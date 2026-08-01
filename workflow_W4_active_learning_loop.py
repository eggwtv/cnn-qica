"""
workflow_W4_active_learning_loop.py
=======================================================================
W4 -- Full-depletion Active Learning loop, REAL version: retrains the
actual cnn_v10 architecture (not a demo CNN) each round, and labels
flagged candidates with a REAL OpenMC quick_check call (not a mock) via
your openmc_beavrs_vver1000_v5.py. Both stubs from the previous version
are gone, per your request.

  Optimizer     : not run here directly -- this script's job is the AL
                  data loop (propose -> flag -> simulate -> retrain).
                  Feed the resulting improved cnn_v10 into W7/W3 for the
                  actual optimizer comparison once a round finishes.
  Fitness source: cnn_v10 architecture (cnn_v10_architecture.py),
                  retrained from scratch each round on the growing pool
                  (original ml_dataset_constrained.csv rows + newly
                  OpenMC-labeled AL rows)
  Sensitivity   : MC-Sobol on the current round's CNN, reported per
                  round to check whether the ranking is STABILIZING
                  (a sign the CNN is converging on real physics) or
                  still shifting round to round
  Entropy       : MC-dropout sigma (primary AL flag) -- H_sens shrinking
                  round-over-round is the expected "AL is working" sign
  DMD role      : DMD reconstruction error on each candidate's REAL
                  predicted ppf_steps curve, as a SECOND, free AL vote.
                  Tie-breaks sigma-flagged candidates only -- never
                  overrides sigma (redundant-not-independent, r=0.504
                  per your earlier finding).

REQUIRES: openmc installed + cross-section libraries configured (same
requirement as running openmc_beavrs_vver1000_v5.py yourself). Each
real OpenMC quick_check costs real wall-clock time -- N_QUERIES_PER_ROUND
and the particle/batch counts below are set conservatively (fast mode)
for a first end-to-end test; raise them once you've confirmed the loop
works and want production-quality labels.

Run standalone: python workflow_W4_active_learning_loop.py
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_utils import load_full_dataset, load_dataset, warm_start_pool
from beavrs_geometry import N_POS, N_TYPES, GRID_ROWS, GRID_COLS, flat_to_grid
from cnn_v10_architecture import build_cnn, v10_loss, IDX_PPF_MAX, IDX_CYCLE, IDX_RHO
from dmd_reduction import fit_dmd_modes, dmd_reconstruction_error
from entropy_sensitivity import sobol_first_order_mc

# Real OpenMC quick_check -- swap this import to your actual filename if
# it differs (e.g. openmc_beavrs_vver1000_v5_FIXED if you keep both names
# around).
from openmc_beavrs_vver1000_v5 import run_quick_check

N_ROUNDS = 3
N_QUERIES_PER_ROUND = 15          # real OpenMC calls per round -- keep
                                    # small for a first test, this is the
                                    # expensive step
SIGMA_PERCENTILE_FLAG = 75
MC_SAMPLES = 25
RETRAIN_EPOCHS = 60                # reduced from cnn_v10's 400 -- each AL
                                    # round retrains from scratch, keep
                                    # this bounded; raise once you've
                                    # clocked real per-round wall time
RETRAIN_PATIENCE = 15

# Fast-mode OpenMC settings for AL labeling (NOT publication-quality --
# these trade accuracy for speed since AL only needs a usable ranking
# signal, not a final answer; your quick_check default is
# particles=2000, batches=40, inactive=15 -- this is a faster variant of
# that same call)
OPENMC_PARTICLES = 1500
OPENMC_BATCHES = 30
OPENMC_INACTIVE = 10


def _retrain_cnn(X_grid, Y_main, Y_rho):
    ym_scaler = StandardScaler()
    Ym_sc = ym_scaler.fit_transform(Y_main)
    yr_scaler = StandardScaler()
    Yr_sc = yr_scaler.fit_transform(Y_rho)
    Y_sc = np.concatenate([Ym_sc, Yr_sc], axis=1).astype(np.float32)

    from tensorflow import keras
    model = build_cnn(GRID_ROWS, GRID_COLS, name='cnn_v10_AL_round')
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss=v10_loss, metrics=['mae'])
    es = keras.callbacks.EarlyStopping(monitor='loss', patience=RETRAIN_PATIENCE,
                                        restore_best_weights=True, verbose=0)
    model.fit(X_grid, Y_sc, epochs=RETRAIN_EPOCHS, batch_size=128,
              callbacks=[es], verbose=0)
    return model, ym_scaler, yr_scaler


def _predict_ppf_max(model, ym_scaler, X_flat):
    grid = flat_to_grid(X_flat, GRID_ROWS, GRID_COLS)
    raw = model.predict(grid, verbose=0)
    real_main = raw[:, :34] * ym_scaler.scale_ + ym_scaler.mean_
    return real_main[:, IDX_PPF_MAX]


def _mc_dropout_ppf(model, ym_scaler, X_flat, n_samples=MC_SAMPLES):
    grid = flat_to_grid(X_flat, GRID_ROWS, GRID_COLS)
    samples = np.stack([model(grid, training=True).numpy() for _ in range(n_samples)])
    mean_raw, std_raw = samples.mean(axis=0), samples.std(axis=0)
    mean_real = mean_raw[:, :34] * ym_scaler.scale_ + ym_scaler.mean_
    std_real = std_raw[:, :34] * ym_scaler.scale_
    return mean_real[:, IDX_PPF_MAX], std_real[:, IDX_PPF_MAX]


def _predict_ppf_steps(model, ym_scaler, X_flat, idx_start, idx_end):
    grid = flat_to_grid(X_flat, GRID_ROWS, GRID_COLS)
    raw = model.predict(grid, verbose=0)
    real_main = raw[:, :34] * ym_scaler.scale_ + ym_scaler.mean_
    return real_main[:, idx_start:idx_end]


def run_workflow(max_rows=None, verbose=True):
    from cnn_v10_architecture import IDX_PPF_STEPS_START, IDX_PPF_STEPS_END

    t0 = time.time()
    print("[W4] Loading full multi-head dataset ...")
    X, Y_main, Y_rho = load_full_dataset(max_rows=max_rows)

    # held-out true labels never enter the AL pool, purely for round-over
    # -round sanity reporting
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    n_test = int(0.15 * len(X))
    test_idx, pool_idx = idx[:n_test], idx[n_test:]
    X_test, Y_main_test = X[test_idx], Y_main[test_idx]

    X_pool = X[pool_idx].copy()
    Y_main_pool = Y_main[pool_idx].copy()
    Y_rho_pool = Y_rho[pool_idx].copy()

    round_log = []
    sens_history = []

    for rnd in range(N_ROUNDS):
        print(f"\n[W4] === Round {rnd+1}/{N_ROUNDS} === training pool size={len(X_pool)}")
        X_grid_pool = flat_to_grid(X_pool, GRID_ROWS, GRID_COLS)
        model, ym_scaler, yr_scaler = _retrain_cnn(X_grid_pool, Y_main_pool, Y_rho_pool)

        # quick round sanity check against the held-out test rows
        test_pred = _predict_ppf_max(model, ym_scaler, X_test)
        test_mae = float(np.mean(np.abs(test_pred - Y_main_test[:, IDX_PPF_MAX])))
        print(f"  [W4] held-out ppf_max MAE this round: {test_mae:.4f}")

        fitness_fn = lambda P: _predict_ppf_max(model, ym_scaler, P)
        sens = sobol_first_order_mc(fitness_fn, X_pool, n_types=N_TYPES, n_samples=150)
        sens_history.append(sens)
        if rnd >= 1:
            rank_corr = float(np.corrcoef(sens_history[-1], sens_history[-2])[0, 1])
            print(f"  [W4] sensitivity ranking stability vs previous round: r={rank_corr:.3f} "
                  f"({'stabilizing' if rank_corr > 0.7 else 'still shifting'})")

        candidate_idx = rng.choice(len(X_pool), size=min(500, len(X_pool)), replace=False)
        candidates = X_pool[candidate_idx]

        mean_pred, sigma = _mc_dropout_ppf(model, ym_scaler, candidates)
        sigma_thresh = np.percentile(sigma, SIGMA_PERCENTILE_FLAG)
        sigma_flag = sigma >= sigma_thresh

        pred_curves = _predict_ppf_steps(model, ym_scaler, candidates,
                                          IDX_PPF_STEPS_START, IDX_PPF_STEPS_END)
        dmd_model = fit_dmd_modes(Y_main_pool[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END], rank=4)
        dmd_err = np.array([dmd_reconstruction_error(c, dmd_model) for c in pred_curves])
        dmd_thresh = np.percentile(dmd_err, SIGMA_PERCENTILE_FLAG)
        dmd_flag = dmd_err >= dmd_thresh

        agreement = float(np.mean(sigma_flag == dmd_flag))
        n_disagree = int(np.sum(sigma_flag != dmd_flag))
        print(f"  [W4] sigma flagged {sigma_flag.sum()}, dmd flagged {dmd_flag.sum()}, "
              f"agreement={agreement:.2f}  -> DMD changed the AL batch on "
              f"{n_disagree}/{len(candidates)} candidates this round")

        flagged_idx = np.where(sigma_flag)[0]
        # tie-break within sigma-flagged set by dmd_err (never overrides sigma)
        order = np.lexsort((-dmd_err[flagged_idx], -sigma[flagged_idx]))
        query_idx = candidate_idx[flagged_idx[order[:N_QUERIES_PER_ROUND]]]
        print(f"  [W4] Selected {len(query_idx)} candidates for REAL OpenMC simulation")

        # ---- REAL OpenMC labeling (no mock) ----
        new_rows_main, new_rows_rho = [], []
        for i, qi in enumerate(query_idx):
            pattern = X_pool[qi].tolist()
            print(f"    [W4] OpenMC quick_check {i+1}/{len(query_idx)} ...")
            result = run_quick_check(
                pattern, particles=OPENMC_PARTICLES, batches=OPENMC_BATCHES,
                inactive=OPENMC_INACTIVE, work_dir=f"al_round{rnd}_q{i}",
                boron_search=False,   # match ml_dataset_constrained.csv's
                                       # unborated (0 ppm) convention
            )
            ppf_arr = result['ppf']
            ppf_max_new = float(ppf_arr.max())
            ppf_boc_new = float(ppf_arr[0])
            keff_new = float(result['keff'])
            rho_new = (keff_new - 1.0) / keff_new * 1e5

            # pad/truncate the 31-position OpenMC ppf array to match the
            # ppf_steps target width (31 steps) -- quick_check gives a
            # single BOC-only snapshot, not the full depletion curve, so
            # broadcast it flat as a conservative placeholder for the
            # ppf_steps target and rely on ppf_max/ppf_boc (the ones
            # quick_check genuinely measures) as the real new labels.
            # Swap in run_one_pattern() (full depletion) instead of
            # run_quick_check() here if you want genuine per-step curves
            # for the newly-labeled AL rows too -- costs hours, not
            # minutes, per pattern.
            ppf_steps_new = np.full(31, ppf_max_new, dtype=np.float32)
            ppf_steps_new[0] = ppf_boc_new

            new_rows_main.append(np.concatenate([[ppf_max_new], [ppf_boc_new],
                                                   ppf_steps_new,
                                                   [np.nan]]))  # cycle_length
                                                                 # unknown from
                                                                 # quick_check
            new_rows_rho.append([rho_new])

        new_rows_main = np.array(new_rows_main, dtype=np.float32)
        new_rows_rho = np.array(new_rows_rho, dtype=np.float32)
        # cycle_length unknown from a BOC-only quick_check -- impute with
        # the pool's current mean so the row is usable in this round's
        # retrain without corrupting the cycle-length loss term; replace
        # with run_one_pattern()'s real cycle_length_in_days once you're
        # running full depletion instead of quick_check.
        new_rows_main[:, IDX_CYCLE] = np.nanmean(Y_main_pool[:, IDX_CYCLE])

        X_pool = np.vstack([X_pool, X_pool[query_idx]])
        Y_main_pool = np.vstack([Y_main_pool, new_rows_main])
        Y_rho_pool = np.vstack([Y_rho_pool, new_rows_rho])

        round_log.append({
            'round': rnd, 'n_flagged_sigma': int(sigma_flag.sum()),
            'n_flagged_dmd': int(dmd_flag.sum()), 'agreement_frac': agreement,
            'n_disagree': n_disagree, 'n_queried': len(query_idx),
            'held_out_ppf_mae': test_mae,
        })

    summary = {
        'workflow': 'W4_ActiveLearning_CNN',
        'n_rounds': N_ROUNDS,
        'total_wall_time_s': float(time.time() - t0),
        'final_training_size': len(X_pool),
        'mean_agreement_sigma_dmd': float(np.mean([r['agreement_frac'] for r in round_log])),
        'final_held_out_ppf_mae': round_log[-1]['held_out_ppf_mae'],
    }
    print(f"\n[W4] SUMMARY: {summary}")
    print("[W4] NOTE: cycle_length for newly-labeled rows was imputed (quick_check is "
          "BOC-only) -- swap in run_one_pattern() for real depletion-derived cycle "
          "lengths before treating a retrained model's cycle predictions as final.")
    return summary, round_log


if __name__ == "__main__":
    summary, round_log = run_workflow()
    pd.DataFrame([summary]).to_csv("w4_summary.csv", index=False)
    pd.DataFrame(round_log).to_csv("w4_round_log.csv", index=False)
    print("[SAVED] w4_summary.csv  w4_round_log.csv")
