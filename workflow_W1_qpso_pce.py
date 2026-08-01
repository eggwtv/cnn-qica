"""
workflow_W1_qpso_pce.py
=======================================================================
W1 -- Quantum PSO + Polynomial Chaos Expansion surrogate. NO neural
network anywhere in this workflow.

  Optimizer     : QuantumPSO (qpso_core.py)
  Fitness source: PCEOracle (pce_oracle.py) -- order-2 (pairwise) PCE
  Sensitivity   : PCE's own analytic Sobol indices (exact, no sampling)
                  cross-checked against the oracle-agnostic MC Sobol
                  estimator from entropy_sensitivity.py
  Entropy       : sensitivity-weighted trust region (freeze high-Sobol
                  positions, free the rest) -- "Mode C" style
  DMD role      : model-order reduction of the burnup-step curve target
                  (31/69 raw timesteps -> ~4 DMD mode coefficients),
                  fit one small PCE per mode instead of one per timestep

Run standalone: python workflow_W1_qpso_pce.py
"""

import time
import numpy as np
import pandas as pd


from data_utils import load_dataset, train_test_split_simple, warm_start_pool, N_TYPES
from pce_oracle import PCEOracle, fit_curve_via_dmd
from dmd_reduction import fit_dmd_modes, variance_explained_by_rank
from entropy_sensitivity import (
    sobol_first_order_mc, sensitivity_trust_region,
    position_entropy_from_training,
)
from qpso_core import QuantumPSO
#fix:
from fitness_utils import (
    make_fitness_fn,
    clamp_predictions_for_reporting,
    n_below_floor as count_below_floor,
)

SEEDS = [42, 137, 271, 509, 1023]
N_GENS = 250
N_PARTICLES = 80
FREE_FRAC = 1.0 #0.65


def run_workflow(max_rows=4000, verbose=True):
    t0 = time.time()

    print("[W1] Loading data ...")
    X, y, curves, pos_cols = load_dataset(max_rows=max_rows)
    n_pos = X.shape[1]
    split = train_test_split_simple(X, y, curves)

    print("[W1] Fitting PCE surrogate (order-2, non-NN) ...")
    pce = PCEOracle(n_types=N_TYPES, order=2)
    pce.fit(split['X_train'], split['y_train'])

    #fix_2: 
    pce.fit_ensemble(split['X_train'], split['y_train'])

    #fix:

    fitness_fn, get_diag = make_fitness_fn(
    pce,
    use_uncertainty=True,      # uses ensemble if your PCE supports it
    )
    pred = pce.predict(split['X_test'])
    mae = float(np.mean(np.abs(pred - split['y_test'])))
    r2 = 1.0 - np.sum((pred - split['y_test']) ** 2) / np.sum(
        (split['y_test'] - split['y_test'].mean()) ** 2)
    print(f"  PCE test MAE={mae:.4f}  R2={r2:.4f}  CV_MAE={pce.cv_mae:.4f}")

    print("[W1] Analytic Sobol sensitivity from PCE coefficients ...")
    sens_analytic = pce.sobol_indices()
    print("[W1] Cross-checking with oracle-agnostic MC Sobol estimator ...")
    sens_mc = sobol_first_order_mc(pce, split['X_train'], n_types=N_TYPES, n_samples=200)
    rank_corr = float(np.corrcoef(sens_analytic, sens_mc)[0, 1])
    print(f"  analytic-vs-MC Sobol correlation r={rank_corr:.3f} "
          f"(should be strongly positive -- sanity check on the MC estimator)")

    if curves is not None:
        print("[W1] Fitting DMD model-order-reduction on burnup curves ...")
        var_explained = variance_explained_by_rank(split['curves_train'], max_rank=8)
        print(f"  variance explained by rank 1..8: {np.round(var_explained, 3)}")
        rank = int(np.searchsorted(var_explained, 0.97)) + 1
        rank = max(2, min(rank, 6))
        dmd_model = fit_dmd_modes(split['curves_train'], rank=rank)
        print(f"  using rank={rank} DMD modes "
              f"({2*rank} real features vs {split['curves_train'].shape[1]} raw timesteps)")
        curve_pce_models = fit_curve_via_dmd(PCEOracle, split['X_train'],
                                              split['curves_train'], dmd_model,
                                              n_types=N_TYPES, order=1)
        print(f"  fit {len(curve_pce_models)} small PCE models "
              f"(one per DMD mode) instead of {split['curves_train'].shape[1]} raw-timestep PCEs")
    else:
        dmd_model = None

    print("[W1] Building sensitivity-defined trust region ...")
    free_mask = sensitivity_trust_region(sens_analytic, free_frac=FREE_FRAC)
    print(f"  free positions: {free_mask.sum()}/{n_pos}")

    H_train = position_entropy_from_training(split['X_train'], n_types=N_TYPES)

    seeds_pool = warm_start_pool(split['X_train'], split['y_train'], n_seeds=50)

    results = []
    for seed in SEEDS:
        opt = QuantumPSO(n_pos=n_pos, n_types=N_TYPES, n_particles=N_PARTICLES,
                          n_gens=N_GENS, seed=seed, free_mask=free_mask)
        t_start = time.time()
        #out = opt.run(pce, warm_start_patterns=seeds_pool, tag=f"W1|s{seed}", verbose=verbose)
        #fix:
        out = opt.run(
            fitness_fn,
            warm_start_patterns=seeds_pool,
            tag=f"W1|s{seed}",
            verbose=verbose
        )

        wall = time.time() - t_start
        mean_r, per_seed_r = None, None
        results.append({
            'seed': seed, 'best_fitness': out['best_fitness'],
            'best_pattern': out['best_pattern'], 'wall_time_s': wall,
            'history': out['history'],
        })
        print(f"  [W1|s{seed}] DONE best_fitness={out['best_fitness']:.4f}  {wall:.1f}s")

    #fix: 
    #fits = np.array([r['best_fitness'] for r in results])
    raw = np.array([r['best_fitness'] for r in results])

    fits = clamp_predictions_for_reporting(raw)

    train_floor = np.percentile(split['y_train'],5)

    below_floor_count = count_below_floor(raw, train_floor)

    # same floor definition as W5 (5th percentile of TRAINING LABELS,
    # not the CNN-predicted-distribution definition your final suite
    # flagged as inconsistent) -- kept identical across every workflow
    # in this suite so n_below_floor is directly comparable
    train_floor = float(np.percentile(split['y_train'], 5))
    n_below_floor = int(np.sum(fits < train_floor))

    summary = {
        'workflow': 'W1_QPSO_PCE',
        'ppf_mean': float(fits.mean()), 'ppf_std': float(fits.std()),
        'ppf_min': float(fits.min()), 'ppf_max': float(fits.max()),
        'total_wall_time_s': float(time.time() - t0),
        'pce_test_mae': mae, 'pce_test_r2': r2,
        'sobol_analytic_vs_mc_r': rank_corr,
        'train_floor_proxy': train_floor, 'n_below_floor': n_below_floor,
        'n_seeds': len(SEEDS),
        'n_free_positions': int(free_mask.sum()),
        'n_total_positions': n_pos,
        'below_floor_count': below_floor_count
    }
    print(f"\n[W1] SUMMARY: ppf={summary['ppf_mean']:.4f}+/-{summary['ppf_std']:.4f}  "
          f"total_time={summary['total_wall_time_s']:.1f}s")
    return summary, results


if __name__ == "__main__":
    summary, results = run_workflow()
    pd.DataFrame([summary]).to_csv("w1_summary.csv", index=False)
    pd.DataFrame([{'seed': r['seed'], 'best_fitness': r['best_fitness'],
                    'wall_time_s': r['wall_time_s']} for r in results]
                 ).to_csv("w1_per_seed.csv", index=False)

    from workflow_plots import save_workflow_plots
    raw = np.array([r['best_fitness'] for r in results])
    save_workflow_plots("w1", results, summary['train_floor_proxy'], raw)

    print("[SAVED] w1_summary.csv  w1_per_seed.csv")