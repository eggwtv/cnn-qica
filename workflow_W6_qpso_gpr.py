"""
workflow_W6_qpso_gpr.py
=======================================================================
W6 -- Quantum PSO + Gaussian Process Regression surrogate. Different
non-NN fitness source than W1 (PCE), same optimizer as W1. This isolates
the SURROGATE CHOICE effect holding the optimizer fixed -- the mirror
image of what W5 does (isolating the OPTIMIZER effect holding the
surrogate fixed). Between W1 and W6 you get a 2x2-style read: does the
choice of non-NN surrogate matter as much as, more than, or less than
the choice of optimizer mechanism.

  Optimizer     : QuantumPSO (same class/hyperparameters as W1)
  Fitness source: GPROracle (gpr_oracle.py) -- RBF kernel over one-hot
                  positions, exact posterior predictive std (no MC
                  sampling needed)
  Sensitivity   : oracle-agnostic MC Sobol estimator (GPR has no closed
                  -form Sobol the way PCE does, so this workflow is the
                  one that actually exercises entropy_sensitivity.py's
                  black-box estimator for real, not just as a
                  cross-check)
  Entropy       : predictive_entropy_gaussian() computed from GPR's
                  NATIVE posterior std -- no MC-dropout loop required,
                  a genuine efficiency advantage over the CNN pipeline
                  worth reporting explicitly in the write-up
  DMD role      : same reduction idea as W1 -- GPR fits the compressed
                  DMD-mode targets instead of raw timesteps, which also
                  helps GPR's own scalability (fewer independent GPR
                  heads to fit)

Run standalone: python workflow_W6_qpso_gpr.py
"""

import time
import numpy as np
import pandas as pd

from data_utils import load_dataset, train_test_split_simple, warm_start_pool, N_TYPES
from gpr_oracle import GPROracle
from dmd_reduction import fit_dmd_modes, reduce_curve_to_modes
from entropy_sensitivity import (
    sobol_first_order_mc, sensitivity_trust_region, predictive_entropy_gaussian,
)
from qpso_core import QuantumPSO
#fix
from fitness_utils import (
    make_fitness_fn,
    clamp_predictions_for_reporting,
    n_below_floor as count_below_floor,
)

SEEDS = [42, 137, 271]  # fewer seeds by default: GPR fitness calls are
# noticeably slower per-batch than PCE, keep this workflow's wall-clock
# comparable to the others; raise back to the full 5-seed SEEDS list
# from W1/W5 once you've confirmed timing on your machine.
N_GENS = 150
N_PARTICLES = 60
FREE_FRAC =1.0 # 0.65
MAX_TRAIN_POINTS = 600  # GPR is O(n^3); keep this bounded, see gpr_oracle.py


def run_workflow(max_rows=2000, verbose=True):
    t0 = time.time()

    print("[W6] Loading data ...")
    X, y, curves, pos_cols = load_dataset(max_rows=max_rows)
    n_pos = X.shape[1]
    split = train_test_split_simple(X, y, curves)

    print(f"[W6] Fitting GPR surrogate (non-NN, max {MAX_TRAIN_POINTS} train pts) ...")
    gpr = GPROracle(n_types=N_TYPES, max_train_points=MAX_TRAIN_POINTS)
    gpr.fit(split['X_train'], split['y_train'])

    #fix:
    fitness_fn, get_diag = make_fitness_fn(
        gpr,
        use_uncertainty=True
    )

    pred = gpr.predict(split['X_test'])
    mae = float(np.mean(np.abs(pred - split['y_test'])))
    r2 = 1.0 - np.sum((pred - split['y_test']) ** 2) / np.sum(
        (split['y_test'] - split['y_test'].mean()) ** 2)
    print(f"  GPR test MAE={mae:.4f}  R2={r2:.4f}")

    print("[W6] MC Sobol sensitivity (GPR has no closed-form Sobol) ...")
    sens = sobol_first_order_mc(gpr, split['X_train'], n_types=N_TYPES, n_samples=150)
    free_mask = sensitivity_trust_region(sens, free_frac=FREE_FRAC)
    print(f"  free positions: {free_mask.sum()}/{n_pos}")

    print("[W6] Checking GPR native uncertainty vs error correlation "
          "(the CNN needed MC-dropout for this; GPR gets it for free) ...")
    mean_pred, std_pred = gpr.predict_with_uncertainty(split['X_test'])
    abs_err = np.abs(mean_pred - split['y_test'])
    unc_err_r = float(np.corrcoef(std_pred, abs_err)[0, 1])
    print(f"  sigma-vs-|error| correlation r={unc_err_r:.3f}")

    seeds_pool = warm_start_pool(split['X_train'], split['y_train'], n_seeds=50)

    results = []
    for seed in SEEDS:
        opt = QuantumPSO(n_pos=n_pos, n_types=N_TYPES, n_particles=N_PARTICLES,
                          n_gens=N_GENS, seed=seed, free_mask=free_mask)
        t_start = time.time()
        #out = opt.run(gpr, warm_start_patterns=seeds_pool, tag=f"W6|s{seed}", verbose=verbose)
        #fix:
        out = opt.run(fitness_fn, warm_start_patterns=seeds_pool, tag=f"W6|s{seed}", verbose=verbose)
        wall = time.time() - t_start

        # per-candidate uncertainty check on the winning pattern -- the
        # GPR analogue of your CNN "sigma < threshold -> trust this"
        # check, computed in one call instead of 25-30 MC-dropout passes
        _, best_std = gpr.predict_with_uncertainty(out['best_pattern'].reshape(1, -1))
        best_H = float(predictive_entropy_gaussian(best_std)[0])

        results.append({'seed': seed, 'best_fitness': out['best_fitness'],
                         'best_pattern': out['best_pattern'], 'wall_time_s': wall,
                         'best_sigma': float(best_std[0]), 'best_entropy': best_H,
                         'history': out['history']})
        print(f"  [W6|s{seed}] DONE best_fitness={out['best_fitness']:.4f}  "
              f"sigma={best_std[0]:.4f}  {wall:.1f}s")

    #fix: 
    # fits = np.array([r['best_fitness'] for r in results])
    raw = np.array([r['best_fitness'] for r in results])

    fits = clamp_predictions_for_reporting(raw)

    # same floor definition as W1/W5 (5th percentile of TRAINING LABELS)
    train_floor = float(np.percentile(split['y_train'], 5))
    below_floor_count = count_below_floor(raw, train_floor)
    n_below_floor = int(np.sum(fits < train_floor))

    summary = {
        'workflow': 'W6_QPSO_GPR',
        'ppf_mean': float(fits.mean()), 'ppf_std': float(fits.std()),
        'ppf_min': float(fits.min()), 'ppf_max': float(fits.max()),
        'total_wall_time_s': float(time.time() - t0),
        'gpr_test_mae': mae, 'gpr_test_r2': r2,
        'gpr_sigma_error_corr': unc_err_r,
        'train_floor_proxy': train_floor, 'n_below_floor': below_floor_count,
        'n_free_positions': int(free_mask.sum()), 'n_total_positions': n_pos,
        'n_seeds': len(SEEDS),
    }
    print(f"\n[W6] SUMMARY: ppf={summary['ppf_mean']:.4f}+/-{summary['ppf_std']:.4f}  "
          f"total_time={summary['total_wall_time_s']:.1f}s")
    return summary, results

if __name__ == "__main__":
    summary, results = run_workflow()
    pd.DataFrame([summary]).to_csv("w6_summary.csv", index=False)
    print("[SAVED] w6_summary.csv")

    from workflow_plots import save_workflow_plots
    raw = np.array([r['best_fitness'] for r in results])
    save_workflow_plots("w6", results, summary['train_floor_proxy'], raw)