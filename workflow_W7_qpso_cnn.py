"""
workflow_W7_qpso_cnn.py
=======================================================================
W7 -- Quantum PSO + your REAL cnn_v10 surrogate. No stub, no demo/mock
CNN fallback -- this now requires cnn_v10_model.keras + cnn_v10_config.json
to actually be present (raises FileNotFoundError with the exact filenames
it looked for otherwise, via cnn_v10_oracle.CNNOracle).

Purpose unchanged: hold the surrogate exactly fixed (same CNN QICA/W3
already uses), swap ONLY the optimizer from QICA (empire mechanics) to
QuantumPSO (swarm mechanics). Compare this row directly against W3 --
any difference in outcome is attributable purely to the optimizer.

  Optimizer     : QuantumPSO (same class as W1/W6)
  Fitness source: cnn_v10 (real, via CNNOracle.predict_ppf_max)
  Sensitivity   : cnn_v10_sens.csv if present (the gradient sensitivity
                  cnn_v10_octant_retrain.py already computed and saved);
                  falls back to the oracle-agnostic MC Sobol estimator
                  if that file isn't found next to the model
  Entropy       : H_sens (sensitivity-weighted population entropy),
                  identical formula to QICA's, for direct comparability
  DMD role      : none in the optimizer loop itself (that's W4's job);
                  DMD reconstruction error is computed once on the
                  winning pattern's REAL predicted ppf_steps curve
                  (from CNNOracle.predict_full) as a free, reported
                  cross-check

Run standalone: python workflow_W7_qpso_cnn.py
"""

import time
import numpy as np
import pandas as pd

from data_utils import load_dataset, train_test_split_simple, warm_start_pool
from beavrs_geometry import N_TYPES
from cnn_v10_oracle import CNNOracle
from dmd_reduction import fit_dmd_modes, dmd_reconstruction_error
from entropy_sensitivity import (
    sobol_first_order_mc, sensitivity_trust_region,
    sensitivity_weighted_entropy_gate,
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


def run_workflow(max_rows=None, verbose=True):
    t0 = time.time()

    print("[W7] Loading data ...")
    X, y, curves, pos_cols = load_dataset(max_rows=max_rows)
    n_pos = X.shape[1]
    split = train_test_split_simple(X, y, curves)

    print("[W7] Loading real cnn_v10 ...")
    cnn = CNNOracle()
    #fix:

    fitness_fn, get_diag = make_fitness_fn(
        cnn,
        use_uncertainty=True,
    )

    sens = cnn.load_saved_sensitivity()
    if sens is None or len(sens) != n_pos:
        print("[W7] cnn_v10_sens.csv not found/mismatched -- computing MC Sobol instead")
        sens = sobol_first_order_mc(cnn.predict_ppf_max, split['X_train'],
                                     n_types=N_TYPES, n_samples=150)
    else:
        print("[W7] Using existing cnn_v10_sens.csv gradient sensitivity")
    free_mask = sensitivity_trust_region(sens, free_frac=FREE_FRAC)
    print(f"[W7] free positions: {free_mask.sum()}/{n_pos}")

    seeds_pool = warm_start_pool(split['X_train'], split['y_train'], n_seeds=50)

    results = []
    for seed in SEEDS:
        opt = QuantumPSO(n_pos=n_pos, n_types=N_TYPES, n_particles=N_PARTICLES,
                          n_gens=N_GENS, seed=seed, free_mask=free_mask)
        t_start = time.time()
        #out = opt.run(cnn.predict_ppf_max, warm_start_patterns=seeds_pool,tag=f"W7|s{seed}", verbose=verbose)

        #fix:
        out = opt.run(
            fitness_fn,
            warm_start_patterns=seeds_pool,
            tag=f"W7|s{seed}",
            verbose=verbose
        )
        wall = time.time() - t_start

        H_final = out['history']['entropy'][-1]
        H_sens = sensitivity_weighted_entropy_gate(np.full(n_pos, H_final), sens)

        # free DMD cross-check on the winner's REAL predicted curve
        full = cnn.predict_full(out['best_pattern'].reshape(1, -1))
        dmd_err = None
        if curves is not None and len(split['curves_train']) >= 50:
            dmd_model = fit_dmd_modes(split['curves_train'], rank=4)
            dmd_err = dmd_reconstruction_error(full['ppf_steps'][0], dmd_model)

        results.append({'seed': seed, 'best_fitness': out['best_fitness'],
                         'best_pattern': out['best_pattern'], 'wall_time_s': wall,
                         'H_sens_final': H_sens, 'dmd_curve_err': dmd_err,
                         'history': out['history']})
        print(f"  [W7|s{seed}] DONE best_fitness={out['best_fitness']:.4f}  "
              f"dmd_curve_err={dmd_err}  {wall:.1f}s")

    #fits = np.array([r['best_fitness'] for r in results])
    #fix:
    raw = np.array([r['best_fitness'] for r in results])

    fits = clamp_predictions_for_reporting(raw)

    train_floor = float(np.percentile(split['y_train'], 5))
    below_floor_count= count_below_floor(raw, train_floor)

    summary = {
        'workflow': 'W7_QPSO_CNN',
        'ppf_mean': float(fits.mean()), 'ppf_std': float(fits.std()),
        'ppf_min': float(fits.min()), 'ppf_max': float(fits.max()),
        'total_wall_time_s': float(time.time() - t0),
        'n_free_positions': int(free_mask.sum()), 'n_total_positions': n_pos,
        'n_seeds': len(SEEDS),

        #fix:
        'train_floor_proxy': train_floor,
        'n_below_floor': below_floor_count,
    }
    print(f"\n[W7] SUMMARY: ppf={summary['ppf_mean']:.4f}+/-{summary['ppf_std']:.4f}  "
          f"total_time={summary['total_wall_time_s']:.1f}s")
    print("[W7] Compare this row directly against W3 (QICA+CNN) -- same fitness "
          "source, different optimizer mechanism.")
    return summary, results


if __name__ == "__main__":
    summary, results = run_workflow()
    pd.DataFrame([summary]).to_csv("w7_summary.csv", index=False)
    print("[SAVED] w7_summary.csv")

    from workflow_plots import save_workflow_plots
    raw = np.array([r['best_fitness'] for r in results])
    save_workflow_plots("w7", results, summary['train_floor_proxy'], raw, sens=None)