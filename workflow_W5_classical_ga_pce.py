"""
workflow_W5_classical_ga_pce.py
=======================================================================
W5 -- Classical GA (NO quantum probability encoding), same PCE oracle as
W1. This is the floor control: it isolates whether QPSO's quantum
representation is earning its keep, using the SAME surrogate as W1, so
any difference in outcome is attributable to the optimizer mechanism
alone, not the fitness source. This mirrors the plain-GA-vs-QICA result
you already have for the NN pipeline (plain GA "won" but only by finding
fake wins below the training floor) -- here you're asking the same
question one layer over: does dropping the quantum representation from
QPSO cause the same kind of exploitation against a PCE surrogate.

  Optimizer     : ClassicalGA (qpso_core.py) -- tournament selection,
                  uniform crossover, mutation. No probability vectors.
  Fitness source: same PCEOracle as W1 (fit once, shared)
  Sensitivity   : same Sobol indices as W1 (shared, for fair comparison)
  Entropy       : plain population diversity (fraction of unique
                  patterns), NOT sensitivity-weighted -- deliberately
                  the "naive" version, since this workflow's whole point
                  is being the unsophisticated control
  DMD role      : none (by design -- this is the floor/null workflow)

Run standalone: python workflow_W5_classical_ga_pce.py
(re-fits its own PCE if w1 hasn't been run first; pass a pre-fit PCE via
run_workflow(pce=...) to guarantee W1 and W5 share the exact same
surrogate, which is the fairer comparison)
"""

import time
import numpy as np
import pandas as pd

from data_utils import load_dataset, train_test_split_simple, warm_start_pool, N_TYPES
from pce_oracle import PCEOracle
from entropy_sensitivity import sobol_first_order_mc, sensitivity_trust_region
from qpso_core import ClassicalGA
#fix:
from fitness_utils import (
    make_fitness_fn,
    clamp_predictions_for_reporting,
    n_below_floor as count_below_floor,
)

SEEDS = [42, 137, 271, 509, 1023]
N_GENS = 250
POP_SIZE = 80
FREE_FRAC = 1.0 #0.65


def run_workflow(max_rows=4000, pce=None, split=None, free_mask=None, verbose=True):
    t0 = time.time()

    if split is None:
        print("[W5] Loading data ...")
        X, y, curves, pos_cols = load_dataset(max_rows=max_rows)
        split = train_test_split_simple(X, y, curves)
    n_pos = split['X_train'].shape[1]

    if pce is None:
        print("[W5] Fitting PCE surrogate (same recipe as W1, independent fit) ...")
        pce = PCEOracle(n_types=N_TYPES, order=2)
        pce.fit(split['X_train'], split['y_train'])

        #fix_2: 
        pce.fit_ensemble(split['X_train'], split['y_train'])

        #fix:
        fitness_fn, get_diag = make_fitness_fn(pce)
    if free_mask is None:
        sens = sobol_first_order_mc(pce, split['X_train'], n_types=N_TYPES, n_samples=200)
        free_mask = sensitivity_trust_region(sens, free_frac=FREE_FRAC)

    seeds_pool = warm_start_pool(split['X_train'], split['y_train'], n_seeds=50)

    results = []
    for seed in SEEDS:
        opt = ClassicalGA(n_pos=n_pos, n_types=N_TYPES, pop_size=POP_SIZE,
                           n_gens=N_GENS, seed=seed, free_mask=free_mask)
        t_start = time.time()

        #fix:
        #out = opt.run(pce, warm_start_patterns=seeds_pool, tag=f"W5|s{seed}", verbose=verbose)
        out = opt.run(fitness_fn, warm_start_patterns=seeds_pool,tag=f"W5|s{seed}", verbose=verbose)
                      
        wall = time.time() - t_start
        results.append({'seed': seed, 'best_fitness': out['best_fitness'],
                         'best_pattern': out['best_pattern'], 'wall_time_s': wall,
                         'history': out['history']})
        print(f"  [W5|s{seed}] DONE best_fitness={out['best_fitness']:.4f}  {wall:.1f}s")

    #fix:
    #fits = np.array([r['best_fitness'] for r in results])
    raw = np.array([r['best_fitness'] for r in results])

    fits = clamp_predictions_for_reporting(raw)

    train_floor = np.percentile(split['y_train'],5)

    below_floor_count = count_below_floor(raw, train_floor)

    # OOD / floor check: is the best pattern's fitness lower than
    # anything a PCE-honest surrogate should trust? Uses the 5th
    # percentile of TRAINING LABELS as the floor -- deliberately the
    # SAME definition as your original QICA floor (~1.697, "5th
    # percentile of real training labels"), NOT the later "5th
    # percentile of the CNN's own PREDICTED distribution" definition
    # your final suite flagged as inconsistent (that alternate
    # definition put QICA's own baseline below its own floor). Keeping
    # every workflow in this suite on the training-LABEL-percentile
    # definition avoids reintroducing that same inconsistency here.
    train_floor = float(np.percentile(split['y_train'], 5))
    n_below_floor = int(np.sum(fits < train_floor))

    summary = {
        'workflow': 'W5_ClassicalGA_PCE',
        'ppf_mean': float(fits.mean()), 'ppf_std': float(fits.std()),
        'ppf_min': float(fits.min()), 'ppf_max': float(fits.max()),
        'total_wall_time_s': float(time.time() - t0),
        'train_floor_proxy': train_floor,
        'n_below_floor': n_below_floor, 'n_seeds': len(SEEDS),
        'n_free_positions': int(free_mask.sum()), 'n_total_positions': n_pos,
        'n_below_floor': below_floor_count,
    }
    print(f"\n[W5] SUMMARY: ppf={summary['ppf_mean']:.4f}+/-{summary['ppf_std']:.4f}  "
          f"below_floor={n_below_floor}/{len(SEEDS)}  "
          f"total_time={summary['total_wall_time_s']:.1f}s")
    return summary, results


if __name__ == "__main__":
    summary, results = run_workflow()
    pd.DataFrame([summary]).to_csv("w5_summary.csv", index=False)

    from workflow_plots import save_workflow_plots
    raw = np.array([r['best_fitness'] for r in results])
    save_workflow_plots("w5", results, summary['train_floor_proxy'], raw)

    print("[SAVED] w5_summary.csv")