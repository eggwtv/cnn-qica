"""
workflow_W3_qica_cnn_adapter.py
=======================================================================
W3 -- your EXISTING QICA + CNN pipeline (qica_v11_production.py /
qica_v5-2.py against cnn_v9/v10). This is NOT reimplemented here --
you already have a mature, validated version of it (5 seeds x 250 gens,
sensitivity + H_sens entropy already wired in, SHAP cross-check, OpenMC
export). Reimplementing it would just create a second, worse copy that
could silently drift from the one you actually trust.

Instead, this is a thin ADAPTER: it reads whichever QICA summary CSV
your production script already writes (qica_v11_summary.csv /
qica_ablation_v10_final_summary.csv / qica_final_summary.csv --
whichever exists) and reshapes it into the SAME schema every other
workflow in this suite reports, so compare_workflows.py can put it in
the same table without you re-running anything.

If you want a genuinely fresh W3 run for a same-day, same-conditions
comparison against W1/W5/W6/W7, just run your existing qica_v11 script
normally and point QICA_SUMMARY_CANDIDATES at its output -- this adapter
will pick it up automatically.
"""

import os
import numpy as np
import pandas as pd

QICA_SUMMARY_CANDIDATES = [
    "qica_v11_summary.csv",
    "qica_ablation_v10_final_summary.csv",
    "qica_final_summary.csv",
]
QICA_HISTORY_CANDIDATES = [
    "qica_v11_history.csv",
    "qica_ablation_v10_final_history.csv",
    "qica_final_history.csv",
]


def load_existing_qica_results():
    """
    Tries each known filename your production QICA scripts already
    write. Expected columns (from your own scripts): seed, best_ppf (or
    ppf), best_cycle (or cycle), wall time is NOT currently saved by
    your QICA scripts (add a per-seed timer if you want this column
    populated automatically instead of filled in by hand below).
    """
    for path in QICA_SUMMARY_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path
    return None, None


def run_workflow(fallback_ppf_mean=1.7534, fallback_ppf_std=0.0139,
                  fallback_cycle=313.6, fallback_wall_time_s=2238.0,
                  fallback_sens_entropy_r=-0.31):
    """
    fallback_* defaults are your own already-reported QICA v11 numbers
    (USE_CYCLE_FITNESS=False run: best_ppf=1.7534+/-0.0139, ~37 min
    wall time across 5 seeds) so this adapter still produces a usable
    comparison row even if no CSV is found on disk in this environment.
    """
    df, path = load_existing_qica_results()

    if df is not None:
        print(f"[W3] Loaded existing QICA+CNN results from {path}")
        ppf_col = 'best_ppf' if 'best_ppf' in df.columns else 'ppf'
        fits = df[ppf_col].values.astype(float)
        summary = {
            'workflow': 'W3_QICA_CNN',
            'ppf_mean': float(fits.mean()), 'ppf_std': float(fits.std()),
            'ppf_min': float(fits.min()), 'ppf_max': float(fits.max()),
            'total_wall_time_s': float('nan'),  # add a timer to your
            # production QICA script if you want this populated
            'n_seeds': len(fits), 'source_file': path,
        }
    else:
        print("[W3] No existing QICA summary CSV found in this environment -- "
              "using your own last-reported production numbers as a placeholder "
              "row. Replace with a fresh run's CSV before treating this as final.")
        summary = {
            'workflow': 'W3_QICA_CNN',
            'ppf_mean': fallback_ppf_mean, 'ppf_std': fallback_ppf_std,
            'ppf_min': fallback_ppf_mean - fallback_ppf_std,
            'ppf_max': fallback_ppf_mean + fallback_ppf_std,
            'total_wall_time_s': fallback_wall_time_s,
            'n_seeds': 5, 'source_file': 'PLACEHOLDER (your last reported run)',
        }

    summary['cycle_mean'] = fallback_cycle
    summary['sens_entropy_per_seed_r'] = fallback_sens_entropy_r
    return summary, df


if __name__ == "__main__":
    summary, df = run_workflow()
    pd.DataFrame([summary]).to_csv("w3_summary.csv", index=False)
    print(f"[W3] SUMMARY: {summary}")
    print("[SAVED] w3_summary.csv")
