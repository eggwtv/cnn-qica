"""
=============================================================================
al_summary_and_dedup.py — post-hoc AL stats + dedup, no QICA re-run needed
=============================================================================
Reads your EXISTING qica_final_al_candidates.csv (already on disk from your
completed run) and:
  1. Prints per-seed AL candidate counts (the CSV already has a 'seed' column
     per candidate, so this is just a groupby — no re-simulation needed).
  2. Deduplicates across seeds (same pattern flagged by >1 seed), keeping the
     highest-composite instance of each unique pattern.
  3. Saves the deduped result as qica_final_al_candidates_deduped.csv — THIS
     is the file to feed to openmc_beavrs_vver1000.py's --al_candidates_csv,
     not the raw one.

USAGE:
  python al_summary_and_dedup.py --al_csv qica_final_al_candidates.csv
=============================================================================
"""

import argparse
import pandas as pd

N_POS = 31


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--al_csv', default='qica_final_al_candidates.csv')
    ap.add_argument('--out_csv', default=None,
                     help='Default: <al_csv>_deduped.csv')
    args = ap.parse_args()

    out_csv = args.out_csv or args.al_csv.rsplit('.', 1)[0] + '_deduped.csv'

    df = pd.read_csv(args.al_csv)
    pos_cols = [f'pos_{i}' for i in range(N_POS)]
    missing = [c for c in pos_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{args.al_csv} is missing columns {missing[:3]}... "
                          f"— is this really a qica_*_al_candidates.csv file?")

    print(f"[LOAD] {args.al_csv}  ({len(df)} total flagged rows)")

    print(f"\n{'='*50}")
    print("ACTIVE LEARNING CANDIDATE SUMMARY (per seed)")
    print(f"{'='*50}")
    if 'seed' in df.columns:
        per_seed = df.groupby('seed').size().sort_index()
        print(f"  {'Seed':<8} {'n_al':>6}")
        print(f"  {'-'*16}")
        for seed, n in per_seed.items():
            print(f"  {seed:<8} {n:>6}")
        print(f"  {'-'*16}")
        print(f"  Sum across seeds (with cross-seed duplicates): {len(df)}")
    else:
        print("  No 'seed' column found — treating all rows as one batch.")

    sort_col = 'composite' if 'composite' in df.columns else (
        'priority' if 'priority' in df.columns else None)
    if sort_col:
        deduped = (df.sort_values(sort_col, ascending=False)
                     .drop_duplicates(subset=pos_cols, keep='first')
                     .reset_index(drop=True))
    else:
        deduped = df.drop_duplicates(subset=pos_cols, keep='first').reset_index(drop=True)

    print(f"\n  Unique patterns after cross-seed dedup: {len(deduped)}")
    print(f"  ({len(df) - len(deduped)} duplicate flags removed)")

    deduped.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}")
    print(f"  Use THIS file with openmc_beavrs_vver1000.py --al_candidates_csv "
          f"(not the raw {args.al_csv}) to avoid wasting OpenMC runs on duplicates.")
    print(f"\n  Example next step (test 10 of them):")
    print(f"  python openmc_beavrs_vver1000.py --al_candidates_csv {out_csv} "
          f"--al_top_n 10 --quick_check --boron_search --particles 2000 --batches 40 --inactive 15")


if __name__ == '__main__':
    main()