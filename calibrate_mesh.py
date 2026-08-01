"""
=============================================================================
calibrate_mesh.py -- combined calibration tool, now with an automatic sweep
=============================================================================
Row-flip was TESTED and DISPROVEN last round (it clipped 11/31 real
positions to zero pin power -- a geometry-corruption artifact, not a fix).
This version drops that path as a serious option and instead sweeps the two
remaining live suspects directly against ml_dataset_constrained.csv's
pattern_00001:

  1. Reflector material: 'water' (old, dead-code steel) vs 'steel' (FIX J)
  2. BP rod rotation variant for the 6-count and 15-count patterns (real
     BEAVRS uses 4 rotations per count depending on core quadrant; this
     model previously hardcoded ONE rotation everywhere)

Usage:
    python calibrate_mesh.py                      # single run, current settings
    python calibrate_mesh.py --sweep               # full automatic sweep (~10-15 min)
    python calibrate_mesh.py --sweep --particles 2000 --batches 40 --inactive 25
                                                    # faster/coarser sweep first pass

The sweep runs in two stages to keep the run count sane:
  Stage 1: reflector material only (2 runs, BP variants held at default)
  Stage 2: BP rotation variants (up to 16 combos) using the WINNING
           reflector material from stage 1
Each combo is scored by mean |delta| (PPF) against the true per-position
values, plus keff error in pcm and whether the argmax position matches.
=============================================================================
"""

import argparse
import itertools
import numpy as np
import pandas as pd

import openmc_beavrs_vver1000_v5_FIXED as sim
from openmc_beavrs_vver1000_v5_FIXED import (
    run_quick_check,
    CALIBRATION_PATTERN,
    CALIBRATION_TRUE_PPF,
    CALIBRATION_TRUE_KEFF,
    CALIBRATION_TRUE_CYCLE_DAYS,
    N_POS,
)

TRUE_PPF = np.array([CALIBRATION_TRUE_PPF[p] for p in range(N_POS)])
TRUE_ARGMAX = int(TRUE_PPF.argmax())


def score_one_run(particles, batches, inactive, work_dir, use_row_flip=False):
    result = run_quick_check(
        CALIBRATION_PATTERN, particles=particles, batches=batches, inactive=inactive,
        work_dir=work_dir, use_row_flip=use_row_flip,
    )
    got_ppf = result['ppf']
    delta = got_ppf - TRUE_PPF
    abs_err = np.abs(delta)
    keff_err_pcm = (result['keff'] - CALIBRATION_TRUE_KEFF) / CALIBRATION_TRUE_KEFF * 1e5
    return dict(
        keff=result['keff'], keff_err_pcm=keff_err_pcm,
        argmax=int(got_ppf.argmax()), argmax_match=(int(got_ppf.argmax()) == TRUE_ARGMAX),
        mean_abs_err=float(abs_err.mean()), max_abs_err=float(abs_err.max()),
        ppf=got_ppf.tolist(),
    )


def print_row(label, r):
    print(f"  {label:<28} keff_err={r['keff_err_pcm']:+7.0f}pcm  "
          f"mean|d|={r['mean_abs_err']:.4f}  max|d|={r['max_abs_err']:.4f}  "
          f"argmax={r['argmax']:>2} ({'MATCH' if r['argmax_match'] else 'miss'})")


def run_sweep(particles, batches, inactive):
    print("=" * 70)
    print("STAGE 1 -- reflector material sweep (BP variants held at default 6N/15NW)")
    print("=" * 70)
    stage1_rows = []
    for refl in ['water', 'steel']:
        sim.REFLECTOR_MATERIAL = refl
        sim.BP_VARIANT_6, sim.BP_VARIANT_15 = '6N', '15NW'
        r = score_one_run(particles, batches, inactive, f'sweep_refl_{refl}')
        r['reflector'] = refl
        stage1_rows.append(r)
        print_row(f"reflector={refl}", r)

    best1 = min(stage1_rows, key=lambda r: r['mean_abs_err'])
    best_reflector = best1['reflector']
    print(f"\n  -> Stage 1 winner: reflector={best_reflector} "
          f"(mean|d|={best1['mean_abs_err']:.4f})")

    print("\n" + "=" * 70)
    print(f"STAGE 2 -- BP rotation variant sweep (reflector fixed at {best_reflector})")
    print("=" * 70)
    sim.REFLECTOR_MATERIAL = best_reflector
    variants_6 = ['6N', '6S', '6E', '6W']
    variants_15 = ['15NW', '15NE', '15SW', '15SE']
    stage2_rows = []
    for v6, v15 in itertools.product(variants_6, variants_15):
        sim.BP_VARIANT_6, sim.BP_VARIANT_15 = v6, v15
        r = score_one_run(particles, batches, inactive, f'sweep_bp_{v6}_{v15}')
        r['bp6'], r['bp15'] = v6, v15
        stage2_rows.append(r)
        print_row(f"BP={v6}/{v15}", r)

    best2 = min(stage2_rows, key=lambda r: r['mean_abs_err'])
    print(f"\n  -> Stage 2 winner: BP variants={best2['bp6']}/{best2['bp15']}  "
          f"(mean|d|={best2['mean_abs_err']:.4f})")

    print("\n" + "=" * 70)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 70)
    print(f"  REFLECTOR_MATERIAL = '{best_reflector}'")
    print(f"  BP_VARIANT_6        = '{best2['bp6']}'")
    print(f"  BP_VARIANT_15       = '{best2['bp15']}'")
    print(f"  keff error          = {best2['keff_err_pcm']:+.0f} pcm  "
          f"(target: within ~a few hundred pcm)")
    print(f"  mean |delta| (PPF)  = {best2['mean_abs_err']:.4f}")
    print(f"  argmax position     = {best2['argmax']} "
          f"({'MATCHES true position 8' if best2['argmax_match'] else 'still MISMATCH vs true position 8'})")

    all_rows = [dict(stage=1, reflector=r['reflector'], bp6='6N', bp15='15NW', **{
        k: v for k, v in r.items() if k not in ('reflector', 'ppf')}) for r in stage1_rows]
    all_rows += [dict(stage=2, reflector=best_reflector, **{
        k: v for k, v in r.items() if k not in ('ppf',)}) for r in stage2_rows]
    pd.DataFrame(all_rows).to_csv('calibration_sweep_results.csv', index=False)
    print(f"\n[SAVED] calibration_sweep_results.csv")

    if not best2['argmax_match'] or best2['mean_abs_err'] > 0.3:
        print("\n  NOTE: even the best combo in this sweep did not fully close the gap.")
        print("  That means the remaining error is NOT explained by reflector material")
        print("  or BP rotation alone -- worth checking next: enrichment/BP-count table")
        print("  in ASSEMBLY_LIBRARY against your actual data generator's assumptions,")
        print("  and whether GUIDE_TUBE_POS / INSTRUMENT_TUBE_POS match the real 17x17")
        print("  BEAVRS assembly pin map (BEAVRS spec section on fuel assembly design).")


def run_single(particles, batches, inactive, use_row_flip):
    print("=" * 70)
    print(f"CALIBRATING against ml_dataset_constrained.csv pattern_00001")
    print(f"  row_flip={use_row_flip}  reflector={sim.REFLECTOR_MATERIAL}  "
          f"bp6={sim.BP_VARIANT_6}  bp15={sim.BP_VARIANT_15}   "
          f"{particles}p x {batches}b, {inactive} inactive")
    print("=" * 70)
    print(f"  pattern: {CALIBRATION_PATTERN}")
    print(f"  TRUE keff_BOC        = {CALIBRATION_TRUE_KEFF:.4f}")
    print(f"  TRUE ppf_max (BOC)   = {TRUE_PPF.max():.3f} at position {TRUE_ARGMAX}")
    print(f"  TRUE cycle_length    = {CALIBRATION_TRUE_CYCLE_DAYS:.1f} days "
          f"(quick_check can't verify this -- BOC only)")

    r = score_one_run(particles, batches, inactive,
                       f"calibrate_{'rowflip' if use_row_flip else 'normal'}",
                       use_row_flip=use_row_flip)
    got_ppf = np.array(r['ppf'])
    delta = got_ppf - TRUE_PPF

    print("\n" + "=" * 70)
    print("PER-POSITION COMPARISON (true vs. computed)")
    print("=" * 70)
    print(f"{'pos':>4} {'true_ppf':>9} {'got_ppf':>9} {'delta':>8}")
    for p in range(N_POS):
        flag = "  <-- TRUE MAX" if p == TRUE_ARGMAX else \
               ("  <-- YOUR MAX" if p == r['argmax'] else "")
        print(f"{p:>4} {TRUE_PPF[p]:>9.3f} {got_ppf[p]:>9.3f} {delta[p]:>+8.3f}{flag}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print_row("this run", r)
    print(f"\n  Run the full automatic sweep instead of guessing manually:")
    print(f"    python calibrate_mesh.py --sweep --particles {particles} "
          f"--batches {batches} --inactive {inactive}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--particles', type=int, default=4000)
    ap.add_argument('--batches', type=int, default=60)
    ap.add_argument('--inactive', type=int, default=40)
    ap.add_argument('--try_row_flip', action='store_true',
                     help='DISPROVEN -- kept only for reference, do not use.')
    ap.add_argument('--sweep', action='store_true',
                     help='Automatic reflector x BP-rotation sweep instead of one run.')
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args.particles, args.batches, args.inactive)
    else:
        run_single(args.particles, args.batches, args.inactive, args.try_row_flip)


if __name__ == '__main__':
    main()