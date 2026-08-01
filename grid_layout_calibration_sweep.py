"""
=============================================================================
grid_layout_calibration_sweep.py
=============================================================================
GOAL: figure out the real loading_0..loading_30 -> (row,col) mapping used by
ml_dataset_constrained.csv, by testing several candidate GRID_LAYOUTs against
known rows and seeing which one's OpenMC per-position PPF map best correlates
with the recorded ppf_s0_a0..a30 labels for the SAME rows.

WHY CORRELATION, NOT ABSOLUTE MATCH:
  Your BP/enrichment template still isn't perfectly calibrated (that's a
  separate, ongoing fix). Absolute PPF magnitude will be off regardless of
  whether the geometry mapping is right. But POSITION-TO-POSITION RANKING
  (is position X higher-power than position Y) is driven mostly by where an
  assembly sits in the core, which is exactly the thing GRID_LAYOUT encodes.
  So: correct mapping -> high position-wise Spearman correlation with the
  recorded labels, even with residual physics error. Wrong mapping -> the
  values get shuffled across positions -> correlation collapses toward 0.

HOW TO USE:
  1. Run this file stand-alone first (no OpenMC needed) — it will print an
     ASCII rendering of each candidate layout. Compare each against your
     labeled core diagram BEFORE spending compute on it. Delete/fix any
     candidate whose printed shape doesn't look right.
  2. Fill in `run_quick_check_ppf_map()` below to call into your existing
     openmc_beavrs_vver1000_FIXED_2.py (this is the ONE integration point —
     everything else is generic).
  3. Run with real rows: `python grid_layout_calibration_sweep.py --score`
=============================================================================
     R  P  N  M  L  K  J  H  G  F  E  D  C  B  A
 1   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 2   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 3   .  .  .  .  .  .  .  .  .  .  .  . 30 31  .
 4   .  .  .  .  .  .  .  .  .  .  . 27 28 29  .
 5   .  .  .  .  .  .  .  .  .  . 22 23 24 25 26
 6   .  .  .  .  .  .  .  .  . 16 17 18 19 20 21
 7   .  .  .  .  .  .  .  .  9 10 11 12 13 14 15 
 8   .  .  .  .  .  .  .  1  2  3  4  5  6  7  8
 9   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
10   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
11   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
12   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
13   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
14   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
15   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
"""


import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Column convention: R P N M L K J H G F E D C B A (15 cols, skip I/O/Q) ──
COLS = ['R', 'P', 'N', 'M', 'L', 'K', 'J', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A']
COL_IDX = {c: i + 1 for i, c in enumerate(COLS)}     # 1..15, H = 8 = center
CENTER_ROW = 8
CENTER_COL = 8   # 'H'
N_POS = 31

# =============================================================================
# STEP 1 — APPROXIMATE FULL-CORE ROW WIDTHS (read off your diagram)
# =============================================================================
# active_cols_in_row[r] = set of column-indices (1..15) that are real fuel
# assemblies in row r (centered on col 8). THESE ARE APPROXIMATE — adjust to
# match your diagram exactly if the printed ASCII in step 2 looks wrong.
# Widths are symmetric about row 8 and about col 8 (standard BEAVRS shape).

ROW_HALF_WIDTH = {   # how many cols to each side of center col 8 are active
    1: 4, 2: 5, 3: 6, 4: 6, 5: 7, 6: 7, 7: 7, 8: 7,
    9: 7, 10: 7, 11: 7, 12: 6, 13: 6, 14: 5, 15: 4,
}

def active_cols_in_row(row):
    hw = ROW_HALF_WIDTH[row]
    return set(range(CENTER_COL - hw, CENTER_COL + hw + 1))

FULL_CORE_MASK = {r: active_cols_in_row(r) for r in range(1, 16)}
print(f"[INFO] Approx full-core assembly count from ROW_HALF_WIDTH: "
      f"{sum(len(v) for v in FULL_CORE_MASK.values())}  (real BEAVRS = 193)")


# =============================================================================
# STEP 2 — OCTANT GENERATORS (candidate hypotheses)
# =============================================================================

def build_octant(rule_fn, mask=FULL_CORE_MASK, exclude_center=True):
    """rule_fn(row, col) -> bool. Returns sorted list of (row,col) tuples."""
    pts = []
    for r in range(1, 16):
        for c in mask[r]:
            if exclude_center and r == CENTER_ROW and c == CENTER_COL:
                continue
            if rule_fn(r, c):
                pts.append((r, c))
    return pts


def order_center_out(pts):
    """Row-major, ordered by increasing distance from centerline row, then
    by increasing distance from center col within each row (nearest-center
    first) — matches the '1,2,3 in row8, then 9,10 in row7' pattern."""
    def key(p):
        r, c = p
        return (abs(r - CENTER_ROW), abs(c - CENTER_COL))
    return sorted(pts, key=key)


# Rule A: strict upper-right wedge, diagonal boundary inclusive
rule_A = lambda r, c: r <= CENTER_ROW and c >= CENTER_COL and (c - CENTER_COL) >= (CENTER_ROW - r)
# Rule B: same wedge, boundary shifted by one row (absorbs the possible off-by-one
# seen when matching '1,2,3'/'9,10' against a strict diagonal)
rule_B = lambda r, c: r <= CENTER_ROW and c >= CENTER_COL and (c - CENTER_COL) >= (CENTER_ROW - r) - 1
# Rule A reflected (row/col swapped) — catches a transpose error
rule_A_T = lambda r, c: c <= CENTER_COL and r >= CENTER_ROW and (r - CENTER_ROW) >= (CENTER_COL - c)

CANDIDATES = {}

for name, rule in [('octant_ruleA', rule_A), ('octant_ruleB', rule_B), ('octant_ruleA_transpose', rule_A_T)]:
    pts = order_center_out(build_octant(rule))
    CANDIDATES[name] = pts[:N_POS]  # truncate/pad-check below

# Control: current placeholder rectangle (6x6, drop the corner), row-major.
# This SHOULD score badly — it's the sanity check that the metric works at all.
placeholder_pts = [(r, c) for r in range(1, 7) for c in range(1, 7)][:31]
CANDIDATES['placeholder_v9_control'] = placeholder_pts


# =============================================================================
# STEP 3 — VISUALIZE (run this before trusting anything)
# =============================================================================

def print_candidate(name, pts):
    print(f"\n=== {name}  (n={len(pts)}, {'OK' if len(pts) == N_POS else 'MISMATCH — fix ROW_HALF_WIDTH or rule'}) ===")
    grid = [['.'] * 15 for _ in range(15)]
    for i, (r, c) in enumerate(pts):
        grid[r - 1][c - 1] = f'{i+1:2d}'
    header = '    ' + ' '.join(f'{c:>2}' for c in COLS)
    print(header)
    for ri, row in enumerate(grid, start=1):
        print(f'{ri:2d}  ' + ' '.join(f'{v:>2}' for v in row))


def visualize_all():
    for name, pts in CANDIDATES.items():
        print_candidate(name, pts)


# =============================================================================
# STEP 4 — INTEGRATION POINT (fill this in — only function you need to edit)
# =============================================================================

def pts_to_grid_layout(pts):
    """Convert a flat (row,col) list into the 2D array format your
    openmc script's GRID_LAYOUT expects (-1 = inactive), 15x15."""
    grid = np.full((15, 15), -1, dtype=int)
    for i, (r, c) in enumerate(pts):
        grid[r - 1, c - 1] = i
    return grid


def run_quick_check_ppf_map(pattern_31, pts):
    """
    *** EDIT THIS FUNCTION ***
    Must return a numpy array of length 31: predicted PPF at BOC for each
    of the 31 positions, in the SAME order as `pts` (i.e. index i in the
    output corresponds to loading position i under this candidate mapping).

    Wire this to your existing openmc_beavrs_vver1000_FIXED_2.py, e.g.:

        import openmc_beavrs_vver1000_FIXED_2 as omc
        omc.GRID_LAYOUT = pts_to_grid_layout(pts)   # monkey-patch the layout
        omc.GUIDE_TUBE_POS = ...                    # keep as-is (per-assembly, unaffected)
        result = omc.quick_check(pattern_31, particles=4000, batches=60, inactive=40)
        return result['ppf_per_position']           # must be length-31, ordered by pos index

    Left as a stub (raises) so you don't accidentally score against fake data.
    """
    raise NotImplementedError(
        "Wire this to your quick_check() — see docstring for the exact hook."
    )


# =============================================================================
# STEP 5 — SCORING
# =============================================================================

def load_known_rows(csv_path='ml_dataset_constrained.csv', n_rows=10, seed=0):
    df = pd.read_csv(csv_path, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    # per-position BOC ppf label columns: ppf_s0_a{i}
    label_cols = [f'ppf_s0_a{i}' for i in range(N_POS) if f'ppf_s0_a{i}' in df.columns]
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(df), size=min(n_rows, len(df)), replace=False)
    rows = []
    for idx in idxs:
        pattern = df.loc[idx, load_cols].values.astype(int)
        labels = df.loc[idx, label_cols].values.astype(float)
        rows.append((pattern, labels))
    return rows


def score_candidate(name, pts, known_rows):
    corrs = []
    for pattern, true_labels in known_rows:
        pred = run_quick_check_ppf_map(pattern, pts)
        if len(pred) != len(true_labels):
            print(f"  [WARN] length mismatch pred={len(pred)} true={len(true_labels)} — skipping row")
            continue
        r, _ = spearmanr(pred, true_labels)
        corrs.append(r)
    if not corrs:
        return np.nan, np.nan
    return float(np.mean(corrs)), float(np.std(corrs))


def run_full_sweep(csv_path='ml_dataset_constrained.csv', n_rows=10):
    known_rows = load_known_rows(csv_path, n_rows=n_rows)
    print(f"\n[SWEEP] Scoring {len(CANDIDATES)} candidates against {len(known_rows)} known rows...")
    results = []
    for name, pts in CANDIDATES.items():
        if len(pts) != N_POS:
            print(f"  Skipping {name}: has {len(pts)} positions, need {N_POS}")
            continue
        mean_r, std_r = score_candidate(name, pts, known_rows)
        results.append((name, mean_r, std_r))
        print(f"  {name:28s}  mean Spearman r = {mean_r:+.3f}  (std {std_r:.3f})")

    results.sort(key=lambda x: (x[1] if x[1] == x[1] else -999), reverse=True)
    print("\n[RANKED]")
    for name, mean_r, std_r in results:
        print(f"  {name:28s}  {mean_r:+.3f}")
    if results:
        print(f"\n[WINNER] {results[0][0]}  (mean r = {results[0][1]:+.3f})")
        print("  Sanity check: 'placeholder_v9_control' should score near 0 —")
        print("  if it scores just as high as the winner, the metric itself isn't")
        print("  discriminating (check that run_quick_check_ppf_map is really")
        print("  using the patched GRID_LAYOUT, not the module's original one).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score', action='store_true', help='run the OpenMC-backed sweep (needs Step 4 wired up)')
    parser.add_argument('--csv', default='ml_dataset_constrained.csv')
    parser.add_argument('--n_rows', type=int, default=10)
    args = parser.parse_args()

    visualize_all()   # always safe to run — no OpenMC needed

    if args.score:
        run_full_sweep(args.csv, args.n_rows)
    else:
        print("\n[INFO] Ran in visualize-only mode. Compare the printed grids above "
              "against your labeled diagram. Once they look right (and you've wired "
              "up run_quick_check_ppf_map), re-run with --score.")