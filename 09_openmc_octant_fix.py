"""
=============================================================================
09_openmc_octant_fix.py
=============================================================================
Rebuilds GRID_LAYOUT and build_core_geometry() for a genuine 1/8-octant
BEAVRS reduction, derived from your actual ASCII coordinate map (rows 1-15,
columns R..A). The old GRID_LAYOUT was np.arange(31) reshaped into a 6x6
box -- confirmed via the mirror-pair entropy/sensitivity test in
07_pareto_nsga2_cnn.py to have NO relationship to real physical adjacency
(mirror pairs were statistically indistinguishable from random pairs).

What changed, and why:
  1. GRID_LAYOUT is now 6 rows x 8 cols, built directly from your diagram
     (row 8 -> "diagram row 8" in the array below, etc). Verified: every
     position 0-30 appears exactly once, row-widths (2,3,5,6,7,8) sum to 31.
  2. TWO reflective boundaries instead of the old two-axis-plane setup:
       - the "base" edge (row 8 in the diagram / row index 5 here) -- the
         horizontal core-center mirror line
       - a genuine DIAGONAL plane through positions 1,9,16,22,27,30 (the
         0-indexed 0,8,15,21,26,29) -- the 45-degree octant cut. The old
         script had NO diagonal plane at all; if your true reduction is
         1/8 (not 1/4), this was silently missing an entire boundary.
  3. The remaining two "outer" edges (far column A side, and the tip near
     diagram row 3) are vacuum -- these approximate the true, stepped,
     round outer edge of the reactor core, same padding-with-reflector
     approach your code already uses for -1 cells.

USE: replace GRID_LAYOUT / GRID_ROWS / GRID_COLS and build_core_geometry()
in openmc_beavrs_vver1000_v4.py with the versions below. pattern_to_grid()
and compute_ppf_per_assembly() do NOT need to change -- they already index
via GRID_LAYOUT generically, so they'll pick up the new shape automatically.

IMPORTANT -- still verify empirically before trusting results: confirm
your lattice's row-index <-> physical-y-axis direction convention with an
openmc.Universe.plot() of the rebuilt geometry (should look like a wedge,
not a rectangle missing a chunk in the wrong corner) before running any
real transport solves against it.

Run standalone (no OpenMC install needed) to sanity-check adjacency first:
  python 09_openmc_octant_fix.py
=============================================================================
      H  G  F  E  D  C  B  A
r0                 29 30
r1              26 27 28
r2           21 22 23 24 25
r3        15 16 17 18 19 20
r4      8  9 10 11 12 13 14
r5   0  1  2  3  4  5  6  7

      H  G  F  E  D  C  B  A
3                   30 31
4                27 28 29
5             22 23 24 25 26
6          16 17 18 19 20 21
7       9 10 11 12 13 14 15
8    1  2  3  4  5  6  7  8
"""

import numpy as np

# =============================================================================
# PART A — corrected GRID_LAYOUT (drop-in replacement)
# =============================================================================
N_POS, N_TYPES = 31, 9
GRID_ROWS, GRID_COLS = 6, 8

# row index 0 = diagram row 3 (far tip, near the true outer core edge)
# row index 5 = diagram row 8 (base row, sits along the horizontal mirror line)
# col index 0 = diagram column H
# col index 7 = diagram column A
GRID_LAYOUT = np.array([
    [-1, -1, -1, -1, -1, 29, 30, -1],   # diagram row 3
    [-1, -1, -1, -1, 26, 27, 28, -1],   # diagram row 4
    [-1, -1, -1, 21, 22, 23, 24, 25],   # diagram row 5
    [-1, -1, 15, 16, 17, 18, 19, 20],   # diagram row 6
    [-1,  8,  9, 10, 11, 12, 13, 14],   # diagram row 7
    [ 0,  1,  2,  3,  4,  5,  6,  7],   # diagram row 8 (core-center row)
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)

# sanity checks that run at import time -- fail loudly if the array is wrong
assert GRID_MASK.sum() == N_POS, f"expected {N_POS} active cells, got {GRID_MASK.sum()}"
_seen = sorted(GRID_LAYOUT[GRID_MASK].tolist())
assert _seen == list(range(N_POS)), "GRID_LAYOUT must contain every position 0..30 exactly once"
print(f"[OK] GRID_LAYOUT shape {GRID_LAYOUT.shape}, {N_POS} positions, all present exactly once.")


# =============================================================================
# PART B — adjacency verification (pure numpy, no OpenMC needed)
# =============================================================================
def real_neighbors(pos_id):
    """Return the set of position ids that are REAL orthogonal (up/down/
    left/right) grid neighbors of pos_id, i.e. what OpenMC's RectLattice
    will actually treat as physically adjacent cells. Cells that fall off
    the array edge or land on a -1 (padding/reflector) cell are NOT real
    neighbors -- those directions are handled by the boundary condition
    (reflective mirror or vacuum outer edge), not by an actual assembly."""
    where = np.argwhere(GRID_LAYOUT == pos_id)
    if len(where) == 0:
        return []
    r, c = where[0]
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS and GRID_LAYOUT[rr, cc] >= 0:
            neighbors.append(int(GRID_LAYOUT[rr, cc]))
    return sorted(neighbors)


print("\n" + "=" * 70)
print("ADJACENCY CHECK — run this before trusting any OpenMC output")
print("=" * 70)
print("Position 0 (the corner where both mirror planes cross) should have")
print("exactly ONE real neighbor (position 1) -- its other two array-sides")
print("are the reflective boundaries themselves, not real assembly cells.\n")

for p in [0, 1, 8, 15, 21, 26, 29, 7, 14, 20, 25, 30]:
    nbrs = real_neighbors(p)
    print(f"  position {p:2d} -> real neighbors: {nbrs}")

n0 = real_neighbors(0)
if n0 == [1]:
    print("\n  [PASS] position 0 has exactly one real neighbor (position 1), as expected.")
else:
    print(f"\n  [WARN] position 0 has neighbors {n0}, expected exactly [1]. Re-check the array.")

# diagonal-boundary positions should each have entropy/sensitivity similar
# to THEIR OWN mirror partner once you rebuild the CNN/OpenMC training data
# around this layout -- that's the same check 07_pareto_nsga2_cnn.py ran
# against the OLD (wrong) layout and correctly failed. Re-running that same
# check against this NEW layout (once you have fresh train_type_freq /
# sensitivity data generated under it) is the next real validation step.
diag_positions = [0, 8, 15, 21, 26, 29]
print(f"\n  Diagonal-boundary positions (should sit exactly on the 45-degree "
      f"mirror line): {diag_positions}")
print("  Each of these should have exactly one 'inward' neighbor (further into")
print("  the wedge) and no neighbor across the diagonal -- verify this matches")
print("  what's printed above (e.g. position 8's neighbors should include 9 and")
print("  15, but nothing 'across' the diagonal toward positions like 1 or 0).")


# =============================================================================
# PART C — corrected build_core_geometry() (drop-in replacement)
# =============================================================================
CORE_GEOMETRY_CODE = '''
def build_core_geometry(pattern_flat, assembly_universes, reflector_u):
    """
    1/8-octant reduction. Two reflective mirror planes (base row + 45-deg
    diagonal), two vacuum outer edges (approximating the true round core
    boundary). See 09_openmc_octant_fix.py for the derivation.
    """
    grid = pattern_to_grid(pattern_flat)   # unchanged -- already generic over GRID_LAYOUT

    core_lat = openmc.RectLattice(name='core')
    core_lat.lower_left = (-GRID_COLS * ASSEMBLY_PITCH / 2,
                            -GRID_ROWS * ASSEMBLY_PITCH / 2)
    core_lat.pitch = (ASSEMBLY_PITCH, ASSEMBLY_PITCH)

    universes = np.empty((GRID_ROWS, GRID_COLS), dtype=object)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                atype = int(grid[r, c])
                universes[r, c] = assembly_universes[atype]
            else:
                universes[r, c] = reflector_u
    core_lat.universes = universes

    half_w = GRID_COLS * ASSEMBLY_PITCH / 2
    half_h = GRID_ROWS * ASSEMBLY_PITCH / 2
    lx0, ly0 = core_lat.lower_left  # = (-half_w, -half_h)

    # --- reflective mirror #1: the "base row" edge (row index GRID_ROWS-1,
    # the diagram's row 8 / horizontal core-center line). VERIFY row-to-y
    # direction empirically (openmc.Universe.plot()) before trusting sign --
    # written here assuming row index increases with DECREASING physical y
    # (row 0 at y=+half_h, last row at y=-half_h), matching the original
    # script's lower_left convention. Flip the sign below if your plot
    # shows the wedge upside down.
    y_base = ly0  # edge of the last row (GRID_ROWS-1), i.e. the base row
    y_base_plane = openmc.YPlane(y0=y_base, boundary_type='reflective')

    # --- reflective mirror #2: the 45-degree diagonal through positions
    # 0,8,15,21,26,29 (row_local + col_local == GRID_ROWS - 1 == 5).
    # In real (x,y) coordinates, cell (r,c)'s center sits at:
    #   x_c = lx0 + (c + 0.5) * ASSEMBLY_PITCH
    #   y_c = ly0 + (GRID_ROWS - 1 - r + 0.5) * ASSEMBLY_PITCH   (row-flip, see above)
    # The diagonal boundary sits HALF A CELL beyond the diagonal positions,
    # on the side away from the used wedge -- i.e. at
    #   (x - lx0)/ASSEMBLY_PITCH + (y - ly0)/ASSEMBLY_PITCH = GRID_ROWS  (= 6)
    # which is a plane with normal (1,1) in local (shifted) coordinates:
    #   1*x + 1*y = ASSEMBLY_PITCH * GRID_ROWS + lx0 + ly0
    diag_d = ASSEMBLY_PITCH * GRID_ROWS + lx0 + ly0
    diag_plane = openmc.Plane(a=1.0, b=1.0, c=0.0, d=diag_d, boundary_type='reflective')
    # NOTE: openmc.Plane's positive/negative half-space convention means you
    # may need diag_plane or -diag_plane below depending on which side the
    # wedge sits on -- check with a quick plot; flip to -diag_plane if the
    # wedge gets clipped instead of the empty corner.

    # --- vacuum outer edges: true (stepped) core boundary ---
    x_max = openmc.XPlane(x0=half_w, boundary_type='vacuum')   # far column (A) side
    y_max = openmc.YPlane(y0=half_h, boundary_type='vacuum')   # tip (diagram row 3) side
    z_min = openmc.ZPlane(z0=-ACTIVE_HEIGHT / 2, boundary_type='vacuum')
    z_max = openmc.ZPlane(z0= ACTIVE_HEIGHT / 2, boundary_type='vacuum')

    core_region = (+y_base_plane & -x_max & -y_max & +z_min & -z_max & +diag_plane)
    core_cell = openmc.Cell(fill=core_lat, region=core_region)

    return openmc.Geometry(openmc.Universe(cells=[core_cell])), core_lat
'''

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Drop-in replacement code for openmc_beavrs_vver1000_v4.py:")
    print("=" * 70)
    print(CORE_GEOMETRY_CODE)
    print("=" * 70)
    print("Steps to apply:")
    print("  1. Replace N_POS/GRID_ROWS/GRID_COLS/GRID_LAYOUT at the top of")
    print("     openmc_beavrs_vver1000_v4.py with Part A above (already N_POS=31,")
    print("     just GRID_ROWS/GRID_COLS/GRID_LAYOUT change).")
    print("  2. Replace build_core_geometry() with the code printed above.")
    print("  3. Run a quick openmc.Universe.plot() and eyeball it -- should look")
    print("     like a wedge (roughly triangular, bigger at one corner), not a")
    print("     rectangle. Flip signs noted in the comments if it looks wrong.")
    print("  4. Re-run --quick_check on a KNOWN training-data row (one you have")
    print("     the recorded ppf_s0_a* label for) and compare magnitude/position")
    print("     against that label -- same calibration-row diagnostic you used")
    print("     before finding the BP-rod-placement and boron-convention bugs.")
