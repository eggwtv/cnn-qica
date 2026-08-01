"""
beavrs_geometry.py
=======================================================================
Single source of truth for the corrected 6x8 octant GRID_LAYOUT, shared
by data_utils.py, cnn_v10_oracle.py, workflow_W7_qpso_cnn.py, and
workflow_W4_active_learning_loop.py -- so there is exactly ONE place
this array is defined instead of three copies silently drifting apart.

Copied verbatim from cnn_v10_octant_retrain.py / openmc_beavrs_vver1000_v5.py
(both already agree on this exact array).
"""

import numpy as np

N_POS, N_TYPES = 31, 9
GRID_ROWS, GRID_COLS = 6, 8

GRID_LAYOUT = np.array([
    [-1, -1, -1, -1, -1, 29, 30, -1],   # diagram row 3
    [-1, -1, -1, -1, 26, 27, 28, -1],   # diagram row 4
    [-1, -1, -1, 21, 22, 23, 24, 25],   # diagram row 5
    [-1, -1, 15, 16, 17, 18, 19, 20],   # diagram row 6
    [-1,  8,  9, 10, 11, 12, 13, 14],   # diagram row 7
    [ 0,  1,  2,  3,  4,  5,  6,  7],   # diagram row 8 (core-center row)
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)

assert GRID_MASK.sum() == N_POS
assert sorted(GRID_LAYOUT[GRID_MASK].tolist()) == list(range(N_POS))


def flat_to_grid(flat, grid_rows=GRID_ROWS, grid_cols=GRID_COLS, grid_layout=GRID_LAYOUT):
    """(N, 31) flat loading pattern -> (N, 6, 8) grid, matching cnn_v10's
    exact reshape convention. flat[:, GRID_LAYOUT[r,c]] goes into grid
    cell (r,c) -- i.e. flat index IS the real position number, identical
    to loading_0..loading_30 in ml_dataset_constrained.csv."""
    Xg = np.zeros((len(flat), grid_rows, grid_cols), dtype=np.int32)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid_layout[r, c] >= 0:
                Xg[:, r, c] = flat[:, grid_layout[r, c]]
    return Xg
