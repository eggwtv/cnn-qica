"""
=============================================================================
rep_cnn_v2.py  —  BEAVRS CNN Surrogate  |  PWR Fuel Loading Pattern  |  PPF
=============================================================================

OVERVIEW
────────
Predict three physics outputs for any proposed fuel loading pattern in a
2-batch Westinghouse 4-loop PWR (BEAVRS benchmark benchmark):

  PRIMARY   → ppf_max    (Power Peaking Factor, global cycle maximum)
  SECONDARY → cycle_length  (effective full power days in cycle)
  BONUS     → keff_boc   (beginning-of-cycle criticality, from react_0)

DATASET
───────
  ml_dataset_constrained.csv  —  10,000 BEAVRS 1/8-core loading patterns
  cycle_length_summary.xlsx   —  monocore EFPD equivalents per assembly type
                                  used to give the embedding meaningful
                                  physical initialisation

PPF GOAL RANGE: 2.0 – 4.5
───────────────────────────
  The BEAVRS dataset PPF statistics are:
      min  ~1.61  |  10th pct ~2.02  |  mean ~2.76  |  90th pct ~3.87  |  max ~7.9

  IMPORTANT — this code does NOT chase an artificial target of 1.7.
  The dataset minimum is ~1.6, so trying to train the model to predict
  values it has never seen would introduce bias.  Instead:
    • The model is trained with PURE MSE — predict what the data says.
    • The "goal range" 2.0–4.5 is used only for REPORTING and FILTERING:
      at evaluation time we show how well the model predicts in that zone.
    • QICA (next step) will use the CNN as a fitness oracle and search
      for patterns with ppf_max closer to 2.0.

WHY CNN INSTEAD OF ANN?
────────────────────────
  The previous ANN treated the 31 assembly positions as a flat vector.
  There was no concept of "position 3 is adjacent to position 8".

  A CNN treats the loading pattern as a SPATIAL IMAGE:
    • Assemblies that are physically close share filter responses.
    • PPF is fundamentally a spatial problem: it peaks wherever adjacent
      fresh (high-enrichment) assemblies reinforce each other's neutron flux.
    • The residual + attention architecture lets the model learn WHICH
      spatial locations matter most for PPF — this is also our sensitivity.

WHAT IS A 2-BATCH SYSTEM?
───────────────────────────
  "Batch" = a cohort of assemblies loaded at one refuelling outage.

  2-BATCH (this model):
    After start-up every refuelling has:
      50% fresh fuel (Batch A)  +  50% once-irradiated (Batch B from cycle n-1)
    → Two distinct fuel types in-core simultaneously.
    → Simpler permutation space.  Typical cycle ≈ 17–18 months (~520 EFPD).

  3-BATCH (current industry standard):
    1/3 fresh  +  1/3 once-burnt  +  1/3 twice-burnt per reload.
    Fuel stays in for 3 cycles → more energy → longer cycles (~24 months).
    More variables, harder to optimise → deferred to future work.

  In our BEAVRS dataset (2-batch):
    Assembly types 1–4  → once-irradiated (Batch B, lower monocore EFPD)
    Assembly types 5–9  → fresh (Batch A, higher monocore EFPD)
    The optimisation question: WHICH position gets which type.

BEAVRS GEOMETRY — 31 POSITIONS (1/8-core symmetry)
────────────────────────────────────────────────────
  BEAVRS is a Westinghouse 4-loop PWR with 193 fuel assemblies in a 17×17 grid.
  Exploiting the 8-fold reflective symmetry of a square cross-section core
  reduces this to 31 unique positions.  (If you used 1/4-symmetry you would
  have ~49 unique positions — that is the "49 assemblies" number you may have
  seen.  Our dataset is built on 1/8-symmetry, so 31.)

  We map the 31 positions onto a 6×6 CNN input grid:
    Positions 0-30 → grid cells (6×6 = 36; 5 cells are padding = reflector)

  A GEOMETRY-PRESERVING MASK (GRID_MASK) marks the 5 padding cells.
  The CNN sees 0 (reflector type) at those positions, and the sensitivity
  analysis treats them as inactive.  The loss is NOT computed on masked cells.

ARCHITECTURE OVERVIEW
──────────────────────
  Input: (B, 6, 6)  integer assembly types  0=reflector, 1–9=fuel
    ↓ Embedding(10→16)          each type → 16-dim physical vector
    ↓ 3 × ConvResBlock (32→64→128 filters, 3×3, same-padding, GELU, BN)
    ↓ Spatial attention gate    (1×1 conv, sigmoid)  — which cells matter
    ↓ GlobalAveragePooling2D
    ↓ Dense(128, GELU) + Dropout
    ↓ Dense(64, GELU)  + Dropout
    ↓ Three separate heads:
        ppf_head    → Dense(1)  → ppf_max
        cycle_head  → Dense(1)  → cycle_length
        keff_head   → Dense(1)  → keff_boc
        step_head   → Dense(31) → ppf at each burnup step  (for QICA profile)

ACTIVE LEARNING LOOP (SCAFFOLD)
─────────────────────────────────
  After training, the code enters an ACTIVE LEARNING loop scaffold:
    1. CNN predicts ppf_max and MC-Dropout uncertainty for all patterns.
    2. Uncertainty sampling: flag high-σ patterns in the goal range.
    3. [ PLACEHOLDER ] — in production, query a simulator for those patterns.
    4. Retrain on expanded dataset.
  No simulator is called here — the loop is ready to plug one in later.
  QICA will ultimately replace step 3 with an optimizer query.

OUTPUT FILES:
  cnn_beavrs_v2_model.keras  — trained model weights
  cnn_beavrs_v2_config.json  — geometry, scalers, sensitivity (for QICA)
  cnn_beavrs_v2_sens.csv     — ∂ppf_max/∂position (QICA assimilation weights)
  cnn_beavrs_v2_results.png  — evaluation + spatial sensitivity plots
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print(f"Running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print("rep_cnn_v2.py — BEAVRS CNN  |  Pure Prediction  |  PPF goal range 2.0–4.5\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── File paths ────────────────────────────────────────────────────────────────
BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'   # monocore EFPD per assembly type
MODEL_NAME  = 'cnn_beavrs_v2_model.keras'
CONFIG_NAME = 'cnn_beavrs_v2_config.json'
SENS_NAME   = 'cnn_beavrs_v2_sens.csv'
PLOT_NAME   = 'cnn_beavrs_v2_results.png'

# ── PPF reporting range (aligned with actual dataset — NOT a loss target) ─────
# These values are used ONLY for evaluation reports and filter plots.
# The CNN is trained with pure MSE — no artificial constraints below data range.
PPF_GOAL_LOW  = 2.0   # lower bound of the "good zone" for QICA to search
PPF_GOAL_HIGH = 4.5   # upper bound — patterns above this are "too peaked"

# ── BEAVRS core geometry ──────────────────────────────────────────────────────
N_POS    = 31   # unique positions in 1/8-core symmetry octant
N_TYPES  = 9    # assembly types 1–9  (0 = reflector/mask)
N_STEPS  = 31   # burnup time-steps with PPF data

# ── 6×6 spatial grid layout ───────────────────────────────────────────────────
# 31 positions mapped row-major from inner (highest flux) to outer.
# 5 slots are padding (value -1) → these become type 0 (reflector) in the CNN.
# This is the GEOMETRY-PRESERVING MASK — the CNN always sees the correct
# spatial relationships between positions.
GRID_ROWS   = 6
GRID_COLS   = 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],   # row 0 — near-centre, highest mean BOC PPF
    [ 6,  7,  8,  9, 10, 11],   # row 1
    [12, 13, 14, 15, 16, 17],   # row 2
    [18, 19, 20, 21, 22, 23],   # row 3
    [24, 25, 26, 27, 28, 29],   # row 4
    [30, -1, -1, -1, -1, -1],   # row 5 — peripheral position + reflector padding
], dtype=np.int32)

# Boolean mask: True where the cell holds an active fuel position
# False (5 cells) = reflector / inactive padding
GRID_MASK = (GRID_LAYOUT >= 0)   # (6, 6) — used in sensitivity + loss

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 300    # with early stopping; typical actual epochs ~100–200
LR         = 0.001
DROPOUT    = 0.15
TEST_FRAC  = 0.15
VAL_FRAC   = 0.15
SEED       = 42

# ── MC Dropout ────────────────────────────────────────────────────────────────
MC_SAMPLES = 30   # stochastic forward passes for uncertainty estimation

# ── Active learning ───────────────────────────────────────────────────────────
AL_UNCERTAINTY_THRESHOLD = 0.08   # σ_ppf above which a pattern is "query-worthy"
AL_MAX_QUERIES           = 50     # how many patterns to flag per AL round


# =============================================================================
# SECTION 2 — LOAD MONOCORE CYCLE LENGTHS FROM XLSX
# =============================================================================
# The cycle_length_summary.xlsx contains one row per assembly type with the
# monocore EFPD (effective full power days if only that assembly type is loaded).
# We use this to build a physically meaningful encoding dictionary.
# Assembly type fa_id matches loading columns 1–9 in the CSV.

print("[XLSX] Loading monocore cycle lengths from cycle_length_summary.xlsx ...")

if os.path.exists(XL_FILE):
    xl_df = pd.read_excel(XL_FILE, sheet_name='Cycle_Lengths')
    # Columns: fa_id, fa_name, monocore_cycle_length, EFPD
    # fa_id 1–9 correspond to assembly types 1–9 in loading columns
    monocore_map = dict(zip(xl_df['fa_id'].astype(int),
                            xl_df['monocore_cycle_length'].astype(float)))
    print(f"  Loaded {len(monocore_map)} assembly type entries from xlsx:")
    for fid, cyc in sorted(monocore_map.items()):
        fa_name = xl_df.loc[xl_df['fa_id'] == fid, 'fa_name'].values[0]
        print(f"    Type {fid}  ({fa_name:<18s})  monocore EFPD = {cyc:.1f}")
else:
    # Fallback hardcoded from the xlsx if file not found
    print(f"  [WARN] {XL_FILE} not found — using hardcoded monocore values from xlsx.")
    monocore_map = {
        1: 172.9,   # FA_16_NOBA   — lowest energy, once-burnt
        2: 366.9,   # FA_24_NOBA
        3: 323.2,   # FA_24_12BA_C1
        4: 299.8,   # FA_24_16BA
        5: 519.9,   # FA_31_NOBA   — highest energy, fresh fuel
        6: 504.9,   # FA_31_6BA
        7: 475.3,   # FA_32_15BA
        8: 471.6,   # FA_31_16BA
        9: 454.7,   # FA_31_20BA
    }

# Assembly encoding dictionary for config export:
#   0 = reflector (no fuel)  → 0.0 EFPD equivalent
ASSEMBLY_CYCLE_EQUIV = {0: 0.0}
ASSEMBLY_CYCLE_EQUIV.update(monocore_map)

print()


# =============================================================================
# SECTION 3 — LOAD DATA + DATASET PPF ANALYSIS
# =============================================================================

print("[DATA] Loading BEAVRS dataset ...")

if not os.path.exists(BEAVRS_CSV):
    print(f"[ERROR] {BEAVRS_CSV} not found.")
    sys.exit(1)

# The CSV has a very wide header; skiprows=1 skips the dataset name row
df = pd.read_csv(BEAVRS_CSV, skiprows=1, engine='python', on_bad_lines='skip')
print(f"  Loaded {len(df)} patterns × {df.shape[1]} columns\n")

# ── Identify column groups ────────────────────────────────────────────────────
load_cols  = [f'loading_{i}' for i in range(N_POS)]
react_cols = sorted([c for c in df.columns if c.startswith('react_')],
                    key=lambda c: int(c.split('_')[1]))

ppf_steps   = sorted(set(int(c.split('_')[1][1:]) for c in df.columns
                          if c.startswith('ppf_')))
ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df.columns
                          if c.startswith('ppf_')))

# Per-step maximum PPF across all assemblies (N, 31)
step_max_ppf = np.stack([
    df[[f'ppf_s{s}_a{i}' for i in ppf_assembs
        if f'ppf_s{s}_a{i}' in df.columns]].values.astype(np.float32).max(axis=1)
    for s in ppf_steps
], axis=1)

ppf_global_max = step_max_ppf.max(axis=1)   # (N,) — cycle-maximum PPF
ppf_boc        = step_max_ppf[:, 0]          # (N,) — BOC (step 0) PPF

# keff_boc: derived from react_0 (reactivity at step 0)
# Reactivity ρ = (keff - 1) / keff  →  keff = 1 / (1 - ρ)
# react_0 values in the CSV are dimensionless ρ
keff_boc = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)

# ── DATASET PPF ANALYSIS — what does the data actually contain? ───────────────
# CRITICAL: The PPF goal must be aligned with the dataset distribution.
# Training a model to predict values outside the data range introduces bias.
# We print this analysis so you always know what "good" means for YOUR dataset.
print("=" * 58)
print("DATASET PPF ANALYSIS  (align your goal with these numbers)")
print("=" * 58)
print(f"  Patterns           : {len(df)}")
print(f"  PPF_max range      : {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  PPF_max mean       : {ppf_global_max.mean():.3f}  (average pattern in dataset)")
print(f"  PPF_max median     : {np.median(ppf_global_max):.3f}")
print(f"  PPF_max std        : {ppf_global_max.std():.3f}")
print(f"  10th percentile    : {np.percentile(ppf_global_max,10):.3f}  ← good patterns")
print(f"  25th percentile    : {np.percentile(ppf_global_max,25):.3f}")
print(f"  Cycle length range : {df.cycle_length.min():.1f} – {df.cycle_length.max():.1f} days")
print()
print(f"  Patterns in goal range {PPF_GOAL_LOW}–{PPF_GOAL_HIGH}:")
in_goal = ((ppf_global_max >= PPF_GOAL_LOW) & (ppf_global_max <= PPF_GOAL_HIGH)).sum()
print(f"    Count  = {in_goal}  ({in_goal/len(df)*100:.1f}% of dataset)")
print(f"  Patterns with ppf_max < {PPF_GOAL_LOW} : {(ppf_global_max<PPF_GOAL_LOW).sum()}")
print(f"  Patterns with ppf_max < 2.5  : {(ppf_global_max<2.5).sum()}")
print(f"  BOC is cycle-max in {(step_max_ppf.argmax(axis=1)==0).mean()*100:.1f}% of cases")
print()
print(f"  keff_boc range     : {keff_boc.min():.4f} – {keff_boc.max():.4f}")
print(f"  keff_boc mean      : {keff_boc.mean():.4f}")
print("=" * 58)
print()
print("  ► CNN is trained with PURE MSE — no soft constraints below data range.")
print("  ► QICA will use this CNN to SEARCH for patterns towards ppf_max ~ 2.0.")
print()


# =============================================================================
# SECTION 4 — TARGETS AND FEATURES
# =============================================================================
#
# Outputs the CNN predicts (all in a single multi-output head):
#   Index 0      : ppf_max       — global cycle maximum PPF    ← PRIMARY
#   Index 1      : ppf_boc       — BOC maximum PPF             ← proxy for above
#   Indices 2-32 : ppf_steps[0..30] — max PPF at each burnup step (for QICA burn profile)
#   Index 33     : cycle_length  — operating days
#   Index 34     : keff_boc      — BOC criticality             ← added v2
#
# Total outputs: 1 + 1 + 31 + 1 + 1 = 35

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_keff      = keff_boc.reshape(-1, 1).astype(np.float32)

Y_combined = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle, Y_keff], axis=1)

IDX_PPF_MAX    = 0
IDX_PPF_BOC    = 1
IDX_PPF_STEPS  = slice(2, 2 + N_STEPS)
IDX_CYCLE      = 2 + N_STEPS          # = 33
IDX_KEFF       = 2 + N_STEPS + 1      # = 34
N_OUTPUTS      = Y_combined.shape[1]  # = 35

print(f"[TARGETS]  Shape: {Y_combined.shape}")
print(f"  Idx 0     : ppf_max (PRIMARY)")
print(f"  Idx 1     : ppf_boc")
print(f"  Idx 2-32  : ppf at each burnup step")
print(f"  Idx 33    : cycle_length")
print(f"  Idx 34    : keff_boc  (NEW in v2)\n")


# =============================================================================
# SECTION 5 — 2D SPATIAL INPUT ENCODING + GEOMETRY-PRESERVING MASK
# =============================================================================
# Each loading pattern is a (31,) vector of integer types (1–9).
# We reshape to (6, 6) using GRID_LAYOUT so the CNN sees the core geometry.
#
# GEOMETRY-PRESERVING MASK:
#   The 5 padding positions (GRID_LAYOUT == -1) always receive type 0.
#   This is NOT learned — it is a hard geometric constraint that tells the CNN
#   "these 5 cells are always reflector; ignore them in spatial comparisons."
#   The CNN will learn to zero-out attention weights for those cells automatically,
#   and we verify this in the sensitivity map (Section 11).

load_int = df[load_cols].values.astype(np.int32)   # (N, 31), values 1–9


def make_grid_input(load_1d: np.ndarray) -> np.ndarray:
    """
    Convert (N, 31) integer loading array → (N, 6, 6) spatial grid.

    Active positions are filled from load_1d.
    Masked positions (GRID_LAYOUT < 0) are always set to 0 (reflector type).

    This is the geometry-preserving mask: the CNN always sees the correct
    physical neighbourhood of every active assembly.
    """
    N    = load_1d.shape[0]
    grid = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)

    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            pos = GRID_LAYOUT[i, j]
            grid[:, i, j] = load_1d[:, pos] if pos >= 0 else 0

    return grid


X_grid = make_grid_input(load_int)   # (N, 6, 6)

print(f"[INPUT]  Grid shape: {X_grid.shape}")
print(f"  Active fuel cells : {GRID_MASK.sum()}/36  (5 are reflector padding)")
active_vals = X_grid[:, GRID_MASK]
print(f"  Type range        : {active_vals.min()} – "
      f"{active_vals.max()}")
print(f"  Example pattern 0 :\n{X_grid[0]}\n")


# =============================================================================
# SECTION 6 — TRAIN / VALIDATION / TEST SPLIT + SCALING
# =============================================================================

# Stratified split on PPF quantile ensures low-PPF (rare) patterns appear in
# all three sets — crucial so the model sees good patterns during training
ppf_bins = pd.qcut(ppf_global_max, q=10, labels=False)

X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X_grid, Y_combined,
    test_size=TEST_FRAC,
    stratify=ppf_bins,
    random_state=SEED
)

ppf_bins_tv = pd.qcut(Y_trainval[:, IDX_PPF_MAX], q=10, labels=False)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval,
    test_size=VAL_FRAC / (1 - TEST_FRAC),
    stratify=ppf_bins_tv,
    random_state=SEED
)

print(f"[SPLIT]")
print(f"  Train : {len(X_train):5d}  ({len(X_train)/len(X_grid)*100:.0f}%)")
print(f"  Val   : {len(X_val):5d}  ({len(X_val)/len(X_grid)*100:.0f}%)")
print(f"  Test  : {len(X_test):5d}  ({len(X_test)/len(X_grid)*100:.0f}%)\n")

# StandardScaler on outputs — fit ONLY on train, never on val/test (no leakage)
y_scaler     = StandardScaler()
Y_train_sc   = y_scaler.fit_transform(Y_train).astype(np.float32)
Y_val_sc     = y_scaler.transform(Y_val).astype(np.float32)
Y_test_sc    = y_scaler.transform(Y_test).astype(np.float32)

print(f"[SCALING]")
print(f"  ppf_max    mean={y_scaler.mean_[IDX_PPF_MAX]:.3f}  std={y_scaler.scale_[IDX_PPF_MAX]:.3f}")
print(f"  cycle_len  mean={y_scaler.mean_[IDX_CYCLE]:.1f}   std={y_scaler.scale_[IDX_CYCLE]:.2f}")
print(f"  keff_boc   mean={y_scaler.mean_[IDX_KEFF]:.4f}  std={y_scaler.scale_[IDX_KEFF]:.4f}\n")

# Precompute the scaled position of PPF_GOAL_LOW for use in reporting
PPF_GOAL_LOW_SCALED = float(
    (PPF_GOAL_LOW - y_scaler.mean_[IDX_PPF_MAX]) / y_scaler.scale_[IDX_PPF_MAX]
)


# =============================================================================
# SECTION 7 — CNN MODEL ARCHITECTURE
# =============================================================================

class ConvResBlock(layers.Layer):
    """
    Residual convolutional block: two 3×3 Conv2D layers with skip connection.

    WHY RESIDUAL CONNECTIONS?
    ─────────────────────────
    In a deep network, gradients can vanish during backpropagation.
    Skip connections let gradients flow directly from the output back to the
    input of this block, enabling training of deeper networks without degradation.
    When the identity mapping is already near-optimal, the block can learn to
    output near-zero (so the skip carries the signal) — very stable.

    For the PPF problem:
    Early blocks learn local patterns (e.g. "two adjacent type-9 assemblies").
    Later blocks combine these into global patterns (e.g. "checkerboard of
    fresh/burnt gives low PPF").  The skip lets useful early features persist.

    Args:
        filters     : output channels
        kernel_size : 3 × 3 by default (local 3-cell neighbourhood in the core)
        dropout     : Dropout rate applied after the block
    """

    def __init__(self, filters: int, kernel_size: int = 3,
                 dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        kw = dict(padding='same', kernel_initializer='he_uniform')
        self.conv1 = layers.Conv2D(filters, kernel_size, **kw)
        self.conv2 = layers.Conv2D(filters, kernel_size, **kw)
        self.bn1   = layers.BatchNormalization()
        self.bn2   = layers.BatchNormalization()
        self.drop  = layers.Dropout(dropout) if dropout > 0 else None
        self._filters     = filters
        self._projection  = None   # built lazily

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self._projection = layers.Conv2D(
                self._filters, 1, padding='same',
                kernel_initializer='he_uniform', name=self.name + '_proj'
            )
        super().build(input_shape)

    def call(self, x, training=False):
        skip = x if self._projection is None else self._projection(x)
        h    = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        h    = tf.nn.gelu(self.bn2(self.conv2(h), training=training))
        if self.drop is not None:
            h = self.drop(h, training=training)
        return h + skip   # residual addition


def build_cnn(
    grid_rows: int   = GRID_ROWS,
    grid_cols: int   = GRID_COLS,
    n_types: int     = N_TYPES + 1,   # 0 (reflector) + 1..9 → 10 values
    embed_dim: int   = 16,
    filters: tuple   = (32, 64, 128),
    dense_units: int = 128,
    dropout: float   = DROPOUT,
    n_outputs: int   = N_OUTPUTS
) -> keras.Model:
    """
    Build the spatial CNN surrogate for BEAVRS PPF/cycle/keff prediction.

    Embedding layer design:
    ─────────────────────────────────────────────────────────────────────
    We initialise the embedding with the MONOCORE EFPD values from the xlsx
    file.  This gives the model a physically meaningful starting point:
      type 5 (FA_31_NOBA, 519.9 EFPD) starts at a higher initial embedding
      value than type 1 (FA_16_NOBA, 172.9 EFPD) — the model already "knows"
      type 5 carries more energy before training begins.
    The embeddings are still freely learned end-to-end; the initialisation
    just provides a useful warm start.

    Spatial attention gate:
    ─────────────────────────────────────────────────────────────────────
    After the three ResBlocks a 1×1 Conv(sigmoid) generates per-cell weights.
    This is an explicit way for the CNN to learn "the centre cells matter more
    for PPF than the periphery" — which aligns with physics (inner assemblies
    contribute most to the flux peak).  The weights are logged in the
    sensitivity analysis (Section 11).

    Args:
        grid_rows, grid_cols : spatial dimensions (6, 6)
        n_types  : embedding vocabulary size (10: types 0–9)
        embed_dim: embedding vector dimension
        filters  : filter counts for each of the 3 ResBlocks
        dense_units: MLP head width
        dropout  : Dropout rate (active at train AND MC inference)
        n_outputs: total outputs (35 in v2)

    Returns:
        keras.Model (compiled)
    """

    # ── Build monocore-informed embedding initialisation ──────────────────────
    # Shape: (n_types, embed_dim)
    # We set dimension 0 of each type's embedding to the normalised EFPD value.
    # All other dimensions start at zero (model will learn them).
    efpd_vals = np.array([ASSEMBLY_CYCLE_EQUIV.get(t, 0.0) for t in range(n_types)])
    efpd_norm = efpd_vals / (efpd_vals.max() + 1e-8)   # normalise to [0, 1]
    emb_init  = np.zeros((n_types, embed_dim), dtype=np.float32)
    emb_init[:, 0] = efpd_norm   # channel 0 = EFPD fingerprint
    emb_initializer = tf.keras.initializers.Constant(emb_init)

    inp = keras.Input(shape=(grid_rows, grid_cols), name='loading_grid', dtype=tf.int32)

    # ── Embedding ──────────────────────────────────────────────────────────────
    x = layers.Embedding(
        input_dim=n_types,
        output_dim=embed_dim,
        embeddings_initializer=emb_initializer,
        name='assembly_embedding'
    )(inp)   # (B, 6, 6, 16)

    # ── Convolutional backbone ─────────────────────────────────────────────────
    # Same-padding keeps spatial size at 6×6 — no downsampling (grid is tiny).
    # filter count doubles per block: 32 → 64 → 128.
    for i, f in enumerate(filters):
        x = ConvResBlock(
            f, kernel_size=3,
            dropout=dropout if i > 0 else 0.0,   # no dropout in first block
            name=f'conv_block_{i+1}'
        )(x)   # (B, 6, 6, f)

    # ── Spatial attention gate ────────────────────────────────────────────────
    # 1×1 conv (one filter, sigmoid) → per-cell weight ∈ (0, 1).
    # Geometry-preserving: the masked cells (reflector) will learn near-zero
    # attention because there is never any signal at those positions.
    attn = layers.Conv2D(1, 1, activation='sigmoid',
                          name='spatial_attention')(x)   # (B, 6, 6, 1)
    x = layers.Multiply(name='attended_features')([x, attn])   # (B, 6, 6, 128)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)   # (B, 128)

    # ── MLP head ───────────────────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation='gelu', name='dense_1')(x)
    x = layers.Dropout(dropout, name='dropout_1')(x)
    x = layers.Dense(dense_units // 2, activation='gelu', name='dense_2')(x)
    x = layers.Dropout(dropout * 0.5, name='dropout_2')(x)

    # ── Output ────────────────────────────────────────────────────────────────
    out = layers.Dense(n_outputs, activation='linear', name='predictions')(x)

    return keras.Model(inputs=inp, outputs=out, name='BEAVRS_CNN_v2')


model = build_cnn()
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")


# =============================================================================
# SECTION 8 — PURE PREDICTION LOSS  (no soft constraints)
# =============================================================================
#
# DESIGN DECISION: PURE WEIGHTED MSE
# ────────────────────────────────────
# Previous versions (rep-6 ANN) added a PPF soft-constraint penalty
# (penalise predictions > 1.7).  This was WRONG for two reasons:
#
#   1. The dataset has very few patterns below PPF 2.0.  Training the model
#      to predict below-data values causes it to under-predict PPF for
#      real patterns — exactly the opposite of what we want for safety.
#
#   2. The CNN is a SURROGATE (predictor), not an optimiser.  It should
#      learn the true physics mapping.  The optimiser (QICA) then searches
#      the input space for low-PPF patterns.  Biasing the surrogate
#      corrupts the fitness landscape QICA sees.
#
# WHAT WE KEEP:
#   • Weighted MSE: ppf_max gets 3× weight — it is the primary objective.
#   • Burnup monotonicity penalty (small): PPF physically decreases after BOC
#     due to burnable absorber burn-out and fuel depletion.  This is a
#     genuine physics constraint that IS present in the data.
#   • keff proximity: soft regularisation so keff_boc stays close to 1.0
#     (critical at BOC) — this is physically necessary for a valid cycle.

OUTPUT_WEIGHTS = np.ones(N_OUTPUTS, dtype=np.float32)
OUTPUT_WEIGHTS[IDX_PPF_MAX]   = 3.0    # PRIMARY target
OUTPUT_WEIGHTS[IDX_PPF_BOC]   = 2.0    # nearly equal to ppf_max
OUTPUT_WEIGHTS[IDX_PPF_STEPS] = 0.5    # 31 correlated step values
OUTPUT_WEIGHTS[IDX_CYCLE]     = 1.0    # secondary target
OUTPUT_WEIGHTS[IDX_KEFF]      = 1.5    # keff_boc — added in v2
OUTPUT_WEIGHTS_TF = tf.constant(OUTPUT_WEIGHTS)

# keff should be near 1.0 at BOC for a valid operating cycle.
# Scaled value of keff=1.0:
KEFF_CRITICAL_SCALED = float(
    (1.0 - y_scaler.mean_[IDX_KEFF]) / y_scaler.scale_[IDX_KEFF]
)


def pure_prediction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Physics-informed PURE PREDICTION loss.

    Three terms:
      1. Weighted MSE across all 35 outputs  ← main driver
      2. Late-burnup PPF monotonicity penalty  ← physics constraint in data
      3. keff proximity to critical (1.0)  ← mild regulariser

    NO ppf target soft-constraint.  The CNN maps inputs → outputs faithfully.
    The optimiser (QICA) will search for low-PPF inputs.

    Args:
        y_true : (batch, 35) ground truth in SCALED space
        y_pred : (batch, 35) CNN predictions in SCALED space

    Returns:
        scalar loss value
    """
    # ── 1. Weighted MSE ───────────────────────────────────────────────────────
    # Each target scaled to ~N(0,1) so they contribute on equal footing before
    # the explicit weights are applied.
    sq_diff      = tf.square(y_true - y_pred)
    weighted_mse = tf.reduce_mean(sq_diff * OUTPUT_WEIGHTS_TF)

    # ── 2. Late-burnup PPF monotonicity ───────────────────────────────────────
    # After ~3 burnup steps the PPF profile should be non-increasing.
    # (Burnable absorbers have burned out; flux is flattening with depletion.)
    # Penalise predicted step-to-step INCREASES after step index 3.
    step_preds  = y_pred[:, IDX_PPF_STEPS]            # (batch, 31) scaled
    late        = step_preds[:, 3:]                    # (batch, 28)
    diffs       = late[:, 1:] - late[:, :-1]           # (batch, 27)
    violations  = tf.maximum(0.0, diffs)               # only penalise increases
    mono_penalty = tf.reduce_mean(tf.square(violations))

    # ── 3. keff proximity to critical ─────────────────────────────────────────
    # A valid cycle must start critical (keff ≈ 1.0).  Penalise predictions
    # that stray far from the scaled keff=1.0 value.
    keff_pred = y_pred[:, IDX_KEFF]
    keff_dev  = tf.reduce_mean(tf.square(keff_pred - KEFF_CRITICAL_SCALED))

    return weighted_mse + 0.01 * mono_penalty + 0.05 * keff_dev


# =============================================================================
# SECTION 9 — TRAIN
# =============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=pure_prediction_loss,
    metrics=['mae']
)

print("[TRAINING] Starting CNN training ...")
print(f"  Dataset : {len(X_train)} train + {len(X_val)} val patterns")
print(f"  Loss    : PURE weighted MSE + monotonicity + keff proximity")
print(f"  Note    : NO ppf target soft-constraint — pure prediction\n")

t_start = time.time()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15,
        min_lr=1e-5, verbose=1
    ),
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda ep, logs: print(
            f"  Ep {ep+1:4d} | loss: {logs['loss']:.5f} | val: {logs['val_loss']:.5f}"
        ) if (ep + 1) % 25 == 0 else None
    ),
]

history = model.fit(
    X_train, Y_train_sc,
    validation_data=(X_val, Y_val_sc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=0
)

t_train     = time.time() - t_start
best_epoch  = int(np.argmin(history.history['val_loss'])) + 1
print(f"\n[TRAINING DONE]  {t_train:.1f}s  |  best epoch: {best_epoch}\n")


# =============================================================================
# SECTION 10 — EVALUATE ON TEST SET
# =============================================================================

print("[EVALUATION] Deterministic predictions on test set ...")

Y_pred_sc   = model.predict(X_test, verbose=0)
Y_pred_real = y_scaler.inverse_transform(Y_pred_sc)
Y_true_real = Y_test

ppf_max_pred  = Y_pred_real[:, IDX_PPF_MAX]
ppf_max_true  = Y_true_real[:, IDX_PPF_MAX]
cycle_pred    = Y_pred_real[:, IDX_CYCLE]
cycle_true    = Y_true_real[:, IDX_CYCLE]
keff_pred     = Y_pred_real[:, IDX_KEFF]
keff_true     = Y_true_real[:, IDX_KEFF]

# ── PPF metrics ───────────────────────────────────────────────────────────────
ppf_mae       = np.abs(ppf_max_pred - ppf_max_true).mean()
ppf_rel_err   = (np.abs(ppf_max_pred - ppf_max_true) / (ppf_max_true + 1e-6)).mean() * 100
ppf_r2        = r2_score(ppf_max_true, ppf_max_pred)
ppf_pearson   = np.corrcoef(ppf_max_pred, ppf_max_true)[0, 1]

# ── Goal zone accuracy (2.0–4.5) ──────────────────────────────────────────────
goal_mask   = (ppf_max_true >= PPF_GOAL_LOW) & (ppf_max_true <= PPF_GOAL_HIGH)
goal_mae    = np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask]).mean()
goal_r2     = r2_score(ppf_max_true[goal_mask], ppf_max_pred[goal_mask])

# ── Cycle length metrics ──────────────────────────────────────────────────────
cycle_mae   = np.abs(cycle_pred - cycle_true).mean()
cycle_r2    = r2_score(cycle_true, cycle_pred)

# ── keff metrics ──────────────────────────────────────────────────────────────
keff_mae    = np.abs(keff_pred - keff_true).mean()
keff_r2     = r2_score(keff_true, keff_pred)

print(f"\n{'='*58}")
print(f"CNN TEST RESULTS  (v2 — Pure Prediction)")
print(f"{'='*58}")
print(f"  PPF_max (all test patterns):")
print(f"    MAE              : {ppf_mae:.4f}")
print(f"    Relative error   : {ppf_rel_err:.2f}%")
print(f"    R²               : {ppf_r2:.4f}")
print(f"    Pearson r        : {ppf_pearson:.4f}")
print(f"  PPF_max (goal zone {PPF_GOAL_LOW}–{PPF_GOAL_HIGH}  n={goal_mask.sum()}):")
print(f"    MAE              : {goal_mae:.4f}  ← performance where QICA will search")
print(f"    R²               : {goal_r2:.4f}")
print(f"  Cycle length:")
print(f"    MAE              : {cycle_mae:.2f} days")
print(f"    R²               : {cycle_r2:.4f}")
print(f"  keff_boc:")
print(f"    MAE              : {keff_mae:.5f}")
print(f"    R²               : {keff_r2:.4f}")
print(f"{'='*58}\n")


# =============================================================================
# SECTION 11 — MONTE CARLO DROPOUT UNCERTAINTY
# =============================================================================
# MC Dropout: run the model MC_SAMPLES times with Dropout ACTIVE (training=True).
# Each run gives a slightly different prediction.  The standard deviation across
# runs estimates the model's EPISTEMIC UNCERTAINTY — how confident it is.
#
# Physical interpretation:
#   Low σ  → pattern similar to many training examples → trust the prediction
#   High σ → unusual pattern (unlike training data) → query a simulator
#
# For the ACTIVE LEARNING LOOP (Section 13), high-σ patterns inside the
# goal zone are the most valuable to query — they are uncertain AND potentially
# useful.

print("[MC DROPOUT] Estimating prediction uncertainty ...")
t_mc = time.time()

mc_predictions = np.stack([
    model(X_test, training=True).numpy()
    for _ in range(MC_SAMPLES)
])   # (MC_SAMPLES, N_test, 35)

mc_mean_sc  = mc_predictions.mean(axis=0)
mc_std_sc   = mc_predictions.std(axis=0)
mc_mean_real = y_scaler.inverse_transform(mc_mean_sc)
mc_std_real  = mc_std_sc * y_scaler.scale_

ppf_mc_mean = mc_mean_real[:, IDX_PPF_MAX]
ppf_mc_std  = mc_std_real[:, IDX_PPF_MAX]

unc_err_corr = np.corrcoef(ppf_mc_std, np.abs(ppf_mc_mean - ppf_max_true))[0, 1]

print(f"  Time              : {time.time()-t_mc:.1f}s")
print(f"  Mean σ(ppf_max)   : {ppf_mc_std.mean():.4f}")
print(f"  Max σ(ppf_max)    : {ppf_mc_std.max():.4f}")
print(f"  Uncertainty–error corr: {unc_err_corr:.3f}  (positive → σ is a useful flag)\n")


# =============================================================================
# SECTION 12 — SENSITIVITY ANALYSIS  (∂ppf_max / ∂position)
# =============================================================================
# Input gradient saliency through the embedding layer.
# For each of the 31 active positions, compute the L2 norm of the gradient
# of ppf_max prediction with respect to that position's embedding vector.
#
# High |gradient| → assembly choice at this position strongly affects ppf_max.
# This is the EXPLAINABILITY output:
#   "Position P1 (near-centre) is the most critical: always put once-burnt
#    (low-energy) fuel there to avoid flux peaking."
#
# The normalised sensitivity values are saved to CSV for QICA to use as
# ASSIMILATION WEIGHTS — positions with high sensitivity get more optimisation
# effort per QICA iteration.

print("[SENSITIVITY]  Computing ∂ppf_max/∂position ...")

n_sens   = min(200, len(X_test))
X_sample = tf.constant(X_test[:n_sens], dtype=tf.int32)

with tf.GradientTape() as tape:
    emb_layer = model.get_layer('assembly_embedding')
    x_emb     = emb_layer(X_sample)          # (n_sens, 6, 6, 16)
    tape.watch(x_emb)

    h = x_emb
    skipped = {'loading_grid', 'assembly_embedding'}
    for layer in model.layers:
        if layer.name in skipped or isinstance(layer, keras.layers.InputLayer):
            continue
        try:
            h = layer(h, training=False)
        except Exception:
            try:
                h = layer(h)
            except Exception:
                pass

    ppf_out = h[:, IDX_PPF_MAX]

grads = tape.gradient(ppf_out, x_emb)   # (n_sens, 6, 6, 16)

if grads is not None:
    sens_grid = tf.norm(grads, axis=-1).numpy().mean(axis=0)   # (6, 6)
    # Zero out masked cells so they don't appear active
    sens_grid[~GRID_MASK] = 0.0

    sens_pos = np.zeros(N_POS, dtype=np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            pos = GRID_LAYOUT[r, c]
            if pos >= 0:
                sens_pos[pos] = sens_grid[r, c]

    sens_norm = sens_pos / (sens_pos.max() + 1e-8)
    top5 = np.argsort(sens_norm)[::-1][:5].tolist()
    print(f"  Top-5 critical positions : {top5}")
    print(f"  Sensitivity range        : {sens_norm.min():.3f} – {sens_norm.max():.3f}\n")
else:
    print("  [WARN] Gradient failed — using uniform sensitivity.\n")
    sens_norm = np.ones(N_POS, dtype=np.float32)
    sens_grid = np.ones((GRID_ROWS, GRID_COLS), dtype=np.float32)
    sens_pos  = np.ones(N_POS, dtype=np.float32)

sens_df = pd.DataFrame({
    'position'           : [f'pos_{i}' for i in range(N_POS)],
    'sensitivity'        : sens_pos,
    'sensitivity_norm'   : sens_norm,
    'monocore_efpd_type' : [ASSEMBLY_CYCLE_EQUIV.get(i % N_TYPES + 1, 0)
                             for i in range(N_POS)],
})
sens_df.to_csv(SENS_NAME, index=False)
print(f"  Saved: {SENS_NAME}")


# =============================================================================
# SECTION 13 — ACTIVE LEARNING LOOP SCAFFOLD
# =============================================================================
# Active learning improves the surrogate by querying the most informative
# patterns — those where the model is UNCERTAIN and results would be USEFUL.
#
# The loop here is a SCAFFOLD (no simulator yet):
#   Round n:
#     1. Predict ppf_max + σ(ppf_max) for ALL patterns (train + val + test).
#     2. Filter to patterns in the goal zone [PPF_GOAL_LOW, PPF_GOAL_HIGH].
#     3. Sort by descending σ — high-uncertainty patterns in the goal zone
#        are most informative (uncertain but potentially good → worth querying).
#     4. Flag the top AL_MAX_QUERIES patterns as "query candidates".
#     5. [ SIMULATOR PLACEHOLDER ] → call PARCS/Serpent/OpenMC here.
#        When the simulator is available, replace the stub with:
#            new_y = simulate(query_patterns)
#            X_train = np.vstack([X_train, make_grid_input(query_patterns)])
#            Y_train = np.vstack([Y_train, new_y])
#            # Retrain: model.fit(X_train, Y_train, ...)
#     6. After QICA is integrated: QICA proposes candidate patterns,
#        CNN screens them, high-σ ones get forwarded to the simulator.
#
# Why uncertainty sampling works:
#   Patterns in the goal zone with high σ are "on the frontier" — similar
#   to training data but with novel assembly arrangements the model hasn't
#   seen enough of.  Adding their true labels reduces σ exactly where
#   QICA will be searching.

print("\n[ACTIVE LEARNING]  Running query candidate selection (scaffold) ...")

# Run MC Dropout on the full dataset
print("  Computing uncertainty over full dataset ...")
mc_all = np.stack([
    model(X_grid, training=True).numpy()
    for _ in range(MC_SAMPLES)
])   # (MC_SAMPLES, N, 35)

mc_std_all  = mc_all.std(axis=0) * y_scaler.scale_
mc_mean_all = y_scaler.inverse_transform(mc_all.mean(axis=0))

ppf_all_pred = mc_mean_all[:, IDX_PPF_MAX]
ppf_all_std  = mc_std_all[:, IDX_PPF_MAX]

# Filter: goal zone + high uncertainty
goal_zone_mask  = (ppf_all_pred >= PPF_GOAL_LOW) & (ppf_all_pred <= PPF_GOAL_HIGH)
uncertainty_mask = ppf_all_std >= AL_UNCERTAINTY_THRESHOLD
query_mask       = goal_zone_mask & uncertainty_mask

query_indices   = np.where(query_mask)[0]
query_sorted    = query_indices[np.argsort(ppf_all_std[query_indices])[::-1]]
query_top       = query_sorted[:AL_MAX_QUERIES]

print(f"  Patterns in goal zone          : {goal_zone_mask.sum()}")
print(f"  High-uncertainty (σ≥{AL_UNCERTAINTY_THRESHOLD}) patterns: {uncertainty_mask.sum()}")
print(f"  Query candidates (intersection): {len(query_indices)}")
print(f"  Top-{AL_MAX_QUERIES} flagged for simulator   : {len(query_top)}")
print()
print("  [ SIMULATOR STUB ] — plug in PARCS/Serpent/OpenMC here:")
print("      for idx in query_top:")
print("          true_ppf, true_cycle, true_keff = simulate(df.iloc[idx])")
print("          # Add (idx, true_ppf, true_cycle, true_keff) to labelled set")
print("          # Retrain CNN on expanded dataset")
print()
print("  QICA integration (next step):")
print("      → QICA proposes loading patterns (not in training set)")
print("      → CNN predicts ppf_max + σ for each proposal")
print("      → Low σ, low ppf_max → accept as QICA candidate")
print("      → High σ → forward to simulator for labelling")

# Save query candidates to CSV for manual review / simulator hand-off
al_df = pd.DataFrame({
    'pattern_id'     : df['pattern_id'].values[query_top] if 'pattern_id' in df.columns
                       else [f'pattern_{i:05d}' for i in query_top],
    'pred_ppf_max'   : ppf_all_pred[query_top],
    'pred_ppf_std'   : ppf_all_std[query_top],
    'cycle_length'   : mc_mean_all[query_top, IDX_CYCLE],
    'keff_boc'       : mc_mean_all[query_top, IDX_KEFF],
})
al_df.to_csv('cnn_al_query_candidates.csv', index=False)
print(f"\n  Saved query candidates → cnn_al_query_candidates.csv")


# =============================================================================
# SECTION 14 — SAVE MODEL + CONFIG
# =============================================================================

model.save(MODEL_NAME)

config = {
    # Geometry
    'N_POS'              : N_POS,
    'N_TYPES'            : N_TYPES,
    'N_STEPS'            : N_STEPS,
    'GRID_ROWS'          : GRID_ROWS,
    'GRID_COLS'          : GRID_COLS,
    'GRID_LAYOUT'        : GRID_LAYOUT.tolist(),
    'GRID_MASK'          : GRID_MASK.tolist(),

    # Output indices
    'IDX_PPF_MAX'        : IDX_PPF_MAX,
    'IDX_PPF_BOC'        : IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': 2,
    'IDX_PPF_STEPS_END'  : 2 + N_STEPS,
    'IDX_CYCLE'          : IDX_CYCLE,
    'IDX_KEFF'           : IDX_KEFF,
    'N_OUTPUTS'          : N_OUTPUTS,

    # Goal range (for QICA to use as search target, NOT a loss constraint)
    'PPF_GOAL_LOW'       : PPF_GOAL_LOW,
    'PPF_GOAL_HIGH'      : PPF_GOAL_HIGH,

    # Scaler (for QICA to inverse-transform CNN outputs)
    'y_scaler_mean'      : y_scaler.mean_.tolist(),
    'y_scaler_scale'     : y_scaler.scale_.tolist(),

    # Assembly encoding (monocore EFPD from xlsx)
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v) for k, v in ASSEMBLY_CYCLE_EQUIV.items()},

    # MC Dropout
    'mc_dropout_enabled' : True,
    'mc_samples'         : MC_SAMPLES,
    'al_uncertainty_thr' : AL_UNCERTAINTY_THRESHOLD,

    # Performance
    'test_ppf_mae'       : float(ppf_mae),
    'test_ppf_r2'        : float(ppf_r2),
    'test_goal_zone_mae' : float(goal_mae),
    'test_cycle_mae'     : float(cycle_mae),
    'test_keff_mae'      : float(keff_mae),
}

with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n[SAVED]  {MODEL_NAME}")
print(f"[SAVED]  {CONFIG_NAME}")
print(f"[SAVED]  {SENS_NAME}")


# =============================================================================
# SECTION 15 — VISUALISATIONS
# =============================================================================

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    f"BEAVRS CNN v2 — Pure Prediction  |  PPF Goal Zone {PPF_GOAL_LOW}–{PPF_GOAL_HIGH}  "
    f"|  Test MAE={ppf_mae:.3f}  R²={ppf_r2:.3f}",
    fontsize=13, fontweight='bold'
)
gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

# ── 1. Training loss ──────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(history.history['loss'],     '#1B4FBF', lw=1.5, label='Train')
ax.plot(history.history['val_loss'], '#F5A623', lw=1.5, label='Val')
ax.axvline(best_epoch-1, color='red', lw=1, ls=':', label=f'Best ep {best_epoch}')
ax.set_yscale('log')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
ax.set_title('Training Curve\n(Pure MSE + physics)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── 2. PPF scatter — coloured by goal zone ────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
# Points in goal zone: teal; outside: grey
colors_sc = np.where(
    (ppf_max_true >= PPF_GOAL_LOW) & (ppf_max_true <= PPF_GOAL_HIGH),
    '#17BECF', '#AAAAAA'
)
lim = [ppf_max_true.min()-0.1, ppf_max_true.max()+0.1]
ax.scatter(ppf_max_true, ppf_max_pred, c=colors_sc, alpha=0.35, s=7)
ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
ax.axhspan(PPF_GOAL_LOW, PPF_GOAL_HIGH, alpha=0.06, color='teal',
           label=f'Goal zone {PPF_GOAL_LOW}–{PPF_GOAL_HIGH}')
ax.axvspan(PPF_GOAL_LOW, PPF_GOAL_HIGH, alpha=0.06, color='teal')
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'PPF Prediction\nMAE={ppf_mae:.3f}  R²={ppf_r2:.3f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.text(0.05, 0.92, f'Goal MAE={goal_mae:.3f}', transform=ax.transAxes,
        fontsize=8, color='teal', fontweight='bold')

# ── 3. Cycle length scatter ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.scatter(cycle_true, cycle_pred, alpha=0.3, s=7, color='#2CA02C')
lim_c = [cycle_true.min()-5, cycle_true.max()+5]
ax.plot(lim_c, lim_c, 'k--', lw=1)
ax.set_xlabel('True cycle length (days)'); ax.set_ylabel('Predicted (days)')
ax.set_title(f'Cycle Length\nMAE={cycle_mae:.1f}d  R²={cycle_r2:.3f}')
ax.grid(alpha=0.3)

# ── 4. keff scatter ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
ax.scatter(keff_true, keff_pred, alpha=0.3, s=7, color='#9467BD')
lim_k = [keff_true.min()-0.005, keff_true.max()+0.005]
ax.plot(lim_k, lim_k, 'k--', lw=1)
ax.axvline(1.0, color='red', lw=1, ls=':', label='keff=1.0 (critical)')
ax.axhline(1.0, color='red', lw=1, ls=':')
ax.set_xlabel('True keff_boc'); ax.set_ylabel('Predicted keff_boc')
ax.set_title(f'keff (BOC)  [NEW in v2]\nMAE={keff_mae:.5f}  R²={keff_r2:.3f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 5. Sensitivity heatmap ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
disp_sens = np.full((GRID_ROWS, GRID_COLS), np.nan)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        pos = GRID_LAYOUT[r, c]
        if pos >= 0:
            disp_sens[r, c] = sens_norm[pos]

cmap_s = plt.cm.RdYlGn_r.copy(); cmap_s.set_bad('lightgrey')
im = ax.imshow(disp_sens, cmap=cmap_s, aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Norm. sensitivity')
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        pos = GRID_LAYOUT[r, c]
        if pos >= 0:
            ax.text(c, r, f'P{pos}', ha='center', va='center',
                    fontsize=6,
                    fontweight='bold' if sens_norm[pos] > 0.7 else 'normal',
                    color='black')
ax.set_title('Sensitivity: ∂ppf_max/∂position\n(Red = critical for PPF)')
ax.set_xticks([]); ax.set_yticks([])

# ── 6. PPF burnup profile ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
steps_range    = np.arange(N_STEPS)
true_smean = Y_true_real[:, IDX_PPF_STEPS].mean(axis=0)
true_sstd  = Y_true_real[:, IDX_PPF_STEPS].std(axis=0)
pred_smean = Y_pred_real[:, IDX_PPF_STEPS].mean(axis=0)
pred_sstd  = Y_pred_real[:, IDX_PPF_STEPS].std(axis=0)
ax.plot(steps_range, true_smean, '#1B4FBF', lw=2, label='True')
ax.fill_between(steps_range, true_smean-true_sstd, true_smean+true_sstd,
                color='#1B4FBF', alpha=0.15)
ax.plot(steps_range, pred_smean, '#F5A623', lw=2, ls='--', label='Predicted')
ax.fill_between(steps_range, pred_smean-pred_sstd, pred_smean+pred_sstd,
                color='#F5A623', alpha=0.15)
ax.axhspan(PPF_GOAL_LOW, PPF_GOAL_HIGH, alpha=0.06, color='teal',
           label=f'Goal {PPF_GOAL_LOW}–{PPF_GOAL_HIGH}')
ax.set_xlabel('Burnup Step'); ax.set_ylabel('Max PPF at Step')
ax.set_title('PPF Burnup Profile\n(mean ± 1σ, test patterns)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 7. PPF distribution with goal zone ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
bins = np.linspace(1.5, 5.5, 60)
ax.hist(ppf_max_true, bins=bins, alpha=0.55, color='#1B4FBF', label='True')
ax.hist(ppf_max_pred, bins=bins, alpha=0.55, color='#F5A623', label='Predicted')
ax.axvspan(PPF_GOAL_LOW, PPF_GOAL_HIGH, alpha=0.12, color='teal',
           label=f'Goal zone')
ax.set_xlabel('ppf_max'); ax.set_ylabel('Count')
ax.set_title(f'PPF Distribution\n(Goal zone n_pred={((ppf_max_pred>=PPF_GOAL_LOW)&(ppf_max_pred<=PPF_GOAL_HIGH)).sum()})')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 8. Uncertainty vs error ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
ppf_abs_err = np.abs(ppf_mc_mean - ppf_max_true)
ax.scatter(ppf_mc_std, ppf_abs_err, alpha=0.2, s=6, color='#9467BD')
m, b = np.polyfit(ppf_mc_std, ppf_abs_err, 1)
xs   = np.linspace(ppf_mc_std.min(), ppf_mc_std.max(), 100)
ax.plot(xs, m*xs + b, 'r--', lw=1.5,
        label=f'r = {unc_err_corr:.3f}')
ax.axvline(AL_UNCERTAINTY_THRESHOLD, color='orange', lw=1.5, ls=':',
           label=f'AL threshold σ={AL_UNCERTAINTY_THRESHOLD}')
ax.set_xlabel('MC σ (uncertainty)'); ax.set_ylabel('|PPF Error|')
ax.set_title('MC Uncertainty vs Error\n(orange = AL query threshold)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 9. Monocore EFPD bar chart ────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
type_ids  = sorted(monocore_map.keys())
efpd_vals = [monocore_map[t] for t in type_ids]
bar_colors = ['#D62728' if monocore_map[t] < 400 else '#1B4FBF' for t in type_ids]
bars = ax.bar(type_ids, efpd_vals, color=bar_colors, edgecolor='white')
ax.set_xlabel('Assembly Type (from xlsx)')
ax.set_ylabel('Monocore EFPD')
ax.set_title('Assembly Monocore Cycle Lengths\n(Red=once-burnt, Blue=fresh)')
for bar, val in zip(bars, efpd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{val:.0f}', ha='center', va='bottom', fontsize=7)
ax.grid(axis='y', alpha=0.3)
legend_patches = [Patch(color='#D62728', label='Once-burnt (Batch B)'),
                  Patch(color='#1B4FBF', label='Fresh (Batch A)')]
ax.legend(handles=legend_patches, fontsize=7)

# ── 10. Example loading grids (lowest vs highest predicted PPF) ───────────────
ax_low  = fig.add_subplot(gs[2, 1])
ax_high = fig.add_subplot(gs[2, 2])
idx_low  = ppf_max_true.argmin()
idx_high = ppf_max_true.argmax()
for ax_ex, idx, label in [
    (ax_low,  idx_low,  f'Best: true PPF={ppf_max_true[idx_low]:.3f}'),
    (ax_high, idx_high, f'Worst: true PPF={ppf_max_true[idx_high]:.3f}'),
]:
    g = X_test[idx].astype(float)
    # Apply geometry mask — show NaN for reflector padding
    g_disp = g.copy()
    g_disp[~GRID_MASK] = np.nan
    cmap_ex = plt.cm.YlOrRd.copy(); cmap_ex.set_bad('lightgrey')
    im_ex = ax_ex.imshow(g_disp, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im_ex, ax=ax_ex, label='Type')
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            pos = GRID_LAYOUT[r, c]
            if pos >= 0:
                ax_ex.text(c, r, f'{X_test[idx,r,c]}',
                           ha='center', va='center', fontsize=8)
    ax_ex.set_title(f'{label}\nPred={ppf_max_pred[idx]:.3f}')
    ax_ex.set_xticks([]); ax_ex.set_yticks([])

# ── 11. Residuals ─────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 3])
resid = ppf_max_pred - ppf_max_true
ax.hist(resid, bins=50, color='#1B4FBF', edgecolor='white', lw=0.5)
ax.axvline(0, color='red', lw=1.5, label='Zero error')
ax.axvline(resid.mean(), color='orange', lw=1.5,
           label=f'Mean={resid.mean():.3f}')
ax.set_xlabel('Predicted − True ppf_max')
ax.set_ylabel('Count')
ax.set_title(f'Residuals\nμ={resid.mean():.3f}  σ={resid.std():.3f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.savefig(PLOT_NAME, dpi=150, bbox_inches='tight')
print(f"\n[SAVED]  {PLOT_NAME}")


# =============================================================================
# SECTION 16 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("rep_cnn_v2.py  FINAL SUMMARY")
print("=" * 60)
print(f"  Architecture     : BEAVRS CNN v2 (6×6, embed+res+attn+mlp)")
print(f"  Loss             : PURE weighted MSE (no PPF target constraint)")
print(f"  Parameters       : {model.count_params():,}")
print(f"  Best epoch       : {best_epoch} / {EPOCHS}")
print(f"  Training time    : {t_train:.1f}s")
print()
print(f"  Dataset PPF      : {ppf_global_max.min():.2f}–{ppf_global_max.max():.2f}")
print(f"  Dataset PPF avg  : {ppf_global_max.mean():.3f}")
print(f"  Goal zone        : {PPF_GOAL_LOW}–{PPF_GOAL_HIGH} (for QICA search)")
print()
print(f"  PPF_max:")
print(f"    MAE            : {ppf_mae:.4f}")
print(f"    Relative error : {ppf_rel_err:.2f}%")
print(f"    R²             : {ppf_r2:.4f}")
print(f"    Goal-zone MAE  : {goal_mae:.4f}")
print()
print(f"  Cycle length:")
print(f"    MAE            : {cycle_mae:.2f} days")
print(f"    R²             : {cycle_r2:.4f}")
print()
print(f"  keff_boc:")
print(f"    MAE            : {keff_mae:.5f}")
print(f"    R²             : {keff_r2:.4f}")
print()
print(f"  MC Dropout σ_ppf (mean) : {ppf_mc_std.mean():.4f}")
print(f"  Active learning candidates: {len(query_top)}")
print()
print(f"  OUTPUT FILES:")
print(f"    {MODEL_NAME}")
print(f"    {CONFIG_NAME}")
print(f"    {SENS_NAME}")
print(f"    {PLOT_NAME}")
print(f"    cnn_al_query_candidates.csv")
print()
print("  NEXT STEP: Run qica_cnn.py — use this CNN as the QICA fitness")
print("  oracle, searching for loading patterns with ppf_max → 2.0.")
print("=" * 60)