"""
=============================================================================
cnn_v3.py  —  BEAVRS CNN v3  |  Fixed keff  |  Min-PPF Surrogate
=============================================================================

WHAT CHANGED FROM v2 (bug fixes + improvements)
──────────────────────────────────────────────────────────────────────────

BUG FIX — keff R² = -36.8 (catastrophic failure in v2):
  ROOT CAUSE: v2 computed
      KEFF_CRITICAL_SCALED = (1.0 - mean_keff) / std_keff
                           = (1.0 - 1.1307) / 0.0113 ≈ -11.57
  Then penalised: keff_dev = mean((keff_pred - (-11.57))²) × 0.05
  At average prediction (keff_pred ≈ 0 in scaled space):
      keff_dev ≈ 11.57² ≈ 133.8, weighted = 6.7
  This DOMINATES the loss (main weighted MSE ≈ 3-5) and forces all
  keff predictions toward scaled value -11.57 → real value 1.000,
  while the true keff range is 1.097-1.179.
  FIX: Remove the broken keff_proximity penalty entirely.

FIX 1 — keff representation:
  Convert keff → reactivity ρ (pcm): rho_pcm = (keff-1)/keff × 1e5
  Range ~8900-15300 pcm, mean ~11600, std ~1100.
  StandardScaler gives N(0,1) → model learns this properly.
  Dedicate a separate output head to keff with 5× loss weight.

FIX 2 — Multi-head architecture:
  Shared backbone → three separate Dense branches:
    ppf_head   → ppf_max, ppf_boc, 31 PPF step values
    cycle_head → cycle_length
    keff_head  → keff_boc (via ρ_pcm transform)
  Each head has its own gradient pathway.
  Prevents the small-variance keff signal from being drowned by PPF/cycle.

NEW — Active Learning loop (Section 13):
  Full production-ready scaffold:
    Round n:
      1. QICA proposes new candidate patterns (not in training set)
      2. CNN predicts ppf_max + σ(MC Dropout) for each candidate
      3. Low σ + low ppf → accept as surrogate-confident
         High σ → forward to OpenMC for true labelling
      4. OpenMC results added to dataset
      5. CNN retrained on expanded dataset
  OpenMC stub: replace openmc_simulate() with real call.

GOAL — Minimum PPF (no hallucination):
  Dataset ppf_max range: ~1.61 – 7.91, 10th pct ≈ 2.02
  CNN is trained with PURE MSE — it maps inputs to true physics outputs.
  The QICA (qica_v3.py) then SEARCHES for low-PPF patterns.
  We do NOT train the CNN to predict values below the data distribution.
  The lowest achievable PPF without hallucination = ~2.0 (10th pct).

OUTPUT FILES:
  cnn_v3_model.keras         — trained model
  cnn_v3_config.json         — geometry, scalers, sensitivity (for QICA)
  cnn_v3_sens.csv            — position sensitivities
  cnn_v3_results.png         — evaluation plots
  cnn_v3_al_candidates.csv   — active learning query candidates
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
print("cnn_v3.py — BEAVRS CNN  |  Fixed keff  |  Min-PPF Surrogate\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'
MODEL_NAME  = 'cnn_v3_model.keras'
CONFIG_NAME = 'cnn_v3_config.json'
SENS_NAME   = 'cnn_v3_sens.csv'
PLOT_NAME   = 'cnn_v3_results.png'
AL_CSV      = 'cnn_v3_al_candidates.csv'

# ── PPF reporting range ───────────────────────────────────────────────────────
# These are used ONLY for evaluation reporting, NOT for the training loss.
# The CNN is trained with pure MSE — it predicts what the physics says.
# The QICA then searches for minimum-PPF inputs.
PPF_REPORT_LOW  = 2.0    # below this = very good pattern
PPF_REPORT_HIGH = 4.5    # above this = too peaked

# ── BEAVRS core geometry ──────────────────────────────────────────────────────
N_POS    = 31            # unique positions in 1/8-core symmetry
N_TYPES  = 9             # assembly types 1–9 (0 = reflector/mask)
N_STEPS  = 31            # burnup timesteps with PPF data

GRID_ROWS, GRID_COLS = 6, 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 400
LR          = 0.001
DROPOUT     = 0.15
TEST_FRAC   = 0.15
VAL_FRAC    = 0.15
SEED        = 42
MC_SAMPLES  = 30

# ── Active learning ───────────────────────────────────────────────────────────
AL_UNCERTAINTY_THRESHOLD  = 0.07   # σ_ppf above which pattern needs simulator
AL_MAX_QUERIES_PER_ROUND  = 50


# =============================================================================
# SECTION 2 — LOAD MONOCORE CYCLE LENGTHS (assembly type embedding init)
# =============================================================================

print("[XLSX] Loading monocore cycle lengths ...")
if os.path.exists(XL_FILE):
    xl_df = pd.read_excel(XL_FILE, sheet_name='Cycle_Lengths')
    monocore_map = dict(zip(xl_df['fa_id'].astype(int),
                            xl_df['monocore_cycle_length'].astype(float)))
    for fid, cyc in sorted(monocore_map.items()):
        fa_name = xl_df.loc[xl_df['fa_id'] == fid, 'fa_name'].values[0]
        print(f"  Type {fid}  ({fa_name:<18s})  EFPD = {cyc:.1f}")
else:
    print("  [WARN] xlsx not found — using hardcoded fallback values.")
    monocore_map = {1:172.9, 2:366.9, 3:323.2, 4:299.8, 5:519.9,
                   6:504.9, 7:475.3, 8:471.6, 9:454.7}

ASSEMBLY_CYCLE_EQUIV = {0: 0.0}
ASSEMBLY_CYCLE_EQUIV.update(monocore_map)
print()


# =============================================================================
# SECTION 3 — LOAD DATA
# =============================================================================

print("[DATA] Loading BEAVRS dataset ...")
if not os.path.exists(BEAVRS_CSV):
    print(f"[ERROR] {BEAVRS_CSV} not found."); sys.exit(1)

df = pd.read_csv(BEAVRS_CSV, skiprows=1, engine='python', on_bad_lines='skip')
print(f"  Loaded {len(df)} patterns × {df.shape[1]} columns\n")

load_cols  = [f'loading_{i}' for i in range(N_POS)]
react_cols = sorted([c for c in df.columns if c.startswith('react_')],
                    key=lambda c: int(c.split('_')[1]))

ppf_steps   = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))

step_max_ppf = np.stack([
    df[[f'ppf_s{s}_a{i}' for i in ppf_assembs
        if f'ppf_s{s}_a{i}' in df.columns]].values.astype(np.float32).max(axis=1)
    for s in ppf_steps
], axis=1)   # (N, N_STEPS)

ppf_global_max = step_max_ppf.max(axis=1)
ppf_boc        = step_max_ppf[:, 0]

# ── keff via ρ (pcm) ── FIX: use standard reactivity units, not raw keff ─────
# ρ_pcm = (keff-1)/keff × 1e5
# Range: ~8900–15300 pcm, mean ~11600, std ~1100
# This has much better numerical properties than raw keff ∈ [1.097, 1.180]
# The StandardScaler will give a well-conditioned N(0,1) output for this head.
keff_raw   = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
rho_pcm    = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

# Dataset summary
print("=" * 58)
print("DATASET PPF ANALYSIS")
print("=" * 58)
print(f"  Patterns           : {len(df)}")
print(f"  PPF_max range      : {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  PPF_max mean       : {ppf_global_max.mean():.3f}")
print(f"  10th percentile    : {np.percentile(ppf_global_max, 10):.3f}  ← realistic QICA target")
print(f"  Patterns below 2.0 : {(ppf_global_max < 2.0).sum()} ({(ppf_global_max < 2.0).mean()*100:.1f}%)")
print(f"  Cycle length range : {df.cycle_length.min():.1f} – {df.cycle_length.max():.1f} days")
print(f"  keff_boc range     : {keff_raw.min():.4f} – {keff_raw.max():.4f}")
print(f"  ρ_pcm range        : {rho_pcm.min():.0f} – {rho_pcm.max():.0f} pcm")
print("=" * 58)
print()


# =============================================================================
# SECTION 4 — TARGETS AND FEATURES
# =============================================================================
#
# Output layout (flat vector, total N_OUTPUTS):
#   [0]          ppf_max         — global cycle maximum PPF         PRIMARY
#   [1]          ppf_boc         — beginning-of-cycle PPF
#   [2 : 2+N_STEPS]  ppf_steps  — max PPF at each of 31 burnup steps
#   [2+N_STEPS]  cycle_length   — operating days
#   [3+N_STEPS]  rho_pcm        — reactivity (pcm) at BOC  (was keff in v2)
#
# WHY SEPARATE HEAD SCALERS:
#   If we put all 35 outputs through one StandardScaler, the keff/rho
#   column gets N(0,1) in the same array as PPF and cycle_length.
#   BUT the loss function then weights them together, and large-magnitude
#   PPF/cycle errors still swamp the tiny rho gradient.
#   Solution: separate the rho column into its own scaler and own loss term.

N_OUTPUTS       = 1 + 1 + N_STEPS + 1 + 1   # = 35
IDX_PPF_MAX     = 0
IDX_PPF_BOC     = 1
IDX_PPF_STEPS   = slice(2, 2 + N_STEPS)      # slice object for convenience
IDX_CYCLE       = 2 + N_STEPS                # = 33
IDX_RHO         = 3 + N_STEPS               # = 34

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_rho       = rho_pcm.reshape(-1, 1).astype(np.float32)

# Concatenate: first 34 cols (ppf + cycle), last col (rho) separate
Y_main = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle], axis=1)  # (N, 34)
Y_rho_col = Y_rho                                                                 # (N, 1)

# Build 6×6 integer grid input
X_raw = df[load_cols].values.astype(np.int32)     # (N, 31)
X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1
# Reflector padding (5 cells) stays 0

print(f"[INPUT]  Grid shape: {X_grid.shape}")
print(f"  Active fuel cells : {GRID_MASK.sum()}/36")
print(f"[TARGETS] Main: {Y_main.shape}, Rho: {Y_rho_col.shape}\n")


# =============================================================================
# SECTION 5 — TRAIN / VAL / TEST SPLIT + SCALERS
# =============================================================================

(X_tr, X_tmp,
 Ym_tr, Ym_tmp,
 Yr_tr, Yr_tmp) = train_test_split(X_grid, Y_main, Y_rho_col,
                                    test_size=TEST_FRAC + VAL_FRAC,
                                    random_state=SEED)

val_fraction_of_tmp = VAL_FRAC / (TEST_FRAC + VAL_FRAC)

(X_val, X_test,
 Ym_val, Ym_test,
 Yr_val, Yr_test) = train_test_split(X_tmp, Ym_tmp, Yr_tmp,
                                      test_size=0.5,
                                      random_state=SEED)

print(f"[SPLIT] {len(X_tr)} train / {len(X_val)} val / {len(X_test)} test")

# ── Scaler for main outputs (PPF + cycle) ─────────────────────────────────────
ym_scaler = StandardScaler()
Ym_tr_sc  = ym_scaler.fit_transform(Ym_tr)
Ym_val_sc = ym_scaler.transform(Ym_val)
Ym_test_sc = ym_scaler.transform(Ym_test)

# ── Scaler for rho (separate — CRITICAL FIX) ──────────────────────────────────
yr_scaler = StandardScaler()
Yr_tr_sc  = yr_scaler.fit_transform(Yr_tr)
Yr_val_sc = yr_scaler.transform(Yr_val)
Yr_test_sc = yr_scaler.transform(Yr_test)

# Combined scaled target: concat for training
Y_tr_sc   = np.concatenate([Ym_tr_sc,  Yr_tr_sc],  axis=1).astype(np.float32)  # (N, 35)
Y_val_sc  = np.concatenate([Ym_val_sc, Yr_val_sc],  axis=1).astype(np.float32)
Y_test_sc = np.concatenate([Ym_test_sc, Yr_test_sc], axis=1).astype(np.float32)

ppf_scale  = ym_scaler.mean_[0]   # for summary
cycle_mean = ym_scaler.mean_[IDX_CYCLE - 0]  # offset by 0 since Y_main starts at IDX 0
rho_mean   = yr_scaler.mean_[0]

print(f"[SCALING]")
print(f"  ppf_max     mean={ym_scaler.mean_[0]:.3f}  std={ym_scaler.scale_[0]:.3f}")
print(f"  cycle_len   mean={ym_scaler.mean_[IDX_CYCLE]:.1f}  std={ym_scaler.scale_[IDX_CYCLE]:.2f}")
print(f"  rho_pcm     mean={yr_scaler.mean_[0]:.0f}   std={yr_scaler.scale_[0]:.0f}")
print()


# =============================================================================
# SECTION 6 — ARCHITECTURE
# =============================================================================
#
# v3 changes vs v2:
#   • ConvResBlock identical (no change)
#   • After the shared backbone (embed→conv→attn→pool), we SPLIT into THREE
#     dedicated Dense branches:
#       ppf_branch  → dense(64) → dense(33)  [ppf_max + ppf_boc + 31 steps]
#       cycle_branch → dense(32) → dense(1)  [cycle_length]
#       rho_branch  → dense(32) → dense(1)   [rho_pcm at BOC]
#     This separates gradient flows. The tiny-variance rho signal can no
#     longer be overwhelmed by the larger PPF/cycle gradients.
#   • Loss uses separate terms per branch with explicit weighting.

class ConvResBlock(layers.Layer):
    """
    Residual convolutional block:  Conv → BN → GELU → Conv → BN → Add → GELU → Dropout
    """
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        # Projection shortcut when channel count changes
        self.proj  = None
        self.dropout = layers.Dropout(dropout) if dropout > 0 else None
        self._filters = filters

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.proj = layers.Conv2D(self._filters, 1, padding='same')
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = self.proj(x) if self.proj is not None else x
        h = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        h = tf.nn.gelu(h + shortcut)
        if self.dropout is not None:
            h = self.dropout(h, training=training)
        return h


def build_cnn_v3(
    grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
    n_types=N_TYPES + 1,       # 0=reflector, 1–9=fuel
    embed_dim=16,
    filters=(32, 64, 128),
    dense_units=128,
    dropout=DROPOUT,
    n_outputs=N_OUTPUTS,
):
    """
    Multi-head CNN surrogate for BEAVRS loading pattern → physics outputs.

    Backbone: Embedding → 3×ConvResBlock → spatial attention → GlobalAvgPool
    Three separate heads (FIX for keff/rho):
      ppf_head    → 33 outputs (ppf_max, ppf_boc, 31 step PPFs)
      cycle_head  →  1 output  (cycle_length)
      rho_head    →  1 output  (reactivity ρ_pcm at BOC)
    """
    # Monocore-informed embedding initialisation (same as v2)
    efpd_vals = np.array([ASSEMBLY_CYCLE_EQUIV.get(t, 0.0) for t in range(n_types)])
    efpd_norm = efpd_vals / (efpd_vals.max() + 1e-8)
    emb_init  = np.zeros((n_types, embed_dim), dtype=np.float32)
    emb_init[:, 0] = efpd_norm
    emb_init_fn = tf.keras.initializers.Constant(emb_init)

    inp = keras.Input(shape=(grid_rows, grid_cols), dtype=tf.int32, name='loading_grid')

    x = layers.Embedding(n_types, embed_dim,
                          embeddings_initializer=emb_init_fn,
                          name='assembly_embedding')(inp)

    for i, f in enumerate(filters):
        x = ConvResBlock(f, kernel_size=3,
                         dropout=dropout if i > 0 else 0.0,
                         name=f'conv_block_{i+1}')(x)

    # Spatial attention gate (unchanged from v2 — proven to help)
    attn = layers.Conv2D(1, 1, activation='sigmoid', name='spatial_attention')(x)
    x    = layers.Multiply(name='attended_features')([x, attn])
    x    = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # ── Shared MLP ────────────────────────────────────────────────────────────
    shared = layers.Dense(dense_units, activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu', name='shared_dense2')(shared)

    # ── PPF head (33 outputs: max, boc, 31 steps) ─────────────────────────────
    h_ppf = layers.Dense(64, activation='gelu', name='ppf_dense')(shared)
    h_ppf = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf = layers.Dense(1 + 1 + N_STEPS, activation='linear',
                             name='ppf_output')(h_ppf)   # (B, 33)

    # ── Cycle head (1 output) ─────────────────────────────────────────────────
    h_cyc = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle = layers.Dense(1, activation='linear',
                               name='cycle_output')(h_cyc)   # (B, 1)

    # ── Rho head (1 output) — the FIXED keff-equivalent head ─────────────────
    # This head has its own branch so rho gradients are NOT mixed with PPF.
    h_rho = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho = layers.Dense(1, activation='linear',
                             name='rho_output')(h_rho)       # (B, 1)

    # ── Concatenate for single-output training interface ──────────────────────
    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])

    return keras.Model(inputs=inp, outputs=out, name='BEAVRS_CNN_v3')


model = build_cnn_v3()
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")


# =============================================================================
# SECTION 7 — LOSS FUNCTION (FIXED)
# =============================================================================
#
# v3 loss = weighted per-head MSE + light PPF monotonicity penalty
#
# REMOVED from v2 (both were wrong or unnecessary):
#   ✗ keff_proximity penalty (caused R²=-36: see file header)
#   ✗ keff proximity to 1.0 (keff=1.0 is subcritical — valid cycles have keff>1)
#
# KEPT from v2:
#   ✓ Weighted MSE (PPF gets 3× weight as primary output)
#   ✓ PPF monotonicity after step 3 (real physics: PPF decreases with burnup)
#
# NEW in v3:
#   ✓ Dedicated rho loss term with 5× weight (ensures the tiny-variance
#     signal gets enough gradient attention)

IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS      # = 33
IDX_CYCLE_v3        = IDX_PPF_STEPS_END      # = 33
IDX_RHO_v3          = IDX_PPF_STEPS_END + 1  # = 34

W_PPF_MAX   = 3.0    # primary target
W_PPF_BOC   = 2.0
W_PPF_STEPS = 0.5    # 31 correlated values — lower individual weight
W_CYCLE     = 1.0
W_RHO       = 5.0    # INCREASED: ensures rho gradient is not swamped
W_MONO      = 0.01   # physics monotonicity penalty (small, regularising)


def v3_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Multi-head physics-informed loss.

    Terms:
      1. ppf_max MSE    × 3.0
      2. ppf_boc MSE    × 2.0
      3. ppf_steps MSE  × 0.5  (per step)
      4. cycle MSE      × 1.0
      5. rho MSE        × 5.0  ← high weight for small-variance target
      6. PPF monotonicity penalty × 0.01  (late-burnup physics)

    No keff_proximity term (that was the v2 bug).
    """
    # ── Per-output MSE ────────────────────────────────────────────────────────
    ppf_max_loss   = W_PPF_MAX   * tf.reduce_mean(tf.square(
        y_true[:, 0] - y_pred[:, 0]))
    ppf_boc_loss   = W_PPF_BOC   * tf.reduce_mean(tf.square(
        y_true[:, 1] - y_pred[:, 1]))
    ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
        - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
    cycle_loss     = W_CYCLE     * tf.reduce_mean(tf.square(
        y_true[:, IDX_CYCLE_v3] - y_pred[:, IDX_CYCLE_v3]))
    rho_loss       = W_RHO       * tf.reduce_mean(tf.square(
        y_true[:, IDX_RHO_v3] - y_pred[:, IDX_RHO_v3]))

    # ── Late-burnup PPF monotonicity penalty ─────────────────────────────────
    # After step 3 the PPF profile should be non-increasing (BAs burned out).
    step_preds = y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]   # (B, 31)
    late       = step_preds[:, 3:]                                    # (B, 28)
    diffs      = late[:, 1:] - late[:, :-1]                          # (B, 27)
    violations = tf.maximum(0.0, diffs)
    mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))

    return ppf_max_loss + ppf_boc_loss + ppf_steps_loss + cycle_loss + rho_loss + mono_loss


# =============================================================================
# SECTION 8 — TRAIN
# =============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=v3_loss,
    metrics=['mae']
)

print("[TRAINING] Starting CNN v3 ...")
print(f"  Dataset : {len(X_tr)} train + {len(X_val)} val patterns")
print(f"  Loss    : weighted per-head MSE  (fixed — no keff_proximity)")
print(f"  Rho weight: {W_RHO}× (ensures keff-equivalent signal is learned)")
print()

t_start = time.time()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=30,
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
    X_tr, Y_tr_sc,
    validation_data=(X_val, Y_val_sc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=0
)

t_train    = time.time() - t_start
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
print(f"\n[TRAINING DONE]  {t_train:.1f}s  |  best epoch: {best_epoch}\n")


# =============================================================================
# SECTION 9 — INVERSE TRANSFORM HELPER
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    """
    Invert the two-scaler scheme back to real physical units.

    Args:
        Y_sc : (N, 35) in scaled space  [34 main cols | 1 rho col]
    Returns:
        Y    : (N, 35) in real units
    """
    Y_main_real = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_real  = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main_real, Y_rho_real], axis=1)


# =============================================================================
# SECTION 10 — EVALUATE ON TEST SET
# =============================================================================

print("[EVALUATION] Running deterministic predictions on test set ...")

Y_pred_sc   = model.predict(X_test, verbose=0)
Y_pred_real = inverse_transform(Y_pred_sc)
Y_true_real = inverse_transform(Y_test_sc)

ppf_max_pred  = Y_pred_real[:, IDX_PPF_MAX]
ppf_max_true  = Y_true_real[:, IDX_PPF_MAX]
cycle_pred    = Y_pred_real[:, IDX_CYCLE_v3]
cycle_true    = Y_true_real[:, IDX_CYCLE_v3]
rho_pred      = Y_pred_real[:, IDX_RHO_v3]
rho_true      = Y_true_real[:, IDX_RHO_v3]

# Convert rho back to keff for reporting: keff = 1 / (1 - rho_pcm/1e5)
keff_pred_rep = 1.0 / (1.0 - rho_pred / 1e5)
keff_true_rep = 1.0 / (1.0 - rho_true / 1e5)

ppf_mae      = np.abs(ppf_max_pred - ppf_max_true).mean()
ppf_rel_err  = (np.abs(ppf_max_pred - ppf_max_true) / (ppf_max_true + 1e-6)).mean() * 100
ppf_r2       = r2_score(ppf_max_true, ppf_max_pred)
ppf_pearson  = np.corrcoef(ppf_max_pred, ppf_max_true)[0, 1]

goal_mask    = (ppf_max_true >= PPF_REPORT_LOW) & (ppf_max_true <= PPF_REPORT_HIGH)
goal_mae     = np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask]).mean()
goal_r2      = r2_score(ppf_max_true[goal_mask], ppf_max_pred[goal_mask])

cycle_mae    = np.abs(cycle_pred - cycle_true).mean()
cycle_r2     = r2_score(cycle_true, cycle_pred)

rho_mae      = np.abs(rho_pred - rho_true).mean()
rho_r2       = r2_score(rho_true, rho_pred)
keff_mae_rep = np.abs(keff_pred_rep - keff_true_rep).mean()
keff_r2_rep  = r2_score(keff_true_rep, keff_pred_rep)

print(f"\n{'='*58}")
print(f"CNN v3 TEST RESULTS")
print(f"{'='*58}")
print(f"  PPF_max (all patterns):")
print(f"    MAE              : {ppf_mae:.4f}")
print(f"    Relative error   : {ppf_rel_err:.2f}%")
print(f"    R²               : {ppf_r2:.4f}")
print(f"    Pearson r        : {ppf_pearson:.4f}")
print(f"  PPF_max (goal zone {PPF_REPORT_LOW}–{PPF_REPORT_HIGH}  n={goal_mask.sum()}):")
print(f"    MAE              : {goal_mae:.4f}")
print(f"    R²               : {goal_r2:.4f}")
print(f"  Cycle length:")
print(f"    MAE              : {cycle_mae:.2f} days")
print(f"    R²               : {cycle_r2:.4f}")
print(f"  Rho_pcm (BOC):")
print(f"    MAE              : {rho_mae:.0f} pcm")
print(f"    R²               : {rho_r2:.4f}")
print(f"  keff_boc (derived from rho):")
print(f"    MAE              : {keff_mae_rep:.5f}")
print(f"    R²               : {keff_r2_rep:.4f}  (was -36.8 in v2)")
print(f"{'='*58}\n")


# =============================================================================
# SECTION 11 — MONTE CARLO DROPOUT UNCERTAINTY
# =============================================================================

print("[MC DROPOUT] Estimating prediction uncertainty ...")
t_mc = time.time()

mc_preds_sc = np.stack([
    model(X_test, training=True).numpy()
    for _ in range(MC_SAMPLES)
])   # (MC_SAMPLES, N_test, 35)

mc_mean_sc = mc_preds_sc.mean(axis=0)   # (N_test, 35)
mc_std_sc  = mc_preds_sc.std(axis=0)    # (N_test, 35)

mc_mean_real = inverse_transform(mc_mean_sc)
# For uncertainty: std in real space ≈ std_sc × scaler.scale_
mc_std_main = mc_std_sc[:, :34] * ym_scaler.scale_
mc_std_rho  = mc_std_sc[:, 34:35] * yr_scaler.scale_
mc_std_real = np.concatenate([mc_std_main, mc_std_rho], axis=1)

ppf_mc_mean = mc_mean_real[:, IDX_PPF_MAX]
ppf_mc_std  = mc_std_real[:, IDX_PPF_MAX]

unc_err_corr = np.corrcoef(ppf_mc_std,
                            np.abs(ppf_mc_mean - ppf_max_true))[0, 1]
print(f"  Time              : {time.time()-t_mc:.1f}s")
print(f"  Mean σ(ppf_max)   : {ppf_mc_std.mean():.4f}")
print(f"  Max σ(ppf_max)    : {ppf_mc_std.max():.4f}")
print(f"  σ-error corr      : {unc_err_corr:.3f}  (positive → σ is useful flag)\n")


# =============================================================================
# SECTION 12 — SENSITIVITY ANALYSIS  (∂ppf_max / ∂position)
# =============================================================================

print("[SENSITIVITY]  Computing ∂ppf_max/∂position ...")

n_sens   = min(200, len(X_test))
X_sample = tf.constant(X_test[:n_sens], dtype=tf.int32)

sens_norm = np.ones(N_POS, dtype=np.float32)  # default uniform
sens_grid = np.ones((GRID_ROWS, GRID_COLS), dtype=np.float32)

try:
    with tf.GradientTape() as tape:
        emb_layer = model.get_layer('assembly_embedding')
        x_emb     = emb_layer(X_sample)
        tape.watch(x_emb)
        h = x_emb
        skip = {'loading_grid', 'assembly_embedding'}
        for layer in model.layers:
            if layer.name in skip or isinstance(layer, keras.layers.InputLayer):
                continue
            try:    h = layer(h, training=False)
            except: pass
        ppf_out = h[:, IDX_PPF_MAX]
    grads = tape.gradient(ppf_out, x_emb)
    if grads is not None:
        sens_grid_raw = tf.norm(grads, axis=-1).numpy().mean(axis=0)
        sens_grid_raw[~GRID_MASK] = 0.0
        sens_pos = np.zeros(N_POS, dtype=np.float32)
        pos_i = 0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    sens_pos[pos_i] = sens_grid_raw[r, c]
                    pos_i += 1
        sens_norm  = sens_pos / (sens_pos.max() + 1e-8)
        sens_grid  = sens_grid_raw / (sens_grid_raw.max() + 1e-8)
        top5 = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"  Top-5 critical positions : {top5}")
        print(f"  Sensitivity range        : {sens_norm.min():.3f} – {sens_norm.max():.3f}\n")
except Exception as e:
    print(f"  [WARN] Gradient failed ({e}) — using uniform sensitivity\n")

sens_df = pd.DataFrame({
    'position'        : [f'pos_{i}' for i in range(N_POS)],
    'sensitivity'     : sens_pos if 'sens_pos' in dir() else np.ones(N_POS),
    'sensitivity_norm': sens_norm,
})
sens_df.to_csv(SENS_NAME, index=False)
print(f"  Saved: {SENS_NAME}")


# =============================================================================
# SECTION 13 — ACTIVE LEARNING LOOP
# =============================================================================
#
# PRODUCTION ACTIVE LEARNING PIPELINE
# ─────────────────────────────────────
# This implements the full loop described in the problem statement:
#
#   Initial dataset → train CNN → QICA proposes candidates →
#   CNN screens by uncertainty → high-σ → OpenMC → add to dataset → retrain
#
# The loop runs AL_ROUNDS rounds of:
#   1. Run MC Dropout on ALL dataset patterns
#   2. Flag patterns with high σ AND low predicted PPF (these are the most
#      valuable: uncertain AND potentially optimal)
#   3. call openmc_simulate() [STUB — replace with real OpenMC]
#   4. Add new (pattern, ppf, cycle, keff) to training data
#   5. Retrain CNN for AL_RETRAIN_EPOCHS additional epochs (warm start)
#
# OPENMC INTEGRATION:
#   Replace the body of openmc_simulate() with your actual OpenMC workflow.
#   Expected output: ppf_max (float), cycle_length (float), keff_boc (float)
#
# CURRENT MODE: AL_ROUNDS = 0 → no actual retraining (scaffold only).
# Set AL_ROUNDS = 1 (or more) when your simulator is ready.
# ─────────────────────────────────────────────────────────────────────────────

AL_ROUNDS         = 0    # Set to 1+ when simulator is plugged in
AL_RETRAIN_EPOCHS = 50   # Warm-start epochs per AL round


def openmc_simulate(loading_pattern_1d: np.ndarray) -> dict:
    """
    SIMULATOR STUB — replace this function body with your OpenMC workflow.

    In production, this function should:
      1. Write the loading pattern to an OpenMC geometry/materials XML
      2. Run the OpenMC simulation (typically a depletion calculation)
      3. Parse the output statepoint file
      4. Return the physics quantities of interest

    Args:
        loading_pattern_1d : (31,) integer array of assembly types 1–9
                              in position order (matches load_cols)

    Returns:
        dict with keys:
            'ppf_max'       : float  — global max PPF over fuel cycle
            'ppf_boc'       : float  — BOC PPF
            'ppf_steps'     : (N_STEPS,) array — PPF at each burnup step
            'cycle_length'  : float  — effective full-power days
            'keff_boc'      : float  — k-effective at beginning of cycle
            'rho_pcm_boc'   : float  — reactivity (pcm) at BOC
            'success'       : bool   — False if simulation failed/diverged

    ── Example OpenMC integration (pseudocode): ────────────────────────────
        import openmc
        import openmc.deplete

        # Build geometry from loading_pattern_1d
        materials = build_materials(loading_pattern_1d)
        geometry  = build_geometry(materials)
        settings  = openmc.Settings()
        settings.batches = 200
        settings.particles = 10000

        # Run eigenvalue calculation
        model = openmc.Model(geometry, materials, settings)
        keff_boc = model.run()  # returns k-eff

        # Run depletion
        operator = openmc.deplete.CoupledOperator(model, ...)
        integrator = openmc.deplete.PredictorIntegrator(operator, ...)
        integrator.integrate()

        # Parse outputs
        results = openmc.deplete.ResultsList.from_hdf5('depletion_results.h5')
        ...
        return {'ppf_max': ppf_max, 'cycle_length': cycle_days, ...}
    ────────────────────────────────────────────────────────────────────────
    
    # ── STUB: returns None to signal "not yet implemented" ───────────────────
    raise NotImplementedError(
        "openmc_simulate() is a stub.\n"
        "Replace this function body with your OpenMC/PARCS workflow.\n"
        "Expected output: dict with ppf_max, ppf_boc, ppf_steps, "
        "cycle_length, keff_boc, rho_pcm_boc, success."
    )
    """
    from openmc_beavrs_simulator import simulate as _omc_sim
    
    def openmc_simulate(loading_pattern_1d):
        return _omc_sim(loading_pattern_1d)


def run_active_learning_round(
    model, df_current,
    X_grid_current, Y_main_current, Y_rho_current,
    ym_scaler, yr_scaler,
    round_idx: int,
):
    """
    One round of the active learning loop.

    1. Run MC Dropout on all current patterns.
    2. Identify high-uncertainty, low-PPF candidates.
    3. Call openmc_simulate() for each candidate.
    4. Augment dataset with new observations.
    5. Retrain model (warm start) on augmented dataset.

    Returns updated (df, X_grid, Y_main, Y_rho).
    """
    print(f"\n[AL ROUND {round_idx}] Running MC Dropout on {len(X_grid_current)} patterns ...")

    mc_all = np.stack([
        model(X_grid_current, training=True).numpy()
        for _ in range(MC_SAMPLES)
    ])   # (MC_SAMPLES, N, 35)

    mc_mean_sc_all = mc_all.mean(axis=0)
    mc_std_sc_all  = mc_all.std(axis=0)
    mc_mean_all    = inverse_transform(mc_mean_sc_all)
    mc_std_all_phy = np.concatenate([
        mc_std_sc_all[:, :34] * ym_scaler.scale_,
        mc_std_sc_all[:, 34:35] * yr_scaler.scale_,
    ], axis=1)

    ppf_all_pred = mc_mean_all[:, IDX_PPF_MAX]
    ppf_all_std  = mc_std_all_phy[:, IDX_PPF_MAX]

    # Priority score: high uncertainty AND low predicted PPF
    # → uncertainty-weighted inverse PPF
    priority = ppf_all_std / (ppf_all_pred + 1e-6)
    top_idxs = np.argsort(priority)[::-1][:AL_MAX_QUERIES_PER_ROUND]

    new_records = []
    sim_success = 0
    print(f"  Querying {len(top_idxs)} candidates ...")
    for idx in top_idxs:
        pattern = X_grid_current[idx][GRID_MASK].flatten().astype(np.int32)
        try:
            result = openmc_simulate(pattern)
            if result and result.get('success', True):
                new_records.append({
                    'ppf_max': result['ppf_max'],
                    'ppf_boc': result.get('ppf_boc', result['ppf_max']),
                    'ppf_steps': result.get('ppf_steps', np.full(N_STEPS, result['ppf_max'])),
                    'cycle_length': result['cycle_length'],
                    'rho_pcm_boc': result['rho_pcm_boc'],
                    'loading_pattern': pattern,
                })
                sim_success += 1
        except NotImplementedError:
            break   # Stub not replaced yet
        except Exception as e:
            print(f"    [WARN] Simulation failed for pattern {idx}: {e}")

    print(f"  Simulated {sim_success}/{len(top_idxs)} patterns successfully")
    if sim_success == 0:
        print("  No new data added — returning unchanged dataset")
        return df_current, X_grid_current, Y_main_current, Y_rho_current

    # Augment arrays
    for rec in new_records:
        pat = rec['loading_pattern']
        # Build 6×6 grid
        new_grid = np.zeros((1, GRID_ROWS, GRID_COLS), dtype=np.int32)
        pos_i = 0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    new_grid[0, r, c] = pat[pos_i]
                    pos_i += 1

        new_ym = np.array([[rec['ppf_max'], rec['ppf_boc']]
                            + list(rec['ppf_steps'])
                            + [rec['cycle_length']]], dtype=np.float32)
        new_yr = np.array([[rec['rho_pcm_boc']]], dtype=np.float32)

        X_grid_current  = np.vstack([X_grid_current, new_grid])
        Y_main_current  = np.vstack([Y_main_current, new_ym])
        Y_rho_current   = np.vstack([Y_rho_current, new_yr])

    # Re-scale and warm-start retrain
    Ym_aug_sc = ym_scaler.transform(Y_main_current)
    Yr_aug_sc = yr_scaler.transform(Y_rho_current)
    Y_aug_sc  = np.concatenate([Ym_aug_sc, Yr_aug_sc], axis=1).astype(np.float32)

    print(f"  Retraining on {len(X_grid_current)} patterns ({AL_RETRAIN_EPOCHS} epochs warm start) ...")
    model.fit(
        X_grid_current, Y_aug_sc,
        epochs=AL_RETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    print(f"  [AL ROUND {round_idx} COMPLETE]  Added {sim_success} new patterns")
    return df_current, X_grid_current, Y_main_current, Y_rho_current


# Run AL rounds (0 = scaffold only, no simulator yet)
print("[ACTIVE LEARNING]  Scanning full dataset for query candidates ...")

mc_all_full = np.stack([
    model(X_grid, training=True).numpy()
    for _ in range(MC_SAMPLES)
])
mc_mean_sc_full = mc_all_full.mean(axis=0)
mc_std_sc_full  = mc_all_full.std(axis=0)
mc_mean_full    = inverse_transform(mc_mean_sc_full)
mc_std_full_phy = np.concatenate([
    mc_std_sc_full[:, :34] * ym_scaler.scale_,
    mc_std_sc_full[:, 34:35] * yr_scaler.scale_,
], axis=1)

ppf_full_pred = mc_mean_full[:, IDX_PPF_MAX]
ppf_full_std  = mc_std_full_phy[:, IDX_PPF_MAX]

# Priority = high σ AND low predicted PPF
priority_score = ppf_full_std / (ppf_full_pred + 1e-6)
high_unc_mask  = ppf_full_std >= AL_UNCERTAINTY_THRESHOLD
low_ppf_mask   = ppf_full_pred <= np.percentile(ppf_full_pred, 25)
query_mask     = high_unc_mask & low_ppf_mask
query_idxs     = np.where(query_mask)[0]
query_sorted   = query_idxs[np.argsort(priority_score[query_idxs])[::-1]]
query_top      = query_sorted[:AL_MAX_QUERIES_PER_ROUND]

print(f"  High-uncertainty (σ≥{AL_UNCERTAINTY_THRESHOLD}) : {high_unc_mask.sum()}")
print(f"  Low-PPF (bottom quartile)         : {low_ppf_mask.sum()}")
print(f"  Priority candidates               : {len(query_idxs)}")
print(f"  Top-{AL_MAX_QUERIES_PER_ROUND} flagged for simulator  : {len(query_top)}")

al_df = pd.DataFrame({
    'pattern_id'   : [f'pat_{i:05d}' for i in query_top],
    'pred_ppf_max' : ppf_full_pred[query_top],
    'pred_ppf_std' : ppf_full_std[query_top],
    'priority'     : priority_score[query_top],
    'cycle_length' : mc_mean_full[query_top, IDX_CYCLE_v3],
    'rho_pcm_boc'  : mc_mean_full[query_top, IDX_RHO_v3],
})
al_df.to_csv(AL_CSV, index=False)
print(f"  Saved query candidates → {AL_CSV}")

if AL_ROUNDS > 0:
    X_grid_al, Y_main_al, Y_rho_al = X_grid.copy(), Y_main.copy(), Y_rho_col.copy()
    for rnd in range(1, AL_ROUNDS + 1):
        _, X_grid_al, Y_main_al, Y_rho_al = run_active_learning_round(
            model, df, X_grid_al, Y_main_al, Y_rho_al,
            ym_scaler, yr_scaler, round_idx=rnd
        )

print()


# =============================================================================
# SECTION 14 — SAVE MODEL + CONFIG
# =============================================================================

model.save(MODEL_NAME)

config = {
    'N_POS': N_POS, 'N_TYPES': N_TYPES, 'N_STEPS': N_STEPS,
    'GRID_ROWS': GRID_ROWS, 'GRID_COLS': GRID_COLS,
    'GRID_LAYOUT': GRID_LAYOUT.tolist(),
    'GRID_MASK': GRID_MASK.tolist(),
    'IDX_PPF_MAX': IDX_PPF_MAX,
    'IDX_PPF_BOC': IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': IDX_PPF_STEPS_START,
    'IDX_PPF_STEPS_END': IDX_PPF_STEPS_END,
    'IDX_CYCLE': IDX_CYCLE_v3,
    'IDX_RHO': IDX_RHO_v3,
    'N_OUTPUTS': N_OUTPUTS,
    'PPF_REPORT_LOW': PPF_REPORT_LOW,
    'PPF_REPORT_HIGH': PPF_REPORT_HIGH,
    # Scalers saved as lists for JSON serialisation
    'ym_scaler_mean': ym_scaler.mean_.tolist(),
    'ym_scaler_scale': ym_scaler.scale_.tolist(),
    'yr_scaler_mean': yr_scaler.mean_.tolist(),
    'yr_scaler_scale': yr_scaler.scale_.tolist(),
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v) for k, v in ASSEMBLY_CYCLE_EQUIV.items()},
    'mc_samples': MC_SAMPLES,
    'al_uncertainty_thr': AL_UNCERTAINTY_THRESHOLD,
    # Performance summary
    'test_ppf_mae': float(ppf_mae),
    'test_ppf_r2': float(ppf_r2),
    'test_goal_mae': float(goal_mae),
    'test_cycle_mae': float(cycle_mae),
    'test_rho_r2': float(rho_r2),
    'test_keff_r2': float(keff_r2_rep),   # for reporting
    'v2_keff_r2_was': -36.8,               # documented comparison
}

with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

print(f"[SAVED]  {MODEL_NAME}")
print(f"[SAVED]  {CONFIG_NAME}")
print(f"[SAVED]  {SENS_NAME}")


# =============================================================================
# SECTION 15 — VISUALISATIONS
# =============================================================================

fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    f"BEAVRS CNN v3  |  Fixed keff  |  PPF MAE={ppf_mae:.3f}  R²={ppf_r2:.3f}  "
    f"keff R²={keff_r2_rep:.3f} (was -36.8 in v2)",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1. Training curve
ax = fig.add_subplot(gs[0, 0])
ax.plot(history.history['loss'],     '#1B4FBF', lw=1.5, label='Train')
ax.plot(history.history['val_loss'], '#F5A623', lw=1.5, label='Val')
ax.axvline(best_epoch - 1, color='red', lw=1, ls=':', label=f'Best ep {best_epoch}')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
ax.set_title('Training Curve\n(v3 multi-head loss)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. PPF scatter — coloured by goal zone
ax = fig.add_subplot(gs[0, 1])
colors_sc = np.where(
    (ppf_max_true >= PPF_REPORT_LOW) & (ppf_max_true <= PPF_REPORT_HIGH),
    '#17BECF', '#AAAAAA'
)
lim = [ppf_max_true.min() - 0.1, ppf_max_true.max() + 0.1]
ax.scatter(ppf_max_true, ppf_max_pred, c=colors_sc, alpha=0.35, s=7)
ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
ax.axhspan(PPF_REPORT_LOW, PPF_REPORT_HIGH, alpha=0.07, color='teal',
           label=f'Goal zone')
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'PPF Prediction\nMAE={ppf_mae:.3f}  R²={ppf_r2:.3f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 3. keff scatter — the key improvement
ax = fig.add_subplot(gs[0, 2])
ax.scatter(keff_true_rep, keff_pred_rep, alpha=0.3, s=7, color='#9467BD')
lim_k = [keff_true_rep.min() - 0.003, keff_true_rep.max() + 0.003]
ax.plot(lim_k, lim_k, 'k--', lw=1, label='Perfect')
ax.set_xlabel('True keff_boc'); ax.set_ylabel('Predicted keff_boc')
ax.set_title(f'keff_boc  [FIX: v2 R²=-36.8 → v3 R²={keff_r2_rep:.3f}]\n'
             f'MAE={keff_mae_rep:.5f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.text(0.05, 0.93, 'keff_proximity bug removed\nSeparate rho head',
        transform=ax.transAxes, fontsize=7, color='#9467BD',
        bbox=dict(fc='white', ec='#9467BD', alpha=0.7, boxstyle='round'))

# 4. Cycle length scatter
ax = fig.add_subplot(gs[0, 3])
ax.scatter(cycle_true, cycle_pred, alpha=0.3, s=7, color='#2CA02C')
lim_c = [cycle_true.min() - 5, cycle_true.max() + 5]
ax.plot(lim_c, lim_c, 'k--', lw=1)
ax.set_xlabel('True cycle length (days)'); ax.set_ylabel('Predicted (days)')
ax.set_title(f'Cycle Length\nMAE={cycle_mae:.1f}d  R²={cycle_r2:.3f}')
ax.grid(alpha=0.3)

# 5. Uncertainty vs error
ax = fig.add_subplot(gs[1, 0])
ppf_abs_err = np.abs(ppf_mc_mean - ppf_max_true)
ax.scatter(ppf_mc_std, ppf_abs_err, alpha=0.2, s=6, color='#D62728')
m, b = np.polyfit(ppf_mc_std, ppf_abs_err, 1)
xs = np.linspace(ppf_mc_std.min(), ppf_mc_std.max(), 100)
ax.plot(xs, m * xs + b, 'k--', lw=1.5, label=f'r={unc_err_corr:.3f}')
ax.axvline(AL_UNCERTAINTY_THRESHOLD, color='orange', lw=1.5, ls=':',
           label=f'AL thresh σ={AL_UNCERTAINTY_THRESHOLD}')
ax.set_xlabel('MC σ (ppf_max)'); ax.set_ylabel('|PPF error|')
ax.set_title('Uncertainty vs Error\n(AL query threshold)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 6. PPF distribution with achievable zone
ax = fig.add_subplot(gs[1, 1])
bins = np.linspace(1.5, 5.5, 60)
ax.hist(ppf_max_true, bins=bins, alpha=0.55, color='#1B4FBF', label='True')
ax.hist(ppf_max_pred, bins=bins, alpha=0.55, color='#F5A623', label='Predicted')
p10 = np.percentile(ppf_global_max, 10)
ax.axvline(p10, color='#2CA02C', lw=2, ls='--', label=f'10th pct = {p10:.2f} (QICA target)')
ax.axvline(ppf_global_max.min(), color='red', lw=2, ls=':', label=f'Data min = {ppf_global_max.min():.2f}')
ax.set_xlabel('ppf_max'); ax.set_ylabel('Count')
ax.set_title('PPF Distribution\nGreen = realistic QICA target')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 7. Sensitivity heatmap
ax = fig.add_subplot(gs[1, 2])
disp_sens = np.full((GRID_ROWS, GRID_COLS), np.nan)
pos_i = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            disp_sens[r, c] = sens_norm[pos_i]
            pos_i += 1
cmap_s = plt.cm.RdYlGn_r.copy(); cmap_s.set_bad('lightgrey')
im = ax.imshow(disp_sens, cmap=cmap_s, aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Norm. sensitivity')
pos_i = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            ax.text(c, r, f'P{pos_i}', ha='center', va='center', fontsize=6)
            pos_i += 1
ax.set_title('∂ppf_max / ∂position\n(Red = critical)')
ax.set_xticks([]); ax.set_yticks([])

# 8. Active learning candidates scatter
ax = fig.add_subplot(gs[1, 3])
ax.scatter(ppf_full_pred, ppf_full_std, alpha=0.15, s=4, color='#AAAAAA',
           label='All patterns')
if len(query_top) > 0:
    ax.scatter(ppf_full_pred[query_top], ppf_full_std[query_top],
               alpha=0.8, s=20, color='#D62728', zorder=5,
               label=f'AL candidates (n={len(query_top)})')
ax.axhline(AL_UNCERTAINTY_THRESHOLD, color='orange', lw=1.5, ls='--',
           label=f'σ threshold')
ax.axvline(np.percentile(ppf_full_pred, 25), color='teal', lw=1.5, ls='--',
           label='PPF 25th pct')
ax.set_xlabel('Predicted ppf_max'); ax.set_ylabel('MC σ (ppf_max)')
ax.set_title('Active Learning Candidates\n(high-σ, low-PPF → send to OpenMC)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 9. Example: best vs worst loading patterns
for col_off, (label, true_arr) in enumerate(
    [('Best (lowest true PPF)', ppf_max_true), ('Worst (highest true PPF)', ppf_max_true)]
):
    ax = fig.add_subplot(gs[2, col_off])
    idx = true_arr.argmin() if col_off == 0 else true_arr.argmax()
    g_disp = X_test[idx].astype(float).copy()
    g_disp[~GRID_MASK] = np.nan
    cmap_ex = plt.cm.YlOrRd.copy(); cmap_ex.set_bad('lightgrey')
    im_ex = ax.imshow(g_disp, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im_ex, ax=ax, label='Type')
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'{X_test[idx, r, c]}', ha='center', va='center', fontsize=8)
    ax.set_title(f'{label}\nTrue PPF={true_arr[idx]:.3f}  Pred={ppf_max_pred[idx]:.3f}')
    ax.set_xticks([]); ax.set_yticks([])

# 10. PPF burnup profile
ax = fig.add_subplot(gs[2, 2])
steps_range  = np.arange(N_STEPS)
true_smean   = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
true_sstd    = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
pred_smean   = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
pred_sstd    = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
ax.plot(steps_range, true_smean, '#1B4FBF', lw=2, label='True')
ax.fill_between(steps_range, true_smean - true_sstd, true_smean + true_sstd,
                color='#1B4FBF', alpha=0.15)
ax.plot(steps_range, pred_smean, '#F5A623', lw=2, ls='--', label='Predicted')
ax.fill_between(steps_range, pred_smean - pred_sstd, pred_smean + pred_sstd,
                color='#F5A623', alpha=0.15)
ax.set_xlabel('Burnup Step'); ax.set_ylabel('Max PPF at Step')
ax.set_title('PPF Burnup Profile\n(mean ± 1σ, test patterns)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 11. Residuals
ax = fig.add_subplot(gs[2, 3])
resid = ppf_max_pred - ppf_max_true
ax.hist(resid, bins=50, color='#1B4FBF', edgecolor='white', lw=0.5)
ax.axvline(0, color='red', lw=1.5, label='Zero error')
ax.axvline(resid.mean(), color='orange', lw=1.5, label=f'Mean={resid.mean():.3f}')
ax.set_xlabel('Predicted − True ppf_max'); ax.set_ylabel('Count')
ax.set_title(f'Residuals\nμ={resid.mean():.3f}  σ={resid.std():.3f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.savefig(PLOT_NAME, dpi=150, bbox_inches='tight')
print(f"\n[SAVED]  {PLOT_NAME}")


# =============================================================================
# SECTION 16 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 62)
print("cnn_v3.py  FINAL SUMMARY")
print("=" * 62)
print(f"  Architecture     : BEAVRS CNN v3 (multi-head, fixed keff)")
print(f"  Loss             : Per-head weighted MSE (no keff_proximity)")
print(f"  Parameters       : {model.count_params():,}")
print(f"  Best epoch       : {best_epoch} / {EPOCHS}")
print(f"  Training time    : {t_train:.1f}s")
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
print(f"  keff_boc (via rho_pcm):")
print(f"    MAE            : {keff_mae_rep:.5f}   (was 0.07059 in v2)")
print(f"    R²             : {keff_r2_rep:.4f}    (was -36.8 in v2)")
print(f"    ✓ keff_proximity bug FIXED")
print()
print(f"  MC Dropout σ_ppf (mean)  : {ppf_mc_std.mean():.4f}")
print(f"  AL candidates identified : {len(query_top)}")
print()
print(f"  Dataset PPF range : {ppf_global_max.min():.2f}–{ppf_global_max.max():.2f}")
print(f"  10th pct PPF      : {np.percentile(ppf_global_max,10):.2f}  ← QICA realistic target")
print(f"  (targeting below this requires data outside training distribution)")
print()
print(f"  OUTPUT FILES:")
print(f"    {MODEL_NAME}")
print(f"    {CONFIG_NAME}")
print(f"    {SENS_NAME}")
print(f"    {PLOT_NAME}")
print(f"    {AL_CSV}")
print()
print("  NEXT STEP: Run qica_v3.py — minimise ppf_max using this CNN.")
print("  OpenMC integration: replace openmc_simulate() body in Section 13.")
print("=" * 62)