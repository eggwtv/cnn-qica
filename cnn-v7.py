"""
=============================================================================
cnn_v7.py  —  BEAVRS CNN v7  |  W_LOG=3.0  +  Inverse-PPF Sample Weighting
=============================================================================

WHAT CHANGED FROM v6
─────────────────────────────────────────────────────────────────────────────
CHANGE 1 — W_LOG raised from 2.0 → 3.0:
  v6 showed no improvement in relative error despite the log-space term.
  The MSE component (W_PPF_MAX=3.0) still dominated because errors in scaled
  space look the same regardless of the absolute PPF level.  Raising W_LOG
  to 3.0 shifts the gradient balance further toward relative-error correction.
  If relative error is still > 3.10% after this run, try W_PPF_MAX=1.5 with
  W_LOG=4.0 so the log term fully dominates for the ppf_max head.

CHANGE 2 — Inverse-PPF sample weighting (USE_SAMPLE_WEIGHTING = True):
  Each training sample is given weight:
      w_i = mean(ppf_max_train) / ppf_max_i,  then normalised so mean(w) = 1.
  Effect on gradient contributions:
      ppf_max = 2.0  →  w ≈ 1.7   (upweighted ~70%)
      ppf_max = 3.4  →  w ≈ 1.0   (neutral)
      ppf_max = 5.0  →  w ≈ 0.7   (downweighted ~30%)
  This makes the optimizer spend more gradient on the low-PPF region without
  changing the architecture or loss function.
  Toggle with USE_SAMPLE_WEIGHTING = True/False.

ARCHITECTURE (unchanged — best from v5 tuner, confirmed at 400 epochs):
  filters=(64, 128)    ← v5 Stage-2 best: '64_128' filter mode
  num_blocks=2         ← v5 Stage-2 best: 2 blocks beats 3 at full length
  dense_units=128      ← reverted: tuner found 64 but that underperformed at 400ep
  ppf_head_units=64    ← tuner result, kept
  No spatial attention ← confirmed OFF in v5 Stage-1 search
  Multi-head           ← confirmed ON in v5 Stage-1 search

  Input (6×6 int grid) → Embedding(10, 16) → ConvResBlock(64) → ConvResBlock(128)
  → GlobalAvgPool → Dense(128, gelu) → Dropout(0.2) → Dense(64, gelu)
  → [PPF head: Dense(64)→Dense(33)] + [Cycle head: Dense(32)→Dense(1)]
                                     + [Rho  head: Dense(32)→Dense(1)]
  → Concatenate → (B, 35)

LOSS (v7):
  L = W_PPF_MAX × MSE(ppf_max)
    + W_PPF_BOC × MSE(ppf_boc)
    + W_PPF_STEPS × MSE(ppf_steps)
    + W_CYCLE × MSE(cycle_length)
    + W_RHO × MSE(rho_pcm)
    + W_MONO × monotonicity penalty on late-cycle PPF
    + W_LOG(=3.0) × MSE(log(ppf_max_pred_real), log(ppf_max_true_real))

OUTPUTS:
  cnn_v7_model.keras
  cnn_v7_config.json
  cnn_v7_sens.csv
  cnn_v7_results.png
  cnn_v7_al_candidates.csv
  train_type_freq_v7.npy

BASELINES:
  v4 (MSE only)      : ppf_rel_err=3.10%  ppf_mae=0.0843  ppf_r2=0.9841
                       cycle_mae=1.28d    keff_r2=0.9912
  v5 (MAPE eps=0.5)  : ppf_rel_err=3.22%  ppf_mae=0.0875  ppf_r2=0.9826
                       cycle_mae=1.61d    keff_r2=0.9889
  v6 (log-MSE W=2.0) : ppf_rel_err=3.25%  ppf_mae=0.0889  ppf_r2=0.9816
                       cycle_mae=1.20d    keff_r2=0.9904
=============================================================================
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
print("cnn_v7.py — BEAVRS CNN  |  W_LOG=3.0 + Inverse-PPF Sample Weighting\n")

t_start = time.time()


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'
MODEL_NAME  = 'cnn_v7_model.keras'
CONFIG_NAME = 'cnn_v7_config.json'
SENS_NAME   = 'cnn_v7_sens.csv'
PLOT_NAME   = 'cnn_v7_results.png'
AL_CSV      = 'cnn_v7_al_candidates.csv'
FREQ_PATH   = 'train_type_freq_v7.npy'

PPF_REPORT_LOW  = 2.0
PPF_REPORT_HIGH = 4.5

# ── BEAVRS core geometry ──────────────────────────────────────────────────────
N_POS    = 31
N_TYPES  = 9
N_STEPS  = 31

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

# ── Output layout (35 total) ──────────────────────────────────────────────────
N_OUTPUTS           = 1 + 1 + N_STEPS + 1 + 1   # 35
IDX_PPF_MAX         = 0
IDX_PPF_BOC         = 1
IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS    # 33
IDX_CYCLE           = 2 + N_STEPS    # 33
IDX_RHO             = 3 + N_STEPS    # 34

# ── Architecture (best from v5 tuner — all confirmed at 400 epochs) ───────────
EMBED_DIM      = 16
FILTERS        = (64, 128)   # Stage-2 confirmed: '64_128' > '32_64' > '32_64_128'
NUM_BLOCKS     = 2            # Stage-2 confirmed: 2 blocks > 3 at full training length
DENSE_UNITS    = 128          # reverted from tuner's 64 — underperforms at 400 ep
PPF_HEAD_UNITS = 64           # tuner result, kept
DROPOUT        = 0.2
CONV_DROP      = 0.1
WEIGHT_DECAY   = 1e-4

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 400
LR         = 1e-3
TEST_FRAC  = 0.15
VAL_FRAC   = 0.15
SEED       = 42
MC_SAMPLES = 30

# ── Loss weights ──────────────────────────────────────────────────────────────
W_PPF_MAX   = 1.5 #3.0
W_PPF_BOC   = 2.0
W_PPF_STEPS = 0.5
W_CYCLE     = 1.0
W_RHO       = 5.0
W_MONO      = 0.01
# v7: raised from 2.0 → 3.0.
# If still > 3.10% after this run, try W_PPF_MAX=1.5 + W_LOG=4.0 together,
# so the log term fully dominates for the ppf_max head.
W_LOG = 5.0 #3.0

# ── Sample weighting ──────────────────────────────────────────────────────────
# Upweights low-PPF training samples so the optimizer spends more gradient
# there.  Disable to reproduce v6 behaviour exactly.
USE_SAMPLE_WEIGHTING = True

# ── Active learning ───────────────────────────────────────────────────────────
AL_UNCERTAINTY_THRESHOLD = 0.07
AL_MAX_QUERIES_PER_ROUND = 50

# Set after scaler.fit() — referenced inside v7_loss
_PPF_MAX_MEAN = None
_PPF_MAX_STD  = None


# =============================================================================
# SECTION 2 — LOAD MONOCORE CYCLE LENGTHS
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
], axis=1)

ppf_global_max = step_max_ppf.max(axis=1)
ppf_boc        = step_max_ppf[:, 0]

keff_raw = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
rho_pcm  = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"  Patterns           : {len(df)}")
print(f"  PPF_max range      : {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  PPF_max mean       : {ppf_global_max.mean():.3f}")
print(f"  10th percentile    : {np.percentile(ppf_global_max, 10):.3f}  ← QICA target")
print(f"  Cycle length range : {df.cycle_length.min():.1f} – {df.cycle_length.max():.1f} days")
print(f"  keff_boc range     : {keff_raw.min():.4f} – {keff_raw.max():.4f}")
print("=" * 60 + "\n")


# =============================================================================
# SECTION 4 — BUILD TARGET AND FEATURE ARRAYS
# =============================================================================

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_rho       = rho_pcm.reshape(-1, 1).astype(np.float32)

# Main targets (34 cols): ppf_max + ppf_boc + 31 step-PPFs + cycle_length
Y_main    = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle], axis=1)
# Rho (1 col): separate scaler so rho variance doesn't swamp PPF/cycle gradients
Y_rho_col = Y_rho

# 6×6 integer grid input (0 = inactive, 1–9 = assembly type)
X_raw  = df[load_cols].values.astype(np.int32)
X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1

print(f"[INPUT]  Grid shape  : {X_grid.shape}")
print(f"  Active fuel cells  : {GRID_MASK.sum()}/36")
print(f"[TARGETS] Main: {Y_main.shape}, Rho: {Y_rho_col.shape}\n")


# =============================================================================
# SECTION 5 — TRAIN / VAL / TEST SPLIT + SCALERS + SAMPLE WEIGHTS
# =============================================================================

(X_tr, X_tmp,
 Ym_tr, Ym_tmp,
 Yr_tr, Yr_tmp) = train_test_split(X_grid, Y_main, Y_rho_col,
                                    test_size=TEST_FRAC + VAL_FRAC,
                                    random_state=SEED)

(X_val, X_test,
 Ym_val, Ym_test,
 Yr_val, Yr_test) = train_test_split(X_tmp, Ym_tmp, Yr_tmp,
                                      test_size=0.5,
                                      random_state=SEED)

print(f"[SPLIT] {len(X_tr)} train / {len(X_val)} val / {len(X_test)} test")

ym_scaler  = StandardScaler()
Ym_tr_sc   = ym_scaler.fit_transform(Ym_tr)
Ym_val_sc  = ym_scaler.transform(Ym_val)
Ym_test_sc = ym_scaler.transform(Ym_test)

yr_scaler  = StandardScaler()
Yr_tr_sc   = yr_scaler.fit_transform(Yr_tr)
Yr_val_sc  = yr_scaler.transform(Yr_val)
Yr_test_sc = yr_scaler.transform(Yr_test)

Y_tr_sc   = np.concatenate([Ym_tr_sc,  Yr_tr_sc],  axis=1).astype(np.float32)
Y_val_sc  = np.concatenate([Ym_val_sc, Yr_val_sc],  axis=1).astype(np.float32)
Y_test_sc = np.concatenate([Ym_test_sc, Yr_test_sc], axis=1).astype(np.float32)

# Store ppf_max scaler constants for use inside the loss function
_PPF_MAX_MEAN = float(ym_scaler.mean_[IDX_PPF_MAX])
_PPF_MAX_STD  = float(ym_scaler.scale_[IDX_PPF_MAX])

print(f"\n[SCALER]  ppf_max  mean={_PPF_MAX_MEAN:.3f}  std={_PPF_MAX_STD:.3f}")
print(f"          cycle    mean={ym_scaler.mean_[IDX_CYCLE]:.1f}  std={ym_scaler.scale_[IDX_CYCLE]:.2f}")
print(f"          rho_pcm  mean={yr_scaler.mean_[0]:.0f}   std={yr_scaler.scale_[0]:.0f}")

# ── Inverse-PPF sample weights ────────────────────────────────────────────────
# w_i = mean(ppf_train) / ppf_i, then normalised so mean(w) = 1.
# Low-PPF patterns (the ones that matter most for QICA) get higher weight,
# directing more gradient budget toward reducing their relative error.
if USE_SAMPLE_WEIGHTING:
    ppf_tr_raw = Ym_tr[:, IDX_PPF_MAX]          # unscaled ppf_max, training set
    mean_ppf   = float(ppf_tr_raw.mean())
    sample_weights = (mean_ppf / (ppf_tr_raw + 1e-6)).astype(np.float32)
    sample_weights /= sample_weights.mean()      # normalise: mean weight = 1
    print(f"\n[SAMPLE WEIGHTS]  enabled")
    print(f"  Weight range       : {sample_weights.min():.3f} – {sample_weights.max():.3f}")
    print(f"  PPF < 2.5  samples : mean weight = "
          f"{sample_weights[ppf_tr_raw < 2.5].mean():.3f}  "
          f"(n={int((ppf_tr_raw < 2.5).sum())})")
    print(f"  PPF 2.5–4.0 samples: mean weight = "
          f"{sample_weights[(ppf_tr_raw >= 2.5) & (ppf_tr_raw < 4.0)].mean():.3f}")
    print(f"  PPF > 4.0  samples : mean weight = "
          f"{sample_weights[ppf_tr_raw > 4.0].mean():.3f}  "
          f"(n={int((ppf_tr_raw > 4.0).sum())})")
else:
    sample_weights = None
    print("\n[SAMPLE WEIGHTS]  disabled (USE_SAMPLE_WEIGHTING=False)")

print()


# =============================================================================
# SECTION 6 — ARCHITECTURE
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """
    Residual convolutional block:
      Conv2D → BN → GELU → Conv2D → BN → Add(shortcut) → GELU → Dropout

    The residual shortcut lets gradients bypass the conv stack, avoiding
    vanishing gradients. Projection Conv2D (1×1) handles filter dim mismatches.
    """
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        self._dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
        self._filters = filters
        self.proj = None

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.proj = layers.Conv2D(self._filters, 1, padding='same')
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = self.proj(x) if self.proj is not None else x
        h = tf.nn.gelu(self.bn1(self.conv1(x),  training=training))
        h = self.bn2(self.conv2(h), training=training)
        h = tf.nn.gelu(h + shortcut)
        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)
        return h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self._filters,
                    'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


def build_cnn_v7(
    embed_dim      = EMBED_DIM,
    filters        = FILTERS,
    num_blocks     = NUM_BLOCKS,
    dense_units    = DENSE_UNITS,
    ppf_head_units = PPF_HEAD_UNITS,
    dropout        = DROPOUT,
    conv_dropout   = CONV_DROP,
):
    """
    Multi-head CNN surrogate (spatial attention OFF — Stage-1 confirmed best).

    Input → Embedding(10, 16) → 2×ConvResBlock([64,128]) → GlobalAvgPool
          → Dense(128, gelu) → Dropout(0.2) → Dense(64, gelu)
          → [PPF head (33)] + [Cycle head (1)] + [Rho head (1)]
          → Concatenate → (B, 35)

    Three heads prevent gradient interference: PPF has many outputs and high
    weight, cycle is a scalar, rho uses a separate scaler.
    """
    inp = keras.Input(shape=(GRID_ROWS, GRID_COLS), dtype='int32',
                      name='loading_grid')
    x   = layers.Embedding(N_TYPES + 1, embed_dim,
                            name='assembly_embedding')(inp)

    for i in range(num_blocks):
        f = filters[i] if i < len(filters) else filters[-1]
        x = ConvResBlock(f, dropout=conv_dropout,
                         name=f'conv_block_{i + 1}')(x)

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Shared trunk
    shared = layers.Dense(dense_units, activation='gelu',
                          name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu',
                          name='shared_dense2')(shared)

    # PPF head: 33 outputs (ppf_max + ppf_boc + 31 step values)
    h_ppf   = layers.Dense(ppf_head_units, activation='gelu',
                           name='ppf_dense')(shared)
    h_ppf   = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf = layers.Dense(1 + 1 + N_STEPS, activation='linear',
                           name='ppf_output')(h_ppf)

    # Cycle head: 1 output
    h_cyc     = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc     = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle = layers.Dense(1, activation='linear', name='cycle_output')(h_cyc)

    # Rho head: 1 output
    h_rho   = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho   = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho = layers.Dense(1, activation='linear', name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    return keras.Model(inputs=inp, outputs=out, name='BEAVRS_CNN_v7')


# =============================================================================
# SECTION 7 — LOSS FUNCTION
# =============================================================================
#
# v7 loss = v6 structure with W_LOG raised to 3.0.
#
# The log-space term gradient in scaled space:
#   d/d(pred_sc) [log(pred_real) - log(true_real)]²
#   = 2 * (log(pred_real) - log(true_real)) / pred_real * _PPF_MAX_STD
#
# At low PPF (small pred_real), the 1/pred_real factor amplifies the gradient.
# At high PPF, the gradient is attenuated. This is the opposite of MSE,
# which gives equal absolute-error gradients regardless of PPF level.
#
# With W_LOG=3.0 vs W_PPF_MAX=3.0, the log term now contributes equal weight
# to the loss sum before the per-sample variance is considered — previously
# at W_LOG=2.0 it was the minority term.
#
# LOG_CLAMP: prevents log(0) if predictions drift negative early in training.
# PPF_min ≈ 1.5 in this dataset so this is a generous safety margin.

LOG_CLAMP = 0.5


def v7_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # ── Standard MSE terms (scaled space) ─────────────────────────────────────
    ppf_max_loss   = W_PPF_MAX   * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_MAX] - y_pred[:, IDX_PPF_MAX]))

    ppf_boc_loss   = W_PPF_BOC   * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_BOC] - y_pred[:, IDX_PPF_BOC]))

    ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
        - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))

    cycle_loss     = W_CYCLE     * tf.reduce_mean(tf.square(
        y_true[:, IDX_CYCLE] - y_pred[:, IDX_CYCLE]))

    rho_loss       = W_RHO       * tf.reduce_mean(tf.square(
        y_true[:, IDX_RHO] - y_pred[:, IDX_RHO]))

    # ── Monotonicity penalty ───────────────────────────────────────────────────
    # After BA burnout (~step 3), PPF should decrease monotonically.
    late       = y_pred[:, IDX_PPF_STEPS_START + 3:IDX_PPF_STEPS_END]
    diffs      = late[:, 1:] - late[:, :-1]
    violations = tf.maximum(0.0, diffs)
    mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))

    # ── Log-space MSE: directly targets relative error (W=3.0 in v7) ──────────
    ppf_true_real = y_true[:, IDX_PPF_MAX] * _PPF_MAX_STD + _PPF_MAX_MEAN
    ppf_pred_real = y_pred[:, IDX_PPF_MAX] * _PPF_MAX_STD + _PPF_MAX_MEAN

    log_true = tf.math.log(tf.maximum(ppf_true_real, LOG_CLAMP))
    log_pred = tf.math.log(tf.maximum(ppf_pred_real, LOG_CLAMP))
    log_loss = W_LOG * tf.reduce_mean(tf.square(log_pred - log_true))

    return (ppf_max_loss + ppf_boc_loss + ppf_steps_loss
            + cycle_loss + rho_loss + mono_loss + log_loss)


# =============================================================================
# SECTION 8 — BUILD + COMPILE + TRAIN
# =============================================================================

model = build_cnn_v7()
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
    loss=v7_loss,
    metrics=['mae']
)
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=40,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20,
        min_lr=1e-5, verbose=1
    ),
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda ep, logs: print(
            f"  Ep {ep+1:4d} | loss: {logs['loss']:.5f} | val: {logs['val_loss']:.5f}"
            + (" ← gap!" if logs['val_loss'] > 1.5 * logs['loss'] else "")
        ) if (ep + 1) % 25 == 0 else None
    ),
]

print("[TRAINING] Starting CNN v7 ...")
print(f"  Filters          : {FILTERS}  (Stage-2 confirmed best)")
print(f"  Dense units      : {DENSE_UNITS}  (reverted from tuner's 64)")
print(f"  W_LOG            : {W_LOG}  (raised from v6's 2.0)")
print(f"  Sample weighting : {USE_SAMPLE_WEIGHTING}")
print()

history = model.fit(
    X_tr, Y_tr_sc,
    validation_data=(X_val, Y_val_sc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    sample_weight=sample_weights,   # None if USE_SAMPLE_WEIGHTING=False
    verbose=0
)

t_train    = time.time() - t_start
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
train_losses = history.history['loss']
val_losses   = history.history['val_loss']
final_ratio  = val_losses[best_epoch - 1] / (train_losses[best_epoch - 1] + 1e-9)

print(f"\n[TRAINING DONE]  {t_train:.1f}s ({t_train/60:.1f} min)  |  best epoch: {best_epoch}")
print(f"  Val/train ratio at best epoch: {final_ratio:.2f}  "
      f"({'✓ good' if final_ratio < 1.3 else '⚠ mild overfit'})\n")


# =============================================================================
# SECTION 9 — INVERSE TRANSFORM HELPER
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    """Invert the two-scaler scheme to real physical units."""
    Y_main_real = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_real  = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main_real, Y_rho_real], axis=1)


# =============================================================================
# SECTION 10 — EVALUATE ON TEST SET
# =============================================================================

print("[EVALUATION] Running predictions on test set ...")

Y_pred_sc   = model.predict(X_test, verbose=0)
Y_pred_real = inverse_transform(Y_pred_sc)
Y_true_real = inverse_transform(Y_test_sc)

ppf_max_pred = Y_pred_real[:, IDX_PPF_MAX]
ppf_max_true = Y_true_real[:, IDX_PPF_MAX]
cycle_pred   = Y_pred_real[:, IDX_CYCLE]
cycle_true   = Y_true_real[:, IDX_CYCLE]
rho_pred     = Y_pred_real[:, IDX_RHO]
rho_true     = Y_true_real[:, IDX_RHO]

keff_pred_rep = 1.0 / (1.0 - rho_pred / 1e5)
keff_true_rep = 1.0 / (1.0 - rho_true / 1e5)

ppf_mae     = np.abs(ppf_max_pred - ppf_max_true).mean()
ppf_rel_err = (np.abs(ppf_max_pred - ppf_max_true)
               / (ppf_max_true + 1e-6)).mean() * 100
ppf_r2      = r2_score(ppf_max_true, ppf_max_pred)
ppf_pearson = np.corrcoef(ppf_max_pred, ppf_max_true)[0, 1]

goal_mask   = (ppf_max_true >= PPF_REPORT_LOW) & (ppf_max_true <= PPF_REPORT_HIGH)
goal_mae    = np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask]).mean()
goal_r2     = r2_score(ppf_max_true[goal_mask], ppf_max_pred[goal_mask])
goal_rel    = (np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask])
               / (ppf_max_true[goal_mask] + 1e-6)).mean() * 100

cycle_mae   = np.abs(cycle_pred - cycle_true).mean()
cycle_r2    = r2_score(cycle_true, cycle_pred)

rho_mae     = np.abs(rho_pred - rho_true).mean()
rho_r2      = r2_score(rho_true, rho_pred)
keff_mae_r  = np.abs(keff_pred_rep - keff_true_rep).mean()
keff_r2_r   = r2_score(keff_true_rep, keff_pred_rep)

# Relative error by PPF zone
low_mask  = ppf_max_true <  2.5
mid_mask  = (ppf_max_true >= 2.5) & (ppf_max_true < 4.0)
high_mask = ppf_max_true >= 4.0

def zone_rel(mask):
    if not mask.any(): return float('nan')
    return (np.abs(ppf_max_pred[mask] - ppf_max_true[mask])
            / (ppf_max_true[mask] + 1e-6)).mean() * 100

rel_low  = zone_rel(low_mask)
rel_mid  = zone_rel(mid_mask)
rel_high = zone_rel(high_mask)

print(f"\n{'='*62}")
print(f"CNN v7 TEST RESULTS")
print(f"{'='*62}")
print(f"  PPF_max (all patterns):")
print(f"    MAE              : {ppf_mae:.4f}   (v4: 0.0843 | v5: 0.0875 | v6: 0.0889)")
print(f"    Relative error   : {ppf_rel_err:.2f}%    (v4: 3.10% | v5: 3.22% | v6: 3.25%)")
print(f"    R²               : {ppf_r2:.4f}   (v4: 0.9841 | v5: 0.9826 | v6: 0.9816)")
print(f"    Pearson r        : {ppf_pearson:.4f}")
print(f"")
print(f"  Relative error breakdown by PPF zone:")
print(f"    PPF < 2.5  (n={low_mask.sum():4d}) : {rel_low:.2f}%   ← target zone (v6: 3.59%)")
print(f"    2.5–4.0    (n={mid_mask.sum():4d}) : {rel_mid:.2f}%")
print(f"    PPF > 4.0  (n={high_mask.sum():4d}) : {rel_high:.2f}%")
print(f"")
print(f"  PPF_max (goal zone {PPF_REPORT_LOW}–{PPF_REPORT_HIGH}  n={goal_mask.sum()}):")
print(f"    MAE              : {goal_mae:.4f}   (v4: 0.0811 | v5: 0.0826 | v6: ~0.083)")
print(f"    R²               : {goal_r2:.4f}   (v4: 0.9673 | v5: 0.9660)")
print(f"    Relative error   : {goal_rel:.2f}%")
print(f"")
print(f"  Cycle length:")
print(f"    MAE              : {cycle_mae:.2f} days  (v4: 1.28 | v5: 1.61 | v6: 1.20)")
print(f"    R²               : {cycle_r2:.4f}")
print(f"  Rho_pcm (BOC):")
print(f"    MAE              : {rho_mae:.0f} pcm   (v4: 65 | v5: 73)")
print(f"    R²               : {rho_r2:.4f}")
print(f"  keff_boc (derived from rho):")
print(f"    MAE              : {keff_mae_r:.5f}  (v4: 0.00083 | v5: 0.00093 | v6: ~0.00084)")
print(f"    R²               : {keff_r2_r:.4f}   (v4: 0.9912 | v5: 0.9889 | v6: 0.9904)")
print(f"{'='*62}\n")


# =============================================================================
# SECTION 11 — MONTE CARLO DROPOUT UNCERTAINTY
# =============================================================================

print("[MC DROPOUT] Estimating prediction uncertainty ...")
t_mc = time.time()

mc_preds_sc = np.stack([
    model(X_test, training=True).numpy()
    for _ in range(MC_SAMPLES)
])

mc_mean_sc = mc_preds_sc.mean(axis=0)
mc_std_sc  = mc_preds_sc.std(axis=0)

mc_mean_real = inverse_transform(mc_mean_sc)
mc_std_main  = mc_std_sc[:, :34] * ym_scaler.scale_
mc_std_rho   = mc_std_sc[:, 34:35] * yr_scaler.scale_
mc_std_real  = np.concatenate([mc_std_main, mc_std_rho], axis=1)

ppf_mc_mean = mc_mean_real[:, IDX_PPF_MAX]
ppf_mc_std  = mc_std_real[:, IDX_PPF_MAX]

unc_err_corr = np.corrcoef(ppf_mc_std,
                            np.abs(ppf_mc_mean - ppf_max_true))[0, 1]

print(f"  Time              : {time.time() - t_mc:.1f}s")
print(f"  Mean σ(ppf_max)   : {ppf_mc_std.mean():.4f}")
print(f"  Max σ(ppf_max)    : {ppf_mc_std.max():.4f}")
print(f"  σ-error corr      : {unc_err_corr:.3f}  (positive → σ is useful flag)\n")


# =============================================================================
# SECTION 12 — POSITION SENSITIVITY  (∂ppf_max / ∂position)
# =============================================================================

print("[SENSITIVITY]  Computing ∂ppf_max/∂position ...")

n_sens   = min(200, len(X_test))
X_sample = tf.constant(X_test[:n_sens], dtype=tf.int32)

sens_norm = np.ones(N_POS, dtype=np.float32)
sens_pos  = np.ones(N_POS, dtype=np.float32)
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
        sens_norm = sens_pos / (sens_pos.max() + 1e-8)
        sens_grid = sens_grid_raw / (sens_grid_raw.max() + 1e-8)
        top5 = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"  Top-5 critical positions : {top5}")
        print(f"  Sensitivity range        : {sens_norm.min():.3f} – {sens_norm.max():.3f}\n")
except Exception as e:
    print(f"  [WARN] Gradient failed ({e}) — using uniform sensitivity\n")

sens_df = pd.DataFrame({
    'position'        : [f'pos_{i}' for i in range(N_POS)],
    'sensitivity'     : sens_pos,
    'sensitivity_norm': sens_norm,
})
sens_df.to_csv(SENS_NAME, index=False)
print(f"  Saved: {SENS_NAME}")


# =============================================================================
# SECTION 13 — ACTIVE LEARNING CANDIDATE IDENTIFICATION
# =============================================================================

print("[ACTIVE LEARNING]  Scanning full dataset for query candidates ...")

mc_all_full = np.stack([
    model(X_grid, training=True).numpy()
    for _ in range(MC_SAMPLES)
])
mc_mean_sc_full  = mc_all_full.mean(axis=0)
mc_std_sc_full   = mc_all_full.std(axis=0)
mc_mean_full     = inverse_transform(mc_mean_sc_full)
mc_std_full_phy  = np.concatenate([
    mc_std_sc_full[:, :34] * ym_scaler.scale_,
    mc_std_sc_full[:, 34:35] * yr_scaler.scale_,
], axis=1)

ppf_full_pred  = mc_mean_full[:, IDX_PPF_MAX]
ppf_full_std   = mc_std_full_phy[:, IDX_PPF_MAX]
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

al_records = []
for idx in query_top:
    pattern_flat = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                pattern_flat.append(int(X_grid[idx, r, c]))
    al_records.append({
        'dataset_idx'  : int(idx),
        'pred_ppf_max' : float(ppf_full_pred[idx]),
        'pred_ppf_std' : float(ppf_full_std[idx]),
        'priority'     : float(priority_score[idx]),
        'cycle_length' : float(mc_mean_full[idx, IDX_CYCLE]),
        'rho_pcm_boc'  : float(mc_mean_full[idx, IDX_RHO]),
        **{f'pos_{j}': pattern_flat[j] for j in range(N_POS)},
    })

al_df = pd.DataFrame(al_records)
al_df.to_csv(AL_CSV, index=False)
print(f"  Saved query candidates → {AL_CSV}\n")


# =============================================================================
# SECTION 14 — SAVE MODEL + CONFIG + TRUST REGION
# =============================================================================

model.save(MODEL_NAME)

config = {
    'version'  : 'v7',
    'N_POS'    : N_POS, 'N_TYPES': N_TYPES, 'N_STEPS': N_STEPS,
    'GRID_ROWS': GRID_ROWS, 'GRID_COLS': GRID_COLS,
    'GRID_LAYOUT': GRID_LAYOUT.tolist(),
    'GRID_MASK'  : GRID_MASK.tolist(),
    'IDX_PPF_MAX'        : IDX_PPF_MAX,
    'IDX_PPF_BOC'        : IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': IDX_PPF_STEPS_START,
    'IDX_PPF_STEPS_END'  : IDX_PPF_STEPS_END,
    'IDX_CYCLE' : IDX_CYCLE,
    'IDX_RHO'   : IDX_RHO,
    'N_OUTPUTS' : N_OUTPUTS,
    'PPF_REPORT_LOW' : PPF_REPORT_LOW,
    'PPF_REPORT_HIGH': PPF_REPORT_HIGH,
    'ym_scaler_mean' : ym_scaler.mean_.tolist(),
    'ym_scaler_scale': ym_scaler.scale_.tolist(),
    'yr_scaler_mean' : yr_scaler.mean_.tolist(),
    'yr_scaler_scale': yr_scaler.scale_.tolist(),
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v)
                             for k, v in ASSEMBLY_CYCLE_EQUIV.items()},
    'mc_samples'         : MC_SAMPLES,
    'al_uncertainty_thr' : AL_UNCERTAINTY_THRESHOLD,
    'arch': {
        'embed_dim': EMBED_DIM, 'filters': list(FILTERS),
        'num_blocks': NUM_BLOCKS, 'dense_units': DENSE_UNITS,
        'ppf_head_units': PPF_HEAD_UNITS, 'dropout': DROPOUT,
        'conv_dropout': CONV_DROP, 'weight_decay': WEIGHT_DECAY,
    },
    'loss_weights': {
        'W_PPF_MAX': W_PPF_MAX, 'W_PPF_BOC': W_PPF_BOC,
        'W_PPF_STEPS': W_PPF_STEPS, 'W_CYCLE': W_CYCLE,
        'W_RHO': W_RHO, 'W_MONO': W_MONO, 'W_LOG': W_LOG,
    },
    'training_config': {
        'USE_SAMPLE_WEIGHTING': USE_SAMPLE_WEIGHTING,
    },
    'v7_results': {
        'ppf_mae': float(ppf_mae), 'ppf_rel_err': float(ppf_rel_err),
        'ppf_r2': float(ppf_r2), 'goal_mae': float(goal_mae),
        'goal_r2': float(goal_r2), 'cycle_mae': float(cycle_mae),
        'rho_r2': float(rho_r2), 'keff_r2': float(keff_r2_r),
        'rel_low': float(rel_low), 'rel_mid': float(rel_mid),
        'rel_high': float(rel_high),
    },
    'baseline': {
        'v4': {'ppf_mae': 0.0843, 'ppf_rel_err': 3.10,
               'ppf_r2': 0.9841, 'cycle_mae': 1.28, 'keff_r2': 0.9912},
        'v5': {'ppf_mae': 0.0875, 'ppf_rel_err': 3.22,
               'ppf_r2': 0.9826, 'cycle_mae': 1.61, 'keff_r2': 0.9889},
        'v6': {'ppf_mae': 0.0889, 'ppf_rel_err': 3.25,
               'ppf_r2': 0.9816, 'cycle_mae': 1.20, 'keff_r2': 0.9904},
    },
}

with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

# Per-position type frequency for QICA trust-region
tr_flat = np.stack([
    X_tr[:, r, c]
    for r in range(GRID_ROWS) for c in range(GRID_COLS)
    if GRID_LAYOUT[r, c] >= 0
], axis=1)

train_type_freq = np.zeros((N_POS, N_TYPES), dtype=np.float32)
for p in range(N_POS):
    for t in range(1, N_TYPES + 1):
        train_type_freq[p, t - 1] = float((tr_flat[:, p] == t).mean())
train_type_freq = np.maximum(train_type_freq, 1e-3)
train_type_freq /= train_type_freq.sum(axis=1, keepdims=True)
np.save(FREQ_PATH, train_type_freq)

print(f"[SAVED]  {MODEL_NAME}")
print(f"[SAVED]  {CONFIG_NAME}")
print(f"[SAVED]  {SENS_NAME}")
print(f"[SAVED]  {FREQ_PATH}  (shape {train_type_freq.shape} — for QICA trust-region)")


# =============================================================================
# SECTION 15 — VISUALISATIONS
# =============================================================================

fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    f"BEAVRS CNN v7  |  PPF MAE={ppf_mae:.3f}  rel_err={ppf_rel_err:.2f}%  "
    f"R²={ppf_r2:.3f}  keff R²={keff_r2_r:.3f}  best_ep={best_epoch}"
    f"  [W_LOG={W_LOG}, sample_wt={'ON' if USE_SAMPLE_WEIGHTING else 'OFF'}]",
    fontsize=11, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1. Training curve
ax = fig.add_subplot(gs[0, 0])
ax.plot(history.history['loss'],     '#1B4FBF', lw=1.5, label='Train')
ax.plot(history.history['val_loss'], '#F5A623', lw=1.5, label='Val')
ax.axvline(best_epoch - 1, color='red', lw=1, ls=':', label=f'Best ep {best_epoch}')
ax.set_yscale('log')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
ax.set_title(f'Training Curve\nval/train={final_ratio:.2f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.text(0.98, 0.95,
        f'val/train={final_ratio:.2f}\n{"✓ good" if final_ratio < 1.3 else "⚠ mild overfit"}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        color='green' if final_ratio < 1.3 else 'darkorange',
        bbox=dict(fc='white', ec='grey', alpha=0.7, boxstyle='round'))

# 2. PPF scatter
ax = fig.add_subplot(gs[0, 1])
colors_sc = np.where(
    (ppf_max_true >= PPF_REPORT_LOW) & (ppf_max_true <= PPF_REPORT_HIGH),
    '#17BECF', '#AAAAAA'
)
lim = [ppf_max_true.min() - 0.1, ppf_max_true.max() + 0.1]
ax.scatter(ppf_max_true, ppf_max_pred, c=colors_sc, alpha=0.35, s=7)
ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
ax.axhspan(PPF_REPORT_LOW, PPF_REPORT_HIGH, alpha=0.07,
           color='teal', label='Goal zone')
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'PPF Prediction\nMAE={ppf_mae:.3f}  R²={ppf_r2:.3f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 3. keff scatter
ax = fig.add_subplot(gs[0, 2])
ax.scatter(keff_true_rep, keff_pred_rep, alpha=0.3, s=7, color='#9467BD')
lim_k = [keff_true_rep.min() - 0.003, keff_true_rep.max() + 0.003]
ax.plot(lim_k, lim_k, 'k--', lw=1)
ax.set_xlabel('True keff_boc'); ax.set_ylabel('Predicted keff_boc')
ax.set_title(f'keff_boc\nMAE={keff_mae_r:.5f}  R²={keff_r2_r:.3f}')
ax.grid(alpha=0.3)

# 4. Cycle length scatter
ax = fig.add_subplot(gs[0, 3])
ax.scatter(cycle_true, cycle_pred, alpha=0.3, s=7, color='#2CA02C')
lim_c = [cycle_true.min() - 5, cycle_true.max() + 5]
ax.plot(lim_c, lim_c, 'k--', lw=1)
ax.set_xlabel('True cycle length (days)'); ax.set_ylabel('Predicted (days)')
ax.set_title(f'Cycle Length\nMAE={cycle_mae:.1f}d  R²={cycle_r2:.3f}')
ax.grid(alpha=0.3)

# 5. Version comparison (relative error)
ax = fig.add_subplot(gs[1, 0])
versions  = ['v4\nMSE', 'v5\nMAPE', 'v6\nlogW=2', 'v7\nlogW=3\n+swt']
rel_errs  = [3.10, 3.22, 3.25, ppf_rel_err]
bar_colors = ['#D62728', '#F5A623', '#1B4FBF',
              '#2CA02C' if ppf_rel_err < 3.10 else '#9467BD']
bars = ax.bar(versions, rel_errs, color=bar_colors, alpha=0.75, width=0.55)
ax.axhline(3.10, color='red', lw=0.8, ls='--', alpha=0.5, label='v4 baseline')
for bar, val in zip(bars, rel_errs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Mean relative error (%)')
ax.set_title('Version Comparison\nOverall rel. error')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 6. Relative error by PPF zone (v7)
ax = fig.add_subplot(gs[1, 1])
zones    = [f'PPF<2.5\n(n={low_mask.sum()})',
            f'2.5–4.0\n(n={mid_mask.sum()})',
            f'PPF>4.0\n(n={high_mask.sum()})']
rel_z    = [rel_low, rel_mid, rel_high]
v6_z     = [3.59, None, None]   # v6 low-zone for comparison
bar_cols = ['#D62728', '#17BECF', '#AAAAAA']
bars_z   = ax.bar(zones, rel_z, color=bar_cols, alpha=0.75)
for bar, val in zip(bars_z, rel_z):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
ax.axhline(3.59, color='blue', lw=1, ls='--', alpha=0.6,
           label='v6 PPF<2.5 = 3.59%')
ax.set_ylabel('Relative error (%)'); ax.set_title('v7 Rel. Error by PPF Zone')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 7. Uncertainty vs error
ax = fig.add_subplot(gs[1, 2])
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

# 8. Active learning candidates
ax = fig.add_subplot(gs[1, 3])
ax.scatter(ppf_full_pred, ppf_full_std, alpha=0.15, s=4,
           color='#AAAAAA', label='All patterns')
if len(query_top) > 0:
    ax.scatter(ppf_full_pred[query_top], ppf_full_std[query_top],
               alpha=0.8, s=20, color='#D62728', zorder=5,
               label=f'AL candidates (n={len(query_top)})')
ax.axhline(AL_UNCERTAINTY_THRESHOLD, color='orange', lw=1.5, ls='--',
           label='σ threshold')
ax.axvline(np.percentile(ppf_full_pred, 25), color='teal', lw=1.5, ls='--',
           label='PPF 25th pct')
ax.set_xlabel('Predicted ppf_max'); ax.set_ylabel('MC σ (ppf_max)')
ax.set_title('Active Learning Candidates\n(high-σ, low-PPF)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 9. Best loading pattern
ax = fig.add_subplot(gs[2, 0])
best_idx = ppf_max_true.argmin()
g_disp   = X_test[best_idx].astype(float).copy()
g_disp[~GRID_MASK] = np.nan
cmap_ex  = plt.cm.YlOrRd.copy(); cmap_ex.set_bad('lightgrey')
ax.imshow(g_disp, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_MASK[r, c]:
            ax.text(c, r, f'{X_test[best_idx, r, c]}',
                    ha='center', va='center', fontsize=8)
ax.set_title(f'Best Pattern (min true PPF)\nTrue={ppf_max_true[best_idx]:.3f}  '
             f'Pred={ppf_max_pred[best_idx]:.3f}')
ax.set_xticks([]); ax.set_yticks([])

# 10. Worst loading pattern
ax = fig.add_subplot(gs[2, 1])
worst_idx = ppf_max_true.argmax()
g_disp2   = X_test[worst_idx].astype(float).copy()
g_disp2[~GRID_MASK] = np.nan
ax.imshow(g_disp2, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_MASK[r, c]:
            ax.text(c, r, f'{X_test[worst_idx, r, c]}',
                    ha='center', va='center', fontsize=8)
ax.set_title(f'Worst Pattern (max true PPF)\nTrue={ppf_max_true[worst_idx]:.3f}  '
             f'Pred={ppf_max_pred[worst_idx]:.3f}')
ax.set_xticks([]); ax.set_yticks([])

# 11. PPF burnup profile
ax = fig.add_subplot(gs[2, 2])
steps_range = np.arange(N_STEPS)
true_smean  = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
true_sstd   = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
pred_smean  = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
pred_sstd   = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
ax.plot(steps_range, true_smean, '#1B4FBF', lw=2, label='True')
ax.fill_between(steps_range, true_smean - true_sstd, true_smean + true_sstd,
                color='#1B4FBF', alpha=0.15)
ax.plot(steps_range, pred_smean, '#F5A623', lw=2, ls='--', label='Predicted')
ax.fill_between(steps_range, pred_smean - pred_sstd, pred_smean + pred_sstd,
                color='#F5A623', alpha=0.15)
ax.set_xlabel('Burnup Step'); ax.set_ylabel('Max PPF at Step')
ax.set_title('PPF Burnup Profile\n(mean ± 1σ, test patterns)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 12. Residuals
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
# SECTION 16 — SENSITIVITY HEATMAP
# =============================================================================

fig_s, ax_s = plt.subplots(figsize=(5, 5))
disp_sens = np.full((GRID_ROWS, GRID_COLS), np.nan)
pos_i = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            disp_sens[r, c] = sens_norm[pos_i]; pos_i += 1
cmap_s = plt.cm.RdYlGn_r.copy(); cmap_s.set_bad('lightgrey')
im = ax_s.imshow(disp_sens, cmap=cmap_s, aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax_s, label='Normalised sensitivity')
pos_i = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            ax_s.text(c, r, f'P{pos_i}', ha='center', va='center', fontsize=7)
            pos_i += 1
ax_s.set_title('∂ppf_max / ∂position  (Red = critical)')
ax_s.set_xticks([]); ax_s.set_yticks([])
plt.tight_layout()
plt.savefig('cnn_v7_sensitivity.png', dpi=150, bbox_inches='tight')
print(f"[SAVED]  cnn_v7_sensitivity.png")


# =============================================================================
# SECTION 17 — FINAL SUMMARY
# =============================================================================

print('\n' + '=' * 68)
print('CNN v7  FINAL SUMMARY')
print('=' * 68)
print(f"  Architecture     : filters={FILTERS}, blocks={NUM_BLOCKS}, "
      f"dense={DENSE_UNITS}, ppf_head={PPF_HEAD_UNITS}")
print(f"  Regularisation   : conv_drop={CONV_DROP}, head_drop={DROPOUT}, "
      f"weight_decay={WEIGHT_DECAY}")
print(f"  Loss terms       : MSE(ppf+boc+steps+cycle+rho) + monotonicity "
      f"+ log-space(W={W_LOG})")
print(f"  Sample weighting : {'enabled (1/ppf_max)' if USE_SAMPLE_WEIGHTING else 'disabled'}")
print(f"  Parameters       : {model.count_params():,}")
print(f"  Best epoch       : {best_epoch} / {EPOCHS}")
print(f"  Training time    : {t_train:.1f}s ({t_train/60:.1f} min)")
print(f"  Val/train ratio  : {final_ratio:.2f}  "
      f"({'✓ good' if final_ratio < 1.3 else '⚠ mild overfit'})")
print()
print(f"  {'Metric':<30} {'v4':>8}  {'v5':>8}  {'v6':>8}  {'v7':>8}")
print(f"  {'-'*30} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
print(f"  {'PPF MAE':<30} {'0.0843':>8}  {'0.0875':>8}  {'0.0889':>8}  {ppf_mae:>8.4f}")
print(f"  {'PPF rel. error':<30} {'3.10%':>8}  {'3.22%':>8}  {'3.25%':>8}  {ppf_rel_err:>7.2f}%")
print(f"  {'PPF R²':<30} {'0.9841':>8}  {'0.9826':>8}  {'0.9816':>8}  {ppf_r2:>8.4f}")
print(f"  {'Cycle MAE (days)':<30} {'1.28':>8}  {'1.61':>8}  {'1.20':>8}  {cycle_mae:>8.2f}")
print(f"  {'keff R²':<30} {'0.9912':>8}  {'0.9889':>8}  {'0.9904':>8}  {keff_r2_r:>8.4f}")
print(f"  {'Rel. error (PPF<2.5)':<30} {'n/a':>8}  {'3.55%':>8}  {'3.59%':>8}  {rel_low:>7.2f}%")
print()
print(f"  OUTPUT FILES:")
print(f"    {MODEL_NAME:<36} — serializable model")
print(f"    {CONFIG_NAME:<36} — scalers, indices, arch config")
print(f"    {SENS_NAME:<36} — position sensitivities")
print(f"    {PLOT_NAME:<36} — evaluation plots")
print(f"    cnn_v7_sensitivity.png              — sensitivity heatmap")
print(f"    {AL_CSV:<36} — AL query candidates")
print(f"    {FREQ_PATH:<36} — trust-region frequencies")
print()
print(f"  RELATIVE ERROR GUIDANCE:")
if ppf_rel_err < 2.5:
    print(f"    ✓ Relative error below 2.5%.  Excellent — proceed to QICA.")
elif ppf_rel_err < 3.10:
    print(f"    ✓ Relative error improved below v4 baseline (3.10%).")
    print(f"      Next: try W_PPF_MAX=1.5 + W_LOG=4.0 to push further.")
else:
    print(f"    ✗ Relative error not yet improved vs v4 (3.10%).")
    print(f"      Recommended next steps:")
    print(f"        1. Set W_PPF_MAX=1.5, W_LOG=4.0  (log term fully dominates)")
    print(f"        2. Add stratified split on binned PPF")
    print(f"        3. Symmetry augmentation (8-fold dihedral) for more low-PPF data")
print()
print(f"  NEXT STEP: Run qica-cnn-v7.py (point it at cnn_v7_model.keras + config).")
print('=' * 68)