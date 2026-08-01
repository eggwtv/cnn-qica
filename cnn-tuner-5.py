"""
=============================================================================
cnn_v5_tuner.py  —  Stage-2 Focused Tuner + Relative Error Fix
=============================================================================

WHAT THIS DOES:
  Stage 1 (Hyperband, 90 trials) found the best coarse config:
    embed_dim=16, dense_units=128, dropout=0.2, conv_dropout=0.1,
    num_blocks=3, lr=0.001, multi_head=True, use_attention=False

  Stage 2 (this script) fixes those winners and fine-tunes:
    filter sizes, ppf_head_units, weight_decay, and minor dropout adjustments.
  Uses BayesianOptimization (not Hyperband) — better for narrow search spaces.
  ~15 trials × 50 epochs each ≈ 20–35 min runtime.

RELATIVE ERROR FIX:
  The 3.10% rel_err comes from using MSE loss, which treats absolute errors
  equally across all PPF values. An error of 0.08 at PPF=2.0 is 4% (bad),
  but the same error at PPF=7.0 is only 1.1%. MSE doesn't know the difference.
  Fix: add a MAPE-style (percentage) term specifically for ppf_max in real space.
  The loss unscales ppf_max back using stored scaler constants, then computes
  |error| / true_value. This forces the model to be equally accurate across
  the full PPF range, not just accurate on big-value patterns.

HOW TO RUN:
  1. Make sure ml_dataset_constrained.csv is in the same folder
  2. python cnn_v5_tuner.py
  3. It prints best config at the end, then retrains best model fully
  4. Outputs: cnn_v5_model.keras, cnn_v5_config.json

EXPECTED IMPROVEMENT:
  Current: MAE=0.0843, rel_err=3.10%, R²=0.9841
  Target:  MAE≈0.07–0.08, rel_err≈1.5–2.2%, R²≥0.985
=============================================================================
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print(f"Running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print("cnn_v5_tuner.py — Stage-2 Tuner + Relative Error Fix\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'
MODEL_NAME  = 'cnn_v5_model.keras'
CONFIG_NAME = 'cnn_v5_config.json'

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

N_OUTPUTS = 1 + 1 + N_STEPS + 1 + 1   # = 35
IDX_PPF_MAX        = 0
IDX_PPF_BOC        = 1
IDX_PPF_STEPS_START= 2
IDX_PPF_STEPS_END  = 2 + N_STEPS        # 33
IDX_CYCLE          = 2 + N_STEPS        # 33
IDX_RHO            = 3 + N_STEPS        # 34

# ── FIXED hyperparams (from Stage 1 best trial) ───────────────────────────────
EMBED_DIM    = 16
LR           = 0.001
USE_ATTENTION= False   # Stage 1: attention OFF was better
MULTI_HEAD   = True    # Stage 1: multi_head ON was better

# ── Training settings ─────────────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS_TUNE  = 50      # short trials for the tuner — fast
EPOCHS_FULL  = 400     # full retrain of best config after tuning
TEST_FRAC    = 0.15
VAL_FRAC     = 0.15
SEED         = 42

# ── Loss weights (same as v4 for comparability) ───────────────────────────────
W_PPF_MAX   = 3.0
W_PPF_BOC   = 2.0
W_PPF_STEPS = 0.5
W_CYCLE     = 1.0
W_RHO       = 5.0
W_MONO      = 0.01

# ── NEW: weight for the MAPE term that reduces relative error ─────────────────
#    Start at 1.0. If rel_err doesn't drop enough, try 2.0.
#    Too high (>3.0) and it can destabilise early training.
W_MAPE      = 1.0

# These globals are set AFTER the scaler is fit — used inside the loss function
# so we don't need to recompute them every call.
_PPF_MAX_MEAN = None   # real-space mean of ppf_max in training set
_PPF_MAX_STD  = None   # real-space std of ppf_max in training set


# =============================================================================
# SECTION 2 — LOAD DATA (identical to cnn_v4)
# =============================================================================

print("[XLSX] Loading monocore cycle lengths ...")
if os.path.exists(XL_FILE):
    xl_df = pd.read_excel(XL_FILE, sheet_name='Cycle_Lengths')
    monocore_map = dict(zip(xl_df['fa_id'].astype(int),
                            xl_df['monocore_cycle_length'].astype(float)))
else:
    print("  [WARN] xlsx not found — using hardcoded fallback values.")
    monocore_map = {1:172.9, 2:366.9, 3:323.2, 4:299.8, 5:519.9,
                   6:504.9, 7:475.3, 8:471.6, 9:454.7}

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

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_rho       = rho_pcm.reshape(-1, 1).astype(np.float32)

Y_main    = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle], axis=1)
Y_rho_col = Y_rho

X_raw  = df[load_cols].values.astype(np.int32)
X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1

print(f"  PPF_max range : {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  10th pct      : {np.percentile(ppf_global_max, 10):.3f}  ← QICA target")


# =============================================================================
# SECTION 3 — SPLIT + SCALE
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

print(f"\n[SPLIT] {len(X_tr)} train / {len(X_val)} val / {len(X_test)} test")

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

# ── Set globals for the loss function — MUST happen after scaler.fit() ────────
_PPF_MAX_MEAN = float(ym_scaler.mean_[IDX_PPF_MAX])
_PPF_MAX_STD  = float(ym_scaler.scale_[IDX_PPF_MAX])

print(f"\n[SCALER]  ppf_max  mean={_PPF_MAX_MEAN:.3f}  std={_PPF_MAX_STD:.3f}")
print(f"  (real-space ppf_max range in training: "
      f"{Ym_tr[:, IDX_PPF_MAX].min():.2f} – {Ym_tr[:, IDX_PPF_MAX].max():.2f})\n")


# =============================================================================
# SECTION 4 — ARCHITECTURE
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """
    Conv → BN → GELU → Conv → BN → Add(shortcut) → GELU → Dropout
    Residual connection: allows gradient to bypass the conv stack.
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
        h = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        h = tf.nn.gelu(h + shortcut)
        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)
        return h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self._filters, 'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


def build_multihead_cnn(
    embed_dim    = 16,
    filters      = (32, 64, 128),
    num_blocks   = 3,
    dense_units  = 128,
    dropout      = 0.2,
    conv_dropout = 0.1,
    ppf_head_units = 64,     # ← NEW tunable param: PPF head first Dense
):
    """
    Multi-head CNN (attention OFF — confirmed best from Stage 1).
      Input  → Embedding → n×ConvResBlock → GlobalAvgPool
             → Shared Dense → [PPF head | Cycle head | Rho head]
             → Concatenate → (B, 35)

    PPF head has its own size param (ppf_head_units) because PPF is the
    primary target and might benefit from a larger dedicated sub-network.
    """
    inp = keras.Input(shape=(GRID_ROWS, GRID_COLS), dtype='int32',
                      name='loading_grid')
    x   = layers.Embedding(N_TYPES + 1, embed_dim, name='assembly_embedding')(inp)

    for i in range(num_blocks):
        f = filters[i] if i < len(filters) else filters[-1]
        x = ConvResBlock(f, dropout=conv_dropout, name=f'conv_block_{i+1}')(x)

    # NO spatial attention (Stage 1 result: use_attention=False was better)
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Shared trunk
    shared = layers.Dense(dense_units, activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu',
                          name='shared_dense2')(shared)

    # PPF head: tunable first layer size
    h_ppf   = layers.Dense(ppf_head_units, activation='gelu', name='ppf_dense')(shared)
    h_ppf   = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf = layers.Dense(1 + 1 + N_STEPS, activation='linear',
                           name='ppf_output')(h_ppf)

    # Cycle head
    h_cyc    = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc    = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle= layers.Dense(1, activation='linear', name='cycle_output')(h_cyc)

    # Rho head (separate gradient pathway — prevents rho signal from being swamped)
    h_rho    = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho    = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho  = layers.Dense(1, activation='linear', name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    return keras.Model(inputs=inp, outputs=out, name='BEAVRS_CNN_v5')


# =============================================================================
# SECTION 5 — LOSS FUNCTION (with MAPE term to reduce relative error)
# =============================================================================
#
# THE RELATIVE ERROR PROBLEM:
#   Metric:  rel_err = mean( |pred - true| / true ) × 100
#   v4 loss: W_PPF_MAX × MSE(pred, true) — minimizes absolute error
#   Problem: a 0.08 error at PPF=2.0 (4% rel) contributes LESS to MSE than
#            a 0.08 error at PPF=7.0 would if it were scaled up. The model
#            naturally focuses on getting high-PPF patterns right (bigger values
#            = bigger squared errors = dominate MSE gradient).
#
# THE FIX:
#   Add W_MAPE × mean(|pred_real - true_real| / true_real) directly to the loss.
#   This term IS the relative error formula, so minimising it minimises rel_err.
#   We unscale ppf_max back to real space inside the loss using stored constants.
#
# KEY: the MAPE term is ADDITIVE — the MSE term stays to ensure stability.
#      W_MAPE=1.0 is a gentle push; the MSE terms still dominate early training.
#
#  ┌───────────────────────────────────────────────────────────────────────────┐
#  │  ppf_true_real = y_true[:,0] × _PPF_MAX_STD + _PPF_MAX_MEAN             │
#  │  ppf_pred_real = y_pred[:,0] × _PPF_MAX_STD + _PPF_MAX_MEAN             │
#  │  mape_term = mean(|ppf_pred_real - ppf_true_real| / ppf_true_real)      │
#  └───────────────────────────────────────────────────────────────────────────┘

def v5_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # ── Standard MSE terms (in scaled space, same as v4) ──────────────────────
    ppf_max_loss   = W_PPF_MAX   * tf.reduce_mean(tf.square(
        y_true[:, 0] - y_pred[:, 0]))
    ppf_boc_loss   = W_PPF_BOC   * tf.reduce_mean(tf.square(
        y_true[:, 1] - y_pred[:, 1]))
    ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
        - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
    cycle_loss     = W_CYCLE     * tf.reduce_mean(tf.square(
        y_true[:, IDX_CYCLE] - y_pred[:, IDX_CYCLE]))
    rho_loss       = W_RHO       * tf.reduce_mean(tf.square(
        y_true[:, IDX_RHO] - y_pred[:, IDX_RHO]))

    # ── Monotonicity penalty ───────────────────────────────────────────────────
    late       = y_pred[:, IDX_PPF_STEPS_START + 3:IDX_PPF_STEPS_END]
    diffs      = late[:, 1:] - late[:, :-1]
    violations = tf.maximum(0.0, diffs)
    mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))

    # ── NEW: MAPE term — unscale ppf_max, compute percentage error ────────────
    # These are scalars baked in as Python floats — no overhead.
    ppf_true_real = y_true[:, 0] * _PPF_MAX_STD + _PPF_MAX_MEAN
    ppf_pred_real = y_pred[:, 0] * _PPF_MAX_STD + _PPF_MAX_MEAN
    # eps = 0.5: chosen so that at PPF_min≈1.6 the denominator is 2.1 (not tiny)
    # This prevents MAPE from exploding if a prediction is near zero during early epochs
    mape_loss = W_MAPE * tf.reduce_mean(
        tf.abs(ppf_true_real - ppf_pred_real) / (ppf_true_real + 0.5)
    )

    return ppf_max_loss + ppf_boc_loss + ppf_steps_loss + cycle_loss + rho_loss \
           + mono_loss + mape_loss


# =============================================================================
# SECTION 6 — TUNABLE MODEL BUILDER (for keras_tuner)
# =============================================================================

def build_tunable_v5(hp):
    """
    Build function for BayesianOptimization.
    Fixes the Stage-1 winners. Searches the remaining space.
    """
    # ── Params fixed from Stage 1 ──────────────────────────────────────────────
    embed_dim    = EMBED_DIM    # 16 (fixed)
    lr           = LR           # 0.001 (fixed)
    dropout      = 0.2          # fixed — was best

    # ── Params we're still tuning ─────────────────────────────────────────────
    conv_dropout = hp.Choice('conv_dropout', [0.05, 0.10, 0.15])
    num_blocks   = hp.Choice('num_blocks',   [2, 3])
    filter_mode  = hp.Choice('filter_mode',  ['32_64', '64_128', '32_64_128'])
    dense_units  = hp.Choice('dense_units',  [64, 128, 256])
    ppf_head_units = hp.Choice('ppf_head_units', [32, 64, 128])
    weight_decay = hp.Choice('weight_decay', [1e-5, 1e-4, 5e-4])

    if filter_mode == '32_64':
        filters = (32, 64)
    elif filter_mode == '64_128':
        filters = (64, 128)
    else:
        filters = (32, 64, 128)

    # Ensure num_blocks doesn't exceed available filter sizes
    effective_blocks = min(num_blocks, len(filters))

    model = build_multihead_cnn(
        embed_dim      = embed_dim,
        filters        = filters,
        num_blocks     = effective_blocks,
        dense_units    = dense_units,
        dropout        = dropout,
        conv_dropout   = conv_dropout,
        ppf_head_units = ppf_head_units,
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay
        ),
        loss=v5_loss,
        metrics=['mae']
    )
    return model


# =============================================================================
# SECTION 7 — RUN BAYESIAN SEARCH
# =============================================================================
#
# WHY BayesianOptimization OVER Hyperband here?
#   Hyperband is great when you have a WIDE search space with many configs.
#   Stage 1 had 8 params × many values → Hyperband makes sense.
#   Stage 2 has 6 params with narrow ranges → Bayesian is more efficient.
#   Bayesian learns from each trial (fits a surrogate to the loss landscape)
#   and proposes trials in promising regions — fewer wasted trials.
#   With 20 trials it will explore more intelligently than Hyperband's random
#   initial phase, and converge faster.
#
# Tuner directory: 'cnn_v5_tuning' — different from stage 1 ('cnn_tuning')
#   so the two searches don't interfere.

print("\n" + "="*65)
print("STAGE-2 BAYESIAN HYPERPARAMETER SEARCH")
print("="*65)
print(f"  FIXED:  embed_dim={EMBED_DIM}, lr={LR}, dropout=0.2,")
print(f"          multi_head=True, use_attention=False")
print(f"  TUNING: conv_dropout, num_blocks, filter_mode,")
print(f"          dense_units, ppf_head_units, weight_decay")
print(f"  TRIALS: 20 × {EPOCHS_TUNE} epochs (early stop patience=8)")
print(f"  FULL RETRAIN: {EPOCHS_FULL} epochs with best config")
print(f"  NEW LOSS: MSE + MAPE term (W_MAPE={W_MAPE}) to reduce rel_err")
print("="*65 + "\n")

tuner = kt.BayesianOptimization(
    build_tunable_v5,
    objective='val_loss',
    max_trials=20,
    num_initial_points=5,    # 5 random trials to bootstrap the Bayesian model
    directory='cnn_v5_tuning',
    project_name='beavrs_cnn_v5',
    overwrite=False,         # resume if interrupted
)

tuner.search(
    X_tr, Y_tr_sc,
    validation_data=(X_val, Y_val_sc),
    epochs=EPOCHS_TUNE,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,               # shorter patience → faster trials
            restore_best_weights=True
        )
    ],
    verbose=1
)


# =============================================================================
# SECTION 8 — PRINT BEST CONFIG
# =============================================================================

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*65)
print("BEST CONFIGURATION FOUND")
print("="*65)
print(f"  embed_dim      : {EMBED_DIM}  (fixed)")
print(f"  learning_rate  : {LR}  (fixed)")
print(f"  dropout        : 0.2  (fixed)")
print(f"  multi_head     : True  (fixed)")
print(f"  use_attention  : False  (fixed)")
print()
for k, v in best_hp.values.items():
    print(f"  {k:<20}: {v}")
print("="*65 + "\n")


# =============================================================================
# SECTION 9 — FULL RETRAIN WITH BEST CONFIG
# =============================================================================
#
# The tuner's trials used only EPOCHS_TUNE=50 epochs — enough to rank configs
# but not long enough to converge. Retrain the best config to convergence now.
# This is the standard practice after any hyperparameter search.

print(f"[RETRAIN] Building best model for full {EPOCHS_FULL}-epoch training...")
model = tuner.hypermodel.build(best_hp)
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")

t_start = time.time()

full_callbacks = [
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

history = model.fit(
    X_tr, Y_tr_sc,
    validation_data=(X_val, Y_val_sc),
    epochs=EPOCHS_FULL,
    batch_size=BATCH_SIZE,
    callbacks=full_callbacks,
    verbose=0
)

t_train    = time.time() - t_start
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
val_losses  = history.history['val_loss']
train_losses= history.history['loss']
final_ratio = val_losses[best_epoch - 1] / (train_losses[best_epoch - 1] + 1e-9)
print(f"\n[TRAINING DONE]  {t_train:.1f}s  |  best epoch: {best_epoch}")
print(f"  val/train ratio: {final_ratio:.2f}  "
      f"({'✓ good' if final_ratio < 1.3 else '⚠ mild overfit'})\n")


# =============================================================================
# SECTION 10 — EVALUATE
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    Y_main_real = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_real  = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main_real, Y_rho_real], axis=1)

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

ppf_mae      = np.abs(ppf_max_pred - ppf_max_true).mean()
ppf_rel_err  = (np.abs(ppf_max_pred - ppf_max_true) / (ppf_max_true + 1e-6)).mean() * 100
ppf_r2       = r2_score(ppf_max_true, ppf_max_pred)
ppf_pearson  = np.corrcoef(ppf_max_pred, ppf_max_true)[0, 1]

PPF_REPORT_LOW, PPF_REPORT_HIGH = 2.0, 4.5
goal_mask    = (ppf_max_true >= PPF_REPORT_LOW) & (ppf_max_true <= PPF_REPORT_HIGH)
goal_mae     = np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask]).mean()
goal_r2      = r2_score(ppf_max_true[goal_mask], ppf_max_pred[goal_mask])
goal_rel_err = (np.abs(ppf_max_pred[goal_mask] - ppf_max_true[goal_mask])
                / (ppf_max_true[goal_mask] + 1e-6)).mean() * 100

cycle_mae    = np.abs(cycle_pred - cycle_true).mean()
cycle_r2     = r2_score(cycle_true, cycle_pred)

rho_mae      = np.abs(rho_pred - rho_true).mean()
rho_r2       = r2_score(rho_true, rho_pred)
keff_mae_rep = np.abs(keff_pred_rep - keff_true_rep).mean()
keff_r2_rep  = r2_score(keff_true_rep, keff_pred_rep)

# Breakdown of relative error by PPF zone — useful diagnostic
low_mask   = ppf_max_true < 2.5
mid_mask   = (ppf_max_true >= 2.5) & (ppf_max_true < 4.0)
high_mask  = ppf_max_true >= 4.0
rel_low  = (np.abs(ppf_max_pred[low_mask]  - ppf_max_true[low_mask])
            / (ppf_max_true[low_mask]  + 1e-6)).mean() * 100 if low_mask.any()  else float('nan')
rel_mid  = (np.abs(ppf_max_pred[mid_mask]  - ppf_max_true[mid_mask])
            / (ppf_max_true[mid_mask]  + 1e-6)).mean() * 100 if mid_mask.any()  else float('nan')
rel_high = (np.abs(ppf_max_pred[high_mask] - ppf_max_true[high_mask])
            / (ppf_max_true[high_mask] + 1e-6)).mean() * 100 if high_mask.any() else float('nan')

print(f"\n{'='*62}")
print(f"CNN v5 TEST RESULTS")
print(f"{'='*62}")
print(f"  PPF_max (all patterns):")
print(f"    MAE              : {ppf_mae:.4f}   (v4: 0.0843)")
print(f"    Relative error   : {ppf_rel_err:.2f}%    (v4: 3.10%)")
print(f"    R²               : {ppf_r2:.4f}   (v4: 0.9841)")
print(f"    Pearson r        : {ppf_pearson:.4f}")
print(f"")
print(f"  Relative error breakdown by PPF zone:")
print(f"    PPF < 2.5  (n={low_mask.sum():4d}) : {rel_low:.2f}%   ← was highest in v4")
print(f"    2.5–4.0    (n={mid_mask.sum():4d}) : {rel_mid:.2f}%")
print(f"    PPF > 4.0  (n={high_mask.sum():4d}) : {rel_high:.2f}%")
print(f"")
print(f"  PPF_max (goal zone {PPF_REPORT_LOW}–{PPF_REPORT_HIGH}  n={goal_mask.sum()}):")
print(f"    MAE              : {goal_mae:.4f}   (v4: 0.0811)")
print(f"    R²               : {goal_r2:.4f}   (v4: 0.9673)")
print(f"    Relative error   : {goal_rel_err:.2f}%   (new metric — not in v4)")
print(f"")
print(f"  Cycle length:")
print(f"    MAE              : {cycle_mae:.2f} days  (v4: 1.28)")
print(f"    R²               : {cycle_r2:.4f}")
print(f"  Rho_pcm (BOC):")
print(f"    MAE              : {rho_mae:.0f} pcm  (v4: 65)")
print(f"    R²               : {rho_r2:.4f}")
print(f"  keff_boc (derived from rho):")
print(f"    MAE              : {keff_mae_rep:.5f}  (v4: 0.00083)")
print(f"    R²               : {keff_r2_rep:.4f}   (v4: 0.9912)")
print(f"{'='*62}\n")


# =============================================================================
# SECTION 11 — SAVE
# =============================================================================

model.save(MODEL_NAME)

config = {
    'N_POS': N_POS, 'N_TYPES': N_TYPES, 'N_STEPS': N_STEPS,
    'GRID_ROWS': GRID_ROWS, 'GRID_COLS': GRID_COLS,
    'GRID_LAYOUT': GRID_LAYOUT.tolist(),
    'GRID_MASK': GRID_MASK.tolist(),
    'IDX_PPF_MAX': IDX_PPF_MAX, 'IDX_PPF_BOC': IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': IDX_PPF_STEPS_START,
    'IDX_PPF_STEPS_END': IDX_PPF_STEPS_END,
    'IDX_CYCLE': IDX_CYCLE, 'IDX_RHO': IDX_RHO,
    'N_OUTPUTS': N_OUTPUTS,
    'ym_scaler_mean' : ym_scaler.mean_.tolist(),
    'ym_scaler_scale': ym_scaler.scale_.tolist(),
    'yr_scaler_mean' : yr_scaler.mean_.tolist(),
    'yr_scaler_scale': yr_scaler.scale_.tolist(),
    'best_hp': {k: v for k, v in best_hp.values.items()},
    'v4_comparison': {
        'ppf_mae': 0.0843, 'ppf_rel_err': 3.10, 'ppf_r2': 0.9841,
        'cycle_mae': 1.28, 'rho_r2': 0.9912, 'keff_r2': 0.9912
    },
    'v5_results': {
        'ppf_mae': float(ppf_mae), 'ppf_rel_err': float(ppf_rel_err),
        'ppf_r2': float(ppf_r2), 'goal_mae': float(goal_mae),
        'goal_r2': float(goal_r2), 'cycle_mae': float(cycle_mae),
        'rho_r2': float(rho_r2), 'keff_r2': float(keff_r2_rep)
    },
}

with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

# Also save type frequencies for QICA trust-region
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
np.save('train_type_freq_v5.npy', train_type_freq)

print(f"[SAVED]  {MODEL_NAME}")
print(f"[SAVED]  {CONFIG_NAME}")
print(f"[SAVED]  train_type_freq_v5.npy")


# =============================================================================
# SECTION 12 — QUICK DIAGNOSTIC PLOT
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"CNN v5  |  PPF rel_err={ppf_rel_err:.2f}%  MAE={ppf_mae:.3f}  R²={ppf_r2:.3f}",
    fontsize=12, fontweight='bold'
)

# 1. Training curve
ax = axes[0]
ax.plot(history.history['loss'],     '#1B4FBF', lw=1.5, label='Train')
ax.plot(history.history['val_loss'], '#F5A623', lw=1.5, label='Val')
ax.axvline(best_epoch - 1, color='red', lw=1, ls=':', label=f'Best ep {best_epoch}')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_title(f'Training Curve\nval/train={final_ratio:.2f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 2. PPF scatter coloured by zone
ax = axes[1]
lim = [ppf_max_true.min() - 0.1, ppf_max_true.max() + 0.1]
colors = np.where(ppf_max_true < 2.5, '#D62728',
          np.where(ppf_max_true < 4.0, '#17BECF', '#AAAAAA'))
ax.scatter(ppf_max_true, ppf_max_pred, c=colors, alpha=0.35, s=6)
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'PPF Scatter\nRed=low(<2.5) Teal=mid Blue=high')
ax.grid(alpha=0.3)

# 3. Relative error histogram per zone
ax = axes[2]
bins = np.linspace(0, 15, 40)
if low_mask.any():
    rel_err_low  = np.abs(ppf_max_pred[low_mask]  - ppf_max_true[low_mask])  / ppf_max_true[low_mask]  * 100
    ax.hist(rel_err_low, bins=bins, alpha=0.6, color='#D62728', label=f'PPF<2.5 ({rel_low:.1f}%)')
if mid_mask.any():
    rel_err_mid  = np.abs(ppf_max_pred[mid_mask]  - ppf_max_true[mid_mask])  / ppf_max_true[mid_mask]  * 100
    ax.hist(rel_err_mid, bins=bins, alpha=0.6, color='#17BECF', label=f'2.5–4.0 ({rel_mid:.1f}%)')
if high_mask.any():
    rel_err_high = np.abs(ppf_max_pred[high_mask] - ppf_max_true[high_mask]) / ppf_max_true[high_mask] * 100
    ax.hist(rel_err_high, bins=bins, alpha=0.6, color='#AAAAAA', label=f'PPF>4.0 ({rel_high:.1f}%)')
ax.set_xlabel('Relative error (%)'); ax.set_ylabel('Count')
ax.set_title(f'Relative Error by Zone\nOverall: {ppf_rel_err:.2f}%')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_v5_results.png', dpi=150, bbox_inches='tight')
print(f"[SAVED]  cnn_v5_results.png")

print("\n" + "="*62)
print("NEXT STEPS")
print("="*62)
print(f"  If rel_err improved: use cnn_v5_model.keras in your QICA.")
print(f"  If rel_err is still >2.5%: increase W_MAPE from {W_MAPE} to 2.0")
print(f"  and retrain (Section 9 only, skip the tuner).")
print(f"  QICA target remains ≈2.0 PPF (10th percentile of training data).")
print("="*62)