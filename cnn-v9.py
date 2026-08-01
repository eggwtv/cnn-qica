"""
=============================================================================
cnn_v9.py  —  BEAVRS CNN v4  |  Fixed keff  |  Min-PPF Surrogate
=============================================================================

CHANGES FROM THE ORIGINAL (bug-fixes):
──────────────────────────────────────────────────────────────────────────
FIX 1 — ConvResBlock serialization (CRITICAL for QICA):
  Added @keras.saving.register_keras_serializable() decorator to ConvResBlock.
  Without this, model.save() writes the model BUT Keras cannot reconstruct
  ConvResBlock when loading — the class name is stored but the definition
  isn't. This caused the QICA to crash with:
      TypeError: Could not locate class 'ConvResBlock'
  The decorator registers the class in a global Keras registry so any
  subsequent load_model() call can find it automatically.

FIX 2 — Overfitting reduction (mild):
  The training log showed a val/train gap at epoch 75 (val≈0.60, train≈0.33).
  This is mild overfitting — test R² stayed ≥0.98 — but worth addressing.
  Root cause: ConvResBlock used dropout=0.0 (NO dropout in conv layers).
  Fix: ConvResBlock now accepts and uses a dropout rate (default 0.1).
  Also added weight_decay=1e-4 to Adam for L2 regularisation.
  Expected effect: slightly slower convergence, better generalisation.

FIX 3 — Broken openmc_simulate stub removed:
  The original openmc_simulate() had the raise NotImplementedError INSIDE
  the docstring (triple-quotes), so it silently fell through to:
      from openmc_beavrs_simulator import simulate as _omc_sim
  which crashes if openmc isn't installed. Since AL_ROUNDS=0 the active
  learning loop never called it, but it's confusing. Removed entirely.
  The AL section now just identifies candidates and saves them to CSV.

FIX 4 — Saves train_type_freq.npy:
  The QICA trust-region uses per-position assembly-type frequencies from
  the training set. If this file is missing the QICA falls back to uniform
  (all 9 types equally likely at every position), which weakens the trust
  penalty. Now saved automatically after training.

OUTPUTS:
  cnn_v4_model.keras         — trained model (serializable, loads in QICA)
  cnn_v4_config.json         — geometry, scalers, indices (for QICA)
  cnn_v4_sens.csv            — position sensitivities
  cnn_v4_results.png         — evaluation plots
  cnn_v4_al_candidates.csv   — active learning query candidates
  train_type_freq.npy        — per-position type frequencies (for QICA trust-region)
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
#from tensorflow.keras.utils import register_keras_serializable

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print(f"Running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print("cnn_v9.py — BEAVRS CNN  |  Fixed keff  |  Min-PPF Surrogate\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'
MODEL_NAME  = 'cnn_v9_model.keras'
CONFIG_NAME = 'cnn_v9_config.json'
SENS_NAME   = 'cnn_v9_sens.csv'
PLOT_NAME   = 'cnn_v9_results.png'
AL_CSV      = 'cnn_v9_al_candidates.csv'
FREQ_PATH   = 'train_type_freq_v9.npy'

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

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 400
LR          = 0.001
DROPOUT     = 0.15      # shared-head dropout
CONV_DROP   = 0.10      # FIX: conv-block dropout (was 0.0 → caused mild overfitting)
WEIGHT_DECAY= 1e-4      # FIX: L2 regularisation via Adam weight_decay
TEST_FRAC   = 0.15
VAL_FRAC    = 0.15
SEED        = 42
MC_SAMPLES  = 30

AL_UNCERTAINTY_THRESHOLD = 0.07
AL_MAX_QUERIES_PER_ROUND = 50


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

# keff → ρ_pcm: better numerical properties than raw keff for the loss
keff_raw = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
rho_pcm  = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

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
print(f"  rho_pcm range      : {rho_pcm.min():.0f} – {rho_pcm.max():.0f} pcm")
print("=" * 58)
print()


# =============================================================================
# SECTION 4 — TARGETS AND FEATURES
# =============================================================================
#
# Output layout (total N_OUTPUTS = 35):
#   [0]             ppf_max         — global cycle maximum PPF  ← PRIMARY
#   [1]             ppf_boc         — beginning-of-cycle PPF
#   [2 : 2+N_STEPS] ppf_steps       — max PPF at each of 31 burnup steps
#   [2+N_STEPS]     cycle_length    — effective full-power days
#   [3+N_STEPS]     rho_pcm         — reactivity at BOC (pcm)
#
# The last column (rho) gets its OWN scaler — this prevents the small-variance
# keff signal from being swamped by PPF/cycle gradients in a shared scaler.

N_OUTPUTS     = 1 + 1 + N_STEPS + 1 + 1   # = 35
IDX_PPF_MAX   = 0
IDX_PPF_BOC   = 1
IDX_PPF_STEPS = slice(2, 2 + N_STEPS)      # 2:33
IDX_CYCLE     = 2 + N_STEPS               # 33
IDX_RHO       = 3 + N_STEPS               # 34

# Aliases for loss function
IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS         # 33
IDX_CYCLE_v4        = IDX_CYCLE           # 33
IDX_RHO_v4          = IDX_RHO             # 34

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_rho       = rho_pcm.reshape(-1, 1).astype(np.float32)

Y_main    = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle], axis=1)  # (N, 34)
Y_rho_col = Y_rho                                                                    # (N, 1)

# 6×6 integer grid input
X_raw  = df[load_cols].values.astype(np.int32)
X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1

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

print(f"[SCALING]")
print(f"  ppf_max     mean={ym_scaler.mean_[0]:.3f}  std={ym_scaler.scale_[0]:.3f}")
print(f"  cycle_len   mean={ym_scaler.mean_[IDX_CYCLE]:.1f}  std={ym_scaler.scale_[IDX_CYCLE]:.2f}")
print(f"  rho_pcm     mean={yr_scaler.mean_[0]:.0f}   std={yr_scaler.scale_[0]:.0f}")
print()


# =============================================================================
# SECTION 6 — ARCHITECTURE
# =============================================================================
#
# WHY @keras.saving.register_keras_serializable() IS MANDATORY HERE:
# ─────────────────────────────────────────────────────────────────────
# When you call model.save('model.keras'), Keras serialises the model's
# layer graph to JSON. For built-in layers (Dense, Conv2D, etc.) Keras
# already knows how to reconstruct them from that JSON.
#
# For custom subclassed layers like ConvResBlock, Keras stores the class
# name as a string. On load, it needs to look up that string in a global
# registry to find the actual Python class.
#
# The decorator registers the class under its name in that global registry.
# Without it:
#   model = keras.models.load_model('model.keras')
#   → TypeError: Could not locate class 'ConvResBlock'
#
# With it:
#   ConvResBlock is in the registry → load succeeds.
#
# The same decorator must be present in any script that calls load_model()
# (i.e. qica-cnn-v4.py), because the registry is Python-process-local.
# The QICA script re-defines the class with the same decorator — this is
# intentional and correct.

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """
    Residual convolutional block:
      Conv → BN → GELU → Conv → BN → Add(shortcut) → GELU → Dropout

    The residual shortcut allows gradients to flow directly back through
    the skip connection, avoiding vanishing gradients in deep stacks.
    The projection conv (1×1) handles dimension changes when filters change.

    OVERFITTING FIX: dropout rate now applied within the conv block itself,
    not just in the Dense head. With CONV_DROP=0.1 and 3 stacked blocks
    this provides substantial regularisation over the spatial features.
    """
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        self.proj  = None
        # Store dropout rate as instance variable so get_config() can save it
        self._dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
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
        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)
        return h

    def get_config(self):
        # get_config() is required for serialisation — Keras calls this
        # when saving, and from_config() when loading.
        cfg = super().get_config()
        cfg.update({'filters': self._filters,
                    'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


def build_cnn_v9(
    grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
    n_types=N_TYPES + 1,
    embed_dim=16,
    filters=(32, 64),
    dense_units=128,
    dropout=DROPOUT,
    conv_dropout=CONV_DROP,
    n_outputs=N_OUTPUTS,
):
    """
    Multi-head CNN surrogate:
      Input (6×6 int grid) → Embedding → 3×ConvResBlock → Spatial Attention
      → GlobalAvgPool → Shared Dense → [PPF head | Cycle head | Rho head]
      → Concatenate → (B, 35)

    Spatial attention: a 1×1 conv that learns which spatial positions
    matter most for the prediction — effectively a position importance map.
    Positions near the core centre typically get higher attention weights
    because neutron flux is highest there.
    """
    inp = keras.Input(shape=(grid_rows, grid_cols), dtype='int32', name='loading_grid')
    x   = layers.Embedding(n_types, embed_dim, name='assembly_embedding')(inp)

    # Three stacked residual conv blocks — CONV_DROP adds within-block regularisation
    x = ConvResBlock(filters[0], dropout=conv_dropout, name='conv_block_1')(x)
    x = ConvResBlock(filters[1], dropout=conv_dropout, name='conv_block_2')(x)
    #x = ConvResBlock(filters[2], dropout=conv_dropout, name='conv_block_3')(x)

    # Spatial attention: learn which grid positions matter most
    #attn  = layers.Conv2D(1, 1, padding='same', activation='sigmoid',
    #                       name='spatial_attention')(x)
    #x     = layers.Multiply(name='attended_features')([x, attn])
    x     = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Shared Dense trunk — features learned jointly before the heads split
    shared = layers.Dense(dense_units, activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu', name='shared_dense2')(shared)

    # PPF head (33 outputs: ppf_max + ppf_boc + 31 step values)
    h_ppf    = layers.Dense(64, activation='gelu', name='ppf_dense')(shared)
    h_ppf    = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf  = layers.Dense(1 + 1 + N_STEPS, activation='linear', name='ppf_output')(h_ppf)

    # Cycle length head (1 output)
    h_cyc    = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc    = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle= layers.Dense(1, activation='linear', name='cycle_output')(h_cyc)

    # Rho/keff head (1 output) — separate branch to avoid gradient swamping
    h_rho    = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho    = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho  = layers.Dense(1, activation='linear', name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    return keras.Model(inputs=inp, outputs=out, name='BEAVRS_CNN_v9')


model = build_cnn_v9()
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")


# =============================================================================
# SECTION 7 — LOSS FUNCTION
# =============================================================================
#
# Weighted per-head MSE + physics monotonicity penalty.
#
# WHY WEIGHTED HEADS:
#   ppf_max is the primary safety output → 3× weight.
#   rho_pcm has small variance relative to PPF/cycle → 5× weight to ensure
#   the gradient signal isn't drowned out.
#
# WHY MONOTONICITY PENALTY:
#   After step 3 (BA burnout), the PPF profile should only decrease.
#   Penalising upward steps teaches the model the physical constraint,
#   improving generalisation to unseen patterns.

W_PPF_MAX   = 3.0
W_PPF_BOC   = 2.0
W_PPF_STEPS = 0.5
W_CYCLE     = 1.0
W_RHO       = 5.0
W_MONO      = 0.01
W_LOG       = 0.0

@tf.keras.utils.register_keras_serializable()
def v9_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    ppf_max_loss   = W_PPF_MAX   * tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
    ppf_boc_loss   = W_PPF_BOC   * tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
    ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
        - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
    cycle_loss     = W_CYCLE     * tf.reduce_mean(tf.square(
        y_true[:, IDX_CYCLE_v4] - y_pred[:, IDX_CYCLE_v4]))
    rho_loss       = W_RHO       * tf.reduce_mean(tf.square(
        y_true[:, IDX_RHO_v4] - y_pred[:, IDX_RHO_v4]))

    # Monotonicity penalty: penalise upward steps in late-cycle PPF
    late       = y_pred[:, IDX_PPF_STEPS_START + 3:IDX_PPF_STEPS_END]
    diffs      = late[:, 1:] - late[:, :-1]
    violations = tf.maximum(0.0, diffs)
    mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))

    return ppf_max_loss + ppf_boc_loss + ppf_steps_loss + cycle_loss + rho_loss + mono_loss


# =============================================================================
# SECTION 8 — TRAIN
# =============================================================================
#
# OVERFITTING FIX: Adam now uses weight_decay=1e-4 (L2 regularisation).
# This penalises large weight magnitudes at every update step, discouraging
# the model from memorising training patterns.
# Combined with the increased conv-block dropout (CONV_DROP=0.1), this
# should close the val/train gap seen at epoch ~75 in the previous run.

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
    loss=v9_loss,
    metrics=['mae']
)

print("[TRAINING] Starting CNN v9 ...")
print(f"  Dataset      : {len(X_tr)} train + {len(X_val)} val patterns")
print(f"  Loss         : weighted per-head MSE + monotonicity penalty")
print(f"  Rho weight   : {W_RHO}× | Conv dropout: {CONV_DROP} | Weight decay: {WEIGHT_DECAY}")
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
            + (" ← gap!" if logs['val_loss'] > 1.5 * logs['loss'] else "")
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
train_losses = history.history['loss']
val_losses   = history.history['val_loss']
# Check overfitting: final val/train ratio (close to 1.0 is ideal)
final_ratio = val_losses[best_epoch - 1] / (train_losses[best_epoch - 1] + 1e-9)

print(f"\n[TRAINING DONE]  {t_train:.1f}s  |  best epoch: {best_epoch}")
print(f"  Final val/train ratio at best epoch: {final_ratio:.2f}  "
      f"({'good' if final_ratio < 1.3 else 'mild overfit — consider more dropout'})\n")


# =============================================================================
# SECTION 9 — INVERSE TRANSFORM HELPER
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    """Invert the two-scaler scheme back to real physical units."""
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
cycle_pred    = Y_pred_real[:, IDX_CYCLE_v4]
cycle_true    = Y_true_real[:, IDX_CYCLE_v4]
rho_pred      = Y_pred_real[:, IDX_RHO_v4]
rho_true      = Y_true_real[:, IDX_RHO_v4]

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
print(f"CNN v9 TEST RESULTS")
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
])  # (MC_SAMPLES, N_test, 35)

mc_mean_sc = mc_preds_sc.mean(axis=0)
mc_std_sc  = mc_preds_sc.std(axis=0)

mc_mean_real = inverse_transform(mc_mean_sc)
mc_std_main  = mc_std_sc[:, :34] * ym_scaler.scale_
mc_std_rho   = mc_std_sc[:, 34:35] * yr_scaler.scale_
mc_std_real  = np.concatenate([mc_std_main, mc_std_rho], axis=1)

ppf_mc_mean  = mc_mean_real[:, IDX_PPF_MAX]
ppf_mc_std   = mc_std_real[:, IDX_PPF_MAX]

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
#
# WHAT THIS SECTION DOES (no simulator required):
#   1. Runs MC Dropout over the ENTIRE dataset
#   2. Identifies patterns that are BOTH:
#        - Uncertain (σ_ppf ≥ AL_UNCERTAINTY_THRESHOLD): the CNN is not
#          confident about these → they need real simulation to label correctly
#        - Low-PPF (bottom 25th percentile prediction): these are the patterns
#          most likely to be genuinely good if the CNN is right
#   3. Saves these candidates to CSV
#
# HOW TO USE THE CANDIDATES:
#   When you have a physics simulator (OpenMC, PARCS, Serpent, etc.):
#     1. Feed cnn_v4_al_candidates.csv patterns to the simulator
#     2. Get ground-truth ppf_max, cycle_length, keff for each
#     3. Add those rows to your training CSV
#     4. Re-run cnn_v4.py to retrain on the expanded dataset
#     5. The new model will be more accurate in the low-PPF region
#   This is "active learning": using the model's own uncertainty to choose
#   which simulations are most valuable to run.
#
# ROUNDS: Set AL_ROUNDS > 0 in qica-cnn-v4.py AFTER connecting a simulator.

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

ppf_full_pred   = mc_mean_full[:, IDX_PPF_MAX]
ppf_full_std    = mc_std_full_phy[:, IDX_PPF_MAX]

priority_score  = ppf_full_std / (ppf_full_pred + 1e-6)
high_unc_mask   = ppf_full_std >= AL_UNCERTAINTY_THRESHOLD
low_ppf_mask    = ppf_full_pred <= np.percentile(ppf_full_pred, 25)
query_mask      = high_unc_mask & low_ppf_mask
query_idxs      = np.where(query_mask)[0]
query_sorted    = query_idxs[np.argsort(priority_score[query_idxs])[::-1]]
query_top       = query_sorted[:AL_MAX_QUERIES_PER_ROUND]

print(f"  High-uncertainty (σ≥{AL_UNCERTAINTY_THRESHOLD}) : {high_unc_mask.sum()}")
print(f"  Low-PPF (bottom quartile)         : {low_ppf_mask.sum()}")
print(f"  Priority candidates               : {len(query_idxs)}")
print(f"  Top-{AL_MAX_QUERIES_PER_ROUND} flagged for simulator  : {len(query_top)}")

# Save the flat (31-position) loading patterns for each candidate
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
        'cycle_length' : float(mc_mean_full[idx, IDX_CYCLE_v4]),
        'rho_pcm_boc'  : float(mc_mean_full[idx, IDX_RHO_v4]),
        **{f'pos_{j}': pattern_flat[j] for j in range(N_POS)},
    })

al_df = pd.DataFrame(al_records)
al_df.to_csv(AL_CSV, index=False)
print(f"  Saved query candidates → {AL_CSV}")
print(f"  → Run these through your simulator, add results to training data,")
print(f"    then retrain to improve accuracy in the low-PPF region.\n")


# =============================================================================
# SECTION 14 — SAVE MODEL + CONFIG + TRUST REGION
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
    'IDX_CYCLE': IDX_CYCLE_v4,
    'IDX_RHO': IDX_RHO_v4,
    'N_OUTPUTS': N_OUTPUTS,
    'PPF_REPORT_LOW': PPF_REPORT_LOW,
    'PPF_REPORT_HIGH': PPF_REPORT_HIGH,
    'ym_scaler_mean' : ym_scaler.mean_.tolist(),
    'ym_scaler_scale': ym_scaler.scale_.tolist(),
    'yr_scaler_mean' : yr_scaler.mean_.tolist(),
    'yr_scaler_scale': yr_scaler.scale_.tolist(),
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v) for k, v in ASSEMBLY_CYCLE_EQUIV.items()},
    'mc_samples': MC_SAMPLES,
    'al_uncertainty_thr': AL_UNCERTAINTY_THRESHOLD,
    'test_ppf_mae'   : float(ppf_mae),
    'test_ppf_r2'    : float(ppf_r2),
    'test_goal_mae'  : float(goal_mae),
    'test_cycle_mae' : float(cycle_mae),
    'test_rho_r2'    : float(rho_r2),
    'test_keff_r2'   : float(keff_r2_rep),
}

with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

# Save per-position type frequency for QICA trust-region
# type_freq[pos, t] = fraction of training patterns that have type (t+1) at position pos
# The QICA uses this to penalise patterns that use unusual type/position combos.
tr_flat = np.stack([
    X_tr[:, r, c]
    for r in range(GRID_ROWS) for c in range(GRID_COLS)
    if GRID_LAYOUT[r, c] >= 0
], axis=1)   # (N_train, 31)

train_type_freq = np.zeros((N_POS, N_TYPES), dtype=np.float32)
for p in range(N_POS):
    for t in range(1, N_TYPES + 1):
        train_type_freq[p, t - 1] = float((tr_flat[:, p] == t).mean())
# Smooth: add small floor so log doesn't blow up for rarely seen combos
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
    f"BEAVRS CNN v4  |  PPF MAE={ppf_mae:.3f}  R²={ppf_r2:.3f}  "
    f"keff R²={keff_r2_rep:.3f}  best_ep={best_epoch}",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1. Training curve + overfitting diagnostic
ax = fig.add_subplot(gs[0, 0])
ax.plot(history.history['loss'],     '#1B4FBF', lw=1.5, label='Train')
ax.plot(history.history['val_loss'], '#F5A623', lw=1.5, label='Val')
ax.axvline(best_epoch - 1, color='red', lw=1, ls=':', label=f'Best ep {best_epoch}')
ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
ax.set_title('Training Curve\n(val/train ratio near best epoch)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
# Add ratio annotation
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
ax.axhspan(PPF_REPORT_LOW, PPF_REPORT_HIGH, alpha=0.07, color='teal', label='Goal zone')
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'PPF Prediction\nMAE={ppf_mae:.3f}  R²={ppf_r2:.3f}')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 3. keff scatter
ax = fig.add_subplot(gs[0, 2])
ax.scatter(keff_true_rep, keff_pred_rep, alpha=0.3, s=7, color='#9467BD')
lim_k = [keff_true_rep.min() - 0.003, keff_true_rep.max() + 0.003]
ax.plot(lim_k, lim_k, 'k--', lw=1)
ax.set_xlabel('True keff_boc'); ax.set_ylabel('Predicted keff_boc')
ax.set_title(f'keff_boc\nMAE={keff_mae_rep:.5f}  R²={keff_r2_rep:.3f}')
ax.grid(alpha=0.3)

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
ax.axvline(p10, color='#2CA02C', lw=2, ls='--', label=f'10th pct = {p10:.2f}')
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
            disp_sens[r, c] = sens_norm[pos_i]; pos_i += 1
cmap_s = plt.cm.RdYlGn_r.copy(); cmap_s.set_bad('lightgrey')
im = ax.imshow(disp_sens, cmap=cmap_s, aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Norm. sensitivity')
pos_i = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            ax.text(c, r, f'P{pos_i}', ha='center', va='center', fontsize=6); pos_i += 1
ax.set_title('∂ppf_max / ∂position\n(Red = critical)'); ax.set_xticks([]); ax.set_yticks([])

# 8. Active learning candidates
ax = fig.add_subplot(gs[1, 3])
ax.scatter(ppf_full_pred, ppf_full_std, alpha=0.15, s=4, color='#AAAAAA', label='All patterns')
if len(query_top) > 0:
    ax.scatter(ppf_full_pred[query_top], ppf_full_std[query_top],
               alpha=0.8, s=20, color='#D62728', zorder=5,
               label=f'AL candidates (n={len(query_top)})')
ax.axhline(AL_UNCERTAINTY_THRESHOLD, color='orange', lw=1.5, ls='--', label='σ threshold')
ax.axvline(np.percentile(ppf_full_pred, 25), color='teal', lw=1.5, ls='--', label='PPF 25th pct')
ax.set_xlabel('Predicted ppf_max'); ax.set_ylabel('MC σ (ppf_max)')
ax.set_title('Active Learning Candidates\n(high-σ, low-PPF → simulate these)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 9. Best loading pattern grid
ax = fig.add_subplot(gs[2, 0])
best_idx = ppf_max_true.argmin()
g_disp = X_test[best_idx].astype(float).copy(); g_disp[~GRID_MASK] = np.nan
cmap_ex = plt.cm.YlOrRd.copy(); cmap_ex.set_bad('lightgrey')
im_ex = ax.imshow(g_disp, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
plt.colorbar(im_ex, ax=ax, label='Type')
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_MASK[r, c]:
            ax.text(c, r, f'{X_test[best_idx, r, c]}', ha='center', va='center', fontsize=8)
ax.set_title(f'Best Pattern (lowest true PPF)\nTrue={ppf_max_true[best_idx]:.3f}  '
             f'Pred={ppf_max_pred[best_idx]:.3f}'); ax.set_xticks([]); ax.set_yticks([])

# 10. Worst loading pattern grid
ax = fig.add_subplot(gs[2, 1])
worst_idx = ppf_max_true.argmax()
g_disp2 = X_test[worst_idx].astype(float).copy(); g_disp2[~GRID_MASK] = np.nan
im_ex2 = ax.imshow(g_disp2, cmap=cmap_ex, aspect='auto', vmin=1, vmax=9)
plt.colorbar(im_ex2, ax=ax, label='Type')
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_MASK[r, c]:
            ax.text(c, r, f'{X_test[worst_idx, r, c]}', ha='center', va='center', fontsize=8)
ax.set_title(f'Worst Pattern (highest true PPF)\nTrue={ppf_max_true[worst_idx]:.3f}  '
             f'Pred={ppf_max_pred[worst_idx]:.3f}'); ax.set_xticks([]); ax.set_yticks([])

# 11. PPF burnup profile
ax = fig.add_subplot(gs[2, 2])
steps_range  = np.arange(N_STEPS)
true_smean   = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
true_sstd    = Y_true_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
pred_smean   = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].mean(axis=0)
pred_sstd    = Y_pred_real[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END].std(axis=0)
ax.plot(steps_range, true_smean, '#1B4FBF', lw=2, label='True')
ax.fill_between(steps_range, true_smean - true_sstd, true_smean + true_sstd, color='#1B4FBF', alpha=0.15)
ax.plot(steps_range, pred_smean, '#F5A623', lw=2, ls='--', label='Predicted')
ax.fill_between(steps_range, pred_smean - pred_sstd, pred_smean + pred_sstd, color='#F5A623', alpha=0.15)
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
# SECTION 16 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 62)
print("cnn_v4.py  FINAL SUMMARY")
print("=" * 62)
print(f"  Architecture     : 32_64_MSE")
print(f"  ConvResBlock     : serializable (@register_keras_serializable)")
print(f"  Conv dropout     : {CONV_DROP}  (was 0.0 — fixes mild overfitting)")
print(f"  Weight decay     : {WEIGHT_DECAY}  (AdamW L2 regularisation)")
print(f"  Parameters       : {model.count_params():,}")
print(f"  Best epoch       : {best_epoch} / {EPOCHS}")
print(f"  Training time    : {t_train:.1f}s")
print(f"  Val/train ratio  : {final_ratio:.2f}  ({'✓ good' if final_ratio < 1.3 else '⚠ mild overfit'})")
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
print(f"    MAE            : {keff_mae_rep:.5f}")
print(f"    R²             : {keff_r2_rep:.4f}")
print()
print(f"  MC Dropout σ_ppf (mean)  : {ppf_mc_std.mean():.4f}")
print(f"  AL candidates identified : {len(query_top)}")
print()
print(f"  OUTPUT FILES:")
print(f"    {MODEL_NAME}    — serializable model (loads correctly in QICA)")
print(f"    {CONFIG_NAME}   — scalers + indices")
print(f"    {SENS_NAME}     — position sensitivities")
print(f"    {PLOT_NAME}     — evaluation plots")
print(f"    {AL_CSV}        — AL query candidates")
print(f"    {FREQ_PATH}     — trust-region frequencies for QICA")
print()
print(f"  NEXT STEP: Run qica-cnn-v4.py to optimise loading patterns.")
print("=" * 62)