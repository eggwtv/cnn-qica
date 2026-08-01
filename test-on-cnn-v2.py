"""
=============================================================================
cnn_arch_compare.py  —  Architecture & Loss Comparison
=============================================================================

PURPOSE:
  Systematically isolates two variables that changed together across v4→v7:
    (1) Conv filter stack: [32,64]  vs  [64,128]  vs  [32,64,128]+attention
    (2) Loss function:     MSE-only  vs  log-dominant (W_PPF=1.5, W_LOG=4.0)

  6 experiments total (3 arch × 2 loss), all multi-head output.

WHAT THIS ANSWERS:
  • Is the [32,64,128]+attention architecture actually better, or was v4's
    edge caused by the lucky val/train=0.89 split?
  • Does log-dominant loss beat MSE across ALL architectures, or only some?
  • Which combination gives rel_err < 3.10% without overfitting?

REGULARISATION STRATEGY (to keep val/train ≤ 1.15):
  • DROPOUT_TRUNK = 0.15  (back to v4 level — v6/v7 used 0.20)
  • CONV_DROP     = 0.10  (kept)
  • WEIGHT_DECAY  = 1e-4  (kept)
  • EarlyStopping patience=30 (v4 level, not 40 like v6/v7)
  These are intentionally conservative. The goal is a clean architecture
  comparison, not squeezing out the last 0.1% from any single config.

EPOCHS: 300 per experiment (enough to converge, faster than 400).

OUTPUTS:
  cnn_compare_results.csv   — full metric table for all 6 experiments
  cnn_compare_results.png   — bar chart comparison
  cnn_compare_{name}.keras  — saved model for each experiment
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
print("cnn_arch_compare.py — Architecture × Loss Comparison\n")


# =============================================================================
# SECTION 1 — EXPERIMENT CONFIGURATION
# =============================================================================
#
# Define all experiments here. Each entry specifies:
#   name        — output file prefix and table label
#   filters     — list of filter counts per conv block
#   use_attn    — whether to add spatial attention after the conv stack
#   loss_mode   — 'mse'  → pure MSE (v4 baseline)
#                 'log'  → log-dominant (W_PPF=1.5, W_LOG=4.0)
#
# WHY THESE THREE ARCHITECTURES:
#   A [32,64]:      Small, fast, same as early v4 prototypes
#   B [64,128]:     What v5–v7 used (tuner result at 40 ep)
#   C [32,64,128]:  The ACTUAL v4 architecture — 3 progressive blocks
#                   + spatial attention. Never retested with v6/v7-era loss.
#
# WHY THESE TWO LOSS FUNCTIONS:
#   MSE:  Baseline — minimises absolute error, ignores relative magnitude.
#         Reproduces v4 training conditions exactly.
#   LOG:  Log-dominant — W_PPF=1.5 < W_LOG=4.0, so the log-space term
#         (which ≈ squared relative error) genuinely LEADS the gradient.
#         In v6/v7 they were equal (both at 3.0), so MSE still competed.
#         This is the version recommended but never run.

EXPERIMENTS = [
    # ── Phase 1: Architecture comparison with MSE loss ───────────────────────
    {
        "name"     : "A_32_64_MSE",
        "filters"  : [32, 64],
        "use_attn" : False,
        "loss_mode": "mse",
        "label"    : "[32,64]  MSE",
    },
    {
        "name"     : "B_64_128_MSE",
        "filters"  : [64, 128],
        "use_attn" : False,
        "loss_mode": "mse",
        "label"    : "[64,128]  MSE",
    },
    {
        "name"     : "C_32_64_128_attn_MSE",
        "filters"  : [32, 64, 128],
        "use_attn" : True,
        "loss_mode": "mse",
        "label"    : "[32,64,128]+attn  MSE  ← v4 architecture",
    },
    # ── Phase 2: Same architectures with log-dominant loss ───────────────────
    {
        "name"     : "A_32_64_LOG",
        "filters"  : [32, 64],
        "use_attn" : False,
        "loss_mode": "log",
        "label"    : "[32,64]  LOG",
    },
    {
        "name"     : "B_64_128_LOG",
        "filters"  : [64, 128],
        "use_attn" : False,
        "loss_mode": "log",
        "label"    : "[64,128]  LOG",
    },
    {
        "name"     : "C_32_64_128_attn_LOG",
        "filters"  : [32, 64, 128],
        "use_attn" : True,
        "loss_mode": "log",
        "label"    : "[32,64,128]+attn  LOG",
    },
]

# ── Shared hyperparameters ────────────────────────────────────────────────────
EMBED_DIM      = 16
DENSE_UNITS    = 128      # trunk width — reverted to confirmed-best
PPF_HEAD_UNITS = 64       # PPF sub-head first layer
DROPOUT_TRUNK  = 0.15     # v4 level (0.20 in v6/v7 → val/train crept up)
CONV_DROP      = 0.10
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 128
EPOCHS         = 300      # shorter for comparison; train winner at 400
LR             = 1e-3
TEST_FRAC      = 0.15
VAL_FRAC       = 0.15
SEED           = 42
MC_SAMPLES     = 20       # fewer for speed in comparison runs

# ── Loss weight configurations ────────────────────────────────────────────────
LOSS_CONFIGS = {
    "mse": {
        # Pure weighted MSE — exactly v4's loss
        "W_PPF_MAX"  : 3.0,
        "W_PPF_BOC"  : 2.0,
        "W_PPF_STEPS": 0.5,
        "W_CYCLE"    : 1.0,
        "W_RHO"      : 5.0,
        "W_MONO"     : 0.01,
        "W_LOG"      : 0.0,   # OFF
        "desc"       : "MSE-only (v4 baseline)",
    },
    "log": {
        # Log-dominant: log-space term LEADS the ppf_max gradient.
        # W_PPF_MAX reduced so MSE doesn't compete with log-MSE.
        # W_LOG=4.0 > W_PPF_MAX=1.5 → relative error is the primary objective.
        "W_PPF_MAX"  : 1.5,   # reduced: let log term lead
        "W_PPF_BOC"  : 2.0,
        "W_PPF_STEPS": 0.5,
        "W_CYCLE"    : 1.0,
        "W_RHO"      : 5.0,
        "W_MONO"     : 0.01,
        "W_LOG"      : 4.0,   # dominant: > W_PPF_MAX
        "desc"       : "Log-dominant (W_PPF=1.5, W_LOG=4.0)",
    },
}

# ── Output files ──────────────────────────────────────────────────────────────
BEAVRS_CSV = 'ml_dataset_constrained.csv'
XL_FILE    = 'cycle_length_summary.xlsx'
RESULTS_CSV = 'cnn_compare_results.csv'
RESULTS_PNG = 'cnn_compare_results.png'

# ── BEAVRS geometry ───────────────────────────────────────────────────────────
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

N_OUTPUTS           = 1 + 1 + N_STEPS + 1 + 1   # 35
IDX_PPF_MAX         = 0
IDX_PPF_BOC         = 1
IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS
IDX_CYCLE           = 2 + N_STEPS
IDX_RHO             = 3 + N_STEPS

LOG_CLAMP = 0.5

# These are set once after scaler.fit() and used inside every loss function
_PPF_MAX_MEAN = None
_PPF_MAX_STD  = None


# =============================================================================
# SECTION 2 — DATA LOADING (once, shared across all experiments)
# =============================================================================

print("=" * 65)
print("DATA LOADING")
print("=" * 65)

if os.path.exists(XL_FILE):
    xl_df = pd.read_excel(XL_FILE, sheet_name='Cycle_Lengths')
    monocore_map = dict(zip(xl_df['fa_id'].astype(int),
                            xl_df['monocore_cycle_length'].astype(float)))
else:
    monocore_map = {1:172.9, 2:366.9, 3:323.2, 4:299.8, 5:519.9,
                   6:504.9, 7:475.3, 8:471.6, 9:454.7}

if not os.path.exists(BEAVRS_CSV):
    print(f"[ERROR] {BEAVRS_CSV} not found."); sys.exit(1)

df = pd.read_csv(BEAVRS_CSV, skiprows=1, engine='python', on_bad_lines='skip')
print(f"  Loaded {len(df)} patterns × {df.shape[1]} columns")

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
keff_raw       = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
rho_pcm        = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

Y_main    = np.concatenate([
    ppf_global_max.reshape(-1,1), ppf_boc.reshape(-1,1),
    step_max_ppf,
    df['cycle_length'].values.reshape(-1,1)
], axis=1).astype(np.float32)
Y_rho_col = rho_pcm.reshape(-1, 1).astype(np.float32)

X_raw  = df[load_cols].values.astype(np.int32)
X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1

(X_tr, X_tmp, Ym_tr, Ym_tmp, Yr_tr, Yr_tmp) = train_test_split(
    X_grid, Y_main, Y_rho_col,
    test_size=TEST_FRAC + VAL_FRAC, random_state=SEED
)
(X_val, X_test, Ym_val, Ym_test, Yr_val, Yr_test) = train_test_split(
    X_tmp, Ym_tmp, Yr_tmp, test_size=0.5, random_state=SEED
)

ym_scaler = StandardScaler()
Ym_tr_sc  = ym_scaler.fit_transform(Ym_tr)
Ym_val_sc = ym_scaler.transform(Ym_val)
Ym_test_sc= ym_scaler.transform(Ym_test)

yr_scaler = StandardScaler()
Yr_tr_sc  = yr_scaler.fit_transform(Yr_tr)
Yr_val_sc = yr_scaler.transform(Yr_val)
Yr_test_sc= yr_scaler.transform(Yr_test)

Y_tr_sc   = np.concatenate([Ym_tr_sc,  Yr_tr_sc],  axis=1).astype(np.float32)
Y_val_sc  = np.concatenate([Ym_val_sc, Yr_val_sc],  axis=1).astype(np.float32)
Y_test_sc = np.concatenate([Ym_test_sc,Yr_test_sc], axis=1).astype(np.float32)

_PPF_MAX_MEAN = float(ym_scaler.mean_[IDX_PPF_MAX])
_PPF_MAX_STD  = float(ym_scaler.scale_[IDX_PPF_MAX])

print(f"  Split: {len(X_tr)} train / {len(X_val)} val / {len(X_test)} test")
print(f"  PPF_max range: {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  PPF_max mean : {ppf_global_max.mean():.3f}")
print(f"  Scaler: ppf_max mean={_PPF_MAX_MEAN:.3f} std={_PPF_MAX_STD:.3f}\n")


# =============================================================================
# SECTION 3 — ARCHITECTURE
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """Residual conv block: Conv→BN→GELU→Conv→BN→Add(shortcut)→GELU→Dropout."""
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


def build_model(
    filters:       list,
    use_attn:      bool,
    embed_dim:     int   = EMBED_DIM,
    dense_units:   int   = DENSE_UNITS,
    ppf_head_units:int   = PPF_HEAD_UNITS,
    dropout:       float = DROPOUT_TRUNK,
    conv_dropout:  float = CONV_DROP,
):
    """
    Multi-head CNN surrogate. Always multi-head output (PPF + Cycle + Rho).

    Parameters
    ----------
    filters    : conv filter sizes per block, e.g. [32,64] or [32,64,128]
    use_attn   : whether to add a 1×1 spatial attention map after the conv stack

    Architecture summary
    ────────────────────
    Input(6×6 int) → Embedding(10,16)
    → n×ConvResBlock(filters[i])      # configurable
    → [optional spatial attention]     # configurable
    → GlobalAvgPool
    → Dense(dense_units, gelu) → Dropout(dropout)
    → Dense(dense_units//2, gelu)
    → PPF head:   Dense(ppf_head_units) → Dense(33)   # ppf_max + boc + 31 steps
    → Cycle head: Dense(32) → Dense(1)
    → Rho head:   Dense(32) → Dense(1)
    → Concatenate → (B, 35)
    """
    num_blocks = len(filters)
    inp = keras.Input(shape=(GRID_ROWS, GRID_COLS), dtype='int32', name='loading_grid')
    x   = layers.Embedding(N_TYPES + 1, embed_dim, name='assembly_embedding')(inp)

    for i, f in enumerate(filters):
        x = ConvResBlock(f, dropout=conv_dropout, name=f'conv_block_{i+1}')(x)

    if use_attn:
        # Spatial attention: a learned importance map over the 6×6 grid.
        # Each cell gets a weight in [0,1]; attended features = x * weight.
        # This teaches the model that cells near the flux peak matter more.
        attn = layers.Conv2D(1, 1, padding='same', activation='sigmoid',
                             name='spatial_attention')(x)
        x    = layers.Multiply(name='attended_features')([x, attn])

    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    # Shared trunk — features learned jointly before heads split
    shared = layers.Dense(dense_units, activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu', name='shared_dense2')(shared)

    # ── PPF head: 33 outputs ──────────────────────────────────────────────────
    h_ppf   = layers.Dense(ppf_head_units, activation='gelu', name='ppf_dense')(shared)
    h_ppf   = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf = layers.Dense(1 + 1 + N_STEPS, activation='linear', name='ppf_output')(h_ppf)

    # ── Cycle head: 1 output ──────────────────────────────────────────────────
    h_cyc     = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc     = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle = layers.Dense(1, activation='linear', name='cycle_output')(h_cyc)

    # ── Rho head: 1 output ───────────────────────────────────────────────────
    h_rho   = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho   = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho = layers.Dense(1, activation='linear', name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    #arch_tag = f"[{'→'.join(str(f) for f in filters)}]{'_attn' if use_attn else ''}"
    arch_tag = "_".join(str(f) for f in filters)
    if use_attn:
        arch_tag += "_attn"
    return keras.Model(inputs=inp, outputs=out, name=f'BEAVRS_CNN_{arch_tag}')


# =============================================================================
# SECTION 4 — LOSS FUNCTIONS
# =============================================================================
#
# MSE loss  (v4 baseline):
#   L = W_PPF_MAX × MSE(ppf_max) + W_PPF_BOC × MSE(ppf_boc)
#     + W_PPF_STEPS × MSE(steps) + W_CYCLE × MSE(cycle)
#     + W_RHO × MSE(rho) + W_MONO × monotonicity_penalty
#   No relative-error term. Gradient is uniform across PPF values.
#
# Log-dominant loss (new):
#   Same MSE terms BUT W_PPF_MAX reduced to 1.5 AND W_LOG=4.0 added.
#   Effect: log-space MSE term contributes MORE gradient than the
#   absolute-error MSE term (4.0 > 1.5). The model is now primarily
#   optimised for relative error on ppf_max.
#
#   MSE(log ppf_pred, log ppf_true) ≈ mean((rel_err)²)  for small errors.
#   Gradient ∝ 1/pred_real → amplified at low PPF exactly where we need it.
#   No eps, no dilution, symmetric over/under-prediction.

def make_loss_fn(cfg: dict):
    """
    Factory: returns a loss function closed over the weight config dict.
    All weights except W_LOG come from cfg. If W_LOG=0 the log term is skipped.
    """
    W_PPF_MAX   = cfg["W_PPF_MAX"]
    W_PPF_BOC   = cfg["W_PPF_BOC"]
    W_PPF_STEPS = cfg["W_PPF_STEPS"]
    W_CYCLE     = cfg["W_CYCLE"]
    W_RHO       = cfg["W_RHO"]
    W_MONO      = cfg["W_MONO"]
    W_LOG       = cfg["W_LOG"]

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # ── MSE terms ─────────────────────────────────────────────────────────
        ppf_max_loss = W_PPF_MAX * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_MAX] - y_pred[:, IDX_PPF_MAX]))
        ppf_boc_loss = W_PPF_BOC * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_BOC] - y_pred[:, IDX_PPF_BOC]))
        ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
            - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
        cycle_loss = W_CYCLE * tf.reduce_mean(tf.square(
            y_true[:, IDX_CYCLE] - y_pred[:, IDX_CYCLE]))
        rho_loss   = W_RHO   * tf.reduce_mean(tf.square(
            y_true[:, IDX_RHO]   - y_pred[:, IDX_RHO]))

        # ── Monotonicity penalty ───────────────────────────────────────────────
        late       = y_pred[:, IDX_PPF_STEPS_START + 3:IDX_PPF_STEPS_END]
        violations = tf.maximum(0.0, late[:, 1:] - late[:, :-1])
        mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))

        total = ppf_max_loss + ppf_boc_loss + ppf_steps_loss + cycle_loss + rho_loss + mono_loss

        # ── Log-space term (only if W_LOG > 0) ────────────────────────────────
        if W_LOG > 0:
            ppf_true_real = y_true[:, IDX_PPF_MAX] * _PPF_MAX_STD + _PPF_MAX_MEAN
            ppf_pred_real = y_pred[:, IDX_PPF_MAX] * _PPF_MAX_STD + _PPF_MAX_MEAN
            log_true = tf.math.log(tf.maximum(ppf_true_real, LOG_CLAMP))
            log_pred = tf.math.log(tf.maximum(ppf_pred_real, LOG_CLAMP))
            total += W_LOG * tf.reduce_mean(tf.square(log_pred - log_true))

        return total

    loss_fn.__name__ = f"loss_{cfg['desc'][:20]}"
    return loss_fn


# =============================================================================
# SECTION 5 — INVERSE TRANSFORM HELPER
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    Y_main_real = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_real  = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main_real, Y_rho_real], axis=1)


# =============================================================================
# SECTION 6 — EVALUATE HELPER
# =============================================================================

def evaluate_model(model) -> dict:
    """Run predictions on test set and return all metrics as a dict arch_tag"""
    Y_pred_sc   = model.predict(X_test, verbose=0)
    Y_pred_real = inverse_transform(Y_pred_sc)
    Y_true_real = inverse_transform(Y_test_sc)

    ppf_pred = Y_pred_real[:, IDX_PPF_MAX]
    ppf_true = Y_true_real[:, IDX_PPF_MAX]
    cyc_pred = Y_pred_real[:, IDX_CYCLE]
    cyc_true = Y_true_real[:, IDX_CYCLE]
    rho_pred = Y_pred_real[:, IDX_RHO]
    rho_true = Y_true_real[:, IDX_RHO]

    keff_pred = 1.0 / (1.0 - rho_pred / 1e5)
    keff_true = 1.0 / (1.0 - rho_true / 1e5)

    low_mask  = ppf_true < 2.5
    mid_mask  = (ppf_true >= 2.5) & (ppf_true < 4.0)
    high_mask = ppf_true >= 4.0

    def zone_rel(mask):
        if not mask.any(): return float('nan')
        return (np.abs(ppf_pred[mask] - ppf_true[mask])
                / (ppf_true[mask] + 1e-6)).mean() * 100

    return {
        "ppf_mae"    : float(np.abs(ppf_pred - ppf_true).mean()),
        "ppf_rel_err": float((np.abs(ppf_pred - ppf_true)
                              / (ppf_true + 1e-6)).mean() * 100),
        "ppf_r2"     : float(r2_score(ppf_true, ppf_pred)),
        "cycle_mae"  : float(np.abs(cyc_pred - cyc_true).mean()),
        "cycle_r2"   : float(r2_score(cyc_true, cyc_pred)),
        "keff_r2"    : float(r2_score(keff_true, keff_pred)),
        "rel_low"    : float(zone_rel(low_mask)),
        "rel_mid"    : float(zone_rel(mid_mask)),
        "rel_high"   : float(zone_rel(high_mask)),
    }


# =============================================================================
# SECTION 7 — TRAINING LOOP OVER ALL EXPERIMENTS
# =============================================================================

print("=" * 65)
print(f"RUNNING {len(EXPERIMENTS)} EXPERIMENTS")
print("=" * 65)
print(f"  DROPOUT_TRUNK : {DROPOUT_TRUNK}  (v4 level — keeps val/train low)")
print(f"  Epochs        : {EPOCHS}  (early stop patience=30)")
print(f"  Baselines     : v4 rel_err=3.10%  val/train=0.89")
print("=" * 65 + "\n")

all_results = []
t_total_start = time.time()

for exp_idx, exp in enumerate(EXPERIMENTS):
    exp_name   = exp["name"]
    filters    = exp["filters"]
    use_attn   = exp["use_attn"]
    loss_mode  = exp["loss_mode"]
    loss_cfg   = LOSS_CONFIGS[loss_mode]

    print(f"\n{'─'*65}")
    print(f"[{exp_idx+1}/{len(EXPERIMENTS)}]  {exp_name}")
    print(f"  Arch   : filters={filters}, attn={use_attn}")
    print(f"  Loss   : {loss_cfg['desc']}")
    print(f"{'─'*65}")

    # Build model
    model = build_model(
        filters       = filters,
        use_attn      = use_attn,
        dropout       = DROPOUT_TRUNK,
        conv_dropout  = CONV_DROP,
    )
    n_params = model.count_params()
    print(f"  Params : {n_params:,}")

    # Compile with the appropriate loss
    loss_fn = make_loss_fn(loss_cfg)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss=loss_fn,
        metrics=['mae']
    )

    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15,
            min_lr=1e-5, verbose=0
        ),
    ]

    t_exp = time.time()
    history = model.fit(
        X_tr, Y_tr_sc,
        validation_data=(X_val, Y_val_sc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )
    t_exp = time.time() - t_exp

    best_epoch    = int(np.argmin(history.history['val_loss'])) + 1
    val_loss_best = history.history['val_loss'][best_epoch - 1]
    tr_loss_best  = history.history['loss'][best_epoch - 1]
    vt_ratio      = val_loss_best / (tr_loss_best + 1e-9)

    print(f"  Best epoch : {best_epoch}  val/train = {vt_ratio:.2f}  "
          f"({'✓ good' if vt_ratio < 1.2 else '⚠ overfit' if vt_ratio > 1.3 else '~ ok'})")

    # Evaluate
    metrics = evaluate_model(model)
    print(f"  PPF rel_err: {metrics['ppf_rel_err']:.2f}%  "
          f"MAE={metrics['ppf_mae']:.4f}  R²={metrics['ppf_r2']:.4f}")
    print(f"  Zone err:  <2.5={metrics['rel_low']:.2f}%  "
          f"2.5-4={metrics['rel_mid']:.2f}%  >4={metrics['rel_high']:.2f}%")
    print(f"  Cycle MAE: {metrics['cycle_mae']:.2f}d  keff R²={metrics['keff_r2']:.4f}")
    print(f"  Time: {t_exp:.0f}s")

    # Save model
    model_path = f'cnn_compare_{exp_name}.keras'
    model.save(model_path)
    print(f"  Saved: {model_path}")

    # Record result
    row = {
        "name"         : exp_name,
        "label"        : exp["label"],
        "filters"      : str(filters),
        "attn"         : use_attn,
        "loss_mode"    : loss_mode,
        "n_params"     : n_params,
        "best_epoch"   : best_epoch,
        "val_train_ratio": round(vt_ratio, 3),
        **{k: round(v, 4) for k, v in metrics.items()},
        "train_time_s" : round(t_exp, 1),
    }
    all_results.append(row)

t_total = time.time() - t_total_start
print(f"\n{'='*65}")
print(f"ALL EXPERIMENTS COMPLETE  ({t_total/60:.1f} min total)")
print(f"{'='*65}")


# =============================================================================
# SECTION 8 — RESULTS TABLE
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n[SAVED]  {RESULTS_CSV}")

print(f"\n{'='*95}")
print(f"COMPARISON TABLE")
print(f"{'='*95}")
hdr = (f"{'Experiment':<26} {'loss':>6} {'rel_err':>8} {'rel<2.5':>8} "
       f"{'ppf_r2':>7} {'cycle':>7} {'keff_r2':>7} {'v/t':>5} {'ep':>4}")
print(hdr)
print("─" * 95)
print(f"  v4 BASELINE (reference)   MSE    3.10%    ~3.5%   0.9841   1.28d   0.9912   0.89  100")
print("─" * 95)
for row in all_results:
    flag = ""
    if row["ppf_rel_err"] < 3.10:  flag = " ← NEW BEST"
    elif row["ppf_rel_err"] < 3.22: flag = " ← beats v5"
    print(
        f"  {row['name']:<24} {row['loss_mode']:>6}  "
        f"{row['ppf_rel_err']:>6.2f}%  "
        f"{row['rel_low']:>6.2f}%  "
        f"{row['ppf_r2']:>7.4f}  "
        f"{row['cycle_mae']:>5.2f}d  "
        f"{row['keff_r2']:>7.4f}  "
        f"{row['val_train_ratio']:>5.2f}  "
        f"{row['best_epoch']:>4}"
        + flag
    )
print("─" * 95)

best_idx = min(range(len(all_results)),
               key=lambda i: all_results[i]["ppf_rel_err"])
best = all_results[best_idx]
print(f"\n  WINNER: {best['name']}")
print(f"    rel_err = {best['ppf_rel_err']:.2f}%   "
      f"R² = {best['ppf_r2']:.4f}   val/train = {best['val_train_ratio']:.2f}")
if best["ppf_rel_err"] < 3.10:
    print(f"    ✓ Beats v4 baseline (3.10%)!")
elif best["ppf_rel_err"] < 3.20:
    print(f"    ~ Close to v4. Try training winner at 400 epochs.")
else:
    print(f"    ✗ All experiments above v4. Investigate val/train ratios.")

print(f"\n  NEXT STEP: Retrain '{best['name']}' config at EPOCHS=400")
print(f"  and add USE_SAMPLE_WEIGHTING=True if loss_mode='log'.")


# =============================================================================
# SECTION 9 — VISUALISATIONS
# =============================================================================

fig = plt.figure(figsize=(20, 14))
fig.suptitle("BEAVRS CNN Architecture × Loss Comparison\n"
             "(v4 baseline: 3.10% rel_err, R²=0.9841, val/train=0.89)",
             fontsize=12, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

names   = [r["name"] for r in all_results]
labels  = [r["label"].replace(" ← v4 architecture", "\n← v4 arch") for r in all_results]
rel_err = [r["ppf_rel_err"]  for r in all_results]
rel_low = [r["rel_low"]      for r in all_results]
ppf_r2  = [r["ppf_r2"]       for r in all_results]
cycle   = [r["cycle_mae"]    for r in all_results]
keff_r2 = [r["keff_r2"]      for r in all_results]
vt      = [r["val_train_ratio"] for r in all_results]

# Color: MSE=blue, LOG=teal
colors = ['#185FA5' if r["loss_mode"] == "mse" else '#0F6E56' for r in all_results]

x = np.arange(len(all_results))
bar_w = 0.65

# 1. Overall rel_err (primary metric)
ax = fig.add_subplot(gs[0, :2])
bars = ax.bar(x, rel_err, color=colors, alpha=0.8, width=bar_w)
ax.axhline(3.10, color='red', lw=1.5, ls='--', label='v4 baseline (3.10%)')
ax.axhline(3.00, color='green', lw=1.0, ls=':', alpha=0.7, label='Target (<3.0%)')
for bar, val in zip(bars, rel_err):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='500')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
ax.set_ylabel('Relative error (%)'); ax.set_title('PPF Relative Error (lower = better)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(2.5, max(rel_err) + 0.5)

# 2. Legend for colors
from matplotlib.patches import Patch
legend_els = [Patch(fc='#185FA5', label='MSE loss (v4-style)'),
              Patch(fc='#0F6E56', label='Log-dominant loss')]
ax.legend(handles=legend_els + [
    plt.Line2D([0],[0], c='red', ls='--', label='v4 baseline'),
    plt.Line2D([0],[0], c='green', ls=':', label='Target')
], fontsize=8)

# 3. Rel error PPF < 2.5 zone
ax = fig.add_subplot(gs[0, 2])
bars_z = ax.bar(x, rel_low, color=colors, alpha=0.8, width=bar_w)
ax.axhline(3.55, color='red', lw=1.5, ls='--', alpha=0.7, label='v5 ref (3.55%)')
for bar, val in zip(bars_z, rel_low):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r["loss_mode"] + "\n" + str(r["filters"])
                                       for r in all_results], fontsize=7)
ax.set_ylabel('Rel error (%)'); ax.set_title('PPF < 2.5 Zone\n(QICA target region)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 4. PPF R²
ax = fig.add_subplot(gs[1, 0])
ax.bar(x, ppf_r2, color=colors, alpha=0.8, width=bar_w)
ax.axhline(0.9841, color='red', lw=1.5, ls='--', label='v4=0.9841')
ax.set_xticks(x); ax.set_xticklabels([r["loss_mode"] for r in all_results], fontsize=8)
ax.set_ylabel('R²'); ax.set_title('PPF R² (higher = better)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.970, 0.995)

# 5. Cycle MAE
ax = fig.add_subplot(gs[1, 1])
ax.bar(x, cycle, color=colors, alpha=0.8, width=bar_w)
ax.axhline(1.28, color='red', lw=1.5, ls='--', label='v4=1.28d')
ax.set_xticks(x); ax.set_xticklabels([r["loss_mode"] for r in all_results], fontsize=8)
ax.set_ylabel('MAE (days)'); ax.set_title('Cycle Length MAE (lower = better)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 6. keff R²
ax = fig.add_subplot(gs[1, 2])
ax.bar(x, keff_r2, color=colors, alpha=0.8, width=bar_w)
ax.axhline(0.9912, color='red', lw=1.5, ls='--', label='v4=0.9912')
ax.set_xticks(x); ax.set_xticklabels([r["loss_mode"] for r in all_results], fontsize=8)
ax.set_ylabel('R²'); ax.set_title('keff R² (higher = better)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.980, 0.998)

# 7. Val/train ratio (overfitting diagnostic)
ax = fig.add_subplot(gs[2, 0])
bars_vt = ax.bar(x, vt, color=colors, alpha=0.8, width=bar_w)
ax.axhline(1.0,  color='green', lw=1.0, ls='--', alpha=0.7, label='Ideal = 1.0')
ax.axhline(1.2,  color='orange', lw=1.0, ls=':', alpha=0.7, label='Warn > 1.2')
ax.axhline(0.89, color='red', lw=1.5, ls='--', label='v4=0.89')
for bar, val in zip(bars_vt, vt):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([r["loss_mode"] for r in all_results], fontsize=8)
ax.set_ylabel('val_loss / train_loss'); ax.set_title('Val/Train Ratio\n(1.0=ideal, ↑=overfit)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 8. Params vs rel_err scatter
ax = fig.add_subplot(gs[2, 1])
params = [r["n_params"] for r in all_results]
for i, r in enumerate(all_results):
    ax.scatter(r["n_params"], r["ppf_rel_err"], s=80,
               color=colors[i], alpha=0.85, zorder=3)
    ax.text(r["n_params"], r["ppf_rel_err"] + 0.02,
            r["loss_mode"] + "\n" + str(r["filters"]),
            ha='center', va='bottom', fontsize=6)
ax.axhline(3.10, color='red', lw=1, ls='--', label='v4 baseline')
ax.scatter([339108], [3.10], s=100, c='red', marker='*', zorder=5, label='v4')
ax.set_xlabel('# Parameters'); ax.set_ylabel('Rel error (%)')
ax.set_title('Params vs Relative Error')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 9. Ranked comparison summary
ax = fig.add_subplot(gs[2, 2])
ranked = sorted(enumerate(all_results), key=lambda x: x[1]["ppf_rel_err"])
y_pos = np.arange(len(ranked))
ax.barh(y_pos, [r["ppf_rel_err"] for _, r in ranked],
        color=[colors[i] for i, _ in ranked], alpha=0.8)
ax.axvline(3.10, color='red', lw=1.5, ls='--', label='v4 (3.10%)')
ax.set_yticks(y_pos)
ax.set_yticklabels([r["name"].replace("_MSE","").replace("_LOG","")
                    + f'\n({r["loss_mode"].upper()})' for _, r in ranked], fontsize=7)
ax.set_xlabel('Relative error (%)'); ax.set_title('Ranking (best at top)')
ax.legend(fontsize=7); ax.grid(axis='x', alpha=0.3)
for i, (_, r) in enumerate(ranked):
    ax.text(r["ppf_rel_err"] + 0.01, i, f'{r["ppf_rel_err"]:.2f}%', va='center', fontsize=7)

plt.savefig(RESULTS_PNG, dpi=150, bbox_inches='tight')
print(f"\n[SAVED]  {RESULTS_PNG}")

print(f"\n{'='*65}")
print(f"INTERPRETATION GUIDE")
print(f"{'='*65}")
print(f"  The table and plots answer:")
print(f"  1. Does [32,64,128]+attn (v4 arch) beat [64,128] when using the")
print(f"     SAME loss? If yes, architecture was the edge, not the loss.")
print(f"  2. Does log-dominant loss beat MSE within the SAME architecture?")
print(f"     If yes, the relative-error loss design is the key lever.")
print(f"  3. Is the winning combo below v4's 3.10% rel_err?")
print(f"     If not, try stratified split + 400-epoch retrain + sample_wt.")
print(f"\n  Retrain the winner with EPOCHS=400 using cnn_v7.py as template,")
print(f"  substituting the best architecture and loss config found here.")
print(f"{'='*65}")