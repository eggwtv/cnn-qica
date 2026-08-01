"""
=============================================================================
cnn_v8.py  —  BEAVRS CNN v8  |  Architecture & Loss Comparison
=============================================================================
Runs 6 experiments to definitively establish the best configuration for the
BEAVRS PPF surrogate, building directly on comparison experiment findings.

COMPARISON RESULTS (prior run, 300 epochs):
  A_32_64_MSE         : 2.83%  R²=0.9873  cycle=1.01d  keff=0.9948  ← BEST
  C_32_64_128+attn    : 2.88%  R²=0.9861  cycle=1.08d  keff=0.9929
  A_32_64_LOG         : 2.89%  R²=0.9871  cycle=0.98d  keff=0.9944

v4 BASELINE (400 epochs, old arch [32,64,128]+attn+v4-loss):
  3.10%  R²=0.9841  cycle=1.28d  keff=0.9912

v8 EXPERIMENTS (400 epochs, same SEED/split):
  1. A_MSE       [32,64], no-attn, MSE  (W_PPF=3.0)
  2. A_LOG       [32,64], no-attn, LOG  (W_PPF=1.5, W_LOG=4.0)
  3. A_LOG_SWT   [32,64], no-attn, LOG  + inverse-PPF sample weights
  4. C_MSE       [32,64,128]+attn, MSE
  5. C_LOG       [32,64,128]+attn, LOG
  6. C_LOG_SWT   [32,64,128]+attn, LOG  + sample weights

To run a subset: set SELECTED_EXPERIMENTS = ['A_MSE', 'C_MSE'] etc.

OUTPUTS:
  cnn_v8_model.keras          — best model across all experiments
  cnn_v8_config.json          — scalers + geometry (for QICA v5)
  train_type_freq_v8.npy      — trust-region frequencies (for QICA v5)
  cnn_v8_results.csv          — per-experiment metrics
  cnn_v8_comparison.png       — comparison plots
  cnn_v8_<name>.keras         — each experiment's model
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print(f"Running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print("cnn_v8.py  —  Architecture & Loss Comparison\n")

t_global = time.time()


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV = 'ml_dataset_constrained.csv'
XL_FILE    = 'cycle_length_summary.xlsx'

# Set to None to run all, or e.g. ['A_MSE', 'A_LOG'] to run a subset
SELECTED_EXPERIMENTS = None

# 6 experiments: (name, filters, use_attention, loss_mode, use_sample_weight)
ALL_EXPERIMENTS = [
    ('A_MSE',     (32, 64),       False, 'mse', False),
    ('A_LOG',     (32, 64),       False, 'log', False),
    ('A_LOG_SWT', (32, 64),       False, 'log', True),
    ('C_MSE',     (32, 64, 128),  True,  'mse', False),
    ('C_LOG',     (32, 64, 128),  True,  'log', False),
    ('C_LOG_SWT', (32, 64, 128),  True,  'log', True),
]

# Loss weight sets
LOSS_WEIGHTS = {
    'mse': dict(W_PPF_MAX=3.0, W_PPF_BOC=2.0, W_PPF_STEPS=0.5,
                W_CYCLE=1.0,   W_RHO=5.0,     W_MONO=0.01, W_LOG=0.0),
    'log': dict(W_PPF_MAX=1.5, W_PPF_BOC=2.0, W_PPF_STEPS=0.5,
                W_CYCLE=1.0,   W_RHO=5.0,     W_MONO=0.01, W_LOG=4.0),
}

# Architecture constants
N_POS, N_TYPES, N_STEPS = 31, 9, 31
GRID_ROWS, GRID_COLS    = 6, 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)

N_OUTPUTS           = 1 + 1 + N_STEPS + 1 + 1  # 35
IDX_PPF_MAX         = 0
IDX_PPF_BOC         = 1
IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS   # 33
IDX_CYCLE           = 2 + N_STEPS   # 33
IDX_RHO             = 3 + N_STEPS   # 34

# Training constants
EMBED_DIM      = 16
DENSE_UNITS    = 128
PPF_HEAD_UNITS = 64
DROPOUT        = 0.15
CONV_DROP      = 0.10
WEIGHT_DECAY   = 1e-4
BATCH_SIZE     = 128
EPOCHS         = 400
LR             = 1e-3
TEST_FRAC      = 0.15
VAL_FRAC       = 0.15
SEED           = 42
MC_SAMPLES     = 30
LOG_CLAMP      = 0.5

PPF_REPORT_LOW  = 2.0
PPF_REPORT_HIGH = 4.5


# =============================================================================
# SECTION 2 — MONOCORE CYCLE LENGTHS
# =============================================================================

print("[XLSX] Loading monocore cycle lengths ...")
if os.path.exists(XL_FILE):
    xl_df = pd.read_excel(XL_FILE, sheet_name='Cycle_Lengths')
    monocore_map = dict(zip(xl_df['fa_id'].astype(int),
                            xl_df['monocore_cycle_length'].astype(float)))
else:
    print("  [WARN] xlsx not found — using hardcoded fallback.")
    monocore_map = {1:172.9, 2:366.9, 3:323.2, 4:299.8, 5:519.9,
                   6:504.9, 7:475.3, 8:471.6, 9:454.7}
print()


# =============================================================================
# SECTION 3 — DATA LOADING
# =============================================================================

print("[DATA] Loading BEAVRS dataset ...")
if not os.path.exists(BEAVRS_CSV):
    print(f"[ERROR] {BEAVRS_CSV} not found."); sys.exit(1)

df = pd.read_csv(BEAVRS_CSV, skiprows=1, engine='python', on_bad_lines='skip')
print(f"  Loaded {len(df)} patterns × {df.shape[1]} columns")

load_cols  = [f'loading_{i}' for i in range(N_POS)]
react_cols = sorted([c for c in df.columns if c.startswith('react_')],
                    key=lambda c: int(c.split('_')[1]))
ppf_steps_idx   = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_assembs_idx = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))

step_max_ppf = np.stack([
    df[[f'ppf_s{s}_a{i}' for i in ppf_assembs_idx
        if f'ppf_s{s}_a{i}' in df.columns]].values.astype(np.float32).max(axis=1)
    for s in ppf_steps_idx
], axis=1)

ppf_global_max = step_max_ppf.max(axis=1)
ppf_boc        = step_max_ppf[:, 0]
keff_raw       = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
rho_pcm        = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

print(f"  PPF_max range : {ppf_global_max.min():.3f} – {ppf_global_max.max():.3f}")
print(f"  PPF 10th pct  : {np.percentile(ppf_global_max, 10):.3f}  (QICA target)")
print(f"  Cycle range   : {df.cycle_length.min():.1f} – {df.cycle_length.max():.1f} days\n")


# =============================================================================
# SECTION 4 — FEATURE / TARGET ARRAYS
# =============================================================================

Y_main  = np.concatenate([
    ppf_global_max.reshape(-1,1),
    ppf_boc.reshape(-1,1),
    step_max_ppf,
    df['cycle_length'].values.reshape(-1,1)
], axis=1).astype(np.float32)   # (N, 34)
Y_rho   = rho_pcm.reshape(-1, 1).astype(np.float32)

X_raw   = df[load_cols].values.astype(np.int32)
X_grid  = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
pos_idx = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            X_grid[:, r, c] = X_raw[:, pos_idx]
            pos_idx += 1

print(f"[INPUT]   Grid : {X_grid.shape}  ({GRID_MASK.sum()} active cells)")
print(f"[TARGETS] Main : {Y_main.shape},  Rho : {Y_rho.shape}\n")


# =============================================================================
# SECTION 5 — TRAIN / VAL / TEST SPLIT  (fixed seed — same as v4)
# =============================================================================

(X_tr, X_tmp,
 Ym_tr, Ym_tmp,
 Yr_tr, Yr_tmp) = train_test_split(X_grid, Y_main, Y_rho,
                                    test_size=TEST_FRAC + VAL_FRAC,
                                    random_state=SEED)
(X_val, X_test,
 Ym_val, Ym_test,
 Yr_val, Yr_test) = train_test_split(X_tmp, Ym_tmp, Yr_tmp,
                                      test_size=0.5, random_state=SEED)

print(f"[SPLIT]  {len(X_tr)} train / {len(X_val)} val / {len(X_test)} test")

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

_PPF_MAX_MEAN  = float(ym_scaler.mean_[IDX_PPF_MAX])
_PPF_MAX_STD   = float(ym_scaler.scale_[IDX_PPF_MAX])

print(f"  ppf_max : mean={_PPF_MAX_MEAN:.3f}  std={_PPF_MAX_STD:.3f}")
print(f"  cycle   : mean={ym_scaler.mean_[IDX_CYCLE]:.1f}  std={ym_scaler.scale_[IDX_CYCLE]:.2f}")
print(f"  rho_pcm : mean={yr_scaler.mean_[0]:.0f}  std={yr_scaler.scale_[0]:.0f}\n")


# =============================================================================
# SECTION 6 — CONVRESBLOCK (serializable — identical to v4/v7)
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """
    Residual conv block: Conv→BN→GELU→Conv→BN→Add(skip)→GELU→Dropout
    Registered for Keras serialization so saved models load correctly in QICA.
    """
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        self._filters      = filters
        self._dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
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


# =============================================================================
# SECTION 7 — FLEXIBLE ARCHITECTURE BUILDER
# =============================================================================

def build_cnn_v8(filters, use_attention, name='BEAVRS_CNN_v8',
                 embed_dim=EMBED_DIM, dense_units=DENSE_UNITS,
                 ppf_head_units=PPF_HEAD_UNITS,
                 dropout=DROPOUT, conv_dropout=CONV_DROP):
    """
    Flexible multi-head CNN.

    filters:       tuple of filter counts, one per ConvResBlock
                   (32, 64)       → 2 blocks, 100K params
                   (32, 64, 128)  → 3 blocks, 339K params
    use_attention: if True, adds a 1×1 spatial attention gate after the blocks
    """
    inp = keras.Input(shape=(GRID_ROWS, GRID_COLS), dtype='int32', name='loading_grid')
    x   = layers.Embedding(N_TYPES + 1, embed_dim, name='assembly_embedding')(inp)

    for i, f in enumerate(filters):
        x = ConvResBlock(f, dropout=conv_dropout, name=f'conv_block_{i+1}')(x)

    if use_attention:
        attn = layers.Conv2D(1, 1, padding='same', activation='sigmoid',
                              name='spatial_attention')(x)
        x    = layers.Multiply(name='attended_features')([x, attn])

    x      = layers.GlobalAveragePooling2D(name='global_pool')(x)
    shared = layers.Dense(dense_units,     activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout,                          name='shared_dropout')(shared)
    shared = layers.Dense(dense_units//2,  activation='gelu', name='shared_dense2')(shared)

    # PPF head: 33 outputs (ppf_max + ppf_boc + 31 step values)
    h_ppf   = layers.Dense(ppf_head_units, activation='gelu', name='ppf_dense')(shared)
    h_ppf   = layers.Dropout(dropout*0.5,                     name='ppf_dropout')(h_ppf)
    out_ppf = layers.Dense(1+1+N_STEPS,    activation='linear',name='ppf_output')(h_ppf)

    # Cycle head
    h_cyc     = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc     = layers.Dropout(dropout*0.3,          name='cycle_dropout')(h_cyc)
    out_cycle = layers.Dense(1,  activation='linear',name='cycle_output')(h_cyc)

    # Rho head
    h_rho   = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho   = layers.Dropout(dropout*0.3,          name='rho_dropout')(h_rho)
    out_rho = layers.Dense(1,  activation='linear',name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    return keras.Model(inputs=inp, outputs=out, name=name)


# =============================================================================
# SECTION 8 — LOSS FUNCTION FACTORY
# =============================================================================

def make_loss_fn(weights: dict, ppf_mean: float, ppf_std: float):
    """
    Returns a Keras-compatible loss function with the given weight settings.

    MSE mode (W_LOG=0.0):
      L = W_PPF_MAX×MSE(ppf_max) + W_PPF_BOC×MSE(ppf_boc)
        + W_PPF_STEPS×MSE(step_ppfs) + W_CYCLE×MSE(cycle) + W_RHO×MSE(rho)
        + W_MONO×monotonicity_penalty

    LOG mode (W_LOG>0):
      Adds W_LOG × MSE(log(pred_ppf_real), log(true_ppf_real))
      This directly minimises relative error for ppf_max:
        log(pred) - log(true) ≈ (pred-true)/true  for small errors
    """
    W_PPF_MAX   = weights['W_PPF_MAX']
    W_PPF_BOC   = weights['W_PPF_BOC']
    W_PPF_STEPS = weights['W_PPF_STEPS']
    W_CYCLE     = weights['W_CYCLE']
    W_RHO       = weights['W_RHO']
    W_MONO      = weights['W_MONO']
    W_LOG       = weights['W_LOG']

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        ppf_max_l = W_PPF_MAX   * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_MAX] - y_pred[:, IDX_PPF_MAX]))
        ppf_boc_l = W_PPF_BOC   * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_BOC] - y_pred[:, IDX_PPF_BOC]))
        ppf_stp_l = W_PPF_STEPS * tf.reduce_mean(tf.square(
            y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
            - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
        cycle_l   = W_CYCLE     * tf.reduce_mean(tf.square(
            y_true[:, IDX_CYCLE] - y_pred[:, IDX_CYCLE]))
        rho_l     = W_RHO       * tf.reduce_mean(tf.square(
            y_true[:, IDX_RHO] - y_pred[:, IDX_RHO]))

        # Late-cycle PPF monotonicity penalty
        late       = y_pred[:, IDX_PPF_STEPS_START+3:IDX_PPF_STEPS_END]
        mono_l     = W_MONO * tf.reduce_mean(tf.square(
            tf.maximum(0.0, late[:, 1:] - late[:, :-1])))

        total = ppf_max_l + ppf_boc_l + ppf_stp_l + cycle_l + rho_l + mono_l

        if W_LOG > 0.0:
            # Recover real-space PPF values from scaled predictions
            ppf_true_real = y_true[:, IDX_PPF_MAX] * ppf_std + ppf_mean
            ppf_pred_real = y_pred[:, IDX_PPF_MAX] * ppf_std + ppf_mean
            log_l = W_LOG * tf.reduce_mean(tf.square(
                tf.math.log(tf.maximum(ppf_pred_real, LOG_CLAMP))
                - tf.math.log(tf.maximum(ppf_true_real, LOG_CLAMP))
            ))
            total += log_l

        return total

    loss_fn.__name__ = f"v8_loss_{'log' if W_LOG > 0 else 'mse'}"
    return loss_fn


# =============================================================================
# SECTION 9 — EVALUATION HELPER
# =============================================================================

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    Y_main = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_ = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main, Y_rho_], axis=1)


def evaluate_model(model, X_test_, Y_test_sc_, label=''):
    """Full evaluation returning metrics dict."""
    Y_pred_sc   = model.predict(X_test_, verbose=0)
    Y_pred_real = inverse_transform(Y_pred_sc)
    Y_true_real = inverse_transform(Y_test_sc_)

    ppf_p = Y_pred_real[:, IDX_PPF_MAX]
    ppf_t = Y_true_real[:, IDX_PPF_MAX]
    cyc_p = Y_pred_real[:, IDX_CYCLE]
    cyc_t = Y_true_real[:, IDX_CYCLE]
    rho_p = Y_pred_real[:, IDX_RHO]
    rho_t = Y_true_real[:, IDX_RHO]

    keff_p = 1.0 / (1.0 - rho_p / 1e5)
    keff_t = 1.0 / (1.0 - rho_t / 1e5)

    # PPF zones
    low_m  = ppf_t < 2.5
    mid_m  = (ppf_t >= 2.5) & (ppf_t < 4.0)
    high_m = ppf_t >= 4.0

    def rel_err(p, t, mask=None):
        if mask is not None:
            p, t = p[mask], t[mask]
        return (np.abs(p - t) / (t + 1e-6)).mean() * 100

    return {
        'ppf_mae'    : float(np.abs(ppf_p - ppf_t).mean()),
        'ppf_rel'    : rel_err(ppf_p, ppf_t),
        'ppf_r2'     : float(r2_score(ppf_t, ppf_p)),
        'rel_low'    : rel_err(ppf_p, ppf_t, low_m),
        'rel_mid'    : rel_err(ppf_p, ppf_t, mid_m),
        'rel_high'   : rel_err(ppf_p, ppf_t, high_m),
        'cycle_mae'  : float(np.abs(cyc_p - cyc_t).mean()),
        'cycle_r2'   : float(r2_score(cyc_t, cyc_p)),
        'keff_r2'    : float(r2_score(keff_t, keff_p)),
        'keff_mae'   : float(np.abs(keff_p - keff_t).mean()),
        'n_low'      : int(low_m.sum()),
        'n_mid'      : int(mid_m.sum()),
        'n_high'     : int(high_m.sum()),
        'ppf_pred'   : ppf_p,
        'ppf_true'   : ppf_t,
        'cyc_pred'   : cyc_p,
        'cyc_true'   : cyc_t,
        'keff_pred'  : keff_p,
        'keff_true'  : keff_t,
    }


# =============================================================================
# SECTION 10 — MAIN EXPERIMENT LOOP
# =============================================================================

EXPERIMENTS = (
    [e for e in ALL_EXPERIMENTS if e[0] in SELECTED_EXPERIMENTS]
    if SELECTED_EXPERIMENTS else ALL_EXPERIMENTS
)

print("=" * 70)
print(f"RUNNING {len(EXPERIMENTS)} EXPERIMENTS  (EPOCHS={EPOCHS}  BATCH={BATCH_SIZE})")
print("=" * 70)
print(f"  v4 BASELINE : 3.10%  R²=0.9841  cycle=1.28d  keff=0.9912")
print(f"  PRIOR BEST  : A_32_64_MSE  2.83%  (300ep)")
print(f"  LOG config  : W_PPF=1.5  W_LOG=4.0")
print()

all_results = {}

for exp_name, filters, use_attn, loss_mode, use_swt in EXPERIMENTS:
    t0 = time.time()
    print(f"{'─'*65}")
    print(f"[{exp_name}]  filters={list(filters)}  attn={use_attn}  "
          f"loss={loss_mode}  sample_wt={use_swt}")
    print(f"{'─'*65}")

    # Build model
    model = build_cnn_v8(filters, use_attn, name=f'CNN_v8_{exp_name}')
    n_params = model.count_params()
    print(f"  Params : {n_params:,}")

    # Sample weights (inverse-PPF, normalised)
    if use_swt:
        ppf_tr_raw   = Ym_tr[:, IDX_PPF_MAX]
        sw           = (ppf_tr_raw.mean() / (ppf_tr_raw + 1e-6)).astype(np.float32)
        sw          /= sw.mean()
        print(f"  Sample weight range: {sw.min():.3f} – {sw.max():.3f}")
    else:
        sw = None

    # Loss function
    loss_fn = make_loss_fn(LOSS_WEIGHTS[loss_mode], _PPF_MAX_MEAN, _PPF_MAX_STD)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
        loss=loss_fn,
        metrics=['mae']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15,
            min_lr=1e-5, verbose=0
        ),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: print(
                f"    Ep {ep+1:4d} | loss: {logs['loss']:.5f}"
                f" | val: {logs['val_loss']:.5f}"
            ) if (ep+1) % 50 == 0 else None
        ),
    ]

    history = model.fit(
        X_tr, Y_tr_sc,
        validation_data=(X_val, Y_val_sc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        sample_weight=sw,
        verbose=0
    )

    t_exp = time.time() - t0
    best_ep = int(np.argmin(history.history['val_loss'])) + 1
    vt_ratio = (history.history['val_loss'][best_ep-1]
                / (history.history['loss'][best_ep-1] + 1e-9))

    metrics = evaluate_model(model, X_test, Y_test_sc, label=exp_name)
    metrics.update({
        'name'      : exp_name,
        'filters'   : str(list(filters)),
        'attn'      : use_attn,
        'loss_mode' : loss_mode,
        'sample_wt' : use_swt,
        'n_params'  : n_params,
        'best_epoch': best_ep,
        'vt_ratio'  : float(vt_ratio),
        'time_s'    : float(t_exp),
        'history'   : history.history,
    })
    all_results[exp_name] = metrics

    # Save this experiment's model
    model.save(f'cnn_v8_{exp_name}.keras')

    print(f"  PPF rel_err : {metrics['ppf_rel']:.2f}%   "
          f"(low={metrics['rel_low']:.2f}%  mid={metrics['rel_mid']:.2f}%)")
    print(f"  PPF MAE     : {metrics['ppf_mae']:.4f}    R²={metrics['ppf_r2']:.4f}")
    print(f"  Cycle MAE   : {metrics['cycle_mae']:.2f}d  keff R²={metrics['keff_r2']:.4f}")
    print(f"  Best epoch  : {best_ep}  val/train={vt_ratio:.2f}  "
          f"time={t_exp:.0f}s ({t_exp/60:.1f}min)")
    print(f"  Saved: cnn_v8_{exp_name}.keras\n")


# =============================================================================
# SECTION 11 — FIND BEST AND SAVE WINNER
# =============================================================================

print("=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
print(f"  {'Experiment':<14} {'loss':<5} {'rel%':>6}  {'<2.5':>6}  "
      f"{'R²':>7}  {'cycle':>6}  {'keff':>7}  {'v/t':>5}  {'ep':>4}")
print(f"  {'─'*14} {'─'*5} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}")
# Print v4 baseline
print(f"  {'v4 BASELINE':<14} {'mse':<5} {'3.10%':>6}  "
      f"{'~3.5%':>6}  {'0.9841':>7}  {'1.28d':>6}  {'0.9912':>7}  {'0.89':>5}  {'100':>4}")
print(f"  {'─'*14} {'─'*5} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*4}")

for name, r in all_results.items():
    marker = ' ← NEW BEST' if r['ppf_rel'] == min(v['ppf_rel'] for v in all_results.values()) else ''
    print(f"  {name:<14} {r['loss_mode']:<5} {r['ppf_rel']:>5.2f}%  "
          f"{r['rel_low']:>5.2f}%  {r['ppf_r2']:>7.4f}  "
          f"{r['cycle_mae']:>5.2f}d  {r['keff_r2']:>7.4f}  "
          f"{r['vt_ratio']:>5.2f}  {r['best_epoch']:>4d}{marker}")
print()

best_name = min(all_results, key=lambda k: all_results[k]['ppf_rel'])
best      = all_results[best_name]
print(f"  WINNER: {best_name}")
print(f"    rel_err = {best['ppf_rel']:.2f}%  MAE={best['ppf_mae']:.4f}  "
      f"R²={best['ppf_r2']:.4f}  val/train={best['vt_ratio']:.2f}\n")

# Save winner model as canonical v8
best_model = keras.models.load_model(
    f'cnn_v8_{best_name}.keras',
    custom_objects={'ConvResBlock': ConvResBlock}
)
best_model.save('cnn_v8_model.keras')
print(f"[SAVED]  cnn_v8_model.keras  ← best model ({best_name})\n")


# =============================================================================
# SECTION 12 — SAVE CONFIG (for QICA v5)
# =============================================================================

config = {
    'version'            : 'v8',
    'best_experiment'    : best_name,
    'N_POS'              : N_POS, 'N_TYPES': N_TYPES, 'N_STEPS': N_STEPS,
    'GRID_ROWS'          : GRID_ROWS, 'GRID_COLS': GRID_COLS,
    'GRID_LAYOUT'        : GRID_LAYOUT.tolist(),
    'GRID_MASK'          : GRID_MASK.tolist(),
    'IDX_PPF_MAX'        : IDX_PPF_MAX,
    'IDX_PPF_BOC'        : IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': IDX_PPF_STEPS_START,
    'IDX_PPF_STEPS_END'  : IDX_PPF_STEPS_END,
    'IDX_CYCLE'          : IDX_CYCLE,
    'IDX_RHO'            : IDX_RHO,
    'N_OUTPUTS'          : N_OUTPUTS,
    'PPF_REPORT_LOW'     : PPF_REPORT_LOW,
    'PPF_REPORT_HIGH'    : PPF_REPORT_HIGH,
    'ym_scaler_mean'     : ym_scaler.mean_.tolist(),
    'ym_scaler_scale'    : ym_scaler.scale_.tolist(),
    'yr_scaler_mean'     : yr_scaler.mean_.tolist(),
    'yr_scaler_scale'    : yr_scaler.scale_.tolist(),
    'mc_samples'         : MC_SAMPLES,
    'al_uncertainty_thr' : 0.07,
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v) for k, v in monocore_map.items()},
    'best_arch'          : {
        'filters'    : list(best['filters'].strip('[]').split(', ')),
        'use_attn'   : best['attn'],
        'loss_mode'  : best['loss_mode'],
        'sample_wt'  : best['sample_wt'],
        'n_params'   : best['n_params'],
    },
    'best_metrics'       : {
        'ppf_rel': best['ppf_rel'], 'ppf_mae': best['ppf_mae'],
        'ppf_r2' : best['ppf_r2'], 'cycle_mae': best['cycle_mae'],
        'keff_r2': best['keff_r2'], 'rel_low': best['rel_low'],
    },
    'all_experiments'    : {
        name: {k: v for k, v in r.items()
               if k not in ('history', 'ppf_pred', 'ppf_true',
                            'cyc_pred', 'cyc_true', 'keff_pred', 'keff_true')}
        for name, r in all_results.items()
    },
    'baselines'          : {
        'v4': dict(ppf_rel=3.10, ppf_mae=0.0843, ppf_r2=0.9841,
                   cycle_mae=1.28, keff_r2=0.9912),
    },
}

with open('cnn_v8_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("[SAVED]  cnn_v8_config.json")


# =============================================================================
# SECTION 13 — SAVE TRUST-REGION FREQUENCIES (for QICA v5)
# =============================================================================

tr_flat = np.stack([
    X_tr[:, r, c]
    for r in range(GRID_ROWS) for c in range(GRID_COLS)
    if GRID_LAYOUT[r, c] >= 0
], axis=1)   # (N_train, 31)

train_type_freq = np.zeros((N_POS, N_TYPES), dtype=np.float32)
for p in range(N_POS):
    for t in range(1, N_TYPES + 1):
        train_type_freq[p, t-1] = float((tr_flat[:, p] == t).mean())
train_type_freq  = np.maximum(train_type_freq, 1e-3)
train_type_freq /= train_type_freq.sum(axis=1, keepdims=True)
np.save('train_type_freq_v8.npy', train_type_freq)
print("[SAVED]  train_type_freq_v8.npy  (31×9 trust-region frequencies)")


# =============================================================================
# SECTION 14 — SAVE RESULTS CSV
# =============================================================================

csv_rows = []
for name, r in all_results.items():
    csv_rows.append({
        'name'       : name, 'filters': r['filters'],
        'attn'       : r['attn'], 'loss_mode': r['loss_mode'],
        'sample_wt'  : r['sample_wt'], 'n_params': r['n_params'],
        'ppf_rel'    : round(r['ppf_rel'], 3),
        'ppf_mae'    : round(r['ppf_mae'], 4),
        'ppf_r2'     : round(r['ppf_r2'], 4),
        'rel_low'    : round(r['rel_low'], 3),
        'rel_mid'    : round(r['rel_mid'], 3),
        'rel_high'   : round(r['rel_high'], 3),
        'cycle_mae'  : round(r['cycle_mae'], 2),
        'keff_r2'    : round(r['keff_r2'], 4),
        'best_epoch' : r['best_epoch'],
        'vt_ratio'   : round(r['vt_ratio'], 2),
        'time_s'     : round(r['time_s'], 0),
    })
pd.DataFrame(csv_rows).to_csv('cnn_v8_results.csv', index=False)
print("[SAVED]  cnn_v8_results.csv")


# =============================================================================
# SECTION 15 — COMPARISON PLOTS
# =============================================================================

names    = list(all_results.keys())
n_exp    = len(names)
colors   = plt.cm.tab10(np.linspace(0, 1, n_exp))

fig = plt.figure(figsize=(24, 18))
fig.suptitle(
    f"CNN v8 Comparison  |  WINNER: {best_name}  "
    f"rel_err={best['ppf_rel']:.2f}%  R²={best['ppf_r2']:.4f}  "
    f"keff_R²={best['keff_r2']:.4f}",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# 1. Relative error bar chart
ax = fig.add_subplot(gs[0, 0])
vals   = [all_results[n]['ppf_rel'] for n in names]
bar_cs = ['#2CA02C' if v == min(vals) else '#1B4FBF' for v in vals]
bars   = ax.bar(names, vals, color=bar_cs, alpha=0.75)
ax.axhline(3.10, color='red', lw=1.5, ls='--', alpha=0.7, label='v4 baseline 3.10%')
ax.axhline(2.83, color='orange', lw=1.2, ls=':', alpha=0.7, label='prior best 2.83%')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{v:.2f}%', ha='center', va='bottom', fontsize=7.5)
ax.set_ylabel('Mean relative error (%)')
ax.set_title('PPF_max Relative Error\n(lower = better)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)

# 2. R² comparison
ax = fig.add_subplot(gs[0, 1])
r2s = [all_results[n]['ppf_r2'] for n in names]
ax.bar(names, r2s, color=colors, alpha=0.75)
ax.axhline(0.9841, color='red', lw=1.5, ls='--', alpha=0.7, label='v4 baseline')
for i, (bar_val, v) in enumerate(zip(ax.patches, r2s)):
    ax.text(i, v-0.002, f'{v:.4f}', ha='center', va='top', fontsize=7)
ax.set_ylim([min(r2s)-0.003, 1.0])
ax.set_ylabel('PPF_max R²')
ax.set_title('PPF R² (higher = better)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)

# 3. Low-PPF zone relative error (<2.5)
ax = fig.add_subplot(gs[0, 2])
low_vals = [all_results[n]['rel_low'] for n in names]
ax.bar(names, low_vals, color=colors, alpha=0.75)
ax.axhline(3.50, color='red', lw=1.5, ls='--', alpha=0.7, label='v4 est. ~3.5%')
for i, v in enumerate(low_vals):
    ax.text(i, v+0.02, f'{v:.2f}%', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Rel. error (%)')
ax.set_title('PPF Rel. Error in Low Zone\nPPF < 2.5 (QICA target)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)

# 4. Cycle MAE + keff R²
ax = fig.add_subplot(gs[0, 3])
cyc_maes = [all_results[n]['cycle_mae'] for n in names]
keff_r2s = [all_results[n]['keff_r2']   for n in names]
x = np.arange(n_exp)
w = 0.35
b1 = ax.bar(x-w/2, cyc_maes, w, label='Cycle MAE (d)', color='#1B4FBF', alpha=0.7)
ax2 = ax.twinx()
b2 = ax2.bar(x+w/2, keff_r2s, w, label='keff R²',     color='#D62728', alpha=0.7)
ax.axhline(1.28, color='#1B4FBF', lw=1, ls='--', alpha=0.5)
ax2.axhline(0.9912, color='#D62728', lw=1, ls='--', alpha=0.5)
ax.set_ylabel('Cycle MAE (days)', color='#1B4FBF')
ax2.set_ylabel('keff R²', color='#D62728')
ax.set_title('Cycle Length MAE & keff R²')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
lines = [b1, b2]
ax.legend(lines, [l.get_label() for l in lines], fontsize=7)

# 5. Training curves (loss)
ax = fig.add_subplot(gs[1, :2])
for (name, _, _, _, _), col in zip(EXPERIMENTS, colors):
    h = all_results[name]['history']
    ax.plot(h['val_loss'], color=col, lw=1.5, label=f'{name}(val)')
    ax.plot(h['loss'],     color=col, lw=0.8, ls='--', alpha=0.5)
ax.set_yscale('log')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
ax.set_title('Training Curves (solid=val, dashed=train)')
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

# 6. PPF scatter for best experiment
ax = fig.add_subplot(gs[1, 2])
r = all_results[best_name]
ppf_t, ppf_p = r['ppf_true'], r['ppf_pred']
lim = [ppf_t.min()-0.1, ppf_t.max()+0.1]
ax.scatter(ppf_t, ppf_p, alpha=0.3, s=6, color='#1B4FBF')
ax.plot(lim, lim, 'k--', lw=1)
ax.axhspan(PPF_REPORT_LOW, PPF_REPORT_HIGH, alpha=0.07, color='teal')
ax.set_xlabel('True ppf_max'); ax.set_ylabel('Predicted ppf_max')
ax.set_title(f'{best_name} — PPF Scatter\nMAE={r["ppf_mae"]:.4f}  R²={r["ppf_r2"]:.4f}')
ax.grid(alpha=0.3)

# 7. keff scatter for best
ax = fig.add_subplot(gs[1, 3])
kt, kp = r['keff_true'], r['keff_pred']
ax.scatter(kt, kp, alpha=0.3, s=6, color='#9467BD')
lim_k = [kt.min()-0.002, kt.max()+0.002]
ax.plot(lim_k, lim_k, 'k--', lw=1)
ax.set_xlabel('True keff'); ax.set_ylabel('Predicted keff')
ax.set_title(f'{best_name} — keff Scatter\nMAE={r["keff_mae"]:.5f}  R²={r["keff_r2"]:.4f}')
ax.grid(alpha=0.3)

# 8. Architecture comparison: [32,64] vs [32,64,128]+attn
ax = fig.add_subplot(gs[2, 0])
arch_a = [n for n in names if n.startswith('A_')]
arch_c = [n for n in names if n.startswith('C_')]
a_rels = [all_results[n]['ppf_rel'] for n in arch_a]
c_rels = [all_results[n]['ppf_rel'] for n in arch_c]
x_ = np.arange(min(len(arch_a), len(arch_c)))
ax.bar(x_-0.2, a_rels[:len(x_)], 0.35, label='[32,64] no-attn', color='#1B4FBF', alpha=0.75)
ax.bar(x_+0.2, c_rels[:len(x_)], 0.35, label='[32,64,128]+attn', color='#D62728', alpha=0.75)
ax.set_xticks(x_)
ax.set_xticklabels([n.split('_', 1)[1] for n in arch_a[:len(x_)]], fontsize=8)
ax.axhline(3.10, color='grey', lw=1, ls='--', label='v4 3.10%')
ax.set_ylabel('Rel. error (%)'); ax.set_title('Architecture Comparison\n[32,64] vs [32,64,128]+attn')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 9. MSE vs LOG comparison
ax = fig.add_subplot(gs[2, 1])
mse_ns = [n for n in names if 'MSE' in n]
log_ns = [n for n in names if 'LOG' in n and 'SWT' not in n]
mse_r  = [all_results[n]['ppf_rel'] for n in mse_ns]
log_r  = [all_results[n]['ppf_rel'] for n in log_ns]
x_ = np.arange(min(len(mse_ns), len(log_ns)))
ax.bar(x_-0.2, mse_r[:len(x_)], 0.35, label='MSE loss', color='#2CA02C', alpha=0.75)
ax.bar(x_+0.2, log_r[:len(x_)], 0.35, label='LOG loss', color='#F5A623', alpha=0.75)
ax.set_xticks(x_)
ax.set_xticklabels([n.split('_', 1)[1] for n in mse_ns[:len(x_)]], fontsize=8)
ax.axhline(3.10, color='grey', lw=1, ls='--', label='v4 3.10%')
ax.set_ylabel('Rel. error (%)'); ax.set_title('Loss Comparison\nMSE vs LOG (W_LOG=4.0)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 10. LOG vs LOG+SWT comparison
ax = fig.add_subplot(gs[2, 2])
log_ns2  = [n for n in names if 'LOG' in n and 'SWT' not in n]
swt_ns   = [n for n in names if 'SWT' in n]
log_r2   = [all_results[n]['ppf_rel'] for n in log_ns2]
swt_r    = [all_results[n]['ppf_rel'] for n in swt_ns]
x_ = np.arange(min(len(log_ns2), len(swt_ns)))
ax.bar(x_-0.2, log_r2[:len(x_)], 0.35, label='LOG (no SWT)', color='#F5A623', alpha=0.75)
ax.bar(x_+0.2, swt_r[:len(x_)],  0.35, label='LOG + SWT',    color='#E377C2', alpha=0.75)
ax.set_xticks(x_)
ax.set_xticklabels([n.split('_', 1)[1] for n in log_ns2[:len(x_)]], fontsize=8)
ax.axhline(3.10, color='grey', lw=1, ls='--', label='v4 3.10%')
ax.set_ylabel('Rel. error (%)')
ax.set_title('LOG vs LOG+SampleWeight\nInverse-PPF weighting effect')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

# 11. Val/train ratio (overfitting check)
ax = fig.add_subplot(gs[2, 3])
vt_vals = [all_results[n]['vt_ratio'] for n in names]
bar_cs  = ['#D62728' if v > 1.3 else '#2CA02C' for v in vt_vals]
ax.bar(names, vt_vals, color=bar_cs, alpha=0.75)
ax.axhline(1.3, color='red', lw=1.5, ls='--', alpha=0.7, label='Overfit threshold')
ax.axhline(1.0, color='green', lw=1, ls=':', alpha=0.7)
ax.set_ylabel('val_loss / train_loss at best epoch')
ax.set_title('Overfitting Check\n(< 1.3 = good generalisation)')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)

plt.savefig('cnn_v8_comparison.png', dpi=150, bbox_inches='tight')
print("\n[SAVED]  cnn_v8_comparison.png")


# =============================================================================
# SECTION 16 — FINAL SUMMARY
# =============================================================================

t_total = time.time() - t_global
print(f"\n{'='*70}")
print(f"CNN v8  FINAL SUMMARY  ({t_total:.0f}s / {t_total/60:.1f}min total)")
print(f"{'='*70}")
print(f"  {'Experiment':<14} {'rel_err':>8}  {'<2.5':>7}  {'R²':>7}  "
      f"{'cycle':>7}  {'keff':>7}  {'v/t':>5}")
print(f"  {'─'*14} {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}")
print(f"  {'v4 BASELINE':<14} {'3.10%':>8}  {'~3.5%':>7}  {'0.9841':>7}  "
      f"{'1.28d':>7}  {'0.9912':>7}  {'0.89':>5}")
for name, r in all_results.items():
    marker = ' ✓' if r['ppf_rel'] < 3.10 else ''
    print(f"  {name:<14} {r['ppf_rel']:>7.2f}%  "
          f"{r['rel_low']:>6.2f}%  {r['ppf_r2']:>7.4f}  "
          f"{r['cycle_mae']:>6.2f}d  {r['keff_r2']:>7.4f}  "
          f"{r['vt_ratio']:>5.2f}{marker}")
print()
print(f"  WINNER: {best_name}")
print(f"    → cnn_v8_model.keras (best model)")
print(f"    → cnn_v8_config.json (use in qica_v5.py)")
print(f"    → train_type_freq_v8.npy (trust-region for QICA)")
print()
print(f"  NEXT STEP: Run qica_v5.py (ENTROPY_MODE='both' for full entropy pipeline)")
print(f"{'='*70}")