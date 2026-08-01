"""
=============================================================================
cnn_v10_octant_retrain.py  —  BEAVRS CNN v10  |  Corrected 6x8 Octant Layout
=============================================================================
Same data, same split, same architecture family as cnn-v9.py -- the ONLY
thing that changes is GRID_LAYOUT: 6x6 arbitrary reshape -> 6x8 real BEAVRS
octant adjacency (from openmc_beavrs_vver1000_v5_FIXED.py). This does NOT
require new OpenMC simulations: ml_dataset_constrained.csv's ppf_max /
cycle_length / keff labels are already correct, physics-derived numbers
indexed by loading_0..loading_30, entirely independent of how you choose to
reshape them into a 2D image for the CNN's convolutions. Only the reshape
changes here.

At the end, loads your EXISTING cnn_v9_model.keras and evaluates it on the
SAME held-out test set (identical SEED, identical train_test_split call
order/fractions as cnn-v9.py, so the test rows line up exactly) for a
direct, fair old-vs-new accuracy comparison -- so you can see whether the
corrected spatial structure helps, hurts, or is roughly a wash before you
commit to it downstream.

OUTPUTS (mirrors cnn-v9.py's, renamed to v10):
  cnn_v10_model.keras, cnn_v10_config.json, cnn_v10_sens.csv,
  cnn_v10_results.png, cnn_v10_al_candidates.csv, train_type_freq_v10.npy,
  cnn_v9_vs_v10_comparison.csv / .png

Run:  python cnn_v10_octant_retrain.py
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print("cnn_v10_octant_retrain.py — corrected 6x8 octant GRID_LAYOUT\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

BEAVRS_CSV  = 'ml_dataset_constrained.csv'
XL_FILE     = 'cycle_length_summary.xlsx'
MODEL_NAME  = 'cnn_v10_model.keras'
CONFIG_NAME = 'cnn_v10_config.json'
SENS_NAME   = 'cnn_v10_sens.csv'
PLOT_NAME   = 'cnn_v10_results.png'
AL_CSV      = 'cnn_v10_al_candidates.csv'
FREQ_PATH   = 'train_type_freq_v10.npy'

OLD_MODEL_NAME = 'cnn_v9_model.keras'   # for the baseline comparison

PPF_REPORT_LOW  = 2.0
PPF_REPORT_HIGH = 4.5

N_POS    = 31
N_TYPES  = 9
N_STEPS  = 31

# NEW, corrected 6x8 octant layout (from openmc_beavrs_vver1000_v5_FIXED.py).
# THIS is the only architectural change relative to cnn-v9.py.
GRID_ROWS, GRID_COLS = 6, 8
GRID_LAYOUT = np.array([
    [-1, -1, -1, -1, -1, 29, 30, -1],
    [-1, -1, -1, -1, 26, 27, 28, -1],
    [-1, -1, -1, 21, 22, 23, 24, 25],
    [-1, -1, 15, 16, 17, 18, 19, 20],
    [-1,  8,  9, 10, 11, 12, 13, 14],
    [ 0,  1,  2,  3,  4,  5,  6,  7],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)
assert GRID_MASK.sum() == N_POS
assert sorted(GRID_LAYOUT[GRID_MASK].tolist()) == list(range(N_POS))

# OLD layout, needed only to re-derive the flat->grid mapping the OLD model
# expects, for the baseline comparison at the end.
OLD_GRID_ROWS, OLD_GRID_COLS = 6, 6
OLD_GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)

BATCH_SIZE  = 128
EPOCHS      = 400
LR          = 0.001
DROPOUT     = 0.15
CONV_DROP   = 0.10
WEIGHT_DECAY= 1e-4
TEST_FRAC   = 0.15
VAL_FRAC    = 0.15
SEED        = 42
MC_SAMPLES  = 30

AL_UNCERTAINTY_THRESHOLD = 0.07
AL_MAX_QUERIES_PER_ROUND = 50


# =============================================================================
# SECTION 2 — MONOCORE CYCLE LENGTHS (unchanged from cnn-v9.py)
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
ASSEMBLY_CYCLE_EQUIV = {0: 0.0}
ASSEMBLY_CYCLE_EQUIV.update(monocore_map)


# =============================================================================
# SECTION 3 — LOAD DATA  (identical to cnn-v9.py)
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


# =============================================================================
# SECTION 4 — TARGETS AND FEATURES  (identical to cnn-v9.py)
# =============================================================================

N_OUTPUTS     = 1 + 1 + N_STEPS + 1 + 1
IDX_PPF_MAX   = 0
IDX_PPF_BOC   = 1
IDX_PPF_STEPS_START = 2
IDX_PPF_STEPS_END   = 2 + N_STEPS
IDX_CYCLE     = 2 + N_STEPS
IDX_RHO       = 3 + N_STEPS

Y_ppf_max   = ppf_global_max.reshape(-1, 1).astype(np.float32)
Y_ppf_boc   = ppf_boc.reshape(-1, 1).astype(np.float32)
Y_ppf_steps = step_max_ppf.astype(np.float32)
Y_cycle     = df['cycle_length'].values.reshape(-1, 1).astype(np.float32)
Y_rho       = rho_pcm.reshape(-1, 1).astype(np.float32)

Y_main    = np.concatenate([Y_ppf_max, Y_ppf_boc, Y_ppf_steps, Y_cycle], axis=1)
Y_rho_col = Y_rho

X_raw  = df[load_cols].values.astype(np.int32)


def flat_to_grid(flat, grid_rows, grid_cols, grid_layout):
    Xg = np.zeros((len(flat), grid_rows, grid_cols), dtype=np.int32)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid_layout[r, c] >= 0:
                Xg[:, r, c] = flat[:, grid_layout[r, c]]
    return Xg


X_grid_new = flat_to_grid(X_raw, GRID_ROWS, GRID_COLS, GRID_LAYOUT)
print(f"[INPUT]  New grid shape: {X_grid_new.shape}  (was 6x6, now {GRID_ROWS}x{GRID_COLS})")
print(f"  Active fuel cells : {GRID_MASK.sum()}/{GRID_ROWS*GRID_COLS}")
print(f"[TARGETS] Main: {Y_main.shape}, Rho: {Y_rho_col.shape}\n")


# =============================================================================
# SECTION 5 — SPLIT + SCALERS  (identical seed/fractions/call order to
# cnn-v9.py so the OLD model's test rows and the NEW model's test rows match)
# =============================================================================

(X_tr, X_tmp,
 Ym_tr, Ym_tmp,
 Yr_tr, Yr_tmp) = train_test_split(X_grid_new, Y_main, Y_rho_col,
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


# =============================================================================
# SECTION 6 — ARCHITECTURE  (identical to cnn-v9.py, just parameterized grid)
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                    kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        self.proj  = None
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
        cfg = super().get_config()
        cfg.update({'filters': self._filters, 'kernel_size': 3, 'dropout': self._dropout_rate})
        return cfg


def build_cnn(grid_rows, grid_cols, n_types=N_TYPES + 1, embed_dim=16,
              filters=(32, 64), dense_units=128, dropout=DROPOUT,
              conv_dropout=CONV_DROP, n_outputs=N_OUTPUTS, name='BEAVRS_CNN'):
    inp = keras.Input(shape=(grid_rows, grid_cols), dtype='int32', name='loading_grid')
    x   = layers.Embedding(n_types, embed_dim, name='assembly_embedding')(inp)

    x = ConvResBlock(filters[0], dropout=conv_dropout, name='conv_block_1')(x)
    x = ConvResBlock(filters[1], dropout=conv_dropout, name='conv_block_2')(x)
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)

    shared = layers.Dense(dense_units, activation='gelu', name='shared_dense')(x)
    shared = layers.Dropout(dropout, name='shared_dropout')(shared)
    shared = layers.Dense(dense_units // 2, activation='gelu', name='shared_dense2')(shared)

    h_ppf    = layers.Dense(64, activation='gelu', name='ppf_dense')(shared)
    h_ppf    = layers.Dropout(dropout * 0.5, name='ppf_dropout')(h_ppf)
    out_ppf  = layers.Dense(1 + 1 + N_STEPS, activation='linear', name='ppf_output')(h_ppf)

    h_cyc    = layers.Dense(32, activation='gelu', name='cycle_dense')(shared)
    h_cyc    = layers.Dropout(dropout * 0.3, name='cycle_dropout')(h_cyc)
    out_cycle= layers.Dense(1, activation='linear', name='cycle_output')(h_cyc)

    h_rho    = layers.Dense(32, activation='gelu', name='rho_dense')(shared)
    h_rho    = layers.Dropout(dropout * 0.3, name='rho_dropout')(h_rho)
    out_rho  = layers.Dense(1, activation='linear', name='rho_output')(h_rho)

    out = layers.Concatenate(name='predictions')([out_ppf, out_cycle, out_rho])
    return keras.Model(inputs=inp, outputs=out, name=name)


model = build_cnn(GRID_ROWS, GRID_COLS, name='BEAVRS_CNN_v10')
model.summary()
print(f"\n[MODEL]  Parameters: {model.count_params():,}\n")


# =============================================================================
# SECTION 7 — LOSS  (identical to cnn-v9.py)
# =============================================================================

W_PPF_MAX, W_PPF_BOC, W_PPF_STEPS, W_CYCLE, W_RHO, W_MONO = 3.0, 2.0, 0.5, 1.0, 5.0, 0.01

@tf.keras.utils.register_keras_serializable()
def v10_loss(y_true, y_pred):
    ppf_max_loss   = W_PPF_MAX   * tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
    ppf_boc_loss   = W_PPF_BOC   * tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
    ppf_steps_loss = W_PPF_STEPS * tf.reduce_mean(tf.square(
        y_true[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]
        - y_pred[:, IDX_PPF_STEPS_START:IDX_PPF_STEPS_END]))
    cycle_loss     = W_CYCLE     * tf.reduce_mean(tf.square(
        y_true[:, IDX_CYCLE] - y_pred[:, IDX_CYCLE]))
    rho_loss       = W_RHO       * tf.reduce_mean(tf.square(
        y_true[:, IDX_RHO] - y_pred[:, IDX_RHO]))
    late       = y_pred[:, IDX_PPF_STEPS_START + 3:IDX_PPF_STEPS_END]
    diffs      = late[:, 1:] - late[:, :-1]
    violations = tf.maximum(0.0, diffs)
    mono_loss  = W_MONO * tf.reduce_mean(tf.square(violations))
    return ppf_max_loss + ppf_boc_loss + ppf_steps_loss + cycle_loss + rho_loss + mono_loss


# =============================================================================
# SECTION 8 — TRAIN
# =============================================================================

model.compile(optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY),
              loss=v10_loss, metrics=['mae'])

print("[TRAINING] Starting CNN v10 (corrected 6x8 octant layout) ...")
t_start = time.time()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15,
                                       min_lr=1e-5, verbose=1),
    keras.callbacks.LambdaCallback(
        on_epoch_end=lambda ep, logs: print(
            f"  Ep {ep+1:4d} | loss: {logs['loss']:.5f} | val: {logs['val_loss']:.5f}"
        ) if (ep + 1) % 25 == 0 else None),
]

history = model.fit(X_tr, Y_tr_sc, validation_data=(X_val, Y_val_sc),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=0)

t_train = time.time() - t_start
best_epoch = int(np.argmin(history.history['val_loss'])) + 1
print(f"\n[TRAINING DONE]  {t_train:.1f}s  |  best epoch: {best_epoch}\n")


def inverse_transform(Y_sc):
    Y_main_real = ym_scaler.inverse_transform(Y_sc[:, :34])
    Y_rho_real  = yr_scaler.inverse_transform(Y_sc[:, 34:35])
    return np.concatenate([Y_main_real, Y_rho_real], axis=1)


# =============================================================================
# SECTION 9 — EVALUATE NEW (v10) MODEL
# =============================================================================

print("[EVALUATION] v10 (new, 6x8 octant layout) on test set ...")
Y_pred_sc   = model.predict(X_test, verbose=0)
Y_pred_real = inverse_transform(Y_pred_sc)
Y_true_real = inverse_transform(Y_test_sc)

def metrics_for(y_true, y_pred):
    mae = np.abs(y_pred - y_true).mean()
    r2  = r2_score(y_true, y_pred)
    return mae, r2

ppf_mae_v10, ppf_r2_v10     = metrics_for(Y_true_real[:, IDX_PPF_MAX], Y_pred_real[:, IDX_PPF_MAX])
cycle_mae_v10, cycle_r2_v10 = metrics_for(Y_true_real[:, IDX_CYCLE],   Y_pred_real[:, IDX_CYCLE])
rho_mae_v10, rho_r2_v10     = metrics_for(Y_true_real[:, IDX_RHO],     Y_pred_real[:, IDX_RHO])

print(f"  PPF_max   : MAE={ppf_mae_v10:.4f}  R²={ppf_r2_v10:.4f}")
print(f"  cycle     : MAE={cycle_mae_v10:.2f}d  R²={cycle_r2_v10:.4f}")
print(f"  rho_pcm   : MAE={rho_mae_v10:.0f}pcm  R²={rho_r2_v10:.4f}\n")


# =============================================================================
# SECTION 10 — BASELINE COMPARISON: old cnn_v9 (6x6) vs new cnn_v10 (6x8),
# on the IDENTICAL held-out test rows
# =============================================================================

print("=" * 70)
print("BASELINE COMPARISON — cnn_v9 (old 6x6 layout) vs cnn_v10 (new 6x8 layout)")
print("Same test rows for both (identical seed/split), so this is a fair,")
print("apples-to-apples check of whether the corrected spatial structure")
print("helps, hurts, or is a wash.")
print("=" * 70)

if os.path.exists(OLD_MODEL_NAME):
    old_model = keras.models.load_model(OLD_MODEL_NAME, compile=False)

    # X_test above is already in NEW grid shape (6,8); rebuild the SAME test
    # rows in the OLD 6x6 grid shape for the old model. We recover which
    # original flat rows ended up in the test split by re-doing the split
    # on X_raw with the identical seed/fractions (deterministic -> same rows).
    # Re-run the identical two-stage split on X_raw (same random_state/
    # test_size/call order as Section 5 above) purely to recover which raw
    # flat-pattern rows ended up in the test set -- train_test_split is
    # deterministic given the same inputs, so these are exactly the same
    # rows as X_test/Y_test above, just still in flat (not gridded) form.
    Xraw_tr, Xraw_tmp = train_test_split(
        X_raw, test_size=TEST_FRAC + VAL_FRAC, random_state=SEED)
    _, Xraw_test = train_test_split(
        Xraw_tmp, test_size=0.5, random_state=SEED)

    X_test_old_grid = flat_to_grid(Xraw_test, OLD_GRID_ROWS, OLD_GRID_COLS, OLD_GRID_LAYOUT)

    Y_pred_sc_old = old_model.predict(X_test_old_grid, verbose=0)
    Y_pred_real_old = inverse_transform(Y_pred_sc_old)

    ppf_mae_v9, ppf_r2_v9     = metrics_for(Y_true_real[:, IDX_PPF_MAX], Y_pred_real_old[:, IDX_PPF_MAX])
    cycle_mae_v9, cycle_r2_v9 = metrics_for(Y_true_real[:, IDX_CYCLE],   Y_pred_real_old[:, IDX_CYCLE])
    rho_mae_v9, rho_r2_v9     = metrics_for(Y_true_real[:, IDX_RHO],     Y_pred_real_old[:, IDX_RHO])

    comparison = pd.DataFrame({
        'metric': ['ppf_max_MAE', 'ppf_max_R2', 'cycle_MAE_days', 'cycle_R2', 'rho_MAE_pcm', 'rho_R2'],
        'cnn_v9_old_6x6': [ppf_mae_v9, ppf_r2_v9, cycle_mae_v9, cycle_r2_v9, rho_mae_v9, rho_r2_v9],
        'cnn_v10_new_6x8': [ppf_mae_v10, ppf_r2_v10, cycle_mae_v10, cycle_r2_v10, rho_mae_v10, rho_r2_v10],
    })
    comparison['delta_v10_minus_v9'] = comparison['cnn_v10_new_6x8'] - comparison['cnn_v9_old_6x6']
    print(comparison.to_string(index=False))
    comparison.to_csv('cnn_v9_vs_v10_comparison.csv', index=False)

    verdict_ppf = "IMPROVED" if ppf_r2_v10 > ppf_r2_v9 else ("ROUGHLY SAME" if abs(ppf_r2_v10-ppf_r2_v9) < 0.005 else "SLIGHTLY WORSE")
    print(f"\nPPF_max R²: v9={ppf_r2_v9:.4f} -> v10={ppf_r2_v10:.4f}  ({verdict_ppf})")
    print("If v10 is comparable or better: safe to switch the whole pipeline (QICA, GA")
    print("scripts, sensitivity/PCE analysis) to point at cnn_v10_model.keras from here on.")
    print("If v10 is meaningfully worse: the old arbitrary layout may have accidentally")
    print("given the CNN an easier optimization landscape for THIS architecture size --")
    print("worth trying a slightly larger conv stack before concluding the fix backfired.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, mae9, mae10, r29, r210) in zip(axes, [
        ('PPF_max', ppf_mae_v9, ppf_mae_v10, ppf_r2_v9, ppf_r2_v10),
        ('Cycle length (d)', cycle_mae_v9, cycle_mae_v10, cycle_r2_v9, cycle_r2_v10),
        ('rho_pcm', rho_mae_v9, rho_mae_v10, rho_r2_v9, rho_r2_v10),
    ]):
        ax.bar(['v9 (old 6x6)', 'v10 (new 6x8)'], [r29, r210], color=['#888888', '#1B4FBF'])
        ax.set_ylabel('R²'); ax.set_title(f'{name}\nMAE: {mae9:.4g} -> {mae10:.4g}')
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('cnn_v9_vs_v10_comparison.png', dpi=150)
    print(f"\n[SAVED] cnn_v9_vs_v10_comparison.csv  cnn_v9_vs_v10_comparison.png")
else:
    print(f"  [SKIP] {OLD_MODEL_NAME} not found — cannot run the baseline comparison. "
          f"Copy your existing cnn_v9_model.keras into this directory and re-run "
          f"just this section if you need the comparison after the fact.")


# =============================================================================
# SECTION 11 — MC DROPOUT (unchanged logic from cnn-v9.py)
# =============================================================================

print("\n[MC DROPOUT] Estimating prediction uncertainty (v10) ...")
mc_preds_sc = np.stack([model(X_test, training=True).numpy() for _ in range(MC_SAMPLES)])
mc_mean_sc = mc_preds_sc.mean(axis=0)
mc_std_sc  = mc_preds_sc.std(axis=0)
mc_mean_real = inverse_transform(mc_mean_sc)
mc_std_main  = mc_std_sc[:, :34] * ym_scaler.scale_
mc_std_rho   = mc_std_sc[:, 34:35] * yr_scaler.scale_
mc_std_real  = np.concatenate([mc_std_main, mc_std_rho], axis=1)
ppf_mc_std   = mc_std_real[:, IDX_PPF_MAX]
print(f"  Mean σ(ppf_max)   : {ppf_mc_std.mean():.4f}")


# =============================================================================
# SECTION 12 — SENSITIVITY ANALYSIS
# =============================================================================

print("\n[SENSITIVITY]  Computing ∂ppf_max/∂position (v10, 6x8 layout) ...")
n_sens   = min(200, len(X_test))
X_sample = tf.constant(X_test[:n_sens], dtype=tf.int32)

sens_norm = np.ones(N_POS, dtype=np.float32)
sens_pos  = np.ones(N_POS, dtype=np.float32)

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
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    sens_pos[GRID_LAYOUT[r, c]] = sens_grid_raw[r, c]
        sens_norm = sens_pos / (sens_pos.max() + 1e-8)
        top5 = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"  Top-5 critical positions (v10) : {top5}")
except Exception as e:
    print(f"  [WARN] Gradient failed ({e}) — using uniform sensitivity")

sens_df = pd.DataFrame({'position': [f'pos_{i}' for i in range(N_POS)],
                         'sensitivity': sens_pos, 'sensitivity_norm': sens_norm})
sens_df.to_csv(SENS_NAME, index=False)
print(f"  Saved: {SENS_NAME}")


# =============================================================================
# SECTION 13 — ACTIVE LEARNING CANDIDATES (unchanged logic)
# =============================================================================

print("\n[ACTIVE LEARNING]  Scanning full dataset for query candidates (v10) ...")
mc_all_full = np.stack([model(X_grid_new, training=True).numpy() for _ in range(MC_SAMPLES)])
mc_mean_full = inverse_transform(mc_all_full.mean(axis=0))
mc_std_full_phy = np.concatenate([
    mc_all_full.std(axis=0)[:, :34] * ym_scaler.scale_,
    mc_all_full.std(axis=0)[:, 34:35] * yr_scaler.scale_,
], axis=1)
ppf_full_pred = mc_mean_full[:, IDX_PPF_MAX]
ppf_full_std  = mc_std_full_phy[:, IDX_PPF_MAX]
priority_score = ppf_full_std / (ppf_full_pred + 1e-6)
high_unc_mask  = ppf_full_std >= AL_UNCERTAINTY_THRESHOLD
low_ppf_mask   = ppf_full_pred <= np.percentile(ppf_full_pred, 25)
query_mask     = high_unc_mask & low_ppf_mask
query_idxs     = np.where(query_mask)[0]
query_sorted   = query_idxs[np.argsort(priority_score[query_idxs])[::-1]]
query_top      = query_sorted[:AL_MAX_QUERIES_PER_ROUND]
print(f"  Priority candidates: {len(query_idxs)}  |  Top-{AL_MAX_QUERIES_PER_ROUND} flagged: {len(query_top)}")

al_records = []
for idx in query_top:
    al_records.append({
        'dataset_idx': int(idx), 'pred_ppf_max': float(ppf_full_pred[idx]),
        'pred_ppf_std': float(ppf_full_std[idx]), 'priority': float(priority_score[idx]),
        'cycle_length': float(mc_mean_full[idx, IDX_CYCLE]),
        'rho_pcm_boc': float(mc_mean_full[idx, IDX_RHO]),
        **{f'pos_{j}': int(X_raw[idx, j]) for j in range(N_POS)},
    })
pd.DataFrame(al_records).to_csv(AL_CSV, index=False)
print(f"  Saved: {AL_CSV}")


# =============================================================================
# SECTION 14 — SAVE MODEL + CONFIG + TRUST REGION
# =============================================================================

model.save(MODEL_NAME)

config = {
    'N_POS': N_POS, 'N_TYPES': N_TYPES, 'N_STEPS': N_STEPS,
    'GRID_ROWS': GRID_ROWS, 'GRID_COLS': GRID_COLS,
    'GRID_LAYOUT': GRID_LAYOUT.tolist(), 'GRID_MASK': GRID_MASK.tolist(),
    'IDX_PPF_MAX': IDX_PPF_MAX, 'IDX_PPF_BOC': IDX_PPF_BOC,
    'IDX_PPF_STEPS_START': IDX_PPF_STEPS_START, 'IDX_PPF_STEPS_END': IDX_PPF_STEPS_END,
    'IDX_CYCLE': IDX_CYCLE, 'IDX_RHO': IDX_RHO, 'N_OUTPUTS': N_OUTPUTS,
    'PPF_REPORT_LOW': PPF_REPORT_LOW, 'PPF_REPORT_HIGH': PPF_REPORT_HIGH,
    'ym_scaler_mean' : ym_scaler.mean_.tolist(), 'ym_scaler_scale': ym_scaler.scale_.tolist(),
    'yr_scaler_mean' : yr_scaler.mean_.tolist(), 'yr_scaler_scale': yr_scaler.scale_.tolist(),
    'ASSEMBLY_CYCLE_EQUIV': {str(k): float(v) for k, v in ASSEMBLY_CYCLE_EQUIV.items()},
    'mc_samples': MC_SAMPLES, 'al_uncertainty_thr': AL_UNCERTAINTY_THRESHOLD,
    'test_ppf_mae': float(ppf_mae_v10), 'test_ppf_r2': float(ppf_r2_v10),
    'test_cycle_mae': float(cycle_mae_v10), 'test_rho_r2': float(rho_r2_v10),
}
with open(CONFIG_NAME, 'w') as f:
    json.dump(config, f, indent=2)

tr_flat = np.stack([X_tr[:, r, c] for r in range(GRID_ROWS) for c in range(GRID_COLS)
                     if GRID_LAYOUT[r, c] >= 0], axis=1)
train_type_freq = np.zeros((N_POS, N_TYPES), dtype=np.float32)
for p in range(N_POS):
    for t in range(1, N_TYPES + 1):
        train_type_freq[p, t - 1] = float((tr_flat[:, p] == t).mean())
train_type_freq = np.maximum(train_type_freq, 1e-3)
train_type_freq /= train_type_freq.sum(axis=1, keepdims=True)
np.save(FREQ_PATH, train_type_freq)

print(f"\n[SAVED]  {MODEL_NAME}\n[SAVED]  {CONFIG_NAME}\n[SAVED]  {SENS_NAME}\n[SAVED]  {FREQ_PATH}")
print("\nNEXT STEP per round4_results_and_plan.md: re-run qica_v11_production.py")
print("pointed at cnn_v10_model.keras / cnn_v10_config.json, USE_CYCLE_FITNESS=False,")
print("to get a FRESH, valid QICA baseline before running 13_final_qica_decision_suite.py.")