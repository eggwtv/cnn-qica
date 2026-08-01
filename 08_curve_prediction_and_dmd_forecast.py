"""
=============================================================================
08_curve_prediction_and_dmd_forecast.py
=============================================================================
Two things, both about "predict a whole curve instead of one number":

  PART A — Functional PCA on the training set's PPF burnup curves.
    Your CNN already outputs 31 ppf_steps values, i.e. it's *sort of*
    already predicting the whole curve, just as 31 independent scalar
    heads. This asks: how compressible are these curves actually? If
    95% of curve-shape variance across your whole training set lives in
    a handful of modes, that's a strong argument for treating the curve
    as ONE structured functional object (predict a short mode-loading
    vector) rather than 31 independent regression targets, and it also
    tells you how many DMD modes are worth keeping in Part B.

  PART B — DMD as a FORECASTER, not a diagnostic. Every DMD use so far
    (04_dmd_signal_validation.py, and Section 5/6 of 05_mentor_...) used
    DMD strictly after the fact: decompose an already-known curve, check
    if the reconstruction error says anything about model confidence.
    That role is now well-evidenced to be redundant with MC-dropout
    sigma (see results_and_cammi_review.md Part 5) -- consensus: drop it
    there. But DMD's ORIGINAL purpose (Schmid 2010) is predictive: fit
    the dynamics operator A on the first half of a curve, then use A to
    FORECAST the second half from just the initial condition. This has
    not been tested at all yet, and it's a genuinely different, still-
    open question. Compares DMD-forecast accuracy against a naive
    persistence baseline and (if available) your CNN's own ppf_steps
    predictions on the same held-out patterns.

Run:  python 08_curve_prediction_and_dmd_forecast.py
=============================================================================
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
rng = np.random.default_rng(42)

DATA_CSV   = 'ml_dataset_constrained.csv'
MODEL_FILE = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
OUT_PREFIX = 'curve_prediction'
N_POS = 31


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


if not has(DATA_CSV):
    print("Need ml_dataset_constrained.csv. Exiting.")
    raise SystemExit

df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
ppf_steps_avail = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_steps_avail = [s for s in ppf_steps_avail
                   if len([c for c in df.columns if c.startswith(f"ppf_s{s}_")]) == len(ppf_assembs)]
n_steps = len(ppf_steps_avail)
load_cols = [f'loading_{i}' for i in range(N_POS)]

if n_steps < 5 or not all(c in df.columns for c in load_cols):
    print("Dataset doesn't have the expected ppf_s*_a* / loading_* columns. Exiting.")
    raise SystemExit

print(f"[DATA] {len(df)} patterns, {n_steps} burnup steps, {len(ppf_assembs)} assemblies")

# curve = ppf_max AT EACH STEP (max over assemblies), matching your CNN's
# own ppf_steps output target -- this is "the whole curve" in the same
# sense your CNN already tries to predict it.
curves = np.zeros((len(df), n_steps), dtype=np.float64)
for si, s in enumerate(ppf_steps_avail):
    cols = [f'ppf_s{s}_a{a}' for a in ppf_assembs if f'ppf_s{s}_a{a}' in df.columns]
    curves[:, si] = df[cols].values.astype(np.float64).max(axis=1)


# =============================================================================
# PART A — Functional PCA on the curve ensemble
# =============================================================================
print("\n" + "=" * 70)
print("PART A — Functional PCA: how compressible is the PPF burnup curve?")
print("=" * 70)

curve_mean = curves.mean(axis=0)
curve_centered = curves - curve_mean

# PCA via SVD (functional PCA for a discretely-sampled curve ensemble is
# just PCA over the sample axis; the "functional" part is in how you read
# the resulting modes -- as curve shapes, not independent scalar features)
U, S, Vt = np.linalg.svd(curve_centered, full_matrices=False)
explained_var = (S ** 2) / np.sum(S ** 2)
cum_var = np.cumsum(explained_var)

n_modes_95 = int(np.searchsorted(cum_var, 0.95) + 1)
n_modes_99 = int(np.searchsorted(cum_var, 0.99) + 1)
print(f"  Modes needed for 95% of curve-shape variance : {n_modes_95}  (out of {n_steps} raw steps)")
print(f"  Modes needed for 99% of curve-shape variance : {n_modes_99}")
print(f"  Variance explained by top 3 modes alone       : {cum_var[2]*100:.1f}%")
if n_modes_95 <= n_steps * 0.3:
    print(f"  -> Curves are HIGHLY compressible: {n_modes_95} numbers capture what 31")
    print(f"     independent scalar CNN outputs are currently trying to learn separately.")
    print(f"     Concretely actionable: replace the 31 ppf_steps output neurons with a")
    print(f"     {n_modes_95}-dim mode-loading head, reconstruct the full curve as")
    print(f"     curve_mean + loadings @ modes[:{n_modes_95}] at inference time. Fewer")
    print(f"     output parameters to fit generally means less overfitting risk per target,")
    print(f"     and the modes themselves are physically interpretable (mode 1 is usually")
    print(f"     an overall-magnitude shift, mode 2 a late/early-cycle tilt, etc. -- plot")
    print(f"     below).")
else:
    print(f"  -> Curves need most of their raw dimensionality to represent well; less")
    print(f"     upside to a mode-based reparametrization than expected.")

modes_df = pd.DataFrame({'mode': range(1, n_steps + 1), 'explained_var': explained_var,
                          'cumulative_var': cum_var})
modes_df.to_csv(f'{OUT_PREFIX}_fpca_modes.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(range(1, n_steps + 1), cum_var, 'o-', color='#1B4FBF')
axes[0].axhline(0.95, color='red', ls='--', lw=1, label='95%')
axes[0].axvline(n_modes_95, color='red', ls=':', lw=1)
axes[0].set_xlabel('Number of modes'); axes[0].set_ylabel('Cumulative variance explained')
axes[0].set_title(f'Functional PCA Scree Plot\n{n_modes_95} modes -> 95%')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(curve_mean, color='black', lw=2, label='mean curve')
for k in range(min(3, n_steps)):
    axes[1].plot(curve_mean + S[k] * Vt[k] / np.sqrt(len(df)), lw=1.3, label=f'+mode {k+1}')
axes[1].set_xlabel('Burnup step'); axes[1].set_ylabel('PPF_max')
axes[1].set_title('Mean Curve + Top Modes'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

for i in rng.choice(len(df), 20, replace=False):
    axes[2].plot(curves[i], alpha=0.3, lw=0.8, color='#1B4FBF')
axes[2].plot(curve_mean, color='red', lw=2, label='mean')
axes[2].set_xlabel('Burnup step'); axes[2].set_ylabel('PPF_max')
axes[2].set_title('20 Random Training Curves'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_fpca.png', dpi=150)
print(f"\n[SAVED] {OUT_PREFIX}_fpca_modes.csv  {OUT_PREFIX}_fpca.png")


# =============================================================================
# PART B — DMD as a FORECASTER (predictive role, not diagnostic role)
# =============================================================================
print("\n" + "=" * 70)
print("PART B — DMD as forecaster: predict 2nd half of curve from 1st half")
print("=" * 70)

N_TEST = 400
SPLIT = n_steps // 2   # fit on steps [0, SPLIT), forecast steps [SPLIT, n_steps)

test_idx = rng.choice(len(df), min(N_TEST, len(df)), replace=False)


def dmd_fit_forecast(curve_1d, split, rank):
    """
    Single-channel DMD needs multiple 'spatial' points to be meaningful, so
    for a 1D curve we use a delay-embedding (Hankel matrix) -- standard
    trick (Hankel-DMD / delay-coordinate DMD) to make a scalar time series
    behave like a multi-channel system DMD can act on.
    """
    known = curve_1d[:split]
    n_future = len(curve_1d) - split
    d = max(2, min(rank + 2, split // 2))     # embedding dimension
    if split - d < 2:
        # too short to embed meaningfully -- fall back to persistence
        return np.full(n_future, known[-1])

    H = np.array([known[i:i + d] for i in range(split - d + 1)]).T  # (d, n_snaps)
    X1, X2 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, len(S))
    Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
    Atilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / Sr)
    eigvals, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vhr.conj().T @ np.diag(1.0 / Sr) @ W
    b, *_ = np.linalg.lstsq(Phi, H[:, -1], rcond=None)

    # roll the delay-embedded state forward n_future steps
    state = H[:, -1].copy()
    forecast = []
    cur_b = b.copy()
    for t in range(1, n_future + 1):
        powers = eigvals ** t
        new_state = (Phi @ (cur_b * powers)).real
        forecast.append(new_state[-1])   # last entry of embedded state = newest value
    return np.array(forecast)


dmd_err, persist_err = [], []
for i in test_idx:
    curve = curves[i]
    true_future = curve[SPLIT:]
    dmd_pred = dmd_fit_forecast(curve, SPLIT, rank=4)
    persist_pred = np.full(len(true_future), curve[SPLIT - 1])  # naive: "stays the same"

    dmd_err.append(np.mean(np.abs(dmd_pred - true_future)))
    persist_err.append(np.mean(np.abs(persist_pred - true_future)))

dmd_err = np.array(dmd_err)
persist_err = np.array(persist_err)
win_rate = float((dmd_err < persist_err).mean())

print(f"  Forecasting steps [{SPLIT}:{n_steps}) from steps [0:{SPLIT}) on {len(test_idx)} patterns")
print(f"  DMD-forecast MAE        : {dmd_err.mean():.4f}")
print(f"  Persistence-baseline MAE: {persist_err.mean():.4f}")
print(f"  DMD beats persistence on {win_rate*100:.1f}% of test patterns")
if dmd_err.mean() < persist_err.mean():
    print("  -> DMD forecasting beats the naive baseline on average. This is a genuinely")
    print("     different, still-open use of DMD (predictive, not diagnostic) worth")
    print("     developing further -- e.g. compare against your CNN's own ppf_steps")
    print("     predictions on these same patterns as the real benchmark to beat.")
else:
    print("  -> DMD forecasting does NOT beat naive persistence here. Combined with the")
    print("     diagnostic-role result (redundant with MC sigma, no independent signal),")
    print("     this would mean DMD isn't earning its complexity in either role on this")
    print("     dataset -- reasonable to deprioritize DMD entirely and focus effort on")
    print("     functional PCA / mode-based curve prediction instead (Part A), which is")
    print("     simpler, already shown compressible, and trains as part of the CNN itself.")

# CNN comparison, if available
if has(MODEL_FILE) and has(CONFIG_FILE):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow import keras

    @tf.keras.utils.register_keras_serializable()
    class ConvResBlock(layers.Layer):
        def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
            super().__init__(**kwargs)
            self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
            self.bn1 = layers.BatchNormalization()
            self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
            self.bn2 = layers.BatchNormalization()
            self.proj = None
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

    print("\n  [LOAD] cnn_v9_model.keras for a head-to-head comparison ...")
    model = keras.models.load_model(MODEL_FILE, compile=False)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    ym_mean, ym_scale = np.array(cfg['ym_scaler_mean'], np.float32), np.array(cfg['ym_scaler_scale'], np.float32)
    step_lo, step_hi = cfg['IDX_PPF_STEPS_START'], cfg['IDX_PPF_STEPS_END']

    GRID_LAYOUT = np.array([
        [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29], [30, -1, -1, -1, -1, -1],
    ], dtype=np.int32)

    def flat_to_grid(flat):
        g = np.zeros((flat.shape[0], 6, 6), dtype=np.int32)
        pi = 0
        for r in range(6):
            for c in range(6):
                if GRID_LAYOUT[r, c] >= 0:
                    g[:, r, c] = flat[:, pi]; pi += 1
        return g

    X_flat = df.iloc[test_idx][load_cols].values.astype(np.int32)
    Xg = tf.constant(flat_to_grid(X_flat), dtype=tf.int32)
    y = model(Xg, training=False).numpy()
    cnn_steps_pred = y[:, step_lo:step_hi] * ym_scale[step_lo:step_hi] + ym_mean[step_lo:step_hi]
    n_cnn_steps = cnn_steps_pred.shape[1]
    # align to the same [SPLIT:n_steps) window if step counts match; otherwise
    # just report on the CNN's own native step grid for a rough comparison
    if n_cnn_steps == n_steps:
        cnn_future_pred = cnn_steps_pred[:, SPLIT:]
        true_future_all = curves[test_idx][:, SPLIT:]
        cnn_err = np.mean(np.abs(cnn_future_pred - true_future_all), axis=1)
        print(f"  CNN ppf_steps MAE on the same forecast window: {cnn_err.mean():.4f}")
        print(f"  (DMD-forecast MAE was {dmd_err.mean():.4f}, persistence was {persist_err.mean():.4f})")
        print("  Remember DMD only saw the first half of the curve; the CNN sees the full")
        print("  loading pattern directly and was trained end-to-end on this exact target --")
        print("  it should generally win. The interesting number is whether DMD-forecast gets")
        print("  CLOSE using much less information (no CNN training, no loading pattern at")
        print("  all beyond the first-half curve shape) -- that's the actual case for using it")
        print("  as a fast, model-free sanity check on CNN curve predictions.")
    else:
        print(f"  [NOTE] CNN has {n_cnn_steps} ppf_steps vs dataset's {n_steps} burnup steps -- "
              f"skipping direct window alignment, counts don't match.")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.hist(dmd_err, bins=30, alpha=0.6, label='DMD forecast', color='#1B4FBF')
ax.hist(persist_err, bins=30, alpha=0.6, label='Persistence baseline', color='#D62728')
ax.set_xlabel('Forecast MAE'); ax.set_ylabel('Count')
ax.set_title(f'DMD Forecast vs Persistence Baseline\nDMD wins on {win_rate*100:.0f}% of patterns')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_dmd_forecast.png', dpi=150)
print(f"\n[SAVED] {OUT_PREFIX}_dmd_forecast.png")
