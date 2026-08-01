"""
=============================================================================
03_dmd_vs_cnn_ppf_dynamics.py
=============================================================================
Tests whether Dynamic Mode Decomposition (DMD) of the per-pattern,
per-assembly, per-burnup-step PPF matrix adds anything over what
cnn_v9.py's ppf_steps head already does — and whether DMD reconstruction
error is a useful complementary AL signal alongside H_hist / MC sigma.

WHY THIS IS A FAIR DMD APPLICATION (not a stretch):
  ml_dataset_constrained.csv has ppf_s{step}_a{assembly} columns (see
  cnn-v9.py Section 3). For ONE loading pattern that's a genuine
  (n_assemblies x n_steps) snapshot matrix evolving over burnup — the
  textbook DMD setup. No reinterpretation needed.

  Caveat to expect, not explain away: burnable-absorber burnout early in
  cycle is a nonlinear regime shift. A single global linear DMD fit per
  pattern will nail the late-cycle linear decay and do worse on the first
  few steps. Watch WHERE reconstruction error concentrates before
  concluding DMD "works."

FOUR ARMS (toggle at top, run repeatedly, results append to one CSV):
  ARM "cnn_raw"       : baseline. cnn_v9_model.keras's own ppf_max /
                         ppf_steps predictions, no DMD involved.
  ARM "dmd_recon"     : rank-sweep DMD reconstruction fidelity of the real
                         assembly x step PPF matrices — is the depletion
                         dynamics actually low-rank/linear enough for this
                         to be worth doing at all? (sanity gate before you
                         invest in wiring DMD into the CNN itself)
  ARM "dmd_features"  : compress each pattern's PPF movie to DMD
                         eigenvalues+amplitudes, train a tiny regressor on
                         those features -> ppf_max, compare against cnn_raw.
  ARM "dmd_al_signal" : correlate DMD reconstruction error against your
                         existing MC-dropout sigma and H_hist-style
                         sensitivity-novelty score — is it flagging the
                         SAME patterns (redundant) or DIFFERENT ones
                         (complementary, worth adding to qica_v11's AL gate)?

  A no-dynamics control (static PCA/POD on the same matrices, same rank)
  is included alongside dmd_recon so you can see whether the *temporal
  operator* is earning its keep or whether plain rank reduction gets you
  the same fidelity for free.

Run:  python 03_dmd_vs_cnn_ppf_dynamics.py
Needs (same dir): ml_dataset_constrained.csv, cnn_v9_model.keras,
                   cnn_v9_config.json
=============================================================================
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#np.random.seed(42)
for seed in [1,2,3,4,5]:
    np.random.seed(seed)

# =============================================================================
# TOGGLES
# =============================================================================
DATA_CSV       = 'ml_dataset_constrained.csv'
MODEL_FILE     = 'cnn_v9_model.keras'
CONFIG_FILE    = 'cnn_v9_config.json'

N_SAMPLE_PATTERNS = 5000          # subsample for speed; set to None for full 10k
DMD_RANK_SWEEP     = [2, 3, 4, 6, 8, 12]
DMD_RANK_FINAL      = 4          # rank used for the feature/AL arms below
RUN_ARM_CNN_RAW      = True
RUN_ARM_DMD_RECON    = True
RUN_ARM_DMD_FEATURES = True
RUN_ARM_DMD_AL_SIGNAL = True

OUT_PREFIX = 'dmd_vs_cnn'


# =============================================================================
# SECTION 0 — MATCH cnn_v9.py's ConvResBlock so the model loads
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
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


# =============================================================================
# SECTION 1 — LOAD DATA + BUILD PER-PATTERN (assembly x step) MATRICES
# =============================================================================
print("[LOAD] dataset ...")

df = pd.read_csv(
    DATA_CSV,
    skiprows=1,
    engine='python',
    on_bad_lines='skip'
)

# Find available PPF steps and assemblies
ppf_steps_avail_raw = sorted(
    set(
        int(c.split('_')[1][1:])
        for c in df.columns
        if c.startswith('ppf_')
    )
)

ppf_assembs_avail = sorted(
    set(
        int(c.split('_')[2][1:])
        for c in df.columns
        if c.startswith('ppf_')
    )
)

# Remove incomplete burnup steps
ppf_steps_avail = [
    s for s in ppf_steps_avail_raw
    if len([
        c for c in df.columns
        if c.startswith(f"ppf_s{s}_")
    ]) == len(ppf_assembs_avail)
]

print(
    f"Using complete burnup steps: {ppf_steps_avail}"
)

n_steps = len(ppf_steps_avail)
n_assem = len(ppf_assembs_avail)

print(
    f"  {len(df)} patterns | "
    f"ppf matrix per pattern = "
    f"{n_assem} assemblies x {n_steps} steps"
)

if N_SAMPLE_PATTERNS is not None and N_SAMPLE_PATTERNS < len(df):
    sample_idx = np.random.choice(len(df), N_SAMPLE_PATTERNS, replace=False)
else:
    sample_idx = np.arange(len(df))
df_s = df.iloc[sample_idx].reset_index(drop=True)
print(f"  Using {len(df_s)} sampled patterns\n")

# ppf_tensor[i, a, s] = PPF of assembly a at step s for pattern i
ppf_tensor = np.zeros((len(df_s), n_assem, n_steps), dtype=np.float32)
for si, s in enumerate(ppf_steps_avail):
    cols = [f'ppf_s{s}_a{a}' for a in ppf_assembs_avail if f'ppf_s{s}_a{a}' in df.columns]
    ppf_tensor[:, :, si] = df_s[cols].values.astype(np.float32)

load_cols = [f'loading_{i}' for i in range(31)]
X_flat = df_s[load_cols].values.astype(np.int32) if all(c in df_s.columns for c in load_cols) else None
ppf_max_true = ppf_tensor.max(axis=(1, 2))  # sanity target for arm comparisons


# =============================================================================
# SECTION 2 — DMD CORE (exact DMD via SVD) + POD/PCA control
# =============================================================================

def exact_dmd(X, rank):
    """X: (n_state, n_time). Returns eigvals, modes Phi, initial amplitudes b."""
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, len(S))
    Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
    Atilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / Sr)
    eigvals, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vhr.conj().T @ np.diag(1.0 / Sr) @ W
    b, *_ = np.linalg.lstsq(Phi, X[:, 0], rcond=None)
    return eigvals, Phi, b


def dmd_reconstruct(eigvals, Phi, b, n_time):
    powers = np.array([eigvals ** t for t in range(n_time)]).T   # (r, n_time)
    return (Phi @ (b[:, None] * powers)).real                     # (n_state, n_time)


def pod_reconstruct(X, rank):
    """Static control: no temporal operator, just rank-r SVD reconstruction."""
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(S))
    return U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]


def rel_err(X_true, X_hat):
    return float(np.linalg.norm(X_true - X_hat) / (np.linalg.norm(X_true) + 1e-9))


# =============================================================================
# ARM: cnn_raw  — existing model's own ppf predictions, for reference
# =============================================================================

def arm_cnn_raw():
    if X_flat is None or not os.path.exists(MODEL_FILE):
        print("[ARM cnn_raw] SKIPPED (loading_* cols or model file missing)")
        return None
    print("[ARM cnn_raw] Loading cnn_v9_model.keras ...")
    model = keras.models.load_model(MODEL_FILE, compile=False)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    GRID_LAYOUT = np.array(cfg['GRID_LAYOUT'])
    ym_mean = np.array(cfg['ym_scaler_mean'], dtype=np.float32)
    ym_scale = np.array(cfg['ym_scaler_scale'], dtype=np.float32)
    IDX_PPF = cfg['IDX_PPF_MAX']

    Xg = np.zeros((len(X_flat), GRID_LAYOUT.shape[0], GRID_LAYOUT.shape[1]), dtype=np.int32)
    pi = 0
    for r in range(GRID_LAYOUT.shape[0]):
        for c in range(GRID_LAYOUT.shape[1]):
            if GRID_LAYOUT[r, c] >= 0:
                Xg[:, r, c] = X_flat[:, pi]; pi += 1

    y_sc = model(tf.constant(Xg, dtype=tf.int32), training=False).numpy()
    ppf_pred = y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]

    mae = mean_absolute_error(ppf_max_true, ppf_pred)
    r2 = r2_score(ppf_max_true, ppf_pred)
    print(f"  CNN ppf_max: MAE={mae:.4f}  R2={r2:.4f}\n")
    return dict(mae=mae, r2=r2, pred=ppf_pred)


# =============================================================================
# ARM: dmd_recon — rank sweep, DMD vs POD reconstruction fidelity
# =============================================================================

def arm_dmd_recon():
    print("[ARM dmd_recon] Rank sweep: DMD vs static-POD reconstruction error ...")
    n_check = min(150, len(df_s))
    results = []
    for rank in DMD_RANK_SWEEP:
        dmd_errs, pod_errs = [], []
        for i in range(n_check):
            X = ppf_tensor[i]  # (n_assem, n_steps)
            try:
                eigvals, Phi, b = exact_dmd(X, rank)
                X_hat = dmd_reconstruct(eigvals, Phi, b, n_steps)
                dmd_errs.append(rel_err(X, X_hat))
            except np.linalg.LinAlgError:
                continue
            X_pod = pod_reconstruct(X, rank)
            pod_errs.append(rel_err(X, X_pod))
        results.append(dict(rank=rank,
                             dmd_relerr_mean=float(np.mean(dmd_errs)),
                             dmd_relerr_std=float(np.std(dmd_errs)),
                             pod_relerr_mean=float(np.mean(pod_errs)),
                             n_features_dmd=4 * rank,   # Re/Im eigval + |b|/angle(b)
                             n_features_raw=n_assem * n_steps))
        print(f"  rank={rank:2d}  DMD relerr={results[-1]['dmd_relerr_mean']:.4f}"
              f"  POD relerr={results[-1]['pod_relerr_mean']:.4f}"
              f"  (compression {results[-1]['n_features_raw']}->{results[-1]['n_features_dmd']})")

    res_df = pd.DataFrame(results)
    res_df.to_csv(f'{OUT_PREFIX}_rank_sweep.csv', index=False)

    # where does error concentrate? (first vs late steps), at DMD_RANK_FINAL
    early_errs, late_errs = [], []
    cut = max(3, n_steps // 4)
    for i in range(n_check):
        X = ppf_tensor[i]
        try:
            eigvals, Phi, b = exact_dmd(X, DMD_RANK_FINAL)
        except np.linalg.LinAlgError:
            continue
        X_hat = dmd_reconstruct(eigvals, Phi, b, n_steps)
        early_errs.append(rel_err(X[:, :cut], X_hat[:, :cut]))
        late_errs.append(rel_err(X[:, cut:], X_hat[:, cut:]))
    print(f"  At rank={DMD_RANK_FINAL}: early-cycle relerr={np.mean(early_errs):.4f} "
          f"vs late-cycle relerr={np.mean(late_errs):.4f}"
          f"  ({'confirms early BA nonlinearity is harder to fit' if np.mean(early_errs) > np.mean(late_errs) else 'no early/late split observed'})\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(res_df['rank'], res_df['dmd_relerr_mean'], 'o-', color='#1B4FBF', label='DMD (dynamic)')
    ax.plot(res_df['rank'], res_df['pod_relerr_mean'], 's--', color='#F5A623', label='POD (static control)')
    ax.set_xlabel('Rank'); ax.set_ylabel('Mean relative reconstruction error')
    ax.set_title('DMD vs static POD: is the temporal operator earning its keep?')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_rank_sweep.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_rank_sweep.csv / .png\n")
    return res_df


# =============================================================================
# ARM: dmd_features — DMD-compressed features -> ppf_max regressor vs cnn_raw
# =============================================================================

def arm_dmd_features(cnn_result):
    print(f"[ARM dmd_features] Extracting rank-{DMD_RANK_FINAL} DMD features per pattern ...")
    feats, keep_idx = [], []
    for i in range(len(df_s)):
        X = ppf_tensor[i]
        try:
            eigvals, Phi, b = exact_dmd(X, DMD_RANK_FINAL)
        except np.linalg.LinAlgError:
            continue
        f = np.concatenate([eigvals.real, eigvals.imag, np.abs(b), np.angle(b)])
        if len(f) == 4 * DMD_RANK_FINAL:
            feats.append(f); keep_idx.append(i)
    feats = np.array(feats, dtype=np.float32)
    y = ppf_max_true[keep_idx]
    print(f"  Feature matrix: {feats.shape} (vs raw {n_assem*n_steps} PPF values/pattern)")

    Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0).fit(Xtr, ytr)
    pred = reg.predict(Xte)
    mae, r2 = mean_absolute_error(yte, pred), r2_score(yte, pred)
    print(f"  Ridge-on-DMD-features ppf_max: MAE={mae:.4f}  R2={r2:.4f}")
    if cnn_result is not None:
        print(f"  (cnn_raw for comparison        : MAE={cnn_result['mae']:.4f}  R2={cnn_result['r2']:.4f})")
        print(f"  -> {'DMD features are competitive with only ' + str(feats.shape[1]) + ' numbers/pattern' if r2 > cnn_result['r2']*0.8 else 'DMD features alone lag the CNN — treat as an AUXILIARY signal, not a replacement'}")
    print()
    return dict(mae=mae, r2=r2, n_features=feats.shape[1])


# =============================================================================
# ARM: dmd_al_signal — is DMD recon error redundant with MC-sigma / H_hist?
# =============================================================================

def arm_dmd_al_signal():
    print("[ARM dmd_al_signal] DMD reconstruction error vs population sensitivity-novelty ...")
    # Proxy for sensitivity_novelty (qica_v11's _sensitivity_novelty) using
    # per-position type frequency, since this script runs standalone from QICA.
    if X_flat is None:
        print("  SKIPPED (no loading_* columns found)\n"); return None
    freq_path = 'train_type_freq_v9.npy'
    if not os.path.exists(freq_path):
        print(f"  SKIPPED ({freq_path} not found)\n"); return None
    type_freq = np.load(freq_path).astype(np.float32)

    dmd_err, novelty = [], []
    for i in range(len(df_s)):
        X = ppf_tensor[i]
        try:
            eigvals, Phi, b = exact_dmd(X, DMD_RANK_FINAL)
        except np.linalg.LinAlgError:
            continue
        X_hat = dmd_reconstruct(eigvals, Phi, b, n_steps)
        dmd_err.append(rel_err(X, X_hat))
        nov = 0.0
        for p in range(min(31, X_flat.shape[1])):
            t = int(X_flat[i, p])
            if 1 <= t <= type_freq.shape[1]:
                nov += -np.log(max(float(type_freq[p, t - 1]), 1e-4))
        novelty.append(nov)

    dmd_err, novelty = np.array(dmd_err), np.array(novelty)
    corr = np.corrcoef(dmd_err, novelty)[0, 1]
    print(f"  corr(DMD reconstruction error, position-rarity novelty) = {corr:.3f}")
    print(f"  -> {'largely redundant with your existing novelty term (skip adding it)' if abs(corr) > 0.5 else 'weakly correlated: DMD error is flagging a genuinely different kind of outlier — worth ablating as an added AL term in qica_v11'}\n")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(novelty, dmd_err, alpha=0.3, s=10, color='#9467BD')
    ax.set_xlabel('Sensitivity-weighted rarity novelty (qica_v11 style)')
    ax.set_ylabel('DMD reconstruction relative error')
    ax.set_title(f'DMD error vs novelty  (r={corr:.3f})')
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_al_signal_corr.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_al_signal_corr.png\n")
    return dict(corr=float(corr))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    cnn_res = arm_cnn_raw() if RUN_ARM_CNN_RAW else None
    if RUN_ARM_DMD_RECON:
        arm_dmd_recon()
    if RUN_ARM_DMD_FEATURES:
        arm_dmd_features(cnn_res)
    if RUN_ARM_DMD_AL_SIGNAL:
        arm_dmd_al_signal()

    print("=" * 60)
    print("READ THE RESULTS IN THIS ORDER:")
    print("  1. dmd_recon rank sweep: if DMD relerr barely beats POD at any")
    print("     rank, the depletion dynamics aren't linear enough for DMD to")
    print("     buy you anything over plain PCA -> not worth wiring into CNN.")
    print("  2. dmd_features R2 vs cnn_raw R2: if close, you've got a >10x")
    print("     compressed physics-structured representation worth trying as")
    print("     an auxiliary CNN input/output, not just a diagnostic.")
    print("  3. dmd_al_signal correlation: only add DMD error to qica_v11's")
    print("     AL gate if |corr| is low — otherwise it's double-counting")
    print("     the novelty term you already have.")
    print("=" * 60)