"""
=============================================================================
04_dmd_signal_validation.py
=============================================================================
Follow-up to 03_dmd_vs_cnn_ppf_dynamics.py.

That script found corr(DMD reconstruction error, sensitivity-weighted
novelty) = r=0.049 on ~800 patterns — essentially zero, but "essentially
zero" needs to be checked, not assumed. This script runs the checks that
decide whether that number is:
  (a) a real (if weak) independent signal worth adding to qica_v11's AL gate, or
  (b) noise around zero that happens to read as 0.049 on this sample.

FIVE TESTS, IN THE ORDER THEY SHOULD CHANGE YOUR DECISION:

  TEST 1 — Permutation test
      Shuffle the novelty scores (fixed DMD error) many times, build the
      null distribution of r, get a two-sided p-value for the observed r.

  TEST 2 — Bootstrap confidence interval
      Resample patterns with replacement, recompute r each time. If the
      95% CI straddles 0, you can't even trust the sign.

  TEST 3 — Split-half stability
      Repeatedly split the sample in half, compute r on each half
      independently. Bouncing sign/magnitude = noise, not signal.

  TEST 4 — Direct signal test (the one that actually matters)
      Correlate DMD error against MC-dropout sigma AND against the CNN's
      real prediction error directly — not just against the novelty proxy.
      If DMD error tracks these better than it tracks novelty, that's a
      real case for adding it as an AL term. If it doesn't track any of
      them, it's not "complementary" — it's just noisy.

  TEST 5 — Outlier inspection
      Pull the highest-DMD-error patterns and compute concrete structural
      diagnostics (assembly-type diversity, same-type neighbor clustering,
      distance from mean per-position type frequency) so you can judge
      by eye whether these are physically unusual patterns or just noise
      in the DMD fit (e.g. a near-degenerate eigenvalue for that pattern).

Needs (same dir): ml_dataset_constrained.csv, cnn_v9_model.keras,
                   cnn_v9_config.json, train_type_freq_v9.npy
Run:  python 04_dmd_signal_validation.py
=============================================================================
"""

import os, json, warnings
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

np.random.seed(42)

# =============================================================================
# TOGGLES
# =============================================================================
DATA_CSV    = 'ml_dataset_constrained.csv'
MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'

N_SAMPLE_PATTERNS = 800
DMD_RANK          = 4
MC_SAMPLES        = 30

N_PERMUTATIONS = 5000
N_BOOTSTRAP    = 5000
N_SPLIT_HALVES = 1000
TOP_N_OUTLIERS = 20

OUT_PREFIX = 'dmd_validation'


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
# SECTION 1 — LOAD DATA + REBUILD PER-PATTERN PPF MATRICES (same as script 03)
# =============================================================================
print("[LOAD] dataset ...")

df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')

ppf_steps_avail_raw = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_assembs_avail   = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))
ppf_steps_avail = [s for s in ppf_steps_avail_raw
                   if len([c for c in df.columns if c.startswith(f"ppf_s{s}_")]) == len(ppf_assembs_avail)]

n_steps = len(ppf_steps_avail)
n_assem = len(ppf_assembs_avail)
print(f"  {len(df)} patterns | ppf matrix per pattern = {n_assem} assemblies x {n_steps} steps")

if N_SAMPLE_PATTERNS is not None and N_SAMPLE_PATTERNS < len(df):
    sample_idx = np.random.choice(len(df), N_SAMPLE_PATTERNS, replace=False)
else:
    sample_idx = np.arange(len(df))
df_s = df.iloc[sample_idx].reset_index(drop=True)
print(f"  Using {len(df_s)} sampled patterns\n")

ppf_tensor = np.zeros((len(df_s), n_assem, n_steps), dtype=np.float32)
for si, s in enumerate(ppf_steps_avail):
    cols = [f'ppf_s{s}_a{a}' for a in ppf_assembs_avail if f'ppf_s{s}_a{a}' in df.columns]
    ppf_tensor[:, :, si] = df_s[cols].values.astype(np.float32)

load_cols = [f'loading_{i}' for i in range(31)]
X_flat = df_s[load_cols].values.astype(np.int32) if all(c in df_s.columns for c in load_cols) else None
ppf_max_true = ppf_tensor.max(axis=(1, 2))


# =============================================================================
# SECTION 2 — DMD CORE (identical to script 03)
# =============================================================================

def exact_dmd(X, rank):
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
    powers = np.array([eigvals ** t for t in range(n_time)]).T
    return (Phi @ (b[:, None] * powers)).real


def rel_err(X_true, X_hat):
    return float(np.linalg.norm(X_true - X_hat) / (np.linalg.norm(X_true) + 1e-9))


# =============================================================================
# SECTION 3 — RECOMPUTE DMD ERROR + NOVELTY (the two series from script 03)
# =============================================================================
print("[RECOMPUTE] DMD reconstruction error + rarity-novelty score ...")

if X_flat is None:
    raise SystemExit("No loading_* columns found — cannot compute novelty/adjacency diagnostics.")
if not os.path.exists(FREQ_FILE):
    raise SystemExit(f"{FREQ_FILE} not found — run cnn_v9.py first.")

type_freq = np.load(FREQ_FILE).astype(np.float32)   # (N_POS, N_TYPES)

dmd_err_list, novelty_list, keep_idx = [], [], []
for i in range(len(df_s)):
    X = ppf_tensor[i]
    try:
        eigvals, Phi, b = exact_dmd(X, DMD_RANK)
    except np.linalg.LinAlgError:
        continue
    X_hat = dmd_reconstruct(eigvals, Phi, b, n_steps)
    dmd_err_list.append(rel_err(X, X_hat))

    nov = 0.0
    for p in range(min(31, X_flat.shape[1])):
        t = int(X_flat[i, p])
        if 1 <= t <= type_freq.shape[1]:
            nov += -np.log(max(float(type_freq[p, t - 1]), 1e-4))
    novelty_list.append(nov)
    keep_idx.append(i)

dmd_err  = np.array(dmd_err_list)
novelty  = np.array(novelty_list)
keep_idx = np.array(keep_idx)
n_valid  = len(keep_idx)
observed_r = float(np.corrcoef(dmd_err, novelty)[0, 1])
print(f"  n valid patterns: {n_valid}")
print(f"  observed r(dmd_err, novelty) = {observed_r:.4f}\n")


# =============================================================================
# SECTION 4 — MC-DROPOUT SIGMA + CNN PREDICTION ERROR (for TEST 4)
# =============================================================================
print("[MODEL] Loading cnn_v9_model.keras for MC-dropout sigma + prediction error ...")

mc_sigma = None
cnn_abs_err = None
if os.path.exists(MODEL_FILE) and os.path.exists(CONFIG_FILE):
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
    Xg = Xg[keep_idx]

    Xg_tf = tf.constant(Xg, dtype=tf.int32)

    # Deterministic prediction -> real error vs ground truth PPF
    y_det = model(Xg_tf, training=False).numpy()
    ppf_pred_det = y_det[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]
    cnn_abs_err = np.abs(ppf_pred_det - ppf_max_true[keep_idx])

    # MC dropout -> sigma
    mc_stack = np.stack([model(Xg_tf, training=True).numpy()[:, IDX_PPF] for _ in range(MC_SAMPLES)])
    mc_sigma = mc_stack.std(axis=0) * ym_scale[IDX_PPF]

    print(f"  mean CNN |pred error| on ppf_max : {cnn_abs_err.mean():.4f}")
    print(f"  mean MC-dropout sigma            : {mc_sigma.mean():.4f}\n")
else:
    print("  [WARN] model/config not found — TEST 4 will skip the direct-error comparisons.\n")


# =============================================================================
# TEST 1 — PERMUTATION TEST
# =============================================================================
print("=" * 70)
print("TEST 1 — Permutation test (is r=%.4f distinguishable from 0?)" % observed_r)
print("=" * 70)

null_rs = np.empty(N_PERMUTATIONS)
rng = np.random.default_rng(42)
for k in range(N_PERMUTATIONS):
    shuffled = rng.permutation(novelty)
    null_rs[k] = np.corrcoef(dmd_err, shuffled)[0, 1]

p_value = float((np.abs(null_rs) >= np.abs(observed_r)).mean())
print(f"  Null distribution: mean={null_rs.mean():.4f}  std={null_rs.std():.4f}")
print(f"  Two-sided p-value for observed r={observed_r:.4f}: p={p_value:.4f}")
print(f"  -> {'SIGNIFICANT at p<0.05 (unlikely to be pure noise)' if p_value < 0.05 else 'NOT significant at p<0.05 (consistent with pure noise)'}\n")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(null_rs, bins=60, color='#AAAAAA', alpha=0.8, label='Null distribution (shuffled novelty)')
ax.axvline(observed_r, color='#D62728', lw=2, label=f'Observed r={observed_r:.4f}')
ax.axvline(-observed_r, color='#D62728', lw=1, ls=':', alpha=0.6)
ax.set_xlabel('Correlation coefficient r'); ax.set_ylabel('Count')
ax.set_title(f'Permutation Test\np={p_value:.4f}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_permutation_test.png', dpi=150)
print(f"[SAVED] {OUT_PREFIX}_permutation_test.png\n")


# =============================================================================
# TEST 2 — BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================
print("=" * 70)
print("TEST 2 — Bootstrap 95% CI on r")
print("=" * 70)

boot_rs = np.empty(N_BOOTSTRAP)
for k in range(N_BOOTSTRAP):
    idx = rng.integers(0, n_valid, n_valid)
    boot_rs[k] = np.corrcoef(dmd_err[idx], novelty[idx])[0, 1]

ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])
print(f"  Bootstrap mean r : {boot_rs.mean():.4f}")
print(f"  95% CI           : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  -> {'CI excludes 0: sign/magnitude are somewhat trustworthy' if ci_lo * ci_hi > 0 else 'CI STRADDLES 0: cannot even trust the sign — treat as noise'}\n")

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(boot_rs, bins=60, color='#1B4FBF', alpha=0.8)
ax.axvline(0, color='black', lw=1, ls='--', label='r=0')
ax.axvline(ci_lo, color='#F5A623', lw=2, label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
ax.axvline(ci_hi, color='#F5A623', lw=2)
ax.set_xlabel('Bootstrap correlation r'); ax.set_ylabel('Count')
ax.set_title('Bootstrap Distribution of r')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_bootstrap_ci.png', dpi=150)
print(f"[SAVED] {OUT_PREFIX}_bootstrap_ci.png\n")


# =============================================================================
# TEST 3 — SPLIT-HALF STABILITY
# =============================================================================
print("=" * 70)
print("TEST 3 — Split-half stability")
print("=" * 70)

half_r1 = np.empty(N_SPLIT_HALVES)
half_r2 = np.empty(N_SPLIT_HALVES)
for k in range(N_SPLIT_HALVES):
    perm = rng.permutation(n_valid)
    a, b_ = perm[: n_valid // 2], perm[n_valid // 2:]
    half_r1[k] = np.corrcoef(dmd_err[a], novelty[a])[0, 1]
    half_r2[k] = np.corrcoef(dmd_err[b_], novelty[b_])[0, 1]

sign_agree = float((np.sign(half_r1) == np.sign(half_r2)).mean())
print(f"  Mean r (half A): {half_r1.mean():.4f}  std: {half_r1.std():.4f}")
print(f"  Mean r (half B): {half_r2.mean():.4f}  std: {half_r2.std():.4f}")
print(f"  Fraction of splits where both halves AGREE in sign: {sign_agree*100:.1f}%")
print(f"  -> {'Reasonably stable sign' if sign_agree > 0.75 else 'Sign is unstable across splits — strong indicator of noise'}\n")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(half_r1, half_r2, alpha=0.15, s=8, color='#2CA02C')
lim = max(np.abs(half_r1).max(), np.abs(half_r2).max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
ax.axhline(0, color='grey', lw=0.8); ax.axvline(0, color='grey', lw=0.8)
ax.set_xlabel('r on random half A'); ax.set_ylabel('r on random half B')
ax.set_title(f'Split-Half Agreement\n{sign_agree*100:.1f}% same-sign')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_split_half.png', dpi=150)
print(f"[SAVED] {OUT_PREFIX}_split_half.png\n")


# =============================================================================
# TEST 4 — DIRECT SIGNAL TEST: does DMD error track REAL uncertainty/error?
# =============================================================================
print("=" * 70)
print("TEST 4 — DMD error vs MC-dropout sigma / actual CNN error (not just novelty proxy)")
print("=" * 70)

if mc_sigma is not None and cnn_abs_err is not None:
    r_sigma = float(np.corrcoef(dmd_err, mc_sigma)[0, 1])
    r_error = float(np.corrcoef(dmd_err, cnn_abs_err)[0, 1])
    print(f"  corr(DMD error, MC-dropout sigma)      : {r_sigma:.4f}")
    print(f"  corr(DMD error, actual CNN |pred err|) : {r_error:.4f}")
    print(f"  corr(DMD error, novelty proxy)         : {observed_r:.4f}  (for reference)")
    best = max(abs(r_sigma), abs(r_error), abs(observed_r))
    if best == abs(observed_r):
        print("  -> DMD error tracks the novelty proxy best (and even that was weak/noisy).")
        print("     It does NOT look like it's catching real model uncertainty here.")
    else:
        which = "MC-dropout sigma" if best == abs(r_sigma) else "actual CNN prediction error"
        print(f"  -> DMD error tracks {which} more strongly than it tracks novelty.")
        print("     This is a more meaningful basis for adding it to the AL gate than the novelty corr alone.")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(mc_sigma, dmd_err, alpha=0.25, s=8, color='#9467BD')
    axes[0].set_xlabel('MC-dropout sigma (ppf_max)'); axes[0].set_ylabel('DMD reconstruction error')
    axes[0].set_title(f'DMD err vs MC sigma\nr={r_sigma:.3f}')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(cnn_abs_err, dmd_err, alpha=0.25, s=8, color='#D62728')
    axes[1].set_xlabel('Actual CNN |prediction error| (ppf_max)'); axes[1].set_ylabel('DMD reconstruction error')
    axes[1].set_title(f'DMD err vs actual CNN error\nr={r_error:.3f}')
    axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_direct_signal_test.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_direct_signal_test.png\n")
else:
    print("  SKIPPED — model/config unavailable.\n")


# =============================================================================
# TEST 5 — OUTLIER INSPECTION: are the high-DMD-error patterns physically odd?
# =============================================================================
print("=" * 70)
print(f"TEST 5 — Structural diagnostics on top-{TOP_N_OUTLIERS} highest-DMD-error patterns")
print("=" * 70)

N_POS = X_flat.shape[1]
N_TYPES = type_freq.shape[1]

# neighbor map from GRID_LAYOUT (orthogonal neighbors within active cells)
neighbor_map = {p: [] for p in range(N_POS)}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    GRID_LAYOUT = np.array(cfg['GRID_LAYOUT'])
    pos_of_rc, rc_of_pos = {}, {}
    pi = 0
    for r in range(GRID_LAYOUT.shape[0]):
        for c in range(GRID_LAYOUT.shape[1]):
            if GRID_LAYOUT[r, c] >= 0:
                pos_of_rc[(r, c)] = pi
                rc_of_pos[pi] = (r, c)
                pi += 1
    for p, (r, c) in rc_of_pos.items():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in pos_of_rc:
                neighbor_map[p].append(pos_of_rc[nb])

def pattern_diagnostics(pattern):
    """pattern: (N_POS,) int array of assembly type labels."""
    diversity = len(np.unique(pattern))
    # avg -log(freq) rarity, same as novelty but reported alone for context
    rarity = np.mean([-np.log(max(float(type_freq[p, int(pattern[p]) - 1]), 1e-4))
                       for p in range(N_POS) if 1 <= pattern[p] <= N_TYPES])
    # same-type-neighbor fraction: how "clustered" is this pattern spatially
    same_type_frac = []
    for p in range(N_POS):
        nbrs = neighbor_map.get(p, [])
        if nbrs:
            same_type_frac.append(np.mean([pattern[p] == pattern[q] for q in nbrs]))
    clustering = float(np.mean(same_type_frac)) if same_type_frac else np.nan
    return diversity, rarity, clustering

order = np.argsort(dmd_err)[::-1][:TOP_N_OUTLIERS]
rows = []
for rank, local_i in enumerate(order):
    global_i = keep_idx[local_i]
    pattern = X_flat[global_i]
    diversity, rarity, clustering = pattern_diagnostics(pattern)
    rows.append({
        'rank': rank + 1,
        'dataset_idx': int(sample_idx[global_i]),
        'dmd_recon_error': float(dmd_err[local_i]),
        'novelty_score': float(novelty[local_i]),
        'mc_sigma_ppf': float(mc_sigma[local_i]) if mc_sigma is not None else np.nan,
        'cnn_abs_err_ppf': float(cnn_abs_err[local_i]) if cnn_abs_err is not None else np.nan,
        'type_diversity': diversity,
        'avg_rarity': float(rarity),
        'same_type_neighbor_frac': clustering,
        **{f'pos_{p}': int(pattern[p]) for p in range(N_POS)},
    })

outlier_df = pd.DataFrame(rows)
outlier_df.to_csv(f'{OUT_PREFIX}_top_outliers.csv', index=False)

print(outlier_df[['rank', 'dataset_idx', 'dmd_recon_error', 'novelty_score',
                   'mc_sigma_ppf', 'cnn_abs_err_ppf', 'type_diversity',
                   'same_type_neighbor_frac']].to_string(index=False))
print(f"\n[SAVED] {OUT_PREFIX}_top_outliers.csv")
print("  -> Eyeball 'type_diversity' and 'same_type_neighbor_frac' against the")
print("     dataset-wide averages below. If outliers cluster at extreme diversity")
print("     or extreme clustering, DMD error is catching genuinely unusual")
print("     assembly arrangements. If they look like typical patterns, the high")
print("     DMD error is more likely a numerical artifact (e.g. a near-repeated")
print("     eigenvalue making that pattern's fit ill-conditioned) than signal.\n")

all_diversity, all_clustering = [], []
for local_i in range(n_valid):
    d, _, cl = pattern_diagnostics(X_flat[keep_idx[local_i]])
    all_diversity.append(d); all_clustering.append(cl)
print(f"  Dataset-wide avg type_diversity        : {np.mean(all_diversity):.2f}")
print(f"  Dataset-wide avg same_type_neighbor_frac: {np.nanmean(all_clustering):.3f}")
print(f"  Outlier-set avg type_diversity          : {outlier_df['type_diversity'].mean():.2f}")
print(f"  Outlier-set avg same_type_neighbor_frac  : {outlier_df['same_type_neighbor_frac'].mean():.3f}\n")


# =============================================================================
# FINAL DECISION SUMMARY
# =============================================================================
print("=" * 70)
print("DECISION SUMMARY — should DMD error go into qica_v11's AL gate?")
print("=" * 70)
print(f"  TEST 1 permutation p-value        : {p_value:.4f}  ({'significant' if p_value < 0.05 else 'NOT significant'})")
print(f"  TEST 2 bootstrap 95% CI            : [{ci_lo:.4f}, {ci_hi:.4f}]  ({'excludes 0' if ci_lo*ci_hi>0 else 'straddles 0'})")
print(f"  TEST 3 split-half sign agreement   : {sign_agree*100:.1f}%")
if mc_sigma is not None:
    print(f"  TEST 4 corr with MC sigma / CNN err: {r_sigma:.4f} / {r_error:.4f}  (vs novelty {observed_r:.4f})")
print(f"  TEST 5 outlier diversity/clustering : see CSV — compare to dataset-wide averages above")
print()
print("  Rule of thumb: only add DMD error as an independent AL term if AT LEAST")
print("  TWO of (significant permutation p, CI excluding 0, >75% split-half sign")
print("  agreement, stronger corr with real error/sigma than with novelty) hold.")
print("  Otherwise, r=0.049 was noise around zero, and 'uncorrelated with novelty'")
print("  just meant 'uncorrelated with anything' — not complementary signal.")
print("=" * 70)