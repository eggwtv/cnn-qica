"""
=============================================================================
tests-for-qica.py  —  QICA Method Ablation Test Suite
=============================================================================
PURPOSE
  Isolate the contribution of each QICA component in a single, reproducible
  run before finalising the production code. Five arms, identical seeds,
  identical budgets — any difference is attributable to the arm flag only.

ARMS (in order):
  1_baseline       σ-only AL  |  no injection  |  —            |  —
  2_h_hist_only    H_hist AL  |  no injection  |  —            |  —
  3_inject_unfixed σ-only AL  |  injection ✓   |  bug present  |  —
  4_h_hist_fixed   H_hist AL  |  injection ✓   |  FIXED        |  —
  5_full_v9        H_hist AL  |  injection ✓   |  FIXED        |  +H_traj

HARD RESET BUG (arm 3 shows it, arm 4/5 fix it):
  _hard_reset_worst_empire() opens with `if len(empires) < 2: return`, but
  the population consistently collapses to a single empire by gen ~50-80 in
  short runs (~125 in full 250-gen runs), so every hard reset silently no-ops.
  FIX: when len(empires)==1, reinit the worst N//2 colonies of that one empire.

H_TRAJ VERDICT (arms 4 vs 5):
  From the v9 full run: CV=0.016, H-σ corr=-0.040  → WEAK discriminator.
  Expected arm 5 ≈ arm 4 in best PPF (H_traj adds nothing).

WHAT TO LOOK FOR IN RESULTS:
  • arm 2 vs arm 1  →  does H_hist alone improve AL candidate quality?
  • arm 3 vs arm 1  →  does injection alone improve search (with broken HR)?
  • arm 4 vs arm 3  →  does fixing the hard reset improve escape from basins?
  • arm 4 vs arm 2  →  does injection matter on top of H_hist?
  • arm 5 vs arm 4  →  does H_traj add anything? (expected: no)

INPUTS:   cnn_v9_model.keras / cnn_v9_config.json / train_type_freq_v9.npy
OUTPUTS:  test_qica_summary.csv, test_qica_history.csv,
          test_qica_al_candidates.csv, test_qica_comparison.png
=============================================================================
"""

"""
=============================================================================
PATCH NOTES (v2) — based on the 5-arm run sent back
=============================================================================
1. HARD RESET TOO DESTRUCTIVE
   Original "fix": once collapsed to a single empire, reinit HALF its
   colonies every time the reset fires. Result: arm4 (fixed) scored WORSE
   than arm3 (bug present, reset never fires) — best_ppf 2.0017 vs 1.8812.
   Fix: HARD_RESET_FRAC shrinks the reinit fraction to 25%, and a
   HARD_RESET_COOLDOWN stops it firing again until the population has had
   time to recover and be judged on its own merits.

2. H_TRAJ COMPUTED EVEN WHEN UNUSED
   trajectory_entropy() runs an SVD per pattern every generation in EVERY
   arm, even arms 1-4 which never read c.h_traj. Confirmed weak anyway
   (CV=0.026-0.039, well under your own 0.05 "real discriminator" bar).
   Fix: evaluate_batch(..., compute_h_traj=...) skips it unless the arm
   flag use_h_traj is True.

3. INJECTION "IMPROVEMENT" METRIC ONLY CHECKED PPF
   Every arm reported 0/N injections "improved" PPF in the next 10 gens —
   but injections target *fitness* (cycle - PPF penalty - sigma penalty),
   not PPF alone, so that diagnostic was too narrow to mean "injection
   failed." Fix: now reports both PPF-window and fitness-window
   improvement so you can tell the difference.

4. NEW ARM 6 — lets you directly test whether the gentler hard reset
   actually closes the arm3-vs-arm4 gap, without touching arms 1-5.
=============================================================================
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(f"TensorFlow {tf.__version__}")
print("tests-for-qica.py  —  QICA Ablation Test Suite\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

MODEL_PATH  = 'cnn_v9_model.keras'     if os.path.exists('cnn_v9_model.keras')     else 'cnn_v4_model.keras'
CONFIG_PATH = 'cnn_v9_config.json'     if os.path.exists('cnn_v9_config.json')     else 'cnn_v4_config.json'
TRUST_PATH  = 'train_type_freq_v9.npy' if os.path.exists('train_type_freq_v9.npy') else 'train_type_freq.npy'

# ── Test budget (reduce TEST_GENS=40 for a quick ~8-min smoke-test) ──────────
TEST_GENS = 80       # generations per arm  — 80 gens ≈ 10-14 min/arm on CPU
TEST_POP  = 60       # population per arm
TEST_MC   = 20       # MC samples per evaluate_batch call
TEST_SEED = 42

# ── Arms ─────────────────────────────────────────────────────────────────────
#   use_h_hist     : H_hist-based AL scoring vs σ-based (baseline)
#   use_injection  : stagnation injection + escalation enabled
#   fix_hard_reset : single-empire hard reset fix enabled
#   use_h_traj     : add H_traj to combined AL score
ARMS = [
    dict(name='1_baseline',      use_h_hist=False, use_injection=False, fix_hard_reset=False, use_h_traj=False,
         label='Baseline (σ-AL, no inj)'),
    dict(name='2_h_hist_only',   use_h_hist=True,  use_injection=False, fix_hard_reset=False, use_h_traj=False,
         label='H_hist AL only'),
    dict(name='3_inject_unfixed',use_h_hist=False, use_injection=True,  fix_hard_reset=False, use_h_traj=False,
         label='Inject only (HR bug)'),
    dict(name='4_h_hist_fixed',  use_h_hist=True,  use_injection=True,  fix_hard_reset=True,  use_h_traj=False,
         label='H_hist + Inj + HR FIX (orig, 50% wipe)'),
    dict(name='5_full_v9',       use_h_hist=True,  use_injection=True,  fix_hard_reset=True,  use_h_traj=True,
         label='Full v9 (+H_traj)'),
    dict(name='6_hr_gentle',     use_h_hist=True,  use_injection=True,  fix_hard_reset=True,  use_h_traj=False,
         label='H_hist + Inj + HR FIX v2 (gentle)'),
]

# ── Hard reset tuning (NEW — see PATCH NOTES) ─────────────────────────────────
HARD_RESET_FRAC      = 0.25   # was hardcoded 0.5 (half the empire) — too destructive
HARD_RESET_COOLDOWN  = 15     # min gens between resets — let the population recover

# ── Core geometry ─────────────────────────────────────────────────────────────
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
N_POS  = int(GRID_MASK.sum())   # 31
N_TYPES = 9

# ── QICA hyperparams (shared across all arms) ─────────────────────────────────
N_EMPIRES            = 6
ASSIMILATION_COEFF   = 0.35
REVOLUTION_RATE      = 0.35
REVOLUTION_MIN       = 0.08
REVOLUTION_MAX       = 0.65
REVOLUTION_BOOST     = 1.8
QUANTUM_TEMP_INIT    = 2.0
QUANTUM_TEMP_FINAL   = 0.1
ELITE_SIZE           = 12
ENTROPY_FREE_FRAC    = 0.65
PPF_LIMIT            = 3.5
W_PPF_PENALTY        = 80.0
W_PPF_SOFT           = 6.0
W_UNCERTAINTY        = 40.0
W_TRUST              = 0.0
W_ENTROPY_BONUS      = 5.0
W_MONOTONICITY       = 10.0
STAGNATION_PATIENCE  = 10      # shorter patience for short-budget runs
STAGNATION_N_INJECT  = 20
STAGNATION_N_ELITES  = 3
STAGNATION_MAX_ESC   = 4       # max escalation level
HARD_RESET_AT_CAP    = 3       # consecutive cap-level rounds before hard reset
STAGNATION_EPS       = 1e-3    # min fitness gain to reset escalation counter
PATTERN_DIV_LOW      = 0.35
POP_ENTROPY_REL_DROP = 0.10
HIST_BINS            = 10
SIGMA_HALF_WIDTH_MULT = 4.0
HIST_HALF_WIDTH_MIN  = 0.03
HIST_HALF_WIDTH_MAX  = 1.20
N_CAL_PATTERNS       = 40
N_CAL_MC_PASSES      = 15
AL_TOP_K             = 50       # max final AL candidates per arm
AL_PERCENTILE        = 70       # flag top-30% by AL score AND bottom-5% PPF
AL_LIVE_CAP          = 150      # running pool cap during search

# Mutable globals, set after calibration
TYPICAL_SIGMA: float  = 0.07
HIST_HALF_WIDTH: float = 0.28

# =============================================================================
# SECTION 2 — MODEL LOAD
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn2   = layers.BatchNormalization()
        self._filters = filters; self._dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
        self.proj = None
    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.proj = layers.Conv2D(self._filters, 1, padding='same')
        super().build(input_shape)
    def call(self, x, training=False):
        s = self.proj(x) if self.proj is not None else x
        h = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        h = tf.nn.gelu(h + s)
        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)
        return h
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self._filters, 'kernel_size': 3, 'dropout': self._dropout_rate})
        return cfg

for p in [MODEL_PATH, CONFIG_PATH]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing {p} — run cnn_v9.py first."); sys.exit(1)

model = keras.models.load_model(MODEL_PATH, compile=False,
                                custom_objects={'ConvResBlock': ConvResBlock})
print(f"[LOAD] {MODEL_PATH}  input={model.input_shape}  output={model.output_shape}")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

ym_mean  = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
ym_scale = np.array(cfg['ym_scaler_scale'], dtype=np.float32)
yr_mean  = np.array(cfg['yr_scaler_mean'],  dtype=np.float32)
yr_scale = np.array(cfg['yr_scaler_scale'], dtype=np.float32)
IDX_PPF_MAX = cfg['IDX_PPF_MAX']
IDX_STEPS_S = cfg['IDX_PPF_STEPS_START']
IDX_STEPS_E = cfg['IDX_PPF_STEPS_END']
IDX_CYCLE   = cfg['IDX_CYCLE']
IDX_RHO     = cfg['IDX_RHO']
N_STEPS     = IDX_STEPS_E - IDX_STEPS_S


# =============================================================================
# SECTION 3 — TRUST REGION
# =============================================================================

if os.path.exists(TRUST_PATH):
    type_freq = np.load(TRUST_PATH).astype(np.float32)
    print(f"[TRUST] {TRUST_PATH}  shape={type_freq.shape}")
else:
    type_freq = np.ones((N_POS, N_TYPES), dtype=np.float32) / N_TYPES
    print("[TRUST] Not found — using uniform fallback.")

def _compute_trust_region(freq, free_frac=ENTROPY_FREE_FRAC):
    h_pos      = (-np.sum(freq * np.log(freq + 1e-10), axis=1)).astype(np.float32)
    n_free     = max(1, int(np.round(N_POS * free_frac)))
    rank       = np.argsort(h_pos)[::-1]
    free_mask  = np.zeros(N_POS, dtype=bool)
    free_mask[rank[:n_free]] = True
    fixed_types = (np.argmax(freq, axis=1) + 1).astype(np.int32)
    return free_mask, fixed_types, n_free

free_mask, fixed_types, n_free = _compute_trust_region(type_freq)
print(f"[TRUST] {n_free}/{N_POS} positions free\n")


# =============================================================================
# SECTION 4 — CALIBRATION + SEED LOAD
# =============================================================================

X_train_seed = None; ppf_cnn_seed = None; X_grid_seed = None

def _calibrate_and_seed():
    global X_train_seed, ppf_cnn_seed, X_grid_seed, TYPICAL_SIGMA, HIST_HALF_WIDTH
    csv_path = 'ml_dataset_constrained.csv'
    if not os.path.exists(csv_path):
        print("[CAL] ml_dataset_constrained.csv not found — skipping seed load + calibration")
        return
    df   = pd.read_csv(csv_path, skiprows=1, engine='python', on_bad_lines='skip')
    lc   = [f'loading_{i}' for i in range(N_POS)]
    if not all(c in df.columns for c in lc):
        print("[CAL] loading_ columns not found"); return
    X_raw = df[lc].values.astype(np.int32)
    N     = len(X_raw)
    grids = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grids[:, r, c] = X_raw[:, pi]; pi += 1
    ppf_preds = []
    for i in range(0, N, 128):
        sc = model(tf.constant(grids[i:i+128], dtype=tf.int32), training=False).numpy()
        ppf_preds.extend((sc[:, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]).tolist())
    ppf_arr = np.array(ppf_preds, dtype=np.float32)
    X_train_seed = X_raw; X_grid_seed = grids; ppf_cnn_seed = ppf_arr
    print(f"[SEED] {N} patterns loaded | PPF {ppf_arr.min():.3f}–{ppf_arr.max():.3f}")

    # Calibrate histogram half-width from actual MC dropout noise
    n_cal     = min(N_CAL_PATTERNS, N)
    cal_idx   = np.random.choice(N, n_cal, replace=False)
    mc_cal_sc = np.stack([
        model(tf.constant(grids[cal_idx], dtype=tf.int32), training=True).numpy()[:, IDX_PPF_MAX]
        for _ in range(N_CAL_MC_PASSES)
    ])
    mc_cal_phys  = mc_cal_sc * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
    sigmas       = mc_cal_phys.std(axis=0)
    typical_sig  = float(np.median(sigmas))
    p95_sig      = float(np.percentile(sigmas, 95))
    TYPICAL_SIGMA   = max(typical_sig, 1e-4)
    raw_hw          = SIGMA_HALF_WIDTH_MULT * max(typical_sig, 0.6 * p95_sig)
    HIST_HALF_WIDTH = float(np.clip(raw_hw, HIST_HALF_WIDTH_MIN, HIST_HALF_WIDTH_MAX))
    print(f"[CAL]  median σ={typical_sig:.4f}  p95 σ={p95_sig:.4f}  →  hist half-width=±{HIST_HALF_WIDTH:.4f}")

_calibrate_and_seed()


# =============================================================================
# SECTION 5 — ENTROPY UTILITIES
# =============================================================================

def gaussian_entropy(mc_ppf: np.ndarray) -> np.ndarray:
    """H = 0.5 * log(2πe * σ²).  mc_ppf: (MC, B) → (B,)"""
    sigma = mc_ppf.std(axis=0) + 1e-10
    return (0.5 * np.log(2.0 * np.pi * np.e * sigma**2)).astype(np.float32)

def histogram_entropy(mc_ppf: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """Calibrated, per-pattern-centered histogram Shannon H.  mc_ppf: (MC, B) → (B,)"""
    MC, B = mc_ppf.shape
    entropies = np.zeros(B, dtype=np.float32)
    centered  = mc_ppf - mc_ppf.mean(axis=0, keepdims=True)
    lo, hi    = -HIST_HALF_WIDTH, HIST_HALF_WIDTH
    for b in range(B):
        hist, _ = np.histogram(centered[:, b], bins=bins, range=(lo, hi))
        p = hist.astype(np.float64) / (hist.sum() + 1e-12)
        p = p[p > 0]
        entropies[b] = float(-np.sum(p * np.log(p)))
    return entropies

def trajectory_entropy(mc_curves: np.ndarray) -> np.ndarray:
    """Multivariate Gaussian H over full PPF burnup curve.  mc_curves: (MC, B, K) → (B,)"""
    MC, B, K = mc_curves.shape
    entropies = np.zeros(B, dtype=np.float32)
    eff_rank  = min(MC - 1, K)
    for b in range(B):
        X   = mc_curves[:, b, :].astype(np.float64)
        X_c = X - X.mean(axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)
            eigvals  = np.maximum((s[:eff_rank]**2) / max(MC - 1, 1), 1e-12)
            H = 0.5 * (eff_rank * np.log(2.0 * np.pi * np.e) + np.sum(np.log(eigvals)))
        except np.linalg.LinAlgError:
            H = 0.0
        entropies[b] = float(H)
    return entropies

def hamming_diversity(empires: list, max_sample: int = 60) -> float:
    """Mean pairwise Hamming distance over free positions only."""
    pats = []
    for emp in empires:
        if emp.imperialist.measured is not None:
            pats.append(emp.imperialist.measured)
        for c in emp.colonies:
            if c.measured is not None:
                pats.append(c.measured)
    if len(pats) < 2:
        return 1.0
    free_idx = np.where(free_mask)[0]
    arr = np.stack(pats)[:, free_idx]
    n = arr.shape[0]
    if n > max_sample:
        arr = arr[np.random.choice(n, max_sample, replace=False)]
        n = max_sample
    total, count = 0.0, 0
    for i in range(n):
        diffs = (arr[i+1:] != arr[i]).mean(axis=1)
        total += float(diffs.sum()); count += diffs.shape[0]
    return total / count if count > 0 else 1.0


# =============================================================================
# SECTION 6 — BATCH EVALUATION
# =============================================================================

def pattern_to_grid(patterns: np.ndarray) -> np.ndarray:
    B   = patterns.shape[0] if patterns.ndim > 1 else 1
    pat = patterns.reshape(B, N_POS)
    g   = np.zeros((B, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi  = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = pat[:, pi]; pi += 1
    return g

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    return np.concatenate([
        Y_sc[:, :34] * ym_scale + ym_mean,
        Y_sc[:, 34:35] * yr_scale + yr_mean,
    ], axis=1)

def evaluate_batch(patterns: np.ndarray, mc_n: int = TEST_MC, compute_h_traj: bool = True) -> dict:
    """Always computes H_hist; H_traj is skipped unless compute_h_traj=True
    (it's an SVD per pattern per call — expensive and, per the v1 ablation,
    a confirmed weak discriminator: CV 0.026-0.039, under the 0.05 bar)."""
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)
    B    = patterns.shape[0]
    grid = pattern_to_grid(patterns)
    X_tf = tf.constant(grid, dtype=tf.int32)

    mc_sc = np.stack([model(X_tf, training=True).numpy() for _ in range(mc_n)])
    mean_sc = mc_sc.mean(axis=0)
    std_sc  = mc_sc.std(axis=0)

    mean_real = inverse_transform(mean_sc)
    std_real  = np.concatenate([
        std_sc[:, :34] * ym_scale,
        std_sc[:, 34:35] * yr_scale,
    ], axis=1)

    ppf_mean   = mean_real[:, IDX_PPF_MAX]
    ppf_std    = std_real[:, IDX_PPF_MAX]
    cycle_mean = mean_real[:, IDX_CYCLE]
    ppf_steps  = mean_real[:, IDX_STEPS_S:IDX_STEPS_E]

    mc_ppf_phys = mc_sc[:, :, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]

    h_gauss = gaussian_entropy(mc_ppf_phys)
    h_hist  = histogram_entropy(mc_ppf_phys)
    if compute_h_traj:
        mc_curves_phys = (mc_sc[:, :, IDX_STEPS_S:IDX_STEPS_E]
                          * ym_scale[IDX_STEPS_S:IDX_STEPS_E]
                          + ym_mean[IDX_STEPS_S:IDX_STEPS_E])
        h_traj = trajectory_entropy(mc_curves_phys)
    else:
        h_traj = np.zeros(B, dtype=np.float32)

    late       = ppf_steps[:, 3:]
    diffs      = late[:, 1:] - late[:, :-1]
    mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / max(late.shape[1]-1, 1))
    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    fitness    = (cycle_mean
                  - W_PPF_SOFT    * ppf_mean
                  - W_PPF_PENALTY * ppf_excess
                  - W_UNCERTAINTY * ppf_std
                  + mono_bonus)

    return {
        'ppf_mean': ppf_mean, 'ppf_std': ppf_std,
        'cycle_mean': cycle_mean, 'fitness': fitness,
        'h_gauss': h_gauss, 'h_hist': h_hist, 'h_traj': h_traj,
    }


# =============================================================================
# SECTION 7 — QUANTUM COUNTRY + EMPIRE
# =============================================================================

class QuantumCountry:
    __slots__ = ('q_state', 'measured', 'fitness', 'ppf_mean', 'ppf_std',
                 'cycle_mean', 'h_gauss', 'h_hist', 'h_traj', 'al_score')
    def __init__(self, q_state: np.ndarray = None):
        if q_state is None:
            raw = np.ones((N_POS, N_TYPES), dtype=np.float32)
            self.q_state = raw / raw.sum(axis=1, keepdims=True)
        else:
            self.q_state = q_state.copy().astype(np.float32)
        for p in range(N_POS):
            if not free_mask[p]:
                self.q_state[p] = 0.0
                self.q_state[p, fixed_types[p]-1] = 1.0
        self.measured   = None
        self.fitness    = -np.inf;  self.ppf_mean   = 9.0
        self.ppf_std    = 0.0;      self.cycle_mean = 0.0
        self.h_gauss    = -10.0;    self.h_hist     = 0.0
        self.h_traj     = 0.0;      self.al_score   = 0.0

    def collapse(self, temperature: float = 1.0) -> np.ndarray:
        logits = np.log(self.q_state + 1e-10) / max(temperature, 0.01)
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits) / (np.exp(logits).sum(axis=1, keepdims=True) + 1e-10)
        self.measured = np.array([
            np.random.choice(N_TYPES, p=probs[i]) + 1 for i in range(N_POS)
        ], dtype=np.int32)
        for p in range(N_POS):
            if not free_mask[p]:
                self.measured[p] = fixed_types[p]
        return self.measured

    def q_entropy(self) -> float:
        return float(-np.sum(self.q_state * np.log(self.q_state + 1e-10)))

    def quantum_assimilate(self, imp: 'QuantumCountry', beta: float, temp: float):
        for p in range(N_POS):
            if free_mask[p]:
                self.q_state[p] = (1.0 - beta) * self.q_state[p] + beta * imp.q_state[p]
        self.q_state = np.maximum(self.q_state, 1e-10)
        self.q_state /= self.q_state.sum(axis=1, keepdims=True)
        for p in range(N_POS):
            if not free_mask[p]:
                self.q_state[p] = 0.0; self.q_state[p, fixed_types[p]-1] = 1.0

    def quantum_revolution(self, rate: float, temperature: float):
        for p in range(N_POS):
            if free_mask[p] and np.random.random() < min(rate, 0.95):
                alpha = np.ones(N_TYPES) * max(temperature, 0.05)
                self.q_state[p] = np.random.dirichlet(alpha)

    def clone(self) -> 'QuantumCountry':
        c = QuantumCountry(self.q_state)
        if self.measured is not None: c.measured = self.measured.copy()
        c.fitness = self.fitness; c.ppf_mean = self.ppf_mean
        c.ppf_std = self.ppf_std; c.cycle_mean = self.cycle_mean
        c.h_gauss = self.h_gauss; c.h_hist = self.h_hist
        c.h_traj  = self.h_traj;  c.al_score = self.al_score
        return c

class Empire:
    def __init__(self, imp: QuantumCountry, cols: list):
        self.imperialist = imp; self.colonies = cols
    @property
    def power(self): return self.imperialist.fitness
    @property
    def total_countries(self): return 1 + len(self.colonies)


# =============================================================================
# SECTION 8 — ABLATION QICA
# =============================================================================

class AblationQICA:
    """
    Single QICA implementation parametrised by arm flags.
    Identical core logic across all arms; flags toggle:
      use_h_hist      → H_hist vs σ for AL scoring
      use_injection   → stagnation injection + escalation
      fix_hard_reset  → single-empire-safe hard reset
      use_h_traj      → include H_traj in combined AL score
    """

    def __init__(self, flags: dict, arm_name: str,
                 n_pop: int = TEST_POP, max_gen: int = TEST_GENS, mc_n: int = TEST_MC):
        self.flags      = flags
        self.arm_name   = arm_name
        self.n_pop      = n_pop
        self.max_gen    = max_gen
        self.mc_n       = mc_n
        self.elite_archive   = []
        self.al_candidates   = []
        self._al_seen        = set()
        self.stagnation_count   = 0
        self.stagnation_rounds  = 0
        self.rounds_at_cap      = 0
        self.last_best_fitness  = None
        self.best_fitness_ever  = -np.inf
        self.last_reset_gen     = -10**9   # NEW — enables HARD_RESET_COOLDOWN
        # Metrics tracked per-generation
        self.history = {
            'gen': [], 'best_ppf': [], 'best_cycle': [], 'best_sigma': [],
            'best_h_hist': [], 'best_h_traj': [],
            'mean_h_hist': [], 'mean_h_traj': [],
            'al_total': [], 'al_added_this_gen': [],
            'inj_fired': [], 'reset_fired': [],
            'ham_div': [], 'rev_rate': [], 'n_empires': [], 'temp': [],
            'stagnation_round': [], 'best_fitness': [],
        }
        # Counters for final summary
        self.total_injections = 0
        self.total_resets     = 0

    def _temperature(self, gen):
        r = gen / self.max_gen
        return QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** r

    def _base_rev_rate(self, gen):
        r = gen / self.max_gen
        return REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * r

    def _adaptive_rev_rate(self, base: float, ham_div: float) -> float:
        if ham_div < PATTERN_DIV_LOW:
            severity = 1.0 - ham_div / PATTERN_DIV_LOW
            return float(min(base * (1.0 + REVOLUTION_BOOST * severity), REVOLUTION_MAX))
        return float(base)

    # ── Population init ───────────────────────────────────────────────────────
    def _init_population(self) -> list:
        countries = []
        for bias_t in range(1, N_TYPES + 1):
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.04
            q[:, bias_t-1] = 0.68
            q /= q.sum(axis=1, keepdims=True)
            countries.append(QuantumCountry(q))
        if X_train_seed is not None:
            n_seeds  = min(8, len(X_train_seed))
            top_idx  = np.argsort(ppf_cnn_seed)[:n_seeds]
            for idx in top_idx:
                pat = X_train_seed[idx]
                q   = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
                for p in range(N_POS):
                    t = int(pat[p])
                    if 1 <= t <= N_TYPES: q[p, t-1] = 0.84
                q /= q.sum(axis=1, keepdims=True)
                countries.append(QuantumCountry(q))
        while len(countries) < self.n_pop:
            countries.append(QuantumCountry())
        return countries

    # ── Evaluate + AL scoring ─────────────────────────────────────────────────
    def _evaluate_all(self, countries: list, temperature: float) -> list:
        patterns = np.stack([c.collapse(temperature) for c in countries])
        result   = evaluate_batch(patterns, mc_n=self.mc_n,
                                   compute_h_traj=self.flags['use_h_traj'])

        ppf_arr   = result['ppf_mean']
        ppf_std   = result['ppf_std']
        h_hist_arr = result['h_hist']
        h_traj_arr = result['h_traj']
        ppf_5pct   = np.percentile(ppf_arr, 5)  # AL only for promising patterns

        # AL score: what this arm uses to rank patterns for simulator queries
        if not self.flags['use_h_hist']:
            # baseline / inject_only: z-score of σ_ppf
            al_scores = (ppf_std - ppf_std.mean()) / (ppf_std.std() + 1e-8)
        elif self.flags['use_h_traj']:
            # full_v9: z(H_hist) + z(H_traj)  — the production v9 rule
            al_scores = (
                (h_hist_arr - h_hist_arr.mean()) / (h_hist_arr.std() + 1e-8)
                + (h_traj_arr - h_traj_arr.mean()) / (h_traj_arr.std() + 1e-8)
            ).astype(np.float32)
        else:
            # h_hist_only / h_hist_fixed: z(H_hist) alone
            al_scores = ((h_hist_arr - h_hist_arr.mean()) / (h_hist_arr.std() + 1e-8)).astype(np.float32)

        al_thr = max(1e-3, float(np.percentile(al_scores, AL_PERCENTILE)))

        ent_bonus = np.array([
            W_ENTROPY_BONUS * c.q_entropy() / (N_POS * N_TYPES) for c in countries
        ], dtype=np.float32)
        ppf_excess = np.maximum(0.0, ppf_arr - PPF_LIMIT)
        fitness_arr = (
            result['cycle_mean']
            - W_PPF_SOFT    * ppf_arr
            - W_PPF_PENALTY * ppf_excess
            - W_UNCERTAINTY * ppf_std
            + result['fitness'] - result['cycle_mean'] + result['cycle_mean']  # monotonicity already in evaluate_batch
            + ent_bonus
        )
        # Recompute cleanly (evaluate_batch already includes monotonicity bonus)
        fitness_arr = result['fitness'] + ent_bonus

        al_added_this_call = 0
        for i, c in enumerate(countries):
            c.fitness    = float(fitness_arr[i])
            c.ppf_mean   = float(ppf_arr[i])
            c.ppf_std    = float(ppf_std[i])
            c.cycle_mean = float(result['cycle_mean'][i])
            c.h_gauss    = float(result['h_gauss'][i])
            c.h_hist     = float(result['h_hist'][i])
            c.h_traj     = float(result['h_traj'][i])
            c.al_score   = float(al_scores[i])

            # AL flagging: uncertain AND promising
            if al_scores[i] >= al_thr and ppf_arr[i] <= ppf_5pct:
                pat_key = tuple(c.measured.tolist())
                if pat_key not in self._al_seen:
                    self._al_seen.add(pat_key)
                    priority = float(c.al_score * c.cycle_mean / (c.ppf_mean + 1e-6))
                    self.al_candidates.append({
                        'arm': self.arm_name,
                        'pattern': c.measured.tolist(),
                        'pred_ppf': c.ppf_mean, 'sigma_ppf': c.ppf_std,
                        'h_hist': c.h_hist, 'h_traj': c.h_traj,
                        'h_gauss': c.h_gauss, 'al_score': c.al_score,
                        'cycle': c.cycle_mean, 'priority': priority,
                    })
                    al_added_this_call += 1
                    if len(self.al_candidates) > AL_LIVE_CAP:
                        self.al_candidates.sort(key=lambda d: d['priority'], reverse=True)
                        kept = self.al_candidates[:AL_LIVE_CAP // 2]
                        self.al_candidates = kept
                        self._al_seen = {tuple(d['pattern']) for d in kept}
        return countries, al_added_this_call

    # ── Empire mechanics ──────────────────────────────────────────────────────
    def _form_empires(self, countries: list) -> list:
        sorted_c = sorted(countries, key=lambda c: c.fitness, reverse=True)
        n_emp    = min(N_EMPIRES, len(sorted_c))
        imps, cols = sorted_c[:n_emp], sorted_c[n_emp:]
        fits    = np.array([i.fitness for i in imps])
        fits_sh = fits - fits.min() + 1e-6
        powers  = fits_sh / fits_sh.sum()
        counts  = np.round(powers * len(cols)).astype(int)
        diff    = len(cols) - counts.sum()
        if diff > 0: counts[np.argmax(powers)] += diff
        elif diff < 0: counts[np.argmax(counts)] += diff
        empires, idx = [], 0
        for i, imp in enumerate(imps):
            empires.append(Empire(imp, list(cols[idx:idx+counts[i]])))
            idx += counts[i]
        return empires

    def _assimilation(self, empires, beta, temp, rev_rate):
        for emp in empires:
            for col in emp.colonies:
                col.quantum_assimilate(emp.imperialist, beta, temp)
                col.quantum_revolution(rev_rate, temp)

    def _intra_competition(self, empires, temp):
        all_cols = [c for emp in empires for c in emp.colonies]
        if not all_cols: return
        self._evaluate_all(all_cols, temp)
        for emp in empires:
            if not emp.colonies: continue
            best_i = max(range(len(emp.colonies)), key=lambda i: emp.colonies[i].fitness)
            if emp.colonies[best_i].fitness > emp.imperialist.fitness:
                emp.colonies[best_i], emp.imperialist = emp.imperialist, emp.colonies[best_i]

    def _empire_collapse(self, empires) -> list:
        if len(empires) <= 1: return empires
        wi = min(range(len(empires)), key=lambda i: empires[i].power)
        si = max(range(len(empires)), key=lambda i: empires[i].power)
        if len(empires[wi].colonies) == 0:
            empires[si].colonies.append(empires[wi].imperialist)
            empires.pop(wi)
        else:
            wci = min(range(len(empires[wi].colonies)), key=lambda i: empires[wi].colonies[i].fitness)
            empires[si].colonies.append(empires[wi].colonies.pop(wci))
        return empires

    def _update_elite(self, empires):
        for emp in empires:
            for c in [emp.imperialist] + emp.colonies:
                if c.measured is not None:
                    self.elite_archive.append((
                        c.fitness, c.measured.copy(),
                        c.ppf_mean, c.cycle_mean, c.ppf_std,
                        c.h_gauss, c.h_hist, c.h_traj
                    ))
        self.elite_archive.sort(key=lambda x: x[0], reverse=True)
        seen, unique = set(), []
        for e in self.elite_archive:
            k = tuple(e[1])
            if k not in seen: seen.add(k); unique.append(e)
        self.elite_archive = unique[:ELITE_SIZE]

    # ── Stagnation injection (FIX 8: escalating multi-elite) ─────────────────
    def _inject_mutations(self, empires, temperature, round_idx: int):
        if not self.elite_archive or not self.flags['use_injection']:
            return None
        esc       = min(round_idx - 1, STAGNATION_MAX_ESC)
        n_elites  = min(STAGNATION_N_ELITES, len(self.elite_archive))
        n_inject  = int(STAGNATION_N_INJECT * (1.0 + 0.5 * esc))
        new_c = []
        for i in range(n_inject):
            seed_pat = self.elite_archive[i % n_elites][1]
            lo_mut = min(5 + 2 * esc, n_free)
            hi_mut = min(14 + 4 * esc, n_free)
            if hi_mut <= lo_mut: hi_mut = lo_mut + 1
            n_mutate = np.random.randint(lo_mut, hi_mut + 1)
            mut_pos  = np.random.choice(
                [p for p in range(N_POS) if free_mask[p]],
                min(n_mutate, n_free), replace=False)
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
            for p, t in enumerate(seed_pat): q[p, int(t)-1] = 0.84
            q /= q.sum(axis=1, keepdims=True)
            for p in mut_pos: q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
            new_c.append(QuantumCountry(q))
        boosted_temp = min(temperature * (2.5 + 0.5 * esc), 2.0)
        patterns = np.stack([c.collapse(boosted_temp) for c in new_c])
        result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
        for i, c in enumerate(new_c):
            c.fitness = float(result['fitness'][i]); c.ppf_mean = float(result['ppf_mean'][i])
            c.ppf_std = float(result['ppf_std'][i]); c.h_hist   = float(result['h_hist'][i])
            c.cycle_mean = float(result['cycle_mean'][i])
        largest = max(range(len(empires)), key=lambda i: empires[i].total_countries)
        empires[largest].colonies.extend(new_c)
        self.stagnation_count = 0
        self.total_injections += 1
        print(f"  [{self.arm_name}] ★ INJECT round={round_idx} esc={esc} | "
              f"{n_inject} mutations from top-{n_elites} elites | "
              f"best_ppf={self.elite_archive[0][2]:.3f}")
        return esc

    # ── Hard reset (FIXED: single-empire support) ─────────────────────────────
    def _hard_reset(self, empires, temperature, gen: int) -> bool:
        """
        v2: two changes vs the original "fix":
          1. HARD_RESET_FRAC (0.25) replaces the hardcoded //2 (50%) wipe of
             the surviving empire's colonies — the 50% wipe is the likely
             cause of arm4 scoring worse than arm3 (bug-present / never
             fires) in the v1 ablation.
          2. HARD_RESET_COOLDOWN stops back-to-back resets so the population
             gets time to actually be judged before being wiped again.
        """
        if gen - self.last_reset_gen < HARD_RESET_COOLDOWN:
            return False
        if not self.elite_archive: return False
        if not self.flags['use_injection']: return False

        if len(empires) == 1:
            if not self.flags['fix_hard_reset']:
                print(f"  [{self.arm_name}] ✗ HARD RESET suppressed — single empire, fix not enabled (BUG)")
                return False
            emp = empires[0]
            if len(emp.colonies) < 2: return False
            # Sort ascending (worst first), reinit worst HARD_RESET_FRAC
            emp.colonies.sort(key=lambda c: c.fitness)
            n_reinit = max(2, int(round(len(emp.colonies) * HARD_RESET_FRAC)))
            new_c    = []
            n_mid    = max(0, len(self.elite_archive) - 2)
            for i in range(n_reinit):
                if n_mid > 0 and i % 2 == 0:
                    seed_pat = self.elite_archive[2 + (i % n_mid)][1]
                    q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.03
                    for p, t in enumerate(seed_pat): q[p, int(t)-1] = 0.7
                    q /= q.sum(axis=1, keepdims=True)
                    n_heavy = min(n_free, max(8, n_free // 2))
                    heavy   = np.random.choice([p for p in range(N_POS) if free_mask[p]], n_heavy, replace=False)
                    for p in heavy: q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                    new_c.append(QuantumCountry(q))
                else:
                    new_c.append(QuantumCountry())
            patterns = np.stack([c.collapse(1.6) for c in new_c])
            result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
            for i, c in enumerate(new_c):
                c.fitness = float(result['fitness'][i]); c.ppf_mean = float(result['ppf_mean'][i])
                c.ppf_std = float(result['ppf_std'][i]); c.h_hist   = float(result['h_hist'][i])
                c.cycle_mean = float(result['cycle_mean'][i])
            emp.colonies[:n_reinit] = new_c
            self.total_resets += 1
            print(f"  [{self.arm_name}] ⚡ HARD RESET SINGLE EMPIRE — "
                  f"reinitialized worst {n_reinit}/{len(emp.colonies)+n_reinit} colonies")
            return True

        else:  # multiple empires — original logic
            worst_idx = min(range(len(empires)), key=lambda i: empires[i].power)
            n_fresh   = max(8, empires[worst_idx].total_countries)
            n_mid     = max(0, len(self.elite_archive) - 2)
            new_c = []
            for i in range(n_fresh):
                if n_mid > 0 and i % 2 == 0:
                    seed_pat = self.elite_archive[2 + (i % n_mid)][1]
                    q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.03
                    for p, t in enumerate(seed_pat): q[p, int(t)-1] = 0.7
                    q /= q.sum(axis=1, keepdims=True)
                    n_heavy = min(n_free, max(8, n_free // 2))
                    heavy   = np.random.choice([p for p in range(N_POS) if free_mask[p]], n_heavy, replace=False)
                    for p in heavy: q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                    new_c.append(QuantumCountry(q))
                else:
                    new_c.append(QuantumCountry())
            patterns = np.stack([c.collapse(1.6) for c in new_c])
            result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
            for i, c in enumerate(new_c):
                c.fitness = float(result['fitness'][i]); c.ppf_mean = float(result['ppf_mean'][i])
                c.ppf_std = float(result['ppf_std'][i]); c.h_hist   = float(result['h_hist'][i])
                c.cycle_mean = float(result['cycle_mean'][i])
            empires[worst_idx] = Empire(new_c[0], new_c[1:])
            self.total_resets += 1
            print(f"  [{self.arm_name}] ⚡ HARD RESET empire #{worst_idx}")
            return True

    # ── Per-generation log ────────────────────────────────────────────────────
    def _log(self, gen, empires, temp, ham_div, rev_rate, al_added, inj_fired, reset_fired):
        best = (self.elite_archive[0] if self.elite_archive
                else (0, None, 9.0, 0.0, 0.0, -10.0, 0.0, 0.0))
        all_c    = [e.imperialist for e in empires] + [c for e in empires for c in e.colonies]
        mean_hh  = float(np.mean([c.h_hist for c in all_c]))
        mean_ht  = float(np.mean([c.h_traj for c in all_c]))
        self.history['gen'].append(gen)
        self.history['best_ppf'].append(float(best[2]))
        self.history['best_cycle'].append(float(best[3]))
        self.history['best_sigma'].append(float(best[4]))
        self.history['best_h_hist'].append(float(best[6]))
        self.history['best_h_traj'].append(float(best[7]))
        self.history['mean_h_hist'].append(mean_hh)
        self.history['mean_h_traj'].append(mean_ht)
        self.history['al_total'].append(len(self.al_candidates))
        self.history['al_added_this_gen'].append(al_added)
        self.history['inj_fired'].append(int(inj_fired))
        self.history['reset_fired'].append(int(reset_fired))
        self.history['ham_div'].append(float(ham_div))
        self.history['rev_rate'].append(float(rev_rate))
        self.history['n_empires'].append(len(empires))
        self.history['temp'].append(float(temp))
        self.history['stagnation_round'].append(self.stagnation_rounds)
        self.history['best_fitness'].append(float(best[0]))

        if gen % 20 == 0 or gen == self.max_gen:
            al_flag = ''
            if inj_fired: al_flag = ' ★INJ'
            if reset_fired: al_flag += ' ⚡RST'
            print(
                f"  [{self.arm_name}] Gen {gen:3d}/{self.max_gen} | "
                f"ppf={best[2]:.3f} σ={best[4]:.4f} | "
                f"H_hist={best[6]:.3f} H_traj={best[7]:.1f} | "
                f"AL_new=+{al_added}(tot={len(self.al_candidates)}) | "
                f"div={ham_div:.2f} rev={rev_rate:.3f} | "
                f"emp={len(empires)} stag={self.stagnation_rounds}{al_flag}"
            )

    # ── Main run ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        np.random.seed(TEST_SEED); tf.random.set_seed(TEST_SEED)  # identical seed every arm
        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"ARM: {self.arm_name}  ({self.flags['label']})")
        print(f"  H_hist={self.flags['use_h_hist']}  Inject={self.flags['use_injection']}  "
              f"HR_fix={self.flags['fix_hard_reset']}  H_traj={self.flags['use_h_traj']}")
        print(f"  N={self.n_pop}  Gens={self.max_gen}  MC={self.mc_n}")
        print(f"{'='*70}")

        countries = self._init_population()
        temp      = self._temperature(0)
        countries, al_added = self._evaluate_all(countries, temp)
        empires   = self._form_empires(countries)
        self._update_elite(empires)
        ham_div   = hamming_diversity(empires)
        base_rate = self._base_rev_rate(0)
        rev_rate  = self._adaptive_rev_rate(base_rate, ham_div)
        self.best_fitness_ever = self.elite_archive[0][0]
        self._log(0, empires, temp, ham_div, rev_rate, al_added, False, False)

        best0 = self.elite_archive[0]
        print(f"  Init: ppf={best0[2]:.3f}  cycle={best0[3]:.1f}d  σ={best0[4]:.4f}  H_hist={best0[6]:.3f}")

        for gen in range(1, self.max_gen + 1):
            temp      = self._temperature(gen)
            base_rate = self._base_rev_rate(gen)
            ham_div   = hamming_diversity(empires)
            rev_rate  = self._adaptive_rev_rate(base_rate, ham_div)

            self._assimilation(empires, ASSIMILATION_COEFF, temp, rev_rate)
            _, al_col = self._evaluate_all(
                [c for emp in empires for c in emp.colonies], temp)
            self._intra_competition(empires, temp)
            self._update_elite(empires)
            empires = self._empire_collapse(empires)

            # Stagnation tracking + injection/reset
            inj_fired = False; reset_fired = False
            if self.elite_archive:
                cur_fit = self.elite_archive[0][0]
                if cur_fit > self.best_fitness_ever + STAGNATION_EPS:
                    self.best_fitness_ever  = cur_fit
                    self.stagnation_rounds  = 0
                    self.rounds_at_cap      = 0
                if self.last_best_fitness is not None:
                    if abs(cur_fit - self.last_best_fitness) < 0.05:
                        self.stagnation_count += 1
                    else:
                        self.stagnation_count = 0
                self.last_best_fitness = cur_fit

                if self.flags['use_injection'] and self.stagnation_count >= STAGNATION_PATIENCE:
                    self.stagnation_rounds += 1
                    esc = self._inject_mutations(empires, temp, round_idx=self.stagnation_rounds)
                    inj_fired = True
                    if esc is not None and esc >= STAGNATION_MAX_ESC:
                        self.rounds_at_cap += 1
                    else:
                        self.rounds_at_cap = 0
                    if self.rounds_at_cap >= HARD_RESET_AT_CAP:
                        fired = self._hard_reset(empires, temp, gen)
                        if fired:
                            reset_fired = True
                            self.rounds_at_cap = 0
                            self.last_reset_gen = gen

            # Re-evaluate imperialists after injections (if any)
            imps = [emp.imperialist for emp in empires]
            _, al_imp = self._evaluate_all(imps, temp)
            al_added = al_col + al_imp

            self._log(gen, empires, temp, ham_div, rev_rate, al_added, inj_fired, reset_fired)

        runtime = time.time() - t0
        # Cap final AL output
        self.al_candidates.sort(key=lambda d: d['priority'], reverse=True)
        self.al_candidates = self.al_candidates[:AL_TOP_K]

        best = self.elite_archive[0] if self.elite_archive else (0, None, 9.0, 0.0, 0.0, -10.0, 0.0, 0.0)
        print(f"\n  DONE [{self.arm_name}]  {runtime:.0f}s  |  "
              f"best_ppf={best[2]:.4f}  cycle={best[3]:.1f}d  "
              f"σ={best[4]:.4f}  H_hist={best[6]:.3f}")
        print(f"  Injections={self.total_injections}  Hard_resets={self.total_resets}  "
              f"AL_candidates={len(self.al_candidates)}")

        return {
            'arm': self.arm_name, 'flags': self.flags,
            'elite': self.elite_archive,
            'history': self.history,
            'al_candidates': self.al_candidates,
            'runtime': runtime,
            'total_injections': self.total_injections,
            'total_resets': self.total_resets,
        }


# =============================================================================
# SECTION 9 — RUN ALL ARMS
# =============================================================================

all_results = []
for arm_cfg in ARMS:
    qica = AblationQICA(
        flags=arm_cfg, arm_name=arm_cfg['name'],
        n_pop=TEST_POP, max_gen=TEST_GENS, mc_n=TEST_MC
    )
    result = qica.run()
    all_results.append(result)


# =============================================================================
# SECTION 10 — ENTROPY DIAGNOSTICS PER ARM
# =============================================================================

print("\n\n" + "=" * 80)
print("ENTROPY DIAGNOSTICS — AL CANDIDATES (per arm)")
print("=" * 80)
print(f"  {'Arm':<22} {'n_AL':>5} {'H_hist_mean':>12} {'H_hist_CV':>10} "
      f"{'H_hist-σ corr':>14} {'H_traj_CV':>10} {'bimodal_det':>12}")
print(f"  {'-'*22} {'-'*5} {'-'*12} {'-'*10} {'-'*14} {'-'*10} {'-'*12}")

al_diag = {}
for r in all_results:
    al = r['al_candidates']
    name = r['arm']
    if not al:
        print(f"  {name:<22}     0    (no AL candidates)")
        al_diag[name] = {}
        continue
    al_df = pd.DataFrame(al)
    n = len(al_df)
    hh = al_df['h_hist'].values
    ht = al_df['h_traj'].values
    sg = al_df['sigma_ppf'].values
    hh_mean = float(hh.mean())
    hh_cv   = float(hh.std() / (abs(hh_mean) + 1e-8))
    hh_corr = float(np.corrcoef(hh, sg[:len(hh)])[0, 1])
    ht_cv   = float(ht.std() / (abs(ht.mean()) + 1e-8)) if ht.std() > 0 else 0.0
    # Bimodal detection: H_hist high (relative) but H_gauss/σ low
    hh_norm = (hh - hh.min()) / (hh.max() - hh.min() + 1e-8)
    hg_norm = (al_df['h_gauss'].values - al_df['h_gauss'].values.min()) / (al_df['h_gauss'].values.max() - al_df['h_gauss'].values.min() + 1e-8)
    bimodal = int(((hh_norm > 0.5) & (hg_norm < 0.3)).sum())
    al_diag[name] = {'n': n, 'hh_mean': hh_mean, 'hh_cv': hh_cv, 'hh_corr': hh_corr,
                     'ht_cv': ht_cv, 'bimodal': bimodal}
    print(f"  {name:<22} {n:>5} {hh_mean:>12.3f} {hh_cv:>10.3f} "
          f"{hh_corr:>14.3f} {ht_cv:>10.4f} {bimodal:>12}")


# =============================================================================
# SECTION 11 — FINAL COMPARISON TABLE
# =============================================================================

print("\n\n" + "=" * 80)
print("FINAL COMPARISON TABLE")
print("=" * 80)
print(f"  {'Arm':<22} {'Label':<28} {'BestPPF':>8} {'Cycle':>8} "
      f"{'σ_best':>8} {'H_hist':>7} {'Injections':>11} {'HR_fired':>9} "
      f"{'AL_cnt':>7} {'Time(s)':>8}")
print(f"  {'-'*22} {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*11} {'-'*9} {'-'*7} {'-'*8}")

summary_rows = []
baseline_ppf = None
for r in all_results:
    best = r['elite'][0] if r['elite'] else (0, None, 9.999, 0.0, 0.0, 0.0, 0.0, 0.0)
    flags = r['flags']
    row = {
        'arm': r['arm'], 'label': flags['label'],
        'best_ppf': float(best[2]), 'cycle': float(best[3]),
        'sigma': float(best[4]), 'h_hist': float(best[6]),
        'injections': r['total_injections'], 'hr_fired': r['total_resets'],
        'al_cnt': len(r['al_candidates']), 'runtime': r['runtime'],
    }
    summary_rows.append(row)
    if r['arm'] == '1_baseline': baseline_ppf = row['best_ppf']
    ppf_delta = f"{'↓' if row['best_ppf'] < (baseline_ppf or 9.9) else '→'}{abs(row['best_ppf']-(baseline_ppf or row['best_ppf'])):.3f}" if baseline_ppf else ''
    print(f"  {row['arm']:<22} {row['label']:<28} "
          f"{row['best_ppf']:>8.4f} {row['cycle']:>8.1f} {row['sigma']:>8.4f} "
          f"{row['h_hist']:>7.3f} {row['injections']:>11} {row['hr_fired']:>9} "
          f"{row['al_cnt']:>7} {row['runtime']:>8.1f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('test_qica_summary_v2.csv', index=False)

print(f"\n  VERDICT GUIDE:")
print(f"  arm2 vs arm1 → H_hist AL alone:    {'+helps' if summary_rows[1]['best_ppf'] < summary_rows[0]['best_ppf'] else 'no benefit'} "
      f"(Δ={summary_rows[0]['best_ppf']-summary_rows[1]['best_ppf']:+.4f} PPF)")
print(f"  arm3 vs arm1 → injection alone:    {'+helps' if summary_rows[2]['best_ppf'] < summary_rows[0]['best_ppf'] else 'no benefit'} "
      f"(Δ={summary_rows[0]['best_ppf']-summary_rows[2]['best_ppf']:+.4f} PPF)  HR_fired={summary_rows[2]['hr_fired']} (expect 0, bug)")
print(f"  arm4 vs arm3 → HR fix value:       {'+helps' if summary_rows[3]['best_ppf'] < summary_rows[2]['best_ppf'] else 'no benefit'} "
      f"(Δ={summary_rows[2]['best_ppf']-summary_rows[3]['best_ppf']:+.4f} PPF)  HR_fired={summary_rows[3]['hr_fired']}")
print(f"  arm5 vs arm4 → H_traj value:       {'+helps' if summary_rows[4]['best_ppf'] < summary_rows[3]['best_ppf'] else 'no benefit'} "
      f"(Δ={summary_rows[3]['best_ppf']-summary_rows[4]['best_ppf']:+.4f} PPF)  (expect ≈0)")


# =============================================================================
# SECTION 12 — SAVE HISTORY + AL CANDIDATES
# =============================================================================

history_rows = []
for r in all_results:
    h = r['history']
    for i in range(len(h['gen'])):
        row = {'arm': r['arm']}
        for k, v in h.items():
            row[k] = v[i] if i < len(v) else None
        history_rows.append(row)

history_df = pd.DataFrame(history_rows)
history_df.to_csv('test_qica_history_v2.csv', index=False)

all_al = []
for r in all_results:
    all_al.extend(r['al_candidates'])
al_df_out = pd.DataFrame(all_al)
if len(al_df_out) > 0:
    al_df_out.to_csv('test_qica_al_candidates_v2.csv', index=False)
print(f"\n[SAVED] test_qica_summary_v2.csv  test_qica_history_v2.csv  test_qica_al_candidates_v2.csv")


# =============================================================================
# SECTION 13 — PLOTS
# =============================================================================

COLORS = ['#1B4FBF', '#F5A623', '#D62728', '#2CA02C', '#9467BD']
ARM_LABELS = [r['flags']['label'] for r in all_results]

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    f"QICA Ablation Test  |  {TEST_GENS} gens / {TEST_POP} pop / {TEST_MC} MC samples",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. Best PPF convergence ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :2])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['best_ppf'], color=COLORS[i], lw=2, label=r['flags']['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF (lower is better)')
ax.set_title('PPF Convergence (lower = better pattern found)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 2. Injection + reset events ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
yticks = []
for i, r in enumerate(all_results):
    h = r['history']
    gens = np.array(h['gen'])
    inj  = np.array(h['inj_fired'])
    rst  = np.array(h['reset_fired'])
    y    = i + 1
    ax.plot(gens, [y] * len(gens), color=COLORS[i], lw=0.5, alpha=0.3)
    inj_gens = gens[inj.astype(bool)]
    rst_gens = gens[rst.astype(bool)]
    if len(inj_gens): ax.scatter(inj_gens, [y]*len(inj_gens), marker='|', s=80, color=COLORS[i], zorder=5)
    if len(rst_gens): ax.scatter(rst_gens, [y]*len(rst_gens), marker='*', s=120, color='red', zorder=6)
    yticks.append((y, r['arm'].split('_', 1)[1][:12]))
ax.set_yticks([y for y, _ in yticks]); ax.set_yticklabels([l for _, l in yticks], fontsize=7)
ax.set_xlabel('Generation')
ax.set_title('Events: | = injection  ★ = hard reset\n(expect ★ only for arms 4+5)')
ax.grid(alpha=0.2)

# ── 3. Stagnation counter ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['stagnation_round'], color=COLORS[i], lw=1.5,
            label=r['flags']['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('Stagnation round')
ax.set_title('Escalation Level\n(arm 3 caps w/o reset; arms 4/5 reset)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 4. H_hist over time (best pattern) ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['best_h_hist'], color=COLORS[i], lw=1.5,
            label=r['flags']['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('H_hist (best pattern)')
ax.set_title('H_hist of Best Pattern\n(lower H → more confident prediction)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 5. H_traj over time (best pattern) ───────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['best_h_traj'], color=COLORS[i], lw=1.5,
            label=r['flags']['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('H_traj (best pattern)')
ax.set_title('H_traj of Best Pattern\n(expect near-constant → weak discriminator)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 6. AL candidates added per gen ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['al_total'], color=COLORS[i], lw=1.5,
            label=r['flags']['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('Cumulative AL candidates')
ax.set_title(f'AL Candidates Accumulated\n(all capped at {AL_TOP_K} in output)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 7. Hamming diversity ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 3])
for i, r in enumerate(all_results):
    h = r['history']
    ax.plot(h['gen'], h['ham_div'], color=COLORS[i], lw=1.5,
            label=r['flags']['label'])
ax.axhline(PATTERN_DIV_LOW, color='red', lw=1, ls='--', label=f'DIV_LOW={PATTERN_DIV_LOW}')
ax.set_xlabel('Generation'); ax.set_ylabel('Hamming diversity')
ax.set_title('Population Diversity\n(crosses red line → rev boost)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 8. AL candidate quality scatter (H_hist vs σ, colored by arm) ────────────
ax = fig.add_subplot(gs[2, :2])
for i, r in enumerate(all_results):
    al = r['al_candidates']
    if al:
        al_df_i = pd.DataFrame(al)
        ax.scatter(al_df_i['sigma_ppf'], al_df_i['h_hist'],
                   alpha=0.5, s=20, color=COLORS[i], label=r['flags']['label'])
ax.set_xlabel('σ_ppf (MC dropout)'); ax.set_ylabel('H_hist (calibrated)')
ax.set_title('AL Candidate Quality: H_hist vs σ\n(ideal: positively correlated — H_hist finds what σ finds)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# ── 9. Final bar comparison ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
names  = [r['arm'].replace('_', '\n') for r in all_results]
ppfs   = [r['elite'][0][2] if r['elite'] else 9.9 for r in all_results]
bars   = ax.bar(range(len(all_results)), ppfs, color=COLORS, edgecolor='k', lw=0.5)
ax.set_xticks(range(len(all_results))); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('Best PPF found')
ax.set_title('Best PPF per Arm\n(lower is better)')
for b, v in zip(bars, ppfs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
ax.grid(axis='y', alpha=0.3)

# ── 10. Injection count + hard reset count ────────────────────────────────────
ax = fig.add_subplot(gs[2, 3])
x   = np.arange(len(all_results))
w   = 0.35
inj_counts = [r['total_injections'] for r in all_results]
rst_counts  = [r['total_resets']    for r in all_results]
ax.bar(x - w/2, inj_counts, w, label='Injections', color='#1B4FBF', alpha=0.8)
ax.bar(x + w/2, rst_counts,  w, label='Hard resets', color='red',     alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('Count'); ax.legend(fontsize=8)
ax.set_title('Injection & Hard-Reset Events\n(arm3: resets=0 confirms bug)')
ax.grid(axis='y', alpha=0.3)

plt.savefig('test_qica_comparison_v2.png', dpi=150, bbox_inches='tight')
print("[SAVED] test_qica_comparison_v2.png")


# =============================================================================
# SECTION 14 — FINAL DIAGNOSTIC SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY — what each metric actually does in this run")
print("=" * 80)
for r in all_results:
    name = r['arm']
    diag = al_diag.get(name, {})
    best = r['elite'][0] if r['elite'] else (0, None, 9.9, 0.0, 0.0, 0.0, 0.0, 0.0)
    h    = r['history']

    # Injection effectiveness: did best PPF improve AFTER injection gens?
    inj_gens = [i for i, fired in enumerate(h['inj_fired']) if fired]
    inj_improvements = 0
    fit_improvements = 0   # NEW — injections target fitness, not PPF alone
    for ig in inj_gens:
        ppf_window = h['best_ppf'][ig:min(ig+10, len(h['best_ppf']))]
        if len(ppf_window) > 1 and min(ppf_window[1:]) < h['best_ppf'][ig]:
            inj_improvements += 1
        fit_window = h['best_fitness'][ig:min(ig+10, len(h['best_fitness']))]
        if len(fit_window) > 1 and max(fit_window[1:]) > h['best_fitness'][ig]:
            fit_improvements += 1

    # H_hist trend: does it drop as best pattern becomes more confident?
    h_hist_trend = 'decreasing' if len(h['best_h_hist']) > 5 and h['best_h_hist'][-1] < h['best_h_hist'][0] else 'flat/increasing'

    print(f"\n  {name} ({r['flags']['label']}):")
    print(f"    Best PPF          : {best[2]:.4f}  σ={best[4]:.4f}  H_hist={best[6]:.3f}")
    print(f"    Injections fired  : {r['total_injections']}  "
          f"(PPF improved next 10 gens: {inj_improvements}/{max(len(inj_gens),1) if inj_gens else 'N/A'}  |  "
          f"fitness improved next 10 gens: {fit_improvements}/{max(len(inj_gens),1) if inj_gens else 'N/A'})")
    print(f"    Hard resets fired : {r['total_resets']}{'  ← EXPECTED 0 (bug)' if name=='3_inject_unfixed' else ''}")
    print(f"    H_hist trend      : {h_hist_trend}  (want decreasing as search converges)")
    if diag:
        print(f"    AL H_hist mean    : {diag.get('hh_mean', 0):.3f}  CV={diag.get('hh_cv', 0):.3f}  σ-corr={diag.get('hh_corr', 0):.3f}")
        print(f"    AL H_traj CV      : {diag.get('ht_cv', 0):.4f}  (expect <0.05 → weak discriminator)")
        print(f"    Bimodal detected  : {diag.get('bimodal', 0)} patterns  (H_hist high but σ low)")
    print(f"    Runtime           : {r['runtime']:.0f}s")

print("\n" + "=" * 80)
print("SEND BACK: test_qica_summary_v2.csv  +  test_qica_comparison_v2.png")
print("  These tell us which methods are worth keeping in the final production code.")
print("=" * 80)