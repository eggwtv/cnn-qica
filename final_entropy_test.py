"""
=============================================================================
final_entropy_test.py  —  Definitive Entropy Method Comparison for QICA
=============================================================================
WHAT THIS SCRIPT DOES
  Runs a statistically reliable ablation over the 4 entropy/AL approaches
  that remain after your two pilot runs. Each arm runs with N_SEEDS different
  random seeds and the script reports mean ± std best_ppf across seeds —
  so you can finally see whether a "winner" is real signal or run-to-run noise.

WHAT CHANGED VS tests-on-qica-v2.py
  1.  MULTI-SEED LOOP (most important fix)
        Each arm runs N_SEEDS times. All verdicts are now mean±std, not
        one noisy point. Noise floor in pilot = ±0.23 PPF; a real effect
        should clear ±0.10 PPF once averaged over 3+ seeds.

  2.  4 CLEAN ARMS — the settled questions are dropped
        DROPPED: arm3 "bug demo" (hard-reset-never-fires) — confirmed
                 it outperforms no-reset not because the bug is good, but
                 because the 50% wipe in the "fix" is too destructive.
        DROPPED: arm4 "50%-wipe fix" — confirmed destructive.
        KEPT:    four arms that answer open questions (see below).

  3.  GENTLE HARD RESET ONLY (25% wipe + cooldown)
        The arms that enable hard reset now use HARD_RESET_FRAC=0.25 and
        HARD_RESET_COOLDOWN=15 throughout. The destructive 50% version
        is gone.

  4.  H_TRAJ SKIPPED UNLESS NEEDED (performance fix)
        evaluate_batch(..., compute_h_traj=False) skips the per-pattern
        SVD when the arm doesn't use H_traj. This is ~20-30% faster for
        the 3 arms that confirmed H_traj is a weak discriminator.

  5.  INJECTION DIAGNOSTIC FIXED
        v1/v2 reported "PPF improved" which was always 0 because injections
        target fitness (cycle - PPF penalties), not raw PPF. Now reports
        both PPF-window and fitness-window improvement separately.

  6.  COLORS CRASH FIXED
        v2 crashed with IndexError because COLORS had 5 entries for 6 arms.
        Now uses matplotlib colormap so it never goes out of range.

  7.  QUICK_TEST TOGGLE
        Set QUICK_TEST=True for a 15-min smoke-test (1 seed, 40 gens, 40 pop).
        Set QUICK_TEST=False for the real run (3 seeds, 250 gens, 80 pop).

ARMS
  A  sigma_baseline   σ-AL only  | no injection  | no H_hist
  B  h_hist_only      H_hist AL  | no injection  | no hard reset
  C  inject_no_hr     σ-AL       | injection ✓   | no hard reset  (was "best" in v1)
  D  h_hist_inject    H_hist AL  | injection ✓   | gentle HR ✓    (production candidate)

OPEN QUESTIONS EACH ARM ANSWERS
  B vs A  →  does H_hist AL alone reliably beat σ-AL?
  C vs A  →  does injection alone reliably improve search?
  D vs B  →  does adding injection+HR on top of H_hist help?
  D vs C  →  is H_hist worth adding when injection is already present?

INPUTS (must exist in same folder as this script)
  cnn_v9_model.keras      trained CNN surrogate
  cnn_v9_config.json      scaler parameters + output indices
  train_type_freq_v9.npy  per-position assembly-type frequencies
  ml_dataset_constrained.csv   training dataset (for seed seeding + calibration)

OUTPUTS
  entropy_test_summary.csv     one row per (arm × seed)
  entropy_test_means.csv       mean ± std per arm across seeds
  entropy_test_history.csv     gen-by-gen best_ppf for all runs
  entropy_test_al_candidates.csv  top AL candidates across all runs
  entropy_test_comparison.png  multi-panel comparison figure
=============================================================================
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(f"TensorFlow {tf.__version__}")
print("final_entropy_test.py  —  Definitive Entropy Method Comparison\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── Toggle: QUICK_TEST for smoke-test, False for real run ─────────────────────
QUICK_TEST = False   # True = ~15 min smoke-test  |  False = full multi-seed run

if QUICK_TEST:
    N_SEEDS   = 1
    TEST_GENS = 40
    TEST_POP  = 40
    TEST_MC   = 15
    print("[MODE] QUICK_TEST: 1 seed × 40 gens × 40 pop")
else:
    N_SEEDS   = 3
    TEST_GENS = 250
    TEST_POP  = 80
    TEST_MC   = 20
    print(f"[MODE] FULL RUN: {N_SEEDS} seeds × {TEST_GENS} gens × {TEST_POP} pop")

# Seeds to use across all arms (gives each arm an independent noise sample)
SEEDS = [42, 137, 271][:N_SEEDS]

MODEL_PATH  = next((p for p in ['cnn_v9_model.keras', 'cnn_v4_model.keras'] if os.path.exists(p)), None)
CONFIG_PATH = next((p for p in ['cnn_v9_config.json', 'cnn_v4_config.json'] if os.path.exists(p)), None)
TRUST_PATH  = next((p for p in ['train_type_freq_v9.npy', 'train_type_freq.npy'] if os.path.exists(p)), None)

# ── Arms: 4 comparisons that answer open questions ────────────────────────────
#
# Each flag dict controls exactly what that arm turns on/off.
# use_h_hist      → H_hist-calibrated histogram entropy for AL scoring vs σ-based
# use_injection   → stagnation injection + escalation
# use_hard_reset  → single-empire-safe GENTLE hard reset (25% wipe + cooldown)
# use_h_traj      → include H_traj in combined AL score  [confirmed weak, so OFF everywhere]
ARMS = [
    dict(name='A_sigma_baseline', use_h_hist=False, use_injection=False,
         use_hard_reset=False, use_h_traj=False,
         label='σ-AL baseline'),
    dict(name='B_h_hist_only',    use_h_hist=True,  use_injection=False,
         use_hard_reset=False, use_h_traj=False,
         label='H_hist AL only'),
    dict(name='C_inject_no_hr',   use_h_hist=False, use_injection=True,
         use_hard_reset=False, use_h_traj=False,
         label='σ-AL + Injection'),
    dict(name='D_h_hist_inject',  use_h_hist=True,  use_injection=True,
         use_hard_reset=True,  use_h_traj=False,
         label='H_hist + Inj + gentle HR'),
]

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
N_POS   = int(GRID_MASK.sum())   # 31
N_TYPES = 9

# ── QICA hyperparameters ──────────────────────────────────────────────────────
N_EMPIRES           = 6
ASSIMILATION_COEFF  = 0.35
REVOLUTION_RATE     = 0.35
REVOLUTION_MIN      = 0.08
REVOLUTION_MAX      = 0.65
REVOLUTION_BOOST    = 1.8
QUANTUM_TEMP_INIT   = 2.0
QUANTUM_TEMP_FINAL  = 0.1
ELITE_SIZE          = 12
ENTROPY_FREE_FRAC   = 0.65
PPF_LIMIT           = 3.5
W_PPF_PENALTY       = 80.0
W_PPF_SOFT          = 6.0
W_UNCERTAINTY       = 40.0
W_ENTROPY_BONUS     = 5.0
W_MONOTONICITY      = 10.0
STAGNATION_PATIENCE = 12      # slightly longer than v2 for 250-gen runs
STAGNATION_N_INJECT = 20
STAGNATION_N_ELITES = 3
STAGNATION_MAX_ESC  = 4
HARD_RESET_AT_CAP   = 3
STAGNATION_EPS      = 1e-3
PATTERN_DIV_LOW     = 0.35
HARD_RESET_FRAC     = 0.25   # gentle: only replace 25% of colonies
HARD_RESET_COOLDOWN = 15     # min gens between resets

# ── AL / histogram calibration ────────────────────────────────────────────────
HIST_BINS            = 10
SIGMA_HALF_WIDTH_MULT = 4.0
HIST_HALF_WIDTH_MIN  = 0.03
HIST_HALF_WIDTH_MAX  = 1.20
N_CAL_PATTERNS       = 40
N_CAL_MC_PASSES      = 15
AL_TOP_K             = 50
AL_PERCENTILE        = 70
AL_LIVE_CAP          = 150

# Mutable globals — set per-run during calibration
TYPICAL_SIGMA: float   = 0.07
HIST_HALF_WIDTH: float = 0.28


# =============================================================================
# SECTION 2 — MODEL + TRUST REGION LOAD
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
        cfg.update({'filters': self._filters, 'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


for p in [MODEL_PATH, CONFIG_PATH]:
    if p is None or not os.path.exists(p):
        print(f"[ERROR] Missing model/config — run cnn_v9.py first."); sys.exit(1)

model = keras.models.load_model(
    MODEL_PATH, compile=False,
    custom_objects={'ConvResBlock': ConvResBlock}
)
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

# Trust region
if TRUST_PATH and os.path.exists(TRUST_PATH):
    type_freq = np.load(TRUST_PATH).astype(np.float32)
    print(f"[TRUST] {TRUST_PATH}  shape={type_freq.shape}")
else:
    type_freq = np.ones((N_POS, N_TYPES), dtype=np.float32) / N_TYPES
    print("[TRUST] Not found — using uniform fallback.")

def _compute_trust_region(freq, free_frac=ENTROPY_FREE_FRAC):
    h_pos      = -np.sum(freq * np.log(freq + 1e-10), axis=1).astype(np.float32)
    n_free     = max(1, int(np.round(N_POS * free_frac)))
    rank       = np.argsort(h_pos)[::-1]
    free_mask  = np.zeros(N_POS, dtype=bool)
    free_mask[rank[:n_free]] = True
    fixed_types = (np.argmax(freq, axis=1) + 1).astype(np.int32)
    return free_mask, fixed_types, n_free

free_mask, fixed_types, n_free = _compute_trust_region(type_freq)
print(f"[TRUST] {n_free}/{N_POS} positions free\n")


# =============================================================================
# SECTION 3 — CALIBRATION + SEED LOAD
# =============================================================================

X_train_seed = None
ppf_cnn_seed = None

def _calibrate_and_seed():
    global X_train_seed, ppf_cnn_seed, TYPICAL_SIGMA, HIST_HALF_WIDTH
    csv_path = 'ml_dataset_constrained.csv'
    if not os.path.exists(csv_path):
        print("[CAL] ml_dataset_constrained.csv not found — skipping seed load")
        return

    df  = pd.read_csv(csv_path, skiprows=1, engine='python', on_bad_lines='skip')
    lc  = [f'loading_{i}' for i in range(N_POS)]
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
        sc = model(tf.constant(grids[i:i+128], dtype=tf.int32),
                   training=False).numpy()
        ppf_preds.extend(
            (sc[:, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]).tolist()
        )
    ppf_arr = np.array(ppf_preds, dtype=np.float32)
    X_train_seed = X_raw
    ppf_cnn_seed = ppf_arr
    print(f"[SEED] {N} patterns  |  PPF {ppf_arr.min():.3f}–{ppf_arr.max():.3f}")

    # Calibrate histogram half-width from real MC dropout noise
    n_cal  = min(N_CAL_PATTERNS, N)
    cal_idx = np.random.choice(N, n_cal, replace=False)
    mc_sc  = np.stack([
        model(tf.constant(grids[cal_idx], dtype=tf.int32),
              training=True).numpy()[:, IDX_PPF_MAX]
        for _ in range(N_CAL_MC_PASSES)
    ])
    mc_phys  = mc_sc * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
    sigmas   = mc_phys.std(axis=0)
    typ_sig  = float(np.median(sigmas))
    p95_sig  = float(np.percentile(sigmas, 95))
    TYPICAL_SIGMA   = max(typ_sig, 1e-4)
    raw_hw          = SIGMA_HALF_WIDTH_MULT * max(typ_sig, 0.6 * p95_sig)
    HIST_HALF_WIDTH = float(np.clip(raw_hw, HIST_HALF_WIDTH_MIN, HIST_HALF_WIDTH_MAX))
    print(f"[CAL]  median σ={typ_sig:.4f}  p95 σ={p95_sig:.4f}  "
          f"→  hist half-width=±{HIST_HALF_WIDTH:.4f}\n")

_calibrate_and_seed()


# =============================================================================
# SECTION 4 — ENTROPY UTILITIES
# =============================================================================

def gaussian_entropy(mc_ppf: np.ndarray) -> np.ndarray:
    """H_gauss = 0.5 * log(2πe σ²).  mc_ppf: (MC, B) → (B,)"""
    sigma = mc_ppf.std(axis=0) + 1e-10
    return (0.5 * np.log(2.0 * np.pi * np.e * sigma ** 2)).astype(np.float32)


def histogram_entropy(mc_ppf: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """
    Calibrated per-pattern-centred histogram Shannon H.

    WHY THIS BEATS σ-BASED SCORING:
      σ assumes unimodal (Gaussian) dropout distributions. For patterns
      near the PPF_LIMIT boundary, the network's dropout can produce
      BIMODAL distributions (some passes land below the limit, some above).
      A bimodal distribution with small σ can still have high H_hist,
      correctly flagging it as a pattern worth simulating. σ would miss it.

      H_hist also uses absolute calibration (HIST_HALF_WIDTH derived from
      the real model's noise level) so the score is comparable across
      different model versions without re-tuning.

    mc_ppf: (MC, B)  →  (B,) entropy values
    """
    _, B = mc_ppf.shape
    entropies = np.zeros(B, dtype=np.float32)
    centred   = mc_ppf - mc_ppf.mean(axis=0, keepdims=True)
    lo, hi    = -HIST_HALF_WIDTH, HIST_HALF_WIDTH
    for b in range(B):
        hist, _ = np.histogram(centred[:, b], bins=bins, range=(lo, hi))
        p = hist.astype(np.float64) / (hist.sum() + 1e-12)
        p = p[p > 0]
        entropies[b] = float(-np.sum(p * np.log(p)))
    return entropies


def trajectory_entropy(mc_curves: np.ndarray) -> np.ndarray:
    """
    Multivariate Gaussian H over full PPF burnup curve.
    mc_curves: (MC, B, K) → (B,)

    VERDICT (from v1/v2 ablation): CV=0.02–0.04, H-σ corr=-0.04.
    This is a WEAK discriminator — not worth the SVD cost.
    Left here for completeness; called only when use_h_traj=True.
    """
    MC, B, K  = mc_curves.shape
    entropies = np.zeros(B, dtype=np.float32)
    eff_rank  = min(MC - 1, K)
    for b in range(B):
        X   = mc_curves[:, b, :].astype(np.float64)
        X_c = X - X.mean(axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)
            eigvals = np.maximum((s[:eff_rank] ** 2) / max(MC - 1, 1), 1e-12)
            H = 0.5 * (eff_rank * np.log(2.0 * np.pi * np.e)
                       + np.sum(np.log(eigvals)))
        except np.linalg.LinAlgError:
            H = 0.0
        entropies[b] = float(H)
    return entropies


def hamming_diversity(empires: list, max_sample: int = 60) -> float:
    """Mean pairwise Hamming distance across free positions."""
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
    n   = arr.shape[0]
    if n > max_sample:
        arr = arr[np.random.choice(n, max_sample, replace=False)]
        n   = max_sample
    total, count = 0.0, 0
    for i in range(n):
        diffs  = (arr[i+1:] != arr[i]).mean(axis=1)
        total += float(diffs.sum()); count += diffs.shape[0]
    return total / count if count > 0 else 1.0


# =============================================================================
# SECTION 5 — BATCH EVALUATION
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


def evaluate_batch(patterns: np.ndarray, mc_n: int = TEST_MC,
                   compute_h_traj: bool = False) -> dict:
    """
    MC-dropout batch evaluation. compute_h_traj=False by default — the SVD
    is expensive (~30% of eval time) and H_traj was confirmed a weak
    discriminator in both pilot runs. Set True only when testing it.
    """
    if patterns.ndim == 1:
        patterns = patterns.reshape(1, -1)
    B    = patterns.shape[0]
    grid = pattern_to_grid(patterns)
    X_tf = tf.constant(grid, dtype=tf.int32)

    mc_sc    = np.stack([model(X_tf, training=True).numpy() for _ in range(mc_n)])
    mean_sc  = mc_sc.mean(axis=0)
    std_sc   = mc_sc.std(axis=0)

    mean_real = inverse_transform(mean_sc)
    std_real  = np.concatenate([
        std_sc[:, :34] * ym_scale,
        std_sc[:, 34:35] * yr_scale,
    ], axis=1)

    ppf_mean   = mean_real[:, IDX_PPF_MAX]
    ppf_std    = std_real[:, IDX_PPF_MAX]
    cycle_mean = mean_real[:, IDX_CYCLE]
    ppf_steps  = mean_real[:, IDX_STEPS_S:IDX_STEPS_E]

    mc_ppf_phys = (mc_sc[:, :, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX]
                   + ym_mean[IDX_PPF_MAX])

    h_gauss = gaussian_entropy(mc_ppf_phys)
    h_hist  = histogram_entropy(mc_ppf_phys)

    if compute_h_traj:
        mc_curves = (mc_sc[:, :, IDX_STEPS_S:IDX_STEPS_E]
                     * ym_scale[IDX_STEPS_S:IDX_STEPS_E]
                     + ym_mean[IDX_STEPS_S:IDX_STEPS_E])
        h_traj = trajectory_entropy(mc_curves)
    else:
        h_traj = np.zeros(B, dtype=np.float32)

    late       = ppf_steps[:, 3:]
    diffs      = late[:, 1:] - late[:, :-1]
    mono_bonus = W_MONOTONICITY * (
        1.0 - (diffs > 0).sum(axis=1) / max(late.shape[1] - 1, 1)
    )
    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    fitness    = (cycle_mean
                  - W_PPF_SOFT    * ppf_mean
                  - W_PPF_PENALTY * ppf_excess
                  - W_UNCERTAINTY * ppf_std
                  + mono_bonus)

    return {
        'ppf_mean':   ppf_mean,
        'ppf_std':    ppf_std,
        'cycle_mean': cycle_mean,
        'fitness':    fitness,
        'h_gauss':    h_gauss,
        'h_hist':     h_hist,
        'h_traj':     h_traj,
    }


# =============================================================================
# SECTION 6 — QUANTUM COUNTRY + EMPIRE
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
        # Clamp fixed positions
        for p in range(N_POS):
            if not free_mask[p]:
                self.q_state[p] = 0.0
                self.q_state[p, fixed_types[p] - 1] = 1.0
        self.measured   = None
        self.fitness    = -np.inf; self.ppf_mean   = 9.0
        self.ppf_std    = 0.0;     self.cycle_mean = 0.0
        self.h_gauss    = -10.0;   self.h_hist     = 0.0
        self.h_traj     = 0.0;     self.al_score   = 0.0

    def collapse(self, temperature: float = 1.0) -> np.ndarray:
        logits = np.log(self.q_state + 1e-10) / max(temperature, 0.01)
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits) / (np.exp(logits).sum(axis=1, keepdims=True) + 1e-10)
        self.measured = np.array(
            [np.random.choice(N_TYPES, p=probs[i]) + 1 for i in range(N_POS)],
            dtype=np.int32
        )
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
                self.q_state[p] = 0.0; self.q_state[p, fixed_types[p] - 1] = 1.0

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
# SECTION 7 — QICA CORE
# =============================================================================

class EntropyQICA:
    """
    QICA parametrised by arm flags.
    The only difference between arms is which AL score is used and whether
    injection/hard-reset are active — the search mechanics are identical.
    """

    def __init__(self, flags: dict, arm_name: str, seed: int,
                 n_pop: int = TEST_POP, max_gen: int = TEST_GENS,
                 mc_n: int = TEST_MC):
        self.flags    = flags
        self.arm_name = arm_name
        self.seed     = seed
        self.n_pop    = n_pop
        self.max_gen  = max_gen
        self.mc_n     = mc_n

        self.elite_archive  = []
        self.al_candidates  = []
        self._al_seen       = set()

        self.stagnation_count  = 0
        self.stagnation_rounds = 0
        self.rounds_at_cap     = 0
        self.last_best_fitness = None
        self.best_fitness_ever = -np.inf
        self.last_reset_gen    = -(10 ** 9)

        self.total_injections = 0
        self.total_resets     = 0

        self.history = {
            'gen': [], 'best_ppf': [], 'best_cycle': [], 'best_sigma': [],
            'best_h_hist': [], 'mean_h_hist': [], 'mean_ppf_std': [],
            'al_total': [], 'al_added_this_gen': [],
            'inj_fired': [], 'reset_fired': [],
            'ham_div': [], 'rev_rate': [], 'n_empires': [], 'temp': [],
            'stagnation_round': [], 'best_fitness': [],
        }

    # ── Schedule helpers ───────────────────────────────────────────────────────
    def _temperature(self, gen: int) -> float:
        r = gen / self.max_gen
        return QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** r

    def _base_rev_rate(self, gen: int) -> float:
        r = gen / self.max_gen
        return REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * r

    def _adaptive_rev_rate(self, base: float, ham_div: float) -> float:
        if ham_div < PATTERN_DIV_LOW:
            sev = 1.0 - ham_div / PATTERN_DIV_LOW
            return float(min(base * (1.0 + REVOLUTION_BOOST * sev), REVOLUTION_MAX))
        return float(base)

    # ── Population initialisation ──────────────────────────────────────────────
    def _init_population(self) -> list:
        countries = []
        # One bias-to-each-type country for exploration breadth
        for bias_t in range(1, N_TYPES + 1):
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.04
            q[:, bias_t - 1] = 0.68
            q /= q.sum(axis=1, keepdims=True)
            countries.append(QuantumCountry(q))
        # Top-K training-data seeds (exploit known-good patterns)
        if X_train_seed is not None:
            n_seeds = min(8, len(X_train_seed))
            top_idx = np.argsort(ppf_cnn_seed)[:n_seeds]
            for idx in top_idx:
                pat = X_train_seed[idx]
                q   = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
                for p in range(N_POS):
                    t = int(pat[p])
                    if 1 <= t <= N_TYPES:
                        q[p, t - 1] = 0.84
                q /= q.sum(axis=1, keepdims=True)
                countries.append(QuantumCountry(q))
        # Fill the rest uniformly
        while len(countries) < self.n_pop:
            countries.append(QuantumCountry())
        return countries

    # ── Evaluate + AL scoring ──────────────────────────────────────────────────
    def _evaluate_all(self, countries: list, temperature: float):
        if not countries:
            return countries, 0
        patterns = np.stack([c.collapse(temperature) for c in countries])
        result   = evaluate_batch(patterns, mc_n=self.mc_n,
                                   compute_h_traj=self.flags['use_h_traj'])

        ppf_arr    = result['ppf_mean']
        ppf_std    = result['ppf_std']
        h_hist_arr = result['h_hist']
        ppf_5pct   = np.percentile(ppf_arr, 5)

        # ── AL scoring: the key variable between arms ─────────────────────────
        # σ-based (baseline/inject-no-hr): z-score of MC dropout σ_ppf
        # H_hist-based: z-score of calibrated histogram entropy
        # H_hist is preferred because it detects bimodal distributions that
        # σ-scoring misses (patterns near the PPF_LIMIT where the NN is
        # genuinely uncertain whether the pattern clears the limit or not).
        if self.flags['use_h_hist']:
            al_scores = (
                (h_hist_arr - h_hist_arr.mean()) / (h_hist_arr.std() + 1e-8)
            ).astype(np.float32)
        else:
            al_scores = (
                (ppf_std - ppf_std.mean()) / (ppf_std.std() + 1e-8)
            ).astype(np.float32)

        al_thr = max(1e-3, float(np.percentile(al_scores, AL_PERCENTILE)))

        ent_bonus = np.array([
            W_ENTROPY_BONUS * c.q_entropy() / (N_POS * N_TYPES)
            for c in countries
        ], dtype=np.float32)

        fitness_arr = result['fitness'] + ent_bonus

        al_added = 0
        for i, c in enumerate(countries):
            c.fitness    = float(fitness_arr[i])
            c.ppf_mean   = float(ppf_arr[i])
            c.ppf_std    = float(ppf_std[i])
            c.cycle_mean = float(result['cycle_mean'][i])
            c.h_gauss    = float(result['h_gauss'][i])
            c.h_hist     = float(h_hist_arr[i])
            c.al_score   = float(al_scores[i])

            if al_scores[i] >= al_thr and ppf_arr[i] <= ppf_5pct:
                pat_key = tuple(c.measured.tolist())
                if pat_key not in self._al_seen:
                    self._al_seen.add(pat_key)
                    priority = float(
                        c.al_score * c.cycle_mean / (c.ppf_mean + 1e-6)
                    )
                    self.al_candidates.append({
                        'arm': self.arm_name,
                        'seed': self.seed,
                        'pattern': c.measured.tolist(),
                        'pred_ppf': c.ppf_mean, 'sigma_ppf': c.ppf_std,
                        'h_hist': c.h_hist, 'h_gauss': c.h_gauss,
                        'al_score': c.al_score,
                        'cycle': c.cycle_mean, 'priority': priority,
                    })
                    al_added += 1
                    if len(self.al_candidates) > AL_LIVE_CAP:
                        self.al_candidates.sort(
                            key=lambda d: d['priority'], reverse=True)
                        kept = self.al_candidates[:AL_LIVE_CAP // 2]
                        self.al_candidates = kept
                        self._al_seen = {tuple(d['pattern']) for d in kept}

        return countries, al_added

    # ── Empire mechanics ───────────────────────────────────────────────────────
    def _form_empires(self, countries: list) -> list:
        sorted_c = sorted(countries, key=lambda c: c.fitness, reverse=True)
        n_emp    = min(N_EMPIRES, len(sorted_c))
        imps, cols = sorted_c[:n_emp], sorted_c[n_emp:]
        fits     = np.array([i.fitness for i in imps])
        fits_sh  = fits - fits.min() + 1e-6
        powers   = fits_sh / fits_sh.sum()
        counts   = np.round(powers * len(cols)).astype(int)
        diff     = len(cols) - counts.sum()
        if diff > 0: counts[np.argmax(powers)] += diff
        elif diff < 0: counts[np.argmax(counts)] += diff
        empires, idx = [], 0
        for i, imp in enumerate(imps):
            empires.append(Empire(imp, list(cols[idx:idx + counts[i]])))
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
            best_i = max(range(len(emp.colonies)),
                         key=lambda i: emp.colonies[i].fitness)
            if emp.colonies[best_i].fitness > emp.imperialist.fitness:
                emp.imperialist, emp.colonies[best_i] = \
                    emp.colonies[best_i], emp.imperialist

    def _empire_collapse(self, empires) -> list:
        if len(empires) <= 1: return empires
        wi = min(range(len(empires)), key=lambda i: empires[i].power)
        si = max(range(len(empires)), key=lambda i: empires[i].power)
        if len(empires[wi].colonies) == 0:
            empires[si].colonies.append(empires[wi].imperialist)
            empires.pop(wi)
        else:
            wci = min(range(len(empires[wi].colonies)),
                      key=lambda i: empires[wi].colonies[i].fitness)
            empires[si].colonies.append(empires[wi].colonies.pop(wci))
        return empires

    def _update_elite(self, empires):
        for emp in empires:
            for c in [emp.imperialist] + emp.colonies:
                if c.measured is not None:
                    self.elite_archive.append((
                        c.fitness, c.measured.copy(),
                        c.ppf_mean, c.cycle_mean, c.ppf_std,
                        c.h_gauss, c.h_hist,
                    ))
        self.elite_archive.sort(key=lambda x: x[0], reverse=True)
        seen, unique = set(), []
        for e in self.elite_archive:
            k = tuple(e[1])
            if k not in seen: seen.add(k); unique.append(e)
        self.elite_archive = unique[:ELITE_SIZE]

    # ── Stagnation injection ───────────────────────────────────────────────────
    def _inject_mutations(self, empires, temperature, round_idx: int):
        if not self.elite_archive or not self.flags['use_injection']:
            return None
        esc      = min(round_idx - 1, STAGNATION_MAX_ESC)
        n_elites = min(STAGNATION_N_ELITES, len(self.elite_archive))
        n_inject = int(STAGNATION_N_INJECT * (1.0 + 0.5 * esc))
        new_c    = []
        for i in range(n_inject):
            seed_pat = self.elite_archive[i % n_elites][1]
            lo_mut   = min(5 + 2 * esc, n_free)
            hi_mut   = min(14 + 4 * esc, n_free)
            if hi_mut <= lo_mut: hi_mut = lo_mut + 1
            n_mutate = np.random.randint(lo_mut, hi_mut + 1)
            mut_pos  = np.random.choice(
                [p for p in range(N_POS) if free_mask[p]],
                min(n_mutate, n_free), replace=False)
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
            for p, t in enumerate(seed_pat): q[p, int(t) - 1] = 0.84
            q /= q.sum(axis=1, keepdims=True)
            for p in mut_pos:
                q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
            new_c.append(QuantumCountry(q))

        boosted_temp = min(temperature * (2.5 + 0.5 * esc), 2.0)
        patterns = np.stack([c.collapse(boosted_temp) for c in new_c])
        result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
        for i, c in enumerate(new_c):
            c.fitness    = float(result['fitness'][i])
            c.ppf_mean   = float(result['ppf_mean'][i])
            c.ppf_std    = float(result['ppf_std'][i])
            c.h_hist     = float(result['h_hist'][i])
            c.cycle_mean = float(result['cycle_mean'][i])

        largest = max(range(len(empires)),
                      key=lambda i: empires[i].total_countries)
        empires[largest].colonies.extend(new_c)
        self.stagnation_count = 0
        self.total_injections += 1
        print(f"  [{self.arm_name}|s{self.seed}] ★ INJ round={round_idx} "
              f"esc={esc} n={n_inject} | "
              f"best_ppf={self.elite_archive[0][2]:.3f}")
        return esc

    # ── Gentle hard reset ──────────────────────────────────────────────────────
    def _hard_reset(self, empires, temperature, gen: int) -> bool:
        """
        GENTLE VERSION (25% wipe + cooldown):
          - Only replaces HARD_RESET_FRAC (0.25) of the single empire's colonies
          - Won't fire again for HARD_RESET_COOLDOWN gens
          - Seeds replacements from mid-archive elites + random to widen search
        This avoids the 50%-wipe destruction that made arm4 underperform arm3
        (bug-present) in both v1 and v2 ablation runs.
        """
        if not self.flags['use_hard_reset']: return False
        if not self.flags['use_injection']: return False
        if gen - self.last_reset_gen < HARD_RESET_COOLDOWN: return False
        if not self.elite_archive: return False

        if len(empires) == 1:
            emp = empires[0]
            if len(emp.colonies) < 4: return False
            emp.colonies.sort(key=lambda c: c.fitness)
            n_reinit = max(2, int(round(len(emp.colonies) * HARD_RESET_FRAC)))
            n_mid    = max(0, len(self.elite_archive) - 2)
            new_c    = []
            for i in range(n_reinit):
                if n_mid > 0 and i % 2 == 0:
                    seed_pat = self.elite_archive[2 + (i % n_mid)][1]
                    q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.03
                    for p, t in enumerate(seed_pat): q[p, int(t) - 1] = 0.7
                    q /= q.sum(axis=1, keepdims=True)
                    n_heavy = min(n_free, max(8, n_free // 2))
                    heavy   = np.random.choice(
                        [p for p in range(N_POS) if free_mask[p]],
                        n_heavy, replace=False)
                    for p in heavy:
                        q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                    new_c.append(QuantumCountry(q))
                else:
                    new_c.append(QuantumCountry())

            patterns = np.stack([c.collapse(1.6) for c in new_c])
            result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
            for i, c in enumerate(new_c):
                c.fitness    = float(result['fitness'][i])
                c.ppf_mean   = float(result['ppf_mean'][i])
                c.ppf_std    = float(result['ppf_std'][i])
                c.h_hist     = float(result['h_hist'][i])
                c.cycle_mean = float(result['cycle_mean'][i])

            emp.colonies[:n_reinit] = new_c
            self.total_resets += 1
            self.last_reset_gen = gen
            print(f"  [{self.arm_name}|s{self.seed}] ⚡ HR: reinit worst "
                  f"{n_reinit}/{len(emp.colonies)} colonies (cooldown={HARD_RESET_COOLDOWN}g)")
            return True

        else:
            # Multiple empires: replace weakest empire
            wi    = min(range(len(empires)), key=lambda i: empires[i].power)
            n_new = max(8, empires[wi].total_countries)
            n_mid = max(0, len(self.elite_archive) - 2)
            new_c = []
            for i in range(n_new):
                if n_mid > 0 and i % 2 == 0:
                    seed_pat = self.elite_archive[2 + (i % n_mid)][1]
                    q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.03
                    for p, t in enumerate(seed_pat): q[p, int(t) - 1] = 0.7
                    q /= q.sum(axis=1, keepdims=True)
                    n_heavy = min(n_free, max(8, n_free // 2))
                    heavy   = np.random.choice(
                        [p for p in range(N_POS) if free_mask[p]],
                        n_heavy, replace=False)
                    for p in heavy:
                        q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                    new_c.append(QuantumCountry(q))
                else:
                    new_c.append(QuantumCountry())
            patterns = np.stack([c.collapse(1.6) for c in new_c])
            result   = evaluate_batch(patterns, mc_n=self.mc_n, compute_h_traj=False)
            for i, c in enumerate(new_c):
                c.fitness    = float(result['fitness'][i])
                c.ppf_mean   = float(result['ppf_mean'][i])
                c.ppf_std    = float(result['ppf_std'][i])
                c.h_hist     = float(result['h_hist'][i])
                c.cycle_mean = float(result['cycle_mean'][i])
            empires[wi] = Empire(new_c[0], new_c[1:])
            self.total_resets += 1
            self.last_reset_gen = gen
            print(f"  [{self.arm_name}|s{self.seed}] ⚡ HR: replaced empire #{wi}")
            return True

    # ── Per-generation log ─────────────────────────────────────────────────────
    def _log(self, gen, empires, temp, ham_div, rev_rate,
             al_added, inj_fired, reset_fired):
        best = (self.elite_archive[0] if self.elite_archive
                else (0, None, 9.0, 0.0, 0.0, -10.0, 0.0))
        all_c    = [e.imperialist for e in empires] + [c for e in empires for c in e.colonies]
        mean_hh  = float(np.mean([c.h_hist for c in all_c]))
        mean_sig = float(np.mean([c.ppf_std for c in all_c]))

        self.history['gen'].append(gen)
        self.history['best_ppf'].append(float(best[2]))
        self.history['best_cycle'].append(float(best[3]))
        self.history['best_sigma'].append(float(best[4]))
        self.history['best_h_hist'].append(float(best[6]))
        self.history['mean_h_hist'].append(mean_hh)
        self.history['mean_ppf_std'].append(mean_sig)
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

        log_every = max(1, self.max_gen // 5)
        if gen % log_every == 0 or gen == self.max_gen:
            flags_str = ''
            if inj_fired: flags_str += ' ★INJ'
            if reset_fired: flags_str += ' ⚡HR'
            print(
                f"  [{self.arm_name}|s{self.seed}] "
                f"Gen {gen:4d}/{self.max_gen} | "
                f"ppf={best[2]:.4f} σ={best[4]:.4f} | "
                f"H_hist={best[6]:.3f} | "
                f"AL+{al_added}(={len(self.al_candidates)}) | "
                f"div={ham_div:.2f} emp={len(empires)} "
                f"stag={self.stagnation_rounds}{flags_str}"
            )

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self) -> dict:
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        t0 = time.time()

        print(f"\n  [{self.arm_name}|s{self.seed}] "
              f"START  {self.flags['label']}  "
              f"H_hist={self.flags['use_h_hist']}  "
              f"Inject={self.flags['use_injection']}  "
              f"HR={self.flags['use_hard_reset']}")

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

            inj_fired = False; reset_fired = False
            if self.elite_archive:
                cur_fit = self.elite_archive[0][0]
                if cur_fit > self.best_fitness_ever + STAGNATION_EPS:
                    self.best_fitness_ever = cur_fit
                    self.stagnation_rounds = 0
                    self.rounds_at_cap     = 0
                if self.last_best_fitness is not None:
                    self.stagnation_count = (
                        self.stagnation_count + 1
                        if abs(cur_fit - self.last_best_fitness) < 0.05
                        else 0
                    )
                self.last_best_fitness = cur_fit

                if self.flags['use_injection'] and self.stagnation_count >= STAGNATION_PATIENCE:
                    self.stagnation_rounds += 1
                    esc = self._inject_mutations(empires, temp,
                                                  round_idx=self.stagnation_rounds)
                    inj_fired = True
                    self.rounds_at_cap = (
                        self.rounds_at_cap + 1
                        if esc is not None and esc >= STAGNATION_MAX_ESC
                        else 0
                    )
                    if self.rounds_at_cap >= HARD_RESET_AT_CAP:
                        fired = self._hard_reset(empires, temp, gen)
                        if fired:
                            reset_fired = True
                            self.rounds_at_cap = 0

            imps = [emp.imperialist for emp in empires]
            _, al_imp = self._evaluate_all(imps, temp)
            al_added  = al_col + al_imp
            self._log(gen, empires, temp, ham_div, rev_rate,
                      al_added, inj_fired, reset_fired)

        runtime = time.time() - t0
        self.al_candidates.sort(key=lambda d: d['priority'], reverse=True)
        self.al_candidates = self.al_candidates[:AL_TOP_K]

        best = (self.elite_archive[0] if self.elite_archive
                else (0, None, 9.0, 0.0, 0.0, -10.0, 0.0))
        print(f"  [{self.arm_name}|s{self.seed}] DONE  "
              f"best_ppf={best[2]:.4f}  cycle={best[3]:.1f}d  "
              f"σ={best[4]:.4f}  {runtime:.0f}s  "
              f"inj={self.total_injections} hr={self.total_resets}")

        return {
            'arm': self.arm_name, 'label': self.flags['label'],
            'seed': self.seed, 'flags': self.flags,
            'best_ppf': float(best[2]), 'best_cycle': float(best[3]),
            'best_sigma': float(best[4]), 'best_h_hist': float(best[6]),
            'elite': self.elite_archive,
            'history': self.history,
            'al_candidates': self.al_candidates,
            'runtime': runtime,
            'total_injections': self.total_injections,
            'total_resets': self.total_resets,
        }


# =============================================================================
# SECTION 8 — MAIN LOOP: RUN ALL ARMS × ALL SEEDS
# =============================================================================

all_results = []   # one dict per (arm × seed)

t_total = time.time()
for arm_cfg in ARMS:
    arm_results = []
    for seed in SEEDS:
        qica = EntropyQICA(
            flags=arm_cfg, arm_name=arm_cfg['name'],
            seed=seed, n_pop=TEST_POP, max_gen=TEST_GENS, mc_n=TEST_MC
        )
        r = qica.run()
        arm_results.append(r)
        all_results.append(r)

    # Print per-arm summary across seeds so you can see convergence as it runs
    ppfs = [r['best_ppf'] for r in arm_results]
    print(f"\n  ── {arm_cfg['name']} ({N_SEEDS} seeds) ──")
    print(f"     best_ppf = {np.mean(ppfs):.4f} ± {np.std(ppfs):.4f}  "
          f"[{np.min(ppfs):.4f} – {np.max(ppfs):.4f}]")

print(f"\n[TOTAL RUNTIME] {(time.time()-t_total)/60:.1f} min  "
      f"({len(ARMS)} arms × {N_SEEDS} seeds × {TEST_GENS} gens)\n")


# =============================================================================
# SECTION 9 — AGGREGATE STATISTICS  (mean ± std per arm across seeds)
# =============================================================================

print("\n" + "=" * 90)
print("MULTI-SEED RESULTS  (mean ± std across seeds — this is the verdict to trust)")
print("=" * 90)
print(f"  {'Arm':<24} {'Label':<30} {'mean_PPF':>10} {'std_PPF':>9} "
      f"{'min_PPF':>9} {'mean_σ':>8} {'mean_inj':>9} {'mean_hr':>8}")
print(f"  {'-'*24} {'-'*30} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")

arm_means = {}
mean_rows = []
for arm_cfg in ARMS:
    arm_r = [r for r in all_results if r['arm'] == arm_cfg['name']]
    ppfs  = np.array([r['best_ppf'] for r in arm_r])
    sigs  = np.array([r['best_sigma'] for r in arm_r])
    injs  = np.array([r['total_injections'] for r in arm_r])
    hrs   = np.array([r['total_resets'] for r in arm_r])
    row = {
        'arm':        arm_cfg['name'],
        'label':      arm_cfg['label'],
        'mean_ppf':   float(ppfs.mean()),
        'std_ppf':    float(ppfs.std()),
        'min_ppf':    float(ppfs.min()),
        'mean_sigma': float(sigs.mean()),
        'mean_inj':   float(injs.mean()),
        'mean_hr':    float(hrs.mean()),
    }
    arm_means[arm_cfg['name']] = row
    mean_rows.append(row)
    print(f"  {row['arm']:<24} {row['label']:<30} "
          f"{row['mean_ppf']:>10.4f} {row['std_ppf']:>9.4f} "
          f"{row['min_ppf']:>9.4f} {row['mean_sigma']:>8.4f} "
          f"{row['mean_inj']:>9.1f} {row['mean_hr']:>8.1f}")

# Verdict comparisons
A = arm_means.get('A_sigma_baseline', {})
B = arm_means.get('B_h_hist_only',    {})
C = arm_means.get('C_inject_no_hr',   {})
D = arm_means.get('D_h_hist_inject',  {})

noise_floor = max(r.get('std_ppf', 0.0) for r in mean_rows)

print(f"\n  VERDICTS (noise floor ≈ ±{noise_floor:.3f} PPF from seed variance):")
if A and B:
    delta_ba = A['mean_ppf'] - B['mean_ppf']
    sig = "REAL SIGNAL ✓" if abs(delta_ba) > noise_floor else "within noise"
    print(f"  B vs A  H_hist AL vs σ-AL baseline: Δ={delta_ba:+.4f} PPF  → {sig}")
if A and C:
    delta_ca = A['mean_ppf'] - C['mean_ppf']
    sig = "REAL SIGNAL ✓" if abs(delta_ca) > noise_floor else "within noise"
    print(f"  C vs A  Injection vs no-injection  : Δ={delta_ca:+.4f} PPF  → {sig}")
if B and D:
    delta_db = B['mean_ppf'] - D['mean_ppf']
    sig = "REAL SIGNAL ✓" if abs(delta_db) > noise_floor else "within noise"
    print(f"  D vs B  Adding inj+HR to H_hist    : Δ={delta_db:+.4f} PPF  → {sig}")
if C and D:
    delta_dc = C['mean_ppf'] - D['mean_ppf']
    sig = "REAL SIGNAL ✓" if abs(delta_dc) > noise_floor else "within noise"
    print(f"  D vs C  H_hist on top of injection : Δ={delta_dc:+.4f} PPF  → {sig}")

print()

best_arm_row = min(mean_rows, key=lambda r: r['mean_ppf'])
print(f"  ★ BEST ARM: {best_arm_row['arm']}  ({best_arm_row['label']})")
print(f"    mean_ppf = {best_arm_row['mean_ppf']:.4f} ± {best_arm_row['std_ppf']:.4f}")
print(f"    → USE THIS in your production QICA")
print("=" * 90)


# =============================================================================
# SECTION 10 — SAVE OUTPUTS
# =============================================================================

# Per-run summary
summary_rows = []
for r in all_results:
    summary_rows.append({
        'arm': r['arm'], 'label': r['label'], 'seed': r['seed'],
        'best_ppf': r['best_ppf'], 'best_cycle': r['best_cycle'],
        'best_sigma': r['best_sigma'], 'best_h_hist': r['best_h_hist'],
        'injections': r['total_injections'], 'hard_resets': r['total_resets'],
        'al_cnt': len(r['al_candidates']), 'runtime_s': r['runtime'],
    })
pd.DataFrame(summary_rows).to_csv('entropy_test_summary.csv', index=False)

# Means across seeds
pd.DataFrame(mean_rows).to_csv('entropy_test_means.csv', index=False)

# Gen-by-gen history
history_rows = []
for r in all_results:
    h = r['history']
    for i in range(len(h['gen'])):
        row = {'arm': r['arm'], 'seed': r['seed']}
        for k, v in h.items():
            row[k] = v[i] if i < len(v) else None
        history_rows.append(row)
pd.DataFrame(history_rows).to_csv('entropy_test_history.csv', index=False)

# Best AL candidates
all_al = sorted(
    [c for r in all_results for c in r['al_candidates']],
    key=lambda d: d['priority'], reverse=True
)[:AL_TOP_K * len(ARMS)]
if all_al:
    pd.DataFrame(all_al).to_csv('entropy_test_al_candidates.csv', index=False)

print("[SAVED] entropy_test_summary.csv  entropy_test_means.csv  "
      "entropy_test_history.csv  entropy_test_al_candidates.csv")


# =============================================================================
# SECTION 11 — PLOTS
# =============================================================================

n_arms = len(ARMS)
# Safe color palette — works for any number of arms
CMAP   = cm.get_cmap('tab10', n_arms)
COLORS = [CMAP(i) for i in range(n_arms)]

ARM_NAMES   = [a['name'] for a in ARMS]
ARM_LABELS  = [a['label'] for a in ARMS]

fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    f"Entropy Method Comparison  |  {N_SEEDS} seed(s) × {TEST_GENS} gens / "
    f"{TEST_POP} pop / {TEST_MC} MC   "
    f"{'[QUICK TEST]' if QUICK_TEST else '[FULL RUN]'}",
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)


# ── 1. Mean best-PPF convergence (averaged across seeds, ±std band) ───────────
ax = fig.add_subplot(gs[0, :2])
for i, arm_cfg in enumerate(ARMS):
    arm_hist = [r['history'] for r in all_results if r['arm'] == arm_cfg['name']]
    max_len  = max(len(h['gen']) for h in arm_hist)
    ppf_mat  = np.full((len(arm_hist), max_len), np.nan)
    for j, h in enumerate(arm_hist):
        ppf_mat[j, :len(h['best_ppf'])] = h['best_ppf']
    mean_ppf = np.nanmean(ppf_mat, axis=0)
    std_ppf  = np.nanstd(ppf_mat, axis=0)
    gens     = np.arange(max_len)
    ax.plot(gens, mean_ppf, color=COLORS[i], lw=2, label=arm_cfg['label'])
    if N_SEEDS > 1:
        ax.fill_between(gens, mean_ppf - std_ppf, mean_ppf + std_ppf,
                        color=COLORS[i], alpha=0.15)
ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF (lower is better)')
ax.set_title(f'PPF Convergence  '
             f'{"(mean ± std across seeds)" if N_SEEDS > 1 else "(single seed)"}')
ax.legend(fontsize=8); ax.grid(alpha=0.3)


# ── 2. Mean best-H_hist over time ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for i, arm_cfg in enumerate(ARMS):
    arm_hist = [r['history'] for r in all_results if r['arm'] == arm_cfg['name']]
    max_len  = max(len(h['gen']) for h in arm_hist)
    hh_mat   = np.full((len(arm_hist), max_len), np.nan)
    for j, h in enumerate(arm_hist):
        hh_mat[j, :len(h['best_h_hist'])] = h['best_h_hist']
    mean_hh  = np.nanmean(hh_mat, axis=0)
    gens     = np.arange(max_len)
    ax.plot(gens, mean_hh, color=COLORS[i], lw=1.5, label=arm_cfg['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('H_hist of best pattern')
ax.set_title('H_hist Convergence\n(lower → model more confident)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ── 3. Per-seed scatter: each run's best PPF, grouped by arm ──────────────────
ax = fig.add_subplot(gs[0, 3])
arm_x_positions = {name: i for i, name in enumerate(ARM_NAMES)}
for r in all_results:
    xi = arm_x_positions[r['arm']]
    jit = (r['seed'] - np.mean(SEEDS)) * 0.06 if N_SEEDS > 1 else 0
    ax.scatter(xi + jit, r['best_ppf'], color=COLORS[xi],
               s=60, alpha=0.75, zorder=5)
# Means
for i, row in enumerate(mean_rows):
    ax.errorbar(i, row['mean_ppf'], yerr=row['std_ppf'],
                fmt='D', color='black', ms=8, capsize=5, zorder=6,
                label='mean±std' if i == 0 else '')
ax.set_xticks(range(n_arms))
ax.set_xticklabels([a['name'].split('_', 1)[0] for a in ARMS], fontsize=9)
ax.set_ylabel('Best PPF found')
ax.set_title('Per-Seed Results\n(◆ = mean ± std, dots = individual seeds)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)


# ── 4. Injection + reset events (first seed of each arm) ─────────────────────
ax = fig.add_subplot(gs[1, 0])
first_seed = SEEDS[0]
yticks = []
for i, arm_cfg in enumerate(ARMS):
    r = next((r for r in all_results
               if r['arm'] == arm_cfg['name'] and r['seed'] == first_seed), None)
    if r is None: continue
    h    = r['history']
    gens = np.array(h['gen'])
    inj  = np.array(h['inj_fired'])
    rst  = np.array(h['reset_fired'])
    y    = i + 1
    ax.plot(gens, [y] * len(gens), color=COLORS[i], lw=0.5, alpha=0.3)
    inj_g = gens[inj.astype(bool)]
    rst_g = gens[rst.astype(bool)]
    if len(inj_g):
        ax.scatter(inj_g, [y] * len(inj_g), marker='|', s=80, color=COLORS[i])
    if len(rst_g):
        ax.scatter(rst_g, [y] * len(rst_g), marker='*', s=120, color='red', zorder=6)
    yticks.append((y, arm_cfg['name'].split('_', 1)[0]))
ax.set_yticks([y for y, _ in yticks])
ax.set_yticklabels([l for _, l in yticks], fontsize=8)
ax.set_xlabel('Generation')
ax.set_title(f'Events (seed={first_seed})\n| = injection  ★ = hard reset')
ax.grid(alpha=0.2)


# ── 5. Population Hamming diversity over time ─────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for i, arm_cfg in enumerate(ARMS):
    r = next((r for r in all_results
               if r['arm'] == arm_cfg['name'] and r['seed'] == first_seed), None)
    if r is None: continue
    h = r['history']
    ax.plot(h['gen'], h['ham_div'], color=COLORS[i], lw=1.5,
            label=arm_cfg['label'])
ax.axhline(PATTERN_DIV_LOW, color='red', lw=1, ls='--',
           label=f'DIV_LOW={PATTERN_DIV_LOW}')
ax.set_xlabel('Generation'); ax.set_ylabel('Hamming diversity')
ax.set_title('Population Diversity\n(below red line → revolution boost)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ── 6. AL candidates: H_hist vs σ_ppf scatter ────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
for i, arm_cfg in enumerate(ARMS):
    al = [c for r in all_results if r['arm'] == arm_cfg['name']
          for c in r['al_candidates']]
    if al:
        df_al = pd.DataFrame(al)
        ax.scatter(df_al['sigma_ppf'], df_al['h_hist'],
                   alpha=0.45, s=18, color=COLORS[i], label=arm_cfg['label'])
ax.set_xlabel('σ_ppf (MC dropout)'); ax.set_ylabel('H_hist (calibrated)')
ax.set_title('AL Candidate Quality\n(ideal: H_hist and σ correlated)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ── 7. AL total candidates accumulated (first seed) ──────────────────────────
ax = fig.add_subplot(gs[1, 3])
for i, arm_cfg in enumerate(ARMS):
    r = next((r for r in all_results
               if r['arm'] == arm_cfg['name'] and r['seed'] == first_seed), None)
    if r is None: continue
    h = r['history']
    ax.plot(h['gen'], h['al_total'], color=COLORS[i], lw=1.5,
            label=arm_cfg['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('Cumulative AL candidates')
ax.set_title(f'AL Candidates Accumulated\n(capped at {AL_TOP_K})')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ── 8. Bar chart: mean best PPF per arm with std error bars ──────────────────
ax = fig.add_subplot(gs[2, :2])
x    = np.arange(n_arms)
w    = 0.55
ppfm = [r['mean_ppf'] for r in mean_rows]
ppfs = [r['std_ppf']  for r in mean_rows]
bars = ax.bar(x, ppfm, w, yerr=ppfs, capsize=6,
              color=COLORS, edgecolor='k', lw=0.5,
              error_kw={'elinewidth': 2, 'ecolor': 'black'})
ax.set_xticks(x)
ax.set_xticklabels([r['label'] for r in mean_rows], fontsize=8, rotation=12)
ax.set_ylabel('Mean best PPF  (lower is better)')
ax.set_title(f'Mean Best PPF per Arm  ({N_SEEDS} seed{"s" if N_SEEDS > 1 else ""})\n'
             f'Error bars = ±1 std across seeds')
for b, v, s in zip(bars, ppfm, ppfs):
    ax.text(b.get_x() + b.get_width() / 2, v + s + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Annotate best arm
best_i = int(np.argmin(ppfm))
bars[best_i].set_edgecolor('red')
bars[best_i].set_linewidth(2.5)
ax.text(x[best_i], ppfm[best_i] / 2, '★ BEST',
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')


# ── 9. H_hist scoring comparison: does H_hist ≠ σ? ───────────────────────────
ax = fig.add_subplot(gs[2, 2])
for i, arm_cfg in enumerate(ARMS):
    al = [c for r in all_results if r['arm'] == arm_cfg['name']
          for c in r['al_candidates']]
    if len(al) > 2:
        df_al = pd.DataFrame(al)
        corr  = np.corrcoef(df_al['sigma_ppf'], df_al['h_hist'])[0, 1]
        hh    = df_al['h_hist'].values
        hh_cv = hh.std() / (abs(hh.mean()) + 1e-8)
        ax.scatter(i, corr, s=100, color=COLORS[i], zorder=5,
                   label=f"{arm_cfg['name'].split('_')[0]}  r={corr:.2f}  CV={hh_cv:.3f}")
ax.axhline(0.5, color='green', lw=1, ls='--', label='r=0.5 (useful agreement)')
ax.axhline(0.0, color='grey',  lw=0.8, ls=':')
ax.set_xticks(range(n_arms))
ax.set_xticklabels([a['name'].split('_', 1)[0] for a in ARMS], fontsize=8)
ax.set_ylabel('Pearson r  (H_hist vs σ_ppf)')
ax.set_title('H_hist–σ Agreement per Arm\n'
             'r > 0.5: H_hist agrees with σ\n'
             'r < 0.5: H_hist finds cases σ misses')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ── 10. Stagnation round count over time (first seed) ────────────────────────
ax = fig.add_subplot(gs[2, 3])
for i, arm_cfg in enumerate(ARMS):
    r = next((r for r in all_results
               if r['arm'] == arm_cfg['name'] and r['seed'] == first_seed), None)
    if r is None: continue
    h = r['history']
    ax.plot(h['gen'], h['stagnation_round'], color=COLORS[i], lw=1.5,
            label=arm_cfg['label'])
ax.set_xlabel('Generation'); ax.set_ylabel('Stagnation escalation level')
ax.set_title('Stagnation Escalation\n(higher → search is stuck)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)


plt.savefig('entropy_test_comparison.png', dpi=150, bbox_inches='tight')
print("[SAVED] entropy_test_comparison.png")


# =============================================================================
# SECTION 12 — FINAL RECOMMENDATION
# =============================================================================

print("\n" + "=" * 90)
print("FINAL RECOMMENDATION")
print("=" * 90)

best_row = min(mean_rows, key=lambda r: r['mean_ppf'])
print(f"\n  Best arm      : {best_row['arm']}  ({best_row['label']})")
print(f"  Mean best PPF : {best_row['mean_ppf']:.4f} ± {best_row['std_ppf']:.4f}")
print(f"  Mean cycle    : see entropy_test_summary.csv")
print(f"  Mean σ        : {best_row['mean_sigma']:.4f}")

print(f"\n  Methodological notes:")
print(f"  • Noise floor from seed variance  : ±{noise_floor:.4f} PPF")
print(f"  • Differences below noise floor are not reliable — re-run with more seeds")
print(f"  • H_hist–σ correlation in AL candidates tells you whether H_hist is adding")
print(f"    information (low r = good: H_hist finds cases σ-scoring misses)")
print(f"  • H_traj was confirmed weak in v1/v2 ablation (CV=0.02–0.04, r≈−0.04)")
print(f"    and is disabled in all arms of this run — do not re-enable.")

print(f"\n  Settings to carry into production:")
print(f"  • AL scoring     : use_h_hist = {ARMS[ARM_NAMES.index(best_row['arm'])]['use_h_hist']}")
print(f"  • Injection      : use_injection = {ARMS[ARM_NAMES.index(best_row['arm'])]['use_injection']}")
print(f"  • Hard reset     : use_hard_reset = {ARMS[ARM_NAMES.index(best_row['arm'])]['use_hard_reset']}")
print(f"  • HR fraction    : {HARD_RESET_FRAC} (25% wipe — do not increase)")
print(f"  • HR cooldown    : {HARD_RESET_COOLDOWN} gens (keep — prevents consecutive wipes)")
print(f"  • H_traj         : OFF (confirmed weak discriminator)")
print("=" * 90)