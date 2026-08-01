"""
=============================================================================
qica_v9.py  —  Production Entropy QICA  |  Combined-AL + Trust-Region Audit
=============================================================================
PURPOSE OF THIS VERSION: v8 ran a one-shot, data-driven "showdown" between
6 candidate AL-selection rules (baseline / sigma-only / H_gauss / H_hist /
H_traj / combined) on a shared pool of every pattern it ever evaluated.
RESULT (v8 run, reproduced in the docstring of qica_v8.py's printed output):

    sigma_only   σ_gain=+59.7%  ppf_cost=+11.9%  → MARGINAL
    h_gauss      σ_gain=+59.7%  ppf_cost=+11.9%  → MARGINAL (≈sigma_only;
                 H_gauss = 0.5*log(2πe*σ²) is a deterministic monotonic
                 transform of σ, so it can never select anything sigma-only
                 didn't already pick — it is NOT a second source of
                 information, just σ on a log scale)
    h_hist       σ_gain=+55.8%  ppf_cost=+12.8%  → MARGINAL
    h_traj       σ_gain=+11.3%  ppf_cost=+7.1%   → MARGINAL  (CV=0.018 across
                 the whole pool — barely varies pattern-to-pattern, weak
                 discriminator on its own)
    combined     σ_gain=+35.5%  ppf_cost=+8.2%   → KEEP — the only rule that
                 finds genuinely MORE uncertain patterns without paying the
                 ppf_cost the single-metric high-σ rules pay.

WHY 'combined' wins, explainably: H_hist (per-pattern, noise-floor-calibrated
Shannon entropy of the MC PPF distribution) answers "how spread/uncertain is
the predicted peak power for this pattern, in units of the surrogate's own
measured MC-dropout noise?" — it is a *level* uncertainty. H_traj (entropy of
the full burnup-curve covariance, not just the final PPF) answers "how
uncertain is the whole shape of the power trajectory over the cycle?" — a
*shape* uncertainty that level-only metrics structurally cannot see. Summing
their batch z-scores selects patterns that are unusual on EITHER axis, so it
surfaces a more diverse, genuinely-informative AL set than any single metric,
without dragging in the high-PPF patterns that pure high-σ selection favors
(because raw σ keeps walking PPF outward as a side-effect of seeking spread).
v9 therefore makes 'combined' the ONLY live AL-selection rule (Section 8) —
no more per-run rule sweep needed, the answer is already known. The full
6-rule showdown function is kept available (RUN_ENTROPY_SHOWDOWN=True) for
re-verification on a future CNN/dataset, but does not run by default.

H_gauss is kept ONLY as a σ-equivalent diagnostic/calibration number (it is
how TYPICAL_SIGMA / HIST_HALF_WIDTH get calibrated) — it no longer competes
for AL selection, since it carries no information beyond raw σ.
H_rank (ranking-disagreement entropy) was never wired into fitness or AL
selection in v6/v7/v8 either — kept here purely as a passive diagnostic plot,
explicitly labelled as such, not implied to be "in use".

NEW IN v9 — TRUST-REGION AUDIT (answers: "is fixing the bottom ~35% of
positions to their modal training type actually helping, or just narrowing
the search for no benefit?"). Section 6b runs two SHORT, identical-seed,
identical-budget mini-optimizations before the main run: one with the
current ENTROPY_FREE_FRAC=0.65 trust region active, one with all 31
positions fully free. Both use the same population size / generation count
/ MC samples, so the only thing that differs is whether the low-entropy
positions are locked. Best PPF and AL-pool diversity from both short runs
are printed side by side so the trust region's effect on THIS dataset/CNN
is measured, not assumed — see Section 6b and the printed
[TRUST-REGION AUDIT] block.

KEY ENTROPY-USAGE MAP (so it stays explainable — what each metric is FOR):
  H_hist + H_traj (combined, z-scored) → SECTION 8: AL candidate flagging.
      The only entropy that touches which patterns get sent to OpenMC.
  H_pop / hamming_diversity             → SECTION 5/13: QICA process control
      (adaptive revolution rate). Search-health monitor, NOT AL-related.
  H_pos (position entropy)              → SECTION 6: trust region — which of
      the 31 assembly positions QICA is allowed to vary at all. NOT used for
      AL or fitness; purely restricts the search space upfront.
  H_gauss                               → calibration constant only (σ scale).
  H_rank                                → passive diagnostic plot only.
None of these touch `fitness` directly except indirectly via ppf_std
(W_UNCERTAINTY term, which uses raw σ, not entropy) — entropy in this
codebase is an AL/search-control signal, not an optimization objective.
This separation is intentional: PPF/cycle quality should come from the
physics-grounded fitness function; entropy's job is deciding what to LEARN
FROM next (AL) and how hard to EXPLORE (revolution rate), not what to
OPTIMIZE FOR.

Carries forward all v6/v7/v8 fixes (calibrated histogram bins, escalating
multi-elite stagnation injection FIX 8, Hamming-distance diversity FIX 9,
hard-reset escape valve FIX 10).

AL candidate output is capped at AL_TOP_K (default 60) in the final saved
CSV — the "AL=NNN" counter printed during the run is just the running size
of the raw candidate pool before that final cap is applied.

INPUTS:   cnn_v9_model.keras, cnn_v9_config.json, train_type_freq_v9.npy
OUTPUTS:  qica_v9_best_patterns.csv, qica_v9_al_candidates.csv,
          qica_v9_convergence.png, qica_v9_entropy_comparison.png,
          qica_v9_trust_region_audit.csv
=============================================================================
"""

import os, sys, json, time, warnings, io, contextlib
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print("qica_v9.py  —  Multi-Entropy QICA  |  Calibrated Shannon Entropy  [FIXED]\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

MODEL_PATH  = 'cnn_v9_model.keras'     if os.path.exists('cnn_v9_model.keras')    else 'cnn_v4_model.keras'
CONFIG_PATH = 'cnn_v9_config.json'     if os.path.exists('cnn_v9_config.json')    else 'cnn_v4_config.json'
TRUST_PATH  = 'train_type_freq_v9.npy' if os.path.exists('train_type_freq_v9.npy') else 'train_type_freq.npy'

# ─── Entropy mode ─────────────────────────────────────────────────────────────
# 'gaussian'  v5 baseline — H = 0.5*log(2πe*σ²), Gaussian assumption
# 'histogram' Option 1    — Real Shannon H from MC histogram bins (FIX 6: calibrated)
# 'trajectory'Option 3    — Multivariate H over full PPF burnup curve
# 'combined'  RECOMMENDED — Histogram H for AL + adaptive revolution rate
# 'full'                  — combined + trajectory as primary AL metric
ENTROPY_MODE = 'combined'

# ─── Trajectory computation flag (independent of mode) ───────────────────────
COMPUTE_TRAJECTORY = True

# ─── Entropy-method showdown (v8's question — already answered, see docstring)
# Off by default: 'combined' (H_hist + H_traj) is now the production AL rule
# baked into Section 8 directly. Flip True only to re-verify on a new
# CNN/dataset; it costs extra post-run compute and is purely diagnostic.
RUN_ENTROPY_SHOWDOWN = False

# ─── Trust-region audit (NEW v9 — answers "is fixing positions helping?") ────
# Runs two short, identical-seed mini-optimizations before the main run:
# one with the trust region active (ENTROPY_FREE_FRAC, current behaviour),
# one fully free (all 31 positions optimisable). Same population/budget for
# both, so any PPF/diversity difference is attributable to the trust region
# itself, not random variation. See Section 6b.
TRUST_REGION_AUDIT = True
AUDIT_GENS   = 40    # short budget — enough to see a trend, not full convergence
AUDIT_N_POP  = 40
AUDIT_MC     = 15

# ─── Histogram entropy (FIX 6: calibrated to actual MC sigma scale) ──────────
HIST_BINS = 10   # bins; max H = log(10) = 2.30 nats
# Half-width of the (per-pattern centered) histogram window, expressed as a
# multiple of the calibrated "typical" MC-dropout sigma. Calibrated once at
# startup in _load_seeds_via_cnn(); these are just the multiplier + bounds.
SIGMA_HALF_WIDTH_MULT = 4.0
HIST_HALF_WIDTH_MIN   = 0.03   # floor, in PPF units
HIST_HALF_WIDTH_MAX   = 1.20   # ceiling, in PPF units
N_CALIBRATION_PATTERNS = 60     # patterns used to measure typical MC sigma
N_CALIBRATION_MC_PASSES = 20    # MC passes per calibration pattern

# ─── AL thresholds ────────────────────────────────────────────────────────────
# v9: percentile is now applied to the COMBINED z-score
# (z(H_hist) + z(H_traj), see Section 8 / _evaluate_all), not raw H_hist —
# this is the rule v8's data-driven showdown verdict marked "KEEP".
AL_HIST_PERCENTILE   = 70     # flag the most-uncertain 30% (among low-PPF patterns)
AL_HIST_MIN_FLOOR    = 1e-3   # tiny floor so a totally flat batch flags nothing
AL_ENTROPY_THRESHOLD = -1.0   # gaussian H (v5 mode only, display/legacy)
AL_SIGMA_THRESHOLD   = 0.08
AL_TOP_K             = 60     # final AL output cap (per request: ~50-60 max)
AL_LIVE_POOL_CAP      = 400    # bound the running collection so it doesn't grow unbounded
AL_ROUNDS            = 0

# ─── Adaptive revolution (relative threshold) ────────────────────────────────
ADAPTIVE_REVOLUTION     = True
POP_ENTROPY_REL_DROP    = 0.10  # trigger when H_pop drops >10% from initial
PATTERN_DIV_LOW         = 0.35  # fraction unique patterns below which to boost rev
REVOLUTION_BOOST        = 1.8
REVOLUTION_MAX          = 0.65

# ─── FIX 8: Escalating, multi-elite stagnation injection ─────────────────────
STAGNATION_PATIENCE     = 15    # gens without fitness improvement before injection
STAGNATION_N_INJECT     = 25    # base number of mutation patterns to inject
STAGNATION_N_ELITES     = 3     # round-robin seed pool size (was: always rank-1 only)
STAGNATION_MAX_ESCALATE = 4     # cap on how far escalation grows (rounds)
# FIX 10: hard-reset escape valve — if escalation pins at the cap this many
# CONSECUTIVE times with no improvement, fully reinitialize the worst empire
# instead of repeating the same nudge forever.
HARD_RESET_AT_CAP_COUNT = 3
STAGNATION_IMPROVE_EPS  = 1e-3  # fitness improvement needed to reset escalation

# ─── Fitness weights ──────────────────────────────────────────────────────────
PPF_LIMIT          = 3.5
W_PPF_PENALTY      = 80.0
W_PPF_SOFT         = 6.0
W_UNCERTAINTY      = 40.0
W_TRUST            = 0.0
W_ENTROPY_BONUS    = 5.0
W_MONOTONICITY     = 10.0

# ─── QICA hyperparameters ─────────────────────────────────────────────────────
N_COUNTRIES        = 100
N_EMPIRES          = 8
ASSIMILATION_COEFF = 0.35
REVOLUTION_RATE    = 0.35
REVOLUTION_MIN     = 0.08
QUANTUM_TEMP_INIT  = 2.0
QUANTUM_TEMP_FINAL = 0.1
MAX_GEN            = 250
ELITE_SIZE         = 15
MC_SAMPLES         = 30
ENTROPY_FREE_FRAC  = 0.65
SEED               = 42

# ─── Core geometry ────────────────────────────────────────────────────────────
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


# =============================================================================
# SECTION 2 — ConvResBlock (must match CNN exactly)
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


# =============================================================================
# SECTION 3 — LOAD MODEL + CONFIG
# =============================================================================

for p in [MODEL_PATH, CONFIG_PATH]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}  → Run cnn_v9.py first."); sys.exit(1)

model = keras.models.load_model(MODEL_PATH, compile=False,
                                custom_objects={'ConvResBlock': ConvResBlock})
print(f"[LOAD] Model  : {MODEL_PATH}")
print(f"       Config : {CONFIG_PATH}")
print(f"  input={model.input_shape}  output={model.output_shape}")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

ym_mean  = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
ym_scale = np.array(cfg['ym_scaler_scale'], dtype=np.float32)
yr_mean  = np.array(cfg['yr_scaler_mean'],  dtype=np.float32)
yr_scale = np.array(cfg['yr_scaler_scale'], dtype=np.float32)

IDX_PPF_MAX = cfg['IDX_PPF_MAX']
IDX_PPF_BOC = cfg['IDX_PPF_BOC']
IDX_STEPS_S = cfg['IDX_PPF_STEPS_START']
IDX_STEPS_E = cfg['IDX_PPF_STEPS_END']
IDX_CYCLE   = cfg['IDX_CYCLE']
IDX_RHO     = cfg['IDX_RHO']
N_STEPS     = IDX_STEPS_E - IDX_STEPS_S
N_OUTPUTS   = cfg['N_OUTPUTS']
print(f"  ppf_max={IDX_PPF_MAX}  steps={IDX_STEPS_S}:{IDX_STEPS_E}  "
      f"cycle={IDX_CYCLE}  rho={IDX_RHO}  N_STEPS={N_STEPS}\n")


# =============================================================================
# SECTION 4 — TRUST REGION FREQUENCIES
# =============================================================================

if os.path.exists(TRUST_PATH):
    type_freq = np.load(TRUST_PATH).astype(np.float32)
    print(f"[TRUST] {TRUST_PATH}  shape={type_freq.shape}")
else:
    print(f"[TRUST] {TRUST_PATH} not found — uniform fallback.")
    type_freq = np.ones((N_POS, N_TYPES), dtype=np.float32) / N_TYPES


# =============================================================================
# SECTION 5 — SHANNON ENTROPY UTILITIES
# =============================================================================

# Global PPF range (legacy, retained for logging only — no longer drives
# the histogram bin window, see FIX 6 below).
PPF_HIST_LO: float = 1.5
PPF_HIST_HI: float = 5.0

# FIX 6: calibrated MC-sigma scale, set by _load_seeds_via_cnn() below.
TYPICAL_SIGMA: float = 0.07
HIST_HALF_WIDTH: float = 0.28   # = SIGMA_HALF_WIDTH_MULT * TYPICAL_SIGMA, clipped


def gaussian_entropy(mc_ppf: np.ndarray) -> np.ndarray:
    """
    v5 BASELINE: H = 0.5 * log(2πe * σ²).
    Assumes Gaussian MC distribution. Misses bimodal structure.
    mc_ppf: (MC, B) → (B,) nats
    """
    sigma = mc_ppf.std(axis=0) + 1e-10
    return (0.5 * np.log(2.0 * np.pi * np.e * sigma**2)).astype(np.float32)


def histogram_entropy(mc_ppf: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """
    FIX 6 — Per-pattern CENTERED histogram with a GLOBALLY CALIBRATED
    half-width (placeholder; overridden below once TYPICAL_SIGMA/
    HIST_HALF_WIDTH are calibrated from real MC passes in
    _load_seeds_via_cnn()). See module docstring for the full rationale.
    mc_ppf: (MC, B) → (B,) nats
    """
    MC, B = mc_ppf.shape
    entropies = np.zeros(B, dtype=np.float32)
    centered = mc_ppf - mc_ppf.mean(axis=0, keepdims=True)
    lo, hi = -HIST_HALF_WIDTH, HIST_HALF_WIDTH
    for b in range(B):
        vals = centered[:, b]
        hist, _ = np.histogram(vals, bins=bins, range=(lo, hi))
        p = hist.astype(np.float64) / (hist.sum() + 1e-12)
        p = p[p > 0]
        entropies[b] = float(-np.sum(p * np.log(p)))
    return entropies


def trajectory_entropy(mc_curves: np.ndarray) -> np.ndarray:
    """
    OPTION 3 — Multivariate Gaussian entropy over full PPF burnup curve.
    H = ½ * (k * log(2πe) + log|Σ|) via SVD (rank-deficient safe).
    Already working correctly in v6 — UNCHANGED.
    mc_curves: (MC, B, N_STEPS) → (B,) nats  (can be negative for small eigenvalues)
    """
    MC, B, K = mc_curves.shape
    entropies = np.zeros(B, dtype=np.float32)
    eff_rank  = min(MC - 1, K)

    for b in range(B):
        X = mc_curves[:, b, :].astype(np.float64)
        X_c = X - X.mean(axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(X_c, full_matrices=False)
            eigvals  = (s[:eff_rank]**2) / max(MC - 1, 1)
            eigvals  = np.maximum(eigvals, 1e-12)
            log_det  = np.sum(np.log(eigvals))
            H = 0.5 * (eff_rank * np.log(2.0 * np.pi * np.e) + log_det)
        except np.linalg.LinAlgError:
            H = 0.0
        entropies[b] = float(H)
    return entropies


def population_entropy(empires: list) -> float:
    """
    OPTION 4 — Shannon entropy of mean quantum probability across population.
    Normalised to [0, 1]. Logging/plotting only — adaptive revolution uses
    pattern diversity as the primary trigger (see compute_adaptive_rev_rate).
    """
    q_states = []
    for emp in empires:
        q_states.append(emp.imperialist.q_state)
        for c in emp.colonies:
            q_states.append(c.q_state)
    if not q_states:
        return 0.5
    mean_q = np.mean(q_states, axis=0)
    mean_q = mean_q / (mean_q.sum(axis=1, keepdims=True) + 1e-10)
    H_raw  = float(-np.sum(mean_q * np.log(mean_q + 1e-10)))
    H_max  = N_POS * np.log(N_TYPES)
    return H_raw / H_max


def pattern_diversity(empires: list) -> float:
    """
    LEGACY METRIC — fraction of unique MEASURED patterns. Kept only for
    logging/comparison. CONFIRMED USELESS at this state-space size (9^20):
    the v7 run showed this reading EXACTLY 1.000 for all 251 generations,
    because quantum collapse() resamples stochastically every generation
    and the combinatorial space is so large that "two countries happen to
    measure the identical pattern" basically never occurs whether or not
    the search is actually converging. It is not a normalization bug — the
    metric itself just carries no information here. See hamming_diversity()
    below (FIX 9) for the real replacement used to drive adaptive revolution.
    """
    pats = []
    for emp in empires:
        if emp.imperialist.measured is not None:
            pats.append(tuple(emp.imperialist.measured.tolist()))
        for c in emp.colonies:
            if c.measured is not None:
                pats.append(tuple(c.measured.tolist()))
    if not pats:
        return 1.0
    return len(set(pats)) / len(pats)


def hamming_diversity(empires: list, max_sample: int = 60) -> float:
    """
    FIX 9 — real diversity signal: mean pairwise Hamming distance over the
    FREE positions only, normalized to [0, 1] (0 = identical population,
    1 = every free position differs between every pair). Unlike fraction-
    unique, this is sensitive to *how similar* patterns are, not just
    whether they're bit-for-bit identical — so it actually moves as the
    population converges, and can meaningfully drive adaptive revolution.

    A random sample of up to `max_sample` measured countries is used to
    keep the O(n^2) pairwise comparison cheap.
    """
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
    arr = np.stack(pats)[:, free_idx]   # (n, n_free)
    n = arr.shape[0]
    if n > max_sample:
        idx = np.random.choice(n, max_sample, replace=False)
        arr = arr[idx]
        n = max_sample
    total = 0.0
    count = 0
    for i in range(n):
        diffs = (arr[i+1:] != arr[i]).mean(axis=1)   # fraction of free positions differing
        total += float(diffs.sum())
        count += diffs.shape[0]
    return total / count if count > 0 else 1.0



def ranking_disagreement_entropy(mc_ppf: np.ndarray) -> float:
    """
    OPTION 5 — Entropy of winner distribution across MC passes.
    mc_ppf: (MC, B) → float
    """
    if mc_ppf.shape[1] < 2:
        return 0.0
    winners = np.argmin(mc_ppf, axis=1)
    B       = mc_ppf.shape[1]
    counts  = np.bincount(winners, minlength=B).astype(np.float64)
    p       = counts / counts.sum()
    p       = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def gaussian_to_sigma(h: float) -> float:
    """Convert Gaussian H back to equivalent σ."""
    return float(np.sqrt(np.exp(2 * h) / (2 * np.pi * np.e)))


def compute_adaptive_rev_rate(base_rate: float, pat_div: float,
                               h_pop: float, initial_h_pop: float) -> float:
    """
    Adaptive revolution using BOTH pattern diversity AND relative H_pop drop.
    Unchanged from v6.
    """
    if not ADAPTIVE_REVOLUTION:
        return base_rate

    if pat_div < PATTERN_DIV_LOW:
        severity = 1.0 - (pat_div / PATTERN_DIV_LOW)
        boosted  = base_rate * (1.0 + REVOLUTION_BOOST * severity)
        return float(min(boosted, REVOLUTION_MAX))

    if initial_h_pop > 0:
        rel_drop = (initial_h_pop - h_pop) / (initial_h_pop + 1e-8)
        if rel_drop > POP_ENTROPY_REL_DROP:
            severity = min(rel_drop / (POP_ENTROPY_REL_DROP * 2.5), 1.0)
            boosted  = base_rate * (1.0 + REVOLUTION_BOOST * severity)
            return float(min(boosted, REVOLUTION_MAX))

    return float(base_rate)


# =============================================================================
# SECTION 6 — TRUST REGION SETUP
# =============================================================================

def compute_position_entropy(freq: np.ndarray) -> np.ndarray:
    return (-np.sum(freq * np.log(freq + 1e-10), axis=1)).astype(np.float32)

def analyze_trust_region(freq: np.ndarray, free_frac: float = ENTROPY_FREE_FRAC):
    h_pos      = compute_position_entropy(freq)
    n_free     = max(1, int(np.round(N_POS * free_frac)))
    rank       = np.argsort(h_pos)[::-1]
    free_mask  = np.zeros(N_POS, dtype=bool)
    free_mask[rank[:n_free]] = True
    fixed_types = (np.argmax(freq, axis=1) + 1).astype(np.int32)
    return free_mask, fixed_types, h_pos, n_free

free_mask, fixed_types, h_pos, n_free = analyze_trust_region(type_freq)
print(f"\n[TRUST REGION]  {n_free}/{N_POS} free  (H_pos: {h_pos.min():.3f}–{h_pos.max():.3f})\n")


# =============================================================================
# SECTION 7 — GRID BUILDER + INVERSE TRANSFORM
# =============================================================================

def pattern_to_grid(pattern_int: np.ndarray) -> np.ndarray:
    B   = pattern_int.shape[0] if pattern_int.ndim > 1 else 1
    pat = pattern_int.reshape(B, N_POS)
    grid = np.zeros((B, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid[:, r, c] = pat[:, pi]; pi += 1
    return grid

def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    return np.concatenate([
        Y_sc[:, :34] * ym_scale + ym_mean,
        Y_sc[:, 34:35] * yr_scale + yr_mean,
    ], axis=1)


# =============================================================================
# SECTION 8 — EVALUATE BATCH
# =============================================================================

def evaluate_batch(patterns_int: np.ndarray, mc_n: int = MC_SAMPLES) -> dict:
    """
    All entropy variants computed in every call.
    h_hist uses the FIX 6 calibrated, per-pattern-centered histogram.
    """
    if patterns_int.ndim == 1:
        patterns_int = patterns_int.reshape(1, -1)
    B    = patterns_int.shape[0]
    grid = pattern_to_grid(patterns_int)
    X_tf = tf.constant(grid, dtype=tf.int32)

    mc_preds_sc = np.stack([
        model(X_tf, training=True).numpy()
        for _ in range(mc_n)
    ])   # (mc_n, B, N_OUTPUTS)

    mean_sc = mc_preds_sc.mean(axis=0)
    std_sc  = mc_preds_sc.std(axis=0)
    mean_real = inverse_transform(mean_sc)
    std_real  = np.concatenate([
        std_sc[:, :34] * ym_scale,
        std_sc[:, 34:35] * yr_scale,
    ], axis=1)

    ppf_mean   = mean_real[:, IDX_PPF_MAX]
    ppf_std    = std_real[:, IDX_PPF_MAX]
    cycle_mean = mean_real[:, IDX_CYCLE]
    rho_mean   = mean_real[:, IDX_RHO]
    ppf_steps  = mean_real[:, IDX_STEPS_S:IDX_STEPS_E]

    # Physical units for entropy computations
    mc_ppf_phys = (mc_preds_sc[:, :, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX]
                   + ym_mean[IDX_PPF_MAX])          # (mc_n, B)

    mc_curves_phys = (mc_preds_sc[:, :, IDX_STEPS_S:IDX_STEPS_E]
                      * ym_scale[IDX_STEPS_S:IDX_STEPS_E]
                      + ym_mean[IDX_STEPS_S:IDX_STEPS_E])   # (mc_n, B, N_STEPS)

    # ── All entropy variants ────────────────────────────────────────────────
    h_gauss = gaussian_entropy(mc_ppf_phys)                  # (B,) — v5 baseline
    h_hist  = histogram_entropy(mc_ppf_phys, bins=HIST_BINS) # (B,) — FIX 6: calibrated, centered
    h_traj  = (trajectory_entropy(mc_curves_phys)
               if (COMPUTE_TRAJECTORY or ENTROPY_MODE in ('trajectory', 'full'))
               else np.zeros(B, dtype=np.float32))
    h_rank  = ranking_disagreement_entropy(mc_ppf_phys)

    # ── Trust penalty ──────────────────────────────────────────────────────────
    trust_penalty = np.array([
        sum(-np.log(float(type_freq[p, int(patterns_int[b, p]) - 1]) + 1e-6)
            for p in range(N_POS)) / N_POS
        for b in range(B)
    ], dtype=np.float32)

    # ── Monotonicity bonus ─────────────────────────────────────────────────────
    late       = ppf_steps[:, 3:]
    diffs      = late[:, 1:] - late[:, :-1]
    mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / max(late.shape[1]-1, 1))

    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    fitness    = (cycle_mean
                  - W_PPF_SOFT    * ppf_mean
                  - W_PPF_PENALTY * ppf_excess
                  - W_UNCERTAINTY * ppf_std
                  - W_TRUST       * trust_penalty
                  + mono_bonus)

    return {
        'ppf_mean': ppf_mean, 'ppf_std': ppf_std,
        'cycle_mean': cycle_mean, 'rho_mean': rho_mean,
        'ppf_steps': ppf_steps, 'fitness': fitness,
        'trust_penalty': trust_penalty,
        'h_gauss': h_gauss, 'h_hist': h_hist,
        'h_traj': h_traj,   'h_rank': float(h_rank),
    }


def is_al_candidate(combined_score: float, al_thr: float) -> bool:
    """
    v9 — PRODUCTION AL RULE (data-driven winner from the v8 showdown):
    combined_score = z(H_hist) + z(H_traj), z-scored within the current
    batch. Caller supplies a percentile-derived threshold (self-calibrating,
    same mechanism as before — just scored on the combined metric now).
    """
    return float(combined_score) >= al_thr


# =============================================================================
# SECTION 9 — QUANTUM COUNTRY
# =============================================================================

class QuantumCountry:
    def __init__(self, q_state: np.ndarray = None):
        if q_state is None:
            raw = np.ones((N_POS, N_TYPES), dtype=np.float32)
            self.q_state = raw / raw.sum(axis=1, keepdims=True)
        else:
            self.q_state = q_state.copy().astype(np.float32)
        for p in range(N_POS):
            if not free_mask[p]:
                self.q_state[p]                   = 0.0
                self.q_state[p, fixed_types[p]-1] = 1.0

        self.measured   = None
        self.fitness    = -np.inf
        self.ppf_mean   = 9.0
        self.ppf_std    = 0.0
        self.cycle_mean = 0.0
        self.keff_mean  = 0.0
        self.h_gauss    = -10.0
        self.h_hist     = 0.0
        self.h_traj     = 0.0
        self.h_combined = 0.0

    def collapse(self, temperature: float = 1.0) -> np.ndarray:
        logits = np.log(self.q_state + 1e-10) / max(temperature, 0.01)
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=1, keepdims=True)
        self.measured = np.array([
            np.random.choice(N_TYPES, p=probs[i]) + 1
            for i in range(N_POS)
        ], dtype=np.int32)
        for p in range(N_POS):
            if not free_mask[p]:
                self.measured[p] = fixed_types[p]
        return self.measured

    def q_entropy(self) -> float:
        return float(-np.sum(self.q_state * np.log(self.q_state + 1e-10)))

    def quantum_assimilate(self, imperialist: 'QuantumCountry', beta: float, temp: float):
        for p in range(N_POS):
            if free_mask[p]:
                self.q_state[p] = ((1.0 - beta) * self.q_state[p]
                                   + beta * imperialist.q_state[p])
        self.q_state = np.maximum(self.q_state, 1e-10)
        self.q_state /= self.q_state.sum(axis=1, keepdims=True)
        for p in range(N_POS):
            if not free_mask[p]:
                self.q_state[p]                   = 0.0
                self.q_state[p, fixed_types[p]-1] = 1.0

    def quantum_revolution(self, rate: float, temperature: float,
                           bias_weights: np.ndarray = None):
        for p in range(N_POS):
            if not free_mask[p]:
                continue
            p_rev = rate if bias_weights is None else rate * (1.0 + bias_weights[p])
            p_rev = min(p_rev, 0.95)
            if np.random.random() < p_rev:
                alpha = np.ones(N_TYPES) * max(temperature, 0.05)
                self.q_state[p] = np.random.dirichlet(alpha)

    def clone(self) -> 'QuantumCountry':
        c = QuantumCountry(self.q_state)
        c.measured   = self.measured.copy() if self.measured is not None else None
        c.fitness    = self.fitness;  c.ppf_mean  = self.ppf_mean
        c.ppf_std    = self.ppf_std;  c.cycle_mean= self.cycle_mean
        c.keff_mean  = self.keff_mean
        c.h_gauss    = self.h_gauss;  c.h_hist = self.h_hist; c.h_traj = self.h_traj
        c.h_combined = self.h_combined
        return c


# =============================================================================
# SECTION 10 — EMPIRE
# =============================================================================

class Empire:
    def __init__(self, imperialist: QuantumCountry, colonies: list):
        self.imperialist = imperialist
        self.colonies    = colonies

    @property
    def power(self) -> float:
        return self.imperialist.fitness

    @property
    def total_countries(self) -> int:
        return 1 + len(self.colonies)


# =============================================================================
# SECTION 11 — WARM START + ENTROPY CALIBRATION (FIX 6)
# =============================================================================

X_train_seed = None; ppf_cnn_seed = None; X_grid_seed = None

def _load_seeds_via_cnn():
    global X_train_seed, ppf_cnn_seed, X_grid_seed
    global PPF_HIST_LO, PPF_HIST_HI, TYPICAL_SIGMA, HIST_HALF_WIDTH
    csv_path = 'ml_dataset_constrained.csv'
    if not os.path.exists(csv_path):
        print("[SEED] ml_dataset_constrained.csv not found — warm-start disabled"); return
    try:
        df   = pd.read_csv(csv_path, skiprows=1, engine='python', on_bad_lines='skip')
        lc   = [f'loading_{i}' for i in range(N_POS)]
        if not all(c in df.columns for c in lc):
            print("[SEED] loading_ columns not found"); return
        X_raw = df[lc].values.astype(np.int32); N = len(X_raw)
        grids = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)
        pi = 0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    grids[:, r, c] = X_raw[:, pi]; pi += 1
        ppf_preds = []
        for i in range(0, N, 128):
            preds   = model(tf.constant(grids[i:i+128], dtype=tf.int32), training=False).numpy()
            ppf_sc  = preds[:, IDX_PPF_MAX]
            ppf_preds.extend((ppf_sc * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]).tolist())
        ppf_arr = np.array(ppf_preds, dtype=np.float32)
        X_train_seed = X_raw; X_grid_seed = grids; ppf_cnn_seed = ppf_arr

        # Legacy display range (no longer drives histogram bins, see FIX 6)
        PPF_HIST_LO = float(max(1.0, np.percentile(ppf_arr, 2)  * 0.95))
        PPF_HIST_HI = float(min(10., np.percentile(ppf_arr, 98) * 1.05))
        print(f"[SEED] CNN-ranked {N} patterns | PPF {ppf_arr.min():.3f}–{ppf_arr.max():.3f}")

        # ── FIX 6: calibrate the histogram entropy bin window from REAL MC spread ──
        n_cal = min(N_CALIBRATION_PATTERNS, N)
        cal_idx   = np.random.choice(N, n_cal, replace=False)
        cal_grids = grids[cal_idx]
        mc_cal_sc = np.stack([
            model(tf.constant(cal_grids, dtype=tf.int32), training=True).numpy()[:, IDX_PPF_MAX]
            for _ in range(N_CALIBRATION_MC_PASSES)
        ])  # (N_CALIBRATION_MC_PASSES, n_cal), scaled space
        mc_cal_phys = mc_cal_sc * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
        sigmas_cal  = mc_cal_phys.std(axis=0)               # (n_cal,)
        typical_sig = float(np.median(sigmas_cal))
        p95_sig     = float(np.percentile(sigmas_cal, 95))

        TYPICAL_SIGMA   = max(typical_sig, 1e-4)
        # Use whichever is larger of (typical*mult) or a fraction of the p95 tail,
        # so a few genuinely high-uncertainty patterns don't all saturate H at max.
        raw_half_width  = SIGMA_HALF_WIDTH_MULT * max(typical_sig, 0.6 * p95_sig)
        HIST_HALF_WIDTH = float(np.clip(raw_half_width, HIST_HALF_WIDTH_MIN, HIST_HALF_WIDTH_MAX))

        print(f"[CAL] MC-sigma calibration ({n_cal} patterns × {N_CALIBRATION_MC_PASSES} passes): "
              f"median σ={typical_sig:.4f}  p95 σ={p95_sig:.4f}")
        print(f"[HIST] Calibrated half-width: ±{HIST_HALF_WIDTH:.4f}  "
              f"(bin width: {2*HIST_HALF_WIDTH/HIST_BINS:.4f}, vs typical σ={TYPICAL_SIGMA:.4f})")
    except Exception as e:
        print(f"[SEED] Failed: {e}")

_load_seeds_via_cnn()


def histogram_entropy(mc_ppf: np.ndarray, bins: int = HIST_BINS) -> np.ndarray:
    """
    FIX 6 (final) — per-pattern centered, globally calibrated half-width.
    Overrides the placeholder defined in Section 5 now that TYPICAL_SIGMA /
    HIST_HALF_WIDTH have been measured from real MC-dropout passes above.

    Centering removes the across-pattern PPF-LEVEL spread (which is just
    "where in PPF-space this design sits", not uncertainty). The calibrated
    half-width keeps the bin grid sized to the ACTUAL MC noise scale,
    fixing both the v5 bug (H≈2.3 always, bins too fine relative to spread
    because each pattern got its own tiny local range) and the v6 bug
    (H≈0 always, bins too coarse because the window spanned the whole
    training-data PPF range instead of the MC noise scale).
    """
    MC, B = mc_ppf.shape
    entropies = np.zeros(B, dtype=np.float32)
    centered = mc_ppf - mc_ppf.mean(axis=0, keepdims=True)
    lo, hi = -HIST_HALF_WIDTH, HIST_HALF_WIDTH
    for b in range(B):
        vals = centered[:, b]
        hist, _ = np.histogram(vals, bins=bins, range=(lo, hi))
        p = hist.astype(np.float64) / (hist.sum() + 1e-12)
        p = p[p > 0]
        entropies[b] = float(-np.sum(p * np.log(p)))
    return entropies


# =============================================================================
# SECTION 12 — POSITION SENSITIVITY
# =============================================================================

def compute_position_sensitivity() -> np.ndarray:
    if X_train_seed is None:
        return np.ones(N_POS, dtype=np.float32) / N_POS
    n_sample    = min(50, len(X_train_seed))
    top_idx     = np.argsort(ppf_cnn_seed)[:n_sample]
    base_grids  = X_grid_seed[top_idx].copy()
    base_ppf    = ppf_cnn_seed[top_idx]
    sensitivities = np.zeros(N_POS, dtype=np.float32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] < 0: continue
            deltas = []
            for t in range(1, N_TYPES + 1):
                perturbed = base_grids.copy(); perturbed[:, r, c] = t
                preds_sc  = model(tf.constant(perturbed, dtype=tf.int32), training=False).numpy()
                ppf_p     = preds_sc[:, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
                deltas.append(float(np.abs(ppf_p - base_ppf).mean()))
            sensitivities[pi] = max(deltas); pi += 1
    total = sensitivities.sum()
    if total > 1e-10: sensitivities /= total
    else:             sensitivities[:] = 1.0 / N_POS
    top3 = np.argsort(sensitivities)[-3:][::-1].tolist()
    print(f"[SENS] Top-3 positions: {top3}  "
          f"(sens: {sensitivities[top3[0]]:.4f}, {sensitivities[top3[1]]:.4f}, "
          f"{sensitivities[top3[2]]:.4f})")
    return sensitivities

pos_sensitivity = compute_position_sensitivity()


# =============================================================================
# SECTION 13 — QICA v7 OPTIMIZER
# =============================================================================

class QICAv9:
    def __init__(self, entropy_mode: str = ENTROPY_MODE,
                 mc_samples: int = MC_SAMPLES,
                 n_countries: int = N_COUNTRIES,
                 max_gen: int = MAX_GEN):
        self.entropy_mode    = entropy_mode
        self.mc_samples      = mc_samples
        self.n_countries     = n_countries
        self.max_gen         = max_gen
        self.elite_archive   = []
        self.al_candidates   = []
        self._al_seen        = set()
        self.universe         = {}    # FIX 11: pat_key -> record, every evaluated country
        self.last_h_rank     = 0.0
        self.last_al_hist_thr = AL_HIST_MIN_FLOOR   # FIX 7: tracked for reporting
        self.initial_h_pop   = None
        self.pat_div_history = []
        # FIX 8: escalating stagnation tracking
        self.stagnation_count    = 0
        self.stagnation_rounds   = 0
        self.rounds_at_cap       = 0   # FIX 10: consecutive cap-level injections w/o improvement
        self.last_best_fitness   = None
        self.best_fitness_ever   = -np.inf
        self.history = {
            'gen': [], 'best_fitness': [], 'mean_fitness': [],
            'best_cycle': [], 'best_ppf': [], 'mean_ppf_std': [],
            'mean_h_gauss': [], 'mean_h_hist': [], 'mean_h_traj': [],
            'h_rank': [], 'h_pop': [], 'pat_div': [],
            'rev_rate': [], 'n_empires': [], 'temperature': [], 'al_count': [],
            'stagnation_round': [],
        }

    def _temperature(self, gen):
        r = gen / self.max_gen
        return QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** r

    def _base_revolution_rate(self, gen):
        r = gen / self.max_gen
        return REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * r

    def _initialize_population(self) -> list:
        countries = []
        for bias_t in range(1, N_TYPES + 1):
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.04
            q[:, bias_t-1] = 0.68
            q /= q.sum(axis=1, keepdims=True)
            countries.append(QuantumCountry(q))
        if X_train_seed is not None and ppf_cnn_seed is not None:
            n_seeds   = min(8, len(X_train_seed))
            top_k_idx = np.argsort(ppf_cnn_seed)[:n_seeds]
            for idx in top_k_idx:
                pat = X_train_seed[idx]
                q   = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
                for p in range(N_POS):
                    t = int(pat[p])
                    if 1 <= t <= N_TYPES:
                        q[p, t - 1] = 0.84
                q /= q.sum(axis=1, keepdims=True)
                countries.append(QuantumCountry(q))
            print(f"  [SEED] {n_seeds} CNN-guided seeds added "
                  f"(PPF range: {ppf_cnn_seed[top_k_idx[0]]:.3f}–"
                  f"{ppf_cnn_seed[top_k_idx[-1]]:.3f})")
        while len(countries) < self.n_countries:
            countries.append(QuantumCountry())
        return countries

    def _evaluate_all(self, countries: list, temperature: float,
                      mc_n: int = None) -> tuple:
        """Returns (countries, h_rank)."""
        if mc_n is None:
            mc_n = self.mc_samples
        patterns = np.stack([c.collapse(temperature) for c in countries])
        result   = evaluate_batch(patterns, mc_n=mc_n)

        ppf_arr   = result['ppf_mean']
        ppf_5pct  = np.percentile(ppf_arr, 5)
        h_hist_arr = result['h_hist']
        h_traj_arr = result['h_traj']

        # v9 — COMBINED AL SCORE (data-driven winner of the v8 showdown):
        # z-score each entropy metric WITHIN this batch (so the combination
        # is unitless and self-calibrating — H_hist is in nats of a
        # noise-floor-calibrated histogram, H_traj is in nats of a
        # multivariate burnup-curve covariance; they are not on the same
        # natural scale, so summing raw values would let whichever metric
        # has the larger natural spread dominate). z(H_hist) captures
        # uncertainty in WHERE the peak PPF lands; z(H_traj) captures
        # uncertainty in the SHAPE of the whole burnup curve — summing them
        # flags patterns that are unusual on either axis, which is exactly
        # what the v8 showdown showed finds more genuinely-uncertain
        # candidates without the PPF cost that pure high-σ selection pays.
        combined_arr = (
            (h_hist_arr - h_hist_arr.mean()) / (h_hist_arr.std() + 1e-8)
            + (h_traj_arr - h_traj_arr.mean()) / (h_traj_arr.std() + 1e-8)
        ).astype(np.float32)

        # FIX 7 (carried forward): AL threshold purely adaptive — percentile
        # of THIS batch's combined-score distribution (self-calibrating to
        # whatever scale the combined metric actually has this generation).
        al_thr = max(AL_HIST_MIN_FLOOR, float(np.percentile(combined_arr, AL_HIST_PERCENTILE)))
        self.last_al_hist_thr = al_thr

        ent_bonus = np.array([
            W_ENTROPY_BONUS * c.q_entropy() / (N_POS * N_TYPES)
            for c in countries
        ], dtype=np.float32)

        ppf_excess = np.maximum(0.0, ppf_arr - PPF_LIMIT)
        late       = result['ppf_steps'][:, 3:]
        diffs      = late[:, 1:] - late[:, :-1]
        mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / max(late.shape[1]-1, 1))

        fitness_arr = (
            result['cycle_mean']
            - W_PPF_SOFT    * ppf_arr
            - W_PPF_PENALTY * ppf_excess
            - W_UNCERTAINTY * result['ppf_std']
            - W_TRUST       * result['trust_penalty']
            + mono_bonus + ent_bonus
        )

        for i, c in enumerate(countries):
            c.fitness    = float(fitness_arr[i])
            c.ppf_mean   = float(ppf_arr[i])
            c.ppf_std    = float(result['ppf_std'][i])
            c.cycle_mean = float(result['cycle_mean'][i])
            c.keff_mean  = float(1.0 / (1.0 - float(result['rho_mean'][i]) / 1e5))
            c.h_gauss    = float(result['h_gauss'][i])
            c.h_hist     = float(result['h_hist'][i])
            c.h_traj     = float(result['h_traj'][i])
            c.h_combined = float(combined_arr[i])

            # FIX 11 (carried forward): log every evaluated country into the
            # universe pool — cheap bookkeeping, used for the unique-pattern
            # count and (optionally) the legacy showdown re-verification.
            pat_key = tuple(c.measured.tolist())
            self.universe[pat_key] = {
                'pattern': c.measured.tolist(), 'pred_ppf': c.ppf_mean,
                'sigma_ppf': c.ppf_std, 'h_gauss': c.h_gauss,
                'h_hist': c.h_hist, 'h_traj': c.h_traj, 'cycle': c.cycle_mean,
            }

            # AL flagging: uncertain (by combined score) AND promising (live collection, capped)
            is_uncertain = is_al_candidate(c.h_combined, al_thr)
            if is_uncertain and c.ppf_mean <= ppf_5pct:
                if pat_key not in self._al_seen:
                    self._al_seen.add(pat_key)
                    priority = c.h_combined * c.cycle_mean / (c.ppf_mean + 1e-6)
                    self.al_candidates.append({
                        'pattern'    : c.measured.tolist(),
                        'pred_ppf'   : c.ppf_mean,
                        'sigma_ppf'  : c.ppf_std,
                        'h_gauss'    : c.h_gauss,
                        'h_hist'     : c.h_hist,
                        'h_traj'     : c.h_traj,
                        'h_combined' : c.h_combined,
                        'cycle'      : c.cycle_mean,
                        'priority'   : float(priority),
                        'entropy_mode': self.entropy_mode,
                    })
                    # Bound the live pool so it doesn't grow unboundedly over
                    # 250 generations (final output is capped at AL_TOP_K
                    # anyway; this just keeps memory/runtime tidy along the way).
                    if len(self.al_candidates) > AL_LIVE_POOL_CAP:
                        self.al_candidates.sort(key=lambda d: d['priority'], reverse=True)
                        kept = self.al_candidates[:AL_LIVE_POOL_CAP // 2]
                        self.al_candidates = kept
                        self._al_seen = {tuple(d['pattern']) for d in kept}

        h_rank = result['h_rank']
        self.last_h_rank = h_rank
        return countries, h_rank

    def _form_empires(self, countries: list) -> list:
        sorted_c = sorted(countries, key=lambda c: c.fitness, reverse=True)
        n_emp    = min(N_EMPIRES, len(sorted_c))
        imps, cols = sorted_c[:n_emp], sorted_c[n_emp:]
        fits    = np.array([imp.fitness for imp in imps])
        fits_sh = fits - fits.min() + 1e-6
        powers  = fits_sh / fits_sh.sum()
        counts  = np.round(powers * len(cols)).astype(int)
        diff    = len(cols) - counts.sum()
        if diff > 0:   counts[np.argmax(powers)] += diff
        elif diff < 0: counts[np.argmax(counts)]  += diff
        empires, idx = [], 0
        for i, imp in enumerate(imps):
            empires.append(Empire(imp, list(cols[idx:idx+counts[i]])))
            idx += counts[i]
        return empires

    def _assimilation_step(self, empires, beta, temp, rev_rate):
        for emp in empires:
            for col in emp.colonies:
                col.quantum_assimilate(emp.imperialist, beta, temp)
                col.quantum_revolution(rev_rate, temp, bias_weights=pos_sensitivity)

    def _intra_competition(self, empires, temperature, mc_n=None):
        all_cols = [c for emp in empires for c in emp.colonies]
        if not all_cols: return
        self._evaluate_all(all_cols, temperature, mc_n=mc_n)
        for emp in empires:
            if not emp.colonies: continue
            best_i = max(range(len(emp.colonies)),
                         key=lambda i: emp.colonies[i].fitness)
            if emp.colonies[best_i].fitness > emp.imperialist.fitness:
                emp.colonies[best_i], emp.imperialist = (
                    emp.imperialist, emp.colonies[best_i])

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
                        c.h_gauss, c.h_hist, c.h_traj, c.h_combined
                    ))
        self.elite_archive.sort(key=lambda x: x[0], reverse=True)
        seen, unique = set(), []
        for entry in self.elite_archive:
            key = tuple(entry[1])
            if key not in seen:
                seen.add(key); unique.append(entry)
        self.elite_archive = unique[:ELITE_SIZE]

    def _inject_mutations(self, empires, temperature, round_idx: int = 1):
        """
        FIX 8 — Escalating, multi-elite stagnation injection.

        v6 ALWAYS reseeded from the single best elite with a fixed mutation
        count, which is exactly why the optimizer kept landing back on the
        same PPF≈1.94-2.01 basin every 15 generations instead of escaping it.

        Now:
          - seeds are drawn round-robin from the top STAGNATION_N_ELITES
            elites, not just rank 1, so injections explore the neighborhood
            of several good-but-different solutions instead of hammering
            the population around one point repeatedly;
          - both batch size and mutation strength (number of randomized
            positions, and the temperature used to re-collapse them) grow
            with the number of CONSECUTIVE stagnation rounds — escalating
            escape attempts instead of repeating the same nudge forever;
          - the escalation counter resets the moment fitness genuinely
            improves (tracked in run()), so a successful escape doesn't
            keep escalating unnecessarily.
        """
        if not self.elite_archive:
            return
        esc = min(round_idx - 1, STAGNATION_MAX_ESCALATE)   # 0..STAGNATION_MAX_ESCALATE
        n_seed_elites = min(STAGNATION_N_ELITES, len(self.elite_archive))
        n_inject = int(STAGNATION_N_INJECT * (1.0 + 0.5 * esc))

        new_c = []
        for i in range(n_inject):
            seed_pat = self.elite_archive[i % n_seed_elites][1]
            lo_mut = min(5 + 2 * esc, n_free)
            hi_mut = min(14 + 4 * esc, n_free)
            if hi_mut <= lo_mut:
                hi_mut = lo_mut + 1
            n_mutate = np.random.randint(lo_mut, hi_mut + 1)
            mut_pos  = np.random.choice(
                [p for p in range(N_POS) if free_mask[p]],
                min(n_mutate, n_free), replace=False
            )
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
            for p, t in enumerate(seed_pat):
                q[p, int(t) - 1] = 0.84
            q /= q.sum(axis=1, keepdims=True)
            for p in mut_pos:
                q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
            new_c.append(QuantumCountry(q))

        boosted_temp = min(temperature * (2.5 + 0.5 * esc), 2.0)
        patterns = np.stack([c.collapse(boosted_temp) for c in new_c])
        result   = evaluate_batch(patterns)
        for i, c in enumerate(new_c):
            c.fitness    = float(result['fitness'][i])
            c.ppf_mean   = float(result['ppf_mean'][i])
            c.ppf_std    = float(result['ppf_std'][i])
            c.h_hist     = float(result['h_hist'][i])
            c.cycle_mean = float(result['cycle_mean'][i])

        largest = max(range(len(empires)), key=lambda i: empires[i].total_countries)
        empires[largest].colonies.extend(new_c)
        print(f"\n  [INJECT round {round_idx}, escalation {esc}] {n_inject} mutations from "
              f"top-{n_seed_elites} elites (best ppf={self.elite_archive[0][2]:.3f}) injected")
        self.stagnation_count = 0
        return esc

    def _hard_reset_worst_empire(self, empires, temperature):
        """
        FIX 10 — hard-reset escape valve.

        The v7 run showed escalation pinning at its cap (4) for rounds 4
        through 10 with ZERO improvement (PPF stuck at 1.920 from gen 100
        to gen 250) — mutating around the same top-3 elites harder and
        harder wasn't enough to leave that basin. This adds a genuinely
        different move: when the cap has been hit HARD_RESET_AT_CAP_COUNT
        times in a row with no improvement, the worst-performing empire is
        COMPLETELY reinitialized (half fresh-random, half seeded from
        mid-tier elites — ranks 3+ rather than the same top-3 already being
        mutated) instead of nudged. This is a real basin jump, not a bigger
        nudge in the same basin.
        """
        if len(empires) < 2 or not self.elite_archive:
            return
        worst_idx = min(range(len(empires)), key=lambda i: empires[i].power)
        n_fresh   = max(8, empires[worst_idx].total_countries)  # preserve population size

        n_mid_elites = max(0, len(self.elite_archive) - 2)   # ranks 3+ only
        new_c = []
        for i in range(n_fresh):
            if n_mid_elites > 0 and i % 2 == 0:
                seed_pat = self.elite_archive[2 + (i % n_mid_elites)][1]
                q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.03
                for p, t in enumerate(seed_pat):
                    q[p, int(t) - 1] = 0.7
                q /= q.sum(axis=1, keepdims=True)
                # heavy randomization on top of the mid-tier seed
                heavy_mut = np.random.choice(
                    [p for p in range(N_POS) if free_mask[p]],
                    min(n_free, max(8, n_free // 2)), replace=False
                )
                for p in heavy_mut:
                    q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                new_c.append(QuantumCountry(q))
            else:
                new_c.append(QuantumCountry())   # fully random, uniform q_state

        patterns = np.stack([c.collapse(1.6) for c in new_c])
        result   = evaluate_batch(patterns)
        for i, c in enumerate(new_c):
            c.fitness    = float(result['fitness'][i])
            c.ppf_mean   = float(result['ppf_mean'][i])
            c.ppf_std    = float(result['ppf_std'][i])
            c.h_hist     = float(result['h_hist'][i])
            c.cycle_mean = float(result['cycle_mean'][i])

        empires[worst_idx] = Empire(new_c[0], new_c[1:])
        print(f"\n  [HARD RESET] Empire #{worst_idx} fully reinitialized "
              f"({n_fresh} countries: half random, half mid-tier-elite-seeded) "
              f"— escalation cap sustained {HARD_RESET_AT_CAP_COUNT}x without improvement")

    def _log(self, gen, empires, temp, h_pop, pat_div, rev_rate):
        all_c  = [e.imperialist for e in empires] + [c for e in empires for c in e.colonies]
        all_fit = [c.fitness    for c in all_c]
        all_std = [c.ppf_std    for c in all_c]
        all_hg  = [c.h_gauss   for c in all_c]
        all_hh  = [c.h_hist    for c in all_c]
        all_ht  = [c.h_traj    for c in all_c]
        best = (self.elite_archive[0] if self.elite_archive
                else (0, None, 9.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0))

        self.history['gen'].append(gen)
        self.history['best_fitness'].append(float(max(all_fit)))
        self.history['mean_fitness'].append(float(np.mean(all_fit)))
        self.history['best_cycle'].append(float(best[3]))
        self.history['best_ppf'].append(float(best[2]))
        self.history['mean_ppf_std'].append(float(np.mean(all_std)))
        self.history['mean_h_gauss'].append(float(np.mean(all_hg)))
        self.history['mean_h_hist'].append(float(np.mean(all_hh)))
        self.history['mean_h_traj'].append(float(np.mean(all_ht)))
        self.history['h_rank'].append(float(self.last_h_rank))
        self.history['h_pop'].append(float(h_pop))
        self.history['pat_div'].append(float(pat_div))
        self.history['rev_rate'].append(float(rev_rate))
        self.history['n_empires'].append(len(empires))
        self.history['temperature'].append(temp)
        self.history['al_count'].append(len(self.al_candidates))
        self.history['stagnation_round'].append(self.stagnation_rounds)

        if gen % 25 == 0 or gen == self.max_gen:
            print(
                f"  Gen {gen:4d}/{self.max_gen} | emp={len(empires):2d} | "
                f"ppf={best[2]:.3f} σ={best[4]:.4f} | "
                f"H_hist={best[6]:.3f} (thr={self.last_al_hist_thr:.3f}) "
                f"H_gauss={best[5]:.2f} H_traj={best[7]:.1f} | "
                f"cycle={best[3]:6.1f}d | fit={best[0]:7.2f} | "
                f"T={temp:.3f} ham_div={pat_div:.2f} rev={rev_rate:.3f} | "
                f"AL={len(self.al_candidates)} | stag_round={self.stagnation_rounds}"
            )

    def run(self) -> dict:
        print("=" * 74)
        print(f"QICA-v9  [FIXED]  ENTROPY_MODE='{self.entropy_mode}'  "
              f"COMPUTE_TRAJECTORY={COMPUTE_TRAJECTORY}")
        print(f"  N={self.n_countries}  Empires={N_EMPIRES}  Gens={self.max_gen}  MC={self.mc_samples}")
        print(f"  H_hist bins={HIST_BINS}  half_width=±{HIST_HALF_WIDTH:.3f} "
              f"(calibrated from typical σ={TYPICAL_SIGMA:.4f})")
        print(f"  AL_HIST_PERCENTILE={AL_HIST_PERCENTILE}  STAGNATION_PATIENCE={STAGNATION_PATIENCE}\n")
        t0 = time.time()

        print("[INIT] Initializing population ...")
        countries = self._initialize_population()
        temp      = self._temperature(0)
        countries, _ = self._evaluate_all(countries, temp)
        empires   = self._form_empires(countries)
        self._update_elite(empires)

        h_pop     = population_entropy(empires)
        pat_div   = hamming_diversity(empires)
        self.initial_h_pop = h_pop
        base_rate = self._base_revolution_rate(0)
        rev_rate  = compute_adaptive_rev_rate(base_rate, pat_div, h_pop, self.initial_h_pop)
        self._log(0, empires, temp, h_pop, pat_div, rev_rate)

        best0 = self.elite_archive[0]
        self.best_fitness_ever = best0[0]
        print(f"  Initial best: ppf={best0[2]:.3f}  cycle={best0[3]:.1f}d  "
              f"σ={best0[4]:.4f}  H_hist={best0[6]:.3f}  H_traj={best0[7]:.1f}\n")

        print("[RUN] Main optimisation loop ...")
        for gen in range(1, self.max_gen + 1):
            temp      = self._temperature(gen)
            base_rate = self._base_revolution_rate(gen)
            h_pop     = population_entropy(empires)
            pat_div   = hamming_diversity(empires)
            rev_rate  = compute_adaptive_rev_rate(base_rate, pat_div, h_pop, self.initial_h_pop)

            self._assimilation_step(empires, ASSIMILATION_COEFF, temp, rev_rate)
            self._intra_competition(empires, temp)
            self._update_elite(empires)
            empires = self._empire_collapse(empires)

            # FIX 8: stagnation detection with escalation + reset-on-improvement
            if self.elite_archive:
                cur_fit = self.elite_archive[0][0]
                if cur_fit > self.best_fitness_ever + STAGNATION_IMPROVE_EPS:
                    self.best_fitness_ever = cur_fit
                    self.stagnation_rounds = 0   # genuine improvement → reset escalation
                    self.rounds_at_cap     = 0   # FIX 10: improvement also clears the cap counter
                if self.last_best_fitness is not None:
                    if abs(cur_fit - self.last_best_fitness) < 0.05:
                        self.stagnation_count += 1
                    else:
                        self.stagnation_count = 0
                self.last_best_fitness = cur_fit
                if self.stagnation_count >= STAGNATION_PATIENCE:
                    self.stagnation_rounds += 1
                    esc = self._inject_mutations(empires, temp, round_idx=self.stagnation_rounds)
                    # FIX 10: track consecutive cap-level injections w/ no improvement
                    if esc is not None and esc >= STAGNATION_MAX_ESCALATE:
                        self.rounds_at_cap += 1
                    else:
                        self.rounds_at_cap = 0
                    if self.rounds_at_cap >= HARD_RESET_AT_CAP_COUNT:
                        self._hard_reset_worst_empire(empires, temp)
                        self.rounds_at_cap = 0

            self._log(gen, empires, temp, h_pop, pat_div, rev_rate)

            if len(empires) == 1 and len(empires[0].colonies) < 3:
                print(f"\n[CONVERGED]  Single empire at gen {gen}")
                break

        t_total = time.time() - t0
        print(f"\n[DONE]  {t_total:.1f}s ({t_total/60:.1f}min)  | "
              f"Unique AL candidates: {len(self.al_candidates)}  | "
              f"Universe pool: {len(self.universe)} unique patterns evaluated")
        return {
            'elite_archive': self.elite_archive,
            'history'      : self.history,
            'al_candidates': self.al_candidates,
            'runtime'      : t_total,
            'al_hist_thr'  : self.last_al_hist_thr,
            'universe'     : self.universe,
        }


# =============================================================================
# SECTION 13b — TRUST-REGION AUDIT  (NEW v9)
# =============================================================================

def run_trust_region_audit():
    """
    Answers: "is fixing the bottom ~(1-ENTROPY_FREE_FRAC) of positions
    (lowest H_pos, locked to their modal training type) actually helping,
    or just shrinking the search for no benefit?"

    Two short, IDENTICAL-SEED, IDENTICAL-BUDGET mini-runs:
      trust_region : current free_mask  (n_free of N_POS positions free)
      fully_free   : all N_POS positions free
    Same population size / generation count / MC samples for both arms —
    the trust region is the only thing that differs, so any PPF or
    AL-diversity gap is attributable to it, not to seed luck or budget.

    This is a cheap (AUDIT_GENS generations, not MAX_GEN) DIRECTIONAL check,
    not a converged final-PPF comparison — it tells you which way the
    effect points before you commit a full 250-gen run to either setting.
    """
    global free_mask, fixed_types, n_free
    if not TRUST_REGION_AUDIT:
        return None

    saved_free_mask, saved_n_free = free_mask.copy(), n_free
    configs = [
        (f'trust_region ({saved_n_free}/{N_POS} free)', saved_free_mask.copy()),
        (f'fully_free ({N_POS}/{N_POS} free)',           np.ones(N_POS, dtype=bool)),
    ]

    print("\n" + "=" * 74)
    print("TRUST-REGION AUDIT  |  is fixing low-entropy positions helping?")
    print(f"  Budget per arm: N={AUDIT_N_POP}  Gens={AUDIT_GENS}  MC={AUDIT_MC}  (identical seed)")
    print("=" * 74)

    rows = []
    for label, mask in configs:
        free_mask = mask
        n_free    = int(mask.sum())
        np.random.seed(SEED); tf.random.set_seed(SEED)   # identical seed per arm
        t0 = time.time()
        with contextlib.redirect_stdout(io.StringIO()):   # suppress per-gen spam
            opt = QICAv9(n_countries=AUDIT_N_POP, max_gen=AUDIT_GENS, mc_samples=AUDIT_MC)
            res = opt.run()
        dt = time.time() - t0
        best = res['elite_archive'][0]
        rows.append({'arm': label, 'n_free': n_free, 'best_ppf': best[2],
                      'best_cycle': best[3], 'al_candidates': len(res['al_candidates']),
                      'runtime_s': dt})
        print(f"  [{label}]  best_ppf={best[2]:.4f}  cycle={best[3]:.1f}d  "
              f"AL_candidates={len(res['al_candidates'])}  ({dt:.0f}s)")

    free_mask, n_free = saved_free_mask, saved_n_free   # restore production trust region

    df = pd.DataFrame(rows)
    df.to_csv('qica_v9_trust_region_audit.csv', index=False)
    ppf_tr = float(df.loc[df['arm'].str.startswith('trust_region'), 'best_ppf'].iloc[0])
    ppf_ff = float(df.loc[df['arm'].str.startswith('fully_free'),   'best_ppf'].iloc[0])
    delta  = ppf_ff - ppf_tr

    print(f"\n  VERDICT (short-budget DIRECTIONAL signal, not a converged comparison):")
    if delta > 0.03:
        print(f"    Trust region BEATS fully-free by {delta:.4f} PPF at this budget — "
              f"locking the modal-type positions is steering the search toward better "
              f"basins faster, not just shrinking the space for no reason. KEEP it.")
    elif delta < -0.03:
        print(f"    Fully-free BEATS trust region by {-delta:.4f} PPF at this budget — "
              f"the locked positions may be excluding genuinely better combinations. "
              f"Consider raising ENTROPY_FREE_FRAC, or re-deriving train_type_freq_v9.npy "
              f"if it was computed from a stale/small training set.")
    else:
        print(f"    Difference ({delta:+.4f} PPF) is within run-to-run noise at this budget — "
              f"roughly NEUTRAL for final PPF. It still earns its keep by cutting the "
              f"effective search space from 9^{N_POS} to 9^{saved_n_free}, which speeds "
              f"convergence and lowers per-generation OpenMC/AL load even at tied PPF.")
    print(f"  (Run at full MAX_GEN={MAX_GEN} budget if you want the converged final-PPF gap "
          f"rather than this {AUDIT_GENS}-gen directional check.)")
    print(f"  [SAVED]  qica_v9_trust_region_audit.csv\n")
    return df


# =============================================================================
# SECTION 14 — ENTROPY METHOD COMPARISON ANALYSIS (post-run)
# =============================================================================

def analyze_entropy_methods(al_candidates: list, history: dict) -> dict:
    if not al_candidates:
        print("[ANALYSIS] No AL candidates — skipping comparison."); return {}
    al_df = pd.DataFrame(al_candidates)
    if len(al_df) < 5:
        print("[ANALYSIS] Too few candidates."); return {}

    variants = {
        'H_gauss (v5 baseline)': 'h_gauss',
        'H_hist  (Option 1)   ': 'h_hist',
        'H_traj  (Option 3)   ': 'h_traj',
    }

    stats = {}
    print("\n" + "=" * 74)
    print("ENTROPY METHOD COMPARISON — POST-RUN ANALYSIS")
    print("=" * 74)
    print(f"  Based on {len(al_df)} unique AL candidates\n")
    print(f"  {'Metric':<26} {'Mean H':>9} {'Std H':>9} {'CV':>7} "
          f"{'H-σ corr':>9} {'Verdict'}")
    print(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*30}")

    best_method = None; best_score = -np.inf

    for label, col in variants.items():
        if col not in al_df.columns:
            print(f"  {label} — column missing"); continue
        h_vals   = al_df[col].dropna().values
        sig_vals = al_df['sigma_ppf'].dropna().values
        if len(h_vals) < 3: continue

        if col == 'h_traj' and np.allclose(h_vals, 0.0):
            print(f"  {label} — NOT COMPUTED "
                  f"(set COMPUTE_TRAJECTORY=True or ENTROPY_MODE='full')")
            continue

        mean_h = float(h_vals.mean())
        std_h  = float(h_vals.std())
        cv     = float(std_h / (abs(mean_h) + 1e-8))
        n      = min(len(h_vals), len(sig_vals))
        corr   = float(np.corrcoef(h_vals[:n], sig_vals[:n])[0, 1])

        if corr > 0.6 and cv > 0.3:
            verdict = "✓ GOOD"
        elif corr > 0.4 or cv > 0.3:
            verdict = "◐ MODERATE"
        else:
            verdict = "✗ WEAK"

        print(f"  {label} {mean_h:>9.3f} {std_h:>9.3f} {cv:>7.3f} {corr:>9.3f}  {verdict}")
        stats[col] = {'mean_h': mean_h, 'std_h': std_h, 'cv': cv, 'corr': corr}
        score = corr * 2.0 + cv * 1.0
        if score > best_score:
            best_score  = score; best_method = label

    if 'h_hist' in al_df.columns and 'h_gauss' in al_df.columns:
        h_hist_n = (
            al_df['h_hist'] - al_df['h_hist'].min()
        ) / (
            (al_df['h_hist'].max() - al_df['h_hist'].min()) + 1e-8
        )
        h_gauss_n = (
            al_df['h_gauss'] - al_df['h_gauss'].min()
        ) / (
            (al_df['h_gauss'].max() - al_df['h_gauss'].min()) + 1e-8
        )
        bimodal = int(((h_hist_n > 0.5) & (h_gauss_n < 0.3)).sum())
        print(f"\n  BIMODAL DETECTION (H_hist high, H_gauss low): {bimodal} patterns "
              f"{'→ multimodal MC structure found!' if bimodal > 0 else '→ none found'}")

    if 'rev_rate' in history and len(history['rev_rate']) > 5:
        rev_arr  = np.array(history['rev_rate'])
        div_arr  = np.array(history['pat_div'])
        boosts   = int((rev_arr > REVOLUTION_RATE * 1.05).sum())
        low_div  = int((div_arr < PATTERN_DIV_LOW).sum())
        print(f"\n  ADAPTIVE REVOLUTION (FIX 9: Hamming-distance diversity, not fraction-unique):")
        print(f"    Hamming diversity range  : {div_arr.min():.3f}–{div_arr.max():.3f}")
        print(f"    Epochs with low diversity: {low_div}/{len(div_arr)}")
        print(f"    Epochs rev_rate boosted  : {boosts}")
        print(f"    Rev rate range           : {rev_arr.min():.3f}–{rev_arr.max():.3f}")
        if boosts > 0:
            print(f"    ✓ Adaptive revolution fired {boosts} times")
        else:
            print(f"    ◐ Adaptive revolution never fired — population stayed genuinely diverse")
            print(f"      by genotype (this is now a real measurement, not a saturated metric)")

    if 'stagnation_round' in history and len(history['stagnation_round']) > 0:
        max_round = max(history['stagnation_round'])
        print(f"\n  ESCALATING STAGNATION INJECTION (FIX 8 + FIX 10 hard reset):")
        print(f"    Max escalation round reached: {max_round}")
        if max_round >= STAGNATION_MAX_ESCALATE:
            print(f"    Hit the escalation cap ({STAGNATION_MAX_ESCALATE}) at least once — "
                  f"FIX 10's hard-reset valve should have engaged after "
                  f"{HARD_RESET_AT_CAP_COUNT} consecutive cap-level injections with no "
                  f"improvement. Check the [HARD RESET] log lines above to confirm it fired.")
        elif max_round > 0:
            print(f"    Escalation engaged but did not max out — escape attempts succeeded "
                  f"before reaching the cap.")
        else:
            print(f"    No stagnation rounds triggered — optimizer kept improving on its own.")

    if best_method:
        print(f"\n  BEST SINGLE AL-FLAGGING METRIC (by σ-correlation + CV): {best_method.strip()}")
    print(f"  NOTE: see the ENTROPY-METHOD SHOWDOWN section below for the real, data-driven")
    print(f"        verdict on which metric(s) are worth keeping over plain QICA.")
    return stats


# =============================================================================
# SECTION 14b — ENTROPY-METHOD SHOWDOWN  (FIX 11: the actual ask)
# =============================================================================

def entropy_method_showdown(universe: dict, top_k: int = AL_TOP_K) -> dict:
    """
    Post-hoc comparison of candidate-selection rules on ONE shared pool of
    evaluated patterns (no re-running the optimizer). Answers: "is using
    entropy for AL selection actually better than just picking the lowest
    PPF patterns (plain QICA, no entropy at all)?"

    Selection rules compared, all drawing from the same "promising" pool
    (safe PPF, bottom percentile by PPF so we're not comparing apples to
    oranges on design quality):
      baseline    — lowest PPF only                  (= plain QICA / v5-2)
      sigma_only  — highest σ                         (what H_gauss is
                    mathematically equivalent to, see FIX 6/v7 finding)
      h_gauss     — highest Gaussian entropy
      h_hist      — highest calibrated histogram entropy
      h_traj      — highest trajectory entropy
      combined    — highest (z(h_hist) + z(h_traj))

    For each rule's selected set we report: mean PPF, mean σ, mean H_hist,
    mean H_traj, internal genotype diversity (mean pairwise Hamming
    distance, free positions only), and Jaccard overlap with baseline.
    A rule "earns its keep" if it finds patterns that are BOTH safe/
    promising AND meaningfully more uncertain than the naive baseline would
    pick anyway (i.e. low overlap + higher mean σ) — that's the whole
    point of spending OpenMC budget on AL verification instead of just
    verifying the top-K lowest-PPF designs.
    """
    if not universe or len(universe) < top_k * 3:
        print(f"[SHOWDOWN] Universe too small ({len(universe)} patterns) for a "
              f"meaningful comparison — skipping.")
        return {}

    df = pd.DataFrame(list(universe.values()))
    df = df[df['pred_ppf'] <= PPF_LIMIT].reset_index(drop=True)
    if len(df) < top_k * 3:
        print(f"[SHOWDOWN] Too few SAFE patterns ({len(df)}) for a meaningful "
              f"comparison — skipping.")
        return {}

    # "Promising" pool: bottom 25th percentile by PPF among safe patterns,
    # with a floor so the pool is always at least 4x top_k for a fair
    # within-pool comparison.
    ppf_cut = np.percentile(df['pred_ppf'], 25)
    pool = df[df['pred_ppf'] <= ppf_cut].copy()
    if len(pool) < top_k * 4:
        pool = df.nsmallest(max(top_k * 4, 200), 'pred_ppf').copy()

    # z-score combined metric
    z_hist = (pool['h_hist'] - pool['h_hist'].mean()) / (pool['h_hist'].std() + 1e-8)
    z_traj = (pool['h_traj'] - pool['h_traj'].mean()) / (pool['h_traj'].std() + 1e-8)
    pool['combined_z'] = z_hist + z_traj

    rules = {
        'baseline (plain QICA, lowest PPF only)': ('pred_ppf', True),   # ascending
        'sigma_only (highest σ)'                : ('sigma_ppf', False),
        'h_gauss (highest Gaussian H)'           : ('h_gauss', False),
        'h_hist (highest calibrated histogram H)': ('h_hist', False),
        'h_traj (highest trajectory H)'          : ('h_traj', False),
        'combined (z(h_hist)+z(h_traj))'         : ('combined_z', False),
    }

    free_idx = np.where(free_mask)[0]

    def hamming_within(patterns: list) -> float:
        if len(patterns) < 2:
            return 0.0
        arr = np.array(patterns)[:, free_idx]
        n = arr.shape[0]
        sample_n = min(n, 50)
        idx = np.random.choice(n, sample_n, replace=False) if n > sample_n else np.arange(n)
        sub = arr[idx]
        total, cnt = 0.0, 0
        for i in range(sub.shape[0]):
            diffs = (sub[i+1:] != sub[i]).mean(axis=1)
            total += float(diffs.sum()); cnt += diffs.shape[0]
        return total / cnt if cnt > 0 else 0.0

    baseline_set = None
    results = {}
    print("\n" + "=" * 88)
    print(f"ENTROPY-METHOD SHOWDOWN  —  one shared pool of {len(pool)} promising/safe patterns, "
          f"top-{top_k} per rule")
    print("=" * 88)
    print(f"  {'Rule':<42} {'meanPPF':>8} {'meanσ':>8} {'meanH_hist':>11} "
          f"{'meanH_traj':>11} {'diversity':>10} {'overlap_base':>13}")
    print(f"  {'-'*42} {'-'*8} {'-'*8} {'-'*11} {'-'*11} {'-'*10} {'-'*13}")

    for label, (col, ascending) in rules.items():
        sel = pool.sort_values(col, ascending=ascending).head(top_k)
        sel_set = set(tuple(p) for p in sel['pattern'])
        if baseline_set is None:
            baseline_set = sel_set
            overlap = 1.0
        else:
            inter = len(sel_set & baseline_set)
            union = len(sel_set | baseline_set)
            overlap = inter / union if union > 0 else 0.0
        div = hamming_within(list(sel['pattern']))
        results[label] = {
            'mean_ppf': float(sel['pred_ppf'].mean()),
            'mean_sigma': float(sel['sigma_ppf'].mean()),
            'mean_h_hist': float(sel['h_hist'].mean()),
            'mean_h_traj': float(sel['h_traj'].mean()),
            'diversity': div,
            'overlap_with_baseline': overlap,
            'n': len(sel),
        }
        print(f"  {label:<42} {sel['pred_ppf'].mean():>8.4f} {sel['sigma_ppf'].mean():>8.4f} "
              f"{sel['h_hist'].mean():>11.3f} {sel['h_traj'].mean():>11.1f} {div:>10.3f} "
              f"{overlap:>13.2f}")

    # ── Data-driven verdicts ──────────────────────────────────────────────────
    base = results['baseline (plain QICA, lowest PPF only)']
    print(f"\n  VERDICTS (relative to baseline = plain QICA, no entropy):")
    verdicts = {}
    for label, r in results.items():
        if label.startswith('baseline'):
            continue
        sigma_gain = (r['mean_sigma'] - base['mean_sigma']) / (base['mean_sigma'] + 1e-8)
        ppf_cost   = (r['mean_ppf'] - base['mean_ppf']) / (base['mean_ppf'] + 1e-8)
        if sigma_gain > 0.15 and r['overlap_with_baseline'] < 0.8 and ppf_cost < 0.10:
            verdict = "KEEP — finds genuinely more-uncertain patterns at little PPF cost"
        elif r['overlap_with_baseline'] >= 0.85:
            verdict = "DROP — picks almost the same patterns baseline already would"
        elif sigma_gain <= 0.05:
            verdict = "DROP — no meaningful uncertainty gain over baseline"
        else:
            verdict = "MARGINAL — some gain, but modest; keep only if AL budget is generous"
        verdicts[label] = verdict
        print(f"    {label:<42} σ_gain={sigma_gain:+.1%}  ppf_cost={ppf_cost:+.1%}  → {verdict}")

    # H_traj-specific discriminativeness check, independent of selection-set stats
    traj_cv = float(pool['h_traj'].std() / (abs(pool['h_traj'].mean()) + 1e-8))
    print(f"\n  H_traj discriminativeness check (CV across the whole promising pool): {traj_cv:.4f}")
    if traj_cv < 0.05:
        print(f"    ⚠ CV<0.05 — H_traj barely varies across patterns in this pool. Even if its")
        print(f"      selection-set stats look OK above, it has very little room to actually")
        print(f"      discriminate good AL candidates from each other. Treat 'KEEP' with caution.")

    print("=" * 88)
    return {'pool_size': len(pool), 'results': results, 'verdicts': verdicts,
            'h_traj_cv': traj_cv}




# =============================================================================
# SECTION 15 — BENCHMARK COMPARISON TABLE
# =============================================================================

def print_benchmark_comparison(best_ppf: float, best_cycle: float):
    benchmarks = [
        ("Traditional GA",       "[1]", 1.42, 330, 121, "2-loop WH"),
        ("Simulated Annealing",  "[4]", 1.38, 280, 193, "4-loop WH"),
        ("ICA (deterministic)",  "[2]", 1.31, 390, 163, "VVER-1000"),
        ("PSO",                  "[3]", 1.40, 310, 121, "2-loop WH"),
        ("ANN + GA hybrid",      "[5]", 1.35, 400, 121, "2-loop WH"),
        ("PSO + ANN surrogate",  "[6]", 1.29, 435, 193, "4-loop WH"),
    ]
    ppf_vals = [b[2] for b in benchmarks]

    print("\n" + "=" * 82)
    print("BENCHMARK COMPARISON: PWR FUEL LOADING OPTIMISATION")
    print("=" * 82)
    print("  ⚠ Different reactor models — order-of-magnitude comparison only")
    print(f"  {'Method':<27} {'Ref':<5} {'PPF':>8} {'Cycle':>8} {'Assem.':>7} {'Notes'}")
    print(f"  {'-'*27} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*15}")
    for method, ref, ppf, cycle, assem, note in benchmarks:
        flag = "★" if ppf == min(ppf_vals) else " "
        print(f"  {flag} {method:<25} {ref:<5} {ppf:>8.2f} {cycle:>8} {assem:>7} {note}")
    print(f"  {'─'*27} {'─'*5} {'─'*8} {'─'*8}")
    print(f"  ► CNN + QICA-v9 (this work)   —    {best_ppf:>8.4f} {best_cycle:>8.1f} {'193':>7} 4-loop BEAVRS")

    better_ppf = sum(1 for b in benchmarks if b[2] > best_ppf)
    print(f"\n  VERDICT:")
    if best_ppf < min(ppf_vals):
        print(f"    ✓ PPF {best_ppf:.4f} BEATS all {len(benchmarks)} published methods!")
    elif best_ppf < np.median(ppf_vals):
        print(f"    ✓ PPF {best_ppf:.4f} better than {better_ppf}/{len(benchmarks)} methods "
              f"(median={np.median(ppf_vals):.2f})")
    else:
        print(f"    ◐ PPF {best_ppf:.4f} — room to improve (best lit. {min(ppf_vals):.2f})")
        print(f"      Consider: higher W_PPF_SOFT (currently {W_PPF_SOFT}), lower W_PPF_PENALTY (currently {W_PPF_PENALTY})")

    print(f"\n  KEY DIFFERENTIATORS vs. prior methods:")
    print(f"    1. CNN surrogate replaces per-pattern simulator calls (~1000× faster)")
    print(f"    2. MC Dropout → per-pattern epistemic uncertainty (σ)")
    print(f"    3. Calibrated histogram Shannon H → real entropy, scaled to actual MC noise")
    print(f"    4. PPF trajectory covariance H → curve-shape uncertainty (novel)")
    print(f"    5. Pattern diversity → adaptive exploration (novel for nuclear fuel opt)")
    print(f"    6. Escalating multi-elite stagnation injection → basin escape")
    print(f"    7. Active learning: uncertainty flags patterns for OpenMC verification")
    print(f"\n  References: [1] Pereira & Lapa (2003) Ann.Nucl.En. 30")
    print(f"              [2] Abdollahzadeh et al. (2012) Nucl.Eng.Des. 248")
    print(f"              [3] Meneses et al. (2006) Ann.Nucl.En. 33")
    print(f"              [4] Mahlers & Martin (2004) Nucl.Technol. 146")
    print(f"              [5] Sadighi et al. (2008) Nucl.Eng.Des. 238")
    print(f"              [6] Kim & Park (2011) J.Nucl.Sci.Technol. 48")
    print("=" * 82)


# =============================================================================
# SECTION 16 — RUN
# =============================================================================

if __name__ == '__main__':

    audit_df = run_trust_region_audit()

    print("\n" + "=" * 74)
    print(f"MAIN RUN  |  ENTROPY_MODE='{ENTROPY_MODE}'  "
          f"COMPUTE_TRAJECTORY={COMPUTE_TRAJECTORY}  MAX_GEN={MAX_GEN}")
    print("=" * 74)

    optimizer = QICAv9()
    results   = optimizer.run()
    elite     = results['elite_archive']
    al_cands  = results['al_candidates']
    hist      = results['history']
    universe  = results.get('universe', {})
    al_hist_thr_final = results.get('al_hist_thr', AL_HIST_MIN_FLOOR)

    entropy_stats  = analyze_entropy_methods(al_cands, hist)

    showdown = None
    if RUN_ENTROPY_SHOWDOWN:
        showdown = entropy_method_showdown(universe, top_k=AL_TOP_K)
    else:
        print(f"\n  [SKIP] Entropy-method showdown not re-run (RUN_ENTROPY_SHOWDOWN=False) — "
              f"v8's run already established 'combined' (H_hist+H_traj) as the only "
              f"rule worth keeping; that rule is what Section 8 uses live above. "
              f"Set RUN_ENTROPY_SHOWDOWN=True to re-verify on a future CNN/dataset.")

    if showdown:
        rows = []
        for label, r in showdown['results'].items():
            row = dict(r)
            row['rule'] = label
            row['verdict'] = showdown['verdicts'].get(label, 'baseline')
            rows.append(row)
        showdown_df = pd.DataFrame(rows)
        showdown_df.to_csv('qica_v9_entropy_showdown.csv', index=False)
        print(f"\n[SAVED]  qica_v9_entropy_showdown.csv  ({len(showdown_df)} rules compared)")

    # ── Top-5 patterns ──────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("TOP LOADING PATTERNS FOUND")
    print("=" * 74)
    print(f"  {'#':<4} {'PPF':>8} {'σ':>7} {'H_hist':>8} {'H_gauss':>9} "
          f"{'H_traj':>8} {'H_comb':>7} {'Cycle':>8} {'Fit':>9}  Status")
    print("-" * 74)
    best_fit, best_pat, best_ppf, best_cyc, best_sig, best_hg, best_hh, best_ht, best_hc = elite[0]
    for rank, (fit, pat, ppf, cyc, sig, hg, hh, ht, hc) in enumerate(elite[:5], 1):
        safe = "✓" if ppf <= PPF_LIMIT else "✗"
        al_h = "H⚠" if hc >= al_hist_thr_final else "H✓"   # flagged by combined z-score, the live AL rule
        al_s = "σ⚠" if sig >= AL_SIGMA_THRESHOLD else "σ✓"
        print(f"  #{rank:<3} {ppf:>8.4f} {sig:>7.4f} {hh:>8.3f} {hg:>9.3f} "
              f"{ht:>8.1f} {hc:>7.2f} {cyc:>8.1f} {fit:>9.2f}  {safe} {al_s} {al_h}")
        print(f"         {list(pat)}")

    print(f"\nBEST:  PPF={best_ppf:.4f}  σ={best_sig:.4f}  "
          f"H_hist={best_hh:.3f}  H_gauss={best_hg:.3f}  H_traj={best_ht:.1f}  H_combined={best_hc:.2f}  "
          f"Cycle={best_cyc:.1f}d")
    print(f"       σ_equiv(H_gauss)={gaussian_to_sigma(best_hg):.4f}")
    print(f"       Pattern: {list(best_pat)}")

    print_benchmark_comparison(best_ppf, best_cyc)

    # ── Save ────────────────────────────────────────────────────────────────
    best_df = pd.DataFrame([
        {'rank': i+1, 'ppf_max': ppf, 'sigma_ppf': sig,
         'h_gauss': hg, 'h_hist': hh, 'h_traj': ht, 'h_combined': hc,
         'cycle_length_days': cyc, 'fitness': fit,
         'ppf_safe': ppf <= PPF_LIMIT,
         'al_combined_flag': hc >= al_hist_thr_final,
         'al_sigma_flag': sig >= AL_SIGMA_THRESHOLD,
         **{f'pos_{j}': int(pat[j]) for j in range(N_POS)}}
        for i, (fit, pat, ppf, cyc, sig, hg, hh, ht, hc) in enumerate(elite)
    ])
    best_df.to_csv('qica_v9_best_patterns.csv', index=False)
    print(f"\n[SAVED]  qica_v9_best_patterns.csv  ({len(best_df)} patterns)")

    if al_cands:
        al_df = (pd.DataFrame(al_cands)
                 .sort_values('priority', ascending=False)
                 .head(AL_TOP_K))
        al_df.to_csv('qica_v9_al_candidates.csv', index=False)
        top = al_df.iloc[0]
        print(f"[SAVED]  qica_v9_al_candidates.csv  ({len(al_df)} candidates)")
        print(f"  Top: ppf={top['pred_ppf']:.3f}  σ={top['sigma_ppf']:.4f}  "
              f"H_hist={top['h_hist']:.3f}  H_traj={top.get('h_traj', 0):.1f}")
    else:
        print(f"\n[WARN] No AL candidates found. If H_hist is still flat across the whole "
              f"population, try lowering AL_HIST_PERCENTILE or increasing N_CALIBRATION_PATTERNS.")

    # ==========================================================================
    # SECTION 17 — PLOTS
    # ==========================================================================
    gen_arr = np.array(hist['gen'])

    fig = plt.figure(figsize=(28, 18))
    fig.suptitle(
        f"QICA-v9 [FIXED]  ENTROPY_MODE='{ENTROPY_MODE}'  COMPUTE_TRAJECTORY={COMPUTE_TRAJECTORY}\n"
        f"Best PPF={best_ppf:.4f}  Cycle={best_cyc:.1f}d  σ={best_sig:.4f}  "
        f"H_hist={best_hh:.3f}  H_traj={best_ht:.1f}  AL={len(al_cands)}",
        fontsize=10, fontweight='bold'
    )
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.42, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gen_arr, hist['best_fitness'], '#1B4FBF', lw=2, label='Best')
    ax.plot(gen_arr, hist['mean_fitness'], '#F5A623', lw=1.2, ls='--', label='Mean')
    ax.set_title('Fitness Convergence'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(gen_arr, hist['best_ppf'], '#D62728', lw=2)
    ax.axhline(PPF_LIMIT, color='orange', lw=1.5, ls='--', label=f'Limit={PPF_LIMIT}')
    ax.set_title('PPF (PRIMARY)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF_max')

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(gen_arr, hist['best_cycle'], '#2CA02C', lw=2)
    ax.set_title('Cycle Length'); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Cycle (days)')

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(gen_arr, hist['mean_h_hist'],  '#17BECF', lw=2.0, label='H_hist (calibrated)')
    ax.plot(gen_arr, hist['mean_h_gauss'], '#9467BD', lw=1.5, ls='--', label='H_gauss (v5)')
    ax.axhline(al_hist_thr_final, color='#17BECF', lw=1, ls=':', alpha=0.7, label='AL thr (final)')
    ax.set_title('H_hist vs H_gauss\n(FIX 6: calibrated to MC σ scale)')
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean H')

    ax = fig.add_subplot(gs[0, 4])
    if any(v != 0 for v in hist['mean_h_traj']):
        ax.plot(gen_arr, hist['mean_h_traj'], '#E377C2', lw=2)
        ax.set_title('H_traj (Option 3)\n(PPF curve covariance entropy)')
    else:
        ax.text(0.5, 0.5, 'H_traj not\ncomputed\n(COMPUTE_TRAJECTORY=False)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.set_title('H_traj (Option 3)')
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean H_traj'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(gen_arr, hist['rev_rate'], '#D62728', lw=2, label='Actual')
    ax.plot(gen_arr, [REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN)*g/MAX_GEN
                      for g in gen_arr], '#AAAAAA', lw=1, ls='--', label='Base')
    ax.axhline(REVOLUTION_MAX, color='orange', lw=1, ls=':', label='Max')
    ax.set_title('Adaptive Revolution Rate')
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Rev rate')

    ax = fig.add_subplot(gs[1, 1])
    pat_div_arr = np.array(hist['pat_div'])
    ax.plot(gen_arr, pat_div_arr, '#8C564B', lw=2, label='Pat. diversity')
    ax.axhline(PATTERN_DIV_LOW, color='red', lw=1.5, ls='--',
               label=f'Low={PATTERN_DIV_LOW} → boost')
    ax.fill_between(gen_arr, 0, PATTERN_DIV_LOW, alpha=0.08, color='red')
    ax.set_title('Pattern Diversity\n(high ≠ no stagnation — see best-fitness plot)')
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Fraction unique patterns')

    ax = fig.add_subplot(gs[1, 2])
    stag_arr = np.array(hist['stagnation_round'])
    ax.plot(gen_arr, stag_arr, '#BCBD22', lw=2)
    ax.set_title('FIX 8: Stagnation Escalation Round\n(resets to 0 on genuine improvement)')
    ax.set_xlabel('Generation'); ax.set_ylabel('Escalation round'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(gen_arr, hist['h_rank'], '#E377C2', lw=2)
    ax.set_title('Ranking Disagreement H_rank')
    ax.set_xlabel('Generation'); ax.set_ylabel('H_rank (nats)'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 4])
    ax.plot(gen_arr, hist['n_empires'], '#1f77b4', lw=2)
    ax.set_title('Empire Collapse'); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('N empires')

    ax = fig.add_subplot(gs[2, 0])
    g_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g_disp[r, c] = float(best_pat[pi]); pi += 1
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad('lightgrey')
    im = ax.imshow(g_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly type')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'T{int(best_pat[pi])}', ha='center', va='center',
                        fontsize=7, fontweight='bold'); pi += 1
    ax.set_title(f'Best Pattern\nPPF={best_ppf:.3f} H_hist={best_hh:.3f}')
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 1])
    e_sigs = [e[4] for e in elite]; e_hh = [e[6] for e in elite]
    e_ht   = [e[7] for e in elite]; e_ppfs = [e[2] for e in elite]
    sc = ax.scatter(e_sigs, e_hh, c=e_ppfs, cmap='RdYlGn_r', s=60, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='PPF_max')
    ax.axvline(AL_SIGMA_THRESHOLD, color='#9467BD', lw=1.5, ls='--')
    ax.axhline(al_hist_thr_final,  color='#17BECF', lw=1.5, ls='--')
    ax.set_title('Elite: σ vs H_hist\n(FIX 6: now discriminative)')
    ax.set_xlabel('σ_ppf'); ax.set_ylabel('H_hist'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 2])
    e_gauss = [e[5] for e in elite]
    ax.scatter(e_gauss, e_hh, c=e_ppfs, cmap='RdYlGn_r', s=60, alpha=0.8)
    ax.set_title('Elite: H_gauss vs H_hist\n(divergence = bimodal detected)')
    ax.set_xlabel('H_gauss (v5)'); ax.set_ylabel('H_hist (v7)'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 3])
    if al_cands:
        al_pl = pd.DataFrame(al_cands)
        sc2 = ax.scatter(al_pl['pred_ppf'], al_pl['h_hist'],
                         c=al_pl['sigma_ppf'], cmap='YlOrRd', s=20, alpha=0.6)
        plt.colorbar(sc2, ax=ax, label='σ_ppf')
        ax.axhline(al_hist_thr_final, color='#17BECF', lw=1.5, ls='--')
        ax.set_title(f'AL Candidates: PPF vs H_hist\n(n={len(al_cands)} unique)')
    else:
        ax.text(0.5, 0.5, 'No AL candidates', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('AL Candidates: PPF vs H_hist')
    ax.set_xlabel('Predicted PPF'); ax.set_ylabel('H_hist'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[2, 4])
    ax.plot(gen_arr, hist['mean_ppf_std'], '#9467BD', lw=2)
    ax.axhline(AL_SIGMA_THRESHOLD, color='orange', lw=1.5, ls='--',
               label=f'σ thr={AL_SIGMA_THRESHOLD}')
    ax.set_title('MC Uncertainty (mean σ)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean σ(PPF)')

    plt.savefig('qica_v9_convergence.png', dpi=150, bbox_inches='tight')
    print("\n[SAVED]  qica_v9_convergence.png")

    # ── Entropy comparison plot ───────────────────────────────────────────────
    if al_cands:
        al_pl = pd.DataFrame(al_cands)
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
        fig2.suptitle(
            f"Entropy Method Comparison [FIXED]  N={len(al_cands)} unique AL candidates\n"
            f"ENTROPY_MODE='{ENTROPY_MODE}'  COMPUTE_TRAJECTORY={COMPUTE_TRAJECTORY}",
            fontsize=10, fontweight='bold'
        )

        ax = axes2[0, 0]
        ax.hist(al_pl['h_gauss'], bins=25, color='#9467BD', alpha=0.7,
                label=f'μ={al_pl["h_gauss"].mean():.3f}')
        ax.axvline(AL_ENTROPY_THRESHOLD, color='k', lw=1.5, ls='--')
        ax.set_title('H_gauss (v5 baseline)\nGaussian approximation')
        ax.legend(fontsize=8); ax.set_xlabel('H_gauss'); ax.grid(alpha=.3)

        ax = axes2[0, 1]
        ax.hist(al_pl['h_hist'], bins=25, color='#17BECF', alpha=0.7,
                label=f'μ={al_pl["h_hist"].mean():.3f}')
        ax.axvline(al_hist_thr_final, color='k', lw=1.5, ls='--')
        ax.set_title('H_hist (Option 1) — FIX 6\nCalibrated to MC σ scale')
        ax.legend(fontsize=8); ax.set_xlabel('H_hist'); ax.grid(alpha=.3)

        ax = axes2[0, 2]
        if 'h_traj' in al_pl.columns and al_pl['h_traj'].abs().sum() > 0:
            ax.hist(al_pl['h_traj'], bins=25, color='#E377C2', alpha=0.7,
                    label=f'μ={al_pl["h_traj"].mean():.1f}')
            ax.set_title('H_traj (Option 3)\nPPF curve covariance entropy')
            ax.legend(fontsize=8); ax.set_xlabel('H_traj')
        else:
            ax.text(0.5, 0.5, 'H_traj not computed\n(set COMPUTE_TRAJECTORY=True)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('H_traj (Option 3) — disabled')
        ax.grid(alpha=.3)

        ax = axes2[1, 0]
        ax.scatter(al_pl['sigma_ppf'], al_pl['h_hist'], alpha=0.5, s=20, c='#17BECF')
        if len(al_pl) > 3:
            m, b = np.polyfit(al_pl['sigma_ppf'], al_pl['h_hist'], 1)
            xs = np.linspace(al_pl['sigma_ppf'].min(), al_pl['sigma_ppf'].max(), 100)
            r  = np.corrcoef(al_pl['sigma_ppf'], al_pl['h_hist'])[0, 1]
            ax.plot(xs, m*xs+b, 'k--', lw=1.5, label=f'r={r:.3f}')
            ax.legend(fontsize=8)
        ax.set_title('H_hist vs σ [KEY VALIDATION]\n(should be positive corr)')
        ax.set_xlabel('σ_ppf'); ax.set_ylabel('H_hist'); ax.grid(alpha=.3)

        ax = axes2[1, 1]
        ax.scatter(al_pl['sigma_ppf'], al_pl['h_gauss'], alpha=0.5, s=20, c='#9467BD')
        if len(al_pl) > 3:
            m, b = np.polyfit(al_pl['sigma_ppf'], al_pl['h_gauss'], 1)
            xs = np.linspace(al_pl['sigma_ppf'].min(), al_pl['sigma_ppf'].max(), 100)
            r  = np.corrcoef(al_pl['sigma_ppf'], al_pl['h_gauss'])[0, 1]
            ax.plot(xs, m*xs+b, 'k--', lw=1.5, label=f'r={r:.3f}')
            ax.legend(fontsize=8)
        ax.set_title('H_gauss vs σ (v5 baseline)\n(should be near-perfect r≈1)')
        ax.set_xlabel('σ_ppf'); ax.set_ylabel('H_gauss'); ax.grid(alpha=.3)

        ax = axes2[1, 2]
        ax.scatter(al_pl['h_gauss'], al_pl['h_hist'],
                   c=al_pl['sigma_ppf'], cmap='YlOrRd', s=30, alpha=0.6)
        ax.set_title('H_gauss vs H_hist\n(divergence from diagonal = bimodal)')
        ax.set_xlabel('H_gauss'); ax.set_ylabel('H_hist'); ax.grid(alpha=.3)

        plt.tight_layout()
        plt.savefig('qica_v9_entropy_comparison.png', dpi=150, bbox_inches='tight')
        print("[SAVED]  qica_v9_entropy_comparison.png")

    # ── Entropy-method showdown plot ──────────────────────────────────────────
    if showdown:
        labels   = list(showdown['results'].keys())
        short_labels = [l.split(' (')[0] for l in labels]
        sigmas   = [showdown['results'][l]['mean_sigma'] for l in labels]
        ppfs     = [showdown['results'][l]['mean_ppf'] for l in labels]
        overlaps = [showdown['results'][l]['overlap_with_baseline'] for l in labels]
        divs     = [showdown['results'][l]['diversity'] for l in labels]
        colors   = ['#888888' if l.startswith('baseline') else '#1B4FBF' for l in labels]

        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle(f"Entropy-Method Showdown — top-{AL_TOP_K} AL selection, "
                      f"pool of {showdown['pool_size']} promising/safe patterns",
                      fontsize=12, fontweight='bold')

        ax = axes3[0]
        ax.bar(short_labels, sigmas, color=colors)
        ax.axhline(sigmas[0], color='red', lw=1.2, ls='--', label='baseline level')
        ax.set_ylabel('Mean σ_ppf of selected set')
        ax.set_title('Does this rule find MORE uncertain\npatterns than plain QICA?')
        ax.tick_params(axis='x', rotation=35, labelsize=8); ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = axes3[1]
        ax.bar(short_labels, overlaps, color=colors)
        ax.axhline(0.8, color='orange', lw=1.2, ls='--', label='high-overlap line (0.8)')
        ax.set_ylabel('Jaccard overlap with baseline set')
        ax.set_title('Is this rule just re-picking\nthe same patterns as baseline?')
        ax.tick_params(axis='x', rotation=35, labelsize=8); ax.legend(fontsize=8); ax.grid(alpha=.3)

        ax = axes3[2]
        ax.bar(short_labels, ppfs, color=colors)
        ax.set_ylabel('Mean PPF of selected set')
        ax.set_title('PPF cost of choosing\nuncertainty over pure ranking')
        ax.tick_params(axis='x', rotation=35, labelsize=8); ax.grid(alpha=.3)

        plt.tight_layout()
        plt.savefig('qica_v9_entropy_showdown.png', dpi=150, bbox_inches='tight')
        print("[SAVED]  qica_v9_entropy_showdown.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("QICA-v9  FINAL SUMMARY  [ALL FIXES APPLIED]")
    print("=" * 74)
    print(f"  CHANGES (v8 → v9):")
    print(f"    FIX 12 Production AL rule  : 'combined' (z(H_hist)+z(H_traj)) is now the ONLY")
    print(f"                                 live AL-selection rule (Section 8), baked in directly —")
    print(f"                                 no per-run rule sweep needed, v8 already proved it wins.")
    print(f"    FIX 13 Trust-region audit  : two short identical-seed mini-runs (trust region vs.")
    print(f"                                 fully-free) measure whether fixing low-entropy positions")
    print(f"                                 actually helps THIS dataset/CNN — see audit block above.")
    print(f"    Carried forward            : FIX 8 escalating stagnation injection, FIX 9 Hamming")
    print(f"                                 diversity, FIX 10 hard-reset escape valve.")
    print()
    if audit_df is not None and len(audit_df) == 2:
        print(f"  TRUST-REGION AUDIT (this run, {AUDIT_GENS}-gen directional check):")
        for _, r in audit_df.iterrows():
            print(f"    {r['arm']:<32} best_ppf={r['best_ppf']:.4f}  AL_candidates={int(r['al_candidates'])}")
    print()
    if showdown and showdown.get('verdicts'):
        print(f"  ENTROPY SHOWDOWN RE-VERIFICATION (this run, RUN_ENTROPY_SHOWDOWN=True):")
        for label, v in showdown['verdicts'].items():
            print(f"    {label:<42} → {v}")
        print(f"\n  → Use the qica_v9_entropy_showdown.csv to confirm 'combined' is still the")
        print(f"    rule worth keeping before relying on it for a new CNN/dataset.")
    print()
    print(f"  RESULTS:")
    print(f"    Best PPF       : {best_ppf:.4f}  ({'SAFE' if best_ppf <= PPF_LIMIT else 'EXCEEDS'})")
    print(f"    Best σ_ppf     : {best_sig:.4f}")
    print(f"    Best H_hist    : {best_hh:.3f}  half_width=±{HIST_HALF_WIDTH:.3f}  bins={HIST_BINS}")
    print(f"    Best H_gauss   : {best_hg:.3f}  σ_equiv={gaussian_to_sigma(best_hg):.4f}")
    print(f"    Best H_traj    : {best_ht:.1f}  (PPF curve covariance)")
    print(f"    Best H_combined: {best_hc:.2f}  (z(H_hist)+z(H_traj) — the live AL-selection score)")
    print(f"    Cycle          : {best_cyc:.1f} days")
    print(f"    AL candidates  : {len(al_cands)} unique  (combined-score threshold used: {al_hist_thr_final:.4f})")
    print(f"    Max stagnation escalation round reached: {max(hist['stagnation_round']) if hist['stagnation_round'] else 0}")
    print()
    print(f"  PAPER CONTRIBUTIONS (v9):")
    print(f"    1. Calibrated histogram Shannon H: first use in nuclear fuel loading AL")
    print(f"       that is correctly scaled to the surrogate's own MC-dropout noise floor")
    print(f"    2. PPF trajectory entropy: first use of burnup curve covariance")
    print(f"       → captures shape uncertainty, not just peak PPF uncertainty")
    print(f"    3. Data-driven combined AL rule: z(H_hist)+z(H_traj) selected via a")
    print(f"       one-run, post-hoc 6-rule showdown rather than hand-picked, then made")
    print(f"       the sole production AL metric — every entropy use has a stated role")
    print(f"       (AL flagging / search-process control / trust-region scoping) and a")
    print(f"       measured justification, not an assumed one.")
    print(f"    4. Trust-region audit: identical-seed, identical-budget A/B check of the")
    print(f"       modal-type position lock, instead of leaving it as an unverified prior.")
    print(f"    5. Escalating multi-elite stagnation escape + hard-reset valve: novel")
    print(f"       basin-escape mechanism for nuclear fuel optimisation.")
    print("=" * 74)