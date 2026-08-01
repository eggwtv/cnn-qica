"""
=============================================================================
qica_v5.py  —  Uncertainty-Aware QICA with Shannon Entropy  |  CNN v8
=============================================================================
Extends qica_v4.py with two independent Shannon entropy implementations:

  ENTROPY_MODE = 'none'   Pure v4 behaviour (σ threshold for AL, full search)
  ENTROPY_MODE = 'mc'     MC Prediction Entropy for AL candidate selection
  ENTROPY_MODE = 'trust'  Trust Region Entropy reduces QICA search space
  ENTROPY_MODE = 'both'   Full pipeline: MC entropy + trust region together

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION 1 — MC Prediction Entropy  (AL use case)
─────────────────────────────────────────────────────────────────────────────
Standard σ picks patterns that are uncertain in amplitude.
Entropy captures the SHAPE of the uncertainty distribution — it is high
when the CNN's predictions split into two clusters (bimodal), even when σ
is only moderate.  This is BALD (Bayesian Active Learning by Disagreement)
adapted for regression.

  H_epistemic[n] = 0.5 × log(2πe × σ²[n])       (Gaussian differential entropy)

Equivalent to BALD for regression when aleatoric noise is constant.
  - High H → CNN is confused about this pattern → simulate it first
  - Low H  → CNN is confident → trust the surrogate

The AL priority score switches from  σ / ppf_pred  (v4)
                                  to  H / ppf_pred  (v5 mc mode)

─────────────────────────────────────────────────────────────────────────────
IMPLEMENTATION 2 — Trust Region Entropy  (QICA search reduction)
─────────────────────────────────────────────────────────────────────────────
From train_type_freq.npy:  freq[p, t] = P(type t at position p in training)

Per-position entropy:
  H_pos[p] = −Σ_t  freq[p,t] × log(freq[p,t])    (in nats)

Physical meaning:
  Low H_pos  → training data consistently uses one type at position p
               → that position is effectively fixed (no freedom to optimise)
  High H_pos → many types appear at position p in training
               → this is a real decision variable worth optimising

Strategy:
  Sort positions by H_pos.  Bottom (1 - ENTROPY_FREE_FRAC) are FIXED to
  their modal training type.  Top ENTROPY_FREE_FRAC are FREE for QICA.

  e.g. ENTROPY_FREE_FRAC = 0.65 → keep ~20 free positions, fix ~11.
  Result: search space shrinks from 9^31 → 9^20, QICA converges faster
  and avoids the fixed positions' implicitly constrained region.

─────────────────────────────────────────────────────────────────────────────
INPUTS:
  cnn_v8_model.keras        (or cnn_v4_model.keras as fallback)
  cnn_v8_config.json        (or cnn_v4_config.json)
  train_type_freq_v8.npy    (or train_type_freq.npy)

OUTPUTS:
  qica_v5_best_patterns.csv
  qica_v5_al_candidates.csv
  qica_v5_entropy_analysis.png
  qica_v5_convergence.png
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print("qica_v5.py  —  Uncertainty-Aware QICA with Shannon Entropy\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# ── File paths (v8 preferred; v4 fallback) ───────────────────────────────────
MODEL_PATH  = 'cnn_v9_model.keras'   if os.path.exists('cnn_v9_model.keras')  else 'cnn_v4_model.keras'
CONFIG_PATH = 'cnn_v9_config.json'   if os.path.exists('cnn_v9_config.json')  else 'cnn_v4_config.json'
TRUST_PATH  = 'train_type_freq_v9.npy' if os.path.exists('train_type_freq_v9.npy') else 'train_type_freq.npy'

# ── Entropy mode ─────────────────────────────────────────────────────────────
ENTROPY_MODE = 'mc'          # 'none' | 'mc' | 'trust' | 'both'

# MC entropy config
AL_ENTROPY_THRESHOLD = -1.0    # log-entropy threshold (≈σ=0.08 for Gaussian)
                               # H = 0.5*log(2πe*σ²) → H(-2.2) ≈ σ=0.08
# Trust region config
ENTROPY_FREE_FRAC    = 0.65    # top 65% of positions by entropy are free to optimise
                               # bottom 35% fixed to their modal training type

# ── Fitness weighting ─────────────────────────────────────────────────────────
PPF_LIMIT         = 3.5
W_PPF_PENALTY     = 80.0  
W_PPF_SOFT        = 6.0    # gentler; was losing ~7 days of cycle for 0.006 PPF gain
W_UNCERTAINTY     = 40.0   # restored to v4 level — 80 was killing late exploration
W_TRUST           = 0.0 #20.0
W_ENTROPY_BONUS   = 5.0
W_MONOTONICITY    = 10.0

# ── Active learning ───────────────────────────────────────────────────────────
AL_SIGMA_THRESHOLD = 0.08      # kept for σ-based comparison / 'none' mode
AL_TOP_K           = 50
AL_ROUNDS          = 0

# ── QICA hyperparameters ──────────────────────────────────────────────────────
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
SEED               = 42

# CNN geometry
GRID_ROWS = 6
GRID_COLS = 6
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
# SECTION 2 — ConvResBlock (must match cnn_v8.py exactly)
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

print(f"[LOAD] Model  : {MODEL_PATH}")
print(f"       Config : {CONFIG_PATH}")
print(f"       Trust  : {TRUST_PATH}")

for p in [MODEL_PATH, CONFIG_PATH]:
    if not os.path.exists(p):
        print(f"[ERROR] Missing: {p}\n  → Run cnn_v8.py (or cnn_v4.py) first.")
        sys.exit(1)

model = keras.models.load_model(MODEL_PATH, compile=False,
                                custom_objects={'ConvResBlock': ConvResBlock})
print(f"  Model loaded  : input={model.input_shape}  output={model.output_shape}")

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

print(f"  Indices : ppf_max={IDX_PPF_MAX}  cycle={IDX_CYCLE}  rho={IDX_RHO}")
print(f"  PPF limit : {PPF_LIMIT}  |  MC samples : {MC_SAMPLES}")
print(f"  Entropy mode : {ENTROPY_MODE}\n")


# =============================================================================
# SECTION 4 — LOAD TRUST-REGION FREQUENCIES
# =============================================================================

if os.path.exists(TRUST_PATH):
    type_freq = np.load(TRUST_PATH).astype(np.float32)
    print(f"[TRUST] Loaded {TRUST_PATH}  shape={type_freq.shape}")
else:
    print(f"[TRUST] {TRUST_PATH} not found — uniform fallback.")
    type_freq = np.ones((N_POS, N_TYPES), dtype=np.float32) / N_TYPES


# =============================================================================
# SECTION 5 — SHANNON ENTROPY UTILITIES
# =============================================================================

def compute_position_entropy(freq: np.ndarray) -> np.ndarray:
    """
    Per-position Shannon entropy from training type frequencies.

    H[p] = -Σ_t  freq[p,t] × log(freq[p,t])

    Args:
        freq : (N_POS, N_TYPES) — per-position type frequencies
    Returns:
        h    : (N_POS,) — entropy in nats [0, log(N_TYPES)]
    """
    h = -np.sum(freq * np.log(freq + 1e-10), axis=1)
    return h.astype(np.float32)


def analyze_trust_region(freq: np.ndarray, free_frac: float = ENTROPY_FREE_FRAC):
    """
    Identify free vs fixed positions using trust region entropy.

    Strategy:
      Sort positions by H_pos.
      Top (free_frac) fraction: high entropy → free to optimise.
      Bottom fraction: low entropy → fix to modal training type.

    Returns:
        free_mask    : (N_POS,) bool — True = QICA optimises this position
        fixed_types  : (N_POS,) int  — modal type for fixed positions (1-indexed)
        h_pos        : (N_POS,) float — per-position entropy values
        n_free       : int — number of free positions
    """
    h_pos      = compute_position_entropy(freq)
    n_free     = max(1, int(np.round(N_POS * free_frac)))
    n_fixed    = N_POS - n_free
    rank       = np.argsort(h_pos)[::-1]   # descending entropy
    free_mask  = np.zeros(N_POS, dtype=bool)
    free_mask[rank[:n_free]] = True

    # Modal type (1-indexed) for each position
    fixed_types = (np.argmax(freq, axis=1) + 1).astype(np.int32)

    return free_mask, fixed_types, h_pos, n_free


def compute_mc_entropy(mc_preds: np.ndarray) -> np.ndarray:
    """
    Differential entropy of MC Dropout prediction distribution (BALD-style).

    For a Gaussian with std σ:
      H = 0.5 × log(2πe × σ²)

    This is equivalent to BALD (epistemic component) for regression when
    aleatoric noise is constant across inputs.

    Args:
        mc_preds : (MC_SAMPLES, N) — PPF predictions from N forward passes
    Returns:
        entropy  : (N,) — in nats; higher = CNN is more uncertain
    """
    sigma   = mc_preds.std(axis=0) + 1e-10   # (N,) in physical units
    entropy = 0.5 * np.log(2.0 * np.pi * np.e * sigma**2)
    return entropy.astype(np.float32)


def entropy_to_sigma_equiv(h: float) -> float:
    """Convert differential entropy back to equivalent σ for comparison."""
    return float(np.sqrt(np.exp(2*h) / (2*np.pi*np.e)))


# =============================================================================
# SECTION 6 — TRUST REGION SETUP (runs once before QICA)
# =============================================================================

free_mask, fixed_types, h_pos, n_free = analyze_trust_region(type_freq, ENTROPY_FREE_FRAC)

print("=" * 60)
print("TRUST REGION ENTROPY ANALYSIS")
print("=" * 60)
print(f"  Per-position entropy range : {h_pos.min():.3f} – {h_pos.max():.3f} nats")
print(f"  Max possible (uniform)     : {np.log(N_TYPES):.3f} nats")
print(f"  Entropy free frac          : {ENTROPY_FREE_FRAC}  ({n_free}/{N_POS} positions free)")
print(f"  Fixed positions            : {N_POS - n_free}")
print()

if ENTROPY_MODE in ('trust', 'both'):
    print("  FREE positions (QICA optimises):")
    free_pos_list = np.where(free_mask)[0].tolist()
    print(f"    {free_pos_list}")
    print("  FIXED positions → modal type:")
    fixed_pos_list = np.where(~free_mask)[0]
    for p in fixed_pos_list:
        print(f"    pos_{p:02d}  H={h_pos[p]:.3f}  modal_type={fixed_types[p]}")
    print()
else:
    print(f"  [ENTROPY_MODE='{ENTROPY_MODE}'] Trust region NOT active — all positions free.\n")


# =============================================================================
# SECTION 7 — ENTROPY ANALYSIS PLOT
# =============================================================================

fig_e, axes_e = plt.subplots(1, 3, figsize=(18, 5))
fig_e.suptitle(
    f"Shannon Entropy Analysis  |  ENTROPY_MODE='{ENTROPY_MODE}'  "
    f"free_frac={ENTROPY_FREE_FRAC}  ({n_free}/{N_POS} free)",
    fontsize=11, fontweight='bold'
)

# Plot 1: Per-position entropy bar chart
ax = axes_e[0]
colors_pos = ['#2CA02C' if free_mask[p] else '#D62728' for p in range(N_POS)]
ax.bar(range(N_POS), h_pos, color=colors_pos, alpha=0.8)
thresh_h = np.sort(h_pos)[::-1][n_free-1]   # entropy at boundary
ax.axhline(thresh_h, color='black', lw=1.5, ls='--',
           label=f'Threshold H={thresh_h:.3f}')
ax.axhline(np.log(N_TYPES), color='grey', lw=1, ls=':',
           label=f'Max (uniform) = {np.log(N_TYPES):.2f}')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#2CA02C', label=f'Free ({n_free} pos)'),
    Patch(facecolor='#D62728', label=f'Fixed ({N_POS-n_free} pos)'),
], fontsize=8)
ax.set_xlabel('Position index'); ax.set_ylabel('Entropy H (nats)')
ax.set_title('Per-Position Entropy\n(green=free, red=fixed)')
ax.grid(alpha=0.3)

# Plot 2: Entropy grid heatmap
ax = axes_e[1]
disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
pi = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_LAYOUT[r, c] >= 0:
            disp[r, c] = h_pos[pi]; pi += 1
cmap_e = plt.cm.RdYlGn.copy(); cmap_e.set_bad('lightgrey')
im = ax.imshow(disp, cmap=cmap_e, aspect='auto', vmin=0, vmax=np.log(N_TYPES))
plt.colorbar(im, ax=ax, label='H (nats)')
pi = 0
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if GRID_MASK[r, c]:
            status = 'F' if free_mask[pi] else 'X'
            ax.text(c, r, f'{status}\n{h_pos[pi]:.1f}',
                    ha='center', va='center', fontsize=6.5)
            pi += 1
ax.set_title('Entropy Heatmap\n(F=Free, X=Fixed, green=flexible)')
ax.set_xticks([]); ax.set_yticks([])

# Plot 3: Type frequency for a high-entropy vs low-entropy position
ax = axes_e[2]
high_p = int(np.argmax(h_pos))
low_p  = int(np.argmin(h_pos))
x_ = np.arange(N_TYPES)
ax.bar(x_-0.2, type_freq[high_p], 0.35,
       label=f'pos_{high_p} (H={h_pos[high_p]:.2f}, {"free" if free_mask[high_p] else "fixed"})',
       color='#2CA02C', alpha=0.8)
ax.bar(x_+0.2, type_freq[low_p],  0.35,
       label=f'pos_{low_p} (H={h_pos[low_p]:.2f}, {"free" if free_mask[low_p] else "fixed"})',
       color='#D62728', alpha=0.8)
ax.set_xticks(x_); ax.set_xticklabels([f'T{i+1}' for i in range(N_TYPES)], fontsize=8)
ax.set_xlabel('Assembly type'); ax.set_ylabel('Frequency in training data')
ax.set_title('Type Frequency:\nHighest vs Lowest Entropy Position')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('qica_v5_entropy_analysis.png', dpi=150, bbox_inches='tight')
print("[SAVED]  qica_v5_entropy_analysis.png\n")
plt.close()


# =============================================================================
# SECTION 8 — GRID BUILDER + INVERSE TRANSFORM
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
# SECTION 9 — MC DROPOUT EVALUATOR (enhanced with entropy output)
# =============================================================================

def evaluate_batch(patterns_int: np.ndarray) -> dict:
    """
    Evaluate a batch of loading patterns with MC Dropout.

    Returns both σ (amplitude uncertainty) and H (entropy / distributional
    uncertainty) for ppf_max. Use H for AL when ENTROPY_MODE includes 'mc'.

    Args:
        patterns_int : (B, 31) integer patterns (types 1–9)
    Returns:
        dict with keys: ppf_mean, ppf_std, ppf_entropy, cycle_mean,
                        rho_mean, keff_mean, ppf_steps, fitness, trust_penalty
    """
    if patterns_int.ndim == 1:
        patterns_int = patterns_int.reshape(1, -1)
    B    = patterns_int.shape[0]
    grid = pattern_to_grid(patterns_int)
    X_tf = tf.constant(grid, dtype=tf.int32)

    mc_preds_sc = np.stack([
        model(X_tf, training=True).numpy()
        for _ in range(MC_SAMPLES)
    ])   # (MC_SAMPLES, B, N_OUTPUTS)

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
    keff_mean  = 1.0 / (1.0 - rho_mean / 1e5)
    ppf_steps  = mean_real[:, IDX_STEPS_S:IDX_STEPS_E]

    # MC Prediction Entropy (BALD-style epistemic uncertainty)
    # Uses the full MC sample array for ppf_max (in physical units)
    mc_ppf_phys = mc_preds_sc[:, :, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
    ppf_entropy = compute_mc_entropy(mc_ppf_phys)   # (B,)

    # Trust-region penalty
    trust_penalty = np.zeros(B, dtype=np.float32)
    for b in range(B):
        pat = patterns_int[b]
        nlp = sum(-np.log(float(type_freq[p, int(pat[p])-1]) + 1e-6)
                  for p in range(N_POS)) / N_POS
        trust_penalty[b] = float(nlp)

    # Monotonicity bonus (PPF should decrease after BA burnout ~step 3)
    late       = ppf_steps[:, 3:]
    diffs      = late[:, 1:] - late[:, :-1]
    mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / 27.0)

    ppf_excess  = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    #fitness     = (cycle_mean
    #               - W_PPF_PENALTY * ppf_excess
    #               - W_UNCERTAINTY  * ppf_std
    #               - W_TRUST        * trust_penalty
    #               + mono_bonus)
    # AFTER
    fitness = (cycle_mean
            - W_PPF_SOFT * ppf_mean          # soft gradient within safe range
            - W_PPF_PENALTY * ppf_excess   # hard penalty above limit (unchanged)
            - W_UNCERTAINTY * ppf_std
            + mono_bonus)

    return {
        'ppf_mean'     : ppf_mean,
        'ppf_std'      : ppf_std,
        'ppf_entropy'  : ppf_entropy,
        'cycle_mean'   : cycle_mean,
        'rho_mean'     : rho_mean,
        'keff_mean'    : keff_mean,
        'ppf_steps'    : ppf_steps,
        'fitness'      : fitness,
        'trust_penalty': trust_penalty,
    }


def is_al_candidate(ppf_std: float, ppf_entropy: float) -> bool:
    """
    Returns True if pattern should be queried by the physics simulator.
    Switches between σ threshold (mode='none') and entropy threshold (mode='mc'/'both').
    """
    if ENTROPY_MODE in ('mc', 'both'):
        return ppf_entropy >= AL_ENTROPY_THRESHOLD
    return ppf_std >= AL_SIGMA_THRESHOLD


# =============================================================================
# SECTION 10 — QUANTUM COUNTRY  (extended with free/fixed position support)
# =============================================================================

class QuantumCountry:
    """
    One candidate loading pattern in quantum superposition.

    When ENTROPY_MODE includes 'trust':
      - Fixed positions (low H_pos) always use their modal training type
      - Free positions (high H_pos) are fully optimised by QICA
      - Revolution and assimilation only update free positions

    When ENTROPY_MODE is 'none' or 'mc':
      - All positions are free (standard v4 behaviour)
    """

    def __init__(self, q_state: np.ndarray = None):
        if q_state is None:
            raw = np.ones((N_POS, N_TYPES), dtype=np.float32)
            self.q_state = raw / raw.sum(axis=1, keepdims=True)
        else:
            self.q_state = q_state.copy().astype(np.float32)

        # Enforce fixed positions immediately
        if ENTROPY_MODE in ('trust', 'both'):
            for p in range(N_POS):
                if not free_mask[p]:
                    self.q_state[p]               = 0.0
                    self.q_state[p, fixed_types[p]-1] = 1.0

        self.measured   = None
        self.fitness    = -np.inf
        self.ppf_mean   = 9.0
        self.ppf_std    = 0.0
        self.ppf_entropy= -np.inf
        self.cycle_mean = 0.0
        self.keff_mean  = 0.0
        self.trust_pen  = 0.0

    def collapse(self, temperature: float = 1.0) -> np.ndarray:
        """Sample a concrete integer pattern from the quantum probability distribution."""
        logits = np.log(self.q_state + 1e-10) / max(temperature, 0.01)
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=1, keepdims=True)

        self.measured = np.array([
            np.random.choice(N_TYPES, p=probs[i]) + 1
            for i in range(N_POS)
        ], dtype=np.int32)

        # Override fixed positions regardless of sampling
        if ENTROPY_MODE in ('trust', 'both'):
            for p in range(N_POS):
                if not free_mask[p]:
                    self.measured[p] = fixed_types[p]

        return self.measured

    def entropy(self) -> float:
        """Shannon entropy of the quantum state (diversity measure for fitness bonus)."""
        return float(-np.sum(self.q_state * np.log(self.q_state + 1e-10)))

    def quantum_assimilate(self, imperialist: 'QuantumCountry', beta: float, temp: float):
        """
        Blend colony's q_state toward imperialist's q_state.
        Only free positions are updated in trust-region mode.
        """
        if ENTROPY_MODE in ('trust', 'both'):
            for p in range(N_POS):
                if free_mask[p]:
                    self.q_state[p] = ((1.0 - beta) * self.q_state[p]
                                        + beta * imperialist.q_state[p])
        else:
            self.q_state = ((1.0 - beta) * self.q_state
                             + beta * imperialist.q_state)

        self.q_state = np.maximum(self.q_state, 1e-10)
        self.q_state /= self.q_state.sum(axis=1, keepdims=True)

        # Re-fix fixed positions
        if ENTROPY_MODE in ('trust', 'both'):
            for p in range(N_POS):
                if not free_mask[p]:
                    self.q_state[p]               = 0.0
                    self.q_state[p, fixed_types[p]-1] = 1.0

    def quantum_revolution(self, rate: float, temperature: float):
        """
        Randomly reset some positions (exploration).
        Only free positions are reset in trust-region mode.
        """
        for p in range(N_POS):
            if ENTROPY_MODE in ('trust', 'both') and not free_mask[p]:
                continue
            if np.random.random() < rate:
                alpha = np.ones(N_TYPES) * max(temperature, 0.05)
                self.q_state[p] = np.random.dirichlet(alpha)

    def clone(self) -> 'QuantumCountry':
        c = QuantumCountry(self.q_state)
        c.measured    = self.measured.copy() if self.measured is not None else None
        c.fitness     = self.fitness
        c.ppf_mean    = self.ppf_mean
        c.ppf_std     = self.ppf_std
        c.ppf_entropy = self.ppf_entropy
        c.cycle_mean  = self.cycle_mean
        c.keff_mean   = self.keff_mean
        c.trust_pen   = self.trust_pen
        return c


# =============================================================================
# SECTION 11 — EMPIRE
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
# SECTION 12 — QICA v5 OPTIMIZER
# =============================================================================
# ==========================================================
# Load training patterns for warm-start seeding
# ==========================================================
# ── CNN-guided warm-start (after model is loaded) ────────────────────────────
# Runs CNN inference on training data to find lowest-predicted-PPF patterns.
# Uses CNN predictions (not labels) for consistency with the QICA objective.

X_train_seed   = None    # (N, 31) int patterns
X_grid_seed    = None    # (N, 6, 6) grids for CNN inference
ppf_cnn_seed   = None    # CNN-predicted ppf_max for each training pattern

def _load_seeds_via_cnn():
    """Load training patterns and rank them by CNN-predicted PPF."""
    global X_train_seed, X_grid_seed, ppf_cnn_seed

    csv_path = 'ml_dataset_constrained.csv'
    if not os.path.exists(csv_path):
        print("[SEED] ml_dataset_constrained.csv not found — warm-start disabled")
        return

    try:
        df  = pd.read_csv(csv_path, skiprows=1, engine='python', on_bad_lines='skip')
        lc  = [f'loading_{i}' for i in range(N_POS)]
        if not all(c in df.columns for c in lc):
            print(f"[SEED] Column 'loading_0' not found in CSV — warm-start disabled")
            return

        X_raw = df[lc].values.astype(np.int32)          # (N, 31)
        N     = len(X_raw)

        # Build 6×6 grids
        grids = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)
        pi = 0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    grids[:, r, c] = X_raw[:, pi]; pi += 1

        # Run CNN inference in batches (no MC dropout — deterministic mean prediction)
        batch_size = 128
        ppf_preds  = []
        for i in range(0, N, batch_size):
            batch = tf.constant(grids[i:i+batch_size], dtype=tf.int32)
            preds = model(batch, training=False).numpy()   # (B, 35)
            ppf_sc  = preds[:, IDX_PPF_MAX]
            ppf_real = ppf_sc * ym_scale[IDX_PPF_MAX] + ym_mean[IDX_PPF_MAX]
            ppf_preds.extend(ppf_real.tolist())

        ppf_arr = np.array(ppf_preds, dtype=np.float32)
        X_train_seed  = X_raw
        X_grid_seed   = grids
        ppf_cnn_seed  = ppf_arr

        n_below_2 = (ppf_arr < 2.0).sum()
        print(f"[SEED] CNN-ranked {N} training patterns")
        print(f"       Predicted PPF range: {ppf_arr.min():.3f} – {ppf_arr.max():.3f}")
        print(f"       Patterns with CNN-pred PPF < 2.0: {n_below_2} ({n_below_2/N*100:.1f}%)")

    except Exception as e:
        print(f"[SEED] Failed: {e} — warm-start disabled")

_load_seeds_via_cnn()


# =============================================================================
# IMPROVEMENT 1 — POSITION SENSITIVITY  (numerical Jacobian via CNN)
# =============================================================================
# Runs once at startup. For each of the 31 positions, we ask:
#   "How much does swapping this position to any other type change predicted PPF?"
# Positions with large sensitivity get a higher revolution rate in QICA —
# meaning QICA spends more exploration budget on positions that actually matter.
#
# Technical note: the CNN takes int32 input via Embedding, so we cannot use
# tf.GradientTape (gradients don't flow through integer lookup). We instead
# use a numerical Jacobian: perturb each position to all 9 types, measure
# |ΔPPF|, and take the max. Runs in ~10s on CPU for 50 patterns × 9 types × 31 pos.

def compute_position_sensitivity() -> np.ndarray:
    """
    Rank positions by their influence on predicted PPF_max.

    For each position p, we replace it with each of the 9 types (keeping
    all other positions at their training values) and measure the mean
    absolute change in CNN-predicted PPF across n_sample low-PPF patterns.

    Returns:
        sensitivities : (N_POS,) float32, normalised to sum=1.
    """
    if X_train_seed is None or ppf_cnn_seed is None:
        print("[SENS] No seed data — uniform position sensitivity (warm-start skipped)")
        return np.ones(N_POS, dtype=np.float32) / N_POS

    n_sample    = min(50, len(X_train_seed))
    top_idx     = np.argsort(ppf_cnn_seed)[:n_sample]
    base_grids  = X_grid_seed[top_idx].copy()                  # (n_sample, 6, 6)
    base_ppf    = ppf_cnn_seed[top_idx]                        # (n_sample,)

    sensitivities = np.zeros(N_POS, dtype=np.float32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] < 0:
                continue
            type_deltas = []
            for t in range(1, N_TYPES + 1):
                perturbed          = base_grids.copy()
                perturbed[:, r, c] = t
                preds_sc = model(tf.constant(perturbed, dtype=tf.int32),
                                 training=False).numpy()        # (n_sample, N_OUTPUTS)
                ppf_p = (preds_sc[:, IDX_PPF_MAX] * ym_scale[IDX_PPF_MAX]
                         + ym_mean[IDX_PPF_MAX])
                type_deltas.append(float(np.abs(ppf_p - base_ppf).mean()))
            sensitivities[pi] = max(type_deltas)
            pi += 1

    # Normalise to sum=1 so we can use as a probability-like weight
    total = sensitivities.sum()
    if total > 1e-10:
        sensitivities /= total
    else:
        sensitivities[:] = 1.0 / N_POS

    top3 = np.argsort(sensitivities)[-3:][::-1].tolist()
    print(f"[SENS] Position sensitivities computed ({n_sample} low-PPF patterns).")
    print(f"       Top-3 most impactful positions: {top3}  "
          f"(sensitivity: {sensitivities[top3[0]]:.4f}, "
          f"{sensitivities[top3[1]]:.4f}, {sensitivities[top3[2]]:.4f})")
    return sensitivities

pos_sensitivity = compute_position_sensitivity()


class QICAv5:
    """
    Quantum ICA with Shannon Entropy — v5.

    Changes from v4:
      - QuantumCountry respects free/fixed positions (trust region entropy)
      - AL candidate selection uses MC entropy or σ based on ENTROPY_MODE
      - History tracks entropy statistics for convergence analysis
    """

    def __init__(self):
        self.elite_archive = []
        self.al_candidates = []
        self.history = {
            'gen': [], 'best_fitness': [], 'mean_fitness': [],
            'best_cycle': [], 'best_ppf': [], 'mean_ppf_std': [],
            'mean_ppf_entropy': [], 'n_empires': [],
            'temperature': [], 'al_count': [],
        }
        self.stagnation_count = 0
        self.last_best_ppf = 9.0
    def _temperature(self, gen):
        r = gen / MAX_GEN
        return QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** r

    def _revolution_rate(self, gen):
        r = gen / MAX_GEN
        return REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * r

    def _initialize_population(self) -> list:
        countries = []
        for bias_t in range(1, N_TYPES + 1):
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.04
            q[:, bias_t-1] = 0.68
            q /= q.sum(axis=1, keepdims=True)
            c = QuantumCountry(q)   # constructor enforces fixed positions
            countries.append(c)
        # Add to _initialize_population() — seed from low-PPF training data
        # Assumes: X_train shape (N, 31), ppf_train shape (N,)
        
        # AFTER (uses the safe globals from _try_load_seeds)
        if X_train_seed is not None and ppf_cnn_seed is not None:
            n_seeds   = min(8, len(X_train_seed))
            top_k_idx = np.argsort(ppf_cnn_seed)[:n_seeds]
            for rank_i, idx in enumerate(top_k_idx):
                pat = X_train_seed[idx]
                q   = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
                for p in range(N_POS):
                    t = int(pat[p])
                    if 1 <= t <= N_TYPES:
                        q[p, t - 1] = 0.84
                q /= q.sum(axis=1, keepdims=True)
                countries.append(QuantumCountry(q))
            print(f"  [SEED] {n_seeds} CNN-guided seeds added "
                f"(pred PPF: {ppf_cnn_seed[top_k_idx[0]]:.3f} – "
                f"{ppf_cnn_seed[top_k_idx[-1]]:.3f})")

        #for _ in range(N_COUNTRIES - N_TYPES):
        #    countries.append(QuantumCountry())
        while len(countries) < N_COUNTRIES:
            countries.append(QuantumCountry())
        return countries

    def _evaluate_all(self, countries: list, temperature: float) -> list:
        patterns = np.stack([c.collapse(temperature) for c in countries])
        result   = evaluate_batch(patterns)

        ppf_arr  = result['ppf_mean']
        std_arr  = result['ppf_std']
        ent_arr  = result['ppf_entropy']
        cyc_arr  = result['cycle_mean']
        rho_arr  = result['rho_mean']
        stp_arr  = result['ppf_steps']
        tpen_arr = result['trust_penalty']

        ent_bonus = np.array([
            W_ENTROPY_BONUS * c.entropy() / (N_POS * N_TYPES)
            for c in countries
        ], dtype=np.float32)

        ppf_excess = np.maximum(0.0, ppf_arr - PPF_LIMIT)
        late       = stp_arr[:, 3:]
        diffs      = late[:, 1:] - late[:, :-1]
        mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / 27.0)

        fitness_arr = (
            cyc_arr
            - W_PPF_SOFT * ppf_arr
            - W_PPF_PENALTY * ppf_excess
            - W_UNCERTAINTY * std_arr
            - W_TRUST * tpen_arr
            + mono_bonus
            + ent_bonus
        )

        for i, c in enumerate(countries):
            c.fitness     = float(fitness_arr[i])
            c.ppf_mean    = float(ppf_arr[i])
            c.ppf_std     = float(std_arr[i])
            c.ppf_entropy = float(ent_arr[i])
            c.cycle_mean  = float(cyc_arr[i])
            c.keff_mean   = float(1.0 / (1.0 - float(rho_arr[i]) / 1e5))
            c.trust_pen   = float(tpen_arr[i])

        # AL candidate flagging (σ or entropy threshold depending on mode)
        #ppf_25pct = np.percentile(ppf_arr, 25)
        ppf_10pct = np.percentile(ppf_arr, 10)
        for i, c in enumerate(countries):
            if (is_al_candidate(c.ppf_std, c.ppf_entropy)
                    and c.ppf_mean <= ppf_10pct):
                #priority = (c.ppf_entropy if ENTROPY_MODE in ('mc', 'both')
                #            else c.ppf_std) / (c.ppf_mean + 1e-6)
                priority = (
                    c.ppf_entropy *
                    c.cycle_mean /
                    (c.ppf_mean + 1e-6)
                )
                self.al_candidates.append({
                    'pattern'    : c.measured.tolist(),
                    'pred_ppf'   : c.ppf_mean,
                    'sigma_ppf'  : c.ppf_std,
                    'entropy_ppf': c.ppf_entropy,
                    'cycle'      : c.cycle_mean,
                    'priority'   : float(priority),
                    'mode'       : ENTROPY_MODE,
                })

        return countries

    def _form_empires(self, countries: list) -> list:
        sorted_c = sorted(countries, key=lambda c: c.fitness, reverse=True)
        imps, cols = sorted_c[:N_EMPIRES], sorted_c[N_EMPIRES:]
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
                col.quantum_revolution(rev_rate, temp)

    def _intra_competition(self, empires, temperature):
        all_cols = [c for emp in empires for c in emp.colonies]
        if not all_cols:
            return
        self._evaluate_all(all_cols, temperature)
        for emp in empires:
            if not emp.colonies:
                continue
            best_i = max(range(len(emp.colonies)),
                         key=lambda i: emp.colonies[i].fitness)
            if emp.colonies[best_i].fitness > emp.imperialist.fitness:
                emp.colonies[best_i], emp.imperialist = (
                    emp.imperialist, emp.colonies[best_i])

    def _empire_collapse(self, empires) -> list:
        if len(empires) <= 1:
            return empires
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
                        c.ppf_mean, c.cycle_mean, c.ppf_std, c.ppf_entropy
                    ))
        self.elite_archive.sort(key=lambda x: x[0], reverse=True)
        seen, unique = set(), []
        for entry in self.elite_archive:
            key = tuple(entry[1])
            if key not in seen:
                seen.add(key); unique.append(entry)
        self.elite_archive = unique[:ELITE_SIZE]

    def _log(self, gen, empires, temp):
        all_fit = ([e.imperialist.fitness for e in empires] +
                   [c.fitness for e in empires for c in e.colonies])
        all_std = ([e.imperialist.ppf_std for e in empires] +
                   [c.ppf_std for e in empires for c in e.colonies])
        all_ent = ([e.imperialist.ppf_entropy for e in empires] +
                   [c.ppf_entropy for e in empires for c in e.colonies])
        best = self.elite_archive[0] if self.elite_archive else (0, None, 9.0, 0.0, 0.0, -10.0)

        self.history['gen'].append(gen)
        self.history['best_fitness'].append(float(max(all_fit)))
        self.history['mean_fitness'].append(float(np.mean(all_fit)))
        self.history['best_cycle'].append(float(best[3]))
        self.history['best_ppf'].append(float(best[2]))
        self.history['mean_ppf_std'].append(float(np.mean(all_std)))
        self.history['mean_ppf_entropy'].append(float(np.mean(all_ent)))
        self.history['n_empires'].append(len(empires))
        self.history['temperature'].append(temp)
        self.history['al_count'].append(len(self.al_candidates))

        if gen % 25 == 0 or gen == MAX_GEN:
            sigma_eq = entropy_to_sigma_equiv(best[5]) if ENTROPY_MODE in ('mc','both') else best[4]
            print(
                f"  Gen {gen:4d}/{MAX_GEN} | empires={len(empires):2d} | "
                f"best_ppf={best[2]:.3f} σ={best[4]:.4f} H={best[5]:.2f} | "
                f"cycle={best[3]:6.1f}d | fit={best[0]:7.2f} | "
                f"T={temp:.3f} | AL_q={len(self.al_candidates)}"
            )

    def run(self) -> dict:
        print("=" * 70)
        print(f"QICA-v5  |  Entropy mode: '{ENTROPY_MODE}'")
        if ENTROPY_MODE in ('trust', 'both'):
            print(f"  Trust region: {n_free}/{N_POS} positions free  "
                  f"(top {ENTROPY_FREE_FRAC*100:.0f}% by H_pos)")
        if ENTROPY_MODE in ('mc', 'both'):
            print(f"  MC entropy AL threshold: H ≥ {AL_ENTROPY_THRESHOLD} "
                  f"(≈ σ_equiv ≥ {entropy_to_sigma_equiv(AL_ENTROPY_THRESHOLD):.3f})")
        print(f"  Population: {N_COUNTRIES}  Empires: {N_EMPIRES}  Gens: {MAX_GEN}")
        print(f"  ≈{MAX_GEN*N_COUNTRIES:,} evaluations × {MC_SAMPLES} MC samples\n")

        t0 = time.time()

        print("[INIT] Initializing population ...")
        countries = self._initialize_population()
        temp      = self._temperature(0)
        countries = self._evaluate_all(countries, temp)
        empires   = self._form_empires(countries)
        self._update_elite(empires)
        best0 = self.elite_archive[0]
        print(f"  Initial best: ppf={best0[2]:.3f}  cycle={best0[3]:.1f}d  "
              f"σ={best0[4]:.4f}  H={best0[5]:.2f}\n")

        print("[RUN] Main optimisation loop ...")
        for gen in range(1, MAX_GEN + 1):
            temp     = self._temperature(gen)
            rev_rate = self._revolution_rate(gen)
            self._assimilation_step(empires, ASSIMILATION_COEFF, temp, rev_rate)
            self._intra_competition(empires, temp)
            self._update_elite(empires)
            empires  = self._empire_collapse(empires)

            # AFTER (mutation-based — explores neighbors of current best)
            # ── Stagnation detection + mutation injection ──────────────────────────────
            if len(empires) == 1:
                current_best_ppf = self.elite_archive[0][2] if self.elite_archive else 9.0
                if abs(current_best_ppf - self.last_best_ppf) < 0.005:
                    self.stagnation_count += 1          # ← this was missing
                else:
                    self.stagnation_count = 0
                    self.last_best_ppf = current_best_ppf

                if self.stagnation_count >= 20:
                    best_pat = self.elite_archive[0][1]
                    n_inject = N_COUNTRIES // 4
                    new_c = []
                    for _ in range(n_inject):
                        n_mutate = np.random.randint(6, 15)
                        mut_pos  = np.random.choice(N_POS, n_mutate, replace=False)
                        q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.02
                        for p, t in enumerate(best_pat):
                            q[p, int(t) - 1] = 0.84
                        q /= q.sum(axis=1, keepdims=True)
                        for p in mut_pos:
                            q[p] = np.ones(N_TYPES, dtype=np.float32) / N_TYPES
                        new_c.append(QuantumCountry(q))
                    patterns = np.stack([c.collapse(min(temp * 3.0, 1.5)) for c in new_c])
                    result   = evaluate_batch(patterns)
                    for i, c in enumerate(new_c):
                        c.fitness     = float(result['fitness'][i])
                        c.ppf_mean    = float(result['ppf_mean'][i])
                        c.ppf_std     = float(result['ppf_std'][i])
                        c.ppf_entropy = float(result['ppf_entropy'][i])
                        c.cycle_mean  = float(result['cycle_mean'][i])
                    empires[0].colonies.extend(new_c)
                    self.stagnation_count = 0
                    print(f"\n  [INJECT] gen={gen}  {n_inject} mutations of best "
                        f"(ppf={self.elite_archive[0][2]:.3f}) injected")
                    
            self._log(gen, empires, temp)
            if len(empires) == 1 and len(empires[0].colonies) < 3:
                print(f"\n[CONVERGED]  Single empire at gen {gen}")
                break

        t_total = time.time() - t0
        print(f"\n[DONE]  {t_total:.1f}s  |  AL candidates: {len(self.al_candidates)}")
        return {
            'elite_archive': self.elite_archive,
            'history'      : self.history,
            'al_candidates': self.al_candidates,
        }


# =============================================================================
# SECTION 13 — SIMULATOR STUB 
# =============================================================================

def simulate_pattern(loading_pattern_1d: np.ndarray) -> dict:
    """
    Physics simulation stub (OpenMC / PARCS / Serpent).
    Set AL_ROUNDS > 0 after implementing this function.
    """
    raise NotImplementedError(
        "simulate_pattern() is a stub.\n"
        "Fill in with your physics simulator call, then set AL_ROUNDS > 0."
    )


# =============================================================================
# SECTION 14 — RUN QICA
# =============================================================================

if __name__ == '__main__':

    optimizer = QICAv5()
    results   = optimizer.run()

    elite    = results['elite_archive']
    al_cands = results['al_candidates']
    hist     = results['history']

    # Active learning rounds (AL_ROUNDS=0 → skip)
    if AL_ROUNDS > 0:
        print(f"\n[AL] {AL_ROUNDS} round(s) requested — implement simulate_pattern() first.")

    # ── Print top-5 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP LOADING PATTERNS FOUND")
    print("=" * 70)
    print(f"{'Rank':<5} {'PPF_max':<10} {'σ_ppf':<9} {'H_ppf':<9} "
          f"{'Cycle(d)':<12} {'Fitness':<12} Status")
    print("-" * 70)
    for rank, entry in enumerate(elite[:5], 1):
        fit, pat, ppf, cyc, sig, ent = entry
        safe = "✓ SAFE"   if ppf <= PPF_LIMIT else "✗ EXCEEDS"
        conf_h = "H✓" if ent < AL_ENTROPY_THRESHOLD else "H⚠"
        conf_s = "σ✓" if sig < AL_SIGMA_THRESHOLD  else "σ⚠"
        print(f"  #{rank}   {ppf:7.4f}    {sig:7.4f}   {ent:7.2f}   "
              f"{cyc:9.1f}    {fit:9.2f}   {safe}  {conf_s}  {conf_h}")
        print(f"         Types: {list(pat)}")

    best_fit, best_pat, best_ppf, best_cyc, best_sig, best_ent = elite[0]
    print(f"\nBEST PATTERN:")
    print(f"  PPF_max      : {best_ppf:.4f}  ({'✓ SAFE' if best_ppf <= PPF_LIMIT else '✗ EXCEEDS'})")
    print(f"  σ_ppf        : {best_sig:.4f}  "
          f"({'σ✓ confident' if best_sig < AL_SIGMA_THRESHOLD else 'σ⚠ verify'})")
    print(f"  H_ppf        : {best_ent:.2f}  "
          f"({'H✓ confident' if best_ent < AL_ENTROPY_THRESHOLD else 'H⚠ uncertain'})")
    print(f"  σ_equiv(H)   : {entropy_to_sigma_equiv(best_ent):.4f}")
    print(f"  Cycle length : {best_cyc:.1f} days")
    print(f"  Fitness      : {best_fit:.2f}")
    print(f"  Pattern      : {list(best_pat)}")

    if ENTROPY_MODE in ('trust', 'both'):
        print(f"\n  Fixed positions (set to modal type, not optimised):")
        for p in np.where(~free_mask)[0]:
            print(f"    pos_{p:02d} → type {fixed_types[p]}  (H={h_pos[p]:.3f})")

    # ── Save results ───────────────────────────────────────────────────────────
    best_df = pd.DataFrame([
        {'rank': i+1,
         'ppf_max': ppf, 'sigma_ppf': sig, 'entropy_ppf': ent,
         'entropy_sigma_equiv': entropy_to_sigma_equiv(ent),
         'cycle_length_days': cyc, 'fitness': fit,
         'ppf_safe': ppf <= PPF_LIMIT,
         'sigma_confident': sig < AL_SIGMA_THRESHOLD,
         'entropy_confident': ent < AL_ENTROPY_THRESHOLD,
         **{f'pos_{j}': int(pat[j]) for j in range(N_POS)}}
        for i, (fit, pat, ppf, cyc, sig, ent) in enumerate(elite)
    ])
    best_df.to_csv('qica_v5_best_patterns.csv', index=False)
    print(f"\n[SAVED]  qica_v5_best_patterns.csv  ({len(best_df)} patterns)")

    if al_cands:
        al_df = (pd.DataFrame(al_cands)
                 .sort_values('priority', ascending=False)
                 .drop_duplicates(subset=['pattern'])
                 .head(AL_TOP_K))
        al_df.to_csv('qica_v5_al_candidates.csv', index=False)
        print(f"[SAVED]  qica_v5_al_candidates.csv  ({len(al_df)} candidates)")
        print(f"  AL selection method: {'MC entropy' if ENTROPY_MODE in ('mc','both') else 'σ threshold'}")
        best_al = al_df.iloc[0]
        print(f"  Top candidate: ppf={best_al['pred_ppf']:.3f}  "
              f"σ={best_al['sigma_ppf']:.4f}  H={best_al['entropy_ppf']:.2f}")

    # ── Convergence plots ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(
        f"QICA-v5  ENTROPY_MODE='{ENTROPY_MODE}'  |  "
        f"Best PPF={best_ppf:.4f}  Cycle={best_cyc:.1f}d  "
        f"σ={best_sig:.4f}  H={best_ent:.2f}  AL={len(al_cands)}",
        fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.40, wspace=0.35)

    # 1. Fitness convergence
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist['gen'], hist['best_fitness'],  '#1B4FBF', lw=2,   label='Best')
    ax.plot(hist['gen'], hist['mean_fitness'],  '#F5A623', lw=1.5, ls='--', label='Mean')
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')
    ax.set_title('Fitness Convergence'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2. PPF convergence
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hist['gen'], hist['best_ppf'], '#D62728', lw=2)
    ax.axhline(PPF_LIMIT, color='orange', lw=1.5, ls='--', label=f'Limit={PPF_LIMIT}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Best ppf_max')
    ax.set_title('PPF Convergence (PRIMARY)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3. Cycle length
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(hist['gen'], hist['best_cycle'], '#2CA02C', lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Cycle length (days)')
    ax.set_title('Cycle Length Convergence'); ax.grid(alpha=0.3)

    # 4. σ and entropy convergence
    ax  = fig.add_subplot(gs[0, 3])
    ax2 = ax.twinx()
    l1, = ax.plot(hist['gen'], hist['mean_ppf_std'], '#9467BD', lw=2, label='Mean σ')
    l2, = ax2.plot(hist['gen'], hist['mean_ppf_entropy'], '#17BECF', lw=2, ls='--', label='Mean H')
    ax.axhline(AL_SIGMA_THRESHOLD,    color='#9467BD', lw=1, ls=':', alpha=0.6)
    ax2.axhline(AL_ENTROPY_THRESHOLD, color='#17BECF', lw=1, ls=':', alpha=0.6)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean σ (ppf_max)', color='#9467BD')
    ax2.set_ylabel('Mean H (entropy)', color='#17BECF')
    ax.set_title('Uncertainty: σ vs Entropy')
    ax.legend(handles=[l1,l2], fontsize=7); ax.grid(alpha=0.3)

    # 5. Empire collapse
    ax = fig.add_subplot(gs[0, 4])
    ax.plot(hist['gen'], hist['n_empires'], '#8C564B', lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('N empires')
    ax.set_title('Empire Collapse (convergence)'); ax.grid(alpha=0.3)

    # 6. AL candidate count
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(hist['gen'], hist['al_count'], '#E377C2', lw=2)
    ax.axhline(AL_TOP_K, color='orange', lw=1.5, ls='--', label=f'Export top-{AL_TOP_K}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cumulative AL candidates')
    ax.set_title(f'AL Candidates\n(mode: {"entropy" if ENTROPY_MODE in ("mc","both") else "sigma"})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 7. Best pattern grid
    ax = fig.add_subplot(gs[1, 1])
    grid_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid_disp[r, c] = float(best_pat[pi]); pi += 1
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad('lightgrey')
    im = ax.imshow(grid_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly type')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                status = '' if free_mask[pi] else '*'   # * = fixed
                ax.text(c, r, f'T{int(best_pat[pi])}{status}',
                        ha='center', va='center', fontsize=7, fontweight='bold')
                pi += 1
    ax.set_title(f'Best Pattern\nPPF={best_ppf:.3f}  H={best_ent:.2f}\n(* = trust-fixed)')
    ax.set_xticks([]); ax.set_yticks([])

    # 8. Position entropy overlay on grid
    ax = fig.add_subplot(gs[1, 2])
    ent_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                ent_disp[r, c] = h_pos[pi]; pi += 1
    cmap2 = plt.cm.RdYlGn.copy(); cmap2.set_bad('lightgrey')
    im2 = ax.imshow(ent_disp, cmap=cmap2, aspect='auto', vmin=0, vmax=np.log(N_TYPES))
    plt.colorbar(im2, ax=ax, label='H_pos (nats)')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                col_tx = 'black' if free_mask[pi] else 'red'
                ax.text(c, r, f'{h_pos[pi]:.1f}',
                        ha='center', va='center', fontsize=7, color=col_tx)
                pi += 1
    ax.set_title('Trust Region Entropy\n(red=fixed, black=free)')
    ax.set_xticks([]); ax.set_yticks([])

    # 9. σ vs H scatter for elite archive
    ax = fig.add_subplot(gs[1, 3])
    e_sigs = [e[4] for e in elite]
    e_ents = [e[5] for e in elite]
    e_ppfs = [e[2] for e in elite]
    sc = ax.scatter(e_sigs, e_ents, c=e_ppfs, cmap='RdYlGn_r', s=60, alpha=0.8,
                    vmin=PPF_LIMIT-0.5, vmax=PPF_LIMIT+0.5)
    plt.colorbar(sc, ax=ax, label='PPF_max')
    ax.axvline(AL_SIGMA_THRESHOLD,    color='#9467BD', lw=1.5, ls='--', label=f'σ thr={AL_SIGMA_THRESHOLD}')
    ax.axhline(AL_ENTROPY_THRESHOLD,  color='#17BECF', lw=1.5, ls='--', label=f'H thr={AL_ENTROPY_THRESHOLD}')
    ax.set_xlabel('σ_ppf'); ax.set_ylabel('H_ppf (entropy)')
    ax.set_title('Elite Archive: σ vs H\n(colour = PPF_max)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 10. AL candidates scatter
    ax = fig.add_subplot(gs[1, 4])
    if al_cands:
        al_df_plot = pd.DataFrame(al_cands)
        ax.scatter(al_df_plot['pred_ppf'], al_df_plot['entropy_ppf'],
                   alpha=0.5, s=20, color='#D62728', label=f'AL (n={len(al_cands)})')
        ax.axhline(AL_ENTROPY_THRESHOLD, color='orange', lw=1.5, ls='--',
                   label=f'H threshold={AL_ENTROPY_THRESHOLD}')
        ax.set_xlabel('Predicted PPF_max')
        ax.set_ylabel('MC entropy H_ppf')
        ax.set_title(f'AL Candidates\n(entropy-flagged, low-PPF)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No AL candidates\nflagged', ha='center', va='center',
                transform=ax.transAxes)

    plt.savefig('qica_v5_convergence.png', dpi=150, bbox_inches='tight')
    print("[SAVED]  qica_v5_convergence.png")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QICA-v5  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Entropy mode         : {ENTROPY_MODE}")
    if ENTROPY_MODE in ('trust', 'both'):
        print(f"  Search reduction     : {N_POS - n_free}/{N_POS} positions fixed")
        print(f"    9^{N_POS} → 9^{n_free} effective search space")
    if ENTROPY_MODE in ('mc', 'both'):
        print(f"  AL method            : MC Prediction Entropy (BALD-style)")
        print(f"  H threshold          : {AL_ENTROPY_THRESHOLD} "
              f"(σ_equiv={entropy_to_sigma_equiv(AL_ENTROPY_THRESHOLD):.3f})")
    print(f"  Best PPF             : {best_ppf:.4f}  "
          f"({'SAFE' if best_ppf <= PPF_LIMIT else 'EXCEEDS — raise W_PPF_PENALTY'})")
    print(f"  Best σ_ppf           : {best_sig:.4f}")
    print(f"  Best H_ppf           : {best_ent:.2f}  "
          f"(σ_equiv={entropy_to_sigma_equiv(best_ent):.4f})")
    print(f"  Cycle length         : {best_cyc:.1f} days")
    print(f"  AL candidates        : {len(al_cands)}")
    print()
    print("  NEXT STEPS:")
    print("  1. Implement simulate_pattern() for patterns flagged by entropy.")
    print("  2. Set AL_ROUNDS=1, re-run to retrain CNN on simulator labels.")
    print("  3. After each AL round, σ and H in the low-PPF region should drop.")
    print("  4. Stop when H_ppf_best < AL_ENTROPY_THRESHOLD consistently.")
    print("=" * 70)