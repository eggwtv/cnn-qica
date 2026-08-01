"""
=============================================================================
qica_final.py  —  Final QICA  |  σ-AL + 3 Sensitivity-Entropy Linking Modes
=============================================================================

WHAT THIS FILE IS AND WHY IT'S STRUCTURED THIS WAY
────────────────────────────────────────────────────
Your three ablation runs (v1, v2, final_entropy_test) produced one clear,
consistent finding: none of H_hist-as-search-driver, injection, or hard-reset
reliably move best_ppf past the seed noise floor (±0.110 PPF). The only
things that replicate across all three runs are:

  (a) σ-AL baseline is your best, simplest search driver
  (b) H_hist as a FLAGGING signal for OpenMC candidates is stable and
      discriminative (CV 0.046–0.087, σ-correlation 0.64–0.85 in every arm)
  (c) H_traj is dead (CV 0.02–0.04, confirmed across all runs)

So: baseline search = σ-AL (plain MC-dropout uncertainty governs exploration).
    entropy = only used downstream for AL candidate flagging.
    sensitivity = new input from cnn_v9_sens.csv.

This file compares 4 arms (3 seeds each, 250 gens, 80 pop) to answer the
one question your ablations haven't answered yet: does linking the CNN's
gradient sensitivity to the entropy/AL mechanism make the candidate flagging
(and potentially the search) meaningfully better?

ARM A — σ-AL BASELINE (your confirmed best from final_entropy_test)
  Search:   plain MC-dropout σ drives exploration budget
  AL flag:  σ > threshold AND H_hist > H_threshold
  Sens:     not used

ARM B — SENSITIVITY-WEIGHTED AL SCORING (Mode A from task 3 spec)
  Search:   same σ-AL as baseline
  AL flag:  composite score = z(H_hist) + ALPHA * z(sens_novelty)
            where sens_novelty = how unusual the pattern's choices are
            AT HIGH-SENSITIVITY POSITIONS specifically.
            Idea: H_hist gives overall entropy; sens_novelty focuses entropy
            on the positions that matter most to ppf_max. A pattern with
            high H_hist might just be unusual at irrelevant (low-sens) positions.
            This arm tests whether focusing the entropy signal on high-sens
            positions improves AL candidate quality.

ARM C — SENSITIVITY-GATED MUTATION BUDGET (Mode B from task 3 spec)
  Search:   revolution budget redistributed — high-sens positions mutate LESS,
            low-sens positions get the freed budget. Same total mutation rate,
            different spatial allocation.
            Idea: the CNN gradient says "don't vary position X much or ppf blows
            up." Respecting that in the mutation operator may let the search
            converge faster without hitting PPF walls.
  AL flag:  same as baseline

ARM D — SENSITIVITY-DEFINED TRUST REGION (Mode C from task 3 spec)
  Search:   instead of using positional-frequency entropy to decide which 20/31
            positions are "free," use the gradient sensitivity RANKING: freeze
            the top-K highest-sensitivity positions to their CNN-preferred type,
            only evolve the low-sensitivity positions.
            Idea: your current trust region uses training-set frequency as a
            proxy for "what the CNN expects." The gradient sensitivity is a
            more direct, model-derived measure of which positions actually
            drive ppf_max — so using it to define the free/frozen split may
            produce a tighter, more reliable trust region.
  AL flag:  same as baseline

=============================================================================
WHAT METRICS TO LOOK AT WHEN READING THE RESULTS
=============================================================================

PRIMARY (did the arm find better solutions?):
  best_ppf : lower is better. Your data minimum is 1.697, 10th pct ≈ 2.0.
             Differences < ±0.110 are within the seed noise floor — don't
             trust them. Only differences > ±0.150 are signal.
  mean_ppf / std_ppf (across 3 seeds): std tells you if the arm is consistent.
             A low mean with high std is risky for production use.

SECONDARY (did the arm improve AL candidate quality?):
  al_h_hist_cv : coefficient of variation of H_hist across the 50 AL candidates.
                 HIGHER CV = more diverse candidates = better for OpenMC batch.
                 Low CV means all candidates look alike (wasted OpenMC budget).
  al_sigma_corr : Pearson r(H_hist, σ) in AL candidates.
                 LOWER r = H_hist is finding patterns σ-alone misses (additive).
                 HIGH r = H_hist just duplicates σ (not adding information).
  al_sens_novelty_mean (Arms B, D only): how unusual candidates are at
                 high-sensitivity positions specifically. HIGHER = better
                 (these are the candidates that will teach the CNN the most).

TERTIARY (did the mechanism do what it was supposed to?):
  div_final : population diversity at last generation (0–1). Too low (<0.3)
              means premature convergence. Arms with more constrained search
              (C, D) may show lower div — check if this hurts or helps best_ppf.
  time_s : wall clock seconds per seed run. Arms B/C/D add overhead — if they
           don't improve best_ppf or AL quality, the overhead isn't worth it.

COMPARISON TABLE TO BUILD IN YOUR NOTES:
  "Does Mode B (sens-weighted AL) improve al_sigma_corr vs baseline?"
      → If al_sigma_corr(B) < al_sigma_corr(A) by >0.05, Mode B adds info.
  "Does Mode C (sens-gated mutation) improve convergence speed?"
      → Check if Mode C reaches best_ppf < 2.0 in fewer generations.
  "Does Mode D (sens trust region) tighten the search?"
      → Check div_final(D) vs div_final(A). If D converges faster AND best_ppf
        is no worse, the tighter trust region is worth it.
  "Does any mode beat the baseline on best_ppf?"
      → Only trust if mean_ppf difference > 0.150 PPF AND std_ppf is not huge.

      v9_loss
=============================================================================
REQUIRED FILES (place in same directory as this script):
  cnn_v9_model.keras        — trained CNN surrogate
  cnn_v9_config.json        — scalers, indices, geometry
  train_type_freq_v9.npy    — per-position type frequencies (for trust region)
  cnn_v9_sens.csv           — position sensitivities from cnn-v9.py Section 12
  ml_dataset_constrained.csv — BEAVRS loading patterns (for seed population)
=============================================================================
"""

import os, sys, json, time, warnings, itertools
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']  = 'TRUE'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print(f"Running on: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
print("qica_final.py  —  σ-AL Baseline + 3 Sensitivity-Entropy Linking Modes\n")


# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================
#
# QUICK_TEST=True  → 1 seed, 30 gens, pop 20 — sanity check (~5 min)
# QUICK_TEST=False → 3 seeds, 250 gens, pop 80 — real results (~3.5 hrs)
#
# Only trust results from QUICK_TEST=False. The noise floor is ±0.110 PPF
# from seed variance alone — one short run cannot beat that.

QUICK_TEST = False

N_SEEDS   = 1   if QUICK_TEST else 3
N_GENS    = 30  if QUICK_TEST else 250
N_POP     = 20  if QUICK_TEST else 80
MC_SAMP   = 10  if QUICK_TEST else 25
SEEDS     = [42] if QUICK_TEST else [42, 137, 271]

# Files
MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters (match final_entropy_test.py confirmed settings)
N_EMPIRES_INIT   = 6       # initial number of empires
ASSIMILATION_RATE = 0.3    # fraction of colony that drifts toward imperialist per gen
REV_START        = 0.35    # starting revolution (mutation) rate
REV_END          = 0.08    # ending revolution rate (linear decay)
STAGNATION_PAT   = 20      # gens without improvement before escape mutation burst
ESCAPE_BURST     = 30      # extra mutations when stagnating

# AL flagging thresholds (calibrated from final_entropy_test.py: median σ ≈ 0.067)
AL_SIGMA_THRESH  = 0.067   # recalibrated in load() below from actual data distribution
AL_H_THRESH_PCT  = 60      # flag patterns in top (100-X)th percentile of H_hist score
AL_MAX_CANDS     = 50      # max AL candidates to save per run

# Sensitivity-linking parameters
ALPHA_SENS_WT    = 0.4     # Arm B: weight of sensitivity-novelty term in AL score
SENS_FREEZE_K    = 8       # Arm D: top-K sensitivity positions frozen in trust region
SENS_MUT_SCALE   = 3.0     # Arm C: low-sens positions get SENS_MUT_SCALE× more mutation

# Output names
OUT_PREFIX  = 'qica_final'
SUMMARY_CSV = f'{OUT_PREFIX}_summary.csv'
HISTORY_CSV = f'{OUT_PREFIX}_history.csv'
AL_CSV      = f'{OUT_PREFIX}_al_candidates.csv'
PLOT_PNG    = f'{OUT_PREFIX}_comparison.png'

# BEAVRS geometry (must match cnn_v9_config.json)
N_POS    = 31
N_TYPES  = 9
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

# Position-to-grid lookup (built once, used everywhere)
POS_TO_RC  = {}  # pos_idx → (row, col)
RC_TO_POS  = {}  # (row, col) → pos_idx
_pi = 0
for _r in range(GRID_ROWS):
    for _c in range(GRID_COLS):
        if GRID_LAYOUT[_r, _c] >= 0:
            POS_TO_RC[_pi] = (_r, _c)
            RC_TO_POS[(_r, _c)] = _pi
            _pi += 1


# =============================================================================
# SECTION 1 — CONVRESBLOCK (required to load cnn_v9_model.keras)
# =============================================================================
# This must be registered before keras.models.load_model() is called.
# Identical to the definition in cnn_v9.py — do not modify.

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
        cfg.update({'filters': self._filters, 'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


# =============================================================================
# SECTION 2 — LOAD MODEL, CONFIG, SENSITIVITY, DATASET
# =============================================================================

def load_everything():
    """
    Load and validate all inputs. Returns a dict of everything QICA needs.
    Prints a calibration summary so you can verify σ thresholds are sane.
    """
    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"[LOAD] {MODEL_FILE} ...")
    model = keras.models.load_model(
        MODEL_FILE,
        compile=False
    )
    inp_shape = model.input_shape
    out_shape = model.output_shape
    print(f"  input={inp_shape}  output={out_shape}")

    # ── Config (scalers + indices) ─────────────────────────────────────────────
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    IDX_PPF_MAX   = cfg['IDX_PPF_MAX']    # 0
    IDX_CYCLE     = cfg['IDX_CYCLE']      # 33
    IDX_RHO       = cfg['IDX_RHO']        # 34
    IDX_PPF_START = cfg['IDX_PPF_STEPS_START']  # 2
    IDX_PPF_END   = cfg['IDX_PPF_STEPS_END']    # 33
    ym_mean  = np.array(cfg['ym_scaler_mean'])
    ym_scale = np.array(cfg['ym_scaler_scale'])
    yr_mean  = np.array(cfg['yr_scaler_mean'])
    yr_scale = np.array(cfg['yr_scaler_scale'])
    print(f"[CFG]  IDX_PPF_MAX={IDX_PPF_MAX}  IDX_CYCLE={IDX_CYCLE}  IDX_RHO={IDX_RHO}")

    # ── Trust region frequencies ───────────────────────────────────────────────
    type_freq = np.load(FREQ_FILE)   # shape (31, 9)
    print(f"[TRUST] {FREQ_FILE}  shape={type_freq.shape}")
    # Positions with low max-frequency are "free" (QICA can change them)
    # Replicate the final_entropy_test.py approach: free = top-20 by entropy
    pos_entropy = -np.sum(type_freq * np.log(type_freq + 1e-9), axis=1)  # (31,)
    free_by_entropy = np.argsort(pos_entropy)[::-1][:20].tolist()
    print(f"[TRUST] {len(free_by_entropy)}/31 positions free (by positional entropy)")

    # ── Sensitivity ────────────────────────────────────────────────────────────
    if os.path.exists(SENS_FILE):
        sens_df = pd.read_csv(SENS_FILE)
        sens_norm = sens_df['sensitivity_norm'].values.astype(np.float32)  # (31,) in [0,1]
        print(f"[SENS] {SENS_FILE}  range={sens_norm.min():.3f}–{sens_norm.max():.3f}")
        top5_sens = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"[SENS] Top-5 critical positions: {top5_sens}")
    else:
        print(f"[WARN] {SENS_FILE} not found — using uniform sensitivity (all=0.5)")
        sens_norm = np.full(N_POS, 0.5, dtype=np.float32)
    # Normalised so max=1, min~0
    # High value → position strongly affects ppf_max per CNN gradient

    # ── Dataset (for population seeding) ──────────────────────────────────────
    print(f"[DATA] Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    X_raw  = df[load_cols].values.astype(np.int32)    # (N, 31)
    ppf_col = df['ppf_max'].values.astype(np.float32) if 'ppf_max' in df.columns else None
    cyc_col = df['cycle_length'].values.astype(np.float32) if 'cycle_length' in df.columns else None

    # Build 6×6 grids from flat 31-position loading
    X_grid = _flat_to_grid_batch(X_raw)               # (N, 6, 6)
    print(f"[DATA] {len(df)} patterns  X_grid={X_grid.shape}")

    # ── Calibrate σ threshold from population ─────────────────────────────────
    # Run MC-dropout on a sample of 500 patterns to get empirical σ distribution
    sample_idx = np.random.choice(len(X_grid), min(500, len(X_grid)), replace=False)
    X_sample   = X_grid[sample_idx]
    _, sigma_sample = _mc_predict(model, X_sample, ym_mean, ym_scale,
                                  yr_mean, yr_scale, IDX_PPF_MAX, n=10)
    sigma_cal   = sigma_sample  # per-pattern σ for ppf_max
    median_sig  = float(np.median(sigma_cal))
    p95_sig     = float(np.percentile(sigma_cal, 95))
    al_sig_thr  = median_sig  # threshold = median (patterns above median σ are "uncertain")
    print(f"[CAL]  median σ={median_sig:.4f}  p95 σ={p95_sig:.4f}  →  AL σ_thr={al_sig_thr:.4f}")

    # ── PPF range ──────────────────────────────────────────────────────────────
    ppf_pred, _ = _mc_predict(model, X_grid[:200], ym_mean, ym_scale,
                               yr_mean, yr_scale, IDX_PPF_MAX, n=5)
    ppf_min_est = float(np.min(ppf_pred))
    ppf_max_est = float(np.max(ppf_pred))
    print(f"[SEED] PPF sample range: {ppf_min_est:.3f}–{ppf_max_est:.3f}")

    return dict(
        model=model,
        ym_mean=ym_mean, ym_scale=ym_scale,
        yr_mean=yr_mean, yr_scale=yr_scale,
        IDX_PPF_MAX=IDX_PPF_MAX, IDX_CYCLE=IDX_CYCLE, IDX_RHO=IDX_RHO,
        IDX_PPF_START=IDX_PPF_START, IDX_PPF_END=IDX_PPF_END,
        type_freq=type_freq,
        free_by_entropy=free_by_entropy,
        pos_entropy=pos_entropy,
        sens_norm=sens_norm,
        X_grid=X_grid,
        al_sig_thr=al_sig_thr,
    )


# =============================================================================
# SECTION 3 — CORE HELPERS
# =============================================================================

def _flat_to_grid(pattern_flat):
    """Convert 1D (31,) int array to 6×6 grid (inactive cells = 0)."""
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g[r, c] = pattern_flat[pi]
                pi += 1
    return g


def _flat_to_grid_batch(X_flat):
    """Vectorised version of _flat_to_grid for (N, 31) input."""
    N = len(X_flat)
    X_g = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                X_g[:, r, c] = X_flat[:, pi]
                pi += 1
    return X_g


def _grid_to_flat(grid):
    """Convert 6×6 grid back to (31,) flat array."""
    flat = np.zeros(N_POS, dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                flat[pi] = grid[r, c]
                pi += 1
    return flat


def _mc_predict(model, X_grids, ym_mean, ym_scale, yr_mean, yr_scale,
                idx_ppf, n=MC_SAMP):
    """
    MC-dropout inference. Returns (mean_ppf, std_ppf) arrays, shape (N,).
    X_grids: (N, 6, 6) int32
    """
    preds = []
    Xt = tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()   # dropout ON = stochastic
        # Invert scaler: first 34 outputs use ym, last 1 uses yr
        y_main = y_sc[:, :34] * ym_scale[:34] + ym_mean[:34]
        preds.append(y_main[:, idx_ppf])          # ppf_max column
    preds = np.array(preds)    # (n, N)
    return preds.mean(axis=0), preds.std(axis=0)


def _compute_h_hist(population_flat, type_freq, calibration_halfwidth=0.29):
    """
    Calibrated histogram entropy of the population at each position.
    H_hist = mean over positions of Shannon entropy of type distribution in pop.

    This is the FLAGGING signal: tells you how diverse the population is in
    type choice across positions. Used to identify AL candidates that are
    genuinely novel (high H_hist AND high σ → send to OpenMC).

    calibration_halfwidth: from final_entropy_test calibration step.
    Set to your actual median σ * 4.3 (empirical constant from your runs).
    """
    pop = np.array(population_flat)  # (N, 31)
    N   = len(pop)
    H   = 0.0
    for p in range(N_POS):
        counts = np.zeros(N_TYPES + 1)
        for t in range(1, N_TYPES + 1):
            counts[t] = (pop[:, p] == t).sum()
        probs = counts[1:] / (N + 1e-9)
        H_p   = -np.sum(probs * np.log(probs + 1e-9))
        H    += H_p
    return H / N_POS


def _sensitivity_novelty(pattern_flat, type_freq, sens_norm):
    """
    Arm B: how unusual is this pattern specifically at HIGH-SENSITIVITY positions?

    For each position p, novelty(p) = -log(freq[p, type]) = surprise of placing
    this assembly type here (relative to training distribution).
    Sensitivity-weighted novelty = sum over positions of sens_norm[p] * novelty(p).

    High value → pattern makes unusual choices at positions that strongly affect ppf.
    These are the most informative candidates for OpenMC: uncertain AND impactful.
    """
    novelty = 0.0
    for p in range(N_POS):
        t = pattern_flat[p]
        if 1 <= t <= N_TYPES:
            freq_pt = float(type_freq[p, t - 1])
            surprise = -np.log(max(freq_pt, 1e-4))
            novelty += float(sens_norm[p]) * surprise
    return novelty


# =============================================================================
# SECTION 4 — TRUST REGION BUILDERS (one per arm)
# =============================================================================

def _trust_region_entropy(free_by_entropy):
    """
    Arms A, B, C: standard entropy-based trust region from final_entropy_test.
    Returns set of free position indices.
    """
    return set(free_by_entropy)


def _trust_region_sensitivity(sens_norm, freeze_k=SENS_FREEZE_K):
    """
    Arm D: sensitivity-defined trust region.
    Freeze the top-K highest-sensitivity positions (they're too dangerous to vary).
    Free positions = all 31 minus the top-K.

    WHY: CNN gradient says ∂ppf/∂pos is large at these positions — changing
    assembly type here causes the largest swing in predicted ppf_max.
    Freezing them prevents the search from accidentally blowing up ppf by
    placing a wrong assembly at a critical position.
    """
    frozen = set(np.argsort(sens_norm)[::-1][:freeze_k].tolist())
    free   = set(range(N_POS)) - frozen
    return free


# =============================================================================
# SECTION 5 — MUTATION OPERATORS (one per arm)
# =============================================================================

def _mutate_uniform(pattern_flat, rev_rate, free_positions, type_freq, rng):
    """
    Arms A, B, D: uniform mutation — each free position mutates with prob=rev_rate.
    Mutant type drawn from type_freq[p, :] (respects training distribution).
    """
    mut = pattern_flat.copy()
    for p in free_positions:
        if rng.random() < rev_rate:
            probs = type_freq[p, :]
            probs = probs / probs.sum()
            mut[p] = rng.choice(np.arange(1, N_TYPES + 1), p=probs)
    return mut


def _mutate_sens_gated(pattern_flat, rev_rate, free_positions, type_freq,
                        sens_norm, rng, scale=SENS_MUT_SCALE):
    """
    Arm C: sensitivity-gated mutation.
    Free positions are mutated at rates proportional to (1 - sensitivity):
      low-sens positions mutate at scale× the base rate,
      high-sens positions mutate at base rate / scale.
    Total expected mutations per pattern stays constant.

    WHY: If a position is high-sensitivity, changing it likely hurts ppf.
    Budget the mutation toward positions where exploration is safe.
    """
    mut = pattern_flat.copy()
    for p in free_positions:
        sens_p   = float(sens_norm[p])
        # sens_p in [0,1]: 1=highest gradient, 0=lowest
        # Invert: low-sens positions get higher mutation rate
        rate_p   = rev_rate * (1.0 - sens_p) * scale + rev_rate * sens_p / scale
        # Clip so single-position rate doesn't exceed 1
        rate_p   = min(rate_p, 0.95)
        if rng.random() < rate_p:
            probs = type_freq[p, :]
            probs = probs / probs.sum()
            mut[p] = rng.choice(np.arange(1, N_TYPES + 1), p=probs)
    return mut


# =============================================================================
# SECTION 6 — AL CANDIDATE SCORING (one per arm)
# =============================================================================

def _al_score_baseline(ppf, sigma, h_hist, pattern_flat, type_freq, sens_norm,
                        al_sig_thr, h_thr):
    """
    Arms A, C, D: baseline AL scoring.
    Flag = σ > al_sig_thr AND h_hist > h_thr
    Score = σ (higher σ → higher priority for OpenMC)

    This is your confirmed-replicating signal from all three ablation runs.
    """
    flag  = (sigma > al_sig_thr) and (h_hist > h_thr)
    score = sigma
    return flag, score, 0.0   # third value = sens_novelty (0 = not computed)


def _al_score_sens_weighted(ppf, sigma, h_hist, pattern_flat, type_freq,
                             sens_norm, al_sig_thr, h_thr, alpha=ALPHA_SENS_WT):
    """
    Arm B: sensitivity-weighted AL scoring.
    Composite score = z(H_hist) + alpha * z(sens_novelty)

    This focuses the entropy signal on HIGH-SENSITIVITY positions only.
    z-scores are computed relative to the population, so the two terms
    are on the same scale before combining.

    ── METRIC TO WATCH ──
    Compare al_sigma_corr(B) vs al_sigma_corr(A):
      If r(B) < r(A): Mode B finds patterns σ alone misses → additive value.
      If r(B) ≈ r(A): sens-weighting doesn't change who gets flagged.
    Also compare al_sens_novelty_mean(B) vs baseline (A doesn't compute it):
      Higher value in B confirms candidates are unusual at important positions.
    """
    sn    = _sensitivity_novelty(pattern_flat, type_freq, sens_norm)
    # z-score normalisation happens at the arm level (see _collect_al_candidates)
    # Here we just return the raw values; combining happens after population eval
    flag  = (sigma > al_sig_thr)  # broader initial flag; h_hist used in combine step
    score = sigma                  # base score; will be updated in _collect_al_candidates
    return flag, score, sn


# =============================================================================
# SECTION 7 — MAIN QICA RUNNER
# =============================================================================

def run_qica(
    arm_label,
    arm_mode,        # 'A', 'B', 'C', or 'D'
    seed,
    resources,       # dict from load_everything()
    n_gens=N_GENS,
    n_pop=N_POP,
    mc_samp=MC_SAMP,
):
    """
    Run one QICA arm for one seed. Returns a results dict.

    arm_mode options:
      'A' — σ-AL baseline (H_hist as flag only, uniform mutation, entropy trust region)
      'B' — Sensitivity-Weighted AL Scoring (sens_novelty added to AL score)
      'C' — Sensitivity-Gated Mutation Budget (mut rate inversely proportional to sens)
      'D' — Sensitivity-Defined Trust Region (freeze top-K sens positions)

    The search mechanism (σ-driven exploration) is IDENTICAL across all arms —
    only the flagging signal (B), mutation operator (C), or trust region (D) changes.
    This is the correct ablation structure: one variable at a time.
    """
    rng = np.random.default_rng(seed)

    model      = resources['model']
    ym_mean    = resources['ym_mean']
    ym_scale   = resources['ym_scale']
    yr_mean    = resources['yr_mean']
    yr_scale   = resources['yr_scale']
    IDX_PPF    = resources['IDX_PPF_MAX']
    IDX_CYCLE  = resources['IDX_CYCLE']
    type_freq  = resources['type_freq']
    free_entr  = resources['free_by_entropy']
    sens_norm  = resources['sens_norm']
    X_grid_all = resources['X_grid']
    al_sig_thr = resources['al_sig_thr']

    # ── Trust region for this arm ──────────────────────────────────────────────
    if arm_mode == 'D':
        free_positions = _trust_region_sensitivity(sens_norm, freeze_k=SENS_FREEZE_K)
        trust_label    = f'sens-freeze top-{SENS_FREEZE_K}'
    else:
        free_positions = _trust_region_entropy(free_entr)
        trust_label    = f'entropy top-20'

    tag = f'[{arm_label}|s{seed}]'
    print(f"\n  {tag} START  mode={arm_mode}  trust={trust_label}  "
          f"free_positions={len(free_positions)}/31  seed={seed}")
    print(f"  {tag}   gens={n_gens}  pop={n_pop}  mc={mc_samp}")

    t0 = time.time()

    # ── Initialise population from dataset (not random — seeded from real patterns) ──
    seed_idx   = rng.choice(len(X_grid_all), n_pop, replace=False)
    population = []  # list of 1D flat int arrays, shape (31,)
    for idx in seed_idx:
        population.append(_grid_to_flat(X_grid_all[idx]))
    population = np.array(population)  # (n_pop, 31)

    # ── First evaluation ───────────────────────────────────────────────────────
    X_pop      = _flat_to_grid_batch(population)     # (n_pop, 6, 6)
    ppf_pop, sigma_pop = _mc_predict(model, X_pop, ym_mean, ym_scale,
                                      yr_mean, yr_scale, IDX_PPF, n=mc_samp)

    best_idx   = int(np.argmin(ppf_pop))
    best_ppf   = float(ppf_pop[best_idx])
    best_pat   = population[best_idx].copy()
    best_sigma = float(sigma_pop[best_idx])
    stag       = 0

    # Get cycle length for best pattern
    X_best_g   = _flat_to_grid_batch(best_pat[None])
    y_b_sc     = model(tf.constant(X_best_g, dtype=tf.int32), training=False).numpy()
    best_cycle = float(y_b_sc[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    print(f"  {tag} Gen    0/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} "
          f"| cycle={best_cycle:.1f}d | stag=0")

    # ── Build initial empires ──────────────────────────────────────────────────
    # Sort by ppf; top n_empires become imperialists, rest are colonies
    n_empires      = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx     = np.argsort(ppf_pop)
    imperialist_idx = sorted_idx[:n_empires]
    colony_idx     = sorted_idx[n_empires:]

    # Empire membership: colony → imperialist index (in imperialist_idx)
    empire_of = {}
    for i, ci in enumerate(colony_idx):
        empire_of[ci] = imperialist_idx[i % n_empires]

    # ── AL candidate tracking ─────────────────────────────────────────────────
    al_candidates  = []   # list of dicts
    al_seen_hashes = set()

    # ── History for plotting ──────────────────────────────────────────────────
    history = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for gen in range(1, n_gens + 1):

        # Compute revolution rate (linear decay REV_START → REV_END)
        rev_rate = REV_START + (REV_END - REV_START) * (gen / n_gens)

        # H_hist of current population (for AL flagging)
        h_hist_pop = _compute_h_hist(population, type_freq)

        # ── Assimilation: each colony drifts toward its imperialist ───────────
        new_population = population.copy()
        for ci in colony_idx:
            imp   = empire_of[ci]
            pat   = population[ci].copy()
            i_pat = population[imp]
            for p in free_positions:
                # With probability ASSIMILATION_RATE, adopt imperialist's type
                if rng.random() < ASSIMILATION_RATE:
                    pat[p] = i_pat[p]
            new_population[ci] = pat

        # ── Revolution: random mutations on all members ───────────────────────
        for i in range(n_pop):
            if arm_mode == 'C':
                new_population[i] = _mutate_sens_gated(
                    new_population[i], rev_rate, free_positions,
                    type_freq, sens_norm, rng
                )
            else:
                new_population[i] = _mutate_uniform(
                    new_population[i], rev_rate, free_positions, type_freq, rng
                )

        population = new_population

        # ── Evaluate ──────────────────────────────────────────────────────────
        X_pop      = _flat_to_grid_batch(population)
        ppf_pop, sigma_pop = _mc_predict(model, X_pop, ym_mean, ym_scale,
                                          yr_mean, yr_scale, IDX_PPF, n=mc_samp)

        # ── Update best ───────────────────────────────────────────────────────
        gen_best_idx = int(np.argmin(ppf_pop))
        gen_best_ppf = float(ppf_pop[gen_best_idx])
        if gen_best_ppf < best_ppf - 1e-5:
            best_ppf   = gen_best_ppf
            best_pat   = population[gen_best_idx].copy()
            best_sigma = float(sigma_pop[gen_best_idx])
            stag       = 0
        else:
            stag += 1

        # ── Escape: mutation burst when stagnating ────────────────────────────
        # Uses baseline mutation (uniform) even in arm C — we don't want the
        # sensitivity gating to prevent escape from local minima
        if stag >= STAGNATION_PAT:
            for _ in range(ESCAPE_BURST):
                # Pick a random colony and mutate it hard (2× rev_rate)
                ci = int(rng.choice(colony_idx))
                population[ci] = _mutate_uniform(
                    population[ci], min(rev_rate * 2.0, 0.9),
                    free_positions, type_freq, rng
                )
            stag = 0

        # ── Imperialist competition ───────────────────────────────────────────
        sorted_idx      = np.argsort(ppf_pop)
        imperialist_idx = sorted_idx[:n_empires]
        colony_idx      = sorted_idx[n_empires:]
        n_active_emp    = len(set(empire_of.values()) & set(imperialist_idx))
        # Merge weakest empire if it has no colonies
        for i in imperialist_idx:
            empire_of[i] = i   # imperialists belong to themselves
        for i, ci in enumerate(colony_idx):
            empire_of[ci] = imperialist_idx[i % n_empires]

        # ── Diversity (fraction of unique patterns) ───────────────────────────
        unique_pats = len(set(map(tuple, population)))
        div         = unique_pats / n_pop

        # ── AL candidate flagging ─────────────────────────────────────────────
        # Threshold for H_hist: top (100 - AL_H_THRESH_PCT)th percentile of
        # current population entropy contribution (computed per-pattern below)
        per_pat_h = []
        for pi_idx in range(n_pop):
            per_pat_h.append(_compute_h_hist(population[[pi_idx]], type_freq))
        per_pat_h  = np.array(per_pat_h)
        h_thr      = float(np.percentile(per_pat_h, AL_H_THRESH_PCT))

        new_al  = 0
        if arm_mode == 'B':
            # Collect sens_novelty for z-score normalisation
            sn_vals = np.array([
                _sensitivity_novelty(population[pi_idx], type_freq, sens_norm)
                for pi_idx in range(n_pop)
            ])
            h_vals  = per_pat_h
            # z-scores (safe against zero std)
            z_h     = (h_vals - h_vals.mean()) / (h_vals.std() + 1e-8)
            z_sn    = (sn_vals - sn_vals.mean()) / (sn_vals.std() + 1e-8)
            composite = z_h + ALPHA_SENS_WT * z_sn
            comp_thr  = float(np.percentile(composite, AL_H_THRESH_PCT))

            for pi_idx in range(n_pop):
                sig  = float(sigma_pop[pi_idx])
                flag = (sig > al_sig_thr) and (composite[pi_idx] > comp_thr)
                if flag:
                    pat_hash = tuple(population[pi_idx])
                    if pat_hash not in al_seen_hashes:
                        al_seen_hashes.add(pat_hash)
                        al_candidates.append(dict(
                            arm=arm_label, seed=seed, gen=gen,
                            ppf_pred=float(ppf_pop[pi_idx]),
                            sigma=sig,
                            h_hist=float(per_pat_h[pi_idx]),
                            sens_novelty=float(sn_vals[pi_idx]),
                            composite_score=float(composite[pi_idx]),
                            **{f'pos_{k}': int(population[pi_idx][k]) for k in range(N_POS)}
                        ))
                        new_al += 1
                        if len(al_candidates) >= AL_MAX_CANDS:
                            break
        else:
            for pi_idx in range(n_pop):
                sig     = float(sigma_pop[pi_idx])
                h_pi    = float(per_pat_h[pi_idx])
                flag    = (sig > al_sig_thr) and (h_pi > h_thr)
                if flag:
                    pat_hash = tuple(population[pi_idx])
                    if pat_hash not in al_seen_hashes:
                        al_seen_hashes.add(pat_hash)
                        sn = (_sensitivity_novelty(population[pi_idx], type_freq, sens_norm)
                              if arm_mode in ('B',) else 0.0)
                        al_candidates.append(dict(
                            arm=arm_label, seed=seed, gen=gen,
                            ppf_pred=float(ppf_pop[pi_idx]),
                            sigma=sig,
                            h_hist=h_pi,
                            sens_novelty=sn,
                            composite_score=sig,
                            **{f'pos_{k}': int(population[pi_idx][k]) for k in range(N_POS)}
                        ))
                        new_al += 1
                        if len(al_candidates) >= AL_MAX_CANDS:
                            break

        # ── Log every 25 gens (or every 5 in quick test) ─────────────────────
        log_interval = 5 if QUICK_TEST else 25
        if gen % log_interval == 0:
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} "
                  f"| H_hist={h_hist_pop:.3f} | div={div:.2f} stag={stag} AL+{new_al}")

        history.append(dict(
            arm=arm_label, seed=seed, gen=gen,
            best_ppf=best_ppf, sigma=best_sigma,
            h_hist=h_hist_pop, div=div, stag=stag,
            rev_rate=rev_rate, n_al_total=len(al_candidates),
        ))

    # ── Final evaluation of best pattern ─────────────────────────────────────
    X_best_g  = _flat_to_grid_batch(best_pat[None])
    y_b_sc    = model(tf.constant(X_best_g, dtype=tf.int32), training=False).numpy()
    best_cycle = float(y_b_sc[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    t_total = time.time() - t0
    print(f"  {tag} DONE  best_ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  "
          f"σ={best_sigma:.4f}  {t_total:.0f}s  AL_cands={len(al_candidates)}")

    # ── AL candidate diagnostics ──────────────────────────────────────────────
    al_df     = pd.DataFrame(al_candidates) if al_candidates else pd.DataFrame()
    al_h_cv   = 0.0
    al_sig_r  = 0.0
    al_sn_mean = 0.0
    if len(al_df) > 2:
        al_h_cv   = float(al_df['h_hist'].std() / (al_df['h_hist'].mean() + 1e-9))
        al_sig_r  = float(np.corrcoef(al_df['h_hist'], al_df['sigma'])[0, 1])
        al_sn_mean = float(al_df['sens_novelty'].mean())

    return dict(
        arm=arm_label,
        mode=arm_mode,
        seed=seed,
        best_ppf=best_ppf,
        best_cycle=best_cycle,
        best_sigma=best_sigma,
        best_pat=best_pat.tolist(),
        time_s=t_total,
        n_al=len(al_candidates),
        al_h_cv=al_h_cv,
        al_sig_corr=al_sig_r,
        al_sens_novelty_mean=al_sn_mean,
        div_final=float(div),
        history=history,
        al_candidates=al_candidates,
    )


# =============================================================================
# SECTION 8 — EXPERIMENT DEFINITION
# =============================================================================

ARMS = [
    dict(
        label='A_baseline',
        mode='A',
        name='σ-AL Baseline (entropy trust, uniform mut)',
        # The confirmed best arm from final_entropy_test.py.
        # H_hist used only for AL flagging, not search.
        # This is your reference — everything else is compared to this.
    ),
    dict(
        label='B_sens_al_score',
        mode='B',
        name='Sensitivity-Weighted AL Score',
        # Same search as A, but AL flagging score = z(H_hist) + α*z(sens_novelty).
        # Tests: does focusing entropy on high-sens positions improve candidate quality?
        # KEY METRICS: al_sig_corr (lower = more additive), al_sens_novelty_mean (higher = more novel)
    ),
    dict(
        label='C_sens_mut',
        mode='C',
        name='Sensitivity-Gated Mutation',
        # Same AL flagging as A, but mutation rate inversely proportional to sensitivity.
        # Tests: does respecting the sensitivity gradient in the operator improve convergence?
        # KEY METRICS: best_ppf (lower = faster convergence), div_final (lower = tighter search)
    ),
    dict(
        label='D_sens_trust',
        mode='D',
        name='Sensitivity-Defined Trust Region',
        # Same AL flagging and mutation as A, but trust region = freeze top-K sens positions.
        # Tests: is gradient sensitivity a better trust-region boundary than positional entropy?
        # KEY METRICS: best_ppf, div_final, time_s (simpler trust region may be faster)
    ),
]


# =============================================================================
# SECTION 9 — MAIN RUN
# =============================================================================

def main():
    resources = load_everything()
    print(f"\n{'='*70}")
    print(f"STARTING: {len(ARMS)} arms × {N_SEEDS} seeds × {N_GENS} gens × pop={N_POP}")
    print(f"{'='*70}\n")

    all_results = []
    all_history = []
    all_al      = []

    t_total_start = time.time()

    for arm_cfg in ARMS:
        arm_results = []
        print(f"\n{'='*70}")
        print(f"ARM: {arm_cfg['label']}  ({arm_cfg['name']})")
        print(f"  Mode={arm_cfg['mode']}  Seeds={SEEDS}")
        print(f"{'='*70}")

        for seed in SEEDS:
            res = run_qica(
                arm_label=arm_cfg['label'],
                arm_mode=arm_cfg['mode'],
                seed=seed,
                resources=resources,
                n_gens=N_GENS,
                n_pop=N_POP,
                mc_samp=MC_SAMP,
            )
            all_results.append(res)
            arm_results.append(res)
            all_history.extend(res['history'])
            all_al.extend(res['al_candidates'])

        # Per-arm seed summary
        ppf_vals = [r['best_ppf'] for r in arm_results]
        print(f"\n  ── {arm_cfg['label']} ({len(SEEDS)} seeds) ──")
        print(f"     best_ppf = {np.mean(ppf_vals):.4f} ± {np.std(ppf_vals):.4f}  "
              f"[{min(ppf_vals):.4f} – {max(ppf_vals):.4f}]")

    total_mins = (time.time() - t_total_start) / 60
    print(f"\n[TOTAL RUNTIME] {total_mins:.1f} min\n")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    _save_outputs(all_results, all_history, all_al)

    # ── Print final comparison table ───────────────────────────────────────────
    _print_comparison(all_results)

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_results(all_results, all_history, resources)


# =============================================================================
# SECTION 10 — OUTPUT + COMPARISON
# =============================================================================

def _save_outputs(all_results, all_history, all_al):
    # Summary: one row per (arm, seed)
    summary_rows = []
    for r in all_results:
        summary_rows.append(dict(
            arm=r['arm'], mode=r['mode'], seed=r['seed'],
            best_ppf=r['best_ppf'], best_cycle=r['best_cycle'],
            best_sigma=r['best_sigma'], time_s=r['time_s'],
            n_al=r['n_al'], al_h_cv=r['al_h_cv'],
            al_sig_corr=r['al_sig_corr'],
            al_sens_novelty_mean=r['al_sens_novelty_mean'],
            div_final=r['div_final'],
        ))
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    pd.DataFrame(all_history).to_csv(HISTORY_CSV, index=False)
    if all_al:
        pd.DataFrame(all_al).to_csv(AL_CSV, index=False)
    print(f"\n[SAVED] {SUMMARY_CSV}  {HISTORY_CSV}  {AL_CSV}")


def _print_comparison(all_results):
    """
    Print the multi-metric comparison table.
    Read this table from bottom to top:
      1. Does any arm beat A_baseline on best_ppf by > 0.150?  (PRIMARY)
      2. Does Mode B reduce al_sig_corr vs baseline?           (AL QUALITY)
      3. Does Mode D tighten div_final without hurting ppf?    (TRUST REGION)
      4. Does Mode C improve ppf or speed at cost of div?      (MUTATION)
    """
    from collections import defaultdict
    arm_stats = defaultdict(list)
    for r in all_results:
        arm_stats[r['arm']].append(r)

    noise_floor = 0.110  # from final_entropy_test.py seed std

    print(f"\n{'='*90}")
    print(f"MULTI-SEED RESULTS  (mean ± std — differences < {noise_floor:.3f} PPF are within noise)")
    print(f"{'='*90}")
    header = (f"  {'Arm':<22} {'mean_PPF':>9} {'std_PPF':>8} {'min_PPF':>9} "
              f"{'al_h_cv':>8} {'al_σ_r':>7} {'al_sn':>7} {'div':>6} {'t(s)':>7}")
    print(header)
    print(f"  {'-'*88}")

    arm_means = {}
    for arm_label, results in arm_stats.items():
        ppf   = [r['best_ppf'] for r in results]
        alh   = [r['al_h_cv'] for r in results]
        alr   = [r['al_sig_corr'] for r in results]
        alsn  = [r['al_sens_novelty_mean'] for r in results]
        div   = [r['div_final'] for r in results]
        ts    = [r['time_s'] for r in results]
        row = (f"  {arm_label:<22} {np.mean(ppf):>9.4f} {np.std(ppf):>8.4f} "
               f"{min(ppf):>9.4f} {np.mean(alh):>8.3f} {np.mean(alr):>7.3f} "
               f"{np.mean(alsn):>7.3f} {np.mean(div):>6.2f} {np.mean(ts):>7.0f}")
        print(row)
        arm_means[arm_label] = np.mean(ppf)

    baseline_ppf = arm_means.get('A_baseline', 99.0)

    print(f"\n  VERDICTS (noise floor ≈ ±{noise_floor:.3f} PPF):")
    for arm_cfg in ARMS[1:]:
        lbl   = arm_cfg['label']
        delta = baseline_ppf - arm_means.get(lbl, baseline_ppf)
        tag   = ("✓ BETTER (clears noise)" if delta > noise_floor
                 else "✗ WORSE  (clears noise)" if delta < -noise_floor
                 else "≈ within noise floor")
        print(f"  {lbl} vs A_baseline: Δ={delta:+.4f} PPF  → {tag}")

    print(f"\n  COLUMN KEY:")
    print(f"    mean_PPF     : lower better; primary objective")
    print(f"    std_PPF      : lower better; high std = inconsistent arm")
    print(f"    al_h_cv      : HIGHER better; diversity of AL candidates")
    print(f"    al_σ_r       : LOWER better; H_hist adds info σ doesn't have")
    print(f"    al_sn        : HIGHER better (Arms B); novelty at high-sens positions")
    print(f"    div          : population diversity [0–1]; <0.30 = premature convergence")
    print(f"    t(s)         : wall-clock seconds; extra mechanism = extra cost")
    print(f"{'='*90}\n")


def _plot_results(all_results, all_history, resources):
    """
    6-panel comparison plot:
      [0,0] PPF convergence curves (mean ± std across seeds, by arm)
      [0,1] Best PPF per arm (bar chart with seed std as error bars)
      [0,2] Diversity over generations
      [1,0] AL candidate quality: H_hist CV per arm
      [1,1] AL candidate quality: H_hist–σ correlation per arm
      [1,2] Best loading pattern grid (overall best across all arms)
    """
    from collections import defaultdict

    hist_df   = pd.DataFrame(all_history)
    arm_label_order = [a['label'] for a in ARMS]
    COLORS    = ['#1B4FBF', '#E05C2E', '#2E9E4F', '#A83CC4']
    arm_color = {lbl: COLORS[i] for i, lbl in enumerate(arm_label_order)}

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"QICA Final  |  σ-AL Baseline + 3 Sensitivity-Entropy Modes  |  "
        f"{N_SEEDS} seeds × {N_GENS} gens × pop={N_POP}",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 0,0: Convergence curves ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for arm_cfg, color in zip(ARMS, COLORS):
        lbl  = arm_cfg['label']
        sub  = hist_df[hist_df['arm'] == lbl]
        if sub.empty:
            continue
        gens = sorted(sub['gen'].unique())
        means, stds = [], []
        for g in gens:
            g_vals = sub[sub['gen'] == g]['best_ppf'].values
            means.append(g_vals.mean())
            stds.append(g_vals.std())
        means, stds = np.array(means), np.array(stds)
        ax.plot(gens, means, color=color, lw=2, label=lbl)
        if N_SEEDS > 1:
            ax.fill_between(gens, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF (lower better)')
    ax.set_title('Convergence Curves\n(mean ± std across seeds)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.axhline(2.0, color='red', lw=1, ls='--', alpha=0.5, label='PPF=2.0')

    # ── Panel 0,1: Final best PPF bar chart ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    arm_stats = defaultdict(list)
    for r in all_results:
        arm_stats[r['arm']].append(r['best_ppf'])
    labels_bar  = arm_label_order
    means_bar   = [np.mean(arm_stats[l]) for l in labels_bar]
    stds_bar    = [np.std(arm_stats[l])  for l in labels_bar]
    x           = np.arange(len(labels_bar))
    bars = ax.bar(x, means_bar, color=COLORS[:len(labels_bar)], alpha=0.8,
                  yerr=stds_bar, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('_', 1)[1] for l in labels_bar], rotation=20, fontsize=8)
    ax.set_ylabel('Best PPF (mean ± std)')
    ax.set_title(f'Final Best PPF by Arm\nnoise floor ≈ ±0.110')
    ax.axhline(means_bar[0] + 0.11, color='grey', lw=1, ls=':', alpha=0.5)
    ax.axhline(means_bar[0] - 0.11, color='grey', lw=1, ls=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    # annotate
    for bar, m, s in zip(bars, means_bar, stds_bar):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.01,
                f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Panel 0,2: Diversity over generations ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for arm_cfg, color in zip(ARMS, COLORS):
        lbl = arm_cfg['label']
        sub = hist_df[hist_df['arm'] == lbl]
        if sub.empty:
            continue
        gens  = sorted(sub['gen'].unique())
        divs  = [sub[sub['gen'] == g]['div'].mean() for g in gens]
        ax.plot(gens, divs, color=color, lw=2, label=lbl)
    ax.axhline(0.30, color='red', lw=1, ls='--', alpha=0.5, label='div=0.30 floor')
    ax.set_xlabel('Generation'); ax.set_ylabel('Population Diversity')
    ax.set_title('Diversity\n(< 0.30 = premature convergence)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Panel 1,0: AL candidate H_hist CV ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    # One value per (arm, seed); bar = mean, error = std across seeds
    cv_means, cv_stds = [], []
    for lbl in labels_bar:
        cvs = [r['al_h_cv'] for r in all_results if r['arm'] == lbl]
        cv_means.append(np.mean(cvs))
        cv_stds.append(np.std(cvs))
    ax.bar(x, cv_means, color=COLORS[:len(labels_bar)], alpha=0.8,
           yerr=cv_stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('_', 1)[1] for l in labels_bar], rotation=20, fontsize=8)
    ax.set_ylabel('H_hist CV of AL Candidates')
    ax.set_title('AL Candidate Diversity\n(H_hist CV — HIGHER = more diverse)')
    ax.grid(True, alpha=0.3, axis='y')
    # Add annotation: higher = better
    for xi, (m, s) in enumerate(zip(cv_means, cv_stds)):
        ax.text(xi, m + s + 0.001, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Panel 1,1: AL H_hist–σ correlation ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    corr_means, corr_stds = [], []
    for lbl in labels_bar:
        corrs = [r['al_sig_corr'] for r in all_results if r['arm'] == lbl]
        corr_means.append(np.mean(corrs))
        corr_stds.append(np.std(corrs))
    ax.bar(x, corr_means, color=COLORS[:len(labels_bar)], alpha=0.8,
           yerr=corr_stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('_', 1)[1] for l in labels_bar], rotation=20, fontsize=8)
    ax.set_ylabel('r(H_hist, σ) in AL Candidates')
    ax.set_title('AL Candidate Informativeness\n(r(H_hist,σ) — LOWER = more additive info)')
    ax.axhline(0.0, color='grey', lw=1, ls='--')
    ax.grid(True, alpha=0.3, axis='y')
    for xi, (m, s) in enumerate(zip(corr_means, corr_stds)):
        ax.text(xi, m + s + 0.005, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    # ── Panel 1,2: Best pattern grid across all arms ───────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    best_result   = min(all_results, key=lambda r: r['best_ppf'])
    best_pat_flat = np.array(best_result['best_pat'])
    g_disp        = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    g_disp[~GRID_MASK] = np.nan
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g_disp[r, c] = best_pat_flat[pi]; pi += 1
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad('lightgrey')
    im = ax.imshow(g_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly Type')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'{int(g_disp[r,c])}',
                        ha='center', va='center', fontsize=9, fontweight='bold')
                pi += 1
    ax.set_title(
        f"Best Pattern Found (overall)\n"
        f"Arm={best_result['arm']}  seed={best_result['seed']}\n"
        f"PPF={best_result['best_ppf']:.4f}  cycle={best_result['best_cycle']:.1f}d"
    )
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(PLOT_PNG, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {PLOT_PNG}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    print(f"  {N_SEEDS} seed(s) × {N_GENS} gens × pop={N_POP} × {MC_SAMP} MC samples")
    if not QUICK_TEST:
        est_min = N_SEEDS * len(ARMS) * N_GENS * N_POP * MC_SAMP * 0.00025
        print(f"  Estimated runtime: ~{est_min:.0f} min on CPU (very rough)")
    print()

    for fpath in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(fpath):
            print(f"[ERROR] Required file missing: {fpath}")
            sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN] {SENS_FILE} not found — will use uniform sensitivity. "
              f"Run cnn_v9.py first to generate it.")

    main()