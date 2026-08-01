"""
=============================================================================
qica_ablation_v10.py — Quantum representation + configurable trust region +
                        warm-start seeding + composite cycle-length fitness,
                        isolated A/B/C/D/F + combined E
=============================================================================
Your qica_v9-final.py kept the "AL scoring + entropy gate" ideas from the
qica_ab_final.py A/B test but quietly dropped FOUR things earlier versions had:
  1. The actual QUANTUM-INSPIRED representation (v9 just mutates plain
     integer arrays — there's no superposition/collapse left at all).
  2. A configurable trust region (v9 hardcodes top-20-by-entropy; there was
     no way to ask "what if the whole 31-position space were free?").
  3. CNN-ranked warm-start seeding (v9 initializes purely from random
     dataset samples).
  4. v5's COMPOSITE FITNESS. v9-final selects/ranks patterns by raw
     min(PPF) alone. That's exactly why its "best" pattern (~1.77 PPF) sits
     below the CNN's own training-distribution floor — pure PPF-minimization
     has no incentive to stay inside the region the CNN was actually
     trained on, and no term rewarding cycle length at all, so the search
     is free to wander toward low-PPF/nonsense-cycle corners the CNN
     extrapolates badly on.

This file puts all four back in as INDEPENDENT, TOGGLEABLE features, runs
each in isolation against the v9-final baseline (multi-seed, so the
comparison isn't noise), then runs ONE combined arm using only whichever
toggles actually beat baseline outside the noise floor.

ARMS:
  A_baseline      : exactly v9-final's behaviour (concrete population,
                    top-20-by-entropy trust region, random dataset init,
                    selection = min(PPF))
  B_quantum       : + quantum representation (q_state per country, blended
                    assimilation, Dirichlet-noise revolution, temperature-
                    annealed collapse to concrete patterns for CNN scoring)
  C_full_trust    : baseline but TRUST_FREE_FRAC=None -> all 31 positions free
  D_warmstart     : baseline but initial population seeded from CNN-ranked
                    low-PPF training patterns instead of pure random sampling
  F_cycle_fitness : baseline but selection/ranking uses v5's composite
                    fitness instead of raw min(PPF) — see below
  E_combined      : baseline + whichever of B/C/D/F beat A by > noise_floor,
                    with SHAP traceability (only run once, here, to keep the
                    earlier arms cheap)

HOW THE QUANTUM REPRESENTATION WORKS:
  Each population member has q_state: (31, 9) — a probability distribution
  over assembly types at every position, not a committed integer.
  - Assimilation: for each free position, with probability ASSIMILATION_RATE,
    blend that position's distribution toward the imperialist's:
      q_state[p] = (1-beta)*q_state[p] + beta*imperialist.q_state[p]
    This mixes PROBABILITY MASS, not committed values — a country can end
    up "50/50 between type 3 and type 7" rather than jumping straight to
    one or the other.
  - Revolution: for each free position, with probability rev_rate, resample
    that position's entire distribution from a Dirichlet(alpha=temperature)
    prior — annealed from broad/exploratory (high temp, early gens) to
    peaked/exploitative (low temp, late gens).
  - Collapse: every generation, each q_state is turned into ONE concrete
    pattern via temperature-scaled categorical sampling (softmax(log(q)/T)),
    which is what actually gets fed to the CNN for scoring. Fixed (non-free)
    positions are pinned as one-hot at their modal training type and always
    collapse deterministically to that type.

HOW THE COMPOSITE FITNESS TOGGLE (F / use_cycle_fitness) WORKS:
  When OFF (v9-final's original behaviour): every "best pattern", empire
  ranking (who is imperialist vs colony), and stagnation check is driven
  purely by minimizing MC-dropout-mean PPF. Nothing rewards cycle length,
  and nothing explicitly penalizes wandering out of the region the CNN
  was actually trained on beyond the raw PPF value itself.

  When ON: every one of those same decision points instead uses v5's
  composite fitness, computed fresh each generation for the whole
  population:
      fitness = cycle_mean
                - W_PPF_SOFT    * ppf_mean                (soft PPF gradient)
                - W_PPF_PENALTY * max(0, ppf_mean - PPF_LIMIT)  (hard safety cutoff)
                - W_UNCERTAINTY * ppf_std                  (penalize CNN's own
                                                             MC-dropout uncertainty
                                                             -> discourages OOD wandering)
  cycle_mean comes from one extra deterministic (training=False) forward
  pass per generation — no MC dropout needed for it since cycle length
  isn't the uncertainty-sensitive term here.

  This is wired into the SAME selection code path as the other toggles:
  a single `selection_key` array (higher = better) drives best-pattern
  tracking, empire imperialist/colony sorting, and stagnation detection.
  When use_cycle_fitness=False, selection_key = -ppf_mean (so "lower PPF
  is better" becomes "higher selection_key is better", same convention).
  When True, selection_key = fitness. Quantum/trust-region/warm-start all
  compose freely with this — e.g. B+F runs quantum representation with
  fitness-driven assimilation targets.

RUN COST: this is a COMPARISON run, deliberately lighter than production
qica_v9-final.py settings (5 comparison arms x 3 seeds x 150 gens x pop 60
x MC 20, then 1 combined arm at heavier settings). Tune COMPARISON_* /
FINAL_* at the top. Set QUICK_TEST=True first to sanity-check in minutes.
=============================================================================
"""

import os, sys, json, time, warnings
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

print(f"TensorFlow {tf.__version__}")
print("qica_ablation_v10.py — Quantum + Trust + WarmStart + CycleFitness, A/B/C/D/F/E\n")


# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

QUICK_TEST = False   # True = 2 seeds, 30 gens, pop 20 (~few min per arm sanity check)

COMPARISON_SEEDS = [42, 137, 271]      if not QUICK_TEST else [42, 137]
COMPARISON_GENS  = 150                  if not QUICK_TEST else 30
COMPARISON_POP   = 60                   if not QUICK_TEST else 20
COMPARISON_MC    = 20                   if not QUICK_TEST else 10

# Final combined-arm settings can be heavier since it only runs once
FINAL_SEEDS = [42, 137, 271, 509, 1023] if not QUICK_TEST else [42]
FINAL_GENS  = 250                       if not QUICK_TEST else 30
FINAL_POP   = 80                        if not QUICK_TEST else 20
FINAL_MC    = 25                        if not QUICK_TEST else 10

NOISE_FLOOR = 0.020   # PPF delta a toggle must beat baseline by to count as a real win
                       # (used for A/B/C/D on the PPF metric; F is judged on fitness —
                       #  see the CYCLE_FITNESS_NOISE_FLOOR note in the verdict section)

# Files (unchanged from cnn_v9.py / qica_v9-final.py)
MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters — unchanged from v9-final
N_EMPIRES_INIT      = 6
ASSIMILATION_RATE    = 0.3
REV_START            = 0.35
REV_END              = 0.08
STAGNATION_PAT       = 20
ESCAPE_BURST         = 30
QUANTUM_BLEND_BETA   = 0.5   # how much of imperialist's distribution mixes in per assimilation hit
QUANTUM_TEMP_INIT    = 2.0   # collapse temperature: high = near-uniform sampling (explore)
QUANTUM_TEMP_FINAL   = 0.15  # low = near-argmax sampling (exploit)

AL_SIGMA_FRAC     = 0.50
AL_H_SENS_THRESH  = 1.30
AL_MAX_CANDS      = 50
ALPHA_SENS_WT     = 0.4

WARM_START_N = 8   # number of CNN-ranked low-PPF training patterns seeded into init population

# ── Composite fitness (v5-style), used only when arm['use_cycle_fitness'] ────
PPF_LIMIT       = 3.5    # hard safety ceiling
W_PPF_SOFT      = 6.0    # soft gradient penalty on PPF within/near the safe range
W_PPF_PENALTY   = 80.0   # hard penalty per unit PPF above PPF_LIMIT
W_UNCERTAINTY   = 40.0   # penalize MC-dropout sigma(ppf) -> discourages OOD wandering

# Stagnation-detection epsilon: fitness lives on a much bigger numeric scale
# (roughly cycle-length units, ~hundreds) than -ppf (~1.6-4.5), so each
# selection metric needs its own "did we actually improve" tolerance.
STAGNATION_EPS_PPF     = 1e-5
STAGNATION_EPS_FITNESS = 1e-2

# SHAP (only used for the final combined arm)
SHAP_BACKGROUND_N = 60
SHAP_NSAMPLES     = 200
SHAP_MAX_EXPLAIN  = 40

OUT_PREFIX = 'qica_ablation_v10'

# BEAVRS geometry (unchanged)
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


# =============================================================================
# SECTION 1 — CONVRESBLOCK (must match cnn_v9.py exactly)
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
        cfg.update({'filters': self._filters, 'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


# =============================================================================
# SECTION 2 — LOAD (extended: also builds CNN-ranked warm-start seed pool)
# =============================================================================

def load_everything():
    print(f"[LOAD] {MODEL_FILE} ...")
    model = keras.models.load_model(MODEL_FILE, compile=False)
    print(f"  input={model.input_shape}  output={model.output_shape}")

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    IDX_PPF   = cfg['IDX_PPF_MAX']
    IDX_CYCLE = cfg['IDX_CYCLE']
    ym_mean   = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
    ym_scale  = np.array(cfg['ym_scaler_scale'], dtype=np.float32)

    type_freq = np.load(FREQ_FILE).astype(np.float32)   # (31, 9)
    pos_ent   = -np.sum(type_freq * np.log(type_freq + 1e-9), axis=1)
    print(f"[TRUST] per-position entropy range: {pos_ent.min():.3f}-{pos_ent.max():.3f} nats "
          f"(max possible = {np.log(N_TYPES):.3f})")

    if os.path.exists(SENS_FILE):
        sens_df   = pd.read_csv(SENS_FILE)
        sens_norm = sens_df['sensitivity_norm'].values.astype(np.float32)
        top5      = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"[SENS]  range={sens_norm.min():.3f}-{sens_norm.max():.3f}  top5={top5}")
    else:
        print("[WARN]  sensitivity file not found — uniform sensitivity")
        sens_norm = np.full(N_POS, 0.5, dtype=np.float32)

    df     = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    lc     = [f'loading_{i}' for i in range(N_POS)]
    X_raw  = df[lc].values.astype(np.int32)
    X_grid = _flat_to_grid_batch(X_raw)
    print(f"[DATA]  {len(df)} patterns loaded")

    idx_s   = np.random.choice(len(X_grid), min(500, len(X_grid)), replace=False)
    _, sigs = _mc_predict(model, X_grid[idx_s], ym_mean, ym_scale, IDX_PPF, n=10)
    al_thr  = float(np.median(sigs))
    print(f"[CAL]   median sigma={al_thr:.4f}  ->  AL sigma_thr={al_thr:.4f}")

    # ── CNN-ranked warm-start pool (deterministic inference, no MC) ──────────
    print(f"[WARM-START] Ranking {len(X_grid)} training patterns by CNN-predicted PPF ...")
    batch_size = 256
    ppf_preds = []
    for i in range(0, len(X_grid), batch_size):
        batch = tf.constant(X_grid[i:i+batch_size], dtype=tf.int32)
        y_sc  = model(batch, training=False).numpy()
        ppf_preds.extend((y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]).tolist())
    ppf_cnn_ranked = np.array(ppf_preds, dtype=np.float32)
    warm_idx = np.argsort(ppf_cnn_ranked)[:max(50, WARM_START_N * 4)]  # small pool to sample from
    print(f"  Warm-start pool: {len(warm_idx)} lowest-CNN-predicted-PPF patterns "
          f"(range {ppf_cnn_ranked[warm_idx].min():.3f}-{ppf_cnn_ranked[warm_idx].max():.3f})\n")

    return dict(
        model=model, ym_mean=ym_mean, ym_scale=ym_scale,
        IDX_PPF=IDX_PPF, IDX_CYCLE=IDX_CYCLE,
        type_freq=type_freq, pos_ent=pos_ent, sens_norm=sens_norm,
        X_grid=X_grid, al_sig_thr=al_thr,
        warm_pool=X_grid[warm_idx],
    )


def build_free_positions(pos_ent, free_frac):
    """
    free_frac=None -> ALL 31 positions free (no trust region restriction).
    free_frac=float in (0,1] -> top round(31*free_frac) positions by entropy
    are free; rest fixed to modal training type. This replaces v9-final's
    hardcoded ":20" slice with a configurable version — 20/31 ~= 0.645 is
    what v9-final actually used, so free_frac=0.645 reproduces it exactly.
    """
    if free_frac is None:
        return set(range(N_POS)), np.zeros(N_POS, dtype=np.int32)  # fixed_types unused
    n_free = max(1, int(np.round(N_POS * free_frac)))
    rank   = np.argsort(pos_ent)[::-1]
    free_pos = set(rank[:n_free].tolist())
    return free_pos, None


# =============================================================================
# SECTION 3 — HELPERS
# =============================================================================

def _flat_to_grid_batch(X_flat):
    N  = len(X_flat)
    Xg = np.zeros((N, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                Xg[:, r, c] = X_flat[:, pi]; pi += 1
    return Xg


def _grid_to_flat(grid):
    flat = np.zeros(N_POS, dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                flat[pi] = grid[r, c]; pi += 1
    return flat


def _mc_predict(model, X_grids, ym_mean, ym_scale, idx_ppf, n):
    """MC-dropout mean/std of PPF. Unaffected by the fitness toggle — the
    composite fitness function consumes this ppf_mean/ppf_std as inputs."""
    preds = []
    Xt    = tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()
        preds.append(y_sc[:, idx_ppf] * ym_scale[idx_ppf] + ym_mean[idx_ppf])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def _predict_cycle_deterministic(model, X_grids, ym_mean, ym_scale, idx_cycle):
    """
    Deterministic (training=False, no dropout) cycle-length prediction.
    Only called when use_cycle_fitness=True — cycle length isn't the
    uncertainty-sensitive term in the composite fitness (PPF's MC sigma
    already carries the "how much do I trust this region" signal), so a
    single forward pass is enough and keeps the extra cost to ~1/mc_samp
    of an additional MC-dropout evaluation.
    """
    Xt   = tf.constant(X_grids, dtype=tf.int32)
    y_sc = model(Xt, training=False).numpy()
    return (y_sc[:, idx_cycle] * ym_scale[idx_cycle] + ym_mean[idx_cycle]).astype(np.float32)


def _composite_fitness(cycle_mean, ppf_mean, ppf_std):
    """
    v5-style composite fitness:
      fitness = cycle_mean
                - W_PPF_SOFT    * ppf_mean                      (soft gradient)
                - W_PPF_PENALTY * max(0, ppf_mean - PPF_LIMIT)   (hard cutoff)
                - W_UNCERTAINTY * ppf_std                        (OOD guard)
    All three inputs are (N,) arrays; returns (N,) fitness array.
    """
    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    return (cycle_mean
            - W_PPF_SOFT * ppf_mean
            - W_PPF_PENALTY * ppf_excess
            - W_UNCERTAINTY * ppf_std).astype(np.float32)


def _selection_key(use_cycle_fitness, ppf_mean, ppf_std, cycle_mean):
    """
    Single entry point for "what does this arm optimize toward" — higher
    is always better, regardless of toggle, so every downstream consumer
    (best-pattern tracking, empire ranking, stagnation check) can share
    one code path.
      use_cycle_fitness=False -> selection_key = -ppf_mean  (lower PPF wins)
      use_cycle_fitness=True  -> selection_key = composite fitness
    Returns (selection_key, fitness_for_logging) where fitness_for_logging
    is the composite fitness value if computed, else NaN (so summaries can
    still report it when available without recomputing).
    """
    if use_cycle_fitness:
        fit = _composite_fitness(cycle_mean, ppf_mean, ppf_std)
        return fit, fit
    else:
        return -ppf_mean, np.full_like(ppf_mean, np.nan)


def _compute_pop_h_sens(population, sens_norm, free_pos):
    """Sensitivity-weighted population entropy (v9-final's AL gate), computed
    only over free positions (fixed positions have zero diversity by design)."""
    N = len(population)
    H_per_pos = np.zeros(N_POS, dtype=np.float64)
    for p in free_pos:
        counts = np.zeros(N_TYPES)
        for t in range(N_TYPES):
            counts[t] = (population[:, p] == (t + 1)).sum()
        probs = (counts + 1e-9) / (N + N_TYPES * 1e-9)
        H_per_pos[p] = -np.sum(probs * np.log(probs))
    w = sens_norm.astype(np.float64)
    denom = sum(w[p] for p in free_pos) + 1e-9
    return float(sum(w[p] * H_per_pos[p] for p in free_pos) / denom)


def _mutate_uniform(pat, rev_rate, free_positions, type_freq, rng):
    """Concrete-mode mutation (v9-final's original behaviour)."""
    mut = pat.copy()
    for p in free_positions:
        if rng.random() < rev_rate:
            probs = type_freq[p] / type_freq[p].sum()
            mut[p] = rng.choice(np.arange(1, N_TYPES + 1), p=probs)
    return mut


def _sensitivity_novelty(pat, type_freq, sens_norm):
    novelty = 0.0
    for p in range(N_POS):
        t = int(pat[p])
        if 1 <= t <= N_TYPES:
            freq_pt = max(float(type_freq[p, t - 1]), 1e-4)
            novelty += float(sens_norm[p]) * (-np.log(freq_pt))
    return novelty


def _modal_types(type_freq):
    """Modal (most frequent) training type per position, 1-indexed."""
    return (np.argmax(type_freq, axis=1) + 1).astype(np.int32)


# ── Quantum representation primitives ────────────────────────────────────────

def quantum_init_state(n_pop, type_freq, free_pos, fixed_types, rng,
                        warm_pool=None, warm_start_n=0):
    """
    Q: (n_pop, N_POS, N_TYPES) probability distributions.
    Fixed positions are pinned one-hot to their modal training type.
    Free positions start near-uniform with slight training-frequency bias,
    EXCEPT for warm_start_n countries, whose free positions start peaked
    toward a CNN-ranked low-PPF training pattern (mirrors v5's seeding).
    """
    Q = np.tile((type_freq * 0.5 + 0.5 / N_TYPES)[None, :, :], (n_pop, 1, 1)).astype(np.float32)
    for p in range(N_POS):
        if p not in free_pos:
            Q[:, p, :] = 0.0
            Q[:, p, fixed_types[p] - 1] = 1.0
    Q /= Q.sum(axis=2, keepdims=True)

    if warm_pool is not None and warm_start_n > 0:
        n_seed = min(warm_start_n, n_pop, len(warm_pool))
        seed_patterns = warm_pool[rng.choice(len(warm_pool), n_seed, replace=False)]
        for i in range(n_seed):
            pat = _grid_to_flat(seed_patterns[i])
            for p in free_pos:
                t = int(pat[p])
                Q[i, p, :] = 0.02
                Q[i, p, t - 1] = 0.84
            Q[i] /= Q[i].sum(axis=1, keepdims=True)
    return Q


def quantum_collapse(Q, temperature, rng):
    """Temperature-scaled categorical sampling -> concrete (n_pop, N_POS) patterns."""
    n_pop = Q.shape[0]
    logits = np.log(Q + 1e-10) / max(temperature, 0.01)
    logits -= logits.max(axis=2, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)
    out = np.zeros((n_pop, N_POS), dtype=np.int32)
    for i in range(n_pop):
        for p in range(N_POS):
            out[i, p] = rng.choice(N_TYPES, p=probs[i, p]) + 1
    return out


def quantum_assimilate(Q, colony_idx, imperialist_idx, free_pos, rate, beta, rng):
    for p in free_pos:
        if rng.random() < rate:
            Q[colony_idx, p] = (1 - beta) * Q[colony_idx, p] + beta * Q[imperialist_idx, p]
    Q[colony_idx] = np.maximum(Q[colony_idx], 1e-10)
    Q[colony_idx] /= Q[colony_idx].sum(axis=1, keepdims=True)


def quantum_revolution(Q, idx, free_pos, rev_rate, temperature, rng):
    for p in free_pos:
        if rng.random() < rev_rate:
            alpha = np.ones(N_TYPES) * max(temperature, 0.05)
            Q[idx, p] = rng.dirichlet(alpha)


# =============================================================================
# SECTION 4 — UNIFIED QICA RUNNER (handles all toggle combinations)
# =============================================================================

def run_qica(arm, seed, res, n_gens, n_pop, mc_samp, run_shap_flag=False):
    """
    arm: dict with keys 'label', 'use_quantum' (bool), 'trust_free_frac'
    (float or None), 'warm_start' (bool), 'use_cycle_fitness' (bool, default
    False via .get so older arm dicts without the key still work).
    """
    rng       = np.random.default_rng(seed)
    model     = res['model']
    ym_mean   = res['ym_mean']
    ym_scale  = res['ym_scale']
    IDX_PPF   = res['IDX_PPF']
    IDX_CYCLE = res['IDX_CYCLE']
    type_freq = res['type_freq']
    pos_ent   = res['pos_ent']
    sens_norm = res['sens_norm']
    X_all     = res['X_grid']
    al_sig_thr= res['al_sig_thr']
    warm_pool = res['warm_pool']

    free_pos, _   = build_free_positions(pos_ent, arm['trust_free_frac'])
    fixed_types   = _modal_types(type_freq)
    use_quantum   = arm['use_quantum']
    use_warm      = arm['warm_start']
    use_cycle_fit = arm.get('use_cycle_fitness', False)
    sel_eps       = STAGNATION_EPS_FITNESS if use_cycle_fit else STAGNATION_EPS_PPF

    tag = f"[{arm['label']}|s{seed}]"
    print(f"\n  {tag} START  quantum={use_quantum}  free={len(free_pos)}/31  "
          f"warm_start={use_warm}  cycle_fitness={use_cycle_fit}  "
          f"gens={n_gens}  pop={n_pop}  mc={mc_samp}")
    t0 = time.time()

    # ── Initialise population ────────────────────────────────────────────────
    if use_quantum:
        Q = quantum_init_state(n_pop, type_freq, free_pos, fixed_types, rng,
                                warm_pool=warm_pool if use_warm else None,
                                warm_start_n=WARM_START_N if use_warm else 0)
        population = quantum_collapse(Q, QUANTUM_TEMP_INIT, rng)
    else:
        if use_warm:
            n_seed = min(WARM_START_N, n_pop, len(warm_pool))
            seed_idx = rng.choice(len(warm_pool), n_seed, replace=False)
            seeded = np.array([_grid_to_flat(warm_pool[i]) for i in seed_idx])
            rest_idx = rng.choice(len(X_all), n_pop - n_seed, replace=False)
            rest = np.array([_grid_to_flat(X_all[i]) for i in rest_idx])
            population = np.concatenate([seeded, rest], axis=0)
        else:
            idx0 = rng.choice(len(X_all), n_pop, replace=False)
            population = np.array([_grid_to_flat(X_all[i]) for i in idx0])
        # Pin fixed positions to modal type if trust region is active
        for p in range(N_POS):
            if p not in free_pos:
                population[:, p] = fixed_types[p]

    # ── First evaluation ──────────────────────────────────────────────────────
    Xg = _flat_to_grid_batch(population)
    ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)
    if use_cycle_fit:
        cycle_pop = _predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
    else:
        cycle_pop = np.zeros_like(ppf_pop)   # not needed for selection; filled at the end for reporting

    selection_key, fitness_pop = _selection_key(use_cycle_fit, ppf_pop, sig_pop, cycle_pop)

    best_idx      = int(np.argmax(selection_key))
    best_sel_key  = float(selection_key[best_idx])
    best_ppf      = float(ppf_pop[best_idx])
    best_pat      = population[best_idx].copy()
    best_sigma    = float(sig_pop[best_idx])
    best_fitness  = float(fitness_pop[best_idx]) if use_cycle_fit else float('nan')
    best_cycle_running = float(cycle_pop[best_idx]) if use_cycle_fit else None
    stag = 0

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    fit_str = f" fitness={best_fitness:.2f}" if use_cycle_fit else ""
    print(f"  {tag} Gen   0/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} "
          f"cycle={best_cycle:.1f}d{fit_str}")

    n_emp      = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx = np.argsort(selection_key)[::-1]   # descending = best first, for either metric
    imp_idx    = sorted_idx[:n_emp]
    col_idx    = sorted_idx[n_emp:]
    empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

    al_cands = []
    al_seen  = set()
    history  = []
    log_interval = max(5, n_gens // 10)

    for gen in range(1, n_gens + 1):
        rev_rate = REV_START + (REV_END - REV_START) * (gen / n_gens)
        temp = QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** (gen / n_gens)

        if use_quantum:
            for ci in col_idx:
                quantum_assimilate(Q, ci, empire_of[ci], free_pos,
                                    ASSIMILATION_RATE, QUANTUM_BLEND_BETA, rng)
            for i in range(n_pop):
                quantum_revolution(Q, i, free_pos, rev_rate, temp, rng)
            population = quantum_collapse(Q, temp, rng)
        else:
            new_pop = population.copy()
            for ci in col_idx:
                imp = empire_of[ci]
                for p in free_pos:
                    if rng.random() < ASSIMILATION_RATE:
                        new_pop[ci, p] = population[imp, p]
            for i in range(n_pop):
                new_pop[i] = _mutate_uniform(new_pop[i], rev_rate, free_pos, type_freq, rng)
            population = new_pop

        Xg = _flat_to_grid_batch(population)
        ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)
        if use_cycle_fit:
            cycle_pop = _predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
        selection_key, fitness_pop = _selection_key(use_cycle_fit, ppf_pop, sig_pop, cycle_pop)

        gi = int(np.argmax(selection_key))
        if float(selection_key[gi]) > best_sel_key + sel_eps:
            best_sel_key = float(selection_key[gi])
            best_ppf     = float(ppf_pop[gi])
            best_pat     = population[gi].copy()
            best_sigma   = float(sig_pop[gi])
            if use_cycle_fit:
                best_fitness = float(fitness_pop[gi])
                best_cycle_running = float(cycle_pop[gi])
            stag = 0
        else:
            stag += 1

        if stag >= STAGNATION_PAT:
            for _ in range(ESCAPE_BURST):
                ci = int(rng.choice(col_idx))
                if use_quantum:
                    quantum_revolution(Q, ci, free_pos, min(rev_rate * 2.0, 0.9), min(temp * 2, 2.0), rng)
                else:
                    population[ci] = _mutate_uniform(
                        population[ci], min(rev_rate * 2.0, 0.9), free_pos, type_freq, rng)
            if use_quantum:
                population = quantum_collapse(Q, temp, rng)
                Xg = _flat_to_grid_batch(population)
                ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)
                if use_cycle_fit:
                    cycle_pop = _predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
                selection_key, fitness_pop = _selection_key(use_cycle_fit, ppf_pop, sig_pop, cycle_pop)
            stag = 0

        sorted_idx = np.argsort(selection_key)[::-1]
        imp_idx    = sorted_idx[:n_emp]
        col_idx    = sorted_idx[n_emp:]
        empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

        div = len(set(map(tuple, population))) / n_pop
        pop_H_sens = _compute_pop_h_sens(population, sens_norm, free_pos)

        if len(al_cands) < AL_MAX_CANDS and pop_H_sens > AL_H_SENS_THRESH:
            pop_sn_vals = np.array([
                _sensitivity_novelty(population[i], type_freq, sens_norm)
                for i in range(n_pop)
            ], dtype=np.float32)
            sig_z = (sig_pop - sig_pop.mean()) / (sig_pop.std() + 1e-8)
            sn_z  = (pop_sn_vals - pop_sn_vals.mean()) / (pop_sn_vals.std() + 1e-8)
            comp  = sig_z + ALPHA_SENS_WT * sn_z
            comp_thr = float(np.percentile(comp, 70))
            for i in range(n_pop):
                if float(sig_pop[i]) > al_sig_thr and float(comp[i]) > comp_thr:
                    ph = tuple(population[i])
                    if ph not in al_seen:
                        al_seen.add(ph)
                        al_cands.append(dict(
                            arm=arm['label'], seed=seed, gen=gen,
                            ppf_pred=float(ppf_pop[i]), sigma=float(sig_pop[i]),
                            h_sens_pop=pop_H_sens, sens_novelty=float(pop_sn_vals[i]),
                            composite=float(comp[i]),
                            **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}
                        ))
                        if len(al_cands) >= AL_MAX_CANDS:
                            break

        if gen % log_interval == 0:
            fit_str = f" fitness={best_fitness:.2f}" if use_cycle_fit else ""
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} "
                  f"| H_sens={pop_H_sens:.3f} div={div:.2f} stag={stag} AL={len(al_cands)}{fit_str}")

        history.append(dict(arm=arm['label'], seed=seed, gen=gen, best_ppf=best_ppf,
                             sigma=best_sigma, best_fitness=best_fitness,
                             h_sens_pop=pop_H_sens, div=div, stag=stag,
                             n_al=len(al_cands)))

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])
    # If cycle fitness was on, best_cycle_running (tracked per-gen already) should
    # match this recomputation up to floating point — recomputing here keeps the
    # non-fitness arms' end-of-run reporting code identical to before.
    if use_cycle_fit and best_cycle_running is not None:
        best_cycle = best_cycle_running

    t_s = time.time() - t0
    fit_str = f"  fitness={best_fitness:.2f}" if use_cycle_fit else ""
    print(f"  {tag} DONE  ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  sigma={best_sigma:.4f}"
          f"{fit_str}  {t_s:.0f}s  AL={len(al_cands)}")

    return dict(
        arm=arm['label'], seed=seed, best_ppf=best_ppf, best_cycle=best_cycle,
        best_sigma=best_sigma, best_fitness=best_fitness, best_pat=best_pat.tolist(),
        time_s=t_s, n_al=len(al_cands), div_final=float(div), history=history,
        al_candidates=al_cands,
    )


# =============================================================================
# SECTION 5 — SHAP (only run once, on the final combined arm)
# =============================================================================

def run_shap(all_results, res, out_prefix):
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' package not installed — skipping traceability layer.")
        print("        Install with: pip install shap --break-system-packages")
        return None

    print("\n[SHAP] Building explainer ...")
    model, ym_mean, ym_scale, IDX_PPF = res['model'], res['ym_mean'], res['ym_scale'], res['IDX_PPF']
    X_all = res['X_grid']

    def predict_fn(flat_batch):
        flat_int = np.round(np.clip(flat_batch, 1, N_TYPES)).astype(np.int32)
        Xg = _flat_to_grid_batch(flat_int)
        y_sc = model(tf.constant(Xg, dtype=tf.int32), training=False).numpy()
        return y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]

    bg_idx  = np.random.choice(len(X_all), SHAP_BACKGROUND_N, replace=False)
    bg_flat = np.array([_grid_to_flat(X_all[i]) for i in bg_idx], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fn, bg_flat)

    explain_records = [dict(source='qica_best', seed=r['seed'],
                             pattern=np.array(r['best_pat'], dtype=np.float32))
                        for r in all_results]
    all_al_flat = [c for r in all_results for c in r['al_candidates']]
    if all_al_flat:
        al_df_full = pd.DataFrame(all_al_flat).sort_values('composite', ascending=False)
        n_al_to_explain = max(0, SHAP_MAX_EXPLAIN - len(explain_records))
        for _, row in al_df_full.head(n_al_to_explain).iterrows():
            pat = np.array([row[f'pos_{k}'] for k in range(N_POS)], dtype=np.float32)
            explain_records.append(dict(source='al_candidate', seed=int(row['seed']), pattern=pat))
    explain_records = explain_records[:SHAP_MAX_EXPLAIN]

    print(f"[SHAP] Explaining {len(explain_records)} patterns ...")
    rows = []
    t0 = time.time()
    for i, rec in enumerate(explain_records):
        sv = explainer.shap_values(rec['pattern'][None, :], nsamples=SHAP_NSAMPLES, silent=True)
        sv = np.array(sv).reshape(-1)
        row = dict(source=rec['source'], seed=rec['seed'],
                   ppf_pred=float(predict_fn(rec['pattern'][None, :])[0]))
        for p in range(N_POS):
            row[f'pos_{p}_type'] = int(rec['pattern'][p])
            row[f'pos_{p}_shap'] = float(sv[p])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  [SHAP] {i+1}/{len(explain_records)}  ({time.time()-t0:.0f}s elapsed)")

    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(f'{out_prefix}_shap.csv', index=False)
    print(f"[SHAP] Saved {out_prefix}_shap.csv  ({time.time()-t0:.0f}s total)")

    sens_norm = res['sens_norm']
    shap_cols = [f'pos_{p}_shap' for p in range(N_POS)]
    mean_abs_shap = shap_df[shap_cols].abs().mean(axis=0).values
    shap_rank = np.argsort(mean_abs_shap)[::-1]
    sens_rank = np.argsort(sens_norm)[::-1]
    overlap5 = len(set(shap_rank[:5]) & set(sens_rank[:5]))
    print(f"[SHAP] Mean |SHAP|-ranked top5 vs sensitivity-ranked top5 overlap: {overlap5}/5")
    return shap_df


# =============================================================================
# SECTION 6 — ABLATION ARMS + MAIN
# =============================================================================

ARMS = [
    dict(label='A_baseline',      use_quantum=False, trust_free_frac=20/31, warm_start=False, use_cycle_fitness=False),
    dict(label='B_quantum',       use_quantum=True,  trust_free_frac=20/31, warm_start=False, use_cycle_fitness=False),
    dict(label='C_full_trust',    use_quantum=False, trust_free_frac=None,  warm_start=False, use_cycle_fitness=False),
    dict(label='D_warmstart',     use_quantum=False, trust_free_frac=20/31, warm_start=True,  use_cycle_fitness=False),
    dict(label='F_cycle_fitness', use_quantum=False, trust_free_frac=20/31, warm_start=False, use_cycle_fitness=True),
]


def main():
    res = load_everything()

    print(f"{'='*70}")
    print(f"COMPARISON PHASE  |  {len(COMPARISON_SEEDS)} seeds x {COMPARISON_GENS} gens x "
          f"pop={COMPARISON_POP} x MC={COMPARISON_MC}  |  {len(ARMS)} arms")
    print(f"{'='*70}")

    all_results, all_history, all_al = [], [], []
    for arm in ARMS:
        print(f"\n{'='*70}\nARM: {arm['label']}\n{'='*70}")
        arm_results = []
        for seed in COMPARISON_SEEDS:
            r = run_qica(arm, seed, res, COMPARISON_GENS, COMPARISON_POP, COMPARISON_MC)
            all_results.append(r); arm_results.append(r)
            all_history.extend(r['history']); all_al.extend(r['al_candidates'])
        ppf_vals   = [r['best_ppf']   for r in arm_results]
        cycle_vals = [r['best_cycle'] for r in arm_results]
        print(f"\n  -- {arm['label']} ({len(COMPARISON_SEEDS)} seeds) --")
        print(f"     best_ppf   = {np.mean(ppf_vals):.4f} +/- {np.std(ppf_vals):.4f}  "
              f"[{min(ppf_vals):.4f} - {max(ppf_vals):.4f}]")
        print(f"     best_cycle = {np.mean(cycle_vals):.1f} +/- {np.std(cycle_vals):.1f} d")
        if arm.get('use_cycle_fitness', False):
            fit_vals = [r['best_fitness'] for r in arm_results]
            print(f"     best_fitness = {np.mean(fit_vals):.2f} +/- {np.std(fit_vals):.2f}")

    # ── Verdict table ─────────────────────────────────────────────────────────
    from collections import defaultdict
    arm_ppf   = defaultdict(list)
    arm_cycle = defaultdict(list)
    for r in all_results:
        arm_ppf[r['arm']].append(r['best_ppf'])
        arm_cycle[r['arm']].append(r['best_cycle'])

    a_mean_ppf   = np.mean(arm_ppf['A_baseline'])
    a_mean_cycle = np.mean(arm_cycle['A_baseline'])
    print(f"\n{'='*70}")
    print(f"VERDICT  (baseline A: ppf={a_mean_ppf:.4f}, cycle={a_mean_cycle:.1f}d, "
          f"noise floor = {NOISE_FLOOR} PPF)")
    print(f"{'='*70}")
    winners = []
    for arm in ARMS[1:]:
        lbl        = arm['label']
        mean_ppf   = np.mean(arm_ppf[lbl])
        mean_cycle = np.mean(arm_cycle[lbl])
        delta_ppf  = a_mean_ppf - mean_ppf   # positive = arm beats baseline (lower PPF)

        if arm.get('use_cycle_fitness', False):
            # F is optimizing a different objective than raw PPF, so judge it on
            # whether it found a BETTER TRADEOFF, not just lower PPF: report both
            # metrics and require it not to be worse on PPF than baseline's own
            # floor while gaining meaningfully on cycle length, OR require the
            # composite fitness itself (which already encodes the tradeoff) to
            # clear the same noise floor once rescaled — simplest robust check
            # here is: cycle length improved AND ppf stayed within noise floor
            # of baseline (i.e. F isn't just finding shorter-cycle/lower-ppf
            # corners, it's finding longer-cycle patterns at comparable safety).
            delta_cycle = mean_cycle - a_mean_cycle
            beats = (delta_cycle > 1.0) and (delta_ppf > -NOISE_FLOOR)
            print(f"  {lbl:<14} mean_ppf={mean_ppf:.4f}  mean_cycle={mean_cycle:.1f}d  "
                  f"delta_cycle={delta_cycle:+.1f}d  delta_ppf={delta_ppf:+.4f}  "
                  f"{'>>> WINS (longer cycle, no PPF regression), will combine' if beats else '(no clear win vs baseline)'}")
        else:
            beats = delta_ppf > NOISE_FLOOR
            print(f"  {lbl:<14} mean_ppf={mean_ppf:.4f}  mean_cycle={mean_cycle:.1f}d  "
                  f"delta(A-arm)={delta_ppf:+.4f}  "
                  f"{'>>> WINS, will combine' if beats else '(not distinguishable from noise)'}")
        if beats:
            winners.append(arm)

    _save(all_results, all_history, all_al, f'{OUT_PREFIX}_comparison')
    _plot_comparison(all_results, all_history)

    # ── Combined final arm ────────────────────────────────────────────────────
    combined = dict(
        label='E_combined',
        use_quantum=any(w['label'] == 'B_quantum' for w in winners),
        trust_free_frac=(None if any(w['label'] == 'C_full_trust' for w in winners) else 20/31),
        warm_start=any(w['label'] == 'D_warmstart' for w in winners),
        use_cycle_fitness=any(w['label'] == 'F_cycle_fitness' for w in winners),
    )
    print(f"\n{'='*70}")
    print(f"FINAL COMBINED ARM: quantum={combined['use_quantum']}  "
          f"trust_free_frac={combined['trust_free_frac']}  warm_start={combined['warm_start']}  "
          f"cycle_fitness={combined['use_cycle_fitness']}")
    if not winners:
        print("  (No individual toggle beat baseline outside the noise floor — "
          "this run will be IDENTICAL to A_baseline, just as a confirmation.)")
    print(f"  {len(FINAL_SEEDS)} seeds x {FINAL_GENS} gens x pop={FINAL_POP} x MC={FINAL_MC}")
    print(f"{'='*70}")

    final_results, final_history, final_al = [], [], []
    for seed in FINAL_SEEDS:
        r = run_qica(combined, seed, res, FINAL_GENS, FINAL_POP, FINAL_MC)
        final_results.append(r)
        final_history.extend(r['history']); final_al.extend(r['al_candidates'])

    final_ppf   = [r['best_ppf']   for r in final_results]
    final_cycle = [r['best_cycle'] for r in final_results]
    print(f"\n{'='*70}")
    print(f"E_combined — {len(FINAL_SEEDS)} seeds")
    print(f"  best_ppf   = {np.mean(final_ppf):.4f} +/- {np.std(final_ppf):.4f}  "
          f"[{min(final_ppf):.4f} - {max(final_ppf):.4f}]")
    print(f"  best_cycle = {np.mean(final_cycle):.1f} +/- {np.std(final_cycle):.1f} d")
    if combined['use_cycle_fitness']:
        final_fit = [r['best_fitness'] for r in final_results]
        print(f"  best_fitness = {np.mean(final_fit):.2f} +/- {np.std(final_fit):.2f}")
    print(f"  (baseline A reference: ppf={a_mean_ppf:.4f}  cycle={a_mean_cycle:.1f}d)")
    print(f"{'='*70}")

    _save(final_results, final_history, final_al, f'{OUT_PREFIX}_final')
    run_shap(final_results, res, f'{OUT_PREFIX}_final')

    best_overall = min(final_results, key=lambda r: r['best_ppf'])
    print(f"\n[BEST OVERALL by PPF, E_combined] seed={best_overall['seed']}  "
          f"ppf={best_overall['best_ppf']:.4f}  cycle={best_overall['best_cycle']:.1f}d")
    if combined['use_cycle_fitness']:
        best_by_fit = max(final_results, key=lambda r: r['best_fitness'])
        print(f"[BEST OVERALL by fitness, E_combined] seed={best_by_fit['seed']}  "
              f"ppf={best_by_fit['best_ppf']:.4f}  cycle={best_by_fit['best_cycle']:.1f}d  "
              f"fitness={best_by_fit['best_fitness']:.2f}")
        print(f"  loading pattern: {best_by_fit['best_pat']}")
    else:
        print(f"  loading pattern: {best_overall['best_pat']}")


def _save(all_results, all_history, all_al, prefix):
    rows = [dict(arm=r['arm'], seed=r['seed'], best_ppf=r['best_ppf'],
                 best_cycle=r['best_cycle'], best_sigma=r['best_sigma'],
                 best_fitness=r.get('best_fitness', float('nan')),
                 time_s=r['time_s'], n_al=r['n_al'], div_final=r['div_final'])
            for r in all_results]
    pd.DataFrame(rows).to_csv(f'{prefix}_summary.csv', index=False)
    pd.DataFrame(all_history).to_csv(f'{prefix}_history.csv', index=False)
    if all_al:
        al_df = pd.DataFrame(all_al)
        pos_cols = [f'pos_{k}' for k in range(N_POS)]
        al_df = (al_df.sort_values('composite', ascending=False)
                       .drop_duplicates(subset=pos_cols, keep='first').reset_index(drop=True))
        al_df.to_csv(f'{prefix}_al_candidates.csv', index=False)
    print(f"[SAVED] {prefix}_summary.csv  {prefix}_history.csv  {prefix}_al_candidates.csv")


def _plot_comparison(all_results, all_history):
    from collections import defaultdict
    hist_df = pd.DataFrame(all_history)
    arm_stats = defaultdict(list)
    for r in all_results:
        arm_stats[r['arm']].append(r)

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("QICA Ablation: Quantum vs FullTrust vs WarmStart vs CycleFitness vs Baseline",
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)
    colors = {'A_baseline': '#888888', 'B_quantum': '#1B4FBF',
              'C_full_trust': '#E05C2E', 'D_warmstart': '#2CA02C',
              'F_cycle_fitness': '#9467BD'}

    ax = fig.add_subplot(gs[0])
    for arm in ARMS:
        lbl = arm['label']
        sub = hist_df[hist_df['arm'] == lbl]
        if sub.empty:
            continue
        gens = sorted(sub['gen'].unique())
        mn = [sub[sub['gen'] == g]['best_ppf'].mean() for g in gens]
        ax.plot(gens, mn, color=colors.get(lbl, 'black'), lw=2, label=lbl)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF (mean across seeds)')
    ax.set_title('PPF Convergence by Arm'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1])
    labels = [a['label'] for a in ARMS]
    means  = [np.mean([r['best_ppf'] for r in arm_stats[l]]) for l in labels]
    stds   = [np.std( [r['best_ppf'] for r in arm_stats[l]]) for l in labels]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=[colors.get(l, 'black') for l in labels], alpha=0.85, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel('Final Best PPF (mean +/- std)')
    ax.set_title('Arm Comparison — PPF'); ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[2])
    cyc_means = [np.mean([r['best_cycle'] for r in arm_stats[l]]) for l in labels]
    cyc_stds  = [np.std( [r['best_cycle'] for r in arm_stats[l]]) for l in labels]
    ax.bar(x, cyc_means, yerr=cyc_stds, color=[colors.get(l, 'black') for l in labels], alpha=0.85, capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel('Final Best Cycle Length (days, mean +/- std)')
    ax.set_title('Arm Comparison — Cycle Length'); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT_PREFIX}_comparison.png', dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {OUT_PREFIX}_comparison.png")


if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL COMPARISON'}")
    for f in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN]  {SENS_FILE} not found — using uniform sensitivity")
    main()