"""
python openmc_beavrs_vver1000_FIXED_2.py --single_pattern "9,3,1,6,5,5,1,5,6,1,4,4,1,1,4,1,1,3,4,5,2,4,1,3,8,1,1,7,1,3,1" \
  --quick_check --particles 4000 --batches 60 --inactive 40
=============================================================================
qica_v11_production.py — Production QICA, all ablation-v10 winners wired in
                          as TOGGLES, with an OpenMC verification exporter.
=============================================================================
Defaults = the ablation-v10 winning config (E_combined):
    USE_QUANTUM      = True   (won: +0.073 PPF vs baseline, clears noise floor)
    TRUST_FREE_FRAC   = None   (won: full 31/31 free, +0.027 vs baseline)
    WARM_START        = True   (won: +0.025 vs baseline)
    USE_CYCLE_FITNESS = False  (lost ablation on raw PPF, but is kept as a
                                 toggle — see "TESTING WITH OPENMC" below,
                                 you asked to verify it physically rather
                                 than trust the CNN-only verdict)

TRUST_FREE_FRAC follows qica-v5-2.py's ENTROPY_FREE_FRAC logic exactly:
    None        -> all 31 positions free (no restriction)
    float (0,1] -> top round(31*frac) positions by per-position Shannon
                   entropy H_pos = -sum_t freq[p,t] log freq[p,t]) are free;
                   the rest are FIXED to their modal training type. This is
                   the same trust-region math as qica-v5-2.py's
                   analyze_trust_region() / compute_position_entropy() and
                   qica_v9-final.py's hardcoded 20/31 slice (= frac=20/31).

TOGGLE ANY OF THE FOUR AT THE TOP AND RE-RUN. Nothing else in the file
needs to change — every code path is unified through the same
_selection_key() / quantum_* / free_pos machinery from qica_ablation_v10.py.

=============================================================================
TESTING WITH OPENMC (read this before trusting any CNN-only verdict)
=============================================================================
Everything above is CNN-surrogate opinion. To ground-truth it:

1) QUICK CHECK (minutes, BOC-only, no depletion — sanity check PPF/keff):
   After this script finishes it writes best_patterns_for_openmc.csv (one
   row per seed) with a ready-to-paste --single_pattern string printed to
   stdout. Run, e.g.:

     python openmc_beavrs_vver1000_FIXED_2.py \
       --single_pattern "<printed pattern>" \
       --quick_check --particles 4000 --batches 60 --inactive 40

   Compare the printed PPF_max (BOC) against this script's best_ppf. They
   won't match exactly (CNN predicts *cycle-max* PPF, quick_check gives
   *BOC* PPF only — see qica-v5-2.py's note "BOC is cycle max in 99.3%",
   so BOC vs cycle-max should be close but not identical) — flag it if the
   gap is large (>0.3-0.4 PPF), that's a surrogate-trust problem worth
   investigating before running full depletion.

2) FULL DEPLETION (hours per pattern — gives real cycle_length + full PPF
   trajectory, the only way to actually verify cycle length):

     python openmc_beavrs_vver1000_FIXED_2.py \
       --al_candidates_csv best_patterns_for_openmc.csv \
       --chain <your_chain_file.xml> \
       --particles 4000 --batches 60 --inactive 15 \
       --out_csv qica_v11_verified.csv

   This appends real (pattern, react, ppf, cycle_length) rows you can
   compare row-by-row against this script's CNN predictions, AND append to
   ml_dataset_constrained.csv for retraining (active learning loop).

3) A/B'ING use_cycle_fitness AGAINST OPENMC SPECIFICALLY:
   The ablation only judged F_cycle_fitness by CNN-predicted PPF, which is
   an unfair test — F is optimizing for a cycle/PPF tradeoff, not raw PPF,
   so of course raw-PPF comparison made it look worse. To judge it fairly:
     a) Run this script twice: once with USE_CYCLE_FITNESS=False (default
        config below) and once with USE_CYCLE_FITNESS=True. Each produces
        its own best_patterns_for_openmc.csv — rename them
        (e.g. *_nofit.csv / *_fit.csv) so they don't overwrite each other.
     b) Full-depletion BOTH best patterns through OpenMC (step 2 above).
     c) Compare REAL cycle_length and REAL ppf_max (cycle-max, not BOC) —
        the composite-fitness pattern should show a real cycle-length gain
        with real PPF staying under PPF_LIMIT=3.5. If OpenMC confirms that
        tradeoff, use_cycle_fitness is actually a win the ablation's
        PPF-only metric couldn't see; if OpenMC shows the "longer cycle"
        prediction doesn't hold up, trust the ablation verdict.
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
import keras
from keras import layers

print(f"TensorFlow {tf.__version__}")
print("qica_v11_production.py — toggleable quantum/trust/warmstart/cycle-fitness\n")


# =============================================================================
# SECTION 0 — TOGGLES  (edit these four, nothing else needs to change)
# =============================================================================

USE_QUANTUM       = True     # quantum-inspired q_state representation (ablation winner)
TRUST_FREE_FRAC   = None     # None = all 31 free | float e.g. 20/31 = qica-v5-2-style trust region
WARM_START        = True     # seed init population from CNN-ranked low-PPF training patterns
USE_CYCLE_FITNESS = False    # False = minimize PPF only | True = v5 composite (cycle - penalties)

QUICK_TEST = False   # True = 1 seed, 30 gens, pop 20 (sanity check, minutes)

N_SEEDS = 1   if QUICK_TEST else 5
N_GENS  = 30  if QUICK_TEST else 250
N_POP   = 20  if QUICK_TEST else 80
MC_SAMP = 10  if QUICK_TEST else 25
SEEDS   = [42] if QUICK_TEST else [42, 137, 271, 509, 1023]

# Files (unchanged locations)
MODEL_FILE  = 'cnn_v10_model.keras'
CONFIG_FILE = 'cnn_v10_config.json'
FREQ_FILE   = 'train_type_freq_v10.npy'
SENS_FILE   = 'cnn_v10_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters (unchanged from v9-final / ablation-v10)
N_EMPIRES_INIT     = 6
ASSIMILATION_RATE  = 0.3
REV_START          = 0.35
REV_END            = 0.08
STAGNATION_PAT     = 20
ESCAPE_BURST       = 30
QUANTUM_BLEND_BETA = 0.5
QUANTUM_TEMP_INIT  = 2.0
QUANTUM_TEMP_FINAL = 0.15

AL_H_SENS_THRESH = 1.30
AL_MAX_CANDS     = 50
ALPHA_SENS_WT    = 0.4
WARM_START_N     = 8

# Composite fitness (v5-style), used only when USE_CYCLE_FITNESS=True
PPF_LIMIT     = 3.5
W_PPF_SOFT    = 6.0
W_PPF_PENALTY = 80.0
W_UNCERTAINTY = 40.0

STAGNATION_EPS_PPF     = 1e-5
STAGNATION_EPS_FITNESS = 1e-2

SHAP_ENABLE       = True
SHAP_BACKGROUND_N = 60
SHAP_NSAMPLES     = 200
SHAP_MAX_EXPLAIN  = 40

OUT_PREFIX  = 'qica_v11'
OPENMC_CSV  = 'best_patterns_for_openmc_T.csv'   # al_candidates_csv-compatible export

N_POS, N_TYPES = 31, 9 
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


# =============================================================================
# SECTION 1 — CONVRESBLOCK (must match cnn_v9.py exactly)
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
# SECTION 2 — LOAD
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

    type_freq = np.load(FREQ_FILE).astype(np.float32)
    pos_ent   = -np.sum(type_freq * np.log(type_freq + 1e-9), axis=1)
    print(f"[TRUST] entropy range {pos_ent.min():.3f}-{pos_ent.max():.3f} nats "
          f"(max={np.log(N_TYPES):.3f})  |  TRUST_FREE_FRAC={TRUST_FREE_FRAC}")

    if os.path.exists(SENS_FILE):
        sens_df   = pd.read_csv(SENS_FILE)
        sens_norm = sens_df['sensitivity_norm'].values.astype(np.float32)
        top5      = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"[SENS]  range={sens_norm.min():.3f}-{sens_norm.max():.3f}  top5={top5}")
    else:
        print("[WARN]  sensitivity file not found — uniform sensitivity")
        sens_norm = np.full(N_POS, 0.5, dtype=np.float32)

    df    = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    lc    = [f'loading_{i}' for i in range(N_POS)]
    X_raw = df[lc].values.astype(np.int32)
    X_grid = _flat_to_grid_batch(X_raw)
    print(f"[DATA]  {len(df)} patterns loaded")

    idx_s   = np.random.choice(len(X_grid), min(500, len(X_grid)), replace=False)
    _, sigs = _mc_predict(model, X_grid[idx_s], ym_mean, ym_scale, IDX_PPF, n=10)
    al_thr  = float(np.median(sigs))
    print(f"[CAL]   median sigma={al_thr:.4f}")

    print(f"[WARM-START] Ranking {len(X_grid)} training patterns by CNN-predicted PPF ...")
    batch_size, ppf_preds = 256, []
    for i in range(0, len(X_grid), batch_size):
        batch = tf.constant(X_grid[i:i+batch_size], dtype=tf.int32)
        y_sc  = model(batch, training=False).numpy()
        ppf_preds.extend((y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]).tolist())
    ppf_cnn_ranked = np.array(ppf_preds, dtype=np.float32)
    warm_idx = np.argsort(ppf_cnn_ranked)[:max(50, WARM_START_N * 4)]
    print(f"  Warm-start pool: {len(warm_idx)} patterns "
          f"(PPF {ppf_cnn_ranked[warm_idx].min():.3f}-{ppf_cnn_ranked[warm_idx].max():.3f})\n")

    return dict(model=model, ym_mean=ym_mean, ym_scale=ym_scale, IDX_PPF=IDX_PPF, IDX_CYCLE=IDX_CYCLE,
                type_freq=type_freq, pos_ent=pos_ent, sens_norm=sens_norm, X_grid=X_grid,
                al_sig_thr=al_thr, warm_pool=X_grid[warm_idx])


def build_free_positions(pos_ent, free_frac):
    """qica-v5-2.py analyze_trust_region() logic. None -> all free."""
    if free_frac is None:
        return set(range(N_POS))
    n_free = max(1, int(np.round(N_POS * free_frac)))
    rank   = np.argsort(pos_ent)[::-1]
    return set(rank[:n_free].tolist())


def _modal_types(type_freq):
    return (np.argmax(type_freq, axis=1) + 1).astype(np.int32)


# =============================================================================
# SECTION 3 — HELPERS
# =============================================================================

def _flat_to_grid_batch(X_flat):
    N, Xg = len(X_flat), np.zeros((len(X_flat), GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                Xg[:, r, c] = X_flat[:, pi]; pi += 1
    return Xg


def _grid_to_flat(grid):
    flat, pi = np.zeros(N_POS, dtype=np.int32), 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                flat[pi] = grid[r, c]; pi += 1
    return flat


def _mc_predict(model, X_grids, ym_mean, ym_scale, idx_ppf, n):
    preds, Xt = [], tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()
        preds.append(y_sc[:, idx_ppf] * ym_scale[idx_ppf] + ym_mean[idx_ppf])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def _predict_cycle_deterministic(model, X_grids, ym_mean, ym_scale, idx_cycle):
    Xt   = tf.constant(X_grids, dtype=tf.int32)
    y_sc = model(Xt, training=False).numpy()
    return (y_sc[:, idx_cycle] * ym_scale[idx_cycle] + ym_mean[idx_cycle]).astype(np.float32)


def _composite_fitness(cycle_mean, ppf_mean, ppf_std):
    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    return (cycle_mean - W_PPF_SOFT * ppf_mean - W_PPF_PENALTY * ppf_excess
            - W_UNCERTAINTY * ppf_std).astype(np.float32)


def _selection_key(use_cycle_fitness, ppf_mean, ppf_std, cycle_mean):
    if use_cycle_fitness:
        fit = _composite_fitness(cycle_mean, ppf_mean, ppf_std)
        return fit, fit
    return -ppf_mean, np.full_like(ppf_mean, np.nan)


def _compute_pop_h_sens(population, sens_norm, free_pos):
    N = len(population)
    H_per_pos = np.zeros(N_POS, dtype=np.float64)
    for p in free_pos:
        counts = np.array([(population[:, p] == (t + 1)).sum() for t in range(N_TYPES)])
        probs = (counts + 1e-9) / (N + N_TYPES * 1e-9)
        H_per_pos[p] = -np.sum(probs * np.log(probs))
    w = sens_norm.astype(np.float64)
    denom = sum(w[p] for p in free_pos) + 1e-9
    return float(sum(w[p] * H_per_pos[p] for p in free_pos) / denom)


def _mutate_uniform(pat, rev_rate, free_positions, type_freq, rng):
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


# ── Quantum representation primitives ────────────────────────────────────────

def quantum_init_state(n_pop, type_freq, free_pos, fixed_types, rng, warm_pool=None, warm_start_n=0):
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
    n_pop = Q.shape[0]
    logits = np.log(Q + 1e-10) / max(temperature, 0.01)
    logits -= logits.max(axis=2, keepdims=True)
    probs = np.exp(logits); probs /= probs.sum(axis=2, keepdims=True)
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
            Q[idx, p] = rng.dirichlet(np.ones(N_TYPES) * max(temperature, 0.05))


# =============================================================================
# SECTION 4 — UNIFIED QICA RUNNER
# =============================================================================

def run_qica(seed, res, n_gens, n_pop, mc_samp):
    rng       = np.random.default_rng(seed)
    model, ym_mean, ym_scale = res['model'], res['ym_mean'], res['ym_scale']
    IDX_PPF, IDX_CYCLE = res['IDX_PPF'], res['IDX_CYCLE']
    type_freq, pos_ent, sens_norm = res['type_freq'], res['pos_ent'], res['sens_norm']
    X_all, al_sig_thr, warm_pool  = res['X_grid'], res['al_sig_thr'], res['warm_pool']

    free_pos    = build_free_positions(pos_ent, TRUST_FREE_FRAC)
    fixed_types = _modal_types(type_freq)
    sel_eps     = STAGNATION_EPS_FITNESS if USE_CYCLE_FITNESS else STAGNATION_EPS_PPF

    tag = f"[s{seed}]"
    print(f"\n  {tag} START  quantum={USE_QUANTUM}  free={len(free_pos)}/31  "
          f"warm_start={WARM_START}  cycle_fitness={USE_CYCLE_FITNESS}  "
          f"gens={n_gens}  pop={n_pop}  mc={mc_samp}")
    t0 = time.time()

    if USE_QUANTUM:
        Q = quantum_init_state(n_pop, type_freq, free_pos, fixed_types, rng,
                                warm_pool=warm_pool if WARM_START else None,
                                warm_start_n=WARM_START_N if WARM_START else 0)
        population = quantum_collapse(Q, QUANTUM_TEMP_INIT, rng)
    else:
        if WARM_START:
            n_seed   = min(WARM_START_N, n_pop, len(warm_pool))
            seed_idx = rng.choice(len(warm_pool), n_seed, replace=False)
            seeded   = np.array([_grid_to_flat(warm_pool[i]) for i in seed_idx])
            rest_idx = rng.choice(len(X_all), n_pop - n_seed, replace=False)
            rest     = np.array([_grid_to_flat(X_all[i]) for i in rest_idx])
            population = np.concatenate([seeded, rest], axis=0)
        else:
            idx0 = rng.choice(len(X_all), n_pop, replace=False)
            population = np.array([_grid_to_flat(X_all[i]) for i in idx0])
        for p in range(N_POS):
            if p not in free_pos:
                population[:, p] = fixed_types[p]

    Xg = _flat_to_grid_batch(population)
    ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)
    cycle_pop = (_predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
                 if USE_CYCLE_FITNESS else np.zeros_like(ppf_pop))
    selection_key, fitness_pop = _selection_key(USE_CYCLE_FITNESS, ppf_pop, sig_pop, cycle_pop)

    best_idx     = int(np.argmax(selection_key))
    best_sel_key = float(selection_key[best_idx])
    best_ppf     = float(ppf_pop[best_idx])
    best_pat     = population[best_idx].copy()
    best_sigma   = float(sig_pop[best_idx])
    best_fitness = float(fitness_pop[best_idx]) if USE_CYCLE_FITNESS else float('nan')
    best_cycle_running = float(cycle_pop[best_idx]) if USE_CYCLE_FITNESS else None
    stag = 0

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    print(f"  {tag} Gen   0/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} cycle={best_cycle:.1f}d")

    n_emp      = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx = np.argsort(selection_key)[::-1]
    imp_idx    = sorted_idx[:n_emp]
    col_idx    = sorted_idx[n_emp:]
    empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

    al_cands, al_seen, history = [], set(), []
    log_interval = max(5, n_gens // 10)

    for gen in range(1, n_gens + 1):
        rev_rate = REV_START + (REV_END - REV_START) * (gen / n_gens)
        temp = QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** (gen / n_gens)

        if USE_QUANTUM:
            for ci in col_idx:
                quantum_assimilate(Q, ci, empire_of[ci], free_pos, ASSIMILATION_RATE, QUANTUM_BLEND_BETA, rng)
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
        if USE_CYCLE_FITNESS:
            cycle_pop = _predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
        selection_key, fitness_pop = _selection_key(USE_CYCLE_FITNESS, ppf_pop, sig_pop, cycle_pop)

        gi = int(np.argmax(selection_key))
        if float(selection_key[gi]) > best_sel_key + sel_eps:
            best_sel_key, best_ppf = float(selection_key[gi]), float(ppf_pop[gi])
            best_pat, best_sigma = population[gi].copy(), float(sig_pop[gi])
            if USE_CYCLE_FITNESS:
                best_fitness, best_cycle_running = float(fitness_pop[gi]), float(cycle_pop[gi])
            stag = 0
        else:
            stag += 1

        if stag >= STAGNATION_PAT:
            for _ in range(ESCAPE_BURST):
                ci = int(rng.choice(col_idx))
                if USE_QUANTUM:
                    quantum_revolution(Q, ci, free_pos, min(rev_rate * 2.0, 0.9), min(temp * 2, 2.0), rng)
                else:
                    population[ci] = _mutate_uniform(population[ci], min(rev_rate * 2.0, 0.9), free_pos, type_freq, rng)
            if USE_QUANTUM:
                population = quantum_collapse(Q, temp, rng)
                Xg = _flat_to_grid_batch(population)
                ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)
                if USE_CYCLE_FITNESS:
                    cycle_pop = _predict_cycle_deterministic(model, Xg, ym_mean, ym_scale, IDX_CYCLE)
                selection_key, fitness_pop = _selection_key(USE_CYCLE_FITNESS, ppf_pop, sig_pop, cycle_pop)
            stag = 0

        sorted_idx = np.argsort(selection_key)[::-1]
        imp_idx, col_idx = sorted_idx[:n_emp], sorted_idx[n_emp:]
        empire_of = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

        div = len(set(map(tuple, population))) / n_pop
        pop_H_sens = _compute_pop_h_sens(population, sens_norm, free_pos)

        if len(al_cands) < AL_MAX_CANDS and pop_H_sens > AL_H_SENS_THRESH:
            pop_sn_vals = np.array([_sensitivity_novelty(population[i], type_freq, sens_norm)
                                     for i in range(n_pop)], dtype=np.float32)
            sig_z = (sig_pop - sig_pop.mean()) / (sig_pop.std() + 1e-8)
            sn_z  = (pop_sn_vals - pop_sn_vals.mean()) / (pop_sn_vals.std() + 1e-8)
            comp  = sig_z + ALPHA_SENS_WT * sn_z
            comp_thr = float(np.percentile(comp, 70))
            for i in range(n_pop):
                if float(sig_pop[i]) > al_sig_thr and float(comp[i]) > comp_thr:
                    ph = tuple(population[i])
                    if ph not in al_seen:
                        al_seen.add(ph)
                        al_cands.append(dict(seed=seed, gen=gen, ppf_pred=float(ppf_pop[i]),
                                              sigma=float(sig_pop[i]), h_sens_pop=pop_H_sens,
                                              sens_novelty=float(pop_sn_vals[i]), composite=float(comp[i]),
                                              **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}))
                        if len(al_cands) >= AL_MAX_CANDS:
                            break

        if gen % log_interval == 0:
            fit_str = f" fitness={best_fitness:.2f}" if USE_CYCLE_FITNESS else ""
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} "
                  f"| H_sens={pop_H_sens:.3f} div={div:.2f} stag={stag} AL={len(al_cands)}{fit_str}")

        history.append(dict(seed=seed, gen=gen, best_ppf=best_ppf, sigma=best_sigma,
                             best_fitness=best_fitness, h_sens_pop=pop_H_sens, div=div,
                             stag=stag, n_al=len(al_cands)))

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])
    if USE_CYCLE_FITNESS and best_cycle_running is not None:
        best_cycle = best_cycle_running

    t_s = time.time() - t0
    fit_str = f"  fitness={best_fitness:.2f}" if USE_CYCLE_FITNESS else ""
    print(f"  {tag} DONE  ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  sigma={best_sigma:.4f}"
          f"{fit_str}  {t_s:.0f}s  AL={len(al_cands)}")

    return dict(seed=seed, best_ppf=best_ppf, best_cycle=best_cycle, best_sigma=best_sigma,
                best_fitness=best_fitness, best_pat=best_pat.tolist(), time_s=t_s,
                n_al=len(al_cands), div_final=float(div), history=history, al_candidates=al_cands)


# =============================================================================
# SECTION 5 — SHAP (optional traceability)
# =============================================================================

def run_shap(all_results, res):
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' not installed — skip. pip install shap --break-system-packages")
        return None
    print("\n[SHAP] Building explainer ...")
    model, ym_mean, ym_scale, IDX_PPF, X_all = res['model'], res['ym_mean'], res['ym_scale'], res['IDX_PPF'], res['X_grid']

    def predict_fn(flat_batch):
        flat_int = np.round(np.clip(flat_batch, 1, N_TYPES)).astype(np.int32)
        Xg = _flat_to_grid_batch(flat_int)
        y_sc = model(tf.constant(Xg, dtype=tf.int32), training=False).numpy()
        return y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]

    bg_idx  = np.random.choice(len(X_all), SHAP_BACKGROUND_N, replace=False)
    bg_flat = np.array([_grid_to_flat(X_all[i]) for i in bg_idx], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fn, bg_flat)

    explain_records = [dict(source='qica_best', seed=r['seed'], pattern=np.array(r['best_pat'], dtype=np.float32))
                        for r in all_results]
    all_al_flat = [c for r in all_results for c in r['al_candidates']]
    if all_al_flat:
        al_df_full = pd.DataFrame(all_al_flat).sort_values('composite', ascending=False)
        n_extra = max(0, SHAP_MAX_EXPLAIN - len(explain_records))
        for _, row in al_df_full.head(n_extra).iterrows():
            pat = np.array([row[f'pos_{k}'] for k in range(N_POS)], dtype=np.float32)
            explain_records.append(dict(source='al_candidate', seed=int(row['seed']), pattern=pat))
    explain_records = explain_records[:SHAP_MAX_EXPLAIN]

    print(f"[SHAP] Explaining {len(explain_records)} patterns ...")
    rows, t0 = [], time.time()
    for i, rec in enumerate(explain_records):
        sv = np.array(explainer.shap_values(rec['pattern'][None, :], nsamples=SHAP_NSAMPLES, silent=True)).reshape(-1)
        row = dict(source=rec['source'], seed=rec['seed'], ppf_pred=float(predict_fn(rec['pattern'][None, :])[0]))
        for p in range(N_POS):
            row[f'pos_{p}_type'], row[f'pos_{p}_shap'] = int(rec['pattern'][p]), float(sv[p])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  [SHAP] {i+1}/{len(explain_records)}  ({time.time()-t0:.0f}s)")

    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(f'{OUT_PREFIX}_shap.csv', index=False)
    print(f"[SHAP] Saved {OUT_PREFIX}_shap.csv")

    sens_norm = res['sens_norm']
    shap_cols = [f'pos_{p}_shap' for p in range(N_POS)]
    mean_abs_shap = shap_df[shap_cols].abs().mean(axis=0).values
    shap_rank, sens_rank = np.argsort(mean_abs_shap)[::-1], np.argsort(sens_norm)[::-1]
    overlap5 = len(set(shap_rank[:5]) & set(sens_rank[:5]))
    print(f"[SHAP] Top5 SHAP vs sensitivity overlap: {overlap5}/5")
    return shap_df


# =============================================================================
# SECTION 6 — OPENMC EXPORT  (this is the key deliverable you asked for)
# =============================================================================

def export_for_openmc(all_results):
    """
    Writes best_patterns_for_openmc.csv in the exact schema
    openmc_beavrs_vver1000_FIXED_2.py's --al_candidates_csv expects
    (pos_0..pos_30 columns), one row per seed's best pattern, PLUS prints
    a ready-to-paste --single_pattern string for the single overall best.
    """
    rows = []
    for r in all_results:
        rows.append({f'pos_{k}': int(r['best_pat'][k]) for k in range(N_POS)})
    pd.DataFrame(rows).to_csv(OPENMC_CSV, index=False)
    print(f"\n[OPENMC EXPORT] Saved {OPENMC_CSV} ({len(rows)} patterns, "
          f"al_candidates_csv-compatible)")

    best = min(all_results, key=lambda r: r['best_ppf'])
    pat_str = ",".join(str(int(x)) for x in best['best_pat'])
    print(f"\n[OPENMC EXPORT] Best overall (seed={best['seed']}, CNN ppf={best['best_ppf']:.4f}, "
          f"CNN cycle={best['best_cycle']:.1f}d):")
    print(f"  --single_pattern \"{pat_str}\"")
    print(f"\n  Quick check (minutes, BOC-only):")
    print(f'  python openmc_beavrs_vver1000_FIXED_2.py --single_pattern "{pat_str}" '
          f'--quick_check --particles 4000 --batches 60 --inactive 40')
    print(f"\n  Full depletion, ALL {len(rows)} seeds' best patterns (hours, real cycle_length):")
    print(f"  python openmc_beavrs_vver1000_FIXED_2.py --al_candidates_csv {OPENMC_CSV} "
          f"--chain <chain_file.xml> --particles 4000 --batches 60 --inactive 15 "
          f"--out_csv {OUT_PREFIX}_verified.csv")


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================

def main():
    res = load_everything()
    print(f"\n{'='*68}")
    print(f"QICA v11  |  {N_SEEDS} seeds x {N_GENS} gens x pop={N_POP} x MC={MC_SAMP}")
    print(f"quantum={USE_QUANTUM}  trust_free_frac={TRUST_FREE_FRAC}  "
          f"warm_start={WARM_START}  cycle_fitness={USE_CYCLE_FITNESS}")
    print(f"{'='*68}\n")

    all_results, all_history, all_al = [], [], []
    t0_all = time.time()
    for seed in SEEDS:
        r = run_qica(seed, res, N_GENS, N_POP, MC_SAMP)
        all_results.append(r)
        all_history.extend(r['history']); all_al.extend(r['al_candidates'])

    ppf_vals = [r['best_ppf'] for r in all_results]
    print(f"\n{'='*68}\nQICA v11 — {N_SEEDS} seeds")
    print(f"  best_ppf = {np.mean(ppf_vals):.4f} +/- {np.std(ppf_vals):.4f}  "
          f"[{min(ppf_vals):.4f} - {max(ppf_vals):.4f}]")
    if USE_CYCLE_FITNESS:
        fit_vals = [r['best_fitness'] for r in all_results]
        cyc_vals = [r['best_cycle'] for r in all_results]
        print(f"  best_fitness = {np.mean(fit_vals):.2f} +/- {np.std(fit_vals):.2f}")
        print(f"  best_cycle   = {np.mean(cyc_vals):.1f} +/- {np.std(cyc_vals):.1f} d")
    print(f"[TOTAL RUNTIME] {(time.time()-t0_all)/60:.1f} min")

    # Save
    rows = [dict(seed=r['seed'], best_ppf=r['best_ppf'], best_cycle=r['best_cycle'],
                 best_sigma=r['best_sigma'], best_fitness=r.get('best_fitness', float('nan')),
                 time_s=r['time_s'], n_al=r['n_al'], div_final=r['div_final']) for r in all_results]
    pd.DataFrame(rows).to_csv(f'{OUT_PREFIX}_summary.csv', index=False)
    pd.DataFrame(all_history).to_csv(f'{OUT_PREFIX}_history.csv', index=False)
    if all_al:
        pd.DataFrame(all_al).to_csv(f'{OUT_PREFIX}_al_candidates.csv', index=False)
    print(f"[SAVED] {OUT_PREFIX}_summary.csv  {OUT_PREFIX}_history.csv  {OUT_PREFIX}_al_candidates.csv")

    if SHAP_ENABLE:
        run_shap(all_results, res)

    export_for_openmc(all_results)

    best_overall = min(all_results, key=lambda r: r['best_ppf'])
    print(f"\n[BEST OVERALL] seed={best_overall['seed']}  ppf={best_overall['best_ppf']:.4f}  "
          f"cycle={best_overall['best_cycle']:.1f}d")
    print(f"  loading pattern: {best_overall['best_pat']}")


if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    for f in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN] {SENS_FILE} not found — uniform sensitivity")
    main()