"""
=============================================================================
qica_v10.py — Anti-Extrapolation QICA  |  Global AL Cap  |  Built-in Goodness Report
=============================================================================
WHY THIS VERSION EXISTS
─────────────────────────
qica_v9-final.py optimizes pure min(predicted PPF) with no floor and no
reward for cycle length. Nothing stops it from riding the CNN into a region
below the training distribution's minimum (data min ≈ 1.612-1.697 depending
on split) — a classic surrogate-exploitation failure mode. qica-v5-2.py never
had this problem because its fitness function only penalizes PPF *above* the
safety limit and actively *rewards* cycle length, so there's no incentive to
chase an ever-lower PPF once it stops being physically meaningful.

v10 = v9-final's sensitivity+entropy AL machinery and SHAP traceability,
grafted onto a v5-style composite fitness function, with an explicit
out-of-distribution (OOD) penalty so the search is pushed AWAY from PPF
values below what the CNN was actually trained on, not just neutral about it.

CHANGES vs qica_v9-final.py
─────────────────────────────
1. FITNESS FUNCTION REWORK (the actual fix):
     fitness = cycle_mean
                - W_PPF_SOFT     * ppf_mean                  (gentle pull down)
                - W_PPF_PENALTY  * max(0, ppf_mean - PPF_LIMIT)   (hard safety wall)
                - W_UNCERTAINTY  * sigma
                - W_OOD_PENALTY  * max(0, DATA_PPF_MIN - ppf_mean) (NEW — extrapolation wall)
                + mono_bonus
   DATA_PPF_MIN is computed empirically at startup (deterministic CNN pass
   over the full training set), not hardcoded — so it tracks whatever data
   you're actually training on. Any pattern predicted below that floor now
   COSTS fitness instead of being free money, which is what was pulling v9
   into (probable) hallucinated territory.

2. GLOBAL AL CAP = 50 (not 50 per seed).
   qica_final_al_candidates.csv from v9-final had up to N_SEEDS x
   AL_MAX_CANDS rows (500 for 10 seeds) because the candidate list and the
   dedup set were both reset per-seed. v10 uses one shared candidate pool
   and one shared seen-set across ALL seeds, periodically trimmed to the
   top AL_MAX_CANDS=50 by composite score. The saved CSV is guaranteed
   <=50 rows, already deduplicated — no more post-hoc dedup script needed.

3. FASTER BY DEFAULT.
   5 seeds (not 10), 150 gens (not 250), pop=60 (not 80), MC=15 (not 25),
   plus an early-stop check per seed once it's been stagnant for a long
   stretch after its last escape burst. Rough estimate: ~10-15 min total
   on the same hardware that took v9-final ~83 min for 10 seeds.

4. BUILT-IN GOODNESS REPORT (new final section).
   Printed automatically at the end, and saved to qica_v10_goodness.txt:
     - mean/std/min best_ppf vs data distribution (min, 5th/10th pctile)
     - explicit "OOD FLAG" if any seed's best_ppf sits below DATA_PPF_MIN
     - AL diagnostics (al_h_cv, al_sig_corr, al_sn_mean) — same meaning as v9
     - SHAP vs gradient-sensitivity top-5 overlap — same meaning as v9
     - a one-line verdict + a concrete "run this pattern through OpenMC next"
       instruction with the exact --single_pattern string to paste in.

5. Everything else — sensitivity-linked entropy gate (H_sens), sensitivity-
   weighted composite AL scoring, CNN-warm-start seeding (from qica-v5-2),
   monotonicity bonus (from qica-v5-2/final_entropy_test), SHAP traceability
   — is unchanged in spirit from v9-final, just wired into the new fitness.

INPUTS  (same as v9-final):
  cnn_v9_model.keras, cnn_v9_config.json, train_type_freq_v9.npy,
  cnn_v9_sens.csv (optional), ml_dataset_constrained.csv

OUTPUTS:
  qica_v10_summary.csv        one row per seed
  qica_v10_history.csv        gen-by-gen history, all seeds
  qica_v10_al_candidates.csv  <=50 rows, globally deduped, sorted by priority
  qica_v10_results.png
  qica_v10_shap.csv / qica_v10_shap_summary.png
  qica_v10_goodness.txt       the built-in goodness/verdict report
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
print("qica_v10.py — Anti-Extrapolation Fitness + Global AL Cap + Goodness Report\n")


# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

QUICK_TEST = False   # True = 1 seed, 20 gens, pop 20 (~2 min sanity check)

N_SEEDS  = 1   if QUICK_TEST else 5
N_GENS   = 20  if QUICK_TEST else 150
N_POP    = 20  if QUICK_TEST else 60
MC_SAMP  = 8   if QUICK_TEST else 15
SEEDS    = [42] if QUICK_TEST else [42, 137, 271, 509, 1023]

# Early stop: if a seed has been stagnant this many gens straight with no
# improvement AND has already fired at least one escape burst, stop early.
EARLY_STOP_STAG = 45

# Files (unchanged locations)
MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters
N_EMPIRES_INIT    = 6
ASSIMILATION_RATE = 0.3
REV_START         = 0.35
REV_END           = 0.08
STAGNATION_PAT    = 15
ESCAPE_BURST      = 20

# ── Fitness weights (v5-style: reward cycle length, only wall off PPF above
#    the safety limit, and now also wall off PPF BELOW the training floor) ──
PPF_LIMIT       = 3.5    # hard safety ceiling (unchanged meaning from v5/v9)
W_PPF_SOFT      = 6.0    # gentle downward pull on PPF (from qica-v5-2)
W_PPF_PENALTY   = 80.0   # hard penalty above PPF_LIMIT (from qica-v5-2)
W_UNCERTAINTY   = 40.0   # penalize high MC-dropout sigma (from qica-v5-2)
W_MONOTONICITY  = 10.0   # reward late-cycle PPF decreasing (from qica-v5-2)
W_OOD_PENALTY   = 120.0  # NEW: penalty per unit below DATA_PPF_MIN — this is
                          # the actual anti-extrapolation fix. Weighted even
                          # higher than the safety-ceiling penalty because a
                          # hallucinated "great" pattern is arguably worse
                          # than a merely-suboptimal-but-real one.
OOD_MARGIN      = 0.02   # small buffer below DATA_PPF_MIN before penalty kicks
                          # in, so the floor isn't punishing legitimate values
                          # that are merely close to (not below) the training min

# AL thresholds
AL_SIGMA_FRAC     = 0.50
AL_H_SENS_THRESH  = 1.30   # see qica_v9-final.py calibration note — unchanged
AL_MAX_CANDS      = 50     # NOW GLOBAL across all seeds, not per-seed
AL_TRIM_AT        = 100    # trim the working pool back down to AL_MAX_CANDS
                            # once it grows past this, to keep memory/sort cost small
ALPHA_SENS_WT     = 0.4

# SHAP
SHAP_ENABLE       = True
SHAP_BACKGROUND_N = 60
SHAP_NSAMPLES     = 150
SHAP_MAX_EXPLAIN  = 25     # 5 bests + 20 AL top candidates (scaled down from 40
                            # since we now have 5 seeds not 10, and a 50-cap AL pool)

# Output
OUT_PREFIX    = 'qica_v10'
SUMMARY_CSV   = f'{OUT_PREFIX}_summary.csv'
HISTORY_CSV   = f'{OUT_PREFIX}_history.csv'
AL_CSV        = f'{OUT_PREFIX}_al_candidates.csv'
SHAP_CSV      = f'{OUT_PREFIX}_shap.csv'
PLOT_PNG      = f'{OUT_PREFIX}_results.png'
SHAP_PNG      = f'{OUT_PREFIX}_shap_summary.png'
GOODNESS_TXT  = f'{OUT_PREFIX}_goodness.txt'

# BEAVRS geometry
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
# SECTION 2 — LOAD + OOD FLOOR CALIBRATION
# =============================================================================

def load_everything():
    print(f"[LOAD] {MODEL_FILE} ...")
    model = keras.models.load_model(MODEL_FILE, compile=False)
    print(f"  input={model.input_shape}  output={model.output_shape}")

    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    IDX_PPF   = cfg['IDX_PPF_MAX']
    IDX_CYCLE = cfg['IDX_CYCLE']
    IDX_STEPS_S = cfg.get('IDX_PPF_STEPS_START')
    IDX_STEPS_E = cfg.get('IDX_PPF_STEPS_END')
    ym_mean   = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
    ym_scale  = np.array(cfg['ym_scaler_scale'], dtype=np.float32)

    type_freq = np.load(FREQ_FILE).astype(np.float32)   # (31, 9)
    pos_ent   = -np.sum(type_freq * np.log(type_freq + 1e-9), axis=1)
    free_entr = set(np.argsort(pos_ent)[::-1][:20].tolist())
    print(f"[TRUST] {len(free_entr)}/31 positions free (entropy)")

    if os.path.exists(SENS_FILE):
        sens_df   = pd.read_csv(SENS_FILE)
        sens_norm = sens_df['sensitivity_norm'].values.astype(np.float32)
        top5      = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"[SENS]  range={sens_norm.min():.3f}-{sens_norm.max():.3f}  top5={top5}")
    else:
        print("[WARN]  sensitivity file not found - uniform sensitivity")
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

    # ── NEW: empirical OOD floor. Deterministic CNN pass over the WHOLE
    # training set (not MC dropout — we want the point estimate distribution,
    # same quantity QICA's fitness function will be compared against). ──────
    print("[CAL]   Computing DATA_PPF_MIN from full-dataset CNN inference ...")
    ppf_all = []
    for i in range(0, len(X_grid), 256):
        batch = tf.constant(X_grid[i:i+256], dtype=tf.int32)
        y_sc  = model(batch, training=False).numpy()
        ppf_all.extend((y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]).tolist())
    ppf_all = np.array(ppf_all, dtype=np.float32)
    data_ppf_min   = float(ppf_all.min())
    data_ppf_p05   = float(np.percentile(ppf_all, 5))
    data_ppf_p10   = float(np.percentile(ppf_all, 10))
    print(f"  DATA_PPF_MIN (CNN pred) = {data_ppf_min:.4f}  |  "
          f"5th pct = {data_ppf_p05:.4f}  |  10th pct = {data_ppf_p10:.4f}")
    print(f"  OOD penalty floor set to {data_ppf_min:.4f} "
          f"(margin {OOD_MARGIN}, weight {W_OOD_PENALTY})\n")

    return dict(
        model=model, ym_mean=ym_mean, ym_scale=ym_scale,
        IDX_PPF=IDX_PPF, IDX_CYCLE=IDX_CYCLE,
        IDX_STEPS_S=IDX_STEPS_S, IDX_STEPS_E=IDX_STEPS_E,
        type_freq=type_freq, free_entr=free_entr, sens_norm=sens_norm,
        X_grid=X_grid, al_sig_thr=al_thr,
        data_ppf_min=data_ppf_min, data_ppf_p05=data_ppf_p05, data_ppf_p10=data_ppf_p10,
        ppf_all=ppf_all,
    )


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


def _mc_predict(model, X_grids, ym_mean, ym_scale, idx_ppf, n=MC_SAMP):
    """Legacy single-index MC predictor (used only for startup calibration)."""
    preds = []
    Xt    = tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()
        preds.append(y_sc[:, idx_ppf] * ym_scale[idx_ppf] + ym_mean[idx_ppf])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def _mc_predict_full(model, X_grids, res, n=MC_SAMP):
    """
    MC-dropout batch evaluator returning everything the v5-style fitness
    function needs: ppf mean/std, cycle mean, and (if available) the PPF
    burnup-step trajectory for the monotonicity bonus.
    """
    ym_mean, ym_scale = res['ym_mean'], res['ym_scale']
    IDX_PPF, IDX_CYCLE = res['IDX_PPF'], res['IDX_CYCLE']
    IDX_STEPS_S, IDX_STEPS_E = res['IDX_STEPS_S'], res['IDX_STEPS_E']

    Xt = tf.constant(X_grids, dtype=tf.int32)
    mc_sc = np.stack([model(Xt, training=True).numpy() for _ in range(n)])  # (n, B, out)

    ppf_sc   = mc_sc[:, :, IDX_PPF]
    ppf_phys = ppf_sc * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]
    ppf_mean = ppf_phys.mean(axis=0)
    ppf_std  = ppf_phys.std(axis=0)

    cyc_sc   = mc_sc[:, :, IDX_CYCLE]
    cyc_phys = cyc_sc * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE]
    cyc_mean = cyc_phys.mean(axis=0)

    mono_bonus = np.zeros(X_grids.shape[0], dtype=np.float32)
    if IDX_STEPS_S is not None and IDX_STEPS_E is not None:
        steps_sc   = mc_sc[:, :, IDX_STEPS_S:IDX_STEPS_E].mean(axis=0)  # (B, K)
        steps_phys = steps_sc * ym_scale[IDX_STEPS_S:IDX_STEPS_E] + ym_mean[IDX_STEPS_S:IDX_STEPS_E]
        late       = steps_phys[:, 3:]
        if late.shape[1] > 1:
            diffs      = late[:, 1:] - late[:, :-1]
            mono_bonus = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / max(late.shape[1] - 1, 1))

    return ppf_mean.astype(np.float32), ppf_std.astype(np.float32), \
           cyc_mean.astype(np.float32), mono_bonus.astype(np.float32)


def _fitness(ppf_mean, ppf_std, cyc_mean, mono_bonus, data_ppf_min):
    """
    v5-style composite fitness + NEW out-of-distribution penalty.
    This is the core anti-extrapolation fix: predictions below data_ppf_min
    (minus a small margin) get punished instead of rewarded, so QICA can no
    longer profit from riding the CNN's surrogate past the edge of what it
    was actually trained on.
    """
    ppf_excess = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    ood_excess = np.maximum(0.0, (data_ppf_min - OOD_MARGIN) - ppf_mean)
    fitness = (cyc_mean
               - W_PPF_SOFT * ppf_mean
               - W_PPF_PENALTY * ppf_excess
               - W_UNCERTAINTY * ppf_std
               - W_OOD_PENALTY * ood_excess
               + mono_bonus)
    return fitness.astype(np.float32)


def _compute_pop_h_sens(population, sens_norm):
    N = len(population)
    H_per_pos = np.zeros(N_POS, dtype=np.float64)
    for p in range(N_POS):
        counts = np.zeros(N_TYPES)
        for t in range(N_TYPES):
            counts[t] = (population[:, p] == (t + 1)).sum()
        probs = (counts + 1e-9) / (N + N_TYPES * 1e-9)
        H_per_pos[p] = -np.sum(probs * np.log(probs))
    w = sens_norm.astype(np.float64)
    return float(np.sum(w * H_per_pos) / (np.sum(w) + 1e-9))


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


# =============================================================================
# SECTION 3.5 — CNN-GUIDED WARM-START SEEDS (from qica-v5-2.py)
# =============================================================================

def build_warmstart_seeds(res, n_seeds=8):
    """
    Rank training patterns by deterministic CNN-predicted PPF and return the
    n_seeds lowest — used to seed part of the initial population instead of
    pure random sampling. This nudges the search to start inside the training
    manifold (helps both convergence speed AND staying away from OOD regions),
    exactly the idea from qica-v5-2._load_seeds_via_cnn / _initialize_population.
    """
    ppf_all = res['ppf_all']
    X_grid  = res['X_grid']
    top_idx = np.argsort(ppf_all)[:n_seeds]
    seeds = np.array([_grid_to_flat(X_grid[i]) for i in top_idx])
    print(f"[SEED] {n_seeds} CNN-ranked warm-start patterns "
          f"(pred PPF {ppf_all[top_idx[0]]:.3f}-{ppf_all[top_idx[-1]]:.3f})")
    return seeds


# =============================================================================
# SECTION 4 — GLOBAL AL POOL  (shared across all seeds, capped at 50)
# =============================================================================

class GlobalALPool:
    """
    Fixes the v9-final bug where AL_MAX_CANDS was a PER-SEED cap (so 10 seeds
    x 50 = 500 rows, already-unique because the seen-set was also per-seed).
    Here there is exactly ONE seen-set and ONE candidate list for the whole
    run, so cross-seed duplicates are structurally impossible AND the final
    saved CSV never exceeds AL_MAX_CANDS rows.
    """
    def __init__(self, max_cands=AL_MAX_CANDS, trim_at=AL_TRIM_AT):
        self.cands = []
        self.seen  = set()
        self.max_cands = max_cands
        self.trim_at   = trim_at

    def add(self, pattern, record):
        ph = tuple(int(x) for x in pattern)
        if ph in self.seen:
            return False
        self.seen.add(ph)
        self.cands.append(record)
        if len(self.cands) > self.trim_at:
            self._trim()
        return True

    def _trim(self):
        self.cands.sort(key=lambda d: d['composite'], reverse=True)
        self.cands = self.cands[:self.max_cands]
        self.seen  = {tuple(int(x) for x in d['pattern']) for d in self.cands}

    def finalize(self):
        self._trim()
        return self.cands

    def __len__(self):
        return len(self.cands)


# =============================================================================
# SECTION 5 — QICA RUNNER
# =============================================================================

def run_qica(seed, res, al_pool, warmstart_seeds, n_gens=N_GENS, n_pop=N_POP, mc_samp=MC_SAMP):
    rng       = np.random.default_rng(seed)
    model     = res['model']
    type_freq = res['type_freq']
    free_pos  = res['free_entr']
    sens_norm = res['sens_norm']
    X_all     = res['X_grid']
    al_sig_thr    = res['al_sig_thr']
    data_ppf_min  = res['data_ppf_min']

    tag = f'[s{seed}]'
    print(f"\n  {tag} START  free={len(free_pos)}/31  gens={n_gens}  pop={n_pop}  mc={mc_samp}")
    t0 = time.time()

    # ── Init population: warm-start seeds + random dataset draws ────────────
    n_warm = min(len(warmstart_seeds), n_pop // 4)
    pop_list = [warmstart_seeds[i].copy() for i in range(n_warm)]
    idx0 = rng.choice(len(X_all), n_pop - n_warm, replace=False)
    pop_list += [_grid_to_flat(X_all[i]) for i in idx0]
    population = np.array(pop_list)

    Xg = _flat_to_grid_batch(population)
    ppf_pop, sig_pop, cyc_pop, mono_pop = _mc_predict_full(model, Xg, res, n=mc_samp)
    fit_pop = _fitness(ppf_pop, sig_pop, cyc_pop, mono_pop, data_ppf_min)

    best_idx   = int(np.argmax(fit_pop))
    best_ppf   = float(ppf_pop[best_idx])
    best_cycle = float(cyc_pop[best_idx])
    best_pat   = population[best_idx].copy()
    best_sigma = float(sig_pop[best_idx])
    best_fit   = float(fit_pop[best_idx])
    stag       = 0
    fired_escape_once = False

    print(f"  {tag} Gen   0/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} "
          f"cycle={best_cycle:.1f}d fit={best_fit:.2f}")

    n_emp      = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx = np.argsort(fit_pop)[::-1]     # descending fitness now (was ascending PPF)
    imp_idx    = sorted_idx[:n_emp]
    col_idx    = sorted_idx[n_emp:]
    empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

    history = []
    log_interval = 5 if QUICK_TEST else 25

    for gen in range(1, n_gens + 1):
        rev_rate = REV_START + (REV_END - REV_START) * (gen / n_gens)

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
        ppf_pop, sig_pop, cyc_pop, mono_pop = _mc_predict_full(model, Xg, res, n=mc_samp)
        fit_pop = _fitness(ppf_pop, sig_pop, cyc_pop, mono_pop, data_ppf_min)

        gi = int(np.argmax(fit_pop))
        if float(fit_pop[gi]) > best_fit + 1e-4:
            best_fit   = float(fit_pop[gi])
            best_ppf   = float(ppf_pop[gi])
            best_cycle = float(cyc_pop[gi])
            best_pat   = population[gi].copy()
            best_sigma = float(sig_pop[gi])
            stag       = 0
        else:
            stag += 1

        if stag >= STAGNATION_PAT:
            for _ in range(ESCAPE_BURST):
                ci = int(rng.choice(col_idx))
                population[ci] = _mutate_uniform(
                    population[ci], min(rev_rate * 2.0, 0.9), free_pos, type_freq, rng)
            stag = 0
            fired_escape_once = True

        sorted_idx = np.argsort(fit_pop)[::-1]
        imp_idx    = sorted_idx[:n_emp]
        col_idx    = sorted_idx[n_emp:]
        empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

        div = len(set(map(tuple, population))) / n_pop
        pop_H_sens = _compute_pop_h_sens(population, sens_norm)

        # ── Sensitivity-weighted composite AL flagging -> GLOBAL pool ───────
        new_al = 0
        if len(al_pool) < al_pool.max_cands and pop_H_sens > AL_H_SENS_THRESH:
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
                    added = al_pool.add(population[i], dict(
                        seed=seed, gen=gen, pattern=population[i].tolist(),
                        ppf_pred=float(ppf_pop[i]), sigma=float(sig_pop[i]),
                        h_sens_pop=pop_H_sens, sens_novelty=float(pop_sn_vals[i]),
                        composite=float(comp[i]),
                        **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}
                    ))
                    if added:
                        new_al += 1

        if gen % log_interval == 0:
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} sigma={best_sigma:.4f} "
                  f"| H_sens={pop_H_sens:.3f} div={div:.2f} stag={stag} AL_global={len(al_pool)}")

        history.append(dict(seed=seed, gen=gen, best_ppf=best_ppf, best_fit=best_fit,
                             sigma=best_sigma, h_sens_pop=pop_H_sens, div=div,
                             stag=stag, n_al_global=len(al_pool)))

        # ── Early stop: long stagnation after at least one escape burst ─────
        if fired_escape_once and stag >= EARLY_STOP_STAG:
            print(f"  {tag} [EARLY STOP] gen={gen} stagnant {stag} gens post-escape")
            break

    t_s = time.time() - t0
    ood_flag = best_ppf < (data_ppf_min - OOD_MARGIN)
    print(f"  {tag} DONE  ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  sigma={best_sigma:.4f}  "
          f"fit={best_fit:.2f}  {t_s:.0f}s  {'** OOD FLAG **' if ood_flag else '(within data range)'}")

    return dict(
        seed=seed, best_ppf=best_ppf, best_cycle=best_cycle, best_sigma=best_sigma,
        best_fit=best_fit, best_pat=best_pat.tolist(), time_s=t_s,
        ood_flag=bool(ood_flag), history=history,
    )


# =============================================================================
# SECTION 6 — SHAP TRACEABILITY  (unchanged logic from v9-final)
# =============================================================================

def run_shap(all_results, res, al_pool):
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' package not installed - skipping traceability layer.")
        print("        Install with: pip install shap --break-system-packages")
        return None

    print("\n[SHAP] Building explainer ...")
    model    = res['model']
    ym_mean  = res['ym_mean']
    ym_scale = res['ym_scale']
    IDX_PPF  = res['IDX_PPF']
    X_all    = res['X_grid']

    def predict_fn(flat_batch):
        flat_int = np.round(np.clip(flat_batch, 1, N_TYPES)).astype(np.int32)
        Xg = _flat_to_grid_batch(flat_int)
        y_sc = model(tf.constant(Xg, dtype=tf.int32), training=False).numpy()
        return y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]

    bg_idx  = np.random.choice(len(X_all), SHAP_BACKGROUND_N, replace=False)
    bg_flat = np.array([_grid_to_flat(X_all[i]) for i in bg_idx], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fn, bg_flat)

    explain_records = []
    for r in all_results:
        explain_records.append(dict(source='qica_best', seed=r['seed'],
                                     pattern=np.array(r['best_pat'], dtype=np.float32)))
    al_final = al_pool.finalize()
    if al_final:
        al_df_full = pd.DataFrame(al_final).sort_values('composite', ascending=False)
        n_al_to_explain = max(0, SHAP_MAX_EXPLAIN - len(explain_records))
        for _, row in al_df_full.head(n_al_to_explain).iterrows():
            pat = np.array([row[f'pos_{k}'] for k in range(N_POS)], dtype=np.float32)
            explain_records.append(dict(source='al_candidate', seed=int(row['seed']), pattern=pat))

    explain_records = explain_records[:SHAP_MAX_EXPLAIN]
    print(f"[SHAP] Explaining {len(explain_records)} patterns "
          f"(background n={SHAP_BACKGROUND_N}, nsamples={SHAP_NSAMPLES}) ...")

    rows = []
    t0 = time.time()
    for i, rec in enumerate(explain_records):
        sv = explainer.shap_values(rec['pattern'][None, :], nsamples=SHAP_NSAMPLES, silent=True)
        sv = np.array(sv).reshape(-1)
        row = dict(source=rec['source'], seed=rec['seed'],
                   ppf_pred=float(predict_fn(rec['pattern'][None, :])[0]),
                   shap_base_value=float(explainer.expected_value)
                                    if np.isscalar(explainer.expected_value)
                                    else float(np.ravel(explainer.expected_value)[0]))
        for p in range(N_POS):
            row[f'pos_{p}_type'] = int(rec['pattern'][p])
            row[f'pos_{p}_shap'] = float(sv[p])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  [SHAP] {i+1}/{len(explain_records)}  ({time.time()-t0:.0f}s elapsed)")

    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(SHAP_CSV, index=False)
    print(f"[SHAP] Saved {SHAP_CSV}  ({time.time()-t0:.0f}s total)")

    sens_norm  = res['sens_norm']
    shap_cols  = [f'pos_{p}_shap' for p in range(N_POS)]
    mean_abs_shap = shap_df[shap_cols].abs().mean(axis=0).values
    shap_rank  = np.argsort(mean_abs_shap)[::-1]
    sens_rank  = np.argsort(sens_norm)[::-1]
    overlap5   = len(set(shap_rank[:5]) & set(sens_rank[:5]))
    print(f"[SHAP] Mean |SHAP|-ranked top5 vs sensitivity-ranked top5 overlap: {overlap5}/5")

    _plot_shap(shap_df, mean_abs_shap, sens_norm, shap_rank)
    return dict(shap_df=shap_df, overlap5=overlap5)


def _plot_shap(shap_df, mean_abs_shap, sens_norm, shap_rank):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    order = shap_rank[:15]
    ax.barh(range(15), mean_abs_shap[order][::-1], color='#1B4FBF')
    ax.set_yticks(range(15))
    ax.set_yticklabels([f'pos_{p}' for p in order][::-1], fontsize=8)
    ax.set_xlabel('Mean |SHAP value| on ppf_max')
    ax.set_title(f'Top-15 Positions by SHAP\n(n={len(shap_df)} explained patterns)')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1]
    sens_norm_arr = np.asarray(sens_norm)
    ax.scatter(sens_norm_arr, mean_abs_shap, alpha=0.7, s=40, color='#E05C2E')
    for p in shap_rank[:5]:
        ax.annotate(f'pos_{p}', (sens_norm_arr[p], mean_abs_shap[p]), fontsize=7)
    r = np.corrcoef(sens_norm_arr, mean_abs_shap)[0, 1]
    ax.set_xlabel('Gradient sensitivity (norm.)')
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title(f'SHAP vs Gradient Sensitivity\nr = {r:.3f} (should be positive if consistent)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Saved {SHAP_PNG}")


# =============================================================================
# SECTION 7 — GOODNESS REPORT  (new)
# =============================================================================

def build_goodness_report(all_results, res, al_pool, shap_result):
    lines = []
    def p(s=''):
        lines.append(s)
        print(s)

    ppf_vals   = [r['best_ppf'] for r in all_results]
    cyc_vals   = [r['best_cycle'] for r in all_results]
    ood_flags  = [r['ood_flag'] for r in all_results]
    data_min   = res['data_ppf_min']
    data_p05   = res['data_ppf_p05']
    data_p10   = res['data_ppf_p10']

    p("\n" + "=" * 78)
    p("QICA v10 - GOODNESS REPORT")
    p("=" * 78)
    p(f"  Seeds run            : {len(all_results)}")
    p(f"  best_ppf  mean+-std   : {np.mean(ppf_vals):.4f} +/- {np.std(ppf_vals):.4f}  "
      f"[{min(ppf_vals):.4f} - {max(ppf_vals):.4f}]")
    p(f"  best_cycle mean       : {np.mean(cyc_vals):.1f} d")
    p("")
    p(f"  Training data (CNN pred) PPF min   : {data_min:.4f}")
    p(f"  Training data 5th / 10th pctile    : {data_p05:.4f} / {data_p10:.4f}")
    n_ood = sum(ood_flags)
    if n_ood > 0:
        p(f"  *** OOD FLAG: {n_ood}/{len(all_results)} seed(s) found a best_ppf BELOW "
          f"the training data minimum. ***")
        p(f"      This means even with the OOD penalty active, the CNN is still being")
        p(f"      pushed past what it saw in training on some seeds. Treat these results")
        p(f"      as UNVERIFIED until checked in OpenMC. Consider raising W_OOD_PENALTY")
        p(f"      further if this persists.")
    else:
        p(f"  No seed's best_ppf fell below the training data minimum ({data_min:.4f}).")
        p(f"  This is the expected/healthy outcome of the anti-extrapolation fitness fix.")
    p("")

    al_final = al_pool.finalize()
    p(f"  AL candidates (global, deduped, capped at {AL_MAX_CANDS}) : {len(al_final)}")
    if len(al_final) > 2:
        al_df = pd.DataFrame(al_final)
        al_h_cv = float(al_df['h_sens_pop'].std() / (al_df['h_sens_pop'].mean() + 1e-9))
        al_sig_r = 0.0
        if al_df['sigma'].std() > 1e-9:
            al_sig_r = float(np.corrcoef(al_df['h_sens_pop'].values, al_df['sigma'].values)[0, 1])
        al_sn_mean = float(al_df['sens_novelty'].mean())
        p(f"    al_h_cv (diversity of flagging phases)      : {al_h_cv:.3f}  (higher = flagged across more diverse search phases)")
        p(f"    al_sig_corr r(H_sens, sigma)                : {al_sig_r:.3f}  (lower = entropy gate adds info beyond sigma alone)")
        p(f"    al_sn_mean (mean sensitivity novelty)       : {al_sn_mean:.2f}  (higher = candidates unusual at high-impact positions)")
    p("")

    if shap_result is not None:
        p(f"  SHAP vs gradient-sensitivity top-5 overlap : {shap_result['overlap5']}/5")
        if shap_result['overlap5'] < 3:
            p(f"    [FLAG] Low agreement between SHAP attributions and the gradient")
            p(f"    sensitivity map - inspect before trusting AL picks blindly.")
    else:
        p(f"  SHAP: not run (package missing or SHAP_ENABLE=False)")
    p("")

    best_overall = min(all_results, key=lambda r: r['best_ppf'])
    pat_str = ",".join(str(int(x)) for x in best_overall['best_pat'])
    p(f"  BEST OVERALL: seed={best_overall['seed']}  ppf={best_overall['best_ppf']:.4f}  "
      f"cycle={best_overall['best_cycle']:.1f}d  OOD={best_overall['ood_flag']}")
    p("")
    p("  VERDICT:")
    if n_ood == 0 and (shap_result is None or shap_result['overlap5'] >= 3):
        p("    Search stayed within the training distribution and SHAP/sensitivity agree.")
        p("    This run is a reasonable candidate for OpenMC verification.")
    elif n_ood > 0:
        p("    Extrapolation risk detected on at least one seed - verify in OpenMC before trusting.")
    else:
        p("    Search stayed in-distribution but SHAP/sensitivity disagree somewhat - worth a look.")
    p("")
    p("  NEXT STEP - run this exact command to verify the best pattern in OpenMC:")
    p(f"    python openmc_beavrs_vver1000.py --single_pattern \"{pat_str}\" \\")
    p(f"      --quick_check --boron_search --particles 4000 --batches 60 --inactive 40")
    p("    (--inactive raised to 40 - see chat notes on fission-source convergence.)")
    p("=" * 78)

    with open(GOODNESS_TXT, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n[SAVED] {GOODNESS_TXT}")


# =============================================================================
# SECTION 8 — MAIN
# =============================================================================

def main():
    res = load_everything()
    warmstart_seeds = build_warmstart_seeds(res, n_seeds=8)
    al_pool = GlobalALPool()

    print(f"\n{'='*68}")
    print(f"QICA v10  |  {N_SEEDS} seeds x {N_GENS} gens x pop={N_POP} x MC={MC_SAMP}")
    print(f"Anti-extrapolation fitness + global AL cap={AL_MAX_CANDS}")
    print(f"{'='*68}\n")

    all_results, all_history = [], []
    t0_all = time.time()

    for seed in SEEDS:
        r = run_qica(seed, res, al_pool, warmstart_seeds)
        all_results.append(r)
        all_history.extend(r['history'])

    ppf_vals = [r['best_ppf'] for r in all_results]
    print(f"\n{'='*68}")
    print(f"QICA v10 - {len(all_results)} seeds")
    print(f"{'='*68}")
    print(f"  best_ppf = {np.mean(ppf_vals):.4f} +/- {np.std(ppf_vals):.4f}  "
          f"[{min(ppf_vals):.4f} - {max(ppf_vals):.4f}]")
    print(f"[TOTAL RUNTIME] {(time.time()-t0_all)/60:.1f} min")

    _save(all_results, all_history, al_pool)
    _plot(all_results, all_history)

    shap_result = None
    if SHAP_ENABLE:
        shap_result = run_shap(all_results, res, al_pool)

    build_goodness_report(all_results, res, al_pool, shap_result)


def _save(all_results, all_history, al_pool):
    rows = []
    for r in all_results:
        rows.append(dict(
            seed=r['seed'], best_ppf=r['best_ppf'], best_cycle=r['best_cycle'],
            best_sigma=r['best_sigma'], best_fit=r['best_fit'],
            ood_flag=r['ood_flag'], time_s=r['time_s'],
        ))
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(all_history).to_csv(HISTORY_CSV, index=False)

    al_final = al_pool.finalize()
    if al_final:
        pd.DataFrame(al_final).to_csv(AL_CSV, index=False)
        print(f"[SAVED] {AL_CSV}  ({len(al_final)} rows, globally deduped, capped at {AL_MAX_CANDS})")
    print(f"[SAVED] {SUMMARY_CSV}  {HISTORY_CSV}")


def _plot(all_results, all_history):
    hist_df = pd.DataFrame(all_history)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"QICA v10 - {len(all_results)} seeds  |  Anti-Extrapolation Fitness + Global AL Cap",
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    gens = sorted(hist_df['gen'].unique())
    mn = np.array([hist_df[hist_df['gen'] == g]['best_ppf'].mean() for g in gens])
    sd = np.array([hist_df[hist_df['gen'] == g]['best_ppf'].std() for g in gens])
    ax.plot(gens, mn, color='#1B4FBF', lw=2)
    ax.fill_between(gens, mn - np.nan_to_num(sd), mn + np.nan_to_num(sd), color='#1B4FBF', alpha=0.15)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF')
    ax.set_title('Convergence (mean +/- 1 std across seeds)')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ppf_vals = [r['best_ppf'] for r in all_results]
    ax.hist(ppf_vals, bins=min(8, max(3, len(all_results))), color='#E05C2E', edgecolor='white')
    ax.axvline(np.mean(ppf_vals), color='black', lw=2, label=f'mean={np.mean(ppf_vals):.4f}')
    ax.set_xlabel('Best PPF per seed'); ax.set_ylabel('Count')
    ax.set_title(f'Final PPF Distribution ({len(all_results)} seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    h_gens = np.array([hist_df[hist_df['gen'] == g]['h_sens_pop'].mean() for g in gens])
    ax.plot(gens, h_gens, color='#2CA02C', lw=2)
    ax.axhline(AL_H_SENS_THRESH, color='orange', lw=1.5, ls='--', label='AL gate thresh')
    ax.set_xlabel('Generation'); ax.set_ylabel('H_sens')
    ax.set_title('Sensitivity-Linked Entropy Gate')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    div_gens = np.array([hist_df[hist_df['gen'] == g]['div'].mean() for g in gens])
    ax.plot(gens, div_gens, color='#9467BD', lw=2)
    ax.axhline(0.80, color='red', lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('Generation'); ax.set_ylabel('Population Diversity')
    ax.set_title('Diversity (> 0.80 = still exploring)')
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    al_gens = np.array([hist_df[hist_df['gen'] == g]['n_al_global'].mean() for g in gens])
    ax.plot(gens, al_gens, color='#D62728', lw=2)
    ax.axhline(AL_MAX_CANDS, color='orange', lw=1.5, ls='--', label=f'Global cap={AL_MAX_CANDS}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Global AL candidates')
    ax.set_title('AL Candidates (shared across all seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    best_r = min(all_results, key=lambda r: r['best_ppf'])
    pat = np.array(best_r['best_pat'])
    g_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g_disp[r, c] = pat[pi]; pi += 1
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad('lightgrey')
    im = ax.imshow(g_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly Type')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'{int(pat[pi])}', ha='center', va='center', fontsize=9, fontweight='bold')
                pi += 1
    ax.set_title(f"Best Pattern (all seeds)\nseed={best_r['seed']}  "
                 f"PPF={best_r['best_ppf']:.4f}  cycle={best_r['best_cycle']:.1f}d\n"
                 f"OOD={best_r['ood_flag']}")
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(PLOT_PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {PLOT_PNG}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    print(f"  {N_SEEDS} seeds x {N_GENS} gens x pop={N_POP} x MC={MC_SAMP}")
    est = N_SEEDS * N_GENS * N_POP * MC_SAMP * 0.00025
    if not QUICK_TEST:
        print(f"  Estimated search: ~{est:.0f} min on CPU (+ SHAP pass, ~1-2 min)")
    print()

    for f in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN]  {SENS_FILE} not found - using uniform sensitivity")

    main()