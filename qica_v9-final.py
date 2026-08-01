"""
CURRENT ONE

============================================================
qica_v9-final.py — Production QICA for BEAVRS/VVER-1000 Loading-Pattern Search
=============================================================================
This is the "ship it" version, built directly from the qica_ab_final.py A/B
test results:

  A_baseline (σ-only AL, uniform-population entropy gate):
      best_ppf = 1.7982 ± 0.0214  [1.7656 – 1.8250]
  B_sens_al_score (sensitivity-weighted composite AL score):
      best_ppf = 1.7889 ± 0.0182  [1.7667 – 1.8138]

  Δ(A−B) = +0.0093 PPF, B has LOWER variance across seeds (0.0182 vs 0.0214)
  and al_sn (sensitivity novelty of flagged candidates) = 34.45 vs 0.00 for A
  — i.e. B's AL batch is actually finding unusual patterns at positions that
  matter, not just wherever the model happens to be uncertain. Verdict was
  "positive but within a single noise-floor pass" — so this file also widens
  N_SEEDS back up and keeps mode B as the ONLY search mode (no more A/B
  branching in the hot loop — that branching is now dead weight).

WHAT'S NEW vs qica_ab_final.py
───────────────────────────────
1. MODE B IS NOW THE ONLY MODE.
   The arm-switch is gone. Sensitivity-weighted composite AL scoring
   (σ z-score + ALPHA_SENS_WT × sensitivity-novelty z-score) is simply how
   this QICA flags candidates now.

2. SENSITIVITY-LINKED ENTROPY GATE (the actual ask).
   qica_ab_final.py's population entropy H_pop was a FLAT mean over all 31
   positions — a position with sensitivity 0.02 counted exactly as much as
   one with sensitivity 1.00 when deciding whether the population was
   "still exploring enough to flag AL candidates." That's backwards: what
   you actually want to gate on is whether the population is still diverse
   at the positions that move ppf_max. This file replaces the flat mean
   with a sensitivity-weighted entropy:

       H_sens = Σ_p  sens_norm[p] · H_p(pop)  /  Σ_p sens_norm[p]

   where H_p(pop) is the per-position Shannon entropy of the population's
   type distribution at position p (same per-position entropy as before —
   only the aggregation changed). This directly operationalises the A/B
   finding that B's edge came from targeting sensitivity, not just σ.
   AL_H_SENS_THRESH is recalibrated for this new (sensitivity-weighted,
   therefore generally lower-magnitude) statistic — see CALIBRATION below.

3. SHAP TRACEABILITY.
   After the search, every seed's best pattern AND every flagged AL
   candidate gets a SHAP explanation (KernelExplainer over the CNN's
   ppf_max output, background = random dataset sample). This answers
   "why did QICA pick this pattern" and "why did AL flag this candidate"
   in terms the CNN's own sensitivity map should agree with — if a
   candidate's top SHAP positions don't overlap its sens_novelty
   positions, that's a real disagreement worth flagging to a human before
   it goes to OpenMC.
   Output: qica_final_shap.csv (per-position SHAP values for every
   explained pattern) + qica_final_shap_summary.png.

4. N_SEEDS = 10 by default (per the A/B verdict's own recommendation to
   replicate with more seeds before trusting the delta).

RUN COST: 10 seeds × 250 gens × pop 80 × MC 25 ≈ 2x qica_ab_final.py's
single-arm cost. SHAP (KernelExplainer, ~40 explained patterns × nsamples=200)
adds a few more minutes on CPU. Set QUICK_TEST=True to sanity check first.
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
print("qica_final.py — Sensitivity-Weighted AL + Sensitivity-Linked Entropy Gate + SHAP\n")


# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

QUICK_TEST = False   # True = 1 seed, 30 gens, pop 20 (~5 min sanity check)

N_SEEDS  = 1   if QUICK_TEST else 10
N_GENS   = 30  if QUICK_TEST else 250
N_POP    = 20  if QUICK_TEST else 80
MC_SAMP  = 10  if QUICK_TEST else 25
SEEDS    = [42] if QUICK_TEST else [42, 137, 271, 509, 1023, 7, 88, 314, 1618, 2718]

# Files (unchanged locations from cnn_v9.py)
MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters — unchanged from the A/B run
N_EMPIRES_INIT    = 6
ASSIMILATION_RATE = 0.3
REV_START         = 0.35
REV_END           = 0.08
STAGNATION_PAT    = 20
ESCAPE_BURST      = 30

# AL thresholds
AL_SIGMA_FRAC     = 0.50
# CALIBRATION NOTE for AL_H_SENS_THRESH:
#   Flat H_pop (qica_ab_final.py) ranged ~1.44–1.74 nats, gate at 1.40.
#   H_sens is a sensitivity-weighted AVERAGE of the same per-position
#   entropies, so its numeric range is similar in magnitude (weights are
#   normalised sens_norm values summing to their own total, not to N_POS) —
#   it is NOT rescaled to [0,1]. It will typically run slightly LOWER than
#   flat H_pop because high-sensitivity positions tend to sit in the
#   entropy-trust-region's more "watched" 20/31 free positions and converge
#   a bit faster than average. Start at 1.30 and inspect al_h_cv in the
#   summary; if AL count saturates too early (<50 gens to fill AL_MAX_CANDS)
#   raise it, if it never fires lower it. This is logged every run.
AL_H_SENS_THRESH  = 1.30
AL_MAX_CANDS      = 50
ALPHA_SENS_WT     = 0.4    # weight of sensitivity-novelty in composite AL score

# SHAP
SHAP_ENABLE       = True
SHAP_BACKGROUND_N = 60     # background sample size for KernelExplainer
SHAP_NSAMPLES     = 200    # perturbation samples per explanation (KernelExplainer)
SHAP_MAX_EXPLAIN  = 40     # cap total patterns explained (best-per-seed + top AL cands)

# Output
OUT_PREFIX   = 'qica_final'
SUMMARY_CSV  = f'{OUT_PREFIX}_summary.csv'
HISTORY_CSV  = f'{OUT_PREFIX}_history.csv'
AL_CSV       = f'{OUT_PREFIX}_al_candidates.csv'
SHAP_CSV     = f'{OUT_PREFIX}_shap.csv'
PLOT_PNG     = f'{OUT_PREFIX}_results.png'
SHAP_PNG     = f'{OUT_PREFIX}_shap_summary.png'

# BEAVRS geometry (31 active positions in 1/8 symmetry, 6x6 packed grid)
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
# SECTION 1 — CONVRESBLOCK (must match cnn_v9.py exactly for load_model to work)
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

    type_freq = np.load(FREQ_FILE).astype(np.float32)   # (31, 9)
    pos_ent   = -np.sum(type_freq * np.log(type_freq + 1e-9), axis=1)
    free_entr = set(np.argsort(pos_ent)[::-1][:20].tolist())
    print(f"[TRUST] {len(free_entr)}/31 positions free (entropy)")

    if os.path.exists(SENS_FILE):
        sens_df   = pd.read_csv(SENS_FILE)
        sens_norm = sens_df['sensitivity_norm'].values.astype(np.float32)
        top5      = np.argsort(sens_norm)[::-1][:5].tolist()
        print(f"[SENS]  range={sens_norm.min():.3f}–{sens_norm.max():.3f}  top5={top5}")
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
    print(f"[CAL]   median σ={al_thr:.4f}  →  AL σ_thr={al_thr:.4f}\n")

    return dict(
        model=model, ym_mean=ym_mean, ym_scale=ym_scale,
        IDX_PPF=IDX_PPF, IDX_CYCLE=IDX_CYCLE,
        type_freq=type_freq, free_entr=free_entr, sens_norm=sens_norm,
        X_grid=X_grid, al_sig_thr=al_thr,
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
    preds = []
    Xt    = tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()
        preds.append(y_sc[:, idx_ppf] * ym_scale[idx_ppf] + ym_mean[idx_ppf])
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)


def _compute_pop_h_sens(population, sens_norm):
    """
    SENSITIVITY-LINKED entropy gate (the core change vs qica_ab_final.py).

    Old (flat): H_pop = mean_p [ H_p(population) ]
    New (this): H_sens = Σ_p sens_norm[p]·H_p(population) / Σ_p sens_norm[p]

    H_p(population) is unchanged — Shannon entropy of the population's type
    distribution at position p (Laplace-smoothed). Only the aggregation is
    reweighted so positions the CNN says matter for ppf_max dominate the
    "are we still exploring" signal, instead of every position (including
    ones the trust region barely lets move) counting equally.
    """
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
# SECTION 4 — QICA RUNNER (mode B only — this is now just "the algorithm")
# =============================================================================

def run_qica(seed, res, n_gens=N_GENS, n_pop=N_POP, mc_samp=MC_SAMP):
    rng       = np.random.default_rng(seed)
    model     = res['model']
    ym_mean   = res['ym_mean']
    ym_scale  = res['ym_scale']
    IDX_PPF   = res['IDX_PPF']
    IDX_CYCLE = res['IDX_CYCLE']
    type_freq = res['type_freq']
    free_pos  = res['free_entr']
    sens_norm = res['sens_norm']
    X_all     = res['X_grid']
    al_sig_thr= res['al_sig_thr']

    tag = f'[s{seed}]'
    print(f"\n  {tag} START  free={len(free_pos)}/31  gens={n_gens}  pop={n_pop}  mc={mc_samp}")
    t0 = time.time()

    idx0       = rng.choice(len(X_all), n_pop, replace=False)
    population = np.array([_grid_to_flat(X_all[i]) for i in idx0])

    Xg               = _flat_to_grid_batch(population)
    ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)

    best_idx   = int(np.argmin(ppf_pop))
    best_ppf   = float(ppf_pop[best_idx])
    best_pat   = population[best_idx].copy()
    best_sigma = float(sig_pop[best_idx])
    stag       = 0

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    print(f"  {tag} Gen   0/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} cycle={best_cycle:.1f}d")

    n_emp      = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx = np.argsort(ppf_pop)
    imp_idx    = sorted_idx[:n_emp]
    col_idx    = sorted_idx[n_emp:]
    empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

    al_cands = []
    al_seen  = set()
    history  = []
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

        Xg               = _flat_to_grid_batch(population)
        ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)

        gi = int(np.argmin(ppf_pop))
        if float(ppf_pop[gi]) < best_ppf - 1e-5:
            best_ppf   = float(ppf_pop[gi])
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

        sorted_idx = np.argsort(ppf_pop)
        imp_idx    = sorted_idx[:n_emp]
        col_idx    = sorted_idx[n_emp:]
        empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

        div = len(set(map(tuple, population))) / n_pop

        # ── SENSITIVITY-LINKED entropy gate ─────────────────────────────────
        pop_H_sens = _compute_pop_h_sens(population, sens_norm)

        # ── Sensitivity-weighted composite AL flagging (was Arm B) ──────────
        new_al = 0
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
                            seed=seed, gen=gen,
                            ppf_pred=float(ppf_pop[i]), sigma=float(sig_pop[i]),
                            h_sens_pop=pop_H_sens, sens_novelty=float(pop_sn_vals[i]),
                            composite=float(comp[i]),
                            **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}
                        ))
                        new_al += 1
                        if len(al_cands) >= AL_MAX_CANDS:
                            break

        if gen % log_interval == 0:
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} "
                  f"| H_sens={pop_H_sens:.3f} div={div:.2f} stag={stag} AL={len(al_cands)}")

        history.append(dict(seed=seed, gen=gen, best_ppf=best_ppf, sigma=best_sigma,
                             h_sens_pop=pop_H_sens, div=div, stag=stag, n_al=len(al_cands)))

    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    t_s = time.time() - t0
    print(f"  {tag} DONE  ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  σ={best_sigma:.4f}  "
          f"{t_s:.0f}s  AL={len(al_cands)}")

    al_df = pd.DataFrame(al_cands) if al_cands else pd.DataFrame()
    al_h_cv = al_sig_r = al_sn_mean = 0.0
    if len(al_df) > 2:
        al_h_cv = float(al_df['h_sens_pop'].std() / (al_df['h_sens_pop'].mean() + 1e-9))
        if al_df['sigma'].std() > 1e-9:
            al_sig_r = float(np.corrcoef(al_df['h_sens_pop'].values, al_df['sigma'].values)[0, 1])
        al_sn_mean = float(al_df['sens_novelty'].mean())

    return dict(
        seed=seed, best_ppf=best_ppf, best_cycle=best_cycle, best_sigma=best_sigma,
        best_pat=best_pat.tolist(), time_s=t_s, n_al=len(al_cands),
        al_h_cv=al_h_cv, al_sig_corr=al_sig_r, al_sn_mean=al_sn_mean,
        div_final=float(div), history=history, al_candidates=al_cands,
    )


# =============================================================================
# SECTION 5 — SHAP TRACEABILITY
# =============================================================================

def run_shap(all_results, res):
    """
    Explain WHY the QICA (and its AL layer) picked the patterns it did,
    using the CNN's own attributions rather than just its sensitivity map.

    Uses shap.KernelExplainer (model-agnostic, works with the Embedding +
    ConvResBlock architecture without needing a differentiable-op-compatible
    explainer). Background = random dataset sample. Explained set = best
    pattern per seed + highest-composite AL candidates (capped at
    SHAP_MAX_EXPLAIN total to keep runtime sane on CPU).

    Deterministic prediction (training=False) is explained — MC-dropout
    uncertainty is a separate, already-reported signal (σ); SHAP here
    answers "which positions drove the point estimate," not "which
    positions drove the uncertainty."
    """
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' package not installed — skipping traceability layer.")
        print("        Install with: pip install shap --break-system-packages")
        return None

    print("\n[SHAP] Building explainer ...")
    model    = res['model']
    ym_mean  = res['ym_mean']
    ym_scale = res['ym_scale']
    IDX_PPF  = res['IDX_PPF']
    X_all    = res['X_grid']

    def predict_fn(flat_batch):
        """flat_batch: (n, 31) float/int array -> (n,) real ppf_max predictions."""
        flat_int = np.round(np.clip(flat_batch, 1, N_TYPES)).astype(np.int32)
        Xg = _flat_to_grid_batch(flat_int)
        y_sc = model(tf.constant(Xg, dtype=tf.int32), training=False).numpy()
        return y_sc[:, IDX_PPF] * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]

    bg_idx  = np.random.choice(len(X_all), SHAP_BACKGROUND_N, replace=False)
    bg_flat = np.array([_grid_to_flat(X_all[i]) for i in bg_idx], dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fn, bg_flat)

    # Build the explanation set: best pattern per seed + top AL candidates by composite
    explain_records = []
    for r in all_results:
        explain_records.append(dict(source='qica_best', seed=r['seed'],
                                     pattern=np.array(r['best_pat'], dtype=np.float32)))
    all_al_flat = []
    for r in all_results:
        for c in r['al_candidates']:
            all_al_flat.append(c)
    if all_al_flat:
        al_df_full = pd.DataFrame(all_al_flat).sort_values('composite', ascending=False)
        n_al_to_explain = max(0, SHAP_MAX_EXPLAIN - len(explain_records))
        for _, row in al_df_full.head(n_al_to_explain).iterrows():
            pat = np.array([row[f'pos_{k}'] for k in range(N_POS)], dtype=np.float32)
            explain_records.append(dict(source='al_candidate', seed=int(row['seed']),
                                         pattern=pat))

    explain_records = explain_records[:SHAP_MAX_EXPLAIN]
    print(f"[SHAP] Explaining {len(explain_records)} patterns "
          f"(background n={SHAP_BACKGROUND_N}, nsamples={SHAP_NSAMPLES}) ...")

    rows = []
    t0 = time.time()
    for i, rec in enumerate(explain_records):
        sv = explainer.shap_values(rec['pattern'][None, :], nsamples=SHAP_NSAMPLES, silent=True)
        sv = np.array(sv).reshape(-1)   # (31,)
        row = dict(source=rec['source'], seed=rec['seed'],
                   ppf_pred=float(predict_fn(rec['pattern'][None, :])[0]),
                   shap_base_value=float(explainer.expected_value)
                                    if np.isscalar(explainer.expected_value)
                                    else float(np.ravel(explainer.expected_value)[0]))
        for p in range(N_POS):
            row[f'pos_{p}_type']  = int(rec['pattern'][p])
            row[f'pos_{p}_shap']  = float(sv[p])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  [SHAP] {i+1}/{len(explain_records)}  ({time.time()-t0:.0f}s elapsed)")

    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(SHAP_CSV, index=False)
    print(f"[SHAP] Saved {SHAP_CSV}  ({time.time()-t0:.0f}s total)")

    # ── Cross-check: do SHAP's top positions agree with the sensitivity map? ──
    sens_norm  = res['sens_norm']
    shap_cols  = [f'pos_{p}_shap' for p in range(N_POS)]
    mean_abs_shap = shap_df[shap_cols].abs().mean(axis=0).values
    shap_rank  = np.argsort(mean_abs_shap)[::-1]
    sens_rank  = np.argsort(sens_norm)[::-1]
    overlap5   = len(set(shap_rank[:5]) & set(sens_rank[:5]))
    print(f"[SHAP] Mean |SHAP|-ranked top5 vs sensitivity-ranked top5 overlap: {overlap5}/5")
    if overlap5 < 3:
        print("       [FLAG] Low agreement — CNN's gradient-sensitivity map and its "
              "SHAP attributions disagree on what matters. Inspect before trusting AL picks.")

    _plot_shap(shap_df, mean_abs_shap, sens_norm, shap_rank)
    return shap_df


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
# SECTION 6 — MAIN
# =============================================================================

def main():
    res = load_everything()

    print(f"\n{'='*68}")
    print(f"QICA FINAL  |  {N_SEEDS} seeds × {N_GENS} gens × pop={N_POP} × MC={MC_SAMP}")
    print(f"Sensitivity-weighted AL + sensitivity-linked entropy gate")
    print(f"{'='*68}\n")

    all_results, all_history, all_al = [], [], []
    t0_all = time.time()

    for seed in SEEDS:
        r = run_qica(seed, res)
        all_results.append(r)
        all_history.extend(r['history'])
        all_al.extend(r['al_candidates'])

    ppf_vals = [r['best_ppf'] for r in all_results]
    print(f"\n{'='*68}")
    print(f"QICA FINAL — {N_SEEDS} seeds")
    print(f"{'='*68}")
    print(f"  best_ppf = {np.mean(ppf_vals):.4f} ± {np.std(ppf_vals):.4f}  "
          f"[{min(ppf_vals):.4f} – {max(ppf_vals):.4f}]")
    print(f"  (A/B reference — A_baseline: 1.7982 ± 0.0214, B (5 seeds): 1.7889 ± 0.0182)")
    print(f"[TOTAL RUNTIME] {(time.time()-t0_all)/60:.1f} min")

    _save(all_results, all_history, all_al)
    _plot(all_results, all_history)

    if SHAP_ENABLE:
        run_shap(all_results, res)

    best_overall = min(all_results, key=lambda r: r['best_ppf'])
    print(f"\n[BEST OVERALL] seed={best_overall['seed']}  ppf={best_overall['best_ppf']:.4f}  "
          f"cycle={best_overall['best_cycle']:.1f}d")
    print(f"  loading pattern: {best_overall['best_pat']}")
    print(f"\n  NEXT STEP: feed this pattern (and qica_final_al_candidates.csv) into")
    print(f"  openmc_vver1000.py / your BEAVRS OpenMC pipeline for ground-truth labels,")
    print(f"  append to the training CSV, and retrain cnn_v9.py.")


def _save(all_results, all_history, all_al):
    rows = []
    for r in all_results:
        rows.append(dict(
            seed=r['seed'], best_ppf=r['best_ppf'], best_cycle=r['best_cycle'],
            best_sigma=r['best_sigma'], time_s=r['time_s'], n_al=r['n_al'],
            al_h_cv=r['al_h_cv'], al_sig_corr=r['al_sig_corr'],
            al_sn_mean=r['al_sn_mean'], div_final=r['div_final'],
        ))
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(all_history).to_csv(HISTORY_CSV, index=False)
    if all_al:
        pd.DataFrame(all_al).to_csv(AL_CSV, index=False)
    print(f"\n[SAVED] {SUMMARY_CSV}  {HISTORY_CSV}  {AL_CSV}")


def _plot(all_results, all_history):
    hist_df = pd.DataFrame(all_history)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"QICA Final — {N_SEEDS} seeds × {N_GENS} gens  |  "
                 f"Sensitivity-Weighted AL + Sensitivity-Linked Entropy Gate",
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    gens = sorted(hist_df['gen'].unique())
    mn = np.array([hist_df[hist_df['gen'] == g]['best_ppf'].mean() for g in gens])
    sd = np.array([hist_df[hist_df['gen'] == g]['best_ppf'].std() for g in gens])
    ax.plot(gens, mn, color='#1B4FBF', lw=2)
    ax.fill_between(gens, mn - sd, mn + sd, color='#1B4FBF', alpha=0.15)
    ax.axhline(1.697, color='red', lw=1, ls='--', alpha=0.6, label='Data min 1.697')
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF')
    ax.set_title('Convergence (mean ± 1σ across seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ppf_vals = [r['best_ppf'] for r in all_results]
    ax.hist(ppf_vals, bins=min(10, N_SEEDS), color='#E05C2E', edgecolor='white')
    ax.axvline(np.mean(ppf_vals), color='black', lw=2, label=f'mean={np.mean(ppf_vals):.4f}')
    ax.set_xlabel('Best PPF per seed'); ax.set_ylabel('Count')
    ax.set_title(f'Final PPF Distribution ({N_SEEDS} seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    h_gens = np.array([hist_df[hist_df['gen'] == g]['h_sens_pop'].mean() for g in gens])
    ax.plot(gens, h_gens, color='#2CA02C', lw=2)
    ax.axhline(AL_H_SENS_THRESH, color='orange', lw=1.5, ls='--', label='AL gate thresh')
    ax.set_xlabel('Generation'); ax.set_ylabel('H_sens (sensitivity-weighted entropy)')
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
    al_gens = np.array([hist_df[hist_df['gen'] == g]['n_al'].mean() for g in gens])
    ax.plot(gens, al_gens, color='#D62728', lw=2)
    ax.axhline(AL_MAX_CANDS, color='orange', lw=1.5, ls='--', label=f'Cap={AL_MAX_CANDS}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Cumulative AL candidates')
    ax.set_title('AL Candidates Accumulated'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

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
                 f"PPF={best_r['best_ppf']:.4f}  cycle={best_r['best_cycle']:.1f}d")
    ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(PLOT_PNG, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {PLOT_PNG}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    print(f"  {N_SEEDS} seeds × {N_GENS} gens × pop={N_POP} × MC={MC_SAMP}")
    est = N_SEEDS * N_GENS * N_POP * MC_SAMP * 0.00025
    if not QUICK_TEST:
        print(f"  Estimated search: ~{est:.0f} min on CPU (+ SHAP pass)")
    print()

    for f in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN]  {SENS_FILE} not found — using uniform sensitivity")

    main()