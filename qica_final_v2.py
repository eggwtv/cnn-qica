"""
=============================================================================
qica_ab_final.py  —  A vs B Replication  |  5 seeds  |  Bugs Fixed
=============================================================================

WHAT CHANGED FROM qica_final.py
─────────────────────────────────
BUG FIX 1 — AL H_hist gate (critical):
  Old code called _compute_h_hist(population[[pi_idx]], ...) on a single-
  pattern slice. A population of 1 always has zero entropy at every position
  (no diversity to measure), so h_pi > h_thr was always 0 > 0 = False.
  Arms A/C/D flagged zero candidates. Arm B dodged this accidentally because
  it used sigma > al_sig_thr without an H_hist gate.

  Fix: compute H_hist once over the FULL population each generation. This
  gives the true population-level diversity signal. Then use a single
  population H_hist value as the gate (pop_H_hist > H_POP_THRESH), meaning
  "only flag AL candidates in generations where the population is still
  diverse enough to be exploring." Per-pattern flagging remains σ > threshold.

BUG FIX 2 — matplotlib crash:
  Figure height of 11 quadrillion pixels because NaN/zero al_h_cv columns
  fed into bar chart ylim computation. Fixed by:
    (a) np.nan_to_num() on all bar values before plotting
    (b) explicit ax.set_ylim(bottom=0, top=max_val * 1.3) on every bar chart
    (c) catching zero-std edge case in error bars

RUN DESIGN:
  Arms A and B only (3 arms was already settled; C and D were within noise).
  5 seeds for tighter confidence interval on the 0.027 PPF delta.
  250 gens, pop 80, MC 25 — same as the run that produced the numbers.

VERDICT THRESHOLD:
  Seed std is now ~0.008–0.020, so effective noise floor ≈ ±0.025 PPF.
  B beats A if mean_ppf(B) < mean_ppf(A) - 0.025 consistently.

METRICS TO READ (in order):
  1. mean_ppf ± std : primary. B wins if Δ > 0.025 and std(B) ≤ std(A).
  2. al_h_cv_fixed  : H_hist CV of flagged AL candidates (HIGHER = more
                      diverse batch for OpenMC). This is now actually computed.
  3. al_sig_corr    : r(H_hist, σ) in AL candidates. LOWER = B finds patterns
                      σ-alone misses (the whole point of sensitivity weighting).
  4. al_sn_mean     : mean sensitivity novelty of B's candidates. HIGHER =
                      candidates are unusual at high-sensitivity positions.
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
print("qica_ab_final.py  —  A vs B  |  5 seeds  |  Bugs Fixed\n")


# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

QUICK_TEST = False   # True = 1 seed, 30 gens, pop 20 (~5 min sanity check)

N_SEEDS  = 1   if QUICK_TEST else 10
N_GENS   = 30  if QUICK_TEST else 250
N_POP    = 20  if QUICK_TEST else 80
MC_SAMP  = 10  if QUICK_TEST else 25
SEEDS    = [42] if QUICK_TEST else [42, 137, 271, 509, 1023]

# Files
MODEL_FILE = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
DATA_CSV    = 'ml_dataset_constrained.csv'

# QICA hyperparameters — unchanged from the run that produced the results
N_EMPIRES_INIT    = 6
ASSIMILATION_RATE = 0.3
REV_START         = 0.35
REV_END           = 0.08
STAGNATION_PAT    = 20
ESCAPE_BURST      = 30

# AL thresholds
AL_SIGMA_FRAC     = 0.50   # flag patterns above median σ (calibrated per run)
AL_H_POP_THRESH   = 1.40   # FIXED: population-level H_hist gate (nat units)
                             # from your runs: population H_hist stays 1.43–1.74
                             # throughout; 1.40 keeps the gate open most gens
AL_MAX_CANDS      = 50

# Arm B parameters
ALPHA_SENS_WT     = 0.4    # weight of sensitivity-novelty in composite AL score
SENS_FREEZE_K     = 8      # not used by A/B, kept for reference

# Output
OUT_PREFIX  = 'qica_ab_final'
SUMMARY_CSV = f'{OUT_PREFIX}_summary.csv'
HISTORY_CSV = f'{OUT_PREFIX}_history.csv'
AL_CSV      = f'{OUT_PREFIX}_al_candidates.csv'
PLOT_PNG    = f'{OUT_PREFIX}_comparison.png'

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
# SECTION 1 — CONVRESBLOCK
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

    # Calibrate AL σ threshold on a 500-pattern sample
    idx_s   = np.random.choice(len(X_grid), min(500, len(X_grid)), replace=False)
    _, sigs = _mc_predict(model, X_grid[idx_s], ym_mean, ym_scale, IDX_PPF, n=10)
    al_thr  = float(np.median(sigs))
    print(f"[CAL]   median σ={al_thr:.4f}  →  AL σ_thr={al_thr:.4f}\n")

    return dict(
        model=model,
        ym_mean=ym_mean, ym_scale=ym_scale,
        IDX_PPF=IDX_PPF, IDX_CYCLE=IDX_CYCLE,
        type_freq=type_freq,
        free_entr=free_entr,
        sens_norm=sens_norm,
        X_grid=X_grid,
        al_sig_thr=al_thr,
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
    """MC-dropout: returns (mean_ppf, std_ppf) arrays shape (N,)."""
    preds = []
    Xt    = tf.constant(X_grids, dtype=tf.int32)
    for _ in range(n):
        y_sc = model(Xt, training=True).numpy()
        preds.append(y_sc[:, idx_ppf] * ym_scale[idx_ppf] + ym_mean[idx_ppf])
    preds = np.array(preds)   # (n, N)
    return preds.mean(axis=0), preds.std(axis=0)


def _compute_pop_h_hist(population):
    """
    FIXED: Shannon entropy of the POPULATION's type distribution at each position.
    population: (N_pop, 31) int array.
    Returns a scalar — mean entropy across positions.
    This measures how diverse the population is at that generation.
    High value = population still exploring; low value = converged.
    """
    N   = len(population)
    H   = 0.0
    for p in range(N_POS):
        counts = np.zeros(N_TYPES)
        for t in range(N_TYPES):
            counts[t] = (population[:, p] == (t + 1)).sum()
        probs = (counts + 1e-9) / (N + N_TYPES * 1e-9)   # Laplace smoothing
        H    += -np.sum(probs * np.log(probs))
    return H / N_POS


def _mutate_uniform(pat, rev_rate, free_positions, type_freq, rng):
    """Uniform mutation: each free position mutates with prob=rev_rate."""
    mut = pat.copy()
    for p in free_positions:
        if rng.random() < rev_rate:
            probs = type_freq[p] / type_freq[p].sum()
            mut[p] = rng.choice(np.arange(1, N_TYPES + 1), p=probs)
    return mut


def _sensitivity_novelty(pat, type_freq, sens_norm):
    """
    Arm B: surprise at high-sensitivity positions.
    = Σ_p  sens_norm[p] * (-log freq[p, type])
    """
    novelty = 0.0
    for p in range(N_POS):
        t = int(pat[p])
        if 1 <= t <= N_TYPES:
            freq_pt = max(float(type_freq[p, t - 1]), 1e-4)
            novelty += float(sens_norm[p]) * (-np.log(freq_pt))
    return novelty


# =============================================================================
# SECTION 4 — QICA RUNNER
# =============================================================================

def run_qica(arm_label, arm_mode, seed, res, n_gens=N_GENS, n_pop=N_POP, mc_samp=MC_SAMP):
    """
    arm_mode 'A': σ-AL baseline, uniform mutation, entropy trust region.
    arm_mode 'B': same search + sensitivity-weighted composite AL flagging.
    """
    rng       = np.random.default_rng(seed)
    model     = res['model']
    ym_mean   = res['ym_mean']
    ym_scale  = res['ym_scale']
    IDX_PPF   = res['IDX_PPF']
    IDX_CYCLE = res['IDX_CYCLE']
    type_freq = res['type_freq']
    free_pos  = res['free_entr']        # set of ints
    sens_norm = res['sens_norm']
    X_all     = res['X_grid']
    al_sig_thr= res['al_sig_thr']

    tag = f'[{arm_label}|s{seed}]'
    print(f"\n  {tag} START  mode={arm_mode}  free={len(free_pos)}/31  "
          f"gens={n_gens}  pop={n_pop}  mc={mc_samp}")

    t0 = time.time()

    # ── Initialise population from dataset ────────────────────────────────────
    idx0       = rng.choice(len(X_all), n_pop, replace=False)
    population = np.array([_grid_to_flat(X_all[i]) for i in idx0])  # (N, 31)

    # ── First evaluation ──────────────────────────────────────────────────────
    Xg               = _flat_to_grid_batch(population)
    ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)

    best_idx   = int(np.argmin(ppf_pop))
    best_ppf   = float(ppf_pop[best_idx])
    best_pat   = population[best_idx].copy()
    best_sigma = float(sig_pop[best_idx])
    stag       = 0

    Xb         = _flat_to_grid_batch(best_pat[None])
    yb         = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    print(f"  {tag} Gen   0/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} "
          f"cycle={best_cycle:.1f}d")

    # ── Empires ───────────────────────────────────────────────────────────────
    n_emp         = min(N_EMPIRES_INIT, n_pop // 4)
    sorted_idx    = np.argsort(ppf_pop)
    imp_idx       = sorted_idx[:n_emp]
    col_idx       = sorted_idx[n_emp:]
    empire_of     = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

    # ── AL tracking ───────────────────────────────────────────────────────────
    al_cands   = []
    al_seen    = set()
    history    = []

    # Pre-compute sensitivity novelty for Arm B normalisation (updated each gen)
    pop_sn_vals = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    log_interval = 5 if QUICK_TEST else 25

    for gen in range(1, n_gens + 1):
        rev_rate = REV_START + (REV_END - REV_START) * (gen / n_gens)

        # ── Assimilation ──────────────────────────────────────────────────────
        new_pop = population.copy()
        for ci in col_idx:
            imp = empire_of[ci]
            for p in free_pos:
                if rng.random() < ASSIMILATION_RATE:
                    new_pop[ci, p] = population[imp, p]

        # ── Revolution (uniform for both arms) ────────────────────────────────
        for i in range(n_pop):
            new_pop[i] = _mutate_uniform(new_pop[i], rev_rate, free_pos, type_freq, rng)

        population = new_pop

        # ── Evaluate ──────────────────────────────────────────────────────────
        Xg               = _flat_to_grid_batch(population)
        ppf_pop, sig_pop = _mc_predict(model, Xg, ym_mean, ym_scale, IDX_PPF, n=mc_samp)

        # ── Update best ───────────────────────────────────────────────────────
        gi = int(np.argmin(ppf_pop))
        if float(ppf_pop[gi]) < best_ppf - 1e-5:
            best_ppf   = float(ppf_pop[gi])
            best_pat   = population[gi].copy()
            best_sigma = float(sig_pop[gi])
            stag       = 0
        else:
            stag += 1

        # ── Escape burst (uniform, same for both arms) ────────────────────────
        if stag >= STAGNATION_PAT:
            for _ in range(ESCAPE_BURST):
                ci = int(rng.choice(col_idx))
                population[ci] = _mutate_uniform(
                    population[ci], min(rev_rate * 2.0, 0.9), free_pos, type_freq, rng)
            stag = 0

        # ── Empire update ─────────────────────────────────────────────────────
        sorted_idx = np.argsort(ppf_pop)
        imp_idx    = sorted_idx[:n_emp]
        col_idx    = sorted_idx[n_emp:]
        empire_of  = {ci: imp_idx[i % n_emp] for i, ci in enumerate(col_idx)}

        # ── Diversity ─────────────────────────────────────────────────────────
        div = len(set(map(tuple, population))) / n_pop

        # ── FIXED: Population-level H_hist gate ───────────────────────────────
        # One scalar per generation — how diverse is the whole population?
        # Used as a gate: only flag AL when pop is still diverse (exploring).
        pop_H = _compute_pop_h_hist(population)

        # ── AL flagging ───────────────────────────────────────────────────────
        new_al = 0
        if len(al_cands) < AL_MAX_CANDS and pop_H > AL_H_POP_THRESH:

            if arm_mode == 'B':
                # Compute sensitivity novelty for all patterns this generation
                pop_sn_vals = np.array([
                    _sensitivity_novelty(population[i], type_freq, sens_norm)
                    for i in range(n_pop)
                ], dtype=np.float32)
                # z-score both signals across population
                sig_z  = (sig_pop - sig_pop.mean()) / (sig_pop.std() + 1e-8)
                sn_z   = (pop_sn_vals - pop_sn_vals.mean()) / (pop_sn_vals.std() + 1e-8)
                comp   = sig_z + ALPHA_SENS_WT * sn_z
                # Flag top-30% by composite score (no hardcoded threshold)
                comp_thr = float(np.percentile(comp, 70))

                for i in range(n_pop):
                    if float(sig_pop[i]) > al_sig_thr and float(comp[i]) > comp_thr:
                        ph = tuple(population[i])
                        if ph not in al_seen:
                            al_seen.add(ph)
                            al_cands.append(dict(
                                arm=arm_label, seed=seed, gen=gen,
                                ppf_pred=float(ppf_pop[i]),
                                sigma=float(sig_pop[i]),
                                h_hist_pop=pop_H,
                                sens_novelty=float(pop_sn_vals[i]),
                                composite=float(comp[i]),
                                **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}
                            ))
                            new_al += 1
                            if len(al_cands) >= AL_MAX_CANDS:
                                break

            else:  # Arm A
                # σ > threshold AND population is diverse (pop_H gate replaces per-pattern H)
                sig_thr_a = float(np.percentile(sig_pop, 70))  # top-30% by σ
                for i in range(n_pop):
                    if float(sig_pop[i]) > sig_thr_a:
                        ph = tuple(population[i])
                        if ph not in al_seen:
                            al_seen.add(ph)
                            al_cands.append(dict(
                                arm=arm_label, seed=seed, gen=gen,
                                ppf_pred=float(ppf_pop[i]),
                                sigma=float(sig_pop[i]),
                                h_hist_pop=pop_H,
                                sens_novelty=0.0,
                                composite=float(sig_pop[i]),
                                **{f'pos_{k}': int(population[i, k]) for k in range(N_POS)}
                            ))
                            new_al += 1
                            if len(al_cands) >= AL_MAX_CANDS:
                                break

        if gen % log_interval == 0:
            print(f"  {tag} Gen {gen:4d}/{n_gens} | ppf={best_ppf:.4f} σ={best_sigma:.4f} "
                  f"| H_pop={pop_H:.3f} div={div:.2f} stag={stag} AL={len(al_cands)}")

        history.append(dict(
            arm=arm_label, seed=seed, gen=gen,
            best_ppf=best_ppf, sigma=best_sigma,
            h_hist_pop=pop_H, div=div, stag=stag, n_al=len(al_cands),
        ))

    # ── Final cycle of best ───────────────────────────────────────────────────
    Xb = _flat_to_grid_batch(best_pat[None])
    yb = model(tf.constant(Xb, dtype=tf.int32), training=False).numpy()
    best_cycle = float(yb[0, IDX_CYCLE] * ym_scale[IDX_CYCLE] + ym_mean[IDX_CYCLE])

    t_s = time.time() - t0
    print(f"  {tag} DONE  ppf={best_ppf:.4f}  cycle={best_cycle:.1f}d  "
          f"σ={best_sigma:.4f}  {t_s:.0f}s  AL={len(al_cands)}")

    # ── AL diagnostics ────────────────────────────────────────────────────────
    al_df = pd.DataFrame(al_cands) if al_cands else pd.DataFrame()
    al_h_cv = al_sig_r = al_sn_mean = 0.0
    if len(al_df) > 2:
        al_h_cv  = float(al_df['h_hist_pop'].std() / (al_df['h_hist_pop'].mean() + 1e-9))
        if al_df['sigma'].std() > 1e-9:
            al_sig_r = float(np.corrcoef(
                al_df['h_hist_pop'].values, al_df['sigma'].values
            )[0, 1])
        al_sn_mean = float(al_df['sens_novelty'].mean())

    return dict(
        arm=arm_label, mode=arm_mode, seed=seed,
        best_ppf=best_ppf, best_cycle=best_cycle, best_sigma=best_sigma,
        best_pat=best_pat.tolist(), time_s=t_s,
        n_al=len(al_cands), al_h_cv=al_h_cv,
        al_sig_corr=al_sig_r, al_sn_mean=al_sn_mean,
        div_final=float(div),
        history=history, al_candidates=al_cands,
    )


# =============================================================================
# SECTION 5 — EXPERIMENT
# =============================================================================

ARMS = [
    dict(label='A_baseline',     mode='A',
         name='σ-AL Baseline (entropy trust, uniform mut)'),
    dict(label='B_sens_al_score', mode='B',
         name='Sensitivity-Weighted AL Composite Score'),
]

COLORS = ['#1B4FBF', '#E05C2E']


# =============================================================================
# SECTION 6 — MAIN
# =============================================================================

def main():
    res = load_everything()

    print(f"\n{'='*68}")
    print(f"A vs B  |  {N_SEEDS} seeds × {N_GENS} gens × pop={N_POP} × MC={MC_SAMP}")
    print(f"{'='*68}\n")

    all_results = []
    all_history = []
    all_al      = []
    t0_all      = time.time()

    for arm in ARMS:
        arm_res = []
        print(f"\n{'='*68}")
        print(f"ARM: {arm['label']}  ({arm['name']})")
        print(f"{'='*68}")

        for seed in SEEDS:
            r = run_qica(arm['label'], arm['mode'], seed, res)
            all_results.append(r)
            arm_res.append(r)
            all_history.extend(r['history'])
            all_al.extend(r['al_candidates'])

        ppf_vals = [r['best_ppf'] for r in arm_res]
        print(f"\n  ── {arm['label']} ({N_SEEDS} seeds) ──")
        print(f"     best_ppf = {np.mean(ppf_vals):.4f} ± {np.std(ppf_vals):.4f}  "
              f"[{min(ppf_vals):.4f} – {max(ppf_vals):.4f}]")

    print(f"\n[TOTAL RUNTIME] {(time.time()-t0_all)/60:.1f} min")

    _save(all_results, all_history, all_al)
    _print_table(all_results)
    _plot(all_results, all_history, res)


# =============================================================================
# SECTION 7 — OUTPUT
# =============================================================================

def _save(all_results, all_history, all_al):
    rows = []
    for r in all_results:
        rows.append(dict(
            arm=r['arm'], mode=r['mode'], seed=r['seed'],
            best_ppf=r['best_ppf'], best_cycle=r['best_cycle'],
            best_sigma=r['best_sigma'], time_s=r['time_s'],
            n_al=r['n_al'], al_h_cv=r['al_h_cv'],
            al_sig_corr=r['al_sig_corr'], al_sn_mean=r['al_sn_mean'],
            div_final=r['div_final'],
        ))
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    pd.DataFrame(all_history).to_csv(HISTORY_CSV, index=False)
    if all_al:
        pd.DataFrame(all_al).to_csv(AL_CSV, index=False)
    print(f"\n[SAVED] {SUMMARY_CSV}  {HISTORY_CSV}  {AL_CSV}")


def _print_table(all_results):
    """
    Print the verdict table with clear noise-floor guidance.

    With 5 seeds and std ~0.008–0.020, the effective noise floor is:
      ±2σ / sqrt(5) ≈ ±0.016–0.018 PPF.
    A difference > 0.025 PPF that holds across all 5 seeds is signal.
    """
    from collections import defaultdict
    arm_stats = defaultdict(list)
    for r in all_results:
        arm_stats[r['arm']].append(r)

    noise_floor = 0.025   # tightened from ±0.110 because std is now 0.008–0.020

    print(f"\n{'='*90}")
    print(f"FINAL RESULTS  (noise floor now ≈ ±{noise_floor:.3f} PPF  with {N_SEEDS} seeds)")
    print(f"{'='*90}")
    hdr = (f"  {'Arm':<22} {'mean_PPF':>9} {'std_PPF':>8} {'min_PPF':>9} "
           f"{'al_h_cv':>8} {'al_σ_r':>7} {'al_sn':>8} {'div':>6} {'t(s)':>6}")
    print(hdr)
    print(f"  {'-'*86}")

    arm_means = {}
    for arm in ARMS:
        lbl     = arm['label']
        results = arm_stats[lbl]
        ppf     = [r['best_ppf']     for r in results]
        alh     = [r['al_h_cv']      for r in results]
        alr     = [r['al_sig_corr']  for r in results]
        alsn    = [r['al_sn_mean']   for r in results]
        div     = [r['div_final']    for r in results]
        ts      = [r['time_s']       for r in results]
        n_al    = [r['n_al']         for r in results]
        print(f"  {lbl:<22} {np.mean(ppf):>9.4f} {np.std(ppf):>8.4f} "
              f"{min(ppf):>9.4f} {np.mean(alh):>8.3f} {np.mean(alr):>7.3f} "
              f"{np.mean(alsn):>8.2f} {np.mean(div):>6.2f} {np.mean(ts):>6.0f}")
        arm_means[lbl] = (np.mean(ppf), np.std(ppf))

    a_mean = arm_means['A_baseline'][0]
    b_mean, b_std = arm_means['B_sens_al_score']
    delta   = a_mean - b_mean    # positive = B is better (lower PPF)

    print(f"\n  VERDICT:")
    print(f"  Δ(A−B) = {delta:+.4f} PPF  (positive = B is lower = B is better)")
    if delta > noise_floor:
        verdict = f"✓ B BEATS A — delta {delta:.4f} > {noise_floor:.3f} noise floor"
        note    = "Sensitivity-weighted AL scoring modestly but consistently helps."
    elif delta > 0:
        verdict = f"≈ B marginally better but within noise — needs more seeds or gens"
        note    = "Trend is positive; replicate with N_SEEDS=10 to confirm."
    else:
        verdict = f"✗ B does NOT beat A — A baseline is sufficient"
        note    = "Use A_baseline in production."
    print(f"  {verdict}")
    print(f"  {note}")

    print(f"\n  COLUMN GUIDE:")
    print(f"    al_h_cv  : diversity of H_pop values across flagged AL gens.")
    print(f"               HIGHER = AL flagging happened across diverse search phases")
    print(f"    al_σ_r   : r(H_pop, σ) across AL flagging events.")
    print(f"               LOWER = entropy gate isn't just duplicating σ signal")
    print(f"    al_sn    : mean sensitivity novelty of B's flagged patterns.")
    print(f"               HIGHER = B's candidates are unusual at impactful positions")
    print(f"    div      : final population diversity. > 0.80 = still exploring")
    print(f"{'='*90}")


def _plot(all_results, all_history, res):
    """
    6-panel figure. All bar charts have explicit ylim to prevent the matplotlib
    height-overflow crash from zero/NaN values.
    """
    hist_df = pd.DataFrame(all_history)

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"QICA A vs B  |  {N_SEEDS} seeds × {N_GENS} gens  |  "
        f"Sensitivity-Weighted AL (Bug-Fixed)",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def _safe_bar(ax, x, means, stds, colors, labels, title, ylabel,
                  higher_better=False, fmt='.3f'):
        """Draw a bar chart with safe ylim (fixes the matplotlib overflow crash)."""
        means = np.nan_to_num(np.array(means, dtype=float))
        stds  = np.nan_to_num(np.array(stds,  dtype=float))
        bars  = ax.bar(x, means, color=colors, alpha=0.8, yerr=stds,
                       capsize=6, error_kw={'linewidth': 1.5})
        max_val = float(np.max(means + stds)) if np.any(means > 0) else 1.0
        ax.set_ylim(bottom=0, top=max(max_val * 1.35, 0.01))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        suffix = ' ↑ better' if higher_better else ' ↓ better'
        ax.text(0.98, 0.97, suffix, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='grey')
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, m + s + max_val * 0.03,
                    f'{m:{fmt}}', ha='center', va='bottom', fontsize=9)

    arm_labels = [a['label'].split('_', 1)[1] for a in ARMS]
    x = np.arange(len(ARMS))

    # ── 0,0: Convergence curves ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for arm, color in zip(ARMS, COLORS):
        sub  = hist_df[hist_df['arm'] == arm['label']]
        if sub.empty:
            continue
        gens = sorted(sub['gen'].unique())
        mn   = [sub[sub['gen'] == g]['best_ppf'].mean() for g in gens]
        sd   = [sub[sub['gen'] == g]['best_ppf'].std()  for g in gens]
        mn, sd = np.array(mn), np.array(sd)
        ax.plot(gens, mn, color=color, lw=2, label=arm['label'])
        ax.fill_between(gens, mn - sd, mn + sd, color=color, alpha=0.15)
    ax.axhline(1.697, color='red', lw=1, ls='--', alpha=0.6, label='Data min 1.697')
    ax.axhline(2.0,   color='grey', lw=1, ls=':', alpha=0.5)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best PPF (lower better)')
    ax.set_title('Convergence (mean ± 1σ across seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 0,1: Final mean PPF bar ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    from collections import defaultdict
    arm_stats = defaultdict(list)
    for r in all_results:
        arm_stats[r['arm']].append(r)
    means = [np.mean([r['best_ppf'] for r in arm_stats[a['label']]]) for a in ARMS]
    stds  = [np.std( [r['best_ppf'] for r in arm_stats[a['label']]]) for a in ARMS]
    _safe_bar(ax, x, means, stds, COLORS, arm_labels,
              f'Final Best PPF (mean ± std, {N_SEEDS} seeds)',
              'Mean Best PPF')
    # Add noise floor lines
    a_mean = means[0]
    ax.axhline(a_mean + 0.025, color='grey', ls=':', lw=1, alpha=0.7)
    ax.axhline(a_mean - 0.025, color='grey', ls=':', lw=1, alpha=0.7)
    ax.text(len(ARMS) - 0.05, a_mean + 0.027, '±0.025 noise', ha='right',
            fontsize=7, color='grey')

    # ── 0,2: Diversity over time ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for arm, color in zip(ARMS, COLORS):
        sub  = hist_df[hist_df['arm'] == arm['label']]
        if sub.empty:
            continue
        gens = sorted(sub['gen'].unique())
        div  = [sub[sub['gen'] == g]['div'].mean() for g in gens]
        ax.plot(gens, div, color=color, lw=2, label=arm['label'])
    ax.axhline(0.80, color='red', lw=1, ls='--', alpha=0.5, label='div=0.80')
    ax.set_xlabel('Generation'); ax.set_ylabel('Population Diversity')
    ax.set_title('Diversity (> 0.80 = still exploring)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # ── 1,0: AL candidate count over time ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    for arm, color in zip(ARMS, COLORS):
        sub  = hist_df[hist_df['arm'] == arm['label']]
        if sub.empty:
            continue
        gens  = sorted(sub['gen'].unique())
        n_als = [sub[sub['gen'] == g]['n_al'].mean() for g in gens]
        ax.plot(gens, n_als, color=color, lw=2, label=arm['label'])
    ax.axhline(AL_MAX_CANDS, color='orange', lw=1.5, ls='--',
               label=f'Cap={AL_MAX_CANDS}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cumulative AL candidates')
    ax.set_title('AL Candidates Accumulated\n(FIXED: now actually flags patterns)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 1,1: AL candidate quality bars ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    sn_means = [np.mean([r['al_sn_mean']  for r in arm_stats[a['label']]]) for a in ARMS]
    sn_stds  = [np.std( [r['al_sn_mean']  for r in arm_stats[a['label']]]) for a in ARMS]
    _safe_bar(ax, x, sn_means, sn_stds, COLORS, arm_labels,
              'Mean Sensitivity Novelty of AL Candidates\n(B higher = unusual at impactful positions)',
              'Sens. Novelty', higher_better=True, fmt='.2f')

    # ── 1,2: Best pattern grid ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    best_r = min(all_results, key=lambda r: r['best_ppf'])
    pat    = np.array(best_r['best_pat'])
    g_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g_disp[r, c] = pat[pi]; pi += 1
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad('lightgrey')
    im   = ax.imshow(g_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly Type')
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'{int(pat[pi])}',
                        ha='center', va='center', fontsize=9, fontweight='bold')
                pi += 1
    ax.set_title(
        f"Best Pattern Found (all arms)\n"
        f"Arm={best_r['arm']}  seed={best_r['seed']}\n"
        f"PPF={best_r['best_ppf']:.4f}  cycle={best_r['best_cycle']:.1f}d"
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Guard against any remaining overflow before saving
    try:
        plt.savefig(PLOT_PNG, dpi=120, bbox_inches='tight')
        print(f"[SAVED] {PLOT_PNG}")
    except Exception as e:
        print(f"[WARN]  Plot save failed: {e}")
        plt.savefig(PLOT_PNG, dpi=72, bbox_inches='tight')
        print(f"[SAVED] {PLOT_PNG} (fallback dpi=72)")
    plt.close()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print(f"MODE: {'QUICK TEST' if QUICK_TEST else 'FULL RUN'}")
    print(f"  Arms: {[a['label'] for a in ARMS]}")
    print(f"  {N_SEEDS} seeds × {N_GENS} gens × pop={N_POP} × MC={MC_SAMP}")
    est = N_SEEDS * len(ARMS) * N_GENS * N_POP * MC_SAMP * 0.00025
    if not QUICK_TEST:
        print(f"  Estimated: ~{est:.0f} min on CPU")
    print()

    for f in [MODEL_FILE, CONFIG_FILE, FREQ_FILE, DATA_CSV]:
        if not os.path.exists(f):
            print(f"[ERROR] Missing: {f}"); sys.exit(1)
    if not os.path.exists(SENS_FILE):
        print(f"[WARN]  {SENS_FILE} not found — using uniform sensitivity")

    main()