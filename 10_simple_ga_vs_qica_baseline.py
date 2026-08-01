"""
=============================================================================
10_simple_ga_vs_qica_baseline.py
=============================================================================
The "simple, normal GA -- not my CNN, not QICA's quantum encoding -- but on
my real problem" experiment. Unlike 06_simple_ga_entropy_sensitivity.py
(synthetic ground truth, isolates the mechanism), THIS script runs a plain
elitist GA -- no quantum probability encoding, no trust region, no MC
dropout in the fitness loop -- directly against your real, trusted CNN
surrogate, using real loading patterns and the real cycle-length-encoded
assembly types. It's the fair, direct, real-data comparison against your
QICA baseline that answers "does the extra QICA machinery actually earn
its complexity, on my actual problem, not a toy one."

Every run prints your CNN's own sanity-check R2 FIRST, unconditionally, so
you always have your trust baseline in view before looking at anything else.

Runs at the SAME seeds as qica_v11_production.py (42, 137, 271, 509, 1023)
for a like-for-like comparison, and loads qica_v11_summary.csv if present
to compute a noise-floor verdict exactly like your v10 ablation study did.

Also runs the real-data version of the entropy<->sensitivity<->stagnation
test from 06_simple_ga_entropy_sensitivity.py -- same per-seed-then-averaged
methodology, but now using your real CNN gradient sensitivity (cnn_v9_sens.csv)
as the weighting instead of a synthetic ground truth.

Run:  python 10_simple_ga_vs_qica_baseline.py
=============================================================================
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# CONFIG
# =============================================================================
MODEL_FILE   = 'cnn_v9_model.keras'
CONFIG_FILE  = 'cnn_v9_config.json'
DATA_CSV     = 'ml_dataset_constrained.csv'
SENS_FILE    = 'cnn_v9_sens.csv'
QICA_SUMMARY = 'qica_v11_summary.csv'
OUT_PREFIX   = 'simple_ga_real'

N_POS, N_TYPES = 31, 9
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)   # NOTE: update to the corrected 6x8 octant layout from
                       # 09_openmc_octant_fix.py once you've retrained the CNN
                       # on data generated under the fixed geometry.

SEEDS = [42, 137, 271, 509, 1023]     # same as qica_v11_production.py
POP_SIZE, N_GENS = 80, 250            # matching QICA's production budget
TOURNAMENT_K = 3
MUTATION_RATE = 0.03
ELITE_FRAC = 0.10
NOISE_FLOOR = 0.02                    # same PPF noise floor your v10 ablation used
W_PPF_SOFT = 6.0                      # set to match whichever QICA config you're comparing against
USE_CYCLE_FITNESS = False             # mirror QICA's toggle for apples-to-apples
SENSITIVITY_WEIGHTED_MUTATION = True # toggle for the "link sensitivity to entropy
                                       # via mutation rate" idea from the review doc


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.proj = None
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


if not (has(MODEL_FILE) and has(CONFIG_FILE)):
    print("Need cnn_v9_model.keras + cnn_v9_config.json. Exiting.")
    raise SystemExit

print("[LOAD] cnn_v9_model.keras ...")
MODEL = keras.models.load_model(MODEL_FILE, compile=False)
with open(CONFIG_FILE) as f:
    CFG = json.load(f)
YM_MEAN = np.array(CFG['ym_scaler_mean'], dtype=np.float32)
YM_SCALE = np.array(CFG['ym_scaler_scale'], dtype=np.float32)
IDX_PPF, IDX_CYCLE = CFG['IDX_PPF_MAX'], CFG['IDX_CYCLE']


def flat_to_grid(flat):
    g = np.zeros((flat.shape[0], GRID_LAYOUT.shape[0], GRID_LAYOUT.shape[1]), dtype=np.int32)
    pi = 0
    for r in range(GRID_LAYOUT.shape[0]):
        for c in range(GRID_LAYOUT.shape[1]):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = flat[:, pi]; pi += 1
    return g


def evaluate(pop):
    Xg = tf.constant(flat_to_grid(pop), dtype=tf.int32)
    y = MODEL(Xg, training=False).numpy()
    ppf = y[:, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
    cyc = y[:, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
    return ppf, cyc


# =============================================================================
# STEP 0 — CNN sanity-check R2, printed unconditionally, every run
# =============================================================================
print("\n" + "=" * 70)
print("STEP 0 — CNN sanity-check R2 (your trust baseline -- always check this first)")
print("=" * 70)

warm_pool = None
if has(DATA_CSV):
    df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    if all(c in df.columns for c in load_cols):
        warm_pool = df[load_cols].values.astype(np.int32)
        ppf_steps_avail = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
        ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))
        if ppf_steps_avail and ppf_assembs:
            step_max = np.stack([
                df[[f'ppf_s{s}_a{a}' for a in ppf_assembs if f'ppf_s{s}_a{a}' in df.columns]]
                .values.astype(np.float32).max(axis=1)
                for s in ppf_steps_avail
            ], axis=1)
            ppf_true_all = step_max.max(axis=1)
            cyc_true_all = df['cycle_length'].values.astype(np.float32) if 'cycle_length' in df.columns else None

            sample_idx = np.random.default_rng(0).choice(len(df), min(2000, len(df)), replace=False)
            ppf_pred_s, cyc_pred_s = evaluate(warm_pool[sample_idx])
            r2_ppf = r_squared(ppf_true_all[sample_idx], ppf_pred_s)
            print(f"  CNN PPF_max sanity R2 (approx, {len(sample_idx)}-row sample of full "
                  f"dataset -- NOT a held-out test set, just a live trust check): {r2_ppf:.4f}")
            if cyc_true_all is not None:
                r2_cyc = r_squared(cyc_true_all[sample_idx], cyc_pred_s)
                print(f"  CNN cycle_length sanity R2 (same caveat)                              : {r2_cyc:.4f}")
            print("  (For the REAL held-out-test-set R2, use the number cnn-v9.py reported at")
            print("   training time -- this is just a quick, always-available live check.)\n")

if warm_pool is None:
    print("  [SKIP] no ml_dataset_constrained.csv found -- GA will use random init only.\n")


# =============================================================================
# STEP 1 — plain elitist GA (no quantum encoding, no trust region, no MC dropout)
# =============================================================================
sens_weights = None
if has(SENS_FILE):
    sens_df = pd.read_csv(SENS_FILE)
    sens_weights = sens_df['sensitivity_norm'].values


def fitness(ppf, cyc):
    if USE_CYCLE_FITNESS:
        return cyc - W_PPF_SOFT * ppf
    return -ppf   # pure min-PPF, higher fitness = lower PPF


def run_simple_ga(seed):
    rs = np.random.default_rng(seed)
    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rs.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)

    ppf, cyc = evaluate(pop)
    fit = fitness(ppf, cyc)
    best_i = np.argmax(fit)
    best = {'ppf': ppf[best_i], 'cycle': cyc[best_i], 'fit': fit[best_i]}
    stag = 0
    log = []

    for gen in range(N_GENS):
        pos_H = np.zeros(N_POS)
        for p in range(N_POS):
            counts = np.bincount(pop[:, p] - 1, minlength=N_TYPES).astype(np.float64)
            probs = counts / counts.sum()
            pos_H[p] = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
        H_mean = pos_H.mean()
        H_sens = np.average(pos_H, weights=sens_weights) if sens_weights is not None else H_mean

        log.append({'seed': seed, 'gen': gen, 'best_ppf': best['ppf'], 'best_cycle': best['cycle'],
                     'best_fit': best['fit'], 'stag': stag, 'H_mean': H_mean, 'H_sens': H_sens})

        elite_n = max(1, int(POP_SIZE * ELITE_FRAC))
        order = np.argsort(-fit)
        elite = pop[order[:elite_n]]
        children = [pop[i].copy() for i in order[:elite_n]]

        # per-position mutation rate: flat by default, or sensitivity-inverse
        # weighted if SENSITIVITY_WEIGHTED_MUTATION is on (protects high-
        # sensitivity positions once the population starts agreeing on them --
        # see review doc Section 6, item 1)
        if SENSITIVITY_WEIGHTED_MUTATION and sens_weights is not None:
            mut_rates = MUTATION_RATE * (1.5 - sens_weights)   # low sens -> more mutation
            mut_rates = np.clip(mut_rates, 0.005, 0.15)
        else:
            mut_rates = np.full(N_POS, MUTATION_RATE)

        while len(children) < POP_SIZE:
            def tournament():
                cand = rs.choice(POP_SIZE, TOURNAMENT_K, replace=False)
                return pop[cand[np.argmax(fit[cand])]]
            p1, p2 = tournament(), tournament()
            mask = rs.integers(0, 2, size=N_POS).astype(bool)
            child = np.where(mask, p1, p2).astype(np.int32)
            mut_mask = rs.random(N_POS) < mut_rates
            child[mut_mask] = rs.integers(1, N_TYPES + 1, size=mut_mask.sum())
            children.append(child)
        pop = np.stack(children[:POP_SIZE])
        ppf, cyc = evaluate(pop)
        fit = fitness(ppf, cyc)
        gi = np.argmax(fit)
        if fit[gi] > best['fit'] + 1e-6:
            best = {'ppf': ppf[gi], 'cycle': cyc[gi], 'fit': fit[gi]}
            stag = 0
        else:
            stag += 1

    return pd.DataFrame(log), best


print("=" * 70)
print(f"STEP 1 — Plain elitist GA  |  {len(SEEDS)} seeds x {N_GENS} gens x pop={POP_SIZE}  |  "
      f"cycle_fitness={USE_CYCLE_FITNESS}  |  sensitivity-weighted mutation={SENSITIVITY_WEIGHTED_MUTATION}")
print("=" * 70)

all_logs, all_bests = [], []
for s in SEEDS:
    log_df, best = run_simple_ga(s)
    all_logs.append(log_df)
    all_bests.append(best)
    print(f"  seed {s:5d}: best_ppf={best['ppf']:.4f}  best_cycle={best['cycle']:.1f}d")

ga_ppf = np.array([b['ppf'] for b in all_bests])
ga_cyc = np.array([b['cycle'] for b in all_bests])
print(f"\n  Simple GA (n={len(SEEDS)}): best_ppf = {ga_ppf.mean():.4f} +/- {ga_ppf.std():.4f}  "
      f"[{ga_ppf.min():.4f} - {ga_ppf.max():.4f}]")
print(f"                              best_cycle = {ga_cyc.mean():.1f} +/- {ga_cyc.std():.1f} d")

full_log = pd.concat(all_logs, ignore_index=True)
full_log.to_csv(f'{OUT_PREFIX}_history.csv', index=False)


# =============================================================================
# STEP 2 — compare against QICA baseline, same noise-floor logic as v10 ablation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2 — Simple GA vs QICA baseline (noise-floor verdict, same rule your")
print("v10 ablation used: a delta smaller than NOISE_FLOOR=%.2f PPF is NOT distinguishable"
      % NOISE_FLOOR)
print("=" * 70)

if has(QICA_SUMMARY):
    qsum = pd.read_csv(QICA_SUMMARY)
    ppf_col = next((c for c in ['best_ppf', 'ppf', 'ppf_max'] if c in qsum.columns), None)
    cyc_col = next((c for c in ['best_cycle', 'cycle', 'cycle_length'] if c in qsum.columns), None)
    if ppf_col:
        qica_ppf = qsum[ppf_col].values
        qica_mean, qica_std = qica_ppf.mean(), qica_ppf.std()
        delta = qica_mean - ga_ppf.mean()   # positive = simple GA found LOWER (better) PPF
        print(f"  QICA baseline : best_ppf = {qica_mean:.4f} +/- {qica_std:.4f}  (n={len(qica_ppf)})")
        print(f"  Simple GA     : best_ppf = {ga_ppf.mean():.4f} +/- {ga_ppf.std():.4f}  (n={len(ga_ppf)})")
        print(f"  delta (QICA - simple GA) = {delta:+.4f} PPF")
        if abs(delta) < NOISE_FLOOR:
            print(f"  -> WITHIN NOISE FLOOR. Cannot distinguish QICA's extra machinery (quantum")
            print(f"     encoding, trust region, MC-dropout-based uncertainty penalty) from a plain")
            print(f"     GA on this objective/budget. If this holds up, it's a real, useful, and")
            print(f"     slightly uncomfortable finding worth reporting honestly: the complexity may")
            print(f"     be earning its keep on EXPLAINABILITY / uncertainty-awareness grounds, not")
            print(f"     on raw best-PPF-found grounds.")
        elif delta > 0:
            print(f"  -> Simple GA found a BETTER (lower) PPF than QICA, clearing the noise floor.")
            print(f"     Worth double-checking QICA's settings (gens/pop/seeds) are truly comparable")
            print(f"     before concluding QICA's extra machinery isn't earning its complexity here.")
        else:
            print(f"  -> QICA found a better (lower) PPF than the simple GA, clearing the noise floor.")
            print(f"     This is the expected direction if the extra machinery is earning its keep.")
        if cyc_col:
            qica_cyc = qsum[cyc_col].values
            print(f"\n  QICA cycle    : {qica_cyc.mean():.1f} +/- {qica_cyc.std():.1f} d")
            print(f"  Simple GA cycle: {ga_cyc.mean():.1f} +/- {ga_cyc.std():.1f} d")
    else:
        print(f"  [SKIP] {QICA_SUMMARY} doesn't have a recognizable PPF column.")
else:
    print(f"  [SKIP] {QICA_SUMMARY} not found -- run qica_v11_production.py first for a baseline,")
    print(f"  or just eyeball the Simple GA numbers above against the QICA numbers you already have")
    print(f"  in your own notes/logs.")


# =============================================================================
# STEP 3 — entropy <-> sensitivity <-> stagnation, on the REAL problem
# (per-seed-then-averaged, same fix as 06_simple_ga_entropy_sensitivity.py)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3 — Entropy <-> sensitivity <-> stagnation, on your REAL CNN/problem")
print("=" * 70)

if sens_weights is None:
    print("  [SKIP] need cnn_v9_sens.csv for the real sensitivity weighting.\n")
else:
    per_seed_r_stag, per_seed_r_impr = [], []
    for log_df in all_logs:
        d_fit = log_df['best_fit'].diff().fillna(0.0).values
        h_drop = np.concatenate([[0.0], -np.diff(log_df['H_sens'].values)])
        if log_df['stag'].std() > 1e-9:
            per_seed_r_stag.append(np.corrcoef(log_df['H_sens'], log_df['stag'])[0, 1])
        if h_drop.std() > 1e-9 and d_fit.std() > 1e-9:
            per_seed_r_impr.append(np.corrcoef(h_drop, d_fit)[0, 1])

    print(f"  mean per-seed corr(H_sens, stagnation)          = {np.mean(per_seed_r_stag):+.3f}  "
          f"(std={np.std(per_seed_r_stag):.3f}, n={len(per_seed_r_stag)})")
    print(f"  mean per-seed corr(H_sens drop, fit improvement) = {np.mean(per_seed_r_impr):+.3f}  "
          f"(std={np.std(per_seed_r_impr):.3f}, n={len(per_seed_r_impr)})")
    print("  Compare directly to 06_simple_ga_entropy_sensitivity.py's synthetic-ground-truth")
    print("  result. Similar sign/magnitude here (on your real CNN, real problem) is strong")
    print("  confirmation the mechanism holds up outside the toy setting too.\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for log_df in all_logs:
        axes[0].plot(log_df.gen, log_df.H_sens, alpha=0.7, lw=1)
    axes[0].set_xlabel('Generation'); axes[0].set_ylabel('H_sens')
    axes[0].set_title('H_sens Trajectories (real CNN, all seeds)')
    axes[0].grid(alpha=0.3)

    for log_df in all_logs:
        axes[1].plot(log_df.gen, log_df.best_ppf, alpha=0.7, lw=1)
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Best PPF')
    axes[1].set_title('Simple GA Convergence (real CNN, all seeds)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_convergence.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_history.csv  {OUT_PREFIX}_convergence.png")

print("\nDone. Toggle SENSITIVITY_WEIGHTED_MUTATION=True and re-run to test the")
print("'sensitivity-proportional mutation rate' idea from the review doc directly")
print("against this same baseline.")
