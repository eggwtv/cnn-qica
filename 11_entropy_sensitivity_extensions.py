"""
=============================================================================
11_entropy_sensitivity_extensions.py
=============================================================================
Three checks, all extending 10_simple_ga_vs_qica_baseline.py's toggle:

PART A — Re-run the mirror-pair adjacency test (07_pareto_nsga2_cnn.py's
  Part A) against the NEW 6x8 octant GRID_LAYOUT, now that you've retrained
  sensitivity/entropy statistics under it. The old (arange-reshape) layout
  failed this test outright (mirror pairs statistically indistinguishable
  from random pairs). This is the independent confirmation the new geometry
  is structurally sound, separate from the OpenMC zero-power fix.

PART B — Sobol-weighted mutation, vs the gradient-weighted version you
  already ran (SENSITIVITY_WEIGHTED_MUTATION=True in script 10, which came
  back basically flat: 1.5091 vs 1.5006 baseline). Your PCE result found
  gradient sensitivity and Sobol indices agree on 4/5 top positions but not
  identically -- this swaps the weight source to Sobol (data-derived, model
  -independent) as a cheap, direct ablation against the same baseline.
  Needs mentor_feedback_pce_sobol.csv from the earlier PCE run; if you don't
  have it, this falls back to gradient sensitivity with a warning so the
  script still runs.

PART C — Entropy-triggered sensitivity re-estimation. Idea #2 from the
  review doc: whenever population entropy at a position drops below a
  threshold (the GA has "committed" to a value there), re-run the CNN's
  gradient sensitivity AT THAT SPECIFIC committed input, and compare it to
  the sensitivity estimated earlier from more varied inputs. Local
  sensitivity at one fixed context can genuinely differ from a broader
  estimate (this is exactly why your PCE additive model only explained 6.3%
  of variance -- most of the story is position-pair interactions, so
  "sensitivity" is context-dependent). This gives a live consistency check
  exactly at the moment the GA locks a position in.

Run:  python 11_entropy_sensitivity_extensions.py
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

rng = np.random.default_rng(42)

MODEL_FILE   = 'cnn_v9_model.keras'
CONFIG_FILE  = 'cnn_v9_config.json'
DATA_CSV     = 'ml_dataset_constrained.csv'
SENS_FILE    = 'cnn_v9_sens.csv'
FREQ_FILE    = 'train_type_freq_v9.npy'
SOBOL_FILE   = 'mentor_feedback_pce_sobol.csv'   # from your earlier PCE run, optional
OUT_PREFIX   = '11_entropy_sens_ext'

N_POS, N_TYPES = 31, 9

# NEW 6x8 octant layout (09_openmc_octant_fix.py). If you've retrained the
# CNN/sensitivity/frequency files under this layout, this matches them. If
# you're still on the old 6x6 CNN, switch this back to the old array before
# running Part A (it needs to match whatever generated FREQ_FILE/SENS_FILE).
GRID_ROWS, GRID_COLS = 6, 8
GRID_LAYOUT = np.array([
    [-1, -1, -1, -1, -1, 29, 30, -1],
    [-1, -1, -1, -1, 26, 27, 28, -1],
    [-1, -1, -1, 21, 22, 23, 24, 25],
    [-1, -1, 15, 16, 17, 18, 19, 20],
    [-1,  8,  9, 10, 11, 12, 13, 14],
    [ 0,  1,  2,  3,  4,  5,  6,  7],
], dtype=np.int32)


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


# =============================================================================
# PART A — mirror-pair adjacency recheck under the NEW layout
# =============================================================================
print("=" * 70)
print("PART A — Mirror-pair adjacency check, NEW 6x8 octant GRID_LAYOUT")
print("=" * 70)

if has(FREQ_FILE) and has(SENS_FILE):
    type_freq = np.load(FREQ_FILE).astype(np.float64)
    pos_entropy = -np.sum(type_freq * np.log(type_freq + 1e-12), axis=1)
    sens_df = pd.read_csv(SENS_FILE)
    sens_norm = sens_df['sensitivity_norm'].values

    pairs = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if r >= c or c >= GRID_COLS or r >= GRID_ROWS:
                continue
            # only meaningful within the square sub-block; the 6x8 layout
            # isn't square, so mirror through (r,c)<->(c,r) only where both
            # indices are valid in both directions
            if c < GRID_ROWS and r < GRID_COLS:
                a, b = GRID_LAYOUT[r, c], GRID_LAYOUT[c, r]
                if a >= 0 and b >= 0:
                    pairs.append((int(a), int(b)))

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} valid mirror-pairs under this (non-square) layout -- "
              f"the diagonal-transpose test doesn't apply cleanly to a 6x8 grid. Use the "
              f"true geometric check instead: compare each of the 6 diagonal-boundary "
              f"positions {[0,8,15,21,26,29]} against ITS OWN reflected neighbor pair, "
              f"since those are the physically meaningful symmetric points here, not a "
              f"generic array transpose.")
        diag_positions = [0, 8, 15, 21, 26, 29]
        print(f"\n  Entropy at diagonal-boundary positions: "
              f"{[round(float(pos_entropy[p]), 3) for p in diag_positions]}")
        print(f"  Sensitivity at diagonal-boundary positions: "
              f"{[round(float(sens_norm[p]), 3) for p in diag_positions]}")
        print(f"  Dataset-wide entropy mean: {pos_entropy.mean():.3f}  "
              f"sensitivity mean: {sens_norm.mean():.3f}")
        print(f"  If the diagonal-boundary values sit noticeably off the dataset-wide mean "
              f"in a consistent direction, that's a (weaker, but still informative) signal "
              f"about whether these true-symmetry-axis positions behave differently, without "
              f"needing a same-shape mirror-pair test.")
    else:
        ent_a = np.array([pos_entropy[a] for a, b in pairs])
        ent_b = np.array([pos_entropy[b] for a, b in pairs])
        sen_a = np.array([sens_norm[a] for a, b in pairs])
        sen_b = np.array([sens_norm[b] for a, b in pairs])
        ent_mad = np.mean(np.abs(ent_a - ent_b))
        sen_mad = np.mean(np.abs(sen_a - sen_b))
        rand_pairs = [(rng.integers(0, N_POS), rng.integers(0, N_POS)) for _ in range(200)]
        rand_ent_mad = np.mean([abs(pos_entropy[a] - pos_entropy[b]) for a, b in rand_pairs])
        rand_sen_mad = np.mean([abs(sens_norm[a] - sens_norm[b]) for a, b in rand_pairs])

        print(f"  {len(pairs)} mirror pairs found.")
        print(f"  Mean |entropy diff| mirror pairs   : {ent_mad:.4f}   random pairs: {rand_ent_mad:.4f}")
        print(f"  Mean |sensitivity diff| mirror pairs: {sen_mad:.4f}   random pairs: {rand_sen_mad:.4f}")
        if ent_mad < 0.5 * rand_ent_mad and sen_mad < 0.5 * rand_sen_mad:
            print("  -> PASS: mirror pairs are meaningfully more similar than random pairs. "
                  "Independent confirmation the new octant geometry is structurally sound.")
        else:
            print("  -> Still not a clean pass. Re-verify GRID_LAYOUT against the actual "
                  "BEAVRS coordinate diagram before trusting downstream OpenMC PPF numbers.")
else:
    print("  [SKIP] need train_type_freq_v9.npy + cnn_v9_sens.csv (regenerate cnn-v9.py "
          "under the NEW GRID_LAYOUT first if you haven't already).\n")


# =============================================================================
# Shared loading (needed for Parts B and C)
# =============================================================================
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


if not (has(MODEL_FILE) and has(CONFIG_FILE) and has(DATA_CSV)):
    print("\nNeed cnn_v9_model.keras + cnn_v9_config.json + ml_dataset_constrained.csv "
          "for Parts B/C. Stopping here.")
    raise SystemExit

print("\n[LOAD] cnn_v9_model.keras ...")
MODEL = keras.models.load_model(MODEL_FILE, compile=False)
with open(CONFIG_FILE) as f:
    CFG = json.load(f)
YM_MEAN, YM_SCALE = np.array(CFG['ym_scaler_mean'], np.float32), np.array(CFG['ym_scaler_scale'], np.float32)
IDX_PPF = CFG['IDX_PPF_MAX']

df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
load_cols = [f'loading_{i}' for i in range(N_POS)]
warm_pool = df[load_cols].values.astype(np.int32) if all(c in df.columns for c in load_cols) else None


def flat_to_grid(flat):
    g = np.zeros((flat.shape[0], GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = flat[:, pi]; pi += 1
    return g


def predict_ppf(pop):
    Xg = tf.constant(flat_to_grid(pop), dtype=tf.int32)
    y = MODEL(Xg, training=False).numpy()
    return y[:, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]


def mc_predict(pop, n=15):
    Xg = tf.constant(flat_to_grid(pop), dtype=tf.int32)
    preds = np.stack([MODEL(Xg, training=True).numpy()[:, IDX_PPF] for _ in range(n)])
    real = preds * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
    return real.mean(axis=0), real.std(axis=0)


# =============================================================================
# PART B — Sobol-weighted vs gradient-weighted mutation
# =============================================================================
print("\n" + "=" * 70)
print("PART B — Sobol-weighted mutation rate vs your already-tested gradient version")
print("=" * 70)

if has(SENS_FILE):
    grad_sens = pd.read_csv(SENS_FILE)['sensitivity_norm'].values.astype(np.float32)
else:
    grad_sens = np.full(N_POS, 0.5, dtype=np.float32)
    print("  [WARN] no cnn_v9_sens.csv -- gradient sensitivity fallback is uniform")

if has(SOBOL_FILE):
    sobol_df = pd.read_csv(SOBOL_FILE)
    # expects a 'position'/'sobol_index' style column pair -- adjust names if yours differ
    pos_col = next((c for c in sobol_df.columns if 'pos' in c.lower()), None)
    val_col = next((c for c in sobol_df.columns if 'sobol' in c.lower() or 'index' in c.lower()), None)
    sobol_sens = np.zeros(N_POS, dtype=np.float32)
    if pos_col and val_col:
        for _, row in sobol_df.iterrows():
            p = int(row[pos_col]) if not isinstance(row[pos_col], str) else int(row[pos_col].split('_')[-1])
            if 0 <= p < N_POS:
                sobol_sens[p] = float(row[val_col])
        sobol_sens = sobol_sens / (sobol_sens.max() + 1e-9)
    else:
        print("  [WARN] couldn't parse position/value columns from mentor_feedback_pce_sobol.csv "
              "-- falling back to gradient sensitivity for this arm too.")
        sobol_sens = grad_sens.copy()
else:
    print("  [WARN] mentor_feedback_pce_sobol.csv not found -- run the PCE script first for a "
          "real Sobol comparison. Falling back to gradient sensitivity (this makes Part B a "
          "no-op vs your already-run arm; skip trusting its numbers).")
    sobol_sens = grad_sens.copy()

POP_SIZE, N_GENS, SEEDS = 80, 250, [42, 137, 271, 509, 1023]
TOURNAMENT_K, BASE_MUT_RATE, ELITE_FRAC = 3, 0.03, 0.10


def run_ga(weight_vec, seed):
    rs = np.random.default_rng(seed)
    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rs.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)

    ppf = predict_ppf(pop)
    fit = -ppf
    best = {'ppf': float(ppf[np.argmax(fit)])}
    mut_rates = np.clip(BASE_MUT_RATE * (1.5 - weight_vec), 0.005, 0.15)

    for gen in range(N_GENS):
        elite_n = max(1, int(POP_SIZE * ELITE_FRAC))
        order = np.argsort(-fit)
        children = [pop[i].copy() for i in order[:elite_n]]
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
        ppf = predict_ppf(pop)
        fit = -ppf
        gi = int(np.argmax(fit))
        if float(ppf[gi]) < best['ppf'] - 1e-6:
            best['ppf'] = float(ppf[gi])
    return best['ppf']


results = {}
for label, wvec in [('gradient_weighted', grad_sens), ('sobol_weighted', sobol_sens)]:
    vals = [run_ga(wvec, s) for s in SEEDS]
    results[label] = np.array(vals)
    print(f"  {label:<18} best_ppf = {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
          f"[{min(vals):.4f} - {max(vals):.4f}]")

delta = results['gradient_weighted'].mean() - results['sobol_weighted'].mean()
print(f"\n  delta (gradient - sobol) = {delta:+.4f} PPF  "
      f"({'sobol wins, clears 0.02 noise floor' if delta > 0.02 else 'gradient wins, clears 0.02 noise floor' if delta < -0.02 else 'not distinguishable from noise'})")

pd.DataFrame({k: v for k, v in results.items()}).to_csv(f'{OUT_PREFIX}_partB_sobol_vs_gradient.csv', index=False)
print(f"[SAVED] {OUT_PREFIX}_partB_sobol_vs_gradient.csv")


# =============================================================================
# PART C — entropy-triggered sensitivity re-estimation
# =============================================================================
print("\n" + "=" * 70)
print("PART C — Entropy-triggered sensitivity re-check (local vs global disagreement)")
print("=" * 70)
print("""  Runs one GA to N_GENS. Every generation, checks per-position population
  entropy. The FIRST time a position's entropy drops below ENTROPY_LOCK_THRESH
  (population has "committed" to a value there), re-computes gradient
  sensitivity at that SPECIFIC committed context (perturb just that position
  across all 9 types, holding the rest of the population's consensus pattern
  fixed) and compares it to the ORIGINAL global sensitivity (cnn_v9_sens.csv,
  averaged over 50 low-PPF training patterns). A large disagreement means
  that position's importance is context-dependent -- consistent with your
  PCE finding that ~94% of PPF variance is pairwise interaction, not
  single-position effects.
""")

ENTROPY_LOCK_THRESH = 0.30   # nats; population entropy below this = "locked"

if not has(SENS_FILE):
    print("  [SKIP] need cnn_v9_sens.csv for the global sensitivity baseline.")
else:
    global_sens = pd.read_csv(SENS_FILE)['sensitivity_norm'].values.astype(np.float32)
    rs = np.random.default_rng(7)
    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rs.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)
    ppf = predict_ppf(pop)
    fit = -ppf
    locked = np.zeros(N_POS, dtype=bool)
    local_sens_at_lock = np.full(N_POS, np.nan)
    lock_gen = np.full(N_POS, -1)

    def local_sensitivity_at(consensus_pattern, position):
        """Perturb ONE position across all 9 types, holding everything else
        fixed at the population's current consensus (modal) pattern -- this
        is 'sensitivity at this specific committed context', not the global
        low-PPF-patterns average cnn_v9_sens.csv used."""
        base = np.tile(consensus_pattern, (N_TYPES, 1)).astype(np.int32)
        for t in range(N_TYPES):
            base[t, position] = t + 1
        preds = predict_ppf(base)
        return float(preds.max() - preds.min())

    for gen in range(N_GENS):
        elite_n = max(1, int(POP_SIZE * ELITE_FRAC))
        order = np.argsort(-fit)
        children = [pop[i].copy() for i in order[:elite_n]]
        while len(children) < POP_SIZE:
            def tournament():
                cand = rs.choice(POP_SIZE, TOURNAMENT_K, replace=False)
                return pop[cand[np.argmax(fit[cand])]]
            p1, p2 = tournament(), tournament()
            mask = rs.integers(0, 2, size=N_POS).astype(bool)
            child = np.where(mask, p1, p2).astype(np.int32)
            mut_mask = rs.random(N_POS) < BASE_MUT_RATE
            child[mut_mask] = rs.integers(1, N_TYPES + 1, size=mut_mask.sum())
            children.append(child)
        pop = np.stack(children[:POP_SIZE])
        ppf = predict_ppf(pop)
        fit = -ppf

        consensus = np.array([np.bincount(pop[:, p] - 1, minlength=N_TYPES).argmax() + 1
                               for p in range(N_POS)])
        for p in range(N_POS):
            if locked[p]:
                continue
            counts = np.bincount(pop[:, p] - 1, minlength=N_TYPES).astype(np.float64)
            probs = counts / counts.sum()
            H = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
            if H < ENTROPY_LOCK_THRESH:
                locked[p] = True
                lock_gen[p] = gen
                local_sens_at_lock[p] = local_sensitivity_at(consensus, p)

        if locked.all():
            print(f"  All 31 positions locked by gen {gen}.")
            break

    n_locked = int(locked.sum())
    print(f"  {n_locked}/{N_POS} positions locked within {N_GENS} generations.")
    if n_locked > 0:
        valid = ~np.isnan(local_sens_at_lock)
        local_norm = local_sens_at_lock.copy()
        local_norm[valid] = local_norm[valid] / (local_norm[valid].max() + 1e-9)
        disagreement = np.abs(local_norm - global_sens)
        r = np.corrcoef(local_norm[valid], global_sens[valid])[0, 1] if valid.sum() > 2 else float('nan')
        print(f"  corr(local sensitivity-at-lock, global gradient sensitivity) = {r:.3f}")
        top_disagree = np.argsort(-np.where(valid, disagreement, -1))[:5]
        print(f"  Top-5 positions where local != global sensitivity: {top_disagree.tolist()}")
        print(f"    (their disagreement: {[round(float(disagreement[p]), 3) for p in top_disagree]})")
        if r < 0.5:
            print("  -> Meaningful local/global disagreement. Confirms sensitivity is genuinely "
                  "context-dependent here -- worth re-deriving sensitivity per-context rather "
                  "than trusting one global map for AL flagging decisions late in a search.")
        else:
            print("  -> Local and global sensitivity broadly agree. Context-dependence exists "
                  "(per the PCE interaction finding) but isn't large enough to change which "
                  "positions look important.")

        out = pd.DataFrame({
            'position': range(N_POS), 'locked': locked, 'lock_gen': lock_gen,
            'local_sens_at_lock_norm': local_norm, 'global_sens': global_sens,
            'disagreement': disagreement,
        })
        out.to_csv(f'{OUT_PREFIX}_partC_entropy_triggered.csv', index=False)
        print(f"[SAVED] {OUT_PREFIX}_partC_entropy_triggered.csv")

        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(global_sens[valid], local_norm[valid], alpha=0.7, s=40, color='#1B4FBF')
        for p in top_disagree:
            if valid[p]:
                ax.annotate(f'pos_{p}', (global_sens[p], local_norm[p]), fontsize=7)
        lims = [0, 1]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('Global sensitivity (cnn_v9_sens.csv)')
        ax.set_ylabel('Local sensitivity at lock-in context')
        ax.set_title(f'Local vs Global Sensitivity at Entropy Lock-In\nr={r:.3f}')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT_PREFIX}_partC_scatter.png', dpi=150)
        print(f"[SAVED] {OUT_PREFIX}_partC_scatter.png")