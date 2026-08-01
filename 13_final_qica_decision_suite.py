"""
=============================================================================
13_final_qica_decision_suite.py
=============================================================================
The "decide the final QICA config before moving to Active Learning" script.
Reads GRID_ROWS/GRID_COLS/GRID_LAYOUT straight from the CNN's own config
JSON (not hardcoded) so it automatically adapts whether you point it at
cnn_v9_config.json (old 6x6) or cnn_v10_config.json (new 6x8) -- no more
shape-mismatch errors from a stale hardcoded grid.

Five components, all against the SAME CNN:
  1. Real Pareto front (NSGA-II, 2 objectives: min PPF, max cycle)
  2. NSGA-II with entropy as an explicit 3rd objective (max genotype spread)
  3. Plain GA + uncertainty penalty + gradient-sensitivity-weighted mutation
  4. Plain GA + uncertainty penalty + Sobol-weighted mutation
  5. DMD-as-forecaster, retried with eigenvalue stabilization (clip |eig|<=1
     before extrapolating -- fixes the specific blowup that caused the
     298-MAE failure last time)

Every PPF result gets stamped against a DYNAMICALLY computed training floor
(5th percentile of this CNN's own predictions across the full training set
-- not a hardcoded number from a different model) and compared to whichever
QICA baseline CSV you point it at.

REQUIRES a fresh QICA baseline CSV generated against the SAME model you
point this script at (see round4_results_and_plan.md Section 4 for why this
isn't optional). If QICA_SUMMARY doesn't exist, the script still runs
everything and reports results -- it just skips the "verdict vs QICA" table
and tells you to go generate one.

Run:  python 13_final_qica_decision_suite.py
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

# =============================================================================
# CONFIG — point these at whichever CNN you've validated as current
# =============================================================================
MODEL_FILE   = 'cnn_v10_model.keras'
CONFIG_FILE  = 'cnn_v10_config.json'
DATA_CSV     = 'ml_dataset_constrained.csv'
SENS_FILE    = 'cnn_v10_sens.csv'
SOBOL_FILE   = 'mentor_feedback_pce_sobol.csv'   # from earlier PCE run; if it
                                                    # doesn't exist for this CNN
                                                    # version, the 'sobol'
                                                    # component is skipped
QICA_SUMMARY = 'qica_v11_summary.csv'              # MUST be regenerated
                                                    # against MODEL_FILE above
                                                    # before this means anything
OUT_PREFIX   = 'final_suite'

N_TYPES = 9
SEEDS = [42, 137, 271, 509, 1023]
NOISE_FLOOR = 0.02

# GA settings (components 3 & 4)
POP_SIZE, N_GENS = 80, 250
TOURNAMENT_K = 3
BASE_MUTATION_RATE = 0.03
ELITE_FRAC = 0.10
W_UNCERTAINTY = 40.0
MC_SAMPLES_FOR_PENALTY = 10

# NSGA-II settings (components 1 & 2)
NSGA_POP, NSGA_GENS = 100, 80

# DMD settings (component 5)
DMD_N_TEST = 400
DMD_RANK = 4


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


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
    print("Need the CNN model + config. Exiting.")
    raise SystemExit

print(f"[LOAD] {MODEL_FILE} ...")
MODEL = keras.models.load_model(MODEL_FILE, compile=False)
with open(CONFIG_FILE) as f:
    CFG = json.load(f)

# Grid geometry read DYNAMICALLY from config -- this is what fixes the
# shape-mismatch error you hit: no more hardcoded 6x6 assumption anywhere.
GRID_ROWS = CFG['GRID_ROWS']
GRID_COLS = CFG['GRID_COLS']
GRID_LAYOUT = np.array(CFG['GRID_LAYOUT'], dtype=np.int32)
N_POS = CFG['N_POS']
YM_MEAN = np.array(CFG['ym_scaler_mean'], dtype=np.float32)
YM_SCALE = np.array(CFG['ym_scaler_scale'], dtype=np.float32)
IDX_PPF, IDX_CYCLE = CFG['IDX_PPF_MAX'], CFG['IDX_CYCLE']
print(f"  Grid geometry from config: {GRID_ROWS}x{GRID_COLS}, N_POS={N_POS}")


def flat_to_grid(flat):
    g = np.zeros((flat.shape[0], GRID_ROWS, GRID_COLS), dtype=np.int32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = flat[:, GRID_LAYOUT[r, c]]
    return g


def evaluate(pop, mc_samples=0):
    Xg = tf.constant(flat_to_grid(pop), dtype=tf.int32)
    if mc_samples <= 1:
        y = MODEL(Xg, training=False).numpy()
        ppf = y[:, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
        cyc = y[:, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
        return ppf, cyc, np.zeros_like(ppf)
    stack = np.stack([MODEL(Xg, training=True).numpy() for _ in range(mc_samples)])
    ppf_s = stack[:, :, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
    cyc_s = stack[:, :, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
    return ppf_s.mean(axis=0), cyc_s.mean(axis=0), ppf_s.std(axis=0)


warm_pool = None
df_full = None
if has(DATA_CSV):
    df_full = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    if all(c in df_full.columns for c in load_cols):
        warm_pool = df_full[load_cols].values.astype(np.int32)

# --- dynamic training floor: 5th percentile of THIS model's own predictions
# over the full training set (not a number carried over from a different CNN) ---
if warm_pool is not None:
    full_ppf_pred, _, _ = evaluate(warm_pool)
    TRAINING_FLOOR = float(np.percentile(full_ppf_pred, 5))
    print(f"  Dynamic training floor (5th percentile of {MODEL_FILE}'s own "
          f"predictions): {TRAINING_FLOOR:.4f}")
else:
    TRAINING_FLOOR = 1.697
    print(f"  [WARN] no dataset found -- falling back to a hardcoded floor "
          f"({TRAINING_FLOOR}) from a DIFFERENT model. Treat OOD flags with extra caution.")

grad_sens_weights, sobol_weights = None, None
if has(SENS_FILE):
    grad_sens_weights = pd.read_csv(SENS_FILE)['sensitivity_norm'].values
if has(SOBOL_FILE):
    sdf = pd.read_csv(SOBOL_FILE)
    col = 'sobol_first_order_norm' if 'sobol_first_order_norm' in sdf.columns else 'sobol_first_order'
    if len(sdf) == N_POS:
        sobol_weights = sdf[col].values
        sobol_weights = sobol_weights / (sobol_weights.max() + 1e-9)
    else:
        print(f"  [WARN] {SOBOL_FILE} has {len(sdf)} rows, expected {N_POS} -- "
              f"was computed for a different N_POS/model. Skipping Sobol component.")

qica_baseline = None
if has(QICA_SUMMARY):
    qsum = pd.read_csv(QICA_SUMMARY)
    ppf_col = next((c for c in ['best_ppf', 'ppf', 'ppf_max'] if c in qsum.columns), None)
    if ppf_col:
        qica_baseline = (qsum[ppf_col].values.mean(), qsum[ppf_col].values.std())
        print(f"  QICA baseline loaded: {qica_baseline[0]:.4f} +/- {qica_baseline[1]:.4f} "
              f"(n={len(qsum)}) -- MAKE SURE this was generated against {MODEL_FILE}, not a "
              f"different CNN version, or this comparison is invalid.")
if qica_baseline is None:
    print(f"  [WARN] {QICA_SUMMARY} not found or unreadable. Every component below will still "
          f"run and report its own numbers, but the 'vs QICA' verdict table will be skipped.")
    print(f"  -> Regenerate it: run qica_v11_production.py pointed at {MODEL_FILE}/{CONFIG_FILE}, "
          f"USE_CYCLE_FITNESS=False, before trusting any comparison here.")


all_results = {}   # name -> dict(ppf_mean, ppf_std, cyc_mean, cyc_std, n_below_floor, n_seeds)


def report(name, ppf_arr, cyc_arr):
    n_below = int((ppf_arr < TRAINING_FLOOR).sum())
    all_results[name] = dict(ppf_mean=ppf_arr.mean(), ppf_std=ppf_arr.std(),
                              cyc_mean=cyc_arr.mean(), cyc_std=cyc_arr.std(),
                              n_below_floor=n_below, n_seeds=len(ppf_arr))
    print(f"  -- {name}: ppf={ppf_arr.mean():.4f}+/-{ppf_arr.std():.4f}  "
          f"cycle={cyc_arr.mean():.1f}+/-{cyc_arr.std():.1f}d  "
          f"({n_below}/{len(ppf_arr)} below training floor {TRAINING_FLOOR:.3f})")


# =============================================================================
# COMPONENT 1 & 2 — NSGA-II, with and without entropy as a 3rd objective
# =============================================================================
print("\n" + "=" * 70)
print("COMPONENT 1 & 2 — NSGA-II Pareto front (2-objective vs 3-objective+entropy)")
print("=" * 70)


def population_entropy(pop):
    counts = np.stack([(pop == t).mean(axis=0) for t in range(1, N_TYPES + 1)], axis=1)
    return -np.sum(counts * np.log(counts + 1e-12), axis=1).mean()


def dominates(obj_a, obj_b, minimize_flags):
    """obj_a, obj_b: tuples of objective values. minimize_flags: bool per objective."""
    better_or_equal, strictly_better = True, False
    for a, b, mn in zip(obj_a, obj_b, minimize_flags):
        if mn:
            if a > b: better_or_equal = False
            if a < b: strictly_better = True
        else:
            if a < b: better_or_equal = False
            if a > b: strictly_better = True
    return better_or_equal and strictly_better


def fast_non_dominated_sort(objectives, minimize_flags):
    n = len(objectives)
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if dominates(objectives[p], objectives[q], minimize_flags):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p], minimize_flags):
                dom_count[p] += 1
        if dom_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [np.array(fr) for fr in fronts if len(fr) > 0], rank


def crowding_distance(objectives, front_idx, minimize_flags):
    n = len(front_idx)
    if n == 0:
        return np.array([])
    dist = np.zeros(n)
    n_obj = len(minimize_flags)
    vals_by_obj = [np.array([objectives[i][k] for i in front_idx]) for k in range(n_obj)]
    for vals in vals_by_obj:
        order = np.argsort(vals)
        dist[order[0]] = dist[order[-1]] = np.inf
        span = vals[order[-1]] - vals[order[0]] + 1e-9
        for k in range(1, n - 1):
            dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span
    return dist


def run_nsga2(use_entropy_objective):
    if warm_pool is not None and len(warm_pool) >= NSGA_POP:
        pop = warm_pool[rng.choice(len(warm_pool), NSGA_POP, replace=False)].copy()
    else:
        pop = rng.integers(1, N_TYPES + 1, size=(NSGA_POP, N_POS)).astype(np.int32)

    minimize_flags = [True, False] + ([False] if use_entropy_objective else [])

    def obj_tuples(p):
        ppf, cyc, _ = evaluate(p)
        if use_entropy_objective:
            # per-individual "entropy contribution" proxy: hamming distance
            # to the population mean pattern (rewards individuals that are
            # genuinely different, not just the population-wide entropy,
            # which isn't a per-individual quantity NSGA-II can select on)
            mode_per_pos = np.array([np.bincount(p[:, i], minlength=N_TYPES + 1).argmax()
                                      for i in range(N_POS)])
            distinctness = (p != mode_per_pos[None, :]).sum(axis=1).astype(np.float64)
            return list(zip(ppf, cyc, distinctness))
        return list(zip(ppf, cyc))

    objectives = obj_tuples(pop)
    for gen in range(NSGA_GENS):
        fronts, rank = fast_non_dominated_sort(objectives, minimize_flags)
        cdist = np.zeros(len(pop))
        for fr in fronts:
            cdist[fr] = crowding_distance(objectives, fr, minimize_flags)

        children = []
        while len(children) < NSGA_POP:
            def tournament():
                cand = rng.choice(len(pop), 2, replace=False)
                i, j = cand
                if rank[i] < rank[j] or (rank[i] == rank[j] and cdist[i] > cdist[j]):
                    return pop[i]
                return pop[j]
            p1, p2 = tournament(), tournament()
            mask = rng.integers(0, 2, size=N_POS).astype(bool)
            child = np.where(mask, p1, p2).astype(np.int32)
            mut_mask = rng.random(N_POS) < 0.04
            child[mut_mask] = rng.integers(1, N_TYPES + 1, size=mut_mask.sum())
            children.append(child)
        offspring = np.stack(children)

        combined = np.concatenate([pop, offspring], axis=0)
        combined_obj = obj_tuples(combined)
        c_fronts, c_rank = fast_non_dominated_sort(combined_obj, minimize_flags)
        new_idx = []
        for fr in c_fronts:
            if len(new_idx) + len(fr) <= NSGA_POP:
                new_idx.extend(fr.tolist())
            else:
                cd = crowding_distance(combined_obj, fr, minimize_flags)
                order = fr[np.argsort(-cd)]
                new_idx.extend(order[:NSGA_POP - len(new_idx)].tolist())
                break
        pop = combined[new_idx]
        objectives = [combined_obj[i] for i in new_idx]

    final_ppf, final_cyc, _ = evaluate(pop)
    fronts, rank = fast_non_dominated_sort(objectives, minimize_flags)
    front1 = fronts[0]
    front_pop = pop[front1]
    H = population_entropy(front_pop)
    return dict(ppf=final_ppf[front1], cyc=final_cyc[front1],
                n_front=len(front1), entropy=H, pop=front_pop)


nsga_2obj = run_nsga2(use_entropy_objective=False)
print(f"  2-objective front: {nsga_2obj['n_front']} solutions, "
      f"PPF [{nsga_2obj['ppf'].min():.3f}-{nsga_2obj['ppf'].max():.3f}], "
      f"cycle [{nsga_2obj['cyc'].min():.1f}-{nsga_2obj['cyc'].max():.1f}]d, "
      f"entropy={nsga_2obj['entropy']:.3f} nats")

nsga_3obj = run_nsga2(use_entropy_objective=True)
print(f"  3-objective(+entropy) front: {nsga_3obj['n_front']} solutions, "
      f"PPF [{nsga_3obj['ppf'].min():.3f}-{nsga_3obj['ppf'].max():.3f}], "
      f"cycle [{nsga_3obj['cyc'].min():.1f}-{nsga_3obj['cyc'].max():.1f}]d, "
      f"entropy={nsga_3obj['entropy']:.3f} nats")
print(f"\n  Entropy went {nsga_2obj['entropy']:.3f} -> {nsga_3obj['entropy']:.3f} nats "
      f"when made an explicit objective "
      f"({'more diverse designs, as intended' if nsga_3obj['entropy'] > nsga_2obj['entropy'] else 'no real change -- entropy objective may need a stronger weight/selection pressure'}).")

report('nsga2_2obj (best-PPF point on front)', np.array([nsga_2obj['ppf'].min()]),
       np.array([nsga_2obj['cyc'][np.argmin(nsga_2obj['ppf'])]]))
report('nsga2_3obj_entropy (best-PPF point on front)', np.array([nsga_3obj['ppf'].min()]),
       np.array([nsga_3obj['cyc'][np.argmin(nsga_3obj['ppf'])]]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(nsga_2obj['ppf'], nsga_2obj['cyc'], label='2-objective', alpha=0.7, s=40)
axes[0].scatter(nsga_3obj['ppf'], nsga_3obj['cyc'], label='3-objective (+entropy)', alpha=0.7, s=40)
axes[0].set_xlabel('PPF_max'); axes[0].set_ylabel('Cycle (days)')
axes[0].set_title('Pareto Fronts: 2-obj vs 3-obj(+entropy)'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(['2-objective', '3-objective\n(+entropy)'], [nsga_2obj['entropy'], nsga_3obj['entropy']],
            color=['#888888', '#1B4FBF'])
axes[1].set_ylabel('Mean per-position entropy (nats)')
axes[1].set_title('Genotype Diversity on the Front')
axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_nsga2_comparison.png', dpi=150)
print(f"[SAVED] {OUT_PREFIX}_nsga2_comparison.png")


# =============================================================================
# COMPONENT 3 & 4 — GA + uncertainty penalty + (gradient / Sobol) mutation
# =============================================================================
print("\n" + "=" * 70)
print("COMPONENT 3 & 4 — GA + uncertainty penalty, combined with sensitivity-weighted mutation")
print("=" * 70)


def mutation_rates_for(weights, base_rate=BASE_MUTATION_RATE):
    if weights is None:
        return np.full(N_POS, base_rate)
    r = base_rate * (1.5 - weights)
    return np.clip(r, 0.005, 0.15)


def run_ga_combo(seed, weights):
    rs = np.random.default_rng(seed)
    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rs.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)

    mut_rates = mutation_rates_for(weights)
    ppf, cyc, ppf_std = evaluate(pop, mc_samples=MC_SAMPLES_FOR_PENALTY)
    fit = -ppf - W_UNCERTAINTY * ppf_std
    best = {'ppf': ppf[np.argmax(fit)], 'cycle': cyc[np.argmax(fit)]}

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
        ppf, cyc, ppf_std = evaluate(pop, mc_samples=MC_SAMPLES_FOR_PENALTY)
        fit = -ppf - W_UNCERTAINTY * ppf_std
        gi = np.argmax(fit)
        if ppf[gi] < best['ppf']:
            best = {'ppf': ppf[gi], 'cycle': cyc[gi]}
    return best


for name, weights in [('unc_penalty+grad_sens', grad_sens_weights),
                       ('unc_penalty+sobol', sobol_weights)]:
    if weights is None:
        print(f"  [SKIP] {name} -- weighting source not available")
        continue
    print(f"\n--- {name} ---")
    bests = [run_ga_combo(s, weights) for s in SEEDS]
    ppf_arr = np.array([b['ppf'] for b in bests])
    cyc_arr = np.array([b['cycle'] for b in bests])
    for s, b in zip(SEEDS, bests):
        flag = "  [BELOW FLOOR]" if b['ppf'] < TRAINING_FLOOR else ""
        print(f"    seed {s:5d}: ppf={b['ppf']:.4f}  cycle={b['cycle']:.1f}d{flag}")
    report(name, ppf_arr, cyc_arr)


# =============================================================================
# COMPONENT 5 — DMD-as-forecaster, retried with eigenvalue stabilization
# =============================================================================
print("\n" + "=" * 70)
print("COMPONENT 5 — DMD-as-forecaster, retry with eigenvalue stabilization")
print("=" * 70)

if df_full is not None:
    ppf_steps_avail = sorted(set(int(c.split('_')[1][1:]) for c in df_full.columns if c.startswith('ppf_')))
    ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df_full.columns if c.startswith('ppf_')))
    ppf_steps_avail = [s for s in ppf_steps_avail
                       if len([c for c in df_full.columns if c.startswith(f"ppf_s{s}_")]) == len(ppf_assembs)]
    n_steps = len(ppf_steps_avail)

    if n_steps > 4:
        idx = rng.choice(len(df_full), min(DMD_N_TEST, len(df_full)), replace=False)
        dfs = df_full.iloc[idx].reset_index(drop=True)
        curves = np.zeros((len(dfs), n_steps), dtype=np.float64)
        for si, s in enumerate(ppf_steps_avail):
            cols = [f'ppf_s{s}_a{a}' for a in ppf_assembs if f'ppf_s{s}_a{a}' in df_full.columns]
            curves[:, si] = dfs[cols].values.astype(np.float64).max(axis=1)

        split = n_steps // 2

        def dmd_forecast_stabilized(curve_1d, split, rank, stabilize=True):
            known = curve_1d[:split]
            n_future = len(curve_1d) - split
            d = max(2, min(rank + 2, split // 2))
            if split - d < 2:
                return np.full(n_future, known[-1])
            H = np.array([known[i:i + d] for i in range(split - d + 1)]).T
            X1, X2 = H[:, :-1], H[:, 1:]
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            r = min(rank, len(S))
            Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
            Atilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / Sr)
            eigvals, W = np.linalg.eig(Atilde)
            if stabilize:
                # THE FIX: clip any eigenvalue magnitude > 1 down to exactly 1.
                # A correctly-behaved burnup curve shouldn't have modes that
                # grow without bound; this removes the specific mechanism
                # (mode^15 blowup) that produced the MAE=298 failure, without
                # being able to make an already-decaying mode diverge.
                mags = np.abs(eigvals)
                scale = np.where(mags > 1.0, 1.0 / mags, 1.0)
                eigvals = eigvals * scale
            Phi = X2 @ Vhr.conj().T @ np.diag(1.0 / Sr) @ W
            b, *_ = np.linalg.lstsq(Phi, H[:, -1], rcond=None)
            forecast = []
            for t in range(1, n_future + 1):
                powers = eigvals ** t
                new_state = (Phi @ (b * powers)).real
                forecast.append(new_state[-1])
            return np.array(forecast)

        dmd_err_unstab, dmd_err_stab, persist_err = [], [], []
        for i in range(len(dfs)):
            curve = curves[i]
            true_future = curve[split:]
            pred_unstab = dmd_forecast_stabilized(curve, split, DMD_RANK, stabilize=False)
            pred_stab   = dmd_forecast_stabilized(curve, split, DMD_RANK, stabilize=True)
            persist_pred = np.full(len(true_future), curve[split - 1])
            dmd_err_unstab.append(np.mean(np.abs(pred_unstab - true_future)))
            dmd_err_stab.append(np.mean(np.abs(pred_stab - true_future)))
            persist_err.append(np.mean(np.abs(persist_pred - true_future)))

        dmd_err_unstab = np.array(dmd_err_unstab)
        dmd_err_stab = np.array(dmd_err_stab)
        persist_err = np.array(persist_err)

        print(f"  DMD (unstabilized, same as before) MAE : {dmd_err_unstab.mean():.4f}")
        print(f"  DMD (eigenvalue-stabilized)         MAE : {dmd_err_stab.mean():.4f}")
        print(f"  Persistence baseline                MAE : {persist_err.mean():.4f}")
        win_rate = float((dmd_err_stab < persist_err).mean())
        print(f"  Stabilized DMD beats persistence on {win_rate*100:.1f}% of patterns")
        if dmd_err_stab.mean() < persist_err.mean():
            print("  -> Stabilization fixed it: DMD-as-forecaster is now genuinely competitive.")
            print("     Worth keeping as a cheap, model-free cross-check on CNN curve predictions.")
        else:
            print("  -> Even stabilized, DMD-as-forecaster does not beat naive persistence.")
            print("     This is now a well-evidenced final answer, not an implementation bug:")
            print("     across FOUR separate attempts (diagnostic role, dropout-free variant,")
            print("     partial correlation, and now a stabilized forecaster), DMD has not earned")
            print("     a place in this pipeline. Safe to close this line of investigation for")
            print("     good and note it as a genuine negative result in your writeup.")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.hist(dmd_err_unstab, bins=30, alpha=0.5, label='DMD (unstabilized)', color='#D62728')
        ax.hist(dmd_err_stab, bins=30, alpha=0.5, label='DMD (stabilized)', color='#1B4FBF')
        ax.hist(persist_err, bins=30, alpha=0.5, label='Persistence', color='#2CA02C')
        ax.set_xlabel('Forecast MAE'); ax.set_ylabel('Count')
        ax.set_title('DMD Forecast: Before/After Eigenvalue Stabilization')
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, max(1.0, np.percentile(dmd_err_stab, 95) * 1.5))
        plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_dmd_stabilized.png', dpi=150)
        print(f"[SAVED] {OUT_PREFIX}_dmd_stabilized.png")
    else:
        print("  [SKIP] not enough burnup steps in dataset for this test.")
else:
    print("  [SKIP] no dataset available.")


# =============================================================================
# FINAL VERDICT TABLE
# =============================================================================
print("\n" + "=" * 70)
print("FINAL VERDICT — everything, compared, one table")
print("=" * 70)

rows = []
for name, r in all_results.items():
    row = {'component': name, 'ppf_mean': r['ppf_mean'], 'ppf_std': r['ppf_std'],
           'cycle_mean': r['cyc_mean'], 'n_below_floor': r['n_below_floor'], 'n_seeds': r['n_seeds']}
    if qica_baseline is not None:
        delta = qica_baseline[0] - r['ppf_mean']
        row['delta_vs_qica'] = delta
        row['verdict'] = ("within noise floor" if abs(delta) < NOISE_FLOOR
                           else ("beats QICA" if delta > 0 else "QICA wins"))
    rows.append(row)

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(f'{OUT_PREFIX}_final_verdict.csv', index=False)
print(f"\n[SAVED] {OUT_PREFIX}_final_verdict.csv")
print("\nRule of thumb for picking your final config: prefer the option with the BEST")
print("ppf_mean among those with n_below_floor == 0 (i.e. don't let an OOD 'win' win the")
print("argument) that also beats or matches QICA's baseline beyond the noise floor.")