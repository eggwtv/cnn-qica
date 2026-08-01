"""
=============================================================================
07_pareto_nsga2_cnn.py
=============================================================================
Fixes the Section 4 weight-sweep problem: instead of 7 independent noisy
single-objective runs (only 1/7 landed on the non-dominated front last
time), this runs ONE proper multi-objective GA (NSGA-II style: Deb et al.
2002) directly against the CNN surrogate. Two real objectives, never
blended into one scalar:
    maximize cycle_length
    minimize ppf_max
Selection uses (non-dominated rank, crowding distance) -- crowding distance
IS the standard diversity-preservation mechanism your entropy machinery is
meant to be an alternative/complement to, so this script reports both side
by side on the final front.

Also includes a standalone GRID_LAYOUT diagonal-symmetry check (Part 2b of
the review doc) -- run this FIRST, it takes seconds and tells you whether
to trust any OpenMC PPF numbers you've generated against this geometry.

Run:  python 07_pareto_nsga2_cnn.py
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

np.random.seed(42)
rng = np.random.default_rng(42)

N_POS, N_TYPES = 31, 9
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)

MODEL_FILE  = 'cnn_v9_model.keras'
CONFIG_FILE = 'cnn_v9_config.json'
DATA_CSV    = 'ml_dataset_constrained.csv'
FREQ_FILE   = 'train_type_freq_v9.npy'
SENS_FILE   = 'cnn_v9_sens.csv'
OUT_PREFIX  = 'pareto_nsga2'


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


# =============================================================================
# PART A — GRID_LAYOUT diagonal-symmetry sanity check (run this first)
# =============================================================================
print("=" * 70)
print("PART A — Is GRID_LAYOUT's row/column adjacency the real physical layout?")
print("=" * 70)
print("""  Logic: if GRID_LAYOUT[r,c] really represents physical adjacency for a
  symmetric core wedge, then position (r,c) and its diagonal-mirror position
  (c,r) -- wherever BOTH are real (non-padding) cells -- should look
  statistically similar in the training data (similar type-entropy, similar
  sensitivity), because a genuine symmetric wedge is symmetric about its own
  diagonal by construction. If they don't, GRID_LAYOUT is very likely just
  an arbitrary reshape-for-CNN-convenience array, not real adjacency -- and
  reusing it to build OpenMC geometry would place assemblies next to the
  wrong physical neighbors, which would show up as exactly the kind of PPF
  magnitude/position mismatch you've been seeing against OpenMC.
""")

if has(FREQ_FILE) and has(SENS_FILE):
    type_freq = np.load(FREQ_FILE).astype(np.float64)
    pos_entropy = -np.sum(type_freq * np.log(type_freq + 1e-12), axis=1)
    sens_df = pd.read_csv(SENS_FILE)
    sens_norm = sens_df['sensitivity_norm'].values

    pos_id_grid = GRID_LAYOUT.copy()
    pairs = []
    for r in range(6):
        for c in range(6):
            if r >= c:
                continue  # only check each pair once, skip the diagonal itself
            a, b = pos_id_grid[r, c], pos_id_grid[c, r]
            if a >= 0 and b >= 0:
                pairs.append((a, b))

    if len(pairs) < 3:
        print(f"  Only {len(pairs)} valid mirror-pairs exist under this GRID_LAYOUT -- "
              f"not enough for a reliable check. (This itself is informative: a genuine\n"
              f"  symmetric wedge usually has many such pairs; very few suggests the array\n"
              f"  isn't laid out as a symmetric wedge at all.)")
    else:
        ent_a = np.array([pos_entropy[a] for a, b in pairs])
        ent_b = np.array([pos_entropy[b] for a, b in pairs])
        sen_a = np.array([sens_norm[a] for a, b in pairs])
        sen_b = np.array([sens_norm[b] for a, b in pairs])
        ent_mad = np.mean(np.abs(ent_a - ent_b))
        sen_mad = np.mean(np.abs(sen_a - sen_b))
        # compare against MAD between RANDOM (non-mirror) position pairs as a baseline
        rand_pairs = [(rng.integers(0, N_POS), rng.integers(0, N_POS)) for _ in range(200)]
        rand_ent_mad = np.mean([abs(pos_entropy[a] - pos_entropy[b]) for a, b in rand_pairs])
        rand_sen_mad = np.mean([abs(sens_norm[a] - sens_norm[b]) for a, b in rand_pairs])

        print(f"  {len(pairs)} diagonal-mirror pairs found under GRID_LAYOUT.")
        print(f"  Mean |entropy difference| within mirror pairs   : {ent_mad:.4f}")
        print(f"  Mean |entropy difference| within RANDOM pairs   : {rand_ent_mad:.4f}")
        print(f"  Mean |sensitivity difference| within mirror pairs: {sen_mad:.4f}")
        print(f"  Mean |sensitivity difference| within RANDOM pairs: {rand_sen_mad:.4f}")
        if ent_mad < 0.5 * rand_ent_mad and sen_mad < 0.5 * rand_sen_mad:
            print("\n  -> Mirror pairs are noticeably MORE similar than random pairs on both")
            print("     measures. Weak supporting evidence GRID_LAYOUT may reflect a real")
            print("     symmetric structure. Still worth a manual cross-check against the")
            print("     actual BEAVRS octant coordinate list before trusting it fully.")
        else:
            print("\n  -> Mirror pairs are NOT meaningfully more similar than random pairs.")
            print("     This is consistent with GRID_LAYOUT being an arbitrary index-packing")
            print("     array (built for CNN convenience) rather than real physical adjacency.")
            print("     STRONG recommendation: before trusting any OpenMC PPF number generated")
            print("     with this geometry, verify against the real BEAVRS/PARCS coordinate")
            print("     map used to generate ml_dataset_constrained.csv's 'loading_i' columns.")
else:
    print("  [SKIP] need train_type_freq_v9.npy and cnn_v9_sens.csv (both from cnn-v9.py)\n")


# =============================================================================
# PART B — NSGA-II style multi-objective GA on the CNN surrogate
# =============================================================================
print("\n" + "=" * 70)
print("PART B — Real Pareto front (NSGA-II style) on the CNN surrogate")
print("=" * 70)


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


if not (has(MODEL_FILE) and has(CONFIG_FILE)):
    print("  [SKIP] needs cnn_v9_model.keras + cnn_v9_config.json\n")
else:
    print("[LOAD] cnn_v9_model.keras ...")
    MODEL = keras.models.load_model(MODEL_FILE, compile=False)
    with open(CONFIG_FILE) as f:
        CFG = json.load(f)
    YM_MEAN, YM_SCALE = np.array(CFG['ym_scaler_mean'], np.float32), np.array(CFG['ym_scaler_scale'], np.float32)
    IDX_PPF, IDX_CYCLE = CFG['IDX_PPF_MAX'], CFG['IDX_CYCLE']

    def flat_to_grid(flat):
        g = np.zeros((flat.shape[0], 6, 6), dtype=np.int32)
        pi = 0
        for r in range(6):
            for c in range(6):
                if GRID_LAYOUT[r, c] >= 0:
                    g[:, r, c] = flat[:, pi]; pi += 1
        return g

    def evaluate(pop):
        Xg = tf.constant(flat_to_grid(pop), dtype=tf.int32)
        y = MODEL(Xg, training=False).numpy()
        ppf = y[:, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
        cyc = y[:, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
        return ppf, cyc

    # --- NSGA-II core machinery ---
    def fast_non_dominated_sort(ppf, cyc):
        """minimize ppf, maximize cyc. Returns list of fronts (arrays of indices)."""
        n = len(ppf)
        S = [[] for _ in range(n)]
        dom_count = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        fronts = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                p_better = (ppf[p] <= ppf[q] and cyc[p] >= cyc[q]) and (ppf[p] < ppf[q] or cyc[p] > cyc[q])
                q_better = (ppf[q] <= ppf[p] and cyc[q] >= cyc[p]) and (ppf[q] < ppf[p] or cyc[q] > cyc[p])
                if p_better:
                    S[p].append(q)
                elif q_better:
                    dom_count[p] += 1
            if dom_count[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return [np.array(f) for f in fronts if len(f) > 0], rank

    def crowding_distance(ppf, cyc, front_idx):
        n = len(front_idx)
        if n == 0:
            return np.array([])
        dist = np.zeros(n)
        for vals, minimize in [(ppf[front_idx], True), (cyc[front_idx], False)]:
            order = np.argsort(vals)
            dist[order[0]] = dist[order[-1]] = np.inf
            span = vals[order[-1]] - vals[order[0]] + 1e-9
            for k in range(1, n - 1):
                dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span
        return dist

    def tournament_select(pop, rank, cdist, k=2):
        idx = rng.integers(0, len(pop), size=k)
        best = idx[0]
        for i in idx[1:]:
            if rank[i] < rank[best] or (rank[i] == rank[best] and cdist[i] > cdist[best]):
                best = i
        return pop[best]

    POP_SIZE, N_GENS = 100, 80
    warm_pool = None
    if has(DATA_CSV):
        df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
        lc = [f'loading_{i}' for i in range(N_POS)]
        if all(c in df.columns for c in lc):
            warm_pool = df[lc].values.astype(np.int32)

    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rng.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rng.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)

    print(f"  Running NSGA-II: pop={POP_SIZE}, gens={N_GENS} (CNN-only, no OpenMC) ...")
    front1_size_hist = []
    for gen in range(N_GENS):
        ppf, cyc = evaluate(pop)
        fronts, rank = fast_non_dominated_sort(ppf, cyc)
        cdist = np.zeros(len(pop))
        for f in fronts:
            cdist[f] = crowding_distance(ppf, cyc, f)
        front1_size_hist.append(len(fronts[0]))

        # offspring via tournament + uniform crossover + mutation
        children = []
        while len(children) < POP_SIZE:
            p1 = tournament_select(pop, rank, cdist)
            p2 = tournament_select(pop, rank, cdist)
            mask = rng.integers(0, 2, size=N_POS).astype(bool)
            child = np.where(mask, p1, p2).astype(np.int32)
            mut_mask = rng.random(N_POS) < 0.04
            child[mut_mask] = rng.integers(1, N_TYPES + 1, size=mut_mask.sum())
            children.append(child)
        offspring = np.stack(children)

        # combine parent + offspring, re-sort, keep best POP_SIZE (elitist NSGA-II)
        combined = np.concatenate([pop, offspring], axis=0)
        c_ppf, c_cyc = evaluate(combined)
        c_fronts, c_rank = fast_non_dominated_sort(c_ppf, c_cyc)
        new_pop_idx = []
        for f in c_fronts:
            if len(new_pop_idx) + len(f) <= POP_SIZE:
                new_pop_idx.extend(f.tolist())
            else:
                cd = crowding_distance(c_ppf, c_cyc, f)
                order = f[np.argsort(-cd)]
                remaining = POP_SIZE - len(new_pop_idx)
                new_pop_idx.extend(order[:remaining].tolist())
                break
        pop = combined[new_pop_idx]

        if (gen + 1) % 20 == 0:
            ppf_now, cyc_now = evaluate(pop)
            print(f"    gen {gen+1:3d}/{N_GENS} | front-1 size={front1_size_hist[-1]:3d} | "
                  f"ppf range=[{ppf_now.min():.3f},{ppf_now.max():.3f}] | "
                  f"cycle range=[{cyc_now.min():.1f},{cyc_now.max():.1f}]d")

    # final front
    final_ppf, final_cyc = evaluate(pop)
    final_fronts, final_rank = fast_non_dominated_sort(final_ppf, final_cyc)
    front1 = final_fronts[0]
    cd_front1 = crowding_distance(final_ppf, final_cyc, front1)

    # entropy-based spread on the front (your actual entropy<->Pareto test, done right)
    front_pop = pop[front1]
    counts = np.stack([(front_pop == t).mean(axis=0) for t in range(1, N_TYPES + 1)], axis=1)
    pos_H_front = -np.sum(counts * np.log(counts + 1e-12), axis=1)
    mean_H_front = pos_H_front.mean()

    print(f"\n  Final Pareto front: {len(front1)} non-dominated solutions "
          f"(out of {POP_SIZE} final population)")
    print(f"  PPF range on front   : {final_ppf[front1].min():.3f} - {final_ppf[front1].max():.3f}")
    print(f"  Cycle range on front : {final_cyc[front1].min():.1f} - {final_cyc[front1].max():.1f} days")
    print(f"  Mean crowding distance on front (excl. inf boundary points): "
          f"{np.mean(cd_front1[np.isfinite(cd_front1)]):.4f}")
    print(f"  Mean per-position Shannon entropy across front population   : {mean_H_front:.4f} nats")
    print("  -> If crowding distance and entropy both stay high across the whole run, the")
    print("     front is genuinely diverse (many meaningfully different solutions along the")
    print("     tradeoff). If crowding distance is high but entropy is low, the front spans")
    print("     a wide PPF/cycle range using only small type-perturbations of one core design")
    print("     -- objective-space diversity without genotype diversity, worth knowing either way.")

    front_df = pd.DataFrame({
        'ppf': final_ppf[front1], 'cycle': final_cyc[front1], 'crowding_distance': cd_front1,
        **{f'pos_{p}': front_pop[:, p] for p in range(N_POS)}
    }).sort_values('ppf')
    front_df.to_csv(f'{OUT_PREFIX}_front.csv', index=False)
    print(f"\n[SAVED] {OUT_PREFIX}_front.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dom_mask = np.ones(len(pop), dtype=bool); dom_mask[front1] = False
    axes[0].scatter(final_ppf[dom_mask], final_cyc[dom_mask], color='grey', s=25, alpha=0.5, label='dominated')
    axes[0].scatter(final_ppf[front1], final_cyc[front1], color='#D62728', s=50, zorder=5, label='Pareto front')
    axes[0].set_xlabel('PPF_max'); axes[0].set_ylabel('Cycle length (days)')
    axes[0].set_title(f'True Pareto Front (NSGA-II)\n{len(front1)} non-dominated solutions')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(front1_size_hist, color='#1B4FBF')
    axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Front-1 size')
    axes[1].set_title('Front Size Over Generations\n(growing/stable = healthy diversity)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_pareto_front.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_pareto_front.png\n")
