"""
=============================================================================
06_simple_ga_entropy_sensitivity.py
=============================================================================
Dr. Cammi asked for entropy <-> sensitivity <-> GA, tested on "a simple GA,
not your CNN". This is that experiment.

Why a synthetic test instead of QICA/CNN logs: QICA's real logs mix three
confounds you can't separate after the fact -- surrogate (CNN) error,
TensorFlow's known non-deterministic-CPU-op noise (the exact thing your own
v1-vs-v2 ablation caught earlier), and unaligned per-seed stagnation timing
when pooled. Here, the fitness function is hand-written, so "sensitivity"
per position is EXACTLY known (not gradient-approximated, not SHAP-
approximated) -- ground truth, not an estimate. This isolates the mechanism
cleanly. Once validated here, the same H_sens formula already living in
qica_v11_production.py is the thing you bring the result back to.

Design:
  - 31 positions, 9 categorical types per position (same shape as your
    real problem).
  - Each position p has a KNOWN sensitivity weight w_p (a few positions
    dominate, most barely matter -- mimicking your real sensitivity
    range 0.352-1.000 with a handful of top positions).
  - Each (position, type) pair has a fixed, known "goodness" value.
  - fitness(individual) = sum_p  w_p * goodness_p[type_p]   (+ optional
    pairwise interaction term, off by default -- see ADD_INTERACTIONS)
  - Plain elitist GA: tournament selection, uniform crossover, per-gene
    mutation. No quantum encoding, no CNN, no dropout -- deliberately boring.

Three things this script directly answers:
  1. Does population-wide, sensitivity-weighted entropy loss predict
     stagnation, in a controlled setting with the confounds removed?
  2. Does per-position entropy decay FASTER at genuinely high-sensitivity
     positions than at low-sensitivity ones? (the "rod position <-> entropy"
     question, now testable against ground truth instead of an approximation)
  3. Does entropy decay just before an improvement (population "homing in")
     or just before a stall (premature convergence)?

Run:  python 06_simple_ga_entropy_sensitivity.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
N_POS, N_TYPES = 31, 9
POP_SIZE       = 80
N_GENS         = 150
N_SEEDS        = 12          # independent GA runs -- averaging across these
                              # is what fixes the pooling problem from Section 1
TOURNAMENT_K   = 3
MUTATION_RATE  = 0.03         # per-gene probability of resampling to a new type
ELITE_FRAC     = 0.10
ADD_INTERACTIONS = True      # set True to add pairwise coupling (harder problem)
STAG_PATIENCE_EPS = 1e-6

OUT_PREFIX = 'simple_ga'


# =============================================================================
# GROUND-TRUTH PROBLEM DEFINITION (fixed once, known exactly -- this is the
# whole point: no approximation error on the "sensitivity" side of any test)
# =============================================================================
def build_ground_truth(seed=0):
    rs = np.random.default_rng(seed)
    # a handful of dominant positions, most positions nearly irrelevant --
    # mirrors your real gradient-sensitivity range (0.352-1.000, top5 dominate)
    raw = rs.pareto(a=2.0, size=N_POS) + 0.1
    w = raw / raw.max()                       # normalize to [~0.1, 1.0]
    w = np.clip(w, 0.05, 1.0)

    goodness = rs.uniform(0, 1, size=(N_POS, N_TYPES))  # fixed lookup table

    interaction = None
    if ADD_INTERACTIONS:
        # sparse pairwise coupling: a handful of position-pairs where the
        # BEST combined type differs from what either position alone wants
        # (mimics real neutron-coupling-between-adjacent-assemblies effects)
        n_pairs = 15
        pairs = rs.choice(N_POS, size=(n_pairs, 2), replace=True)
        interaction = {tuple(p): rs.uniform(-0.3, 0.3, size=(N_TYPES, N_TYPES))
                       for p in pairs if p[0] != p[1]}
    return w, goodness, interaction


def fitness_fn(pop, w, goodness, interaction=None):
    """pop: (N, 31) int in [0, N_TYPES-1]. Returns (N,) fitness, higher=better."""
    base = np.zeros(len(pop))
    for p in range(N_POS):
        base += w[p] * goodness[p, pop[:, p]]
    if interaction:
        for (p1, p2), tbl in interaction.items():
            base += tbl[pop[:, p1], pop[:, p2]]
    return base


# =============================================================================
# PLAIN ELITIST GA
# =============================================================================
def run_ga(w, goodness, interaction, seed):
    rs = np.random.default_rng(seed)
    pop = rs.integers(0, N_TYPES, size=(POP_SIZE, N_POS))
    fit = fitness_fn(pop, w, goodness, interaction)

    best_fit = fit.max()
    stag = 0
    log = []

    for gen in range(N_GENS):
        # --- per-position Shannon entropy of the CURRENT population ---
        pos_H = np.zeros(N_POS)
        for p in range(N_POS):
            counts = np.bincount(pop[:, p], minlength=N_TYPES).astype(np.float64)
            probs = counts / counts.sum()
            pos_H[p] = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))

        H_mean = pos_H.mean()
        H_sens = np.average(pos_H, weights=w)     # sensitivity-weighted, same
                                                     # formula spirit as qica_v11's H_sens

        log.append({'seed': seed, 'gen': gen, 'best_fit': best_fit, 'stag': stag,
                    'H_mean': H_mean, 'H_sens': H_sens,
                    **{f'H_pos_{p}': pos_H[p] for p in range(N_POS)}})

        # --- selection: tournament ---
        elite_n = max(1, int(POP_SIZE * ELITE_FRAC))
        order = np.argsort(-fit)
        elite = pop[order[:elite_n]]

        children = [pop[i].copy() for i in order[:elite_n]]  # elitism carries over
        while len(children) < POP_SIZE:
            def tournament():
                cand = rs.choice(POP_SIZE, TOURNAMENT_K, replace=False)
                return pop[cand[np.argmax(fit[cand])]]
            p1, p2 = tournament(), tournament()
            mask = rs.integers(0, 2, size=N_POS).astype(bool)
            child = np.where(mask, p1, p2)
            mut_mask = rs.random(N_POS) < MUTATION_RATE
            child[mut_mask] = rs.integers(0, N_TYPES, size=mut_mask.sum())
            children.append(child)
        pop = np.stack(children[:POP_SIZE])
        fit = fitness_fn(pop, w, goodness, interaction)

        if fit.max() > best_fit + STAG_PATIENCE_EPS:
            best_fit = fit.max()
            stag = 0
        else:
            stag += 1

    return pd.DataFrame(log), pos_H  # pos_H here = FINAL generation's per-position entropy


# =============================================================================
# RUN MULTIPLE SEEDS, ANALYZE PROPERLY (per-seed, then aggregate -- fixes the
# pooling problem that likely explains Section 1/3's near-zero correlations)
# =============================================================================
print("=" * 70)
print(f"SIMPLE GA  |  {N_SEEDS} seeds x {N_GENS} gens x pop={POP_SIZE}  |  "
      f"interactions={'ON' if ADD_INTERACTIONS else 'OFF'}")
print("=" * 70)

w, goodness, interaction = build_ground_truth(seed=0)
print(f"  Ground-truth sensitivity range: {w.min():.3f} - {w.max():.3f}")
print(f"  Top-5 ground-truth positions  : {np.argsort(w)[::-1][:5].tolist()}\n")

all_logs = []
final_entropy_by_seed = []
for s in range(N_SEEDS):
    log_df, final_H = run_ga(w, goodness, interaction, seed=1000 + s)
    all_logs.append(log_df)
    final_entropy_by_seed.append(final_H)
    print(f"  seed {s:2d}: best_fit={log_df.best_fit.iloc[-1]:.4f}  "
          f"final H_mean={log_df.H_mean.iloc[-1]:.3f}  "
          f"final H_sens={log_df.H_sens.iloc[-1]:.3f}")

full = pd.concat(all_logs, ignore_index=True)
full.to_csv(f'{OUT_PREFIX}_history.csv', index=False)

# ---- Test 1: H_sens vs stagnation, computed WITHIN each seed then averaged ----
per_seed_r_stag = []
per_seed_r_impr = []
for s, log_df in enumerate(all_logs):
    d_fit = log_df['best_fit'].diff().fillna(0.0).values
    h_drop = np.concatenate([[0.0], -np.diff(log_df['H_sens'].values)])
    if log_df['stag'].std() > 1e-9:
        per_seed_r_stag.append(np.corrcoef(log_df['H_sens'], log_df['stag'])[0, 1])
    if h_drop.std() > 1e-9 and d_fit.std() > 1e-9:
        per_seed_r_impr.append(np.corrcoef(h_drop, d_fit)[0, 1])

print("\n" + "=" * 70)
print("TEST 1 — H_sens vs stagnation (per-seed, then averaged -- not pooled)")
print("=" * 70)
print(f"  mean per-seed corr(H_sens, stagnation)          = {np.mean(per_seed_r_stag):+.3f}  "
      f"(std across seeds: {np.std(per_seed_r_stag):.3f}, n={len(per_seed_r_stag)})")
print(f"  mean per-seed corr(H_sens drop, fit improvement) = {np.mean(per_seed_r_impr):+.3f}  "
      f"(std across seeds: {np.std(per_seed_r_impr):.3f}, n={len(per_seed_r_impr)})")
print("  If |mean| is meaningfully bigger than the pooled QICA result (~0.09-0.10)")
print("  AND consistent in sign across seeds (small std relative to mean), that's")
print("  evidence the relationship IS real but was being washed out by pooling")
print("  seeds with unaligned stagnation timing in the QICA analysis -- i.e. the")
print("  mechanism works, the earlier test just measured it wrong.\n")

# ---- Test 2: does entropy decay FASTER at high-sensitivity positions? ----
# per-position entropy decay = H(gen 0) - H(final gen), averaged across seeds
H0 = np.zeros(N_POS)
Hfinal = np.zeros(N_POS)
for log_df in all_logs:
    g0 = log_df[log_df.gen == 0].iloc[0]
    gN = log_df[log_df.gen == N_GENS - 1].iloc[0]
    H0 += np.array([g0[f'H_pos_{p}'] for p in range(N_POS)])
    Hfinal += np.array([gN[f'H_pos_{p}'] for p in range(N_POS)])
H0 /= N_SEEDS
Hfinal /= N_SEEDS
decay = H0 - Hfinal  # positive = lost diversity

r_decay_w = np.corrcoef(decay, w)[0, 1]
print("=" * 70)
print("TEST 2 — Does per-position entropy decay track KNOWN ground-truth sensitivity?")
print("=" * 70)
print(f"  corr(entropy decay, ground-truth sensitivity w_p) = {r_decay_w:+.3f}")
if r_decay_w > 0.3:
    print("  -> POSITIVE and clear: high-sensitivity positions really do converge")
    print("     (lose diversity) faster than low-sensitivity ones, as expected --")
    print("     the search is correctly 'locking in' on what matters most.")
    print("     This validates the mechanism Dr. Cammi is asking about, in a setting")
    print("     with zero approximation error. The real-data near-zero result in")
    print("     Section 3 is then better read as 'the training data's sampling wasn't")
    print("     targeted at high-sensitivity positions' (a data problem) rather than")
    print("     'entropy and sensitivity are unrelated in general' (a mechanism problem).")
else:
    print("  -> Weak/no relationship even in this clean-room setting. This would be a")
    print("     genuinely more surprising finding -- would suggest entropy decay in an")
    print("     elitist GA is driven mostly by selection pressure and population size,")
    print("     not by which genes matter most for fitness. Worth double-checking with")
    print("     ADD_INTERACTIONS=True and/or a stronger elitism setting before concluding.")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].scatter(w, decay, alpha=0.7, s=40, color='#1B4FBF')
axes[0].set_xlabel('Ground-truth sensitivity w_p'); axes[0].set_ylabel('Entropy decay (H0 - Hfinal)')
axes[0].set_title(f'Test 2: Entropy Decay vs Sensitivity\nr={r_decay_w:.3f}')
axes[0].grid(alpha=0.3)

for log_df in all_logs[:6]:
    axes[1].plot(log_df.gen, log_df.H_sens, alpha=0.6, lw=1)
axes[1].set_xlabel('Generation'); axes[1].set_ylabel('H_sens (sensitivity-weighted entropy)')
axes[1].set_title('H_sens Trajectories (first 6 seeds)')
axes[1].grid(alpha=0.3)

for log_df in all_logs[:6]:
    axes[2].plot(log_df.gen, log_df.best_fit, alpha=0.6, lw=1)
axes[2].set_xlabel('Generation'); axes[2].set_ylabel('Best fitness')
axes[2].set_title('Fitness Convergence (first 6 seeds)')
axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_results.png', dpi=150)
print(f"\n[SAVED] {OUT_PREFIX}_history.csv  {OUT_PREFIX}_results.png")
print("\nNext: if Test 2 confirms the mechanism, re-run qica_v11_production.py's")
print("Section-1-style analysis but WITHIN each seed (not pooled) before concluding")
print("the real system shows no relationship -- the pooling artifact may be the")
print("actual explanation for the near-zero result you got on real QICA logs.")
