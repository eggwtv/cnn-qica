"""
=============================================================================
12_ga_variants_vs_matched_baseline.py
=============================================================================
Follow-up to 10_simple_ga_vs_qica_baseline.py, fixing two things per
round3_results_explained.md:

  1. USES THE OLD 6x6 GRID_LAYOUT for every CNN call. cnn_v9_model.keras has
     a hard-coded (6,6) input shape -- it was trained under the old,
     physically-arbitrary layout, and cannot accept the new 6x8 octant
     layout without retraining. The 6x8 layout is for OpenMC geometry only
     (see openmc_beavrs_vver1000_v5_FIXED.py) -- do NOT import/reuse it here.

  2. Compares against the CORRECT matched QICA baseline
     (cycle_fitness=False: best_ppf=1.7534+/-0.0139), not the
     cycle_fitness=True one used by mistake last time (2.089+/-0.19 --
     a different objective, not a fair comparison).

Adds, per round3_results_explained.md Section 6:
  - TRAINING_FLOOR flag: every reported best-PPF gets stamped with whether
    it's below the CNN's documented training-data floor (~1.697). A "win"
    below the floor is very likely surrogate exploitation, not a real
    design -- see the Section-5 explanation of why your last plain-GA
    "win" is probably this, not a genuine improvement over QICA.
  - variant 'uncertainty_penalty': plain GA + QICA's MC-dropout ppf_std
    penalty term, WITHOUT quantum encoding or trust region. Isolates
    whether the uncertainty penalty alone (not QICA's other machinery)
    is what keeps QICA out of the OOD region.
  - variant 'grad_sens': gradient-sensitivity-weighted mutation (protects
    high-sensitivity positions once population commits to them).
  - variant 'sobol': same idea, weighted by PCE Sobol indices instead
    (model-independent -- from mentor_feedback_pce_sobol.csv if present).

Run:  python 12_ga_variants_vs_matched_baseline.py
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
SOBOL_FILE   = 'mentor_feedback_pce_sobol.csv'
OUT_PREFIX   = 'ga_variants'

N_POS, N_TYPES = 31, 9
# OLD 6x6 layout -- matches cnn_v9_model.keras's trained input shape (6,6).
# Do NOT swap in the new 6x8 octant layout here; see module docstring.
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)

SEEDS = [42, 137, 271, 509, 1023]
POP_SIZE, N_GENS = 80, 250
TOURNAMENT_K = 3
BASE_MUTATION_RATE = 0.03
ELITE_FRAC = 0.10

# Correct, matched QICA baseline (cycle_fitness=False), from your own logs.
QICA_BASELINE_PPF_MEAN = 1.7534
QICA_BASELINE_PPF_STD  = 0.0139
QICA_BASELINE_CYCLE_MEAN = 313.6   # from your v10 E_combined run
NOISE_FLOOR = 0.02

# Training-data floor -- below this, trust the CNN's own prediction much
# less (it's extrapolating). From your own QICA logs:
# "Warm-start pool: 50 lowest-CNN-predicted-PPF patterns (range 1.697-1.815)"
TRAINING_FLOOR = 1.697

# Uncertainty-penalty variant settings (matches QICA's own weighting)
W_UNCERTAINTY = 40.0
MC_SAMPLES_FOR_PENALTY = 10   # lower than QICA's 25 -- keeps this affordable;
                                # note in your writeup if you bump this up later


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
    print("Need cnn_v9_model.keras + cnn_v9_config.json. Exiting.")
    raise SystemExit

print("[LOAD] cnn_v9_model.keras (expects (6,6) input -- old layout, correct here) ...")
MODEL = keras.models.load_model(MODEL_FILE, compile=False)
with open(CONFIG_FILE) as f:
    CFG = json.load(f)
YM_MEAN = np.array(CFG['ym_scaler_mean'], dtype=np.float32)
YM_SCALE = np.array(CFG['ym_scaler_scale'], dtype=np.float32)
IDX_PPF, IDX_CYCLE = CFG['IDX_PPF_MAX'], CFG['IDX_CYCLE']


def flat_to_grid(flat):
    g = np.zeros((flat.shape[0], 6, 6), dtype=np.int32)
    pi = 0
    for r in range(6):
        for c in range(6):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = flat[:, pi]; pi += 1
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
if has(DATA_CSV):
    df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    if all(c in df.columns for c in load_cols):
        warm_pool = df[load_cols].values.astype(np.int32)

grad_sens_weights, sobol_weights = None, None
if has(SENS_FILE):
    grad_sens_weights = pd.read_csv(SENS_FILE)['sensitivity_norm'].values
if has(SOBOL_FILE):
    sdf = pd.read_csv(SOBOL_FILE)
    col = 'sobol_first_order_norm' if 'sobol_first_order_norm' in sdf.columns else 'sobol_first_order'
    sobol_weights = sdf[col].values
    sobol_weights = sobol_weights / (sobol_weights.max() + 1e-9)


def mutation_rates_for(variant, base_rate=BASE_MUTATION_RATE):
    if variant == 'grad_sens' and grad_sens_weights is not None:
        r = base_rate * (1.5 - grad_sens_weights)
        return np.clip(r, 0.005, 0.15)
    if variant == 'sobol' and sobol_weights is not None:
        r = base_rate * (1.5 - sobol_weights)
        return np.clip(r, 0.005, 0.15)
    return np.full(N_POS, base_rate)


def run_ga(seed, variant):
    """variant in {'flat', 'grad_sens', 'sobol', 'uncertainty_penalty'}."""
    rs = np.random.default_rng(seed)
    if warm_pool is not None and len(warm_pool) >= POP_SIZE:
        pop = warm_pool[rs.choice(len(warm_pool), POP_SIZE, replace=False)].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(POP_SIZE, N_POS)).astype(np.int32)

    use_penalty = (variant == 'uncertainty_penalty')
    mc = MC_SAMPLES_FOR_PENALTY if use_penalty else 0
    mut_rates = mutation_rates_for(variant)

    ppf, cyc, ppf_std = evaluate(pop, mc_samples=mc)
    fit = (-ppf - W_UNCERTAINTY * ppf_std) if use_penalty else (-ppf)
    best_i = np.argmax(fit)
    best = {'ppf': ppf[best_i], 'cycle': cyc[best_i]}

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
        ppf, cyc, ppf_std = evaluate(pop, mc_samples=mc)
        fit = (-ppf - W_UNCERTAINTY * ppf_std) if use_penalty else (-ppf)
        gi = np.argmax(fit)
        cand_ppf = ppf[gi]
        # track best by RAW ppf (not fitness, which may include penalty) so
        # comparisons across variants are on the same footing
        if cand_ppf < best['ppf']:
            best = {'ppf': cand_ppf, 'cycle': cyc[gi]}

    return best


VARIANTS = ['flat']
if grad_sens_weights is not None:
    VARIANTS.append('grad_sens')
if sobol_weights is not None:
    VARIANTS.append('sobol')
VARIANTS.append('uncertainty_penalty')

print("=" * 70)
print(f"Running variants {VARIANTS} at {len(SEEDS)} seeds x {N_GENS} gens x pop={POP_SIZE}")
print(f"(uncertainty_penalty uses mc_samples={MC_SAMPLES_FOR_PENALTY} per generation -- slower)")
print("=" * 70)

results = {}
for variant in VARIANTS:
    print(f"\n--- variant: {variant} ---")
    bests = []
    for s in SEEDS:
        b = run_ga(s, variant)
        bests.append(b)
        below_floor = b['ppf'] < TRAINING_FLOOR
        flag = "  [BELOW TRAINING FLOOR -- likely surrogate exploitation, not a real design]" if below_floor else ""
        print(f"  seed {s:5d}: best_ppf={b['ppf']:.4f}  best_cycle={b['cycle']:.1f}d{flag}")
    ppf_arr = np.array([b['ppf'] for b in bests])
    cyc_arr = np.array([b['cycle'] for b in bests])
    n_below = int((ppf_arr < TRAINING_FLOOR).sum())
    results[variant] = dict(ppf_mean=ppf_arr.mean(), ppf_std=ppf_arr.std(),
                             cyc_mean=cyc_arr.mean(), cyc_std=cyc_arr.std(),
                             n_below_floor=n_below, ppf_arr=ppf_arr)
    print(f"  -- {variant}: best_ppf = {ppf_arr.mean():.4f} +/- {ppf_arr.std():.4f}  "
          f"[{ppf_arr.min():.4f} - {ppf_arr.max():.4f}]  "
          f"({n_below}/{len(SEEDS)} seeds below training floor {TRAINING_FLOOR})")


# =============================================================================
# Compare all variants against the CORRECT matched QICA baseline
# =============================================================================
print("\n" + "=" * 70)
print(f"VERDICT vs matched QICA baseline (cycle_fitness=False): "
      f"best_ppf = {QICA_BASELINE_PPF_MEAN:.4f} +/- {QICA_BASELINE_PPF_STD:.4f}")
print(f"Noise floor = {NOISE_FLOOR} PPF. Training floor = {TRAINING_FLOOR} "
      f"(below this, treat any 'win' with suspicion, not celebration).")
print("=" * 70)

summary_rows = []
for variant, r in results.items():
    delta = QICA_BASELINE_PPF_MEAN - r['ppf_mean']
    if abs(delta) < NOISE_FLOOR:
        verdict = "WITHIN NOISE FLOOR vs QICA"
    elif delta > 0:
        verdict = "beats QICA (lower PPF), clears noise floor"
    else:
        verdict = "QICA beats this variant, clears noise floor"
    ood_note = (f" -- {r['n_below_floor']}/{len(SEEDS)} seeds went BELOW the training floor; "
                f"treat this 'win' as likely surrogate exploitation, not a validated design"
                if r['n_below_floor'] > 0 else " -- stayed within the training-supported region")
    print(f"  {variant:20s}: ppf={r['ppf_mean']:.4f}+/-{r['ppf_std']:.4f}  "
          f"delta(QICA-variant)={delta:+.4f}  -> {verdict}{ood_note}")
    summary_rows.append({'variant': variant, 'ppf_mean': r['ppf_mean'], 'ppf_std': r['ppf_std'],
                          'cycle_mean': r['cyc_mean'], 'cycle_std': r['cyc_std'],
                          'delta_vs_qica': delta, 'n_below_training_floor': r['n_below_floor']})

pd.DataFrame(summary_rows).to_csv(f'{OUT_PREFIX}_summary.csv', index=False)

print("\nKey question this answers: does 'uncertainty_penalty' (plain GA + QICA's MC-dropout")
print("sigma penalty, nothing else from QICA) land ABOVE the training floor while 'flat' does")
print("not? If so, that's direct evidence the uncertainty penalty specifically -- not the")
print("quantum encoding or trust region -- is what keeps QICA out of the surrogate's blind spot.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(results.keys())
means = [results[v]['ppf_mean'] for v in names]
stds = [results[v]['ppf_std'] for v in names]
colors = ['#D62728' if results[v]['n_below_floor'] > 0 else '#2CA02C' for v in names]
axes[0].bar(names, means, yerr=stds, color=colors, capsize=4)
axes[0].axhline(QICA_BASELINE_PPF_MEAN, color='black', ls='--', lw=1.5, label='QICA baseline')
axes[0].axhline(TRAINING_FLOOR, color='orange', ls=':', lw=1.5, label='training floor')
axes[0].set_ylabel('best PPF (mean +/- std)')
axes[0].set_title('GA Variants vs Matched QICA Baseline\n(red bars = went below training floor)')
axes[0].legend(fontsize=8); axes[0].tick_params(axis='x', rotation=20)

for v in names:
    axes[1].scatter([v] * len(results[v]['ppf_arr']), results[v]['ppf_arr'],
                     alpha=0.7, s=40)
axes[1].axhline(TRAINING_FLOOR, color='orange', ls=':', lw=1.5, label='training floor')
axes[1].set_ylabel('best PPF per seed'); axes[1].set_title('Per-Seed Spread')
axes[1].legend(fontsize=8); axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_comparison.png', dpi=150)
print(f"\n[SAVED] {OUT_PREFIX}_summary.csv  {OUT_PREFIX}_comparison.png")