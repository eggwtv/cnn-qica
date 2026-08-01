"""
=============================================================================
05_feedback_tests.py
=============================================================================
Follow-up to 04_dmd_signal_validation.py, and a direct response to
Dr. Cammi's feedback after your presentation. Six independent tests:

  SECTION 1 — Entropy <-> GA search <-> sensitivity
      Does sensitivity-weighted entropy loss (H_sens, already logged by
      qica_v11_production.py every generation) predict search stagnation
      better than raw population diversity? Regression on qica_v11_history.csv.

  SECTION 2 — Polynomial Chaos Expansion (PCE) global sensitivity
      A THIRD, model-independent ranking of "which of the 31 positions
      matter most for PPF_max" -- built directly from the training CSV,
      not from the CNN. Compared against gradient sensitivity (cnn_v9_sens.csv)
      and SHAP (qica_v11_shap.csv) if present.

  SECTION 3 — Rod/assembly position <-> entropy
      Per-position training-data entropy (already computed as your Trust
      Region logic) plotted spatially and correlated against sensitivity.
      Tests whether QICA's most influential positions are also the ones
      your training data explored the least.

  SECTION 4 — Pareto front <-> entropy
      A lightweight weight-sweep over W_PPF_SOFT (cheap CNN-only QICA
      restarts, no OpenMC) to approximate a real Pareto front instead of
      one blended-fitness point, plus an entropy-based spread/crowding
      metric across the sweep (population-diversity analogue of NSGA-II
      crowding distance).

  SECTION 5 — DMD WITHOUT MC dropout
      Replaces the MC-dropout uncertainty proxy with a deterministic
      finite-difference input-perturbation ensemble (no stochastic
      forward passes), and re-checks whether DMD reconstruction error
      still tracks *that* uncertainty proxy.

  SECTION 6 — Partial correlation (the actually-informative next check
      flagged at the end of 04_dmd_signal_validation.py)
      Does DMD error predict real CNN error AFTER controlling for MC
      sigma? If the partial correlation is ~0, DMD error is just a more
      expensive, redundant proxy for sigma you already compute. If it's
      still nontrivial, that's real evidence of complementary signal.

Needs (same dir, as many as are available -- each section degrades
gracefully and prints what it skipped and why):
  ml_dataset_constrained.csv, cnn_v9_model.keras, cnn_v9_config.json,
  cnn_v9_sens.csv, train_type_freq_v9.npy,
  qica_v11_summary.csv, qica_v11_history.csv, qica_v11_shap.csv (optional)

Run:  python 05_mentor_feedback_tests.py
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

# =============================================================================
# FILE NAMES (edit if yours differ)
# =============================================================================
DATA_CSV     = 'ml_dataset_constrained.csv'
MODEL_FILE   = 'cnn_v9_model.keras'
CONFIG_FILE  = 'cnn_v9_config.json'
SENS_FILE    = 'cnn_v9_sens.csv'
FREQ_FILE    = 'train_type_freq_v9.npy'
HISTORY_FILE = 'qica_v11_history.csv'
SUMMARY_FILE = 'qica_v11_summary.csv'
SHAP_FILE    = 'qica_v11_shap.csv'

OUT_PREFIX = 'mentor_feedback'
N_POS, N_TYPES = 31, 9
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)


def has(f):
    ok = os.path.exists(f)
    if not ok:
        print(f"  [SKIP] {f} not found")
    return ok


# =============================================================================
# SECTION 0 — shared: load CNN + config + ConvResBlock (needed by 2, 4, 5, 6)
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


MODEL, CFG = None, None
if has(MODEL_FILE) and has(CONFIG_FILE):
    print("[LOAD] cnn_v9_model.keras ...")
    MODEL = keras.models.load_model(MODEL_FILE, compile=False)
    with open(CONFIG_FILE) as f:
        CFG = json.load(f)
    YM_MEAN = np.array(CFG['ym_scaler_mean'], dtype=np.float32)
    YM_SCALE = np.array(CFG['ym_scaler_scale'], dtype=np.float32)
    IDX_PPF = CFG['IDX_PPF_MAX']
    IDX_CYCLE = CFG['IDX_CYCLE']


def flat_to_grid(flat):
    """flat: (N, 31) -> grid: (N, 6, 6)"""
    g = np.zeros((flat.shape[0], 6, 6), dtype=np.int32)
    pi = 0
    for r in range(6):
        for c in range(6):
            if GRID_LAYOUT[r, c] >= 0:
                g[:, r, c] = flat[:, pi]
                pi += 1
    return g


def cnn_predict_ppf_cycle(flat_patterns, mc_samples=0):
    """flat_patterns: (N, 31) int in [1,9]. Returns (ppf_mean, ppf_sigma, cycle_mean)."""
    Xg = tf.constant(flat_to_grid(flat_patterns), dtype=tf.int32)
    if mc_samples <= 1:
        y = MODEL(Xg, training=False).numpy()
        ppf = y[:, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
        cyc = y[:, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
        return ppf, np.zeros_like(ppf), cyc
    stack = np.stack([MODEL(Xg, training=True).numpy() for _ in range(mc_samples)])
    ppf_s = stack[:, :, IDX_PPF] * YM_SCALE[IDX_PPF] + YM_MEAN[IDX_PPF]
    cyc_s = stack[:, :, IDX_CYCLE] * YM_SCALE[IDX_CYCLE] + YM_MEAN[IDX_CYCLE]
    return ppf_s.mean(axis=0), ppf_s.std(axis=0), cyc_s.mean(axis=0)


# =============================================================================
# SECTION 1 — Entropy <-> GA search <-> sensitivity
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 1 — Does sensitivity-weighted entropy (H_sens) predict stagnation?")
print("=" * 70)

if has(HISTORY_FILE):
    hist = pd.read_csv(HISTORY_FILE)
    needed = {'gen', 'h_sens_pop', 'stag', 'best_ppf'}
    if needed.issubset(hist.columns):
        # per-seed: correlate H_sens level against how long stagnation persists
        # and against generation-over-generation PPF improvement.
        hist = hist.sort_values(['seed', 'gen']) if 'seed' in hist.columns else hist.sort_values('gen')
        group_col = 'seed' if 'seed' in hist.columns else None
        rows = []
        groups = hist.groupby(group_col) if group_col else [(None, hist)]
        for seed, g in groups:
            g = g.reset_index(drop=True)
            d_ppf = -g['best_ppf'].diff().fillna(0.0).values   # positive = improvement
            h = g['h_sens_pop'].values
            stag = g['stag'].values
            # entropy DROP over generations (loss of diversity, sensitivity-weighted)
            h_drop = np.concatenate([[0.0], -np.diff(h)])
            rows.append(pd.DataFrame({
                'seed': seed, 'gen': g['gen'], 'h_sens_pop': h, 'h_sens_drop': h_drop,
                'stag': stag, 'ppf_improvement': d_ppf
            }))
        long_df = pd.concat(rows, ignore_index=True)

        r_h_stag   = np.corrcoef(long_df['h_sens_pop'], long_df['stag'])[0, 1]
        r_hdrop_imp = np.corrcoef(long_df['h_sens_drop'], long_df['ppf_improvement'])[0, 1]
        print(f"  corr(H_sens level, stagnation counter)         : {r_h_stag:.4f}")
        print(f"  corr(H_sens drop this gen, PPF improvement)    : {r_hdrop_imp:.4f}")
        print("  Interpretation: a negative corr(H_sens, stag) means entropy tends to be")
        print("  LOWER during long stagnation stretches -- i.e. diversity loss precedes/")
        print("  co-occurs with stalling, which is the claim this test is checking.")
        print("  A positive corr(H_sens_drop, ppf_improvement) would mean fast entropy")
        print("  loss right before an improvement (the population is 'homing in'),")
        print("  vs a negative one meaning entropy loss precedes stalling, not progress.\n")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(long_df['h_sens_pop'], long_df['stag'], alpha=0.15, s=8, color='#1B4FBF')
        axes[0].set_xlabel('H_sens (population, sensitivity-weighted entropy)')
        axes[0].set_ylabel('Stagnation counter')
        axes[0].set_title(f'H_sens vs Stagnation\nr={r_h_stag:.3f}')
        axes[0].grid(alpha=0.3)

        axes[1].scatter(long_df['h_sens_drop'], long_df['ppf_improvement'], alpha=0.15, s=8, color='#D62728')
        axes[1].set_xlabel('H_sens drop this generation')
        axes[1].set_ylabel('PPF improvement this generation')
        axes[1].set_title(f'Entropy Drop vs Improvement\nr={r_hdrop_imp:.3f}')
        axes[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_entropy_vs_stagnation.png', dpi=150)
        print(f"[SAVED] {OUT_PREFIX}_entropy_vs_stagnation.png\n")
    else:
        print(f"  [SKIP] {HISTORY_FILE} missing columns {needed - set(hist.columns)}\n")
else:
    print("  Run qica_v11_production.py first to produce qica_v11_history.csv.\n")


# =============================================================================
# SECTION 2 — PCE-based global sensitivity (Sobol indices), model-independent
# =============================================================================
print("=" * 70)
print("SECTION 2 — PCE global sensitivity (Sobol indices) vs gradient sensitivity vs SHAP")
print("=" * 70)


def fit_pce_sobol(X_int, y, n_types=N_TYPES, degree=2):
    """
    Minimal from-scratch PCE for CATEGORICAL inputs (assembly types 1..n_types).

    Since inputs are discrete/categorical (not continuous uniform/normal),
    we build an orthonormal polynomial basis empirically per position from
    the *observed* discrete distribution at that position (this is the
    "arbitrary polynomial chaos" / data-driven PCE approach -- see
    Oladyshkin & Nowak 2012, and Skarbeli & Alvarez Velarde 2020 for the
    fuel-cycle-specific precedent).

    Returns: sobol_first_order (N_POS,) -- fraction of output variance
    attributable to each position, summing to <=1 (total-order truncated
    at `degree`, so some higher-order interaction variance is left out).
    """
    n, n_pos = X_int.shape
    y = y.astype(np.float64)
    y_mean, y_var = y.mean(), y.var()
    if y_var < 1e-12:
        return np.zeros(n_pos)

    # Build per-position empirical orthonormal polynomials (degree 0..`degree`)
    # via Gram-Schmidt on the discrete empirical measure at that position.
    def basis_for_position(col):
        vals = col.astype(np.float64)
        # start from raw monomials 1, x, x^2, ... and orthonormalize under
        # the empirical measure (equivalent to arbitrary/data-driven PCE)
        basis = [np.ones_like(vals)]
        for d in range(1, degree + 1):
            v = vals ** d
            for b in basis:
                proj = np.mean(v * b) / (np.mean(b * b) + 1e-12)
                v = v - proj * b
            norm = np.sqrt(np.mean(v * v) + 1e-12)
            basis.append(v / norm)
        return np.stack(basis[1:], axis=1)  # drop the constant term, (n, degree)

    # design matrix: for each position, its (degree) orthonormal polynomial
    # features; least-squares fit of y onto the union (first-order / additive
    # PCE truncation -- ignores cross-position interaction terms, which is
    # the standard first pass before adding interaction terms if needed)
    feats = []
    slices = []
    cursor = 0
    for p in range(n_pos):
        b = basis_for_position(X_int[:, p])
        feats.append(b)
        slices.append((cursor, cursor + b.shape[1]))
        cursor += b.shape[1]
    Phi = np.concatenate(feats, axis=1)  # (n, n_pos*degree)

    # ridge-regularized least squares (small ridge for numerical stability
    # given many near-collinear low-order polynomial terms)
    lam = 1e-6 * n
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b_vec = Phi.T @ (y - y_mean)
    coeffs, *_ = np.linalg.lstsq(A, b_vec, rcond=None)

    # first-order Sobol index per position = variance explained by that
    # position's own coefficients / total y variance (PCE orthonormality
    # makes this a direct sum of squared coefficients, Sudret 2008 eq. 10-12)
    sobol = np.zeros(n_pos)
    y_hat_var_total = 0.0
    for p, (lo, hi) in enumerate(slices):
        var_p = np.sum(coeffs[lo:hi] ** 2)
        sobol[p] = var_p
        y_hat_var_total += var_p
    sobol = sobol / (y_var + 1e-12)
    return sobol, y_hat_var_total / (y_var + 1e-12)


if has(DATA_CSV):
    df = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    if all(c in df.columns for c in load_cols):
        ppf_steps_avail = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
        ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))
        step_cols_max = None
        ppf_global_max = None
        if ppf_steps_avail and ppf_assembs:
            step_max = np.stack([
                df[[f'ppf_s{s}_a{a}' for a in ppf_assembs if f'ppf_s{s}_a{a}' in df.columns]]
                .values.astype(np.float32).max(axis=1)
                for s in ppf_steps_avail
            ], axis=1)
            ppf_global_max = step_max.max(axis=1)

        if ppf_global_max is not None:
            X_int = df[load_cols].values.astype(np.int32)
            print(f"  Fitting data-driven PCE (degree=2, additive/first-order truncation) "
                  f"on {len(df)} patterns ...")
            sobol_idx, explained_frac = fit_pce_sobol(X_int, ppf_global_max, degree=2)
            print(f"  PCE additive model explains {explained_frac*100:.1f}% of PPF_max variance "
                  f"(remainder is higher-order interaction + noise)")
            pce_top5 = np.argsort(sobol_idx)[::-1][:5]
            print(f"  PCE (Sobol first-order) top-5 positions : {pce_top5.tolist()}")

            pce_df = pd.DataFrame({
                'position': [f'pos_{i}' for i in range(N_POS)],
                'sobol_first_order': sobol_idx,
                'sobol_first_order_norm': sobol_idx / (sobol_idx.max() + 1e-9),
            })

            # cross-check against gradient sensitivity and SHAP if available
            compare_cols = {}
            if has(SENS_FILE):
                sens_df = pd.read_csv(SENS_FILE)
                grad_norm = sens_df['sensitivity_norm'].values
                compare_cols['gradient_sensitivity_norm'] = grad_norm
                grad_top5 = np.argsort(grad_norm)[::-1][:5]
                overlap_grad = len(set(pce_top5.tolist()) & set(grad_top5.tolist()))
                r_pce_grad = np.corrcoef(sobol_idx, grad_norm)[0, 1]
                print(f"  Gradient-sensitivity top-5               : {grad_top5.tolist()}")
                print(f"  PCE vs gradient-sensitivity: top-5 overlap = {overlap_grad}/5, "
                      f"rank correlation r = {r_pce_grad:.3f}")

            if has(SHAP_FILE):
                shap_df = pd.read_csv(SHAP_FILE)
                shap_cols = [c for c in shap_df.columns if c.endswith('_shap')]
                if shap_cols:
                    mean_abs_shap = shap_df[shap_cols].abs().mean().values
                    compare_cols['mean_abs_shap'] = mean_abs_shap
                    shap_top5 = np.argsort(mean_abs_shap)[::-1][:5]
                    overlap_shap = len(set(pce_top5.tolist()) & set(shap_top5.tolist()))
                    r_pce_shap = np.corrcoef(sobol_idx, mean_abs_shap)[0, 1]
                    print(f"  SHAP top-5                                : {shap_top5.tolist()}")
                    print(f"  PCE vs SHAP: top-5 overlap = {overlap_shap}/5, "
                          f"rank correlation r = {r_pce_shap:.3f}")

            for k, v in compare_cols.items():
                pce_df[k] = v
            pce_df.to_csv(f'{OUT_PREFIX}_pce_sobol.csv', index=False)
            print(f"\n  [SAVED] {OUT_PREFIX}_pce_sobol.csv")
            print("  -> Three independent rankings ('which positions matter') now exist:")
            print("     gradient sensitivity (local, CNN-derived), SHAP (local, CNN-derived),")
            print("     and PCE Sobol (global, data-derived, model-independent). High overlap")
            print("     across all three is strong evidence the CNN learned real physics there;")
            print("     disagreement flags positions worth a human look before trusting AL picks.\n")

            fig, ax = plt.subplots(figsize=(10, 5))
            order = np.argsort(sobol_idx)[::-1]
            ax.bar(range(N_POS), sobol_idx[order], color='#2CA02C')
            ax.set_xticks(range(N_POS))
            ax.set_xticklabels([f'{p}' for p in order], fontsize=7, rotation=90)
            ax.set_xlabel('Position (sorted by PCE Sobol index)')
            ax.set_ylabel('First-order Sobol index (fraction of PPF_max variance)')
            ax.set_title(f'PCE Global Sensitivity\n(additive model explains {explained_frac*100:.1f}% of variance)')
            ax.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_pce_sobol.png', dpi=150)
            print(f"[SAVED] {OUT_PREFIX}_pce_sobol.png\n")
        else:
            print("  [SKIP] could not reconstruct ppf_max from ppf_s*_a* columns.\n")
    else:
        print(f"  [SKIP] {DATA_CSV} missing loading_0..loading_30 columns.\n")


# =============================================================================
# SECTION 3 — Rod/assembly position <-> entropy (spatial)
# =============================================================================
print("=" * 70)
print("SECTION 3 — Per-position training entropy vs sensitivity (spatial)")
print("=" * 70)

if has(FREQ_FILE) and has(SENS_FILE):
    type_freq = np.load(FREQ_FILE).astype(np.float64)  # (N_POS, N_TYPES)
    pos_entropy = -np.sum(type_freq * np.log(type_freq + 1e-12), axis=1)  # nats
    sens_df = pd.read_csv(SENS_FILE)
    sens_norm = sens_df['sensitivity_norm'].values

    r_ent_sens = np.corrcoef(pos_entropy, sens_norm)[0, 1]
    print(f"  Per-position entropy range     : {pos_entropy.min():.3f} - {pos_entropy.max():.3f} nats "
          f"(max possible = {np.log(N_TYPES):.3f})")
    print(f"  corr(position entropy, sensitivity) = {r_ent_sens:.4f}")
    if r_ent_sens < -0.2:
        print("  -> NEGATIVE: your most influential (high-sensitivity) positions tend to be")
        print("     the ones your training data explored LEAST. This is a coverage gap --")
        print("     a concrete, quantified argument for prioritizing these positions in AL.")
    elif r_ent_sens > 0.2:
        print("  -> POSITIVE: high-sensitivity positions were already well-explored in")
        print("     training. Reassuring -- the CNN had real signal where it matters most.")
    else:
        print("  -> Weak/no relationship: entropy and sensitivity are largely independent")
        print("     here, i.e. the Trust Region entropy gate and the sensitivity map are")
        print("     capturing genuinely different information (which is fine -- it just")
        print("     means neither one can substitute for the other).")

    # spatial heatmaps, side by side, same 6x6 layout as cnn-v9.py's sensitivity plot
    disp_ent = np.full((6, 6), np.nan)
    disp_sen = np.full((6, 6), np.nan)
    pi = 0
    for r in range(6):
        for c in range(6):
            if GRID_LAYOUT[r, c] >= 0:
                disp_ent[r, c] = pos_entropy[pi]
                disp_sen[r, c] = sens_norm[pi]
                pi += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    cmap = plt.cm.viridis.copy(); cmap.set_bad('lightgrey')
    im0 = axes[0].imshow(disp_ent, cmap=cmap, aspect='auto')
    plt.colorbar(im0, ax=axes[0], label='Entropy (nats)')
    axes[0].set_title('Per-position training entropy'); axes[0].set_xticks([]); axes[0].set_yticks([])
    cmap2 = plt.cm.RdYlGn_r.copy(); cmap2.set_bad('lightgrey')
    im1 = axes[1].imshow(disp_sen, cmap=cmap2, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[1], label='Norm. sensitivity')
    axes[1].set_title(f'Gradient sensitivity\ncorr with entropy: r={r_ent_sens:.3f}')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_entropy_sensitivity_map.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_entropy_sensitivity_map.png\n")
else:
    print(f"  [SKIP] need both {FREQ_FILE} and {SENS_FILE} (both produced by cnn-v9.py)\n")


# =============================================================================
# SECTION 4 — Pareto front <-> entropy (weight sweep + entropy-based spread)
# =============================================================================
print("=" * 70)
print("SECTION 4 — Pareto weight sweep (W_PPF_SOFT) + entropy-based front spread")
print("=" * 70)

W_SWEEP = [1.0, 3.0, 6.0, 9.0, 12.0, 18.0, 26.0]
SWEEP_POP = 60
SWEEP_GENS = 40   # cheap: CNN-only, no OpenMC, just enough to see the tradeoff shape


def cheap_qica_point(w_ppf_soft, warm_pool=None, seed=0):
    """
    Minimal single-objective hill-climber (not the full quantum-ICA machinery
    -- this is intentionally cheap, purely to sample one point on the
    PPF/cycle tradeoff curve per weight, fast). Uses the CNN directly.
    Returns best (ppf, cycle, fitness, final-population-entropy).
    """
    rs = np.random.default_rng(seed)
    if warm_pool is not None and len(warm_pool) >= SWEEP_POP:
        idx = rs.choice(len(warm_pool), SWEEP_POP, replace=False)
        pop = warm_pool[idx].copy()
    else:
        pop = rs.integers(1, N_TYPES + 1, size=(SWEEP_POP, N_POS)).astype(np.int32)

    ppf, _, cyc = cnn_predict_ppf_cycle(pop)
    fitness = cyc - w_ppf_soft * ppf
    best_i = np.argmax(fitness)
    best = (ppf[best_i], cyc[best_i], fitness[best_i])

    for gen in range(SWEEP_GENS):
        order = np.argsort(-fitness)
        elite = pop[order[:SWEEP_POP // 4]]
        children = []
        for _ in range(SWEEP_POP):
            parent = elite[rs.integers(0, len(elite))].copy()
            n_mut = rs.integers(1, 6)
            mut_pos = rs.choice(N_POS, n_mut, replace=False)
            parent[mut_pos] = rs.integers(1, N_TYPES + 1, size=n_mut)
            children.append(parent)
        pop = np.stack(children)
        ppf, _, cyc = cnn_predict_ppf_cycle(pop)
        fitness = cyc - w_ppf_soft * ppf
        gi = np.argmax(fitness)
        if fitness[gi] > best[2]:
            best = (ppf[gi], cyc[gi], fitness[gi])

    # final population entropy (per-position, sensitivity-agnostic here --
    # this is population SPREAD along the front, the diversity-preservation
    # quantity relevant to Pareto crowding, not the trust-region entropy)
    counts = np.stack([(pop == t).mean(axis=0) for t in range(1, N_TYPES + 1)], axis=1)
    pos_H = -np.sum(counts * np.log(counts + 1e-12), axis=1)
    return best[0], best[1], best[2], float(pos_H.mean())


if MODEL is not None:
    warm_pool = None
    if has(DATA_CSV):
        df2 = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
        lc = [f'loading_{i}' for i in range(N_POS)]
        if all(c in df2.columns for c in lc):
            warm_pool = df2[lc].values.astype(np.int32)

    print(f"  Sweeping W_PPF_SOFT over {W_SWEEP} ({SWEEP_POP} pop x {SWEEP_GENS} gens, CNN-only) ...")
    sweep_rows = []
    for w in W_SWEEP:
        p, c, fit, h = cheap_qica_point(w, warm_pool, seed=int(w * 100) + 1)
        sweep_rows.append({'W_PPF_SOFT': w, 'ppf': p, 'cycle': c, 'fitness': fit, 'pop_entropy': h})
        print(f"    W={w:5.1f} -> ppf={p:.3f}  cycle={c:.1f}d  pop_entropy={h:.3f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f'{OUT_PREFIX}_pareto_sweep.csv', index=False)

    # non-dominated filter (true Pareto front among the swept points)
    is_dom = np.zeros(len(sweep_df), dtype=bool)
    for i in range(len(sweep_df)):
        for j in range(len(sweep_df)):
            if i == j:
                continue
            # j dominates i if j has >= cycle and <= ppf, with at least one strict
            if (sweep_df.cycle[j] >= sweep_df.cycle[i] and sweep_df.ppf[j] <= sweep_df.ppf[i]
                    and (sweep_df.cycle[j] > sweep_df.cycle[i] or sweep_df.ppf[j] < sweep_df.ppf[i])):
                is_dom[i] = True
                break
    sweep_df['on_pareto_front'] = ~is_dom
    n_front = int((~is_dom).sum())
    print(f"\n  {n_front}/{len(sweep_df)} swept weights landed on the non-dominated front.")
    print("  Entropy-spread metric (population diversity at each weight) tells you whether")
    print("  the search is exploring a genuinely different region at each weight, or")
    print("  collapsing to similar solutions regardless of W_PPF_SOFT (in which case the")
    print("  'front' is really just noise around one solution, not real tradeoff diversity).")
    corr_w_h = np.corrcoef(sweep_df['W_PPF_SOFT'], sweep_df['pop_entropy'])[0, 1]
    print(f"  corr(W_PPF_SOFT, final population entropy) = {corr_w_h:.3f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    front = sweep_df[sweep_df.on_pareto_front]
    dom = sweep_df[~sweep_df.on_pareto_front]
    axes[0].scatter(dom.ppf, dom.cycle, color='grey', s=40, label='dominated')
    axes[0].scatter(front.ppf, front.cycle, color='#D62728', s=60, zorder=5, label='Pareto front')
    for _, row in sweep_df.iterrows():
        axes[0].annotate(f"W={row.W_PPF_SOFT:.0f}", (row.ppf, row.cycle), fontsize=7,
                          xytext=(3, 3), textcoords='offset points')
    axes[0].set_xlabel('PPF_max'); axes[0].set_ylabel('Cycle length (days)')
    axes[0].set_title('Weight-Sweep Approximation to Pareto Front')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_df.W_PPF_SOFT, sweep_df.pop_entropy, 'o-', color='#1B4FBF')
    axes[1].set_xlabel('W_PPF_SOFT'); axes[1].set_ylabel('Final population entropy (spread)')
    axes[1].set_title(f'Diversity Across the Sweep\ncorr(W, entropy)={corr_w_h:.3f}')
    axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_pareto_entropy_sweep.png', dpi=150)
    print(f"[SAVED] {OUT_PREFIX}_pareto_sweep.csv  {OUT_PREFIX}_pareto_entropy_sweep.png\n")
else:
    print("  [SKIP] needs cnn_v9_model.keras + cnn_v9_config.json\n")


# =============================================================================
# SECTION 5 — DMD WITHOUT MC dropout (deterministic perturbation ensemble)
# =============================================================================
print("=" * 70)
print("SECTION 5 — DMD reconstruction error vs a DETERMINISTIC (non-dropout) uncertainty proxy")
print("=" * 70)


def exact_dmd(X, rank):
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, len(S))
    Ur, Sr, Vhr = U[:, :r], S[:r], Vh[:r, :]
    Atilde = Ur.conj().T @ X2 @ Vhr.conj().T @ np.diag(1.0 / Sr)
    eigvals, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vhr.conj().T @ np.diag(1.0 / Sr) @ W
    b, *_ = np.linalg.lstsq(Phi, X[:, 0], rcond=None)
    return eigvals, Phi, b


def dmd_reconstruct(eigvals, Phi, b, n_time):
    powers = np.array([eigvals ** t for t in range(n_time)]).T
    return (Phi @ (b[:, None] * powers)).real


def rel_err(X_true, X_hat):
    return float(np.linalg.norm(X_true - X_hat) / (np.linalg.norm(X_true) + 1e-9))


N_SAMPLE = 500
DMD_RANK = 4

if has(DATA_CSV) and MODEL is not None:
    df3 = pd.read_csv(DATA_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    ppf_steps_avail = sorted(set(int(c.split('_')[1][1:]) for c in df3.columns if c.startswith('ppf_')))
    ppf_assembs = sorted(set(int(c.split('_')[2][1:]) for c in df3.columns if c.startswith('ppf_')))
    ppf_steps_avail = [s for s in ppf_steps_avail
                       if len([c for c in df3.columns if c.startswith(f"ppf_s{s}_")]) == len(ppf_assembs)]
    n_steps, n_assem = len(ppf_steps_avail), len(ppf_assembs)
    load_cols = [f'loading_{i}' for i in range(N_POS)]

    if all(c in df3.columns for c in load_cols) and n_steps > 1:
        idx = rng.choice(len(df3), min(N_SAMPLE, len(df3)), replace=False)
        df3s = df3.iloc[idx].reset_index(drop=True)
        ppf_tensor = np.zeros((len(df3s), n_assem, n_steps), dtype=np.float32)
        for si, s in enumerate(ppf_steps_avail):
            cols = [f'ppf_s{s}_a{a}' for a in ppf_assembs if f'ppf_s{s}_a{a}' in df3.columns]
            ppf_tensor[:, :, si] = df3s[cols].values.astype(np.float32)
        X_flat_s = df3s[load_cols].values.astype(np.int32)

        print(f"  Computing DMD reconstruction error on {len(df3s)} sampled patterns ...")
        dmd_err = np.array([
            rel_err(ppf_tensor[i], dmd_reconstruct(*exact_dmd(ppf_tensor[i], DMD_RANK), n_steps))
            for i in range(len(df3s))
        ])

        # --- deterministic uncertainty proxy: finite-difference input
        # perturbation ensemble. For each pattern, flip ONE randomly chosen
        # position to each of the other 8 assembly types (all 8 alternatives,
        # not random resampling), and measure the spread of the CNN's
        # ppf_max prediction across those 8 deterministic single-position
        # perturbations. This has no stochastic dropout in it at all --
        # it's a first-order finite-difference sensitivity-of-output-to-input
        # perturbation, i.e. a purely deterministic local uncertainty proxy.
        print("  Building deterministic finite-difference perturbation ensemble "
              "(no MC dropout) ...")
        det_sigma = np.zeros(len(df3s))
        base_ppf, _, _ = cnn_predict_ppf_cycle(X_flat_s, mc_samples=0)
        for i in range(len(df3s)):
            pos = rng.integers(0, N_POS)
            variants = []
            for t in range(1, N_TYPES + 1):
                if t == X_flat_s[i, pos]:
                    continue
                v = X_flat_s[i].copy()
                v[pos] = t
                variants.append(v)
            variants = np.stack(variants)
            v_ppf, _, _ = cnn_predict_ppf_cycle(variants, mc_samples=0)
            det_sigma[i] = v_ppf.std()

        r_det = float(np.corrcoef(dmd_err, det_sigma)[0, 1])
        print(f"  corr(DMD error, deterministic perturbation-ensemble sigma) = {r_det:.4f}")
        print("  Compare this to corr(DMD error, MC-dropout sigma) from 04_dmd_signal_validation.py")
        print("  (r=0.504 in your last run). If this deterministic version gives a similar r,")
        print("  DMD error is tracking a general 'how sensitive is the CNN here' property that")
        print("  doesn't depend on dropout noise specifically -- a stronger, cleaner result.")
        print("  If it's much weaker, MC-dropout's stochastic epistemic-uncertainty signal was")
        print("  doing something the deterministic perturbation doesn't capture.\n")

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(det_sigma, dmd_err, alpha=0.25, s=10, color='#8C564B')
        ax.set_xlabel('Deterministic perturbation-ensemble sigma (ppf_max)')
        ax.set_ylabel('DMD reconstruction error')
        ax.set_title(f'DMD error vs deterministic (non-dropout) uncertainty\nr={r_det:.3f}')
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_dmd_no_dropout.png', dpi=150)
        print(f"[SAVED] {OUT_PREFIX}_dmd_no_dropout.png\n")

        # stash for section 6
        _dmd_err_s6 = dmd_err
        _X_flat_s6 = X_flat_s
        _ppf_true_s6 = ppf_tensor.max(axis=(1, 2))
    else:
        _dmd_err_s6 = None
        print("  [SKIP] missing loading_* or ppf_s*_a* columns.\n")
else:
    _dmd_err_s6 = None
    print("  [SKIP] needs dataset + CNN model.\n")


# =============================================================================
# SECTION 6 — Partial correlation: DMD error vs real CNN error, controlling
# for MC sigma (the exact next step flagged at the end of 04_dmd_...)
# =============================================================================
print("=" * 70)
print("SECTION 6 — Partial correlation: does DMD error add anything beyond MC sigma?")
print("=" * 70)

if _dmd_err_s6 is not None and MODEL is not None:
    print("  Computing MC-dropout sigma + actual CNN error on the same sampled patterns ...")
    ppf_mean_mc, ppf_sigma_mc, _ = cnn_predict_ppf_cycle(_X_flat_s6, mc_samples=30)
    cnn_abs_err = np.abs(ppf_mean_mc - _ppf_true_s6)

    def partial_corr(x, y, z):
        """partial correlation of x,y controlling for z (linear residualization)."""
        def resid(a, b):
            b1 = np.vstack([b, np.ones_like(b)]).T
            coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
            return a - b1 @ coef
        rx = resid(x, z)
        ry = resid(y, z)
        return float(np.corrcoef(rx, ry)[0, 1])

    r_raw = float(np.corrcoef(_dmd_err_s6, cnn_abs_err)[0, 1])
    r_partial = partial_corr(_dmd_err_s6, cnn_abs_err, ppf_sigma_mc)

    print(f"  raw corr(DMD error, actual CNN error)                       = {r_raw:.4f}")
    print(f"  partial corr(DMD error, actual CNN error | MC sigma held fixed) = {r_partial:.4f}")
    print()
    if abs(r_partial) < 0.10:
        print("  -> Partial correlation collapses toward zero once MC sigma is controlled for.")
        print("     This means DMD error was mostly RE-DETECTING what MC-dropout sigma already")
        print("     tells you, at the extra cost of an SVD/eig fit per pattern. Verdict: DON'T")
        print("     add DMD error as an independent AL gating term -- it's a redundant, more")
        print("     expensive proxy for sigma you already compute every generation.")
    else:
        print(f"  -> Partial correlation stays nontrivial ({r_partial:.3f}) even after removing")
        print("     everything MC sigma explains. This IS evidence of complementary signal --")
        print("     worth a focused follow-up (e.g. add DMD error as a secondary AL gate term,")
        print("     weighted much lower than sigma, and check if it flags a genuinely different")
        print("     subset of patterns than sigma alone would.")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter([r_raw], [r_partial], s=0)  # invisible, just to set up axes cleanly
    ax.bar(['raw corr', 'partial corr\n(control MC sigma)'], [r_raw, r_partial],
           color=['#1B4FBF', '#D62728'])
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Correlation with actual CNN |prediction error|')
    ax.set_title('Does DMD Error Add Information Beyond MC-Dropout Sigma?')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(f'{OUT_PREFIX}_partial_correlation.png', dpi=150)
    print(f"\n[SAVED] {OUT_PREFIX}_partial_correlation.png\n")
else:
    print("  [SKIP] needs Section 5 to have run successfully.\n")


print("=" * 70)
print("ALL SECTIONS COMPLETE — see mentor_feedback_*.png / *.csv in this directory")
print("=" * 70)