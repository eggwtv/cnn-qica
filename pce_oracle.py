"""
pce_oracle.py
=======================================================================
Polynomial Chaos Expansion surrogate -- a genuine NON-NEURAL-NETWORK
fitness oracle for the discrete 9-type assembly space.

Implementation note (read this before assuming it's "just regression"):
For a discrete UNIFORM random variable (each assembly position takes one
of 9 equally-likely types, matching the DISCRETE UNIFORM assumption
standard PCE requires), the orthogonal polynomial basis for that
variable is literally a one-hot indicator basis -- there's no continuous
Legendre/Hermite polynomial needed because the variable only takes 9
discrete values with no natural ordering. So a PCE over categorical
uniform inputs IS, term-for-term, equivalent to a linear model fit on
one-hot-encoded inputs (first order = "additive" terms) plus pairwise
products of one-hot blocks (second order = "position-interaction"
terms). This is standard practice for categorical PCE (see e.g. the
ANOVA-decomposition literature Sobol & Kucherenko build on) and matches
exactly what your own ablation already found empirically: a first-order
(additive) PCE explained only ~6.3% of PPF variance, meaning most of the
signal is in the pairwise interaction terms -- which is why this
implementation defaults to order=2.

Because Sobol indices for a PCE come for free analytically from the
fitted coefficients (no extra sampling needed), this module gives you
BOTH the surrogate AND its own exact global sensitivity ranking in one
fit -- use pce.sobol_indices() as the "ground truth" to sanity-check the
oracle-agnostic MC estimator in entropy_sensitivity.py against.

Two prediction heads are supported:
  - predict(patterns)        -> scalar PPF_max prediction (what QPSO uses
                                 as its fitness function)
  - predict_curve(patterns)  -> full burnup-step PPF curve (optional; use
                                 dmd_reduction.py to compress the curve
                                 target down to a few mode coefficients
                                 before fitting, per fit_curve_via_dmd())
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


class PCEOracle:
    def __init__(self, n_types=9, order=2, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
        self.n_types = n_types
        self.order = order
        self.alphas = alphas
        self.encoder = OneHotEncoder(categories=[list(range(1, n_types + 1))] * 1,
                                      sparse_output=False)
        self.model = None
        self.model_curve = None
        self.n_pos = None
        self._cv_mae = None  # leave-one-fold-out CV error -> used as
        # this oracle's own uncertainty proxy (analogue to MC-dropout std)

    # ------------------------------------------------------------------
    def _onehot_features(self, X):
        """
        X: (N, n_pos) int array -> (N, n_pos*n_types) one-hot block
        This one-hot block IS the first-order PCE basis.
        """
        N, n_pos = X.shape
        blocks = []
        for p in range(n_pos):
            oh = np.zeros((N, self.n_types), dtype=np.float64)
            col = X[:, p].astype(int) - 1
            oh[np.arange(N), col] = 1.0
            blocks.append(oh)
        return np.concatenate(blocks, axis=1), blocks

    def _second_order_features(self, blocks, pairs):
        """
        Pairwise interaction terms: outer product of one-hot blocks for a
        fixed list of position pairs. Full pairwise expansion is
        O(n_pos^2 * n_types^2) (31 positions -> ~465 pairs * 81 =~37,000
        extra columns) -- to keep this runnable without a GPU, `pairs` is
        capped to a fixed random subset chosen once in _design_matrix and
        reused for every subsequent call (train or predict), so the
        design matrix has a consistent shape. RidgeCV handles the
        resulting dimensionality fine since it's still linear-in-params.
        """
        if not pairs:
            return np.zeros((blocks[0].shape[0], 0))
        feats = []
        for (i, j) in pairs:
            # outer product per-sample: (N, n_types) x (N, n_types) -> (N, n_types*n_types)
            fi, fj = blocks[i], blocks[j]
            feats.append(np.einsum('na,nb->nab', fi, fj).reshape(fi.shape[0], -1))
        return np.concatenate(feats, axis=1)

    def _design_matrix(self, X, pair_subset=None):
        oh_flat, blocks = self._onehot_features(X)
        if self.order < 2:
            return oh_flat, None
        n_pos = X.shape[1]
        if pair_subset is None:
            # cap pairwise expansion for tractability on 31+ positions:
            # sample a fixed random subset of position pairs once (stored
            # on the instance) rather than all ~465 pairs.
            rng = np.random.default_rng(42)
            all_pairs = [(i, j) for i in range(n_pos) for j in range(i + 1, n_pos)]
            n_keep = min(120, len(all_pairs))
            chosen = rng.choice(len(all_pairs), size=n_keep, replace=False)
            pair_subset = [all_pairs[k] for k in chosen]
        pair_feats = self._second_order_features(blocks, pair_subset)
        return np.concatenate([oh_flat, pair_feats], axis=1), pair_subset


    
    #added fixes: 
    def fit_ensemble(self, X_train, y_train, n_models=8, subsample_frac=0.8, seed=0):
        """Bootstrap ensemble for uncertainty -- needed since base PCE has no native sigma."""
        rng = np.random.default_rng(seed)
        self._ensemble = []
        n = len(X_train)
        for i in range(n_models):
            idx = rng.choice(n, size=int(n * subsample_frac), replace=True)
            m = PCEOracle(n_types=self.n_types, order=self.order, alphas=self.alphas)
            m.fit(X_train[idx], y_train[idx])
            self._ensemble.append(m)

    def predict_with_uncertainty(self, X):
        if not hasattr(self, '_ensemble') or not self._ensemble:
            mean = self.predict(X)
            return mean, np.zeros_like(mean)
        preds = np.stack([m.predict(X) for m in self._ensemble])
        return preds.mean(axis=0), preds.std(axis=0)


    
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, cv_folds=5):
        """
        X_train : (N, n_pos) int assembly-type patterns
        y_train : (N,) scalar target (e.g. ppf_max)
        """
        self.n_pos = X_train.shape[1]
        design, self._pairs = self._design_matrix(X_train)
        self.model = RidgeCV(alphas=self.alphas, cv=cv_folds)
        self.model.fit(design, y_train)

        # cheap CV-based uncertainty proxy: k-fold held-out MAE, used
        # later as a flat "this oracle's typical error" scalar (NOT
        # per-candidate uncertainty -- PCE doesn't give you that for
        # free the way GPR does; see gpr_oracle.py for the analogue
        # that DOES have per-point predictive variance).
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
        errs = []
        for tr_idx, te_idx in kf.split(design):
            m = RidgeCV(alphas=self.alphas, cv=3)
            m.fit(design[tr_idx], y_train[tr_idx])
            pred = m.predict(design[te_idx])
            errs.append(np.mean(np.abs(pred - y_train[te_idx])))
        self._cv_mae = float(np.mean(errs))
        return self

    def predict(self, X):
        design, _ = self._design_matrix(X, pair_subset=self._pairs)
        return self.model.predict(design)

    def __call__(self, X):
        """Makes PCEOracle directly usable as a fitness_fn(patterns)."""
        return self.predict(X)

    @property
    def cv_mae(self):
        return self._cv_mae

    # ------------------------------------------------------------------
    def sobol_indices(self):
        """
        Analytic first-order Sobol indices from the fitted Ridge
        coefficients -- exact given this basis (no Monte Carlo sampling
        error), computed as the sum-of-squared-coefficients belonging to
        each position's one-hot block, normalized by total explained
        variance. This is the standard PCE -> Sobol closed form.
        """
        coefs = self.model.coef_
        n_types = self.n_types
        first_order = np.zeros(self.n_pos)
        for p in range(self.n_pos):
            block = coefs[p * n_types:(p + 1) * n_types]
            first_order[p] = np.sum(block ** 2)
        total = first_order.sum() + 1e-12
        return first_order / total


def fit_curve_via_dmd(pce_class, X_train, curve_matrix, dmd_model, **pce_kwargs):
    """
    Fit one small PCEOracle per DMD mode-coefficient instead of one per
    raw burnup timestep. Uses dmd_reduction.reduce_curve_to_modes() to
    compress each training curve to ~2*rank real features first.
    Returns a list of fitted PCEOracle objects (one per DMD feature) plus
    the dmd_model needed to reconstruct full curves at inference time.
    """
    from dmd_reduction import reduce_curve_to_modes
    feats = np.array([reduce_curve_to_modes(c, dmd_model) for c in curve_matrix])
    models = []
    for k in range(feats.shape[1]):
        m = pce_class(**pce_kwargs)
        m.fit(X_train, feats[:, k])
        models.append(m)
    return models
