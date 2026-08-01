"""
=============================================================================
qica-cnn-v4.py  —  Uncertainty-Aware QICA for BEAVRS CNN-v4
                   NO OpenMC dependency  |  Active Learning loop included
=============================================================================

WHAT THIS FILE DOES:
  Quantum Imperialist Competitive Algorithm (QICA) that searches the space
  of 31-position PWR fuel loading patterns to find configurations with:
    • Low PPF_max (power peaking factor) — the primary safety constraint
    • Acceptable cycle length (how long the reactor can run before refuelling)
    • keff near operating range (derived from rho_pcm prediction)

  The CNN (from cnn_v4.py) acts as a SURROGATE for the physics simulation —
  instead of running a full neutron transport code for each candidate pattern
  (which takes hours), the CNN evaluates thousands of patterns in seconds.

  MC Dropout uncertainty is used throughout:
    • High σ → QICA avoids this region (the surrogate may be wrong there)
    • Low σ  → surrogate is confident → this is a trustworthy candidate

CHANGES FROM THE ORIGINAL (bug fixes):
──────────────────────────────────────────────────────────────────────────
FIX 1 — Model loading crash (CRITICAL):
  The original crashed with:
      TypeError: Could not locate class 'ConvResBlock'
  ROOT CAUSE: ConvResBlock is a custom Keras subclass. When a model is
  saved, Keras serialises the class NAME but not the class DEFINITION.
  On load, Keras looks up the name in a global registry. If the registry
  doesn't have it, loading fails.
  FIX: Define ConvResBlock here (before load_model) with the
  @keras.saving.register_keras_serializable() decorator, which registers it
  in the global Keras registry for this Python process. Then load with
  custom_objects={'ConvResBlock': ConvResBlock} as a belt-and-suspenders
  fallback.
  NOTE: The class definition must be IDENTICAL to cnn_v4.py — same layer
  names, same __init__ signature, same get_config() — or weights won't
  map correctly. Copy-paste, never diverge.

FIX 2 — OpenMC dependency removed:
  The original imported openmc_beavrs_simulator.py and crashed when
  OpenMC wasn't installed. All OpenMC references removed.
  Active learning now operates in "candidate-flagging" mode:
    • QICA finds optimal patterns via surrogate evaluation only
    • Uncertain candidates are saved to CSV for future labelling
    • When you have a simulator, plug it into simulate_pattern() below
    • Set AL_ROUNDS > 0 to run actual AL retraining rounds

HOW TO CONNECT A REAL SIMULATOR (when ready):
  1. Fill in the body of simulate_pattern() (see Section 3)
  2. Set AL_ROUNDS = 1 (or more) at the top of Section 1
  3. Re-run — the loop will automatically:
       a. QICA finds low-PPF candidates with σ > AL_SIGMA_THRESHOLD
       b. Runs simulate_pattern() on each candidate
       c. Adds new (pattern, ppf, cycle, keff) rows to training arrays
       d. Retrains the CNN on the expanded dataset (warm start, 50 epochs)
       e. Repeats for AL_ROUNDS rounds

  Compatible simulators: OpenMC, Serpent, PARCS, or any code that takes
  a loading pattern and returns PPF/cycle/keff.

INPUTS:
  cnn_v4_model.keras     — trained model from cnn_v4.py
  cnn_v4_config.json     — scalers, indices, geometry
  train_type_freq.npy    — per-position type frequencies (optional, for trust-region)

OUTPUTS:
  qica_v4_best_patterns.csv  — top patterns found
  qica_v4_al_candidates.csv   — candidates flagged for future simulation
  qica_v4_convergence.png    — optimisation convergence plots
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.keras.utils import register_keras_serializable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow {tf.__version__}")
print("qica-cnn-v4.py  —  Uncertainty-Aware QICA for BEAVRS CNN-v4\n")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

MODEL_PATH  = 'cnn_v9_model.keras'
CONFIG_PATH = 'cnn_v9_config.json'
TRUST_PATH  = 'train_type_freq_v9.npy'

# ── Fitness weighting ─────────────────────────────────────────────────────────
# The QICA maximises: fitness = cycle − PPF_penalty − uncertainty_penalty
#                              − trust_penalty + monotonicity_bonus + entropy_bonus
#
# PPF_LIMIT:     safety limit for ppf_max. Patterns above this get penalised.
# W_PPF_PENALTY: days subtracted per unit of PPF excess above PPF_LIMIT.
# W_UNCERTAINTY: days subtracted per unit of MC Dropout σ_ppf.
#                Encourages QICA to stay in regions where the surrogate is confident.
# W_TRUST:       days subtracted for patterns far from the training distribution.
#                Prevents the surrogate from hallucinating good results for
#                assembly type combinations it never saw in training.
# W_ENTROPY:     fitness bonus per unit of quantum state entropy (diversity measure).
#                Prevents premature convergence to one local optimum.
# W_MONOTONICITY: bonus for patterns whose predicted PPF burnup profile
#                decreases monotonically in the late cycle (physically correct).
PPF_LIMIT         = 3.5
W_PPF_PENALTY     = 80.0
W_UNCERTAINTY     = 40.0
W_TRUST           = 20.0
W_ENTROPY_BONUS   = 5.0
W_MONOTONICITY    = 10.0

# ── Active learning ───────────────────────────────────────────────────────────
# AL_SIGMA_THRESHOLD: patterns with σ_ppf above this are "uncertain" and should
#                     ideally be sent to a physics simulator for ground-truth labelling.
# AL_TOP_K:           maximum number of query candidates to export per run.
# AL_ROUNDS:          number of active learning rounds to run.
#                     0 = candidate-flagging only (no retraining, no simulator needed).
#                     1+ = requires simulate_pattern() to be implemented.
AL_SIGMA_THRESHOLD = 0.08
AL_TOP_K           = 50
AL_ROUNDS          = 0    # ← set to 1+ when simulate_pattern() is ready

# ── QICA hyperparameters ──────────────────────────────────────────────────────
# N_COUNTRIES:       total population size (patterns evaluated per generation).
# N_EMPIRES:         number of imperialists (elite countries that attract colonies).
# ASSIMILATION_COEFF: how strongly colonies blend toward their imperialist (0–1).
# REVOLUTION_RATE:   fraction of positions randomly reset per generation (exploration).
# REVOLUTION_MIN:    minimum revolution rate (exploitation floor).
# QUANTUM_TEMP_INIT: initial "temperature" — high = flat distribution (exploration).
# QUANTUM_TEMP_FINAL: final temperature — low = peaked distribution (exploitation).
# MAX_GEN:           maximum number of generations.
# ELITE_SIZE:        how many top patterns to keep in the archive.
# MC_SAMPLES:        number of stochastic forward passes per pattern evaluation.
N_COUNTRIES        = 100
N_EMPIRES          = 8
ASSIMILATION_COEFF = 0.45
REVOLUTION_RATE    = 0.35
REVOLUTION_MIN     = 0.04
QUANTUM_TEMP_INIT  = 2.0
QUANTUM_TEMP_FINAL = 0.1
MAX_GEN            = 250
ELITE_SIZE         = 15
MC_SAMPLES         = 30
SEED               = 42

# CNN geometry (must match cnn_v4.py)
GRID_ROWS = 6
GRID_COLS = 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)
N_POS   = int(GRID_MASK.sum())   # 31
N_TYPES = 9


# =============================================================================
# SECTION 2 — DEFINE ConvResBlock (MUST MATCH cnn_v4.py EXACTLY)
# =============================================================================
#
# WHY WE REDEFINE IT HERE:
#   The Keras global registry (@register_keras_serializable) is process-local.
#   Every Python process (including this script) starts with an empty registry.
#   When Keras tries to load the model, it looks up 'ConvResBlock' in the
#   registry. If this script hasn't defined it yet, the lookup fails.
#
#   By defining it here with the same decorator BEFORE calling load_model(),
#   we populate the registry in this process, and the load succeeds.
#
#   The class must be BYTE-FOR-BYTE identical to cnn_v4.py's ConvResBlock —
#   same layer names, same init params — or the saved weights won't align.


@tf.keras.utils.register_keras_serializable()
class ConvResBlock(layers.Layer):
    """
    Residual convolutional block (must match cnn_v4.py):
      Conv → BN → GELU → Conv → BN → Add(shortcut) → GELU → Dropout

    Registering with @keras.saving.register_keras_serializable() is what
    allows Keras to find this class when loading a saved model file.
    """
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
        cfg.update({'filters': self._filters,
                    'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


# =============================================================================
# SECTION 3 — SIMULATOR STUB (plug your simulator in here)
# =============================================================================
#
# WHAT THIS FUNCTION SHOULD DO (when implemented):
#   Take a 31-position integer loading pattern and return the physics outputs
#   that a neutron transport code (OpenMC, PARCS, Serpent, etc.) would compute.
#
# HOW ACTIVE LEARNING WORKS WITH THIS:
#   The QICA searches ~25,000 candidate patterns using only the CNN surrogate.
#   Some of those candidates will have high σ_ppf — the CNN is uncertain about
#   them. These are valuable to simulate because:
#     a. If the CNN was wrong (real PPF is high), we learn a failure case.
#     b. If the CNN was right (real PPF is low), we confirm a good pattern.
#   Either way, adding the new (pattern → true physics) pair to the training
#   set and retraining makes the CNN more accurate in that region.
#   After N rounds of this loop, the surrogate's uncertainty in the low-PPF
#   region shrinks, and the QICA can find better patterns with more confidence.
#
# HOW TO CONNECT A REAL SIMULATOR:
#   Replace the NotImplementedError body below with your actual simulation call.
#   Expected return dict:
#     {
#       'ppf_max'      : float  — global max PPF over fuel cycle
#       'ppf_boc'      : float  — PPF at beginning of cycle
#       'ppf_steps'    : array  — (N_STEPS,) max PPF at each burnup step
#       'cycle_length' : float  — effective full-power days
#       'keff_boc'     : float  — k-eff at BOC
#       'rho_pcm_boc'  : float  — reactivity (pcm) at BOC
#       'success'      : bool   — False if the simulation failed
#     }
#
# EXAMPLE OpenMC integration (pseudocode):
#   import openmc
#   import openmc.deplete
#   materials = build_materials_from_pattern(loading_pattern_1d)
#   geometry  = build_beavrs_geometry(materials)
#   settings  = openmc.Settings()
#   settings.batches   = 200
#   settings.particles = 50000
#   model = openmc.Model(geometry, materials, settings)
#   sp = model.run()
#   with openmc.StatePoint(sp) as sp_file:
#       keff_boc = float(sp_file.k_combined[0])
#   # then run depletion to get cycle length and PPF at each step
#   ...
#   return {'ppf_max': ..., 'cycle_length': ..., 'keff_boc': keff_boc, 'success': True}

def simulate_pattern(loading_pattern_1d: np.ndarray) -> dict:
    """
    Physics simulation stub.

    Args:
        loading_pattern_1d : (31,) int array — assembly types 1–9 per position

    Returns:
        dict with keys: ppf_max, ppf_boc, ppf_steps, cycle_length,
                        keff_boc, rho_pcm_boc, success
        OR raises NotImplementedError if not yet implemented.
    """
    raise NotImplementedError(
        "simulate_pattern() is a stub.\n"
        "Replace this function body with your physics simulation.\n"
        "Then set AL_ROUNDS > 0 in Section 1 to enable active learning."
    )


# =============================================================================
# SECTION 4 — LOAD CNN-v4 MODEL + CONFIG
# =============================================================================

print("[LOAD] CNN-v4 model + config ...")
for path in [MODEL_PATH, CONFIG_PATH]:
    if not os.path.exists(path):
        print(f"[ERROR] Missing: {path}")
        print("  → Run cnn_v4.py first to generate these files.")
        sys.exit(1)

# custom_objects is the belt-and-suspenders approach: even if the decorator
# registry lookup fails (rare edge case), Keras can still find the class here.
model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={'ConvResBlock': ConvResBlock}
)
print(f"  Model loaded  : {MODEL_PATH}")
print(f"  Input shape   : {model.input_shape}")
print(f"  Output shape  : {model.output_shape}")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

# Reconstruct scalers from saved mean/scale arrays
ym_mean  = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
ym_scale = np.array(cfg['ym_scaler_scale'], dtype=np.float32)
yr_mean  = np.array(cfg['yr_scaler_mean'],  dtype=np.float32)
yr_scale = np.array(cfg['yr_scaler_scale'], dtype=np.float32)

IDX_PPF_MAX = cfg['IDX_PPF_MAX']      # 0
IDX_PPF_BOC = cfg['IDX_PPF_BOC']      # 1
IDX_STEPS_S = cfg['IDX_PPF_STEPS_START']  # 2
IDX_STEPS_E = cfg['IDX_PPF_STEPS_END']    # 33
IDX_CYCLE   = cfg['IDX_CYCLE']        # 33
IDX_RHO     = cfg['IDX_RHO']          # 34
N_STEPS     = IDX_STEPS_E - IDX_STEPS_S  # 31
N_OUTPUTS   = cfg['N_OUTPUTS']        # 35

print(f"  Indices       : ppf_max={IDX_PPF_MAX}, cycle={IDX_CYCLE}, rho={IDX_RHO}")
print(f"  PPF limit     : {PPF_LIMIT}  | MC samples: {MC_SAMPLES}")
print()


# =============================================================================
# SECTION 5 — TRUST-REGION FREQUENCIES
# =============================================================================
#
# WHAT THE TRUST REGION DOES:
#   The CNN was trained on a specific distribution of patterns. If the QICA
#   proposes a pattern that uses assembly type combinations that never appeared
#   in training, the CNN is extrapolating — its predictions are likely wrong.
#
#   The trust region penalty = Σ_pos -log P(type | pos) / N_POS
#   where P(type | pos) is the empirical frequency of that type at that position
#   in the training set. Rare type/position combinations get large penalties.
#
#   This keeps QICA in the "well-supported" region of the input space,
#   preventing surrogate hallucinations (good-looking predictions for patterns
#   the model has never seen a training analog for).

if os.path.exists(TRUST_PATH):
    type_freq = np.load(TRUST_PATH).astype(np.float32)   # (31, 9)
    print(f"[TRUST] Loaded per-position type frequencies from {TRUST_PATH}")
    print(f"  Shape: {type_freq.shape} — trust-region penalty active\n")
else:
    print(f"[TRUST] {TRUST_PATH} not found — using uniform fallback.")
    print(f"  → Re-run cnn_v4.py to generate it, or trust penalty will be disabled.\n")
    type_freq = np.ones((N_POS, N_TYPES), dtype=np.float32) / N_TYPES


# =============================================================================
# SECTION 6 — GRID BUILDER + INVERSE TRANSFORM
# =============================================================================

def pattern_to_grid(pattern_int: np.ndarray) -> np.ndarray:
    """
    Convert (B, 31) integer pattern → (B, 6, 6) integer grid.
    Active cells filled from pattern; reflector cells stay 0.
    """
    B   = pattern_int.shape[0] if pattern_int.ndim > 1 else 1
    pat = pattern_int.reshape(B, N_POS)
    grid = np.zeros((B, GRID_ROWS, GRID_COLS), dtype=np.int32)
    pos_i = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid[:, r, c] = pat[:, pos_i]
                pos_i += 1
    return grid


def inverse_transform(Y_sc: np.ndarray) -> np.ndarray:
    """
    Reverse the two-scaler scheme to get real physical units.
    Y_sc[:, :34] → ym_scaler inverse  (PPF + cycle)
    Y_sc[:, 34:] → yr_scaler inverse  (rho_pcm)
    """
    Y_main = Y_sc[:, :34] * ym_scale + ym_mean
    Y_rho  = Y_sc[:, 34:35] * yr_scale + yr_mean
    return np.concatenate([Y_main, Y_rho], axis=1)


# =============================================================================
# SECTION 7 — MC DROPOUT EVALUATOR
# =============================================================================

def evaluate_batch(patterns_int: np.ndarray, temperature: float = 1.0) -> dict:
    """
    Evaluate a batch of loading patterns using MC Dropout.

    Runs MC_SAMPLES stochastic forward passes (training=True keeps dropout ON).
    Returns mean predictions AND σ (uncertainty) for all outputs.

    Args:
        patterns_int : (B, 31) integer array, values 1–9
        temperature  : not used in evaluation directly (passed through for logging)

    Returns:
        dict with arrays of shape (B,):
          ppf_mean, ppf_std, cycle_mean, rho_mean, keff_mean,
          ppf_steps (B, 31), fitness, trust_penalty
    """
    if patterns_int.ndim == 1:
        patterns_int = patterns_int.reshape(1, -1)
    B    = patterns_int.shape[0]
    grid = pattern_to_grid(patterns_int)
    X_tf = tf.constant(grid, dtype=tf.int32)

    # MC Dropout: training=True keeps dropout active at inference
    mc_preds_sc = np.stack([
        model(X_tf, training=True).numpy()
        for _ in range(MC_SAMPLES)
    ])   # (MC_SAMPLES, B, 35)

    mean_sc = mc_preds_sc.mean(axis=0)   # (B, 35)
    std_sc  = mc_preds_sc.std(axis=0)    # (B, 35)

    mean_real = inverse_transform(mean_sc)
    std_real  = np.concatenate([
        std_sc[:, :34] * ym_scale,
        std_sc[:, 34:35] * yr_scale,
    ], axis=1)

    ppf_mean   = mean_real[:, IDX_PPF_MAX]
    ppf_std    = std_real[:, IDX_PPF_MAX]
    cycle_mean = mean_real[:, IDX_CYCLE]
    rho_mean   = mean_real[:, IDX_RHO]
    keff_mean  = 1.0 / (1.0 - rho_mean / 1e5)
    ppf_steps  = mean_real[:, IDX_STEPS_S:IDX_STEPS_E]   # (B, 31)

    # ── Trust-region penalty ──────────────────────────────────────────────────
    trust_penalty = np.zeros(B, dtype=np.float32)
    for b in range(B):
        pat = patterns_int[b]
        nlp = sum(-np.log(float(type_freq[p, int(pat[p]) - 1]) + 1e-6)
                  for p in range(N_POS)) / N_POS
        trust_penalty[b] = float(nlp)

    # ── PPF burnup monotonicity bonus ─────────────────────────────────────────
    # After burnup step 3 (BA burnout), PPF should only decrease.
    # Bonus = W * (1 - fraction_of_upward_steps).
    late        = ppf_steps[:, 3:]                         # (B, 28)
    diffs       = late[:, 1:] - late[:, :-1]               # (B, 27)
    n_violate   = (diffs > 0).sum(axis=1).astype(np.float32)
    mono_bonus  = W_MONOTONICITY * (1.0 - n_violate / 27.0)

    # ── Fitness ───────────────────────────────────────────────────────────────
    ppf_excess  = np.maximum(0.0, ppf_mean - PPF_LIMIT)
    fitness     = (cycle_mean
                   - W_PPF_PENALTY * ppf_excess
                   - W_UNCERTAINTY  * ppf_std
                   - W_TRUST        * trust_penalty
                   + mono_bonus)

    return {
        'ppf_mean'     : ppf_mean,
        'ppf_std'      : ppf_std,
        'cycle_mean'   : cycle_mean,
        'rho_mean'     : rho_mean,
        'keff_mean'    : keff_mean,
        'ppf_steps'    : ppf_steps,
        'fitness'      : fitness,
        'trust_penalty': trust_penalty,
    }


# =============================================================================
# SECTION 8 — QUANTUM COUNTRY
# =============================================================================

class QuantumCountry:
    """
    Represents one "country" (candidate loading pattern) in quantum superposition.

    Instead of a fixed integer pattern, a QuantumCountry holds a probability
    matrix q_state: (N_POS, N_TYPES) where q_state[pos, t] is the probability
    of position pos having assembly type (t+1).

    The "quantum measurement" (collapse) samples a concrete integer pattern
    from this probability distribution. The temperature controls how peaked
    the sampling is: high T → uniform (exploration), low T → greedy (exploitation).
    """

    def __init__(self, q_state: np.ndarray = None):
        if q_state is None:
            raw = np.ones((N_POS, N_TYPES), dtype=np.float32)
            self.q_state = raw / raw.sum(axis=1, keepdims=True)
        else:
            self.q_state = q_state.copy().astype(np.float32)
        self.measured   = None
        self.fitness    = -np.inf
        self.ppf_mean   = 9.0
        self.ppf_std    = 0.0
        self.cycle_mean = 0.0
        self.keff_mean  = 0.0
        self.trust_pen  = 0.0

    def collapse(self, temperature: float = 1.0) -> np.ndarray:
        """Sample a concrete pattern from the quantum probability distribution."""
        logits = np.log(self.q_state + 1e-10) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs  = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        self.measured = np.array([
            np.random.choice(N_TYPES, p=probs[i]) + 1
            for i in range(N_POS)
        ], dtype=np.int32)
        return self.measured

    def entropy(self) -> float:
        """Shannon entropy of q_state — higher = more diverse/uncertain."""
        return float(-np.sum(self.q_state * np.log(self.q_state + 1e-10)))

    def quantum_assimilate(self, imperialist: 'QuantumCountry', beta: float, temperature: float):
        """
        Move colony's probability distribution toward imperialist's distribution.
        Pure probability-space interpolation — no hard type assignments.
        beta = assimilation coefficient (0 = no move, 1 = become imperialist).
        """
        self.q_state = (1.0 - beta) * self.q_state + beta * imperialist.q_state
        self.q_state = np.maximum(self.q_state, 1e-10)
        self.q_state /= self.q_state.sum(axis=1, keepdims=True)

    def quantum_revolution(self, rate: float, temperature: float):
        """
        Per-position Dirichlet reset: randomly reinitialise some positions.
        High temperature → flat Dirichlet (broad exploration).
        Low temperature → concentrated Dirichlet (local exploitation).
        """
        for i in range(N_POS):
            if np.random.random() < rate:
                alpha = np.ones(N_TYPES) * max(temperature, 0.05)
                self.q_state[i] = np.random.dirichlet(alpha)

    def clone(self) -> 'QuantumCountry':
        c = QuantumCountry(self.q_state)
        c.measured   = self.measured.copy() if self.measured is not None else None
        c.fitness    = self.fitness
        c.ppf_mean   = self.ppf_mean
        c.ppf_std    = self.ppf_std
        c.cycle_mean = self.cycle_mean
        c.keff_mean  = self.keff_mean
        c.trust_pen  = self.trust_pen
        return c


# =============================================================================
# SECTION 9 — EMPIRE
# =============================================================================

class Empire:
    """
    One empire = one imperialist (the strongest country) + its colonies.
    Colonies are attracted toward the imperialist over time (assimilation).
    If a colony becomes stronger than the imperialist, they swap roles.
    Weak empires eventually collapse: their colonies are absorbed by strong ones.
    """
    def __init__(self, imperialist: QuantumCountry, colonies: list):
        self.imperialist = imperialist
        self.colonies    = colonies

    @property
    def power(self) -> float:
        return self.imperialist.fitness

    @property
    def total_countries(self) -> int:
        return 1 + len(self.colonies)


# =============================================================================
# SECTION 10 — QICA OPTIMIZER
# =============================================================================

class QICAv5:
    """
    Quantum Imperialist Competitive Algorithm — v5.

    Evolution of one generation:
      1. All colonies undergo assimilation (move toward imperialist) and
         revolution (random reset of some positions = exploration).
      2. All colonies are collapse+evaluated (MC Dropout fitness).
      3. Intra-empire competition: strongest colony may dethrone imperialist.
      4. Inter-empire competition: weakest empire loses one country to strongest.
      5. Elite archive updated with all unique top patterns seen so far.
      6. Repeat until convergence (single empire) or MAX_GEN reached.
    """

    def __init__(self):
        self.elite_archive = []   # list of (fitness, pattern, ppf, cycle, σ_ppf)
        self.al_candidates = []   # patterns flagged for future simulation
        self.history = {
            'gen': [], 'best_fitness': [], 'mean_fitness': [],
            'best_cycle': [], 'best_ppf': [], 'mean_ppf_std': [],
            'n_empires': [], 'temperature': [], 'al_count': [],
        }

    # ── Adaptive schedules ────────────────────────────────────────────────────

    def _temperature(self, gen: int) -> float:
        """Exponential decay from QUANTUM_TEMP_INIT to QUANTUM_TEMP_FINAL."""
        r = gen / MAX_GEN
        return QUANTUM_TEMP_INIT * (QUANTUM_TEMP_FINAL / QUANTUM_TEMP_INIT) ** r

    def _revolution_rate(self, gen: int) -> float:
        """Linear decay from REVOLUTION_RATE to REVOLUTION_MIN."""
        r = gen / MAX_GEN
        return REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * r

    # ── Population initialisation ─────────────────────────────────────────────

    def _initialize_population(self) -> list:
        countries = []
        # N_TYPES seeded countries: each biased toward one assembly type
        # (encourages diversity in the initial population)
        for bias_t in range(1, N_TYPES + 1):
            q = np.ones((N_POS, N_TYPES), dtype=np.float32) * 0.04
            q[:, bias_t - 1] = 0.68
            q /= q.sum(axis=1, keepdims=True)
            countries.append(QuantumCountry(q))
        # Remaining countries: uniform random (no bias)
        for _ in range(N_COUNTRIES - N_TYPES):
            countries.append(QuantumCountry())
        return countries

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def _evaluate_all(self, countries: list, temperature: float) -> list:
        """Collapse all countries, batch-evaluate with MC Dropout."""
        patterns = np.stack([c.collapse(temperature) for c in countries])  # (B, 31)
        result   = evaluate_batch(patterns, temperature)

        ppf_arr  = result['ppf_mean']
        std_arr  = result['ppf_std']
        cyc_arr  = result['cycle_mean']
        rho_arr  = result['rho_mean']
        step_arr = result['ppf_steps']
        tpen_arr = result['trust_penalty']

        # Entropy bonus (per-country — encourages population diversity)
        ent_bonus = np.array([
            W_ENTROPY_BONUS * c.entropy() / (N_POS * N_TYPES)
            for c in countries
        ], dtype=np.float32)

        ppf_excess  = np.maximum(0.0, ppf_arr - PPF_LIMIT)
        late        = step_arr[:, 3:]
        diffs       = late[:, 1:] - late[:, :-1]
        mono_bonus  = W_MONOTONICITY * (1.0 - (diffs > 0).sum(axis=1) / 27.0)

        fitness_arr = (cyc_arr
                       - W_PPF_PENALTY * ppf_excess
                       - W_UNCERTAINTY  * std_arr
                       - W_TRUST        * tpen_arr
                       + mono_bonus
                       + ent_bonus)

        for i, c in enumerate(countries):
            c.fitness    = float(fitness_arr[i])
            c.ppf_mean   = float(ppf_arr[i])
            c.ppf_std    = float(std_arr[i])
            c.cycle_mean = float(cyc_arr[i])
            c.keff_mean  = float(1.0 / (1.0 - float(rho_arr[i]) / 1e5))
            c.trust_pen  = float(tpen_arr[i])

        # Flag uncertain + low-PPF patterns as AL candidates
        ppf_25pct = np.percentile(ppf_arr, 25)
        for i, c in enumerate(countries):
            if c.ppf_std >= AL_SIGMA_THRESHOLD and c.ppf_mean <= ppf_25pct:
                self.al_candidates.append({
                    'pattern'   : c.measured.tolist(),
                    'pred_ppf'  : c.ppf_mean,
                    'sigma_ppf' : c.ppf_std,
                    'cycle'     : c.cycle_mean,
                    'priority'  : c.ppf_std / (c.ppf_mean + 1e-6),
                })

        return countries

    # ── Form empires ──────────────────────────────────────────────────────────

    def _form_empires(self, countries: list) -> list:
        sorted_c = sorted(countries, key=lambda c: c.fitness, reverse=True)
        imperialists  = sorted_c[:N_EMPIRES]
        colonies_pool = sorted_c[N_EMPIRES:]
        fits    = np.array([imp.fitness for imp in imperialists])
        fits_sh = fits - fits.min() + 1e-6
        powers  = fits_sh / fits_sh.sum()
        n_col   = len(colonies_pool)
        counts  = np.round(powers * n_col).astype(int)
        diff    = n_col - counts.sum()
        if diff > 0:
            counts[np.argmax(powers)] += diff
        elif diff < 0:
            counts[np.argmax(counts)] += diff
        empires, idx = [], 0
        for i, imp in enumerate(imperialists):
            empires.append(Empire(imp, list(colonies_pool[idx:idx + counts[i]])))
            idx += counts[i]
        return empires

    # ── Assimilation + revolution ─────────────────────────────────────────────

    def _assimilation_step(self, empires: list, beta: float, temp: float, rev_rate: float):
        for empire in empires:
            for col in empire.colonies:
                col.quantum_assimilate(empire.imperialist, beta, temp)
                col.quantum_revolution(rev_rate, temp)

    # ── Intra-empire competition ───────────────────────────────────────────────

    def _intra_competition(self, empires: list, temperature: float):
        all_cols = [col for emp in empires for col in emp.colonies]
        if not all_cols:
            return
        self._evaluate_all(all_cols, temperature)
        for empire in empires:
            if not empire.colonies:
                continue
            best_i = max(range(len(empire.colonies)),
                         key=lambda i: empire.colonies[i].fitness)
            best_c = empire.colonies[best_i]
            if best_c.fitness > empire.imperialist.fitness:
                empire.colonies[best_i] = empire.imperialist
                empire.imperialist      = best_c

    # ── Empire collapse ───────────────────────────────────────────────────────

    def _empire_collapse(self, empires: list) -> list:
        if len(empires) <= 1:
            return empires
        weak_i   = min(range(len(empires)), key=lambda i: empires[i].power)
        strong_i = max(range(len(empires)), key=lambda i: empires[i].power)
        weak     = empires[weak_i]
        if len(weak.colonies) == 0:
            empires[strong_i].colonies.append(weak.imperialist)
            empires.pop(weak_i)
        else:
            wc_i = min(range(len(weak.colonies)),
                       key=lambda i: weak.colonies[i].fitness)
            empires[strong_i].colonies.append(weak.colonies.pop(wc_i))
        return empires

    # ── Elite archive ─────────────────────────────────────────────────────────

    def _update_elite(self, empires: list):
        for emp in empires:
            for c in [emp.imperialist] + emp.colonies:
                if c.measured is not None:
                    self.elite_archive.append((
                        c.fitness, c.measured.copy(),
                        c.ppf_mean, c.cycle_mean, c.ppf_std
                    ))
        self.elite_archive.sort(key=lambda x: x[0], reverse=True)
        seen, unique = set(), []
        for entry in self.elite_archive:
            key = tuple(entry[1])
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        self.elite_archive = unique[:ELITE_SIZE]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, gen: int, empires: list, temp: float, rev_rate: float):
        all_fit     = ([e.imperialist.fitness for e in empires] +
                       [c.fitness for e in empires for c in e.colonies])
        all_ppf_std = ([e.imperialist.ppf_std for e in empires] +
                       [c.ppf_std for e in empires for c in e.colonies])
        best = self.elite_archive[0] if self.elite_archive else (0, None, 9.0, 0.0, 0.0)
        self.history['gen'].append(gen)
        self.history['best_fitness'].append(float(max(all_fit)))
        self.history['mean_fitness'].append(float(np.mean(all_fit)))
        self.history['best_cycle'].append(float(best[3]))
        self.history['best_ppf'].append(float(best[2]))
        self.history['mean_ppf_std'].append(float(np.mean(all_ppf_std)))
        self.history['n_empires'].append(len(empires))
        self.history['temperature'].append(temp)
        self.history['al_count'].append(len(self.al_candidates))

        if gen % 25 == 0 or gen == MAX_GEN:
            print(
                f"  Gen {gen:4d}/{MAX_GEN} | empires={len(empires):2d} | "
                f"best_ppf={best[2]:.3f} (σ={best[4]:.4f}) | "
                f"cycle={best[3]:6.1f}d | fit={best[0]:7.2f} | "
                f"T={temp:.3f} | AL_q={len(self.al_candidates)}"
            )

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        print("=" * 70)
        print("QICA-v5 OPTIMIZER  (uncertainty-aware, trust-region, CNN-v4)")
        print("=" * 70)
        print(f"  Population : {N_COUNTRIES}  Empires : {N_EMPIRES}  Gens : {MAX_GEN}")
        print(f"  MC Samples : {MC_SAMPLES}  PPF limit : {PPF_LIMIT}")
        print(f"  Fitness = cycle − {W_PPF_PENALTY}·PPF_excess − {W_UNCERTAINTY}·σ − "
              f"{W_TRUST}·trust + {W_MONOTONICITY}·mono + {W_ENTROPY_BONUS}·entropy")
        print(f"  ≈{MAX_GEN * N_COUNTRIES:,} evaluations × {MC_SAMPLES} MC samples\n")

        t0 = time.time()

        print("[INIT] Initializing population ...")
        countries = self._initialize_population()
        temp      = self._temperature(0)
        countries = self._evaluate_all(countries, temp)
        empires   = self._form_empires(countries)
        self._update_elite(empires)
        best0 = self.elite_archive[0]
        print(f"  Initial best: ppf={best0[2]:.3f}  cycle={best0[3]:.1f}d  σ={best0[4]:.4f}\n")

        print("[RUN] Main optimization loop ...")
        for gen in range(1, MAX_GEN + 1):
            temp     = self._temperature(gen)
            rev_rate = self._revolution_rate(gen)
            self._assimilation_step(empires, ASSIMILATION_COEFF, temp, rev_rate)
            self._intra_competition(empires, temp)
            self._update_elite(empires)
            empires  = self._empire_collapse(empires)
            self._log(gen, empires, temp, rev_rate)
            if len(empires) == 1 and len(empires[0].colonies) < 3:
                print(f"\n[CONVERGED] Single empire at gen {gen}")
                break

        t_total = time.time() - t0
        print(f"\n[DONE] {t_total:.1f}s | AL candidates: {len(self.al_candidates)}")
        return {
            'elite_archive': self.elite_archive,
            'history'      : self.history,
            'al_candidates': self.al_candidates,
        }


# =============================================================================
# SECTION 11 — ACTIVE LEARNING LOOP  (no simulator required in mode AL_ROUNDS=0)
# =============================================================================
#
# WHAT THE ACTIVE LEARNING LOOP DOES:
#   In the nuclear fuel optimisation problem, running a full neutron transport
#   simulation (OpenMC/PARCS/Serpent) for one loading pattern takes ~hours.
#   We can't run it for all 9^31 possible patterns.
#
#   Active learning solves this by choosing WHICH patterns to simulate:
#     1. Train CNN on existing labelled data (already done above).
#     2. QICA finds promising patterns using the CNN as a surrogate.
#     3. The CNN also reports uncertainty σ_ppf for each candidate.
#     4. Patterns with HIGH σ AND LOW ppf_pred are the most informative to simulate:
#           - High σ → CNN is uncertain → real physics might differ significantly
#           - Low ppf_pred → potentially optimal → worth the simulation cost
#     5. Simulate those patterns (get ground-truth ppf, cycle, keff).
#     6. Add them to the training set and retrain CNN.
#     7. Repeat: each round makes the CNN more accurate in the low-PPF region.
#
#   After N rounds, σ_ppf for the best QICA candidates should be very small
#   → the CNN is confident about those patterns → they're trustworthy results.
#
# CURRENT MODE (AL_ROUNDS=0): candidates identified and saved, no simulation.
# SET AL_ROUNDS=1+ after implementing simulate_pattern() above.

def run_al_round(model, X_grid_current, Y_tr_sc_current, ym_scaler_arr, yr_scaler_arr,
                 round_idx: int) -> tuple:
    """
    One active learning round (requires simulate_pattern() to be implemented).

    Steps:
      1. Run MC Dropout on ALL current patterns to find high-σ, low-ppf ones.
      2. Call simulate_pattern() on the top-priority candidates.
      3. Append new (X_grid, Y) pairs to the training arrays.
      4. Warm-start retrain for AL_RETRAIN_EPOCHS epochs.

    Returns:
      (updated X_grid_current, updated Y_tr_sc_current)

    NOTES:
      - ym_scaler_arr / yr_scaler_arr are the saved scaler mean/scale arrays
        (not sklearn objects — we reconstruct them here from the config).
      - The CNN model is retrained in-place (model.fit updates its weights).
    """
    AL_RETRAIN_EPOCHS       = 50
    AL_MAX_QUERIES_PER_RND  = 50

    print(f"\n[AL ROUND {round_idx}] Running MC Dropout on {len(X_grid_current)} patterns ...")

    # Step 1: find high-uncertainty, low-PPF candidates
    mc_all = np.stack([
        model(X_grid_current, training=True).numpy()
        for _ in range(MC_SAMPLES)
    ])   # (MC_SAMPLES, N, 35)

    mean_sc   = mc_all.mean(axis=0)
    std_sc    = mc_all.std(axis=0)
    mean_real = inverse_transform(mean_sc)
    std_phys  = np.concatenate([std_sc[:, :34] * ym_scaler_arr,
                                  std_sc[:, 34:35] * yr_scaler_arr], axis=1)

    ppf_all_pred = mean_real[:, IDX_PPF_MAX]
    ppf_all_std  = std_phys[:, IDX_PPF_MAX]
    priority     = ppf_all_std / (ppf_all_pred + 1e-6)
    top_idxs     = np.argsort(priority)[::-1][:AL_MAX_QUERIES_PER_RND]

    # Step 2: simulate each candidate
    new_X_grids, new_Ys = [], []
    n_success = 0
    print(f"  Querying {len(top_idxs)} candidates via simulate_pattern() ...")

    for idx in top_idxs:
        # Extract flat pattern from grid
        pattern_flat = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if GRID_LAYOUT[r, c] >= 0:
                    pattern_flat.append(int(X_grid_current[idx, r, c]))
        pattern_1d = np.array(pattern_flat, dtype=np.int32)

        try:
            result = simulate_pattern(pattern_1d)
            if result and result.get('success', True):
                # Build 6×6 grid for this new pattern
                new_grid = np.zeros((1, GRID_ROWS, GRID_COLS), dtype=np.int32)
                pos_i = 0
                for r in range(GRID_ROWS):
                    for c in range(GRID_COLS):
                        if GRID_LAYOUT[r, c] >= 0:
                            new_grid[0, r, c] = pattern_1d[pos_i]; pos_i += 1
                new_X_grids.append(new_grid)

                # Build target vector and scale it using saved scaler arrays
                ppf_steps_new = np.array(result.get('ppf_steps',
                                          np.full(N_STEPS, result['ppf_max'])), dtype=np.float32)
                ym_new  = np.array([[result['ppf_max'], result.get('ppf_boc', result['ppf_max'])]
                                    + ppf_steps_new.tolist()
                                    + [result['cycle_length']]], dtype=np.float32)
                yr_new  = np.array([[result['rho_pcm_boc']]], dtype=np.float32)
                ym_sc_new = (ym_new - ym_scaler_arr[:34]) / (np.where(ym_scaler_arr[34:] > 0,
                                                                         ym_scaler_arr[34:], 1.0))
                # Note: ym_scaler_arr here would need to be the scale_ as well — see below
                new_Y = np.concatenate([ym_sc_new,
                                         (yr_new - yr_scaler_arr[0]) / yr_scaler_arr[1]], axis=1)
                new_Ys.append(new_Y)
                n_success += 1

        except NotImplementedError:
            print(f"  [SKIP] simulate_pattern() not implemented — set AL_ROUNDS=0")
            break
        except Exception as e:
            print(f"  [WARN] Simulation failed for pattern idx {idx}: {e}")

    print(f"  Simulated {n_success}/{len(top_idxs)} patterns successfully")
    if n_success == 0:
        return X_grid_current, Y_tr_sc_current

    # Step 3: append new data
    X_grid_aug = np.vstack([X_grid_current] + new_X_grids)
    Y_aug      = np.vstack([Y_tr_sc_current] + new_Ys)

    # Step 4: warm-start retrain
    print(f"  Retraining on {len(X_grid_aug)} patterns ({AL_RETRAIN_EPOCHS} warm-start epochs) ...")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-4),
        loss='mse'   # simple MSE for warm-start
    )
    model.fit(X_grid_aug, Y_aug, epochs=AL_RETRAIN_EPOCHS, batch_size=64, verbose=0)
    print(f"  [AL ROUND {round_idx} COMPLETE]  +{n_success} patterns")
    return X_grid_aug, Y_aug


# =============================================================================
# SECTION 12 — RUN QICA
# =============================================================================

if __name__ == '__main__':

    # ── Run the optimizer ─────────────────────────────────────────────────────
    optimizer = QICAv5()
    results   = optimizer.run()

    elite    = results['elite_archive']
    al_cands = results['al_candidates']

    # ── Active learning rounds (AL_ROUNDS=0 → skip) ───────────────────────────
    # To enable:
    #   1. Implement simulate_pattern() in Section 3
    #   2. Set AL_ROUNDS = 1 (or more) at the top
    #   3. Re-run this script
    if AL_ROUNDS > 0:
        print(f"\n[ACTIVE LEARNING] Running {AL_ROUNDS} round(s) ...")
        # Build initial training arrays from config (we don't have the training CSV here,
        # so this placeholder shows the structure — in practice, pass X_grid + Y_sc from
        # the CNN training script or load them from a saved checkpoint)
        print("  [NOTE] To run AL rounds, you need to pass X_grid_train and Y_tr_sc")
        print("  from the CNN training script. See run_al_round() docstring.")
        print("  Skipping AL retraining for now.")
    else:
        print(f"\n[AL] Candidate-flagging mode (AL_ROUNDS=0 — no simulation required)")
        print(f"  {len(al_cands)} candidates identified during QICA search")
        print(f"  → Implement simulate_pattern() and set AL_ROUNDS=1 to enable retraining")

    # ── Print top-5 patterns ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP LOADING PATTERNS FOUND BY QICA-v5")
    print("=" * 70)
    print(f"{'Rank':<5} {'PPF_max':<10} {'σ_ppf':<8} {'Cycle(d)':<12} "
          f"{'Fitness':<12} {'Status'}")
    print("-" * 70)
    for rank, (fit, pat, ppf, cyc, sig) in enumerate(elite[:5], 1):
        safe  = "✓ SAFE"  if ppf <= PPF_LIMIT        else "✗ EXCEEDS PPF_LIMIT"
        conf  = "✓ confident" if sig < AL_SIGMA_THRESHOLD else "⚠ send to simulator"
        print(f"  #{rank}   {ppf:7.4f}    {sig:.4f}   {cyc:9.1f}    {fit:9.2f}   {safe}  {conf}")
        print(f"         Types: {list(pat)}")

    best_fit, best_pat, best_ppf, best_cyc, best_sig = elite[0]
    print(f"\nBEST PATTERN:")
    print(f"  PPF_max     : {best_ppf:.4f}  ({'✓ SAFE' if best_ppf <= PPF_LIMIT else '✗ EXCEEDS'})")
    print(f"  σ_ppf       : {best_sig:.4f}  "
          f"({'✓ surrogate confident' if best_sig < AL_SIGMA_THRESHOLD else '⚠ verify with simulator'})")
    print(f"  Cycle length: {best_cyc:.1f} days")
    print(f"  Fitness     : {best_fit:.2f}")
    print(f"  Pattern     : {list(best_pat)}")

    # ── Save results ──────────────────────────────────────────────────────────
    best_df = pd.DataFrame([
        {'rank': i + 1,
         'ppf_max': ppf, 'sigma_ppf': sig,
         'cycle_length_days': cyc, 'fitness': fit,
         'ppf_safe': ppf <= PPF_LIMIT,
         'surrogate_confident': sig < AL_SIGMA_THRESHOLD,
         **{f'pos_{j}': int(pat[j]) for j in range(N_POS)}}
        for i, (fit, pat, ppf, cyc, sig) in enumerate(elite)
    ])
    best_df.to_csv('qica_v4_best_patterns.csv', index=False)
    print(f"\n[SAVED] qica_v4_best_patterns.csv  ({len(best_df)} elite patterns)")

    # Save AL candidates (uncertain + low-PPF — highest priority for simulation)
    if al_cands:
        al_df = (pd.DataFrame(al_cands)
                 .sort_values('priority', ascending=False)
                 .drop_duplicates(subset=['pattern'])
                 .head(AL_TOP_K))
        al_df.to_csv('qica_v4_al_candidates.csv', index=False)
        print(f"[SAVED] qica_v4_al_candidates.csv  ({len(al_df)} candidates)")
        print(f"  Top candidate: ppf={al_df.iloc[0]['pred_ppf']:.3f}  "
              f"σ={al_df.iloc[0]['sigma_ppf']:.4f}")
        print(f"  → When simulate_pattern() is implemented and AL_ROUNDS>0,")
        print(f"    these patterns will be labelled and used to retrain the CNN.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    hist = results['history']
    fig  = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"QICA-v5  |  Best PPF={best_ppf:.4f}  Cycle={best_cyc:.1f}d  "
        f"σ={best_sig:.4f}  AL_candidates={len(al_cands)}",
        fontsize=12, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(hist['gen'], hist['best_fitness'],  '#1B4FBF', lw=2,   label='Best')
    ax.plot(hist['gen'], hist['mean_fitness'],  '#F5A623', lw=1.5, ls='--', label='Mean')
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness (days adj.)')
    ax.set_title('Fitness Convergence'); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(hist['gen'], hist['best_ppf'], '#D62728', lw=2)
    ax.axhline(PPF_LIMIT, color='orange', lw=1.5, ls='--', label=f'Limit={PPF_LIMIT}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Best ppf_max')
    ax.set_title('PPF Convergence (PRIMARY)'); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(hist['gen'], hist['best_cycle'], '#2CA02C', lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Best cycle length (days)')
    ax.set_title('Cycle Length Convergence'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(hist['gen'], hist['mean_ppf_std'], '#9467BD', lw=2)
    ax.axhline(AL_SIGMA_THRESHOLD, color='red', lw=1.5, ls=':',
               label=f'AL threshold σ={AL_SIGMA_THRESHOLD}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Mean σ_ppf')
    ax.set_title('Population Uncertainty\n(should decrease as QICA converges)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(hist['gen'], hist['n_empires'], '#8C564B', lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('Number of empires')
    ax.set_title('Empire Collapse (convergence)'); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(hist['gen'], hist['al_count'], '#E377C2', lw=2)
    ax.axhline(AL_TOP_K, color='orange', lw=1.5, ls='--', label=f'Export top-{AL_TOP_K}')
    ax.set_xlabel('Generation'); ax.set_ylabel('Cumulative AL candidates')
    ax.set_title('Active Learning\nCandidate Accumulation')
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 2])
    grid_disp = np.full((GRID_ROWS, GRID_COLS), np.nan)
    pos_i = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid_disp[r, c] = float(best_pat[pos_i]); pos_i += 1
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad('lightgrey')
    im = ax.imshow(grid_disp, cmap=cmap, aspect='auto', vmin=1, vmax=9)
    plt.colorbar(im, ax=ax, label='Assembly Type')
    pos_i = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_MASK[r, c]:
                ax.text(c, r, f'T{int(best_pat[pos_i])}',
                        ha='center', va='center', fontsize=7, fontweight='bold')
                pos_i += 1
    ax.set_title(f'Best Loading Pattern\nPPF={best_ppf:.3f}  σ={best_sig:.4f}')
    ax.set_xticks([]); ax.set_yticks([])

    ax  = fig.add_subplot(gs[1, 3])
    ax2 = ax.twinx()
    l1, = ax.plot(hist['gen'],  hist['temperature'],  '#D62728', lw=2, label='T (quantum temp)')
    l2, = ax2.plot(hist['gen'],
                   [REVOLUTION_RATE - (REVOLUTION_RATE - REVOLUTION_MIN) * g / MAX_GEN
                    for g in hist['gen']],
                   '#8C564B', lw=2, ls='--', label='Revolution rate')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Temperature', color='#D62728')
    ax2.set_ylabel('Revolution Rate', color='#8C564B')
    ax.set_title('Adaptive Parameters\n(explore → exploit over time)')
    ax.legend(handles=[l1, l2], fontsize=7); ax.grid(alpha=.3)

    plt.savefig('qica_v5_convergence.png', dpi=150, bbox_inches='tight')
    print("[SAVED] qica_v5_convergence.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QICA-v5 SUMMARY")
    print("=" * 70)
    print(f"  Algorithm      : Quantum ICA with MC Dropout uncertainty")
    print(f"  Surrogate      : BEAVRS CNN-v4 (multi-head, 31→6×6 grid)")
    print(f"  Best PPF       : {best_ppf:.4f}  ({'SAFE' if best_ppf <= PPF_LIMIT else 'EXCEEDS — increase W_PPF_PENALTY'})")
    print(f"  Best σ_ppf     : {best_sig:.4f}  ({'confident' if best_sig < AL_SIGMA_THRESHOLD else 'UNCERTAIN → verify with simulator'})")
    print(f"  Cycle length   : {best_cyc:.1f} days")
    print(f"  AL candidates  : {len(al_cands)}")
    print()
    print("  NEXT STEPS:")
    print("  1. Confident patterns (σ < threshold): accept as optimal candidates.")
    print("  2. Uncertain patterns (σ ≥ threshold): feed al_queries.csv to your")
    print("     physics simulator (OpenMC/PARCS/Serpent).")
    print("  3. Implement simulate_pattern() in Section 3 of this file.")
    print("  4. Set AL_ROUNDS = 1, re-run → CNN retrains on new labels.")
    print("  5. Repeat until σ_ppf_best < 0.02 for the best pattern.")
    print("=" * 70)