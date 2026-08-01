"""
=============================================================================
al_loop.py  —  AL Coordinator for BEAVRS CNN + QICA + OpenMC
=============================================================================
RENAMED from al-loop.py → al_loop.py  (hyphens are fine for python al_loop.py
but this is cleaner and consistent with the rest of the pipeline).

FIXES vs original al-loop.py:
  FIX 1 — FREQ_PATH now checks train_type_freq_v9.npy first (v9 is current model)
  FIX 2 — _find_model / _find_config already prioritise v9 correctly (no change needed)
  FIX 3 — openmc_beavrs_fast is now the correct import name (file renamed from
           openmc-fast.py which Python cannot import due to the hyphen)

WHAT THIS DOES
  Runs one or more rounds of active learning:
    1. Loads your trained CNN (cnn_v9_model.keras)
    2. Scans training dataset with MC Dropout to find uncertain low-PPF patterns
    3. Simulates each with OpenMC to get ground-truth PPF/cycle/keff
    4. Appends new rows to ml_dataset_al.csv  ← NEW file, original NEVER touched
    5. Warm-start retrains the CNN on original + all AL data so far
    6. Saves the retrained model as cnn_al_round_N.keras
    7. Repeats for the next round

DATASET POLICY
  ml_dataset_constrained.csv   ← NEVER modified (original, read-only)
  ml_dataset_al.csv            ← grows with each round  (new patterns only)
  Training uses both in memory: pd.concat([original_df, al_df])

TUNING PARAMS (change these at the top of SECTION 1)
  AL_ROUNDS           : number of rounds to run
  PATTERNS_PER_ROUND  : OpenMC simulations per round (start with 5-10)
  OPENMC_SPEED        : 'debug'|'fast'|'balanced'|'accurate'
  CNN_RETRAIN_EPOCHS  : epochs to warm-start retrain after each round

ESTIMATED TIME PER ROUND
  speed='debug',   patterns=5  : ~5-10  min
  speed='fast',    patterns=5  : ~15-25 min
  speed='fast',    patterns=10 : ~30-50 min
  speed='balanced',patterns=10 : ~2-3 hrs

RESUMING
  Progress is saved to al_progress.json after each round.
  Re-run the script to continue from where you left off.
  Delete al_progress.json to restart from scratch.

USAGE
  python al_loop.py             # run with defaults
  python al_loop.py --rounds 1  # single round
  python al_loop.py --patterns 5 --speed debug  # quick test
=============================================================================
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(f"TensorFlow {tf.__version__}")
print("al_loop.py  —  AL Coordinator\n")


# =============================================================================
# SECTION 1 — CONFIGURATION  (edit these)
# =============================================================================

# ── Dataset files ─────────────────────────────────────────────────────────────
ORIGINAL_CSV   = 'ml_dataset_constrained.csv'   # NEVER touched
AL_CSV         = 'ml_dataset_al.csv'             # grows with each AL round
PROGRESS_FILE  = 'al_progress.json'

# ── Model files ───────────────────────────────────────────────────────────────
# Priority order: v9 > v8 > v4 (cnn_v9.py output is cnn_v9_model.keras)
def _find_model():
    for name in ['cnn_v9_model.keras', 'cnn_v8_model.keras', 'cnn_v4_model.keras']:
        if os.path.exists(name):
            return name
    return None

def _find_config():
    for name in ['cnn_v9_config.json', 'cnn_v8_config.json', 'cnn_v4_config.json']:
        if os.path.exists(name):
            return name
    return None

BASE_MODEL_PATH  = _find_model()
BASE_CONFIG_PATH = _find_config()

# FIX 1: Check v9 trust-region file first (cnn_v9.py saves train_type_freq_v9.npy)
FREQ_PATH = (
    'train_type_freq_v9.npy' if os.path.exists('train_type_freq_v9.npy') else
    'train_type_freq_v8.npy' if os.path.exists('train_type_freq_v8.npy') else
    'train_type_freq.npy'
)

# ── Active learning settings ──────────────────────────────────────────────────
AL_ROUNDS          = 3       # total rounds to run
PATTERNS_PER_ROUND = 5       # OpenMC simulations per round (start small!)
OPENMC_SPEED       = 'fast'  # 'debug' | 'fast' | 'balanced' | 'accurate'
UNCERTAINTY_THRESH = 0.08    # σ_ppf threshold for candidate selection
MIN_PPF_PERCENTILE = 25      # only consider patterns in bottom X% of predicted PPF

# ── CNN retraining settings ───────────────────────────────────────────────────
CNN_RETRAIN_EPOCHS = 50      # warm-start epochs per round (not full retrain)
CNN_RETRAIN_LR     = 5e-5    # lower LR for warm-start

# ── QICA settings for candidate finding ──────────────────────────────────────
QICA_N_COUNTRIES  = 60       # population (fewer = faster candidate search)
QICA_MAX_GEN      = 100      # generations for candidate search
QICA_MC_SAMPLES   = 20       # MC Dropout passes per evaluation
QICA_AL_TOP_K     = 30       # candidates to collect from QICA run

# ── Geometry (must match cnn_v9.py exactly) ───────────────────────────────────
N_POS, N_TYPES, N_STEPS = 31, 9, 31
GRID_ROWS, GRID_COLS    = 6, 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)
SEED = 42


# =============================================================================
# SECTION 2 — CONVRESBLOCK  (must match cnn_v9.py exactly for model loading)
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
        self._filters = filters; self._dropout_rate = dropout
        self.dropout_layer = layers.Dropout(dropout) if dropout > 0 else None
        self.proj = None

    def build(self, input_shape):
        if input_shape[-1] != self._filters:
            self.proj = layers.Conv2D(self._filters, 1, padding='same')
        super().build(input_shape)

    def call(self, x, training=False):
        s = self.proj(x) if self.proj is not None else x
        h = tf.nn.gelu(self.bn1(self.conv1(x), training=training))
        h = self.bn2(self.conv2(h), training=training)
        h = tf.nn.gelu(h + s)
        if self.dropout_layer:
            h = self.dropout_layer(h, training=training)
        return h

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self._filters, 'kernel_size': 3,
                    'dropout': self._dropout_rate})
        return cfg


# =============================================================================
# SECTION 3 — LOAD CNN + CONFIG
# =============================================================================

def load_cnn_and_config(model_path: str = None, config_path: str = None) -> tuple:
    """Load CNN model and scalers from config JSON."""
    mpath = model_path or BASE_MODEL_PATH
    cpath = config_path or BASE_CONFIG_PATH

    if mpath is None or not os.path.exists(mpath):
        print(f"[ERROR] No CNN model found. Run cnn_v9.py first.")
        sys.exit(1)

    if cpath is None or not os.path.exists(cpath):
        print(f"[ERROR] No CNN config found. Run cnn_v9.py first.")
        sys.exit(1)

    # compile=False avoids needing v9_loss defined here — inference only
    model = keras.models.load_model(mpath, compile=False,
                                     custom_objects={'ConvResBlock': ConvResBlock})
    with open(cpath) as f:
        cfg = json.load(f)

    ym_mean  = np.array(cfg['ym_scaler_mean'],  dtype=np.float32)
    ym_scale = np.array(cfg['ym_scaler_scale'], dtype=np.float32)
    yr_mean  = np.array(cfg['yr_scaler_mean'],  dtype=np.float32)
    yr_scale = np.array(cfg['yr_scaler_scale'], dtype=np.float32)

    indices = {k: cfg[k] for k in ['IDX_PPF_MAX', 'IDX_PPF_BOC',
                                     'IDX_PPF_STEPS_START', 'IDX_PPF_STEPS_END',
                                     'IDX_CYCLE', 'IDX_RHO', 'N_OUTPUTS']}
    print(f"[CNN] Loaded: {mpath}  input={model.input_shape}  output={model.output_shape}")
    return model, ym_mean, ym_scale, yr_mean, yr_scale, indices


# =============================================================================
# SECTION 4 — DATA LOADING
# =============================================================================

def load_original_dataset() -> pd.DataFrame:
    """Load original dataset (read-only)."""
    if not os.path.exists(ORIGINAL_CSV):
        print(f"[ERROR] {ORIGINAL_CSV} not found."); sys.exit(1)
    df = pd.read_csv(ORIGINAL_CSV, skiprows=1, engine='python', on_bad_lines='skip')
    print(f"[DATA] Original: {len(df)} patterns × {df.shape[1]} columns  (read-only)")
    return df


def load_al_dataset() -> pd.DataFrame:
    """Load accumulated AL data, or return empty frame if none yet."""
    if os.path.exists(AL_CSV):
        df = pd.read_csv(AL_CSV)
        print(f"[DATA] AL dataset: {len(df)} patterns  ({AL_CSV})")
        return df
    print(f"[DATA] No AL data yet — starting fresh")
    return pd.DataFrame()


def build_grid_array(df: pd.DataFrame) -> np.ndarray:
    """Convert loading columns to (N, 6, 6) grid array."""
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    X_raw  = df[load_cols].values.astype(np.int32)
    X_grid = np.zeros((len(df), GRID_ROWS, GRID_COLS), dtype=np.int32)
    pi = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                X_grid[:, r, c] = X_raw[:, pi]; pi += 1
    return X_grid


def build_targets(df: pd.DataFrame) -> tuple:
    """Extract Y_main (34 cols) and Y_rho (1 col) from dataframe."""
    react_cols   = sorted([c for c in df.columns if c.startswith('react_')],
                          key=lambda c: int(c.split('_')[1]))
    ppf_steps_s  = sorted(set(int(c.split('_')[1][1:]) for c in df.columns if c.startswith('ppf_')))
    ppf_assembs  = sorted(set(int(c.split('_')[2][1:]) for c in df.columns if c.startswith('ppf_')))

    step_max = np.stack([
        df[[f'ppf_s{s}_a{i}' for i in ppf_assembs if f'ppf_s{s}_a{i}' in df.columns
            ]].values.astype(np.float32).max(axis=1)
        for s in ppf_steps_s
    ], axis=1)

    ppf_global_max = step_max.max(axis=1)
    ppf_boc        = step_max[:, 0]
    keff_raw       = (1.0 / (1.0 - df[react_cols[0]].values)).astype(np.float32)
    rho_pcm        = ((keff_raw - 1.0) / keff_raw * 1e5).astype(np.float32)

    Y_main = np.concatenate([
        ppf_global_max.reshape(-1,1),
        ppf_boc.reshape(-1,1),
        step_max,
        df['cycle_length'].values.reshape(-1,1)
    ], axis=1).astype(np.float32)

    Y_rho = rho_pcm.reshape(-1,1).astype(np.float32)
    return Y_main, Y_rho


# =============================================================================
# SECTION 5 — MC DROPOUT CANDIDATE FINDER
# =============================================================================

def find_candidates_mc_dropout(
        model, X_grid_all: np.ndarray,
        ym_mean, ym_scale, yr_mean, yr_scale, indices,
        ppf_true_all: np.ndarray = None,
        n_mc: int = QICA_MC_SAMPLES,
        top_k: int = QICA_AL_TOP_K) -> list:
    """
    Scan the existing dataset with MC Dropout to find the most uncertain
    low-PPF patterns.  Much faster than running QICA.

    Returns:
        list of dicts with dataset_idx, pred_ppf, sigma_ppf, entropy_ppf, priority
    """
    IDX_PPF = indices['IDX_PPF_MAX']
    print(f"  [Candidates] MC Dropout scan ({n_mc} passes, {len(X_grid_all)} patterns)...")
    t0 = time.time()

    X_tf  = tf.constant(X_grid_all, dtype=tf.int32)
    mc_sc = np.stack([model(X_tf, training=True).numpy() for _ in range(n_mc)])

    ppf_mean_sc = mc_sc[:, :, IDX_PPF].mean(axis=0)
    ppf_std_sc  = mc_sc[:, :, IDX_PPF].std(axis=0)
    ppf_pred    = ppf_mean_sc * ym_scale[IDX_PPF] + ym_mean[IDX_PPF]
    ppf_std     = ppf_std_sc  * ym_scale[IDX_PPF]

    # Gaussian differential entropy  H = 0.5 * log(2πe σ²)
    ppf_entropy = 0.5 * np.log(2 * np.pi * np.e * (ppf_std + 1e-10)**2)

    # Priority: high uncertainty AND low predicted PPF
    ppf_25pct    = np.percentile(ppf_pred, MIN_PPF_PERCENTILE)
    low_ppf_mask  = (ppf_pred <= ppf_25pct)
    high_unc_mask = (ppf_std >= UNCERTAINTY_THRESH)
    candidate_mask = low_ppf_mask & high_unc_mask

    print(f"  [Candidates] High-σ: {high_unc_mask.sum()}  Low-PPF: {low_ppf_mask.sum()}  "
          f"Combined: {candidate_mask.sum()}  ({time.time()-t0:.1f}s)")

    priority  = ppf_entropy / (ppf_pred + 1e-6)
    cand_idx  = np.where(candidate_mask)[0]

    if len(cand_idx) == 0:
        # Fallback: take highest uncertainty patterns regardless of PPF
        print("  [WARN] No combined candidates — using top-σ patterns as fallback")
        cand_idx = np.argsort(ppf_std)[::-1][:top_k]
    else:
        cand_idx  = cand_idx[np.argsort(priority[cand_idx])[::-1]][:top_k]

    result = [
        {
            'dataset_idx': int(i),
            'pred_ppf'   : float(ppf_pred[i]),
            'sigma_ppf'  : float(ppf_std[i]),
            'entropy_ppf': float(ppf_entropy[i]),
            'priority'   : float(priority[i]),
        }
        for i in cand_idx
    ]
    print(f"  [Candidates] Top candidate: ppf={result[0]['pred_ppf']:.3f}  "
          f"σ={result[0]['sigma_ppf']:.4f}")
    return result


# =============================================================================
# SECTION 6 — BUILD NEW DATA ROW (convert OpenMC result → CSV row)
# =============================================================================

def result_to_row(pattern_1d: np.ndarray, result: dict,
                   original_df: pd.DataFrame) -> dict:
    """
    Convert an OpenMC result dict to a row matching ml_dataset_constrained.csv schema.
    PPF columns stored as ppf_s{step}_a0 (one assembly per step = max PPF at that step).
    """
    react_cols = sorted([c for c in original_df.columns if c.startswith('react_')],
                        key=lambda c: int(c.split('_')[1]))

    keff_boc = result['keff_boc']
    rho_frac = (keff_boc - 1.0) / keff_boc if keff_boc > 0 else 0.0

    row = {}
    for j, pos in enumerate(pattern_1d):
        row[f'loading_{j}'] = int(pos)

    if react_cols:
        row[react_cols[0]] = float(rho_frac)
        for rc in react_cols[1:]:
            row[rc] = float(rho_frac * 0.9)

    ppf_steps = result['ppf_steps']
    for step_i, ppf_val in enumerate(ppf_steps):
        row[f'ppf_s{step_i}_a0'] = float(ppf_val)

    row['cycle_length'] = float(result['cycle_length'])
    row['al_round']     = -1
    row['omc_speed']    = OPENMC_SPEED
    return row


# =============================================================================
# SECTION 7 — CNN RETRAINING
# =============================================================================

def retrain_cnn_warm_start(
        model, X_grid_combined, Y_main_combined, Y_rho_combined,
        ym_scaler, yr_scaler, round_idx: int) -> tuple:
    """
    Warm-start retrain the CNN on original + all AL data accumulated so far.
    Uses a lower LR and fewer epochs than the initial training.
    Uses MSE loss (simpler than v9_loss for warm-start).
    """
    print(f"\n  [Retrain] Combined dataset: {len(X_grid_combined)} patterns")
    print(f"  [Retrain] Warm-start: {CNN_RETRAIN_EPOCHS} epochs, LR={CNN_RETRAIN_LR}")

    Ym_sc = ym_scaler.transform(Y_main_combined)
    Yr_sc = yr_scaler.transform(Y_rho_combined)
    Y_sc  = np.concatenate([Ym_sc, Yr_sc], axis=1).astype(np.float32)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=CNN_RETRAIN_LR,
                                          weight_decay=1e-4),
        loss='mse'   # simpler MSE for warm-start; v9_loss not needed here
    )

    history = model.fit(
        X_grid_combined, Y_sc,
        validation_split=0.1,
        epochs=CNN_RETRAIN_EPOCHS,
        batch_size=128,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                           restore_best_weights=True, verbose=0),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda ep, logs: print(
                    f"    Ep {ep+1:3d} | loss: {logs['loss']:.5f} | val: {logs['val_loss']:.5f}"
                ) if (ep+1) % 10 == 0 else None
            ),
        ]
    )

    model_path = f'cnn_al_round_{round_idx}.keras'
    model.save(model_path)
    print(f"  [Retrain] Saved: {model_path}  (best val_loss at ep "
          f"{int(np.argmin(history.history['val_loss']))+1})")
    return model, history


# =============================================================================
# SECTION 8 — PROGRESS TRACKING
# =============================================================================

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            p = json.load(f)
        print(f"[PROGRESS] Resuming from round {p['last_completed_round']+1}")
        return p
    return {'last_completed_round': 0, 'rounds': {}, 'total_al_patterns': 0}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# SECTION 9 — MAIN AL LOOP
# =============================================================================

def run_active_learning(n_rounds: int = AL_ROUNDS,
                         patterns_per_round: int = PATTERNS_PER_ROUND,
                         openmc_speed: str = OPENMC_SPEED):

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    # FIX 3: import from openmc_beavrs_fast (the renamed file, no hyphen)
    try:
        import openmc
        from openmc_beavrs_fast import simulate_pattern as omc_simulate
        from openmc_beavrs_fast import _APPROX_TIME_MIN
        print("[OpenMC] Available ✓")
    except ImportError as e:
        print(f"[ERROR] OpenMC or openmc_beavrs_fast.py not available: {e}")
        print("  Make sure openmc_beavrs_fast.py is in the same directory")
        print("  (renamed from openmc-fast.py — Python cannot import hyphen filenames)")
        print("  Install OpenMC: conda install -c conda-forge openmc")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"ACTIVE LEARNING LOOP")
    print(f"{'='*65}")
    print(f"  Original dataset : {ORIGINAL_CSV}  (read-only)")
    print(f"  AL dataset       : {AL_CSV}  (grows each round)")
    print(f"  Rounds           : {n_rounds}")
    print(f"  Patterns/round   : {patterns_per_round}")
    print(f"  OpenMC speed     : {openmc_speed}")
    est_total = _APPROX_TIME_MIN[openmc_speed] * patterns_per_round * n_rounds
    print(f"  Estimated total  : ~{est_total:.0f} min  ({est_total/60:.1f} hrs)")
    print(f"{'='*65}\n")

    # ── Load data + model ─────────────────────────────────────────────────────
    progress   = load_progress()
    orig_df    = load_original_dataset()
    al_df      = load_al_dataset()

    model, ym_mean, ym_scale, yr_mean, yr_scale, indices = load_cnn_and_config()

    # Build scalers from original data (fit once, reuse every round)
    orig_Y_main, orig_Y_rho = build_targets(orig_df)
    orig_X_grid              = build_grid_array(orig_df)

    ym_scaler = StandardScaler().fit(orig_Y_main)
    yr_scaler = StandardScaler().fit(orig_Y_rho)

    start_round = progress['last_completed_round'] + 1

    for round_idx in range(start_round, start_round + n_rounds):

        print(f"\n{'─'*65}")
        print(f"AL ROUND {round_idx}  |  "
              f"AL data so far: {len(al_df)} patterns  |  "
              f"Original: {len(orig_df)} patterns")
        print(f"{'─'*65}")
        t_round = time.time()

        # ── Step 1: Find candidates ───────────────────────────────────────────
        print(f"\n  Step 1 / 4 — Finding uncertain low-PPF candidates...")
        candidates = find_candidates_mc_dropout(
            model, orig_X_grid, ym_mean, ym_scale, yr_mean, yr_scale, indices,
            top_k=QICA_AL_TOP_K
        )

        # De-duplicate against already-simulated patterns
        already_done = set()
        if len(al_df) > 0:
            load_cols = [f'loading_{i}' for i in range(N_POS)]
            for _, row in al_df.iterrows():
                pat_key = tuple(int(row[c]) for c in load_cols if c in al_df.columns)
                already_done.add(pat_key)

        new_candidates = []
        for c in candidates:
            i  = c['dataset_idx']
            pt = tuple(int(orig_df.iloc[i][f'loading_{j}']) for j in range(N_POS))
            if pt not in already_done:
                new_candidates.append(c)
            if len(new_candidates) >= patterns_per_round:
                break

        print(f"  {len(new_candidates)} new (unsimulated) candidates selected")
        if len(new_candidates) == 0:
            print("  No new candidates — all top candidates already simulated. "
                  "Consider increasing QICA_AL_TOP_K.")
            break

        # ── Step 2: Run OpenMC simulations ────────────────────────────────────
        print(f"\n  Step 2 / 4 — Running {len(new_candidates)} OpenMC simulations "
              f"(speed='{openmc_speed}')...")

        new_rows = []
        n_success = 0
        for sim_i, cand in enumerate(new_candidates):
            idx     = cand['dataset_idx']
            pat_arr = np.array([int(orig_df.iloc[idx][f'loading_{j}'])
                                for j in range(N_POS)], dtype=np.int32)
            print(f"\n  Simulation {sim_i+1}/{len(new_candidates)}: "
                  f"idx={idx}  pred_ppf={cand['pred_ppf']:.3f}  σ={cand['sigma_ppf']:.4f}")

            result = omc_simulate(pat_arr, speed_mode=openmc_speed, verbose=True)

            if result['success']:
                row = result_to_row(pat_arr, result, orig_df)
                row['al_round'] = round_idx
                row['src_idx']  = idx
                row['pred_ppf_before'] = cand['pred_ppf']
                row['pred_sigma']      = cand['sigma_ppf']
                new_rows.append(row)
                n_success += 1
                print(f"  ✓ PPF_max={result['ppf_max']:.4f}  "
                      f"cycle={result['cycle_length']:.1f}d  "
                      f"keff={result['keff_boc']:.5f}")
            else:
                print(f"  ✗ Simulation failed — skipping")

        print(f"\n  {n_success}/{len(new_candidates)} simulations succeeded")
        if n_success == 0:
            print("  No successful simulations this round. Check OpenMC setup.")
            continue

        # ── Step 3: Append to AL dataset ──────────────────────────────────────
        print(f"\n  Step 3 / 4 — Appending {n_success} rows to {AL_CSV}...")
        new_al_df = pd.DataFrame(new_rows)

        if len(al_df) == 0:
            al_df = new_al_df
        else:
            al_df = pd.concat([al_df, new_al_df], ignore_index=True)

        al_df.to_csv(AL_CSV, index=False)
        print(f"  AL dataset now has {len(al_df)} patterns  ({AL_CSV})")

        # ── Step 4: Retrain CNN ───────────────────────────────────────────────
        print(f"\n  Step 4 / 4 — Warm-start retraining CNN...")

        X_grid_al   = build_grid_array(al_df)
        Y_main_al   = np.zeros((len(al_df), 34), dtype=np.float32)
        Y_rho_al    = np.zeros((len(al_df), 1),  dtype=np.float32)

        for al_i, (_, row) in enumerate(al_df.iterrows()):
            ppf_steps_al = np.array([float(row.get(f'ppf_s{s}_a0', 3.0))
                                      for s in range(N_STEPS)], dtype=np.float32)
            ppf_max_al   = float(ppf_steps_al.max())
            ppf_boc_al   = float(ppf_steps_al[0])
            cycle_al     = float(row.get('cycle_length', 350.0))
            react_al     = float(row.get(f'react_0', 0.0))
            keff_al      = 1.0 / (1.0 - react_al) if abs(react_al) < 1 else 1.02
            rho_al       = (keff_al - 1.0) / keff_al * 1e5

            Y_main_al[al_i, 0]    = ppf_max_al
            Y_main_al[al_i, 1]    = ppf_boc_al
            Y_main_al[al_i, 2:33] = ppf_steps_al
            Y_main_al[al_i, 33]   = cycle_al
            Y_rho_al[al_i, 0]     = rho_al

        X_combined      = np.concatenate([orig_X_grid, X_grid_al], axis=0)
        Y_main_combined = np.concatenate([orig_Y_main, Y_main_al], axis=0)
        Y_rho_combined  = np.concatenate([orig_Y_rho,  Y_rho_al],  axis=0)

        model, hist = retrain_cnn_warm_start(
            model, X_combined, Y_main_combined, Y_rho_combined,
            ym_scaler, yr_scaler, round_idx
        )

        # Quick eval on original test set to track improvement
        _, _, X_test, _, _, Ym_test, _, _, Yr_test = _quick_split(
            orig_X_grid, orig_Y_main, orig_Y_rho)
        Y_pred_sc   = model.predict(X_test, verbose=0)
        ppf_pred_r  = Y_pred_sc[:, 0] * ym_scaler.scale_[0] + ym_scaler.mean_[0]
        ppf_true_r  = Ym_test[:, 0]
        rel_err     = (np.abs(ppf_pred_r - ppf_true_r) / (ppf_true_r + 1e-6)).mean() * 100

        t_round_min = (time.time() - t_round) / 60
        print(f"\n  Round {round_idx} complete  ({t_round_min:.1f} min)")
        print(f"    New patterns simulated : {n_success}")
        print(f"    Total AL patterns      : {len(al_df)}")
        print(f"    PPF relative error     : {rel_err:.2f}%  (on original test set)")

        # ── Save progress ─────────────────────────────────────────────────────
        progress['last_completed_round'] = round_idx
        progress['total_al_patterns']    = len(al_df)
        progress['rounds'][str(round_idx)] = {
            'n_simulated'    : n_success,
            'ppf_rel_err'    : round(rel_err, 3),
            'total_al'       : len(al_df),
            'time_min'       : round(t_round_min, 1),
            'model_saved'    : f'cnn_al_round_{round_idx}.keras',
        }
        save_progress(progress)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"ACTIVE LEARNING COMPLETE")
    print(f"{'='*65}")
    print(f"  Total rounds run   : {progress['last_completed_round']}")
    print(f"  Total AL patterns  : {progress['total_al_patterns']}")
    print(f"  Original dataset   : {ORIGINAL_CSV}  (unchanged)")
    print(f"  AL data file       : {AL_CSV}")
    print()
    for rnd, info in progress.get('rounds', {}).items():
        print(f"  Round {rnd}: {info['n_simulated']} simulated  "
              f"PPF_err={info['ppf_rel_err']:.2f}%  ({info['time_min']:.1f}min)")
    print()
    print(f"  Best model: cnn_al_round_{progress['last_completed_round']}.keras")
    print(f"  Next step: run qica_v5.py — it auto-detects cnn_v9_model.keras")
    print(f"  After AL: point qica_v5.py at cnn_al_round_N.keras manually if desired")
    print(f"{'='*65}")


def _quick_split(X, Ym, Yr):
    """Same split as training (seed=42, 15% test)."""
    X_tr, X_tmp, Ym_tr, Ym_tmp, Yr_tr, Yr_tmp = train_test_split(
        X, Ym, Yr, test_size=0.30, random_state=SEED)
    X_val, X_test, Ym_val, Ym_test, Yr_val, Yr_test = train_test_split(
        X_tmp, Ym_tmp, Yr_tmp, test_size=0.5, random_state=SEED)
    return X_tr, X_val, X_test, Ym_tr, Ym_val, Ym_test, Yr_tr, Yr_val, Yr_test


# =============================================================================
# SECTION 10 — CLI
# =============================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Active learning coordinator')
    p.add_argument('--rounds',   type=int, default=AL_ROUNDS,
                   help=f'Number of AL rounds (default: {AL_ROUNDS})')
    p.add_argument('--patterns', type=int, default=PATTERNS_PER_ROUND,
                   help=f'OpenMC sims per round (default: {PATTERNS_PER_ROUND})')
    p.add_argument('--speed',    default=OPENMC_SPEED,
                   choices=['debug','fast','balanced','accurate'],
                   help=f'OpenMC speed mode (default: {OPENMC_SPEED})')
    p.add_argument('--reset',    action='store_true',
                   help='Delete al_progress.json and restart from scratch')
    args = p.parse_args()

    if args.reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"[RESET] Deleted {PROGRESS_FILE} — starting fresh")

    run_active_learning(
        n_rounds=args.rounds,
        patterns_per_round=args.patterns,
        openmc_speed=args.speed
    )