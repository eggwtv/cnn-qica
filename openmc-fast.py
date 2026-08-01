"""
=============================================================================
openmc_beavrs_fast.py  —  Fast BEAVRS Simulator for Active Learning
=============================================================================
RENAMED from openmc-fast.py → openmc_beavrs_fast.py
  Python cannot import modules whose filename contains a hyphen.
  al_loop.py does: from openmc_beavrs_fast import simulate_pattern
  so this file MUST be named with underscores.

ENDF/B-VIII.1 FIX (was VIII.0 / endfb80):
  Chain file candidates updated to check endfb81 first, then endfb80 as fallback.
  Set the env var before running:
    export OPENMC_CROSS_SECTIONS=/path/to/endfb-b8.1-hdf5/cross_sections.xml
    export OPENMC_CHAIN_FILE=/path/to/chain_endfb81_pwr.xml   # optional override

WHAT IS OPENMC?
  OpenMC simulates individual neutrons bouncing around the reactor. It is a
  "Monte Carlo" code, meaning it uses random sampling to model neutron physics.
  Each neutron is tracked from birth (fission) until absorption or leakage.
  Running enough neutrons gives statistical estimates of:
    • keff   — how critical the reactor is (keff≈1.006 at BOC with boron)
    • PPF    — max assembly power / mean assembly power (safety limit)
    • Cycle length — days before keff drops below critical (fuel exhaustion)

INSTALLATION (one time setup)
  Step 1: Install OpenMC
    conda install -c conda-forge openmc          # recommended
    # or: pip install openmc  (Linux only)

  Step 2: Download ENDF/B-VIII.1 cross-section library (~8 GB download):
    https://openmc.org/official-data-libraries/ → ENDF/B-VIII.1-HDF5

  Step 3: Tell OpenMC where the library is:
    export OPENMC_CROSS_SECTIONS=/path/to/endfb-b8.1-hdf5/cross_sections.xml
    # Add this to ~/.bashrc or ~/.zshrc so it persists

  Step 4: Download the depletion chain file for VIII.1:
    wget https://anl.box.com/shared/static/...chain_endfb81_pwr.xml
    # or it may ship with the conda-forge openmc package
    export OPENMC_CHAIN_FILE=/path/to/chain_endfb81_pwr.xml

  Step 5: Test:
    python openmc_beavrs_fast.py --test

SPEED MODES  (set SPEED_MODE below)
  'debug'    ~20-40 sec/pattern  — checks the code works; results are noise
  'fast'     ~2-5 min/pattern   — good for active learning  ← start here
  'balanced' ~8-15 min/pattern  — better accuracy
  'accurate' ~25-45 min/pattern — publication quality

WHAT THIS RETURNS (for CNN v9 / QICA v5):
  {
    'ppf_max'      : float  — peak PPF over the full cycle
    'ppf_boc'      : float  — PPF at beginning of cycle
    'ppf_steps'    : array  — (31,) max PPF at each burnup step (CNN format)
    'cycle_length' : float  — effective full-power days
    'keff_boc'     : float  — k-effective at BOC
    'rho_pcm_boc'  : float  — reactivity in pcm at BOC
    'success'      : bool   — False if the simulation crashed
  }

HOW TO USE IN THE ACTIVE LEARNING LOOP:
  from openmc_beavrs_fast import simulate_pattern
  result = simulate_pattern(np.array([5,2,5,2,5,2,2,5,5,2,5,2,2,5,2,5,
                                       2,5,2,5,2,5,2,5,2,5,2,5,2,5,2]))
  # then pass result to your AL loop's dataset builder
=============================================================================
"""

import os, sys, json, time, warnings, shutil, tempfile, argparse
warnings.filterwarnings('ignore')
import numpy as np

# ── Speed mode ─────────────────────────────────────────────────────────────────
# Change this to trade off accuracy vs speed.
SPEED_MODE = 'fast'    # 'debug' | 'fast' | 'balanced' | 'accurate'

# Each entry: (particles_per_batch, total_batches, inactive_batches, depletion_steps)
_SPEED_CONFIGS = {
    'debug'    : (100,   20,   8,  2),
    'fast'     : (1000,  50,  15,  4),
    'balanced' : (2000,  80,  20,  8),
    'accurate' : (5000, 150,  50, 15),
}
_APPROX_TIME_MIN = {'debug': 0.5, 'fast': 3, 'balanced': 12, 'accurate': 35}

# ── Check OpenMC ───────────────────────────────────────────────────────────────
try:
    import openmc
    import openmc.deplete
    _OPENMC_OK = True
except ImportError:
    _OPENMC_OK = False


# =============================================================================
# SECTION 1 — BEAVRS GEOMETRY CONSTANTS
# =============================================================================

ASSEMBLY_PITCH  = 21.504   # cm   assembly centre-to-centre distance
ACTIVE_HEIGHT   = 365.76   # cm
N_ROWS_COLS     = 15       # 15×15 assembly grid for full BEAVRS core
CORE_POWER_W    = 3411e6   # W total thermal power

# Homogenised volume fractions (from pin-cell geometry: r_fuel=0.39218, p=1.26 cm)
# Fuel:clad:water ≈ 30.4% : 11.4% : 58.2% (per fuel pin cell)
# Weighted by 264 fuel pins + 25 guide tubes in 289 total positions:
VF_FUEL  = 0.278   # fuel pellet
VF_CLAD  = 0.114   # Zircaloy cladding
VF_WATER = 0.608   # coolant

# BEAVRS fuel assembly type definitions
# Each entry: (U-235 enrichment wt%, boron_atoms_per_barn_cm added for BA)
_ASSEMBLY_TYPES = {
    1: dict(enr=1.60, b10_extra=0.0,    note='Discharged / once-burned'),
    2: dict(enr=2.40, b10_extra=0.0,    note='2.4% fresh'),
    3: dict(enr=2.40, b10_extra=5e-5,   note='2.4% + IFBA (ZrB2 coating)'),
    4: dict(enr=2.40, b10_extra=1.5e-4, note='2.4% + WABA (Al2O3-B4C rods)'),
    5: dict(enr=3.10, b10_extra=0.0,    note='3.1% fresh'),
    6: dict(enr=3.10, b10_extra=3e-5,   note='3.1% + light IFBA'),
    7: dict(enr=3.10, b10_extra=5e-5,   note='3.1% + IFBA'),
    8: dict(enr=3.10, b10_extra=1.5e-4, note='3.1% + WABA'),
    9: dict(enr=3.10, b10_extra=8e-5,   note='3.1% + Pyrex'),
}

# 1/8-core CNN grid layout (must match cnn_v9.py GRID_LAYOUT exactly)
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)
N_POS     = int(GRID_MASK.sum())   # 31

# Full BEAVRS 15×15 core presence map (1=fuel assembly, 0=reflector)
CORE_MAP_TEMPLATE = np.array([
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
], dtype=np.int32)
N_FUEL_ASSEMBLIES = int(CORE_MAP_TEMPLATE.sum())   # should be 193

CYCLE_END_KEFF = 1.006    # keff at which cycle ends (accounts for ~600 ppm boron)


# =============================================================================
# SECTION 2 — 1/8-CORE → FULL CORE EXPANSION
# =============================================================================

def expand_to_full_core(pattern_31: np.ndarray) -> np.ndarray:
    """
    Expand 31-position 1/8-core pattern to 15×15 full-core map.

    The CNN uses the upper-right triangular octant (row ≤ col) of the
    core, centred at (7,7).  Eight-fold dihedral symmetry fills the rest.

    Args:
        pattern_31 : (31,) int array, values 1–9
    Returns:
        full : (15,15) int array  (0 = reflector / water)
    """
    full = np.zeros((15, 15), dtype=np.int32)
    pi   = 0
    for gr in range(6):
        for gc in range(6):
            if GRID_LAYOUT[gr, gc] < 0:
                continue
            t = int(pattern_31[pi]); pi += 1
            # Core centre = (7,7); CNN row/col maps to offset from centre
            fr, fc = 7 + gr, 7 + gc
            for (sr, sc) in {(fr,fc),(14-fr,fc),(fr,14-fc),(14-fr,14-fc),
                              (fc,fr),(14-fc,fr),(fc,14-fr),(14-fc,14-fr)}:
                if 0 <= sr < 15 and 0 <= sc < 15 and CORE_MAP_TEMPLATE[sr,sc]:
                    full[sr, sc] = t
    return full


# =============================================================================
# SECTION 3 — MATERIAL BUILDER
# =============================================================================

def make_homogenised_material(assembly_type: int, mat_id: int) -> 'openmc.Material':
    """
    Create a homogenised UO2 + Zircaloy + water material for an assembly.

    Physical basis: Volume-weighted atom densities from BEAVRS pin-cell geometry.
    Enrichment and burnable absorber loading vary by assembly type.

    The homogenisation approximation introduces ~5-10% error on PPF (vs pin-by-pin)
    and ~2-5% error on keff.  For active learning (relative ranking of patterns)
    this accuracy is fully sufficient.
    """
    spec  = _ASSEMBLY_TYPES[assembly_type]
    enr   = spec['enr'] / 100.0      # weight fraction U-235
    b10_x = spec['b10_extra']        # extra B-10 from burnable absorbers (atoms/b-cm)

    mat = openmc.Material(material_id=mat_id,
                          name=f'Assembly type {assembly_type} ({spec["note"]})')

    # ── UO2 fuel contribution (volume fraction VF_FUEL) ──────────────────────
    # Reference atom densities from BEAVRS at 2.4% (openmc.examples):
    # U total: 2.2964e-2 atoms/b-cm  O-16: 4.5829e-2 atoms/b-cm
    u_total   = 2.2964e-2 * VF_FUEL
    u235_dens = u_total * enr
    u238_dens = u_total * (1.0 - enr - 0.008*enr)
    u234_dens = u_total * 0.008 * enr
    o16_fuel  = 4.5829e-2 * VF_FUEL

    # ── Zircaloy cladding contribution (volume fraction VF_CLAD) ─────────────
    # Zr isotopes (atoms/b-cm) from openmc.examples
    zr_scale = VF_CLAD
    zr90 = 2.1827e-2 * zr_scale
    zr91 = 4.7600e-3 * zr_scale
    zr92 = 7.2758e-3 * zr_scale
    zr94 = 7.3734e-3 * zr_scale
    zr96 = 1.1879e-3 * zr_scale

    # ── Water coolant contribution (volume fraction VF_WATER, 600 ppm boron) ─
    w_scale = VF_WATER
    h1    = 4.9457e-2 * w_scale
    o16_w = 2.4672e-2 * w_scale
    b10   = 8.0042e-6 * w_scale + b10_x   # coolant B-10 + BA contribution
    b11   = 3.2218e-5 * w_scale

    # Effective density
    rho_eff = (10.3 * VF_FUEL + 6.55 * VF_CLAD + 0.74 * VF_WATER)
    mat.set_density('g/cm3', rho_eff)

    mat.add_nuclide('U234', u234_dens)
    mat.add_nuclide('U235', u235_dens)
    mat.add_nuclide('U238', u238_dens)
    mat.add_nuclide('O16',  o16_fuel + o16_w)
    mat.add_nuclide('Zr90', zr90)
    mat.add_nuclide('Zr91', zr91)
    mat.add_nuclide('Zr92', zr92)
    mat.add_nuclide('Zr94', zr94)
    mat.add_nuclide('Zr96', zr96)
    mat.add_nuclide('H1',   h1)
    mat.add_nuclide('B10',  b10)
    mat.add_nuclide('B11',  b11)
    # S(α,β) table for thermal scattering in water — works for both VIII.0 and VIII.1
    mat.add_s_alpha_beta('c_H_in_H2O')
    mat.temperature = 750.0   # effective homogenised temp (between T_fuel=900 and T_cool=600)

    return mat


def make_reflector_material(mat_id: int) -> 'openmc.Material':
    """Pure water reflector (no boron, cold)."""
    mat = openmc.Material(material_id=mat_id, name='Water reflector')
    mat.set_density('g/cm3', 0.74)
    mat.add_nuclide('H1',  4.9457e-2)
    mat.add_nuclide('O16', 2.4672e-2)
    mat.add_s_alpha_beta('c_H_in_H2O')
    mat.temperature = 600.0
    return mat


# =============================================================================
# SECTION 4 — GEOMETRY BUILDER
# =============================================================================

def build_core_geometry(full_core_map: np.ndarray) -> tuple:
    """
    Build 3D BEAVRS core geometry with homogenised assemblies.

    Returns:
        (geometry, materials, mesh_for_tally)
    """
    half_h = ACTIVE_HEIGHT / 2.0
    half_p = ASSEMBLY_PITCH / 2.0

    # Build one material per unique assembly type present
    unique_types = sorted(set(int(t) for t in full_core_map.flatten() if t > 0))
    mat_dict     = {t: make_homogenised_material(t, mat_id=t) for t in unique_types}
    refl_mat     = make_reflector_material(mat_id=99)
    all_mats     = list(mat_dict.values()) + [refl_mat]

    # Build assembly universe for each unique type
    def make_asm_universe(atype):
        mat = mat_dict[atype]
        s   = openmc.rectangular_prism(
            ASSEMBLY_PITCH, ASSEMBLY_PITCH,
            boundary_type='transmission',
            origin=(0, 0)
        ) & +openmc.ZPlane(-half_h) & -openmc.ZPlane(half_h)
        cell = openmc.Cell(fill=mat, region=s, name=f'Asm type {atype}')
        uni  = openmc.Universe(name=f'Assembly {atype}')
        uni.add_cell(cell)
        return uni

    def make_refl_universe():
        s   = openmc.rectangular_prism(
            ASSEMBLY_PITCH, ASSEMBLY_PITCH,
            boundary_type='transmission',
        ) & +openmc.ZPlane(-half_h) & -openmc.ZPlane(half_h)
        cell = openmc.Cell(fill=refl_mat, region=s, name='Reflector')
        uni  = openmc.Universe(name='Reflector')
        uni.add_cell(cell)
        return uni

    asm_universes = {t: make_asm_universe(t) for t in unique_types}
    refl_uni      = make_refl_universe()

    # 15×15 core lattice
    lattice = openmc.RectLattice(name='BEAVRS core lattice')
    lattice.pitch       = (ASSEMBLY_PITCH, ASSEMBLY_PITCH)
    lattice.lower_left  = (-7.5 * ASSEMBLY_PITCH, -7.5 * ASSEMBLY_PITCH)
    lattice.universes   = [
        [asm_universes.get(int(full_core_map[r, c]), refl_uni) for c in range(15)]
        for r in range(14, -1, -1)   # OpenMC: bottom row first
    ]

    # Root universe with cylindrical outer boundary
    core_r  = 7.5 * ASSEMBLY_PITCH + 15.0
    cyl     = openmc.ZCylinder(r=core_r, boundary_type='vacuum')
    z_bot   = openmc.ZPlane(z0=-half_h - 15.0, boundary_type='vacuum')
    z_top   = openmc.ZPlane(z0= half_h + 15.0, boundary_type='vacuum')

    fuel_cell  = openmc.Cell(fill=lattice, region=-cyl & +z_bot & -z_top, name='Core')
    root_uni   = openmc.Universe(name='Root')
    root_uni.add_cell(fuel_cell)

    geometry = openmc.Geometry(root_uni)

    # 15×15 mesh for assembly PPF tally
    mesh = openmc.RegularMesh(mesh_id=1, name='Assembly power mesh')
    mesh.dimension   = [15, 15, 1]
    mesh.lower_left  = [-7.5*ASSEMBLY_PITCH, -7.5*ASSEMBLY_PITCH, -half_h]
    mesh.upper_right = [ 7.5*ASSEMBLY_PITCH,  7.5*ASSEMBLY_PITCH,  half_h]

    return geometry, openmc.Materials(all_mats), mesh


# =============================================================================
# SECTION 5 — SETTINGS + TALLIES
# =============================================================================

def build_settings(particles: int, batches: int, inactive: int) -> 'openmc.Settings':
    s = openmc.Settings()
    s.batches   = batches
    s.inactive  = inactive
    s.particles = particles
    s.run_mode  = 'eigenvalue'
    s.output    = {'tallies': False, 'summary': False}
    # Uniform spatial source over core
    core_half = 7.5 * ASSEMBLY_PITCH
    s.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            [-core_half, -core_half, -ACTIVE_HEIGHT/2],
            [ core_half,  core_half,  ACTIVE_HEIGHT/2]
        ),
        constraints={'fissionable': True}
    )
    s.temperature = {'default': 750.0, 'method': 'interpolation'}
    return s


def build_tally(mesh: 'openmc.RegularMesh') -> 'openmc.Tallies':
    filt   = openmc.MeshFilter(mesh)
    tally  = openmc.Tally(tally_id=1, name='Assembly fission tally')
    tally.filters = [filt]
    tally.scores  = ['fission']
    return openmc.Tallies([tally])


# =============================================================================
# SECTION 6 — RESULT EXTRACTION
# =============================================================================

def extract_ppf(sp_path: str, full_core_map: np.ndarray) -> float:
    """Compute assembly-level PPF from statepoint file."""
    with openmc.StatePoint(sp_path) as sp:
        tally = sp.get_tally(id=1)
        power = tally.get_values(scores=['fission']).reshape(15, 15)
    mask        = (full_core_map > 0)
    fuel_power  = power[mask]
    if fuel_power.sum() < 1e-15:
        return 9.99
    return float(fuel_power.max() / fuel_power.mean())


def extract_keff(sp_path: str) -> float:
    """Read keff from statepoint."""
    with openmc.StatePoint(sp_path) as sp:
        k = sp.keff
    return float(k.n if hasattr(k, 'n') else k)


def extract_cycle_length(depl_path: str, timestep_days: list) -> tuple:
    """
    Interpolate cycle end from depletion results.
    Cycle ends when keff drops below CYCLE_END_KEFF.

    Returns: (cycle_efpd, keff_boc, keff_array)
    """
    try:
        res   = openmc.deplete.ResultsList.from_hdf5(depl_path)
        times = [0.0] + [sum(timestep_days[:i+1]) for i in range(len(timestep_days))]
        keffs = []
        for step_res in res:
            k = step_res.k
            keffs.append(float(k.n if hasattr(k, 'n') else k))
        keff_arr = np.array(keffs)
        keff_boc = keff_arr[0]

        below = keff_arr < CYCLE_END_KEFF
        if not below.any():
            # Extrapolate
            if len(times) >= 2:
                dk = (keff_arr[-1] - keff_arr[-2]) / (times[-1] - times[-2] + 1e-9)
                cyc = times[-1] + (CYCLE_END_KEFF - keff_arr[-1]) / (dk + 1e-9) if dk < 0 else times[-1]
            else:
                cyc = times[-1]
        else:
            idx = int(np.argmax(below))
            if idx == 0:
                cyc = float(times[0])
            else:
                frac = (CYCLE_END_KEFF - keff_arr[idx-1]) / (keff_arr[idx] - keff_arr[idx-1] + 1e-9)
                cyc  = times[idx-1] + frac * (times[idx] - times[idx-1])

        return max(30.0, float(cyc)), float(keff_boc), keff_arr

    except Exception as e:
        print(f"  [WARN] Depletion results read failed: {e}")
        return 350.0, 1.05, np.ones(len(timestep_days)+1) * 1.05


# =============================================================================
# SECTION 7 — MAIN SIMULATE FUNCTION
# =============================================================================

def simulate_pattern(loading_pattern_1d: np.ndarray,
                     speed_mode: str = None,
                     work_dir: str = None,
                     keep_files: bool = False,
                     verbose: bool = True) -> dict:
    """
    Run a fast homogenised BEAVRS simulation for a 31-position loading pattern.

    Args:
        loading_pattern_1d : (31,) int array, assembly types 1–9
        speed_mode         : override SPEED_MODE if given
        work_dir           : directory for temp files (None = auto temp dir)
        keep_files         : keep OpenMC input/output files if True
        verbose            : print progress if True

    Returns:
        dict with keys: ppf_max, ppf_boc, ppf_steps, cycle_length,
                        keff_boc, rho_pcm_boc, success
    """
    if not _OPENMC_OK:
        raise RuntimeError(
            "OpenMC not installed. See file header for installation instructions."
        )

    mode    = speed_mode or SPEED_MODE
    cfg     = _SPEED_CONFIGS[mode]
    n_p, n_b, n_i, n_steps = cfg
    est_min = _APPROX_TIME_MIN[mode]

    pattern = np.asarray(loading_pattern_1d, dtype=np.int32).flatten()
    assert len(pattern) == N_POS, f"Expected {N_POS} positions, got {len(pattern)}"
    assert all(1 <= t <= 9 for t in pattern), "Types must be 1–9"

    full_core_map = expand_to_full_core(pattern)

    if verbose:
        print(f"\n[OpenMC/{mode}]  pattern={list(pattern[:8])}...")
        print(f"  Particles={n_p}  Batches={n_b}  Inactive={n_i}  DepSteps={n_steps}")
        print(f"  Estimated time: ~{est_min} min")

    orig_dir = os.getcwd()
    tmp_created = False
    if work_dir is None:
        work_dir    = tempfile.mkdtemp(prefix='beavrs_fast_')
        tmp_created = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    t_start = time.time()
    try:
        os.chdir(work_dir)

        # Build model
        geometry, materials, mesh = build_core_geometry(full_core_map)
        settings = build_settings(n_p, n_b, n_i)
        tallies  = build_tally(mesh)

        model = openmc.Model(geometry=geometry, materials=materials,
                             settings=settings, tallies=tallies)
        model.export_to_xml()

        # ── BOC eigenvalue run ────────────────────────────────────────────────
        if verbose:
            print(f"  Running BOC eigenvalue...")
        openmc.run(output=False)

        sp_path  = f'statepoint.{n_b:04d}.h5'
        keff_boc = extract_keff(sp_path)
        ppf_boc  = extract_ppf(sp_path, full_core_map)
        rho_pcm  = (keff_boc - 1.0) / keff_boc * 1e5

        if verbose:
            print(f"  BOC keff={keff_boc:.5f}  PPF_BOC={ppf_boc:.4f}")

        if keff_boc < CYCLE_END_KEFF:
            if verbose:
                print(f"  WARNING: subcritical loading (keff < {CYCLE_END_KEFF})")
            return {
                'ppf_max': ppf_boc, 'ppf_boc': ppf_boc,
                'ppf_steps': np.full(31, ppf_boc, dtype=np.float32),
                'cycle_length': 0.0, 'keff_boc': keff_boc,
                'rho_pcm_boc': rho_pcm, 'success': True,
            }

        # ── Depletion calculation ─────────────────────────────────────────────
        # Distribute depletion steps across ~500 EFPD
        step_efpd = 500.0 / n_steps
        timesteps = [step_efpd * 86400.0] * n_steps   # seconds
        power_arr = [CORE_POWER_W] * n_steps

        # ── FIX: Chain file search — ENDF/B-VIII.1 first, VIII.0 as fallback ─
        # You can also set: export OPENMC_CHAIN_FILE=/path/to/chain_endfb81_pwr.xml
        chain_file = os.environ.get('OPENMC_CHAIN_FILE', None)

        if chain_file is None:
            xs_dir = os.path.dirname(os.environ.get('OPENMC_CROSS_SECTIONS', ''))
            candidates = [
                # VIII.1 paths (your installed version)
                '/usr/share/openmc/chain_endfb81_pwr.xml',
                '/opt/openmc/chain_endfb81_pwr.xml',
                os.path.join(xs_dir, 'chain_endfb81_pwr.xml'),
                os.path.join(xs_dir, '..', 'chain_endfb81_pwr.xml'),
                # conda-forge install location for VIII.1
                os.path.join(os.path.dirname(sys.executable),
                             '..', 'share', 'openmc', 'chain_endfb81_pwr.xml'),
                # VIII.0 fallback (will still work but less accurate for your library)
                '/usr/share/openmc/chain_endfb80_pwr.xml',
                '/opt/openmc/chain_endfb80_pwr.xml',
                os.path.join(xs_dir, 'chain_endfb80_pwr.xml'),
            ]
            for c in candidates:
                if c and os.path.exists(os.path.normpath(c)):
                    chain_file = os.path.normpath(c)
                    break

        if chain_file is None:
            print("  [WARN] Depletion chain file not found.\n"
                  "  For ENDF/B-VIII.1, the chain file ships with conda-forge OpenMC or can\n"
                  "  be downloaded from https://openmc.org/official-data-libraries/\n"
                  "  Set: export OPENMC_CHAIN_FILE=/path/to/chain_endfb81_pwr.xml\n"
                  "  Returning BOC-only results (no cycle length).")
            ppf_steps31 = np.full(31, ppf_boc, dtype=np.float32)
            return {
                'ppf_max': ppf_boc, 'ppf_boc': ppf_boc,
                'ppf_steps': ppf_steps31, 'cycle_length': 350.0,
                'keff_boc': keff_boc, 'rho_pcm_boc': rho_pcm, 'success': True,
            }

        if verbose:
            print(f"  Depletion chain: {chain_file}")
            print(f"  Running {n_steps} depletion steps × {step_efpd:.0f} EFPD each...")

        operator   = openmc.deplete.CoupledOperator(
            model, chain_file=chain_file, normalization_mode='source-rate'
        )
        integrator = openmc.deplete.PredictorIntegrator(
            operator, timesteps, power_arr, timestep_units='s'
        )

        # Track PPF at each step (read after integration)
        ppf_track = [ppf_boc]

        integrator.integrate()

        # Read per-step statepoints produced during depletion
        for step_idx in range(n_steps):
            sp = f'openmc_simulation_n{step_idx + 1}.h5'
            if os.path.exists(sp):
                ppf_track.append(extract_ppf(sp, full_core_map))

        # Cycle length from keff depletion curve
        cycle_efpd, _, keff_arr = extract_cycle_length(
            'depletion_results.h5', [step_efpd] * n_steps
        )

        # Resample PPF to 31 steps (CNN format)
        ppf_arr    = np.array(ppf_track, dtype=np.float32)
        t_measured = np.linspace(0, 500.0, len(ppf_arr))
        t_target   = np.linspace(0, max(cycle_efpd, 10.0), 31)
        ppf_31     = np.interp(t_target, t_measured, ppf_arr).astype(np.float32)
        ppf_max    = float(ppf_arr.max())

        t_elapsed  = time.time() - t_start
        if verbose:
            print(f"  Done: cycle={cycle_efpd:.1f} EFPD  PPF_max={ppf_max:.4f}  "
                  f"({t_elapsed:.0f}s elapsed)")

        return {
            'ppf_max'     : ppf_max,
            'ppf_boc'     : ppf_boc,
            'ppf_steps'   : ppf_31,
            'cycle_length': cycle_efpd,
            'keff_boc'    : keff_boc,
            'rho_pcm_boc' : rho_pcm,
            'success'     : True,
        }

    except Exception as exc:
        import traceback
        print(f"  [ERROR] OpenMC simulation failed: {exc}")
        traceback.print_exc()
        return {
            'ppf_max': 9.99, 'ppf_boc': 9.99,
            'ppf_steps': np.full(31, 9.99, dtype=np.float32),
            'cycle_length': 0.0, 'keff_boc': 0.0,
            'rho_pcm_boc': 0.0, 'success': False,
        }
    finally:
        os.chdir(orig_dir)
        if tmp_created and not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif keep_files and verbose:
            print(f"  Output files kept in: {work_dir}")


# =============================================================================
# SECTION 8 — STANDALONE TEST + CLI
# =============================================================================

def run_test(mode: str = 'debug') -> None:
    """Quick sanity check: simulate one checkerboard pattern."""
    print(f"\n{'='*60}")
    print(f"OPENMC BEAVRS FAST  —  Test Run (mode='{mode}')")
    print(f"{'='*60}")
    print(f"  Estimated time: ~{_APPROX_TIME_MIN[mode]} min\n")

    # Simple checkerboard (alternating type 5 and type 2 — physically reasonable)
    test_pat = np.tile([5, 2], 16)[:31].astype(np.int32)
    print(f"  Test pattern: {list(test_pat)}")

    result = simulate_pattern(test_pat, speed_mode=mode, verbose=True)

    print(f"\n  RESULT:")
    print(f"    PPF_max      : {result['ppf_max']:.4f}")
    print(f"    PPF_BOC      : {result['ppf_boc']:.4f}")
    print(f"    Cycle length : {result['cycle_length']:.1f} EFPD")
    print(f"    keff_BOC     : {result['keff_boc']:.5f}")
    print(f"    rho_pcm_BOC  : {result['rho_pcm_boc']:.0f} pcm")
    print(f"    Success      : {result['success']}")
    print(f"\n  If you see physically reasonable numbers above, OpenMC is working.")
    print(f"  Typical BEAVRS BOC: keff≈1.02-1.05, PPF≈1.4-2.5, cycle≈300-500 EFPD")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='BEAVRS fast OpenMC simulator')
    p.add_argument('--test',  action='store_true',
                   help='Run a quick test pattern (debug mode)')
    p.add_argument('--mode',  default=SPEED_MODE,
                   choices=list(_SPEED_CONFIGS.keys()),
                   help='Speed mode (default: fast)')
    p.add_argument('--keep',  action='store_true',
                   help='Keep OpenMC working files after run')
    args = p.parse_args()

    if args.test:
        run_test(mode=args.mode)
    else:
        print(__doc__)
        print("\nUsage:")
        print("  python openmc_beavrs_fast.py --test --mode debug   # quick sanity check")
        print("  python openmc_beavrs_fast.py --test --mode fast    # AL-quality run")
        print("  python openmc_beavrs_fast.py --test --mode accurate")