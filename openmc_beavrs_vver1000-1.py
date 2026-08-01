"""
conda --version
conda activate openmc-env

=============================================================================
openmc_beavrs_vver1000.py  —  High-Fidelity Generator for cnn_v9 / qica_v9-final
=============================================================================
Produces ml_dataset_constrained.csv rows that cnn-v9.py can train on directly,
and/or verifies AL-flagged candidates from qica_v9-final_al_candidates.csv.

IMPORTANT GEOMETRY NOTE (read this first):
──────────────────────────────────────────
Your cnn-v9.py GRID_LAYOUT is a 6×6 SQUARE grid (31 active positions, 1/8-core
BEAVRS-style symmetry) — not a VVER-1000 hexagonal assembly layout. Everything
downstream (CNN input embedding, QICA trust-region/free_mask, sensitivity map)
is built on that square indexing. So this script builds a BEAVRS-convention
square-assembly core (17×17 pin assemblies) to stay bit-compatible with your
existing pipeline. If/when you want literal VVER-1000 hex assemblies, swap
`build_assembly_universe()` and the core `RectLattice` for an `openmc.HexLattice`
— the depletion / tally / CSV-export machinery below doesn't change, only the
geometry section (SECTION 2) does.

WHAT THIS SCRIPT DOES
──────────────────────
  1. Defines 9 assembly "types" (enrichment + optional Gd2O3 burnable absorber
     pins), matching the ASSEMBLY_CYCLE_EQUIV fallback ordering in cnn-v9.py.
  2. Builds a 6×6 (31-active-position) core from a flat 31-length loading
     pattern, using the IDENTICAL GRID_LAYOUT and position-ordering loop as
     cnn-v9.py — so loading_i in the CSV means the same physical position in
     both pipelines.
  3. Runs coupled depletion (openmc.deplete) for N_STEPS burnup steps.
  4. At every step: extracts k-eff -> reactivity (rho), a 2-group (thermal /
     fast) flux tally, an assembly-boundary surface current tally, and a
     pin-power mesh tally -> per-assembly PPF.
  5. Finds cycle_length via linear interpolation of the rho(t) zero-crossing
     (same method as your 01_reproduce_paper.py find_cycle_length()).
  6. Writes rows to a CSV with columns:
       loading_0..loading_30, react_0..react_30,
       ppf_s{step}_a{assembly} for step in 0..30, assembly in 0..30,
       cycle_length
     with a leading comment line so cnn-v9.py's `pd.read_csv(..., skiprows=1)`
     works unmodified.

REQUIREMENTS (environment-specific — fill these in before running):
  - OpenMC compiled with a depletion-capable build
  - Cross-section library: set OPENMC_CROSS_SECTIONS env var
        export OPENMC_CROSS_SECTIONS=/path/to/endfb-viii.1-hdf5/cross_sections.xml
  - Depletion chain file: set OPENMC_CHAIN or pass --chain
        (matches memory note: ENDF/B-VIII.1 chain, e.g. chain_endfb71_pwr.xml
         renamed/rebuilt for VIII.1 if you have one; a VIII.0/71 chain works
         fine for a template run)
  - This is a TEMPLATE: pin/assembly dimensions below are standard published
    BEAVRS benchmark values (public MIT spec), not proprietary VVER data.
    Verify against your own spec sheet before treating outputs as final.

RUNTIME: full-core depletion with pin-resolved tallies is NOT cheap. A single
31-position pattern × 31 burnup steps with pin-power mesh tallies will run
from tens of minutes to several hours on CPU depending on particles/batch.
Start with --n_patterns 1 --particles 2000 --batches 20 to sanity check.
=============================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import openmc
import openmc.deplete


# =============================================================================
# SECTION 0 — CONFIGURATION  (kept in lockstep with cnn-v9.py / qica_v9-final.py)
# =============================================================================

N_POS   = 31     # active fuel positions (1/8-core symmetry) — matches cnn-v9.py
N_TYPES = 9      # assembly types — matches cnn-v9.py
N_STEPS = 31     # burnup steps — matches cnn-v9.py N_STEPS

GRID_ROWS, GRID_COLS = 6, 6
GRID_LAYOUT = np.array([
    [ 0,  1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10, 11],
    [12, 13, 14, 15, 16, 17],
    [18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29],
    [30, -1, -1, -1, -1, -1],
], dtype=np.int32)
GRID_MASK = (GRID_LAYOUT >= 0)

# ── Standard published BEAVRS pin-cell dimensions (cm) ───────────────────────
PIN_PITCH        = 1.26
ASSEMBLY_PITCH   = 21.42          # 17 * 1.26
FUEL_OR          = 0.4096
CLAD_IR          = 0.4180
CLAD_OR          = 0.4750
GT_IR            = 0.5610         # guide tube inner radius
GT_OR            = 0.6020         # guide tube outer radius
ACTIVE_HEIGHT    = 366.0          # cm (approx BEAVRS active fuel height)
PINS_PER_SIDE    = 17

# 25 guide-tube lattice positions in a standard 17x17 Westinghouse assembly
# (row, col) 0-indexed, includes the 24 GT + 1 central instrument tube.
GUIDE_TUBE_POS = {
    (2,5),(2,8),(2,11),
    (3,3),(3,13),
    (5,2),(5,5),(5,8),(5,11),(5,14),
    (8,2),(8,5),(8,11),(8,14),      # (8,8) reserved for instrument tube
    (11,2),(11,5),(11,8),(11,11),(11,14),
    (13,3),(13,13),
    (14,5),(14,8),(14,11),
}
INSTRUMENT_TUBE_POS = (8, 8)

# ── Assembly type library ─────────────────────────────────────────────────────
# Enrichment (wt% U-235) and burnable-absorber (Gd2O3 wt% in a subset of pins)
# per type. Ordered loosely to echo the monocore EFPD fallback in cnn-v9.py
# (type 1 = short-cycle/low-worth ... type 9 = long-cycle/high-worth), but
# feel free to replace with your xlsx-derived per-type values.
ASSEMBLY_LIBRARY = {
    1: dict(enrich=1.60, gd2o3_wt=0.0, n_bp_pins=0),
    2: dict(enrich=3.20, gd2o3_wt=0.0, n_bp_pins=0),
    3: dict(enrich=2.80, gd2o3_wt=4.0, n_bp_pins=8),
    4: dict(enrich=2.60, gd2o3_wt=4.0, n_bp_pins=12),
    5: dict(enrich=4.45, gd2o3_wt=0.0, n_bp_pins=0),
    6: dict(enrich=4.20, gd2o3_wt=6.0, n_bp_pins=16),
    7: dict(enrich=3.95, gd2o3_wt=6.0, n_bp_pins=12),
    8: dict(enrich=3.90, gd2o3_wt=4.0, n_bp_pins=8),
    9: dict(enrich=4.95, gd2o3_wt=0.0, n_bp_pins=0),
}

# Two-group energy boundary (eV) for the thermal/fast flux tally
THERMAL_FAST_BOUNDARY_EV = 0.625

# Core power for depletion normalisation (W) — scaled down from a real VVER-1000
# (~3000 MWth) by the fraction of the core this 1/8-symmetry model represents.
# Adjust FULL_CORE_POWER_W / SYMMETRY_FACTOR to your actual core size.
FULL_CORE_POWER_W = 3000.0e6
SYMMETRY_FACTOR   = 8.0
MODEL_POWER_W     = FULL_CORE_POWER_W / SYMMETRY_FACTOR

# Non-uniform burnup step schedule (days), dense early (BA burnout region),
# sparser late — mirrors the time-axis logic in 01_reproduce_paper.py.
def build_step_days(n_steps=N_STEPS):
    early = [2, 3, 5, 10, 15]
    remaining = n_steps - len(early)
    rest = list(np.diff(np.linspace(20, 620, remaining + 1)))
    return (early + rest)[:n_steps]

STEP_DAYS = build_step_days()


# =============================================================================
# SECTION 1 — MATERIALS
# =============================================================================

def build_materials(type_id):
    """Return (fuel_mat, bp_fuel_mat_or_None, clad_mat, gt_mat, water_mat)."""
    lib = ASSEMBLY_LIBRARY[type_id]

    fuel = openmc.Material(name=f'fuel_type{type_id}')
    fuel.add_element('U', 1.0, enrichment=lib['enrich'])
    fuel.add_element('O', 2.0)
    fuel.set_density('g/cm3', 10.31)

    bp_fuel = None
    if lib['gd2o3_wt'] > 0:
        bp_fuel = openmc.Material(name=f'fuel_bp_type{type_id}')
        # UO2 + Gd2O3 integral burnable absorber, simple wt%-mixed approximation
        bp_fuel.add_element('U', (1.0 - lib['gd2o3_wt'] / 100.0), enrichment=lib['enrich'])
        bp_fuel.add_element('O', 2.0 * (1.0 - lib['gd2o3_wt'] / 100.0))
        bp_fuel.add_element('Gd', 2.0 * (lib['gd2o3_wt'] / 100.0))
        bp_fuel.add_element('O', 3.0 * (lib['gd2o3_wt'] / 100.0))
        bp_fuel.set_density('g/cm3', 10.31 * (1 - 0.05 * lib['gd2o3_wt'] / 100.0))

    clad = openmc.Material(name='zircaloy4')
    clad.add_element('Zr', 0.982)
    clad.add_element('Sn', 0.014)
    clad.add_element('Fe', 0.002)
    clad.add_element('Cr', 0.001)
    clad.add_element('Ni', 0.001)
    clad.set_density('g/cm3', 6.55)

    gt = openmc.Material(name='zircaloy4_gt')
    gt.add_element('Zr', 1.0)
    gt.set_density('g/cm3', 6.55)

    water = openmc.Material(name='borated_water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    # ~800 wtppm soluble boron, BOC-ish. Must match H/O's percent_type ('ao')
    # since OpenMC can't mix atom and weight percents within one material —
    # converted here to an approximate atom fraction (molar-mass ratio).
    water.add_element('B', 800e-6 * (18.015 / 10.811), percent_type='ao')
    water.set_density('g/cm3', 0.712)                    # ~305 C PWR/VVER-ish hot density
    water.add_s_alpha_beta('c_H_in_H2O')

    return fuel, bp_fuel, clad, gt, water


# =============================================================================
# SECTION 2 — GEOMETRY
# =============================================================================

def build_pin_cells(fuel, bp_fuel, clad, gt, water, n_bp_pins):
    """
    Build the two pin-cell universes used to tile a 17x17 assembly:
      - fuel pin universe (plain UO2, used for non-BP pins)
      - bp pin universe   (UO2+Gd2O3, used for n_bp_pins BA rods, if any)
      - guide/instrument tube universe (water-filled Zr tube)
    """
    def _fuel_pin_universe(fuel_mat, tag):
        fuel_surf = openmc.ZCylinder(r=FUEL_OR)
        clad_ir   = openmc.ZCylinder(r=CLAD_IR)
        clad_or   = openmc.ZCylinder(r=CLAD_OR)

        c_fuel = openmc.Cell(fill=fuel_mat, region=-fuel_surf)
        c_gap  = openmc.Cell(fill=None, region=+fuel_surf & -clad_ir)  # He gap (void, simplified)
        c_clad = openmc.Cell(fill=clad, region=+clad_ir & -clad_or)
        c_mod  = openmc.Cell(fill=water, region=+clad_or)

        u = openmc.Universe(name=f'pin_{tag}', cells=[c_fuel, c_gap, c_clad, c_mod])
        return u

    u_fuel = _fuel_pin_universe(fuel, 'fuel')
    u_bp   = _fuel_pin_universe(bp_fuel, 'bp') if bp_fuel is not None else None

    gt_ir_surf = openmc.ZCylinder(r=GT_IR)
    gt_or_surf = openmc.ZCylinder(r=GT_OR)
    c_gt_water_in = openmc.Cell(fill=water, region=-gt_ir_surf)
    c_gt_clad     = openmc.Cell(fill=gt, region=+gt_ir_surf & -gt_or_surf)
    c_gt_water_out= openmc.Cell(fill=water, region=+gt_or_surf)
    u_gt = openmc.Universe(name='guide_tube',
                            cells=[c_gt_water_in, c_gt_clad, c_gt_water_out])

    return u_fuel, u_bp, u_gt


def build_assembly_universe(type_id):
    """
    Build one 17x17 BEAVRS-style square fuel assembly universe for the given
    assembly type (1..9). BP pins (if any) are placed at a fixed symmetric
    subset of non-guide-tube lattice positions — replace with your own
    per-type BP maps if you have exact rod maps.
    """
    fuel, bp_fuel, clad, gt, water = build_materials(type_id)
    u_fuel, u_bp, u_gt = build_pin_cells(fuel, bp_fuel, clad, gt, water,
                                          ASSEMBLY_LIBRARY[type_id]['n_bp_pins'])

    lat = openmc.RectLattice(name=f'assembly_type{type_id}')
    lat.lower_left = (-ASSEMBLY_PITCH / 2, -ASSEMBLY_PITCH / 2)
    lat.pitch = (PIN_PITCH, PIN_PITCH)

    universes = np.empty((PINS_PER_SIDE, PINS_PER_SIDE), dtype=object)
    universes[:, :] = u_fuel

    for pos in GUIDE_TUBE_POS:
        universes[pos] = u_gt
    universes[INSTRUMENT_TUBE_POS] = u_gt

    if u_bp is not None:
        # Deterministic, symmetric-ish BP pin placement: pick the first
        # n_bp_pins non-guide-tube, non-instrument-tube positions on a
        # coarse diagonal sweep. Replace with an exact rod map if available.
        n_bp = ASSEMBLY_LIBRARY[type_id]['n_bp_pins']
        placed = 0
        for offset in range(1, PINS_PER_SIDE):
            if placed >= n_bp:
                break
            for (r, c) in [(offset, offset), (offset, PINS_PER_SIDE - 1 - offset)]:
                if placed >= n_bp:
                    break
                if (r, c) in GUIDE_TUBE_POS or (r, c) == INSTRUMENT_TUBE_POS:
                    continue
                if 0 <= r < PINS_PER_SIDE and 0 <= c < PINS_PER_SIDE:
                    universes[r, c] = u_bp
                    placed += 1

    lat.universes = universes

    outer_bound = openmc.model.RectangularPrism(
        ASSEMBLY_PITCH, ASSEMBLY_PITCH, boundary_type='transmission')
    assembly_cell = openmc.Cell(fill=lat, region=-outer_bound)
    return openmc.Universe(name=f'assy_u_type{type_id}', cells=[assembly_cell])


def build_reflector_universe():
    """Fills inactive (-1) GRID_LAYOUT slots and, doubled up, the core baffle."""
    _, _, _, _, water = build_materials(1)   # water material only needed here
    steel = openmc.Material(name='baffle_steel')
    steel.add_element('Fe', 0.70)
    steel.add_element('Cr', 0.19)
    steel.add_element('Ni', 0.10)
    steel.add_element('Mn', 0.01)
    steel.set_density('g/cm3', 7.9)

    outer_bound = openmc.model.RectangularPrism(
        ASSEMBLY_PITCH, ASSEMBLY_PITCH, boundary_type='transmission')
    c_water = openmc.Cell(fill=water, region=-outer_bound)
    return openmc.Universe(name='reflector_u', cells=[c_water])


def pattern_to_grid(pattern_flat):
    """
    IDENTICAL mapping to cnn-v9.py's flat->grid loop, so loading_i in the CSV
    means the same physical position in both pipelines.
    """
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    pos_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid[r, c] = pattern_flat[pos_idx]
                pos_idx += 1
    return grid


def build_core_geometry(pattern_flat, assembly_universes, reflector_u):
    """
    Build the full OpenMC geometry for one 31-length loading pattern.
    Reflective boundaries on the two core-center-adjacent edges (standard
    BEAVRS 1/8-symmetry construction); vacuum on the outward-facing edges as
    a simplified core/baffle/reflector boundary (see module docstring note
    re: diagonal symmetry cut — add an openmc.Plane(a=1,b=-1,...) if you need
    the exact eighth-core diagonal for a fully rigorous 1/8 model).
    """
    grid = pattern_to_grid(pattern_flat)

    core_lat = openmc.RectLattice(name='core')
    core_lat.lower_left = (-GRID_COLS * ASSEMBLY_PITCH / 2,
                            -GRID_ROWS * ASSEMBLY_PITCH / 2)
    core_lat.pitch = (ASSEMBLY_PITCH, ASSEMBLY_PITCH)

    universes = np.empty((GRID_ROWS, GRID_COLS), dtype=object)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                atype = int(grid[r, c])
                universes[r, c] = assembly_universes[atype]
            else:
                universes[r, c] = reflector_u
    core_lat.universes = universes

    half_w = GRID_COLS * ASSEMBLY_PITCH / 2
    half_h = GRID_ROWS * ASSEMBLY_PITCH / 2

    x_min = openmc.XPlane(x0=-half_w, boundary_type='reflective')  # core-center plane
    x_max = openmc.XPlane(x0= half_w, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=-half_h, boundary_type='reflective')  # core-center plane
    y_max = openmc.YPlane(y0= half_h, boundary_type='vacuum')
    z_min = openmc.ZPlane(z0=-ACTIVE_HEIGHT / 2, boundary_type='vacuum')
    z_max = openmc.ZPlane(z0= ACTIVE_HEIGHT / 2, boundary_type='vacuum')

    core_region = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max
    core_cell = openmc.Cell(fill=core_lat, region=core_region)

    return openmc.Geometry(openmc.Universe(cells=[core_cell])), core_lat


# =============================================================================
# SECTION 3 — TALLIES: thermal/fast flux, assembly surface current, pin power
# =============================================================================

def build_tallies():
    """
    Three tally groups, all requested by name in your prompt:
      1. energy_flux_tally : 2-group (thermal/fast) scalar flux over the
         whole core mesh — diagnostic for spectrum shift vs. loading pattern.
      2. current_tally     : net partial current through the surfaces of a
         mesh matching the assembly pitch — assembly-to-assembly leakage,
         useful for nodal/diffusion cross-checks or SHAP-style attribution
         sanity checks against the CNN's gradient sensitivity map.
      3. power_mesh_tally  : fine (pin-pitch) mesh kappa-fission tally used
         to derive per-assembly PPF at each burnup step.
    """
    core_mesh = openmc.RegularMesh()
    core_mesh.lower_left  = (-GRID_COLS * ASSEMBLY_PITCH / 2,
                              -GRID_ROWS * ASSEMBLY_PITCH / 2,
                              -ACTIVE_HEIGHT / 2)
    core_mesh.upper_right = ( GRID_COLS * ASSEMBLY_PITCH / 2,
                               GRID_ROWS * ASSEMBLY_PITCH / 2,
                               ACTIVE_HEIGHT / 2)
    core_mesh.dimension = (GRID_COLS, GRID_ROWS, 1)   # one bin per assembly slot

    energy_filter = openmc.EnergyFilter([0.0, THERMAL_FAST_BOUNDARY_EV, 20.0e6])
    mesh_filter   = openmc.MeshFilter(core_mesh)

    flux_tally = openmc.Tally(name='thermal_fast_flux')
    flux_tally.filters = [mesh_filter, energy_filter]
    flux_tally.scores  = ['flux']

    mesh_surf_filter = openmc.MeshSurfaceFilter(core_mesh)
    current_tally = openmc.Tally(name='assembly_surface_current')
    current_tally.filters = [mesh_surf_filter]
    current_tally.scores  = ['current']

    pin_mesh = openmc.RegularMesh()
    pin_mesh.lower_left  = core_mesh.lower_left
    pin_mesh.upper_right = core_mesh.upper_right
    n_pins_x = GRID_COLS * PINS_PER_SIDE
    n_pins_y = GRID_ROWS * PINS_PER_SIDE
    pin_mesh.dimension = (n_pins_x, n_pins_y, 1)
    pin_mesh_filter = openmc.MeshFilter(pin_mesh)

    power_tally = openmc.Tally(name='pin_power_mesh')
    power_tally.filters = [pin_mesh_filter]
    power_tally.scores  = ['kappa-fission']

    tallies = openmc.Tallies([flux_tally, current_tally, power_tally])
    return tallies, core_mesh, pin_mesh


# =============================================================================
# SECTION 4 — PPF EXTRACTION FROM PIN-POWER MESH
# =============================================================================

def compute_ppf_per_assembly(pin_power_mesh_values, n_pins_x, n_pins_y):
    """
    pin_power_mesh_values: flat array of kappa-fission tally means, ordered
    per OpenMC's MeshFilter convention (x fastest, then y).
    Returns ppf_per_assembly: (N_POS,) array, one PPF per active assembly
    position, in the SAME 0..30 ordering as loading_/ppf_s{step}_a{i}.
    """
    grid2d = pin_power_mesh_values.reshape(n_pins_y, n_pins_x)

    core_mean_pin_power = grid2d[grid2d > 0].mean() if np.any(grid2d > 0) else 1.0

    ppf_flat = np.zeros(N_POS, dtype=np.float64)
    pos_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] < 0:
                continue
            sub = grid2d[r * PINS_PER_SIDE:(r + 1) * PINS_PER_SIDE,
                         c * PINS_PER_SIDE:(c + 1) * PINS_PER_SIDE]
            max_pin_power = sub.max() if sub.size else 0.0
            ppf_flat[pos_idx] = max_pin_power / (core_mean_pin_power + 1e-12)
            pos_idx += 1
    return ppf_flat


# =============================================================================
# SECTION 4.5 — QUICK CHECK (single static transport solve, no depletion)
# =============================================================================

def run_quick_check(pattern_flat, particles=2000, batches=40, inactive=15,
                     work_dir='openmc_quickcheck'):
    """
    Fast sanity check for one loading pattern: BOC k-eff and PPF only, no
    depletion loop. This is a single transport solve — minutes on CPU, not
    hours — meant to answer "does this pattern look reasonable before I
    commit to a full 31-step depletion run."
    Prints results directly; does not write to the training CSV.
    """
    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        assembly_universes = {t: build_assembly_universe(t) for t in range(1, N_TYPES + 1)}
        reflector_u = build_reflector_universe()
        geometry, core_lat = build_core_geometry(pattern_flat, assembly_universes, reflector_u)
        geometry.export_to_xml()

        materials = openmc.Materials(geometry.get_all_materials().values())
        materials.export_to_xml()

        settings = openmc.Settings()
        settings.batches = batches
        settings.inactive = inactive
        settings.particles = particles
        settings.temperature = {'default': 600.0}
        bbox = geometry.bounding_box
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(bbox[0], bbox[1], only_fissionable=True))
        settings.export_to_xml()

        tallies, core_mesh, pin_mesh = build_tallies()
        tallies.export_to_xml()

        t0 = time.time()
        openmc.run()
        elapsed = time.time() - t0

        sp_files = sorted(glob.glob('statepoint.*.h5'))
        n_pins_x = GRID_COLS * PINS_PER_SIDE
        n_pins_y = GRID_ROWS * PINS_PER_SIDE

        with openmc.StatePoint(sp_files[-1]) as sp:
            keff = sp.keff
            react = (keff.n - 1.0) / keff.n
            power_tally = sp.get_tally(name='pin_power_mesh')
            ppf = compute_ppf_per_assembly(power_tally.mean.ravel(), n_pins_x, n_pins_y)

        print(f'\n{"="*58}')
        print(f'QUICK CHECK RESULT  ({elapsed:.0f}s, {particles}p x {batches}b)')
        print(f'{"="*58}')
        print(f'  k-eff (BOC)     : {keff.n:.5f} +/- {keff.s:.5f}')
        print(f'  reactivity      : {react*1e5:+.0f} pcm')
        print(f'  PPF_max (BOC)   : {ppf.max():.3f}  (position {int(ppf.argmax())})')
        print(f'  PPF_min         : {ppf.min():.3f}')
        print(f'  PPF per position:')
        for i in range(0, N_POS, 8):
            print('   ', ' '.join(f'{v:.2f}' for v in ppf[i:i+8]))
        print(f'{"="*58}')
        print('  Reference: your realistic PPF target range is 2.0-4.5 (per cnn-v9 data).')
        print('  If PPF_max here is way outside that, don\'t bother running full depletion yet.')
        return dict(keff=keff.n, react=react, ppf=ppf)
    finally:
        os.chdir(cwd0)


# =============================================================================
# SECTION 5 — SINGLE-PATTERN DEPLETION RUN
# =============================================================================

def run_one_pattern(pattern_flat, chain_file, particles=4000, batches=60,
                     inactive=15, work_dir='openmc_run'):
    """
    Runs full depletion for one 31-length loading pattern and returns a dict
    with everything cnn-v9.py's CSV schema needs, plus the diagnostic flux/
    current tallies for your own inspection (saved to .npy alongside the CSV
    row rather than crammed into the training CSV).
    """
    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)

    try:
        assembly_universes = {t: build_assembly_universe(t) for t in range(1, N_TYPES + 1)}
        reflector_u = build_reflector_universe()

        geometry, core_lat = build_core_geometry(pattern_flat, assembly_universes, reflector_u)
        geometry.export_to_xml()

        materials = openmc.Materials(geometry.get_all_materials().values())
        materials.export_to_xml()

        settings = openmc.Settings()
        settings.batches   = batches
        settings.inactive  = inactive
        settings.particles = particles
        settings.temperature = {'default': 600.0}
        bbox = geometry.bounding_box
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(bbox[0], bbox[1], only_fissionable=True))

        tallies, core_mesh, pin_mesh = build_tallies()
        tallies.export_to_xml()
        settings.export_to_xml()

        model = openmc.Model(geometry=geometry, materials=materials,
                              settings=settings, tallies=tallies)

        fuel_materials = [m for m in materials if m.name.startswith('fuel_')]
        operator = openmc.deplete.CoupledOperator(model, chain_file)

        power_W = [MODEL_POWER_W] * N_STEPS
        integrator = openmc.deplete.PredictorIntegrator(
            operator, STEP_DAYS[:N_STEPS], power=power_W, timestep_units='d')
        integrator.integrate()

        # ── Collect per-step results ─────────────────────────────────────────
        results = openmc.deplete.Results('depletion_results.h5')
        n_pins_x = GRID_COLS * PINS_PER_SIDE
        n_pins_y = GRID_ROWS * PINS_PER_SIDE

        react = np.zeros(N_STEPS, dtype=np.float64)
        ppf   = np.zeros((N_STEPS, N_POS), dtype=np.float64)

        # Statepoint files are written once per transport solve during
        # depletion; naming follows OpenMC's convention for the installed
        # version (commonly 'openmc_simulation_n{step}.h5' or
        # 'statepoint.{batches}.h5' per step directory) — adjust the glob
        # below if your OpenMC version names them differently.
        sp_files = sorted(glob.glob('openmc_simulation_n*.h5'),
                           key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
        if not sp_files:
            sp_files = sorted(glob.glob('statepoint*.h5'))

        for step, sp_path in enumerate(sp_files[:N_STEPS]):
            with openmc.StatePoint(sp_path) as sp:
                keff = sp.keff.n
                react[step] = (keff - 1.0) / keff   # rho as a fraction

                power_tally = sp.get_tally(name='pin_power_mesh')
                pin_vals = power_tally.mean.ravel()
                ppf[step, :] = compute_ppf_per_assembly(pin_vals, n_pins_x, n_pins_y)

                if step == 0:
                    flux_tally = sp.get_tally(name='thermal_fast_flux')
                    current_tally = sp.get_tally(name='assembly_surface_current')
                    np.save('boc_thermal_fast_flux.npy', flux_tally.mean.ravel())
                    np.save('boc_assembly_current.npy', current_tally.mean.ravel())

        cycle_length = find_cycle_length(react, np.cumsum([0] + STEP_DAYS[:N_STEPS])[1:])

        return dict(pattern=pattern_flat, react=react, ppf=ppf,
                     cycle_length=cycle_length)

    finally:
        os.chdir(cwd0)


def find_cycle_length(react_curve, days):
    """Linear-interpolated zero-crossing of rho(t) — same method as
    01_reproduce_paper.py's find_cycle_length()."""
    for i in range(1, len(react_curve)):
        if react_curve[i] < 0 and react_curve[i - 1] >= 0:
            frac = react_curve[i - 1] / (react_curve[i - 1] - react_curve[i])
            return days[i - 1] + frac * (days[i] - days[i - 1])
    return float(days[-1])   # never went subcritical within the schedule


# =============================================================================
# SECTION 6 — CSV WRITER  (schema-compatible with cnn-v9.py, skiprows=1)
# =============================================================================

def build_csv_header():
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    react_cols = [f'react_{i}' for i in range(N_STEPS)]
    ppf_cols = [f'ppf_s{s}_a{a}' for s in range(N_STEPS) for a in range(N_POS)]
    return load_cols + react_cols + ppf_cols + ['cycle_length']


def result_to_row(result):
    row = list(int(x) for x in result['pattern'])
    row += list(result['react'])
    row += list(result['ppf'].reshape(-1))
    row += [result['cycle_length']]
    return row


def append_row_to_csv(csv_path, header, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('# openmc_beavrs_vver1000.py export — see cnn-v9.py for schema\n')
            f.write(','.join(header) + '\n')
        f.write(','.join(str(x) for x in row) + '\n')


# =============================================================================
# SECTION 7 — DATASET GENERATION LOOP
# =============================================================================

def random_pattern(rng, n_pos=N_POS, n_types=N_TYPES):
    """Positional-only sampling: fixed 9-type inventory rearranged across 31
    positions, matching your memory note on dataset construction. Adjust the
    per-type counts here to match your actual assembly inventory exactly."""
    base_counts = np.full(n_types, n_pos // n_types, dtype=int)
    base_counts[: n_pos % n_types] += 1
    types_pool = np.repeat(np.arange(1, n_types + 1), base_counts)[:n_pos]
    return rng.permutation(types_pool)


def load_al_candidates(al_csv_path):
    """Alternative to random sampling: verify qica_v9-final's flagged AL
    candidates instead of generating fresh random patterns."""
    df = pd.read_csv(al_csv_path)
    pos_cols = [f'pos_{i}' for i in range(N_POS)]
    return df[pos_cols].values.astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_patterns', type=int, default=1)
    ap.add_argument('--chain', type=str, default=os.environ.get('OPENMC_CHAIN', ''))
    ap.add_argument('--particles', type=int, default=4000)
    ap.add_argument('--batches', type=int, default=60)
    ap.add_argument('--inactive', type=int, default=15)
    ap.add_argument('--out_csv', type=str, default='ml_dataset_constrained.csv')
    ap.add_argument('--al_candidates_csv', type=str, default=None,
                     help='If set, verify these patterns instead of random ones '
                          '(e.g. qica_final_al_candidates.csv)')
    ap.add_argument('--al_top_n', type=int, default=None,
                     help='If set with --al_candidates_csv, only run the first N rows '
                          '(e.g. --al_top_n 10 to test 10 candidates instead of all 50)')
    ap.add_argument('--single_pattern', type=str, default=None,
                     help='Comma-separated 31 ints, e.g. the QICA best_pat output: '
                          '"4,1,4,1,4,1,4,4,1,1,3,3,4,5,2,3,1,1,5,1,5,4,3,5,7,1,1,1,1,6,1"')
    ap.add_argument('--quick_check', action='store_true',
                     help='Skip depletion; run one static transport solve for BOC '
                          'k-eff/PPF only. Minutes, not hours. Does not write to CSV.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if 'OPENMC_CROSS_SECTIONS' not in os.environ:
        print('[WARN] OPENMC_CROSS_SECTIONS is not set — OpenMC will fail to find nuclear data.')

    rng = np.random.default_rng(args.seed)
    header = build_csv_header()

    if args.single_pattern:
        pat = np.array([int(x) for x in args.single_pattern.split(',')], dtype=int)
        assert len(pat) == N_POS, f'Expected {N_POS} values, got {len(pat)}'
        patterns = [pat]
        print(f'[MODE] Single specified pattern: {list(pat)}')
    elif args.al_candidates_csv:
        patterns = load_al_candidates(args.al_candidates_csv)
        if args.al_top_n:
            patterns = patterns[:args.al_top_n]
        print(f'[MODE] Verifying {len(patterns)} AL candidates from {args.al_candidates_csv}')
    else:
        patterns = [random_pattern(rng) for _ in range(args.n_patterns)]
        print(f'[MODE] Generating {len(patterns)} random loading patterns')

    if args.quick_check:
        for i, pattern in enumerate(patterns):
            print(f'\n[QUICK CHECK {i+1}/{len(patterns)}]')
            run_quick_check(pattern, particles=args.particles, batches=args.batches,
                             inactive=args.inactive, work_dir=f'openmc_quickcheck_{i:04d}')
        return

    if not args.chain:
        print('[ERROR] No depletion chain file given (required for full runs). '
              'Set --chain or OPENMC_CHAIN, or use --quick_check for a depletion-free test.')
        sys.exit(1)

    for i, pattern in enumerate(patterns):
        print(f'\n[PATTERN {i+1}/{len(patterns)}]  {list(pattern)}')
        t0 = time.time()
        result = run_one_pattern(
            pattern, chain_file=args.chain,
            particles=args.particles, batches=args.batches, inactive=args.inactive,
            work_dir=f'openmc_run_{i:04d}')
        row = result_to_row(result)
        append_row_to_csv(args.out_csv, header, row)
        print(f'  ppf_max={result["ppf"].max():.3f}  '
              f'cycle_length={result["cycle_length"]:.1f}d  '
              f'({time.time()-t0:.0f}s)')

    print(f'\n[DONE] Wrote {len(patterns)} row(s) to {args.out_csv}')
    print('  Feed this file directly into cnn-v9.py (BEAVRS_CSV) or append it')
    print('  to your existing ml_dataset_constrained.csv for retraining.')


if __name__ == '__main__':
    main()