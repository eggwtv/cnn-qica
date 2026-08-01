"""
cd ~/Desktop/qica/cnn-qica
export OPENMC_CROSS_SECTIONS=$PWD/endfb-viii.1-hdf5/cross_sections.xml
echo $OPENMC_CROSS_SECTIONS
conda activate openmc-env

python openmc_beavrs_vver1000_FIXED_2.py --single_pattern "4,1,4,1,4,1,4,4,1,1,3,3,4,5,2,3,1,1,5,1,5,4,3,5,7,1,1,1,1,6,1" \
  --quick_check --particles 4000 --batches 60 --inactive 40

python openmc_beavrs_vver1000_FIXED_2.py --single_pattern "1,3,1,1,2,9,5,1,3,9,7,4,1,5,9,1,5,7,5,5,9,1,8,5,4,1,4,1,5,1,1" \
  --quick_check --particles 4000 --batches 60 --inactive 40
=============================================================================
openmc_beavrs_vver1000_FIXED_2.py  —  High-Fidelity Generator for cnn_v9 / qica_v9-final
=============================================================================
Fixes applied relative to your current openmc_beavrs_vver1000.py, in order
of how much they matter:

FIX 1 — ASSEMBLY_LIBRARY now matches cycle_length_summary.xlsx exactly
────────────────────────────────────────────────────────────────────────
Your xlsx names every type explicitly (fa_name = FA_<enrich*10>_<nBA>BA):
    1: FA_16_NOBA      -> 1.6%,  0 BA
    2: FA_24_NOBA      -> 2.4%,  0 BA
    3: FA_24_12BA_C1   -> 2.4%, 12 BA
    4: FA_24_16BA      -> 2.4%, 16 BA
    5: FA_31_NOBA      -> 3.1%,  0 BA
    6: FA_31_6BA       -> 3.1%,  6 BA
    7: FA_32_15BA      -> 3.2%, 15 BA
    8: FA_31_16BA      -> 3.1%, 16 BA
    9: FA_31_20BA      -> 3.1%, 20 BA
The OLD ASSEMBLY_LIBRARY had invented placeholder enrichments (up to 4.95%!)
and wrong BA pin counts (e.g. type 9 had ZERO BA pins coded, when the real
type 9 has the MOST of any type, 20). This means OpenMC was building a
physically different core than the one that produced your training CSV —
by itself this is large enough to explain a 4x PPF blowout, independent of
anything about surrogate extrapolation.

FIX 2 — Boron convention unified between training generation and quick-check
────────────────────────────────────────────────────────────────────────
OLD run_one_pattern() (training data / full depletion) always used a
hardcoded, uncritical 800 ppm boron. OLD quick_check --boron_search solved
for CRITICAL boron (k=1.000, typically 2000-2500 ppm here). Those are two
different physical states — boron level changes spectral hardening, and
different assembly types (different BA loadings) respond to that
differently, which redistributes power. Comparing a CNN trained on
800-ppm-uncritical labels against an OpenMC check run at critical boron is
not an apples-to-apples comparison.
FIXED: both run_one_pattern() and run_quick_check() now share the SAME
convention, controlled by --boron_search / --boron_ppm:
  - If --boron_search: solve for critical boron ONCE at BOC (fresh fuel),
    then hold that ppm FIXED for the entire depletion (this is the
    standard "BOC-critical, held constant" approximation — a full
    critical-boron-letdown curve would need a boron search at every
    burnup step, which is far more expensive; if you need that level of
    rigor later, call find_critical_boron() inside the per-step loop of
    run_one_pattern() instead of once before it).
  - If not: use the fixed --boron_ppm for both, so at least training and
    verification agree with each other even if it isn't literally critical.
IMPORTANT: if your existing ml_dataset_constrained.csv was generated with
the OLD 800-ppm-hardcoded run_one_pattern(), it is on a DIFFERENT boron
convention than any of your future quick_checks run with this fixed
script. To retrain cleanly, regenerate the dataset with this script.

FIX 3 — PPF normalization: confirmed consistent, but check your CSV's provenance
────────────────────────────────────────────────────────────────────────
compute_ppf_per_assembly() (max pin power / mean pin power) is called
identically by both training-data generation and quick_check, so the
*formula* was never the issue. HOWEVER: that function contains a
row-flip fix for a previous top/bottom mesh-mirroring bug. If
ml_dataset_constrained.csv predates that fix, the CNN was trained on
mirrored/mislabeled PPF targets and no amount of matching assembly types
or boron will make quick_check agree with it. Confirm which script
version produced your current training CSV; if it's the buggy one,
regenerate.

FIX 4 — Boron search ceiling: was fixed at 2500 ppm, now configurable + auto-widening
────────────────────────────────────────────────────────────────────────
find_critical_boron() now takes --boron_hi (default raised to 3500 ppm)
and, if criticality still isn't bracketed, automatically doubles the
upper bound up to --boron_max (default 6000 ppm) before giving up. Note:
real PWR/VVER boric-acid solubility limits typically cap practical boron
around ~2500-3000 ppm — if your search needs much more than that to reach
k=1.000, that loading pattern may not be operationally realizable with
boron control alone (rod insertion / different fuel would be needed in
practice). The wider ceiling here is for diagnosing/comparing patterns
computationally, not a claim that >3000 ppm is achievable in a real core.

RUNTIME: same as before — full-core depletion with pin-resolved tallies
takes tens of minutes to hours on CPU. quick_check remains the fast path.
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

from beavrs_bp_patch import BP_ROD_MAPS
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

# ── Assembly type library — FIX 1: now taken directly from cycle_length_summary.xlsx
# (fa_id, fa_name, monocore_cycle_length). fa_name encodes "FA_<enrich*10>_<nBA>BA".
# gd2o3_wt (integral absorber wt%) is NOT given by the xlsx (it only has
# enrichment + BA rod count baked into the name + the resulting monocore
# cycle length) — held at a single representative value (4.0 wt%) across all
# BA-bearing types as a stated assumption. If you have the real per-type
# Gd2O3 loading (or know these use discrete Pyrex/WABA rods rather than
# integral Gd2O3, which is the more common BEAVRS convention), replace the
# gd2o3_wt values below — the BA *pin count* (which drives most of the power
# suppression) and enrichment (which drives most of the reactivity) are now
# correct either way.
ASSEMBLY_LIBRARY = {
    # id: enrich(wt% U-235), gd2o3_wt(%), n_bp_pins   | xlsx fa_name (monocore EFPD)
    1: dict(enrich=1.6, gd2o3_wt=0.0, n_bp_pins=0),   # FA_16_NOBA     (172.9 d)
    2: dict(enrich=2.4, gd2o3_wt=0.0, n_bp_pins=0),   # FA_24_NOBA     (366.9 d)
    3: dict(enrich=2.4, gd2o3_wt=4.0, n_bp_pins=12),  # FA_24_12BA_C1  (323.2 d)
    4: dict(enrich=2.4, gd2o3_wt=4.0, n_bp_pins=16),  # FA_24_16BA     (299.8 d)
    5: dict(enrich=3.1, gd2o3_wt=0.0, n_bp_pins=0),   # FA_31_NOBA     (520.0 d)
    6: dict(enrich=3.1, gd2o3_wt=4.0, n_bp_pins=6),   # FA_31_6BA      (504.9 d)
    7: dict(enrich=3.2, gd2o3_wt=4.0, n_bp_pins=15),  # FA_32_15BA     (475.3 d)
    8: dict(enrich=3.1, gd2o3_wt=4.0, n_bp_pins=16),  # FA_31_16BA     (471.6 d)
    9: dict(enrich=3.1, gd2o3_wt=4.0, n_bp_pins=20),  # FA_31_20BA     (454.7 d)
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

# ── FIX 2 (REVISED) — boron convention now matches your ACTUAL ml_dataset_constrained.csv ──
# Evidence from the real CSV (10,000 rows, checked directly): implied BOC k-eff via
# react_0 ranges 1.097-1.180 across ALL patterns. A critical-boron search would force
# EVERY pattern to k-eff ~= 1.000 by construction -- it doesn't. That spread instead sits
# squarely inside the individual monocore BOC k-eff values in cycle_length_summary.xlsx's
# All_Cases_Data sheet (1.05-1.28), which are boron-free single-assembly calcs. Conclusion:
# ml_dataset_constrained.csv was generated with little/no soluble boron and NO criticality
# search -- not at 800 ppm, and not at critical boron. Since this CSV is fixed and can't be
# regenerated, OpenMC has to match IT, not some other convention. Default is now 0 ppm.
# --boron_search is kept available (e.g. for exploring a genuinely different, borated
# operating point) but is NOT the right setting to reproduce this dataset's physics --
# using it will evaluate patterns at a k-eff~1.000 state the CNN never saw a single
# training example of, which is its own extrapolation problem layered on top.
BORON_PPM_DEFAULT = 0.0        # matches the unborated convention evident in the real CSV
BORON_LO_DEFAULT  = 200.0
BORON_HI_DEFAULT  = 3500.0     # FIX 4 — raised from 2500 (only relevant if you opt into --boron_search)
BORON_MAX_DEFAULT = 6000.0     # FIX 4 — auto-widening hard stop


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
    water.add_element('B', BORON_PPM_DEFAULT * 1e-6 * (18.015 / 10.811), percent_type='ao')
    water.set_density('g/cm3', 0.712)                    # ~305 C PWR/VVER-ish hot density
    water.add_s_alpha_beta('c_H_in_H2O')

    return fuel, bp_fuel, clad, gt, water


# Fixed reference to the ORIGINAL build_materials, captured once here before
# any monkeypatching. build_materials_with_boron() calls this, never the
# module-level `build_materials` name (which gets temporarily reassigned
# during boron search / boron-fixed runs) — calling the live name would
# recurse into itself infinitely.
_ORIGINAL_build_materials = build_materials


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

'''
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

'''
def build_assembly_universe(type_id):
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
        # REAL BEAVRS BP rod placement (from mit-crpg/BEAVRS official OpenMC
        # model, _add_bpra_layouts() in assemblies.py), replacing the old
        # diagonal-sweep placeholder. BP_ROD_MAPS is keyed by rod count and
        # gives the exact (row,col) positions that carry Gd2O3 rods in the
        # real benchmark, for that count.
        n_bp = ASSEMBLY_LIBRARY[type_id]['n_bp_pins']
        bp_positions = BP_ROD_MAPS.get(n_bp)
        if bp_positions is None:
            raise ValueError(
                f"No real BEAVRS BP map for n_bp_pins={n_bp} (type {type_id}). "
                f"Known counts: {sorted(BP_ROD_MAPS.keys())}"
            )
        for (r, c) in bp_positions:
            # Guard: BP positions are always a subset of GUIDE_TUBE_POS in
            # the real spec (BP rods replace what would otherwise be guide
            # tubes at those 24 labeled positions) — this should never fire,
            # but flags it loudly if BP_ROD_MAPS and GUIDE_TUBE_POS diverge.
            if (r, c) not in GUIDE_TUBE_POS:
                raise ValueError(
                    f"BP position {(r, c)} for count {n_bp} is not in "
                    f"GUIDE_TUBE_POS — label/coordinate mismatch, check "
                    f"GT_LABEL_POS against GUIDE_TUBE_POS."
                )
            universes[r, c] = u_bp
 
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
    Three tally groups:
      1. energy_flux_tally : 2-group (thermal/fast) scalar flux over the
         whole core mesh — diagnostic for spectrum shift vs. loading pattern.
      2. current_tally     : net partial current through the surfaces of a
         mesh matching the assembly pitch — assembly-to-assembly leakage.
      3. power_mesh_tally  : fine (pin-pitch) mesh kappa-fission tally used
         to derive per-assembly PPF at each burnup step. THIS is the tally
         that both cnn-v9's training labels and quick_check's PPF report
         are derived from — keep this function identical everywhere it's
         used, or the CNN and OpenMC are no longer speaking the same
         language regardless of anything else being fixed.
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
# SECTION 3.5 — BORON CRITICAL SEARCH
# =============================================================================

def build_materials_with_boron(type_id, boron_ppm):
    """Same as build_materials() but with a caller-specified boron ppm."""
    fuel, bp_fuel, clad, gt, water = _ORIGINAL_build_materials(type_id)
    water = openmc.Material(name='borated_water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.add_element('B', max(boron_ppm, 1e-6) * 1e-6 * (18.015 / 10.811), percent_type='ao')
    water.set_density('g/cm3', 0.712)
    water.add_s_alpha_beta('c_H_in_H2O')
    return fuel, bp_fuel, clad, gt, water


def _run_static_keff(pattern_flat, boron_ppm, particles, batches, inactive, work_dir):
    """One short static (non-depleted) transport solve at a given boron ppm.
    Returns k-eff (ufloat-like openmc.Tally value with .n / .s)."""
    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        global build_materials
        orig_build_materials = build_materials
        build_materials = lambda t: build_materials_with_boron(t, boron_ppm)
        try:
            assembly_universes = {t: build_assembly_universe(t) for t in range(1, N_TYPES + 1)}
            reflector_u = build_reflector_universe()
            geometry, _ = build_core_geometry(pattern_flat, assembly_universes, reflector_u)
        finally:
            build_materials = orig_build_materials

        geometry.export_to_xml()
        materials = openmc.Materials(geometry.get_all_materials().values())
        materials.export_to_xml()

        settings = openmc.Settings()
        settings.batches, settings.inactive, settings.particles = batches, inactive, particles
        settings.temperature = {'default': 600.0}
        bbox = geometry.bounding_box
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Box(bbox[0], bbox[1], only_fissionable=True))
        settings.export_to_xml()

        tallies, _, _ = build_tallies()
        tallies.export_to_xml()

        openmc.run(output=False)
        sp_path = sorted(glob.glob('statepoint.*.h5'))[-1]
        with openmc.StatePoint(sp_path) as sp:
            keff = sp.keff
        return keff
    finally:
        os.chdir(cwd0)


def find_critical_boron(pattern_flat, particles=1000, batches=25, inactive=10,
                         target_keff=1.0, tol_pcm=100, max_iter=6,
                         work_dir='boron_search',
                         boron_lo=BORON_LO_DEFAULT, boron_hi=BORON_HI_DEFAULT,
                         boron_max=BORON_MAX_DEFAULT, auto_widen=True):
    """
    Secant-method search for the soluble boron ppm that brings this loading
    pattern's BOC k-eff to target_keff (default 1.000).

    FIX 4: if [boron_lo, boron_hi] doesn't bracket criticality on the high
    side (k_hi still > target — pattern is "too reactive" for the current
    ceiling), this now DOUBLES boron_hi (up to boron_max) and retries,
    instead of silently returning the unconverged endpoint as if it were
    an answer. If it still can't bracket at boron_max, it says so plainly:
    that loading pattern likely can't be made critical with boron alone
    within a realistic/allowed range, which is itself a useful diagnostic
    (flag it as a bad/over-reactive pattern rather than reporting a PPF at
    a non-critical, physically inconsistent state).

    Returns (boron_ppm_critical, keff_at_that_ppm, converged: bool).
    """
    print(f'\n[BORON SEARCH] target k-eff={target_keff:.4f} (tol {tol_pcm} pcm), '
          f'{particles}p x {batches}b per trial, up to {max_iter} iterations, '
          f'range [{boron_lo:.0f}, {boron_hi:.0f}] ppm')

    b_lo, b_hi = boron_lo, boron_hi
    k_lo = _run_static_keff(pattern_flat, b_lo, particles, batches, inactive,
                             f'{work_dir}_lo').n
    k_hi = _run_static_keff(pattern_flat, b_hi, particles, batches, inactive,
                             f'{work_dir}_hi').n
    print(f'  trial 1: boron={b_lo:.0f}ppm  k-eff={k_lo:.5f}')
    print(f'  trial 2: boron={b_hi:.0f}ppm  k-eff={k_hi:.5f}')

    # Auto-widen: keep doubling boron_hi while the pattern is still
    # supercritical even at the current ceiling.
    widen_tries = 0
    while k_hi > target_keff and auto_widen and b_hi < boron_max:
        widen_tries += 1
        new_hi = min(b_hi * 2.0, boron_max)
        print(f'  [WIDEN] k_hi={k_hi:.5f} still > target at {b_hi:.0f} ppm — '
              f'expanding ceiling to {new_hi:.0f} ppm (try {widen_tries})')
        b_hi = new_hi
        k_hi = _run_static_keff(pattern_flat, b_hi, particles, batches, inactive,
                                 f'{work_dir}_hi_widen{widen_tries}').n
        print(f'  trial (widen {widen_tries}): boron={b_hi:.0f}ppm  k-eff={k_hi:.5f}')

    if not (k_lo >= target_keff >= k_hi):
        print(f'  [WARN] target k-eff NOT bracketed even at boron_max={boron_max:.0f} ppm '
              f'(k_lo={k_lo:.5f}, k_hi={k_hi:.5f}). This pattern is likely too reactive to '
              f'control with soluble boron alone within the allowed range. Returning nearest '
              f'endpoint, flagged as NON-CONVERGED — treat any PPF computed at this boron '
              f'level as provisional, not a true critical-state result.')
        b_final, k_final = (b_lo, k_lo) if abs(k_lo - target_keff) < abs(k_hi - target_keff) else (b_hi, k_hi)
        return b_final, k_final, False

    b_a, k_a, b_b, k_b = b_lo, k_lo, b_hi, k_hi
    for it in range(3, max_iter + 1):
        b_mid = b_a + (target_keff - k_a) * (b_b - b_a) / (k_b - k_a)
        b_mid = float(np.clip(b_mid, min(b_lo, b_hi), max(b_lo, b_hi)))
        k_mid = _run_static_keff(pattern_flat, b_mid, particles, batches, inactive,
                                  f'{work_dir}_it{it}').n
        print(f'  trial {it}: boron={b_mid:.0f}ppm  k-eff={k_mid:.5f}  '
              f'(target {target_keff:.4f} +/- {tol_pcm/1e5:.5f})')
        if abs(k_mid - target_keff) * 1e5 <= tol_pcm:
            print(f'  [CONVERGED] critical boron ~= {b_mid:.0f} ppm')
            return b_mid, k_mid, True
        if k_mid > target_keff:
            b_a, k_a = b_mid, k_mid
        else:
            b_b, k_b = b_mid, k_mid

    print(f'  [STOP] max_iter reached, best estimate boron={b_mid:.0f} ppm, k-eff={k_mid:.5f}')
    return b_mid, k_mid, False


# =============================================================================
# SECTION 4 — PPF EXTRACTION FROM PIN-POWER MESH
# =============================================================================

def compute_ppf_per_assembly(pin_power_mesh_values, n_pins_x, n_pins_y):
    """
    pin_power_mesh_values: flat array of kappa-fission tally means, ordered
    per OpenMC's MeshFilter convention (x fastest, then y, y increasing
    from the mesh's lower_left).

    NOTE ON CONSISTENCY: this function must be byte-for-byte identical
    between whatever generated ml_dataset_constrained.csv and whatever you
    run quick_check with. If they ever diverge (e.g. one has the row-flip
    fix and one doesn't), the CNN and OpenMC are scoring PPF on different
    definitions and no other fix in this file will make them agree.

    Returns ppf_per_assembly: (N_POS,) array, one PPF per active assembly
    position, in the SAME 0..30 ordering as loading_/ppf_s{step}_a{i}.
    """
    grid2d = pin_power_mesh_values.reshape(n_pins_y, n_pins_x)
    grid2d = grid2d[::-1, :]   # row 0 -> max y, matching RectLattice convention

    core_mean_pin_power = grid2d[grid2d > 0].mean() if np.any(grid2d > 0) else 1.0

    ppf_flat = np.zeros(N_POS, dtype=np.float64)
    zero_block_positions = []
    pos_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] < 0:
                continue
            sub = grid2d[r * PINS_PER_SIDE:(r + 1) * PINS_PER_SIDE,
                         c * PINS_PER_SIDE:(c + 1) * PINS_PER_SIDE]
            max_pin_power = sub.max() if sub.size else 0.0
            if max_pin_power <= 0.0:
                zero_block_positions.append(pos_idx)
            ppf_flat[pos_idx] = max_pin_power / (core_mean_pin_power + 1e-12)
            pos_idx += 1

    if zero_block_positions:
        print(f'  [WARN] {len(zero_block_positions)} assembly position(s) came back with '
              f'ZERO pin power: {zero_block_positions}')
    return ppf_flat


# =============================================================================
# SECTION 4.5 — QUICK CHECK (single static transport solve, no depletion)
# =============================================================================

def run_quick_check(pattern_flat, particles=2000, batches=40, inactive=15,
                     work_dir='openmc_quickcheck', boron_search=False,
                     boron_ppm=BORON_PPM_DEFAULT,
                     boron_lo=BORON_LO_DEFAULT, boron_hi=BORON_HI_DEFAULT,
                     boron_max=BORON_MAX_DEFAULT):
    """
    Fast sanity check for one loading pattern: BOC k-eff and PPF only.
    FIX 2: boron_search here and in run_one_pattern() below now go through
    the exact same find_critical_boron() call with the same defaults, so
    a quick_check and the training-data-generating depletion run are
    evaluated under the same boron convention.
    """
    converged = True
    if boron_search:
        print('[WARN] --boron_search evaluates this pattern at k-eff~=1.000 (critical boron), '
              'but ml_dataset_constrained.csv was generated UNBORATED (BOC k-eff 1.10-1.18 '
              'across all 10,000 real patterns, never near 1.000). This will NOT be an '
              'apples-to-apples comparison against your CNN, which never saw a critical-boron '
              'training example. Omit --boron_search (default boron_ppm=0) to match the CSV.')
        boron_ppm, keff_est, converged = find_critical_boron(
            pattern_flat, particles=max(1000, particles // 2),
            batches=max(25, batches // 2), inactive=max(10, inactive // 2),
            work_dir=f'{work_dir}_boronsearch',
            boron_lo=boron_lo, boron_hi=boron_hi, boron_max=boron_max)
        tag = 'CONVERGED' if converged else 'NOT CONVERGED — treat PPF below as provisional'
        print(f'[BORON SEARCH] using boron={boron_ppm:.0f} ppm '
              f'(est. k-eff={keff_est:.5f}, {tag}) for the PPF report below')

    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)
    try:
        global build_materials
        orig_build_materials = build_materials
        if boron_search or boron_ppm != BORON_PPM_DEFAULT:
            build_materials = lambda t: build_materials_with_boron(t, boron_ppm)
        try:
            assembly_universes = {t: build_assembly_universe(t) for t in range(1, N_TYPES + 1)}
            reflector_u = build_reflector_universe()
            geometry, core_lat = build_core_geometry(pattern_flat, assembly_universes, reflector_u)
        finally:
            build_materials = orig_build_materials
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
        print(f'QUICK CHECK RESULT  ({elapsed:.0f}s, {particles}p x {batches}b, '
              f'boron={boron_ppm:.0f}ppm, boron_converged={converged})')
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
        return dict(keff=keff.n, react=react, ppf=ppf, boron_ppm=boron_ppm, boron_converged=converged)
    finally:
        os.chdir(cwd0)


# =============================================================================
# SECTION 5 — SINGLE-PATTERN DEPLETION RUN
# =============================================================================

def run_one_pattern(pattern_flat, chain_file, particles=4000, batches=60,
                     inactive=15, work_dir='openmc_run',
                     boron_search=False, boron_ppm=BORON_PPM_DEFAULT,
                     boron_lo=BORON_LO_DEFAULT, boron_hi=BORON_HI_DEFAULT,
                     boron_max=BORON_MAX_DEFAULT):
    """
    Runs full depletion for one 31-length loading pattern and returns a dict
    with everything cnn-v9.py's CSV schema needs.

    FIX 2: this now accepts boron_search / boron_ppm with the SAME meaning
    and SAME find_critical_boron() call as run_quick_check(). If
    boron_search=True, a critical-boron search is run ONCE on the fresh
    (BOC) fuel, and that ppm is then held FIXED across the whole depletion
    (a standard "BOC-critical held constant" approximation — not a full
    per-step critical boron letdown curve, which would need a search at
    every burnup step). Whichever convention you pick, use the SAME flags
    for both this function and run_quick_check() so training labels and
    verification checks are evaluated under matching physics.
    """
    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)

    try:
        converged = True
        if boron_search:
            print('[WARN] --boron_search targets k-eff~=1.000, but ml_dataset_constrained.csv '
                  'is unborated (BOC k-eff 1.10-1.18). Use this only if you intend to generate '
                  'genuinely NEW, borated-state training data as a separate/future dataset — '
                  'not to verify against the existing CSV.')
            boron_ppm, keff_est, converged = find_critical_boron(
                pattern_flat, particles=max(1000, particles // 2),
                batches=max(25, batches // 2), inactive=max(10, inactive // 2),
                work_dir='boronsearch_boc',
                boron_lo=boron_lo, boron_hi=boron_hi, boron_max=boron_max)
            print(f'[BORON SEARCH] BOC critical boron = {boron_ppm:.0f} ppm '
                  f'(k_est={keff_est:.5f}, converged={converged}) — held fixed for depletion')

        global build_materials
        orig_build_materials = build_materials
        if boron_search or boron_ppm != BORON_PPM_DEFAULT:
            build_materials = lambda t: build_materials_with_boron(t, boron_ppm)
        try:
            assembly_universes = {t: build_assembly_universe(t) for t in range(1, N_TYPES + 1)}
            reflector_u = build_reflector_universe()
            geometry, core_lat = build_core_geometry(pattern_flat, assembly_universes, reflector_u)
        finally:
            build_materials = orig_build_materials

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

        sp_files = sorted(glob.glob('openmc_simulation_n*.h5'),
                           key=lambda f: int(''.join(filter(str.isdigit, f)) or -1))
        if not sp_files:
            sp_files = sorted(glob.glob('statepoint*.h5'))

        for step, sp_path in enumerate(sp_files[:N_STEPS]):
            with openmc.StatePoint(sp_path) as sp:
                keff = sp.keff.n
                react[step] = (keff - 1.0) / keff

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
                     cycle_length=cycle_length, boron_ppm=boron_ppm,
                     boron_converged=converged)

    finally:
        os.chdir(cwd0)


def find_cycle_length(react_curve, days):
    """Linear-interpolated zero-crossing of rho(t)."""
    for i in range(1, len(react_curve)):
        if react_curve[i] < 0 and react_curve[i - 1] >= 0:
            frac = react_curve[i - 1] / (react_curve[i - 1] - react_curve[i])
            return days[i - 1] + frac * (days[i] - days[i - 1])
    return float(days[-1])


# =============================================================================
# SECTION 6 — CSV WRITER  (schema-compatible with cnn-v9.py, skiprows=1)
# =============================================================================

def build_csv_header():
    load_cols = [f'loading_{i}' for i in range(N_POS)]
    react_cols = [f'react_{i}' for i in range(N_STEPS)]
    ppf_cols = [f'ppf_s{s}_a{a}' for s in range(N_STEPS) for a in range(N_POS)]
    return load_cols + react_cols + ppf_cols + ['cycle_length', 'boron_ppm', 'boron_converged']


def result_to_row(result):
    row = list(int(x) for x in result['pattern'])
    row += list(result['react'])
    row += list(result['ppf'].reshape(-1))
    row += [result['cycle_length'], result.get('boron_ppm', BORON_PPM_DEFAULT),
            int(result.get('boron_converged', True))]
    return row


def append_row_to_csv(csv_path, header, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('# openmc_beavrs_vver1000_FIXED.py export — see cnn-v9.py for schema\n')
            f.write(','.join(header) + '\n')
        f.write(','.join(str(x) for x in row) + '\n')


# =============================================================================
# SECTION 7 — DATASET GENERATION LOOP
# =============================================================================

def random_pattern(rng, n_pos=N_POS, n_types=N_TYPES):
    base_counts = np.full(n_types, n_pos // n_types, dtype=int)
    base_counts[: n_pos % n_types] += 1
    types_pool = np.repeat(np.arange(1, n_types + 1), base_counts)[:n_pos]
    return rng.permutation(types_pool)


def load_al_candidates(al_csv_path):
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
    ap.add_argument('--al_candidates_csv', type=str, default=None)
    ap.add_argument('--al_top_n', type=int, default=None)
    ap.add_argument('--single_pattern', type=str, default=None)
    ap.add_argument('--quick_check', action='store_true')
    ap.add_argument('--boron_search', action='store_true',
                     help='Solve for critical boron (k=1.000) instead of using a fixed ppm. '
                          'Applies to BOTH --quick_check and full depletion runs now.')
    ap.add_argument('--boron_ppm', type=float, default=BORON_PPM_DEFAULT,
                     help='Fixed boron ppm to use if --boron_search is not set.')
    ap.add_argument('--boron_lo', type=float, default=BORON_LO_DEFAULT,
                     help='Lower bound (ppm) for the critical boron search.')
    ap.add_argument('--boron_hi', type=float, default=BORON_HI_DEFAULT,
                     help='Upper bound (ppm) for the critical boron search (was fixed at 2500; '
                          'default is now %(default)s and will auto-widen further, see --boron_max).')
    ap.add_argument('--boron_max', type=float, default=BORON_MAX_DEFAULT,
                     help='Hard ceiling (ppm) the auto-widening search will not exceed.')
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
                             inactive=args.inactive, work_dir=f'openmc_quickcheck_{i:04d}',
                             boron_search=args.boron_search, boron_ppm=args.boron_ppm,
                             boron_lo=args.boron_lo, boron_hi=args.boron_hi,
                             boron_max=args.boron_max)
        return

    if not args.chain:
        print('[ERROR] No depletion chain file given. Set --chain or OPENMC_CHAIN, '
              'or use --quick_check for a depletion-free test.')
        sys.exit(1)

    for i, pattern in enumerate(patterns):
        print(f'\n[PATTERN {i+1}/{len(patterns)}]  {list(pattern)}')
        t0 = time.time()
        result = run_one_pattern(
            pattern, chain_file=args.chain,
            particles=args.particles, batches=args.batches, inactive=args.inactive,
            work_dir=f'openmc_run_{i:04d}',
            boron_search=args.boron_search, boron_ppm=args.boron_ppm,
            boron_lo=args.boron_lo, boron_hi=args.boron_hi, boron_max=args.boron_max)
        row = result_to_row(result)
        append_row_to_csv(args.out_csv, header, row)
        print(f'  ppf_max={result["ppf"].max():.3f}  '
              f'cycle_length={result["cycle_length"]:.1f}d  '
              f'boron={result["boron_ppm"]:.0f}ppm  '
              f'({time.time()-t0:.0f}s)')

    print(f'\n[DONE] Wrote {len(patterns)} row(s) to {args.out_csv}')
    print('  NOTE: this CSV now has boron_ppm / boron_converged columns appended.')
    print('  cnn-v9.py currently only reads loading_/react_/ppf_/cycle_length columns,')
    print('  so it will ignore the two new trailing columns and keep working unmodified —')
    print('  but they let you sanity-check, per row, what boron state each label came from.')


if __name__ == '__main__':
    main()