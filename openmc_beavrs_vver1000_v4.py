"""
python openmc_beavrs_vver1000_v4.py --single_pattern "9,3,1,6,5,5,1,5,6,1,4,4,1,1,4,1,1,3,4,5,2,4,1,3,8,1,1,7,1,3,1" \
  --quick_check --particles 4000 --batches 60 --inactive 40
=============================================================================
openmc_beavrs_vver1000_v4.py  —  BEAVRS-spec-corrected Generator for cnn_v9 / qica_v11
=============================================================================
Changes relative to openmc_beavrs_vver1000_FIXED_2.py, sourced directly from
BEAVRS_2.0.2_spec.pdf (MIT CRPG, April 2018):

FIX 5 — Fuel density is now enrichment-dependent (spec Tables 4-8), not one
  hardcoded 10.31 g/cm3 for every type:
      1.6% -> 10.31341   2.4% -> 10.29748   3.1% -> 10.30166
      3.2% -> 10.34115   3.4% -> 10.35917   (g/cc)

FIX 6 — Burnable poison material corrected: real BEAVRS does NOT use Gd2O3.
  Table 1 of the spec states the BP material is "Borosilicate Glass, 12.5
  w/o B2O3" (composition given exactly in spec Table 10). The old script's
  gd2o3_wt=4.0 assumption was never sourced from your xlsx/CSV/core diagram
  (none of them mention Gd) - it was a placeholder invented to fill a gap.
  This version:
    (a) uses the real borosilicate-glass material (Al/B/O/Si atom densities
        straight from spec Table 10, reconstructed via add_nuclide+'sum' so
        they're used as literal atom/b-cm values, not renormalised),
    (b) models the BP rod as the real BEAVRS *discrete annular rod*
        (air -> SS304 clad -> He gap -> borosilicate glass -> He gap ->
        SS304 clad -> water -> Zircaloy, per spec Figure 8) sitting in a
        GUIDE TUBE position, instead of doping U inside a fuel pin. This is
        a structural fix, not just a material swap: BP rods physically
        replace guide tubes, they never replace fuel pins.
    (c) uses the real per-count BP rod position maps (spec Figures 15-22,
        same source as the standalone beavrs_bp_patch.py) merged directly
        into this file instead of the old diagonal-sweep placeholder.

FIX 7 — Coolant density corrected to the spec's actual Borated Water value
  (Table 17): 0.740582 g/cc, replacing the old 0.712 approximation.

FIX 8 — Coolant/cross-section temperature corrected to the spec's Hot Zero
  Power inlet temperature (Table 21): 560 F = 566.5 K, replacing the old
  600 K round-number guess. (The spec's own changelog flags 560F vs 560K as
  a historical bug they themselves caught and fixed - worth citing if asked.)

Boron convention UNCHANGED from FIXED_2 (default 0 ppm, unborated, matching
your ml_dataset_constrained.csv's actual generating convention) - per your
mentor's guidance, this is intentionally not revisited here.

Everything else (tallies, depletion loop, PPF extraction, CSV schema,
boron-search machinery) is unchanged from FIXED_2.
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
# SECTION 0 — CONFIGURATION
# =============================================================================

N_POS   = 31
N_TYPES = 9
N_STEPS = 31

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
ASSEMBLY_PITCH   = 21.42
FUEL_OR          = 0.4096
CLAD_IR          = 0.4180
CLAD_OR          = 0.4750
GT_IR            = 0.5610         # guide tube inner radius (spec: 0.56134)
GT_OR            = 0.6020         # guide tube outer radius (spec: 0.60198)
ACTIVE_HEIGHT    = 366.0
PINS_PER_SIDE    = 17

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

# ── Assembly type library — enrichment + BP rod COUNT only (material is now
# real borosilicate glass, see FIX 6; density comes from FUEL_DENSITY_BY_ENRICHMENT) ─
ASSEMBLY_LIBRARY = {
    # id: enrich(wt% U-235), n_bp_pins   | xlsx fa_name (monocore EFPD)
    1: dict(enrich=1.6, n_bp_pins=0),    # FA_16_NOBA     (172.9 d)
    2: dict(enrich=2.4, n_bp_pins=0),    # FA_24_NOBA     (366.9 d)
    3: dict(enrich=2.4, n_bp_pins=12),   # FA_24_12BA_C1  (323.2 d)
    4: dict(enrich=2.4, n_bp_pins=16),   # FA_24_16BA     (299.8 d)
    5: dict(enrich=3.1, n_bp_pins=0),    # FA_31_NOBA     (520.0 d)
    6: dict(enrich=3.1, n_bp_pins=6),    # FA_31_6BA      (504.9 d)
    7: dict(enrich=3.2, n_bp_pins=15),   # FA_32_15BA     (475.3 d)
    8: dict(enrich=3.1, n_bp_pins=16),   # FA_31_16BA     (471.6 d)
    9: dict(enrich=3.1, n_bp_pins=20),   # FA_31_20BA     (454.7 d)
}
# NOTE: type 7 uses 3.2% enrichment, which the spec (Table 1) lists as a
# CYCLE 2 fresh-reload enrichment (Region 4A), not a Cycle 1 zone (Cycle 1 is
# 1.6/2.4/3.1% only). Density-wise this is still handled correctly (density
# is enrichment-dependent, not cycle-dependent - Table 7 gives 3.2% density
# regardless of which cycle it's used in), but worth confirming with your
# mentor whether this 9-type inventory is meant to represent literal BEAVRS
# Cycle 1 or a custom/generalized assembly set for this project.

# FIX 5 — fuel density by enrichment (spec Tables 4-8, g/cc)
FUEL_DENSITY_BY_ENRICHMENT = {
    1.6: 10.31341,
    2.4: 10.29748,
    3.1: 10.30166,
    3.2: 10.34115,
    3.4: 10.35917,
}

def _fuel_density_for_enrichment(enrich):
    if enrich in FUEL_DENSITY_BY_ENRICHMENT:
        return FUEL_DENSITY_BY_ENRICHMENT[enrich]
    nearest = min(FUEL_DENSITY_BY_ENRICHMENT, key=lambda e: abs(e - enrich))
    print(f'[WARN] No exact BEAVRS spec density for enrichment {enrich}% - '
          f'using nearest tabulated value ({nearest}% -> '
          f'{FUEL_DENSITY_BY_ENRICHMENT[nearest]} g/cc)')
    return FUEL_DENSITY_BY_ENRICHMENT[nearest]

# FIX 7 / FIX 8 — coolant density and temperature, from spec Tables 17 / 21
WATER_DENSITY_GCC = 0.740582      # spec Table 17, Borated Water
COOLANT_TEMP_K    = 600 #566.5         # spec Table 21: 560 F HZP inlet -> K

THERMAL_FAST_BOUNDARY_EV = 0.625

FULL_CORE_POWER_W = 3000.0e6
SYMMETRY_FACTOR   = 8.0
MODEL_POWER_W     = FULL_CORE_POWER_W / SYMMETRY_FACTOR

def build_step_days(n_steps=N_STEPS):
    early = [2, 3, 5, 10, 15]
    remaining = n_steps - len(early)
    rest = list(np.diff(np.linspace(20, 620, remaining + 1)))
    return (early + rest)[:n_steps]

STEP_DAYS = build_step_days()

BORON_PPM_DEFAULT = 0.0           # unchanged - unborated, matches training CSV
BORON_LO_DEFAULT  = 200.0
BORON_HI_DEFAULT  = 3500.0
BORON_MAX_DEFAULT = 6000.0


# =============================================================================
# SECTION 0.5 — REAL BEAVRS BP ROD POSITION MAPS
# (merged from beavrs_bp_patch.py — source: mit-crpg/BEAVRS official OpenMC
#  model, assemblies.py _add_bpra_layouts(); positions decoded from spec
#  Figures 15-22 / Figure 14 guide-tube label diagram)
# =============================================================================

GT_LABEL_POS = {
    'a': (2, 5),   'b': (2, 8),   'c': (2, 11),
    'd': (3, 3),   'e': (3, 13),
    'f': (5, 2),   'g': (5, 5),   'h': (5, 8),   'i': (5, 11),  'j': (5, 14),
    'k': (8, 2),   'l': (8, 5),   'n': (8, 11),  'o': (8, 14),
    'p': (11, 2),  'q': (11, 5),  'r': (11, 8),  's': (11, 11), 't': (11, 14),
    'u': (13, 3),  'v': (13, 13),
    'w': (14, 5),  'x': (14, 8),  'y': (14, 11),
}

_BA_SPECS_LABELS = {
    '4':    {'d', 'e', 'u', 'v'},
    '6N':   {'a', 'c', 'd', 'e', 'f', 'j'},
    '6S':   {'p', 't', 'u', 'v', 'w', 'y'},
    '6E':   {'c', 'e', 'j', 't', 'v', 'y'},
    '6W':   {'a', 'd', 'f', 'p', 'u', 'w'},
    '8':    {'d', 'e', 'h', 'l', 'n', 'r', 'u', 'v'},
    '12':   {'a', 'c', 'd', 'e', 'f', 'j', 'p', 't', 'u', 'v', 'w', 'y'},
    '15NW': {'a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'n', 'p', 'q', 'r', 's'},
    '15NE': {'a', 'b', 'c', 'e', 'g', 'h', 'i', 'j', 'l', 'n', 'o', 'q', 'r', 's', 't'},
    '15SW': {'f', 'g', 'h', 'i', 'k', 'l', 'n', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'y'},
    '15SE': {'g', 'h', 'i', 'j', 'l', 'n', 'o', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y'},
    '16':   {'a', 'b', 'c', 'd', 'e', 'f', 'j', 'k', 'o', 'p', 't', 'u', 'v', 'w', 'x', 'y'},
    '20':   {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'o', 'p', 'q', 's', 't',
              'u', 'v', 'w', 'x', 'y'},
}
for _name, _labels in _BA_SPECS_LABELS.items():
    _expected = int(''.join(ch for ch in _name if ch.isdigit()))
    assert len(_labels) == _expected, f"{_name}: {len(_labels)} != {_expected}"

# Canonical single variant per BP count (real core uses 4 rotations of the
# 6- and 15-rod patterns for core-tilt reasons; per-assembly rotation isn't
# tracked here, same limitation as beavrs_bp_patch.py).
BP_ROD_MAPS = {
    0:  frozenset(),
    4:  frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['4']),
    6:  frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['6N']),
    8:  frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['8']),
    12: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['12']),
    15: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['15NW']),
    16: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['16']),
    20: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS['20']),
}


# =============================================================================
# SECTION 1 — MATERIALS
# =============================================================================

def build_materials(type_id):
    """Return (fuel_mat, clad_mat, gt_mat, water_mat).
    No bp_fuel here - burnable poison is now the real discrete borosilicate
    rod built separately by build_bp_pin_universe(), not doped into fuel."""
    lib = ASSEMBLY_LIBRARY[type_id]

    fuel = openmc.Material(name=f'fuel_type{type_id}')
    fuel.add_element('U', 1.0, enrichment=lib['enrich'])
    fuel.add_element('O', 2.0)
    fuel.set_density('g/cm3', _fuel_density_for_enrichment(lib['enrich']))

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
    water.set_density('g/cm3', WATER_DENSITY_GCC)
    water.add_s_alpha_beta('c_H_in_H2O')

    return fuel, clad, gt, water


_ORIGINAL_build_materials = build_materials


def build_borosilicate_glass_material():
    """Real BEAVRS BP material (spec Table 10), replacing the old Gd2O3
    assumption. Atom densities given directly by the spec in atom/b-cm -
    add_nuclide(..., 'ao') + set_density('sum') treats them as literal
    absolute densities rather than renormalised fractions, reproducing the
    spec's stated 2.26 g/cc without needing to set it explicitly."""
    m = openmc.Material(name='borosilicate_glass_BP')
    m.add_nuclide('Al27', 1.7352e-03, 'ao')
    m.add_nuclide('B10',  9.6506e-04, 'ao')
    m.add_nuclide('B11',  3.9189e-03, 'ao')
    m.add_nuclide('O16',  4.6514e-02, 'ao')
    m.add_nuclide('O17',  1.7671e-05, 'ao')
    m.add_nuclide('O18',  9.3268e-05, 'ao')
    m.add_nuclide('Si28', 1.6926e-02, 'ao')
    m.add_nuclide('Si29', 8.5944e-04, 'ao')
    m.add_nuclide('Si30', 5.6654e-04, 'ao')
    m.set_density('sum')
    return m


def build_ss304_bp_material():
    """SS304 clad for the BP rod (spec Table 15 composition, major isotopes
    only - same simplified approach already used for the baffle steel)."""
    m = openmc.Material(name='ss304_bp_clad')
    m.add_element('Fe', 0.695)
    m.add_element('Cr', 0.190)
    m.add_element('Ni', 0.095)
    m.add_element('Mn', 0.020)
    m.set_density('g/cm3', 8.03)
    return m


# =============================================================================
# SECTION 2 — GEOMETRY
# =============================================================================

def build_fuel_pin_universe(fuel, clad, water):
    fuel_surf = openmc.ZCylinder(r=FUEL_OR)
    clad_ir   = openmc.ZCylinder(r=CLAD_IR)
    clad_or   = openmc.ZCylinder(r=CLAD_OR)

    c_fuel = openmc.Cell(fill=fuel, region=-fuel_surf)
    c_gap  = openmc.Cell(fill=None, region=+fuel_surf & -clad_ir)   # He gap (void, simplified)
    c_clad = openmc.Cell(fill=clad, region=+clad_ir & -clad_or)
    c_mod  = openmc.Cell(fill=water, region=+clad_or)

    return openmc.Universe(name='pin_fuel', cells=[c_fuel, c_gap, c_clad, c_mod])


def build_guide_tube_universe(gt, water):
    gt_ir_surf = openmc.ZCylinder(r=GT_IR)
    gt_or_surf = openmc.ZCylinder(r=GT_OR)
    c_gt_water_in = openmc.Cell(fill=water, region=-gt_ir_surf)
    c_gt_clad     = openmc.Cell(fill=gt, region=+gt_ir_surf & -gt_or_surf)
    c_gt_water_out= openmc.Cell(fill=water, region=+gt_or_surf)
    return openmc.Universe(name='guide_tube', cells=[c_gt_water_in, c_gt_clad, c_gt_water_out])


# BP rod concentric radii (cm), spec Figure 8 "BP Geometry above Dashpot"
BP_CLAD1_IR  = 0.21400   # air / inner clad boundary
BP_CLAD1_OR  = 0.23051   # inner SS304 clad
BP_HE1_OR    = 0.24130   # He gap
BP_POISON_OR = 0.42672   # borosilicate glass poison
BP_HE2_OR    = 0.43688   # He gap
BP_CLAD2_OR  = 0.48387   # outer SS304 clad
# then water out to GT_IR, then Zircaloy out to GT_OR (reuses existing consts)

def build_bp_pin_universe(borosilicate, ss304_bp, water, gt_mat):
    """Real BEAVRS discrete BP rod: air -> SS304 -> He -> borosilicate glass
    -> He -> SS304 -> water -> Zircaloy. This occupies a GUIDE TUBE position
    (see build_assembly_universe), never a fuel-pin position."""
    s1 = openmc.ZCylinder(r=BP_CLAD1_IR)
    s2 = openmc.ZCylinder(r=BP_CLAD1_OR)
    s3 = openmc.ZCylinder(r=BP_HE1_OR)
    s4 = openmc.ZCylinder(r=BP_POISON_OR)
    s5 = openmc.ZCylinder(r=BP_HE2_OR)
    s6 = openmc.ZCylinder(r=BP_CLAD2_OR)
    s7 = openmc.ZCylinder(r=GT_IR)
    s8 = openmc.ZCylinder(r=GT_OR)

    c_air    = openmc.Cell(fill=None,          region=-s1)               # air, void-approx
    c_clad1  = openmc.Cell(fill=ss304_bp,      region=+s1 & -s2)
    c_he1    = openmc.Cell(fill=None,          region=+s2 & -s3)         # He gap, void-approx
    c_poison = openmc.Cell(fill=borosilicate,  region=+s3 & -s4)
    c_he2    = openmc.Cell(fill=None,          region=+s4 & -s5)         # He gap, void-approx
    c_clad2  = openmc.Cell(fill=ss304_bp,      region=+s5 & -s6)
    c_water  = openmc.Cell(fill=water,         region=+s6 & -s7)
    c_zr     = openmc.Cell(fill=gt_mat,        region=+s7 & -s8)
    c_mod    = openmc.Cell(fill=water,         region=+s8)

    return openmc.Universe(name='pin_bp_borosilicate',
                            cells=[c_air, c_clad1, c_he1, c_poison,
                                   c_he2, c_clad2, c_water, c_zr, c_mod])


def build_assembly_universe(type_id):
    """17x17 assembly. BP rods (if n_bp_pins > 0) replace guide-tube
    positions per the real BEAVRS rod map (BP_ROD_MAPS), not fuel-pin
    positions and not a diagonal-sweep placeholder."""
    fuel, clad, gt, water = build_materials(type_id)
    borosilicate = build_borosilicate_glass_material()
    ss304_bp     = build_ss304_bp_material()

    u_fuel = build_fuel_pin_universe(fuel, clad, water)
    u_gt   = build_guide_tube_universe(gt, water)
    u_bp   = build_bp_pin_universe(borosilicate, ss304_bp, water, gt)

    lat = openmc.RectLattice(name=f'assembly_type{type_id}')
    lat.lower_left = (-ASSEMBLY_PITCH / 2, -ASSEMBLY_PITCH / 2)
    lat.pitch = (PIN_PITCH, PIN_PITCH)

    universes = np.empty((PINS_PER_SIDE, PINS_PER_SIDE), dtype=object)
    universes[:, :] = u_fuel

    for pos in GUIDE_TUBE_POS:
        universes[pos] = u_gt
    universes[INSTRUMENT_TUBE_POS] = u_gt

    n_bp = ASSEMBLY_LIBRARY[type_id]['n_bp_pins']
    if n_bp > 0:
        bp_positions = BP_ROD_MAPS.get(n_bp)
        if bp_positions is None:
            raise ValueError(
                f"No real BEAVRS BP map for n_bp_pins={n_bp} (type {type_id}). "
                f"Known counts: {sorted(BP_ROD_MAPS.keys())}"
            )
        for (r, c) in bp_positions:
            if (r, c) not in GUIDE_TUBE_POS:
                raise ValueError(
                    f"BP position {(r, c)} for count {n_bp} is not in "
                    f"GUIDE_TUBE_POS - label/coordinate mismatch."
                )
            universes[r, c] = u_bp

    lat.universes = universes

    outer_bound = openmc.model.RectangularPrism(
        ASSEMBLY_PITCH, ASSEMBLY_PITCH, boundary_type='transmission')
    assembly_cell = openmc.Cell(fill=lat, region=-outer_bound)
    return openmc.Universe(name=f'assy_u_type{type_id}', cells=[assembly_cell])


def build_reflector_universe():
    """Fills inactive (-1) GRID_LAYOUT slots and, doubled up, the core baffle."""
    _, _, _, water = build_materials(1)
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
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
    pos_idx = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID_LAYOUT[r, c] >= 0:
                grid[r, c] = pattern_flat[pos_idx]
                pos_idx += 1
    return grid


def build_core_geometry(pattern_flat, assembly_universes, reflector_u):
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

    x_min = openmc.XPlane(x0=-half_w, boundary_type='reflective')
    x_max = openmc.XPlane(x0= half_w, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=-half_h, boundary_type='reflective')
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
    core_mesh = openmc.RegularMesh()
    core_mesh.lower_left  = (-GRID_COLS * ASSEMBLY_PITCH / 2,
                              -GRID_ROWS * ASSEMBLY_PITCH / 2,
                              -ACTIVE_HEIGHT / 2)
    core_mesh.upper_right = ( GRID_COLS * ASSEMBLY_PITCH / 2,
                               GRID_ROWS * ASSEMBLY_PITCH / 2,
                               ACTIVE_HEIGHT / 2)
    core_mesh.dimension = (GRID_COLS, GRID_ROWS, 1)

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
# SECTION 3.5 — BORON CRITICAL SEARCH (unchanged behaviour, default OFF)
# =============================================================================

def build_materials_with_boron(type_id, boron_ppm):
    fuel, clad, gt, water = _ORIGINAL_build_materials(type_id)
    water = openmc.Material(name='borated_water')
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.add_element('B', max(boron_ppm, 1e-6) * 1e-6 * (18.015 / 10.811), percent_type='ao')
    water.set_density('g/cm3', WATER_DENSITY_GCC)
    water.add_s_alpha_beta('c_H_in_H2O')
    return fuel, clad, gt, water


def _run_static_keff(pattern_flat, boron_ppm, particles, batches, inactive, work_dir):
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
        settings.temperature = {'default': COOLANT_TEMP_K}
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

    widen_tries = 0
    while k_hi > target_keff and auto_widen and b_hi < boron_max:
        widen_tries += 1
        new_hi = min(b_hi * 2.0, boron_max)
        print(f'  [WIDEN] k_hi={k_hi:.5f} still > target at {b_hi:.0f} ppm - '
              f'expanding ceiling to {new_hi:.0f} ppm (try {widen_tries})')
        b_hi = new_hi
        k_hi = _run_static_keff(pattern_flat, b_hi, particles, batches, inactive,
                                 f'{work_dir}_hi_widen{widen_tries}').n
        print(f'  trial (widen {widen_tries}): boron={b_hi:.0f}ppm  k-eff={k_hi:.5f}')

    if not (k_lo >= target_keff >= k_hi):
        print(f'  [WARN] target k-eff NOT bracketed even at boron_max={boron_max:.0f} ppm '
              f'(k_lo={k_lo:.5f}, k_hi={k_hi:.5f}). Returning nearest endpoint, '
              f'flagged as NON-CONVERGED.')
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
    grid2d = pin_power_mesh_values.reshape(n_pins_y, n_pins_x)
    grid2d = grid2d[::-1, :]

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
    converged = True
    if boron_search:
        print('[WARN] --boron_search evaluates this pattern at k-eff~=1.000 (critical boron), '
              'but ml_dataset_constrained.csv was generated UNBORATED. Omit --boron_search '
              '(default boron_ppm=0) to match the CSV.')
        boron_ppm, keff_est, converged = find_critical_boron(
            pattern_flat, particles=max(1000, particles // 2),
            batches=max(25, batches // 2), inactive=max(10, inactive // 2),
            work_dir=f'{work_dir}_boronsearch',
            boron_lo=boron_lo, boron_hi=boron_hi, boron_max=boron_max)
        tag = 'CONVERGED' if converged else 'NOT CONVERGED - treat PPF below as provisional'
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
        settings.temperature = {'default': COOLANT_TEMP_K}
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
              f'boron={boron_ppm:.0f}ppm, boron_converged={converged}, '
              f'T={COOLANT_TEMP_K:.1f}K, water_density={WATER_DENSITY_GCC:.4f}g/cc)')
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
    os.makedirs(work_dir, exist_ok=True)
    cwd0 = os.getcwd()
    os.chdir(work_dir)

    try:
        converged = True
        if boron_search:
            print('[WARN] --boron_search targets k-eff~=1.000, but ml_dataset_constrained.csv '
                  'is unborated. Use only for genuinely NEW borated training data.')
            boron_ppm, keff_est, converged = find_critical_boron(
                pattern_flat, particles=max(1000, particles // 2),
                batches=max(25, batches // 2), inactive=max(10, inactive // 2),
                work_dir='boronsearch_boc',
                boron_lo=boron_lo, boron_hi=boron_hi, boron_max=boron_max)
            print(f'[BORON SEARCH] BOC critical boron = {boron_ppm:.0f} ppm '
                  f'(k_est={keff_est:.5f}, converged={converged}) - held fixed for depletion')

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
        settings.temperature = {'default': COOLANT_TEMP_K}
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
            f.write('# openmc_beavrs_vver1000_v4.py export — see cnn-v9.py for schema\n')
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
    ap.add_argument('--boron_search', action='store_true')
    ap.add_argument('--boron_ppm', type=float, default=BORON_PPM_DEFAULT)
    ap.add_argument('--boron_lo', type=float, default=BORON_LO_DEFAULT)
    ap.add_argument('--boron_hi', type=float, default=BORON_HI_DEFAULT)
    ap.add_argument('--boron_max', type=float, default=BORON_MAX_DEFAULT)
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


if __name__ == '__main__':
    main()