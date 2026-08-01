"""
=============================================================================
PATCH — Real BEAVRS BP rod placement (replaces the diagonal-sweep placeholder)
=============================================================================
Source: mit-crpg/BEAVRS/models/openmc/beavrs/assemblies.py, _add_bpra_layouts()
         (the official BEAVRS OpenMC model, fetched directly from GitHub)

This is NOT a guess — it's the real per-count BP rod position map used to
generate the actual BEAVRS benchmark geometry. It replaces the diagonal-sweep
`for offset in range(1, PINS_PER_SIDE): ...` block in your
build_assembly_universe() function.

WHAT CHANGED:
  - GT_LABEL_POS: the 24 non-center guide-tube-eligible (row,col) positions,
    labeled a-y exactly as in the official model. This is IDENTICAL to your
    existing GUIDE_TUBE_POS set (24 positions) + INSTRUMENT_TUBE_POS at
    (8,8) — confirms that part of your script was already correct.
  - BP_ROD_MAPS: dict of {n_bp_pins: frozenset of (row,col) positions that
    get u_bp instead of u_gt}, built directly from the real ba_specs.

NOTE ON ROTATIONAL VARIANTS (6 and 15 rod counts only):
  The real BEAVRS core uses 4 rotations of the 6-rod pattern (6N/6S/6E/6W)
  and 4 rotations of the 15-rod pattern (15NW/15NE/15SW/15SE) at different
  physical locations, for core-tilt reasons. Your model doesn't track
  per-assembly rotation (only a count per type), so this patch picks ONE
  canonical variant per count (6N, 15NW). This is still a large improvement
  over the diagonal sweep — same rod COUNT and a REAL BEAVRS rod PATTERN,
  just not location-specific rotation. If you later want rotation-awareness,
  BP_ROD_MAPS_ALL below keeps all 4 variants per count so you can wire in a
  per-position rotation choice.

Counts NOT in your ASSEMBLY_LIBRARY (4, 8) are included for completeness
since they appear in the real spec, but you don't need them.
=============================================================================
"""

# ── Label -> (row, col), 0-indexed within the 17x17 assembly ────────────────
# Decoded directly from Assemblies.pin_lattice_template in the official model.
GT_LABEL_POS = {
    'a': (2, 5),   'b': (2, 8),   'c': (2, 11),
    'd': (3, 3),   'e': (3, 13),
    'f': (5, 2),   'g': (5, 5),   'h': (5, 8),   'i': (5, 11),  'j': (5, 14),
    'k': (8, 2),   'l': (8, 5),   'n': (8, 11),  'o': (8, 14),
    'p': (11, 2),  'q': (11, 5),  'r': (11, 8),  's': (11, 11), 't': (11, 14),
    'u': (13, 3),  'v': (13, 13),
    'w': (14, 5),  'x': (14, 8),  'y': (14, 11),
}
# 'center' (8,8) is the instrument tube in every spec — never u_bp, handled
# separately by your existing INSTRUMENT_TUBE_POS logic.

# ── Real per-count BP label sets, straight from _add_bpra_layouts() ─────────
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

# Sanity check counts match their labels (run once at import time)
for _name, _labels in _BA_SPECS_LABELS.items():
    _expected = int(''.join(ch for ch in _name if ch.isdigit()))
    assert len(_labels) == _expected, f"{_name}: {len(_labels)} != {_expected}"

# ── Canonical single variant per BP count (what build_assembly_universe uses) ─
# Picked one rotation for 6 and 15 (see docstring). 12/16/20/4/8 have only
# one variant in the real spec (rotationally symmetric BP pattern).
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

# All 4 rotational variants per count, kept for future rotation-aware use.
BP_ROD_MAPS_ALL = {
    6:  {v: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS[v])
         for v in ('6N', '6S', '6E', '6W')},
    15: {v: frozenset(GT_LABEL_POS[l] for l in _BA_SPECS_LABELS[v])
         for v in ('15NW', '15NE', '15SW', '15SE')},
}


# =============================================================================
# DROP-IN REPLACEMENT for build_assembly_universe() in
# openmc_beavrs_vver1000_FIXED_2.py — replaces the diagonal-sweep BP block.
# =============================================================================
"""
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
"""