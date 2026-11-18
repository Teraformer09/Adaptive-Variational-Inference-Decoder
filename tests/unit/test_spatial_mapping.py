"""OBS-01 & OBS-02 Unit Tests: Canonical 21-Element <-> 7x3 Lattice Spatial Mapping."""

from __future__ import annotations

import numpy as np
import pytest

from training.observations.spatial_mapping import (
    CANONICAL_12_COUPLERS,
    CANONICAL_21_PARAM_NAMES,
    CANONICAL_21_TO_7X3_COORDS,
    CANONICAL_7X3_TO_21_INDEX,
    CANONICAL_QUBIT_GRID_3X3,
    COUPLER_PAIR_TO_VEC_INDEX,
    SPATIAL_CELL_TABLE,
    matrix7x3_to_vector21,
    vector21_to_matrix7x3,
)


def test_obs01_exact_roundtrip_vector21_and_matrix7x3() -> None:
    """OBS-01: Exact bijection and round-trip identity between 21-vector and 7x3 matrix."""
    rng = np.random.default_rng(20261118)
    vec = rng.uniform(0.0001, 0.48, size=21).astype(np.float64)

    mat = vector21_to_matrix7x3(vec)
    assert mat.shape == (7, 3)

    rec = matrix7x3_to_vector21(mat)
    assert rec.shape == (21,)
    assert np.array_equal(vec, rec)

    # Check basis indicator vectors e_0 .. e_20 map to 21 distinct (row, col) cells
    seen_coords = set()
    for i in range(21):
        ei = np.zeros(21, dtype=np.float64)
        ei[i] = 1.0
        mi = vector21_to_matrix7x3(ei)
        nz = np.argwhere(mi == 1.0)
        assert nz.shape == (1, 2)
        coord = (int(nz[0, 0]), int(nz[0, 1]))
        assert coord == CANONICAL_21_TO_7X3_COORDS[i]
        seen_coords.add(coord)
        assert CANONICAL_7X3_TO_21_INDEX[coord[0]][coord[1]] == i

    assert len(seen_coords) == 21


def test_obs02_physical_geometry_qubits_and_couplers() -> None:
    """OBS-02: All 9 qubits and 12 nearest-neighbour couplers match rotated d=3 lattice indexing."""
    assert len(SPATIAL_CELL_TABLE) == 21
    assert len(CANONICAL_21_PARAM_NAMES) == 21

    # Rows 0..2 must match the 3x3 data-qubit grid q_0..q_8
    for r in range(3):
        for c in range(3):
            expected_q = CANONICAL_QUBIT_GRID_3X3[r][c]
            vec_idx = CANONICAL_7X3_TO_21_INDEX[r][c]
            cell = SPATIAL_CELL_TABLE[vec_idx]
            assert vec_idx == expected_q
            assert cell.kind == "qubit"
            assert cell.qubits == (expected_q,)
            assert cell.name == f"p_{expected_q}"

    # Verify every coupler is an actual Manhattan-distance-1 edge on the 3x3 grid
    qubit_rc = {CANONICAL_QUBIT_GRID_3X3[r][c]: (r, c) for r in range(3) for c in range(3)}
    horiz_seen = []
    vert_seen = []
    for k, pair in enumerate(CANONICAL_12_COUPLERS):
        qa, qb = pair
        ra, ca = qubit_rc[qa]
        rb, cb = qubit_rc[qb]
        manhattan = abs(ra - rb) + abs(ca - cb)
        assert manhattan == 1, f"Non-adjacent coupler {pair} has Manhattan distance {manhattan}"
        vec_idx = 9 + k
        assert COUPLER_PAIR_TO_VEC_INDEX[pair] == vec_idx
        cell = SPATIAL_CELL_TABLE[vec_idx]
        if ra == rb:
            assert cell.kind == "horizontal_coupler"
            assert cell.row in (3, 4)
            horiz_seen.append(pair)
        else:
            assert cell.kind == "vertical_coupler"
            assert cell.row in (5, 6)
            vert_seen.append(pair)

    assert len(horiz_seen) == 6
    assert len(vert_seen) == 6

    with pytest.raises(ValueError):
        vector21_to_matrix7x3(np.zeros(20))
    with pytest.raises(ValueError):
        matrix7x3_to_vector21(np.zeros((3, 7)))
