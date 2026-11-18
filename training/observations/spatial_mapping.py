"""Canonical 21-Mechanism <-> 7x3 Physical Lattice Spatial Mapping (`spatial_mapping.py`).

Uses explicit, immutable index tables linking the canonical 21-parameter vector ordering
`[p_0..p_8, p_01..p_58]` to the 7x3 rotated d=3 surface-code lattice matrix:
  - Rows 0..2 (3x3): Individual data-qubit fault probabilities q_0 .. q_8
  - Rows 3..4 (2x3): Horizontal nearest-neighbour coupler probabilities
                     Row 3: (0,1), (1,2), (3,4)
                     Row 4: (4,5), (6,7), (7,8)
  - Rows 5..6 (2x3): Vertical nearest-neighbour coupler probabilities
                     Row 5: (0,3), (3,6), (1,4)
                     Row 6: (4,7), (2,5), (5,8)

Never relies on blind array reshaping without verifying the explicit index table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

NUM_QUBITS_D3: int = 9
NUM_COUPLERS_D3: int = 12
NUM_PHYSICAL_PARAMS_D3: int = 21
SPATIAL_MATRIX_SHAPE: Tuple[int, int] = (7, 3)

CANONICAL_QUBIT_GRID_3X3: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
)

CANONICAL_12_COUPLERS: Tuple[Tuple[int, int], ...] = (
    # Horizontal couplers (pair_idx 0..5 -> vec_idx 9..14)
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    # Vertical couplers (pair_idx 6..11 -> vec_idx 15..20)
    (0, 3),
    (3, 6),
    (1, 4),
    (4, 7),
    (2, 5),
    (5, 8),
)

CANONICAL_21_PARAM_NAMES: Tuple[str, ...] = (
    "p_0", "p_1", "p_2",
    "p_3", "p_4", "p_5",
    "p_6", "p_7", "p_8",
    "p_01", "p_12", "p_34",
    "p_45", "p_67", "p_78",
    "p_03", "p_36", "p_14",
    "p_47", "p_25", "p_58",
)


@dataclass(frozen=True)
class SpatialCellSpec:
    """Immutable descriptor of a single entry in the 21-parameter / 7x3 lattice contract."""
    vec_index: int
    row: int
    col: int
    name: str
    kind: str  # "qubit" | "horizontal_coupler" | "vertical_coupler"
    qubits: Tuple[int, ...]


# Explicit, immutable 21-entry table defining the bijection between vec[0..20] and M[0..6, 0..2]
SPATIAL_CELL_TABLE: Tuple[SpatialCellSpec, ...] = (
    # Rows 0..2: 3x3 Data Qubits
    SpatialCellSpec(0,  0, 0, "p_0",  "qubit", (0,)),
    SpatialCellSpec(1,  0, 1, "p_1",  "qubit", (1,)),
    SpatialCellSpec(2,  0, 2, "p_2",  "qubit", (2,)),
    SpatialCellSpec(3,  1, 0, "p_3",  "qubit", (3,)),
    SpatialCellSpec(4,  1, 1, "p_4",  "qubit", (4,)),
    SpatialCellSpec(5,  1, 2, "p_5",  "qubit", (5,)),
    SpatialCellSpec(6,  2, 0, "p_6",  "qubit", (6,)),
    SpatialCellSpec(7,  2, 1, "p_7",  "qubit", (7,)),
    SpatialCellSpec(8,  2, 2, "p_8",  "qubit", (8,)),
    # Rows 3..4: 6 Horizontal Couplers
    SpatialCellSpec(9,  3, 0, "p_01", "horizontal_coupler", (0, 1)),
    SpatialCellSpec(10, 3, 1, "p_12", "horizontal_coupler", (1, 2)),
    SpatialCellSpec(11, 3, 2, "p_34", "horizontal_coupler", (3, 4)),
    SpatialCellSpec(12, 4, 0, "p_45", "horizontal_coupler", (4, 5)),
    SpatialCellSpec(13, 4, 1, "p_67", "horizontal_coupler", (6, 7)),
    SpatialCellSpec(14, 4, 2, "p_78", "horizontal_coupler", (7, 8)),
    # Rows 5..6: 6 Vertical Couplers
    SpatialCellSpec(15, 5, 0, "p_03", "vertical_coupler", (0, 3)),
    SpatialCellSpec(16, 5, 1, "p_36", "vertical_coupler", (3, 6)),
    SpatialCellSpec(17, 5, 2, "p_14", "vertical_coupler", (1, 4)),
    SpatialCellSpec(18, 6, 0, "p_47", "vertical_coupler", (4, 7)),
    SpatialCellSpec(19, 6, 1, "p_25", "vertical_coupler", (2, 5)),
    SpatialCellSpec(20, 6, 2, "p_58", "vertical_coupler", (5, 8)),
)

# Immutable forward and inverse index lookup tables
CANONICAL_21_TO_7X3_COORDS: Tuple[Tuple[int, int], ...] = tuple(
    (cell.row, cell.col) for cell in SPATIAL_CELL_TABLE
)

_INV_GRID = [[0 for _ in range(3)] for _ in range(7)]
for _c in SPATIAL_CELL_TABLE:
    _INV_GRID[_c.row][_c.col] = _c.vec_index

CANONICAL_7X3_TO_21_INDEX: Tuple[Tuple[int, int, int], ...] = tuple(
    (int(r[0]), int(r[1]), int(r[2])) for r in _INV_GRID
)

COUPLER_PAIR_TO_VEC_INDEX: Dict[Tuple[int, int], int] = {
    cell.qubits: cell.vec_index
    for cell in SPATIAL_CELL_TABLE
    if len(cell.qubits) == 2
}


def vector21_to_matrix7x3(vec: np.ndarray) -> np.ndarray:
    """Map a 21-element parameter/uncertainty vector onto the canonical 7x3 lattice matrix."""
    arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (NUM_PHYSICAL_PARAMS_D3,):
        raise ValueError(f"Expected 1-D vector of shape (21,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector21_to_matrix7x3 received non-finite values")

    mat = np.empty(SPATIAL_MATRIX_SHAPE, dtype=np.float64)
    for idx, (r, c) in enumerate(CANONICAL_21_TO_7X3_COORDS):
        mat[r, c] = arr[idx]
    return mat


def matrix7x3_to_vector21(mat: np.ndarray) -> np.ndarray:
    """Map a 7x3 lattice matrix back to the canonical 21-element vector ordering."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != SPATIAL_MATRIX_SHAPE:
        raise ValueError(f"Expected 2-D matrix of shape (7, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix7x3_to_vector21 received non-finite values")

    vec = np.empty(NUM_PHYSICAL_PARAMS_D3, dtype=np.float64)
    for r in range(7):
        for c in range(3):
            vec[CANONICAL_7X3_TO_21_INDEX[r][c]] = arr[r, c]
    return vec
