"""Physical Binary Pauli Error Mask Composition (`pauli_composition.py`).

Enforces modulo-2 bitwise XOR composition across independent generator masks:
  e_Z = XOR_{j=1..G} e_{Z,j}
  e_X = XOR_{j=1..G} e_{X,j}
so two identical Pauli flips on the same qubit at the same instant cancel (Z XOR Z = I).
"""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


def compose_pauli_masks_xor(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Combine binary Pauli error masks via bitwise XOR modulo 2."""
    if len(masks) == 0:
        raise ValueError("compose_pauli_masks_xor requires at least one mask")
    first = np.asarray(masks[0], dtype=np.uint8) & 1
    acc = first.copy()
    for m in masks[1:]:
        arr = np.asarray(m, dtype=np.uint8)
        if arr.shape != acc.shape:
            raise ValueError(f"Mask shape mismatch in XOR composition: {arr.shape} != {acc.shape}")
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("Pauli mask contains non-binary values outside {0, 1}")
        acc ^= arr & 1
    return acc


def compose_mixed_pauli(
    x_masks: Sequence[np.ndarray],
    z_masks: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose X and Z Pauli masks ignoring global phase (Y = i X Z -> (x=1, z=1))."""
    comp_x = compose_pauli_masks_xor(x_masks)
    comp_z = compose_pauli_masks_xor(z_masks)
    return comp_x, comp_z
