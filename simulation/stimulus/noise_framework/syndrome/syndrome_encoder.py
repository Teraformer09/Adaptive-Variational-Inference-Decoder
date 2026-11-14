"""Canonical Rotated d=3 CSS Syndrome Encoder (`syndrome_encoder.py`).

Reuses `simulation/reference/code_model/css_code_builder.py` so there is only one
authoritative H_X and H_Z code geometry implementation across the entire repository:
  s_X = H_X e_Z mod 2
  s_Z = H_Z e_X mod 2
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code


class SyndromeEncoder:
    """Encodes physical X and Z data-qubit fault masks into ideal 4-bit CSS syndromes."""

    def __init__(self) -> None:
        self.code = build_rotated_d3_code()
        self.hx = np.asarray(self.code.hx, dtype=np.uint8)
        self.hz = np.asarray(self.code.hz, dtype=np.uint8)

    def encode_x_basis(self, z_errors: np.ndarray) -> np.ndarray:
        """Compute X-stabilizer syndromes s_X = H_X e_Z mod 2 (detects Z data faults)."""
        err = np.asarray(z_errors, dtype=np.uint8) & 1
        if err.ndim == 1:
            if err.shape != (9,):
                raise ValueError(f"z_errors must have shape (9,) or (N, 9), got {err.shape}")
            return (self.hx @ err) & 1
        if err.ndim == 2 and err.shape[1] == 9:
            return (err @ self.hx.T) & 1
        raise ValueError(f"z_errors must have shape (9,) or (N, 9), got {err.shape}")

    def encode_z_basis(self, x_errors: np.ndarray) -> np.ndarray:
        """Compute Z-stabilizer syndromes s_Z = H_Z e_X mod 2 (detects X data faults)."""
        err = np.asarray(x_errors, dtype=np.uint8) & 1
        if err.ndim == 1:
            if err.shape != (9,):
                raise ValueError(f"x_errors must have shape (9,) or (N, 9), got {err.shape}")
            return (self.hz @ err) & 1
        if err.ndim == 2 and err.shape[1] == 9:
            return (err @ self.hz.T) & 1
        raise ValueError(f"x_errors must have shape (9,) or (N, 9), got {err.shape}")

    def encode(
        self,
        x_errors: np.ndarray,
        z_errors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(ideal_syndromes_x, ideal_syndromes_z)`."""
        sx = self.encode_x_basis(z_errors)
        sz = self.encode_z_basis(x_errors)
        return sx, sz
