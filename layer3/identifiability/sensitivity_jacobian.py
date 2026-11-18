"""Exact Analytical 16x21 Syndrome Sensitivity Jacobian (`sensitivity_jacobian.py`)."""

from __future__ import annotations

import numpy as np

from layer3.forward_model.syndrome_likelihood import (
    HADAMARD_16,
    PARITY_MASK_X_16X21,
    PARITY_MASK_Z_16X21,
)


def compute_syndrome_jacobian(
    theta_21: np.ndarray,
    basis_checks: str = "X",
    readout_flip_prob: float = 0.0,
) -> np.ndarray:
    """Compute exact 16x21 analytical Jacobian J_{s, m}(theta) = dP(s | theta) / dtheta_m."""
    th = np.clip(np.asarray(theta_21, dtype=np.float64), 0.0, 0.495)
    if th.shape != (21,):
        raise ValueError(f"Expected theta_21 shape (21,), got {th.shape}")

    mask = PARITY_MASK_X_16X21 if basis_checks.upper() == "X" else PARITY_MASK_Z_16X21
    factors = 1.0 - 2.0 * mask * th[None, :]  # (16, 21), strictly >= 0.01 > 0
    spec_full = np.prod(factors, axis=1)      # (16,)

    if readout_flip_prob > 0.0:
        q = float(np.clip(readout_flip_prob, 0.0, 0.49))
        w = np.arange(16, dtype=np.uint8)
        wt = ((w >> 0) & 1) + ((w >> 1) & 1) + ((w >> 2) & 1) + ((w >> 3) & 1)
        spec_full = spec_full * np.power(1.0 - 2.0 * q, wt)

    # Leave-one-out spectral product spec_full[w] / factors[w, m]
    spec_without_m = spec_full[:, None] / factors  # (16, 21)
    dspec_dtheta = -2.0 * mask * spec_without_m    # (16, 21)
    jacobian = (HADAMARD_16 @ dspec_dtheta) / 16.0
    return jacobian
