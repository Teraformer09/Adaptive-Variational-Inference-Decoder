"""Exact 16-State Walsh-Hadamard Syndrome Likelihood (`syndrome_likelihood.py`).

Computes the exact forward probability distribution P(s | theta) over s in {0..15}
for the 21-mechanism rotated d=3 surface code (9 single-qubit + 12 coupler rates).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

CROSSTALK_PAIRS_12: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
    (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),
)

# 4-bit X-stabilizer signatures detecting Z-basis mechanisms (c0={0,1}, c1={1,2,4,5}, c2={3,4,6,7}, c3={7,8})
EVENT_SIGS_X_21 = np.array(
    [
        0x1, 0x3, 0x2, 0x4, 0x6, 0x2, 0x4, 0xC, 0x8,
        0x2, 0x1, 0x2, 0x4, 0x8, 0x4, 0x5, 0x0, 0x5, 0xA, 0x0, 0xA,
    ],
    dtype=np.uint8,
)

# 4-bit Z-stabilizer signatures detecting X-basis mechanisms (z0={0,1,3,4}, z1={2,5}, z2={3,6}, z3={4,5,7,8})
EVENT_SIGS_Z_21 = np.array(
    [
        0x1, 0x1, 0x2, 0x5, 0x9, 0xA, 0x4, 0x8, 0x8,
        0x0, 0x3, 0x4, 0x3, 0xC, 0x0, 0x4, 0x1, 0x8, 0x1, 0x8, 0x2,
    ],
    dtype=np.uint8,
)


def _build_hadamard_16() -> np.ndarray:
    """Construct unnormalized 16x16 Sylvester-Walsh-Hadamard matrix H_{w, s} = (-1)^{<w, s>}."""
    idx = np.arange(16, dtype=np.uint8)
    xor_prod = idx[:, None] & idx[None, :]
    popcnt = (
        ((xor_prod >> 0) & 1)
        + ((xor_prod >> 1) & 1)
        + ((xor_prod >> 2) & 1)
        + ((xor_prod >> 3) & 1)
    )
    return np.where((popcnt & 1) == 1, -1.0, 1.0).astype(np.float64)


HADAMARD_16 = _build_hadamard_16()


def _build_parity_mask(sigs_21: np.ndarray) -> np.ndarray:
    """Return 16x21 binary matrix M[w, m] = 1 iff <w, sigs_21[m]> == 1 (mod 2)."""
    w = np.arange(16, dtype=np.uint8)[:, None]
    g = sigs_21[None, :]
    prod = w & g
    pop = ((prod >> 0) & 1) + ((prod >> 1) & 1) + ((prod >> 2) & 1) + ((prod >> 3) & 1)
    return (pop & 1).astype(np.float64)


PARITY_MASK_X_16X21 = _build_parity_mask(EVENT_SIGS_X_21)
PARITY_MASK_Z_16X21 = _build_parity_mask(EVENT_SIGS_Z_21)


def exact_syndrome_probabilities(
    theta_21: np.ndarray,
    basis_checks: str = "X",
    readout_flip_prob: float = 0.0,
) -> np.ndarray:
    """Compute exact 16-state syndrome distribution P(s | theta_21) via Walsh-Hadamard transform.

    - `basis_checks="X"`: X-stabilizer distribution induced by 21 Z-fault rates `theta_21`.
    - `basis_checks="Z"`: Z-stabilizer distribution induced by 21 X-fault rates `theta_21`.
    """
    th = np.clip(np.asarray(theta_21, dtype=np.float64), 0.0, 0.495)
    if th.shape != (21,):
        raise ValueError(f"Expected theta_21 shape (21,), got {th.shape}")

    mask = PARITY_MASK_X_16X21 if basis_checks.upper() == "X" else PARITY_MASK_Z_16X21
    factors = 1.0 - 2.0 * mask * th[None, :]  # (16, 21)
    spec = np.prod(factors, axis=1)           # (16,)

    if readout_flip_prob > 0.0:
        q = float(np.clip(readout_flip_prob, 0.0, 0.49))
        w = np.arange(16, dtype=np.uint8)
        wt = ((w >> 0) & 1) + ((w >> 1) & 1) + ((w >> 2) & 1) + ((w >> 3) & 1)
        spec = spec * np.power(1.0 - 2.0 * q, wt)

    probs = (HADAMARD_16 @ spec) / 16.0
    probs = np.clip(probs, 1.0e-12, 1.0)
    return probs / float(np.sum(probs))


def syndrome_negative_log_likelihood(
    hist_16: np.ndarray,
    theta_21: np.ndarray,
    basis_checks: str = "X",
    readout_flip_prob: float = 0.0,
) -> float:
    """Compute cross-entropy negative log-likelihood -sum_s f_hat(s) * log P(s | theta)."""
    f = np.asarray(hist_16, dtype=np.float64)
    p = exact_syndrome_probabilities(
        theta_21=theta_21, basis_checks=basis_checks, readout_flip_prob=readout_flip_prob
    )
    return float(-np.sum(f * np.log(np.clip(p, 1.0e-12, 1.0))))
