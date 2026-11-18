"""Separate X/Z 16-State Syndrome Histogram & Co-Activation Builder (`syndrome_histogram.py`)."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def _pack_4bit_syndromes(syn: np.ndarray) -> np.ndarray:
    """Convert either integer nibbles (N,) in 0..15 or bit matrices (N, 4) into uint8 nibbles."""
    arr = np.asarray(syn)
    if arr.ndim == 1:
        if np.any((arr < 0) | (arr > 15)):
            raise ValueError("Packed 4-bit syndrome values must lie in [0, 15]")
        return arr.astype(np.uint8)
    if arr.ndim == 2 and arr.shape[1] == 4:
        b = (arr != 0).astype(np.uint8)
        return (b[:, 0] | (b[:, 1] << 1) | (b[:, 2] << 2) | (b[:, 3] << 3)).astype(np.uint8)
    raise ValueError(f"Unsupported syndrome array shape {arr.shape}; expected (N,) or (N, 4)")


def compute_basis_syndrome_stats(
    syndromes: np.ndarray,
    validity_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute normalized 16-state histogram, check density, and adjacent co-activation rate.

    Normalizes strictly by the number of valid rounds (`valid_rounds > 0`). If all rounds in the
    window were dropped (`valid_rounds == 0`), returns the neutral trivial-syndrome prior `f[0]=1.0`
    and marks `all_dropped=True`.
    """
    packed = _pack_4bit_syndromes(syndromes)
    n_total = int(packed.shape[0])
    if n_total == 0:
        raise ValueError("Syndrome window must contain at least 1 round")

    if validity_mask is not None:
        vm = np.asarray(validity_mask)
        if vm.ndim == 2:
            valid_bool = np.all(vm != 0, axis=1)
        elif vm.ndim == 1 and vm.shape[0] == n_total:
            valid_bool = vm != 0
        else:
            raise ValueError(f"Incompatible validity_mask shape {vm.shape} for {n_total} rounds")
        valid_packed = packed[valid_bool]
    else:
        valid_packed = packed

    n_valid = int(valid_packed.shape[0])
    if n_valid == 0:
        hist = np.zeros(16, dtype=np.float64)
        hist[0] = 1.0
        return {
            "histogram": hist,
            "valid_rounds": 0,
            "total_rounds": n_total,
            "density": 0.0,
            "adjacent_coactivation": 0.0,
            "burst_subwindow_score": 0.0,
            "all_dropped": True,
        }

    counts = np.bincount(valid_packed, minlength=16).astype(np.float64)
    hist = counts / float(n_valid)

    b0 = (valid_packed >> 0) & 1
    b1 = (valid_packed >> 1) & 1
    b2 = (valid_packed >> 2) & 1
    b3 = (valid_packed >> 3) & 1

    weights = (b0 + b1 + b2 + b3).astype(np.float64)
    density = float(np.mean(weights) / 4.0)

    # Adjacent check pairs: (0,1), (1,2), (2,3)
    adj = ((b0 & b1) + (b1 & b2) + (b2 & b3)).astype(np.float64) / 3.0
    adj_rate = float(np.mean(adj))

    # Sub-window peak excitation score (rolling 16-round max non-trivial rate vs window mean)
    non_triv = (valid_packed != 0).astype(np.float64)
    mean_nt = float(np.mean(non_triv))
    if n_valid >= 16:
        kernel = np.ones(16, dtype=np.float64) / 16.0
        sub_rates = np.convolve(non_triv, kernel, mode="valid")
        peak_nt = float(np.max(sub_rates))
    else:
        peak_nt = mean_nt
    burst_sub = float(np.clip((peak_nt - mean_nt) * 2.5 + 0.65 * max(0.0, peak_nt - 0.25), 0.0, 1.0))

    return {
        "histogram": hist,
        "valid_rounds": n_valid,
        "total_rounds": n_total,
        "density": density,
        "adjacent_coactivation": adj_rate,
        "burst_subwindow_score": burst_sub,
        "all_dropped": False,
    }
