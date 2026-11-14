"""Probability-Domain Physical Composition (`probability_composition.py`).

Implements:
  1. Independent activation union probability:
       p_combined = 1 - prod_j (1 - p_j)
  2. Independent Bernoulli modulo-2 XOR parity probability:
       p_xor = 0.5 * (1 - prod_j (1 - 2 p_j))
  3. Slot placement into the canonical 21-mechanism parameter vector.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def compose_independent_bernoulli(prob_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Compute p_combined = 1 - prod_j (1 - p_j) for independent Bernoulli sources."""
    if len(prob_vectors) == 0:
        raise ValueError("compose_independent_bernoulli requires at least one probability vector")
    first = np.asarray(prob_vectors[0], dtype=np.float64)
    if np.any((first < 0.0) | (first > 1.0)) or not np.all(np.isfinite(first)):
        raise ValueError("Probability values must be finite and within [0, 1]")
    survival = 1.0 - first.copy()
    for vec in prob_vectors[1:]:
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != first.shape:
            raise ValueError(f"Shape mismatch in probability composition: {arr.shape} != {first.shape}")
        if np.any((arr < 0.0) | (arr > 1.0)) or not np.all(np.isfinite(arr)):
            raise ValueError("Probability values must be finite and within [0, 1]")
        survival *= 1.0 - arr
    return np.clip(1.0 - survival, 0.0, 1.0)


def compose_xor_parity_bernoulli(prob_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Compute odd-parity XOR Bernoulli probability 0.5 * (1 - prod_j (1 - 2 p_j))."""
    if len(prob_vectors) == 0:
        raise ValueError("compose_xor_parity_bernoulli requires at least one probability vector")
    first = np.asarray(prob_vectors[0], dtype=np.float64)
    if np.any((first < 0.0) | (first > 1.0)) or not np.all(np.isfinite(first)):
        raise ValueError("Probability values must be finite and within [0, 1]")
    parity_prod = 1.0 - 2.0 * first.copy()
    for vec in prob_vectors[1:]:
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != first.shape:
            raise ValueError(f"Shape mismatch in XOR probability composition: {arr.shape} != {first.shape}")
        if np.any((arr < 0.0) | (arr > 1.0)) or not np.all(np.isfinite(arr)):
            raise ValueError("Probability values must be finite and within [0, 1]")
        parity_prod *= 1.0 - 2.0 * arr
    return np.clip(0.5 * (1.0 - parity_prod), 0.0, 1.0)


def place_into_21_mechanism_vector(
    single_probs: np.ndarray | None = None,
    pair_probs: np.ndarray | None = None,
) -> np.ndarray:
    """Place 9 single-qubit probabilities and 12 pair probabilities into their canonical 21 slots."""
    out = np.zeros(21, dtype=np.float64)
    if single_probs is not None:
        s = np.asarray(single_probs, dtype=np.float64)
        if s.shape != (9,):
            raise ValueError(f"single_probs must have shape (9,), got {s.shape}")
        out[:9] = s
    if pair_probs is not None:
        p = np.asarray(pair_probs, dtype=np.float64)
        if p.shape != (12,):
            raise ValueError(f"pair_probs must have shape (12,), got {p.shape}")
        out[9:21] = p
    return out
