"""Authoritative Physical CSS Fault Realizer (`fault_realizer.py`).

Implements Stage 5.1:
  - Single-qubit fault mechanisms k=0..8: E_i ~ Bernoulli(p_i)
  - Nearest-neighbour pair fault mechanisms k=9..20: E_{ij} ~ Bernoulli(p_{ij})
    affecting both qubits (i, j) in `CANONICAL_12_PAIRS` via modulo-2 XOR.
  - Never samples the same physical mechanism twice when multiple generators contribute
    to its composite probability parameter.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.config_schema import (
    CANONICAL_12_PAIRS,
    CompositeNoiseSample,
)

_D3_CODE = build_rotated_d3_code()
_LX_VEC = np.asarray(_D3_CODE.logical_X[0], dtype=np.uint8)
_LZ_VEC = np.asarray(_D3_CODE.logical_Z[0], dtype=np.uint8)


class FaultRealizer:
    """Turns composite 21-mechanism noise parameters into physical X/Z data-qubit fault masks."""

    def __init__(self) -> None:
        self.pairs = CANONICAL_12_PAIRS

    def _realize_single_basis(
        self,
        theta_21: np.ndarray,
        rng: np.random.Generator,
        n_rounds: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta_21, dtype=np.float64)
        if theta.shape != (21,):
            raise ValueError(f"Expected 21-parameter vector, got shape {theta.shape}")
        if np.any((theta < 0.0) | (theta > 1.0)) or not np.all(np.isfinite(theta)):
            raise ValueError("Fault probabilities must be finite and within [0, 1]")

        events_21 = (rng.uniform(0.0, 1.0, size=(n_rounds, 21)) < theta[None, :]).astype(np.uint8)
        qubit_mask = events_21[:, :9].copy()
        for pair_idx, (q_i, q_j) in enumerate(self.pairs):
            pair_ev = events_21[:, 9 + pair_idx]
            qubit_mask[:, q_i] ^= pair_ev
            qubit_mask[:, q_j] ^= pair_ev

        if n_rounds == 1:
            return qubit_mask[0], events_21[0]
        return qubit_mask, events_21

    def realize_z_faults(
        self,
        parameters: np.ndarray | Dict[str, Any],
        rng: np.random.Generator,
        n_rounds: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta_z = parameters["theta_z"] if isinstance(parameters, dict) else parameters
        return self._realize_single_basis(theta_z, rng=rng, n_rounds=n_rounds)

    def realize_x_faults(
        self,
        parameters: np.ndarray | Dict[str, Any],
        rng: np.random.Generator,
        n_rounds: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta_x = parameters["theta_x"] if isinstance(parameters, dict) else parameters
        return self._realize_single_basis(theta_x, rng=rng, n_rounds=n_rounds)

    def realize(
        self,
        composite_sample: CompositeNoiseSample,
        rng: np.random.Generator,
        n_rounds: int = 1,
    ) -> Dict[str, Any]:
        """Realize physical X and Z data-qubit faults for a `CompositeNoiseSample`."""
        if composite_sample.composite_faults is not None and (
            "err_z" in composite_sample.composite_faults
            and "err_x" in composite_sample.composite_faults
        ):
            err_z = np.asarray(composite_sample.composite_faults["err_z"], dtype=np.uint8)
            err_x = np.asarray(composite_sample.composite_faults["err_x"], dtype=np.uint8)
            events_z = np.zeros(21, dtype=np.uint8)
            events_x = np.zeros(21, dtype=np.uint8)
            events_z[:9] = err_z[:9]
            events_x[:9] = err_x[:9]
        else:
            params = composite_sample.composite_parameters
            err_z, events_z = self.realize_z_faults(params["theta_z"], rng=rng, n_rounds=n_rounds)
            err_x, events_x = self.realize_x_faults(params["theta_x"], rng=rng, n_rounds=n_rounds)

        if err_z.ndim == 1:
            log_z = int(np.dot(_LX_VEC, err_z) & 1)
            log_x = int(np.dot(_LZ_VEC, err_x) & 1)
        else:
            log_z = (err_z @ _LX_VEC) & 1
            log_x = (err_x @ _LZ_VEC) & 1

        realized = {
            "err_z": err_z,
            "err_x": err_x,
            "events_z_21": events_z,
            "events_x_21": events_x,
            "logical_z_flip": log_z,
            "logical_x_flip": log_x,
        }
        composite_sample.composite_faults = realized
        return realized
