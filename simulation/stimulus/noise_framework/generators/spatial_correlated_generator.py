"""Spatially Correlated 21-Mechanism Fault Generator (`spatial_correlated_generator.py`).

Implements Stage 3:
  p_k(t | u_t) = sigmoid(alpha_k + beta_k * u_t)
over the 9 single-qubit and 12 nearest-neighbour pair fault mechanisms.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


def _logit(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_c / (1.0 - p_c))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SpatialCorrelatedGenerator(BaseNoiseGenerator):
    """Generates spatially correlated single/pair fault probabilities via shared latent factor u_t."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.latent_beta_scale = float(spec.parameters.get("latent_beta_scale", 1.15))

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        base_probs = prof.sample_values(self.rng, size=21)
        # Emphasize pair mechanisms (indices 9..20) and dominant pairs declared in the profile
        base_probs[:9] *= 0.45
        alpha_k = _logit(base_probs)

        beta_k = np.full(21, 0.35 * self.latent_beta_scale, dtype=np.float64)
        for q in prof.metadata.get("dominant_qubits", []):
            if 0 <= int(q) < 9:
                beta_k[int(q)] = 0.85 * self.latent_beta_scale
        for p_idx in prof.metadata.get("dominant_pairs", []):
            if 0 <= int(p_idx) < 12:
                beta_k[9 + int(p_idx)] = 1.35 * self.latent_beta_scale
                alpha_k[9 + int(p_idx)] += 0.65

        u_t = float(self.rng.standard_normal())
        p_k = _sigmoid(alpha_k + beta_k * u_t)
        p_k = np.clip(p_k, 1e-5, 0.45)

        self.samples_generated += 1
        self.last_timestamp = float(timestamp)

        return NoiseContribution(
            generator_id=self.generator_id,
            profile_id=prof.profile_id,
            window_id=window_id,
            sample_id=int(sample_id),
            timestamp=float(timestamp),
            output_domain=self.output_domain,
            values=p_k,
            metadata={
                "basis": self.basis,
                "latent_u_t": u_t,
            },
        )
