"""Gaussian Temporal & Profile-Sampling Noise Generator (`gaussian_generator.py`)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class GaussianNoiseGenerator(BaseNoiseGenerator):
    """Samples 21-mechanism fault probability vectors modulated by the assigned PD's Gaussian envelope."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.single_weight = float(spec.parameters.get("single_weight", 1.0))
        self.pair_weight = float(spec.parameters.get("pair_weight", 0.35))

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        # Local time within the 100-second window t_local in [0, 100)
        t_local = float(timestamp) - 100.0 * float(window_id)
        mu_t = float(prof.parameters.get("temporal_mu_sec", 50.0))
        sigma_t = max(float(prof.parameters.get("temporal_sigma_sec", 10.0)), 0.5)
        amp = float(prof.parameters.get("burst_amplitude", 0.015))
        envelope = math.exp(-0.5 * ((t_local - mu_t) / sigma_t) ** 2)

        raw = prof.sample_values(self.rng, size=21)
        vec = np.zeros(21, dtype=np.float64)
        vec[:9] = raw[:9] * self.single_weight
        vec[9:] = raw[9:] * self.pair_weight

        dom_q = prof.metadata.get("dominant_qubits", [])
        for q in dom_q:
            if 0 <= int(q) < 9:
                vec[int(q)] += 0.45 * amp * envelope

        dom_p = prof.metadata.get("dominant_pairs", [])
        for p_idx in dom_p:
            if 0 <= int(p_idx) < 12:
                vec[9 + int(p_idx)] += 0.65 * amp * envelope

        vec = np.clip(vec, prof.min_val * 0.25, min(0.45, prof.max_val * 1.5))
        self.samples_generated += 1
        self.last_timestamp = float(timestamp)

        return NoiseContribution(
            generator_id=self.generator_id,
            profile_id=prof.profile_id,
            window_id=window_id,
            sample_id=int(sample_id),
            timestamp=float(timestamp),
            output_domain=self.output_domain,
            values=vec,
            metadata={
                "basis": self.basis,
                "envelope_value": float(envelope),
                "t_local_sec": float(t_local),
            },
        )
