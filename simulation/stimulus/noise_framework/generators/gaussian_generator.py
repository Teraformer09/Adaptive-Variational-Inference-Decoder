"""Gaussian Temporal & Profile-Sampling Noise Generator (`gaussian_generator.py`)."""

from __future__ import annotations

from typing import Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class GaussianNoiseGenerator(BaseNoiseGenerator):
    """Samples 21-mechanism fault probability vectors modulated by the assigned PD's contract."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.single_weight = float(spec.parameters.get("single_weight", 1.0))
        self.pair_weight = float(spec.parameters.get("pair_weight", 0.35))
        self.dom_single_boost = float(spec.parameters.get("dom_single_boost", 0.45))
        self.dom_pair_boost = float(spec.parameters.get("dom_pair_boost", 0.65))
        self.envelope_floor = float(spec.parameters.get("envelope_floor", 0.40))
        self.envelope_gain = float(spec.parameters.get("envelope_gain", 0.85))

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        t_local = float(timestamp) - 100.0 * float(window_id)
        level_vec, envelope, dom_mask, amp = prof.evaluate_pd_contract(
            rng=self.rng, t_local_sec=t_local, size=21
        )

        env_mod = self.envelope_floor + self.envelope_gain * envelope
        vec = np.zeros(21, dtype=np.float64)
        vec[:9] = level_vec[:9] * self.single_weight * env_mod
        vec[9:] = level_vec[9:] * self.pair_weight * env_mod

        vec[:9] += self.dom_single_boost * amp * envelope * dom_mask[:9]
        vec[9:] += self.dom_pair_boost * amp * envelope * dom_mask[9:]

        vec = np.clip(vec, prof.min_val * 0.20, min(0.45, prof.max_val * 1.5))
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
