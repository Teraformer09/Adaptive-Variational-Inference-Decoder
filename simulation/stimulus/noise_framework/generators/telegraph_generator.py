"""Random Telegraph Noise (RTN) Two-Level System Fluctuator (`telegraph_generator.py`)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class TelegraphNoiseGenerator(BaseNoiseGenerator):
    """Simulates discrete switching noise between low/high TLS defect states."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.rate_up = float(spec.parameters.get("rate_up_per_sec", 0.05))
        self.rate_down = float(spec.parameters.get("rate_down_per_sec", 0.18))
        self.high_multiplier = float(spec.parameters.get("high_multiplier", 2.2))
        self.tls_states = np.zeros(9, dtype=np.int32)

    def reset(self) -> None:
        super().reset()
        self.tls_states = np.zeros(9, dtype=np.int32)

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        dt = (
            0.1
            if self.last_timestamp is None
            else max(float(timestamp) - float(self.last_timestamp), 1e-4)
        )
        p_01 = 1.0 - math.exp(-self.rate_up * dt)
        p_10 = 1.0 - math.exp(-self.rate_down * dt)

        u = self.rng.uniform(0.0, 1.0, size=9)
        for q in range(9):
            if self.tls_states[q] == 0 and u[q] < p_01:
                self.tls_states[q] = 1
            elif self.tls_states[q] == 1 and u[q] < p_10:
                self.tls_states[q] = 0

        base_draw = prof.sample_values(self.rng, size=21)
        vec = np.zeros(21, dtype=np.float64)
        mult_q = np.where(self.tls_states == 1, self.high_multiplier, 0.65)
        vec[:9] = base_draw[:9] * mult_q
        vec[9:] = base_draw[9:] * 0.25
        vec = np.clip(vec, 1e-5, 0.45)

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
                "active_tls_count": int(np.sum(self.tls_states)),
                "tls_states": self.tls_states.tolist(),
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        st = super().state_dict()
        st["tls_states"] = self.tls_states.tolist()
        return st

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.tls_states = np.asarray(state["tls_states"], dtype=np.int32)
