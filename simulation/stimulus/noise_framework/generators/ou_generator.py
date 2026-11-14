"""Ornstein-Uhlenbeck (OU) Continuous Stochastic Drift Generator (`ou_generator.py`).

Uses the exact discrete-time Ornstein-Uhlenbeck transition:
  x_{t + dt} = mu + (x_t - mu) * exp(-dt / tau) + sigma * sqrt(1 - exp(-2 * dt / tau)) * eps_t
preserving continuous state `x_t` across 100-second PD assignment boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class OUNoiseGenerator(BaseNoiseGenerator):
    """Generates temporally correlated OU stochastic drift over the 21 fault mechanisms."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "X"))
        self.tau_sec = max(float(spec.parameters.get("tau_sec", 8.0)), 0.1)
        self.volatility_scale = float(spec.parameters.get("volatility_scale", 0.30))
        self.student_df: Optional[float] = (
            float(spec.parameters["student_df"])
            if "student_df" in spec.parameters and spec.parameters["student_df"] is not None
            else None
        )
        self.x_state: Optional[np.ndarray] = None

    def reset(self) -> None:
        super().reset()
        self.x_state = None

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        dt = (
            0.1
            if self.last_timestamp is None
            else max(float(timestamp) - float(self.last_timestamp), 1e-4)
        )
        mu_vec = np.full(21, prof.mean, dtype=np.float64)
        mu_vec[9:] *= 0.40

        if self.x_state is None:
            self.x_state = prof.sample_values(self.rng, size=21)
            self.x_state[9:] *= 0.40

        decay = math.exp(-dt / self.tau_sec)
        ou_sigma = prof.std * self.volatility_scale * math.sqrt(max(1.0 - decay * decay, 1e-12))

        if self.student_df is not None and self.student_df > 2.0:
            raw_t = self.rng.standard_t(df=self.student_df, size=21)
            eps = raw_t / math.sqrt(self.student_df / (self.student_df - 2.0))
        else:
            eps = self.rng.standard_normal(size=21)

        self.x_state = mu_vec + (self.x_state - mu_vec) * decay + ou_sigma * eps
        vec = np.clip(self.x_state, prof.min_val * 0.2, min(0.45, prof.max_val * 1.4))

        self.samples_generated += 1
        self.last_timestamp = float(timestamp)

        return NoiseContribution(
            generator_id=self.generator_id,
            profile_id=prof.profile_id,
            window_id=window_id,
            sample_id=int(sample_id),
            timestamp=float(timestamp),
            output_domain=self.output_domain,
            values=vec.copy(),
            metadata={
                "basis": self.basis,
                "dt_sec": float(dt),
                "tau_sec": self.tau_sec,
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        st = super().state_dict()
        st["x_state"] = None if self.x_state is None else self.x_state.tolist()
        return st

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.x_state = (
            None
            if state.get("x_state") is None
            else np.asarray(state["x_state"], dtype=np.float64)
        )
