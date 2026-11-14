"""Markov-Modulated Regime-Switching Noise Generator (`markov_generator.py`)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class MarkovNoiseGenerator(BaseNoiseGenerator):
    """Generates state-dependent stochastic transitions P(z_{t+1} = b | z_t = a) = T_{ab}."""

    DEFAULT_TRANSITION = np.array(
        [
            [0.970, 0.025, 0.005],
            [0.040, 0.920, 0.040],
            [0.020, 0.100, 0.880],
        ],
        dtype=np.float64,
    )

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "X"))
        self.multipliers = np.asarray(
            spec.parameters.get("state_multipliers", [0.75, 1.0, 1.65]), dtype=np.float64
        )
        self.transition_matrix = np.asarray(
            spec.parameters.get("transition_matrix", self.DEFAULT_TRANSITION), dtype=np.float64
        )
        self.current_state: int = 0

    def reset(self) -> None:
        super().reset()
        self.current_state = 0

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        probs = self.transition_matrix[self.current_state]
        self.current_state = int(self.rng.choice(len(probs), p=probs))
        mult = float(self.multipliers[self.current_state])

        base_draw = prof.sample_values(self.rng, size=21)
        vec = np.zeros(21, dtype=np.float64)
        vec[:9] = base_draw[:9] * mult
        vec[9:] = base_draw[9:] * 0.35 * mult
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
                "markov_state": self.current_state,
                "state_multiplier": mult,
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        st = super().state_dict()
        st["current_state"] = self.current_state
        return st

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.current_state = int(state["current_state"])
