"""Markov-Modulated Regime-Switching Noise Generator (`markov_generator.py`).

Honors the assigned PD's unified contract (`evaluate_pd_contract`):
  - Regime transition probabilities are biased toward active states during `env(t)` peaks
  - Output mechanism vector incorporates `env(t)` and `dom_mask`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.generators.base_generator import BaseNoiseGenerator


class MarkovNoiseGenerator(BaseNoiseGenerator):
    """Generates state-dependent stochastic transitions P(z_{t+1} = b | z_t = a) = T_{ab}(t)."""

    DEFAULT_TRANSITION = np.array(
        [
            [0.950, 0.040, 0.010],
            [0.060, 0.880, 0.060],
            [0.040, 0.140, 0.820],
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
        self.pair_scale = float(spec.parameters.get("pair_scale", 0.35))
        self.envelope_floor = float(spec.parameters.get("envelope_floor", 0.45))
        self.envelope_gain = float(spec.parameters.get("envelope_gain", 1.10))
        self.dom_boost_scale = float(spec.parameters.get("dom_boost_scale", 0.55))
        self.current_state: int = 0

    def reset(self) -> None:
        super().reset()
        self.current_state = 0

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        t_local = float(timestamp) - 100.0 * float(window_id)
        level_vec, env_t, dom_mask, burst_amp = prof.evaluate_pd_contract(
            rng=self.rng, t_local_sec=t_local, size=21
        )

        # Modulate regime transition weights by env(t) so elevated regimes track the PD time envelope
        base_row = self.transition_matrix[self.current_state].copy()
        n_states = len(base_row)
        if n_states >= 3:
            env_bias = np.array([1.0 - 0.55 * env_t, 1.0 + 0.25 * env_t, 1.0 + 1.60 * env_t])
            probs = base_row * env_bias[:n_states]
            probs = probs / np.sum(probs)
        else:
            probs = base_row / np.sum(base_row)

        self.current_state = int(self.rng.choice(len(probs), p=probs))
        mult = float(self.multipliers[self.current_state])
        env_mod = self.envelope_floor + self.envelope_gain * env_t

        vec = np.zeros(21, dtype=np.float64)
        vec[:9] = (
            level_vec[:9] * mult * env_mod
            + self.dom_boost_scale * burst_amp * env_t * dom_mask[:9]
        )
        vec[9:] = (
            level_vec[9:] * self.pair_scale * mult * env_mod
            + self.dom_boost_scale * burst_amp * env_t * dom_mask[9:]
        )
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
                "envelope_value": float(env_t),
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        st = super().state_dict()
        st["current_state"] = self.current_state
        return st

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.current_state = int(state["current_state"])
