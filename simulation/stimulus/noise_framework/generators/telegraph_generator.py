"""Random Telegraph Noise (RTN) Two-Level System Fluctuator (`telegraph_generator.py`).

Honors the assigned PD's unified contract (`evaluate_pd_contract`):
  - Up-switching rate and high-state occupancy are modulated by `env(t)` and `dom_mask`
  - Dominant qubits and pairs receive targeted envelope amplification.
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


class TelegraphNoiseGenerator(BaseNoiseGenerator):
    """Simulates discrete switching noise between low/high TLS defect states."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.rate_up = float(spec.parameters.get("rate_up_per_sec", 0.08))
        self.rate_down = float(spec.parameters.get("rate_down_per_sec", 0.25))
        self.high_multiplier = float(spec.parameters.get("high_multiplier", 1.85))
        self.low_multiplier = float(spec.parameters.get("low_multiplier", 0.65))
        self.pair_scale = float(spec.parameters.get("pair_scale", 0.35))
        self.envelope_floor = float(spec.parameters.get("envelope_floor", 0.45))
        self.envelope_gain = float(spec.parameters.get("envelope_gain", 1.10))
        self.dom_boost_scale = float(spec.parameters.get("dom_boost_scale", 0.50))
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
        t_local = float(timestamp) - 100.0 * float(window_id)
        level_vec, env_t, dom_mask, burst_amp = prof.evaluate_pd_contract(
            rng=self.rng, t_local_sec=t_local, size=21
        )

        u = self.rng.uniform(0.0, 1.0, size=9)
        for q in range(9):
            q_dom = 1.0 + 1.2 * dom_mask[q]
            r_up_q = self.rate_up * (0.4 + 2.6 * env_t) * q_dom
            r_dn_q = self.rate_down * (1.5 - 0.8 * env_t)
            p_01 = 1.0 - math.exp(-r_up_q * dt)
            p_10 = 1.0 - math.exp(-max(r_dn_q, 0.02) * dt)
            if self.tls_states[q] == 0 and u[q] < p_01:
                self.tls_states[q] = 1
            elif self.tls_states[q] == 1 and u[q] < p_10:
                self.tls_states[q] = 0

        env_mod = self.envelope_floor + self.envelope_gain * env_t
        mult_q = np.where(self.tls_states == 1, self.high_multiplier, self.low_multiplier)
        vec = np.zeros(21, dtype=np.float64)
        vec[:9] = level_vec[:9] * mult_q * env_mod + self.dom_boost_scale * burst_amp * env_t * dom_mask[:9]
        vec[9:] = (
            level_vec[9:] * self.pair_scale * env_mod
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
                "active_tls_count": int(np.sum(self.tls_states)),
                "tls_states": self.tls_states.tolist(),
                "envelope_value": float(env_t),
            },
        )

    def state_dict(self) -> Dict[str, Any]:
        st = super().state_dict()
        st["tls_states"] = self.tls_states.tolist()
        return st

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.tls_states = np.asarray(state["tls_states"], dtype=np.int32)
