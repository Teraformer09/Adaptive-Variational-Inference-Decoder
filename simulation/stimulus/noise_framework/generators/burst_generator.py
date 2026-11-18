"""Non-Stationary Correlated Burst Generator (`burst_generator.py`).

Implements Stage 4:
  p_k(t) = sigmoid(alpha_k + x_{k,t} + sum_j A_j * w_{j,k} * g_j(t - tau_j))
using the unified `NoiseProfile.evaluate_pd_contract()` interface.
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


class BurstNoiseGenerator(BaseNoiseGenerator):
    """Generates overlapping spatio-temporal burst contributions."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        super().__init__(spec=spec, seed=seed)
        self.basis = str(spec.parameters.get("basis", "Z"))
        self.kernel = str(spec.parameters.get("kernel", "gaussian_bell"))
        self.base_scale = float(spec.parameters.get("base_scale", 0.35))
        self.logit_jitter_std = float(spec.parameters.get("logit_jitter_std", 0.10))
        self.dom_qubit_weight = float(spec.parameters.get("dom_qubit_weight", 0.85))
        self.dom_pair_weight = float(spec.parameters.get("dom_pair_weight", 1.00))
        self.secondary_burst_weight = float(spec.parameters.get("secondary_burst_weight", 0.55))
        self.amp_logit_scale = float(spec.parameters.get("amp_logit_scale", 40.0))

    def _eval_kernel(self, dt_sec: float, width_sec: float) -> float:
        w = max(width_sec, 0.5)
        if self.kernel == "fast_attack_exp_decay":
            if dt_sec < 0.0:
                return 0.0
            return (1.0 - math.exp(-dt_sec / (0.25 * w))) * math.exp(-dt_sec / w)
        if self.kernel == "rectangular_window":
            return 1.0 if (0.0 <= dt_sec <= w) else 0.0
        return math.exp(-0.5 * (dt_sec / w) ** 2)

    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution:
        window_id = self._check_ready_and_timestamp(timestamp, sample_id)
        prof = self.current_profile
        assert prof is not None

        t_local = float(timestamp) - 100.0 * float(window_id)
        level_vec, g1_env, dom_mask, burst_amp = prof.evaluate_pd_contract(
            rng=self.rng, t_local_sec=t_local, size=21
        )
        # Primary burst kernel follows the PD's exact time envelope; secondary echo trails slightly
        g1 = g1_env if self.kernel == "gaussian_bell" else self._eval_kernel(t_local - 50.0, 6.0) * g1_env
        g2_env, _ = prof.time_envelope(max(0.0, t_local - 4.0))
        g2 = g2_env

        base_draw = level_vec * self.base_scale
        alpha_k = _logit(base_draw)
        x_kt = self.rng.normal(0.0, self.logit_jitter_std, size=21)

        w1 = np.full(21, 0.15, dtype=np.float64)
        w2 = np.full(21, 0.10, dtype=np.float64)
        w1[:9] = np.where(dom_mask[:9] > 0.5, self.dom_qubit_weight, 0.15)
        w1[9:] = np.where(dom_mask[9:] > 0.5, self.dom_pair_weight, 0.15)
        w2[9:] = np.where(dom_mask[9:] > 0.5, 0.80, 0.10)

        amp_logit = burst_amp * self.amp_logit_scale
        total_logit = alpha_k + x_kt + amp_logit * (w1 * g1 + self.secondary_burst_weight * w2 * g2)
        p_k = np.clip(_sigmoid(total_logit), 1e-5, 0.45)

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
                "kernel": self.kernel,
                "g1": float(g1),
                "g2": float(g2),
                "in_burst_window": bool(max(g1, g2) >= 0.35),
            },
        )
