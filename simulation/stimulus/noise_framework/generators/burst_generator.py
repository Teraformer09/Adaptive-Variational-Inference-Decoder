"""Non-Stationary Correlated Burst Generator (`burst_generator.py`).

Implements Stage 4:
  p_k(t) = sigmoid(alpha_k + x_{k,t} + sum_j A_j * w_{j,k} * g_j(t - tau_j))
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
        tau_1 = float(prof.parameters.get("temporal_mu_sec", 50.0))
        width_1 = float(prof.parameters.get("temporal_sigma_sec", 5.0))
        tau_2 = min(95.0, tau_1 + 1.25 * width_1)
        width_2 = max(2.0, 0.75 * width_1)

        g1 = self._eval_kernel(t_local - tau_1, width_1)
        g2 = self._eval_kernel(t_local - tau_2, width_2)

        base_draw = prof.sample_values(self.rng, size=21) * 0.35
        alpha_k = _logit(base_draw)
        x_kt = self.rng.normal(0.0, 0.12, size=21)

        w1 = np.full(21, 0.15, dtype=np.float64)
        w2 = np.full(21, 0.10, dtype=np.float64)
        for q in prof.metadata.get("dominant_qubits", []):
            if 0 <= int(q) < 9:
                w1[int(q)] = 0.85
        for p_idx in prof.metadata.get("dominant_pairs", []):
            if 0 <= int(p_idx) < 12:
                w1[9 + int(p_idx)] = 1.0
                w2[9 + int(p_idx)] = 0.80

        amp_logit = float(prof.parameters.get("burst_amplitude", 0.03)) * 42.0
        total_logit = alpha_k + x_kt + amp_logit * (w1 * g1 + 0.65 * w2 * g2)
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
