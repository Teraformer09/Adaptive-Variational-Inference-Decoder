"""Synchronized Multi-Generator Sampler (`synchronized_sampler.py`).

Coordinates all active generators (including `measurement_noise_01`) at the exact same
10 Hz simulation timestamps `t_n = t_0 + 0.1 * n`, advancing the 100-second PD assignment
scheduler before sampling and verifying timestamp & sample_id equality across all contributions.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.config_schema import NoiseContribution
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer


class SynchronizedSampler:
    """Orchestrates simultaneous 10 Hz sampling across all active noise generators."""

    def __init__(
        self,
        generators: Sequence[Any],
        clock: SamplingClock,
        scheduler: AssignmentScheduler,
        composer: Optional[Any] = None,
    ) -> None:
        if not generators:
            raise ValueError("SynchronizedSampler requires at least one active generator")
        self.generators = list(generators)
        self.clock = clock
        self.scheduler = scheduler
        self.composer = composer

    def sample_all(self, timestamp: float, sample_id: int) -> List[Any]:
        """Advance assignment scheduler to `timestamp` and sample every active generator."""
        self.scheduler.advance(timestamp)
        expected_window = self.scheduler.current_window()
        raw_outputs: List[Any] = []

        for gen in self.generators:
            out = gen.sample(timestamp=timestamp, sample_id=sample_id)
            if int(out.sample_id) != int(sample_id):
                raise RuntimeError(
                    f"Sample ID mismatch from {gen.generator_id}: {out.sample_id} != {sample_id}"
                )
            if abs(float(out.timestamp) - float(timestamp)) > 1e-9:
                raise RuntimeError(
                    f"Timestamp mismatch from {gen.generator_id}: {out.timestamp} != {timestamp}"
                )
            if int(out.window_id) != expected_window:
                raise RuntimeError(
                    f"Window ID mismatch from {gen.generator_id}: {out.window_id} != {expected_window}"
                )
            raw_outputs.append(out)

        return raw_outputs

    def generate_trajectory(
        self,
        duration_seconds: float,
        buffer: Optional[TrajectoryBuffer] = None,
    ) -> List[Any]:
        """Generate `N = round(duration_seconds * 10)` synchronized samples over `duration_seconds`."""
        if duration_seconds <= 0.0 or not math.isfinite(duration_seconds):
            raise ValueError(f"duration_seconds must be positive and finite, got {duration_seconds}")
        num_samples = int(round(duration_seconds * self.clock.rate_hz))
        results: List[Any] = []

        for _ in range(num_samples):
            sid, ts, _, _ = self.clock.step()
            step_outputs = self.sample_all(timestamp=ts, sample_id=sid)
            if self.composer is not None:
                composed = self.composer.compose(
                    contributions=step_outputs, timestamp=ts, sample_id=sid
                )
                if buffer is not None:
                    norm_contribs = [
                        c.to_contribution() if hasattr(c, "to_contribution") else c
                        for c in step_outputs
                    ]
                    buffer.append_step(norm_contribs, composed)
                results.append(composed)
            else:
                results.append(step_outputs)

        return results
