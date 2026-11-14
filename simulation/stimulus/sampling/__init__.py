"""Compatibility re-exports of sampling modules."""

from simulation.stimulus.noise_framework.sampling import (
    SamplingClock,
    SynchronizedSampler,
    TrajectoryBuffer,
)

__all__ = ["SamplingClock", "SynchronizedSampler", "TrajectoryBuffer"]
