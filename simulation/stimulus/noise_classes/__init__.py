"""Compatibility re-exports of noise generator classes."""

from simulation.stimulus.noise_framework.generators import (
    BaseNoiseGenerator,
    BurstNoiseGenerator,
    GaussianNoiseGenerator,
    MarkovNoiseGenerator,
    MeasurementNoiseGenerator,
    OUNoiseGenerator,
    SpatialCorrelatedGenerator,
    TelegraphNoiseGenerator,
)

__all__ = [
    "BaseNoiseGenerator",
    "BurstNoiseGenerator",
    "GaussianNoiseGenerator",
    "MarkovNoiseGenerator",
    "MeasurementNoiseGenerator",
    "OUNoiseGenerator",
    "SpatialCorrelatedGenerator",
    "TelegraphNoiseGenerator",
]
