"""AVID Seven-Stage Persistent-PD Multi-Generator Noise Framework."""

from simulation.stimulus.noise_framework.config_schema import (
    CANONICAL_12_PAIRS,
    CANONICAL_21_PARAM_NAMES,
    CompositeNoiseSample,
    GeneratorSpec,
    NoiseContribution,
    NoiseProfile,
    derive_deterministic_seed,
    sample_truncated_gaussian,
    theoretical_truncated_gaussian_moments,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

__all__ = [
    "CANONICAL_12_PAIRS",
    "CANONICAL_21_PARAM_NAMES",
    "CompositeNoiseSample",
    "GeneratorSpec",
    "NoiseContribution",
    "NoiseProfile",
    "PDRegistry",
    "derive_deterministic_seed",
    "sample_truncated_gaussian",
    "theoretical_truncated_gaussian_moments",
]
