"""Compatibility re-export of `simulation.stimulus.noise_framework.config_schema`."""

from simulation.stimulus.noise_framework.config_schema import (
    CANONICAL_12_PAIRS,
    CANONICAL_21_PARAM_NAMES,
    CompositeNoiseSample,
    GeneratorSpec,
    NoiseContribution,
    NoiseProfile,
    compute_canonical_config_hash,
    derive_deterministic_seed,
    load_yaml_validated,
    sample_truncated_gaussian,
    theoretical_truncated_gaussian_moments,
)

__all__ = [
    "CANONICAL_12_PAIRS",
    "CANONICAL_21_PARAM_NAMES",
    "CompositeNoiseSample",
    "GeneratorSpec",
    "NoiseContribution",
    "NoiseProfile",
    "compute_canonical_config_hash",
    "derive_deterministic_seed",
    "load_yaml_validated",
    "sample_truncated_gaussian",
    "theoretical_truncated_gaussian_moments",
]
