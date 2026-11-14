"""Compatibility re-export of `GeneratorRegistry` and `GeneratorSpec`."""

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry

__all__ = ["GeneratorRegistry", "GeneratorSpec"]
