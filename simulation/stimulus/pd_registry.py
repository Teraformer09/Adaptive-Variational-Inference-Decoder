"""Compatibility re-export of `PDRegistry` and `NoiseProfile`."""

from simulation.stimulus.noise_framework.config_schema import NoiseProfile
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

__all__ = ["NoiseProfile", "PDRegistry"]
