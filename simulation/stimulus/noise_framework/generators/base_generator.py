"""Base interface for all independent noise generators (`base_generator.py`).

Enforces:
  - Separation between `GeneratorSpec` (generator identity/type/seed) and `NoiseProfile` (assigned PD).
  - Independent RNG stream per generator instance.
  - Preservation of internal stochastic state across 100-second window PD reassignments unless `reset()` is called.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Optional
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    GeneratorSpec,
    NoiseContribution,
    NoiseProfile,
    derive_deterministic_seed,
)


class BaseNoiseGenerator(ABC):
    """Common abstract interface for all noise generators in the AVID framework."""

    def __init__(self, spec: GeneratorSpec, seed: Optional[int] = None) -> None:
        if not isinstance(spec, GeneratorSpec):
            raise TypeError(f"Expected GeneratorSpec, got {type(spec)}")
        self.spec = spec
        self.generator_id = spec.generator_id
        self.generator_type = spec.generator_type
        self.output_domain = spec.output_domain
        self.base_seed = int(spec.seed if seed is None else seed)
        self._derived_seed = derive_deterministic_seed(
            self.base_seed, "generator_stream", self.generator_id
        )
        self.rng = np.random.default_rng(self._derived_seed)

        self.current_profile: Optional[NoiseProfile] = None
        self.current_window_id: int = 0
        self.samples_generated: int = 0
        self.last_timestamp: Optional[float] = None

    def assign_profile(
        self,
        profile: NoiseProfile | str,
        window_id: Optional[int] = None,
        registry: Optional[Any] = None,
    ) -> None:
        """Assign a new PD profile for the current 100s window without resetting internal state."""
        if isinstance(profile, str):
            reg = registry if registry is not None else getattr(self, "pd_registry", None)
            if reg is None:
                from simulation.stimulus.noise_framework.pd_registry import PDRegistry
                from pathlib import Path
                reg = PDRegistry(
                    Path(__file__).resolve().parents[4]
                    / "configs"
                    / "noise_framework"
                    / "pd_profiles_16.yaml"
                )
            try:
                profile = reg.get(profile)
            except KeyError as exc:
                raise ValueError(f"Unknown PD profile_id '{profile}'") from exc
        if not isinstance(profile, NoiseProfile):
            raise TypeError(f"Expected NoiseProfile, got {type(profile)}")
        self._validate_profile_compatibility(profile)
        self.current_profile = profile
        if window_id is not None:
            if int(window_id) < 0:
                raise ValueError(f"window_id must be >= 0, got {window_id}")
            self.current_window_id = int(window_id)

    def _validate_profile_compatibility(self, profile: NoiseProfile) -> None:
        if profile.support.get("variable") != "fault_probability":
            raise ValueError(
                f"Generator '{self.generator_id}' rejects profile '{profile.profile_id}' "
                f"with incompatible variable '{profile.support.get('variable')}'"
            )

    def _check_ready_and_timestamp(self, timestamp: float, sample_id: int) -> int:
        if self.current_profile is None:
            raise RuntimeError(
                f"Generator '{self.generator_id}' cannot sample before assign_profile() is called"
            )
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"Invalid timestamp {timestamp}")
        if sample_id < 0:
            raise ValueError(f"Invalid sample_id {sample_id}")
        window_id = int(math.floor((timestamp + 1e-12) / 100.0))
        self.current_window_id = window_id
        return window_id

    @abstractmethod
    def sample(self, timestamp: float, sample_id: int) -> NoiseContribution | Any:
        """Produce one synchronized noise contribution at `(timestamp, sample_id)`."""

    def reset(self) -> None:
        """Reset internal stochastic state and re-seed the generator's RNG stream."""
        self.rng = np.random.default_rng(self._derived_seed)
        self.samples_generated = 0
        self.last_timestamp = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "generator_type": self.generator_type,
            "base_seed": self.base_seed,
            "current_profile_id": self.current_profile.profile_id if self.current_profile else None,
            "current_window_id": self.current_window_id,
            "samples_generated": self.samples_generated,
            "last_timestamp": self.last_timestamp,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("generator_id") != self.generator_id:
            raise ValueError(
                f"Generator ID mismatch in load_state_dict: {state.get('generator_id')} != {self.generator_id}"
            )
        self.current_window_id = int(state["current_window_id"])
        self.samples_generated = int(state["samples_generated"])
        self.last_timestamp = state["last_timestamp"]
        self.rng.bit_generator.state = state["rng_state"]
