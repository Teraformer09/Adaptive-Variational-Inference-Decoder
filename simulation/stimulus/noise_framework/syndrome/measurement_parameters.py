"""Typed, provenance-aware Measurement Noise Parameter Contract (`measurement_parameters.py`).

Implements Stage 5 Step 2 (`inst_0` .. `inst_10`):
  - `q_base_x`: (4,) float64 array in [0, 1]
  - `q_base_z`: (4,) float64 array in [0, 1]
  - `temporal_sticky_prob`: float in [0, 1]
  - `detector_dropout_prob`: float in [0, 1]
  - `observation_delay_samples`: int >= 0
  - `generator_id`, `profile_id`, `sample_id`, `timestamp`, `window_id`
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict
import numpy as np

from simulation.stimulus.noise_framework.config_schema import NoiseContribution


@dataclass
class MeasurementNoiseParameters:
    """Validated measurement-noise parameter realization at a single synchronized sample."""

    q_base_x: np.ndarray
    q_base_z: np.ndarray
    temporal_sticky_prob: float
    detector_dropout_prob: float
    observation_delay_samples: int
    generator_id: str
    profile_id: str
    sample_id: int
    timestamp: float
    window_id: int
    output_domain: str = field(default="measurement_fault_parameters", init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q_base_x = np.asarray(self.q_base_x, dtype=np.float64)
        self.q_base_z = np.asarray(self.q_base_z, dtype=np.float64)

        if self.q_base_x.shape != (4,):
            raise ValueError(f"q_base_x must have shape (4,), got {self.q_base_x.shape}")
        if self.q_base_z.shape != (4,):
            raise ValueError(f"q_base_z must have shape (4,), got {self.q_base_z.shape}")

        if not (np.all(np.isfinite(self.q_base_x)) and np.all(np.isfinite(self.q_base_z))):
            raise ValueError("q_base_x and q_base_z must contain finite numbers")
        if np.any((self.q_base_x < 0.0) | (self.q_base_x > 1.0)):
            raise ValueError(f"q_base_x probabilities out of [0, 1]: {self.q_base_x}")
        if np.any((self.q_base_z < 0.0) | (self.q_base_z > 1.0)):
            raise ValueError(f"q_base_z probabilities out of [0, 1]: {self.q_base_z}")

        if isinstance(self.temporal_sticky_prob, (bool, np.bool_)):
            raise ValueError("temporal_sticky_prob must be a float in [0, 1], not bool")
        self.temporal_sticky_prob = float(self.temporal_sticky_prob)
        if not math.isfinite(self.temporal_sticky_prob) or not (0.0 <= self.temporal_sticky_prob <= 1.0):
            raise ValueError(
                f"temporal_sticky_prob must be in [0, 1], got {self.temporal_sticky_prob}"
            )

        if isinstance(self.detector_dropout_prob, (bool, np.bool_)):
            raise ValueError("detector_dropout_prob must be a float in [0, 1], not bool")
        self.detector_dropout_prob = float(self.detector_dropout_prob)
        if not math.isfinite(self.detector_dropout_prob) or not (0.0 <= self.detector_dropout_prob <= 1.0):
            raise ValueError(
                f"detector_dropout_prob must be in [0, 1], got {self.detector_dropout_prob}"
            )

        if isinstance(self.observation_delay_samples, (bool, np.bool_)) or not isinstance(
            self.observation_delay_samples, (int, np.integer)
        ):
            raise ValueError(
                f"observation_delay_samples must be a non-negative integer, got {type(self.observation_delay_samples)}"
            )
        self.observation_delay_samples = int(self.observation_delay_samples)
        if self.observation_delay_samples < 0:
            raise ValueError(
                f"observation_delay_samples must be >= 0, got {self.observation_delay_samples}"
            )

        if not self.generator_id or not isinstance(self.generator_id, str):
            raise ValueError("generator_id must be a non-empty string")
        if not self.profile_id or not isinstance(self.profile_id, str):
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(self.sample_id, (int, np.integer)) or int(self.sample_id) < 0:
            raise ValueError(f"sample_id must be a non-negative integer, got {self.sample_id}")
        self.sample_id = int(self.sample_id)
        if not isinstance(self.window_id, (int, np.integer)) or int(self.window_id) < 0:
            raise ValueError(f"window_id must be a non-negative integer, got {self.window_id}")
        self.window_id = int(self.window_id)
        self.timestamp = float(self.timestamp)
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(f"timestamp must be finite and >= 0, got {self.timestamp}")

    @property
    def values(self) -> np.ndarray:
        return self.to_vector()

    def to_vector(self) -> np.ndarray:
        """Pack into an 11-element float64 vector `[q_x(4), q_z(4), sticky, dropout, delay]`."""
        return np.concatenate(
            [
                self.q_base_x,
                self.q_base_z,
                np.array(
                    [
                        self.temporal_sticky_prob,
                        self.detector_dropout_prob,
                        float(self.observation_delay_samples),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

    def to_contribution(self) -> NoiseContribution:
        """Convert to a `NoiseContribution` retaining the full `MeasurementNoiseParameters` object."""
        meta = dict(self.metadata)
        meta["measurement_parameters"] = self
        return NoiseContribution(
            generator_id=self.generator_id,
            profile_id=self.profile_id,
            window_id=self.window_id,
            sample_id=self.sample_id,
            timestamp=self.timestamp,
            output_domain=self.output_domain,
            values=self.to_vector(),
            metadata=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q_base_x": self.q_base_x.tolist(),
            "q_base_z": self.q_base_z.tolist(),
            "temporal_sticky_prob": self.temporal_sticky_prob,
            "detector_dropout_prob": self.detector_dropout_prob,
            "observation_delay_samples": self.observation_delay_samples,
            "generator_id": self.generator_id,
            "profile_id": self.profile_id,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "window_id": self.window_id,
        }
