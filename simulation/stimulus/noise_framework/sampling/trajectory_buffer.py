"""Trajectory Buffer storing individual generator contributions and composite samples."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from simulation.stimulus.noise_framework.config_schema import (
    CompositeNoiseSample,
    NoiseContribution,
)


class TrajectoryBuffer:
    """Preserves synchronized per-generator contributions and composed samples across a trajectory."""

    def __init__(self) -> None:
        self.contribution_sets: List[List[NoiseContribution]] = []
        self.composite_samples: List[CompositeNoiseSample] = []

    def append_step(
        self,
        contributions: List[NoiseContribution],
        composite_sample: CompositeNoiseSample,
    ) -> None:
        if not contributions:
            raise ValueError("Cannot append empty contribution list to TrajectoryBuffer")
        self.contribution_sets.append(list(contributions))
        self.composite_samples.append(composite_sample)

    def __len__(self) -> int:
        return len(self.composite_samples)

    def timestamps(self) -> np.ndarray:
        return np.array([c.timestamp for c in self.composite_samples], dtype=np.float64)

    def window_ids(self) -> np.ndarray:
        return np.array([c.window_id for c in self.composite_samples], dtype=np.int32)

    def contributions_for_generator(self, generator_id: str) -> List[NoiseContribution]:
        out: List[NoiseContribution] = []
        for step_contribs in self.contribution_sets:
            for c in step_contribs:
                if c.generator_id == generator_id:
                    out.append(c)
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "num_samples": len(self.composite_samples),
            "unique_windows": sorted({int(c.window_id) for c in self.composite_samples}),
            "generators": sorted(
                {
                    c.generator_id
                    for step in self.contribution_sets
                    for c in step
                }
            ),
        }
