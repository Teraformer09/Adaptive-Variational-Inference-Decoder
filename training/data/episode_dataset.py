"""Paired Syndrome-Observation & Ground-Truth Episode Dataset (`episode_dataset.py`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
import numpy as np


@dataclass
class EpisodeRecord:
    """Single multi-epoch QEC control trajectory separating policy observations from privileged ground truth."""

    trajectory_id: str
    split_name: str
    regime: str
    seed: int
    obs: np.ndarray                 # (T, 82) syndrome-conditioned policy observations
    actions: np.ndarray             # (T,) int64
    rewards: np.ndarray             # (T,) float64
    dones: np.ndarray               # (T,) float64
    target_gains: np.ndarray        # (T, 9) float64
    optimal_actions: np.ndarray     # (T,) int64 (privileged diagnostic reference only)
    h_states: np.ndarray            # (T, 64) float64
    c_states: np.ndarray            # (T, 64) float64
    posterior_means: np.ndarray     # (T, 21) float64
    posterior_vars: np.ndarray      # (T, 21) float64
    true_theta_z: np.ndarray        # (T, 21) float64 privileged ground truth
    decoded_failures: np.ndarray    # (T,) int64
    static_failures: np.ndarray     # (T,) int64
    window_rounds: int = 256
    reward_terms: List[Dict[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeDataset:
    """In-memory collection of `EpisodeRecord` trajectories with split provenance."""

    def __init__(self, split_name: str) -> None:
        self.split_name = str(split_name)
        self.episodes: List[EpisodeRecord] = []
        self._ids: Set[str] = set()

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def trajectory_ids(self) -> Set[str]:
        return set(self._ids)

    def add(self, record: EpisodeRecord) -> None:
        if record.split_name != self.split_name:
            raise ValueError(
                f"Cannot insert split '{record.split_name}' episode into '{self.split_name}' dataset"
            )
        if record.trajectory_id in self._ids:
            raise ValueError(f"Duplicate trajectory_id '{record.trajectory_id}'")
        self._ids.add(record.trajectory_id)
        self.episodes.append(record)
