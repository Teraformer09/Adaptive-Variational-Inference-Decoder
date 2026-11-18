"""Dataset Integrity & Anti-Leakage Validator (`dataset_validation.py`)."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from training.data.episode_dataset import EpisodeDataset
from training.data.trajectory_split import TrajectorySplitManager
from training.observations.observation_schema import OBS_V2_DIM, validate_observation_v2


def validate_episode_datasets(datasets: Sequence[EpisodeDataset]) -> None:
    """Verify every episode's 82-D observation contract and strict pairwise split disjointness."""
    seen_by_split = {}
    for ds in datasets:
        if len(ds) == 0:
            raise ValueError(f"Dataset split '{ds.split_name}' is empty")
        seen_by_split[ds.split_name] = ds.trajectory_ids
        for ep in ds.episodes:
            if ep.obs.ndim != 2 or ep.obs.shape[1] != OBS_V2_DIM:
                raise ValueError(f"Episode '{ep.trajectory_id}' invalid obs shape {ep.obs.shape}")
            for t in range(ep.obs.shape[0]):
                validate_observation_v2(ep.obs[t], strict_probabilities=True)
            if not np.all(np.isfinite(ep.rewards)):
                raise ValueError(f"Episode '{ep.trajectory_id}' contains non-finite rewards")

    names = list(seen_by_split.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            TrajectorySplitManager.assert_no_trajectory_leakage(
                seen_by_split[names[i]], seen_by_split[names[j]]
            )
