"""Trajectory-Level Disjoint Split Enforcement (`trajectory_split.py`).

Prevents any trajectory or window leakage across `train`, `val`, `test`, and `heldout_mixtures`
partitions (EVAL-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    seed_range: Tuple[int, int]
    regimes: Tuple[str, ...]


DEFAULT_PARTITIONS_V2: Dict[str, PartitionSpec] = {
    "train": PartitionSpec(
        name="train",
        seed_range=(1000, 1099),
        regimes=(
            "gaussian_stationary",
            "ou_drift",
            "zz_crosstalk",
            "tls_telegraph",
            "nonstationary_burst",
        ),
    ),
    "val": PartitionSpec(
        name="val",
        seed_range=(1500, 1599),
        regimes=(
            "gaussian_stationary",
            "ou_drift",
            "zz_crosstalk",
            "tls_telegraph",
            "nonstationary_burst",
        ),
    ),
    "test": PartitionSpec(
        name="test",
        seed_range=(2026, 2099),
        regimes=(
            "gaussian_stationary",
            "ou_drift",
            "zz_crosstalk",
            "tls_telegraph",
            "nonstationary_burst",
        ),
    ),
    "heldout_mixtures": PartitionSpec(
        name="heldout_mixtures",
        seed_range=(3000, 3099),
        regimes=(
            "heldout_multi_pair_12_67",
            "heldout_spatial_3qubit_streak",
            "heldout_phenomenological_readout",
        ),
    ),
}


class TrajectorySplitManager:
    """Validates and generates strictly disjoint trajectory IDs and seeds across partitions."""

    def __init__(self, partitions: Dict[str, PartitionSpec] | None = None) -> None:
        self.partitions = dict(partitions) if partitions is not None else dict(DEFAULT_PARTITIONS_V2)
        self.verify_disjoint_seed_ranges()

    def verify_disjoint_seed_ranges(self) -> None:
        names = list(self.partitions.keys())
        for i in range(len(names)):
            a = self.partitions[names[i]]
            if a.seed_range[0] > a.seed_range[1]:
                raise ValueError(f"Invalid seed_range {a.seed_range} for split '{a.name}'")
            for j in range(i + 1, len(names)):
                b = self.partitions[names[j]]
                overlap = max(a.seed_range[0], b.seed_range[0]) <= min(a.seed_range[1], b.seed_range[1])
                if overlap:
                    raise ValueError(
                        f"Trajectory seed overlap between '{a.name}' {a.seed_range} and '{b.name}' {b.seed_range}"
                    )

    def episode_seed(self, split_name: str, base_seed: int, regime_idx: int) -> int:
        spec = self.partitions[split_name]
        if not (spec.seed_range[0] <= int(base_seed) <= spec.seed_range[1]):
            raise ValueError(
                f"Seed {base_seed} lies outside '{split_name}' partition range {spec.seed_range}"
            )
        # Disjoint injective trajectory seed encoding (split_offset + seed*37 + regime_idx)
        split_offset = {"train": 100_000, "val": 200_000, "test": 300_000, "heldout_mixtures": 400_000}.get(
            split_name, 500_000
        )
        return int(split_offset + int(base_seed) * 37 + int(regime_idx))

    @staticmethod
    def assert_no_trajectory_leakage(
        train_trajectory_ids: Set[str],
        eval_trajectory_ids: Set[str],
    ) -> None:
        leaked = train_trajectory_ids.intersection(eval_trajectory_ids)
        if leaked:
            raise ValueError(f"Trajectory leakage detected across splits: {sorted(leaked)}")
