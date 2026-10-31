"""Schema validation, split-integrity verification, and inference input sanitization for AVID trajectories."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "trajectory.schema.json"


class TrajectorySchemaValidator:
    """Validates trajectory dicts against `world_model/dataset/trajectory.schema.json`."""

    def __init__(self, schema_path: Path = SCHEMA_PATH) -> None:
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema: Dict[str, Any] = json.load(f)
        self.validator = jsonschema.Draft202012Validator(self.schema)

    def validate(self, record: Dict[str, Any]) -> None:
        """Raise `jsonschema.ValidationError` if `record` violates the trajectory schema."""
        self.validator.validate(record)

    def validate_batch(self, records: Iterable[Dict[str, Any]]) -> int:
        """Validate a sequence of trajectory records and return the validated count."""
        count = 0
        for rec in records:
            self.validate(rec)
            count += 1
        return count


def verify_zero_leakage_splits(
    splits: Dict[str, Sequence[Dict[str, Any]]]
) -> Tuple[bool, Dict[str, Any]]:
    """Verify that train/val/test splits share zero trajectory IDs and zero simulator seeds.

    Args:
        splits: Mapping from split name (e.g. 'train', 'val', 'test_id', 'test_ood')
                to a sequence of trajectory records.

    Returns:
        (is_valid, report_dict)
    """
    seen_ids: Dict[str, str] = {}
    seen_seeds: Dict[int, str] = {}
    overlaps: List[str] = []

    split_counts: Dict[str, int] = {}
    for split_name, records in splits.items():
        split_counts[split_name] = len(records)
        for rec in records:
            tid = rec["trajectory_id"]
            seed = int(rec["seed"])
            if tid in seen_ids and seen_ids[tid] != split_name:
                overlaps.append(f"trajectory_id={tid} in both {seen_ids[tid]} and {split_name}")
            if seed in seen_seeds and seen_seeds[seed] != split_name:
                overlaps.append(f"seed={seed} in both {seen_seeds[seed]} and {split_name}")
            seen_ids[tid] = split_name
            seen_seeds[seed] = split_name

    is_valid = len(overlaps) == 0
    return is_valid, {
        "zero_leakage_verified": is_valid,
        "split_counts": split_counts,
        "total_unique_trajectories": len(seen_ids),
        "total_unique_seeds": len(seen_seeds),
        "overlaps": overlaps,
    }


def strip_latent_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of a trajectory record with `latent_truth` removed for inference safety."""
    sanitized = copy.copy(record)
    sanitized.pop("latent_truth", None)
    return sanitized
