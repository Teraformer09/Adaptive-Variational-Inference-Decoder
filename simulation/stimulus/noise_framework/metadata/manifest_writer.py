"""Dataset Manifest Writer (`manifest_writer.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from simulation.stimulus.noise_framework.metadata.provenance import sha256_file_bytes


class ManifestWriter:
    """Records reproducible dataset manifests and trajectory statistics."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def add_trajectory_record(
        self,
        encoded_traj: Dict[str, Any],
        npz_path: str | Path,
        provenance: Dict[str, Any],
        assignment_manifest: Dict[str, Any],
        bin_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        npz_p = Path(npz_path)
        rec: Dict[str, Any] = {
            "trajectory_id": encoded_traj["trajectory_id"],
            "split_id": encoded_traj["split_id"],
            "master_seed": int(encoded_traj["master_seed"]),
            "num_composite_samples": int(encoded_traj["num_composite_samples"]),
            "total_rounds": int(encoded_traj["total_rounds"]),
            "npz_file": npz_p.name,
            "npz_sha256": sha256_file_bytes(npz_p),
            "theta_hash": provenance["theta_hash"],
            "profile_config_hash": provenance["profile_config_hash"],
            "generator_config_hash": provenance["generator_config_hash"],
            "code_geometry_hash": provenance["code_geometry_hash"],
            "assignment_counts": assignment_manifest["assignment_counts"],
            "assignment_history": {
                str(k): v for k, v in encoded_traj.get("assignment_history", {}).items()
            },
            "fault_and_syndrome_stats": {
                "mean_z_fault_weight": float(
                    np.mean(np.sum(encoded_traj["fault_masks_z"], axis=1))
                ),
                "mean_x_fault_weight": float(
                    np.mean(np.sum(encoded_traj["fault_masks_x"], axis=1))
                ),
                "non_trivial_syndrome_rate_x": float(
                    np.mean(np.any(encoded_traj["syndromes_x"] != 0, axis=1))
                ),
                "non_trivial_syndrome_rate_z": float(
                    np.mean(np.any(encoded_traj["syndromes_z"] != 0, axis=1))
                ),
                "logical_z_flip_rate": float(np.mean(encoded_traj["logical_error_z"])),
                "logical_x_flip_rate": float(np.mean(encoded_traj["logical_error_x"])),
                "measurement_flip_rate_x": float(np.mean(encoded_traj["measurement_faults_x"])),
                "measurement_flip_rate_z": float(np.mean(encoded_traj["measurement_faults_z"])),
                "dropout_rate_x": float(np.mean(encoded_traj["validity_mask_x"] == 0)),
            },
        }
        if bin_path is not None and Path(bin_path).exists():
            bp = Path(bin_path)
            rec["bin_file"] = bp.name
            rec["bin_sha256"] = sha256_file_bytes(bp)
        self.entries.append(rec)
        return rec

    def write_manifest(self, output_path: str | Path, extra_header: Dict[str, Any] | None = None) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        by_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        theta_by_split: Dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for e in self.entries:
            sp = e["split_id"]
            by_split.setdefault(sp, []).append(e["trajectory_id"])
            theta_by_split.setdefault(sp, set()).add(e["theta_hash"])

        disjoint_train_val = len(theta_by_split["train"] & theta_by_split["val"]) == 0
        disjoint_train_test = len(theta_by_split["train"] & theta_by_split["test"]) == 0
        disjoint_val_test = len(theta_by_split["val"] & theta_by_split["test"]) == 0

        manifest = {
            "schema_version": "1.0",
            "total_trajectories": len(self.entries),
            "total_composite_samples": int(sum(e["num_composite_samples"] for e in self.entries)),
            "total_syndrome_rounds": int(sum(e["total_rounds"] for e in self.entries)),
            "splits": {k: len(v) for k, v in by_split.items()},
            "partition_disjointness": {
                "train_intersect_val_empty": disjoint_train_val,
                "train_intersect_test_empty": disjoint_train_test,
                "val_intersect_test_empty": disjoint_val_test,
                "all_disjoint": bool(
                    disjoint_train_val and disjoint_train_test and disjoint_val_test
                ),
            },
            "trajectories": self.entries,
        }
        if extra_header:
            manifest.update(extra_header)
        out_p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return out_p
