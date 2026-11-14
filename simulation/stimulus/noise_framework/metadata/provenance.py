"""Cryptographic Provenance Tracker (`provenance.py`).

Calculates stable SHA-256 hashes for:
  - Profile configuration (`pd_profiles_16.yaml`)
  - Generator registry (`generator_registry.yaml`)
  - Assignment policy (`generator_assignment.yaml`)
  - Composition rules (`composition_rules.yaml`)
  - Code geometry (`Hx.json`, `Hz.json`)
  - Random seeds and trajectory configuration identity `Theta`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.config_schema import (
    compute_canonical_config_hash,
    load_yaml_validated,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
FRAMEWORK_VERSION = "1.0.0-seven-stage"


def sha256_file_bytes(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compute_code_geometry_hash() -> str:
    code = build_rotated_d3_code()
    payload = {
        "hx": code.hx.tolist(),
        "hz": code.hz.tolist(),
        "logical_x": code.logical_X.tolist(),
        "logical_z": code.logical_Z.tolist(),
    }
    return compute_canonical_config_hash(payload)


def build_provenance_record(
    master_seed: int,
    assignment_policy: str,
    duration_seconds: float,
    split_id: str = "train",
    extra_theta: Dict[str, Any] | None = None,
    framework_yaml_path: str | Path | None = None,
) -> Dict[str, Any]:
    fw_path = (
        Path(framework_yaml_path)
        if framework_yaml_path is not None
        else (REPO_ROOT / "configs" / "noise_framework" / "framework_v1.yaml")
    )
    fw_cfg = load_yaml_validated(fw_path, required_keys=("schema_version", "configs"))
    cfgs = fw_cfg["configs"]

    profile_hash = sha256_file_bytes(REPO_ROOT / cfgs["pd_profiles"])
    gen_reg_hash = sha256_file_bytes(REPO_ROOT / cfgs["generator_registry"])
    assign_hash = sha256_file_bytes(REPO_ROOT / cfgs["generator_assignment"])
    comp_hash = sha256_file_bytes(REPO_ROOT / cfgs["composition_rules"])
    meas_hash = sha256_file_bytes(REPO_ROOT / cfgs["stage5_measurement"])
    geom_hash = compute_code_geometry_hash()

    theta_vector: Dict[str, Any] = {
        "master_seed": int(master_seed),
        "assignment_policy": str(assignment_policy),
        "duration_seconds": float(duration_seconds),
        "split_id": str(split_id),
        "profile_config_hash": profile_hash,
        "generator_config_hash": gen_reg_hash,
    }
    if extra_theta:
        theta_vector.update(extra_theta)

    theta_hash = compute_canonical_config_hash(theta_vector)
    return {
        "framework_version": FRAMEWORK_VERSION,
        "master_seed": int(master_seed),
        "split_id": str(split_id),
        "assignment_policy": str(assignment_policy),
        "duration_seconds": float(duration_seconds),
        "profile_config_hash": profile_hash,
        "generator_config_hash": gen_reg_hash,
        "assignment_config_hash": assign_hash,
        "composition_config_hash": comp_hash,
        "measurement_config_hash": meas_hash,
        "code_geometry_hash": geom_hash,
        "theta_hash": theta_hash,
        "theta_vector": theta_vector,
    }
