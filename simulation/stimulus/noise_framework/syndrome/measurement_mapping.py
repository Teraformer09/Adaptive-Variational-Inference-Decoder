"""Step 3: Validated PD-to-Parameter Mapping (`measurement_mapping.py`).

Resolves explicit YAML mappings from `measurement_parameter_mapping.yaml` against the
16-PD `PDRegistry` without implicit parameter interpretation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from simulation.stimulus.noise_framework.config_schema import NoiseProfile, load_yaml_validated
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REQUIRED_TARGETS = (
    "q_base_x",
    "q_base_z",
    "temporal_sticky_prob",
    "detector_dropout_prob",
    "observation_delay_samples",
)
EXPECTED_DIMENSIONS = {
    "q_base_x": 4,
    "q_base_z": 4,
    "temporal_sticky_prob": 1,
    "detector_dropout_prob": 1,
    "observation_delay_samples": 1,
}
SUPPORTED_SOURCE_VARIABLES = {
    "fault_probability",
    "sticky_scale",
    "dropout_scale",
    "delay_quantiles",
}
SUPPORTED_DISTRIBUTIONS = {"truncated_gaussian"}


class MeasurementParameterMapping:
    """Validated mapping specification converting PD samples into `MeasurementNoiseParameters`."""

    def __init__(self, config_or_path: str | Path | Dict[str, Any] | None = None) -> None:
        if config_or_path is None:
            default_path = (
                Path(__file__).resolve().parents[4]
                / "configs"
                / "noise_framework"
                / "measurement_parameter_mapping.yaml"
            )
            self.raw: Dict[str, Any] = load_yaml_validated(
                default_path,
                required_keys=("schema_version", "generator_type", "target_mappings", "validation"),
            )
        elif isinstance(config_or_path, (str, Path)):
            self.raw = load_yaml_validated(
                config_or_path,
                required_keys=("schema_version", "generator_type", "target_mappings", "validation"),
            )
        elif isinstance(config_or_path, dict):
            self.raw = copy.deepcopy(config_or_path)
        else:
            raise TypeError(f"Unsupported mapping input: {type(config_or_path)}")

        self.generator_type: str = str(self.raw.get("generator_type", ""))
        self.target_entries: List[Dict[str, Any]] = list(self.raw.get("target_mappings", []))
        self.validation_cfg: Dict[str, Any] = dict(self.raw.get("validation", {}))

    @property
    def targets(self) -> List[str]:
        return [str(entry.get("target", "")) for entry in self.target_entries]

    def validate(self, registry: PDRegistry | None = None) -> bool:
        if self.generator_type != "measurement_noise":
            raise ValueError(
                f"Expected generator_type='measurement_noise', got '{self.generator_type}'"
            )

        target_list = self.targets
        if len(target_list) != len(set(target_list)):
            raise ValueError(f"Duplicate target detected in measurement parameter mapping: {target_list}")

        missing = set(REQUIRED_TARGETS) - set(target_list)
        if missing:
            raise ValueError(f"Missing required target(s) in mapping: {sorted(missing)}")

        extra = set(target_list) - set(REQUIRED_TARGETS)
        if extra:
            raise ValueError(f"Unsupported extra target(s) in mapping: {sorted(extra)}")

        for entry in self.target_entries:
            tgt = str(entry.get("target", ""))
            dim = entry.get("dimension")
            if dim != EXPECTED_DIMENSIONS[tgt]:
                raise ValueError(
                    f"Invalid dimension {dim} for target '{tgt}' (expected {EXPECTED_DIMENSIONS[tgt]})"
                )

            src_var = str(entry.get("source_variable", ""))
            if not src_var or src_var not in SUPPORTED_SOURCE_VARIABLES:
                raise ValueError(f"Unsupported or unknown source_variable '{src_var}' for '{tgt}'")

            prof_param = entry.get("use_profile_parameter")
            if prof_param is not None and str(prof_param) not in SUPPORTED_SOURCE_VARIABLES:
                raise ValueError(f"Unsupported profile parameter '{prof_param}' for '{tgt}'")

            bounds = entry.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid bounds {bounds} for target '{tgt}'")

            val_type = str(entry.get("value_type", "float"))
            if tgt == "observation_delay_samples":
                if val_type != "integer":
                    raise ValueError("observation_delay_samples must have value_type='integer'")
                q_rule = str(entry.get("quantization_rule", ""))
                if q_rule not in ("threshold_quantiles", "round_nearest"):
                    raise ValueError(f"Invalid integer conversion rule '{q_rule}'")
                int_vals = entry.get("integer_values", [])
                if not int_vals or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < bounds[0] or v > bounds[1]
                    for v in int_vals
                ):
                    raise ValueError(f"Invalid integer_values {int_vals} for '{tgt}'")
                thresholds = entry.get("thresholds", [])
                if len(thresholds) != len(int_vals) - 1:
                    raise ValueError("thresholds length must equal len(integer_values) - 1")
            else:
                if val_type != "float":
                    raise ValueError(f"Probability target '{tgt}' must have value_type='float'")
                if bounds[0] < 0.0 or bounds[1] > 1.0:
                    raise ValueError(f"Probability target '{tgt}' bounds {bounds} outside [0, 1]")

        if registry is not None:
            for pid in registry.list_profiles():
                prof = registry.get(pid)
                self.validate_profile_compatibility(prof)

        return True

    def validate_profile_compatibility(self, profile: NoiseProfile) -> bool:
        if profile.distribution_type not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Incompatible distribution_type '{profile.distribution_type}' in {profile.profile_id}"
            )
        prof_var = str(profile.support.get("variable", ""))
        if prof_var not in SUPPORTED_SOURCE_VARIABLES:
            raise ValueError(
                f"Profile {profile.profile_id} variable '{prof_var}' is incompatible with mapping"
            )
        if profile.min_val < 0.0 or profile.max_val > 1.0 or profile.min_val >= profile.max_val:
            raise ValueError(
                f"Profile {profile.profile_id} support [{profile.min_val}, {profile.max_val}] is incompatible"
            )
        for entry in self.target_entries:
            src_var = str(entry["source_variable"])
            if src_var != prof_var and src_var not in profile.parameters:
                raise ValueError(
                    f"Source variable '{src_var}' not found in profile {profile.profile_id}"
                )
            prof_param = entry.get("use_profile_parameter")
            if prof_param is not None and str(prof_param) not in profile.parameters:
                raise ValueError(
                    f"Required profile parameter '{prof_param}' missing in {profile.profile_id}"
                )
        return True
