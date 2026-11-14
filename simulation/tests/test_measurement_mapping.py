"""Step 3 Tests — Implement the PD-to-parameter mapping (`test_measurement_mapping.py`)."""

from __future__ import annotations

import copy
from pathlib import Path
import pytest

from simulation.stimulus.noise_framework.config_schema import NoiseProfile
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.syndrome.measurement_mapping import (
    MeasurementParameterMapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PD_YAML = REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml"
MAP_YAML = REPO_ROOT / "configs" / "noise_framework" / "measurement_parameter_mapping.yaml"


@pytest.fixture
def registry() -> PDRegistry:
    return PDRegistry(PD_YAML)


@pytest.fixture
def mapping() -> MeasurementParameterMapping:
    return MeasurementParameterMapping(MAP_YAML)


def test_all_targets_defined(mapping: MeasurementParameterMapping, registry: PDRegistry) -> None:
    expected = {
        "q_base_x",
        "q_base_z",
        "temporal_sticky_prob",
        "detector_dropout_prob",
        "observation_delay_samples",
    }
    assert set(mapping.targets) == expected
    assert mapping.validate(registry) is True


def test_unknown_source_variable_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    bad_raw = copy.deepcopy(mapping.raw)
    bad_raw["target_mappings"][0]["source_variable"] = "nonexistent_variable"
    bad_map = MeasurementParameterMapping(bad_raw)
    with pytest.raises(ValueError, match="source_variable"):
        bad_map.validate(registry)


def test_incompatible_distribution_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    base_prof = registry.get("PD_01")
    bad_prof = object.__new__(NoiseProfile)
    object.__setattr__(bad_prof, "profile_id", "PD_BAD")
    object.__setattr__(bad_prof, "distribution_type", "cauchy_unbounded")
    object.__setattr__(bad_prof, "parameters", dict(base_prof.parameters))
    object.__setattr__(bad_prof, "support", dict(base_prof.support))
    object.__setattr__(bad_prof, "metadata", dict(base_prof.metadata))
    with pytest.raises(ValueError, match="distribution_type"):
        mapping.validate_profile_compatibility(bad_prof)


def test_missing_target_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    bad_raw = copy.deepcopy(mapping.raw)
    bad_raw["target_mappings"] = [
        e for e in bad_raw["target_mappings"] if e["target"] != "q_base_z"
    ]
    bad_map = MeasurementParameterMapping(bad_raw)
    with pytest.raises(ValueError, match="Missing required target"):
        bad_map.validate(registry)


def test_duplicate_target_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    bad_raw = copy.deepcopy(mapping.raw)
    bad_raw["target_mappings"].append(copy.deepcopy(bad_raw["target_mappings"][0]))
    bad_map = MeasurementParameterMapping(bad_raw)
    with pytest.raises(ValueError, match="Duplicate target"):
        bad_map.validate(registry)


def test_invalid_mapping_dimensions_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    bad_raw = copy.deepcopy(mapping.raw)
    bad_raw["target_mappings"][0]["dimension"] = 3
    bad_map = MeasurementParameterMapping(bad_raw)
    with pytest.raises(ValueError, match="Invalid dimension"):
        bad_map.validate(registry)


def test_invalid_integer_conversion_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    bad_raw = copy.deepcopy(mapping.raw)
    bad_raw["target_mappings"][4]["integer_values"] = [0, 1.5, 2]
    bad_map = MeasurementParameterMapping(bad_raw)
    with pytest.raises(ValueError, match="integer_values"):
        bad_map.validate(registry)


def test_incompatible_support_rejected(
    mapping: MeasurementParameterMapping, registry: PDRegistry
) -> None:
    base_prof = registry.get("PD_01")
    bad_prof = object.__new__(NoiseProfile)
    object.__setattr__(bad_prof, "profile_id", "PD_BAD_SUPP")
    object.__setattr__(bad_prof, "distribution_type", "truncated_gaussian")
    object.__setattr__(bad_prof, "parameters", dict(base_prof.parameters))
    object.__setattr__(
        bad_prof,
        "support",
        {"min": -0.5, "max": 1.5, "variable": "fault_probability", "units": "prob"},
    )
    object.__setattr__(bad_prof, "metadata", dict(base_prof.metadata))
    with pytest.raises(ValueError, match="support"):
        mapping.validate_profile_compatibility(bad_prof)
