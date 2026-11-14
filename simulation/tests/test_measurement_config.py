"""Step 1 Tests — Refactor the measurement-noise YAML (`test_measurement_config.py`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pytest

from simulation.stimulus.noise_framework.config_schema import load_yaml_validated

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE5_YAML = REPO_ROOT / "configs" / "noise_framework" / "stage5_measurement_imperfections.yaml"
BASELINES_YAML = (
    REPO_ROOT
    / "configs"
    / "noise_framework"
    / "baselines"
    / "stage5_measurement_baselines.yaml"
)


@pytest.fixture
def config() -> Dict[str, Any]:
    return load_yaml_validated(
        STAGE5_YAML,
        required_keys=("schema_version", "stage", "enabled", "generator", "composition", "validation"),
    )


def test_no_hardcoded_regimes(config: Dict[str, Any]) -> None:
    assert "regimes" not in config


def test_generator_identity(config: Dict[str, Any]) -> None:
    assert config["generator"]["generator_id"] == "measurement_noise_01"
    assert config["generator"]["generator_type"] == "measurement_noise"
    assert config["generator"]["output_domain"] == "measurement_fault_parameters"


def test_assignment_source(config: Dict[str, Any]) -> None:
    assert config["generator"]["profile_assignment"]["source"] == "pd_registry"


def test_assignment_persistence(config: Dict[str, Any]) -> None:
    assert config["generator"]["profile_assignment"]["persistence_seconds"] == 100


def test_sampling_frequency(config: Dict[str, Any]) -> None:
    assert config["generator"]["sampling"]["samples_per_second"] == 10
    assert config["generator"]["sampling"]["synchronized"] is True


def test_supported_mechanisms(config: Dict[str, Any]) -> None:
    mechanisms = config["generator"]["supported_mechanisms"]
    assert set(mechanisms) == {
        "independent_readout",
        "temporally_correlated_readout",
        "detector_dropout",
        "observation_delay",
    }


def test_output_domain_validation_flags(config: Dict[str, Any]) -> None:
    val = config["validation"]
    assert val["enforce_probability_bounds"] is True
    assert val["require_finite_values"] is True
    assert val["require_valid_delay"] is True
