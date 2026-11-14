"""Step 5 Tests — Register the generator (`test_measurement_registry.py`)."""

from __future__ import annotations

from pathlib import Path
import pytest

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REG_YAML = REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml"


def test_generator_type_resolves_and_active_generators_returns_instance() -> None:
    reg = GeneratorRegistry(REG_YAML, master_seed=42)
    active = reg.active_generators()
    ids = [g.generator_id for g in active]
    assert len(ids) == len(set(ids))
    assert "measurement_noise_01" in ids
    meas = reg.get("measurement_noise_01")
    assert isinstance(meas, MeasurementNoiseGenerator)
    assert meas.output_domain == "measurement_fault_parameters"


def test_duplicate_id_rejected() -> None:
    reg = GeneratorRegistry()
    spec = GeneratorSpec(
        generator_id="measurement_noise_01",
        generator_type="measurement_noise",
        enabled=True,
        seed=10,
        output_domain="measurement_fault_parameters",
        output_shape=(11,),
        composition_group="measurement_faults",
    )
    reg.register(spec)
    with pytest.raises(ValueError, match="Duplicate generator_id"):
        reg.register(spec)


def test_disabled_generator_not_instantiated() -> None:
    reg = GeneratorRegistry()
    spec_disabled = GeneratorSpec(
        generator_id="measurement_disabled",
        generator_type="measurement_noise",
        enabled=False,
        seed=11,
        output_domain="measurement_fault_parameters",
        output_shape=(11,),
        composition_group="measurement_faults",
    )
    reg.register(spec_disabled)
    active = reg.instantiate_all()
    assert len(active) == 0
    with pytest.raises(KeyError):
        reg.get("measurement_disabled")


def test_unsupported_generator_type_rejected() -> None:
    reg = GeneratorRegistry()
    spec_bad = GeneratorSpec(
        generator_id="bad_gen",
        generator_type="nonexistent_generator_type",
        enabled=True,
        seed=12,
        output_domain="measurement_fault_parameters",
        output_shape=(11,),
        composition_group="measurement_faults",
    )
    with pytest.raises(ValueError, match="Unknown generator_type"):
        reg.register(spec_bad)
