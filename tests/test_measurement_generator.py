"""Step 4 Tests — Implement the measurement-noise generator (`tests/test_measurement_generator.py`)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec, NoiseProfile
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PD_YAML = REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml"


def _make_spec(seed: int = 42) -> GeneratorSpec:
    return GeneratorSpec(
        generator_id="measurement_noise_01",
        generator_type="measurement_noise",
        output_domain="measurement_fault_parameters",
        enabled=True,
        seed=seed,
        parameters={},
    )


def test_generator_initialization() -> None:
    gen = MeasurementNoiseGenerator(_make_spec())
    assert gen.generator_id == "measurement_noise_01"
    assert gen.generator_type == "measurement_noise"
    assert gen.output_domain == "measurement_fault_parameters"


def test_valid_pd_assignment() -> None:
    reg = PDRegistry(PD_YAML)
    gen = MeasurementNoiseGenerator(_make_spec())
    gen.assign_profile(reg.get("PD_04"), window_id=0)
    assert gen.current_profile is not None
    assert gen.current_profile.profile_id == "PD_04"


def test_unknown_pd_assignment_rejected() -> None:
    reg = PDRegistry(PD_YAML)
    gen = MeasurementNoiseGenerator(_make_spec())
    with pytest.raises(ValueError, match="Unknown PD"):
        gen.assign_profile("PD_99", window_id=0, registry=reg)


def test_valid_sample_generation_and_shapes_and_ids() -> None:
    reg = PDRegistry(PD_YAML)
    gen = MeasurementNoiseGenerator(_make_spec())
    gen.assign_profile(reg.get("PD_04"), window_id=0)
    sample = gen.sample(timestamp=1.0, sample_id=10)
    assert isinstance(sample, MeasurementNoiseParameters)
    assert sample.q_base_x.shape == (4,)
    assert sample.q_base_z.shape == (4,)
    assert sample.sample_id == 10
    assert sample.timestamp == 1.0
    assert sample.profile_id == "PD_04"


def test_independent_rng_reproducibility_and_different_seeds() -> None:
    reg = PDRegistry(PD_YAML)
    g1 = MeasurementNoiseGenerator(_make_spec(seed=100))
    g2 = MeasurementNoiseGenerator(_make_spec(seed=100))
    g3 = MeasurementNoiseGenerator(_make_spec(seed=200))
    for g in (g1, g2, g3):
        g.assign_profile(reg.get("PD_07"), window_id=0)

    s1 = g1.sample(timestamp=0.5, sample_id=5)
    s2 = g2.sample(timestamp=0.5, sample_id=5)
    s3 = g3.sample(timestamp=0.5, sample_id=5)
    assert np.allclose(s1.to_vector(), s2.to_vector())
    assert not np.allclose(s1.to_vector(), s3.to_vector())


def test_state_save_and_load_reproduces_subsequent_samples() -> None:
    reg = PDRegistry(PD_YAML)
    g1 = MeasurementNoiseGenerator(_make_spec(seed=321))
    g2 = MeasurementNoiseGenerator(_make_spec(seed=999))
    g1.assign_profile(reg.get("PD_02"), window_id=0)
    g2.assign_profile(reg.get("PD_02"), window_id=0)

    _ = g1.sample(timestamp=0.0, sample_id=0)
    saved = g1.state_dict()
    g2.load_state_dict(saved)

    next1 = g1.sample(timestamp=0.1, sample_id=1)
    next2 = g2.sample(timestamp=0.1, sample_id=1)
    assert np.allclose(next1.to_vector(), next2.to_vector())


def test_incompatible_pd_rejected() -> None:
    reg = PDRegistry(PD_YAML)
    base_prof = reg.get("PD_01")
    bad_prof = object.__new__(NoiseProfile)
    object.__setattr__(bad_prof, "profile_id", "PD_INCOMPAT")
    object.__setattr__(bad_prof, "distribution_type", "truncated_gaussian")
    object.__setattr__(bad_prof, "parameters", dict(base_prof.parameters))
    object.__setattr__(
        bad_prof,
        "support",
        {"min": 0.01, "max": 0.05, "variable": "qubit_frequency_mhz", "units": "MHz"},
    )
    object.__setattr__(bad_prof, "metadata", dict(base_prof.metadata))

    gen = MeasurementNoiseGenerator(_make_spec())
    with pytest.raises(ValueError, match="incompatible"):
        gen.assign_profile(bad_prof, window_id=0)
