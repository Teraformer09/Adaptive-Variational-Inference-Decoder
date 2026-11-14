"""Step 7 Tests — Implement measurement parameter sampling (`test_measurement_sampling.py`)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest
import yaml

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PD_YAML = REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml"


def _make_gen(seed: int = 42, mapping_path: str | None = None) -> MeasurementNoiseGenerator:
    params = {}
    if mapping_path is not None:
        params["mapping_path"] = mapping_path
    spec = GeneratorSpec(
        generator_id="measurement_noise_01",
        generator_type="measurement_noise",
        enabled=True,
        seed=seed,
        output_domain="measurement_fault_parameters",
        output_shape=(11,),
        composition_group="measurement_faults",
        parameters=params,
    )
    return MeasurementNoiseGenerator(spec)


def test_ten_seconds_generates_100_valid_samples_with_unique_ids_and_spacing() -> None:
    pd_reg = PDRegistry(PD_YAML)
    gen = _make_gen(seed=42)
    gen.assign_profile(pd_reg.get("PD_04"), window_id=0)
    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)

    samples = []
    for sample_id, ts, _ in clock.iter_duration(10.0):
        s = gen.sample(timestamp=ts, sample_id=sample_id)
        assert isinstance(s, MeasurementNoiseParameters)
        assert s.profile_id == "PD_04"
        samples.append(s)

    # N = 10 * 10 = 100 samples
    assert len(samples) == 100
    ids = [s.sample_id for s in samples]
    assert ids == list(range(100))
    assert len(set(ids)) == 100

    timestamps = np.array([s.timestamp for s in samples], dtype=np.float64)
    diffs = np.diff(timestamps)
    assert np.allclose(diffs, 0.1, atol=1e-9)
    assert gen.current_profile is not None
    assert gen.current_profile.profile_id == "PD_04"


def test_samples_follow_assigned_pd_statistical_distribution() -> None:
    pd_reg = PDRegistry(PD_YAML)
    prof = pd_reg.get("PD_04")
    theo_mean, _ = prof.theoretical_moments()

    gen = _make_gen(seed=101)
    gen.assign_profile(prof, window_id=0)
    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)

    qx_draws = []
    for sample_id, ts, _ in clock.iter_duration(90.0):
        s = gen.sample(timestamp=ts, sample_id=sample_id)
        qx_draws.append(s.q_base_x / 0.85)

    emp_mean = float(np.mean(qx_draws))
    assert abs(emp_mean - theo_mean) < 0.0004


def test_invalid_mapping_fails_immediately() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad_map_path = Path(tmp) / "bad_map.yaml"
        bad_cfg = {
            "schema_version": "1.0",
            "generator_type": "measurement_noise",
            "mappings": {
                "independent_readout": {
                    "target": "q_base_x_z",
                    # missing check_sampling_mode and basis_sampling_mode
                }
            },
            "validation": {"probability_min": 0.0, "probability_max": 1.0},
        }
        bad_map_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
        with pytest.raises(ValueError):
            _make_gen(seed=42, mapping_path=str(bad_map_path))
