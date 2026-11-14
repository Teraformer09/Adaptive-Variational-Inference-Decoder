"""Step 10 Tests — Implement measurement-noise composition (`test_measurement_composition.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.config_schema import NoiseContribution
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


def _mp(
    gid: str,
    qx: list[float],
    qz: list[float],
    sticky: float = 0.10,
    dropout: float = 0.02,
    delay: int = 1,
) -> MeasurementNoiseParameters:
    return MeasurementNoiseParameters(
        q_base_x=np.array(qx, dtype=np.float64),
        q_base_z=np.array(qz, dtype=np.float64),
        temporal_sticky_prob=sticky,
        detector_dropout_prob=dropout,
        observation_delay_samples=delay,
        generator_id=gid,
        profile_id="PD_02",
        sample_id=3,
        timestamp=0.3,
        window_id=0,
    )


def test_single_generator_produces_unchanged_contribution() -> None:
    mp1 = _mp("measurement_noise_01", [0.01, 0.02, 0.03, 0.04], [0.02, 0.01, 0.04, 0.03])
    composer = NoiseComposer()
    comp = composer.compose([mp1], timestamp=0.3, sample_id=3)
    out = comp.composite_parameters["measurement"]
    assert np.allclose(out.q_base_x, mp1.q_base_x)
    assert np.allclose(out.q_base_z, mp1.q_base_z)
    assert abs(out.temporal_sticky_prob - mp1.temporal_sticky_prob) < 1e-12
    assert abs(out.detector_dropout_prob - mp1.detector_dropout_prob) < 1e-12
    assert out.observation_delay_samples == mp1.observation_delay_samples


def test_independent_sources_combine_correctly_and_reproducibly_with_provenance() -> None:
    mp1 = _mp("measurement_noise_01", [0.10, 0.20, 0.05, 0.00], [0.05, 0.05, 0.05, 0.05], 0.10, 0.02, 1)
    mp2 = _mp("measurement_noise_02", [0.20, 0.10, 0.05, 0.15], [0.10, 0.10, 0.10, 0.10], 0.20, 0.03, 2)

    composer = NoiseComposer()
    comp_a = composer.compose([mp1, mp2], timestamp=0.3, sample_id=3)
    comp_b = composer.compose([mp1, mp2], timestamp=0.3, sample_id=3)
    out_a = comp_a.composite_parameters["measurement"]
    out_b = comp_b.composite_parameters["measurement"]

    expected_qx = 1.0 - (1.0 - mp1.q_base_x) * (1.0 - mp2.q_base_x)
    expected_qz = 1.0 - (1.0 - mp1.q_base_z) * (1.0 - mp2.q_base_z)
    assert np.allclose(out_a.q_base_x, expected_qx)
    assert np.allclose(out_a.q_base_z, expected_qz)
    assert np.allclose(out_a.to_vector(), out_b.to_vector())
    assert out_a.metadata["source_generators"] == ["measurement_noise_01", "measurement_noise_02"]
    assert len(comp_a.contributions) == 2


def test_identical_source_contributions_not_silently_double_counted() -> None:
    mp1 = _mp("measurement_noise_01", [0.10, 0.20, 0.05, 0.00], [0.05, 0.05, 0.05, 0.05])
    composer = NoiseComposer()
    with pytest.raises(ValueError, match="Duplicate contribution"):
        composer.compose([mp1, mp1], timestamp=0.3, sample_id=3)


def test_incompatible_parameter_dimensions_rejected() -> None:
    bad_contrib = NoiseContribution(
        generator_id="measurement_noise_bad_dim",
        profile_id="PD_01",
        window_id=0,
        sample_id=3,
        timestamp=0.3,
        output_domain="measurement_fault_parameters",
        values=np.zeros(7, dtype=np.float64),
    )
    composer = NoiseComposer()
    with pytest.raises(ValueError, match="shape"):
        composer.compose([bad_contrib], timestamp=0.3, sample_id=3)


def test_invalid_probability_values_rejected() -> None:
    bad_vals = np.array([1.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.01, 0.0], dtype=np.float64)
    bad_contrib = NoiseContribution(
        generator_id="measurement_noise_bad_prob",
        profile_id="PD_01",
        window_id=0,
        sample_id=3,
        timestamp=0.3,
        output_domain="measurement_fault_parameters",
        values=bad_vals,
    )
    composer = NoiseComposer()
    with pytest.raises(ValueError):
        composer.compose([bad_contrib], timestamp=0.3, sample_id=3)
