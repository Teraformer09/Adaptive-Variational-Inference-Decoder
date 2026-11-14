"""Step 2 Tests — Implement the measurement parameter data structure (`test_measurement_parameters.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


def _valid_kwargs() -> dict:
    return {
        "q_base_x": np.array([0.01, 0.02, 0.01, 0.02]),
        "q_base_z": np.array([0.02, 0.01, 0.02, 0.01]),
        "temporal_sticky_prob": 0.1,
        "detector_dropout_prob": 0.005,
        "observation_delay_samples": 1,
        "generator_id": "measurement_noise_01",
        "profile_id": "PD_04",
        "sample_id": 10,
        "timestamp": 1.0,
        "window_id": 0,
    }


def test_valid_four_element_vectors() -> None:
    obj = MeasurementNoiseParameters(**_valid_kwargs())
    assert obj.q_base_x.shape == (4,)
    assert obj.q_base_z.shape == (4,)


def test_three_element_x_vector_rejected() -> None:
    kw = _valid_kwargs()
    kw["q_base_x"] = np.array([0.01, 0.02, 0.01])
    with pytest.raises(ValueError, match="q_base_x"):
        MeasurementNoiseParameters(**kw)


def test_three_element_z_vector_rejected() -> None:
    kw = _valid_kwargs()
    kw["q_base_z"] = np.array([0.02, 0.01, 0.02])
    with pytest.raises(ValueError, match="q_base_z"):
        MeasurementNoiseParameters(**kw)


def test_probability_below_zero_rejected() -> None:
    kw = _valid_kwargs()
    kw["q_base_x"] = np.array([-0.01, 0.02, 0.01, 0.02])
    with pytest.raises(ValueError, match="q_base_x"):
        MeasurementNoiseParameters(**kw)

    kw2 = _valid_kwargs()
    kw2["temporal_sticky_prob"] = -0.05
    with pytest.raises(ValueError, match="temporal_sticky_prob"):
        MeasurementNoiseParameters(**kw2)


def test_probability_above_one_rejected() -> None:
    kw = _valid_kwargs()
    kw["q_base_z"] = np.array([0.02, 1.01, 0.02, 0.01])
    with pytest.raises(ValueError, match="q_base_z"):
        MeasurementNoiseParameters(**kw)

    kw2 = _valid_kwargs()
    kw2["detector_dropout_prob"] = 1.2
    with pytest.raises(ValueError, match="detector_dropout_prob"):
        MeasurementNoiseParameters(**kw2)


def test_nan_or_infinity_rejected() -> None:
    kw_nan = _valid_kwargs()
    kw_nan["q_base_x"] = np.array([np.nan, 0.02, 0.01, 0.02])
    with pytest.raises(ValueError, match="finite"):
        MeasurementNoiseParameters(**kw_nan)

    kw_inf = _valid_kwargs()
    kw_inf["temporal_sticky_prob"] = float("inf")
    with pytest.raises(ValueError, match="temporal_sticky_prob"):
        MeasurementNoiseParameters(**kw_inf)


def test_negative_delay_rejected() -> None:
    kw = _valid_kwargs()
    kw["observation_delay_samples"] = -1
    with pytest.raises(ValueError, match="observation_delay_samples"):
        MeasurementNoiseParameters(**kw)


def test_floating_point_delay_rejected() -> None:
    kw = _valid_kwargs()
    kw["observation_delay_samples"] = 1.5  # type: ignore[assignment]
    with pytest.raises(ValueError, match="observation_delay_samples"):
        MeasurementNoiseParameters(**kw)


def test_missing_generator_id_rejected() -> None:
    kw = _valid_kwargs()
    kw["generator_id"] = ""
    with pytest.raises(ValueError, match="generator_id"):
        MeasurementNoiseParameters(**kw)


def test_valid_parameter_object() -> None:
    obj = MeasurementNoiseParameters(**_valid_kwargs())
    assert obj.generator_id == "measurement_noise_01"
    assert obj.profile_id == "PD_04"
    assert obj.sample_id == 10
    assert obj.timestamp == 1.0
    assert obj.window_id == 0
    assert obj.observation_delay_samples == 1
    assert obj.to_vector().shape == (11,)
