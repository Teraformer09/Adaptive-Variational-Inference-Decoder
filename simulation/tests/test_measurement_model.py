"""Step 8 Tests — Implement the measurement observation model (`test_measurement_model.py`)."""

from __future__ import annotations

import numpy as np

from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


def _params(
    qx: float = 0.0,
    qz: float = 0.0,
    sticky: float = 0.0,
    dropout: float = 0.0,
    delay: int = 0,
    sample_id: int = 0,
    timestamp: float = 0.0,
) -> MeasurementNoiseParameters:
    return MeasurementNoiseParameters(
        q_base_x=np.full(4, qx, dtype=np.float64),
        q_base_z=np.full(4, qz, dtype=np.float64),
        temporal_sticky_prob=sticky,
        detector_dropout_prob=dropout,
        observation_delay_samples=delay,
        generator_id="measurement_noise_01",
        profile_id="PD_01",
        sample_id=sample_id,
        timestamp=timestamp,
        window_id=0,
    )


def test_zero_readout_probability_produces_no_flips() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(42)
    ideal_x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    ideal_z = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)
    obs = model.apply_trajectory(
        ideal_x, ideal_z, [_params(sample_id=0, timestamp=0.0), _params(sample_id=1, timestamp=0.1)], rng
    )
    assert np.all(obs.measurement_error_mask_x == 0)
    assert np.all(obs.measurement_error_mask_z == 0)
    assert np.array_equal(obs.observed_syndrome_x, ideal_x)
    assert np.array_equal(obs.observed_syndrome_z, ideal_z)


def test_unit_readout_probability_flips_every_valid_bit() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(42)
    ideal_x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    ideal_z = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)
    obs = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [
            _params(qx=1.0, qz=1.0, sample_id=0, timestamp=0.0),
            _params(qx=1.0, qz=1.0, sample_id=1, timestamp=0.1),
        ],
        rng,
    )
    assert np.all(obs.measurement_error_mask_x == 1)
    assert np.all(obs.measurement_error_mask_z == 1)
    assert np.array_equal(obs.observed_syndrome_x, 1 - ideal_x)
    assert np.array_equal(obs.observed_syndrome_z, 1 - ideal_z)


def test_zero_and_unit_dropout_probability_and_validity_masks() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(42)
    ideal_x = np.ones((4, 4), dtype=np.uint8)
    ideal_z = np.ones((4, 4), dtype=np.uint8)

    obs_zero = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [_params(dropout=0.0, sample_id=i, timestamp=0.1 * i) for i in range(4)],
        rng,
    )
    assert np.all(obs_zero.validity_mask_x == 1)
    assert np.all(obs_zero.validity_mask_z == 1)

    obs_unit = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [_params(dropout=1.0, sample_id=i, timestamp=0.1 * i) for i in range(4)],
        rng,
    )
    assert np.all(obs_unit.validity_mask_x == 0)
    assert np.all(obs_unit.validity_mask_z == 0)


def test_delay_zero_and_delay_one_preserve_event_and_delivery_timestamps() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(42)
    ideal_x = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    ideal_z = np.zeros_like(ideal_x)

    obs0 = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [_params(delay=0, sample_id=i, timestamp=0.1 * i) for i in range(3)],
        rng,
    )
    assert np.allclose(obs0.event_timestamps, obs0.delivery_timestamps)

    obs1 = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [_params(delay=1, sample_id=i, timestamp=0.1 * i) for i in range(3)],
        rng,
    )
    assert np.allclose(obs1.delivery_timestamps - obs1.event_timestamps, 0.1)
    assert np.array_equal(obs1.observed_syndrome_x[1], ideal_x[0])
    assert np.array_equal(obs1.observed_syndrome_x[2], ideal_x[1])


def test_temporal_persistence_matches_markov_transition_model() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(123)
    n = 3000
    ideal = np.zeros((n, 4), dtype=np.uint8)
    obs = model.apply_trajectory(
        ideal,
        ideal,
        [_params(qx=0.05, qz=0.05, sticky=0.50, sample_id=i, timestamp=0.1 * i) for i in range(n)],
        rng,
    )
    mx = obs.measurement_error_mask_x
    prev_1 = mx[:-1] == 1
    next_1 = mx[1:] == 1
    emp_p11 = float(np.sum(prev_1 & next_1) / max(1, np.sum(prev_1)))
    theo_p11 = (1.0 - 0.50) * 0.05 + 0.50  # 0.525
    assert abs(emp_p11 - theo_p11) < 0.05


def test_input_immutability() -> None:
    model = MeasurementModel(sampling_interval_sec=0.1)
    rng = np.random.default_rng(42)
    ideal_x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    ideal_z = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)
    copy_x = ideal_x.copy()
    copy_z = ideal_z.copy()
    _ = model.apply_trajectory(
        ideal_x,
        ideal_z,
        [
            _params(qx=0.9, qz=0.9, dropout=0.5, delay=1, sample_id=0, timestamp=0.0),
            _params(qx=0.9, qz=0.9, dropout=0.5, delay=1, sample_id=1, timestamp=0.1),
        ],
        rng,
    )
    assert np.array_equal(ideal_x, copy_x)
    assert np.array_equal(ideal_z, copy_z)
