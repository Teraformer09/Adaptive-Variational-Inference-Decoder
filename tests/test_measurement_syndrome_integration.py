"""Step 11 — Measurement Noise and Syndrome Generation Integration Tests."""

from __future__ import annotations

import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.noise_framework.config_schema import (
    CompositeNoiseSample,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)
from simulation.stimulus.noise_framework.syndrome.syndrome_encoder import SyndromeEncoder


def _make_composite_samples(
    n_samples: int,
    meas_params_fn,
    theta_z_val: float = 0.08,
    theta_x_val: float = 0.08,
):
    samples = []
    for k in range(n_samples):
        t = round(k * 0.1, 6)
        w = k // 1000
        mp = meas_params_fn(k, t, w)
        contrib = NoiseContribution(
            generator_id="measurement_noise_01",
            profile_id=mp.profile_id,
            window_id=w,
            sample_id=k,
            timestamp=t,
            output_domain="measurement_fault_parameters",
            values=mp.to_vector(),
            metadata={"measurement_parameters": mp},
        )
        comp = CompositeNoiseSample(
            sample_id=k,
            timestamp=t,
            window_id=w,
            contributions=[contrib],
            composite_parameters={
                "theta_z": np.full(21, theta_z_val, dtype=np.float64),
                "theta_x": np.full(21, theta_x_val, dtype=np.float64),
                "measurement": mp,
            },
            metadata={"profile_assignments": {"measurement_noise_01": mp.profile_id}},
        )
        samples.append(comp)
    return samples


def test_trajectory_contains_all_required_step11_keys():
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    samples = _make_composite_samples(
        10,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=0.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_15",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj = encoder.encode_composite_trajectory(samples, master_seed=42)
    required_keys = {
        "physical_faults",
        "ideal_syndromes",
        "measurement_faults",
        "observed_syndromes",
        "validity_masks",
        "event_timestamps",
        "delivery_timestamps",
    }
    assert required_keys.issubset(set(traj.keys()))


def test_zero_measurement_noise_produces_ideal_observations():
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    samples = _make_composite_samples(
        40,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=0.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_15",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj = encoder.encode_composite_trajectory(samples, master_seed=101)
    np.testing.assert_array_equal(
        traj["observed_syndromes"]["x"], traj["ideal_syndromes"]["x"]
    )
    np.testing.assert_array_equal(
        traj["observed_syndromes"]["z"], traj["ideal_syndromes"]["z"]
    )
    assert np.all(traj["measurement_faults"]["x"] == 0)
    assert np.all(traj["measurement_faults"]["z"] == 0)
    assert np.all(traj["validity_masks"]["x"] == 1)
    assert np.all(traj["validity_masks"]["z"] == 1)


def test_measurement_noise_changes_observations_without_changing_physical_faults():
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    noiseless_samples = _make_composite_samples(
        60,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=0.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_15",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    noisy_samples = _make_composite_samples(
        60,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.full(4, 0.35),
            q_base_z=np.full(4, 0.35),
            temporal_sticky_prob=0.25,
            detector_dropout_prob=0.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_16",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj_clean = encoder.encode_composite_trajectory(
        noiseless_samples, master_seed=777, trajectory_id="same_traj"
    )
    traj_noisy = encoder.encode_composite_trajectory(
        noisy_samples, master_seed=777, trajectory_id="same_traj"
    )

    np.testing.assert_array_equal(
        traj_clean["physical_faults"]["x"], traj_noisy["physical_faults"]["x"]
    )
    np.testing.assert_array_equal(
        traj_clean["physical_faults"]["z"], traj_noisy["physical_faults"]["z"]
    )
    np.testing.assert_array_equal(
        traj_clean["ideal_syndromes"]["x"], traj_noisy["ideal_syndromes"]["x"]
    )
    np.testing.assert_array_equal(
        traj_clean["ideal_syndromes"]["z"], traj_noisy["ideal_syndromes"]["z"]
    )

    assert not np.array_equal(
        traj_clean["observed_syndromes"]["x"], traj_noisy["observed_syndromes"]["x"]
    )
    np.testing.assert_array_equal(
        traj_noisy["observed_syndromes"]["x"],
        traj_noisy["ideal_syndromes"]["x"] ^ traj_noisy["measurement_faults"]["x"],
    )
    np.testing.assert_array_equal(
        traj_noisy["observed_syndromes"]["z"],
        traj_noisy["ideal_syndromes"]["z"] ^ traj_noisy["measurement_faults"]["z"],
    )


def test_syndrome_calculations_match_canonical_css_matrices():
    code = build_rotated_d3_code()
    hx = np.asarray(code.hx, dtype=np.uint8)
    hz = np.asarray(code.hz, dtype=np.uint8)
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    samples = _make_composite_samples(
        30,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=0.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_15",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj = encoder.encode_composite_trajectory(samples, master_seed=55)
    expected_sx = (traj["physical_faults"]["z"] @ hx.T) & 1
    expected_sz = (traj["physical_faults"]["x"] @ hz.T) & 1
    np.testing.assert_array_equal(traj["ideal_syndromes"]["x"], expected_sx)
    np.testing.assert_array_equal(traj["ideal_syndromes"]["z"], expected_sz)


def test_dropout_represented_by_invalidity_not_valid_zero_bit():
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    samples = _make_composite_samples(
        25,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=1.0,
            observation_delay_samples=0,
            generator_id="measurement_noise_01",
            profile_id="PD_16",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj = encoder.encode_composite_trajectory(samples, master_seed=99)
    assert np.all(traj["validity_masks"]["x"] == 0)
    assert np.all(traj["validity_masks"]["z"] == 0)
    assert np.all(traj["observed_syndromes"]["x"] == 0)
    assert np.all(traj["observed_syndromes"]["z"] == 0)


def test_delayed_observations_retain_original_event_identity():
    encoder = SampleEncoder(rounds_per_composite_sample=1, sampling_interval_sec=0.1)
    samples = _make_composite_samples(
        15,
        lambda k, t, w: MeasurementNoiseParameters(
            q_base_x=np.zeros(4),
            q_base_z=np.zeros(4),
            temporal_sticky_prob=0.0,
            detector_dropout_prob=0.0,
            observation_delay_samples=2,
            generator_id="measurement_noise_01",
            profile_id="PD_12",
            sample_id=k,
            timestamp=t,
            window_id=w,
        ),
    )
    traj = encoder.encode_composite_trajectory(samples, master_seed=123)
    np.testing.assert_allclose(traj["event_timestamps"], traj["timestamps"], atol=1e-9)
    np.testing.assert_allclose(
        traj["delivery_timestamps"] - traj["event_timestamps"], 0.2, atol=1e-9
    )
    np.testing.assert_array_equal(traj["sample_ids"], np.arange(15, dtype=np.int32))


def test_xz_basis_separation_is_preserved():
    syn_enc = SyndromeEncoder()
    single_q = np.zeros((5, 9), dtype=np.uint8)
    single_q[:, 4] = 1
    zero_q = np.zeros((5, 9), dtype=np.uint8)

    sx, sz = syn_enc.encode(x_errors=zero_q, z_errors=single_q)
    assert np.any(sx == 1)
    assert np.all(sz == 0)

    sx2, sz2 = syn_enc.encode(x_errors=single_q, z_errors=zero_q)
    assert np.all(sx2 == 0)
    assert np.any(sz2 == 1)
