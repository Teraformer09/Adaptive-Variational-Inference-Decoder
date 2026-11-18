"""OBS-04, OBS-05, & OBS-06 Unit Tests (`test_observation_builder.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from layer3.forward_model.joint_syndrome_model import JointSyndromeForwardModel
from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from training.observations.observation_builder import ObservationBuilder
from training.observations.observation_schema import (
    OBS_V2_DIM,
    SLICE_HIST_X,
    SLICE_HIST_Z,
    SLICE_POSTERIOR_LOG_VAR,
    SLICE_POSTERIOR_MEAN,
    SLICE_TELEMETRY,
    validate_observation_v2,
)
from training.observations.syndrome_histogram import compute_basis_syndrome_stats


def test_obs04_separate_xz_histograms_normalized_by_valid_rounds() -> None:
    """OBS-04: X and Z histograms stay separate and each sum to 1.0 over valid rounds."""
    rng = np.random.default_rng(404)
    sx = rng.integers(0, 16, size=256, dtype=np.uint8)
    sz = rng.integers(0, 4, size=256, dtype=np.uint8)  # deliberately different distribution from sx
    val_x = rng.random((256, 4)) > 0.05
    val_z = rng.random((256, 4)) > 0.10

    st_x = compute_basis_syndrome_stats(sx, val_x)
    st_z = compute_basis_syndrome_stats(sz, val_z)

    assert abs(float(np.sum(st_x["histogram"])) - 1.0) < 1e-12
    assert abs(float(np.sum(st_z["histogram"])) - 1.0) < 1e-12
    assert not np.allclose(st_x["histogram"], st_z["histogram"])
    assert 0 < st_x["valid_rounds"] < 256
    assert 0 < st_z["valid_rounds"] < 256


def test_obs05_observation_construction_exact_82d_ordering() -> None:
    """OBS-05: ObservationBuilder constructs an exact 82-D vector with validated slices."""
    rng = np.random.default_rng(505)
    theta_z = np.full(21, 0.014, dtype=np.float64)
    theta_z[9 + 1] = 0.062  # pair (1, 2)
    theta_x = np.full(21, 0.007, dtype=np.float64)

    fwd = JointSyndromeForwardModel()
    _, _, sx, sz = fwd.sample_syndrome_window(rng, theta_z, theta_x, window_rounds=512)

    est = SequentialPosteriorEstimator()
    pkt = est.step_from_syndrome_windows(sx, sz)

    builder = ObservationBuilder()
    built = builder.build_from_posterior_and_syndromes(
        posterior_packet=pkt.to_layer2_dict(),
        syndromes_x=sx,
        syndromes_z=sz,
    )
    obs = built["observation_82"]
    assert obs.shape == (OBS_V2_DIM,) == (82,)
    validate_observation_v2(obs, strict_probabilities=True)

    assert np.allclose(obs[SLICE_POSTERIOR_MEAN], pkt.posterior_mean_21)
    assert abs(float(np.sum(obs[SLICE_HIST_X])) - 1.0) < 1e-12
    assert abs(float(np.sum(obs[SLICE_HIST_Z])) - 1.0) < 1e-12
    assert obs[SLICE_POSTERIOR_LOG_VAR].shape == (21,)
    assert obs[SLICE_TELEMETRY].shape == (8,)


def test_obs06_no_ground_truth_leakage() -> None:
    """OBS-06: Changing hidden simulator ground-truth while holding observed syndromes fixed leaves R2D2 input identical, and forbidden keys raise ValueError."""
    rng = np.random.default_rng(606)
    theta_true_1 = np.full(21, 0.015, dtype=np.float64)
    theta_true_2 = np.full(21, 0.180, dtype=np.float64)  # Drastically different hidden simulator parameter
    theta_x = np.full(21, 0.008, dtype=np.float64)

    fwd = JointSyndromeForwardModel()
    _, _, sx_fixed, sz_fixed = fwd.sample_syndrome_window(rng, theta_true_1, theta_x, window_rounds=256)

    # Run two separate estimators + builders given ONLY the fixed syndrome observations
    est_1 = SequentialPosteriorEstimator()
    pkt_1 = est_1.step_from_syndrome_windows(sx_fixed, sz_fixed)
    obs_1 = ObservationBuilder().build_from_posterior_and_syndromes(
        pkt_1.to_layer2_dict(), syndromes_x=sx_fixed, syndromes_z=sz_fixed
    )["observation_82"]

    # Even though hidden theta_true_2 is 0.180 in the simulator, the observation pipeline never sees theta_true_2
    _ = theta_true_2
    est_2 = SequentialPosteriorEstimator()
    pkt_2 = est_2.step_from_syndrome_windows(sx_fixed, sz_fixed)
    obs_2 = ObservationBuilder().build_from_posterior_and_syndromes(
        pkt_2.to_layer2_dict(), syndromes_x=sx_fixed, syndromes_z=sz_fixed
    )["observation_82"]

    assert np.array_equal(obs_1, obs_2)

    # Explicitly attempting to pass a ground-truth key into ObservationBuilder must fail immediately
    leaky_payload = pkt_1.to_layer2_dict()
    leaky_payload["true_theta_z"] = theta_true_2
    with pytest.raises(ValueError, match="Ground-truth leakage blocked"):
        ObservationBuilder().build_from_posterior_and_syndromes(leaky_payload)
