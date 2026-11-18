"""Integration Test: Layer 3 Sequential Posterior -> Layer 2 82-D Observation Interface."""

from __future__ import annotations

import numpy as np

from layer3.forward_model.joint_syndrome_model import JointSyndromeForwardModel
from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from layer3.interfaces.layer2_observation_export import export_posterior_for_layer2
from training.observations.observation_builder import ObservationBuilder
from training.observations.observation_schema import (
    SLICE_POSTERIOR_MEAN,
    SLICE_TELEMETRY,
    validate_observation_v2,
)


def test_multi_epoch_layer3_to_layer2_sequential_interface() -> None:
    """Verify multi-epoch OU drift + burst tracking across the Layer 3 -> Layer 2 boundary."""
    rng = np.random.default_rng(202607)
    fwd = JointSyndromeForwardModel()
    estimator = SequentialPosteriorEstimator(temporal_beta=0.50)
    builder = ObservationBuilder(max_config_age_epochs=16)

    obs_history = []
    for ep in range(8):
        theta_z = np.full(21, 0.010 + 0.004 * ep, dtype=np.float64)
        if ep >= 5:
            theta_z[9 + 3] = 0.085  # Burst on ZZ pair (4, 5)
        theta_x = np.full(21, 0.006, dtype=np.float64)

        _, _, sx, sz = fwd.sample_syndrome_window(rng, theta_z, theta_x, window_rounds=512)
        recal = ep == 5
        pkt = estimator.step_from_syndrome_windows(sx, sz, recalibrate=recal)
        exported = export_posterior_for_layer2(pkt)

        built = builder.build_from_posterior_and_syndromes(
            exported, syndromes_x=sx, syndromes_z=sz
        )
        builder.record_action_update(config_changed=(ep in (0, 5)))
        obs = built["observation_82"]
        validate_observation_v2(obs)
        obs_history.append(obs)

    obs_arr = np.stack(obs_history, axis=0)
    assert obs_arr.shape == (8, 82)
    # Posterior mean norm and X syndrome density should increase from epoch 0 to epoch 7
    assert float(np.mean(obs_arr[7, SLICE_POSTERIOR_MEAN])) > float(
        np.mean(obs_arr[0, SLICE_POSTERIOR_MEAN])
    )
    assert obs_arr[7, SLICE_TELEMETRY][0] > obs_arr[0, SLICE_TELEMETRY][0]
