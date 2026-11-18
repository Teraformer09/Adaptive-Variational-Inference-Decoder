"""Layer 3 Bayesian Posterior & Identifiability Interface Tests (`test_layer3_posterior_interface.py`)."""

from __future__ import annotations

import numpy as np

from layer3.forward_model import JointSyndromeForwardModel, exact_syndrome_probabilities
from layer3.identifiability import (
    analyze_jacobian_svd,
    compute_syndrome_jacobian,
    project_identifiable_and_null_coordinates,
    sample_nullspace_perturbation,
)
from layer3.inference import (
    SequentialPosteriorEstimator,
    compute_posterior_predictive_checks,
    evaluate_uncertainty_calibration,
)
from layer3.interfaces import PosteriorPacket, export_posterior_for_layer2


def test_layer3_analytical_jacobian_vs_finite_differences_and_svd() -> None:
    rng = np.random.default_rng(3001)
    theta = rng.uniform(0.005, 0.045, size=21).astype(np.float64)
    J_ana = compute_syndrome_jacobian(theta, basis_checks="X")

    eps = 1e-6
    J_fd = np.zeros_like(J_ana)
    for m in range(21):
        tp = theta.copy()
        tm = theta.copy()
        tp[m] += eps
        tm[m] -= eps
        J_fd[:, m] = (
            exact_syndrome_probabilities(tp, "X") - exact_syndrome_probabilities(tm, "X")
        ) / (2.0 * eps)

    assert float(np.max(np.abs(J_ana - J_fd))) < 1e-9

    svd = analyze_jacobian_svd(theta, basis_checks="X")
    assert svd["rank"] == 9
    assert svd["nullity"] == 12


def test_layer3_sequential_posterior_packet_export_and_calibration() -> None:
    rng = np.random.default_rng(3002)
    theta_z = np.full(21, 0.012, dtype=np.float64)
    theta_z[4] = 0.052
    theta_z[9 + 3] = 0.076  # ZZ pair (4, 5)
    theta_x = np.full(21, 0.006, dtype=np.float64)

    fwd = JointSyndromeForwardModel()
    _, _, sx, sz = fwd.sample_syndrome_window(rng, theta_z, theta_x, window_rounds=1024)

    est = SequentialPosteriorEstimator(temporal_beta=0.50)
    pkt = est.step_from_syndrome_windows(sx, sz)
    assert isinstance(pkt, PosteriorPacket)

    payload = export_posterior_for_layer2(pkt)
    assert payload["posterior_mean_21"].shape == (21,)
    assert payload["posterior_var_21"].shape == (21,)

    ppc = compute_posterior_predictive_checks(pkt.hist_x_16, pkt.posterior_mean_21, "X")
    assert ppc["tvd"] < 0.05

    cal = evaluate_uncertainty_calibration(theta_z, pkt.posterior_mean_21, pkt.posterior_var_21, "X")
    assert cal["identifiable_orbit_9_rmse"] < 0.03
    assert cal["uncertainty_null_to_obs_ratio"] > 1.2

    proj = project_identifiable_and_null_coordinates(theta_z)
    assert proj["psi_observable"].shape == (9,)
    assert proj["psi_nullspace"].shape == (12,)

    theta_null = sample_nullspace_perturbation(theta_z, rng=rng, scale=0.01)
    assert theta_null.shape == (21,)
