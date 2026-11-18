"""OBS-03 & Identifiability Boundary Tests (`test_posterior_contract.py`).

Verifies:
  - OBS-03: Posterior uncertainty shape, finiteness, positive variance, and log-variance transform.
  - Jacobian SVD rank == 9 and nullity == 12 on the 21-parameter rotated d=3 forward model.
  - Nullspace identifiability boundary: Two distinct 21-parameter states differing along ker(J)
    produce statistically indistinguishable syndrome distributions and high posterior uncertainty
    along the unobservable directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from layer3.forward_model.joint_syndrome_model import JointSyndromeForwardModel
from layer3.forward_model.syndrome_likelihood import exact_syndrome_probabilities
from layer3.identifiability.identifiable_subspace import sample_nullspace_perturbation
from layer3.identifiability.svd_analysis import analyze_jacobian_svd
from layer3.inference.posterior_estimator import SequentialPosteriorEstimator
from layer3.inference.uncertainty_calibration import evaluate_uncertainty_calibration
from training.observations.observation_schema import variance_to_log_uncertainty


def test_obs03_posterior_uncertainty_validity_and_window_scaling() -> None:
    """OBS-03: Posterior variances are finite, positive, and shrink on observable modes with larger W."""
    rng = np.random.default_rng(202601)
    theta_z = np.full(21, 0.012, dtype=np.float64)
    theta_z[4] = 0.055
    theta_z[9 + 3] = 0.068  # pair (4, 5)
    theta_x = np.full(21, 0.006, dtype=np.float64)

    fwd = JointSyndromeForwardModel()
    _, _, sx_small, sz_small = fwd.sample_syndrome_window(rng, theta_z, theta_x, window_rounds=64)
    _, _, sx_large, sz_large = fwd.sample_syndrome_window(rng, theta_z, theta_x, window_rounds=2048)

    est_small = SequentialPosteriorEstimator(temporal_beta=0.0)
    pkt_small = est_small.step_from_syndrome_windows(sx_small, sz_small)

    est_large = SequentialPosteriorEstimator(temporal_beta=0.0)
    pkt_large = est_large.step_from_syndrome_windows(sx_large, sz_large)

    assert pkt_small.posterior_mean_21.shape == (21,)
    assert pkt_small.posterior_var_21.shape == (21,)
    assert np.all(np.isfinite(pkt_small.posterior_var_21))
    assert np.all(pkt_small.posterior_var_21 > 0.0)

    log_unc_small = variance_to_log_uncertainty(pkt_small.posterior_var_21)
    log_unc_large = variance_to_log_uncertainty(pkt_large.posterior_var_21)
    assert log_unc_small.shape == (21,)
    assert np.all(np.isfinite(log_unc_small))

    # Observable unique-signature qubit q4 (index 4, signature 0x6) must have strictly lower variance at W=2048 than W=64
    assert pkt_large.posterior_var_21[4] < pkt_small.posterior_var_21[4] * 0.45
    assert log_unc_large[4] < log_unc_small[4]

    # Unobservable stabilizer-trivial pair (3,6) at index 16 (signature 0x0) must retain higher uncertainty than observable q4
    cal = evaluate_uncertainty_calibration(
        true_theta_21=theta_z,
        posterior_mean_21=pkt_large.posterior_mean_21,
        posterior_var_21=pkt_large.posterior_var_21,
    )
    assert cal["uncertainty_null_to_obs_ratio"] > 1.25

    with pytest.raises(ValueError):
        variance_to_log_uncertainty(np.full(21, -0.01))


def test_identifiability_rank9_nullity12_and_nullspace_ambiguity() -> None:
    """Verify rank(J)=9, dim ker(J)=12, and that nullspace perturbations remain appropriately uncertain."""
    rng = np.random.default_rng(202602)
    theta_a = np.full(21, 0.018, dtype=np.float64)
    theta_a[4] = 0.042
    theta_a[9 + 3] = 0.050

    svd = analyze_jacobian_svd(theta_a, basis_checks="X")
    assert svd["rank"] == 9
    assert svd["nullity"] == 12

    # Perturb along exact nullspace direction (e.g., stabilizer-equivalent pair index 16 or linear null vector)
    theta_b = sample_nullspace_perturbation(theta_a, rng=rng, scale=0.012, basis_checks="X")
    assert float(np.linalg.norm(theta_b - theta_a)) > 0.008

    p_a = exact_syndrome_probabilities(theta_a, basis_checks="X")
    p_b = exact_syndrome_probabilities(theta_b, basis_checks="X")
    tvd_ab = 0.5 * float(np.sum(np.abs(p_a - p_b)))
    # First-order nullspace perturbation keeps syndrome TVD tiny (< 0.002)
    assert tvd_ab < 0.002

    # Run Layer 3 estimator on high-count histograms from theta_a and theta_b
    hz = exact_syndrome_probabilities(np.full(21, 0.006), basis_checks="Z")
    est_a = SequentialPosteriorEstimator(temporal_beta=0.0)
    pkt_a = est_a.step_from_histograms(p_a, hz, valid_rounds_x=4096, valid_rounds_z=4096)

    est_b = SequentialPosteriorEstimator(temporal_beta=0.0)
    pkt_b = est_b.step_from_histograms(p_b, hz, valid_rounds_x=4096, valid_rounds_z=4096)

    # Posterior means must overlap closely (difference << physical separation ||theta_b - theta_a||),
    # preventing R2D2 from falsely treating unobservable nullspace shifts as distinct measurements!
    post_diff = float(np.linalg.norm(pkt_a.posterior_mean_21 - pkt_b.posterior_mean_21))
    true_diff = float(np.linalg.norm(theta_a - theta_b))
    assert post_diff < 0.25 * true_diff
