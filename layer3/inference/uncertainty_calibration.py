"""Posterior Uncertainty Calibration & Subspace Diagnostics (`uncertainty_calibration.py`)."""

from __future__ import annotations

from typing import Dict
import numpy as np

from layer3.identifiability.svd_analysis import analyze_jacobian_svd


def evaluate_uncertainty_calibration(
    true_theta_21: np.ndarray,
    posterior_mean_21: np.ndarray,
    posterior_var_21: np.ndarray,
    basis_checks: str = "X",
) -> Dict[str, float]:
    """Evaluate standardized z-scores, observable-orbit RMSE, and nullspace vs observable variance ratio."""
    th_true = np.asarray(true_theta_21, dtype=np.float64)
    th_hat = np.asarray(posterior_mean_21, dtype=np.float64)
    var_hat = np.maximum(np.asarray(posterior_var_21, dtype=np.float64), 1.0e-12)
    std_hat = np.sqrt(var_hat)

    abs_err = np.abs(th_hat - th_true)
    z_scores = abs_err / std_hat
    within_2sigma = float(np.mean(z_scores <= 2.0))

    svd_info = analyze_jacobian_svd(th_hat, basis_checks=basis_checks)
    V_obs = np.asarray(svd_info["V_observable"], dtype=np.float64)
    leverage = np.asarray(svd_info["parameter_observability_leverage"], dtype=np.float64)

    orbit_err = V_obs.T @ (th_hat - th_true)
    orbit_rmse = float(np.sqrt(np.mean(orbit_err ** 2)))
    param_rmse = float(np.sqrt(np.mean((th_hat - th_true) ** 2)))

    high_obs_mask = leverage >= 0.45
    low_obs_mask = leverage < 0.25
    rel_cv = std_hat / np.maximum(th_hat, 1.0e-5)
    mean_std_obs = float(np.mean(std_hat[high_obs_mask])) if np.any(high_obs_mask) else float(np.mean(std_hat))
    mean_std_null = float(np.mean(std_hat[low_obs_mask])) if np.any(low_obs_mask) else float(np.mean(std_hat))
    mean_cv_obs = float(np.mean(rel_cv[high_obs_mask])) if np.any(high_obs_mask) else float(np.mean(rel_cv))
    mean_cv_null = float(np.mean(rel_cv[low_obs_mask])) if np.any(low_obs_mask) else float(np.mean(rel_cv))

    return {
        "param_21_rmse": param_rmse,
        "identifiable_orbit_9_rmse": orbit_rmse,
        "coverage_within_2sigma": within_2sigma,
        "mean_std_observable_params": mean_std_obs,
        "mean_std_unobservable_params": mean_std_null,
        "mean_cv_observable_params": mean_cv_obs,
        "mean_cv_unobservable_params": mean_cv_null,
        "uncertainty_null_to_obs_ratio": mean_cv_null / max(mean_cv_obs, 1.0e-9),
    }
