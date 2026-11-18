"""Posterior Predictive Syndrome Distribution Checks (`posterior_predictive.py`)."""

from __future__ import annotations

from typing import Dict
import numpy as np

from layer3.forward_model.syndrome_likelihood import exact_syndrome_probabilities


def compute_posterior_predictive_checks(
    observed_hist_16: np.ndarray,
    posterior_mean_21: np.ndarray,
    basis_checks: str = "X",
    readout_flip_prob: float = 0.0,
) -> Dict[str, float]:
    """Compute TVD and KL residual between empirical syndrome histogram and posterior predictive."""
    f_obs = np.asarray(observed_hist_16, dtype=np.float64)
    p_pred = exact_syndrome_probabilities(
        posterior_mean_21, basis_checks=basis_checks, readout_flip_prob=readout_flip_prob
    )
    tvd = 0.5 * float(np.sum(np.abs(f_obs - p_pred)))
    mask = f_obs > 1.0e-12
    kl = float(
        np.sum(
            f_obs[mask]
            * np.log(f_obs[mask] / np.maximum(p_pred[mask], 1.0e-12))
        )
    )
    return {
        "tvd": float(np.clip(tvd, 0.0, 1.0)),
        "kl_divergence": float(max(0.0, kl)),
    }
