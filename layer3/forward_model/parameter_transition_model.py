"""Logit-Space Temporal Parameter Transition Model (`parameter_transition_model.py`)."""

from __future__ import annotations

from typing import Tuple
import numpy as np


def prob_to_logit(theta: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
    p = np.clip(np.asarray(theta, dtype=np.float64), eps, 0.495)
    return np.log(p / (1.0 - p))


def logit_to_prob(z: np.ndarray) -> np.ndarray:
    zc = np.clip(np.asarray(z, dtype=np.float64), -14.0, -0.02)
    return 1.0 / (1.0 + np.exp(-zc))


class ParameterTransitionPrior:
    """Logit-Gaussian state transition prior p(z_t | z_{t-1}) for sequential Bayesian filtering."""

    def __init__(
        self,
        base_single_prob: float = 0.012,
        base_pair_prob: float = 0.005,
        ar_persistence: float = 0.72,
        process_logit_std: float = 0.45,
        stationary_logit_std: float = 0.68,
    ) -> None:
        base_theta = np.empty(21, dtype=np.float64)
        base_theta[:9] = float(base_single_prob)
        base_theta[9:] = float(base_pair_prob)
        self.anchor_logit = prob_to_logit(base_theta)
        self.ar_persistence = float(np.clip(ar_persistence, 0.0, 0.99))
        self.process_logit_var = float(process_logit_std) ** 2
        self.stationary_logit_var = float(stationary_logit_std) ** 2

    def predict_prior(
        self,
        prev_mean_logit: np.ndarray | None = None,
        prev_var_logit: np.ndarray | None = None,
        reset_prior: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict `(prior_mean_logit[21], prior_var_logit[21])` for the current window."""
        if reset_prior or prev_mean_logit is None or prev_var_logit is None:
            return (
                self.anchor_logit.copy(),
                np.full(21, self.stationary_logit_var, dtype=np.float64),
            )
        a = self.ar_persistence
        mean_pred = a * prev_mean_logit + (1.0 - a) * self.anchor_logit
        var_pred = np.minimum(
            (a * a) * prev_var_logit + self.process_logit_var,
            self.stationary_logit_var * 1.35,
        )
        return mean_pred, var_pred
