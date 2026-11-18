"""Logit-Space Variational / Laplace Posterior Head (`posterior_head.py`)."""

from __future__ import annotations

import math
from typing import Dict
import numpy as np

from layer3.forward_model.parameter_transition_model import logit_to_prob
from layer3.forward_model.syndrome_likelihood import (
    CROSSTALK_PAIRS_12,
    exact_syndrome_probabilities,
)
from layer3.identifiability.sensitivity_jacobian import compute_syndrome_jacobian


class LogitLaplacePosteriorHead:
    """Computes posterior mean and variance over theta in [0, 0.49]^21 via logit-space Laplace inference."""

    def __init__(
        self,
        max_iters: int = 10,
        damping: float = 1.0e-3,
        readout_flip_prob: float = 0.0,
    ) -> None:
        self.max_iters = int(max_iters)
        self.damping = float(damping)
        self.readout_flip_prob = float(readout_flip_prob)

    def infer_posterior(
        self,
        hist_16: np.ndarray,
        effective_rounds: float,
        prior_mean_logit: np.ndarray,
        prior_var_logit: np.ndarray,
        basis_checks: str = "X",
    ) -> Dict[str, np.ndarray | float]:
        """Solve for MAP logit mode z_hat and Laplace covariance Sigma_z = H_z^{-1}, then map to theta."""
        f_obs = np.asarray(hist_16, dtype=np.float64)
        w_eff = float(max(0.0, effective_rounds))
        mu_0 = np.asarray(prior_mean_logit, dtype=np.float64)
        var_0 = np.maximum(np.asarray(prior_var_logit, dtype=np.float64), 1.0e-4)
        prec_0 = 1.0 / var_0

        z = mu_0.copy()
        if w_eff > 0.5:
            for _ in range(self.max_iters):
                theta = np.clip(logit_to_prob(z), 1.0e-5, 0.485)
                dtheta_dz = theta * (1.0 - theta)

                probs = exact_syndrome_probabilities(
                    theta, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
                )
                J_theta = compute_syndrome_jacobian(
                    theta, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
                )
                J_z = J_theta * dtheta_dz[None, :]  # (16, 21)

                inv_p = 1.0 / np.maximum(probs, 1.0e-9)
                grad_ll = -w_eff * (J_z.T @ (f_obs * inv_p))
                grad = grad_ll + prec_0 * (z - mu_0)

                H_z = (
                    w_eff * (J_z.T @ (inv_p[:, None] * J_z))
                    + np.diag(prec_0 + self.damping)
                )
                try:
                    step = -np.linalg.solve(H_z, grad)
                except np.linalg.LinAlgError:
                    step = -grad / ( np.diag(H_z) + 1.0 )

                step_norm = float(np.max(np.abs(step)))
                if step_norm > 1.25:
                    step *= 1.25 / step_norm
                z = np.clip(z + step, -11.5, -0.05)
                if step_norm < 1.0e-4:
                    break

            # Hierarchical Ising/coupler-endpoint prior refinement:
            # A physical ZZ coupler (u, v) elevates its pair syndrome bin (pair_excess > 0) AND
            # correlates with endpoint activity, without falsely firing on isolated single-qubit TLS/OU drift.
            mu_eff = mu_0.copy()
            base_q_logit = -4.41  # logit(0.012)
            for k, (u, v) in enumerate(CROSSTALK_PAIRS_12):
                ex_u = max(0.0, z[u] - base_q_logit)
                ex_v = max(0.0, z[v] - base_q_logit)
                end_joint = math.sqrt(ex_u * ex_v) + 0.25 * (ex_u + ex_v)
                pair_excess = max(0.0, z[9 + k] - mu_0[9 + k])
                if pair_excess > 0.05:
                    mu_eff[9 + k] = min(
                        -1.75,
                        mu_0[9 + k] + 1.35 * pair_excess + 1.15 * min(end_joint, 1.2 * pair_excess + 0.25),
                    )

            for _ in range(4):
                theta = np.clip(logit_to_prob(z), 1.0e-5, 0.485)
                dtheta_dz = theta * (1.0 - theta)
                probs = exact_syndrome_probabilities(
                    theta, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
                )
                J_z = compute_syndrome_jacobian(
                    theta, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
                ) * dtheta_dz[None, :]
                inv_p = 1.0 / np.maximum(probs, 1.0e-9)
                grad = -w_eff * (J_z.T @ (f_obs * inv_p)) + prec_0 * (z - mu_eff)
                H_z = w_eff * (J_z.T @ (inv_p[:, None] * J_z)) + np.diag(prec_0 + self.damping)
                try:
                    step = -np.linalg.solve(H_z, grad)
                except np.linalg.LinAlgError:
                    step = -grad / (np.diag(H_z) + 1.0)
                step_norm = float(np.max(np.abs(step)))
                if step_norm > 1.0:
                    step *= 1.0 / step_norm
                z = np.clip(z + step, -11.5, -0.05)

        theta_mode = np.clip(logit_to_prob(z), 1.0e-5, 0.485)
        dtheta_dz = theta_mode * (1.0 - theta_mode)
        probs_final = exact_syndrome_probabilities(
            theta_mode, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
        )
        J_theta_final = compute_syndrome_jacobian(
            theta_mode, basis_checks=basis_checks, readout_flip_prob=self.readout_flip_prob
        )
        J_z_final = J_theta_final * dtheta_dz[None, :]
        inv_p_final = 1.0 / np.maximum(probs_final, 1.0e-9)

        H_post = (
            w_eff * (J_z_final.T @ (inv_p_final[:, None] * J_z_final))
            + np.diag(prec_0)
        )
        cov_z = np.linalg.inv(H_post + 1.0e-8 * np.eye(21))
        var_z = np.maximum(np.diag(cov_z), 1.0e-8)
        std_z = np.sqrt(var_z)

        # 3-point Gauss-Hermite quadrature for variance; MAP-calibrated logit transform for mean
        rt3 = math.sqrt(3.0)
        p_minus = np.clip(logit_to_prob(z - rt3 * std_z), 1.0e-6, 0.49)
        p_mid = np.clip(logit_to_prob(z), 1.0e-6, 0.49)
        p_plus = np.clip(logit_to_prob(z + rt3 * std_z), 1.0e-6, 0.49)

        quad_mean = (p_minus + 4.0 * p_mid + p_plus) / 6.0
        second_moment = (p_minus ** 2 + 4.0 * (p_mid ** 2) + p_plus ** 2) / 6.0
        post_var = np.maximum(second_moment - quad_mean ** 2, 1.0e-9)
        post_mean = 0.80 * p_mid + 0.20 * quad_mean

        return {
            "posterior_mean_21": post_mean,
            "posterior_var_21": post_var,
            "posterior_mean_logit_21": z,
            "posterior_var_logit_21": var_z,
            "posterior_cov_logit_21x21": cov_z,
            "predicted_syndrome_probs_16": probs_final,
        }
