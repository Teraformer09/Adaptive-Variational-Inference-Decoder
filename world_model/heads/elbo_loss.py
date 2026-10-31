"""Multi-task sequential variational ELBO objective defined in Section 6 of `world_model/README.md`."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModelELBOLoss(nn.Module):
    """Computes L = lambda_obs * L_obs + beta * KL + lambda_aux * L_aux + lambda_forecast * L_forecast."""

    def __init__(
        self,
        lambda_obs: float = 1.0,
        beta_kl: float = 0.05,
        lambda_aux: float = 2.0,
        lambda_forecast: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_obs = lambda_obs
        self.beta_kl = beta_kl
        self.lambda_aux = lambda_aux
        self.lambda_forecast = lambda_forecast

    def forward(
        self,
        model_out: Dict[str, Any],
        obs: Dict[str, torch.Tensor],
        latent_truth: Dict[str, torch.Tensor],
        burn_in_steps: int = 16,
        train_steps: int = 32,
        beta_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total training loss and per-component metrics over `[burn_in_steps : burn_in_steps + train_steps]`."""
        active_end = burn_in_steps + train_steps
        post = model_out["posterior"]
        forecasts = model_out["forecasts"]

        # 1. Observation Likelihood Loss (L_obs)
        target_det = obs["detection_vec"][:, burn_in_steps:active_end]
        l_det = F.binary_cross_entropy_with_logits(post["det_logits"], target_det)

        # Masked calibration telemetry reconstruction (t1_norm and detuning_norm)
        target_cal = obs["telemetry_val"][:, burn_in_steps:active_end, :2]
        mask_cal = obs["telemetry_mask"][:, burn_in_steps:active_end, :2]
        cal_diff = ((post["calib_pred"] - target_cal) ** 2) * mask_cal
        l_cal = cal_diff.sum() / (mask_cal.sum() + 1e-6)
        l_obs = l_det + 0.5 * l_cal

        # 2. Variational Transition KL Divergence
        l_kl = model_out["kl_loss"]

        # 3. Auxiliary Physical Mechanism Supervision Loss (L_aux — training only!)
        gt_leak = latent_truth["z_leak"][:, burn_in_steps:active_end]
        # Weighted BCE for sparse leakage events
        pos_weight_leak = torch.tensor(6.0, device=gt_leak.device)
        l_leak = F.binary_cross_entropy_with_logits(
            post["leak_logits"], gt_leak, pos_weight=pos_weight_leak
        )

        gt_corr_pairs = latent_truth["z_corr_pairs"][:, burn_in_steps:active_end]
        l_corr = F.mse_loss(post["pair_corr_prob"], gt_corr_pairs) * 25.0

        gt_burst_active = latent_truth["z_burst_active"][:, burn_in_steps:active_end]
        pos_weight_burst = torch.tensor(5.0, device=gt_burst_active.device)
        l_burst_cls = F.binary_cross_entropy_with_logits(
            post["burst_logits"], gt_burst_active, pos_weight=pos_weight_burst
        )
        gt_footprint = latent_truth["z_burst_footprint"][:, burn_in_steps:active_end]
        l_footprint = F.mse_loss(post["burst_footprint_pred"], gt_footprint) * 15.0

        gt_drift = latent_truth["z_drift"][:, burn_in_steps:active_end]
        gt_mem = latent_truth["z_memory"][:, burn_in_steps:active_end]
        l_drift_mem = (
            F.mse_loss(post["drift_pred"], gt_drift) * 30.0
            + F.mse_loss(post["memory_pred"], gt_mem) * 30.0
        )

        gt_det_prob = latent_truth["true_detection_prob"][:, burn_in_steps:active_end]
        l_rate_calib = F.mse_loss(post["det_prob"], gt_det_prob) * 10.0

        l_aux = l_leak + l_corr + l_burst_cls + l_footprint + l_drift_mem + l_rate_calib

        # 4. Multi-Horizon Forecast Loss (L_forecast across H in {1, 2, 4, 8})
        l_forecast = torch.tensor(0.0, device=target_det.device)
        if len(forecasts) > 0:
            for h_step, f_pred in forecasts.items():
                f_target_det = obs["detection_vec"][:, burn_in_steps + h_step : active_end + h_step]
                f_gt_prob = latent_truth["true_detection_prob"][:, burn_in_steps + h_step : active_end + h_step]
                f_gt_drift = latent_truth["z_drift"][:, burn_in_steps + h_step : active_end + h_step]
                f_gt_mem = latent_truth["z_memory"][:, burn_in_steps + h_step : active_end + h_step]

                l_h_det = F.binary_cross_entropy_with_logits(f_pred["det_logits"], f_target_det)
                l_h_prob = F.mse_loss(f_pred["det_prob"], f_gt_prob) * 10.0
                l_h_dm = (
                    F.mse_loss(f_pred["drift_pred"], f_gt_drift) * 20.0
                    + F.mse_loss(f_pred["memory_pred"], f_gt_mem) * 20.0
                )
                l_forecast = l_forecast + (l_h_det + l_h_prob + l_h_dm) / len(forecasts)

        total_loss = (
            self.lambda_obs * l_obs
            + (self.beta_kl * beta_scale) * l_kl
            + self.lambda_aux * l_aux
            + self.lambda_forecast * l_forecast
        )

        return {
            "total_loss": total_loss,
            "l_obs": l_obs.detach(),
            "l_kl": l_kl.detach(),
            "l_aux": l_aux.detach(),
            "l_forecast": l_forecast.detach(),
            "l_leak": l_leak.detach(),
            "l_corr": l_corr.detach(),
            "l_burst": (l_burst_cls + l_footprint).detach(),
            "l_drift_mem": l_drift_mem.detach(),
        }
