"""The 6 physics-specific noise prediction heads defined in Section 5 of `world_model/README.md`."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class SixNoisePredictionHeads(nn.Module):
    """Implements all 6 structured noise prediction heads over the factored latent subspaces:
    1. Detection head (`det_logits`, `det_prob`)
    2. Correlation head (`pair_corr_prob`, `scalar_corr_rate`)
    3. Leakage head (`leak_logits`, `leak_prob`)
    4. Drift/memory head (`drift_pred`, `memory_pred`, `effective_rate_pred`)
    5. Calibration head (`calib_pred`: [t1_norm, detuning_norm])
    6. Common-event head (`burst_logits`, `burst_prob`, `burst_footprint_pred`)
    """

    def __init__(
        self,
        d_hidden: int = 64,
        d_perception: int = 64,
        d_subspace: int = 8,
    ) -> None:
        super().__init__()
        d_lat = 5 * d_subspace

        # 1. Detection Head: uses full latent z_t + h_t + schedule u_t
        self.det_head = nn.Sequential(
            nn.Linear(d_hidden + d_lat + d_perception, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

        # 2. Correlation Head: conditioned on z_corr subspace + schedule u_t
        self.corr_node_proj = nn.Sequential(
            nn.Linear(d_subspace + d_perception, d_hidden // 2),
            nn.SiLU(),
        )
        self.corr_pair_mlp = nn.Sequential(
            nn.Linear(d_hidden + 1, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # 3. Leakage Head: conditioned on z_leak subspace + h_t
        self.leak_head = nn.Sequential(
            nn.Linear(d_subspace + d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # 4. Drift / Memory Head: conditioned on z_drift + z_memory + schedule u_t
        self.drift_head = nn.Sequential(
            nn.Linear(d_subspace * 2 + d_perception, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 2),  # [ou_drift, memory_revival]
        )

        # 5. Calibration Head: predicts global/node telemetry [t1_norm, detuning_norm]
        self.calib_head = nn.Sequential(
            nn.Linear(d_subspace * 3, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 2),
        )

        # 6. Common-Event Head: conditioned on z_event subspace + pooled chip state
        self.footprint_head = nn.Sequential(
            nn.Linear(d_subspace + d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
        )
        self.burst_global_head = nn.Sequential(
            nn.Linear(d_subspace * 2, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def decode_single_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        z_subspaces: Dict[str, torch.Tensor],
        u_emb: torch.Tensor,
        schedule_adj: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode all 6 heads for tensor sequence (B, T, C, ...)."""
        z_corr = z_subspaces["z_corr"]
        z_leak = z_subspaces["z_leak"]
        z_drift = z_subspaces["z_drift"]
        z_memory = z_subspaces["z_memory"]
        z_event = z_subspaces["z_event"]

        # 1. Detection Head
        det_logits = self.det_head(torch.cat([h, z, u_emb], dim=-1)).squeeze(-1)  # (B, T, C)
        det_prob = torch.sigmoid(det_logits)

        # 2. Correlation Head (pairwise over C x C checks conditioned on schedule_adj)
        B, T, C, _ = z_corr.shape
        c_feat = self.corr_node_proj(torch.cat([z_corr, u_emb], dim=-1))  # (B, T, C, D/2)
        c_i = c_feat.unsqueeze(3).expand(B, T, C, C, -1)
        c_j = c_feat.unsqueeze(2).expand(B, T, C, C, -1)
        pair_in = torch.cat([c_i, c_j, schedule_adj.unsqueeze(-1)], dim=-1)
        pair_raw = torch.sigmoid(self.corr_pair_mlp(pair_in).squeeze(-1)) * 0.12
        # Symmetrize and zero diagonal
        eye = torch.eye(C, device=z.device)[None, None, :, :]
        pair_corr_prob = 0.5 * (pair_raw + pair_raw.transpose(-1, -2)) * (1.0 - eye)
        scalar_corr_rate = pair_corr_prob.amax(dim=(-1, -2))  # (B, T)

        # 3. Leakage Head
        leak_logits = self.leak_head(torch.cat([z_leak, h], dim=-1)).squeeze(-1)  # (B, T, C)
        leak_prob = torch.sigmoid(leak_logits)

        # 4. Drift & Non-Markovian Memory Head
        dm_out = self.drift_head(torch.cat([z_drift, z_memory, u_emb], dim=-1))  # (B, T, C, 2)
        drift_pred = torch.tanh(dm_out[..., 0]) * 0.05
        memory_pred = torch.sigmoid(dm_out[..., 1]) * 0.09

        # 5. Calibration Telemetry Head (pooled across checks)
        cal_in = torch.cat([z_drift, z_memory, z_event], dim=-1).mean(dim=2)  # (B, T, 3*d_sub)
        calib_pred = self.calib_head(cal_in)  # (B, T, 2) -> [t1_norm, detuning_norm]

        # 6. Common-Event (Chip-Wide Burst) Head
        burst_footprint_pred = torch.clamp(
            self.footprint_head(torch.cat([z_event, h], dim=-1)).squeeze(-1), 0.0, 0.60
        )  # (B, T, C)
        z_ev_pool = torch.cat([z_event.mean(dim=2), z_event.amax(dim=2)], dim=-1)  # (B, T, 2*d_sub)
        burst_logits = self.burst_global_head(z_ev_pool).squeeze(-1)  # (B, T)
        burst_prob = torch.sigmoid(burst_logits)

        return {
            "det_logits": det_logits,
            "det_prob": det_prob,
            "pair_corr_prob": pair_corr_prob,
            "scalar_corr_rate": scalar_corr_rate,
            "leak_logits": leak_logits,
            "leak_prob": leak_prob,
            "drift_pred": drift_pred,
            "memory_pred": memory_pred,
            "calib_pred": calib_pred,
            "burst_logits": burst_logits,
            "burst_prob": burst_prob,
            "burst_footprint_pred": burst_footprint_pred,
        }

    def forward(
        self,
        core_out: Dict[str, Any],
        obs: Dict[str, torch.Tensor],
        burn_in_steps: int = 16,
        train_steps: int = 32,
        latent_space: Any = None,
    ) -> Dict[str, Any]:
        """Decode posterior predictions on the active window and multi-horizon forecasts H in {1,2,4,8}."""
        active_end = burn_in_steps + train_steps
        sched_active = obs["schedule_adj"][:, burn_in_steps:active_end]

        posterior_preds = self.decode_single_step(
            h=core_out["h"],
            z=core_out["z"],
            z_subspaces=core_out["z_subspaces"],
            u_emb=core_out["u_active"],
            schedule_adj=sched_active,
        )

        forecast_preds: Dict[int, Dict[str, torch.Tensor]] = {}
        for h_step, f_state in core_out.get("forecast_states", {}).items():
            f_h = f_state["h"]
            f_z = f_state["z"]
            f_u = f_state["u"]
            f_sub = latent_space.split_subspaces(f_z)
            f_det_logits = self.det_head(torch.cat([f_h, f_z, f_u], dim=-1)).squeeze(-1)
            f_dm = self.drift_head(torch.cat([f_sub["z_drift"], f_sub["z_memory"], f_u], dim=-1))
            f_drift = torch.tanh(f_dm[..., 0]) * 0.05
            f_mem = torch.sigmoid(f_dm[..., 1]) * 0.09
            forecast_preds[h_step] = {
                "det_logits": f_det_logits,
                "det_prob": torch.sigmoid(f_det_logits),
                "drift_pred": f_drift,
                "memory_pred": f_mem,
            }

        return {
            "posterior": posterior_preds,
            "forecasts": forecast_preds,
            "kl_loss": core_out["kl_loss"],
        }
