"""Version B: Small Spatio-Temporal Causal Transformer Core.

Uses causal multi-head self-attention across the round history (with `burn_in_steps` serving as causal
context warmup) and projects causal hidden states into the same 5 physical latent subspaces
(`z_corr, z_leak, z_drift, z_memory, z_event`) and multi-horizon schedule-conditioned forecasts.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.models.latent_space import FactoredLatentSpace


class CausalTemporalAttentionLayer(nn.Module):
    """Causal multi-head self-attention along the QEC round dimension T for each device node."""

    def __init__(self, d_model: int, n_heads: int = 4, max_len: int = 256) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Sinusoidal relative/round positional encoding
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, allow_history: bool = True) -> torch.Tensor:
        """
        Args:
            x: (BC, T, D) node sequences along time
            allow_history: if False (burn_in_steps=0 or temporal_mode='none'), restricts attention to diagonal or current window
        """
        BC, T, D = x.shape
        x_pe = x + self.pe[:T].unsqueeze(0)
        qkv = self.qkv(x_pe).reshape(BC, T, 3, self.n_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)  # (BC, H, T, d_head)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # (BC, H, T, T)
        if allow_history:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        else:
            causal_mask = torch.eye(T, device=x.device, dtype=torch.bool)

        scores = scores.masked_fill(~causal_mask[None, None, :, :], -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(BC, T, D)
        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class SmallSpatioTemporalTransformerCore(nn.Module):
    """Causal temporal Transformer backbone matching the R2D2-Graph-RSSM latent and forecast contract."""

    def __init__(
        self,
        d_perception: int = 64,
        d_hidden: int = 64,
        d_subspace: int = 8,
        num_layers: int = 2,
        n_heads: int = 4,
        temporal_mode: str = "causal_transformer",
    ) -> None:
        super().__init__()
        self.d_perception = d_perception
        self.d_hidden = d_hidden
        self.temporal_mode = temporal_mode
        self.latent_space = FactoredLatentSpace(
            d_hidden=d_hidden, d_perception=d_perception, d_subspace=d_subspace
        )
        d_lat = self.latent_space.d_latent_total

        self.sched_cond_proj = nn.Embedding(5, d_perception)
        self.drive_cond_proj = nn.Linear(1, d_perception)
        self.edge_summary_proj = nn.Linear(1, d_perception)

        self.in_proj = nn.Linear(d_perception * 2, d_hidden)
        self.layers = nn.ModuleList(
            [CausalTemporalAttentionLayer(d_model=d_hidden, n_heads=n_heads) for _ in range(num_layers)]
        )
        self.horizon_step_emb = nn.Embedding(16, d_hidden)
        self.forecast_mlp = nn.Sequential(
            nn.Linear(d_hidden + d_lat + d_perception, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
        )

    def compute_schedule_embedding(
        self,
        concurrency_group: torch.Tensor,
        drive_norm: torch.Tensor,
        schedule_adj: torch.Tensor,
        use_schedule: bool = True,
    ) -> torch.Tensor:
        """Build per-node schedule conditioning vector u_emb_t of shape (..., C, d_perception)."""
        C = schedule_adj.shape[-1]
        if not use_schedule:
            shape = list(schedule_adj.shape[:-2]) + [C, self.d_perception]
            return torch.zeros(*shape, device=schedule_adj.device)

        grp_emb = self.sched_cond_proj(concurrency_group.clamp(0, 4)).unsqueeze(-2)
        drv_emb = self.drive_cond_proj(drive_norm).unsqueeze(-2)
        active_deg = schedule_adj.sum(dim=-1, keepdim=True)
        deg_emb = self.edge_summary_proj(active_deg)
        return grp_emb + drv_emb + deg_emb

    def forward(
        self,
        g_nodes: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        burn_in_steps: int = 16,
        train_steps: int = 32,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        use_schedule: bool = True,
    ) -> Dict[str, Any]:
        """Run causal temporal Transformer over context and project to 5 physical latent subspaces."""
        B, T_total, C, D = g_nodes.shape
        device = g_nodes.device
        active_end = burn_in_steps + train_steps

        u_all = self.compute_schedule_embedding(
            obs["concurrency_group"],
            obs["drive_norm"],
            obs["schedule_adj"],
            use_schedule=use_schedule,
        )  # (B, T_total, C, D)

        # If burn_in_steps > 0, include [0 .. active_end] as causal context; otherwise slice only [burn_in_steps .. active_end]
        g_window = g_nodes[:, :active_end]  # (B, active_end, C, D)
        u_window = u_all[:, :active_end]

        # Rearrange to (B * C, active_end, d_hidden)
        BC = B * C
        x_in = torch.cat([g_window, u_window], dim=-1).permute(0, 2, 1, 3).reshape(BC, active_end, D * 2)
        h_seq = self.in_proj(x_in)

        allow_history = self.temporal_mode != "none"
        for layer in self.layers:
            h_seq = layer(h_seq, allow_history=allow_history)

        # Reshape back to (B, active_end, C, d_hidden) and slice active training window [burn_in_steps : active_end]
        h_full = h_seq.reshape(B, C, active_end, self.d_hidden).permute(0, 2, 1, 3)
        h_active = h_full[:, burn_in_steps:active_end]  # (B, train_steps, C, d_hidden)
        g_active = g_nodes[:, burn_in_steps:active_end]
        u_active = u_all[:, burn_in_steps:active_end]

        # Prior conditioned on previous step causal state h_{t-1} and current schedule u_t
        if burn_in_steps > 0 and allow_history:
            h_prev = h_full[:, burn_in_steps - 1 : active_end - 1]
        else:
            h_prev = torch.cat(
                [torch.zeros_like(h_active[:, :1]), h_active[:, :-1]], dim=1
            )

        _, prior_mean, prior_std = self.latent_space.prior(h_prev, u_active)
        z_post, post_mean, post_std = self.latent_space.posterior(h_active, g_active)

        kl_loss = self.latent_space.kl_divergence(post_mean, post_std, prior_mean, prior_std)

        # Multi-horizon schedule-conditioned rollout queries
        forecast_states: Dict[int, Dict[str, torch.Tensor]] = {}
        if len(horizons) > 0:
            h_im = h_active
            z_im = post_mean
            for h_step in sorted(horizons):
                u_future = u_all[:, burn_in_steps + h_step : active_end + h_step]
                step_emb = self.horizon_step_emb(torch.tensor(h_step, device=device))[None, None, None, :]
                h_im = self.forecast_mlp(torch.cat([h_im + step_emb, z_im, u_future], dim=-1))
                _, z_im, _ = self.latent_space.prior(h_im, u_future)
                forecast_states[h_step] = {
                    "h": h_im,
                    "z": z_im,
                    "u": u_future,
                }

        return {
            "h": h_active,
            "z": z_post,
            "z_mean": post_mean,
            "z_subspaces": self.latent_space.split_subspaces(post_mean),
            "u_active": u_active,
            "kl_loss": kl_loss,
            "forecast_states": forecast_states,
        }
