"""Version A: R2D2-Enhanced Dual-Timescale Recurrent State-Space Model (RSSM) Core.

Combines:
1. Chrono-Initialized Dual-Timescale LSTM cell state (h_t, c_t) (Tallec & Ollivier, ICLR 2018;
   Hochreiter & Schmidhuber, 1997) for simultaneous fast crosstalk/burst tracking and slow
   leakage/TLS memory retention.
2. R2D2 Stored-State Burn-In Warmup (Kapturowski et al., ICLR 2019) rolling through the first
   `burn_in_steps` rounds under `torch.no_grad()` to eliminate cold-start bias under partial observability.
3. Factored Variational RSSM prior/posterior over the 5 physical noise subspaces.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from world_model.models.latent_space import FactoredLatentSpace


class R2D2GraphRSSMCore(nn.Module):
    """Per-node recurrent state-space core with R2D2 burn-in and multi-horizon imagination."""

    def __init__(
        self,
        d_perception: int = 64,
        d_hidden: int = 64,
        d_subspace: int = 8,
        temporal_mode: str = "r2d2_lstm_rssm",
        t_min_rounds: float = 1.0,
        t_max_rounds: float = 96.0,
    ) -> None:
        super().__init__()
        self.d_perception = d_perception
        self.d_hidden = d_hidden
        self.temporal_mode = temporal_mode
        self.latent_space = FactoredLatentSpace(
            d_hidden=d_hidden, d_perception=d_perception, d_subspace=d_subspace
        )
        d_lat = self.latent_space.d_latent_total

        self.sched_cond_proj = nn.Sequential(
            nn.Embedding(5, d_perception),
        )
        self.drive_cond_proj = nn.Linear(1, d_perception)
        self.edge_summary_proj = nn.Linear(1, d_perception)

        if temporal_mode == "r2d2_lstm_rssm":
            self.rnn_cell = nn.LSTMCell(d_perception + d_lat, d_hidden)
            self._apply_chrono_initialization(t_min_rounds, t_max_rounds)
        elif temporal_mode == "gru_rssm":
            self.rnn_cell = nn.GRUCell(d_perception + d_lat, d_hidden)
        elif temporal_mode == "none":
            self.stateless_mlp = nn.Sequential(
                nn.Linear(d_perception + d_lat, d_hidden),
                nn.SiLU(),
                nn.LayerNorm(d_hidden),
            )
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")

        # Transition cell for multi-step prior imagination (uses schedule embedding u_{t+h} instead of observation g_{t+h})
        self.imagination_proj = nn.Linear(d_perception + d_lat, d_perception + d_lat)
        self.norm_h = nn.LayerNorm(d_hidden)

    def _apply_chrono_initialization(self, t_min: float, t_max: float) -> None:
        """Chrono-initialize LSTM forget and input gate biases across timescales [t_min, t_max] (Tallec & Ollivier, ICLR 2018)."""
        with torch.no_grad():
            H = self.d_hidden
            # Log-uniform characteristic timescales across hidden units
            u = torch.rand(H)
            tau = torch.exp(u * (np.log(t_max) - np.log(t_min)) + np.log(t_min))
            forget_bias = torch.log(tau)
            # PyTorch LSTMCell bias order: [b_i, b_f, b_g, b_o] each of size H
            self.rnn_cell.bias_ih.zero_()
            self.rnn_cell.bias_hh.zero_()
            self.rnn_cell.bias_ih[H : 2 * H].copy_(forget_bias)
            self.rnn_cell.bias_ih[0:H].copy_(-forget_bias)

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

        grp_emb = self.sched_cond_proj(concurrency_group.clamp(0, 4)).unsqueeze(-2)  # (..., 1, D)
        drv_emb = self.drive_cond_proj(drive_norm).unsqueeze(-2)  # (..., 1, D)
        active_deg = schedule_adj.sum(dim=-1, keepdim=True)  # (..., C, 1)
        deg_emb = self.edge_summary_proj(active_deg)  # (..., C, D)
        return grp_emb + drv_emb + deg_emb

    def _step_recurrent(
        self,
        inp: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance recurrent memory by 1 round across all B * C nodes."""
        BC, _ = inp.shape
        if self.temporal_mode == "r2d2_lstm_rssm":
            h_next, c_next = self.rnn_cell(inp, (h_prev, c_prev))
            return self.norm_h(h_next), c_next
        elif self.temporal_mode == "gru_rssm":
            h_next = self.rnn_cell(inp, h_prev)
            return self.norm_h(h_next), c_prev
        else:
            # Stateless memory ablation (temporal_mode="none")
            h_next = self.stateless_mlp(inp)
            return h_next, c_prev

    def forward(
        self,
        g_nodes: torch.Tensor,
        obs: Dict[str, torch.Tensor],
        burn_in_steps: int = 16,
        train_steps: int = 32,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
        use_schedule: bool = True,
    ) -> Dict[str, Any]:
        """Unroll R2D2 burn-in + active posterior filtering + multi-horizon prior forecasting.

        Args:
            g_nodes: (B, T_total, C, d_perception) output of QuantumGraphPerceptionEncoder
            obs: dictionary of observation tensors with T_total >= burn_in_steps + train_steps + max(horizons)
            burn_in_steps: number of initial rounds rolled forward under `torch.no_grad()`
            train_steps: number of active rounds for filtering and forecasting
            horizons: forecast horizons H to evaluate from each active round
        """
        B, T_total, C, D = g_nodes.shape
        device = g_nodes.device
        BC = B * C
        d_lat = self.latent_space.d_latent_total

        u_all = self.compute_schedule_embedding(
            obs["concurrency_group"],
            obs["drive_norm"],
            obs["schedule_adj"],
            use_schedule=use_schedule,
        )  # (B, T_total, C, D)

        h_t = torch.zeros(BC, self.d_hidden, device=device)
        c_t = torch.zeros(BC, self.d_hidden, device=device)
        z_t = torch.zeros(BC, d_lat, device=device)

        # 1. R2D2 Burn-In Warmup under torch.no_grad()
        if burn_in_steps > 0 and self.temporal_mode != "none":
            with torch.no_grad():
                for t in range(burn_in_steps):
                    g_t = g_nodes[:, t].reshape(BC, D)
                    inp = torch.cat([g_t, z_t], dim=-1)
                    h_t, c_t = self._step_recurrent(inp, h_t, c_t)
                    z_t, _, _ = self.latent_space.posterior(h_t, g_t)
            h_t = h_t.detach()
            c_t = c_t.detach()
            z_t = z_t.detach()

        # 2. Active Window Filtering over [burn_in_steps .. burn_in_steps + train_steps - 1]
        h_seq = []
        z_post_seq = []
        post_mean_seq = []
        post_std_seq = []
        prior_mean_seq = []
        prior_std_seq = []
        c_seq = []

        active_end = burn_in_steps + train_steps
        for t in range(burn_in_steps, active_end):
            g_t = g_nodes[:, t].reshape(BC, D)
            u_t = u_all[:, t].reshape(BC, D)

            # In stateless mode, zero out previous z_t to prevent temporal leakage
            z_prev = torch.zeros_like(z_t) if self.temporal_mode == "none" else z_t
            inp = torch.cat([g_t, z_prev], dim=-1)
            h_t, c_t = self._step_recurrent(inp, h_t, c_t)

            _, prior_mean, prior_std = self.latent_space.prior(h_t, u_t)
            z_t, post_mean, post_std = self.latent_space.posterior(h_t, g_t)

            h_seq.append(h_t.reshape(B, C, self.d_hidden))
            c_seq.append(c_t.reshape(B, C, self.d_hidden))
            z_post_seq.append(z_t.reshape(B, C, d_lat))
            post_mean_seq.append(post_mean.reshape(B, C, d_lat))
            post_std_seq.append(post_std.reshape(B, C, d_lat))
            prior_mean_seq.append(prior_mean.reshape(B, C, d_lat))
            prior_std_seq.append(prior_std.reshape(B, C, d_lat))

        h_stack = torch.stack(h_seq, dim=1)            # (B, T_train, C, d_hidden)
        c_stack = torch.stack(c_seq, dim=1)            # (B, T_train, C, d_hidden)
        z_post_stack = torch.stack(z_post_seq, dim=1)  # (B, T_train, C, d_lat)
        post_mean_stack = torch.stack(post_mean_seq, dim=1)
        post_std_stack = torch.stack(post_std_seq, dim=1)
        prior_mean_stack = torch.stack(prior_mean_seq, dim=1)
        prior_std_stack = torch.stack(prior_std_seq, dim=1)

        kl_loss = self.latent_space.kl_divergence(
            post_mean_stack, post_std_stack, prior_mean_stack, prior_std_stack
        )

        # 3. Multi-Horizon Schedule-Conditioned Prior Imagination for H in horizons
        forecast_states: Dict[int, Dict[str, torch.Tensor]] = {}
        if len(horizons) > 0:
            # Flatten across (B, T_train, C) to unroll all starting points in parallel
            BTC = B * train_steps * C
            h_im = h_stack.reshape(BTC, self.d_hidden)
            c_im = c_stack.reshape(BTC, self.d_hidden)
            z_im = z_post_stack.reshape(BTC, d_lat)

            for h_step in sorted(horizons):
                u_future = u_all[:, burn_in_steps + h_step : active_end + h_step].reshape(BTC, D)
                z_prev_im = torch.zeros_like(z_im) if self.temporal_mode == "none" else z_im
                im_inp = self.imagination_proj(torch.cat([u_future, z_prev_im], dim=-1))
                h_im, c_im = self._step_recurrent(im_inp, h_im, c_im)
                z_im, z_im_mean, _ = self.latent_space.prior(h_im, u_future)

                forecast_states[h_step] = {
                    "h": h_im.reshape(B, train_steps, C, self.d_hidden),
                    "z": z_im_mean.reshape(B, train_steps, C, d_lat),
                    "u": u_future.reshape(B, train_steps, C, D),
                }

        return {
            "h": h_stack,
            "z": z_post_stack,
            "z_mean": post_mean_stack,
            "z_subspaces": self.latent_space.split_subspaces(post_mean_stack),
            "u_active": u_all[:, burn_in_steps:active_end],
            "kl_loss": kl_loss,
            "forecast_states": forecast_states,
        }
