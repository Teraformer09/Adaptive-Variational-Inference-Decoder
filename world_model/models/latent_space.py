"""Factored physical latent space (z_corr, z_leak, z_drift, z_memory, z_event) for the AVID World Model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_COMPONENTS = ("z_corr", "z_leak", "z_drift", "z_memory", "z_event")


class FactoredLatentSpace(nn.Module):
    """Computes prior p_theta(z_t | h_t, u_t) and posterior q_phi(z_t | h_t, g_t) over 5 typed noise subspaces."""

    def __init__(self, d_hidden: int = 64, d_perception: int = 64, d_subspace: int = 8) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.d_perception = d_perception
        self.d_subspace = d_subspace
        self.d_latent_total = len(LATENT_COMPONENTS) * d_subspace

        # Posterior q_phi(z_t | h_t, g_t)
        self.post_net = nn.Sequential(
            nn.Linear(d_hidden + d_perception, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, self.d_latent_total * 2),
        )

        # Schedule-conditioned prior p_theta(z_t | h_t, u_t)
        self.prior_net = nn.Sequential(
            nn.Linear(d_hidden + d_perception, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, self.d_latent_total * 2),
        )

    def _split_params(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(raw, 2, dim=-1)
        mean = torch.tanh(mean) * 4.0
        std = F.softplus(raw_std) + 0.05
        return mean, std

    def posterior(self, h_t: torch.Tensor, g_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior distribution q_phi(z_t | h_t, g_t) and sample z_t."""
        raw = self.post_net(torch.cat([h_t, g_t], dim=-1))
        mean, std = self._split_params(raw)
        if self.training:
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean
        return z, mean, std

    def prior(self, h_t: torch.Tensor, u_emb_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute schedule-conditioned prior distribution p_theta(z_t | h_t, u_t) and sample z_t."""
        raw = self.prior_net(torch.cat([h_t, u_emb_t], dim=-1))
        mean, std = self._split_params(raw)
        if self.training:
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean
        return z, mean, std

    def split_subspaces(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Unpack flat latent tensor (..., 5 * d_subspace) into typed physical latent components."""
        chunks = torch.chunk(z, len(LATENT_COMPONENTS), dim=-1)
        return {name: chunk for name, chunk in zip(LATENT_COMPONENTS, chunks)}

    @staticmethod
    def kl_divergence(
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
        free_nats: float = 0.05,
    ) -> torch.Tensor:
        """Analytical Gaussian KL divergence D_KL(q || p) with free-nats floor to prevent posterior collapse."""
        var_q = post_std ** 2
        var_p = prior_std ** 2
        kl = torch.log(prior_std / post_std) + (var_q + (post_mean - prior_mean) ** 2) / (2.0 * var_p) - 0.5
        kl_per_node = torch.sum(kl, dim=-1)
        return torch.clamp(kl_per_node, min=free_nats).mean()
