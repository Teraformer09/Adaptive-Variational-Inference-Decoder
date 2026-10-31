"""Lightweight Low-Rank (LoRA) and Residual Regime Adapters for the AVID World Model.

Allows freezing the main pretrained World Model (`WM_R2D2_Graph_RSSM_Full` or `WM_Small_ST_Transformer_Full`)
and training/exporting compact (~3K parameter) regime-specific or device-calibration adapters for
Hugging Face Hub deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from world_model.models.world_model import AVIDWorldModel


class LoRALinearAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) residual branch delta_W = B @ A with rank r."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = alpha / float(rank)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


class NoiseRegimeAdapter(nn.Module):
    """Modular adapter that specializes a frozen base AVIDWorldModel to a specific noise regime or schedule."""

    def __init__(
        self,
        regime_name: str,
        d_model: int = 64,
        d_subspace: int = 8,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.regime_name = regime_name
        d_lat = 5 * d_subspace

        # Low-rank adapters on perception node embedding, latent posterior mean, and schedule conditioning
        self.perception_lora = LoRALinearAdapter(d_model, d_model, rank=rank)
        self.latent_lora = LoRALinearAdapter(d_lat, d_lat, rank=rank)
        self.schedule_lora = LoRALinearAdapter(d_model, d_model, rank=rank)
        self.det_bias_head = nn.Sequential(
            nn.Linear(d_lat + d_model, rank),
            nn.SiLU(),
            nn.Linear(rank, 1),
        )
        nn.init.zeros_(self.det_bias_head[-1].weight)
        nn.init.zeros_(self.det_bias_head[-1].bias)

    def forward_adapted(
        self,
        base_model: AVIDWorldModel,
        obs: Dict[str, torch.Tensor],
        burn_in_steps: Optional[int] = None,
        train_steps: Optional[int] = None,
        horizons: tuple[int, ...] = (1, 2, 4, 8),
    ) -> Dict[str, Any]:
        """Run the base model with low-rank adapter residuals injected into perception, latent space, and heads."""
        b_steps = base_model.config.burn_in_steps if burn_in_steps is None else burn_in_steps
        t_steps = base_model.config.train_steps if train_steps is None else train_steps

        g_nodes, _ = base_model.perception(obs)
        g_nodes = g_nodes + self.perception_lora(g_nodes)

        core_out = base_model.core(
            g_nodes=g_nodes,
            obs=obs,
            burn_in_steps=b_steps,
            train_steps=t_steps,
            horizons=horizons,
            use_schedule=base_model.config.use_schedule,
        )

        # Apply low-rank adapters to posterior latent state and schedule embedding
        z_adapted = core_out["z"] + self.latent_lora(core_out["z"])
        u_adapted = core_out["u_active"] + self.schedule_lora(core_out["u_active"])
        core_out_adapted = dict(core_out)
        core_out_adapted["z"] = z_adapted
        core_out_adapted["z_subspaces"] = base_model.core.latent_space.split_subspaces(z_adapted)
        core_out_adapted["u_active"] = u_adapted

        head_out = base_model.heads(
            core_out=core_out_adapted,
            obs=obs,
            burn_in_steps=b_steps,
            train_steps=t_steps,
            latent_space=base_model.core.latent_space,
        )

        # Apply adapted detection calibration residual
        det_delta = self.det_bias_head(torch.cat([z_adapted, u_adapted], dim=-1)).squeeze(-1)
        head_out["posterior"]["det_logits"] = head_out["posterior"]["det_logits"] + det_delta
        head_out["posterior"]["det_prob"] = torch.sigmoid(head_out["posterior"]["det_logits"])
        head_out["z_subspaces"] = core_out_adapted["z_subspaces"]
        return head_out

    def save_adapter(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "regime_name": self.regime_name,
                "num_parameters": sum(p.numel() for p in self.parameters()),
                "state_dict": self.state_dict(),
            },
            path,
        )
