"""Unified AVID World Model wrapping the Non-Vision Perception Layer, Dual Backbones, and 6 Prediction Heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from world_model.data.generator import DeviceTopology, build_surface17_topology
from world_model.heads.prediction_heads import SixNoisePredictionHeads
from world_model.models.r2d2_rssm_core import R2D2GraphRSSMCore
from world_model.models.st_transformer_core import SmallSpatioTemporalTransformerCore
from world_model.perception.graph_encoder import QuantumGraphPerceptionEncoder


@dataclass
class WorldModelConfig:
    """Unified configuration & ablation switches (Section 4.4 of `world_model/README.md`)."""

    backbone: str = "r2d2_graph_rssm"  # "r2d2_graph_rssm" | "spatio_temporal_transformer" | "hybrid_graph_transformer_rssm"
    spatial_mode: str = "gnn"          # "gnn" | "graph_transformer" | "dense_transformer" | "none"
    temporal_mode: str = "r2d2_lstm_rssm"  # "r2d2_lstm_rssm" | "gru_rssm" | "causal_transformer" | "none"
    burn_in_steps: int = 16
    train_steps: int = 32
    use_schedule: bool = True
    use_analog: bool = True
    d_model: int = 64
    d_edge: int = 16
    d_subspace: int = 8
    num_spatial_layers: int = 2


class AVIDWorldModel(nn.Module):
    """Predictive latent noise-state estimator and multi-horizon noise generator/forecaster."""

    def __init__(
        self,
        config: Optional[WorldModelConfig] = None,
        topology: Optional[DeviceTopology] = None,
    ) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        self.topology = topology or build_surface17_topology()

        # Resolve hybrid shorthand
        spatial_mode = self.config.spatial_mode
        temporal_mode = self.config.temporal_mode
        if self.config.backbone == "hybrid_graph_transformer_rssm":
            spatial_mode = "graph_transformer"
            temporal_mode = "r2d2_lstm_rssm"
        elif self.config.backbone == "spatio_temporal_transformer" and temporal_mode == "r2d2_lstm_rssm":
            temporal_mode = "causal_transformer"

        self.perception = QuantumGraphPerceptionEncoder(
            topology=self.topology,
            d_model=self.config.d_model,
            d_edge=self.config.d_edge,
            num_layers=self.config.num_spatial_layers,
            spatial_mode=spatial_mode,
            use_schedule=self.config.use_schedule,
            use_analog=self.config.use_analog,
        )

        if self.config.backbone in ("r2d2_graph_rssm", "hybrid_graph_transformer_rssm"):
            self.core = R2D2GraphRSSMCore(
                d_perception=self.config.d_model,
                d_hidden=self.config.d_model,
                d_subspace=self.config.d_subspace,
                temporal_mode=temporal_mode,
            )
        elif self.config.backbone == "spatio_temporal_transformer":
            self.core = SmallSpatioTemporalTransformerCore(
                d_perception=self.config.d_model,
                d_hidden=self.config.d_model,
                d_subspace=self.config.d_subspace,
                num_layers=2,
                n_heads=4,
                temporal_mode=temporal_mode,
            )
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.heads = SixNoisePredictionHeads(
            d_hidden=self.config.d_model,
            d_perception=self.config.d_model,
            d_subspace=self.config.d_subspace,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        burn_in_steps: Optional[int] = None,
        train_steps: Optional[int] = None,
        horizons: Tuple[int, ...] = (1, 2, 4, 8),
    ) -> Dict[str, Any]:
        """Forward pass consuming ONLY classical observations `obs` (never `latent_truth`)."""
        if "latent_truth" in obs:
            raise ValueError("Inference violation: `latent_truth` must never be passed inside `obs`!")

        b_steps = self.config.burn_in_steps if burn_in_steps is None else burn_in_steps
        t_steps = self.config.train_steps if train_steps is None else train_steps

        g_nodes, _ = self.perception(obs)
        core_out = self.core(
            g_nodes=g_nodes,
            obs=obs,
            burn_in_steps=b_steps,
            train_steps=t_steps,
            horizons=horizons,
            use_schedule=self.config.use_schedule,
        )
        head_out = self.heads(
            core_out=core_out,
            obs=obs,
            burn_in_steps=b_steps,
            train_steps=t_steps,
            latent_space=self.core.latent_space,
        )
        head_out["z_subspaces"] = core_out["z_subspaces"]
        return head_out

    @torch.no_grad()
    def predict(
        self,
        observation_window: Dict[str, torch.Tensor],
        topology: Optional[DeviceTopology] = None,
        future_schedule: Optional[Dict[str, torch.Tensor]] = None,
        horizon: int = 8,
        burn_in_steps: Optional[int] = None,
        **forbidden_kwargs: Any,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        """Strict inference API defined in Section 2 of `world_model/README.md`.

        Rejects `latent_truth` if passed either in `observation_window` or via keyword arguments.
        """
        if "latent_truth" in forbidden_kwargs or "latent_truth" in observation_window:
            raise ValueError(
                "Inference Contract Violation: `predict()` must never accept `latent_truth` or simulator hidden state."
            )

        was_training = self.training
        self.eval()
        try:
            obs_in = dict(observation_window)
            if future_schedule is not None:
                for k, v in future_schedule.items():
                    if k in obs_in:
                        obs_in[k] = torch.cat([obs_in[k], v], dim=1)

            T_total = obs_in["detection_vec"].shape[1]
            b_steps = self.config.burn_in_steps if burn_in_steps is None else burn_in_steps
            active_steps = T_total - b_steps - horizon
            if active_steps <= 0:
                b_steps = 0
                active_steps = T_total - horizon

            horizons = tuple(h for h in (1, 2, 4, 8) if h <= horizon)
            out = self.forward(
                obs=obs_in,
                burn_in_steps=b_steps,
                train_steps=active_steps,
                horizons=horizons,
            )
            belief = {
                "z_subspaces": out["z_subspaces"],
                "posterior_estimates": out["posterior"],
            }
            forecast = out["forecasts"]
            return belief, forecast
        finally:
            if was_training:
                self.train()
