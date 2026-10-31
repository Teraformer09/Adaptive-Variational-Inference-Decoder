"""Quantum Graph Perception Layer supporting GNN, Graph-Biased Transformer, Dense Transformer, and Local MLP."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.data.generator import DeviceTopology
from world_model.perception.feature_stems import (
    DiscreteSyndromeStem,
    ScheduleCouplerEdgeStem,
    SpatialGeometryRBFStem,
    ValidityMaskedAnalogTelemetryStem,
)


class EdgeConditionedGNNLayer(nn.Module):
    """GATv2-style edge-conditioned message passing over physical couplers and schedule edges."""

    def __init__(self, d_model: int, d_edge: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.edge_gate = nn.Sequential(
            nn.Linear(d_edge, d_model),
            nn.Sigmoid(),
        )
        self.edge_bias = nn.Linear(d_edge, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        edge_feat: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, d_model)
            edge_feat: (B, T, C, C, d_edge)
            adj_mask: (B, T, C, C) binary mask of allowed message-passing edges (couplers + self + active schedule)
        """
        D = x.shape[-1]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Pairwise attention logits + learned edge bias
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # (B, T, C, C)
        scores = scores + self.edge_bias(edge_feat).squeeze(-1)
        scores = scores.masked_fill(adj_mask <= 0.0, -1e9)
        attn = F.softmax(scores, dim=-1)

        # Edge-gated value messages
        gates = self.edge_gate(edge_feat)  # (B, T, C, C, D)
        v_expanded = v.unsqueeze(-3) * gates  # (B, T, C, C, D)
        agg = torch.sum(attn.unsqueeze(-1) * v_expanded, dim=-2)  # (B, T, C, D)

        x = self.norm1(x + self.out_proj(agg))
        x = self.norm2(x + self.ffn(x))
        return x


class SpatialTransformerLayer(nn.Module):
    """Multi-head spatial self-attention with optional coupler/schedule/RBF additive edge bias."""

    def __init__(self, d_model: int, d_edge: int, n_heads: int = 4, use_edge_bias: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_edge_bias = use_edge_bias

        self.qkv = nn.Linear(d_model, d_model * 3)
        if use_edge_bias:
            self.edge_to_heads = nn.Linear(d_edge, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, d_model)
            edge_feat: (B, T, C, C, d_edge)
        """
        B, T, C, D = x.shape
        qkv = self.qkv(x).reshape(B, T, C, 3, self.n_heads, self.d_head)
        q = qkv[..., 0, :, :].transpose(-2, -3)  # (B, T, H, C, d_head)
        k = qkv[..., 1, :, :].transpose(-2, -3)
        v = qkv[..., 2, :, :].transpose(-2, -3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)  # (B, T, H, C, C)
        if self.use_edge_bias:
            bias = self.edge_to_heads(edge_feat).permute(0, 1, 4, 2, 3)  # (B, T, H, C, C)
            scores = scores + bias

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(-2, -3).reshape(B, T, C, D)
        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.ffn(x))
        return x


class QuantumGraphPerceptionEncoder(nn.Module):
    """Non-vision perception layer mapping round observations o_t to node & edge embeddings g_t."""

    def __init__(
        self,
        topology: DeviceTopology,
        d_model: int = 64,
        d_edge: int = 16,
        num_layers: int = 2,
        spatial_mode: str = "gnn",
        use_schedule: bool = True,
        use_analog: bool = True,
    ) -> None:
        super().__init__()
        self.topology = topology
        self.num_checks = topology.num_checks
        self.d_model = d_model
        self.d_edge = d_edge
        self.spatial_mode = spatial_mode
        self.use_schedule = use_schedule
        self.use_analog = use_analog

        # Precompute static topology buffers on the check graph
        C = topology.num_checks
        static_adj = np.eye(C, dtype=np.float32)
        for c1, c2 in topology.spectator_edges:
            static_adj[c1, c2] = 1.0
            static_adj[c2, c1] = 1.0
        self.register_buffer("static_coupler_adj", torch.from_numpy(static_adj))
        self.register_buffer("check_dist_um", torch.from_numpy(topology.check_dist_um))
        self.register_buffer(
            "check_roles",
            torch.tensor(topology.node_roles[:C], dtype=torch.long),
        )

        # 4 Physics-informed non-vision stems
        self.syndrome_stem = DiscreteSyndromeStem(num_checks=C, d_model=d_model)
        self.analog_stem = ValidityMaskedAnalogTelemetryStem(d_model=d_model)
        self.geometry_stem = SpatialGeometryRBFStem(num_rbf=8, max_dist_um=1200.0, d_edge=d_edge)
        self.schedule_stem = ScheduleCouplerEdgeStem(d_model=d_model, d_edge=d_edge)

        self.input_norm = nn.LayerNorm(d_model)

        # Configurable Spatial Message-Passing / Attention Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if spatial_mode == "gnn":
                self.layers.append(EdgeConditionedGNNLayer(d_model=d_model, d_edge=d_edge))
            elif spatial_mode == "graph_transformer":
                self.layers.append(
                    SpatialTransformerLayer(d_model=d_model, d_edge=d_edge, n_heads=4, use_edge_bias=True)
                )
            elif spatial_mode == "dense_transformer":
                self.layers.append(
                    SpatialTransformerLayer(d_model=d_model, d_edge=d_edge, n_heads=4, use_edge_bias=False)
                )
            elif spatial_mode == "none":
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.SiLU(),
                        nn.LayerNorm(d_model),
                    )
                )
            else:
                raise ValueError(f"Unknown spatial_mode: {spatial_mode}")

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dictionary of observation tensors with leading dims (B, T, ...)
        Returns:
            g_nodes: (B, T, C, d_model) spatial-perceived node representations
            edge_feat: (B, T, C, C, d_edge) combined static + dynamic edge features
        """
        det = obs["detection_vec"]
        syn = obs["syndrome_vec"]
        B, T, C = det.shape

        x_syn = self.syndrome_stem(det, syn, self.check_roles)
        x_ana = self.analog_stem(
            obs["analog_llr"],
            obs["analog_mask"],
            obs["telemetry_val"],
            obs["telemetry_mask"],
            use_analog=self.use_analog,
        )
        node_sched, dyn_edge = self.schedule_stem(
            obs["schedule_adj"],
            obs["concurrency_group"],
            obs["drive_norm"],
            use_schedule=self.use_schedule,
        )

        static_edge = self.geometry_stem(self.check_dist_um, self.static_coupler_adj)  # (C, C, d_edge)
        edge_feat = static_edge[None, None, :, :, :] + dyn_edge  # (B, T, C, C, d_edge)

        # Allowed message-passing adjacency for GNN: physical couplers + active schedule + radial neighborhood (<650 um)
        radial_adj = (self.check_dist_um < 650.0).float()[None, None, :, :]
        adj_mask = torch.clamp(
            self.static_coupler_adj[None, None, :, :] + obs["schedule_adj"] + radial_adj,
            0.0,
            1.0,
        )

        x = self.input_norm(x_syn + x_ana + node_sched)
        for layer in self.layers:
            if self.spatial_mode == "gnn":
                x = layer(x, edge_feat, adj_mask)
            elif self.spatial_mode in ("graph_transformer", "dense_transformer"):
                x = layer(x, edge_feat)
            else:
                x = x + layer(x)

        return x, edge_feat
