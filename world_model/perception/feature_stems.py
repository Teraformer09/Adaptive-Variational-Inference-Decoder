"""Non-vision physics-informed feature stems for the Quantum Graph Perception Layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiscreteSyndromeStem(nn.Module):
    """Embeds binary detection events d_{i,t}, raw stabilizer parity s_{i,t}, and check role."""

    def __init__(self, num_checks: int, d_model: int) -> None:
        super().__init__()
        self.num_checks = num_checks
        self.node_id_emb = nn.Embedding(num_checks, d_model)
        self.role_emb = nn.Embedding(3, d_model)  # 0=data, 1=X_check, 2=Z_check
        self.proj = nn.Linear(2, d_model)

    def forward(
        self,
        detection_vec: torch.Tensor,
        syndrome_vec: torch.Tensor,
        node_roles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            detection_vec: (B, T, C) binary detection events
            syndrome_vec: (B, T, C) binary raw syndromes
            node_roles: (C,) integer roles in {1, 2}
        Returns:
            (B, T, C, d_model)
        """
        B, T, C = detection_vec.shape
        device = detection_vec.device
        x_pair = torch.stack([detection_vec, syndrome_vec], dim=-1)  # (B, T, C, 2)
        node_ids = torch.arange(C, device=device)
        return (
            self.proj(x_pair)
            + self.node_id_emb(node_ids)[None, None, :, :]
            + self.role_emb(node_roles.to(device))[None, None, :, :]
        )


class ValidityMaskedAnalogTelemetryStem(nn.Module):
    """Encodes analog readout LLRs and calibration telemetry with explicit missingness masks.

    Ensures zero is never conflated with a missing value by feeding `[m * x, m]` into the projection.
    Supports `use_analog=False` ablation by zeroing out both features and masks.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # Per-check input: [masked_llr, llr_mask, masked_t1, t1_mask, masked_detun, detun_mask, masked_drive, drive_mask] -> 8 dims
        self.net = nn.Sequential(
            nn.Linear(8, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        analog_llr: torch.Tensor,
        analog_mask: torch.Tensor,
        telemetry_val: torch.Tensor,
        telemetry_mask: torch.Tensor,
        use_analog: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            analog_llr: (B, T, C)
            analog_mask: (B, T, 1)
            telemetry_val: (B, T, 3)
            telemetry_mask: (B, T, 3)
            use_analog: whether to include analog LLR and telemetry features
        Returns:
            (B, T, C, d_model)
        """
        B, T, C = analog_llr.shape
        if not use_analog:
            return torch.zeros(B, T, C, self.net[-1].out_features, device=analog_llr.device)

        m_llr = analog_mask.expand(B, T, C)
        x_llr = analog_llr * m_llr

        m_tel = telemetry_mask[:, :, None, :].expand(B, T, C, 3)
        x_tel = telemetry_val[:, :, None, :].expand(B, T, C, 3) * m_tel

        feat = torch.cat(
            [
                x_llr.unsqueeze(-1),
                m_llr.unsqueeze(-1),
                x_tel,
                m_tel,
            ],
            dim=-1,
        )  # (B, T, C, 8)
        return self.net(feat)


class SpatialGeometryRBFStem(nn.Module):
    """Encodes 2D chip coordinates and pairwise Euclidean distances via Gaussian RBF kernels."""

    def __init__(self, num_rbf: int = 8, max_dist_um: float = 1200.0, d_edge: int = 16) -> None:
        super().__init__()
        centers = torch.linspace(0.0, max_dist_um, num_rbf)
        widths = torch.full((num_rbf,), (max_dist_um / num_rbf) * 0.85)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.edge_proj = nn.Linear(num_rbf + 1, d_edge)

    def forward(
        self, check_dist_um: torch.Tensor, static_coupler_adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            check_dist_um: (C, C) pairwise Euclidean distance in micrometers
            static_coupler_adj: (C, C) binary physical spectator/coupler adjacency
        Returns:
            static_edge_emb: (C, C, d_edge)
        """
        dist = check_dist_um.unsqueeze(-1)  # (C, C, 1)
        rbf = torch.exp(-((dist - self.centers) ** 2) / (2.0 * (self.widths ** 2) + 1e-6))
        cat = torch.cat([rbf, static_coupler_adj.unsqueeze(-1)], dim=-1)
        return self.edge_proj(cat)


class ScheduleCouplerEdgeStem(nn.Module):
    """Encodes time-varying simultaneous gate schedules u_t onto coupler edges and node conditioning."""

    def __init__(self, d_model: int, d_edge: int = 16) -> None:
        super().__init__()
        self.concurrency_emb = nn.Embedding(5, d_model)
        self.drive_proj = nn.Linear(1, d_model)
        self.edge_sched_proj = nn.Linear(2, d_edge)

    def forward(
        self,
        schedule_adj: torch.Tensor,
        concurrency_group: torch.Tensor,
        drive_norm: torch.Tensor,
        use_schedule: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            schedule_adj: (B, T, C, C) active parallel gate adjacency at round t
            concurrency_group: (B, T) integer concurrency group ID
            drive_norm: (B, T, 1) normalized Rabi drive frequency
            use_schedule: if False, masks schedule features to test schedule-blind ablation
        Returns:
            node_sched_emb: (B, T, 1, d_model)
            dynamic_edge_emb: (B, T, C, C, d_edge)
        """
        B, T, C, _ = schedule_adj.shape
        if not use_schedule:
            return (
                torch.zeros(B, T, 1, self.drive_proj.out_features, device=schedule_adj.device),
                torch.zeros(B, T, C, C, self.edge_sched_proj.out_features, device=schedule_adj.device),
            )

        grp_clamped = concurrency_group.clamp(0, 4)
        node_sched = (
            self.concurrency_emb(grp_clamped) + self.drive_proj(drive_norm)
        ).unsqueeze(2)  # (B, T, 1, d_model)

        drive_grid = drive_norm[:, :, :, None].expand(B, T, C, C)
        edge_in = torch.stack([schedule_adj, schedule_adj * drive_grid], dim=-1)  # (B, T, C, C, 2)
        dyn_edge = self.edge_sched_proj(edge_in)
        return node_sched, dyn_edge
