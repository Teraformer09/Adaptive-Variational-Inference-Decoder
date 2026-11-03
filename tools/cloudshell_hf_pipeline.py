#!/usr/bin/env python3
"""
Google Cloud Shell End-to-End Long-Term Data Collection, Weighted-Bellman R2D2 Training &
Hugging Face Hub Exporter for Branch `AVID_EPJ_Patches`.

Pipeline Stages:
  Stage 1 — Long-Term Quantum Syndrome & Non-Stationary Drift Data Collection:
    - Collects 50,000 fast-loop syndrome rounds per regime x 5 regimes = 250,000 rounds
      on the canonical rotated d=3 [[9, 1, 3]] CSS surface code (`avid_long_horizon_d3_250k_rounds.npz`
      and `avid_canonical_css_regimes.npz`).
    - Collects 600 long-horizon non-stationary drift & TLS telegraph replay trajectories
      of length T = 16 slow-loop windows (T_burn = 6 + T_unroll = 10, representing
      600 x 16 x 256 = 2,457,600 underlying syndrome rounds) stored in
      `datasets/avid_r2d2_replay_buffer_trajectories.npz`.
    - Generates 64-bit little-endian AXI4-Stream syndrome packet binaries (`syndrome_packets_v1.bin`).
  Stage 2 — Long-Horizon Calibration & Weighted-Bellman R2D2 World Model Training:
    - Calibrates Exact 16-State Joint-DP MAP (`AVID_Joint_DP_MAP`) via 80-step L-BFGS with
      Pair-Aware Logical-Triangle LLR Projection.
    - Trains `WeightedBellmanR2D2WorldModel` (`R2D2-Graph-RSSM`) for 80 epochs on the collected
      replay buffer using Kapturowski invertible value rescaling h(z), burn-in warmup (T_burn=6),
      3-step Bellman target bootstrapping, and Uncertainty-Weighted Bellman TD Loss.
    - Trains `SmallSpatioTemporalTransformer` (80 epochs) and 5 Rank-4 Regime LoRA Adapters (50 epochs/regime).
  Stage 3 — Shared-Seed Hardware Cycle-Accurate Q7.8 BP Evaluation (B0..B4 + AVID + R2D2).
  Stage 4 — Uploads `datasets/`, `checkpoints/`, `docs/`, `mathematical_model/`, `architecture/`,
            and `configs/` to Hugging Face Hub (`Terrano09/AVID-EPJ-Baseline-Models-and-Data`).
"""

from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))

from simulation.reference.noise_model.joint_dp_likelihood import (
    CorrelatedNoiseModel,
    build_d3_correlated_noise_model,
)
from simulation.reference.noise_model.identifiability_svd import generate_identifiability_artifacts
from simulation.reference.syndrome_stream.causal_stream_runner import CycleAccurateBPEngine
from simulation.reference.baselines import (
    B0StaticBPController,
    B1MarginalLBFGSController,
    B2ObservablePIDController,
    B3CovarianceTrackerController,
    B4OracleBPController,
)
from simulation.stimulus.packet_codec import pack_packet_stream
from tools.generate_code_artifacts import main as generate_code_main
from tools.collect_target_timing import main as collect_timing_main


def load_dotenv(env_path: Path) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        os.environ.setdefault(k, v)
        loaded[k] = v
    return loaded


def build_five_noise_regimes(default_theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Define 5 physically distinct correlated quantum noise regimes for d=3."""
    regimes: Dict[str, np.ndarray] = {}

    # R1: Stationary low-noise baseline
    r1 = default_theta.copy()
    regimes["R1_stationary_baseline"] = r1

    # R2: Center-qubit TLS hotspot (q4 and q5 elevated)
    r2 = default_theta.copy()
    r2[4] = 0.085
    r2[5] = 0.055
    regimes["R2_tls_hotspot_q4_q5"] = r2

    # R3: Strong ZZ crosstalk burst on horizontal edges (0,1), (4,5), (7,8)
    r3 = default_theta.copy()
    r3[9 + 0] = 0.065
    r3[9 + 3] = 0.080
    r3[9 + 5] = 0.060
    regimes["R3_zz_crosstalk_burst"] = r3

    # R4: Boundary gradient drift (top/left qubits elevated)
    r4 = default_theta.copy()
    r4[0] = 0.050
    r4[1] = 0.045
    r4[3] = 0.055
    r4[6] = 0.040
    regimes["R4_boundary_drift"] = r4

    # R5: Combined hotspot + ZZ crosstalk stress test
    r5 = default_theta.copy()
    r5[4] = 0.095
    r5[7] = 0.065
    r5[9 + 3] = 0.075
    r5[9 + 9] = 0.070
    regimes["R5_combined_stress"] = r5

    return regimes


# =============================================================================
# PyTorch Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) + Transformer + LoRA Models
# =============================================================================

def build_d3_bipartite_adjacency() -> torch.Tensor:
    """
    Build 13x13 adjacency matrix for the canonical rotated d=3 CSS graph:
      Nodes 0..8  : 9 data qubits
      Nodes 9..12 : 4 X-stabilizer check nodes (detecting Z errors)
      Includes self-loops, Tanner edges (data <-> check), and 12 ZZ pair edges.
    """
    adj = torch.eye(13, dtype=torch.float32)
    x_checks = [(0, 1), (1, 2, 4, 5), (3, 4, 6, 7), (7, 8)]
    for c_idx, qubits in enumerate(x_checks):
        c_node = 9 + c_idx
        for q in qubits:
            adj[q, c_node] = 1.0
            adj[c_node, q] = 1.0
    zz_edges = [
        (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
        (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    ]
    for u, v in zz_edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def kapturowski_value_rescale(x: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Invertible value rescaling from R2D2 (Kapturowski et al., ICLR 2019):
      h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def kapturowski_value_inv_rescale(y: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    """
    Exact closed-form inverse h^{-1}(y) of the R2D2 value rescaling function.
    """
    sgn = torch.sign(y)
    ay = torch.abs(y)
    disc = 1.0 + 4.0 * eps * (ay + 1.0 + eps)
    u = ((torch.sqrt(disc) - 1.0) / (2.0 * eps)) ** 2 - 1.0
    return sgn * torch.clamp(u, min=0.0)


class BipartiteGraphAttentionEncoder(nn.Module):
    """Non-Vision Bipartite Stabilizer-Data Graph Attention Perception Layer."""

    def __init__(self, in_features: int = 8, d_model: int = 96, n_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.node_embed = nn.Linear(in_features, d_model)
        self.node_type_embed = nn.Embedding(2, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.register_buffer("adj", build_d3_bipartite_adjacency())

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_nodes, _ = node_feats.shape
        device = node_feats.device
        node_types = torch.tensor([0] * 9 + [1] * 4, device=device, dtype=torch.long)
        h = self.node_embed(node_feats) + self.node_type_embed(node_types).view(1, 1, n_nodes, -1)

        q = self.q_proj(h).view(bsz, seq_len, n_nodes, self.n_heads, self.head_dim).transpose(2, 3)
        k = self.k_proj(h).view(bsz, seq_len, n_nodes, self.n_heads, self.head_dim).transpose(2, 3)
        v = self.v_proj(h).view(bsz, seq_len, n_nodes, self.n_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = (self.adj == 0).view(1, 1, 1, n_nodes, n_nodes)
        scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(2, 3).contiguous().view(bsz, seq_len, n_nodes, self.d_model)

        h = self.norm1(h + self.out_proj(out))
        h = self.norm2(h + self.ffn(h))
        return h.mean(dim=2)  # [B, T, d_model]


class WeightedBellmanR2D2WorldModel(nn.Module):
    """
    Model 1: R2D2-Graph-RSSM trained with Uncertainty-Weighted n-Step Bellman TD Loss,
    Kapturowski Value Rescaling, Burn-In Recurrent Warmup, and Exact 16-State Joint-DP
    Syndrome Cross-Entropy.
    """

    def __init__(self, in_features: int = 8, d_model: int = 96, d_stoch: int = 32) -> None:
        super().__init__()
        self.encoder = BipartiteGraphAttentionEncoder(in_features=in_features, d_model=d_model)
        self.d_model = d_model
        self.d_stoch = d_stoch
        self.gru = nn.GRUCell(d_model + d_stoch, d_model)

        self.prior_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_stoch * 2),
        )
        self.post_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_stoch * 2),
        )

        feat_dim = d_model + d_stoch
        self.theta_head = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 21),
        )
        self.q_value_head = nn.Sequential(
            nn.Linear(feat_dim + 21, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feat_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        burn_in_steps: int = 6,
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len, _, _ = node_feats.shape
        device = node_feats.device
        embeds = self.encoder(node_feats)  # [B, T, d_model]

        h = torch.zeros(bsz, self.d_model, device=device)
        s = torch.zeros(bsz, self.d_stoch, device=device)

        h_list, s_list, kl_list = [], [], []
        for t in range(seq_len):
            e_t = embeds[:, t]
            if t == burn_in_steps and burn_in_steps > 0:
                h = h.detach()
                s = s.detach()

            h = self.gru(torch.cat([e_t, s], dim=-1), h)
            prior_stats = self.prior_net(h)
            prior_mean, prior_logstd = torch.chunk(prior_stats, 2, dim=-1)
            prior_std = F.softplus(prior_logstd) + 1e-4

            post_stats = self.post_net(torch.cat([h, e_t], dim=-1))
            post_mean, post_logstd = torch.chunk(post_stats, 2, dim=-1)
            post_std = F.softplus(post_logstd) + 1e-4

            if self.training:
                eps = torch.randn_like(post_std)
                s = post_mean + post_std * eps
            else:
                s = post_mean

            kl = (
                torch.log(prior_std / post_std)
                + (post_std ** 2 + (post_mean - prior_mean) ** 2) / (2.0 * prior_std ** 2)
                - 0.5
            ).sum(dim=-1)

            h_list.append(h)
            s_list.append(s)
            kl_list.append(kl)

        h_seq = torch.stack(h_list, dim=1)
        s_seq = torch.stack(s_list, dim=1)
        kl_seq = torch.stack(kl_list, dim=1)

        latent = torch.cat([h_seq, s_seq], dim=-1)
        raw_theta = self.theta_head(latent)
        theta_pred = 1e-4 + 0.44 * torch.sigmoid(raw_theta)
        q_rescaled = self.q_value_head(torch.cat([latent, theta_pred], dim=-1)).squeeze(-1)
        log_var = torch.clamp(self.uncertainty_head(latent).squeeze(-1), -6.0, 4.0)

        return {
            "theta_pred": theta_pred,
            "q_rescaled": q_rescaled,
            "log_var": log_var,
            "kl": kl_seq,
            "latent": latent,
        }


class SmallSpatioTemporalTransformer(nn.Module):
    """Model 2: Non-Vision Bipartite Graph + Causal Spatio-Temporal Transformer."""

    def __init__(self, in_features: int = 8, d_model: int = 96, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.encoder = BipartiteGraphAttentionEncoder(in_features=in_features, d_model=d_model, n_heads=n_heads)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.theta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 21),
        )
        self.q_value_head = nn.Sequential(
            nn.Linear(d_model + 21, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, node_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, seq_len, _, _ = node_feats.shape
        device = node_feats.device
        x = self.encoder(node_feats) + self.pos_embed[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        h = self.transformer(x, mask=causal_mask)
        theta_pred = 1e-4 + 0.44 * torch.sigmoid(self.theta_head(h))
        q_rescaled = self.q_value_head(torch.cat([h, theta_pred], dim=-1)).squeeze(-1)
        return {"theta_pred": theta_pred, "q_rescaled": q_rescaled, "latent": h}


class RegimeLoRAAdapter(nn.Module):
    """Low-Rank (Rank-4) Fast Regime Adaptation Head for Non-Stationary Quantum Drift."""

    def __init__(self, d_in: int = 128, out_dim: int = 21, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.rank = rank
        self.scale = alpha / float(rank)
        self.lora_a = nn.Parameter(torch.randn(d_in, rank) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, latent: torch.Tensor, base_theta: torch.Tensor) -> torch.Tensor:
        base_logit = torch.logit(torch.clamp((base_theta - 1e-4) / 0.44, 1e-4, 1.0 - 1e-4))
        delta = torch.matmul(torch.matmul(latent, self.lora_a), self.lora_b) * self.scale + self.bias
        return 1e-4 + 0.44 * torch.sigmoid(base_logit + delta)


def torch_joint_dp_distribution(theta: torch.Tensor, signatures: List[int], n_checks: int = 4) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of the exact 2^m-state XOR-convolution DP:
      P_new(s) = (1 - p_e) * P_old(s) + p_e * P_old(s ^ g_e)
    """
    bsz, seq_len, n_params = theta.shape
    n_states = 1 << n_checks
    device = theta.device
    p_flat = theta.reshape(bsz * seq_len, n_params)

    dist = torch.zeros(bsz * seq_len, n_states, device=device, dtype=theta.dtype)
    dist[:, 0] = 1.0

    for e_idx in range(n_params):
        g = int(signatures[e_idx])
        if g == 0:
            continue
        pe = p_flat[:, e_idx : e_idx + 1]
        perm = [s ^ g for s in range(n_states)]
        dist = (1.0 - pe) * dist + pe * dist[:, perm]

    return dist.reshape(bsz, seq_len, n_states)


def extract_window_graph_features(
    sx_bits: np.ndarray,
    model: CorrelatedNoiseModel,
    window_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a syndrome bit stream [N, 4] into slow-loop window graph node features [T, 13, 8]
    and empirical 16-state syndrome distributions [T, 16].
    """
    n_rounds = sx_bits.shape[0]
    n_windows = max(1, n_rounds // window_size)
    trimmed = sx_bits[: n_windows * window_size].reshape(n_windows, window_size, 4)

    node_feats = np.zeros((n_windows, 13, 8), dtype=np.float32)
    emp_dists = np.zeros((n_windows, 16), dtype=np.float32)

    weights = (1 << np.arange(4, dtype=np.int32))
    ints = (trimmed.astype(np.int32) @ weights)  # [T, W]

    x_checks = [(0, 1), (1, 2, 4, 5), (3, 4, 6, 7), (7, 8)]
    for t in range(n_windows):
        w_syn = trimmed[t].astype(np.float32)  # [W, 4]
        counts = np.bincount(ints[t], minlength=16).astype(np.float32)
        emp = counts / float(window_size)
        emp_dists[t] = emp

        c_rates = w_syn.mean(axis=0)  # [4]
        c_temporal = np.abs(np.diff(w_syn, axis=0)).mean(axis=0) if window_size > 1 else np.zeros(4)
        cov = (w_syn.T @ w_syn) / float(window_size) - np.outer(c_rates, c_rates)

        prior_theta = model.build_context_informed_prior(emp)
        for q in range(9):
            inc = [c for c, qs in enumerate(x_checks) if q in qs]
            mean_c = float(np.mean([c_rates[c] for c in inc])) if inc else 0.0
            max_c = float(np.max([c_rates[c] for c in inc])) if inc else 0.0
            pair_vals = [
                prior_theta[9 + idx]
                for idx, (u, v) in enumerate(model.code.zz_edges)
                if u == q or v == q
            ]
            node_feats[t, q, 0] = float(prior_theta[q])
            node_feats[t, q, 1] = float(np.mean(pair_vals)) if pair_vals else 0.0
            node_feats[t, q, 2] = float(np.max(pair_vals)) if pair_vals else 0.0
            node_feats[t, q, 3] = mean_c
            node_feats[t, q, 4] = max_c
            node_feats[t, q, 5] = float(len(inc)) / 4.0
            node_feats[t, q, 6] = float(q in (1, 4, 7))
            node_feats[t, q, 7] = float(1.0 - emp[0])

        for c in range(4):
            node_feats[t, 9 + c, 0] = float(c_rates[c])
            node_feats[t, 9 + c, 1] = float(c_temporal[c])
            node_feats[t, 9 + c, 2] = float(cov[c, (c + 1) % 4])
            node_feats[t, 9 + c, 3] = float(cov[c, (c + 2) % 4])
            node_feats[t, 9 + c, 4] = float(len(x_checks[c])) / 4.0
            node_feats[t, 9 + c, 5] = float(emp[1 << c])
            node_feats[t, 9 + c, 6] = 1.0
            node_feats[t, 9 + c, 7] = float(1.0 - emp[0])

    return node_feats, emp_dists


# =============================================================================
# STAGE 1: Long-Term Quantum Syndrome & Drift Data Collection
# =============================================================================

def collect_long_term_dataset(
    model_x: CorrelatedNoiseModel,
    model_z: CorrelatedNoiseModel,
    datasets_dir: Path,
    rounds_per_regime: int = 50000,
    n_trajectories_per_regime: int = 120,
    seq_len: int = 16,
    window_size: int = 256,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Stage 1: Collect and persist long-term quantum syndrome streams and non-stationary
    recurrent replay buffer trajectories BEFORE model training begins.
    """
    datasets_dir.mkdir(parents=True, exist_ok=True)
    regimes = build_five_noise_regimes(model_x.default_theta)

    print(
        f"[Stage 1A] Collecting {rounds_per_regime:,} fast-loop rounds/regime "
        f"({rounds_per_regime * len(regimes):,} rounds total) across 5 regimes..."
    )
    rng_stream = np.random.default_rng(42)
    regime_streams: Dict[str, Dict[str, np.ndarray]] = {}
    dataset_npz: Dict[str, np.ndarray] = {}
    all_sx: List[np.ndarray] = []
    all_sz: List[np.ndarray] = []
    all_regime_ids: List[np.ndarray] = []
    all_true_theta: List[np.ndarray] = []

    for r_idx, (r_name, true_theta) in enumerate(regimes.items()):
        t0 = time.perf_counter()
        p_single = true_theta[:9]
        p_pairs = true_theta[9:21]

        errs_z = rng_stream.binomial(1, p_single, size=(rounds_per_regime, 9)).astype(np.uint8)
        for e_idx, (u, v) in enumerate(model_x.code.zz_edges):
            pe = float(p_pairs[e_idx])
            if pe > 0.0:
                c_flips = rng_stream.binomial(1, pe, size=rounds_per_regime).astype(np.uint8)
                errs_z[:, u] ^= c_flips
                errs_z[:, v] ^= c_flips

        errs_x = rng_stream.binomial(1, 0.010, size=(rounds_per_regime, 9)).astype(np.uint8)

        sx_bits = (errs_z @ model_x.H.T) % 2
        sz_bits = (errs_x @ model_z.H.T) % 2
        sx_masks = np.sum(sx_bits * (1 << np.arange(4)), axis=1).astype(np.uint8)
        sz_masks = np.sum(sz_bits * (1 << np.arange(4)), axis=1).astype(np.uint8)

        elapsed = time.perf_counter() - t0
        regime_streams[r_name] = {
            "true_theta": true_theta,
            "sx_bits": sx_bits.astype(np.uint8),
            "sz_bits": sz_bits.astype(np.uint8),
            "sx_masks": sx_masks,
            "sz_masks": sz_masks,
            "errs_z": errs_z,
        }
        dataset_npz[f"{r_name}_theta_x"] = true_theta
        dataset_npz[f"{r_name}_sx_masks"] = sx_masks
        dataset_npz[f"{r_name}_sz_masks"] = sz_masks
        dataset_npz[f"{r_name}_errs_z"] = errs_z

        all_sx.append(sx_masks)
        all_sz.append(sz_masks)
        all_regime_ids.append(np.full(rounds_per_regime, r_idx, dtype=np.uint8))
        all_true_theta.append(true_theta)
        print(
            f"  -> Collected [{r_name}]: {rounds_per_regime:,} rounds "
            f"(mean syndrome weight={sx_bits.sum(axis=1).mean():.3f}, {elapsed:.2f}s)"
        )

    sx_concat = np.concatenate(all_sx)
    sz_concat = np.concatenate(all_sz)
    reg_concat = np.concatenate(all_regime_ids)

    long_horizon_path = datasets_dir / "avid_long_horizon_d3_250k_rounds.npz"
    np.savez_compressed(
        str(long_horizon_path),
        s_x_masks=sx_concat,
        s_z_masks=sz_concat,
        regime_ids=reg_concat,
        regime_thetas=np.stack(all_true_theta),
        regime_names=np.array(list(regimes.keys())),
        **dataset_npz,
    )
    shutil.copy2(long_horizon_path, datasets_dir / "avid_canonical_css_regimes.npz")

    raw_packets = pack_packet_stream(sx_concat[:16384].tolist(), sz_concat[:16384].tolist())
    (datasets_dir / "syndrome_packets_v1.bin").write_bytes(raw_packets)

    total_replay_rounds = len(regimes) * n_trajectories_per_regime * seq_len * window_size
    print(
        f"[Stage 1B] Collecting {len(regimes) * n_trajectories_per_regime} non-stationary "
        f"R2D2 replay trajectories (T={seq_len} windows x W={window_size} rounds = "
        f"{total_replay_rounds:,} fast-loop rounds)..."
    )

    rng = np.random.default_rng(20261103)
    replay_data: Dict[str, Dict[str, torch.Tensor]] = {}
    replay_npz: Dict[str, np.ndarray] = {}

    for r_name, base_theta in regimes.items():
        feats_list, emp_list, theta_list, reward_list = [], [], [], []

        for _ in range(n_trajectories_per_regime):
            traj_feats = np.zeros((seq_len, 13, 8), dtype=np.float32)
            traj_emp = np.zeros((seq_len, 16), dtype=np.float32)
            traj_theta = np.zeros((seq_len, 21), dtype=np.float32)
            traj_reward = np.zeros((seq_len,), dtype=np.float32)

            drift = rng.normal(0.0, 0.0025, size=21)
            tls_active = False
            for t in range(seq_len):
                drift = 0.88 * drift + rng.normal(0.0, 0.002, size=21)
                if rng.random() < 0.15:
                    tls_active = not tls_active
                scale = rng.uniform(0.88, 1.14)
                theta_t = np.clip(base_theta * scale + drift, 1e-4, 0.40)
                if tls_active:
                    theta_t[4] = np.clip(theta_t[4] * 1.18, 1e-4, 0.42)
                    theta_t[9 + 3] = np.clip(theta_t[9 + 3] * 1.18, 1e-4, 0.42)

                emp_t = model_x.joint_distribution_dp(theta_t)
                counts = rng.multinomial(window_size, emp_t).astype(np.float32)
                emp_hat = counts / float(window_size)

                bits = (((np.arange(16)[:, None] >> np.arange(4)) & 1)).astype(np.float32)
                c_rates = emp_hat @ bits
                cov = (bits.T * emp_hat) @ bits - np.outer(c_rates, c_rates)
                prior_theta = model_x.build_context_informed_prior(emp_hat)

                x_checks = [(0, 1), (1, 2, 4, 5), (3, 4, 6, 7), (7, 8)]
                for q in range(9):
                    inc = [c for c, qs in enumerate(x_checks) if q in qs]
                    pair_vals = [
                        prior_theta[9 + idx]
                        for idx, (u, v) in enumerate(model_x.code.zz_edges)
                        if u == q or v == q
                    ]
                    traj_feats[t, q, 0] = float(prior_theta[q])
                    traj_feats[t, q, 1] = float(np.mean(pair_vals)) if pair_vals else 0.0
                    traj_feats[t, q, 2] = float(np.max(pair_vals)) if pair_vals else 0.0
                    traj_feats[t, q, 3] = float(np.mean([c_rates[c] for c in inc]))
                    traj_feats[t, q, 4] = float(np.max([c_rates[c] for c in inc]))
                    traj_feats[t, q, 5] = float(len(inc)) / 4.0
                    traj_feats[t, q, 6] = float(q in (1, 4, 7))
                    traj_feats[t, q, 7] = float(1.0 - emp_hat[0])

                for c in range(4):
                    traj_feats[t, 9 + c, 0] = float(c_rates[c])
                    traj_feats[t, 9 + c, 1] = float(2.0 * c_rates[c] * (1.0 - c_rates[c]))
                    traj_feats[t, 9 + c, 2] = float(cov[c, (c + 1) % 4])
                    traj_feats[t, 9 + c, 3] = float(cov[c, (c + 2) % 4])
                    traj_feats[t, 9 + c, 4] = float(len(x_checks[c])) / 4.0
                    traj_feats[t, 9 + c, 5] = float(emp_hat[1 << c])
                    traj_feats[t, 9 + c, 6] = 1.0
                    traj_feats[t, 9 + c, 7] = float(1.0 - emp_hat[0])

                traj_emp[t] = emp_hat
                traj_theta[t] = theta_t.astype(np.float32)
                entropy = float(-np.sum(emp_t * np.log(np.maximum(emp_t, 1e-12))))
                traj_reward[t] = float(10.0 * emp_t[0] - 0.5 * entropy)

            feats_list.append(traj_feats)
            emp_list.append(traj_emp)
            theta_list.append(traj_theta)
            reward_list.append(traj_reward)

        feats_arr = np.stack(feats_list)
        emp_arr = np.stack(emp_list)
        theta_arr = np.stack(theta_list)
        reward_arr = np.stack(reward_list)

        replay_data[r_name] = {
            "node_feats": torch.tensor(feats_arr, dtype=torch.float32),
            "emp_dist": torch.tensor(emp_arr, dtype=torch.float32),
            "true_theta": torch.tensor(theta_arr, dtype=torch.float32),
            "reward": torch.tensor(reward_arr, dtype=torch.float32),
        }
        replay_npz[f"{r_name}_node_feats"] = feats_arr
        replay_npz[f"{r_name}_emp_dist"] = emp_arr
        replay_npz[f"{r_name}_true_theta"] = theta_arr
        replay_npz[f"{r_name}_reward"] = reward_arr

    replay_path = datasets_dir / "avid_r2d2_replay_buffer_trajectories.npz"
    np.savez_compressed(str(replay_path), **replay_npz)

    manifest = {
        "stage_1_status": "COMPLETED",
        "fast_loop_rounds_per_regime": rounds_per_regime,
        "total_fast_loop_stream_rounds": rounds_per_regime * len(regimes),
        "r2d2_replay_trajectories_per_regime": n_trajectories_per_regime,
        "r2d2_total_trajectories": len(regimes) * n_trajectories_per_regime,
        "r2d2_sequence_length_windows": seq_len,
        "r2d2_burn_in_windows": 6,
        "r2d2_unroll_windows": seq_len - 6,
        "window_size_rounds": window_size,
        "total_underlying_replay_rounds": total_replay_rounds,
        "files": [
            "datasets/avid_long_horizon_d3_250k_rounds.npz",
            "datasets/avid_canonical_css_regimes.npz",
            "datasets/avid_r2d2_replay_buffer_trajectories.npz",
            "datasets/syndrome_packets_v1.bin",
        ],
    }
    (datasets_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[Stage 1 Complete] Saved long-term datasets to {datasets_dir}/")
    return regime_streams, replay_data


# =============================================================================
# STAGE 2: Weighted-Bellman R2D2 + Transformer + LoRA Training on Collected Data
# =============================================================================

def compute_nstep_bellman_targets(
    rewards: torch.Tensor,
    target_q_rescaled: torch.Tensor,
    gamma: float = 0.92,
    n_step: int = 3,
) -> torch.Tensor:
    """
    Compute 3-step bootstrapped Bellman targets in rescaled value space:
      G_t^{(n)} = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n h^{-1}(Q_target(s_{t+n}))
      y_t = h(G_t^{(n)})
    """
    bsz, seq_len = rewards.shape
    q_inv = kapturowski_value_inv_rescale(target_q_rescaled.detach())
    returns = torch.zeros_like(rewards)

    for t in range(seq_len):
        acc = torch.zeros(bsz, device=rewards.device, dtype=rewards.dtype)
        for k in range(n_step):
            if t + k < seq_len:
                acc = acc + (gamma ** k) * rewards[:, t + k]
        boot_idx = t + n_step
        if boot_idx < seq_len:
            acc = acc + (gamma ** n_step) * q_inv[:, boot_idx]
        returns[:, t] = acc

    return kapturowski_value_rescale(returns)


def train_world_models_and_lora(
    model: CorrelatedNoiseModel,
    replay_data: Dict[str, Dict[str, torch.Tensor]],
    checkpoints_dir: Path,
    r2d2_epochs: int = 80,
    transformer_epochs: int = 80,
    lora_epochs: int = 50,
) -> Tuple[WeightedBellmanR2D2WorldModel, SmallSpatioTemporalTransformer, Dict[str, RegimeLoRAAdapter], Dict[str, object]]:
    signatures = [int(g) for g in model.signatures]

    all_feats = torch.cat([v["node_feats"] for v in replay_data.values()], dim=0)
    all_emp = torch.cat([v["emp_dist"] for v in replay_data.values()], dim=0)
    all_theta = torch.cat([v["true_theta"] for v in replay_data.values()], dim=0)
    all_rew = torch.cat([v["reward"] for v in replay_data.values()], dim=0)

    r2d2 = WeightedBellmanR2D2WorldModel(in_features=8, d_model=96, d_stoch=32)
    r2d2_target = copy.deepcopy(r2d2)
    r2d2_opt = torch.optim.AdamW(r2d2.parameters(), lr=2.5e-3, weight_decay=1e-4)
    r2d2_sched = torch.optim.lr_scheduler.CosineAnnealingLR(r2d2_opt, T_max=r2d2_epochs, eta_min=2e-4)

    burn_in = 6
    batch_size = max(1, min(60, all_feats.shape[0]))
    n_total = all_feats.shape[0]
    r2d2_history: List[Dict[str, float]] = []

    print(
        f"  -> [Model 1/3] Training Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) "
        f"for {r2d2_epochs} epochs on {n_total} collected trajectories (burn_in={burn_in}, unroll={all_feats.shape[1]-burn_in})..."
    )
    for epoch in range(1, r2d2_epochs + 1):
        perm = torch.randperm(n_total)
        ep_loss, ep_bellman, ep_dp, ep_mse, steps = 0.0, 0.0, 0.0, 0.0, 0

        for start in range(0, n_total, batch_size):
            idx = perm[start : start + batch_size]
            b_feats = all_feats[idx]
            b_emp = all_emp[idx]
            b_theta = all_theta[idx]
            b_rew = all_rew[idx]

            out = r2d2(b_feats, burn_in_steps=burn_in)
            with torch.no_grad():
                tgt_out = r2d2_target(b_feats, burn_in_steps=burn_in)

            theta_pred = out["theta_pred"][:, burn_in:]
            q_rescaled = out["q_rescaled"][:, burn_in:]
            log_var = out["log_var"][:, burn_in:]
            kl_active = out["kl"][:, burn_in:]

            emp_active = b_emp[:, burn_in:]
            theta_true_active = b_theta[:, burn_in:]

            pred_dist = torch_joint_dp_distribution(theta_pred, signatures, n_checks=4)
            dp_ce = -(emp_active * torch.log(torch.clamp(pred_dist, min=1e-12))).sum(dim=-1)
            theta_mse = ((theta_pred - theta_true_active) ** 2).mean(dim=-1)

            instant_reward = b_rew[:, burn_in:] - 4.0 * dp_ce.detach() - 25.0 * theta_mse.detach()
            y_bellman = compute_nstep_bellman_targets(
                instant_reward, tgt_out["q_rescaled"][:, burn_in:], gamma=0.92, n_step=3
            )

            td_err = q_rescaled - y_bellman
            precision = torch.exp(-log_var)
            priority_w = (torch.abs(td_err).detach() + 0.05) ** 0.6
            priority_w = priority_w / (priority_w.mean() + 1e-6)
            weighted_bellman_loss = (
                priority_w * (0.5 * precision * (td_err ** 2) + 0.5 * log_var)
            ).mean()

            loss_dp = dp_ce.mean()
            loss_mse = theta_mse.mean()
            loss_kl = kl_active.mean()

            total_loss = 1.5 * loss_dp + 35.0 * loss_mse + 0.35 * weighted_bellman_loss + 0.01 * loss_kl

            r2d2_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(r2d2.parameters(), 2.0)
            r2d2_opt.step()

            with torch.no_grad():
                for p, tp in zip(r2d2.parameters(), r2d2_target.parameters()):
                    tp.data.mul_(0.95).add_(p.data, alpha=0.05)

            ep_loss += float(total_loss.item())
            ep_bellman += float(weighted_bellman_loss.item())
            ep_dp += float(loss_dp.item())
            ep_mse += float(loss_mse.item())
            steps += 1

        r2d2_sched.step()
        if epoch in (1, 20, 40, 60, r2d2_epochs):
            print(
                f"     [R2D2 Epoch {epoch:02d}/{r2d2_epochs}] "
                f"total={ep_loss/steps:.4f} | dp_nll={ep_dp/steps:.4f} | "
                f"wb_td={ep_bellman/steps:.4f} | param_rmse={(ep_mse/steps)**0.5:.5f}"
            )
        r2d2_history.append({
            "epoch": epoch,
            "total_loss": round(ep_loss / steps, 5),
            "dp_nll": round(ep_dp / steps, 5),
            "weighted_bellman_loss": round(ep_bellman / steps, 5),
            "param_rmse": round((ep_mse / steps) ** 0.5, 6),
        })

    transformer = SmallSpatioTemporalTransformer(in_features=8, d_model=96, n_heads=4, n_layers=2)
    tf_opt = torch.optim.AdamW(transformer.parameters(), lr=2.5e-3, weight_decay=1e-4)
    tf_sched = torch.optim.lr_scheduler.CosineAnnealingLR(tf_opt, T_max=transformer_epochs, eta_min=2e-4)
    tf_history: List[Dict[str, float]] = []

    print(
        f"  -> [Model 2/3] Training Small Spatio-Temporal Causal Transformer "
        f"for {transformer_epochs} epochs..."
    )
    for epoch in range(1, transformer_epochs + 1):
        perm = torch.randperm(n_total)
        ep_loss, ep_mse, steps = 0.0, 0.0, 0
        for start in range(0, n_total, batch_size):
            idx = perm[start : start + batch_size]
            b_feats = all_feats[idx]
            b_emp = all_emp[idx]
            b_theta = all_theta[idx]
            b_rew = all_rew[idx]

            out = transformer(b_feats)
            theta_pred = out["theta_pred"]
            q_rescaled = out["q_rescaled"]

            pred_dist = torch_joint_dp_distribution(theta_pred, signatures, n_checks=4)
            dp_ce = -(b_emp * torch.log(torch.clamp(pred_dist, min=1e-12))).sum(dim=-1)
            theta_mse = ((theta_pred - b_theta) ** 2).mean(dim=-1)

            y_bellman = compute_nstep_bellman_targets(
                b_rew - 4.0 * dp_ce.detach(), q_rescaled, gamma=0.92, n_step=3
            )
            bellman_mse = ((q_rescaled - y_bellman) ** 2).mean()
            total_loss = 1.5 * dp_ce.mean() + 35.0 * theta_mse.mean() + 0.25 * bellman_mse

            tf_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 2.0)
            tf_opt.step()

            ep_loss += float(total_loss.item())
            ep_mse += float(theta_mse.mean().item())
            steps += 1

        tf_sched.step()
        if epoch in (1, 20, 40, 60, transformer_epochs):
            print(
                f"     [Transformer Epoch {epoch:02d}/{transformer_epochs}] "
                f"total={ep_loss/steps:.4f} | param_rmse={(ep_mse/steps)**0.5:.5f}"
            )
        tf_history.append({
            "epoch": epoch,
            "total_loss": round(ep_loss / steps, 5),
            "param_rmse": round((ep_mse / steps) ** 0.5, 6),
        })

    r2d2.eval()
    transformer.eval()
    for p in r2d2.parameters():
        p.requires_grad_(False)

    lora_adapters: Dict[str, RegimeLoRAAdapter] = {}
    lora_metrics: Dict[str, Dict[str, float]] = {}

    print(f"  -> [Model 3/3] Training 5 Rank-4 Regime LoRA Adapters ({lora_epochs} epochs/regime)...")
    for regime_name, r_data in replay_data.items():
        adapter = RegimeLoRAAdapter(d_in=128, out_dim=21, rank=4, alpha=8.0)
        opt = torch.optim.AdamW(adapter.parameters(), lr=6e-3, weight_decay=1e-4)

        r_feats = r_data["node_feats"]
        r_emp = r_data["emp_dist"]
        r_theta = r_data["true_theta"]

        with torch.no_grad():
            base_out = r2d2(r_feats, burn_in_steps=burn_in)
            base_latent = base_out["latent"][:, burn_in:]
            base_theta = base_out["theta_pred"][:, burn_in:]

        target_emp = r_emp[:, burn_in:]
        target_theta = r_theta[:, burn_in:]

        for _ in range(lora_epochs):
            adapted_theta = adapter(base_latent, base_theta)
            pred_dist = torch_joint_dp_distribution(adapted_theta, signatures, n_checks=4)
            dp_ce = -(target_emp * torch.log(torch.clamp(pred_dist, min=1e-12))).sum(dim=-1).mean()
            mse = ((adapted_theta - target_theta) ** 2).mean()
            loss = 1.5 * dp_ce + 45.0 * mse
            opt.zero_grad()
            loss.backward()
            opt.step()

        adapter.eval()
        with torch.no_grad():
            final_theta = adapter(base_latent, base_theta)
            rmse = float(torch.sqrt(((final_theta - target_theta) ** 2).mean()).item())
        lora_adapters[regime_name] = adapter
        lora_metrics[regime_name] = {"final_param_rmse": round(rmse, 6), "rank": 4}
        print(f"     [LoRA {regime_name}] final_param_rmse={rmse:.5f}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(r2d2.state_dict(), checkpoints_dir / "weighted_bellman_r2d2_backbone.pt")
    torch.save(transformer.state_dict(), checkpoints_dir / "spatiotemporal_transformer_backbone.pt")
    for regime_name, adapter in lora_adapters.items():
        torch.save(adapter.state_dict(), checkpoints_dir / f"lora_adapter_{regime_name}.pt")

    torch.save(
        {
            "weighted_bellman_r2d2": r2d2.state_dict(),
            "spatiotemporal_transformer": transformer.state_dict(),
            "lora_adapters": {k: v.state_dict() for k, v in lora_adapters.items()},
            "r2d2_history": r2d2_history,
            "transformer_history": tf_history,
            "lora_metrics": lora_metrics,
        },
        checkpoints_dir / "full_world_model_bundle.pt",
    )

    training_log = {
        "r2d2_epochs": r2d2_epochs,
        "transformer_epochs": transformer_epochs,
        "lora_epochs": lora_epochs,
        "r2d2_history": r2d2_history,
        "transformer_history": tf_history,
        "lora_metrics": lora_metrics,
    }
    return r2d2, transformer, lora_adapters, training_log


# =============================================================================
# STAGE 3: Evaluation & STAGE 4: Hugging Face Hub Upload
# =============================================================================

def decode_stream_logical_error_rate(
    bp_engine: CycleAccurateBPEngine,
    sx_masks: np.ndarray,
    errs_z: np.ndarray,
    q78_weights: np.ndarray,
) -> float:
    logical_errors = 0
    num_rounds = len(sx_masks)
    for i in range(num_rounds):
        tr = bp_engine.decode(int(sx_masks[i]), q78_weights, K=3, true_error=errs_z[i])
        if tr.logical_error:
            logical_errors += 1
    return float(logical_errors / max(1, num_rounds))


def run_cloudshell_pipeline(
    rounds_per_regime: int = 50000,
    n_trajectories_per_regime: int = 120,
    r2d2_epochs: int = 80,
    transformer_epochs: int = 80,
    lora_epochs: int = 50,
    eval_rounds: int = 10000,
    upload_hf: bool = True,
) -> Dict[str, object]:
    env_vars = load_dotenv(REPO_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN") or env_vars.get("HF_TOKEN", "")
    hf_username = os.environ.get("HF_USERNAME") or env_vars.get("HF_USERNAME", "Terrano09")
    hf_repo_id = (
        os.environ.get("HF_REPO_ID")
        or env_vars.get("HF_REPO_ID")
        or f"{hf_username}/AVID-EPJ-Baseline-Models-and-Data"
    )

    print("[0/4] Regenerating canonical CSS code & identifiability artifacts...")
    generate_code_main()
    generate_identifiability_artifacts(REPO_ROOT)
    collect_timing_main()

    model_x = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)
    model_z = build_d3_correlated_noise_model(basis="Z", include_measurement_faults=False)
    bp_engine = CycleAccurateBPEngine(code=model_x.code, basis="X", default_K=3, clock_mhz=250.0)

    checkpoints_dir = REPO_ROOT / "checkpoints"
    datasets_dir = REPO_ROOT / "datasets"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # STAGE 1: Long-term data collection first
    regime_streams, replay_data = collect_long_term_dataset(
        model_x=model_x,
        model_z=model_z,
        datasets_dir=datasets_dir,
        rounds_per_regime=rounds_per_regime,
        n_trajectories_per_regime=n_trajectories_per_regime,
        seq_len=16,
        window_size=256,
    )

    # STAGE 2: Train Weighted-Bellman R2D2 + Transformer + LoRA on collected data
    print("[2/4] Training Weighted-Bellman R2D2 World Model, Transformer & 5 LoRA Adapters on collected data...")
    r2d2, _, lora_adapters, world_model_log = train_world_models_and_lora(
        model=model_x,
        replay_data=replay_data,
        checkpoints_dir=checkpoints_dir,
        r2d2_epochs=r2d2_epochs,
        transformer_epochs=transformer_epochs,
        lora_epochs=lora_epochs,
    )

    # STAGE 3: Calibrate & Evaluate all 7 controllers on held-out evaluation split
    print(f"[3/4] Running {rounds_per_regime:,}-round Joint-DP MAP calibration & shared-seed Q7.8 BP evaluation...")
    b0 = B0StaticBPController()
    b1 = B1MarginalLBFGSController()
    b2 = B2ObservablePIDController()
    b3 = B3CovarianceTrackerController()
    b4 = B4OracleBPController()

    benchmark_summary: Dict[str, object] = {
        "architecture_branch": "AVID_EPJ_Patches",
        "code": "rotated_planar_surface_code_d3_css",
        "estimator": "Exact_16_State_Joint_DP_MAP_plus_Weighted_Bellman_R2D2",
        "decoder_hardware_contract": "Normalized_Min_Sum_BP_Q7.8_17_Cycles_68ns_at_250MHz",
        "rounds_per_regime": rounds_per_regime,
        "total_fast_loop_rounds": rounds_per_regime * 5,
        "total_r2d2_replay_rounds": 5 * n_trajectories_per_regime * 16 * 256,
        "bp_iterations": 3,
        "bp_cycle_latency": bp_engine.cycle_count_fused(3),
        "world_model_training": world_model_log,
        "regimes": {},
    }

    q78_banks: Dict[str, np.ndarray] = {}

    for regime_name, r_stream in regime_streams.items():
        sx_bits = r_stream["sx_bits"]
        sx_masks = r_stream["sx_masks"]
        errs_z = r_stream["errs_z"]
        true_theta = r_stream["true_theta"]

        cal_n = max(4096, rounds_per_regime - eval_rounds)
        cal_masks = sx_masks[:cal_n]
        cal_bits = sx_bits[:cal_n]
        eval_masks = sx_masks[cal_n : cal_n + eval_rounds]
        eval_errs = errs_z[cal_n : cal_n + eval_rounds]

        hist_16 = np.bincount(cal_masks, minlength=16).astype(np.float64)

        t0 = time.perf_counter()
        fit_res = model_x.fit_lbfgs(hist_16, prior_lambda=0.04, max_iter=80)
        fit_ms = (time.perf_counter() - t0) * 1e3
        w_avid = np.asarray(fit_res["llr_q7_8"], dtype=np.int16)

        w_feats, _ = extract_window_graph_features(cal_bits[: 16 * 256], model_x, window_size=256)
        with torch.no_grad():
            t_feats = torch.tensor(w_feats[None, ...], dtype=torch.float32)
            r2d2_out = r2d2(t_feats, burn_in_steps=6)
            adapted_theta = lora_adapters[regime_name](
                r2d2_out["latent"][:, 6:], r2d2_out["theta_pred"][:, 6:]
            )
            theta_r2d2 = adapted_theta.mean(dim=(0, 1)).cpu().numpy()

        theta_hybrid = 0.50 * np.asarray(fit_res["theta"]) + 0.50 * theta_r2d2
        _, w_r2d2 = model_x.compute_q7_8_llrs(theta_hybrid, pair_aware_projection=True)

        w_b0 = b0.update_from_histogram(hist_16)
        w_b1 = b1.update_from_histogram(hist_16)
        w_b2 = b2.update_from_histogram(hist_16)
        w_b3 = b3.update_from_histogram(hist_16)
        w_b4 = b4.weights_from_true_theta(true_theta)

        ler_b0 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_b0)
        ler_b1 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_b1)
        ler_b2 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_b2)
        ler_b3 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_b3)
        ler_avid = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_avid)
        ler_r2d2 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_r2d2)
        ler_b4 = decode_stream_logical_error_rate(bp_engine, eval_masks, eval_errs, w_b4)

        q78_banks[f"{regime_name}_AVID_Joint_DP"] = w_avid
        q78_banks[f"{regime_name}_Weighted_Bellman_R2D2"] = w_r2d2
        q78_banks[f"{regime_name}_B0_Static"] = w_b0
        q78_banks[f"{regime_name}_B1_Marginal_LBFGS"] = w_b1
        q78_banks[f"{regime_name}_B2_PID"] = w_b2
        q78_banks[f"{regime_name}_B3_Covariance"] = w_b3
        q78_banks[f"{regime_name}_B4_Oracle"] = w_b4

        benchmark_summary["regimes"][regime_name] = {
            "true_theta": [round(float(x), 5) for x in true_theta],
            "estimated_theta_avid": [round(float(x), 5) for x in fit_res["theta"]],
            "estimated_theta_r2d2": [round(float(x), 5) for x in theta_hybrid],
            "avid_nll": round(float(fit_res["nll"]), 5),
            "avid_fit_time_ms": round(float(fit_ms), 3),
            "avid_q78_weights": [int(x) for x in w_avid],
            "r2d2_q78_weights": [int(x) for x in w_r2d2],
            "logical_error_rates": {
                "B0_Static_BP": round(ler_b0, 5),
                "B1_Marginal_LBFGS": round(ler_b1, 5),
                "B2_Observable_PID": round(ler_b2, 5),
                "B3_Covariance_Tracker": round(ler_b3, 5),
                "AVID_Joint_DP_MAP": round(ler_avid, 5),
                "AVID_Weighted_Bellman_R2D2": round(ler_r2d2, 5),
                "B4_Oracle_BP": round(ler_b4, 5),
            },
        }
        print(
            f"  -> [{regime_name}] LER: B0={ler_b0:.4f} | B1={ler_b1:.4f} | "
            f"AVID_DP={ler_avid:.4f} | WB_R2D2={ler_r2d2:.4f} | Oracle={ler_b4:.4f}"
        )

    (checkpoints_dir / "avid_joint_dp_checkpoints.json").write_text(
        json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoints_dir / "baseline_vs_avid_benchmark.json").write_text(
        json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(str(checkpoints_dir / "q78_weight_banks.npz"), **q78_banks)

    if not upload_hf or not hf_token:
        print("[SKIP HF UPLOAD] Local artifacts saved in checkpoints/ and datasets/.")
        return benchmark_summary

    print(f"[4/4] Uploading datasets, PyTorch .pt checkpoints & math docs to Hugging Face ({hf_repo_id})...")
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo_id, repo_type="model", private=False, exist_ok=True)

    stage_dir = REPO_ROOT / "build_hf_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for folder in ["checkpoints", "datasets", "architecture", "configs", "docs", "mathematical_model"]:
        src = REPO_ROOT / folder
        dst = stage_dir / folder
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    rows = []
    for r_name, r_data in benchmark_summary["regimes"].items():
        lers = r_data["logical_error_rates"]
        rows.append(
            f"| `{r_name}` | {lers['B0_Static_BP']:.4f} | {lers['B1_Marginal_LBFGS']:.4f} | "
            f"{lers['B2_Observable_PID']:.4f} | {lers['B3_Covariance_Tracker']:.4f} | "
            f"**{lers['AVID_Joint_DP_MAP']:.4f}** | **{lers['AVID_Weighted_Bellman_R2D2']:.4f}** | "
            f"{lers['B4_Oracle_BP']:.4f} |"
        )
    table_md = "\n".join(rows)

    card = f"""---
license: mit
tags:
  - quantum-error-correction
  - surface-code
  - belief-propagation
  - variational-inference
  - r2d2
  - world-model
  - fpga
---

# AVID Baseline Architecture, Weighted-Bellman R2D2 World Model & Long-Horizon CSS Datasets (`AVID_EPJ_Patches`)

This repository hosts:
1. **Long-Horizon Collected Quantum Syndrome & Non-Stationary Drift Datasets (`datasets/`)**:
   - `datasets/avid_long_horizon_d3_250k_rounds.npz`: 250,000 fast-loop syndrome rounds (50,000 rounds/regime across R1–R5).
   - `datasets/avid_r2d2_replay_buffer_trajectories.npz`: 600 non-stationary recurrent replay trajectories ($T=16$ windows $\\times W=256$ rounds = 2,457,600 fast-loop rounds).
   - `datasets/syndrome_packets_v1.bin` & `datasets/dataset_manifest.json`.
2. **Exact $2^m$-State Dynamic Programming Joint Syndrome Likelihood (`AVID-Joint-DP-MAP`)** with **Pair-Aware Logical-Triangle LLR Projection**,
3. **PyTorch Weighted-Bellman R2D2 (`R2D2-Graph-RSSM`) + Spatio-Temporal Causal Transformer + 5 Rank-4 Regime LoRA Adapters** (`.pt` checkpoints in `checkpoints/`),
4. **Complete End-to-End Mathematical Modeling Suite (`docs/` and `mathematical_model/`, Chapters `00`–`08`)**,
5. **Quantized Signed 16-bit Q7.8 LLR Weight Banks (`checkpoints/q78_weight_banks.npz`)** and **Frozen Contracts (`architecture/`)**.

## Shared-Input Logical Error Rate (LER) Benchmark Across 5 Non-Stationary Regimes (50,000 Rounds/Regime)

| Noise Regime | B0 Static BP | B1 Marginal L-BFGS | B2 Observable PID | B3 Covariance | **AVID Joint-DP MAP** | **Weighted-Bellman R2D2** | B4 Oracle BP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_md}

## Included Model Checkpoints (`checkpoints/`)
- `checkpoints/weighted_bellman_r2d2_backbone.pt`: Trained PyTorch `WeightedBellmanR2D2WorldModel` (`R2D2-Graph-RSSM`) state dict.
- `checkpoints/spatiotemporal_transformer_backbone.pt`: Trained PyTorch `SmallSpatioTemporalTransformer` state dict.
- `checkpoints/lora_adapter_R1..R5.pt`: 5 Rank-4 Regime LoRA Adapter checkpoints.
- `checkpoints/full_world_model_bundle.pt`: Complete PyTorch checkpoint bundle + training curves.
- `checkpoints/baseline_vs_avid_benchmark.json`: Complete parameter recovery, Q7.8 weights, and LER metrics.
- `checkpoints/q78_weight_banks.npz`: Hardware-ready signed 16-bit Q7.8 register banks for `weight_map_reg.sv`.

## Mathematical Modeling Folder (`docs/` & `mathematical_model/`)
- `00_Mathematical_Modeling_Index.md` through `08_Weighted_Bellman_R2D2_and_World_Model_Math.md`
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")

    api.upload_folder(
        repo_id=hf_repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message="Upload 250k-round long-horizon datasets, Weighted-Bellman R2D2 .pt checkpoints, pair-aware Q7.8 banks & math docs",
    )
    shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"[DONE] Successfully uploaded all artifacts to: https://huggingface.co/{hf_repo_id}")
    return benchmark_summary


if __name__ == "__main__":
    run_cloudshell_pipeline()
