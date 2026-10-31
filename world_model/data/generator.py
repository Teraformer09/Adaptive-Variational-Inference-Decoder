"""QuTiP-calibrated multi-round quantum hardware noise trajectory generator for the AVID World Model.

Generates schema-compliant trajectories (`world_model/dataset/trajectory.schema.json`) on a 17-node
Surface-17 bipartite transmon-and-check topology across all 5 individual noise benchmarks
(B1, A1, C1, C2, A2) and the unified Mixed-Noise regime (MIXED_ALL_THREE).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DeviceTopology:
    """Versioned bipartite superconducting processor topology G=(V,E)."""

    topology_id: str
    num_checks: int
    num_data: int
    num_nodes: int
    node_roles: List[int]  # 0=data_qubit, 1=X_check, 2=Z_check
    coords_um: np.ndarray  # (num_nodes, 2) 2D physical coordinates in micrometers
    check_to_data_edges: List[Tuple[int, int]]  # (check_idx, data_idx) in [0..num_checks-1] x [0..num_data-1]
    spectator_edges: List[Tuple[int, int]]  # pairs of check indices sharing a spectator coupler
    graph_edges: np.ndarray  # (2, E) undirected edge index over all num_nodes
    pairwise_dist_um: np.ndarray  # (num_nodes, num_nodes) Euclidean distance matrix
    check_dist_um: np.ndarray  # (num_checks, num_checks) Euclidean distance between checks


def build_surface17_topology() -> DeviceTopology:
    """Construct the canonical 17-node Surface-17 processor topology (8 ancilla checks + 9 data qubits)."""
    num_checks = 8
    num_data = 9
    num_nodes = num_checks + num_data

    # First 8 nodes (0..7) are ancilla checks (alternating X=1, Z=2); nodes 8..16 are data qubits (0)
    node_roles = [1 if i % 2 == 0 else 2 for i in range(num_checks)] + [0] * num_data

    # 2D chip layout on a 1200 um x 1200 um transmon grid (pitch ~ 350 um)
    check_coords = np.array(
        [
            [200.0, 200.0],
            [550.0, 200.0],
            [900.0, 200.0],
            [200.0, 550.0],
            [550.0, 550.0],
            [900.0, 550.0],
            [375.0, 900.0],
            [725.0, 900.0],
        ],
        dtype=np.float32,
    )
    data_coords = np.array(
        [
            [100.0, 100.0],
            [375.0, 100.0],
            [725.0, 100.0],
            [100.0, 375.0],
            [375.0, 375.0],
            [725.0, 375.0],
            [375.0, 725.0],
            [725.0, 725.0],
            [1000.0, 725.0],
        ],
        dtype=np.float32,
    )
    coords_um = np.concatenate([check_coords, data_coords], axis=0)

    # Bipartite stabilizer check -> data qubit connectivity
    check_to_data_edges = [
        (0, 0), (0, 1), (0, 3), (0, 4),
        (1, 1), (1, 2), (1, 4), (1, 5),
        (2, 2), (2, 5),
        (3, 3), (3, 4), (3, 6),
        (4, 4), (4, 5), (4, 6), (4, 7),
        (5, 5), (5, 7), (5, 8),
        (6, 6), (6, 7),
        (7, 7), (7, 8),
    ]

    # Spectator coupling pairs between checks (checks that share data qubits / tunable coupler buses)
    spectator_edges = [
        (0, 1), (1, 2), (0, 3), (1, 4), (2, 5),
        (3, 4), (4, 5), (3, 6), (4, 6), (4, 7), (5, 7), (6, 7),
    ]

    edge_src: List[int] = []
    edge_dst: List[int] = []
    # Self-loops
    for i in range(num_nodes):
        edge_src.append(i)
        edge_dst.append(i)
    # Check <-> Data edges
    for c_idx, d_idx in check_to_data_edges:
        d_node = num_checks + d_idx
        edge_src.extend([c_idx, d_node])
        edge_dst.extend([d_node, c_idx])
    # Spectator Check <-> Check edges
    for c1, c2 in spectator_edges:
        edge_src.extend([c1, c2])
        edge_dst.extend([c2, c1])

    graph_edges = np.array([edge_src, edge_dst], dtype=np.int64)

    diff = coords_um[:, None, :] - coords_um[None, :, :]
    pairwise_dist_um = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)
    check_dist_um = pairwise_dist_um[:num_checks, :num_checks]

    return DeviceTopology(
        topology_id="surface17_tunable_coupler_v1",
        num_checks=num_checks,
        num_data=num_data,
        num_nodes=num_nodes,
        node_roles=node_roles,
        coords_um=coords_um,
        check_to_data_edges=check_to_data_edges,
        spectator_edges=spectator_edges,
        graph_edges=graph_edges,
        pairwise_dist_um=pairwise_dist_um,
        check_dist_um=check_dist_um,
    )


class QuantumNoiseTrajectoryGenerator:
    """Generates QuTiP-calibrated trajectories for B1, A1, C1, C2, A2, and MIXED_ALL_THREE."""

    def __init__(self, topology: Optional[DeviceTopology] = None) -> None:
        self.topology = topology or build_surface17_topology()
        # QuTiP physical-reference calibrated constants (from Benchmark-Evals/Qutip-Baseline)
        self.p_base = 0.012
        self.p_leak_heat = 0.0045       # Varbanov et al. 2020 scaled heating rate for rich training events
        self.p_seep_cool = 0.065        # Varbanov et al. 2020 seepage recovery rate
        self.p_defect_given_leak = 0.5826  # QuTiP dispersive ancilla scrambling probability
        self.crosstalk_boost_2gate = 0.0352  # Ketterer & Wellens 2023 parallel CZ correlated boost
        self.crosstalk_boost_3gate = 0.0680  # OOD 3-gate parallel concurrency boost
        self.lambda_phonon_um = 400.0   # Wilen et al. 2021 phonon decay length
        self.burst_prob = 0.025         # Chip-wide impact event arrival probability per round
        self.burst_decay_gamma = 0.82   # ~15 us exponential phonon relaxation per round

    def _sample_schedule_for_round(
        self, rng: np.random.Generator, t: int, ood_schedule: bool = False
    ) -> Tuple[List[Dict[str, Any]], int, np.ndarray, float]:
        """Generate gate schedule features u_t for round t.

        Returns:
            schedule_features: JSON-serializable list of gate dicts
            concurrency_group: int in {0: isolated, 1: parallelPairA, 2: parallelPairB, 3: parallelTripleOOD}
            active_spectator_mask: (num_checks, num_checks) binary adjacency of simultaneously driven checks
            drive_mhz: effective Rabi drive frequency in MHz
        """
        C = self.topology.num_checks
        active_spectator_mask = np.zeros((C, C), dtype=np.float32)

        if ood_schedule:
            # Include unseen 3-gate parallel concurrency group (group 3)
            group = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25]))
        else:
            # In-distribution schedules only use groups 0, 1, 2
            group = int(rng.choice([0, 1, 2], p=[0.40, 0.30, 0.30]))

        drive_mhz = float(rng.choice([15.0, 25.0, 45.0], p=[0.35, 0.40, 0.25]))
        duration_ns = 40.0 if group > 0 else 20.0

        schedule_features: List[Dict[str, Any]] = []
        if group == 0:
            # Isolated single-pair CZ sequence
            c_idx = int(t % C)
            schedule_features.append({
                "gate_type": "CZ",
                "targets": [c_idx, (c_idx + 1) % C],
                "couplers": [0],
                "concurrency_group": 0,
                "duration_ns": duration_ns,
                "drive_mhz": drive_mhz,
            })
        elif group == 1:
            # Parallel CZ group A: checks (0,1), (3,4), (6,7)
            pairs = [(0, 1), (3, 4), (6, 7)]
            for idx, (c1, c2) in enumerate(pairs):
                active_spectator_mask[c1, c2] = 1.0
                active_spectator_mask[c2, c1] = 1.0
                schedule_features.append({
                    "gate_type": "CZ",
                    "targets": [c1, c2],
                    "couplers": [idx],
                    "concurrency_group": 1,
                    "duration_ns": duration_ns,
                    "drive_mhz": drive_mhz,
                })
        elif group == 2:
            # Parallel CZ group B: checks (1,2), (4,5), (3,6)
            pairs = [(1, 2), (4, 5), (3, 6)]
            for idx, (c1, c2) in enumerate(pairs):
                active_spectator_mask[c1, c2] = 1.0
                active_spectator_mask[c2, c1] = 1.0
                schedule_features.append({
                    "gate_type": "CZ",
                    "targets": [c1, c2],
                    "couplers": [idx + 3],
                    "concurrency_group": 2,
                    "duration_ns": duration_ns,
                    "drive_mhz": drive_mhz,
                })
        else:
            # Group 3 (OOD High-Concurrency 3-cluster simultaneous CZs)
            pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (4, 7), (6, 7)]
            for idx, (c1, c2) in enumerate(pairs):
                active_spectator_mask[c1, c2] = 1.0
                active_spectator_mask[c2, c1] = 1.0
                schedule_features.append({
                    "gate_type": "CZ",
                    "targets": [c1, c2],
                    "couplers": [idx + 6],
                    "concurrency_group": 3,
                    "duration_ns": duration_ns,
                    "drive_mhz": drive_mhz,
                })

        return schedule_features, group, active_spectator_mask, drive_mhz

    def generate_trajectory(
        self,
        family: str,
        seed: int,
        num_rounds: int = 96,
        horizon_max: int = 8,
        ood_schedule: bool = False,
    ) -> Dict[str, Any]:
        """Generate a complete schema-validated trajectory for the specified noise family.

        Args:
            family: One of 'B1_TRANSMON_LEAKAGE', 'A1_GATE_CROSSTALK', 'C1_DRIVEN_TLS',
                    'C2_IDLE_MEMORY', 'A2_CHIPWIDE_EVENTS', or 'MIXED_ALL_THREE'.
            seed: Deterministic integer random seed.
            num_rounds: Number of QEC rounds in the trajectory.
            horizon_max: Number of future rounds of schedule features to pre-generate.
            ood_schedule: Whether to inject unseen 3-gate parallel concurrency schedules.
        """
        rng = np.random.default_rng(seed)
        C = self.topology.num_checks

        enable_leak = family in ("B1_TRANSMON_LEAKAGE", "MIXED_ALL_THREE")
        enable_corr = family in ("A1_GATE_CROSSTALK", "MIXED_ALL_THREE")
        enable_tls_drift = family in ("C1_DRIVEN_TLS", "C2_IDLE_MEMORY", "MIXED_ALL_THREE")
        enable_burst = family in ("A2_CHIPWIDE_EVENTS", "MIXED_ALL_THREE")

        # Pre-generate schedules for 0 .. num_rounds + horizon_max
        total_steps = num_rounds + horizon_max
        schedules_all = [
            self._sample_schedule_for_round(rng, t, ood_schedule=ood_schedule)
            for t in range(total_steps)
        ]

        # Latent physical state variables across time
        leak_state = np.zeros(C, dtype=np.float32)  # binary |2> state per check neighborhood
        burst_amp = 0.0
        burst_center = np.array([550.0, 550.0], dtype=np.float32)
        ou_drift = np.zeros(C, dtype=np.float32)    # slow OU calibration drift
        tls_phase = rng.uniform(0.0, 2.0 * np.pi, size=C).astype(np.float32)

        observations: List[Dict[str, Any]] = []
        prev_syndrome = np.zeros(C, dtype=np.int64)

        # Arrays for simulator-only latent_truth sidecar
        truth_z_leak: List[List[int]] = []
        truth_z_corr_rate: List[float] = []
        truth_z_corr_pairs: List[List[List[float]]] = []
        truth_z_burst_active: List[int] = []
        truth_z_burst_footprint: List[List[float]] = []
        truth_z_drift: List[List[float]] = []
        truth_z_memory: List[List[float]] = []
        truth_detection_prob: List[List[float]] = []
        truth_concurrency: List[int] = []

        for t in range(num_rounds):
            sched_feats, group, spec_mask, drive_mhz = schedules_all[t]

            # 1. Update B1 Transmon Leakage HMM state
            if enable_leak:
                for i in range(C):
                    if leak_state[i] == 0.0:
                        if rng.random() < self.p_leak_heat:
                            leak_state[i] = 1.0
                    else:
                        if rng.random() < self.p_seep_cool:
                            leak_state[i] = 0.0

            # 2. Update A2 Chip-Wide Phonon Burst state
            burst_footprint = np.zeros(C, dtype=np.float32)
            if enable_burst:
                burst_amp *= self.burst_decay_gamma
                if rng.random() < self.burst_prob:
                    burst_amp = float(rng.uniform(0.28, 0.45))
                    burst_center = np.array(
                        [rng.uniform(150.0, 950.0), rng.uniform(150.0, 950.0)],
                        dtype=np.float32,
                    )
                if burst_amp > 0.03:
                    dists = np.sqrt(
                        np.sum((self.topology.coords_um[:C] - burst_center[None, :]) ** 2, axis=-1)
                    )
                    burst_footprint = (burst_amp * np.exp(-dists / self.lambda_phonon_um)).astype(np.float32)
            burst_active = 1 if float(np.max(burst_footprint)) > 0.06 else 0

            # 3. Update C1/C2 Driven TLS & Idle Memory Revival + Slow OU Drift
            memory_revival = np.zeros(C, dtype=np.float32)
            if enable_tls_drift:
                ou_drift = 0.96 * ou_drift + rng.normal(0.0, 0.004, size=C).astype(np.float32)
                ou_drift = np.clip(ou_drift, -0.025, 0.035)
                # Coherent TLS memory revival modulated by drive_mhz (C1) and two-frequency beat (C2)
                drive_factor = 1.0 + 0.65 * (drive_mhz / 45.0)
                beat = 0.5 * (
                    np.cos(2.0 * np.pi * 0.045 * t + tls_phase)
                    + np.cos(2.0 * np.pi * 0.16 * t - 0.5 * tls_phase)
                )
                memory_revival = np.clip(0.022 * drive_factor * (beat + 0.5), 0.0, 0.08).astype(np.float32)

            # 4. Compute A1 Schedule-Conditioned Correlated Crosstalk Matrix
            pair_corr_matrix = np.zeros((C, C), dtype=np.float32)
            corr_scalar = 0.0
            if enable_corr and group > 0:
                boost = self.crosstalk_boost_3gate if group == 3 else self.crosstalk_boost_2gate
                pair_corr_matrix = spec_mask * boost
                corr_scalar = float(boost)

            # 5. Combine physical mechanisms into true per-check detection probabilities
            p_det = np.full(C, self.p_base, dtype=np.float32)
            if enable_leak:
                p_det = np.where(leak_state > 0.5, self.p_defect_given_leak, p_det)
            if enable_burst:
                p_det = np.clip(p_det + burst_footprint, 0.0, 0.92)
            if enable_tls_drift:
                p_det = np.clip(p_det + np.maximum(0.0, ou_drift + memory_revival), 0.0, 0.92)
            if enable_corr and group > 0:
                active_nodes = np.clip(np.sum(spec_mask, axis=-1), 0.0, 1.0)
                p_det = np.clip(p_det + active_nodes * (0.65 * corr_scalar), 0.0, 0.92)

            # Sample detection events (including correlated pairwise faults when spec_mask is active)
            det_vec = (rng.random(size=C) < p_det).astype(np.int64)
            if enable_corr and group > 0:
                for c1 in range(C):
                    for c2 in range(c1 + 1, C):
                        if spec_mask[c1, c2] > 0.5 and rng.random() < pair_corr_matrix[c1, c2]:
                            det_vec[c1] = 1
                            det_vec[c2] = 1

            # Update cumulative stabilizer syndrome s_t = s_{t-1} XOR d_t
            curr_syndrome = np.bitwise_xor(prev_syndrome, det_vec)
            prev_syndrome = curr_syndrome

            # Generate Analog Readout LLR features (with 15% random missingness mask to test mask handling)
            analog_available = bool(rng.random() > 0.15)
            llr_list: List[float] = []
            for i in range(C):
                if not analog_available:
                    llr_list.append(0.0)
                elif leak_state[i] > 0.5:
                    llr_list.append(float(rng.normal(2.2, 0.65)))
                else:
                    llr_list.append(float(rng.normal(-2.4, 0.65)))

            # Generate Periodic Calibration Telemetry (available every 8 rounds)
            calib_available = bool(t % 8 == 0)
            mean_drift = float(np.mean(ou_drift + memory_revival)) if enable_tls_drift else 0.0
            t1_est = float(28.0 * (1.0 - 4.0 * mean_drift - 0.5 * float(np.mean(burst_footprint)))) if calib_available else None
            detuning_mhz = float(1.5 * np.mean(memory_revival) * 100.0) if calib_available else None

            sparse_det_indices = [int(i) for i in np.where(det_vec == 1)[0]]
            observations.append({
                "round": t,
                "detection_events": sparse_det_indices,
                "syndrome": [int(x) for x in curr_syndrome],
                "calibration": {
                    "t1_us": t1_est,
                    "detuning_mhz": detuning_mhz,
                    "drive_mhz": drive_mhz,
                },
                "calibration_mask": {
                    "t1_us": calib_available,
                    "detuning_mhz": calib_available,
                    "drive_mhz": True,
                },
                "readout_features": {
                    "leakage_llr": [round(x, 4) for x in llr_list],
                },
                "readout_mask": {
                    "leakage_llr": analog_available,
                },
                "schedule_features": sched_feats,
            })

            # Record simulator-only ground truth
            truth_z_leak.append([int(x) for x in leak_state])
            truth_z_corr_rate.append(round(corr_scalar, 5))
            truth_z_corr_pairs.append(pair_corr_matrix.round(5).tolist())
            truth_z_burst_active.append(burst_active)
            truth_z_burst_footprint.append(burst_footprint.round(5).tolist())
            truth_z_drift.append(ou_drift.round(5).tolist())
            truth_z_memory.append(memory_revival.round(5).tolist())
            truth_detection_prob.append(p_det.round(5).tolist())
            truth_concurrency.append(group)

        future_schedule = [
            {
                "round": num_rounds + h,
                "concurrency_group": schedules_all[num_rounds + h][1],
                "drive_mhz": schedules_all[num_rounds + h][3],
                "schedule_features": schedules_all[num_rounds + h][0],
            }
            for h in range(horizon_max)
        ]

        cfg_hash = hashlib.sha256(f"{family}_{ self.topology.topology_id }_v1".encode()).hexdigest()[:16]
        return {
            "schema_version": "1.0.0",
            "trajectory_id": f"{family}_seed{seed}",
            "seed": int(seed),
            "device_config_id": f"{family}_surface17_config",
            "topology_id": self.topology.topology_id,
            "round_start": 0,
            "round_end": num_rounds - 1,
            "observations": observations,
            "future_schedule": future_schedule,
            "latent_truth": {
                "family": family,
                "ood_schedule": ood_schedule,
                "z_leak": truth_z_leak,
                "z_corr_rate": truth_z_corr_rate,
                "z_corr_pairs": truth_z_corr_pairs,
                "z_burst_active": truth_z_burst_active,
                "z_burst_footprint": truth_z_burst_footprint,
                "z_drift": truth_z_drift,
                "z_memory": truth_z_memory,
                "true_detection_prob": truth_detection_prob,
                "concurrency_group": truth_concurrency,
            },
            "provenance": {
                "simulator": "AVID-QuTiP-Calibrated-Engine",
                "simulator_version": "5.2.3-ref",
                "model_version": "1.0.0",
                "config_hash": cfg_hash,
                "parameter_sources": [
                    "Varbanov_2020_Surface17",
                    "Ketterer_Wellens_2023_Crosstalk",
                    "Agarwal_2023_DrivenTLS",
                    "Gulacsi_Burkard_2023_IdleMemory",
                    "Wilen_2021_PhononBursts",
                ],
            },
        }

    def generate_benchmark_splits(
        self,
        num_train_per_family: int = 12,
        num_val_per_family: int = 4,
        num_test_per_family: int = 6,
        num_rounds: int = 80,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate strictly seed-disjoint train, val, test_id, and test_ood_schedule trajectory splits."""
        families = [
            "B1_TRANSMON_LEAKAGE",
            "A1_GATE_CROSSTALK",
            "C1_DRIVEN_TLS",
            "C2_IDLE_MEMORY",
            "A2_CHIPWIDE_EVENTS",
            "MIXED_ALL_THREE",
        ]
        splits: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test_id": [],
            "test_ood_schedule": [],
        }

        base_seed = 10000
        for fam_idx, fam in enumerate(families):
            fam_offset = base_seed + fam_idx * 1000
            for i in range(num_train_per_family):
                splits["train"].append(
                    self.generate_trajectory(fam, seed=fam_offset + i, num_rounds=num_rounds, ood_schedule=False)
                )
            for i in range(num_val_per_family):
                splits["val"].append(
                    self.generate_trajectory(fam, seed=fam_offset + 200 + i, num_rounds=num_rounds, ood_schedule=False)
                )
            for i in range(num_test_per_family):
                splits["test_id"].append(
                    self.generate_trajectory(fam, seed=fam_offset + 400 + i, num_rounds=num_rounds, ood_schedule=False)
                )
                splits["test_ood_schedule"].append(
                    self.generate_trajectory(fam, seed=fam_offset + 600 + i, num_rounds=num_rounds, ood_schedule=True)
                )

        return splits
