"""PyTorch R2D2 Replay Dataset with burn-in window slicing and strict observation/latent separation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from world_model.data.generator import DeviceTopology, build_surface17_topology


def trajectory_to_tensors(
    record: Dict[str, Any],
    topology: Optional[DeviceTopology] = None,
    include_latent_truth: bool = True,
) -> Dict[str, Any]:
    """Convert a single schema-validated trajectory record into dense NumPy/PyTorch arrays.

    Returns a dictionary with:
        - `observations`: inference-safe dict of tensors over T rounds
        - `latent_truth`: simulator-only ground truth dict (only if `include_latent_truth=True` and present)
    """
    topo = topology or build_surface17_topology()
    C = topo.num_checks
    obs_list = record["observations"]
    T = len(obs_list)

    detection_vec = np.zeros((T, C), dtype=np.float32)
    syndrome_vec = np.zeros((T, C), dtype=np.float32)
    analog_llr = np.zeros((T, C), dtype=np.float32)
    analog_mask = np.zeros((T, 1), dtype=np.float32)
    telemetry_val = np.zeros((T, 3), dtype=np.float32)  # [t1_us/30, detuning_mhz/10, drive_mhz/50]
    telemetry_mask = np.zeros((T, 3), dtype=np.float32)
    schedule_adj = np.zeros((T, C, C), dtype=np.float32)
    concurrency_group = np.zeros(T, dtype=np.int64)
    drive_norm = np.zeros((T, 1), dtype=np.float32)

    for t, obs in enumerate(obs_list):
        for idx in obs.get("detection_events", []):
            if 0 <= idx < C:
                detection_vec[t, idx] = 1.0
        if "syndrome" in obs and obs["syndrome"] is not None:
            syndrome_vec[t] = np.array(obs["syndrome"][:C], dtype=np.float32)

        r_mask = bool(obs.get("readout_mask", {}).get("leakage_llr", False))
        analog_mask[t, 0] = 1.0 if r_mask else 0.0
        if r_mask and "readout_features" in obs:
            analog_llr[t] = np.array(obs["readout_features"].get("leakage_llr", [0.0] * C)[:C], dtype=np.float32)

        cal = obs.get("calibration", {})
        cmask = obs.get("calibration_mask", {})
        if cmask.get("t1_us", False) and cal.get("t1_us") is not None:
            telemetry_val[t, 0] = float(cal["t1_us"]) / 30.0
            telemetry_mask[t, 0] = 1.0
        if cmask.get("detuning_mhz", False) and cal.get("detuning_mhz") is not None:
            telemetry_val[t, 1] = float(cal["detuning_mhz"]) / 10.0
            telemetry_mask[t, 1] = 1.0
        if cmask.get("drive_mhz", False) and cal.get("drive_mhz") is not None:
            telemetry_val[t, 2] = float(cal["drive_mhz"]) / 50.0
            telemetry_mask[t, 2] = 1.0
            drive_norm[t, 0] = float(cal["drive_mhz"]) / 50.0

        s_feats = obs.get("schedule_features", [])
        grp = 0
        for gate in s_feats:
            g_grp = gate.get("concurrency_group")
            if g_grp is not None:
                grp = max(grp, int(g_grp))
            targets = gate.get("targets", [])
            if len(targets) >= 2 and grp > 0:
                c1, c2 = int(targets[0]), int(targets[1])
                if 0 <= c1 < C and 0 <= c2 < C:
                    schedule_adj[t, c1, c2] = 1.0
                    schedule_adj[t, c2, c1] = 1.0
        concurrency_group[t] = grp

    out: Dict[str, Any] = {
        "trajectory_id": record["trajectory_id"],
        "seed": int(record["seed"]),
        "observations": {
            "detection_vec": torch.from_numpy(detection_vec),
            "syndrome_vec": torch.from_numpy(syndrome_vec),
            "analog_llr": torch.from_numpy(analog_llr),
            "analog_mask": torch.from_numpy(analog_mask),
            "telemetry_val": torch.from_numpy(telemetry_val),
            "telemetry_mask": torch.from_numpy(telemetry_mask),
            "schedule_adj": torch.from_numpy(schedule_adj),
            "concurrency_group": torch.from_numpy(concurrency_group),
            "drive_norm": torch.from_numpy(drive_norm),
        },
    }

    if include_latent_truth and "latent_truth" in record and record["latent_truth"] is not None:
        lt = record["latent_truth"]
        out["latent_truth"] = {
            "family": lt.get("family", "UNKNOWN"),
            "z_leak": torch.tensor(lt["z_leak"], dtype=torch.float32),
            "z_corr_rate": torch.tensor(lt["z_corr_rate"], dtype=torch.float32),
            "z_corr_pairs": torch.tensor(lt["z_corr_pairs"], dtype=torch.float32),
            "z_burst_active": torch.tensor(lt["z_burst_active"], dtype=torch.float32),
            "z_burst_footprint": torch.tensor(lt["z_burst_footprint"], dtype=torch.float32),
            "z_drift": torch.tensor(lt["z_drift"], dtype=torch.float32),
            "z_memory": torch.tensor(lt["z_memory"], dtype=torch.float32),
            "true_detection_prob": torch.tensor(lt["true_detection_prob"], dtype=torch.float32),
        }

    return out


class R2D2TrajectoryReplayDataset(Dataset):
    """Slices trajectories into R2D2 `(burn_in_steps + train_steps + horizon)` windows.

    During training:
      - First `burn_in_steps` rounds are rolled forward under `torch.no_grad()` to warm up the
        recurrent hidden/cell belief state `(h_t, c_t)` from real observations.
      - Next `train_steps` rounds are used for active variational ELBO and multi-horizon forecasting
        across `H in {1, 2, 4, 8}`.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        topology: Optional[DeviceTopology] = None,
        burn_in_steps: int = 16,
        train_steps: int = 32,
        horizon_max: int = 8,
        stride: int = 16,
        include_latent_truth: bool = True,
    ) -> None:
        super().__init__()
        self.topology = topology or build_surface17_topology()
        self.burn_in_steps = burn_in_steps
        self.train_steps = train_steps
        self.horizon_max = horizon_max
        self.window_len = burn_in_steps + train_steps + horizon_max
        self.include_latent_truth = include_latent_truth

        self.trajectories = [
            trajectory_to_tensors(rec, topology=self.topology, include_latent_truth=include_latent_truth)
            for rec in records
        ]

        self.windows: List[Tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            T = traj["observations"]["detection_vec"].shape[0]
            if T < self.window_len:
                raise ValueError(f"Trajectory length {T} is shorter than window_len={self.window_len}")
            for start_t in range(0, T - self.window_len + 1, stride):
                self.windows.append((traj_idx, start_t))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, start_t = self.windows[idx]
        end_t = start_t + self.window_len
        traj = self.trajectories[traj_idx]

        obs_slice = {
            k: v[start_t:end_t] for k, v in traj["observations"].items()
        }
        item: Dict[str, Any] = {
            "trajectory_id": traj["trajectory_id"],
            "seed": traj["seed"],
            "start_round": start_t,
            "burn_in_steps": self.burn_in_steps,
            "train_steps": self.train_steps,
            "horizon_max": self.horizon_max,
            "observations": obs_slice,
        }
        if self.include_latent_truth and "latent_truth" in traj:
            lt_slice = {
                k: (v[start_t:end_t] if isinstance(v, torch.Tensor) else v)
                for k, v in traj["latent_truth"].items()
            }
            item["latent_truth"] = lt_slice
        return item
