"""AVID Input Adapter (`avid_input_adapter.py`).

Boundary between the Seven-Stage Noise Framework and existing AVID components:
  - Reuses `simulation/stimulus/stim_to_bin.py` and `simulation/stimulus/packet_codec.py`
    without introducing YAML parsing or PD sampling into the serializer.
  - Preserves the 64-bit `AVID-SYNDROME-PACKET-V1` binary stream format.
  - Stores observation validity masks, delivery timestamps, and assignment history in a
    companion `.meta.json` and compressed `.npz` dataset record.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Dict, Sequence
import numpy as np

from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.noise_framework.config_schema import CompositeNoiseSample
from simulation.stimulus.packet_codec import unpack_packet_u64
from simulation.stimulus.stim_to_bin import MAGIC, bits_to_nibble, convert_npz


class AvidInputAdapter:
    """Serializes encoded trajectories into NPZ and AVID binary packet streams."""

    def __init__(self, encoder: SampleEncoder | None = None) -> None:
        self.encoder = encoder if encoder is not None else SampleEncoder()

    def encode_trajectory(
        self,
        trajectory: Sequence[CompositeNoiseSample] | Dict[str, Any],
        master_seed: int = 42,
        trajectory_id: str = "traj_000",
        split_id: str = "train",
    ) -> Dict[str, Any]:
        if isinstance(trajectory, dict) and "syndromes_x" in trajectory:
            return trajectory
        if isinstance(trajectory, Sequence):
            return self.encoder.encode_composite_trajectory(
                composite_samples=trajectory,
                master_seed=master_seed,
                trajectory_id=trajectory_id,
                split_id=split_id,
            )
        raise TypeError(f"Unsupported trajectory type: {type(trajectory)}")

    def write_npz(
        self,
        trajectory: Sequence[CompositeNoiseSample] | Dict[str, Any],
        path: str | Path,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> Path:
        encoded = self.encode_trajectory(trajectory)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sx = np.asarray(encoded["syndromes_x"], dtype=np.uint8)
        sz = np.asarray(encoded["syndromes_z"], dtype=np.uint8)
        theta_z = np.asarray(encoded["theta_z"], dtype=np.float64)
        theta_x = np.asarray(encoded["theta_x"], dtype=np.float64)

        np.savez_compressed(
            str(out_path),
            syndromes=sx,
            syndromes_x=sx,
            syndromes_z=sz,
            ideal_syndromes_x=np.asarray(encoded["ideal_syndromes_x"], dtype=np.uint8),
            ideal_syndromes_z=np.asarray(encoded["ideal_syndromes_z"], dtype=np.uint8),
            validity_mask_x=np.asarray(encoded["validity_mask_x"], dtype=np.uint8),
            validity_mask_z=np.asarray(encoded["validity_mask_z"], dtype=np.uint8),
            measurement_faults_x=np.asarray(encoded["measurement_faults_x"], dtype=np.uint8),
            measurement_faults_z=np.asarray(encoded["measurement_faults_z"], dtype=np.uint8),
            measurement_parameters_vec=np.asarray(
                encoded["measurement_parameters_vec"], dtype=np.float64
            ),
            fault_masks_z=np.asarray(encoded["fault_masks_z"], dtype=np.uint8),
            fault_masks_x=np.asarray(encoded["fault_masks_x"], dtype=np.uint8),
            events_z_21=np.asarray(encoded["events_z_21"], dtype=np.uint8),
            events_x_21=np.asarray(encoded["events_x_21"], dtype=np.uint8),
            logical_error_z=np.asarray(encoded["logical_error_z"], dtype=np.uint8),
            logical_error_x=np.asarray(encoded["logical_error_x"], dtype=np.uint8),
            composite_noise_z=theta_z,
            composite_noise_x=theta_x,
            true_p=theta_z[:, :9].astype(np.float32),
            true_pzz=theta_z[:, 12].astype(np.float32),
            timestamps=np.asarray(encoded["timestamps"], dtype=np.float64),
            event_timestamps=np.asarray(encoded["event_timestamps"], dtype=np.float64),
            delivery_timestamps=np.asarray(encoded["delivery_timestamps"], dtype=np.float64),
            sample_ids=np.asarray(encoded["sample_ids"], dtype=np.int32),
            window_ids=np.asarray(encoded["window_ids"], dtype=np.int32),
            in_burst_window=np.asarray(encoded["in_burst_window"], dtype=np.uint8),
            n_check=4,
            n_data=9,
            total_rounds=int(encoded["total_rounds"]),
        )

        meta_path = out_path.with_suffix(".meta.json")
        meta_obj: Dict[str, Any] = {
            "trajectory_id": encoded.get("trajectory_id", "traj_000"),
            "split_id": encoded.get("split_id", "train"),
            "master_seed": int(encoded.get("master_seed", 42)),
            "num_composite_samples": int(encoded.get("num_composite_samples", len(sx))),
            "rounds_per_composite_sample": int(encoded.get("rounds_per_composite_sample", 1)),
            "total_rounds": int(encoded["total_rounds"]),
            "assignment_history": {
                str(k): v for k, v in encoded.get("assignment_history", {}).items()
            },
            "dropout_events_x": int(np.sum(encoded["validity_mask_x"] == 0)),
            "dropout_events_z": int(np.sum(encoded["validity_mask_z"] == 0)),
            "max_observation_delay_samples": int(np.max(encoded["observation_delays"])),
        }
        if extra_metadata:
            meta_obj.update(extra_metadata)
        meta_path.write_text(json.dumps(meta_obj, indent=2) + "\n", encoding="utf-8")
        return out_path

    def write_packet_stream(
        self,
        trajectory: Sequence[CompositeNoiseSample] | Dict[str, Any],
        path: str | Path,
        u64_format: bool = True,
    ) -> Path:
        """Serialize trajectory to binary packet stream via `stim_to_bin.convert_npz`."""
        encoded = self.encode_trajectory(trajectory)
        bin_path = Path(path)
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        npz_temp = bin_path.with_suffix(".tmp_input.npz")
        try:
            self.write_npz(encoded, npz_temp)
            convert_npz(str(npz_temp), str(bin_path), verify=False, u64_format=u64_format)
        finally:
            if npz_temp.exists():
                npz_temp.unlink()
            meta_tmp = npz_temp.with_suffix(".meta.json")
            if meta_tmp.exists():
                meta_tmp.unlink()
        return bin_path

    def validate_stream(
        self,
        path: str | Path,
        expected_trajectory: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Validate a binary stream header and round-trip decode every 64-bit packet."""
        bin_path = Path(path)
        raw = bin_path.read_bytes()
        if len(raw) < 16:
            raise ValueError(f"Binary packet stream {bin_path} is shorter than 16-byte header")
        magic, n_rounds, is_u64, n_cx, n_cz, dist, _ = struct.unpack_from("<IIIBBBB", raw, 0)
        if magic != MAGIC:
            raise ValueError(f"Invalid header magic 0x{magic:08X} != 0x{MAGIC:08X}")
        if n_cx != 4 or n_cz != 4 or dist != 3:
            raise ValueError(f"Unexpected code dimensions ({n_cx}, {n_cz}, d={dist})")

        payload = raw[16:]
        if is_u64 == 1:
            if len(payload) != n_rounds * 8:
                raise ValueError(f"Payload byte size {len(payload)} != {n_rounds * 8}")
            sx_unpacked = np.zeros(n_rounds, dtype=np.uint8)
            sz_unpacked = np.zeros(n_rounds, dtype=np.uint8)
            seq_gaps = 0
            for r in range(n_rounds):
                (word,) = struct.unpack_from("<Q", payload, r * 8)
                pkt = unpack_packet_u64(word)
                sx_unpacked[r] = pkt.s_x
                sz_unpacked[r] = pkt.s_z
                if r > 0 and pkt.seq_id != r:
                    seq_gaps += 1

            if expected_trajectory is not None:
                exp_sx = np.array(
                    [bits_to_nibble(row) for row in expected_trajectory["syndromes_x"]],
                    dtype=np.uint8,
                )
                exp_sz = np.array(
                    [bits_to_nibble(row) for row in expected_trajectory["syndromes_z"]],
                    dtype=np.uint8,
                )
                if not (
                    np.array_equal(sx_unpacked, exp_sx) and np.array_equal(sz_unpacked, exp_sz)
                ):
                    raise ValueError("Round-trip syndrome mismatch between trajectory and binary stream")
        else:
            if len(payload) != n_rounds * 2:
                raise ValueError(f"Legacy u16 payload byte size {len(payload)} != {n_rounds * 2}")
            seq_gaps = 0

        return {
            "valid": True,
            "magic_hex": f"0x{magic:08X}",
            "n_rounds": int(n_rounds),
            "u64_format": bool(is_u64 == 1),
            "seq_gaps": int(seq_gaps),
        }
