"""Step 12 — AVID Input Adapter & Packet Stream Integration Tests."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import pytest

from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import AvidInputAdapter
from simulation.stimulus.noise_framework.avid_interface.sample_encoder import SampleEncoder
from simulation.stimulus.noise_framework.config_schema import (
    CompositeNoiseSample,
    NoiseContribution,
)
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)
from simulation.stimulus.packet_codec import unpack_packet_u64
from simulation.stimulus.stim_to_bin import MAGIC, bits_to_nibble


def _build_test_trajectory(n_samples: int = 32) -> dict:
    samples = []
    for k in range(n_samples):
        t = round(k * 0.1, 6)
        w = k // 1000
        mp = MeasurementNoiseParameters(
            q_base_x=np.array([0.05, 0.06, 0.05, 0.06]),
            q_base_z=np.array([0.04, 0.05, 0.04, 0.05]),
            temporal_sticky_prob=0.12,
            detector_dropout_prob=0.05,
            observation_delay_samples=1,
            generator_id="measurement_noise_01",
            profile_id="PD_06",
            sample_id=k,
            timestamp=t,
            window_id=w,
        )
        contrib = NoiseContribution(
            generator_id="measurement_noise_01",
            profile_id="PD_06",
            window_id=w,
            sample_id=k,
            timestamp=t,
            output_domain="measurement_fault_parameters",
            values=mp.to_vector(),
            metadata={"measurement_parameters": mp},
        )
        samples.append(
            CompositeNoiseSample(
                sample_id=k,
                timestamp=t,
                window_id=w,
                contributions=[contrib],
                composite_parameters={
                    "theta_z": np.full(21, 0.06, dtype=np.float64),
                    "theta_x": np.full(21, 0.04, dtype=np.float64),
                    "measurement": mp,
                },
                metadata={"profile_assignments": {"measurement_noise_01": "PD_06"}},
            )
        )
    encoder = SampleEncoder(rounds_per_composite_sample=1)
    return encoder.encode_composite_trajectory(
        samples, master_seed=2026, trajectory_id="traj_step12", split_id="val"
    )


def test_correct_syndrome_bits_survive_encoding(tmp_path: Path):
    traj = _build_test_trajectory(24)
    adapter = AvidInputAdapter()
    npz_path = adapter.write_npz(traj, tmp_path / "traj_step12.npz")
    loaded = np.load(npz_path)

    np.testing.assert_array_equal(loaded["syndromes_x"], traj["syndromes_x"])
    np.testing.assert_array_equal(loaded["syndromes_z"], traj["syndromes_z"])
    np.testing.assert_array_equal(loaded["validity_mask_x"], traj["validity_mask_x"])
    np.testing.assert_array_equal(loaded["validity_mask_z"], traj["validity_mask_z"])
    np.testing.assert_array_equal(loaded["measurement_faults_x"], traj["measurement_faults_x"])
    np.testing.assert_array_equal(loaded["measurement_faults_z"], traj["measurement_faults_z"])


def test_packet_round_trip_returns_identical_syndrome_bits(tmp_path: Path):
    traj = _build_test_trajectory(28)
    adapter = AvidInputAdapter()
    bin_path = adapter.write_packet_stream(traj, tmp_path / "traj_step12.bin", u64_format=True)

    check = adapter.validate_stream(bin_path, expected_trajectory=traj)
    assert check["valid"] is True
    assert check["n_rounds"] == 28
    assert check["u64_format"] is True
    assert check["seq_gaps"] == 0


def test_xz_basis_separation_maintained_in_packet_stream(tmp_path: Path):
    traj = _build_test_trajectory(20)
    # Force distinct X and Z observed syndromes
    traj["syndromes_x"] = np.tile(np.array([1, 0, 1, 0], dtype=np.uint8), (20, 1))
    traj["syndromes_z"] = np.tile(np.array([0, 1, 0, 1], dtype=np.uint8), (20, 1))

    adapter = AvidInputAdapter()
    bin_path = adapter.write_packet_stream(traj, tmp_path / "xz_sep.bin", u64_format=True)
    raw = bin_path.read_bytes()
    payload = raw[16:]

    exp_sx = bits_to_nibble(np.array([1, 0, 1, 0], dtype=np.uint8))
    exp_sz = bits_to_nibble(np.array([0, 1, 0, 1], dtype=np.uint8))
    assert exp_sx != exp_sz

    for r in range(20):
        (word,) = struct.unpack_from("<Q", payload, r * 8)
        pkt = unpack_packet_u64(word)
        assert pkt.s_x == exp_sx
        assert pkt.s_z == exp_sz


def test_packet_length_and_format_unchanged(tmp_path: Path):
    traj = _build_test_trajectory(16)
    adapter = AvidInputAdapter()
    bin_path = adapter.write_packet_stream(traj, tmp_path / "format_check.bin", u64_format=True)
    raw = bin_path.read_bytes()

    # 16-byte header + 16 rounds * 8 bytes/round = 144 bytes
    assert len(raw) == 16 + 16 * 8
    magic, n_rounds, is_u64, n_cx, n_cz, dist, _ = struct.unpack_from("<IIIBBBB", raw, 0)
    assert magic == MAGIC
    assert n_rounds == 16
    assert is_u64 == 1
    assert (n_cx, n_cz, dist) == (4, 4, 3)


def test_metadata_matches_original_trajectory(tmp_path: Path):
    traj = _build_test_trajectory(30)
    adapter = AvidInputAdapter()
    npz_path = adapter.write_npz(traj, tmp_path / "meta_check.npz")
    meta_path = npz_path.with_suffix(".meta.json")
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["trajectory_id"] == "traj_step12"
    assert meta["split_id"] == "val"
    assert meta["master_seed"] == 2026
    assert meta["total_rounds"] == 30
    assert meta["assignment_history"]["0"]["measurement_noise_01"] == "PD_06"
    assert meta["max_observation_delay_samples"] == 1


def test_invalid_packet_input_rejected(tmp_path: Path):
    adapter = AvidInputAdapter()

    # 1. Shorter than 16-byte header
    short_bin = tmp_path / "short.bin"
    short_bin.write_bytes(b"\x00" * 8)
    with pytest.raises(ValueError, match="shorter than 16-byte header"):
        adapter.validate_stream(short_bin)

    # 2. Bad magic
    bad_magic_bin = tmp_path / "bad_magic.bin"
    bad_magic_bin.write_bytes(struct.pack("<IIIBBBB", 0xDEADBEEF, 2, 1, 4, 4, 3, 0) + b"\x00" * 16)
    with pytest.raises(ValueError, match="Invalid header magic"):
        adapter.validate_stream(bad_magic_bin)

    # 3. Truncated payload
    trunc_bin = tmp_path / "trunc.bin"
    trunc_bin.write_bytes(struct.pack("<IIIBBBB", MAGIC, 4, 1, 4, 4, 3, 0) + b"\x00" * 8)
    with pytest.raises(ValueError, match="Payload byte size"):
        adapter.validate_stream(trunc_bin)
