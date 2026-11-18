"""Verification Suite 2: V1 & V2 Checkpoint Export & Schema Validation (`tests/test_checkpoint_export.py`)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from training.export.export_checkpoint import (
    TENSOR_ORDER,
    TENSOR_ORDER_V1,
    export_r2d2_checkpoint,
    load_r2d2_checkpoint_bin,
)
from training.models.r2d2_network import R2D2Network


def test_v1_frozen_checkpoint_files_and_schema_integrity() -> None:
    """Verify frozen V1 checkpoint (`157,984` bytes, 8 inputs) remains intact and reproducible."""
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "checkpoints" / "v1" / "r2d2_v1_trained.bin"
    json_path = repo_root / "checkpoints" / "v1" / "r2d2_v1_trained.json"
    onnx_path = repo_root / "checkpoints" / "v1" / "r2d2_v1_trained.onnx"
    schema_path = repo_root / "deployment" / "model_schema" / "r2d2_schema_v1.json"

    assert bin_path.exists()
    assert json_path.exists()
    assert onnx_path.exists()
    assert schema_path.exists()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    raw_bin = bin_path.read_bytes()
    assert len(raw_bin) == schema["total_file_bytes"] == 157984
    assert hashlib.sha256(raw_bin).hexdigest() == meta["binary_checkpoint"]["sha256"]


def test_v2_checkpoint_files_and_schema_integrity() -> None:
    """Verify V2 82-input checkpoint (`310,720` bytes, 38,672 trainable params / 38,836 floats) matches V2 schema."""
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    json_path = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.json"
    onnx_path = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.onnx"
    schema_path = repo_root / "deployment" / "model_schema" / "r2d2_schema_v2.json"

    assert bin_path.exists()
    assert json_path.exists()
    assert onnx_path.exists()
    assert schema_path.exists()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    raw_bin = bin_path.read_bytes()
    assert len(raw_bin) == schema["total_file_bytes"] == 310720
    assert meta["schema_version"] == 2
    assert meta["architecture"]["input_dim"] == 82
    assert meta["architecture"]["trainable_parameters"] == 38672
    assert meta["binary_checkpoint"]["total_float64_elements"] == 38836
    assert hashlib.sha256(raw_bin).hexdigest() == meta["binary_checkpoint"]["sha256"]

    raw_onnx = onnx_path.read_bytes()
    assert b"AVID_R2D2_Dueling_LSTM_V2_Graph" in raw_onnx
    assert hashlib.sha256(raw_onnx).hexdigest() == meta["onnx_checkpoint"]["sha256"]


def test_roundtrip_export_and_load_exact_parity_v2_and_v1(tmp_path: Path) -> None:
    """Verify bit-exact round-trip export -> load parity for V2 (82-D) and V1 (8-D), plus version mismatch checks."""
    net_v2 = R2D2Network(input_dim=82, seed=777)
    bin_v2 = tmp_path / "test_v2.bin"
    json_v2 = tmp_path / "test_v2.json"
    onnx_v2 = tmp_path / "test_v2.onnx"

    export_r2d2_checkpoint(net_v2, bin_v2, json_v2, onnx_v2)
    loaded_v2, state_v2 = load_r2d2_checkpoint_bin(bin_v2, expected_version=2)

    orig_state = net_v2.get_state_dict()
    for name, _ in TENSOR_ORDER:
        assert np.array_equal(orig_state[name], state_v2[name])

    rng = np.random.default_rng(778)
    obs_seq = rng.uniform(0.0, 0.2, size=(10, 82)).astype(np.float64)
    out_orig = net_v2.forward_sequence(obs_seq)
    out_load = loaded_v2.forward_sequence(obs_seq)

    assert np.max(np.abs(out_orig["q_seq"] - out_load["q_seq"])) == 0.0
    assert np.max(np.abs(out_orig["gain_seq"] - out_load["gain_seq"])) == 0.0
    assert np.max(np.abs(out_orig["h_seq"] - out_load["h_seq"])) == 0.0

    # V1 loader expectation must reject V2 file
    with pytest.raises(ValueError, match="Checkpoint version mismatch"):
        load_r2d2_checkpoint_bin(bin_v2, expected_version=1)

    # Also verify V1 round-trip
    net_v1 = R2D2Network(input_dim=8, seed=779)
    bin_v1 = tmp_path / "test_v1.bin"
    json_v1 = tmp_path / "test_v1.json"
    export_r2d2_checkpoint(net_v1, bin_v1, json_v1)
    loaded_v1, state_v1 = load_r2d2_checkpoint_bin(bin_v1, expected_version=1)
    for name, _ in TENSOR_ORDER_V1:
        assert np.array_equal(net_v1.get_state_dict()[name], state_v1[name])


def test_corrupted_checkpoint_rejected(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.bin"
    bad_file.write_bytes(b"BADMAGIC" + b"\x00" * 100)
    with pytest.raises(ValueError, match="Invalid checkpoint magic"):
        load_r2d2_checkpoint_bin(bad_file)
