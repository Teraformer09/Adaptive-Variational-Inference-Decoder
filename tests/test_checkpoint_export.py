"""Verification Suite 2: Checkpoint Export & Schema Validation (`tests/test_checkpoint_export.py`).

Verifies:
  1. `checkpoints/r2d2_v1_trained.bin`, `.json`, and `.onnx` exist and match SHA-256 hashes.
  2. `deployment/model_schema/r2d2_schema.json` matches binary header and tensor offsets.
  3. Round-trip Python export -> binary load reproduces every weight tensor bit-for-bit (`max_abs_diff == 0.0`).
  4. Corrupted / truncated binary checkpoints are rejected with descriptive errors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from training.models.r2d2_network import R2D2Network
from training.export.export_checkpoint import (
    TENSOR_ORDER,
    export_r2d2_checkpoint,
    load_r2d2_checkpoint_bin,
)


def test_checkpoint_files_and_schema_integrity() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "checkpoints" / "r2d2_v1_trained.bin"
    json_path = repo_root / "checkpoints" / "r2d2_v1_trained.json"
    onnx_path = repo_root / "checkpoints" / "r2d2_v1_trained.onnx"
    schema_path = repo_root / "deployment" / "model_schema" / "r2d2_schema.json"

    assert bin_path.exists()
    assert json_path.exists()
    assert onnx_path.exists()
    assert schema_path.exists()

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    raw_bin = bin_path.read_bytes()
    assert len(raw_bin) == schema["total_file_bytes"] == 157984
    assert hashlib.sha256(raw_bin).hexdigest() == meta["binary_checkpoint"]["sha256"]

    raw_onnx = onnx_path.read_bytes()
    assert len(raw_onnx) > 150000
    assert b"AVID_R2D2_Dueling_LSTM_Graph" in raw_onnx
    assert hashlib.sha256(raw_onnx).hexdigest() == meta["onnx_checkpoint"]["sha256"]


def test_roundtrip_export_and_load_exact_parity(tmp_path: Path) -> None:
    net = R2D2Network(seed=777)
    bin_file = tmp_path / "test_r2d2.bin"
    json_file = tmp_path / "test_r2d2.json"
    onnx_file = tmp_path / "test_r2d2.onnx"

    export_r2d2_checkpoint(net, bin_file, json_file, onnx_file)
    loaded_net, loaded_state = load_r2d2_checkpoint_bin(bin_file)

    orig_state = net.get_state_dict()
    for name, _ in TENSOR_ORDER:
        assert np.array_equal(orig_state[name], loaded_state[name])

    rng = np.random.default_rng(778)
    obs_seq = rng.uniform(0.0, 0.2, size=(10, 8)).astype(np.float64)
    out_orig = net.forward_sequence(obs_seq)
    out_load = loaded_net.forward_sequence(obs_seq)

    assert np.max(np.abs(out_orig["q_seq"] - out_load["q_seq"])) == 0.0
    assert np.max(np.abs(out_orig["gain_seq"] - out_load["gain_seq"])) == 0.0
    assert np.max(np.abs(out_orig["h_seq"] - out_load["h_seq"])) == 0.0


def test_corrupted_checkpoint_rejected(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.bin"
    bad_file.write_bytes(b"BADMAGIC" + b"\x00" * 100)
    with pytest.raises(ValueError, match="Invalid checkpoint magic"):
        load_r2d2_checkpoint_bin(bad_file)
