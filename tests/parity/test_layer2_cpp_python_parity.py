"""DEP-01 & DEP-02 Parity & Compatibility Tests (`test_layer2_cpp_python_parity.py`)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import numpy as np

from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def _compile_parity_runner(repo_root: Path, out_bin: Path) -> Path:
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-Wall",
            "-I",
            str(repo_root),
            str(repo_root / "tests" / "parity" / "cpp_layer2_parity_runner.cpp"),
            str(repo_root / "deployment" / "cpp" / "model_loader.cpp"),
            str(repo_root / "deployment" / "cpp" / "r2d2_runtime.cpp"),
            str(repo_root / "deployment" / "cpp" / "observation_adapter.cpp"),
            "-o",
            str(out_bin),
            "-lm",
        ],
        check=True,
    )
    return out_bin


def test_dep01_v2_82d_cpp_python_numerical_parity_and_latency(tmp_path: Path) -> None:
    """DEP-01: Verify < 1e-12 numerical parity between Python R2D2 V2 and C++17 runtime across 82-D sequences, plus C++ latency."""
    repo_root = Path(__file__).resolve().parents[2]
    v2_bin = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    assert v2_bin.exists(), "Missing checkpoints/v2/r2d2_v2_trained.bin"

    cpp_bin = repo_root / "tests" / "build_layer2" / "cpp_layer2_parity_runner"
    if not cpp_bin.exists():
        cpp_bin = _compile_parity_runner(repo_root, tmp_path / "cpp_layer2_parity_runner")

    py_net, state_dict = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)
    assert py_net.input_dim == 82
    assert state_dict["W_ih"].shape == (256, 82)

    # Construct multi-regime 82-D syndrome-conditioned observation stream
    ep_a = generate_v2_qec_episode("zz_crosstalk", seed=202601, split_name="test", num_epochs=12)
    ep_b = generate_v2_qec_episode("nonstationary_burst", seed=202602, split_name="test", num_epochs=12)
    obs_seq = np.vstack([ep_a.obs, ep_b.obs])
    assert obs_seq.shape == (24, 82)

    in_csv = tmp_path / "v2_ip.csv"
    out_csv = tmp_path / "v2_op.csv"
    with in_csv.open("w", encoding="utf-8") as f:
        for row in obs_seq:
            f.write(",".join(f"{v:.17e}" for v in row) + "\n")

    proc = subprocess.run(
        [str(cpp_bin), "--bench-latency", str(v2_bin), str(in_csv), str(out_csv)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "LATENCY_BENCH_V2:" in proc.stdout

    lines = [l.strip() for l in out_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 24

    h, c = py_net.zero_state()
    max_q_err = 0.0
    max_gain_err = 0.0
    for t, line in enumerate(lines):
        py_out = py_net.step(obs_seq[t], h, c)
        h, c = py_out["h_next"], py_out["c_next"]
        parts = line.split(",")
        assert int(parts[0]) == py_out["action_id"]
        assert int(parts[3]) == py_out["bp_iterations"]
        assert abs(float(parts[2]) - py_out["pair_weight_scale"]) < 1e-12

        cpp_q = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        cpp_g = np.array([float(x) for x in parts[10:19]], dtype=np.float64)
        max_q_err = max(max_q_err, float(np.max(np.abs(cpp_q - py_out["q_values"]))))
        max_gain_err = max(max_gain_err, float(np.max(np.abs(cpp_g - py_out["qubit_gains"]))))
        assert np.all(np.abs(cpp_g) <= 0.30 + 1e-12)

    assert max_q_err < 1e-12, f"V2 Q-value parity error {max_q_err}"
    assert max_gain_err < 1e-12, f"V2 Gain parity error {max_gain_err}"


def test_dep02_checkpoint_version_distinction_and_v1_reproducibility(tmp_path: Path) -> None:
    """DEP-02: V2 loader rejects V1 checkpoint, V1 loader rejects V2 checkpoint, and frozen V1 retains < 1e-12 parity."""
    repo_root = Path(__file__).resolve().parents[2]
    v1_bin = repo_root / "checkpoints" / "v1" / "r2d2_v1_trained.bin"
    v2_bin = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    assert v1_bin.exists()
    assert v2_bin.exists()

    cpp_bin = repo_root / "tests" / "build_layer2" / "cpp_layer2_parity_runner"
    if not cpp_bin.exists():
        cpp_bin = _compile_parity_runner(repo_root, tmp_path / "cpp_layer2_parity_runner")

    # 1. V2 C++ runtime must strictly reject V1 checkpoint
    r1 = subprocess.run(
        [str(cpp_bin), "--expect-fail-v2", str(v1_bin)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "EXPECTED_REJECT_V2" in r1.stdout

    # 2. V1 C++ runtime must strictly reject V2 checkpoint
    r2 = subprocess.run(
        [str(cpp_bin), "--expect-fail-v1", str(v2_bin)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "EXPECTED_REJECT_V1" in r2.stdout

    # 3. Frozen V1 checkpoint still reproduces < 1e-12 parity via `--v1`
    py_v1, _ = load_r2d2_checkpoint_bin(v1_bin, expected_version=1)
    rng = np.random.default_rng(20261115)
    obs_v1 = rng.uniform(0.01, 0.20, size=(8, 8)).astype(np.float64)
    in_v1 = tmp_path / "v1_ip.csv"
    out_v1 = tmp_path / "v1_op.csv"
    with in_v1.open("w", encoding="utf-8") as f:
        for row in obs_v1:
            f.write(",".join(f"{v:.17e}" for v in row) + "\n")

    subprocess.run([str(cpp_bin), "--v1", str(v1_bin), str(in_v1), str(out_v1)], check=True)
    lines = [l.strip() for l in out_v1.read_text(encoding="utf-8").splitlines() if l.strip()]
    h, c = py_v1.zero_state()
    for t, line in enumerate(lines):
        py_out = py_v1.step(obs_v1[t], h, c)
        h, c = py_out["h_next"], py_out["c_next"]
        parts = line.split(",")
        cpp_q = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        assert float(np.max(np.abs(cpp_q - py_out["q_values"]))) < 1e-12
