"""Layer 2 IP/OP Validation Suite: R2D2 V2 82-Input Syndrome-Conditioned Parity & Frozen V1 Check."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import numpy as np

from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def test_layer2_cpp_python_ip_op_parity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    v2_bin = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
    cpp_bin = repo_root / "tests" / "build_layer2" / "cpp_layer2_parity_runner"
    if not cpp_bin.exists():
        cpp_bin = Path(os.environ.get("LAYER2_CPP_BIN", str(cpp_bin)))
    if not cpp_bin.exists():
        cpp_bin = tmp_path / "cpp_layer2_parity_runner"
        subprocess.run(
            [
                "g++", "-O2", "-std=c++17", "-Wall", f"-I{repo_root}",
                str(repo_root / "tests" / "cpp_layer2_parity_runner.cpp"),
                str(repo_root / "deployment" / "cpp" / "model_loader.cpp"),
                str(repo_root / "deployment" / "cpp" / "r2d2_runtime.cpp"),
                str(repo_root / "deployment" / "cpp" / "observation_adapter.cpp"),
                "-o", str(cpp_bin), "-lm",
            ],
            check=True,
        )

    assert v2_bin.exists(), "Missing checkpoints/v2/r2d2_v2_trained.bin"
    py_net, state_dict = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)
    assert py_net.input_dim == 82
    assert py_net.hidden_dim == 64
    assert state_dict["W_ih"].shape == (256, 82)
    assert state_dict["W_hh"].shape == (256, 64)

    regimes = [
        "gaussian_stationary",
        "ou_drift",
        "zz_crosstalk",
        "tls_telegraph",
        "nonstationary_burst",
    ]
    obs_blocks = [
        generate_v2_qec_episode(r, seed=202610 + i, split_name="test", num_epochs=6).obs
        for i, r in enumerate(regimes)
    ]
    obs_seq = np.vstack(obs_blocks)
    assert obs_seq.shape == (30, 82)

    in_csv = tmp_path / "ip_v2.csv"
    out_csv = tmp_path / "op_v2.csv"
    with in_csv.open("w", encoding="utf-8") as f:
        for row in obs_seq:
            f.write(",".join(f"{v:.17e}" for v in row) + "\n")

    subprocess.run([str(cpp_bin), str(v2_bin), str(in_csv), str(out_csv)], check=True)
    lines = [l.strip() for l in out_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 30

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
        assert int(parts[3]) in {3, 4, 5}

    assert max_q_err < 1e-12, f"Q-value parity error {max_q_err}"
    assert max_gain_err < 1e-12, f"Gain parity error {max_gain_err}"
