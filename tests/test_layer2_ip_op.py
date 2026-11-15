"""Layer 2 IP/OP Validation Suite: R2D2 LSTM Training, Export & <1e-12 C++ Parity."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import numpy as np

from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def test_layer2_cpp_python_ip_op_parity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "checkpoints" / "r2d2_v1_trained.bin"
    cpp_bin = repo_root / "tests" / "build_layer2" / "cpp_layer2_parity_runner"
    if not cpp_bin.exists():
        cpp_bin = Path(os.environ.get("LAYER2_CPP_BIN", str(cpp_bin)))
    assert bin_path.exists(), "Missing checkpoints/r2d2_v1_trained.bin"
    assert cpp_bin.exists(), f"Missing compiled {cpp_bin}"

    py_net, state_dict = load_r2d2_checkpoint_bin(bin_path)
    assert py_net.hidden_dim == 64
    assert state_dict["W_hh"].shape == (256, 64)

    # 8-feature observation IP spanning all 6 operating regimes
    rng = np.random.default_rng(20261115)
    base = np.array([
        [0.014, 0.016, 0.006, 0.009, 0.055, 0.015, 0.002, 0.05],
        [0.018, 0.024, 0.016, 0.045, 0.085, 0.055, 0.006, 0.08],
        [0.022, 0.034, 0.022, 0.092, 0.115, 0.105, 0.008, 0.10],
        [0.032, 0.036, 0.007, 0.010, 0.095, 0.018, 0.004, 0.06],
        [0.036, 0.052, 0.026, 0.095, 0.145, 0.115, 0.048, 0.82],
        [0.210, 0.250, 0.125, 0.160, 0.550, 0.250, 0.090, 0.15],
    ], dtype=np.float64)
    obs_seq = np.clip(np.vstack([base] * 4) + rng.normal(0.0, 0.0005, (24, 8)), 0.001, 0.95)

    in_csv = tmp_path / "ip.csv"
    out_csv = tmp_path / "op.csv"
    with in_csv.open("w", encoding="utf-8") as f:
        for row in obs_seq:
            f.write(",".join(f"{v:.15e}" for v in row) + "\n")

    subprocess.run([str(cpp_bin), str(bin_path), str(in_csv), str(out_csv)], check=True)
    lines = [l.strip() for l in out_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 24

    h, c = py_net.zero_state()
    max_q_err = 0.0
    max_gain_err = 0.0
    actions_seen = set()
    for t, line in enumerate(lines):
        py_out = py_net.step(obs_seq[t], h, c)
        h, c = py_out["h_next"], py_out["c_next"]
        parts = line.split(",")
        assert int(parts[0]) == py_out["action_id"]
        assert int(parts[3]) == py_out["bp_iterations"]
        assert abs(float(parts[2]) - py_out["pair_weight_scale"]) < 1e-12
        actions_seen.add(int(parts[0]))
        cpp_q = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        cpp_g = np.array([float(x) for x in parts[10:19]], dtype=np.float64)
        max_q_err = max(max_q_err, float(np.max(np.abs(cpp_q - py_out["q_values"]))))
        max_gain_err = max(max_gain_err, float(np.max(np.abs(cpp_g - py_out["qubit_gains"]))))
        assert np.all(np.abs(cpp_g) <= 0.30 + 1e-12)
        assert int(parts[3]) in {3, 4, 5}

    assert max_q_err < 1e-12, f"Q-value parity error {max_q_err}"
    assert max_gain_err < 1e-12, f"Gain parity error {max_gain_err}"
    assert len(actions_seen) >= 4
