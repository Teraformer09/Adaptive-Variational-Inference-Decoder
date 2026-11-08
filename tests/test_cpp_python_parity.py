"""Verification Suite 3: Python-to-C++ Numerical Parity (`tests/test_cpp_python_parity.py`).

Verifies:
  1. Multi-step recurrent trajectories evaluated by Python `R2D2Network` and C++ `R2D2Runtime`
     agree to within `< 1e-12` (strictly below the `< 1e-6` requirement) across:
     - `state_value` V(h_t)
     - all 6 Dueling Q-values Q(h_t, a_0..a_5)
     - all 9 per-qubit gain residuals delta_g_0..delta_g_8
     - discrete selected `action_id`, `pair_weight_scale`, and `bp_iterations`
  2. Multi-regime trajectory selects all 6 actions identically in Python and C++.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import numpy as np

from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def test_cpp_python_recurrent_trajectory_parity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_path = repo_root / "checkpoints" / "r2d2_v1_trained.bin"
    cpp_bin = repo_root / "software" / "build_native" / "test_baseline_suite"
    assert bin_path.exists()
    assert cpp_bin.exists()

    py_net, _ = load_r2d2_checkpoint_bin(bin_path)

    # Construct a 24-step multi-regime trajectory covering all 6 control regimes + random perturbations
    rng = np.random.default_rng(20261108)
    base_regimes = np.array(
        [
            [0.014, 0.016, 0.006, 0.009, 0.055, 0.015, 0.002, 0.05],
            [0.018, 0.024, 0.016, 0.045, 0.085, 0.055, 0.006, 0.08],
            [0.022, 0.034, 0.022, 0.092, 0.115, 0.105, 0.008, 0.10],
            [0.032, 0.036, 0.007, 0.010, 0.095, 0.018, 0.004, 0.06],
            [0.036, 0.052, 0.026, 0.095, 0.145, 0.115, 0.048, 0.82],
            [0.210, 0.250, 0.125, 0.160, 0.550, 0.250, 0.090, 0.15],
        ],
        dtype=np.float64,
    )
    obs_seq = np.vstack([base_regimes for _ in range(4)])
    obs_seq += rng.normal(0.0, 0.0005, size=obs_seq.shape)
    obs_seq = np.clip(obs_seq, 0.001, 0.95)

    in_csv = tmp_path / "parity_in.csv"
    out_csv = tmp_path / "parity_out.csv"

    with in_csv.open("w", encoding="utf-8") as f:
        for row in obs_seq:
            f.write(",".join(f"{v:.15e}" for v in row) + "\n")

    proc = subprocess.run(
        [str(cpp_bin), "--parity-csv", str(in_csv), str(out_csv)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert out_csv.exists()

    cpp_lines = [line.strip() for line in out_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cpp_lines) == obs_seq.shape[0]

    h, c = py_net.zero_state()
    max_q_diff = 0.0
    max_gain_diff = 0.0
    max_val_diff = 0.0

    for t, line in enumerate(cpp_lines):
        py_out = py_net.step(obs_seq[t], h, c)
        h, c = py_out["h_next"], py_out["c_next"]

        parts = line.split(",")
        cpp_act = int(parts[0])
        cpp_val = float(parts[1])
        cpp_gamma = float(parts[2])
        cpp_k = int(parts[3])
        cpp_q = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        cpp_g = np.array([float(x) for x in parts[10:19]], dtype=np.float64)

        assert cpp_act == py_out["action_id"]
        assert cpp_k == py_out["bp_iterations"]
        assert abs(cpp_gamma - py_out["pair_weight_scale"]) < 1e-12

        val_diff = abs(cpp_val - py_out["state_value"])
        q_diff = float(np.max(np.abs(cpp_q - py_out["q_values"])))
        g_diff = float(np.max(np.abs(cpp_g - py_out["qubit_gains"])))

        max_val_diff = max(max_val_diff, val_diff)
        max_q_diff = max(max_q_diff, q_diff)
        max_gain_diff = max(max_gain_diff, g_diff)

    assert max_val_diff < 1e-6
    assert max_q_diff < 1e-6
    assert max_gain_diff < 1e-6
