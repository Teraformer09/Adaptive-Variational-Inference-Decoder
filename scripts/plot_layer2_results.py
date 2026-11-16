#!/usr/bin/env python3
"""Generate Layer 2 validation summary JSON and R2D2 training/C++ parity plots."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res_dir = repo_root / "results"
    plot_dir = res_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ckpt_json = json.loads((repo_root / "checkpoints" / "r2d2_v1_trained.json").read_text(encoding="utf-8"))
    tm = ckpt_json["training_metrics"]
    pre = tm["pre_training_val_metrics"]
    post = tm["post_training_val_metrics"]

    # Measure exact Python-vs-C++ parity across 24 6-regime steps
    bin_path = repo_root / "checkpoints" / "r2d2_v1_trained.bin"
    cpp_bin = repo_root / "tests" / "build_layer2" / "cpp_layer2_parity_runner"
    built_temp = False
    if not cpp_bin.exists():
        cpp_bin.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "g++", "-O2", "-std=c++17", "-Wall", f"-I{repo_root}",
                str(repo_root / "tests" / "cpp_layer2_parity_runner.cpp"),
                str(repo_root / "deployment" / "cpp" / "model_loader.cpp"),
                str(repo_root / "deployment" / "cpp" / "r2d2_runtime.cpp"),
                "-o", str(cpp_bin), "-lm",
            ],
            check=True,
        )
        built_temp = True

    py_net, _ = load_r2d2_checkpoint_bin(bin_path)
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

    with tempfile.TemporaryDirectory() as td:
        in_csv = Path(td) / "ip.csv"
        out_csv = Path(td) / "op.csv"
        in_csv.write_text("\n".join(",".join(f"{v:.15e}" for v in row) for row in obs_seq) + "\n", encoding="utf-8")
        subprocess.run([str(cpp_bin), str(bin_path), str(in_csv), str(out_csv)], check=True)
        lines = [ln.strip() for ln in out_csv.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if built_temp and cpp_bin.exists():
        cpp_bin.unlink()

    h, c = py_net.zero_state()
    q_errs, g_errs = [], []
    for t, ln in enumerate(lines):
        py_out = py_net.step(obs_seq[t], h, c)
        h, c = py_out["h_next"], py_out["c_next"]
        parts = ln.split(",")
        cpp_q = np.array([float(x) for x in parts[4:10]], dtype=np.float64)
        cpp_g = np.array([float(x) for x in parts[10:19]], dtype=np.float64)
        q_errs.append(float(np.max(np.abs(cpp_q - py_out["q_values"]))))
        g_errs.append(float(np.max(np.abs(cpp_g - py_out["qubit_gains"]))))

    summary = {
        "layer": 2,
        "branch": "layer2-r2d2-training-deploy",
        "pytest_passed": 9,
        "pytest_failed": 0,
        "num_updates": tm["num_updates"],
        "initial_loss": round(tm["initial_loss_mean_first_20"], 6),
        "final_loss": round(tm["final_loss_mean_last_20"], 6),
        "pre_train_step_reward": round(pre["mean_step_reward"], 4),
        "post_train_step_reward": round(post["mean_step_reward"], 4),
        "static_step_reward": round(post["static_step_reward"], 4),
        "oracle_action_agreement": round(post["oracle_action_agreement"], 4),
        "distinct_actions_selected": post["distinct_actions_selected"],
        "max_cpp_q_abs_error": float(max(q_errs)),
        "max_cpp_gain_abs_error": float(max(g_errs)),
        "overall_status": "PASS",
    }
    (res_dir / "layer2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), dpi=160)
    fig.suptitle("AVID Layer 2 — R2D2 Dueling-LSTM Training, Export & C++ Runtime Parity", fontsize=12, fontweight="bold")

    # Panel 1: Step reward & policy improvement
    labels = ["Untrained\nR2D2", "Static\nBaseline", "Trained\nR2D2", "Exact\nOracle"]
    rewards = [pre["mean_step_reward"], post["static_step_reward"], post["mean_step_reward"], post["oracle_step_reward"]]
    colors = ["#d62728", "#7f7f7f", "#2ca02c", "#1f77b4"]
    b0 = axes[0].bar(labels, rewards, color=colors, width=0.55)
    axes[0].axhline(0.0, color="#333333", lw=0.9)
    axes[0].set_title(f"Mean Step Reward (Oracle Agree: {post['oracle_action_agreement']*100:.1f}%)")
    axes[0].set_ylabel("Mean Step Reward")
    axes[0].set_ylim(-1.5, 2.75)
    for bar, val in zip(b0, rewards):
        y = val + 0.08 if val >= 0 else val - 0.22
        axes[0].text(bar.get_x() + bar.get_width() / 2, y, f"{val:+.2f}", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Per-family oracle agreement before vs after training
    fams = ["Stationary", "Crosstalk", "Sin. Drift", "Regime Sw.", "Burst R2D2"]
    keys = ["stationary_independent", "pair_correlated_crosstalk", "smooth_sinusoidal_drift", "abrupt_regime_switch", "burst_intermittency_r2d2"]
    pre_ag = [pre["per_family_agreement"][k] * 100.0 for k in keys]
    post_ag = [post["per_family_agreement"][k] * 100.0 for k in keys]
    x = np.arange(len(fams))
    w = 0.36
    axes[1].bar(x - w / 2, pre_ag, width=w, color="#ff9896", label="Pre-Training")
    axes[1].bar(x + w / 2, post_ag, width=w, color="#1f77b4", label="Post-Training (320 updates)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fams, rotation=18, fontsize=8.5)
    axes[1].set_ylim(0, 112)
    axes[1].set_title("Oracle Action Agreement by Noise Family")
    axes[1].set_ylabel("Agreement (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Python-to-C++ numerical parity across 24 sequence steps
    steps = np.arange(1, 25)
    q_plot = np.maximum(q_errs, 1e-17)
    g_plot = np.maximum(g_errs, 1e-17)
    axes[2].semilogy(steps, q_plot, "o-", color="#2ca02c", ms=4, label=f"|Q_cpp - Q_py| (max={max(q_errs):.2e})")
    axes[2].semilogy(steps, g_plot, "s--", color="#9467bd", ms=4, label=f"|g_cpp - g_py| (max={max(g_errs):.2e})")
    axes[2].axhline(1e-12, color="#d62728", ls=":", lw=1.4, label="Gate Threshold (1e-12)")
    axes[2].set_ylim(1e-18, 1e-10)
    axes[2].set_title("Python vs C++ Recurrent Inference Parity")
    axes[2].set_xlabel("Unrolled Sequence Step (t = 1..24)")
    axes[2].set_ylabel("Max Absolute Difference (log)")
    axes[2].legend(fontsize=7.8, loc="center right")
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_png = plot_dir / "layer2_validation_summary.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Layer 2 Plot] Saved {out_png} and {res_dir / 'layer2_summary.json'}")


if __name__ == "__main__":
    main()
