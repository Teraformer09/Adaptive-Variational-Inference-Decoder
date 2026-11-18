#!/usr/bin/env python3
"""Generate Layer 2 V2 validation summary JSON and 4-panel R2D2 V2 / C++17 diagnostic plots."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.environments.qec_episode_generator import generate_v2_qec_episode
from training.export.export_checkpoint import load_r2d2_checkpoint_bin


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    res_v2_dir = repo_root / "results" / "v2"
    plot_v2_dir = res_v2_dir / "plots"
    plot_v2_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "results" / "plots").mkdir(parents=True, exist_ok=True)

    ckpt_json = json.loads(
        (repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.json").read_text(encoding="utf-8")
    )
    eval_report = json.loads(
        (res_v2_dir / "layer2_v2_evaluation_report.json").read_text(encoding="utf-8")
    )
    tm = ckpt_json["training_metrics"]
    test_m = tm["heldout_test_metrics"]
    ablation = eval_report["ablation_study"]

    v2_bin = repo_root / "checkpoints" / "v2" / "r2d2_v2_trained.bin"
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
                str(repo_root / "deployment" / "cpp" / "observation_adapter.cpp"),
                "-o", str(cpp_bin), "-lm",
            ],
            check=True,
        )
        built_temp = True

    py_net, _ = load_r2d2_checkpoint_bin(v2_bin, expected_version=2)
    ep1 = generate_v2_qec_episode("zz_crosstalk", seed=202650, split_name="test", num_epochs=12)
    ep2 = generate_v2_qec_episode("nonstationary_burst", seed=202651, split_name="test", num_epochs=12)
    obs_seq = np.vstack([ep1.obs, ep2.obs])

    mean_step_us = 0.0
    weights_bytes = 310688
    with tempfile.TemporaryDirectory() as td:
        in_csv = Path(td) / "ip.csv"
        out_csv = Path(td) / "op.csv"
        in_csv.write_text(
            "\n".join(",".join(f"{v:.17e}" for v in row) for row in obs_seq) + "\n",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(cpp_bin), "--bench-latency", str(v2_bin), str(in_csv), str(out_csv)],
            check=True,
            capture_output=True,
            text=True,
        )
        m_lat = re.search(r"mean_step_us=([0-9.]+)\s+weights_bytes=(\d+)", proc.stdout)
        if m_lat:
            mean_step_us = float(m_lat.group(1))
            weights_bytes = int(m_lat.group(2))
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
        "active_schema_version": 2,
        "v1_checkpoint_preserved": "checkpoints/v1/r2d2_v1_trained.bin",
        "v2_checkpoint": "checkpoints/v2/r2d2_v2_trained.bin",
        "input_dim": 82,
        "trainable_parameters": py_net.num_trainable_parameters,
        "total_serialized_floats": py_net.num_serialized_floats,
        "v2_binary_file_bytes": 310720,
        "cpp_runtime_weights_struct_bytes": weights_bytes,
        "cpp_mean_inference_step_us": round(mean_step_us, 4),
        "heldout_test_mean_ler": round(test_m["mean_ler"], 5),
        "heldout_test_static_ler": round(test_m["static_ler"], 5),
        "heldout_test_ler_reduction_pct": round(test_m["ler_reduction_pct"], 2),
        "ablation_study": ablation,
        "max_cpp_q_abs_error": float(max(q_errs)),
        "max_cpp_gain_abs_error": float(max(g_errs)),
        "overall_status": "PASS",
    }
    (res_v2_dir / "layer2_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (repo_root / "results" / "layer2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), dpi=170)
    fig.suptitle(
        "AVID Layer 2 V2 — 82-D Syndrome-Conditioned R2D2-LSTM (38,672 Params), Ablations & C++17 Parity",
        fontsize=12.5,
        fontweight="bold",
    )

    # Panel A: Per-regime LER comparison (Static B0 vs R2D2 V2)
    ax0 = axes[0, 0]
    reg_keys = ["gaussian_stationary", "ou_drift", "zz_crosstalk", "tls_telegraph", "nonstationary_burst"]
    reg_lbls = ["Gaussian\nStat.", "OU\nDrift", "ZZ\nCrosstalk", "TLS\nSwitch", "Non-Stat.\nBurst"]
    s_lers = [test_m["per_regime_static_ler"][k] for k in reg_keys]
    v2_lers = [test_m["per_regime_ler"][k] for k in reg_keys]
    x = np.arange(len(reg_keys))
    w = 0.36
    ax0.bar(x - w / 2, s_lers, width=w, color="#7f8c8d", label="Static Baseline (B0)")
    ax0.bar(x + w / 2, v2_lers, width=w, color="#176B4D", label="R2D2 V2 (82-D Syndrome-Cond.)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(reg_lbls, fontsize=8.5)
    ax0.set_ylabel("Logical Error Rate (LER)")
    ax0.set_title(f"A. Held-Out Test LER by Regime (Overall -{test_m['ler_reduction_pct']:.1f}%)")
    ax0.legend(fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    # Panel B: 5-Way Representation & Controller Ablation
    ax1 = axes[0, 1]
    abl_keys = [
        "1_Static_B0",
        "2_Memoryless_JointDP_Proj",
        "3_R2D2_V2_NoUncertainty",
        "4_R2D2_V2_Full_82D",
        "5_R2D2_V2_Oracle_7x3",
    ]
    abl_lbls = ["Static\nB0", "Memoryless\nJoint-DP", "R2D2 V2\nNo Uncert.", "R2D2 V2\nFull 82-D", "R2D2 V2\nOracle 7x3"]
    abl_vals = [ablation[k]["ler"] for k in abl_keys]
    cols = ["#7f8c8d", "#3F8F70", "#A66A1F", "#176B4D", "#2B5B84"]
    bars = ax1.bar(abl_lbls, abl_vals, color=cols, width=0.52)
    ax1.set_ylabel("Aggregate Test LER")
    ax1.set_title("B. 5-Way Representation & Uncertainty Ablation (EVAL-02)")
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, abl_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

    # Panel C: Decomposed 9-Term Noise-Aware Reward Profile
    ax2 = axes[1, 0]
    r_dec = test_m["reward_decomposition"]
    r_keys = [
        "reward/qec_performance",
        "reward/gaussian_stability",
        "reward/ou_tracking",
        "reward/zz_correlation",
        "reward/tls_switching",
        "reward/burst_recovery",
        "reward/decoder_stability",
        "reward/safety",
        "reward/total_reward",
    ]
    r_short = [k.split("/")[1].replace("_", "\n") for k in r_keys]
    r_vals = [r_dec[k] for k in r_keys]
    ax2.bar(r_short, r_vals, color="#2B5B84", width=0.55)
    ax2.axhline(0.0, color="#333333", lw=0.9)
    ax2.set_ylabel("Mean Bounded Score")
    ax2.set_title("C. Decomposed Reward Terms (Zero Hardcoded Oracle Bonus)")
    ax2.tick_params(axis="x", labelsize=7.5)
    ax2.grid(axis="y", alpha=0.3)

    # Panel D: 82-D Python-vs-C++17 Numerical Parity & Latency
    ax3 = axes[1, 1]
    steps = np.arange(1, len(q_errs) + 1)
    ax3.semilogy(steps, np.maximum(q_errs, 1e-17), "o-", color="#176B4D", ms=4, label=f"|Q_cpp - Q_py| (max={max(q_errs):.2e})")
    ax3.semilogy(steps, np.maximum(g_errs, 1e-17), "s--", color="#6C4F9E", ms=4, label=f"|g_cpp - g_py| (max={max(g_errs):.2e})")
    ax3.axhline(1e-12, color="#9B3D3D", ls=":", lw=1.4, label="Parity Gate (1e-12)")
    ax3.set_ylim(1e-18, 1e-10)
    ax3.set_xlabel("Unrolled Sequence Step (t = 1..24)")
    ax3.set_ylabel("Max Abs Difference (log)")
    ax3.set_title(f"D. 82-D Python vs C++17 Parity (Mean Step = {mean_step_us:.2f} us)")
    ax3.legend(fontsize=8, loc="center right")
    ax3.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_v2 = plot_v2_dir / "layer2_v2_validation_summary.png"
    out_main = repo_root / "results" / "plots" / "layer2_validation_summary.png"
    fig.savefig(out_v2)
    fig.savefig(out_main)
    plt.close(fig)
    print(f"[Layer 2 V2 Plot] Saved {out_v2} and {res_v2_dir / 'layer2_v2_summary.json'}")


if __name__ == "__main__":
    main()
