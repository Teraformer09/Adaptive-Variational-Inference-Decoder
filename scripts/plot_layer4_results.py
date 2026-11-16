#!/usr/bin/env python3
"""Generate Layer 4 validation summary JSON and 5-set / G1-G8 / RTL cycle plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default=str(repo_root / "validation" / "layer4" / "runs" / "remediated_full_layer4_v1"),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gate_doc = json.loads((run_dir / "acceptance" / "gate_evaluation.json").read_text(encoding="utf-8"))

    res_dir = repo_root / "results"
    plot_dir = res_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    sets = gate_doc["set_summaries"]
    gates = gate_doc["gates"]
    tb = gate_doc["standalone_rtl_tb"]

    summary = {
        "layer": 4,
        "branch": "layer4-cosim-bridge-emu",
        "run_id": gate_doc["run_id"],
        "overall_status": gate_doc["overall_status"],
        "canonical_id_count": gate_doc["canonical_id_count"],
        "pass_count": gate_doc["pass_count"],
        "code_git_dirty": gate_doc["code_git_dirty"],
        "total_rtl_cycles_simulated": sum(s["rtl_cycles_total"] for s in sets.values()),
        "gates_passed": [gid for gid, g in gates.items() if g["status"] == "PASS"],
        "k3_bp_core_cycles": tb["k3_bp_core_cycles"],
        "k3_axis_total_cycles": tb["k3_axis_total_cycles"],
        "native_verilator_compiled": tb["native_verilator_compiled"],
    }
    (res_dir / "layer4_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=160)
    fig.suptitle("AVID Layer 4 — Co-Simulation Bridge, QEMU ARM/ARM64 & Verilator RTL (223/223 PASS)", fontsize=12, fontweight="bold")

    # Panel 1: 5-Set Pass Counts
    s_order = ["Set1", "Set2", "Set3", "Set4", "Set5"]
    s_labels = ["Set 1\nInput/RPC", "Set 2\nACK/Cycle", "Set 3\nQEMU Lock", "Set 4\nFault/Stress", "Set 5\nAXI/RTL"]
    s_pass = [sets[k]["counts"].get("PASS", 0) for k in s_order]
    s_tot = [sets[k]["total"] for k in s_order]
    b0 = axes[0].bar(s_labels, s_pass, color="#1f77b4", width=0.55)
    axes[0].set_ylim(0, 66)
    axes[0].set_title("5-Set Canonical Suite (223/223 PASS)")
    axes[0].set_ylabel("Passing Test IDs")
    for bar, p, t in zip(b0, s_pass, s_tot):
        axes[0].text(bar.get_x() + bar.get_width() / 2, p + 1.2, f"{p}/{t}", ha="center", fontsize=8.3, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Gates G1..G8
    g_order = [f"G{i}" for i in range(1, 9)]
    g_pass = [gates[g]["pass_cases"] for g in g_order]
    g_tot = [gates[g]["total_cases"] for g in g_order]
    b1 = axes[1].bar(g_order, g_pass, color="#2ca02c", width=0.58)
    axes[1].set_ylim(0, 88)
    axes[1].set_title("Acceptance Gates G1–G8 (All PASS)")
    axes[1].set_xlabel("Acceptance Gate")
    axes[1].set_ylabel("Verified Gate Cases")
    for bar, p, t in zip(b1, g_pass, g_tot):
        axes[1].text(bar.get_x() + bar.get_width() / 2, p + 1.5, f"{p}/{t}", ha="center", fontsize=7.8, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: Native Verilator Cycle Matrix K=1..5
    cmat = tb["cycle_matrix_K1_to_K5"]
    ks = [r["K"] for r in cmat]
    core_c = [r["bp_core_cycles"] for r in cmat]
    axis_c = [r["axis_total_cycles"] for r in cmat]
    axes[2].plot(ks, core_c, "o-", color="#9467bd", lw=2.0, label="Vaa_qec_top Core (2+5K)")
    axes[2].plot(ks, axis_c, "s--", color="#ff7f0e", lw=2.0, label="Bridge AXI-S Total (7+5K)")
    for k, cc, ac in zip(ks, core_c, axis_c):
        axes[2].annotate(f"{cc}c", (k, cc), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)
        axes[2].annotate(f"{ac}c", (k, ac), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axes[2].set_xticks(ks)
    axes[2].set_ylim(0, 38)
    axes[2].set_title("Verilator Vaa_qec_top Cycle Accounting")
    axes[2].set_xlabel("BP Iterations (K)")
    axes[2].set_ylabel("Clock Cycles (@ 4.0 ns)")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    out_png = plot_dir / "layer4_validation_summary.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[Layer 4 Plot] Saved {out_png} and {res_dir / 'layer4_summary.json'}")


if __name__ == "__main__":
    main()
