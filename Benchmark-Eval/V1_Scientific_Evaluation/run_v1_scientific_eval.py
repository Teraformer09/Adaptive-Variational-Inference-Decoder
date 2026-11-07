#!/usr/bin/env python3
'''
Master Runner for Benchmark-Eval/V1_Scientific_Evaluation/:
Executes the compiled C++ `avid_daemon` + Verilator/Bridge co-simulation across:
  1. `baselines/`: Shared-stream comparison of B0, B1, B2, B3, AVID V1 (Joint-DP + 64-Unit LSTM R2D2-WB), and B4 Oracle.
  2. `ablations/`: 6-Way V1 C++ Runtime & Hardware OSD Mode Ablation (`cfg_osd_mode = 0, 1, 2`).
  3. `multi_seed/`: 10-Seed (`2026..2035`) live C++ `avid_daemon` co-simulation across key non-stationary & correlated regimes.
  4. `statistical_analysis/`: Two-proportion z-tests, 10-seed paired t-tests, Wilcoxon signed-rank tests, Cohen's d, and Holm-Bonferroni adjusted p-values.
  5. `results/`: Master JSON summary and publication-ready Markdown comparison tables.
'''
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Tuple

V1_SCI_DIR = Path(__file__).resolve().parent
BM_ROOT = V1_SCI_DIR.parent
REPO_ROOT = BM_ROOT.parent
sys.path.insert(0, str(BM_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.metrics.qec_metrics import wilson_ci95, ler_reduction_pct
from common.statistical_tests.paired_significance import (
    two_proportion_z_test,
    paired_multiseed_tests,
    holm_bonferroni_correction,
)


def ensure_binaries_built() -> Tuple[Path, Path]:
    daemon_bin = REPO_ROOT / "emulator/qemu/images/avid_daemon_x86_64"
    bridge_bin = REPO_ROOT / "emulator/verilator/models/avid_cosim_bridge"
    if not daemon_bin.exists() or not bridge_bin.exists():
        subprocess.run(["bash", str(REPO_ROOT / "scripts/build_all.sh")], check=True)
    return daemon_bin, bridge_bin


def run_daemon_bridge_pair(
    daemon_bin: Path,
    bridge_bin: Path,
    scenario: str,
    rounds: int,
    window: int,
    seed: int,
    controller_mode: str = "avid_r2d2",
    osd_mode: int = 1,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="avid_v1_eval_") as tmpdir:
        tmp = Path(tmpdir)
        sock = tmp / "bridge.sock"
        b_json = tmp / "bridge.json"
        b_jsonl = tmp / "bridge.jsonl"
        q_json = tmp / "qemu.json"

        b_proc = subprocess.Popen(
            [
                str(bridge_bin),
                "--socket", str(sock),
                "--clock-period-ns", "4",
                "--max-clients", "1",
                "--metrics-out", str(b_json),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(200):
                if sock.exists():
                    break
                time.sleep(0.005)
            subprocess.run(
                [
                    str(daemon_bin),
                    "--backend", "sim",
                    "--socket", str(sock),
                    "--scenario", scenario,
                    "--rounds", str(rounds),
                    "--window", str(window),
                    "--seed", str(seed),
                    "--controller-mode", controller_mode,
                    "--osd-mode", str(osd_mode),
                    "--metrics-out", str(q_json),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            b_proc.wait(timeout=15)
        finally:
            if b_proc.poll() is None:
                b_proc.kill()
        return json.loads(q_json.read_text(encoding="utf-8"))


def run_step_1_baselines(daemon_bin: Path, bridge_bin: Path) -> dict:
    base_dir = V1_SCI_DIR / "baselines"
    scenarios = ["stationary_noise", "slowly_drifting_noise", "correlated_noise", "burst_disturbance", "short_burst_60us_vs_10ms"]
    rows = []
    md_rows = []
    for scen in scenarios:
        m_static = run_daemon_bridge_pair(daemon_bin, bridge_bin, scen, 1536, 256, 2026, "static", 1)
        m_marg = run_daemon_bridge_pair(daemon_bin, bridge_bin, scen, 1536, 256, 2026, "marginal_lbfgs", 1)
        m_proj = run_daemon_bridge_pair(daemon_bin, bridge_bin, scen, 1536, 256, 2026, "joint_dp_proj", 1)
        m_avid = run_daemon_bridge_pair(daemon_bin, bridge_bin, scen, 1536, 256, 2026, "avid_r2d2", 1)

        row = {
            "scenario": scen,
            "rounds": 1536,
            "seed": 2026,
            "B0_Static_BP_LER": m_static["logical_error_rate"],
            "B1_Marginal_LBFGS_LER": m_marg["logical_error_rate"],
            "B3_JointDP_Memoryless_PairProj_LER": m_proj["logical_error_rate"],
            "B4_Full_AVID_LSTM64_R2D2_WB_LER": m_avid["logical_error_rate"],
            "AVID_vs_B0_Reduction_Pct": m_avid["ler_reduction_vs_static_pct"],
            "AVID_Burst_Window_Reduction_Pct": m_avid["burst_window_ler_reduction_pct"],
            "AVID_CI95_Low": m_avid["logical_error_rate_ci95_low"],
            "AVID_CI95_High": m_avid["logical_error_rate_ci95_high"],
        }
        rows.append(row)
        md_rows.append(
            f"| `{scen}` | `{row['B0_Static_BP_LER']:.5f}` | `{row['B1_Marginal_LBFGS_LER']:.5f}` | "
            f"`{row['B3_JointDP_Memoryless_PairProj_LER']:.5f}` | **`{row['B4_Full_AVID_LSTM64_R2D2_WB_LER']:.5f}`** | "
            f"**`-{row['AVID_vs_B0_Reduction_Pct']:.2f}%`** | **`-{row['AVID_Burst_Window_Reduction_Pct']:.2f}%`** |"
        )

    with (base_dir / "v1_shared_stream_baselines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = {"experiment": "V1_Shared_Stream_Fair_Baselines", "seed": 2026, "rounds_per_scenario": 1536, "results": rows}
    (base_dir / "v1_shared_stream_baselines.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (base_dir / "README.md").write_text(
        "# V1 Fair Shared-Stream Baseline Comparison (`B0`, `B1`, `Memoryless Joint-DP`, `Full AVID R2D2-WB`)\n\n"
        "| Scenario (`1,536` Rounds, `seed=2026`) | `B0` Static BP LER | `B1` 4-Marginal L-BFGS LER | `Joint-DP` Memoryless Pair-Proj LER | **Full AVID V1 (`LSTM-64 R2D2-WB`) LER** | Whole-Run Reduction | Burst-Window Reduction |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(md_rows)
        + "\n\n## Artifacts\n"
        "- [`v1_shared_stream_baselines.json`](v1_shared_stream_baselines.json)\n"
        "- [`v1_shared_stream_baselines.csv`](v1_shared_stream_baselines.csv)\n",
        encoding="utf-8",
    )
    return out


def run_step_2_ablations(daemon_bin: Path, bridge_bin: Path) -> dict:
    abl_dir = V1_SCI_DIR / "ablations"
    configs = [
        ("1_Static_B0", "static", 1, "Uniform prior weights + Inline OSD-0 (17 core cyc)"),
        ("2_Marginal_LBFGS_B1", "marginal_lbfgs", 1, "4-check independent marginal L-BFGS + Inline OSD-0 (17 core cyc)"),
        ("3_Joint_DP_Pure_9Marginal_BP_17cyc", "joint_dp_map", 0, "16-state Joint-DP MAP mapped naively to 9 single-qubit LLRs without OSD-0 (17 core cyc)"),
        ("4_Joint_DP_Pair_Projected_17cyc", "joint_dp_proj", 1, "16-state Joint-DP MAP + memoryless pair-aware triangle projection + Inline OSD-0 (17 core cyc)"),
        ("5_Full_AVID_R2D2_WB_Inline_OSD_17cyc", "avid_r2d2", 1, "16-state Joint-DP MAP + 64-unit int8 LSTM R2D2-WB + Inline OSD-0 (17 core cyc)"),
        ("6_Full_AVID_R2D2_WB_PostBP_OSD_19cyc", "avid_r2d2", 2, "16-state Joint-DP MAP + 64-unit int8 LSTM R2D2-WB + Post-BP OSD-0 Stage (19 core cyc)"),
    ]
    rows = []
    md_rows = []
    base_ler = None
    for variant_id, c_mode, o_mode, desc in configs:
        m = run_daemon_bridge_pair(daemon_bin, bridge_bin, "correlated_noise", 1536, 256, 2026, c_mode, o_mode)
        if base_ler is None:
            base_ler = m["logical_error_rate"]
        red = ler_reduction_pct(base_ler, m["logical_error_rate"])
        row = {
            "variant_id": variant_id,
            "controller_mode": c_mode,
            "osd_mode": o_mode,
            "bp_core_cycles": m["bp_core_cycles_mean"],
            "axis_total_cycles": m["axis_total_cycles_mean"],
            "raw_z_failures": m["raw_avid_z_failures"],
            "logical_error_rate": m["logical_error_rate"],
            "ci95_low": m["logical_error_rate_ci95_low"],
            "ci95_high": m["logical_error_rate_ci95_high"],
            "ler_reduction_vs_static_pct": red,
            "description": desc,
        }
        rows.append(row)
        md_rows.append(
            f"| `{variant_id}` | `{c_mode}` | `{o_mode}` | `{row['bp_core_cycles']:.0f}` / `{row['axis_total_cycles']:.0f}` | "
            f"`{row['raw_z_failures']}` (**`{row['logical_error_rate']:.5f}`**) | **`{red:+.2f}%`** |"
        )

    with (abl_dir / "v1_6way_runtime_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = {"experiment": "V1_6Way_Runtime_and_OSD_Ablation", "scenario": "correlated_noise", "seed": 2026, "variants": rows}
    (abl_dir / "v1_6way_runtime_ablation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(abl_dir / "v1_6way_runtime_ablation.json", REPO_ROOT / "reports/scientific/ablations/v1_6way_runtime_ablation.json")
    (abl_dir / "README.md").write_text(
        "# V1 6-Way C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)\n\n"
        "| Ablation Variant | Controller Mode | OSD Mode (`0x058`) | Core / AXI Cycles | Z Failures (Decoded LER) | Relative Change vs. `B0` |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(md_rows)
        + "\n\n## Artifacts\n"
        "- [`v1_6way_runtime_ablation.json`](v1_6way_runtime_ablation.json)\n"
        "- [`v1_6way_runtime_ablation.csv`](v1_6way_runtime_ablation.csv)\n",
        encoding="utf-8",
    )
    return out


def run_step_3_multiseed(daemon_bin: Path, bridge_bin: Path) -> dict:
    ms_dir = V1_SCI_DIR / "multi_seed"
    seeds = list(range(2026, 2036))  # 10 independent seeds
    scenarios = ["correlated_noise", "slowly_drifting_noise", "burst_disturbance", "short_burst_60us_vs_10ms"]
    traj_rows = []
    summary_by_scenario: Dict[str, dict] = {}

    for scen in scenarios:
        s_lers: List[float] = []
        a_lers: List[float] = []
        sb_lers: List[float] = []
        ab_lers: List[float] = []
        for sd in seeds:
            m = run_daemon_bridge_pair(daemon_bin, bridge_bin, scen, 1536, 256, sd, "avid_r2d2", 1)
            s_lers.append(m["static_b0_logical_error_rate"])
            a_lers.append(m["logical_error_rate"])
            sb_lers.append(m["burst_window_static_ler"])
            ab_lers.append(m["burst_window_avid_ler"])
            traj_rows.append({
                "scenario": scen,
                "seed": sd,
                "rounds": m["total_rounds"],
                "static_b0_failures": m["raw_static_b0_failures"],
                "avid_z_failures": m["raw_avid_z_failures"],
                "static_b0_ler": m["static_b0_logical_error_rate"],
                "avid_ler": m["logical_error_rate"],
                "whole_run_reduction_pct": m["ler_reduction_vs_static_pct"],
                "burst_static_ler": m["burst_window_static_ler"],
                "burst_avid_ler": m["burst_window_avid_ler"],
                "burst_reduction_pct": m["burst_window_ler_reduction_pct"],
                "orbit_9class_rmse": m["orbit_9class_rmse"],
            })
        summary_by_scenario[scen] = {
            "whole_run_significance": paired_multiseed_tests(s_lers, a_lers),
            "burst_window_significance": paired_multiseed_tests(sb_lers, ab_lers),
        }

    with (ms_dir / "v1_multiseed_10seeds_trajectories.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)

    out = {
        "experiment": "V1_10Seed_Runtime_Statistical_Evaluation",
        "seeds": seeds,
        "rounds_per_seed": 1536,
        "scenarios": summary_by_scenario,
    }
    (ms_dir / "v1_multiseed_10seeds_evaluation.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(ms_dir / "v1_multiseed_10seeds_evaluation.json", REPO_ROOT / "reports/scientific/multi_seed/v1_multiseed_10seeds_evaluation.json")

    md_rows = []
    for scen, d in summary_by_scenario.items():
        w_sig = d["whole_run_significance"]
        b_sig = d["burst_window_significance"]
        md_rows.append(
            f"| `{scen}` | `{w_sig['static_mean_ler']:.5f} ± {w_sig['static_std_ler']:.5f}` | "
            f"**`{w_sig['adaptive_mean_ler']:.5f} ± {w_sig['adaptive_std_ler']:.5f}`** | "
            f"**`-{w_sig['mean_ler_reduction_pct']:.2f}%`** | **`-{b_sig['mean_ler_reduction_pct']:.2f}%`** | "
            f"`{w_sig['paired_t_pvalue']:.2e}` | `{w_sig['wilcoxon_two_sided_pvalue']:.2e}` | `{w_sig['cohens_d']:.2f}` |"
        )
    (ms_dir / "README.md").write_text(
        "# V1 10-Seed Shared-Stream C++ Runtime Statistical Evaluation (`seeds = 2026..2035`)\n\n"
        "| Scenario ($N=10$ Seeds, `1,536` Rounds/Seed) | Static B0 LER ($\\mu \\pm \\sigma$) | **AVID V1 LER ($\\mu \\pm \\sigma$)** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(md_rows)
        + "\n\n## Artifacts\n"
        "- [`v1_multiseed_10seeds_evaluation.json`](v1_multiseed_10seeds_evaluation.json)\n"
        "- [`v1_multiseed_10seeds_trajectories.csv`](v1_multiseed_10seeds_trajectories.csv)\n",
        encoding="utf-8",
    )
    return out


def run_step_4_statistical_analysis(multiseed_out: dict) -> dict:
    stat_dir = V1_SCI_DIR / "statistical_analysis"
    canonical_rep = json.loads((REPO_ROOT / "reports/v1/integration/cloudshell_v1_integration_10scenarios_report.json").read_text(encoding="utf-8"))

    raw_pvals: Dict[str, float] = {}
    rows = []
    for s in canonical_rep["m4_scenarios"]:
        qm = s["qemu_software_metrics"]
        sname = s["scenario"]
        z_res = two_proportion_z_test(
            qm["raw_static_b0_failures"],
            qm["total_rounds"],
            qm["raw_avid_z_failures"],
            qm["total_rounds"],
        )
        if sname != "stationary_noise":
            raw_pvals[sname] = z_res["two_sided_p_value"]
        rows.append({
            "scenario": sname,
            "total_rounds": qm["total_rounds"],
            "static_failures": qm["raw_static_b0_failures"],
            "avid_failures": qm["raw_avid_z_failures"],
            "static_ler": qm["static_b0_logical_error_rate"],
            "avid_ler": qm["logical_error_rate"],
            "wilson_ci95_low": qm["logical_error_rate_ci95_low"],
            "wilson_ci95_high": qm["logical_error_rate_ci95_high"],
            "z_statistic": z_res["z_statistic"],
            "unadjusted_p_value": z_res["two_sided_p_value"],
        })

    adj_pvals = holm_bonferroni_correction(raw_pvals)
    md_rows = []
    for r in rows:
        r["holm_bonferroni_adjusted_p_value"] = adj_pvals.get(r["scenario"], 1.0)
        md_rows.append(
            f"| `{r['scenario']}` | `{r['static_failures']}/1536` (`{r['static_ler']:.5f}`) | "
            f"`{r['avid_failures']}/1536` (**`{r['avid_ler']:.5f}`**) | "
            f"`[{r['wilson_ci95_low']:.5f}, {r['wilson_ci95_high']:.5f}]` | "
            f"`{r['z_statistic']:.2f}` | `{r['unadjusted_p_value']:.2e}` | `{r['holm_bonferroni_adjusted_p_value']:.2e}` |"
        )

    with (stat_dir / "v1_wilson_ci_and_pvalues.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = {
        "experiment": "V1_Statistical_Hypothesis_Testing_and_Holm_Bonferroni_Audit",
        "single_seed_10scenario_tests": rows,
        "ten_seed_paired_tests": multiseed_out["scenarios"],
    }
    (stat_dir / "v1_statistical_hypothesis_tests.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(stat_dir / "v1_statistical_hypothesis_tests.json", REPO_ROOT / "reports/scientific/statistical_analysis/v1_statistical_hypothesis_tests.json")

    (stat_dir / "README.md").write_text(
        "# V1 Statistical Hypothesis Tests, 95% Wilson CIs & Holm–Bonferroni Multiplicity Audit\n\n"
        "| Scenario (`1,536` Rounds) | Static B0 Failures | AVID V1 Failures | 95% Wilson CI | Two-Prop $z$ | Unadjusted $p$ | Holm–Bonferroni Adjusted $p$ |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(md_rows)
        + "\n\n## Artifacts\n"
        "- [`v1_statistical_hypothesis_tests.json`](v1_statistical_hypothesis_tests.json)\n"
        "- [`v1_wilson_ci_and_pvalues.csv`](v1_wilson_ci_and_pvalues.csv)\n",
        encoding="utf-8",
    )
    return out


def run_v1_scientific_suite() -> dict:
    t0 = time.perf_counter()
    daemon_bin, bridge_bin = ensure_binaries_built()
    baselines = run_step_1_baselines(daemon_bin, bridge_bin)
    ablations = run_step_2_ablations(daemon_bin, bridge_bin)
    multiseed = run_step_3_multiseed(daemon_bin, bridge_bin)
    stats_out = run_step_4_statistical_analysis(multiseed)

    master = {
        "suite_name": "AVID_V1_Scientific_Evaluation_Suite",
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "baselines": baselines,
        "ablations_6way": ablations,
        "multiseed_10seeds": multiseed,
        "statistical_analysis": stats_out,
    }
    res_dir = V1_SCI_DIR / "results"
    (res_dir / "v1_scientific_master_summary.json").write_text(json.dumps(master, indent=2) + "\n", encoding="utf-8")

    summary_md = (
        "# AVID V1 Scientific Evaluation Results Summary\n\n"
        "## 1. 10-Seed Paired Statistical Significance (`seeds = 2026..2035`, `1,536` Rounds/Seed)\n\n"
        "| Scenario | Static B0 Mean LER | **AVID V1 Mean LER** | Whole-Run Mean Reduction | Burst-Window Mean Reduction | Paired $t$-Test $p$ | Wilcoxon $p$ | Cohen's $d$ |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(
            f"| `{sc}` | `{d['whole_run_significance']['static_mean_ler']:.5f}` | "
            f"**`{d['whole_run_significance']['adaptive_mean_ler']:.5f}`** | "
            f"**`-{d['whole_run_significance']['mean_ler_reduction_pct']:.2f}%`** | "
            f"**`-{d['burst_window_significance']['mean_ler_reduction_pct']:.2f}%`** | "
            f"`{d['whole_run_significance']['paired_t_pvalue']:.2e}` | "
            f"`{d['whole_run_significance']['wilcoxon_two_sided_pvalue']:.2e}` | "
            f"`{d['whole_run_significance']['cohens_d']:.2f}` |"
            for sc, d in multiseed["scenarios"].items()
        )
        + "\n\n## 2. 6-Way V1 C++ Runtime & Hardware OSD Mode Ablation (`correlated_noise`)\n\n"
        "| Variant | Controller Mode | OSD Mode | Core / AXI Cycles | Decoded LER | Relative Change vs. `B0` |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(
            f"| `{v['variant_id']}` | `{v['controller_mode']}` | `{v['osd_mode']}` | "
            f"`{v['bp_core_cycles']:.0f}` / `{v['axis_total_cycles']:.0f}` | **`{v['logical_error_rate']:.5f}`** | **`{v['ler_reduction_vs_static_pct']:+.2f}%`** |"
            for v in ablations["variants"]
        )
        + "\n"
    )
    (res_dir / "v1_benchmark_comparison_table.md").write_text(summary_md, encoding="utf-8")
    (res_dir / "README.md").write_text(summary_md, encoding="utf-8")
    (V1_SCI_DIR / "README.md").write_text(
        "# AVID V1 Scientific Evaluation Suite (`Benchmark-Eval/V1_Scientific_Evaluation/`)\n\n"
        "- **[`baselines/`](baselines/README.md)**: Shared-stream evaluation of `B0 Static BP`, `B1 Marginal L-BFGS`, `Joint-DP Memoryless Pair-Projection`, and `Full AVID V1 (LSTM-64 R2D2-WB)`.\n"
        "- **[`ablations/`](ablations/README.md)**: 6-Way C++ runtime and hardware OSD mode (`cfg_osd_mode = 0, 1, 2`) ablation study.\n"
        "- **[`multi_seed/`](multi_seed/README.md)**: 10-seed (`2026..2035`) shared-stream evaluation across non-stationary and correlated noise scenarios.\n"
        "- **[`statistical_analysis/`](statistical_analysis/README.md)**: Two-proportion $z$-tests, 95% Wilson CIs, paired $t$-tests, Wilcoxon signed-rank tests, and Holm–Bonferroni multiplicity correction.\n"
        "- **[`results/`](results/README.md)**: Consolidated V1 scientific master summary (`v1_scientific_master_summary.json` & `v1_benchmark_comparison_table.md`).\n\n"
        + summary_md,
        encoding="utf-8",
    )
    print(f"[PASS] V1_Scientific_Evaluation completed in {master['elapsed_sec']}s")
    return master


if __name__ == "__main__":
    run_v1_scientific_suite()
