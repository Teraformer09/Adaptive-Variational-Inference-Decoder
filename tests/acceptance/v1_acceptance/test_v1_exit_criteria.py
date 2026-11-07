#!/usr/bin/env python3
"""Acceptance test verifying V1 dual-emulator exit criteria and Benchmark-Eval V1 suites."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_v1_canonical_report():
    rep = json.loads((REPO_ROOT / "reports/v1/integration/cloudshell_v1_integration_10scenarios_report.json").read_text(encoding="utf-8"))
    assert rep["overall_status"] == "PASS"
    assert rep["m1_verilator_standalone_rtl"]["simulator"] == "verilator_c++_rtl"
    assert rep["m1_verilator_standalone_rtl"]["native_verilator_compiled"] is True
    assert rep["m1_verilator_standalone_rtl"]["reference_equivalence_mismatches"] == 0
    assert len(rep["m4_scenarios"]) == 10


def test_v1_benchmark_eval_suites():
    v1_int = json.loads((REPO_ROOT / "Benchmark-Eval/V1_Integration/v1_integration_master_summary.json").read_text(encoding="utf-8"))
    assert len(v1_int["milestones"]) == 6
    assert v1_int["milestones"]["M1_RTL_Correctness"]["reference_equivalence_mismatches"] == 0

    v1_sci = json.loads((REPO_ROOT / "Benchmark-Eval/V1_Scientific_Evaluation/results/v1_scientific_master_summary.json").read_text(encoding="utf-8"))
    assert len(v1_sci["ablations_6way"]["variants"]) == 6
    assert len(v1_sci["multiseed_10seeds"]["seeds"]) == 10
    for scen, stats in v1_sci["multiseed_10seeds"]["scenarios"].items():
        assert stats["whole_run_significance"]["paired_t_pvalue"] < 1e-4
        assert stats["whole_run_significance"]["wilcoxon_two_sided_pvalue"] < 0.01
