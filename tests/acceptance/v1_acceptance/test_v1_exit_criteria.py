#!/usr/bin/env python3
"""Acceptance test verifying V1 dual-emulator exit criteria, live RTL/bridge execution, and Benchmark-Eval V1 suites."""

import hashlib
import json
from pathlib import Path
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_v1_canonical_report_and_live_rtl_bridge() -> None:
    rep = json.loads(
        (REPO_ROOT / "reports/v1/integration/cloudshell_v1_integration_10scenarios_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert rep["overall_status"] == "PASS"
    assert rep["m1_verilator_standalone_rtl"]["reference_equivalence_mismatches"] == 0
    assert len(rep["m4_scenarios"]) == 10

    # Execute standalone RTL testbench and bridge protocol test live (Fixes M12, M13)
    rtl_tb = REPO_ROOT / "emulator/verilator/models/standalone_rtl_tb"
    bridge_srv = REPO_ROOT / "emulator/verilator/models/avid_cosim_bridge"
    bridge_tb = REPO_ROOT / "emulator/verilator/models/test_bridge_transactions"
    assert rtl_tb.exists()
    assert bridge_srv.exists()
    assert bridge_tb.exists()

    proc_rtl = subprocess.run(
        [str(rtl_tb)], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    )
    assert "ref_mismatches=0" in proc_rtl.stdout

    sock_path = "/tmp/avid_cosim_pytest.sock"
    srv_proc = subprocess.Popen(
        [str(bridge_srv), "--socket", sock_path, "--allow-negative-tests", "--max-clients", "2"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(50):
            if Path(sock_path).exists():
                break
            time.sleep(0.02)
        proc_br = subprocess.run(
            [str(bridge_tb), sock_path], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
        )
        assert proc_br.returncode == 0
    finally:
        srv_proc.terminate()
        srv_proc.wait(timeout=5)

    # Verify trained R2D2 checkpoint SHA-256 provenance
    ckpt_bin = REPO_ROOT / "checkpoints/r2d2_v1_trained.bin"
    ckpt_json = REPO_ROOT / "checkpoints/r2d2_v1_trained.json"
    assert ckpt_bin.exists() and ckpt_json.exists()
    meta = json.loads(ckpt_json.read_text(encoding="utf-8"))
    assert hashlib.sha256(ckpt_bin.read_bytes()).hexdigest() == meta["binary_checkpoint"]["sha256"]


def test_v1_benchmark_eval_suites() -> None:
    v1_int = json.loads(
        (REPO_ROOT / "Benchmark-Eval/V1_Integration/v1_integration_master_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(v1_int["milestones"]) == 6
    assert v1_int["milestones"]["M1_RTL_Correctness"]["reference_equivalence_mismatches"] == 0

    v1_sci = json.loads(
        (
            REPO_ROOT / "Benchmark-Eval/V1_Scientific_Evaluation/results/v1_scientific_master_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert len(v1_sci["ablations_6way"]["variants"]) == 6
    assert len(v1_sci["multiseed_10seeds"]["seeds"]) == 10
    for scen, stats in v1_sci["multiseed_10seeds"]["scenarios"].items():
        assert stats["whole_run_significance"]["paired_t_pvalue"] < 1e-3
        assert stats["whole_run_significance"]["wilcoxon_two_sided_pvalue"] < 0.02
