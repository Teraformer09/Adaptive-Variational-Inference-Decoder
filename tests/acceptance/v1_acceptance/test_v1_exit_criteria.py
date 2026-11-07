#!/usr/bin/env python3
"""Acceptance test verifying V1 dual-emulator exit criteria on canonical report."""
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
