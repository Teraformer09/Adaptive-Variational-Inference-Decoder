#!/usr/bin/env python3
"""Stress test verifying mid-flight reset recovery scenario metrics."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_reset_recovery():
    rep = json.loads((REPO_ROOT / "reports/latest_report.json").read_text(encoding="utf-8"))
    reset_scen = [s for s in rep["m4_scenarios"] if s["scenario"] == "reset_recovery"][0]
    assert reset_scen["status"] == "PASS"
    assert reset_scen["bridge_rtl_metrics"]["reset_operations"] >= 1
