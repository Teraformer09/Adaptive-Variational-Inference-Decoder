#!/usr/bin/env python3
"""Stress test verifying the 57,344-round long-duration benchmark report."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_long_duration_report():
    rep = json.loads((REPO_ROOT / "reports/v1/performance/cloudshell_full_benchmark_57344_rounds_report.json").read_text(encoding="utf-8"))
    assert rep["overall_status"] == "PASS"
    assert len(rep["m4_scenarios"]) == 7
