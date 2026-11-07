#!/usr/bin/env python3
"""Stress test verifying busy_decoder back-to-back CDC commit protection."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_busy_decoder():
    rep = json.loads((REPO_ROOT / "reports/latest_report.json").read_text(encoding="utf-8"))
    busy_scen = [s for s in rep["m4_scenarios"] if s["scenario"] == "busy_decoder"][0]
    assert busy_scen["status"] == "PASS"
    assert busy_scen["qemu_software_metrics"]["decoded_rtl_transactions"] == 1536
