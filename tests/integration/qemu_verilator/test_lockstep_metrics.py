#!/usr/bin/env python3
"""Integration check verifying zero HAL rejected transactions across all 10 M4 scenarios."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_zero_hal_rejections():
    rep = json.loads((REPO_ROOT / "reports/latest_report.json").read_text(encoding="utf-8"))
    for s in rep["m4_scenarios"]:
        assert s["status"] == "PASS"
        assert s["qemu_software_metrics"]["hal_rejected_transactions"] == 0
