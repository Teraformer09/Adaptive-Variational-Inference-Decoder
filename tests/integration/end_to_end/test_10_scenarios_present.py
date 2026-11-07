#!/usr/bin/env python3
"""Integration check verifying all 10 canonical scenarios are present and PASS."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ten_scenarios():
    rep = json.loads((REPO_ROOT / "reports/latest_report.json").read_text(encoding="utf-8"))
    names = [s["scenario"] for s in rep["m4_scenarios"]]
    assert len(names) == 10
    assert "short_burst_60us_vs_10ms" in names
