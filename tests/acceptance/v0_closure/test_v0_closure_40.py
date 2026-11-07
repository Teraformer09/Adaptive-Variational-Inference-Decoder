#!/usr/bin/env python3
"""Acceptance test verifying all 40/40 V0 closure items."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_v0_closure_report():
    rep = json.loads((REPO_ROOT / "reports/v0/closure/v0_closure_40_40_report.json").read_text(encoding="utf-8"))
    assert rep["v0_closure_score"] == "40/40"
    assert rep["all_40_items_closed"] is True
