#!/usr/bin/env python3
"""Integration check verifying epoch and shadow/active commit reconciliation."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_epoch_commit_reconciliation():
    rep = json.loads((REPO_ROOT / "reports/latest_report.json").read_text(encoding="utf-8"))
    for s in rep["m4_scenarios"]:
        m = s["qemu_software_metrics"]
        assert m["epoch_commit_reconciled"] is True
