#!/usr/bin/env python3
"""Generate a concise summary of the canonical latest report."""
from __future__ import annotations
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    rep = json.loads((REPO_ROOT / "reports" / "latest_report.json").read_text(encoding="utf-8"))
    print(f"Run ID: {rep['run_id']} | Status: {rep['overall_status']} | Scenarios: {len(rep.get('m4_scenarios', []))}")
