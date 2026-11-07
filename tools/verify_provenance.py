#!/usr/bin/env python3
"""Verify that execution reports carry explicit simulator and run provenance."""
from __future__ import annotations
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rep_path = REPO_ROOT / "reports" / "latest_report.json"
    if not rep_path.exists():
        print("[FAIL] reports/latest_report.json missing", file=sys.stderr)
        return 1
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    sim = rep.get("m1_verilator_standalone_rtl", {}).get("simulator", "")
    native = rep.get("m1_verilator_standalone_rtl", {}).get("native_verilator_compiled", None)
    if sim not in ("verilator_c++_rtl", "cpp_cycle_exact_reference_fallback") or native is None:
        print("[FAIL] Missing explicit simulator provenance in latest_report.json", file=sys.stderr)
        return 1
    print(f"[PASS] Provenance verified: run_id={rep.get('run_id')}, simulator={sim}, native_verilator_compiled={native}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
