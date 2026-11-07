#!/usr/bin/env python3
"""Compare two AVID execution reports across M1 and M4 scenarios."""
from __future__ import annotations
import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: compare_reports.py <report_a.json> <report_b.json>")
        return 0
    a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    print(f"Report A: {a.get('run_id')} ({a.get('m1_verilator_standalone_rtl', {}).get('simulator')})")
    print(f"Report B: {b.get('run_id')} ({b.get('m1_verilator_standalone_rtl', {}).get('simulator')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
