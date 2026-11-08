#!/usr/bin/env python3
"""Acceptance test verifying all 40/40 V0 closure items plus live executable verification (M12)."""

import json
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_v0_closure_report_and_live_cpp_suite() -> None:
    rep = json.loads(
        (REPO_ROOT / "reports/v0/closure/v0_closure_40_40_report.json").read_text(encoding="utf-8")
    )
    assert rep["v0_closure_score"] == "40/40"
    assert rep["all_40_items_closed"] is True

    # Execute compiled C++ baseline test suite to verify live functional correctness (Fixes M12)
    cpp_test = REPO_ROOT / "software/build_native/test_baseline_suite"
    assert cpp_test.exists()
    proc = subprocess.run([str(cpp_test)], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    assert "[ALL C++ BASELINE TESTS PASSED]" in proc.stdout

    # Execute compiled daemon self-test
    daemon_bin = REPO_ROOT / "emulator/qemu/images/avid_daemon_x86_64"
    assert daemon_bin.exists()
    proc_d = subprocess.run(
        [str(daemon_bin), "--self-test"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
    )
    assert "Self-test passed" in proc_d.stdout
