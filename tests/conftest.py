"""Pytest configuration ensuring repository root is on sys.path and C++/RTL binaries are built."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_compiled_binaries() -> None:
    """Build Verilator RTL/bridge models, C++ baseline suite, and daemon if missing on a clean checkout."""
    required_bins = [
        REPO_ROOT / "emulator/verilator/models/standalone_rtl_tb",
        REPO_ROOT / "emulator/verilator/models/avid_cosim_bridge",
        REPO_ROOT / "emulator/verilator/models/test_bridge_transactions",
        REPO_ROOT / "emulator/qemu/images/avid_daemon_x86_64",
        REPO_ROOT / "software/build_native/test_baseline_suite",
    ]
    if any(not p.exists() for p in required_bins):
        subprocess.run(
            ["bash", str(REPO_ROOT / "scripts/build_all.sh")],
            cwd=str(REPO_ROOT),
            check=True,
        )
