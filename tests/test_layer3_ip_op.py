"""Layer 3 IP/OP Validation Suite: Embedded Slow-Loop Software (Joint-DP, L-BFGS, FSM, Scheduler, DMA)."""
from __future__ import annotations

from pathlib import Path
import subprocess


def test_layer3_cpp_ip_op_suite() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "software" / "build_layer3"
    assert (build_dir / "test_baseline_suite").exists()
    assert (build_dir / "avid_daemon").exists()

    res1 = subprocess.run([str(build_dir / "test_baseline_suite")], cwd=str(repo_root), capture_output=True, text=True, check=True)
    assert "[PASS]" in res1.stdout

    res2 = subprocess.run([str(build_dir / "avid_daemon"), "--self-test"], cwd=str(repo_root), capture_output=True, text=True, check=True)
    assert res2.returncode == 0
