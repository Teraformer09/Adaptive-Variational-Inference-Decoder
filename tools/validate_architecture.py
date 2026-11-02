#!/usr/bin/env python3
"""
Master End-to-End Architecture Validator (Policy Items A01–A09 / Gates 0–8).
Checks:
  1. All 6 frozen YAML contracts exist in `architecture/contracts/`
  2. All generated artifacts in `architecture/01_code_model/` through `07_fault_management/` exist
  3. Runs all 6 Python unit test suites in `simulation/tests/`
  4. Compiles and runs the C++ test suite (`test_baseline_suite`) and `avid_daemon --self-test`
"""

from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_FILES = [
    "architecture/contracts/system_timing.yaml",
    "architecture/contracts/syndrome_packet_v1.yaml",
    "architecture/contracts/code_layout_v1.yaml",
    "architecture/contracts/noise_channel_v1.yaml",
    "architecture/contracts/llr_numeric_contract.yaml",
    "architecture/contracts/fault_policy.yaml",
    "architecture/01_code_model/Hx.json",
    "architecture/01_code_model/Hz.json",
    "architecture/01_code_model/coordinates.csv",
    "architecture/01_code_model/logical_operators.json",
    "architecture/01_code_model/tanner_tables.json",
    "architecture/02_noise_model/event_signatures.json",
    "architecture/02_noise_model/observation_model.md",
    "architecture/02_noise_model/identifiability_report.md",
    "architecture/03_ingress/dma_topology.md",
    "architecture/03_ingress/packet_layout.md",
    "architecture/03_ingress/buffer_ownership.md",
    "architecture/04_estimator/likelihood_spec.md",
    "architecture/04_estimator/parameter_schema.json",
    "architecture/04_estimator/estimator_io.md",
    "architecture/05_realtime_control/fast_slow_partition.md",
    "architecture/05_realtime_control/deadline_contract.md",
    "architecture/05_realtime_control/epoch_protocol.md",
    "architecture/06_fpga_decoder/cycle_contract.md",
    "architecture/06_fpga_decoder/stream_interface.md",
    "architecture/06_fpga_decoder/resource_envelope.md",
    "architecture/07_fault_management/fallback_fsm.md",
    "architecture/07_fault_management/fault_matrix.csv",
    "configs/d3_reference.yaml",
    "configs/d5_simulation.yaml",
    "configs/target_board.yaml",
    "firmware/rtl/ingress/axis_ingress_tap.sv",
    "firmware/rtl/decoder/bp_engine.sv",
    "firmware/rtl/config/weight_map_reg.sv",
    "firmware/rtl/top/aa_qec_top.sv",
]


def main() -> int:
    print("=================================================================")
    print("  AVID Baseline Architecture Gate Validator (A01-A09 / Q01-Q21)")
    print("=================================================================")

    missing = [rel for rel in REQUIRED_FILES if not (REPO_ROOT / rel).exists()]
    if missing:
        print(f"[FAIL] Missing required files: {missing}")
        return 1
    print(f"[PASS] All {len(REQUIRED_FILES)} required contracts, specs, configs, and RTL files present.")

    # Run Python verification test suites
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(REPO_ROOT / "simulation" / "tests"),
        pattern="test_*.py",
    )
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("[FAIL] Python verification test suites failed.")
        return 1
    print("[PASS] All 6 Python mathematical & architectural verification suites passed.")

    # Build and run C++ verification test suite
    build_dir = REPO_ROOT / "software" / "build_verify"
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cmake", ".."], cwd=str(build_dir), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["make", "-j2"], cwd=str(build_dir), check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["./test_baseline_suite"], cwd=str(build_dir), check=True)
        subprocess.run(["./avid_daemon", "--self-test"], cwd=str(build_dir), check=True)
    finally:
        subprocess.run(["rm", "-rf", str(build_dir)], check=False)

    print("=================================================================")
    print("  [ALL GATES PASSED] AVID Baseline Architecture Verified 100%")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
