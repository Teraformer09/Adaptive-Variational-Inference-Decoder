#!/usr/bin/env python3
"""Verify that execution reports and trained R2D2 checkpoints carry explicit cryptographic provenance."""
from __future__ import annotations

import hashlib
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

    # Verify trained R2D2 checkpoint cryptographic SHA-256 provenance
    ckpt_json = REPO_ROOT / "checkpoints" / "r2d2_v1_trained.json"
    ckpt_bin = REPO_ROOT / "checkpoints" / "r2d2_v1_trained.bin"
    ckpt_onnx = REPO_ROOT / "checkpoints" / "r2d2_v1_trained.onnx"
    if not (ckpt_json.exists() and ckpt_bin.exists() and ckpt_onnx.exists()):
        print("[FAIL] Missing trained R2D2 checkpoint artifacts in checkpoints/", file=sys.stderr)
        return 1

    meta = json.loads(ckpt_json.read_text(encoding="utf-8"))
    bin_sha = hashlib.sha256(ckpt_bin.read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256(ckpt_onnx.read_bytes()).hexdigest()
    if bin_sha != meta["binary_checkpoint"]["sha256"]:
        print("[FAIL] SHA-256 mismatch for checkpoints/r2d2_v1_trained.bin", file=sys.stderr)
        return 1
    if onnx_sha != meta["onnx_checkpoint"]["sha256"]:
        print("[FAIL] SHA-256 mismatch for checkpoints/r2d2_v1_trained.onnx", file=sys.stderr)
        return 1

    superseded = REPO_ROOT / "reports" / "archive" / "pre_remediation_b001b8d" / "SUPERSEDED_PROVENANCE.md"
    if not superseded.exists():
        print("[FAIL] Missing pre-remediation provenance archive notice", file=sys.stderr)
        return 1

    print(
        f"[PASS] Provenance verified: run_id={rep.get('run_id')}, simulator={sim}, "
        f"r2d2_bin_sha256={bin_sha[:16]}..."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
