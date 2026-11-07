#!/usr/bin/env python3
"""Validate canonical repository structure, documentation policy, and link hygiene."""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "AVID_EPJ_Weekly_Update.md",
    "architecture/README.md",
    "architecture/system/system_overview.md",
    "architecture/code_model/Hx.json",
    "architecture/noise_model/event_signatures.json",
    "architecture/hardware/target_platforms.yaml",
    "architecture/contracts/system_timing.yaml",
    "architecture/verification/requirements_traceability.yaml",
    "mathematical_model/README.md",
    "mathematical_model/proofs/identifiability_proof.md",
    "firmware/rtl/top/aa_qec_top.sv",
    "software/src/inference/avid_daemon.cpp",
    "software/src/control/causal_scheduler.cpp",
    "emulator/verilator/build.sh",
    "emulator/qemu/build.sh",
    "configs/schema.yaml",
    "configs/system/v1_integration.yaml",
    "tests/README.md",
    "Benchmark-Eval/README.md",
    "reports/README.md",
    "runs/README.md",
    "logs/README.md",
    "checkpoints/README.md",
    "datasets/dataset_manifest.json",
    "artifacts/README.md",
    "dependencies/requirements.txt",
]


def main() -> int:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (REPO_ROOT / rel).exists():
            errors.append(f"Missing required path: {rel}")

    # Ensure duplicate docs/AVID_EPJ_Weekly_Update.md does not exist
    if (REPO_ROOT / "docs" / "AVID_EPJ_Weekly_Update.md").exists():
        errors.append("Duplicate weekly update found at docs/AVID_EPJ_Weekly_Update.md")

    # Ensure no machine-specific file:///home/ links exist in tracked Markdown files
    for md in REPO_ROOT.rglob("*.md"):
        if ".git" in md.parts:
            continue
        txt = md.read_text(encoding="utf-8", errors="ignore")
        if "file:///home/" in txt:
            errors.append(f"Machine-specific file:///home/ link found in {md.relative_to(REPO_ROOT)}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print("[PASS] Repository structure, single-source documentation, and link hygiene verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
