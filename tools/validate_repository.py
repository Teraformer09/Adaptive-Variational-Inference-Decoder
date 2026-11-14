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
    "checkpoints/r2d2_v1_trained.bin",
    "checkpoints/r2d2_v1_trained.json",
    "checkpoints/r2d2_v1_trained.onnx",
    "training/models/r2d2_network.py",
    "training/export/export_checkpoint.py",
    "deployment/cpp/model_loader.cpp",
    "deployment/cpp/r2d2_runtime.cpp",
    "deployment/model_schema/r2d2_schema.json",
    "tests/test_training.py",
    "tests/test_checkpoint_export.py",
    "tests/test_cpp_python_parity.py",
    "tests/test_decoder_action_effect.py",
    "tests/test_seven_stage_noise_framework.py",
    "configs/noise_framework/framework_v1.yaml",
    "configs/noise_framework/pd_profiles_16.yaml",
    "configs/noise_framework/generator_registry.yaml",
    "configs/noise_framework/generator_assignment.yaml",
    "configs/noise_framework/sampling_schedule.yaml",
    "configs/noise_framework/composition_rules.yaml",
    "configs/noise_framework/stage5_measurement_imperfections.yaml",
    "configs/noise_framework/measurement_parameter_mapping.yaml",
    "configs/noise_framework/baselines/stage5_measurement_baselines.yaml",
    "configs/noise_framework/stage6_stratified_splits.yaml",
    "simulation/stimulus/noise_framework/__init__.py",
    "simulation/stimulus/noise_framework/syndrome/measurement_parameters.py",
    "simulation/stimulus/noise_framework/syndrome/measurement_model.py",
    "simulation/stimulus/noise_framework/generators/measurement_noise_generator.py",
    "simulation/stimulus/seven_stage/stage5_measurement_noise.py",
    "simulation/reference/noise_model/logical_class_distribution.py",
    "simulation/reference/noise_model/stage7_statistical_validator.py",
    "Benchmark-Eval/V1_Seven_Stage_Noise_Validation/run_seven_stage_eval.py",
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

    # Ensure Stage 5 default measurement YAML does not hard-code fixed regimes
    stage5_yaml = REPO_ROOT / "configs" / "noise_framework" / "stage5_measurement_imperfections.yaml"
    if stage5_yaml.exists() and "\nregimes:" in stage5_yaml.read_text(encoding="utf-8"):
        errors.append("Default stage5_measurement_imperfections.yaml must not contain hard-coded 'regimes:'")

    # Ensure no machine-specific file:///home/ links or accidental LaTeX control characters exist in tracked Markdown files
    for md in REPO_ROOT.rglob("*.md"):
        if ".git" in md.parts:
            continue
        txt = md.read_text(encoding="utf-8", errors="ignore")
        if "file:///home/" in txt:
            errors.append(f"Machine-specific file:///home/ link found in {md.relative_to(REPO_ROOT)}")
        bad_ctrl = sorted({repr(c) for c in txt if ord(c) < 32 and c not in ("\n",)})
        if bad_ctrl:
            errors.append(f"Unescaped control character(s) {bad_ctrl} found in {md.relative_to(REPO_ROOT)}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    print("[PASS] Repository structure, single-source documentation, and link hygiene verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
