"""Stage 1: 16-Profile Probability Distribution Library Validation (`stage1_pd_profiles.py`).

For each of the 16 configured PDs (`PD_01` .. `PD_16`):
  - Generates a large independent sample.
  - Compares empirical mean and variance against theoretical truncated-Gaussian moments.
  - Checks support bounds, quantiles, and tail probabilities.
  - Verifies deterministic reproduction using fixed seeds.
  - Writes a Stage 1 validation report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

from simulation.reference.noise_model.stage7_statistical_validator import (
    Stage7StatisticalValidator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_stage1_pd_validation(
    pd_yaml_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    num_samples: int = 25000,
    seed: int = 101,
) -> Dict[str, Any]:
    yaml_path = (
        Path(pd_yaml_path)
        if pd_yaml_path is not None
        else (REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    )
    registry = PDRegistry(yaml_path)
    validator = Stage7StatisticalValidator()
    val_report = validator.validate_pd_library(
        pd_registry=registry, num_samples=num_samples, seed=seed
    )

    # Verify deterministic seed reproduction across all 16 profiles
    repro_ok = True
    for pid in registry.list_profiles():
        prof = registry.get(pid)
        s1 = prof.sample_values(np.random.default_rng(seed + 99), size=128)
        s2 = prof.sample_values(np.random.default_rng(seed + 99), size=128)
        if not np.array_equal(s1, s2):
            repro_ok = False
            break

    report = {
        "stage": "stage1_pd_profiles",
        "pd_config_hash": registry.config_hash,
        "deterministic_reproduction_verified": repro_ok,
        "validation_summary": val_report,
        "passed": bool(val_report["all_passed"] and repro_ok),
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "stage1_pd_validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    res = run_stage1_pd_validation(
        output_dir=REPO_ROOT
        / "Benchmark-Eval"
        / "V1_Seven_Stage_Noise_Validation"
        / "stage1_pd_validation"
    )
    if not res["passed"]:
        raise SystemExit(1)
    print(f"[PASS] Stage 1 validated {res['validation_summary']['num_profiles_validated']} PDs.")
