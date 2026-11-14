"""Stage 3: Spatially Correlated 21-Mechanism Fault Generation Validation (`stage3_spatial_correlated.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.spatial_correlated_generator import (
    SpatialCorrelatedGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_stage3_spatial_validation(
    output_dir: str | Path | None = None,
    num_samples: int = 1000,
    seed: int = 303,
) -> Dict[str, Any]:
    pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    spec = GeneratorSpec(
        generator_id="zz_crosstalk_01",
        generator_type="spatial_correlated",
        enabled=True,
        seed=seed,
        output_domain="probability_parameter",
        output_shape=(21,),
        composition_group="data_faults_z",
        parameters={"basis": "Z", "latent_beta_scale": 1.25},
    )
    gen = SpatialCorrelatedGenerator(spec=spec, seed=seed)
    gen.assign_profile(pd_reg.get("PD_12"), window_id=0)

    samples = np.stack([gen.sample(i * 0.1, i).values for i in range(num_samples)])
    # PD_12 has dominant pairs [1, 4] -> mechanism indices 10 and 13
    corr_matrix = np.corrcoef(samples.T)
    corr_dom_pairs = float(corr_matrix[10, 13])
    mean_dom_pair = float(np.mean(samples[:, [10, 13]]))
    mean_bg_pair = float(np.mean(samples[:, [9, 11, 15, 16]]))

    passed = bool(
        samples.shape == (num_samples, 21)
        and corr_dom_pairs > 0.70
        and mean_dom_pair > mean_bg_pair * 1.5
    )
    report = {
        "stage": "stage3_spatial_correlated",
        "num_samples": num_samples,
        "dominant_pair_correlation_10_13": corr_dom_pairs,
        "mean_dominant_pair_prob": mean_dom_pair,
        "mean_background_pair_prob": mean_bg_pair,
        "passed": passed,
    }
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "stage3_spatial_validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    res = run_stage3_spatial_validation(
        output_dir=REPO_ROOT
        / "Benchmark-Eval"
        / "V1_Seven_Stage_Noise_Validation"
        / "stage3_generator_validation"
    )
    if not res["passed"]:
        raise SystemExit(1)
    print("[PASS] Stage 3 spatial correlation validation passed.")
