"""Stage 4: Non-Stationary Burst & Multi-Generator Composition Validation (`stage4_nonstationary_bursts.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from simulation.reference.noise_model.stage7_statistical_validator import (
    Stage7StatisticalValidator,
)
from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_stage4_composition_validation(
    output_dir: str | Path | None = None,
    duration_seconds: float = 100.0,
    seed: int = 404,
) -> Dict[str, Any]:
    validator = Stage7StatisticalValidator()
    comp_rules_check = validator.validate_composition_rules(seed=seed)

    pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    gen_reg = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=seed
    )
    generators = gen_reg.instantiate_all(master_seed=seed)
    pd_assign = PDAssignment(pd_registry=pd_reg, master_seed=seed, policy_name="uniform_16")
    scheduler = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    scheduler.initialize(generators=generators, pd_registry=pd_reg)

    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
    composer = NoiseComposer()
    sampler = SynchronizedSampler(
        generators=generators, clock=clock, scheduler=scheduler, composer=composer
    )
    buf = TrajectoryBuffer()
    composed_samples = sampler.generate_trajectory(duration_seconds=duration_seconds, buffer=buf)

    # Verify every composite sample retains all active generator contributions
    expected_g = len(generators)
    attribution_preserved = all(
        len(cs.contributions) == expected_g for cs in composed_samples
    )

    passed = bool(
        comp_rules_check["passed"]
        and len(composed_samples) == int(round(duration_seconds * 10.0))
        and attribution_preserved
    )
    report = {
        "stage": "stage4_composition_and_bursts",
        "num_composite_samples": len(composed_samples),
        "num_active_generators": expected_g,
        "individual_attribution_preserved": attribution_preserved,
        "composition_math_validation": comp_rules_check,
        "passed": passed,
    }
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "stage4_composition_validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    res = run_stage4_composition_validation(
        output_dir=REPO_ROOT
        / "Benchmark-Eval"
        / "V1_Seven_Stage_Noise_Validation"
        / "stage4_composition_validation"
    )
    if not res["passed"]:
        raise SystemExit(1)
    print("[PASS] Stage 4 physical noise composition and burst validation passed.")
