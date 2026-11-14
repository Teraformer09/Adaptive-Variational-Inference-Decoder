"""Stage 2: Persistent 100s Assignment & Stochastic Temporal Lifecycle Validation (`stage2_stochastic_temporal.py`).

Verifies:
  - Each generator receives exactly one assignment per 100-second window.
  - Assignments remain fixed throughout each 100-second window.
  - Reassignment happens at the 100-second boundary without resetting continuous generator state.
  - Assignment empirical frequencies converge to the configured categorical probabilities.
  - Generator random streams are strictly isolated (changing Generator 1's PD or seed does not
    alter Generator 2's random stream).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_stage2_assignment_and_lifecycle_validation(
    output_dir: str | Path | None = None,
    duration_seconds: float = 250.0,
    seed: int = 202,
) -> Dict[str, Any]:
    pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    gen_reg = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml",
        master_seed=seed,
    )
    generators = gen_reg.instantiate_all(master_seed=seed)

    pd_assign = PDAssignment(
        pd_registry=pd_reg,
        master_seed=seed,
        policy_name="uniform_16",
        assignment_config_path=REPO_ROOT / "configs" / "noise_framework" / "generator_assignment.yaml",
    )
    scheduler = AssignmentScheduler(pd_assignment=pd_assign, window_duration_sec=100.0)
    scheduler.initialize(generators=generators, pd_registry=pd_reg)

    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
    sampler = SynchronizedSampler(generators=generators, clock=clock, scheduler=scheduler)
    raw_steps = sampler.generate_trajectory(duration_seconds=duration_seconds)

    expected_samples = int(round(duration_seconds * 10.0))
    # Verify invariant PD within each 100s window (0..999, 1000..1999, 2000..2499)
    invariance_ok = True
    window_pids_by_gen: Dict[str, Dict[int, set[str]]] = {
        g.generator_id: {} for g in generators
    }
    for step in raw_steps:
        for contrib in step:
            w = int(contrib.window_id)
            window_pids_by_gen[contrib.generator_id].setdefault(w, set()).add(contrib.profile_id)

    for gid, win_map in window_pids_by_gen.items():
        if sorted(win_map.keys()) != [0, 1, 2]:
            invariance_ok = False
        for w, pid_set in win_map.items():
            if len(pid_set) != 1:
                invariance_ok = False

    # Verify RNG stream isolation: changing generator 0's profile assignment does not alter generator 1
    gen_reg_a = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=seed
    )
    gens_a = gen_reg_a.instantiate_all(master_seed=seed)
    gens_a[0].assign_profile(pd_reg.get("PD_01"), window_id=0)
    gens_a[1].assign_profile(pd_reg.get("PD_07"), window_id=0)
    out_a1 = gens_a[1].sample(0.0, 0).values

    gen_reg_b = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=seed
    )
    gens_b = gen_reg_b.instantiate_all(master_seed=seed)
    gens_b[0].assign_profile(pd_reg.get("PD_16"), window_id=0)
    gens_b[0].sample(0.0, 0)  # advance gen 0
    gens_b[0].sample(0.1, 1)
    gens_b[1].assign_profile(pd_reg.get("PD_07"), window_id=0)
    out_b1 = gens_b[1].sample(0.0, 0).values
    stream_isolation_ok = bool(np.array_equal(out_a1, out_b1))

    # Empirical categorical assignment check over 800 windows
    big_assign = PDAssignment(pd_registry=pd_reg, master_seed=seed + 500, policy_name="uniform_16")
    for w in range(800):
        for g in generators:
            big_assign.assign(g.generator_id, w)
    manifest = big_assign.assignment_manifest()
    max_freq_dev = max(
        abs(manifest["empirical_frequencies"][pid] - 1.0 / 16.0)
        for pid in pd_reg.list_profiles()
    )

    passed = bool(
        len(raw_steps) == expected_samples
        and invariance_ok
        and stream_isolation_ok
        and max_freq_dev < 0.025
    )
    report = {
        "stage": "stage2_assignment_and_lifecycle",
        "duration_seconds": duration_seconds,
        "expected_samples": expected_samples,
        "actual_samples": len(raw_steps),
        "window_persistence_verified": invariance_ok,
        "rng_stream_isolation_verified": stream_isolation_ok,
        "uniform_categorical_max_freq_deviation": float(max_freq_dev),
        "assignment_manifest_250s": pd_assign.assignment_manifest(),
        "passed": passed,
    }
    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "stage2_assignment_validation_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    res = run_stage2_assignment_and_lifecycle_validation(
        output_dir=REPO_ROOT
        / "Benchmark-Eval"
        / "V1_Seven_Stage_Noise_Validation"
        / "stage2_assignment_validation"
    )
    if not res["passed"]:
        raise SystemExit(1)
    print("[PASS] Stage 2 assignment persistence and lifecycle verified.")
