"""Step 9 Tests — Integrate synchronized sampling (`tests/test_synchronized_sampling.py`)."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_generators_synchronized_at_2500_instants() -> None:
    pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
    gen_reg = GeneratorRegistry(
        REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml", master_seed=77
    )
    generators = gen_reg.instantiate_all(master_seed=77)
    assigner = PDAssignment(pd_registry=pd_reg, master_seed=77, policy_name="uniform_16")
    scheduler = AssignmentScheduler(pd_assignment=assigner, window_duration_sec=100.0)
    scheduler.initialize(generators, pd_reg)

    clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
    composer = NoiseComposer()
    sampler = SynchronizedSampler(
        generators=generators, clock=clock, scheduler=scheduler, composer=composer
    )
    buf = TrajectoryBuffer()
    samples = sampler.generate_trajectory(duration_seconds=250.0, buffer=buf)

    assert len(samples) == 2500
    assert len(buf) == 2500
    expected_ids = {g.generator_id for g in generators}
    assert "measurement_noise_01" in expected_ids

    ts_list = []
    windows_seen = set()
    for idx, comp in enumerate(samples):
        assert comp.sample_id == idx
        assert abs(comp.timestamp - round(idx * 0.1, 9)) < 1e-8
        ts_list.append(comp.timestamp)
        windows_seen.add(comp.window_id)
        assert len(comp.contributions) == len(generators)
        contrib_gids = [c.generator_id for c in comp.contributions]
        assert len(contrib_gids) == len(set(contrib_gids))
        assert set(contrib_gids) == expected_ids
        for c in comp.contributions:
            assert c.sample_id == idx
            assert abs(c.timestamp - comp.timestamp) < 1e-8

    assert np.allclose(np.diff(ts_list), 0.1, atol=1e-8)
    assert windows_seen == {0, 1, 2}
