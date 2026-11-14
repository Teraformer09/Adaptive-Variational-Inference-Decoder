"""Stage 5 Test 3: 100-Second Window Profile Persistence (`test_measurement_profile_persistence.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMeasurementProfilePersistence(unittest.TestCase):
    def test_persistence_across_250_seconds_three_windows(self) -> None:
        pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
        spec = GeneratorSpec(
            generator_id="measurement_noise_01",
            generator_type="measurement_noise",
            enabled=True,
            seed=1007,
            output_domain="measurement_fault_parameters",
            output_shape=(11,),
            composition_group="measurement_faults",
            parameters={},
        )
        gen = MeasurementNoiseGenerator(spec=spec, seed=1007)
        assigner = PDAssignment(pd_registry=pd_reg, master_seed=99, policy_name="uniform_16")
        scheduler = AssignmentScheduler(pd_assignment=assigner, window_duration_sec=100.0)
        scheduler.initialize([gen], pd_reg)

        clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
        sampler = SynchronizedSampler(generators=[gen], clock=clock, scheduler=scheduler)
        steps = sampler.generate_trajectory(duration_seconds=250.0)

        self.assertEqual(len(steps), 2500)
        win_profiles = {0: set(), 1: set(), 2: set()}
        win_counts = {0: 0, 1: 0, 2: 0}
        for step in steps:
            mp = step[0]
            win_profiles[mp.window_id].add(mp.profile_id)
            win_counts[mp.window_id] += 1

        self.assertEqual(win_counts[0], 1000)
        self.assertEqual(win_counts[1], 1000)
        self.assertEqual(win_counts[2], 500)
        self.assertEqual(len(win_profiles[0]), 1)
        self.assertEqual(len(win_profiles[1]), 1)
        self.assertEqual(len(win_profiles[2]), 1)


if __name__ == "__main__":
    unittest.main()
