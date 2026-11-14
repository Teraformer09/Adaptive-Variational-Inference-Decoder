"""Stage 5 Test 4: 10 Hz Synchronized Measurement Sampling (`test_measurement_sampling.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMeasurementSampling(unittest.TestCase):
    def test_2500_samples_in_250_seconds_aligned_with_all_generators(self) -> None:
        pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
        gen_reg = GeneratorRegistry(
            REPO_ROOT / "configs" / "noise_framework" / "generator_registry.yaml",
            master_seed=123,
        )
        generators = gen_reg.instantiate_all(master_seed=123)
        assigner = PDAssignment(pd_registry=pd_reg, master_seed=123, policy_name="uniform_16")
        scheduler = AssignmentScheduler(pd_assignment=assigner, window_duration_sec=100.0)
        scheduler.initialize(generators, pd_reg)

        clock = SamplingClock(t0=0.0, rate_hz=10.0, window_duration_sec=100.0)
        sampler = SynchronizedSampler(generators=generators, clock=clock, scheduler=scheduler)
        steps = sampler.generate_trajectory(duration_seconds=250.0)

        self.assertEqual(len(steps), 2500)
        meas_gen = gen_reg.get("measurement_noise_01")
        self.assertEqual(meas_gen.samples_generated, 2500)

        for idx, step in enumerate(steps):
            expected_ts = round(idx * 0.1, 9)
            for out in step:
                self.assertEqual(out.sample_id, idx)
                self.assertAlmostEqual(out.timestamp, expected_ts, places=8)


if __name__ == "__main__":
    unittest.main()
