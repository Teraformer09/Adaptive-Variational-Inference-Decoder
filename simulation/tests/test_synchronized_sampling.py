"""Framework Test 4: Synchronized 10 Hz Multi-Generator Sampling (`test_synchronized_sampling.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.assignment.assignment_scheduler import AssignmentScheduler
from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.generator_registry import GeneratorRegistry
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.sampling.sampling_clock import SamplingClock
from simulation.stimulus.noise_framework.sampling.synchronized_sampler import SynchronizedSampler
from simulation.stimulus.noise_framework.sampling.trajectory_buffer import TrajectoryBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestSynchronizedSampling(unittest.TestCase):
    def test_all_generators_synchronized_at_2500_instants(self) -> None:
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

        self.assertEqual(len(samples), 2500)
        self.assertEqual(len(buf), 2500)
        for idx, comp in enumerate(samples):
            self.assertEqual(comp.sample_id, idx)
            self.assertAlmostEqual(comp.timestamp, round(idx * 0.1, 9), places=8)
            self.assertEqual(len(comp.contributions), len(generators))
            for c in comp.contributions:
                self.assertEqual(c.sample_id, idx)
                self.assertAlmostEqual(c.timestamp, comp.timestamp, places=8)


if __name__ == "__main__":
    unittest.main()
