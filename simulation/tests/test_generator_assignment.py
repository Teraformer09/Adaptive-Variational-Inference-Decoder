"""Framework Test 2: Generator Assignment & RNG Isolation (`test_generator_assignment.py`)."""

from __future__ import annotations

import unittest

from simulation.stimulus.seven_stage.stage2_stochastic_temporal import (
    run_stage2_assignment_and_lifecycle_validation,
)


class TestGeneratorAssignment(unittest.TestCase):
    def test_independent_assignment_and_rng_isolation(self) -> None:
        rep = run_stage2_assignment_and_lifecycle_validation(duration_seconds=250.0, seed=202)
        self.assertTrue(rep["passed"])
        self.assertTrue(rep["window_persistence_verified"])
        self.assertTrue(rep["rng_stream_isolation_verified"])
        self.assertEqual(rep["actual_samples"], 2500)


if __name__ == "__main__":
    unittest.main()
