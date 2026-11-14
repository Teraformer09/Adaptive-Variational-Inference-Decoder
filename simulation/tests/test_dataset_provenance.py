"""Framework Test 7: Dataset Provenance & Reproducibility (`test_dataset_provenance.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.seven_stage.stage6_stratified_sampler import (
    generate_single_stratified_trajectory,
)


class TestDatasetProvenance(unittest.TestCase):
    def test_same_seed_reproduces_identical_trajectory_and_provenance(self) -> None:
        t1, p1, m1 = generate_single_stratified_trajectory(
            trajectory_id="repro_test",
            split_id="train",
            master_seed=777,
            duration_seconds=20.0,
            assignment_policy="uniform_16",
        )
        t2, p2, m2 = generate_single_stratified_trajectory(
            trajectory_id="repro_test",
            split_id="train",
            master_seed=777,
            duration_seconds=20.0,
            assignment_policy="uniform_16",
        )
        self.assertEqual(p1["theta_hash"], p2["theta_hash"])
        self.assertEqual(m1["assignment_counts"], m2["assignment_counts"])
        self.assertTrue(np.array_equal(t1["syndromes_x"], t2["syndromes_x"]))
        self.assertTrue(np.array_equal(t1["syndromes_z"], t2["syndromes_z"]))
        self.assertTrue(np.array_equal(t1["fault_masks_z"], t2["fault_masks_z"]))


if __name__ == "__main__":
    unittest.main()
