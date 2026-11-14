"""Framework Test 1: 16-Profile Library Validation (`test_pd_profiles.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.seven_stage.stage1_pd_profiles import run_stage1_pd_validation

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPDProfiles(unittest.TestCase):
    def test_all_16_profiles_load_and_validate_moments(self) -> None:
        reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
        self.assertEqual(len(reg.list_profiles()), 16)
        rep = run_stage1_pd_validation(num_samples=10000, seed=101)
        self.assertTrue(rep["passed"])
        self.assertTrue(rep["deterministic_reproduction_verified"])


if __name__ == "__main__":
    unittest.main()
