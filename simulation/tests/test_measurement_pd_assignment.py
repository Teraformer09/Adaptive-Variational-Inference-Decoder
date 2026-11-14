"""Stage 5 Test 2: Measurement Generator PD Assignment (`test_measurement_pd_assignment.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.assignment.pd_assignment import PDAssignment
from simulation.stimulus.noise_framework.config_schema import GeneratorSpec, NoiseProfile
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMeasurementPDAssignment(unittest.TestCase):
    def setUp(self) -> None:
        self.pd_reg = PDRegistry(REPO_ROOT / "configs" / "noise_framework" / "pd_profiles_16.yaml")
        self.spec = GeneratorSpec(
            generator_id="measurement_noise_01",
            generator_type="measurement_noise",
            enabled=True,
            seed=1007,
            output_domain="measurement_fault_parameters",
            output_shape=(11,),
            composition_group="measurement_faults",
            parameters={},
        )

    def test_measurement_generator_accepts_all_16_registry_pds(self) -> None:
        gen = MeasurementNoiseGenerator(spec=self.spec, seed=1007)
        assigner = PDAssignment(pd_registry=self.pd_reg, master_seed=42, policy_name="uniform_16")
        for w in range(16):
            pid = assigner.assign("measurement_noise_01", w)
            self.assertIn(pid, self.pd_reg.list_profiles())
            prof = self.pd_reg.get(pid)
            gen.assign_profile(prof, window_id=w)
            out = gen.sample(timestamp=w * 100.0, sample_id=w * 1000)
            self.assertEqual(out.profile_id, pid)
            self.assertEqual(out.generator_id, "measurement_noise_01")

    def test_incompatible_profile_variable_rejected(self) -> None:
        gen = MeasurementNoiseGenerator(spec=self.spec, seed=1007)
        bad_prof = NoiseProfile(
            profile_id="PD_BAD",
            distribution_type="truncated_gaussian",
            parameters={"mean": 0.01, "std": 0.002},
            support={"min": 0.001, "max": 0.05, "variable": "Incompatible_Time_Variable", "units": "s"},
            metadata={},
        )
        with self.assertRaises(ValueError):
            gen.assign_profile(bad_prof, window_id=0)


if __name__ == "__main__":
    unittest.main()
