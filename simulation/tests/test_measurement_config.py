"""Stage 5 Test 1: Measurement Configuration & Baseline Separation (`test_measurement_config.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest

from simulation.stimulus.noise_framework.config_schema import load_yaml_validated

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMeasurementConfig(unittest.TestCase):
    def test_stage5_yaml_has_no_hardcoded_default_regimes(self) -> None:
        cfg_path = REPO_ROOT / "configs" / "noise_framework" / "stage5_measurement_imperfections.yaml"
        cfg = load_yaml_validated(
            cfg_path, required_keys=("schema_version", "stage", "enabled", "generator")
        )
        self.assertNotIn("regimes", cfg)
        gen = cfg["generator"]
        self.assertEqual(gen["generator_id"], "measurement_noise_01")
        self.assertEqual(gen["generator_type"], "measurement_noise")
        self.assertEqual(gen["output_domain"], "measurement_fault_parameters")
        self.assertEqual(gen["profile_assignment"]["source"], "pd_registry")
        self.assertEqual(int(gen["profile_assignment"]["persistence_seconds"]), 100)
        self.assertEqual(int(gen["sampling"]["samples_per_second"]), 10)
        self.assertTrue(bool(gen["sampling"]["synchronized"]))

        expected_mechs = {
            "independent_readout",
            "temporally_correlated_readout",
            "detector_dropout",
            "observation_delay",
        }
        self.assertEqual(set(gen["supported_mechanisms"]), expected_mechs)

    def test_baselines_preserved_in_separate_fixture_file(self) -> None:
        base_path = (
            REPO_ROOT
            / "configs"
            / "noise_framework"
            / "baselines"
            / "stage5_measurement_baselines.yaml"
        )
        base_cfg = load_yaml_validated(base_path, required_keys=("schema_version", "baselines"))
        baselines = base_cfg["baselines"]
        self.assertIn("ideal_noiseless", baselines)
        self.assertIn("moderate_readout_noise", baselines)
        self.assertIn("correlated_readout_and_dropout", baselines)


if __name__ == "__main__":
    unittest.main()
