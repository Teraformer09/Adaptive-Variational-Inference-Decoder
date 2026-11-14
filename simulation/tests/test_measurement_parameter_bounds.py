"""Stage 5 Test 5: Measurement Parameter Bounds & Validation (`test_measurement_parameter_bounds.py`)."""

from __future__ import annotations

from pathlib import Path
import unittest
import numpy as np

from simulation.stimulus.noise_framework.config_schema import GeneratorSpec
from simulation.stimulus.noise_framework.generators.measurement_noise_generator import (
    MeasurementNoiseGenerator,
)
from simulation.stimulus.noise_framework.pd_registry import PDRegistry
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMeasurementParameterBounds(unittest.TestCase):
    def test_sampled_parameters_within_valid_bounds(self) -> None:
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
        for pid in pd_reg.list_profiles():
            gen.assign_profile(pd_reg.get(pid), window_id=0)
            for sid in range(25):
                mp = gen.sample(timestamp=sid * 0.1, sample_id=sid)
                self.assertEqual(mp.q_base_x.shape, (4,))
                self.assertEqual(mp.q_base_z.shape, (4,))
                self.assertTrue(np.all((mp.q_base_x >= 0.0) & (mp.q_base_x <= 1.0)))
                self.assertTrue(np.all((mp.q_base_z >= 0.0) & (mp.q_base_z <= 1.0)))
                self.assertTrue(0.0 <= mp.temporal_sticky_prob <= 1.0)
                self.assertTrue(0.0 <= mp.detector_dropout_prob <= 1.0)
                self.assertIsInstance(mp.observation_delay_samples, int)
                self.assertGreaterEqual(mp.observation_delay_samples, 0)

    def test_invalid_parameters_rejected_on_construction(self) -> None:
        with self.assertRaises(ValueError):
            MeasurementNoiseParameters(
                q_base_x=np.array([0.1, 0.2, 1.2, 0.1]),
                q_base_z=np.zeros(4),
                temporal_sticky_prob=0.1,
                detector_dropout_prob=0.01,
                observation_delay_samples=0,
                generator_id="g",
                profile_id="PD_01",
                sample_id=0,
                timestamp=0.0,
                window_id=0,
            )
        with self.assertRaises(ValueError):
            MeasurementNoiseParameters(
                q_base_x=np.zeros(4),
                q_base_z=np.zeros(4),
                temporal_sticky_prob=0.1,
                detector_dropout_prob=0.01,
                observation_delay_samples=-1,
                generator_id="g",
                profile_id="PD_01",
                sample_id=0,
                timestamp=0.0,
                window_id=0,
            )


if __name__ == "__main__":
    unittest.main()
