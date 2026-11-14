"""Stage 5 Test 9: Multi-Source Measurement Composition (`test_measurement_composition.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.noise_framework.composition.noise_composer import NoiseComposer
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class TestMeasurementComposition(unittest.TestCase):
    def test_independent_bernoulli_measurement_composition_rule(self) -> None:
        mp1 = MeasurementNoiseParameters(
            q_base_x=np.array([0.10, 0.20, 0.05, 0.00], dtype=np.float64),
            q_base_z=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64),
            temporal_sticky_prob=0.10,
            detector_dropout_prob=0.02,
            observation_delay_samples=1,
            generator_id="measurement_noise_01",
            profile_id="PD_02",
            sample_id=3,
            timestamp=0.3,
            window_id=0,
        )
        mp2 = MeasurementNoiseParameters(
            q_base_x=np.array([0.20, 0.10, 0.05, 0.15], dtype=np.float64),
            q_base_z=np.array([0.10, 0.10, 0.10, 0.10], dtype=np.float64),
            temporal_sticky_prob=0.20,
            detector_dropout_prob=0.03,
            observation_delay_samples=2,
            generator_id="measurement_noise_02",
            profile_id="PD_07",
            sample_id=3,
            timestamp=0.3,
            window_id=0,
        )

        composer = NoiseComposer()
        comp = composer.compose([mp1, mp2], timestamp=0.3, sample_id=3)
        comp_mp = comp.composite_parameters["measurement"]

        expected_qx = 1.0 - (1.0 - mp1.q_base_x) * (1.0 - mp2.q_base_x)
        expected_qz = 1.0 - (1.0 - mp1.q_base_z) * (1.0 - mp2.q_base_z)
        expected_sticky = 1.0 - (1.0 - 0.10) * (1.0 - 0.20)
        expected_drop = 1.0 - (1.0 - 0.02) * (1.0 - 0.03)

        self.assertTrue(np.allclose(comp_mp.q_base_x, expected_qx))
        self.assertTrue(np.allclose(comp_mp.q_base_z, expected_qz))
        self.assertAlmostEqual(comp_mp.temporal_sticky_prob, expected_sticky, places=12)
        self.assertAlmostEqual(comp_mp.detector_dropout_prob, expected_drop, places=12)
        self.assertEqual(comp_mp.observation_delay_samples, 2)
        self.assertEqual(len(comp.contributions), 2)


if __name__ == "__main__":
    unittest.main()
