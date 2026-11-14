"""Stage 5 Test 7: Detector Dropout Validity Masks (`test_measurement_dropout.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class TestMeasurementDropout(unittest.TestCase):
    def test_dropped_observations_carry_explicit_zero_validity_mask(self) -> None:
        n_steps = 2000
        ideal_sx = np.ones((n_steps, 4), dtype=np.uint8)
        ideal_sz = np.ones((n_steps, 4), dtype=np.uint8)
        params = [
            MeasurementNoiseParameters(
                q_base_x=np.zeros(4, dtype=np.float64),
                q_base_z=np.zeros(4, dtype=np.float64),
                temporal_sticky_prob=0.0,
                detector_dropout_prob=0.20,
                observation_delay_samples=0,
                generator_id="measurement_noise_01",
                profile_id="PD_12",
                sample_id=i,
                timestamp=i * 0.1,
                window_id=0,
            )
            for i in range(n_steps)
        ]

        model = MeasurementModel(sampling_interval_sec=0.1)
        out = model.apply_trajectory(
            ideal_syndromes_x=ideal_sx,
            ideal_syndromes_z=ideal_sz,
            parameters_seq=params,
            rng=np.random.default_rng(88),
        )
        val_x = out["validity_mask_x"]
        obs_x = out["observed_syndromes_x"]
        drop_rate = float(np.mean(val_x == 0))
        self.assertAlmostEqual(drop_rate, 0.20, delta=0.025)
        # Wherever validity_mask_x == 0, the observation is flagged as invalid and masked to 0
        self.assertTrue(np.all(obs_x[val_x == 0] == 0))
        # Wherever validity_mask_x == 1 (and q_base_x == 0), ideal_sx == 1 is preserved
        self.assertTrue(np.all(obs_x[val_x == 1] == 1))


if __name__ == "__main__":
    unittest.main()
