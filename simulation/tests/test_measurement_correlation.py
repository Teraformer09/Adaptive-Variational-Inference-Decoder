"""Stage 5 Test 6: Measurement Temporal Correlation Transition Model (`test_measurement_correlation.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class TestMeasurementCorrelation(unittest.TestCase):
    def test_temporal_sticky_transition_increases_repeat_error_probability(self) -> None:
        n_steps = 8000
        ideal_sx = np.zeros((n_steps, 4), dtype=np.uint8)
        ideal_sz = np.zeros((n_steps, 4), dtype=np.uint8)

        params = [
            MeasurementNoiseParameters(
                q_base_x=np.full(4, 0.04, dtype=np.float64),
                q_base_z=np.full(4, 0.04, dtype=np.float64),
                temporal_sticky_prob=0.45,
                detector_dropout_prob=0.0,
                observation_delay_samples=0,
                generator_id="measurement_noise_01",
                profile_id="PD_07",
                sample_id=i,
                timestamp=i * 0.1,
                window_id=i // 1000,
            )
            for i in range(n_steps)
        ]

        model = MeasurementModel(sampling_interval_sec=0.1)
        out = model.apply_trajectory(
            ideal_syndromes_x=ideal_sx,
            ideal_syndromes_z=ideal_sz,
            parameters_seq=params,
            rng=np.random.default_rng(77),
        )
        m_x = out["measurement_faults_x"][:, 0]
        prev_one = m_x[:-1] == 1
        prev_zero = m_x[:-1] == 0
        p_1_given_1 = float(np.mean(m_x[1:][prev_one]))
        p_1_given_0 = float(np.mean(m_x[1:][prev_zero]))

        # Theoretical: P(1|0) = 0.04, P(1|1) = (1 - 0.45)*0.04 + 0.45 = 0.472
        self.assertAlmostEqual(p_1_given_0, 0.04, delta=0.015)
        self.assertAlmostEqual(p_1_given_1, 0.472, delta=0.06)
        self.assertGreater(p_1_given_1, p_1_given_0 + 0.30)


if __name__ == "__main__":
    unittest.main()
