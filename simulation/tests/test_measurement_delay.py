"""Stage 5 Test 8: Observation Delay & Timestamp Preservation (`test_measurement_delay.py`)."""

from __future__ import annotations

import unittest
import numpy as np

from simulation.stimulus.noise_framework.syndrome.measurement_model import MeasurementModel
from simulation.stimulus.noise_framework.syndrome.measurement_parameters import (
    MeasurementNoiseParameters,
)


class TestMeasurementDelay(unittest.TestCase):
    def test_event_and_delivery_timestamps_preserved_under_delay(self) -> None:
        n_steps = 20
        ideal_sx = np.zeros((n_steps, 4), dtype=np.uint8)
        ideal_sz = np.zeros((n_steps, 4), dtype=np.uint8)
        # Impulse syndrome at sample 5
        ideal_sx[5] = np.array([1, 0, 1, 0], dtype=np.uint8)

        delay_samples = 2
        params = [
            MeasurementNoiseParameters(
                q_base_x=np.zeros(4, dtype=np.float64),
                q_base_z=np.zeros(4, dtype=np.float64),
                temporal_sticky_prob=0.0,
                detector_dropout_prob=0.0,
                observation_delay_samples=delay_samples,
                generator_id="measurement_noise_01",
                profile_id="PD_16",
                sample_id=i,
                timestamp=round(i * 0.1, 9),
                window_id=0,
            )
            for i in range(n_steps)
        ]

        model = MeasurementModel(sampling_interval_sec=0.1)
        out = model.apply_trajectory(
            ideal_syndromes_x=ideal_sx,
            ideal_syndromes_z=ideal_sz,
            parameters_seq=params,
            rng=np.random.default_rng(99),
        )
        ev_ts = out["event_timestamps"]
        del_ts = out["delivery_timestamps"]
        self.assertTrue(np.allclose(ev_ts, np.arange(n_steps) * 0.1))
        self.assertTrue(np.allclose(del_ts, np.arange(n_steps) * 0.1 + 0.2))
        # Impulse at event index 5 arrives at observed index 5 + 2 = 7
        self.assertTrue(np.array_equal(out["observed_syndromes_x"][5], np.zeros(4, dtype=np.uint8)))
        self.assertTrue(
            np.array_equal(out["observed_syndromes_x"][7], np.array([1, 0, 1, 0], dtype=np.uint8))
        )


if __name__ == "__main__":
    unittest.main()
