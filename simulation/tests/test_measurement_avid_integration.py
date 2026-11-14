"""Stage 5 Test 10: AVID Input Adapter & Packet Stream Integration (`test_measurement_avid_integration.py`)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simulation.stimulus.seven_stage.stage5_measurement_noise import (
    run_stage5_measurement_noise_pipeline,
)


class TestMeasurementAvidIntegration(unittest.TestCase):
    def test_stage5_pipeline_produces_valid_avid_packet_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = run_stage5_measurement_noise_pipeline(
                duration_seconds=25.0,
                seed=55,
                output_dir=Path(tmp),
            )
            self.assertTrue(rep["passed"])
            self.assertEqual(rep["actual_samples"], 250)
            self.assertTrue(rep["validation_checks"]["avid_packet_stream_compatible"])
            self.assertTrue(rep["validation_checks"]["no_default_hardcoded_regimes"])


if __name__ == "__main__":
    unittest.main()
