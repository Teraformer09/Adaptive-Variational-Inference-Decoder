"""Framework Test 8: End-to-End Seven-Stage Pipeline & Causal Runner Compatibility (`test_seven_stage_pipeline.py`)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simulation.reference.syndrome_stream.causal_stream_runner import CausalStreamRunner
from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import AvidInputAdapter
from simulation.stimulus.seven_stage.stage6_stratified_sampler import (
    generate_single_stratified_trajectory,
)


class TestSevenStagePipeline(unittest.TestCase):
    def test_generated_trajectory_roundtrips_packet_stream_and_runs_in_causal_runner(self) -> None:
        traj, prov, _ = generate_single_stratified_trajectory(
            trajectory_id="e2e_pipe_test",
            split_id="test",
            master_seed=2026,
            duration_seconds=60.0,
            assignment_policy="curriculum_burst_enriched",
        )
        adapter = AvidInputAdapter()
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "stream.bin"
            adapter.write_packet_stream(traj, bin_path)
            validation = adapter.validate_stream(bin_path, expected_trajectory=traj)
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["n_rounds"], 600)
            self.assertEqual(validation["seq_gaps"], 0)

        runner = CausalStreamRunner(window_rounds=200, commit_delay_rounds=10, K=3)
        res = runner.run_trajectory(trajectory=traj, controller_mode="avid_r2d2")
        self.assertEqual(res["total_rounds"], 600)
        self.assertGreaterEqual(res["swap_count"], 2)
        self.assertLessEqual(res["ler_adaptive"], res["ler_static"] + 0.02)


if __name__ == "__main__":
    unittest.main()
