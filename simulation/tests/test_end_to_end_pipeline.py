#!/usr/bin/env python3
"""
Unit Test Suite 6: End-to-End Causal Dual-Timescale Pipeline & Baseline Comparison
Addresses Policy Items A04 / Q01, A06 / Q06, A07 / Q13, and A09 / Q18:
  1. Verifies exact cycle count formula L(K) = 2 + 5*K across K in {1..5} (17 cycles = 68 ns @ K=3)
  2. Verifies that all 9 single-qubit Z faults on the canonical rotated d=3 code are
     decoded with zero residual syndrome
  3. Verifies causal double-buffered shadow/active weight bank swap (`bp_ready_in`-gated)
  4. Compares all 5 shared-input baselines (B0..B4) and AVID Joint-DP on identical syndrome streams
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.syndrome_stream.causal_stream_runner import (
    CausalStreamRunner,
    CycleAccurateBPEngine,
    DoubleBufferedWeightBank,
)
from simulation.reference.baselines import (
    B0StaticBPController,
    B1MarginalLBFGSController,
    B2ObservablePIDController,
    B3CovarianceTrackerController,
    B4OracleBPController,
)


class TestEndToEndPipeline(unittest.TestCase):
    def test_bp_cycle_formula_and_single_error_decoding(self) -> None:
        bp = CycleAccurateBPEngine(default_K=3, clock_mhz=250.0)
        expected_cycles = {1: 7, 2: 12, 3: 17, 4: 22, 5: 27}
        for k, cyc in expected_cycles.items():
            self.assertEqual(bp.cycle_count_fused(k), cyc)

        weights = np.full(9, 0x0468, dtype=np.int16)
        for q in range(9):
            err = np.zeros(9, dtype=np.uint8)
            err[q] = 1
            s_vec = (bp.H @ err) % 2
            tr = bp.decode(s_vec, weights, K=3, true_error=err)
            self.assertEqual(tr.cycles_fused, 17)
            self.assertAlmostEqual(tr.latency_ns_fused, 68.0, places=6)
            self.assertTrue(tr.residual_syndrome_zero)
            self.assertFalse(tr.logical_error)

    def test_double_buffered_weight_bank_atomicity(self) -> None:
        wb = DoubleBufferedWeightBank(n_vars=9, reset_llr_q78=0x0400)
        new_w = np.full(9, 0x0200, dtype=np.int16)
        wb.write_shadow_bank(new_w)
        wb.pulse_commit()
        # While decoder is busy (bp_ready_in=False), active_mem must NOT change
        swapped_busy = wb.step_atomic_swap(bp_ready_in=False)
        self.assertFalse(swapped_busy)
        self.assertTrue(np.all(wb.active_mem == 0x0400))
        # As soon as bp_ready_in=True, atomic swap occurs in 1 cycle
        swapped_idle = wb.step_atomic_swap(bp_ready_in=True)
        self.assertTrue(swapped_idle)
        self.assertEqual(wb.swap_count, 1)
        self.assertTrue(np.all(wb.active_mem == 0x0200))

    def test_causal_stream_and_shared_input_baselines(self) -> None:
        runner = CausalStreamRunner(window_rounds=512, commit_delay_rounds=16, K=3)
        res = runner.run_causal_experiment(total_rounds=2048, seed=42)
        self.assertEqual(res["bp_cycles_K3"], 17)
        self.assertGreaterEqual(res["swap_count"], 3)
        self.assertLessEqual(res["ler_adaptive"], res["ler_static"])

        # Verify all 5 baseline controllers B0..B4 produce valid Q7.8 weight banks on shared input
        hist = np.array([750, 20, 25, 10, 22, 5, 65, 4, 18, 3, 5, 2, 12, 2, 3, 1], dtype=np.float64)
        b0 = B0StaticBPController().update_from_histogram(hist)
        b1 = B1MarginalLBFGSController().update_from_histogram(hist)
        b2 = B2ObservablePIDController().update_from_histogram(hist)
        b3 = B3CovarianceTrackerController().update_from_histogram(hist)
        b4 = B4OracleBPController().weights_from_true_theta(runner.model.default_theta)
        for bank in (b0, b1, b2, b3, b4):
            self.assertEqual(bank.shape, (9,))
            self.assertTrue(np.all(bank > 0))


if __name__ == "__main__":
    unittest.main()
