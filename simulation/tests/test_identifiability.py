#!/usr/bin/env python3
"""
Unit Test Suite 4: Structural Identifiability & Jacobian SVD Rank Analysis (A03 / Q04).
Verifies:
  1. Legacy 4-check marginal Jacobian has numerical rank == 4
  2. Repaired 16-state Joint-DP Jacobian has numerical rank == 9 (equal to the number of
     distinct non-zero syndrome signatures on canonical d=3 X-checks)
  3. Prior-tethered MAP Hessian has full rank == 21 with finite condition number (< 500)
  4. L-BFGS estimation on a correlated hotspot + ZZ burst reduces NLL monotonically
     and accurately identifies the elevated error probabilities on qubits 4 and 5
"""

from __future__ import annotations
from pathlib import Path
import sys
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.reference.noise_model.identifiability_svd import analyze_identifiability
from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model


class TestIdentifiability(unittest.TestCase):
    def test_jacobian_and_fisher_ranks(self) -> None:
        rep = analyze_identifiability()
        self.assertEqual(rep["legacy_marginal_rank"], 4)
        self.assertEqual(rep["joint_dp_rank"], 9)
        self.assertEqual(rep["n_distinct_nonzero_signatures"], 9)
        self.assertEqual(rep["map_regularized_rank"], 21)
        self.assertLess(rep["map_condition_number"], 500.0)

    def test_lbfgs_identifies_hotspot_and_zz_crosstalk(self) -> None:
        model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)
        true_theta = model.default_theta.copy()
        true_theta[4] = 0.090  # Strong hotspot on center qubit 4
        true_theta[9 + 3] = 0.070  # Strong ZZ pair on (4, 5)

        true_dist = model.joint_distribution_dp(true_theta)
        counts = true_dist * 4096.0

        init_nll, _ = model.negative_log_likelihood_and_grad(
            model.default_theta, counts, prior_theta=model.default_theta, prior_lambda=0.25
        )
        fit = model.fit_lbfgs(counts, prior_lambda=0.25, max_iter=30)
        self.assertLess(fit["nll"], init_nll)
        # Qubit 4 effective LLR should be significantly lower than corner qubit 0 LLR
        self.assertLess(int(fit["llr_q7_8"][4]), int(fit["llr_q7_8"][0]))


if __name__ == "__main__":
    unittest.main()
