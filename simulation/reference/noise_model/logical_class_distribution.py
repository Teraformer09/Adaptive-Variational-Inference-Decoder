"""Exact 32-State Joint Syndrome-Logical Distribution & Bayes-Optimal Oracle (`logical_class_distribution.py`).

For the rotated d=3 CSS surface code (21 fault mechanisms per basis: 9 single-qubit + 12 pair),
each fault mechanism e in {0..20} has:
  - a 4-bit syndrome signature g_e in {0..15}
  - a 1-bit logical operator flip l_e in {0, 1}
Together these form a 5-bit extended state (s, l) in {0..15} x {0, 1} (32 states).

Using exact 32-state XOR-convolution dynamic programming:
  P_new(s, l) = (1 - p_e) * P_old(s, l) + p_e * P_old(s XOR g_e, l XOR l_e)
we obtain:
  1. Exact joint distribution P(s, l | theta) for s in {0..15}, l in {0, 1}
  2. Exact marginal syndrome distribution P(s | theta) = P(s, 0 | theta) + P(s, 1 | theta)
  3. Exact conditional logical posterior:
       P(l | s, theta) = P(s, l | theta) / (P(s, 0 | theta) + P(s, 1 | theta))
  4. Exact Bayes-optimal Oracle Logical Error Rate floor:
       LER_oracle(theta) = sum_{s=0}^{15} min_{l in {0, 1}} P(s, l | theta)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from simulation.reference.code_model.css_code_builder import build_rotated_d3_code
from simulation.stimulus.noise_framework.config_schema import CANONICAL_12_PAIRS


class LogicalClassDistributionEngine:
    """Computes exact P(s, l | theta), P(l | s, theta), and LER_oracle(theta) for d=3 CSS."""

    def __init__(self, basis: str = "X") -> None:
        self.code = build_rotated_d3_code()
        self.basis = basis.upper()
        self.H = np.asarray(
            self.code.hx if self.basis == "X" else self.code.hz, dtype=np.uint8
        )
        self.logical_vec = np.asarray(
            self.code.logical_X[0] if self.basis == "X" else self.code.logical_Z[0],
            dtype=np.uint8,
        )

        # Precompute 21 fault support vectors in {0,1}^9, 4-bit syndromes g_e, and 1-bit logical flips l_e
        self.mechanism_supports = np.zeros((21, 9), dtype=np.uint8)
        for q in range(9):
            self.mechanism_supports[q, q] = 1
        for p_idx, (q_i, q_j) in enumerate(CANONICAL_12_PAIRS):
            self.mechanism_supports[9 + p_idx, q_i] = 1
            self.mechanism_supports[9 + p_idx, q_j] = 1

        self.syndrome_signatures = np.zeros(21, dtype=np.int32)
        self.logical_flips = np.zeros(21, dtype=np.int32)
        for m in range(21):
            supp = self.mechanism_supports[m]
            s_bits = (self.H @ supp) & 1
            s_int = int(sum(int(s_bits[c]) << c for c in range(4)))
            l_bit = int(np.dot(self.logical_vec, supp) & 1)
            self.syndrome_signatures[m] = s_int
            self.logical_flips[m] = l_bit

        # Precompute an exact syndrome-aliased mechanism pair (m1, m2) with identical
        # non-zero 4-bit syndrome signature g_m1 == g_m2 != 0 and opposite logical flip l_m1 != l_m2
        self.aliased_pair: Tuple[int, int] = (3, 12)
        for m1 in range(21):
            for m2 in range(m1 + 1, 21):
                if (
                    self.syndrome_signatures[m1] != 0
                    and self.syndrome_signatures[m1] == self.syndrome_signatures[m2]
                    and self.logical_flips[m1] != self.logical_flips[m2]
                ):
                    self.aliased_pair = (m1, m2)
                    break

    def compute_joint_distribution(self, theta_21: np.ndarray) -> np.ndarray:
        """Return exact `(16, 2)` array `P[s, l] = P(s, l | theta_21)`."""
        theta = np.asarray(theta_21, dtype=np.float64)
        if theta.shape != (21,):
            raise ValueError(f"Expected 21-parameter vector, got {theta.shape}")
        if np.any((theta < 0.0) | (theta > 1.0)) or not np.all(np.isfinite(theta)):
            raise ValueError("theta_21 must contain finite probabilities in [0, 1]")

        dp = np.zeros((16, 2), dtype=np.float64)
        dp[0, 0] = 1.0

        s_idx = np.arange(16, dtype=np.int32)[:, None]
        l_idx = np.arange(2, dtype=np.int32)[None, :]

        for m in range(21):
            p = float(theta[m])
            if p <= 0.0:
                continue
            g = int(self.syndrome_signatures[m])
            lf = int(self.logical_flips[m])
            dp = (1.0 - p) * dp + p * dp[s_idx ^ g, l_idx ^ lf]

        total = float(np.sum(dp))
        if total > 0.0:
            dp /= total
        return dp

    def compute_posterior_and_oracle_ler(self, theta_21: np.ndarray) -> Dict[str, Any]:
        """Compute `P(s, l | theta)`, `P(s | theta)`, `P(l | s, theta)`, and `LER_oracle(theta)`."""
        joint = self.compute_joint_distribution(theta_21)
        p_syndrome = np.sum(joint, axis=1)
        denom = np.maximum(p_syndrome[:, None], 1e-15)
        posterior_l_given_s = joint / denom
        optimal_l_for_s = np.argmax(joint, axis=1).astype(np.uint8)
        ler_oracle = float(np.sum(np.min(joint, axis=1)))

        return {
            "joint_s_l": joint,
            "marginal_s": p_syndrome,
            "posterior_l_given_s": posterior_l_given_s,
            "optimal_logical_decision": optimal_l_for_s,
            "ler_oracle": ler_oracle,
        }

    def construct_syndrome_aliased_pair(
        self, base_theta: np.ndarray, delta: float = 0.018
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Construct two distinct parameter vectors theta_A != theta_B along an exact
        syndrome-collision direction (g_m1 == g_m2, l_m1 != l_m2) that yield identical
        observable syndrome distributions P(s | theta_A) == P(s | theta_B) while
        differing in conditional logical-class posterior P(l | s).
        """
        theta_a = np.asarray(base_theta, dtype=np.float64).copy()
        theta_b = np.asarray(base_theta, dtype=np.float64).copy()

        m1, m2 = self.aliased_pair
        common_base = float(min(theta_a[m1], theta_a[m2]))
        theta_a[m1] = min(0.25, common_base + delta)
        theta_a[m2] = common_base
        theta_b[m1] = common_base
        theta_b[m2] = min(0.25, common_base + delta)

        res_a = self.compute_posterior_and_oracle_ler(theta_a)
        res_b = self.compute_posterior_and_oracle_ler(theta_b)

        tvd_syndrome = float(0.5 * np.sum(np.abs(res_a["marginal_s"] - res_b["marginal_s"])))
        max_posterior_diff = float(
            np.max(np.abs(res_a["posterior_l_given_s"][:, 1] - res_b["posterior_l_given_s"][:, 1]))
        )
        return theta_a, theta_b, {
            "aliased_mechanisms": [int(m1), int(m2)],
            "shared_syndrome_signature": int(self.syndrome_signatures[m1]),
            "tvd_marginal_syndrome": tvd_syndrome,
            "max_logical_posterior_diff": max_posterior_diff,
            "ler_oracle_a": res_a["ler_oracle"],
            "ler_oracle_b": res_b["ler_oracle"],
        }
