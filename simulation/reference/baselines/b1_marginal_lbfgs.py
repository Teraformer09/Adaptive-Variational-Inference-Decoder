"""B1 Baseline: Legacy 4-Check Marginal Bernoulli L-BFGS Estimator (A07 / Q13)."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model


class B1MarginalLBFGSController:
    """
    Fits the 21 parameters against only the 4 marginal check firing rates q_0..q_3
    (representing the rank-4 pre-repair baseline).
    """

    def __init__(self) -> None:
        self.model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)

    def update_from_histogram(self, hist_16: np.ndarray) -> np.ndarray:
        total = max(1.0, float(np.sum(hist_16)))
        probs = hist_16 / total
        emp_q = np.zeros(4, dtype=np.float64)
        for s in range(16):
            for c in range(4):
                if (s >> c) & 1:
                    emp_q[c] += probs[s]

        def obj(theta: np.ndarray) -> float:
            p_eff = self.model.compute_effective_qubit_probs(theta)
            loss = 0.0
            for c in range(4):
                prod = 1.0
                for v in range(9):
                    if self.model.H[c, v] == 1:
                        prod *= 1.0 - 2.0 * p_eff[v]
                qc = np.clip(0.5 * (1.0 - prod), 1e-6, 1.0 - 1e-6)
                loss -= emp_q[c] * np.log(qc) + (1.0 - emp_q[c]) * np.log(1.0 - qc)
            diff = theta - self.model.default_theta
            return float(loss + 0.5 * np.dot(diff, diff))

        res = minimize(
            obj,
            x0=self.model.default_theta.copy(),
            method="L-BFGS-B",
            bounds=[(1e-5, 0.45)] * self.model.n_params,
            options={"maxiter": 10},
        )
        _, q78 = self.model.compute_q7_8_llrs(res.x, pair_aware_projection=False)
        return q78
