"""B2 Baseline: Check-Marginal Proportional-Integral (PI) Heuristic Controller (A07 / Q13)."""
from __future__ import annotations
import numpy as np
from simulation.reference.code_model.css_code_builder import build_rotated_d3_code


class B2ObservablePIDController:
    """
    Adjusts each qubit's effective error probability directly from the firing rates
    of its incident X-stabilizer checks using a first-order PI filter.
    """

    def __init__(self, kp: float = 0.35, ki: float = 0.05) -> None:
        self.code = build_rotated_d3_code()
        self.H = self.code.Hx
        self.kp = kp
        self.ki = ki
        self.p_est = np.full(9, 0.012, dtype=np.float64)
        self.integral = np.zeros(9, dtype=np.float64)

    def update_from_histogram(self, hist_16: np.ndarray) -> np.ndarray:
        total = max(1.0, float(np.sum(hist_16)))
        probs = hist_16 / total
        emp_q = np.zeros(4, dtype=np.float64)
        for s in range(16):
            for c in range(4):
                if (s >> c) & 1:
                    emp_q[c] += probs[s]

        for v in range(9):
            inc_checks = np.where(self.H[:, v] == 1)[0]
            target_p = np.mean(emp_q[inc_checks]) / 2.5
            err = target_p - self.p_est[v]
            self.integral[v] = np.clip(self.integral[v] + err, -0.05, 0.05)
            self.p_est[v] = np.clip(
                self.p_est[v] + self.kp * err + self.ki * self.integral[v],
                1e-4,
                0.40,
            )

        llr = np.log((1.0 - self.p_est) / self.p_est)
        return np.clip(np.round(llr * 256.0), -32768, 32767).astype(np.int16)
