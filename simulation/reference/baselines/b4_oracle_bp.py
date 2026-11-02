"""B4 Baseline: Instantaneous Ground-Truth Noise Parameter Oracle BP (A07 / Q13)."""
from __future__ import annotations
import numpy as np
from simulation.reference.noise_model.joint_dp_likelihood import build_d3_correlated_noise_model


class B4OracleBPController:
    """
    Receives the exact instantaneous ground-truth physical parameter vector theta(t)
    with zero estimation latency and converts it to Q7.8 LLRs.
    """

    def __init__(self) -> None:
        self.model = build_d3_correlated_noise_model(basis="X", include_measurement_faults=False)

    def weights_from_true_theta(self, true_theta: np.ndarray) -> np.ndarray:
        _, q78 = self.model.compute_q7_8_llrs(true_theta)
        return q78
