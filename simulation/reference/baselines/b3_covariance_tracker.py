"""B3 Baseline: Empirical Syndrome Covariance Tracker (Spitz /rembrandts-style) (A07 / Q13)."""
from __future__ import annotations
import numpy as np
from simulation.reference.code_model.css_code_builder import build_rotated_d3_code


class B3CovarianceTrackerController:
    """
    Estimates qubit error rates from the diagonal and off-diagonal pairwise check
    covariances <s_i s_j> - <s_i><s_j>.
    """

    def __init__(self) -> None:
        self.code = build_rotated_d3_code()
        self.H = self.code.Hx

    def update_from_histogram(self, hist_16: np.ndarray) -> np.ndarray:
        total = max(1.0, float(np.sum(hist_16)))
        probs = hist_16 / total
        mean_s = np.zeros(4, dtype=np.float64)
        corr_s = np.zeros((4, 4), dtype=np.float64)
        for s in range(16):
            bits = np.array([(s >> c) & 1 for c in range(4)], dtype=np.float64)
            mean_s += probs[s] * bits
            corr_s += probs[s] * np.outer(bits, bits)
        cov = corr_s - np.outer(mean_s, mean_s)

        p_eff = np.full(9, 0.012, dtype=np.float64)
        for v in range(9):
            checks = np.where(self.H[:, v] == 1)[0]
            if len(checks) == 2:
                c1, c2 = checks[0], checks[1]
                denom = max(0.1, 1.0 - 2.0 * (mean_s[c1] + mean_s[c2]) + 4.0 * corr_s[c1, c2])
                val = 0.5 - 0.5 * np.sqrt(max(0.01, 1.0 - (4.0 * max(0.0, cov[c1, c2])) / denom))
                p_eff[v] = np.clip(val, 1e-4, 0.40)
            elif len(checks) == 1:
                c1 = checks[0]
                p_eff[v] = np.clip(0.5 * mean_s[c1], 1e-4, 0.40)

        llr = np.log((1.0 - p_eff) / p_eff)
        return np.clip(np.round(llr * 256.0), -32768, 32767).astype(np.int16)
