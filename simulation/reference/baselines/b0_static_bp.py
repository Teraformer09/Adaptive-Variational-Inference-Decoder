"""B0 Baseline: Static Uniform-Prior Normalized Min-Sum BP (A07 / Q13)."""
from __future__ import annotations
import numpy as np


class B0StaticBPController:
    """Keeps Q7.8 LLR weights fixed at the initial calibrated t=0 prior."""

    def __init__(self, n_vars: int = 9, base_p: float = 0.012) -> None:
        self.n_vars = n_vars
        llr = np.log((1.0 - base_p) / base_p)
        q78 = int(np.clip(round(llr * 256.0), -32768, 32767))
        self.weights_q78 = np.full(n_vars, q78, dtype=np.int16)

    def update_from_histogram(self, hist_16: np.ndarray) -> np.ndarray:
        return self.weights_q78.copy()
