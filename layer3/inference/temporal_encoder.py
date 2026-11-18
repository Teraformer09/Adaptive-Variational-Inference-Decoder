"""Temporal Syndrome Sufficient-Statistics Encoder (`temporal_encoder.py`)."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np


class TemporalSyndromeEncoder:
    """Maintains causal exponentially weighted syndrome histograms across epochs."""

    def __init__(self, temporal_beta: float = 0.60) -> None:
        self.temporal_beta = float(np.clip(temporal_beta, 0.0, 0.95))
        self.ema_hist_x: Optional[np.ndarray] = None
        self.ema_hist_z: Optional[np.ndarray] = None
        self.ema_valid_x: float = 0.0
        self.ema_valid_z: float = 0.0

    def reset(self) -> None:
        self.ema_hist_x = None
        self.ema_hist_z = None
        self.ema_valid_x = 0.0
        self.ema_valid_z = 0.0

    def set_temporal_beta(self, beta: float) -> None:
        self.temporal_beta = float(np.clip(beta, 0.0, 0.95))

    def update(
        self,
        hist_x_16: np.ndarray,
        hist_z_16: np.ndarray,
        valid_rounds_x: int,
        valid_rounds_z: int,
        recalibrate: bool = False,
    ) -> Dict[str, object]:
        """Update filtered histograms; if `recalibrate=True`, flush stale history immediately."""
        hx = np.asarray(hist_x_16, dtype=np.float64)
        hz = np.asarray(hist_z_16, dtype=np.float64)
        vx = float(max(0, int(valid_rounds_x)))
        vz = float(max(0, int(valid_rounds_z)))

        if recalibrate or self.ema_hist_x is None or self.ema_hist_z is None or self.temporal_beta <= 0.0:
            self.ema_hist_x = hx.copy()
            self.ema_hist_z = hz.copy()
            self.ema_valid_x = vx
            self.ema_valid_z = vz
        else:
            b = self.temporal_beta
            if vx > 0.0:
                self.ema_hist_x = b * self.ema_hist_x + (1.0 - b) * hx
                self.ema_hist_x /= max(float(np.sum(self.ema_hist_x)), 1e-12)
                self.ema_valid_x = b * self.ema_valid_x + vx
            else:
                self.ema_valid_x *= b

            if vz > 0.0:
                self.ema_hist_z = b * self.ema_hist_z + (1.0 - b) * hz
                self.ema_hist_z /= max(float(np.sum(self.ema_hist_z)), 1e-12)
                self.ema_valid_z = b * self.ema_valid_z + vz
            else:
                self.ema_valid_z *= b

        return {
            "filtered_hist_x": self.ema_hist_x.copy(),
            "filtered_hist_z": self.ema_hist_z.copy(),
            "effective_rounds_x": float(self.ema_valid_x),
            "effective_rounds_z": float(self.ema_valid_z),
        }
