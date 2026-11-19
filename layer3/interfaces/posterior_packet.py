"""Immutable Posterior Packet Contract Between Layer 3 and Layer 2 (`posterior_packet.py`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class PosteriorPacket:
    """Syndrome-conditioned 21-parameter posterior state exported by Layer 3 to Layer 2."""

    epoch_id: int
    posterior_mean_21: np.ndarray
    posterior_var_21: np.ndarray
    hist_x_16: np.ndarray
    hist_z_16: np.ndarray
    valid_rounds_x: int
    valid_rounds_z: int
    estimator_residual_tvd: float
    posterior_innovation_l2: float = 0.0
    burst_score: float = -1.0
    observability_leverage_21: np.ndarray = field(
        default_factory=lambda: np.zeros(21, dtype=np.float64)
    )
    posterior_mean_x_21: Optional[np.ndarray] = None
    posterior_var_x_21: Optional[np.ndarray] = None
    estimator_residual_tvd_z: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.posterior_mean_21.shape != (21,):
            raise ValueError(f"posterior_mean_21 must have shape (21,), got {self.posterior_mean_21.shape}")
        if self.posterior_var_21.shape != (21,):
            raise ValueError(f"posterior_var_21 must have shape (21,), got {self.posterior_var_21.shape}")
        if self.hist_x_16.shape != (16,) or self.hist_z_16.shape != (16,):
            raise ValueError("hist_x_16 and hist_z_16 must each have shape (16,)")
        if not (
            np.all(np.isfinite(self.posterior_mean_21))
            and np.all(np.isfinite(self.posterior_var_21))
        ):
            raise ValueError("Posterior mean and variance must be finite")
        if np.any(self.posterior_var_21 < 0.0):
            raise ValueError("Posterior variances must be non-negative")

    def to_layer2_dict(self) -> Dict[str, Any]:
        """Serialize into the clean syndrome-conditioned dictionary consumed by Layer 2 `ObservationBuilder`."""
        out: Dict[str, Any] = {
            "epoch_id": int(self.epoch_id),
            "posterior_mean_21": self.posterior_mean_21.copy(),
            "posterior_var_21": self.posterior_var_21.copy(),
            "hist_x_16": self.hist_x_16.copy(),
            "hist_z_16": self.hist_z_16.copy(),
            "valid_rounds_x": int(self.valid_rounds_x),
            "valid_rounds_z": int(self.valid_rounds_z),
            "estimator_residual_tvd": float(self.estimator_residual_tvd),
            "posterior_innovation_l2": float(self.posterior_innovation_l2),
            "burst_score": float(self.burst_score),
            "observability_leverage_21": self.observability_leverage_21.copy(),
        }
        if self.posterior_mean_x_21 is not None:
            out["posterior_mean_x_21"] = self.posterior_mean_x_21.copy()
        if self.posterior_var_x_21 is not None:
            out["posterior_var_x_21"] = self.posterior_var_x_21.copy()
        if self.estimator_residual_tvd_z > 0.0 or self.posterior_mean_x_21 is not None:
            out["estimator_residual_tvd_z"] = float(self.estimator_residual_tvd_z)
        return out
