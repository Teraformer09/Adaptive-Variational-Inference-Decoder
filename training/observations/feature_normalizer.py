"""82-Element Feature Normalizer for R2D2 V2 (`feature_normalizer.py`)."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from training.observations.observation_schema import OBS_V1_DIM, OBS_V2_DIM

# Legacy V1 8-element constants preserved identically for V1 checkpoint compatibility
V1_FEATURE_SCALE_8 = np.array(
    [20.0, 15.0, 20.0, 15.0, 4.0, 6.0, 15.0, 1.0], dtype=np.float64
)
V1_FEATURE_OFFSET_8 = np.array(
    [-0.35, -0.45, -0.30, -0.60, -0.70, -0.35, -0.25, -0.50], dtype=np.float64
)


def build_v2_feature_normalization_82() -> Tuple[np.ndarray, np.ndarray]:
    """Return `(feature_scale[82], feature_offset[82])` mapping raw 82-D observations into [-4, +4]."""
    scale = np.ones(OBS_V2_DIM, dtype=np.float64)
    offset = np.zeros(OBS_V2_DIM, dtype=np.float64)

    # A. [0:21] Posterior physical means (9 single-qubit + 12 ZZ-coupler)
    scale[0:9] = 22.0
    offset[0:9] = -0.55
    scale[9:21] = 18.0
    offset[9:21] = -0.45

    # B. [21:42] Posterior log-variance log(sigma^2 + 1e-8) in [-18.42, -2.0]
    scale[21:42] = 0.35
    offset[21:42] = 3.50

    # C. [42:58] 16-state X syndrome histogram (s=0 trivial state vs s=1..15 excited states)
    scale[42] = 4.0
    offset[42] = -2.80
    scale[43:58] = 16.0
    offset[43:58] = -0.55

    # D. [58:74] 16-state Z syndrome histogram
    scale[58] = 4.0
    offset[58] = -2.80
    scale[59:74] = 16.0
    offset[59:74] = -0.55

    # E. [74:82] 8 temporal & system telemetry features
    scale[74:82] = np.array([4.5, 4.5, 8.0, 8.0, 12.0, 2.5, 10.0, 2.5], dtype=np.float64)
    offset[74:82] = np.array(
        [-0.65, -0.65, -0.40, -0.40, -0.35, -0.60, -0.40, -0.50], dtype=np.float64
    )
    return scale, offset


V2_FEATURE_SCALE_82, V2_FEATURE_OFFSET_82 = build_v2_feature_normalization_82()


def default_feature_norm_for_dim(input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return canonical `(scale, offset)` arrays for `input_dim` (82 for V2, 8 for V1)."""
    d = int(input_dim)
    if d == OBS_V2_DIM:
        return V2_FEATURE_SCALE_82.copy(), V2_FEATURE_OFFSET_82.copy()
    if d == OBS_V1_DIM:
        return V1_FEATURE_SCALE_8.copy(), V1_FEATURE_OFFSET_8.copy()
    return np.ones(d, dtype=np.float64), np.zeros(d, dtype=np.float64)
