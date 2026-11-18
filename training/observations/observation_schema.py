"""Canonical 82-Dimensional Layer 2 Observation Schema (`observation_schema.py`).

Composition of the 82-D syndrome-conditioned observation per control epoch:
  - [0:21]   Posterior physical state mean: vec(M_hat_stab) in [0.0, 0.49]^21
  - [21:42]  Posterior log-variance uncertainty: vec(log(sigma^2 + eps)) in R^21
  - [42:58]  Empirical 16-state X-check syndrome histogram: f_hat_X[0..15], sum = 1
  - [58:74]  Empirical 16-state Z-check syndrome histogram: f_hat_Z[0..15], sum = 1
  - [74:82]  8 temporal & system telemetry features:
               74: syndrome_density_x
               75: syndrome_density_z
               76: adjacent_coactivation_x
               77: adjacent_coactivation_z
               78: posterior_innovation_l2
               79: burst_score
               80: estimator_residual_tvd
               81: decoder_config_age_norm
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from training.observations.spatial_mapping import CANONICAL_21_PARAM_NAMES

OBSERVATION_SCHEMA_VERSION: int = 2
OBS_V1_DIM: int = 8
OBS_V2_DIM: int = 82
POSTERIOR_VAR_EPS: float = 1.0e-8

SLICE_POSTERIOR_MEAN = slice(0, 21)
SLICE_POSTERIOR_LOG_VAR = slice(21, 42)
SLICE_HIST_X = slice(42, 58)
SLICE_HIST_Z = slice(58, 74)
SLICE_TELEMETRY = slice(74, 82)

TELEMETRY_NAMES_8: Tuple[str, ...] = (
    "syndrome_density_x",
    "syndrome_density_z",
    "adjacent_coactivation_x",
    "adjacent_coactivation_z",
    "posterior_innovation_l2",
    "burst_score",
    "estimator_residual_tvd",
    "decoder_config_age_norm",
)

FEATURE_NAMES_82: Tuple[str, ...] = (
    tuple(f"post_mean_{n}" for n in CANONICAL_21_PARAM_NAMES)
    + tuple(f"post_logvar_{n}" for n in CANONICAL_21_PARAM_NAMES)
    + tuple(f"hist_x_{s:02d}" for s in range(16))
    + tuple(f"hist_z_{s:02d}" for s in range(16))
    + TELEMETRY_NAMES_8
)

assert len(FEATURE_NAMES_82) == OBS_V2_DIM


def variance_to_log_uncertainty(
    variance_21: np.ndarray, eps: float = POSTERIOR_VAR_EPS
) -> np.ndarray:
    """Convert 21 non-negative posterior variances sigma^2 into log(sigma^2 + eps)."""
    var_arr = np.asarray(variance_21, dtype=np.float64)
    if var_arr.shape != (21,):
        raise ValueError(f"Expected posterior variance shape (21,), got {var_arr.shape}")
    if not np.all(np.isfinite(var_arr)) or np.any(var_arr < 0.0):
        raise ValueError("Posterior variances must be finite and non-negative")
    return np.log(var_arr + float(eps))


def validate_observation_v2(
    obs: np.ndarray,
    strict_probabilities: bool = True,
    hist_tol: float = 1.0e-5,
) -> np.ndarray:
    """Validate shape, finiteness, probability intervals, and histogram normalization of an 82-D observation."""
    arr = np.asarray(obs, dtype=np.float64)
    if arr.shape != (OBS_V2_DIM,):
        raise ValueError(f"Expected 82-D observation shape (82,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Observation contains NaN or Inf values")

    if strict_probabilities:
        post_mean = arr[SLICE_POSTERIOR_MEAN]
        if np.any(post_mean < -1e-9) or np.any(post_mean > 0.50 + 1e-9):
            raise ValueError(
                f"Posterior means out of physical bounds [0, 0.50]: min={post_mean.min()}, max={post_mean.max()}"
            )

        post_logvar = arr[SLICE_POSTERIOR_LOG_VAR]
        min_logvar = float(np.log(POSTERIOR_VAR_EPS)) - 1e-6
        if np.any(post_logvar < min_logvar) or np.any(post_logvar > 2.0):
            raise ValueError(
                f"Posterior log-variance out of bounds [{min_logvar:.2f}, 2.0]: min={post_logvar.min()}, max={post_logvar.max()}"
            )

        hist_x = arr[SLICE_HIST_X]
        hist_z = arr[SLICE_HIST_Z]
        if np.any(hist_x < -1e-9) or np.any(hist_z < -1e-9):
            raise ValueError("Syndrome histogram entries must be non-negative")
        if abs(float(np.sum(hist_x)) - 1.0) > hist_tol:
            raise ValueError(f"X syndrome histogram sum {float(np.sum(hist_x)):.6f} != 1.0")
        if abs(float(np.sum(hist_z)) - 1.0) > hist_tol:
            raise ValueError(f"Z syndrome histogram sum {float(np.sum(hist_z)):.6f} != 1.0")

        telem = arr[SLICE_TELEMETRY]
        if np.any(telem < -1e-9):
            raise ValueError("Telemetry features must be non-negative")

    return arr
