"""82-Dimensional Syndrome-Conditioned Observation Builder (`observation_builder.py`).

Converts Layer 3's Bayesian posterior state `(posterior_mean_21, posterior_var_21, estimator_residual)`,
observed X/Z syndrome histograms, and decoder telemetry into the canonical 82-element R2D2 V2 input.

Strictly forbids any simulator ground-truth noise parameters (`true_theta`, `qubit_p`, `alphas`)
from entering the policy observation vector.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import numpy as np

from training.observations.observation_schema import (
    OBS_V2_DIM,
    POSTERIOR_VAR_EPS,
    SLICE_HIST_X,
    SLICE_HIST_Z,
    SLICE_POSTERIOR_LOG_VAR,
    SLICE_POSTERIOR_MEAN,
    SLICE_TELEMETRY,
    validate_observation_v2,
    variance_to_log_uncertainty,
)
from training.observations.spatial_mapping import (
    matrix7x3_to_vector21,
    vector21_to_matrix7x3,
)
from training.observations.syndrome_histogram import compute_basis_syndrome_stats

FORBIDDEN_GROUND_TRUTH_KEYS = frozenset(
    {
        "true_theta",
        "true_theta_z",
        "true_theta_x",
        "ground_truth_theta",
        "qubit_p_true",
        "alphas_true",
        "oracle_theta",
    }
)


class ObservationBuilder:
    """Stateful/stateless builder producing validated 82-D observations for R2D2 V2."""

    def __init__(self, max_config_age_epochs: int = 16) -> None:
        self.max_config_age_epochs = max(1, int(max_config_age_epochs))
        self._prev_posterior_mean: Optional[np.ndarray] = None
        self._config_age_epochs: int = 0

    def reset(self) -> None:
        self._prev_posterior_mean = None
        self._config_age_epochs = 0

    def record_action_update(self, config_changed: bool) -> None:
        if config_changed:
            self._config_age_epochs = 0
        else:
            self._config_age_epochs += 1

    def build_from_posterior_and_syndromes(
        self,
        posterior_packet: Mapping[str, Any],
        syndromes_x: Optional[np.ndarray] = None,
        syndromes_z: Optional[np.ndarray] = None,
        validity_mask_x: Optional[np.ndarray] = None,
        validity_mask_z: Optional[np.ndarray] = None,
        burst_score_override: Optional[float] = None,
        decoder_config_age_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct the 82-D observation and companion 7x3 posterior matrices.

        Raises `ValueError` if any forbidden simulator ground-truth key is passed in `posterior_packet`.
        """
        leaked = FORBIDDEN_GROUND_TRUTH_KEYS.intersection(posterior_packet.keys())
        if leaked:
            raise ValueError(
                f"Ground-truth leakage blocked in ObservationBuilder: forbidden keys {sorted(leaked)}"
            )

        raw_mean = np.asarray(posterior_packet["posterior_mean_21"], dtype=np.float64)
        raw_var = np.asarray(posterior_packet["posterior_var_21"], dtype=np.float64)
        raw_mean = np.clip(raw_mean, 0.0, 0.49)

        # Round-trip through canonical 7x3 lattice mapping to enforce spatial index contract
        mat_mean_7x3 = vector21_to_matrix7x3(raw_mean)
        vec_mean_21 = matrix7x3_to_vector21(mat_mean_7x3)

        log_var_21 = variance_to_log_uncertainty(raw_var, eps=POSTERIOR_VAR_EPS)
        mat_logvar_7x3 = vector21_to_matrix7x3(log_var_21)
        vec_logvar_21 = matrix7x3_to_vector21(mat_logvar_7x3)

        # X and Z 16-state syndrome histograms and summary statistics
        if syndromes_x is not None:
            stats_x = compute_basis_syndrome_stats(syndromes_x, validity_mask_x)
        else:
            hx = np.asarray(posterior_packet["hist_x_16"], dtype=np.float64)
            hx_sum = float(np.sum(hx))
            hx = hx / hx_sum if hx_sum > 0.0 else np.eye(16, dtype=np.float64)[0]
            stats_x = {
                "histogram": hx,
                "density": float(posterior_packet.get("syndrome_density_x", 0.0)),
                "adjacent_coactivation": float(posterior_packet.get("adjacent_coactivation_x", 0.0)),
                "burst_subwindow_score": float(posterior_packet.get("burst_score", 0.0)),
            }

        if syndromes_z is not None:
            stats_z = compute_basis_syndrome_stats(syndromes_z, validity_mask_z)
        else:
            hz = np.asarray(posterior_packet["hist_z_16"], dtype=np.float64)
            hz_sum = float(np.sum(hz))
            hz = hz / hz_sum if hz_sum > 0.0 else np.eye(16, dtype=np.float64)[0]
            stats_z = {
                "histogram": hz,
                "density": float(posterior_packet.get("syndrome_density_z", 0.0)),
                "adjacent_coactivation": float(posterior_packet.get("adjacent_coactivation_z", 0.0)),
                "burst_subwindow_score": 0.0,
            }

        # Posterior innovation computed strictly from sequential posterior estimates (never true theta)
        if self._prev_posterior_mean is None:
            innovation_l2 = float(posterior_packet.get("posterior_innovation_l2", 0.0))
        else:
            innovation_l2 = float(np.linalg.norm(vec_mean_21 - self._prev_posterior_mean))
        self._prev_posterior_mean = vec_mean_21.copy()

        if burst_score_override is not None:
            burst_score = float(np.clip(burst_score_override, 0.0, 1.0))
        else:
            burst_from_pkt = float(posterior_packet.get("burst_score", -1.0))
            if burst_from_pkt >= 0.0:
                burst_score = float(np.clip(burst_from_pkt, 0.0, 1.0))
            else:
                burst_score = float(
                    np.clip(
                        max(
                            float(stats_x["burst_subwindow_score"]),
                            float(stats_z["burst_subwindow_score"]),
                            min(1.0, innovation_l2 * 5.5),
                        ),
                        0.0,
                        1.0,
                    )
                )

        estimator_residual = float(
            np.clip(posterior_packet.get("estimator_residual_tvd", 0.0), 0.0, 1.0)
        )
        age_epochs = (
            int(decoder_config_age_epochs)
            if decoder_config_age_epochs is not None
            else self._config_age_epochs
        )
        age_norm = float(np.clip(age_epochs / float(self.max_config_age_epochs), 0.0, 1.0))

        obs = np.zeros(OBS_V2_DIM, dtype=np.float64)
        obs[SLICE_POSTERIOR_MEAN] = vec_mean_21
        obs[SLICE_POSTERIOR_LOG_VAR] = vec_logvar_21
        obs[SLICE_HIST_X] = np.asarray(stats_x["histogram"], dtype=np.float64)
        obs[SLICE_HIST_Z] = np.asarray(stats_z["histogram"], dtype=np.float64)
        obs[SLICE_TELEMETRY] = np.array(
            [
                float(np.clip(stats_x["density"], 0.0, 1.0)),
                float(np.clip(stats_z["density"], 0.0, 1.0)),
                float(np.clip(stats_x["adjacent_coactivation"], 0.0, 1.0)),
                float(np.clip(stats_z["adjacent_coactivation"], 0.0, 1.0)),
                float(max(0.0, innovation_l2)),
                burst_score,
                estimator_residual,
                age_norm,
            ],
            dtype=np.float64,
        )

        validate_observation_v2(obs, strict_probabilities=True)

        return {
            "observation_82": obs,
            "posterior_mean_7x3": mat_mean_7x3,
            "posterior_logvar_7x3": mat_logvar_7x3,
            "hist_x_16": obs[SLICE_HIST_X].copy(),
            "hist_z_16": obs[SLICE_HIST_Z].copy(),
            "telemetry_8": obs[SLICE_TELEMETRY].copy(),
        }
