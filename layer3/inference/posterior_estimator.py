"""Sequential Bayesian Posterior Estimator (`posterior_estimator.py`).

Bridges observed X/Z syndrome windows to a calibrated 21-parameter posterior state
`PosteriorPacket(posterior_mean_21, posterior_var_21, ...)` for Layer 2 R2D2 V2.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from layer3.forward_model.parameter_transition_model import ParameterTransitionPrior
from layer3.identifiability.svd_analysis import analyze_jacobian_svd
from layer3.inference.posterior_head import LogitLaplacePosteriorHead
from layer3.inference.posterior_predictive import compute_posterior_predictive_checks
from layer3.inference.temporal_encoder import TemporalSyndromeEncoder
from layer3.interfaces.posterior_packet import PosteriorPacket


def _hist_from_syndromes(
    syndromes: np.ndarray, validity_mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, int]:
    arr = np.asarray(syndromes)
    if arr.ndim == 2 and arr.shape[1] == 4:
        b = (arr != 0).astype(np.uint8)
        packed = (b[:, 0] | (b[:, 1] << 1) | (b[:, 2] << 2) | (b[:, 3] << 3)).astype(np.uint8)
    else:
        packed = arr.astype(np.uint8)

    if validity_mask is not None:
        vm = np.asarray(validity_mask)
        valid_bool = np.all(vm != 0, axis=1) if vm.ndim == 2 else (vm != 0)
        packed = packed[valid_bool]

    n_valid = int(packed.shape[0])
    if n_valid == 0:
        h = np.zeros(16, dtype=np.float64)
        h[0] = 1.0
        return h, 0
    counts = np.bincount(packed, minlength=16).astype(np.float64)
    return counts / float(n_valid), n_valid


class SequentialPosteriorEstimator:
    """Layer 3 sequential Bayesian inverse estimator producing `PosteriorPacket` outputs."""

    def __init__(
        self,
        temporal_beta: float = 0.55,
        readout_flip_prob: float = 0.0,
        ar_persistence: float = 0.72,
    ) -> None:
        self.temporal_encoder = TemporalSyndromeEncoder(temporal_beta=temporal_beta)
        self.transition_prior = ParameterTransitionPrior(ar_persistence=ar_persistence)
        self.posterior_head = LogitLaplacePosteriorHead(max_iters=6, readout_flip_prob=readout_flip_prob)
        self.readout_flip_prob = float(readout_flip_prob)

        ref_svd = analyze_jacobian_svd(
            np.full(21, 0.012, dtype=np.float64),
            basis_checks="X",
            readout_flip_prob=self.readout_flip_prob,
        )
        self._cached_leverage = np.asarray(
            ref_svd["parameter_observability_leverage"], dtype=np.float64
        )
        self._cached_rank = int(ref_svd["rank"])
        self._cached_nullity = int(ref_svd["nullity"])

        self._prev_mean_logit: Optional[np.ndarray] = None
        self._prev_var_logit: Optional[np.ndarray] = None
        self._prev_mean_prob: Optional[np.ndarray] = None
        self._epoch_counter: int = 0

    def reset(self) -> None:
        self.temporal_encoder.reset()
        self._prev_mean_logit = None
        self._prev_var_logit = None
        self._prev_mean_prob = None
        self._epoch_counter = 0

    def configure_from_action(self, temporal_beta: float, recalibrate: bool = False) -> None:
        self.temporal_encoder.set_temporal_beta(temporal_beta)
        if recalibrate:
            self._prev_mean_logit = None
            self._prev_var_logit = None

    def step_from_histograms(
        self,
        hist_x_16: np.ndarray,
        hist_z_16: np.ndarray,
        valid_rounds_x: int,
        valid_rounds_z: int,
        recalibrate: bool = False,
        burst_score: float = -1.0,
    ) -> PosteriorPacket:
        """Update sequential 21-parameter Z-fault posterior from X/Z syndrome histograms."""
        filt = self.temporal_encoder.update(
            hist_x_16=hist_x_16,
            hist_z_16=hist_z_16,
            valid_rounds_x=valid_rounds_x,
            valid_rounds_z=valid_rounds_z,
            recalibrate=recalibrate,
        )
        prior_mu, prior_var = self.transition_prior.predict_prior(
            prev_mean_logit=self._prev_mean_logit,
            prev_var_logit=self._prev_var_logit,
            reset_prior=recalibrate,
        )

        # Infer 21 Z-basis physical parameters from X-stabilizer histogram
        post = self.posterior_head.infer_posterior(
            hist_16=np.asarray(filt["filtered_hist_x"], dtype=np.float64),
            effective_rounds=float(filt["effective_rounds_x"]),
            prior_mean_logit=prior_mu,
            prior_var_logit=prior_var,
            basis_checks="X",
        )
        mean_21 = np.asarray(post["posterior_mean_21"], dtype=np.float64)
        var_21 = np.asarray(post["posterior_var_21"], dtype=np.float64)

        self._prev_mean_logit = np.asarray(post["posterior_mean_logit_21"], dtype=np.float64)
        self._prev_var_logit = np.asarray(post["posterior_var_logit_21"], dtype=np.float64)

        innovation = (
            0.0
            if self._prev_mean_prob is None
            else float(np.linalg.norm(mean_21 - self._prev_mean_prob))
        )
        self._prev_mean_prob = mean_21.copy()

        ppc = compute_posterior_predictive_checks(
            observed_hist_16=hist_x_16,
            posterior_mean_21=mean_21,
            basis_checks="X",
            readout_flip_prob=self.readout_flip_prob,
        )

        pkt = PosteriorPacket(
            epoch_id=self._epoch_counter,
            posterior_mean_21=mean_21,
            posterior_var_21=var_21,
            hist_x_16=np.asarray(hist_x_16, dtype=np.float64),
            hist_z_16=np.asarray(hist_z_16, dtype=np.float64),
            valid_rounds_x=int(valid_rounds_x),
            valid_rounds_z=int(valid_rounds_z),
            estimator_residual_tvd=float(ppc["tvd"]),
            posterior_innovation_l2=innovation,
            burst_score=float(burst_score),
            observability_leverage_21=self._cached_leverage.copy(),
            diagnostics={
                "kl_divergence": float(ppc["kl_divergence"]),
                "jacobian_rank": self._cached_rank,
                "jacobian_nullity": self._cached_nullity,
            },
        )
        self._epoch_counter += 1
        return pkt

    def step_from_syndrome_windows(
        self,
        syndromes_x: np.ndarray,
        syndromes_z: np.ndarray,
        validity_mask_x: Optional[np.ndarray] = None,
        validity_mask_z: Optional[np.ndarray] = None,
        recalibrate: bool = False,
    ) -> PosteriorPacket:
        """Compute histograms from raw syndrome windows and run sequential posterior update."""
        hx, vx = _hist_from_syndromes(syndromes_x, validity_mask_x)
        hz, vz = _hist_from_syndromes(syndromes_z, validity_mask_z)
        return self.step_from_histograms(
            hist_x_16=hx,
            hist_z_16=hz,
            valid_rounds_x=vx,
            valid_rounds_z=vz,
            recalibrate=recalibrate,
        )
