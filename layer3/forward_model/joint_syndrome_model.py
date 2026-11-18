"""Combined Dual-Basis Joint Syndrome Forward Model (`joint_syndrome_model.py`)."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from layer3.forward_model.syndrome_likelihood import (
    CROSSTALK_PAIRS_12,
    exact_syndrome_probabilities,
)


class JointSyndromeForwardModel:
    """Forward simulator and exact analytical predictor for rotated d=3 X and Z syndromes."""

    def __init__(self, readout_flip_prob: float = 0.0) -> None:
        self.readout_flip_prob = float(np.clip(readout_flip_prob, 0.0, 0.49))

    def predict_distributions(
        self, theta_z_21: np.ndarray, theta_x_21: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Predict exact 16-state `p_sx` (from Z errors) and `p_sz` (from X errors)."""
        p_sx = exact_syndrome_probabilities(
            theta_z_21, basis_checks="X", readout_flip_prob=self.readout_flip_prob
        )
        p_sz = exact_syndrome_probabilities(
            theta_x_21, basis_checks="Z", readout_flip_prob=self.readout_flip_prob
        )
        return {"p_sx": p_sx, "p_sz": p_sz}

    def sample_syndrome_window(
        self,
        rng: np.random.Generator,
        theta_z_21: np.ndarray,
        theta_x_21: np.ndarray,
        window_rounds: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate `window_rounds` physical circuit shots returning `(err_z, err_x, sx, sz)`."""
        tz = np.clip(np.asarray(theta_z_21, dtype=np.float64), 0.0, 0.49)
        tx = np.clip(np.asarray(theta_x_21, dtype=np.float64), 0.0, 0.49)

        uz_q = rng.random((window_rounds, 9)) < tz[None, :9]
        ux_q = rng.random((window_rounds, 9)) < tx[None, :9]
        err_z = np.zeros(window_rounds, dtype=np.uint16)
        err_x = np.zeros(window_rounds, dtype=np.uint16)
        for q in range(9):
            err_z ^= uz_q[:, q].astype(np.uint16) << q
            err_x ^= ux_q[:, q].astype(np.uint16) << q

        uz_p = rng.random((window_rounds, 12)) < tz[None, 9:]
        ux_p = rng.random((window_rounds, 12)) < tx[None, 9:]
        for k, (qa, qb) in enumerate(CROSSTALK_PAIRS_12):
            mask = np.uint16((1 << qa) | (1 << qb))
            err_z ^= np.where(uz_p[:, k], mask, np.uint16(0))
            err_x ^= np.where(ux_p[:, k], mask, np.uint16(0))

        # X-checks detect Z errors
        bx0 = ((err_z >> 0) ^ (err_z >> 1)) & 1
        bx1 = ((err_z >> 1) ^ (err_z >> 2) ^ (err_z >> 4) ^ (err_z >> 5)) & 1
        bx2 = ((err_z >> 3) ^ (err_z >> 4) ^ (err_z >> 6) ^ (err_z >> 7)) & 1
        bx3 = ((err_z >> 7) ^ (err_z >> 8)) & 1
        sx = (bx0 | (bx1 << 1) | (bx2 << 2) | (bx3 << 3)).astype(np.uint8)

        # Z-checks detect X errors
        bz0 = ((err_x >> 0) ^ (err_x >> 1) ^ (err_x >> 3) ^ (err_x >> 4)) & 1
        bz1 = ((err_x >> 2) ^ (err_x >> 5)) & 1
        bz2 = ((err_x >> 3) ^ (err_x >> 6)) & 1
        bz3 = ((err_x >> 4) ^ (err_x >> 5) ^ (err_x >> 7) ^ (err_x >> 8)) & 1
        sz = (bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3)).astype(np.uint8)

        if self.readout_flip_prob > 0.0:
            fx = rng.random((window_rounds, 4)) < self.readout_flip_prob
            fz = rng.random((window_rounds, 4)) < self.readout_flip_prob
            for b in range(4):
                sx ^= fx[:, b].astype(np.uint8) << b
                sz ^= fz[:, b].astype(np.uint8) << b

        return err_z, err_x, sx, sz
