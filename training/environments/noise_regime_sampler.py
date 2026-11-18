"""Physical 21-Parameter Trajectory Sampler Across Regimes & Held-Out Mixtures (`noise_regime_sampler.py`)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from training.observations.spatial_mapping import CANONICAL_12_COUPLERS
from training.rewards.noise_aware_reward import LEGACY_FAMILY_TO_REGIME


def sample_physical_noise_trajectory(
    family: str,
    seed: int,
    num_epochs: int = 16,
) -> Dict[str, object]:
    """Generate a multi-epoch ground-truth 21-parameter trajectory `(theta_z, theta_x)` and regime flags.

    Simulator ground truth is returned strictly for forward circuit simulation and training-time
    reward evaluation; it must NEVER be passed directly into R2D2's observation builder.
    """
    rng = np.random.default_rng(int(seed))
    canonical_regime = LEGACY_FAMILY_TO_REGIME.get(family, "gaussian_stationary")

    theta_z_seq = np.zeros((num_epochs, 21), dtype=np.float64)
    theta_x_seq = np.zeros((num_epochs, 21), dtype=np.float64)
    in_burst_seq = np.zeros(num_epochs, dtype=bool)
    post_burst_seq = np.zeros(num_epochs, dtype=bool)
    is_corrupted_seq = np.zeros(num_epochs, dtype=bool)
    readout_flip_seq = np.zeros(num_epochs, dtype=np.float64)
    streak3_prob_seq = np.zeros(num_epochs, dtype=np.float64)

    hot_pair_primary = int(seed % 6)  # Horizontal X-check-visible ZZ pair in 0..5
    tls_qubit = int((seed // 7) % 9)
    ou_state = rng.uniform(0.011, 0.016, size=9)

    for ep in range(num_epochs):
        qz = np.full(9, 0.012, dtype=np.float64)
        pz = np.full(12, 0.004, dtype=np.float64)
        qx = rng.uniform(0.004, 0.008, size=9)
        px = rng.uniform(0.001, 0.003, size=12)

        if family in ("gaussian_stationary", "stationary_independent"):
            base_p = rng.uniform(0.010, 0.015)
            qz = base_p + rng.normal(0.0, 0.0012, size=9)
            pz = rng.uniform(0.002, 0.006, size=12)

        elif family in ("ou_drift", "smooth_sinusoidal_drift"):
            phase = float(ep) / max(1, num_epochs - 1)
            target = np.full(9, 0.013 + 0.012 * math.sin(math.pi * phase))
            target[4] += 0.024 * phase
            target[5] += 0.014 * phase
            ou_state = 0.72 * ou_state + 0.28 * target + rng.normal(0.0, 0.0025, size=9)
            qz = np.clip(ou_state, 0.004, 0.085)
            pz = rng.uniform(0.004, 0.010, size=12)
            pz[3] = 0.018 + 0.038 * phase

        elif family in ("zz_crosstalk", "pair_correlated_crosstalk"):
            qz = rng.uniform(0.010, 0.015, size=9)
            pz = rng.uniform(0.003, 0.007, size=12)
            hot_val = rng.uniform(0.072, 0.098) if (ep % 4 != 0) else rng.uniform(0.045, 0.062)
            pz[hot_pair_primary] = hot_val
            u, v = CANONICAL_12_COUPLERS[hot_pair_primary]
            qz[u] += 0.018
            qz[v] += 0.012

        elif family in ("tls_telegraph", "abrupt_regime_switch"):
            qz = rng.uniform(0.010, 0.015, size=9)
            pz = rng.uniform(0.003, 0.008, size=12)
            tls_active = (4 <= ep < 11) or (ep >= 14)
            if tls_active:
                qz[tls_qubit] = rng.uniform(0.058, 0.082)
                # Couple adjacent edge moderately
                for k, (ua, ub) in enumerate(CANONICAL_12_COUPLERS):
                    if tls_qubit in (ua, ub):
                        pz[k] = max(pz[k], rng.uniform(0.028, 0.045))

        elif family in ("nonstationary_burst", "burst_intermittency_r2d2"):
            if ep in (4, 5, 10, 11):
                qz = rng.uniform(0.028, 0.044, size=9)
                pz = rng.uniform(0.015, 0.028, size=12)
                pz[hot_pair_primary] = rng.uniform(0.082, 0.112)
                u, v = CANONICAL_12_COUPLERS[hot_pair_primary]
                qz[u] += 0.020
                in_burst_seq[ep] = True
            elif ep in (6, 12):
                qz = rng.uniform(0.011, 0.016, size=9)
                pz = rng.uniform(0.004, 0.008, size=12)
                post_burst_seq[ep] = True
            elif ep == 8 and (seed % 3 == 0):
                # Corrupted unphysical / readout storm epoch testing safety fallback
                qz = rng.uniform(0.185, 0.245, size=9)
                pz = rng.uniform(0.105, 0.150, size=12)
                readout_flip_seq[ep] = 0.18
                is_corrupted_seq[ep] = True
            else:
                qz = rng.uniform(0.010, 0.015, size=9)
                pz = rng.uniform(0.003, 0.008, size=12)

        elif family == "heldout_multi_pair_12_67":
            qz = rng.uniform(0.011, 0.016, size=9)
            pz = rng.uniform(0.003, 0.008, size=12)
            qz[[1, 2, 6, 7]] = rng.uniform(0.036, 0.046, size=4)
            pz[1] = rng.uniform(0.072, 0.090)  # pair (1, 2)
            pz[4] = rng.uniform(0.074, 0.092)  # pair (6, 7)

        elif family == "heldout_spatial_3qubit_streak":
            qz = rng.uniform(0.011, 0.016, size=9)
            pz = rng.uniform(0.003, 0.008, size=12)
            qz[[3, 4, 5]] = [0.030, 0.046, 0.030]
            pz[2] = 0.044  # pair (3, 4)
            pz[3] = 0.068  # pair (4, 5)
            streak3_prob_seq[ep] = 0.018

        elif family == "heldout_phenomenological_readout":
            qz = rng.uniform(0.012, 0.018, size=9)
            pz = rng.uniform(0.004, 0.009, size=12)
            qz[4] = 0.046
            pz[3] = 0.072
            readout_flip_seq[ep] = 0.014
            if ep in (5, 6):
                in_burst_seq[ep] = True

        else:
            qz = rng.uniform(0.012, 0.022, size=9)
            pz = rng.uniform(0.005, 0.035, size=12)

        theta_z_seq[ep, :9] = np.clip(qz, 0.001, 0.35)
        theta_z_seq[ep, 9:] = np.clip(pz, 0.0005, 0.30)
        theta_x_seq[ep, :9] = np.clip(qx, 0.001, 0.20)
        theta_x_seq[ep, 9:] = np.clip(px, 0.0005, 0.15)

    return {
        "family": family,
        "canonical_regime": canonical_regime,
        "theta_z_seq": theta_z_seq,
        "theta_x_seq": theta_x_seq,
        "in_burst_seq": in_burst_seq,
        "post_burst_seq": post_burst_seq,
        "is_corrupted_seq": is_corrupted_seq,
        "readout_flip_seq": readout_flip_seq,
        "streak3_prob_seq": streak3_prob_seq,
    }
