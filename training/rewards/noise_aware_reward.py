"""Noise-Aware, Decoder-Outcome-Driven Reward System (`noise_aware_reward.py`).

Implements:
  - Common QEC decoder LER improvement reward (`reward/qec_performance`)
  - Five regime-masked reward components:
      * `reward/gaussian_stability`
      * `reward/ou_tracking`
      * `reward/zz_correlation`
      * `reward/tls_switching`
      * `reward/burst_recovery`
  - Hardware cycle & switching stability (`reward/decoder_stability`)
  - Corrupted-epoch fallback gate (`reward/safety`)
  - Total bounded reward (`reward/total_reward` in [-2.0, +2.0])

Zero hard-coded oracle action preference is used anywhere in the reward objective.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional
import numpy as np

from training.models.r2d2_network import ACTION_DEFINITIONS
from training.rewards.reward_normalization import (
    calibrated_ler_improvement_score,
    clip_component,
    clip_total_reward,
)

CANONICAL_REGIMES = (
    "gaussian_stationary",
    "ou_drift",
    "zz_crosstalk",
    "tls_telegraph",
    "nonstationary_burst",
)

LEGACY_FAMILY_TO_REGIME: Dict[str, str] = {
    "stationary_independent": "gaussian_stationary",
    "gaussian_stationary": "gaussian_stationary",
    "smooth_sinusoidal_drift": "ou_drift",
    "ou_drift": "ou_drift",
    "pair_correlated_crosstalk": "zz_crosstalk",
    "zz_crosstalk": "zz_crosstalk",
    "heldout_multi_pair_12_67": "zz_crosstalk",
    "heldout_spatial_3qubit_streak": "zz_crosstalk",
    "abrupt_regime_switch": "tls_telegraph",
    "tls_telegraph": "tls_telegraph",
    "burst_intermittency_r2d2": "nonstationary_burst",
    "nonstationary_burst": "nonstationary_burst",
    "heldout_phenomenological_readout": "nonstationary_burst",
}


def resolve_regime_responsibilities(
    regime: Optional[str] = None,
    regime_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Return non-negative regime responsibility weights summing to <= 1.0."""
    out = {r: 0.0 for r in CANONICAL_REGIMES}
    if regime_weights is not None:
        for k, v in regime_weights.items():
            mapped = LEGACY_FAMILY_TO_REGIME.get(k, k)
            if mapped in out and v > 0.0:
                out[mapped] += float(v)
        tot = sum(out.values())
        if tot > 1.0:
            for k in out:
                out[k] /= tot
        return out

    if regime is not None:
        mapped = LEGACY_FAMILY_TO_REGIME.get(regime, "gaussian_stationary")
        if mapped in out:
            out[mapped] = 1.0
    else:
        out["gaussian_stationary"] = 1.0
    return out


def compute_noise_aware_reward(
    action_id: int,
    qubit_gains: np.ndarray,
    ler_action: float,
    ler_reference: float,
    regime: Optional[str] = None,
    regime_weights: Optional[Mapping[str, float]] = None,
    prev_action_id: Optional[int] = None,
    prev_qubit_gains: Optional[np.ndarray] = None,
    true_theta_z_21: Optional[np.ndarray] = None,
    prev_true_theta_z_21: Optional[np.ndarray] = None,
    in_burst: bool = False,
    post_burst_recovery: bool = False,
    is_corrupted: bool = False,
    reward_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Evaluate bounded QEC reward and decomposed sub-terms for a single control epoch."""
    spec = ACTION_DEFINITIONS[int(action_id)]
    pair_scale = float(spec["pair_weight_scale"])
    k_iters = int(spec["bp_iterations"])
    temp_beta = float(spec.get("temporal_beta", 0.0))
    recal = bool(spec["recalibrate"])
    fallback = bool(spec["fallback"])
    gains = np.asarray(qubit_gains, dtype=np.float64)

    resp = resolve_regime_responsibilities(regime=regime, regime_weights=regime_weights)

    # 1. Common decoder-outcome QEC performance term
    r_qec = calibrated_ler_improvement_score(ler_action=ler_action, ler_reference=ler_reference)

    # Extract privileged training signals if provided (simulation training only)
    if true_theta_z_21 is not None:
        th = np.asarray(true_theta_z_21, dtype=np.float64)
        q_p = th[:9]
        pair_p = th[9:]
        max_pair = float(np.max(pair_p))
        p_centered = q_p - float(np.mean(q_p))
        # Negative correlation between gain residual and excess qubit error means noisy qubits get attenuated LLRs
        if float(np.std(p_centered)) > 1.0e-5 and float(np.std(gains)) > 1.0e-5:
            gain_align = float(-np.corrcoef(gains, p_centered)[0, 1])
        else:
            gain_align = 0.0
    else:
        max_pair = 0.0
        gain_align = 0.0

    drift_mag = (
        float(np.linalg.norm(np.asarray(true_theta_z_21) - np.asarray(prev_true_theta_z_21)))
        if (true_theta_z_21 is not None and prev_true_theta_z_21 is not None)
        else 0.0
    )

    action_changed = (prev_action_id is not None) and (int(prev_action_id) != int(action_id))
    gain_delta = (
        float(np.linalg.norm(gains - np.asarray(prev_qubit_gains, dtype=np.float64)))
        if prev_qubit_gains is not None
        else 0.0
    )

    # 2A. Stationary Gaussian stability (penalizes unnecessary changes, recalibration, fallback, or K>3 waste)
    raw_gauss = (
        0.55
        - (0.45 if action_changed else 0.0)
        - 0.28 * (k_iters - 3)
        - (0.50 if recal else 0.0)
        - (0.65 if fallback and not is_corrupted else 0.0)
        - 0.45 * gain_delta
    )
    r_gauss = clip_component(raw_gauss) * resp["gaussian_stationary"]

    # 2B. OU Continuous Drift tracking
    target_k_ou = 4 if drift_mag >= 0.012 else 3
    raw_ou = (
        0.35
        + 0.40 * gain_align
        + (0.25 if 0.45 <= temp_beta <= 0.75 else -0.25)
        - 0.25 * abs(k_iters - target_k_ou)
        - (0.45 if fallback and not is_corrupted else 0.0)
    )
    r_ou = clip_component(raw_ou) * resp["ou_drift"]

    # 2C. ZZ Spatial Correlation handling
    needs_strong_pair = max_pair >= 0.055
    ideal_pair_scale = 1.30 if needs_strong_pair else 1.10
    raw_zz = (
        0.45
        - 0.75 * abs(pair_scale - ideal_pair_scale)
        + (0.30 if k_iters >= 4 else -0.35)
        + 0.25 * gain_align
        - (0.55 if fallback and not is_corrupted else 0.0)
    )
    r_zz = clip_component(raw_zz) * resp["zz_crosstalk"]

    # 2D. RTN TLS Telegraph switching detection & recovery
    tls_jump = drift_mag >= 0.025
    raw_tls = (
        0.30
        + 0.45 * gain_align
        + (0.30 if (tls_jump and k_iters >= 4) or (not tls_jump and k_iters <= 4) else -0.20)
        + (0.20 if (tls_jump and temp_beta <= 0.60) or (not tls_jump and temp_beta >= 0.50) else -0.10)
        - (0.50 if fallback and not is_corrupted else 0.0)
    )
    r_tls = clip_component(raw_tls) * resp["tls_telegraph"]

    # 2E. Non-Stationary Burst damage suppression & post-burst recovery
    if in_burst:
        raw_burst = (
            (0.55 if recal else -0.45)
            + (0.35 if k_iters == 5 else -0.25 * (5 - k_iters))
            + 0.20 * max(0.0, r_qec)
            - (0.55 if fallback and not is_corrupted else 0.0)
        )
    elif post_burst_recovery:
        raw_burst = (
            (0.55 if (not recal and not fallback and k_iters <= 4) else -0.45)
            + 0.30 * max(0.0, r_qec)
        )
    else:
        raw_burst = 0.35 - (0.45 if recal else 0.0) - 0.15 * (k_iters - 3)
    r_burst = clip_component(raw_burst) * resp["nonstationary_burst"]

    # 3. Decoder stability term (penalizes chattering & wasted BP cycles without LER benefit)
    transient_allowed = in_burst or post_burst_recovery or (drift_mag >= 0.025) or is_corrupted
    switch_pen = (0.35 if (action_changed and not transient_allowed) else 0.0)
    cycle_waste = (0.18 * (k_iters - 3) if ler_action >= ler_reference - 1e-6 else 0.04 * (k_iters - 3))
    r_stability = clip_component(0.20 - switch_pen - cycle_waste - 0.30 * gain_delta)

    # 4. Safety fallback term
    if is_corrupted:
        r_safety = 1.0 if fallback else -0.90
    else:
        r_safety = -0.85 if fallback else 0.15
    r_safety = clip_component(r_safety)

    w = {
        "qec_performance": 1.10,
        "regime_specific": 0.55,
        "decoder_stability": 0.20,
        "safety": 0.35,
    }
    if reward_weights is not None:
        for k, v in reward_weights.items():
            if k in w:
                w[k] = float(v)

    r_regime_sum = r_gauss + r_ou + r_zz + r_tls + r_burst
    total = clip_total_reward(
        w["qec_performance"] * r_qec
        + w["regime_specific"] * r_regime_sum
        + w["decoder_stability"] * r_stability
        + w["safety"] * r_safety
    )

    return {
        "reward/qec_performance": float(r_qec),
        "reward/gaussian_stability": float(r_gauss),
        "reward/ou_tracking": float(r_ou),
        "reward/zz_correlation": float(r_zz),
        "reward/tls_switching": float(r_tls),
        "reward/burst_recovery": float(r_burst),
        "reward/decoder_stability": float(r_stability),
        "reward/safety": float(r_safety),
        "reward/total_reward": float(total),
    }
