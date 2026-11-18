"""RWD-01 .. RWD-07 Mandatory Reward Unit Tests (`test_reward_components.py`)."""

from __future__ import annotations

import numpy as np

from training.rewards import (
    CANONICAL_REGIMES,
    COMPONENT_MAX,
    COMPONENT_MIN,
    REWARD_TERM_KEYS,
    RewardDiagnosticsTracker,
    TOTAL_REWARD_MAX,
    TOTAL_REWARD_MIN,
    compute_noise_aware_reward,
)


def test_rwd01_all_components_and_total_strictly_bounded() -> None:
    """RWD-01: Every component stays within [-1, +1] and total reward within [-2, +2]."""
    rng = np.random.default_rng(701)
    for _ in range(120):
        act = int(rng.integers(0, 6))
        gains = rng.uniform(-0.30, 0.30, size=9)
        prev_gains = rng.uniform(-0.30, 0.30, size=9)
        ler_a = float(rng.uniform(0.0, 0.45))
        ler_r = float(rng.uniform(0.0, 0.45))
        reg = CANONICAL_REGIMES[int(rng.integers(0, len(CANONICAL_REGIMES)))]
        th = rng.uniform(0.001, 0.25, size=21)
        prev_th = rng.uniform(0.001, 0.25, size=21)

        res = compute_noise_aware_reward(
            action_id=act,
            qubit_gains=gains,
            ler_action=ler_a,
            ler_reference=ler_r,
            regime=reg,
            prev_action_id=int(rng.integers(0, 6)),
            prev_qubit_gains=prev_gains,
            true_theta_z_21=th,
            prev_true_theta_z_21=prev_th,
            in_burst=bool(rng.random() < 0.3),
            post_burst_recovery=bool(rng.random() < 0.2),
            is_corrupted=bool(rng.random() < 0.2),
        )
        for k in REWARD_TERM_KEYS:
            if k == "reward/total_reward":
                assert TOTAL_REWARD_MIN <= res[k] <= TOTAL_REWARD_MAX
            else:
                assert COMPONENT_MIN <= res[k] <= COMPONENT_MAX


def test_rwd02_stationary_gaussian_penalizes_unnecessary_churn_and_masks_other_regimes() -> None:
    """RWD-02: Unnecessary action/K/recal changes are penalized in stationary noise, and inactive regime terms are zero."""
    zeros_g = np.zeros(9)
    steady = compute_noise_aware_reward(
        action_id=0,  # STATIC_PRIOR_K3
        qubit_gains=zeros_g,
        ler_action=0.020,
        ler_reference=0.020,
        regime="gaussian_stationary",
        prev_action_id=0,
        prev_qubit_gains=zeros_g,
    )
    churning = compute_noise_aware_reward(
        action_id=4,  # FAST_RECALIBRATE_K5 (unnecessary recalibrate + K=5 + action switch)
        qubit_gains=np.full(9, 0.25),
        ler_action=0.020,
        ler_reference=0.020,
        regime="gaussian_stationary",
        prev_action_id=0,
        prev_qubit_gains=zeros_g,
    )
    assert steady["reward/gaussian_stability"] > churning["reward/gaussian_stability"]
    assert steady["reward/decoder_stability"] > churning["reward/decoder_stability"]
    assert steady["reward/total_reward"] > churning["reward/total_reward"]
    # Verify regime responsibility masking keeps non-active regime rewards strictly 0
    assert steady["reward/burst_recovery"] == 0.0
    assert steady["reward/zz_correlation"] == 0.0


def test_rwd03_ou_drift_tracking_score() -> None:
    """RWD-03: OU drift rewards smooth adaptive tracking and aligned per-qubit gains."""
    prev_th = np.full(21, 0.012)
    curr_th = np.full(21, 0.012)
    curr_th[4] = 0.045  # drifting noisy central qubit
    aligned_gains = np.full(9, 0.05)
    aligned_gains[4] = -0.22  # attenuate noisy q4 LLR
    anti_gains = -aligned_gains

    good = compute_noise_aware_reward(
        action_id=3,  # DAMPED_PRIOR_K4 (beta=0.55, K=4)
        qubit_gains=aligned_gains,
        ler_action=0.032,
        ler_reference=0.048,
        regime="ou_drift",
        true_theta_z_21=curr_th,
        prev_true_theta_z_21=prev_th,
    )
    poor = compute_noise_aware_reward(
        action_id=5,  # fallback
        qubit_gains=anti_gains,
        ler_action=0.048,
        ler_reference=0.048,
        regime="ou_drift",
        true_theta_z_21=curr_th,
        prev_true_theta_z_21=prev_th,
    )
    assert good["reward/ou_tracking"] > poor["reward/ou_tracking"]
    assert good["reward/total_reward"] > poor["reward/total_reward"]


def test_rwd04_zz_correlations_evaluated_separately() -> None:
    """RWD-04: Pairwise crosstalk regime rewards pair-aware projection over static pair_weight_scale=0."""
    th = np.full(21, 0.012)
    th[9 + 3] = 0.082  # strong ZZ pair (4, 5)
    th[4] = 0.035
    th[5] = 0.030
    gains = np.zeros(9)
    gains[4] = -0.18
    gains[5] = -0.18

    pair_aware = compute_noise_aware_reward(
        action_id=2,  # AGGRESSIVE_PAIR_K5 (pair_weight_scale=1.35, K=5)
        qubit_gains=gains,
        ler_action=0.042,
        ler_reference=0.088,
        regime="zz_crosstalk",
        true_theta_z_21=th,
    )
    pair_blind = compute_noise_aware_reward(
        action_id=0,  # STATIC_PRIOR_K3 (pair_weight_scale=0.0, K=3)
        qubit_gains=np.zeros(9),
        ler_action=0.088,
        ler_reference=0.088,
        regime="zz_crosstalk",
        true_theta_z_21=th,
    )
    assert pair_aware["reward/zz_correlation"] > pair_blind["reward/zz_correlation"]
    assert pair_aware["reward/qec_performance"] > pair_blind["reward/qec_performance"]


def test_rwd05_tls_switching_detection_and_recovery() -> None:
    """RWD-05: TLS telegraph switching rewards rapid defect attenuation and recovery."""
    prev_th = np.full(21, 0.010)
    curr_th = np.full(21, 0.010)
    curr_th[2] = 0.065  # sudden TLS defect switch on q2
    gains_good = np.full(9, 0.04)
    gains_good[2] = -0.24

    detected = compute_noise_aware_reward(
        action_id=3,  # K=4, beta=0.55
        qubit_gains=gains_good,
        ler_action=0.030,
        ler_reference=0.052,
        regime="tls_telegraph",
        true_theta_z_21=curr_th,
        prev_true_theta_z_21=prev_th,
    )
    missed = compute_noise_aware_reward(
        action_id=0,
        qubit_gains=-gains_good,
        ler_action=0.055,
        ler_reference=0.052,
        regime="tls_telegraph",
        true_theta_z_21=curr_th,
        prev_true_theta_z_21=prev_th,
    )
    assert detected["reward/tls_switching"] > missed["reward/tls_switching"]


def test_rwd06_burst_damage_and_post_burst_recovery() -> None:
    """RWD-06: Burst onset rewards recalibration + K=5; post-burst recovery rewards returning to steady control."""
    g = np.zeros(9)
    during_good = compute_noise_aware_reward(
        action_id=4,  # FAST_RECALIBRATE_K5
        qubit_gains=g,
        ler_action=0.045,
        ler_reference=0.090,
        regime="nonstationary_burst",
        in_burst=True,
    )
    during_bad = compute_noise_aware_reward(
        action_id=0,  # no recalibrate, K=3
        qubit_gains=g,
        ler_action=0.090,
        ler_reference=0.090,
        regime="nonstationary_burst",
        in_burst=True,
    )
    assert during_good["reward/burst_recovery"] > during_bad["reward/burst_recovery"]

    rec_good = compute_noise_aware_reward(
        action_id=1,  # steady ADAPTIVE_PAIR_K4 without recalibrate/fallback
        qubit_gains=g,
        ler_action=0.025,
        ler_reference=0.035,
        regime="nonstationary_burst",
        in_burst=False,
        post_burst_recovery=True,
    )
    rec_stuck = compute_noise_aware_reward(
        action_id=4,  # stuck flushing EMA every epoch after burst ended
        qubit_gains=g,
        ler_action=0.035,
        ler_reference=0.035,
        regime="nonstationary_burst",
        in_burst=False,
        post_burst_recovery=True,
    )
    assert rec_good["reward/burst_recovery"] > rec_stuck["reward/burst_recovery"]


def test_rwd07_reward_decomposition_and_diagnostics_logging() -> None:
    """RWD-07: All 9 reward keys are logged independently and worst-regime score is tracked."""
    tracker = RewardDiagnosticsTracker()
    r1 = compute_noise_aware_reward(0, np.zeros(9), 0.02, 0.02, regime="gaussian_stationary")
    r2 = compute_noise_aware_reward(0, np.zeros(9), 0.09, 0.05, regime="zz_crosstalk")
    tracker.log_step(r1, regime="gaussian_stationary")
    tracker.log_step(r2, regime="zz_crosstalk")

    summary = tracker.summary()
    assert summary["num_steps"] == 2
    assert set(summary["mean_terms"].keys()) == set(REWARD_TERM_KEYS)
    assert summary["worst_regime"] == "zz_crosstalk"
    assert summary["worst_regime_reward"] < summary["per_regime_total_reward"]["gaussian_stationary"]
