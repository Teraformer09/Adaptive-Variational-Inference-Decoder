"""R2D2 Trainer with hardware-equivalent d=3 decoder-in-the-loop reward evaluation and Adam optimizer."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from training.models.r2d2_network import ACTION_DEFINITIONS, R2D2Network
from training.replay.sequence_sampler import SequenceSegment, slice_episode_into_segments
from training.replay.prioritized_replay import PrioritizedSequenceReplayBuffer
from training.algorithms.loss import compute_n_step_transformed_td_loss
from training.algorithms.target_network import TargetNetworkManager

# Canonical d=3 rotated surface code geometry and 21-event X-check signatures
CROSSTALK_PAIRS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
    (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),
]

EVENT_SIGS_X = np.array(
    [
        0x1, 0x3, 0x2, 0x4, 0x6, 0x2, 0x4, 0xC, 0x8,
        0x2, 0x1, 0x2, 0x4, 0x8, 0x4, 0x5, 0x0, 0x5, 0xA, 0x0, 0xA,
    ],
    dtype=np.uint8,
)

STATIC_LUT_Z = np.array(
    [
        0x000, 0x001, 0x020, 0x002,
        0x008, 0x009, 0x010, 0x011,
        0x100, 0x101, 0x120, 0x102,
        0x080, 0x081, 0x110, 0x082,
    ],
    dtype=np.uint16,
)


def sample_d3_z_window(
    rng: np.random.Generator,
    qubit_p: np.ndarray,
    alphas: np.ndarray,
    window_rounds: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample `window_rounds` shots of 9-qubit Z errors + 12 ZZ pair errors on rotated d=3."""
    u_q = rng.random((window_rounds, 9)) < qubit_p[None, :]
    err_z = np.zeros(window_rounds, dtype=np.uint16)
    for q in range(9):
        err_z ^= (u_q[:, q].astype(np.uint16) << q)

    u_p = rng.random((window_rounds, 12)) < alphas[None, :]
    for k, (qa, qb) in enumerate(CROSSTALK_PAIRS):
        mask = np.uint16((1 << qa) | (1 << qb))
        err_z ^= np.where(u_p[:, k], mask, np.uint16(0))

    b0 = ((err_z >> 0) ^ (err_z >> 1)) & 1
    b1 = ((err_z >> 1) ^ (err_z >> 2) ^ (err_z >> 4) ^ (err_z >> 5)) & 1
    b2 = ((err_z >> 3) ^ (err_z >> 4) ^ (err_z >> 6) ^ (err_z >> 7)) & 1
    b3 = ((err_z >> 7) ^ (err_z >> 8)) & 1
    sx = (b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)).astype(np.uint8)
    return err_z, sx


def project_q78_weights_d3(
    qubit_p: np.ndarray,
    alphas: np.ndarray,
    pair_weight_scale: float,
    qubit_gains: Optional[np.ndarray] = None,
    fallback: bool = False,
) -> np.ndarray:
    """Hardware-equivalent 12-pair Q7.8 LLR projection matching `NoiseModel::to_pair_projected_q7_8_llr`."""
    if fallback:
        return np.full(9, 0x0468, dtype=np.int32)

    p_eff = np.clip(qubit_p.copy(), 1e-4, 0.45)
    for k, (u, v) in enumerate(CROSSTALK_PAIRS):
        ak = float(np.clip(alphas[k], 0.0, 0.45))
        p_eff[u] = p_eff[u] * (1.0 - ak) + (1.0 - p_eff[u]) * ak
        p_eff[v] = p_eff[v] * (1.0 - ak) + (1.0 - p_eff[v]) * ak
    p_eff = np.clip(p_eff, 1e-4, 0.45)

    llr = np.log((1.0 - p_eff) / p_eff)
    if qubit_gains is not None:
        llr = llr * qubit_gains
    w_q78 = np.clip(np.round(llr * 256.0).astype(np.int32), 16, 0x3FFF)

    if pair_weight_scale <= 0.0:
        return w_q78

    for g in range(1, 16):
        best_k = -1
        best_score = -1.0
        for k, (u, v) in enumerate(CROSSTALK_PAIRS):
            if EVENT_SIGS_X[9 + k] == g and EVENT_SIGS_X[u] != EVENT_SIGS_X[v]:
                score = float(alphas[k]) + 0.25 * float(qubit_p[u] + qubit_p[v])
                if qubit_gains is not None:
                    end_g = max(0.5, 0.5 * float(qubit_gains[u] + qubit_gains[v]))
                    score /= end_g
                if score > best_score:
                    best_score = score
                    best_k = k
        if best_k < 0:
            continue

        aliased = [w for w in range(9) if EVENT_SIGS_X[w] == g]
        if not aliased:
            continue

        u, v = CROSSTALK_PAIRS[best_k]
        p_pair = float(alphas[best_k])
        p_end_max = float(max(qubit_p[u], qubit_p[v]))
        p_base_prior = 0.012

        gain_boost = 1.0
        if qubit_gains is not None:
            end_g_avg = max(0.5, 0.5 * float(qubit_gains[u] + qubit_gains[v]))
            single_g = float(qubit_gains[aliased[0]])
            gain_boost = float(np.clip(single_g / end_g_avg, 0.72, 1.42))

        end_support = (0.35 * max(0.0, p_end_max - p_base_prior)) if (p_pair > 0.013 / gain_boost) else 0.0
        pair_ev = pair_weight_scale * gain_boost * (p_pair + end_support)
        if pair_ev > p_base_prior + 0.008:
            p_eff_pair = float(np.clip(max(p_pair, 0.65 * p_end_max), 0.015, 0.38))
            eff_scale = 0.85 + 0.15 * pair_weight_scale + 0.85 * (gain_boost - 1.0)
            scale_inv = 1.0 / float(np.clip(eff_scale, 0.50, 2.20))
            pair_half = int(np.clip(round(0.48 * scale_inv * np.log((1.0 - p_eff_pair) / p_eff_pair) * 256.0), 16, 0x0200))
            u_half = int(np.clip(round(pair_half * float(qubit_gains[u])), 16, 0x0220)) if qubit_gains is not None else pair_half
            v_half = int(np.clip(round(pair_half * float(qubit_gains[v])), 16, 0x0220)) if qubit_gains is not None else pair_half
            w_q78[u] = min(int(w_q78[u]), u_half)
            w_q78[v] = min(int(w_q78[v]), v_half)
            pair_sum = int(w_q78[u] + w_q78[v])
            for w in aliased:
                margin = int(np.clip(round(32.0 * float(qubit_gains[w])), 16, 96)) if qubit_gains is not None else 32
                w_q78[w] = max(int(w_q78[w]), pair_sum + margin)

    return w_q78


def build_hardware_osd_lut_z(w: np.ndarray) -> np.ndarray:
    """Build the 16-entry Z-correction lookup table matching `bp_engine.sv` for weight bank `w[0..8]`."""
    lut = np.zeros(16, dtype=np.uint16)
    lut[0x0] = 0x000
    lut[0x1] = 0x006 if (w[1] + w[2] < w[0]) else 0x001

    if (w[0] + w[1] <= w[3] + w[4]) and (w[0] + w[1] < w[2]) and (w[0] + w[1] < w[5]):
        lut[0x2] = 0x003
    elif (w[3] + w[4] < w[2]) and (w[3] + w[4] < w[5]):
        lut[0x2] = 0x018
    elif w[2] <= w[5]:
        lut[0x2] = 0x004
    else:
        lut[0x2] = 0x020

    lut[0x3] = 0x002 if (w[1] <= w[0] + w[5]) else 0x021

    if (w[4] + w[5] <= w[7] + w[8]) and (w[4] + w[5] < w[3]) and (w[4] + w[5] < w[6]):
        lut[0x4] = 0x030
    elif (w[7] + w[8] < w[3]) and (w[7] + w[8] < w[6]):
        lut[0x4] = 0x180
    elif w[6] < w[3]:
        lut[0x4] = 0x040
    else:
        lut[0x4] = 0x008

    if ((w[4] + w[5]) << 1) < w[3] and ((w[4] + w[5]) << 1) < w[1]:
        lut[0x5] = 0x031
    elif ((w[1] + w[2]) << 1) < w[0] and ((w[1] + w[2]) << 1) < w[3]:
        lut[0x5] = 0x046
    elif (w[1] + w[4]) < (w[0] + w[3]):
        lut[0x5] = 0x012
    else:
        lut[0x5] = 0x009

    lut[0x6] = 0x010 if (w[4] <= w[3] + w[5]) else 0x028

    if ((w[4] + w[5] + 40) < w[3]) and ((w[4] + w[5]) < w[0]):
        lut[0x7] = 0x032
    elif (w[0] + w[4]) <= (w[1] + w[3]):
        lut[0x7] = 0x011
    else:
        lut[0x7] = 0x00A

    lut[0x8] = 0x0C0 if (w[6] + w[7] < w[8]) else 0x100

    if (w[1] + w[2] + w[6] + w[7]) < (w[0] + w[8]):
        lut[0x9] = 0x0C6
    elif ((w[1] + w[2]) << 1) < w[0]:
        lut[0x9] = 0x106
    elif ((w[6] + w[7]) << 1) < w[8]:
        lut[0x9] = 0x0C1
    else:
        lut[0x9] = 0x101

    if ((w[6] + w[7]) << 1) < w[8] and ((w[6] + w[7]) << 1) < w[4]:
        lut[0xA] = 0x0C4
    elif (w[4] + w[7]) < (w[5] + w[8]):
        lut[0xA] = 0x090
    else:
        lut[0xA] = 0x120

    lut[0xB] = 0x0C2 if (w[6] + w[7] < w[8]) else 0x102
    lut[0xC] = 0x080 if (w[7] <= w[3] + w[8]) else 0x108
    lut[0xD] = 0x086 if (w[1] + w[2] < w[0]) else 0x081

    if ((w[6] + w[7]) << 1) < w[8] and ((w[6] + w[7]) << 1) < w[5]:
        lut[0xE] = 0x0D0
    elif (w[4] + w[8]) <= (w[5] + w[7]):
        lut[0xE] = 0x110
    else:
        lut[0xE] = 0x0A0

    lut[0xF] = 0x082
    return lut


def count_logical_z_failures(err_z: np.ndarray, corr_z: np.ndarray) -> int:
    """Count logical Z_L errors (`X_L = X_0 X_3 X_6` homology check) on a batch of decoded shots."""
    res = (err_z ^ corr_z) & np.uint16(0x1FF)
    lz = ((res >> 0) ^ (res >> 3) ^ (res >> 6)) & np.uint16(1)
    return int(np.sum(lz != 0))


def _oracle_action_and_gains_for_state(
    feat: np.ndarray,
    qubit_p: np.ndarray,
    pair_hotspot_mask: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    err_z_win: Optional[np.ndarray] = None,
    sx_win: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """Determine optimal control action and target per-qubit gain residual using decoder-in-the-loop evaluation."""
    # Per-qubit gain residual target in [-0.24, +0.24]^9:
    # Quiet qubits get positive LLR boost (+gain -> stronger prior confidence);
    # Noisy/crosstalk hotspot qubits get negative LLR attenuation (-gain -> lower LLR so pair wins)
    max_alpha = float(feat[3])
    p_centered = qubit_p - np.mean(qubit_p)
    raw_gain = -8.0 * p_centered - 0.14 * pair_hotspot_mask * max(0.35, min(1.5, max_alpha / 0.08))
    target_gains = np.clip(raw_gain, -0.24, 0.24).astype(np.float64)

    mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst = feat
    is_corrupted = (mean_p > 0.135 or max_p > 0.24 or syn_dens > 0.46)
    is_burst = (not is_corrupted) and (burst > 0.45 or innov > 0.055)
    is_strong_corr = (not is_corrupted) and (not is_burst) and (
        max_alpha >= 0.068 and mean_p < 0.0235
    )
    is_damped = (not is_corrupted) and (not is_burst) and (not is_strong_corr) and (
        (mean_p >= 0.0235) or (max_p >= 0.024 and max_alpha < 0.036) or (0.015 <= max_alpha < 0.036)
    )
    is_mod_corr = (not is_corrupted) and (not is_burst) and (not is_strong_corr) and (not is_damped) and (
        max_alpha >= 0.036 or mean_alpha >= 0.012
    )

    if is_corrupted:
        opt_action = 5
    elif is_burst:
        opt_action = 4
    elif is_strong_corr:
        opt_action = 2
    elif is_damped:
        opt_action = 3
    elif is_mod_corr:
        opt_action = 1
    else:
        opt_action = 0

    # Evaluate candidate actions via decoder-in-the-loop reward when syndrome stream is provided
    if alphas is not None and err_z_win is not None and sx_win is not None:
        best_act = opt_action
        best_score = -1e9
        gains_mult = np.clip(1.0 + target_gains, 0.72, 1.28)
        f_static = count_logical_z_failures(err_z_win, STATIC_LUT_Z[sx_win])
        for a in range(len(ACTION_DEFINITIONS)):
            spec = ACTION_DEFINITIONS[a]
            w_a = project_q78_weights_d3(
                qubit_p=qubit_p,
                alphas=alphas,
                pair_weight_scale=float(spec["pair_weight_scale"]),
                qubit_gains=gains_mult if float(spec["pair_weight_scale"]) > 0.0 else None,
                fallback=bool(spec["fallback"]),
            )
            lut_a = build_hardware_osd_lut_z(w_a)
            f_a = count_logical_z_failures(err_z_win, lut_a[sx_win])
            ler_gain = (f_static - f_a) / max(4.0, float(f_static))
            score = compute_qec_action_reward(feat, a, opt_action, decoded_ler_gain=ler_gain)
            if score > best_score:
                best_score = score
                best_act = a
        return best_act, target_gains

    return opt_action, target_gains


def compute_qec_action_reward(
    feat: np.ndarray,
    action_id: int,
    opt_action: int = -1,
    decoded_ler_gain: Optional[float] = None,
) -> float:
    """Compute QEC control reward from decoded LER reduction, cycle efficiency, and regime stability."""
    mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst = feat
    act_spec = ACTION_DEFINITIONS[action_id]
    gamma_pair = float(act_spec["pair_weight_scale"])
    k_iters = int(act_spec["bp_iterations"])
    beta_temp = float(act_spec.get("temporal_beta", 0.0))
    recal = bool(act_spec["recalibrate"])
    fallback = bool(act_spec["fallback"])

    is_corrupted = (mean_p > 0.135 or max_p > 0.24 or syn_dens > 0.46)
    is_burst = (not is_corrupted) and (burst > 0.45 or innov > 0.055)
    is_strong_corr = (not is_corrupted) and (not is_burst) and (
        max_alpha >= 0.068 and mean_p < 0.0235
    )
    is_damped = (not is_corrupted) and (not is_burst) and (not is_strong_corr) and (
        (mean_p >= 0.0235) or (max_p >= 0.024 and max_alpha < 0.036) or (0.015 <= max_alpha < 0.036)
    )
    is_mod_corr = (not is_corrupted) and (not is_burst) and (not is_strong_corr) and (not is_damped) and (
        max_alpha >= 0.036 or mean_alpha >= 0.012
    )

    if is_corrupted:
        ideal_gamma, ideal_k, ideal_beta = 0.00, 3, 0.00
    elif is_burst:
        ideal_gamma, ideal_k, ideal_beta = 1.25, 5, 0.00
    elif is_strong_corr:
        ideal_gamma, ideal_k, ideal_beta = 1.35, 5, 0.50
    elif is_damped:
        ideal_gamma, ideal_k, ideal_beta = 1.05, 4, 0.55
    elif is_mod_corr:
        ideal_gamma, ideal_k, ideal_beta = 1.15, 4, 0.65
    else:
        ideal_gamma, ideal_k, ideal_beta = 0.00, 3, 0.75

    # 1. Measured decoder LER reduction term + projection calibration
    gamma_err = abs(gamma_pair - ideal_gamma)
    beta_err = abs(beta_temp - ideal_beta)
    fidelity_term = 1.65 - 2.2 * gamma_err - 0.75 * beta_err
    if decoded_ler_gain is not None:
        fidelity_term += 0.35 * float(np.clip(decoded_ler_gain, -0.5, 0.8))

    # 2. Hardware BP iteration cycle budget (L(K) = 2 + 5*K cycles)
    if k_iters < ideal_k:
        fidelity_term -= 0.55 * (ideal_k - k_iters)
    else:
        fidelity_term -= 0.18 * (k_iters - ideal_k)

    # 3. Temporal histogram flush (recalibrate) on burst transitions vs steady state
    if is_burst:
        fidelity_term += 0.95 if recal else -0.75
    elif recal:
        fidelity_term -= 0.65

    # 4. Safety fallback gate during unphysical / corrupted estimator epochs
    if is_corrupted:
        fidelity_term += 1.35 if fallback else -1.30
    elif fallback:
        fidelity_term -= 0.95

    if opt_action >= 0 and action_id == opt_action:
        fidelity_term += 0.45

    return float(fidelity_term)


def generate_qec_control_episode(
    family: str,
    seed: int,
    num_epochs: int = 16,
    policy_net: Optional[R2D2Network] = None,
    epsilon: float = 0.25,
    window_rounds: int = 256,
) -> Dict[str, np.ndarray]:
    """Generate a multi-epoch QEC control trajectory with real d=3 syndrome decoding in the loop."""
    rng = np.random.default_rng(seed)
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    done_list: List[float] = []
    gain_list: List[np.ndarray] = []
    opt_act_list: List[int] = []
    h_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    dec_failures_list: List[int] = []
    static_failures_list: List[int] = []

    if policy_net is not None:
        h_curr, c_curr = policy_net.zero_state()
    else:
        h_curr = np.zeros(64, dtype=np.float64)
        c_curr = np.zeros(64, dtype=np.float64)

    prev_p = np.full(9, 0.015, dtype=np.float64)
    prev_alpha = np.full(12, 0.008, dtype=np.float64)

    for ep in range(num_epochs):
        qubit_p = np.full(9, 0.015, dtype=np.float64)
        alphas = np.full(12, 0.006, dtype=np.float64)
        hotspot_mask = np.zeros(9, dtype=np.float64)
        burst_score = 0.05 + 0.04 * rng.random()

        if family == "stationary_independent":
            base_p = rng.uniform(0.011, 0.015)
            qubit_p = base_p + rng.normal(0.0, 0.0010, size=9)
            alphas = rng.uniform(0.003, 0.008, size=12)
            if ep in (3, 7, 11, 15):
                qubit_p = rng.uniform(0.026, 0.036, size=9)
                alphas = rng.uniform(0.016, 0.028, size=12)
        elif family == "pair_correlated_crosstalk":
            base_p = rng.uniform(0.011, 0.014)
            qubit_p = base_p + rng.normal(0.0, 0.0010, size=9)
            hot_pair = int(seed % 6)  # Persistent horizontal Z-check crosstalk pair per episode
            hot_val = rng.uniform(0.074, 0.098) if ep % 3 != 0 else rng.uniform(0.042, 0.060)
            alphas = rng.uniform(0.003, 0.008, size=12)
            alphas[hot_pair] = hot_val
            q_a, q_b = CROSSTALK_PAIRS[hot_pair]
            hotspot_mask[q_a] = 1.0
            hotspot_mask[q_b] = 1.0
            qubit_p[q_a] += 0.024
            qubit_p[q_b] += 0.010
        elif family == "smooth_sinusoidal_drift":
            phase = float(ep) / max(1, num_epochs - 1)
            base_p = 0.012
            qubit_p = base_p + rng.normal(0.0, 0.0010, size=9)
            alphas = rng.uniform(0.003, 0.007, size=12)
            if seed % 2 == 0:
                # Multi-pair crosstalk regime (pairs (1,2) and (6,7))
                qubit_p[1] = rng.uniform(0.038, 0.048)
                qubit_p[2] = rng.uniform(0.038, 0.048)
                qubit_p[6] = rng.uniform(0.040, 0.050)
                qubit_p[7] = rng.uniform(0.040, 0.050)
                alphas[1] = rng.uniform(0.066, 0.084)
                alphas[4] = rng.uniform(0.068, 0.086)
                hotspot_mask[[1, 2, 6, 7]] = 1.0
            else:
                # Single-hotspot monotonic/smooth drift regime (matching slowly_drifting_noise)
                if ep < num_epochs // 2:
                    qubit_p[4] = 0.026 + 0.010 * phase
                    alphas[3] = 0.019 + 0.012 * phase
                else:
                    qubit_p[4] = 0.038 + 0.016 * phase
                    alphas[3] = 0.042 + 0.018 * phase
                hotspot_mask[4] = 1.0
                hotspot_mask[5] = 1.0
        elif family == "abrupt_regime_switch":
            if ep < num_epochs // 2:
                qubit_p = rng.uniform(0.011, 0.016, size=9)
                alphas = rng.uniform(0.004, 0.009, size=12)
            else:
                qubit_p = rng.uniform(0.027, 0.038, size=9)
                alphas = rng.uniform(0.014, 0.026, size=12)
                alphas[3] = rng.uniform(0.074, 0.102)
                hotspot_mask[4] = 1.0
                hotspot_mask[5] = 1.0
                qubit_p[4] += 0.022
                if ep == num_epochs // 2 or ep == num_epochs // 2 + 1:
                    burst_score = rng.uniform(0.65, 0.90)
        elif family == "burst_intermittency_r2d2":
            if ep in (4, 5, 9, 10):
                qubit_p = rng.uniform(0.030, 0.045, size=9)
                alphas = rng.uniform(0.018, 0.032, size=12)
                hot_edge = int(seed % 6)
                alphas[hot_edge] = rng.uniform(0.080, 0.115)
                burst_score = rng.uniform(0.65, 0.94)
                u, v = CROSSTALK_PAIRS[hot_edge]
                hotspot_mask[u] = 1.0
                hotspot_mask[v] = 1.0
                qubit_p[u] += 0.020
            elif ep in (6, 7, 11, 12):
                qubit_p = rng.uniform(0.175, 0.250, size=9)
                alphas = rng.uniform(0.100, 0.165, size=12)
                burst_score = 0.15
            else:
                qubit_p = rng.uniform(0.011, 0.016, size=9)
                alphas = rng.uniform(0.004, 0.009, size=12)
        else:
            qubit_p = rng.uniform(0.014, 0.024, size=9)
            alphas = rng.uniform(0.008, 0.045, size=12)

        qubit_p = np.clip(qubit_p, 0.001, 0.30)
        alphas = np.clip(alphas, 0.0, 0.25)

        # Sample real d=3 errors and syndromes for this window
        err_z_win, sx_win = sample_d3_z_window(rng, qubit_p, alphas, window_rounds=window_rounds)

        mean_p = float(np.mean(qubit_p))
        max_p = float(np.max(qubit_p))
        mean_alpha = float(np.mean(alphas))
        max_alpha = float(np.max(alphas))
        syn_dens = float(np.clip(2.4 * mean_p + 1.6 * mean_alpha + rng.normal(0.0, 0.005), 0.01, 0.65))
        adj_rate = float(np.clip(0.9 * max_alpha + 0.8 * mean_alpha + rng.normal(0.0, 0.003), 0.005, 0.40))
        innov = float(
            np.sqrt((np.sum((qubit_p - prev_p) ** 2) + np.sum((alphas - prev_alpha) ** 2)) / 21.0)
        )
        prev_p = qubit_p.copy()
        prev_alpha = alphas.copy()

        feat = np.array(
            [mean_p, max_p, mean_alpha, max_alpha, syn_dens, adj_rate, innov, burst_score],
            dtype=np.float64,
        )
        opt_act, targ_gains = _oracle_action_and_gains_for_state(
            feat, qubit_p, hotspot_mask, alphas=alphas, err_z_win=err_z_win, sx_win=sx_win
        )

        h_list.append(h_curr.copy())
        c_list.append(c_curr.copy())

        if policy_net is not None:
            step_out = policy_net.step(feat, h_curr, c_curr)
            h_curr = step_out["h_next"]
            c_curr = step_out["c_next"]
            if rng.random() < epsilon:
                chosen_act = int(rng.integers(0, policy_net.num_actions))
            else:
                chosen_act = int(step_out["action_id"])
            policy_gains = np.clip(1.0 + step_out["qubit_gains"], 0.72, 1.28)
        else:
            chosen_act = opt_act if rng.random() > epsilon else int(rng.integers(0, 6))
            policy_gains = np.clip(1.0 + targ_gains, 0.72, 1.28)

        # Decode the sampled window using the chosen action's projected Q7.8 hardware weights
        act_spec = ACTION_DEFINITIONS[chosen_act]
        w_chosen = project_q78_weights_d3(
            qubit_p=qubit_p,
            alphas=alphas,
            pair_weight_scale=float(act_spec["pair_weight_scale"]),
            qubit_gains=policy_gains if float(act_spec["pair_weight_scale"]) > 0.0 else None,
            fallback=bool(act_spec["fallback"]),
        )
        lut_chosen = build_hardware_osd_lut_z(w_chosen)
        f_chosen = count_logical_z_failures(err_z_win, lut_chosen[sx_win])
        f_static = count_logical_z_failures(err_z_win, STATIC_LUT_Z[sx_win])
        ler_gain = (f_static - f_chosen) / max(4.0, float(f_static))

        rew = compute_qec_action_reward(feat, chosen_act, opt_act, decoded_ler_gain=ler_gain)
        done = 1.0 if (ep == num_epochs - 1) else 0.0

        obs_list.append(feat)
        act_list.append(chosen_act)
        rew_list.append(rew)
        done_list.append(done)
        gain_list.append(targ_gains)
        opt_act_list.append(opt_act)
        dec_failures_list.append(f_chosen)
        static_failures_list.append(f_static)

    return {
        "obs": np.asarray(obs_list, dtype=np.float64),
        "actions": np.asarray(act_list, dtype=np.int64),
        "rewards": np.asarray(rew_list, dtype=np.float64),
        "dones": np.asarray(done_list, dtype=np.float64),
        "target_gains": np.asarray(gain_list, dtype=np.float64),
        "optimal_actions": np.asarray(opt_act_list, dtype=np.int64),
        "h_states": np.asarray(h_list, dtype=np.float64),
        "c_states": np.asarray(c_list, dtype=np.float64),
        "decoded_failures": np.asarray(dec_failures_list, dtype=np.int64),
        "static_failures": np.asarray(static_failures_list, dtype=np.int64),
    }


class R2D2Trainer:
    """Recurrent R2D2 Trainer with Adam optimizer, target network, and prioritized sequence replay."""

    def __init__(
        self,
        online_net: R2D2Network,
        learning_rate: float = 0.0025,
        gamma: float = 0.96,
        burn_in_steps: int = 4,
        unroll_steps: int = 4,
        n_step_td: int = 3,
        target_sync_interval: int = 25,
        grad_clip_norm: float = 5.0,
        replay_capacity: int = 4096,
        seed: int = 20260405,
    ) -> None:
        self.online_net = online_net
        self.target_mgr = TargetNetworkManager(
            online_net=online_net,
            sync_interval=target_sync_interval,
            tau=1.0,
        )
        self.replay = PrioritizedSequenceReplayBuffer(
            capacity=replay_capacity,
            alpha=0.6,
            eta=0.9,
            seed=seed,
        )
        self.lr = float(learning_rate)
        self.gamma = float(gamma)
        self.burn_in_steps = int(burn_in_steps)
        self.unroll_steps = int(unroll_steps)
        self.n_step_td = int(n_step_td)
        self.grad_clip_norm = float(grad_clip_norm)

        # Adam optimizer moments
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.m: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }
        self.v: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }

    def populate_replay_from_seeds(
        self,
        families: List[str],
        seed_start: int = 1000,
        seed_end: int = 1039,
        epsilon: float = 0.35,
    ) -> int:
        """Collect episodes across training seeds and slice into replay buffer."""
        added = 0
        for s in range(seed_start, seed_end + 1):
            for fam_idx, fam in enumerate(families):
                ep_seed = s * 31 + fam_idx * 7
                for eps_val in (epsilon, 0.05):
                    ep = generate_qec_control_episode(
                        family=fam,
                        seed=ep_seed + int(eps_val * 100),
                        num_epochs=16,
                        policy_net=self.online_net,
                        epsilon=eps_val,
                    )
                    segs = slice_episode_into_segments(
                        obs=ep["obs"],
                        actions=ep["actions"],
                        rewards=ep["rewards"],
                        dones=ep["dones"],
                        target_gains=ep["target_gains"],
                        optimal_actions=ep["optimal_actions"],
                        h_states=ep["h_states"],
                        c_states=ep["c_states"],
                        burn_in_steps=self.burn_in_steps,
                        unroll_steps=self.unroll_steps,
                        n_step_td=self.n_step_td,
                        stride=2,
                        family=fam,
                        seed=s,
                    )
                    for seg in segs:
                        self.replay.add(seg)
                        added += 1
        return added

    def _apply_adam_step(self, grads: Dict[str, np.ndarray]) -> float:
        """Clip global gradient norm and apply one Adam parameter update."""
        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        grad_norm = float(np.sqrt(max(total_sq, 0.0)))
        clip_scale = 1.0
        if grad_norm > self.grad_clip_norm:
            clip_scale = self.grad_clip_norm / (grad_norm + 1e-12)

        self.adam_t += 1
        b1_corr = 1.0 - (self.beta1 ** self.adam_t)
        b2_corr = 1.0 - (self.beta2 ** self.adam_t)

        params = self.online_net.parameters()
        for k, param in params.items():
            g = grads[k] * clip_scale
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[k] / b1_corr
            v_hat = self.v[k] / b2_corr
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)

        return grad_norm

    def train_step(self, batch_size: int = 16, beta: float = 0.6) -> Dict[str, float]:
        """Sample a prioritized batch of sequences, compute BPTT gradients, and update weights."""
        batch, indices, is_weights = self.replay.sample(batch_size=batch_size, beta=beta)
        accum_grads = {
            k: np.zeros_like(v) for k, v in self.online_net.parameters().items()
        }
        batch_loss = 0.0
        seq_td_errors: List[np.ndarray] = []

        for seg, w_i in zip(batch, is_weights):
            loss_i, td_err_i, grads_i = compute_n_step_transformed_td_loss(
                online_net=self.online_net,
                target_net=self.target_mgr.target_net,
                segment=seg,
                burn_in_steps=self.burn_in_steps,
                unroll_steps=self.unroll_steps,
                n_step_td=self.n_step_td,
                gamma=self.gamma,
                is_weight=float(w_i),
            )
            batch_loss += loss_i / len(batch)
            seq_td_errors.append(td_err_i)
            for k in accum_grads:
                accum_grads[k] += grads_i[k] / len(batch)

        self.replay.update_priorities(indices, seq_td_errors)
        grad_norm = self._apply_adam_step(accum_grads)
        synced = self.target_mgr.step()
        mean_abs_td = float(np.mean([np.mean(np.abs(e)) for e in seq_td_errors]))

        return {
            "loss": float(batch_loss),
            "mean_abs_td": mean_abs_td,
            "grad_norm": grad_norm,
            "target_synced": 1.0 if synced else 0.0,
        }
