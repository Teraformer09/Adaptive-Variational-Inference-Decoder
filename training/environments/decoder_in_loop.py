"""Hardware-Equivalent Q7.8 Normalized Min-Sum BP + Inline OSD-0 Decoder-in-the-Loop (`decoder_in_loop.py`)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from training.observations.spatial_mapping import CANONICAL_12_COUPLERS

CROSSTALK_PAIRS: List[Tuple[int, int]] = list(CANONICAL_12_COUPLERS)

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

X_CHECK_SUPPORTS: Tuple[Tuple[int, ...], ...] = (
    (0, 1),
    (1, 2, 4, 5),
    (3, 4, 6, 7),
    (7, 8),
)


def sample_d3_circuit_window(
    rng: np.random.Generator,
    theta_z_21: np.ndarray,
    theta_x_21: np.ndarray,
    window_rounds: int = 256,
    readout_flip_prob: float = 0.0,
    streak3_prob: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Simulate `window_rounds` shots of Z/X physical faults and X/Z stabilizer syndromes."""
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
    for k, (qa, qb) in enumerate(CROSSTALK_PAIRS):
        mask = np.uint16((1 << qa) | (1 << qb))
        err_z ^= np.where(uz_p[:, k], mask, np.uint16(0))
        err_x ^= np.where(ux_p[:, k], mask, np.uint16(0))

    if streak3_prob > 0.0:
        st_hit = rng.random(window_rounds) < float(streak3_prob)
        err_z ^= np.where(st_hit, np.uint16((1 << 3) | (1 << 4) | (1 << 5)), np.uint16(0))

    bx0 = ((err_z >> 0) ^ (err_z >> 1)) & 1
    bx1 = ((err_z >> 1) ^ (err_z >> 2) ^ (err_z >> 4) ^ (err_z >> 5)) & 1
    bx2 = ((err_z >> 3) ^ (err_z >> 4) ^ (err_z >> 6) ^ (err_z >> 7)) & 1
    bx3 = ((err_z >> 7) ^ (err_z >> 8)) & 1
    ideal_sx = (bx0 | (bx1 << 1) | (bx2 << 2) | (bx3 << 3)).astype(np.uint8)

    bz0 = ((err_x >> 0) ^ (err_x >> 1) ^ (err_x >> 3) ^ (err_x >> 4)) & 1
    bz1 = ((err_x >> 2) ^ (err_x >> 5)) & 1
    bz2 = ((err_x >> 3) ^ (err_x >> 6)) & 1
    bz3 = ((err_x >> 4) ^ (err_x >> 5) ^ (err_x >> 7) ^ (err_x >> 8)) & 1
    ideal_sz = (bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3)).astype(np.uint8)

    sx = ideal_sx.copy()
    sz = ideal_sz.copy()
    if readout_flip_prob > 0.0:
        fx = rng.random((window_rounds, 4)) < float(readout_flip_prob)
        fz = rng.random((window_rounds, 4)) < float(readout_flip_prob)
        for b in range(4):
            sx ^= fx[:, b].astype(np.uint8) << b
            sz ^= fz[:, b].astype(np.uint8) << b

    return {
        "err_z": err_z,
        "err_x": err_x,
        "sx": sx,
        "sz": sz,
        "ideal_sx": ideal_sx,
        "ideal_sz": ideal_sz,
    }


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

    p_eff = np.clip(np.asarray(qubit_p, dtype=np.float64).copy(), 1e-4, 0.45)
    alphas_arr = np.asarray(alphas, dtype=np.float64)
    if pair_weight_scale > 0.0:
        # Fold only excess coupler probability above isotropic baseline floor (0.008) so degree-4 center
        # vs degree-2 corner qubits are not artificially distorted under stationary uniform noise.
        for k, (u, v) in enumerate(CROSSTALK_PAIRS):
            ak_ex = float(np.clip(alphas_arr[k] - 0.008, 0.0, 0.45)) * min(1.0, pair_weight_scale)
            if ak_ex > 0.0:
                p_eff[u] = p_eff[u] * (1.0 - ak_ex) + (1.0 - p_eff[u]) * ak_ex
                p_eff[v] = p_eff[v] * (1.0 - ak_ex) + (1.0 - p_eff[v]) * ak_ex
    p_eff = np.clip(p_eff, 1e-4, 0.45)

    llr = np.log((1.0 - p_eff) / p_eff)
    if qubit_gains is not None:
        llr = llr * np.asarray(qubit_gains, dtype=np.float64)
    w_q78 = np.clip(np.round(llr * 256.0).astype(np.int32), 16, 0x3FFF)

    if pair_weight_scale <= 0.0:
        return w_q78

    for g in range(1, 16):
        best_k = -1
        best_score = -1.0
        for k, (u, v) in enumerate(CROSSTALK_PAIRS):
            if EVENT_SIGS_X[9 + k] == g and EVENT_SIGS_X[u] != EVENT_SIGS_X[v]:
                score = float(alphas_arr[k]) + 0.25 * float(qubit_p[u] + qubit_p[v])
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
        p_pair = float(alphas_arr[best_k])
        p_end_max = float(max(qubit_p[u], qubit_p[v]))
        p_base_prior = max(0.012, float(np.median(qubit_p)))

        gain_boost = 1.0
        if qubit_gains is not None:
            end_g_avg = max(0.5, 0.5 * float(qubit_gains[u] + qubit_gains[v]))
            single_g = float(qubit_gains[aliased[0]])
            gain_boost = float(np.clip(single_g / end_g_avg, 0.72, 1.42))

        end_support = (
            (0.35 * max(0.0, p_end_max - p_base_prior))
            if (p_pair > (1.15 * p_base_prior) / gain_boost)
            else 0.0
        )
        pair_ev = pair_weight_scale * gain_boost * (p_pair + end_support)
        if pair_ev > 1.85 * p_base_prior:
            p_eff_pair = float(np.clip(max(p_pair, 0.65 * p_end_max), 0.015, 0.38))
            eff_scale = 0.85 + 0.15 * pair_weight_scale + 0.85 * (gain_boost - 1.0)
            scale_inv = 1.0 / float(np.clip(eff_scale, 0.50, 2.20))
            pair_half = int(
                np.clip(
                    round(0.48 * scale_inv * np.log((1.0 - p_eff_pair) / p_eff_pair) * 256.0),
                    16,
                    0x0200,
                )
            )
            u_half = (
                int(np.clip(round(pair_half * float(qubit_gains[u])), 16, 0x0220))
                if qubit_gains is not None
                else pair_half
            )
            v_half = (
                int(np.clip(round(pair_half * float(qubit_gains[v])), 16, 0x0220))
                if qubit_gains is not None
                else pair_half
            )
            w_q78[u] = min(int(w_q78[u]), u_half)
            w_q78[v] = min(int(w_q78[v]), v_half)
            pair_sum = int(w_q78[u] + w_q78[v])
            for w in aliased:
                margin = (
                    int(np.clip(round(32.0 * float(qubit_gains[w])), 16, 96))
                    if qubit_gains is not None
                    else 32
                )
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


_H_X_BIN = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
    ],
    dtype=np.uint8,
)
_SYND_SIGNS_15X4 = np.array(
    [[-1.0 if ((s >> c) & 1) else 1.0 for c in range(4)] for s in range(1, 16)],
    dtype=np.float64,
)
_BIT_WEIGHTS_9 = (np.uint16(1) << np.arange(9, dtype=np.uint16))


def decode_window_minsum_osd0_z(
    sx: np.ndarray,
    w_q78: np.ndarray,
    bp_iterations: int = 3,
) -> np.ndarray:
    """Fast hardware-equivalent Mode-1 Q7.8 Min-Sum + Inline OSD-0 decoder LUT for `(w_q78, K)`.

    Evaluates the exact 16-state `bp_engine.sv` Mode-1 OSD-0 table in O(1) per window while
    modeling the K=3 short-cycle convergence fallback on weight-3/4 multi-check syndromes
    (`0x7`, `0xB`, `0xD`, `0xE`, `0xF`) when `bp_iterations <= 3` under near-tied LLRs.
    """
    k_max = max(1, min(8, int(bp_iterations)))
    lut = build_hardware_osd_lut_z(w_q78)
    if k_max <= 3:
        # At K=3, 4-cycle Tanner traps on multi-check syndromes fall back to static single/boundary cosets when LLRs are closely tied
        if abs(int(w_q78[4] + w_q78[5]) - int(w_q78[3])) < 96:
            lut[0x7] = STATIC_LUT_Z[0x7]
        if abs(int(w_q78[6] + w_q78[7]) - int(w_q78[8])) < 96:
            lut[0xB] = STATIC_LUT_Z[0xB]
            lut[0xE] = STATIC_LUT_Z[0xE]
    return lut[np.asarray(sx, dtype=np.uint8)]


def count_logical_z_failures(err_z: np.ndarray, corr_z: np.ndarray) -> int:
    """Count logical Z_L errors (`X_L = X_0 X_3 X_6` homology check, or unsatisfied syndrome)."""
    res = (err_z ^ corr_z) & np.uint16(0x1FF)
    bx0 = ((res >> 0) ^ (res >> 1)) & 1
    bx1 = ((res >> 1) ^ (res >> 2) ^ (res >> 4) ^ (res >> 5)) & 1
    bx2 = ((res >> 3) ^ (res >> 4) ^ (res >> 6) ^ (res >> 7)) & 1
    bx3 = ((res >> 7) ^ (res >> 8)) & 1
    unsat = (bx0 | bx1 | bx2 | bx3) != 0
    lz = (((res >> 0) ^ (res >> 3) ^ (res >> 6)) & np.uint16(1)) != 0
    return int(np.sum(unsat | lz))
