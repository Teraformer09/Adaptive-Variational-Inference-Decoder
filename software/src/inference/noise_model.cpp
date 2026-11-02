/* ===========================================================================
 * noise_model.cpp — Repaired Exact 16-State Joint DP Correlated Likelihood Engine
 * Policy Items: A01 / Q02, A02 / Q03, A03 / Q04, A08 / Q17
 * =========================================================================*/
#include "noise_model.h"
#include <cmath>
#include <algorithm>
#include <cstring>

/* Canonical rotated d=3 X-stabilizer checks (padded to 4 slots + active mask):
 *   c0 = {0, 1}       -> active_mask = 0x3
 *   c1 = {1, 2, 4, 5} -> active_mask = 0xF
 *   c2 = {3, 4, 6, 7} -> active_mask = 0xF
 *   c3 = {7, 8}       -> active_mask = 0x3
 */
const int NM_CHECK_CONN[NM_N_CHECK][4] = {
    {0, 1, 0, 0},
    {1, 2, 4, 5},
    {3, 4, 6, 7},
    {7, 8, 0, 0}
};

const uint8_t NM_CHECK_ACTIVE_MASK[NM_N_CHECK] = {
    0x03u, 0x0Fu, 0x0Fu, 0x03u
};

/* Nearest-neighbor qubit pairs (grid adjacency on 3x3 layout) */
const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK] = {
    {0, 1}, {1, 2},
    {3, 4}, {4, 5},
    {6, 7}, {7, 8},
    {0, 3}, {3, 6},
    {1, 4}, {4, 7},
    {2, 5}, {5, 8}
};

/* Exact 4-bit X-syndrome signatures g_e for the 21 fault events:
 * Single-qubit Z_0..Z_8:
 *   Z_0: {c0}       = 0x1
 *   Z_1: {c0, c1}   = 0x3
 *   Z_2: {c1}       = 0x2
 *   Z_3: {c2}       = 0x4
 *   Z_4: {c1, c2}   = 0x6
 *   Z_5: {c1}       = 0x2
 *   Z_6: {c2}       = 0x4
 *   Z_7: {c2, c3}   = 0xC
 *   Z_8: {c3}       = 0x8
 * Correlated pairs Z_i Z_j (g_ij = g_i ^ g_j):
 *   (0,1): 0x1 ^ 0x3 = 0x2
 *   (1,2): 0x3 ^ 0x2 = 0x1
 *   (3,4): 0x4 ^ 0x6 = 0x2
 *   (4,5): 0x6 ^ 0x2 = 0x4
 *   (6,7): 0x4 ^ 0xC = 0x8
 *   (7,8): 0xC ^ 0x8 = 0x4
 *   (0,3): 0x1 ^ 0x4 = 0x5
 *   (3,6): 0x4 ^ 0x4 = 0x0
 *   (1,4): 0x3 ^ 0x6 = 0x5
 *   (4,7): 0x6 ^ 0xC = 0xA
 *   (2,5): 0x2 ^ 0x2 = 0x0
 *   (5,8): 0x2 ^ 0x8 = 0xA
 */
const uint8_t NM_EVENT_SIGNATURES[NM_N_PARAMS] = {
    /* 9 single-qubit Z_0..Z_8 */
    0x1u, 0x3u, 0x2u, 0x4u, 0x6u, 0x2u, 0x4u, 0xCu, 0x8u,
    /* 12 ZZ pairs */
    0x2u, 0x1u, 0x2u, 0x4u, 0x8u, 0x4u,
    0x5u, 0x0u, 0x5u, 0xAu, 0x0u, 0xAu
};

NoiseModel::NoiseModel() : prior_lambda_(0.50) {
    init(0.012, 0.004);
}

void NoiseModel::init(double p0, double p_pair0) {
    for (int i = 0; i < NM_N_QUBIT; i++) {
        theta_(i) = p0;
        prior_theta_(i) = p0;
    }
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        theta_(NM_N_QUBIT + k) = p_pair0;
        prior_theta_(NM_N_QUBIT + k) = p_pair0;
    }
    grad_.setZero();
    compute_joint_distribution();
    compute_p_eff();
}

void NoiseModel::clamp_params() {
    for (int i = 0; i < NM_N_PARAMS; i++) {
        theta_(i) = std::clamp(theta_(i), 1e-5, 0.45);
    }
}

void NoiseModel::compute_joint_distribution() {
    p_joint_.setZero();
    p_joint_(0) = 1.0;

    double next_dist[NM_N_STATES];
    for (int e = 0; e < NM_N_PARAMS; e++) {
        const uint8_t g = NM_EVENT_SIGNATURES[e];
        const double pe = std::clamp(theta_(e), 1e-6, 0.49);
        if (g == 0u || pe == 0.0) {
            continue;
        }
        for (int s = 0; s < NM_N_STATES; s++) {
            next_dist[s] = (1.0 - pe) * p_joint_(s) + pe * p_joint_(s ^ g);
        }
        for (int s = 0; s < NM_N_STATES; s++) {
            p_joint_(s) = next_dist[s];
        }
    }
}

void NoiseModel::compute_p_eff() {
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double p_acc = std::clamp(theta_(v), 1e-6, 0.49);
        for (int k = 0; k < NM_N_CROSSTALK; k++) {
            if (NM_CROSSTALK_PAIRS[k].i == v || NM_CROSSTALK_PAIRS[k].j == v) {
                double pe = std::clamp(theta_(NM_N_QUBIT + k), 1e-6, 0.49);
                p_acc = p_acc * (1.0 - pe) + (1.0 - p_acc) * pe;
            }
        }
        p_eff_(v) = std::clamp(p_acc, 1e-5, 0.49);
    }
}

void NoiseModel::extract_histogram(const ring_buffer_t &rb, uint32_t hist_x[NM_N_STATES]) const {
    std::memset(hist_x, 0, sizeof(uint32_t) * NM_N_STATES);
    const int W = rb.count;
    for (int t = 0; t < W; t++) {
        const syndrome_round_t *sr = ring_at(&rb, t);
        uint8_t mask = 0;
        for (int c = 0; c < NM_N_CHECK; c++) {
            mask |= (uint8_t)((sr->x[c] & 1u) << c);
        }
        hist_x[mask & 0x0Fu]++;
    }
}

double NoiseModel::compute_nll_from_histogram(const uint32_t hist_x[NM_N_STATES]) {
    compute_joint_distribution();
    uint64_t total = 0;
    for (int s = 0; s < NM_N_STATES; s++) {
        total += hist_x[s];
    }
    if (total == 0) {
        return 0.0;
    }
    const double inv_total = 1.0 / (double)total;
    double nll = 0.0;
    for (int s = 0; s < NM_N_STATES; s++) {
        const double emp_p = (double)hist_x[s] * inv_total;
        const double mod_p = std::max(p_joint_(s), 1e-15);
        nll -= emp_p * std::log(mod_p);
    }
    if (prior_lambda_ > 0.0) {
        ParamVec diff = theta_ - prior_theta_;
        nll += 0.5 * prior_lambda_ * diff.squaredNorm();
    }
    return nll;
}

double NoiseModel::compute_nll(const ring_buffer_t &rb) {
    uint32_t hist_x[NM_N_STATES];
    extract_histogram(rb, hist_x);
    return compute_nll_from_histogram(hist_x);
}

void NoiseModel::compute_gradient_from_histogram(const uint32_t hist_x[NM_N_STATES]) {
    compute_joint_distribution();
    grad_.setZero();

    uint64_t total = 0;
    for (int s = 0; s < NM_N_STATES; s++) {
        total += hist_x[s];
    }
    if (total == 0) {
        return;
    }
    const double inv_total = 1.0 / (double)total;

    double weights[NM_N_STATES];
    for (int s = 0; s < NM_N_STATES; s++) {
        const double emp_p = (double)hist_x[s] * inv_total;
        const double mod_p = std::max(p_joint_(s), 1e-15);
        weights[s] = -(emp_p / mod_p);
    }

    for (int e = 0; e < NM_N_PARAMS; e++) {
        const uint8_t g = NM_EVENT_SIGNATURES[e];
        if (g == 0u) {
            grad_(e) = 0.0;
            continue;
        }
        const double pe = std::clamp(theta_(e), 1e-6, 0.49);
        const double denom = 1.0 - 2.0 * pe;
        double ge_acc = 0.0;
        for (int s = 0; s < NM_N_STATES; s++) {
            const double dp_s = (p_joint_(s ^ g) - p_joint_(s)) / denom;
            ge_acc += weights[s] * dp_s;
        }
        grad_(e) = ge_acc;
    }

    if (prior_lambda_ > 0.0) {
        grad_ += prior_lambda_ * (theta_ - prior_theta_);
    }
}

void NoiseModel::compute_gradient(const ring_buffer_t &rb) {
    uint32_t hist_x[NM_N_STATES];
    extract_histogram(rb, hist_x);
    compute_gradient_from_histogram(hist_x);
}

double NoiseModel::effective_p(int v) const {
    double p_acc = std::clamp(theta_(v), 1e-6, 0.49);
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        if (NM_CROSSTALK_PAIRS[k].i == v || NM_CROSSTALK_PAIRS[k].j == v) {
            double pe = std::clamp(theta_(NM_N_QUBIT + k), 1e-6, 0.49);
            p_acc = p_acc * (1.0 - pe) + (1.0 - p_acc) * pe;
        }
    }
    return std::clamp(p_acc, 1e-5, 0.49);
}

void NoiseModel::to_llr(float llr_x[NM_N_QUBIT], float llr_z[NM_N_QUBIT]) const {
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double p = effective_p(v);
        double llr = std::log((1.0 - p) / p);
        llr = std::clamp(llr, -120.0, 120.0);
        llr_x[v] = (float)llr;
        llr_z[v] = (float)llr;
    }
}

void NoiseModel::to_q7_8_llr(int16_t llr_q78[NM_N_QUBIT]) const {
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double p = effective_p(v);
        double llr = std::log((1.0 - p) / p);
        long scaled = std::lround(llr * 256.0);
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;
        llr_q78[v] = (int16_t)scaled;
    }
}
