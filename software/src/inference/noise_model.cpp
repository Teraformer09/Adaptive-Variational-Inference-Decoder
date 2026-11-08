/* ===========================================================================
 * noise_model.cpp — Repaired Exact 16-State Joint DP Correlated Likelihood Engine
 * Policy Items: A01 / Q02, A02 / Q03, A03 / Q04, A08 / Q17
 * =========================================================================*/
#include "noise_model.h"
#include <cmath>
#include <algorithm>
#include <cstring>

const int NM_CHECK_CONN[NM_N_CHECK][4] = {
    {0, 1, 0, 0},
    {1, 2, 4, 5},
    {3, 4, 6, 7},
    {7, 8, 0, 0}
};

const uint8_t NM_CHECK_ACTIVE_MASK[NM_N_CHECK] = {
    0x03u, 0x0Fu, 0x0Fu, 0x03u
};

/* Nearest-neighbor qubit pairs (frozen 21-parameter ordering 9..20) */
const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK] = {
    {0, 1}, {1, 2},
    {3, 4}, {4, 5},
    {6, 7}, {7, 8},
    {0, 3}, {3, 6},
    {1, 4}, {4, 7},
    {2, 5}, {5, 8}
};

/* Exact 4-bit X-syndrome signatures g_e for the 21 Z-fault events */
const uint8_t NM_EVENT_SIGNATURES[NM_N_PARAMS] = {
    /* 9 single-qubit Z_0..Z_8 */
    0x1u, 0x3u, 0x2u, 0x4u, 0x6u, 0x2u, 0x4u, 0xCu, 0x8u,
    /* 12 ZZ pairs */
    0x2u, 0x1u, 0x2u, 0x4u, 0x8u, 0x4u,
    0x5u, 0x0u, 0x5u, 0xAu, 0x0u, 0xAu
};

/* Exact 4-bit Z-syndrome signatures g_e^(Z) for the 21 X-fault events on H_Z */
const uint8_t NM_EVENT_SIGNATURES_Z_BASIS[NM_N_PARAMS] = {
    /* 9 single-qubit X_0..X_8 on z0={0,1,3,4}, z1={2,5}, z2={3,6}, z3={4,5,7,8} */
    0x1u, 0x1u, 0x2u, 0x5u, 0x9u, 0xAu, 0x4u, 0x8u, 0x8u,
    /* 12 XX pairs (g_ij = g_i ^ g_j) */
    0x0u, 0x3u, 0xCu, 0x3u, 0xCu, 0x0u,
    0x4u, 0x1u, 0x8u, 0x1u, 0x8u, 0x2u
};

/* The 9 non-zero observable syndrome-signature orbit codes in X-check and Z-check bases */
const uint8_t NM_OBSERVABLE_ORBIT_CODES_X[NM_N_ORBITS] = {
    0x1u, 0x2u, 0x3u, 0x4u, 0x5u, 0x6u, 0x8u, 0xAu, 0xCu
};

const uint8_t NM_OBSERVABLE_ORBIT_CODES_Z[NM_N_ORBITS] = {
    0x1u, 0x2u, 0x3u, 0x4u, 0x5u, 0x8u, 0x9u, 0xAu, 0xCu
};

NoiseModel::NoiseModel() : prior_lambda_(0.05), basis_z_(false) {
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

bool NoiseModel::validate_parameter_bounds(const ParamVec &theta) {
    for (int i = 0; i < NM_N_PARAMS; i++) {
        if (!std::isfinite(theta(i)) || theta(i) < 0.0 || theta(i) >= 0.5) {
            return false;
        }
    }
    return true;
}

void NoiseModel::compute_joint_distribution() {
    p_joint_.setZero();
    p_joint_(0) = 1.0;

    double next_dist[NM_N_STATES];
    for (int e = 0; e < NM_N_PARAMS; e++) {
        const uint8_t g = event_signature(e);
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

void NoiseModel::extract_histogram(const ring_buffer_t &rb, uint32_t hist[NM_N_STATES]) const {
    std::memset(hist, 0, sizeof(uint32_t) * NM_N_STATES);
    const int W = rb.count;
    for (int t = 0; t < W; t++) {
        const syndrome_round_t *sr = ring_at(&rb, t);
        uint8_t mask = 0;
        for (int c = 0; c < NM_N_CHECK; c++) {
            uint8_t bit = basis_z_ ? (sr->z[c] & 1u) : (sr->x[c] & 1u);
            mask |= (uint8_t)(bit << c);
        }
        hist[mask & 0x0Fu]++;
    }
}

double NoiseModel::compute_nll_from_histogram(const uint32_t hist[NM_N_STATES]) {
    compute_joint_distribution();
    uint64_t total = 0;
    for (int s = 0; s < NM_N_STATES; s++) {
        total += hist[s];
    }
    if (total == 0) {
        return 0.0;
    }
    const double inv_total = 1.0 / (double)total;
    double nll = 0.0;
    for (int s = 0; s < NM_N_STATES; s++) {
        const double emp_p = (double)hist[s] * inv_total;
        const double mod_p = std::max(p_joint_(s), 1e-15);
        nll -= emp_p * std::log(mod_p);
    }
    if (prior_lambda_ > 0.0) {
        ParamVec diff = theta_ - prior_theta_;
        nll += 0.5 * prior_lambda_ * diff.squaredNorm();
    }
    return nll;
}

double NoiseModel::compute_legacy_marginal_nll_from_histogram(const uint32_t hist[NM_N_STATES]) const {
    uint64_t total = 0;
    uint64_t check_ones[NM_N_CHECK] = {0};
    for (int s = 0; s < NM_N_STATES; s++) {
        total += hist[s];
        for (int c = 0; c < NM_N_CHECK; c++) {
            if ((s >> c) & 1) check_ones[c] += hist[s];
        }
    }
    if (total == 0) return 0.0;
    const double inv_total = 1.0 / (double)total;

    /* Compute 4 independent check marginal flip probabilities from theta_ */
    double q_check[NM_N_CHECK] = {0.0};
    for (int c = 0; c < NM_N_CHECK; c++) {
        double prod = 1.0;
        for (int e = 0; e < NM_N_PARAMS; e++) {
            uint8_t g = event_signature(e);
            if ((g >> c) & 1u) {
                double pe = std::clamp(theta_(e), 1e-6, 0.49);
                prod *= (1.0 - 2.0 * pe);
            }
        }
        q_check[c] = std::clamp(0.5 * (1.0 - prod), 1e-9, 1.0 - 1e-9);
    }

    double marginal_nll = 0.0;
    for (int c = 0; c < NM_N_CHECK; c++) {
        double emp_c = (double)check_ones[c] * inv_total;
        marginal_nll -= emp_c * std::log(q_check[c]) + (1.0 - emp_c) * std::log(1.0 - q_check[c]);
    }
    return marginal_nll;
}

double NoiseModel::compute_nll(const ring_buffer_t &rb) {
    uint32_t hist[NM_N_STATES];
    extract_histogram(rb, hist);
    return compute_nll_from_histogram(hist);
}

void NoiseModel::compute_gradient_from_histogram(const uint32_t hist[NM_N_STATES]) {
    compute_joint_distribution();
    grad_.setZero();

    uint64_t total = 0;
    for (int s = 0; s < NM_N_STATES; s++) {
        total += hist[s];
    }
    if (total == 0) {
        return;
    }
    const double inv_total = 1.0 / (double)total;

    double weights[NM_N_STATES];
    for (int s = 0; s < NM_N_STATES; s++) {
        const double emp_p = (double)hist[s] * inv_total;
        const double mod_p = std::max(p_joint_(s), 1e-15);
        weights[s] = -(emp_p / mod_p);
    }

    for (int e = 0; e < NM_N_PARAMS; e++) {
        const uint8_t g = event_signature(e);
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
    uint32_t hist[NM_N_STATES];
    extract_histogram(rb, hist);
    compute_gradient_from_histogram(hist);
}

void NoiseModel::compute_orbit_probabilities(const ParamVec &theta, bool basis_z, double orbit_probs[NM_N_ORBITS]) {
    const uint8_t *codes = basis_z ? NM_OBSERVABLE_ORBIT_CODES_Z : NM_OBSERVABLE_ORBIT_CODES_X;
    const uint8_t *sigs  = basis_z ? NM_EVENT_SIGNATURES_Z_BASIS  : NM_EVENT_SIGNATURES;
    for (int k = 0; k < NM_N_ORBITS; k++) {
        uint8_t code = codes[k];
        double prod = 1.0;
        for (int e = 0; e < NM_N_PARAMS; e++) {
            if (sigs[e] == code) {
                double pe = std::clamp(theta(e), 1e-6, 0.49);
                prod *= (1.0 - 2.0 * pe);
            }
        }
        orbit_probs[k] = 0.5 * (1.0 - prod);
    }
}

double NoiseModel::compute_orbit_rmse(const ParamVec &true_theta) const {
    double est_orb[NM_N_ORBITS], tru_orb[NM_N_ORBITS];
    compute_orbit_probabilities(theta_, basis_z_, est_orb);
    compute_orbit_probabilities(true_theta, basis_z_, tru_orb);
    double sq_sum = 0.0;
    for (int k = 0; k < NM_N_ORBITS; k++) {
        double d = est_orb[k] - tru_orb[k];
        sq_sum += d * d;
    }
    return std::sqrt(sq_sum / (double)NM_N_ORBITS);
}

double NoiseModel::compute_param_21_rmse(const ParamVec &true_theta) const {
    ParamVec d = theta_ - true_theta;
    return std::sqrt(d.squaredNorm() / (double)NM_N_PARAMS);
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

void NoiseModel::to_pair_projected_q7_8_llr(int16_t llr_q78[NM_N_QUBIT],
                                            const double *qubit_gains,
                                            double pair_weight_scale) const {
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double p = effective_p(v);
        double llr = std::log((1.0 - p) / p);
        if (qubit_gains) {
            llr *= qubit_gains[v];
        }
        long scaled = std::lround(llr * 256.0);
        llr_q78[v] = (int16_t)std::clamp(scaled, 16L, 0x3FFFL);
    }

    if (pair_weight_scale <= 0.0) {
        return;
    }

    /* General 12-Pair Syndrome-Alias MAP Log-Odds Projection:
     * Iterates over every non-zero syndrome signature g in 1..15 and identifies
     * all nearest-neighbor pairs k in 0..11 whose syndrome signature g_{u,v} == g
     * aliases with one or more single-qubit signatures g_w == g.
     * Zero hardcoded qubit or pair indices; works identically across all 6 horizontal
     * Z-error pairs and all 6 vertical X-error pairs. */
    for (uint8_t g = 1u; g < NM_N_STATES; g++) {
        int best_pair_k = -1;
        double best_pair_score = -1.0;

        for (int k = 0; k < NM_N_CROSSTALK; k++) {
            uint8_t g_pair = event_signature(NM_N_QUBIT + k);
            int u = NM_CROSSTALK_PAIRS[k].i;
            int v = NM_CROSSTALK_PAIRS[k].j;
            if (g_pair == g && event_signature(u) != event_signature(v)) {
                double score = theta_(NM_N_QUBIT + k) + 0.5 * (theta_(u) + theta_(v));
                if (score > best_pair_score) {
                    best_pair_score = score;
                    best_pair_k = k;
                }
            }
        }
        if (best_pair_k < 0) {
            continue;
        }

        int aliased_singles[NM_N_QUBIT];
        int n_aliased = 0;
        for (int w = 0; w < NM_N_QUBIT; w++) {
            if (event_signature(w) == g) {
                aliased_singles[n_aliased++] = w;
            }
        }
        if (n_aliased == 0) {
            continue;
        }

        int u = NM_CROSSTALK_PAIRS[best_pair_k].i;
        int v = NM_CROSSTALK_PAIRS[best_pair_k].j;
        double p_pair       = theta_(NM_N_QUBIT + best_pair_k);
        double p_end_max    = std::max(theta_(u), theta_(v));
        double p_base_prior = prior_theta_(aliased_singles[0]);

        /* Bayesian syndrome-alias dominance condition modulated by pair_weight_scale */
        double pair_evidence = pair_weight_scale * (p_pair + 0.5 * std::max(0.0, p_end_max - p_base_prior));
        if (pair_evidence > p_base_prior + 0.008) {
            double p_eff_pair = std::clamp(std::max(p_pair, 0.65 * p_end_max), 0.015, 0.38);
            double scale_inv  = 1.0 / std::clamp(pair_weight_scale, 0.5, 2.0);
            int16_t pair_half_q78 = (int16_t)std::clamp(
                (long)std::lround(0.45 * scale_inv * std::log((1.0 - p_eff_pair) / p_eff_pair) * 256.0),
                16L, 0x0200L);
            llr_q78[u] = std::min(llr_q78[u], pair_half_q78);
            llr_q78[v] = std::min(llr_q78[v], pair_half_q78);

            int16_t pair_sum = (int16_t)(llr_q78[u] + llr_q78[v]);
            for (int idx = 0; idx < n_aliased; idx++) {
                int w = aliased_singles[idx];
                llr_q78[w] = std::max<int16_t>(llr_q78[w], (int16_t)(pair_sum + 0x0020));
            }
        }
    }
}
