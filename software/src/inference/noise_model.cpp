/* ===========================================================================
 * noise_model.cpp — AVID Phase 3.2
 * HMM variational inference engine implementation.
 * See noise_model.h for full mathematical derivation.
 * =========================================================================*/
#include "noise_model.h"
#include <cmath>
#include <algorithm>

/* ==========================================================================
 * Tanner graph / crosstalk topology for d=3 surface code
 * ========================================================================== */

/* X-type check connectivity (each row = 4 adjacent qubit indices) */
const int NM_CHECK_CONN[NM_N_CHECK][4] = {
    {0, 1, 3, 4},   /* c0 */
    {1, 2, 4, 5},   /* c1 */
    {3, 4, 6, 7},   /* c2 */
    {4, 5, 7, 8}    /* c3 */
};

/* Nearest-neighbor qubit pairs (grid adjacency on 3×3 layout) */
const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK] = {
    {0,1},{1,2},   /* top row    */
    {3,4},{4,5},   /* middle row */
    {6,7},{7,8},   /* bottom row */
    {0,3},{3,6},   /* left col   */
    {1,4},{4,7},   /* center col */
    {2,5},{5,8}    /* right col  */
};

/* Helper: crosstalk parameter index for pair (i,j) — returns -1 if not adjacent */
static int crosstalk_idx(int i, int j) {
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        if ((NM_CROSSTALK_PAIRS[k].i == i && NM_CROSSTALK_PAIRS[k].j == j) ||
            (NM_CROSSTALK_PAIRS[k].i == j && NM_CROSSTALK_PAIRS[k].j == i))
            return k;
    }
    return -1;
}

/* ==========================================================================
 * NoiseModel implementation
 * ========================================================================== */

NoiseModel::NoiseModel() {
    theta_.setZero();
    grad_.setZero();
    q_c_.setZero();
    p_eff_.setZero();
}

void NoiseModel::init(double p0) {
    for (int i = 0; i < NM_N_QUBIT; i++)
        theta_(i) = p0;
    for (int k = 0; k < NM_N_CROSSTALK; k++)
        theta_(NM_N_QUBIT + k) = 0.0;
}

void NoiseModel::clamp_params() {
    for (int i = 0; i < NM_N_PARAMS; i++)
        theta_(i) = std::clamp(theta_(i), 1e-4, 0.499);
}

/* -------------------------------------------------------------------------
 * compute_p_eff() — effective single-qubit rate including crosstalk
 * p_v_eff = p_v + Σ_{j ∈ NB(v)} p_vj - 2 Σ_j p_v · p_vj
 * -------------------------------------------------------------------------*/
void NoiseModel::compute_p_eff() {
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double pv = theta_(v);
        double correction = 0.0;
        for (int k = 0; k < NM_N_CROSSTALK; k++) {
            if (NM_CROSSTALK_PAIRS[k].i == v || NM_CROSSTALK_PAIRS[k].j == v) {
                double p_cross = theta_(NM_N_QUBIT + k);
                correction += p_cross - 2.0 * pv * p_cross;
            }
        }
        p_eff_(v) = std::clamp(pv + correction, 1e-4, 0.499);
    }
}

/* -------------------------------------------------------------------------
 * compute_check_probs() — exact parity formula (no approximation)
 * P(s_c=1|θ) = [1 - ∏_{v∈N(c)} (1 - 2·p_v_eff)] / 2
 * -------------------------------------------------------------------------*/
void NoiseModel::compute_check_probs() {
    for (int c = 0; c < NM_N_CHECK; c++) {
        double prod = 1.0;
        for (int j = 0; j < 4; j++) {
            int v = NM_CHECK_CONN[c][j];
            prod *= (1.0 - 2.0 * p_eff_(v));
        }
        q_c_(c) = (1.0 - prod) * 0.5;
        q_c_(c) = std::clamp(q_c_(c), 1e-9, 1.0 - 1e-9);
    }
}

/* -------------------------------------------------------------------------
 * compute_nll() — Negative Log-Likelihood over syndrome window W
 * L(θ) = -Σ_{t,c} [ s_c^t log q_c + (1-s_c^t) log(1-q_c) ]
 * -------------------------------------------------------------------------*/
double NoiseModel::compute_nll(const ring_buffer_t &rb) {
    compute_p_eff();
    compute_check_probs();

    double nll = 0.0;
    const int W = rb.count;
    for (int t = 0; t < W; t++) {
        const syndrome_round_t *s = ring_at(&rb, t);
        for (int c = 0; c < NM_N_CHECK; c++) {
            double q = q_c_(c);
            if (s->x[c])
                nll -= std::log(q);
            else
                nll -= std::log(1.0 - q);
        }
    }
    return nll;
}

/* -------------------------------------------------------------------------
 * compute_gradient() — Analytical gradient of NLL
 *
 * ∂L/∂p_i = -Σ_{t,c: i∈N(c)} [ (s_c^t - q_c)/(q_c(1-q_c)) ] · ∂q_c/∂p_i
 *
 * ∂q_c/∂p_i = ∏_{v∈N(c), v≠i}(1-2p_v_eff) · ∂p_i_eff/∂p_i
 * ∂p_i_eff/∂p_i = 1 - 2·Σ_{j ∈ NB(i)} p_ij
 *
 * ∂L/∂p_ij = -Σ_{t} Σ_{c: i∈N(c) or j∈N(c)} [ (s_c^t - q_c)/(q_c(1-q_c)) ]
 *              · ∂q_c/∂p_ij
 * ∂q_c/∂p_ij = ∂q_c/∂p_i_eff · (1 - 2p_i) + ∂q_c/∂p_j_eff · (1 - 2p_j)
 * -------------------------------------------------------------------------*/
void NoiseModel::compute_gradient(const ring_buffer_t &rb) {
    compute_p_eff();
    compute_check_probs();
    grad_.setZero();

    const int W = rb.count;
    if (W == 0) return;

    /* Pre-compute ∂p_i_eff/∂p_i = 1 - 2·Σ_j p_ij */
    double dp_eff_dpi[NM_N_QUBIT];
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double s = 0.0;
        for (int k = 0; k < NM_N_CROSSTALK; k++)
            if (NM_CROSSTALK_PAIRS[k].i == v || NM_CROSSTALK_PAIRS[k].j == v)
                s += theta_(NM_N_QUBIT + k);
        dp_eff_dpi[v] = 1.0 - 2.0 * s;
    }

    /* Accumulate gradient over syndrome window */
    for (int t = 0; t < W; t++) {
        const syndrome_round_t *s = ring_at(&rb, t);
        for (int c = 0; c < NM_N_CHECK; c++) {
            double q    = q_c_(c);
            double denom = q * (1.0 - q);
            /* score residual: (s_c - q_c) */
            double resid = (double)s->x[c] - q;

            /* ∂q_c/∂p_i_eff = ∏_{v≠i}(1-2p_eff_v)  (partial product) */
            /* Compute full product and divide out each qubit's contribution */
            double full_prod = 1.0;
            for (int j = 0; j < 4; j++)
                full_prod *= (1.0 - 2.0 * p_eff_(NM_CHECK_CONN[c][j]));

            for (int j = 0; j < 4; j++) {
                int vi = NM_CHECK_CONN[c][j];
                double factor_vi = 1.0 - 2.0 * p_eff_(vi);
                double dqc_dpeff = (std::abs(factor_vi) > 1e-12)
                                   ? (full_prod / factor_vi) : 0.0;
                /* Gradient contribution to p_vi */
                grad_(vi) -= (resid / denom) * dqc_dpeff * dp_eff_dpi[vi];

                /* Gradient contribution to crosstalk parameters sharing vi */
                for (int k = 0; k < NM_N_CROSSTALK; k++) {
                    int a = NM_CROSSTALK_PAIRS[k].i;
                    int b = NM_CROSSTALK_PAIRS[k].j;
                    if (a != vi && b != vi) continue;
                    int other = (a == vi) ? b : a;
                    /* ∂p_vi_eff/∂p_vij = 1 - 2*p_vi */
                    double dpeff_cross = (1.0 - 2.0 * theta_(vi));
                    grad_(NM_N_QUBIT + k) -=
                        (resid / denom) * dqc_dpeff * dpeff_cross;
                    (void)other;
                }
            }
        }
    }

    /* Average over window */
    grad_ /= (double)W;
}

/* -------------------------------------------------------------------------
 * effective_p() — single query of p_v_eff
 * -------------------------------------------------------------------------*/
double NoiseModel::effective_p(int v) const {
    double pv = theta_(v);
    double correction = 0.0;
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        if (NM_CROSSTALK_PAIRS[k].i == v || NM_CROSSTALK_PAIRS[k].j == v) {
            double pc = theta_(NM_N_QUBIT + k);
            correction += pc - 2.0 * pv * pc;
        }
    }
    return std::clamp(pv + correction, 1e-4, 0.499);
}

/* -------------------------------------------------------------------------
 * to_llr() — convert effective error rates to Q7.8 LLRs for FPGA
 * LLR_v = log((1-p_v_eff)/p_v_eff)
 * -------------------------------------------------------------------------*/
void NoiseModel::to_llr(float llr_x[NM_N_QUBIT], float llr_z[NM_N_QUBIT]) const {
    /* For simplicity d=3 uses the same p_eff model for both X and Z errors.
     * In a more complete model, X-error params (detected by Z-stabilizers)
     * and Z-error params (detected by X-stabilizers) are fitted separately. */
    for (int v = 0; v < NM_N_QUBIT; v++) {
        double p = const_cast<NoiseModel*>(this)->effective_p(v);
        double llr = std::log((1.0 - p) / p);
        llr = std::clamp(llr, -100.0, 100.0);
        llr_x[v] = (float)llr;
        llr_z[v] = (float)llr;   /* Symmetric model; extend for anisotropic noise */
    }
}
