/* ===========================================================================
 * noise_model.h — Repaired Exact 2^m-State Joint DP Correlated Likelihood Engine
 * Policy Items: A01 / Q02, A02 / Q03, A03 / Q04, A08 / Q17
 *
 * Canonical Rotated d=3 Planar Surface Code X-Stabilizers:
 *   c0 = {0, 1}          (weight-2 top boundary,   active_mask = 0x3)
 *   c1 = {1, 2, 4, 5}    (weight-4 upper-right,    active_mask = 0xF)
 *   c2 = {3, 4, 6, 7}    (weight-4 lower-left,     active_mask = 0xF)
 *   c3 = {7, 8}          (weight-2 bottom boundary,active_mask = 0x3)
 *
 * Exact 16-State Dynamic Programming Joint Syndrome Distribution:
 *   P^(k)(s) = (1 - p_e) P^(k-1)(s) + p_e P^(k-1)(s ^ g_e),  s in 0..15
 *   where single-qubit Z_i has signature g_i = H_X(:, i) and correlated pair
 *   Z_i Z_j has XOR signature g_ij = g_i ^ g_j.
 *
 * Exact Closed-Form Coset-Inversion Analytic Gradient:
 *   dP(s)/dp_e = (P(s ^ g_e) - P(s)) / (1 - 2 p_e)   for g_e != 0.
 * =========================================================================*/
#pragma once

#include <Eigen/Dense>
#include <stdint.h>
#include "ring_buffer.h"

#define NM_N_QUBIT     9    /* Single-qubit Z error rates */
#define NM_N_CROSSTALK 12   /* Nearest-neighbor ZZ pair error rates */
#define NM_N_PARAMS    (NM_N_QUBIT + NM_N_CROSSTALK)  /* 21 total */
#define NM_N_CHECK     4    /* Canonical d=3 X-stabilizer checks */
#define NM_N_STATES    16   /* 2^4 = 16 joint syndrome states */

struct CrosstalkPair { int i, j; };
extern const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK];

/* Padded 4-slot check connectivity and active-edge bitmasks */
extern const int     NM_CHECK_CONN[NM_N_CHECK][4];
extern const uint8_t NM_CHECK_ACTIVE_MASK[NM_N_CHECK];

/* Syndrome bitmask signatures g_e in 0..15 for all 21 fault mechanisms */
extern const uint8_t NM_EVENT_SIGNATURES[NM_N_PARAMS];

class NoiseModel {
public:
    using ParamVec = Eigen::Matrix<double, NM_N_PARAMS, 1>;
    using GradVec  = Eigen::Matrix<double, NM_N_PARAMS, 1>;
    using StateVec = Eigen::Matrix<double, NM_N_STATES, 1>;

    NoiseModel();

    /** Initialise parameters to single-qubit p_i = p0, pair p_ij = p_pair0 */
    void init(double p0 = 0.012, double p_pair0 = 0.004);

    /**
     * compute_joint_distribution() — Exact 16-state XOR-convolution DP.
     * Populates p_joint_[0..15] in 21 * 16 = 336 multiply-add operations.
     */
    void compute_joint_distribution();

    /**
     * compute_nll() — Exact prior-tethered Joint-DP Negative Log-Likelihood
     * over the syndrome ring buffer window.
     */
    double compute_nll(const ring_buffer_t &rb);

    /**
     * compute_nll_from_histogram() — Exact Joint-DP NLL directly from a 16-bin
     * syndrome histogram (zero per-round loop overhead).
     */
    double compute_nll_from_histogram(const uint32_t hist_x[NM_N_STATES]);

    /**
     * compute_gradient() — Exact analytical gradient of Joint-DP NLL w.r.t. θ.
     */
    void compute_gradient(const ring_buffer_t &rb);

    /**
     * compute_gradient_from_histogram() — Exact analytical gradient from 16-bin histogram.
     */
    void compute_gradient_from_histogram(const uint32_t hist_x[NM_N_STATES]);

    const ParamVec &params() const { return theta_; }
    const GradVec  &grad()   const { return grad_;  }
    const StateVec &joint_probs() const { return p_joint_; }

    /**
     * effective_p(qubit_idx) — Exact XOR-marginalized single-qubit error probability
     * p_eff(v) = p_v XOR_{e=(u,v)} p_e for initializing BP variable-node priors.
     */
    double effective_p(int v) const;

    /** Convert effective error probabilities to floating-point LLRs */
    void to_llr(float llr_x[NM_N_QUBIT], float llr_z[NM_N_QUBIT]) const;

    /** Convert effective error probabilities to signed 16-bit Q7.8 fixed-point LLRs */
    void to_q7_8_llr(int16_t llr_q78[NM_N_QUBIT]) const;

    /** Clamp parameters to valid interior range [1e-5, 0.45] */
    void clamp_params();

    ParamVec theta_;
    ParamVec prior_theta_;
    double   prior_lambda_;

private:
    GradVec  grad_;
    StateVec p_joint_;
    Eigen::Matrix<double, NM_N_QUBIT, 1> p_eff_;

    void compute_p_eff();
    void extract_histogram(const ring_buffer_t &rb, uint32_t hist_x[NM_N_STATES]) const;
};
