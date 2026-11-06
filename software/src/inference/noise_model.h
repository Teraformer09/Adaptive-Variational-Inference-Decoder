/* ===========================================================================
 * noise_model.h — Repaired Exact 2^m-State Joint DP Correlated Likelihood Engine
 * Policy Items: A01 / Q02, A02 / Q03, A03 / Q04, A08 / Q17
 *
 * Canonical Rotated d=3 Planar Surface Code Stabilizers:
 *   - X-Stabilizers (detect Z-errors, basis_z = false):
 *       c0 = {0, 1}, c1 = {1, 2, 4, 5}, c2 = {3, 4, 6, 7}, c3 = {7, 8}
 *   - Z-Stabilizers (detect X-errors, basis_z = true):
 *       z0 = {0, 1, 3, 4}, z1 = {2, 5}, z2 = {3, 6}, z3 = {4, 5, 7, 8}
 *
 * Frozen 21-Parameter Ordering (theta_[0..20]):
 *   - theta_[0..8]  : 9 single-qubit error probabilities p_0 .. p_8
 *   - theta_[9..20] : 12 nearest-neighbor two-qubit correlated error probabilities:
 *       9:(0,1), 10:(1,2), 11:(3,4), 12:(4,5), 13:(6,7), 14:(7,8),
 *      15:(0,3), 16:(3,6), 17:(1,4), 18:(4,7), 19:(2,5), 20:(5,8)
 * =========================================================================*/
#pragma once

#include <Eigen/Dense>
#include <stdint.h>
#include "ring_buffer.h"

#define NM_N_QUBIT     9    /* Single-qubit error rates */
#define NM_N_CROSSTALK 12   /* Nearest-neighbor pair error rates */
#define NM_N_PARAMS    (NM_N_QUBIT + NM_N_CROSSTALK)  /* 21 total */
#define NM_N_CHECK     4    /* Canonical d=3 stabilizer checks */
#define NM_N_STATES    16   /* 2^4 = 16 joint syndrome states */
#define NM_N_ORBITS    9    /* 9 non-zero observable syndrome-signature orbits */

struct CrosstalkPair { int i, j; };
extern const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK];

/* Padded 4-slot check connectivity and active-edge bitmasks */
extern const int     NM_CHECK_CONN[NM_N_CHECK][4];
extern const uint8_t NM_CHECK_ACTIVE_MASK[NM_N_CHECK];

/* Syndrome bitmask signatures g_e in 0..15 for all 21 fault mechanisms */
extern const uint8_t NM_EVENT_SIGNATURES[NM_N_PARAMS];
extern const uint8_t NM_EVENT_SIGNATURES_Z_BASIS[NM_N_PARAMS];
extern const uint8_t NM_OBSERVABLE_ORBIT_CODES_X[NM_N_ORBITS];
extern const uint8_t NM_OBSERVABLE_ORBIT_CODES_Z[NM_N_ORBITS];

class NoiseModel {
public:
    using ParamVec = Eigen::Matrix<double, NM_N_PARAMS, 1>;
    using GradVec  = Eigen::Matrix<double, NM_N_PARAMS, 1>;
    using StateVec = Eigen::Matrix<double, NM_N_STATES, 1>;

    NoiseModel();

    /** Initialise parameters to single-qubit p_i = p0, pair p_ij = p_pair0 */
    void init(double p0 = 0.012, double p_pair0 = 0.004);

    /** Select X-check basis (false, default: decodes Z errors) or Z-check basis (true: decodes X errors) */
    void set_basis_z(bool basis_z) { basis_z_ = basis_z; }
    bool is_basis_z() const { return basis_z_; }

    /** Return syndrome signature g_e for event e in active basis */
    uint8_t event_signature(int e) const {
        return basis_z_ ? NM_EVENT_SIGNATURES_Z_BASIS[e] : NM_EVENT_SIGNATURES[e];
    }

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
    double compute_nll_from_histogram(const uint32_t hist[NM_N_STATES]);

    /**
     * compute_legacy_marginal_nll_from_histogram() — Legacy 4-check independent
     * Bernoulli marginal NLL (used for 6 Nov old-vs-new NLL comparison & B1 baseline).
     */
    double compute_legacy_marginal_nll_from_histogram(const uint32_t hist[NM_N_STATES]) const;

    /**
     * compute_gradient() — Exact analytical gradient of Joint-DP NLL w.r.t. θ.
     */
    void compute_gradient(const ring_buffer_t &rb);

    /**
     * compute_gradient_from_histogram() — Exact analytical gradient from 16-bin histogram.
     */
    void compute_gradient_from_histogram(const uint32_t hist[NM_N_STATES]);

    /**
     * compute_orbit_probabilities() — Computes the 9 data-identifiable syndrome-signature
     * orbit aggregate probabilities P(O_g) = (1 - prod_{e in O_g} (1 - 2 p_e)) / 2.
     */
    static void compute_orbit_probabilities(const ParamVec &theta, bool basis_z, double orbit_probs[NM_N_ORBITS]);

    /**
     * compute_orbit_rmse() — RMSE across the 9 identifiable signature orbits between
     * current theta_ and a ground-truth parameter vector.
     */
    double compute_orbit_rmse(const ParamVec &true_theta) const;

    /**
     * compute_param_21_rmse() — RMSE across all 21 parameters between current theta_
     * and a ground-truth parameter vector.
     */
    double compute_param_21_rmse(const ParamVec &true_theta) const;

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

    /**
     * to_pair_projected_q7_8_llr() — Causal Pair-Aware Logical-Triangle LLR Projection (Q04):
     * Projects estimated correlated pair probabilities theta_[9..20] across logical-triangle
     * endpoints (e.g. (4,5) vs q3 on syndrome 0x4, (3,4) vs q5 on syndrome 0x2)
     * strictly from the estimated parameter vector theta_ without any scenario shortcut.
     */
    void to_pair_projected_q7_8_llr(int16_t llr_q78[NM_N_QUBIT], const double *r2d2_gains = nullptr) const;

    /** Clamp parameters to valid interior range [1e-5, 0.45] */
    void clamp_params();

    /** Validate that a parameter vector lies within valid probability bounds */
    static bool validate_parameter_bounds(const ParamVec &theta);

    ParamVec theta_;
    ParamVec prior_theta_;
    double   prior_lambda_;
    bool     basis_z_;

private:
    GradVec  grad_;
    StateVec p_joint_;
    Eigen::Matrix<double, NM_N_QUBIT, 1> p_eff_;

    void compute_p_eff();
    void extract_histogram(const ring_buffer_t &rb, uint32_t hist[NM_N_STATES]) const;
};
