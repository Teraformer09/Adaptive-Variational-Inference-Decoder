/* ===========================================================================
 * noise_model.h — Repaired Exact 2^m-State Joint DP Correlated Likelihood Engine
 * Policy Items: A01 / Q02, A02 / Q03, A03 / Q04, A08 / Q17
 * =========================================================================*/
#pragma once

#include <Eigen/Dense>
#include <stddef.h>
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

extern const int     NM_CHECK_CONN[NM_N_CHECK][4];
extern const uint8_t NM_CHECK_ACTIVE_MASK[NM_N_CHECK];

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

    /** Initialise parameters to single-qubit p_i = p0, pair p_ij = p_pair0 (aligned with parameter_schema.json) */
    void init(double p0 = 0.012, double p_pair0 = 0.008);

    void set_basis_z(bool basis_z) { basis_z_ = basis_z; }
    bool is_basis_z() const { return basis_z_; }

    uint8_t event_signature(int e) const {
        return basis_z_ ? NM_EVENT_SIGNATURES_Z_BASIS[e] : NM_EVENT_SIGNATURES[e];
    }

    void compute_joint_distribution();

    double compute_nll(const ring_buffer_t &rb);

    double compute_nll_from_histogram(const uint32_t hist[NM_N_STATES]);

    /** Checked 16-bin histogram NLL entrypoint rejecting nullptr or n_bins != 16 */
    bool compute_nll_from_histogram_checked(const uint32_t *hist,
                                            size_t n_bins,
                                            double *out_nll);

    /** Normalized empirical PMF validator & NLL entrypoint (rejects nullptr, n_bins!=16, NaN, <0, |sum-1|>tol) */
    static bool validate_empirical_pmf(const double *pmf,
                                       size_t n_bins,
                                       double sum_tol = 1e-6);

    bool compute_nll_from_pmf(const double *pmf,
                              size_t n_bins,
                              double *out_nll,
                              double sum_tol = 1e-6);

    double compute_legacy_marginal_nll_from_histogram(const uint32_t hist[NM_N_STATES]) const;

    void compute_gradient(const ring_buffer_t &rb);

    void compute_gradient_from_histogram(const uint32_t hist[NM_N_STATES]);

    static void compute_orbit_probabilities(const ParamVec &theta, bool basis_z, double orbit_probs[NM_N_ORBITS]);

    double compute_orbit_rmse(const ParamVec &true_theta) const;

    double compute_param_21_rmse(const ParamVec &true_theta) const;

    const ParamVec &params() const { return theta_; }
    const GradVec  &grad()   const { return grad_;  }
    const StateVec &joint_probs() const { return p_joint_; }

    double effective_p(int v) const;

    void to_llr(float llr_x[NM_N_QUBIT], float llr_z[NM_N_QUBIT]) const;

    void to_q7_8_llr(int16_t llr_q78[NM_N_QUBIT]) const;

    void to_pair_projected_q7_8_llr(int16_t llr_q78[NM_N_QUBIT],
                                    const double *qubit_gains = nullptr,
                                    double pair_weight_scale = 1.0) const;

    void clamp_params();

    static bool validate_parameter_bounds(const ParamVec &theta);
    bool has_valid_hyperparameters() const;

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
