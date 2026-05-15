/* ===========================================================================
 * noise_model.h — AVID Phase 3.2
 * HMM-based variational inference engine for correlated Pauli noise.
 *
 * COMMANDMENT #6: No VQE.  Classical inference only.
 * COMMANDMENT #7: Tracks ZZ/XX crosstalk and single-qubit drift.
 *                 Does NOT track coherent over-rotations (Pauli-twirled).
 * COMMANDMENT #8: Only learns parameters that leave a topological signature.
 * COMMANDMENT #10: Mathematical objective defined below before code.
 *
 * ----------------------------------------------------------------------------
 * MATHEMATICAL FOUNDATION
 * ----------------------------------------------------------------------------
 *
 * Hidden Markov Model for Correlated Pauli Noise on d=3 Surface Code
 * -------------------------------------------------------------------
 *
 * Parameters:
 *   θ = { p_i ∈ (0,0.5)  : single-qubit Z-error rate, i = 0..8
 *         p_ij ∈ (0,0.5) : ZZ crosstalk rate for adjacent pairs (i,j) }
 *
 *   Total parameters for d=3: 9 single + 12 nearest-neighbor = 21
 *
 * Observable: syndrome window {s_c^t} for t = 0..W-1, c = 0..N_CHECK-1
 *
 * Identifiability (COMMANDMENT #8):
 *   Correlated ZZ error on (i,j) creates a characteristic *pair* of
 *   syndrome changes: both check nodes adjacent to both i and j flip
 *   simultaneously.  This is distinct from two independent single-qubit
 *   errors, which have a different temporal correlation structure.
 *   Coherent over-rotations (e.g., X-axis rotations) do not change the
 *   Pauli-frame syndrome signature → they are NOT identifiable from
 *   stabilizer syndromes alone and are explicitly excluded.
 *
 * Observation model (exact formula for independent errors):
 *   P(s_c = 1 | θ) = [1 - ∏_{v ∈ N(c)} (1 - 2 p_v_eff)] / 2
 *
 *   where p_v_eff = p_v + Σ_{j ∈ NB(v)} p_vj - 2 Σ_j p_v p_vj
 *   (first-order inclusion-exclusion for correlated pairs)
 *
 * Objective function (Negative Log-Likelihood):
 *   L(θ) = -Σ_{t=0}^{W-1} Σ_{c=0}^{N_CHECK-1}
 *             [ s_c^t · log q_c(θ) + (1-s_c^t) · log(1 - q_c(θ)) ]
 *   where q_c(θ) = P(s_c=1|θ)
 *
 * Gradient:
 *   ∂L/∂p_i = -Σ_{t,c: i∈N(c)}
 *               [ (s_c^t - q_c) / (q_c(1-q_c)) ] · ∂q_c/∂p_i
 *
 *   ∂q_c/∂p_i = ∏_{v∈N(c), v≠i} (1 - 2 p_v_eff)
 *              × ∂p_i_eff/∂p_i
 *   ∂p_i_eff/∂p_i = 1 - 2 Σ_j p_ij
 *
 * LLR from estimated parameters:
 *   LLR_v = log( (1 - p_v_eff) / p_v_eff )
 *
 * Weight injection: LLR_v → Q7.8 → FPGA via weight_reg_driver.
 * =========================================================================*/
#pragma once

#include <Eigen/Dense>
#include "ring_buffer.h"

/* d=3 parameters */
#define NM_N_QUBIT     9    /* Single-qubit error rates */
#define NM_N_CROSSTALK 12   /* Nearest-neighbor ZZ pairs */
#define NM_N_PARAMS    (NM_N_QUBIT + NM_N_CROSSTALK)  /* 21 total */
#define NM_N_CHECK     4    /* d=3 X-type check nodes */

/* Adjacent qubit pair table (nearest-neighbor on d=3 grid) */
struct CrosstalkPair { int i, j; };
extern const CrosstalkPair NM_CROSSTALK_PAIRS[NM_N_CROSSTALK];

/* d=3 X-type Tanner graph check connectivity */
extern const int NM_CHECK_CONN[NM_N_CHECK][4];  /* 4 qubits per check */

/* -------------------------------------------------------------------------
 * NoiseModel
 * Pre-allocates all Eigen matrices as static members (COMMANDMENT #3).
 * -------------------------------------------------------------------------*/
class NoiseModel {
public:
    /* Parameter vector θ: first N_QUBIT entries = p_i, rest = p_ij */
    using ParamVec = Eigen::Matrix<double, NM_N_PARAMS, 1>;
    using GradVec  = Eigen::Matrix<double, NM_N_PARAMS, 1>;

    NoiseModel();

    /** Initialise parameters to p_i = p0, p_ij = 0 */
    void init(double p0 = 0.01);

    /**
     * compute_nll() — Negative log-likelihood of syndrome window given θ.
     * O(W × N_CHECK × MAX_DEG) per call.
     */
    double compute_nll(const ring_buffer_t &rb);

    /**
     * compute_gradient() — Analytical gradient of NLL w.r.t. θ.
     * Fills grad_ in-place.  Must call before lbfgs_step().
     */
    void compute_gradient(const ring_buffer_t &rb);

    /** Read-only access to current parameters */
    const ParamVec &params() const { return theta_; }

    /** Read-only access to last computed gradient */
    const GradVec  &grad()   const { return grad_;  }

    /**
     * effective_p(qubit_idx) — p_v_eff including crosstalk corrections.
     * Used to compute LLR for weight injection.
     */
    double effective_p(int v) const;

    /**
     * to_llr(float llr_x[9], float llr_z[9])
     * Converts current parameters to per-qubit LLRs for FPGA injection.
     * llr_x and llr_z are filled with log((1-p_eff)/p_eff) for each qubit.
     * (For X-errors use Z-stabilizer model; for Z-errors use X-stabilizer model.)
     */
    void to_llr(float llr_x[NM_N_QUBIT], float llr_z[NM_N_QUBIT]) const;

    /** Clamp parameters to valid range (0.001, 0.499) */
    void clamp_params();

    ParamVec theta_;   /* Public for L-BFGS optimizer to update */

private:
    /* Pre-allocated temporaries — no heap allocation in hot loop */
    GradVec grad_;
    Eigen::Matrix<double, NM_N_CHECK, 1> q_c_;   /* Check fire probs */
    Eigen::Matrix<double, NM_N_QUBIT,  1> p_eff_; /* Effective single-qubit rates */

    void compute_p_eff();
    void compute_check_probs();
};
