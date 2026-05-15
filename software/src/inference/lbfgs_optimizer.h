/* ===========================================================================
 * lbfgs_optimizer.h — AVID Phase 3.3
 * Limited-memory BFGS optimizer for the NLL objective.
 *
 * COMMANDMENT #3: NO std::vector inside the hot loop.
 *   All history buffers (s_hist, y_hist, rho) are statically sized arrays.
 *   No heap allocation occurs after construction.
 *
 * Budget enforcement:
 *   The optimizer runs for at most MAX_ITER steps OR until the gradient
 *   norm < TOL, whichever comes first.  This ensures continuous ~10ms
 *   update cadence regardless of convergence speed.
 * =========================================================================*/
#pragma once

#include <Eigen/Dense>
#include "noise_model.h"

#define LBFGS_M          5       /* History depth */
#define LBFGS_MAX_ITER   50      /* Hard iteration cap (COMMANDMENT #3) */
#define LBFGS_TOL        1e-6    /* Gradient norm convergence threshold */
#define LBFGS_ALPHA0     0.01    /* Initial step size */
#define LBFGS_C1         1e-4    /* Armijo sufficient decrease constant */
#define LBFGS_LS_MAX     10      /* Max line search steps */

class LBFGSOptimizer {
public:
    using Vec = Eigen::Matrix<double, NM_N_PARAMS, 1>;

    LBFGSOptimizer();

    /**
     * reset() — Clear history.  Call when noise model is re-initialised
     * or after a large parameter jump (e.g., crosstalk burst detected).
     */
    void reset();

    /**
     * step() — Run up to LBFGS_MAX_ITER L-BFGS updates on the noise model.
     * @model: NoiseModel whose theta_ is updated in-place.
     * @rb:    Current syndrome window (used to evaluate NLL and gradient).
     * Returns final NLL value.
     */
    double step(NoiseModel &model, const ring_buffer_t &rb);

private:
    /* --- Pre-allocated history (NO dynamic allocation) --- */
    Vec s_hist_[LBFGS_M];   /* Parameter differences: s_k = θ_{k+1} - θ_k */
    Vec y_hist_[LBFGS_M];   /* Gradient differences:  y_k = g_{k+1} - g_k  */
    double rho_[LBFGS_M];   /* ρ_k = 1 / (y_k^T s_k) */
    int    hist_count_;     /* Valid history entries */
    int    hist_head_;      /* Circular write pointer */

    Vec    q_;              /* Working vector for two-loop recursion */
    Vec    alpha_;          /* Step sizes from two-loop recursion */
    Vec    prev_grad_;      /* g_{k-1} for y_k computation */
    Vec    prev_theta_;     /* θ_{k-1} for s_k computation */
    bool   has_prev_;

    /* Two-loop recursion: compute L-BFGS direction from current gradient */
    void two_loop_recursion(const Vec &grad, Vec &direction);

    /* Backtracking Armijo line search */
    double line_search(NoiseModel &model, const ring_buffer_t &rb,
                       const Vec &direction, double f0, const Vec &g0);
};
