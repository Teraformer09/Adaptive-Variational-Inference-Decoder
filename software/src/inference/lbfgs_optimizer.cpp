/* ===========================================================================
 * lbfgs_optimizer.cpp — AVID Phase 3.3
 * L-BFGS implementation with pre-allocated memory and time-budget enforcement.
 * =========================================================================*/
#include "lbfgs_optimizer.h"
#include <cmath>
#include <algorithm>

LBFGSOptimizer::LBFGSOptimizer() {
    reset();
    for (int i = 0; i < LBFGS_M; i++) {
        s_hist_[i].setZero();
        y_hist_[i].setZero();
    }
    q_.setZero();
    alpha_.setZero();
    prev_grad_.setZero();
    prev_theta_.setZero();
}

void LBFGSOptimizer::reset() {
    hist_count_ = 0;
    hist_head_  = 0;
    has_prev_   = false;
}

/* -------------------------------------------------------------------------
 * Two-loop L-BFGS recursion (Nocedal & Wright Algorithm 7.4)
 * Input:  current gradient g
 * Output: search direction d ≈ -H^{-1} g
 * -------------------------------------------------------------------------*/
void LBFGSOptimizer::two_loop_recursion(const Vec &grad, Vec &direction) {
    direction = -grad;
    if (hist_count_ == 0) return;  /* Pure gradient descent on first step */

    q_ = grad;
    double alpha_arr[LBFGS_M] = {};

    /* Forward pass */
    for (int i = hist_count_ - 1; i >= 0; i--) {
        int idx = (hist_head_ - 1 - i + LBFGS_M * 2) % LBFGS_M;
        alpha_arr[i] = rho_[idx] * s_hist_[idx].dot(q_);
        q_ -= alpha_arr[i] * y_hist_[idx];
    }

    /* Scale by initial Hessian estimate: H_0 = (s^T y / y^T y) I */
    int last_idx = (hist_head_ - 1 + LBFGS_M) % LBFGS_M;
    double sTy = s_hist_[last_idx].dot(y_hist_[last_idx]);
    double yTy = y_hist_[last_idx].dot(y_hist_[last_idx]);
    double gamma = (yTy > 1e-12) ? (sTy / yTy) : 1.0;
    direction = gamma * q_;

    /* Backward pass */
    for (int i = 0; i < hist_count_; i++) {
        int idx = (hist_head_ - hist_count_ + i + LBFGS_M * 2) % LBFGS_M;
        double beta = rho_[idx] * y_hist_[idx].dot(direction);
        direction += s_hist_[idx] * (alpha_arr[i] - beta);
    }
    direction = -direction;
}

/* -------------------------------------------------------------------------
 * Backtracking Armijo line search
 * -------------------------------------------------------------------------*/
double LBFGSOptimizer::line_search(NoiseModel &model, const ring_buffer_t &rb,
                                   const Vec &direction,
                                   double f0, const Vec &g0) {
    double alpha  = LBFGS_ALPHA0;
    double slope  = g0.dot(direction);
    Vec    theta0 = model.theta_;

    for (int ls = 0; ls < LBFGS_LS_MAX; ls++) {
        model.theta_ = theta0 + alpha * direction;
        model.clamp_params();
        double f1 = model.compute_nll(rb);
        if (f1 <= f0 + LBFGS_C1 * alpha * slope)
            return f1;
        alpha *= 0.5;
    }
    /* Backtrack failed — restore original params */
    model.theta_ = theta0;
    return f0;
}

/* -------------------------------------------------------------------------
 * step() — main optimization loop
 * -------------------------------------------------------------------------*/
double LBFGSOptimizer::step(NoiseModel &model, const ring_buffer_t &rb) {
    Vec direction;
    double f = 0.0;

    for (int iter = 0; iter < LBFGS_MAX_ITER; iter++) {
        /* Evaluate NLL and gradient */
        f = model.compute_nll(rb);
        model.compute_gradient(rb);
        const Vec &g = model.grad();

        double gnorm = g.norm();
        if (gnorm < LBFGS_TOL) break;

        /* Compute search direction via two-loop recursion */
        two_loop_recursion(g, direction);

        /* Update history with s_k and y_k */
        if (has_prev_) {
            Vec s_k = model.theta_ - prev_theta_;
            Vec y_k = g - prev_grad_;
            double sTy = s_k.dot(y_k);
            if (sTy > 1e-12) {
                s_hist_[hist_head_] = s_k;
                y_hist_[hist_head_] = y_k;
                rho_[hist_head_]    = 1.0 / sTy;
                hist_head_ = (hist_head_ + 1) % LBFGS_M;
                if (hist_count_ < LBFGS_M) hist_count_++;
            }
        }

        prev_theta_ = model.theta_;
        prev_grad_  = g;
        has_prev_   = true;

        /* Line search */
        f = line_search(model, rb, direction, f, g);
        model.clamp_params();
    }
    return f;
}
