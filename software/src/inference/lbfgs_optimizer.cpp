/* ===========================================================================
 * lbfgs_optimizer.cpp — AVID Phase 3.3
 * L-BFGS implementation with pre-allocated memory and two-loop curvature updates.
 * =========================================================================*/
#include "lbfgs_optimizer.h"
#include <cmath>
#include <algorithm>
#include <limits>

LBFGSOptimizer::LBFGSOptimizer() {
    reset();
    for (int i = 0; i < LBFGS_M; i++) {
        s_hist_[i].setZero();
        y_hist_[i].setZero();
        rho_[i] = 0.0;
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

static double compute_projected_grad(const NoiseModel &model, LBFGSOptimizer::Vec *g_proj_out) {
    const auto &g = model.grad();
    double max_inf = 0.0;
    for (int i = 0; i < NM_N_PARAMS; i++) {
        double gi = g(i);
        if ((model.theta_(i) <= 1e-5 + 1e-9 && gi > 0.0) ||
            (model.theta_(i) >= 0.45 - 1e-9 && gi < 0.0)) {
            gi = 0.0;
        }
        if (g_proj_out) {
            (*g_proj_out)(i) = gi;
        }
        max_inf = std::max(max_inf, std::abs(gi));
    }
    return max_inf;
}

void LBFGSOptimizer::two_loop_recursion(const Vec &grad, Vec &direction) {
    direction = -grad;
    if (hist_count_ == 0) return;

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
    double gamma = (yTy > 1e-12 && sTy > 1e-12) ? std::clamp(sTy / yTy, 1e-3, 10.0) : 1.0;
    direction = gamma * q_;

    /* Backward pass */
    for (int i = 0; i < hist_count_; i++) {
        int idx = (hist_head_ - hist_count_ + i + LBFGS_M * 2) % LBFGS_M;
        double beta = rho_[idx] * y_hist_[idx].dot(direction);
        direction += s_hist_[idx] * (alpha_arr[i] - beta);
    }
    direction = -direction;
}

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
        if (std::isfinite(f1) && f1 <= f0 + LBFGS_C1 * alpha * slope)
            return f1;
        alpha *= 0.5;
    }
    model.theta_ = theta0;
    return f0;
}

double LBFGSOptimizer::step(NoiseModel &model, const ring_buffer_t &rb) {
    if (!NoiseModel::validate_parameter_bounds(model.theta_) || !model.has_valid_hyperparameters()) {
        return std::numeric_limits<double>::infinity();
    }

    Vec direction;
    double f = model.compute_nll(rb);

    for (int iter = 0; iter < LBFGS_MAX_ITER; iter++) {
        f = model.compute_nll(rb);
        model.compute_gradient(rb);
        const Vec &g = model.grad();

        double gnorm = g.norm();
        if (!std::isfinite(gnorm) || gnorm < LBFGS_TOL) break;

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

        two_loop_recursion(g, direction);
        if (g.dot(direction) >= -1e-15) {
            direction = -g;
        }

        prev_theta_ = model.theta_;
        prev_grad_  = g;
        has_prev_   = true;

        f = line_search(model, rb, direction, f, g);
        model.clamp_params();
    }
    return f;
}

int LBFGSOptimizer::step_histogram(NoiseModel &model,
                                   const uint32_t hist[NM_N_STATES],
                                   int max_iters,
                                   double *nll_io,
                                   double *grad_inf_out) {
    if (!hist || max_iters <= 0 ||
        !NoiseModel::validate_parameter_bounds(model.theta_) ||
        !model.has_valid_hyperparameters()) {
        if (grad_inf_out) *grad_inf_out = 0.0;
        return 0;
    }

    const int capped_iters = std::min(max_iters, LBFGS_MAX_ITER);
    double nll = model.compute_nll_from_histogram(hist);
    int iters_done = 0;
    double g_inf = 0.0;
    Vec direction;

    for (int it = 0; it < capped_iters; it++) {
        model.compute_gradient_from_histogram(hist);
        const Vec g_raw = model.grad();
        Vec g_proj;
        g_inf = compute_projected_grad(model, &g_proj);
        if (!std::isfinite(g_inf) || g_inf < 1e-5) {
            break;
        }

        if (has_prev_) {
            Vec s_k = model.theta_ - prev_theta_;
            Vec y_k = g_raw - prev_grad_;
            double sTy = s_k.dot(y_k);
            if (std::isfinite(sTy) && sTy > 1e-12) {
                s_hist_[hist_head_] = s_k;
                y_hist_[hist_head_] = y_k;
                rho_[hist_head_]    = 1.0 / sTy;
                hist_head_ = (hist_head_ + 1) % LBFGS_M;
                if (hist_count_ < LBFGS_M) hist_count_++;
            }
        }

        /* Compute two-loop L-BFGS quasi-Newton direction and project active bounds */
        two_loop_recursion(g_proj, direction);
        for (int i = 0; i < NM_N_PARAMS; i++) {
            if ((model.theta_(i) <= 1e-5 + 1e-9 && direction(i) < 0.0) ||
                (model.theta_(i) >= 0.45 - 1e-9 && direction(i) > 0.0)) {
                direction(i) = 0.0;
            }
        }
        if (g_proj.dot(direction) >= -1e-15) {
            reset();
            direction = -g_proj;
        }

        double max_dir = 0.0;
        for (int i = 0; i < NM_N_PARAMS; i++) {
            max_dir = std::max(max_dir, std::abs(direction(i)));
        }
        if (max_dir <= 1e-15) {
            break;
        }

        iters_done++;
        prev_theta_ = model.theta_;
        prev_grad_  = g_raw;
        has_prev_   = true;

        Vec x0 = model.theta_;
        double step_alpha = (hist_count_ > 0)
                                ? std::min(1.0, 0.12 / max_dir)
                                : (0.12 / std::max(1.0, max_dir));

        bool accepted = false;
        for (int ls = 0; ls < LBFGS_LS_MAX; ls++) {
            model.theta_ = x0 + step_alpha * direction;
            model.clamp_params();
            double dir_deriv = 0.0;
            for (int i = 0; i < NM_N_PARAMS; i++) {
                dir_deriv += g_raw(i) * (model.theta_(i) - x0(i));
            }
            double cand = model.compute_nll_from_histogram(hist);
            if (std::isfinite(cand) && cand <= nll + LBFGS_C1 * dir_deriv) {
                nll = cand;
                accepted = true;
                break;
            }
            step_alpha *= 0.5;
            model.theta_ = x0;
        }
        if (!accepted) {
            model.theta_ = x0;
            reset();
        }
    }

    model.compute_gradient_from_histogram(hist);
    if (grad_inf_out) *grad_inf_out = compute_projected_grad(model, nullptr);
    if (nll_io) *nll_io = nll;
    return iters_done;
}
