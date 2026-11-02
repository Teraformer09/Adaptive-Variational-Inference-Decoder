/* ===========================================================================
 * lbfgs_solver.cpp — Bounded-Iteration Histogram L-BFGS Solver (A02 / Q03, SF4)
 * =========================================================================*/
#include "lbfgs_solver.h"
#include <algorithm>

HistogramLBFGSSolver::HistogramLBFGSSolver() {
    opt_.reset();
}

double HistogramLBFGSSolver::fit_histogram(NoiseModel &model,
                                           const uint32_t hist_x[NM_N_STATES],
                                           int max_iters) {
    double nll = model.compute_nll_from_histogram(hist_x);
    for (int it = 0; it < max_iters; it++) {
        model.compute_gradient_from_histogram(hist_x);
        const auto &g = model.grad();
        if (g.norm() < 1e-6) {
            break;
        }
        /* Projected Armijo step along negative gradient / quasi-Newton step */
        NoiseModel::ParamVec x0 = model.theta_;
        double step = 0.04;
        for (int ls = 0; ls < 8; ls++) {
            model.theta_ = x0 - step * g;
            model.clamp_params();
            double cand = model.compute_nll_from_histogram(hist_x);
            if (cand <= nll - 1e-4 * step * g.squaredNorm()) {
                nll = cand;
                break;
            }
            step *= 0.5;
            model.theta_ = x0;
        }
    }
    return nll;
}
