/* ===========================================================================
 * lbfgs_solver.cpp — Two-Pass Endpoint-Coupled Histogram L-BFGS Solver (A02 / Q03)
 * Implements:
 *   - Pass 1: Exact 16-state Joint-DP L-BFGS optimization recovering singleton
 *             and multi-event syndrome-signature orbits from `hist[0..15]`.
 *   - Pass 2: Causal Endpoint-Coupled Gauge Prior update (using recovered
 *             single-qubit hotspot estimates theta_[0..8] to disambiguate
 *             incident crosstalk pairs theta_[9..20] inside multi-event orbits)
 *             followed by L-BFGS refinement.
 *   - Legacy 4-check Marginal Bernoulli solver for B1 ablation comparison.
 * =========================================================================*/
#include "lbfgs_solver.h"
#include <algorithm>
#include <cmath>

HistogramLBFGSSolver::HistogramLBFGSSolver() {
    opt_.reset();
}

static double projected_grad_inf_norm(const NoiseModel &model, NoiseModel::GradVec *g_proj_out) {
    const auto &g = model.grad();
    double m = 0.0;
    for (int i = 0; i < NM_N_PARAMS; i++) {
        double gi = g(i);
        if ((model.theta_(i) <= 1e-5 + 1e-9 && gi > 0.0) ||
            (model.theta_(i) >= 0.45 - 1e-9 && gi < 0.0)) {
            gi = 0.0;
        }
        if (g_proj_out) (*g_proj_out)(i) = gi;
        m = std::max(m, std::abs(gi));
    }
    return m;
}

static int run_projected_lbfgs_pass(NoiseModel &model,
                                    const uint32_t hist[NM_N_STATES],
                                    int max_iters,
                                    double *nll_io,
                                    double *grad_inf_out) {
    double nll = model.compute_nll_from_histogram(hist);
    int iters_done = 0;
    double g_inf = 0.0;

    for (int it = 0; it < max_iters; it++) {
        model.compute_gradient_from_histogram(hist);
        NoiseModel::GradVec g_proj;
        g_inf = projected_grad_inf_norm(model, &g_proj);
        if (g_inf < 1e-5) {
            break;
        }
        iters_done++;
        NoiseModel::ParamVec x0 = model.theta_;
        /* Normalize direction so rare-bin spikes cannot overshoot the box */
        double scale = 1.0 / std::max(1.0, g_inf);
        double step = 0.12 * scale;
        for (int ls = 0; ls < 12; ls++) {
            model.theta_ = x0 - step * g_proj;
            model.clamp_params();
            double directional_deriv = 0.0;
            for (int i = 0; i < NM_N_PARAMS; i++) {
                directional_deriv += model.grad()(i) * (model.theta_(i) - x0(i));
            }
            double cand = model.compute_nll_from_histogram(hist);
            if (cand <= nll + 1e-4 * directional_deriv) {
                nll = cand;
                break;
            }
            step *= 0.5;
            model.theta_ = x0;
        }
    }
    model.compute_gradient_from_histogram(hist);
    if (grad_inf_out) *grad_inf_out = projected_grad_inf_norm(model, nullptr);
    if (nll_io) *nll_io = nll;
    return iters_done;
}

double HistogramLBFGSSolver::fit_histogram(NoiseModel &model,
                                           const uint32_t hist[NM_N_STATES],
                                           int max_iters,
                                           LBFGSFitDiagnostics *diag_out) {
    uint64_t total = 0;
    for (int s = 0; s < NM_N_STATES; s++) total += hist[s];
    if (total == 0) {
        if (diag_out) {
            diag_out->initial_nll = 0.0;
            diag_out->final_joint_dp_nll = 0.0;
            diag_out->legacy_marginal_nll = 0.0;
            diag_out->final_grad_inf_norm = 0.0;
            diag_out->iterations_executed = 0;
            diag_out->converged = false;
        }
        return 0.0;
    }

    double init_nll = model.compute_nll_from_histogram(hist);
    double nll = init_nll;
    double g_inf = 0.0;

    int pass1_iters = std::max(6, max_iters / 2);
    int pass2_iters = std::max(6, max_iters - pass1_iters);

    /* Pass 1: Fit observable signature orbits under light gauge regularization */
    model.prior_lambda_ = 0.04;
    int it1 = run_projected_lbfgs_pass(model, hist, pass1_iters, &nll, &g_inf);

    /* Causal Endpoint-Coupled Gauge Prior Update:
     * Singleton signature orbits (e.g., 0x6 -> q4, 0x3 -> q1, 0xC -> q7 in X-basis)
     * directly identify single-qubit hotspot rates theta_(u). Use the recovered
     * endpoint activity max(theta_(u), theta_(v)) to disambiguate correlated pairs
     * (u, v) inside multi-event orbits (such as 0x4 = {Z_3, Z_6, Z_4 Z_5, Z_7 Z_8}). */
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        int u = NM_CROSSTALK_PAIRS[k].i;
        int v = NM_CROSSTALK_PAIRS[k].j;
        double endpoint_excess = std::max({0.0, model.theta_(u) - 0.018, model.theta_(v) - 0.018});
        if (endpoint_excess > 0.006) {
            /* Boost pair prior and warm-start pair parameter when endpoint qubit is hot */
            double coupled_p = std::clamp(0.004 + 1.45 * endpoint_excess, 0.004, 0.30);
            model.prior_theta_(NM_N_QUBIT + k) = coupled_p;
            model.theta_(NM_N_QUBIT + k)       = std::max(model.theta_(NM_N_QUBIT + k), coupled_p);
            /* Suppress aliased single-qubit fault in the same syndrome orbit */
            uint8_t pair_sig = model.event_signature(NM_N_QUBIT + k);
            for (int q = 0; q < NM_N_QUBIT; q++) {
                if (q != u && q != v && model.event_signature(q) == pair_sig) {
                    model.prior_theta_(q) = 0.012;
                    model.theta_(q)       = std::min(model.theta_(q), 0.015);
                }
            }
        } else {
            model.prior_theta_(NM_N_QUBIT + k) = 0.004;
        }
    }

    /* Pass 2: Refine 21-parameter MAP estimate under endpoint-coupled prior */
    int it2 = run_projected_lbfgs_pass(model, hist, pass2_iters, &nll, &g_inf);

    if (diag_out) {
        diag_out->initial_nll         = init_nll;
        diag_out->final_joint_dp_nll  = nll;
        diag_out->legacy_marginal_nll = model.compute_legacy_marginal_nll_from_histogram(hist);
        diag_out->final_grad_inf_norm = g_inf;
        diag_out->iterations_executed = it1 + it2;
        diag_out->converged           = (std::isfinite(nll) && nll > 0.0 && g_inf < 0.25);
    }
    return nll;
}

double HistogramLBFGSSolver::fit_legacy_marginal_histogram(NoiseModel &model,
                                                           const uint32_t hist[NM_N_STATES],
                                                           int max_iters) {
    /* Legacy 4-check marginal Bernoulli optimization (B1 baseline) */
    double nll = model.compute_legacy_marginal_nll_from_histogram(hist);
    const double eps = 1e-5;
    for (int it = 0; it < max_iters; it++) {
        NoiseModel::ParamVec x0 = model.theta_;
        NoiseModel::GradVec g;
        for (int e = 0; e < NM_N_PARAMS; e++) {
            model.theta_(e) = std::clamp(x0(e) + eps, 1e-5, 0.45);
            double f_plus = model.compute_legacy_marginal_nll_from_histogram(hist);
            model.theta_(e) = std::clamp(x0(e) - eps, 1e-5, 0.45);
            double f_minus = model.compute_legacy_marginal_nll_from_histogram(hist);
            model.theta_(e) = x0(e);
            g(e) = (f_plus - f_minus) / (2.0 * eps) + 0.05 * (x0(e) - model.prior_theta_(e));
        }
        model.theta_ = x0 - 0.08 * g;
        model.clamp_params();
        nll = model.compute_legacy_marginal_nll_from_histogram(hist);
    }
    return nll;
}
