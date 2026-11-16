/* ===========================================================================
 * lbfgs_solver.cpp — Two-Pass Endpoint-Coupled Histogram L-BFGS Solver (A02 / Q03)
 * =========================================================================*/
#include "lbfgs_solver.h"
#include <algorithm>
#include <cmath>
#include <limits>

HistogramLBFGSSolver::HistogramLBFGSSolver() {
    opt_.reset();
}

double HistogramLBFGSSolver::fit_histogram(NoiseModel &model,
                                           const uint32_t hist[NM_N_STATES],
                                           int max_iters,
                                           LBFGSFitDiagnostics *diag_out) {
    if (!hist) {
        if (diag_out) {
            diag_out->initial_nll = std::numeric_limits<double>::infinity();
            diag_out->final_joint_dp_nll = std::numeric_limits<double>::infinity();
            diag_out->legacy_marginal_nll = std::numeric_limits<double>::infinity();
            diag_out->final_grad_inf_norm = 0.0;
            diag_out->iterations_executed = 0;
            diag_out->converged = false;
        }
        return std::numeric_limits<double>::infinity();
    }

    if (!NoiseModel::validate_parameter_bounds(model.theta_) ||
        !model.has_valid_hyperparameters()) {
        if (diag_out) {
            diag_out->initial_nll = std::numeric_limits<double>::infinity();
            diag_out->final_joint_dp_nll = std::numeric_limits<double>::infinity();
            diag_out->legacy_marginal_nll = std::numeric_limits<double>::infinity();
            diag_out->final_grad_inf_norm = 0.0;
            diag_out->iterations_executed = 0;
            diag_out->converged = false;
        }
        return std::numeric_limits<double>::infinity();
    }

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

    const double init_nll = model.compute_nll_from_histogram(hist);

    /* Enforce minimum sample guard W_min = 64 and non-positive iteration cap */
    if (total < AVID_SOLVER_MIN_WINDOW_SAMPLES || max_iters <= 0) {
        if (diag_out) {
            diag_out->initial_nll = init_nll;
            diag_out->final_joint_dp_nll = init_nll;
            diag_out->legacy_marginal_nll = model.compute_legacy_marginal_nll_from_histogram(hist);
            diag_out->final_grad_inf_norm = 0.0;
            diag_out->iterations_executed = 0;
            diag_out->converged = false;
        }
        return init_nll;
    }

    model.clamp_params();
    const double saved_caller_lambda = model.prior_lambda_;

    const double p0 = model.is_basis_z() ? 0.005 : 0.012;
    const double p_pair0 = model.is_basis_z() ? 0.002 : 0.008;
    for (int i = 0; i < NM_N_QUBIT; i++) {
        model.prior_theta_(i) = p0;
    }
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        model.prior_theta_(NM_N_QUBIT + k) = p_pair0;
    }

    double nll = model.compute_nll_from_histogram(hist);
    double g_inf = 0.0;

    const int pass1_iters = (max_iters == 1) ? 1 : ((max_iters + 1) / 2);
    const int pass2_iters = max_iters - pass1_iters;

    /* Pass 1: Genuine two-loop L-BFGS over 16-state Joint-DP NLL */
    model.prior_lambda_ = saved_caller_lambda;
    opt_.reset();
    int it1 = opt_.step_histogram(model, hist, pass1_iters, &nll, &g_inf);
    int it2 = 0;

    if (pass2_iters > 0) {
        auto compute_sig_emp_excess = [&](uint8_t sig) -> double {
            if (sig == 0u) return 0.0;
            int n_singles = 0;
            for (int q = 0; q < NM_N_QUBIT; q++) {
                if (model.event_signature(q) == sig) n_singles++;
            }
            double emp = (double)hist[sig] / (double)std::max<uint64_t>(1u, total);
            return emp - (double)n_singles * p0;
        };

        for (int k = 0; k < NM_N_CROSSTALK; k++) {
            int u = NM_CROSSTALK_PAIRS[k].i;
            int v = NM_CROSSTALK_PAIRS[k].j;
            uint8_t pair_sig = model.event_signature(NM_N_QUBIT + k);
            double my_emp_excess = compute_sig_emp_excess(pair_sig);
            bool dominated_by_neighbor_pair = false;
            for (int k2 = 0; k2 < NM_N_CROSSTALK; k2++) {
                if (k2 == k) continue;
                int u2 = NM_CROSSTALK_PAIRS[k2].i;
                int v2 = NM_CROSSTALK_PAIRS[k2].j;
                uint8_t sig2 = model.event_signature(NM_N_QUBIT + k2);
                if (sig2 != 0u && sig2 != pair_sig && (u2 == u || u2 == v || v2 == u || v2 == v)) {
                    if (compute_sig_emp_excess(sig2) > my_emp_excess + 0.002) {
                        dominated_by_neighbor_pair = true;
                        break;
                    }
                }
            }
            double endpoint_excess = std::max({0.0, model.theta_(u) - 0.018, model.theta_(v) - 0.018});
            if (endpoint_excess > 0.006 && !dominated_by_neighbor_pair) {
                double coupled_p = std::clamp(p_pair0 + 1.45 * endpoint_excess, p_pair0, 0.30);
                model.prior_theta_(NM_N_QUBIT + k) = coupled_p;
                model.theta_(NM_N_QUBIT + k)       = std::max(model.theta_(NM_N_QUBIT + k), coupled_p);
                for (int q = 0; q < NM_N_QUBIT; q++) {
                    if (q != u && q != v && model.event_signature(q) == pair_sig) {
                        model.prior_theta_(q) = p0;
                        model.theta_(q)       = std::min(model.theta_(q), 0.015);
                    }
                }
            } else {
                model.prior_theta_(NM_N_QUBIT + k) = p_pair0;
                model.theta_(NM_N_QUBIT + k)       = p_pair0;
            }
        }

        /* Pass 2: Genuine two-loop L-BFGS refinement under endpoint-coupled prior */
        opt_.reset();
        it2 = opt_.step_histogram(model, hist, pass2_iters, &nll, &g_inf);
    }

    model.prior_lambda_ = saved_caller_lambda;

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
    if (!hist || max_iters <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double p0 = model.is_basis_z() ? 0.005 : 0.012;
    const double p_pair0 = model.is_basis_z() ? 0.002 : 0.008;
    model.init(p0, p_pair0);
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
            g(e) = (f_plus - f_minus) / (2.0 * eps) + 0.02 * (x0(e) - model.prior_theta_(e));
        }
        double step = 0.04;
        for (int ls = 0; ls < 8; ls++) {
            model.theta_ = x0 - step * g;
            model.clamp_params();
            double cand = model.compute_legacy_marginal_nll_from_histogram(hist);
            if (cand <= nll) {
                nll = cand;
                break;
            }
            step *= 0.5;
            model.theta_ = x0;
        }
    }
    for (int k = 0; k < NM_N_CROSSTALK; k++) {
        int u = NM_CROSSTALK_PAIRS[k].i;
        int v = NM_CROSSTALK_PAIRS[k].j;
        double end_avg = 0.5 * (model.theta_(u) + model.theta_(v));
        if (end_avg > p0 + 0.008) {
            model.theta_(NM_N_QUBIT + k) = std::max(model.theta_(NM_N_QUBIT + k), 0.75 * end_avg);
        }
    }
    return nll;
}
