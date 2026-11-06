/* ===========================================================================
 * lbfgs_solver.h — Two-Pass Endpoint-Coupled Histogram L-BFGS Solver (A02 / Q03)
 * =========================================================================*/
#pragma once
#include "../inference/lbfgs_optimizer.h"

struct LBFGSFitDiagnostics {
    double initial_nll;
    double final_joint_dp_nll;
    double legacy_marginal_nll;
    double final_grad_inf_norm;
    int    iterations_executed;
    bool   converged;
};

class HistogramLBFGSSolver {
public:
    HistogramLBFGSSolver();
    double fit_histogram(NoiseModel &model,
                         const uint32_t hist[NM_N_STATES],
                         int max_iters = 20,
                         LBFGSFitDiagnostics *diag_out = nullptr);

    double fit_legacy_marginal_histogram(NoiseModel &model,
                                         const uint32_t hist[NM_N_STATES],
                                         int max_iters = 20);
private:
    LBFGSOptimizer opt_;
};
