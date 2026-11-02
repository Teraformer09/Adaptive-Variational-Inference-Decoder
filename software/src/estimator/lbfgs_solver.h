/* ===========================================================================
 * lbfgs_solver.h — Bounded-Iteration Histogram L-BFGS Solver (A02 / Q03, SF4)
 * =========================================================================*/
#pragma once
#include "../inference/lbfgs_optimizer.h"

class HistogramLBFGSSolver {
public:
    HistogramLBFGSSolver();
    double fit_histogram(NoiseModel &model, const uint32_t hist_x[NM_N_STATES], int max_iters = 15);
private:
    LBFGSOptimizer opt_;
};
