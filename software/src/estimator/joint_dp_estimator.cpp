/* ===========================================================================
 * joint_dp_estimator.cpp — Exact 2^m Joint DP Estimator Wrapper (A02 / Q03)
 * =========================================================================*/
#include "joint_dp_estimator.h"

JointDPEstimator::JointDPEstimator() {
    model_.init(0.012, 0.004);
}

void JointDPEstimator::reset(double p_single, double p_pair) {
    model_.init(p_single, p_pair);
}

double JointDPEstimator::evaluate_nll_and_grad(const uint32_t hist_x[NM_N_STATES],
                                               NoiseModel::GradVec &out_grad) {
    double nll = model_.compute_nll_from_histogram(hist_x);
    model_.compute_gradient_from_histogram(hist_x);
    out_grad = model_.grad();
    return nll;
}
