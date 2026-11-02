/* ===========================================================================
 * joint_dp_estimator.h — Exact 2^m Joint DP Estimator Wrapper (A02 / Q03)
 * =========================================================================*/
#pragma once
#include "../inference/noise_model.h"

class JointDPEstimator {
public:
    JointDPEstimator();
    void reset(double p_single = 0.012, double p_pair = 0.004);
    double evaluate_nll_and_grad(const uint32_t hist_x[NM_N_STATES],
                                 NoiseModel::GradVec &out_grad);
    NoiseModel &model() { return model_; }
    const NoiseModel &model() const { return model_; }
private:
    NoiseModel model_;
};
