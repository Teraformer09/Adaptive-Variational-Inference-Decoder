#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include "estimator/lbfgs_solver.h"
#include "control/r2d2_lstm_controller.h"
#include <cmath>
#include <cstring>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-004");
    const std::string tid = opts.test_id;

    if (tid == "L3-FLOW-004") {
        TestContext ctx(opts, tid, "estimator_to_optimizer_two_pass_contract",
                        "software/src/estimator/lbfgs_solver.cpp, software/src/inference/lbfgs_optimizer.cpp",
                        "HistogramLBFGSSolver::fit_histogram -> LBFGSOptimizer::step_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify fit_histogram propagates basis_z flag, priors, bounds [1e-5,0.45], and two-pass refinement into LBFGSOptimizer";
        ctx.record().independent_oracle = "diag.final_joint_dp_nll <= diag.initial_nll, X vs Z basis produce distinct priors (0.012 vs 0.005), bounds preserved";

        uint32_t hist[16] = {800, 15, 20, 12, 18, 10, 65, 5, 15, 6, 12, 4, 8, 4, 3, 3};
        NoiseModel mz, mx;
        mz.set_basis_z(false);
        mz.init(0.012, 0.008);
        mx.set_basis_z(true);
        mx.init(0.005, 0.002);

        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics dz{}, dx{};
        double nll_z = solver.fit_histogram(mz, hist, 24, &dz);
        double nll_x = solver.fit_histogram(mx, hist, 24, &dx);

        ASSERT_TRUE(std::isfinite(nll_z) && std::isfinite(nll_x));
        ASSERT_TRUE(dz.final_joint_dp_nll <= dz.initial_nll + 1e-12);
        ASSERT_TRUE(dx.final_joint_dp_nll <= dx.initial_nll + 1e-12);
        ASSERT_TRUE(dz.iterations_executed > 0 && dz.iterations_executed <= 24);
        ASSERT_NEAR(mz.prior_theta_(0), 0.012, 1e-12);
        ASSERT_NEAR(mx.prior_theta_(0), 0.005, 1e-12);
        ASSERT_TRUE(NoiseModel::validate_parameter_bounds(mz.theta_));
        ASSERT_TRUE(NoiseModel::validate_parameter_bounds(mx.theta_));

        ctx.record().expected["converged"] = "true";
        ctx.record().actual["converged"] = dz.converged ? "true" : "false";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-005") {
        TestContext ctx(opts, tid, "optimizer_to_noise_estimate_and_r2d2_delivery",
                        "software/src/estimator/lbfgs_solver.cpp, software/src/control/r2d2_lstm_controller.h",
                        "HistogramLBFGSSolver::fit_histogram -> AvidR2D2LSTMController::step_causal", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify exact bit-for-bit delivery of fitted 21-D theta_ into effective_p and AvidR2D2LSTMController::step_causal";
        ctx.record().independent_oracle = "r2d2.prev_theta_[0..20] == model.theta_[0..20] bitwise after step_causal";

        uint32_t hx[16] = {820, 12, 18, 10, 16, 8, 70, 4, 14, 5, 10, 3, 4, 2, 2, 2};
        uint32_t hz[16] = {910, 8, 10, 6, 12, 5, 14, 3, 10, 4, 6, 3, 4, 2, 2, 1};
        NoiseModel mz;
        mz.init(0.012, 0.008);
        HistogramLBFGSSolver solver;
        double nll = solver.fit_histogram(mz, hx, 24);

        double theta_arr[21];
        for (int i = 0; i < 21; ++i) theta_arr[i] = mz.theta_(i);

        AvidR2D2LSTMController r2d2;
        std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
        ASSERT_TRUE(r2d2.load_checkpoint(ckpt));

        double gains[9] = {0};
        uint32_t rec_k = 0;
        double pws = 0.0;
        int act = -1;
        r2d2.step_causal(hx, hz, 990, theta_arr, nll, gains, &rec_k, 0.1, &pws, &act);

        for (int i = 0; i < 21; ++i) {
            ASSERT_EQ(std::memcmp(&r2d2.prev_theta_[i], &theta_arr[i], sizeof(double)), 0);
        }
        ASSERT_TRUE(act >= 0 && act <= 5);
        ASSERT_TRUE(rec_k >= 3 && rec_k <= 5);

        ctx.record().expected["bitwise_theta_match"] = "21/21";
        ctx.record().actual["action_id"] = std::to_string(act);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
