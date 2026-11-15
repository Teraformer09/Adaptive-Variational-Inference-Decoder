#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include <array>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-023");

    if (opts.test_id == "L3-IP-027") {
        TestContext ctx(opts, "L3-IP-027", "valid_initial_guess_and_optimizer_reset",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram, LBFGSOptimizer::reset");
        ctx.record().test_description =
            "Valid initial guess — initialize theta^(0) in [1e-5, 0.45], run fit_histogram, and verify repeated calls reset history cleanly";
        ctx.record().input_contract = "Valid interior initial parameter vector theta^(0) in [1e-5, 0.45]^21";
        ctx.record().independent_oracle = "Deterministic repeat-run identity proves LBFGSOptimizer::reset() clears s_hist_/y_hist_";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/correlated/corr_total1000.txt";

        std::array<uint32_t, 16> hist{};
        ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/correlated/corr_total1000.txt"), hist));

        HistogramLBFGSSolver solver;
        NoiseModel m1, m2;
        m1.init(0.012, 0.004);
        m2.init(0.012, 0.004);

        LBFGSFitDiagnostics d1{}, d2{};
        double nll1 = solver.fit_histogram(m1, hist.data(), 20, &d1);
        /* Pollute solver with a completely different histogram in between, then re-run on m2 */
        NoiseModel m_mid;
        m_mid.init(0.20, 0.10);
        uint32_t alt_hist[16] = {100, 900, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        solver.fit_histogram(m_mid, alt_hist, 20, nullptr);

        double nll2 = solver.fit_histogram(m2, hist.data(), 20, &d2);

        ASSERT_FINITE(nll1);
        ASSERT_NEAR(nll1, nll2, 1e-12);
        ASSERT_TRUE(d1.final_joint_dp_nll <= d1.initial_nll + 1e-9);
        ASSERT_TRUE(NoiseModel::validate_parameter_bounds(m1.theta_));
        for (int i = 0; i < NM_N_PARAMS; ++i) {
            ASSERT_IN_RANGE(m1.theta_(i), 1e-5, 0.45);
            ASSERT_NEAR(m1.theta_(i), m2.theta_(i), 1e-12);
        }

        ctx.record().expected["fit_status"] = "Finite non-increasing NLL, interior theta in [1e-5, 0.45], bit-identical repeat after reset";
        ctx.record().actual["fit_status"] =
            "initial_nll=" + std::to_string(d1.initial_nll) + ", final_nll=" + std::to_string(nll1) +
            ", repeat_diff=" + std::to_string(std::abs(nll1 - nll2));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    TestContext ctx(opts, "L3-IP-023", "valid_parameter_bounds_and_clamping",
                    "software/src/inference/noise_model.cpp",
                    "NoiseModel::validate_parameter_bounds, NoiseModel::clamp_params");
    ctx.record().test_description =
        "Valid parameter bounds — verify validate_parameter_bounds accepts [0.0, 0.5) and rejects >=0.5/<0, and clamp_params maps to [1e-5, 0.45]";
    ctx.record().input_contract = "Physical Bernoulli error probabilities in [0, 0.5), interior clamp [1e-5, 0.45]";
    ctx.record().independent_oracle = "architecture/estimator/parameter_schema.json prob_bounds=[1e-5, 0.45]";
    ctx.record().fixture = "tests/layer3/fixtures/parameters/boundary/interior_bounds_21.txt";

    std::vector<double> bvec;
    ASSERT_TRUE(load_f64_vector_fixture(ctx.fixture_path("parameters/boundary/interior_bounds_21.txt"), bvec));
    ASSERT_EQ(bvec.size(), static_cast<size_t>(21));

    NoiseModel m;
    for (int i = 0; i < 21; ++i) m.theta_(i) = bvec[i];
    ASSERT_TRUE(NoiseModel::validate_parameter_bounds(m.theta_));

    /* Exact lower boundary 0.0 is accepted by validate_parameter_bounds; 0.5 is rejected */
    NoiseModel::ParamVec zvec;
    zvec.setZero();
    ASSERT_TRUE(NoiseModel::validate_parameter_bounds(zvec));

    NoiseModel::ParamVec half_vec = m.theta_;
    half_vec(0) = 0.5;
    ASSERT_FALSE(NoiseModel::validate_parameter_bounds(half_vec));

    /* Verify clamp_params clamps <1e-5 to 1e-5 and >0.45 to 0.45 */
    m.theta_(0) = 0.0;
    m.theta_(1) = 0.49;
    m.clamp_params();
    ASSERT_NEAR(m.theta_(0), 1e-5, 1e-12);
    ASSERT_NEAR(m.theta_(1), 0.45, 1e-12);

    ctx.record().expected["bounds_validation"] = "[0, 0.5) accepted, 0.5 rejected, clamp_params -> [1e-5, 0.45]";
    ctx.record().actual["bounds_validation"] =
        "clamped[0]=" + std::to_string(m.theta_(0)) + ", clamped[1]=" + std::to_string(m.theta_(1));
    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
