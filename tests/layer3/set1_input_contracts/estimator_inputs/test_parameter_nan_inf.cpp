#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include <array>
#include <cmath>
#include <limits>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-024");
    std::array<uint32_t, 16> hist{};
    load_u32_histogram_fixture(
        (opts.repo_root.empty() ? "/tmp/avid_layer3_work" : opts.repo_root) +
        "/tests/layer3/fixtures/histograms/correlated/corr_total1000.txt", hist);

    if (opts.test_id == "L3-IP-028") {
        TestContext ctx(opts, "L3-IP-028", "out_of_range_and_nan_initial_guess",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram");
        ctx.record().test_description =
            "Out-of-range / NaN initial guess — supply negative (-0.1), >0.5 (0.85), and NaN initial theta^(0) into HistogramLBFGSSolver::fit_histogram";
        ctx.record().input_contract = "Solver must reject invalid/non-finite initial guess or document deterministic sanitization without NaN corruption";
        ctx.record().independent_oracle = "validate_parameter_bounds(theta^(0)) == false -> explicit error or NaN-free safe handling";
        ctx.record().fixture = "tests/layer3/fixtures/parameters/invalid/out_of_range_21.txt";

        HistogramLBFGSSolver solver;

        /* Case A: Negative and >0.5 initial guess -> rejected cleanly (+Inf, converged=false, 0 iters) */
        NoiseModel m_oob;
        m_oob.init(0.012, 0.004);
        m_oob.theta_(0) = -0.10;
        m_oob.theta_(1) = 0.85;
        LBFGSFitDiagnostics diag_oob{};
        double nll_oob = solver.fit_histogram(m_oob, hist.data(), 12, &diag_oob);
        ASSERT_TRUE(std::isinf(nll_oob));
        ASSERT_FALSE(std::isnan(nll_oob));
        ASSERT_FALSE(diag_oob.converged);
        ASSERT_EQ(diag_oob.iterations_executed, 0);

        /* Case B: NaN in initial guess -> rejected cleanly (+Inf, converged=false, 0 iters, zero NaN propagation) */
        NoiseModel m_nan;
        m_nan.init(0.012, 0.004);
        m_nan.theta_(0) = std::numeric_limits<double>::quiet_NaN();
        LBFGSFitDiagnostics diag_nan{};
        double nll_nan = solver.fit_histogram(m_nan, hist.data(), 12, &diag_nan);
        ASSERT_TRUE(std::isinf(nll_nan));
        ASSERT_FALSE(std::isnan(nll_nan));
        ASSERT_FALSE(diag_nan.converged);
        ASSERT_EQ(diag_nan.iterations_executed, 0);

        ctx.record().expected["nan_initial_guess"] = "Explicit rejection (+Inf, converged=false, 0 iters) without NaN propagation";
        ctx.record().actual["nan_initial_guess"] =
            "Both OOB (-0.10, 0.85) and NaN initial guesses rejected cleanly (+Inf, converged=false, iterations_executed=0)";

        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    TestContext ctx(opts, "L3-IP-024", "invalid_parameter_values_nan_inf_oob",
                    "software/src/inference/noise_model.cpp",
                    "NoiseModel::validate_parameter_bounds, NoiseModel::compute_joint_distribution, NoiseModel::compute_nll_from_histogram");
    ctx.record().test_description =
        "Invalid parameter values — test -1e-6, 0.51, 1.2, NaN, +Inf, -Inf in validate_parameter_bounds AND direct compute_nll_from_histogram";
    ctx.record().input_contract = "Invalid probabilities (<0, >=0.5, NaN, +-Inf) must be rejected and never propagate NaN into likelihoods";
    ctx.record().independent_oracle = "validate_parameter_bounds == false AND compute_nll_from_histogram guards non-finite parameters";
    ctx.record().fixture = "tests/layer3/fixtures/parameters/invalid/out_of_range_21.txt";

    const double bad_values[6] = {
        -1e-6, 0.51, 1.2,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()
    };

    for (double bv : bad_values) {
        NoiseModel::ParamVec v;
        for (int i = 0; i < 21; ++i) v(i) = 0.012;
        v(0) = bv;
        ASSERT_FALSE(NoiseModel::validate_parameter_bounds(v));
    }

    /* Now test direct evaluation path in NoiseModel::compute_nll_from_histogram */
    NoiseModel m_eval;
    m_eval.init(0.012, 0.004);
    m_eval.theta_(0) = std::numeric_limits<double>::quiet_NaN();
    double nll_after_nan = m_eval.compute_nll_from_histogram(hist.data());
    bool joint_has_nan = std::isnan(m_eval.joint_probs()(0)) || std::isnan(nll_after_nan);
    ASSERT_FALSE(joint_has_nan);
    ASSERT_TRUE(std::isinf(nll_after_nan));

    NoiseModel m_neg;
    m_neg.init(0.012, 0.004);
    m_neg.theta_(0) = -1e-6;
    double nll_neg = m_neg.compute_nll_from_histogram(hist.data());
    ASSERT_TRUE(std::isinf(nll_neg));
    ASSERT_FALSE(std::isnan(nll_neg));

    ctx.record().expected["invalid_param_handling"] = "Rejected cleanly across both validate_parameter_bounds and compute_nll_from_histogram (+Inf, zero NaN propagation)";
    ctx.record().actual["invalid_param_handling"] =
        "validate_parameter_bounds rejects all 6 bad inputs; compute_nll_from_histogram returns +Inf on NaN and -1e-6 without NaN propagation";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
