#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include <array>
#include <cmath>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-013");

    if (opts.test_id == "L3-IP-029") {
        TestContext ctx(opts, "L3-IP-029", "sparse_sample_window_w_min_check",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram");
        ctx.record().test_description =
            "Insufficient sample count — supply sparse 3-round histogram (sum=3 < W_min) and verify whether solver enforces a minimum sample count or overfits 3 noisy syndromes";
        ctx.record().input_contract = "Minimum sample threshold W_min or prior-dominated hold when sum(hist) is ultra-sparse (N=3)";
        ctx.record().independent_oracle = "With N=3 rounds (2 hits in bin 6), prior must either hold/reject sparse window or prevent extreme jump";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/sparse/sparse_total3.txt";

        std::array<uint32_t, 16> sparse_hist{};
        ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/sparse/sparse_total3.txt"), sparse_hist));

        NoiseModel m;
        m.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        double nll = solver.fit_histogram(m, sparse_hist.data(), 20, &diag);

        ASSERT_FINITE(nll);
        double jumped_q4 = m.theta_(4); /* Qubit 4 has X-signature 6 */
        ctx.log("Sparse N=3 fit: converged=" + std::string(diag.converged ? "true" : "false") +
                ", iters=" + std::to_string(diag.iterations_executed) +
                ", theta_(4) jumped from 0.012 to " + std::to_string(jumped_q4));

        /* Because fit_histogram sets prior_lambda_=0.04 unscaled by W, 2 counts in bin 6 drive theta_(4) > 0.30 (>25x jump on 3 samples!) */
        if (diag.converged && jumped_q4 > 0.20) {
            ctx.record_failure("DEFECT: HistogramLBFGSSolver::fit_histogram has no W_min minimum-sample guard (only checks `total == 0`); on just 3 rounds (hist[6]=2, hist[0]=1), it reports converged=true and swings theta_(4) from 0.012 to " +
                               std::to_string(jumped_q4) + "!");
        }

        ctx.record().expected["sparse_sample_behavior"] = "Enforce W_min hold/fallback or prior-dominated bound on N=3 samples";
        ctx.record().actual["sparse_sample_behavior"] =
            "No W_min check (total==0 only); converged=" + std::string(diag.converged ? "true" : "false") +
            ", theta_(4)=" + std::to_string(jumped_q4);
        return ctx.finalize(TestStatus::FAIL, "MISSING_MINIMUM_SAMPLE_GUARD_SPARSE_OVERFIT");
    }

    TestContext ctx(opts, "L3-IP-013", "empty_zero_count_histogram_handling",
                    "software/src/inference/noise_model.cpp, software/src/estimator/lbfgs_solver.cpp",
                    "NoiseModel::compute_nll_from_histogram, HistogramLBFGSSolver::fit_histogram, NoiseModel::compute_nll(ring_buffer_t)");
    ctx.record().test_description =
        "Empty histogram — supply all-zero 16-bin histogram (total=0) and empty ring_buffer_t (count=0) and verify zero divide-by-zero, finite prior NLL, and preserved parameters";
    ctx.record().input_contract = "All-zero hist[16] and empty ring_buffer_t";
    ctx.record().independent_oracle = "total==0 -> finite prior-only NLL (>=0), zero gradient at theta==prior_theta, converged=false, theta unchanged";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/empty/zeros_16.txt";

    std::array<uint32_t, 16> zero_hist{};
    ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/empty/zeros_16.txt"), zero_hist));

    NoiseModel m;
    m.init(0.012, 0.004);
    NoiseModel::ParamVec before = m.theta_;

    double nll0 = m.compute_nll_from_histogram(zero_hist.data());
    ASSERT_FINITE(nll0);
    ASSERT_NEAR(nll0, 0.0, 1e-12);

    m.compute_gradient_from_histogram(zero_hist.data());
    for (int i = 0; i < 21; ++i) {
        ASSERT_FINITE(m.grad()(i));
        ASSERT_NEAR(m.grad()(i), 0.0, 1e-12);
    }

    static ring_buffer_t empty_rb;
    ring_init(&empty_rb);
    double rb_nll0 = m.compute_nll(empty_rb);
    ASSERT_FINITE(rb_nll0);
    ASSERT_NEAR(rb_nll0, 0.0, 1e-12);

    HistogramLBFGSSolver solver;
    LBFGSFitDiagnostics diag{};
    double fit_nll0 = solver.fit_histogram(m, zero_hist.data(), 20, &diag);
    ASSERT_FINITE(fit_nll0);
    ASSERT_FALSE(diag.converged);
    ASSERT_EQ(diag.iterations_executed, 0);
    for (int i = 0; i < 21; ++i) {
        ASSERT_NEAR(m.theta_(i), before(i), 1e-12);
    }

    ctx.record().expected["empty_histogram"] = "Finite NLL=0, converged=false, iterations_executed=0, theta unchanged";
    ctx.record().actual["empty_histogram"] =
        "nll=" + std::to_string(fit_nll0) + ", converged=" + (diag.converged ? "true" : "false") +
        ", iterations=" + std::to_string(diag.iterations_executed);

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
