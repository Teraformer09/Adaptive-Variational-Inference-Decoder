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
            "Insufficient sample count — supply sparse N=3 and boundary N=63 (< W_min=64) vs N=64 (== W_min=64) histograms and verify W_min=64 hold/fit boundary";
        ctx.record().input_contract = "Authoritative minimum sample threshold W_min = 64 (AVID_SOLVER_MIN_WINDOW_SAMPLES = 64u): N < 64 holds prior; N >= 64 executes L-BFGS fit";
        ctx.record().independent_oracle = "N=3 and N=63 -> converged=false, iterations_executed=0, theta==prior; N=64 -> iterations_executed >= 1";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/sparse/sparse_total3.txt";

        ASSERT_EQ(static_cast<uint32_t>(AVID_SOLVER_MIN_WINDOW_SAMPLES), 64u);

        std::array<uint32_t, 16> sparse_hist{};
        ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/sparse/sparse_total3.txt"), sparse_hist));

        /* 1. Ultra-sparse N=3 (< 64): must hold prior */
        NoiseModel m3;
        m3.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag3{};
        double nll3 = solver.fit_histogram(m3, sparse_hist.data(), 20, &diag3);

        ASSERT_FINITE(nll3);
        ASSERT_FALSE(diag3.converged);
        ASSERT_EQ(diag3.iterations_executed, 0);
        double held_q4_3 = m3.theta_(4); /* Qubit 4 has X-signature 6 */
        ASSERT_NEAR(held_q4_3, 0.012, 1e-12);

        /* 2. Exact below-threshold boundary N=63 (W_min - 1): must also hold prior */
        std::array<uint32_t, 16> hist63{};
        hist63[0] = 42u;
        hist63[6] = 21u;
        NoiseModel m63;
        m63.init(0.012, 0.004);
        LBFGSFitDiagnostics diag63{};
        double nll63 = solver.fit_histogram(m63, hist63.data(), 20, &diag63);
        ASSERT_FINITE(nll63);
        ASSERT_FALSE(diag63.converged);
        ASSERT_EQ(diag63.iterations_executed, 0);
        ASSERT_NEAR(m63.theta_(4), 0.012, 1e-12);

        /* 3. Exact threshold boundary N=64 (W_min): must execute L-BFGS optimization */
        std::array<uint32_t, 16> hist64{};
        hist64[0] = 42u;
        hist64[6] = 22u;
        NoiseModel m64;
        m64.init(0.012, 0.004);
        LBFGSFitDiagnostics diag64{};
        double nll64 = solver.fit_histogram(m64, hist64.data(), 20, &diag64);
        ASSERT_FINITE(nll64);
        ASSERT_TRUE(diag64.iterations_executed >= 1);
        ASSERT_TRUE(std::fabs(m64.theta_(4) - 0.012) > 1e-4);

        ctx.log("Verified W_min=64 boundary: N=3 held (iters=0), N=63 held (iters=0), N=64 fitted (iters=" +
                std::to_string(diag64.iterations_executed) + ", theta_(4)=" + std::to_string(m64.theta_(4)) + ")");

        ctx.record().expected["sparse_sample_behavior"] =
            "Enforce authoritative W_min=64 hold/fallback on N=3 and N=63 samples, and fit on N=64";
        ctx.record().actual["sparse_sample_behavior"] =
            "W_min=64 enforced; N=3 iters=" + std::to_string(diag3.iterations_executed) +
            ", N=63 iters=" + std::to_string(diag63.iterations_executed) +
            " (theta_(4)=0.012 held), N=64 iters=" + std::to_string(diag64.iterations_executed) +
            " (theta_(4)=" + std::to_string(m64.theta_(4)) + ")";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
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
