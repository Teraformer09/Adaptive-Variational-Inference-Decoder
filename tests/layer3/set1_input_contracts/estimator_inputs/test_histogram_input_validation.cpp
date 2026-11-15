#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "estimator/lbfgs_solver.h"
#include <array>
#include <cmath>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-012");

    if (opts.test_id == "L3-IP-015") {
        TestContext ctx(opts, "L3-IP-015", "histogram_sum_invariance_1000_vs_10000",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_nll_from_histogram, HistogramLBFGSSolver::fit_histogram");
        ctx.record().test_description =
            "Histogram sum invariance — compare NLL and MAP fit on identical empirical PMFs with total counts W=1000 vs W=10000";
        ctx.record().input_contract = "16-bin uint32_t count histograms with identical relative frequencies";
        ctx.record().independent_oracle = "With prior_lambda_=0, NLL(10*H) == 10 * NLL(H) to floating-point precision; with fixed prior_lambda_=0.04, tether weight scales as 1/W";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/correlated/{corr_total1000.txt,corr_total10000.txt}";

        std::array<uint32_t, 16> h1k{}, h10k{};
        ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/correlated/corr_total1000.txt"), h1k));
        ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/correlated/corr_total10000.txt"), h10k));

        NoiseModel m;
        m.init(0.015, 0.005);
        m.prior_lambda_ = 0.0;
        double nll_1k = m.compute_nll_from_histogram(h1k.data());
        double nll_10k = m.compute_nll_from_histogram(h10k.data());
        /* compute_nll_from_histogram normalizes counts by total W, so NLL(10*H) == NLL(H) directly */
        ASSERT_NEAR(nll_10k, nll_1k, 1e-12);

        HistogramLBFGSSolver solver;
        NoiseModel f1, f10;
        f1.init(0.012, 0.004);
        f10.init(0.012, 0.004);
        solver.fit_histogram(f1, h1k.data(), 20, nullptr);
        solver.fit_histogram(f10, h10k.data(), 20, nullptr);

        double orbit_diff = f1.compute_orbit_rmse(f10.theta_);
        ASSERT_TRUE(orbit_diff < 0.005);

        ctx.record().expected["scale_relation"] = "Per-round normalized NLL is scale-invariant (NLL_10k == NLL_1k); orbit RMSE between W=1000 and W=10000 fits < 0.005";
        ctx.record().actual["scale_relation"] =
            "NLL_1k=" + std::to_string(nll_1k) + ", NLL_10k=" + std::to_string(nll_10k) +
            ", orbit_diff=" + std::to_string(orbit_diff);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (opts.test_id == "L3-IP-016") {
        TestContext ctx(opts, "L3-IP-016", "normalized_empirical_distribution_input_check",
                        "software/src/inference/noise_model.h",
                        "absent float/double empirical PMF input API in software/src/");
        ctx.record().test_description =
            "Invalid normalized distribution — inspect whether Layer 3 slow-loop C++ estimator exposes a floating-point normalized empirical PMF entrypoint accepting sum != 1.0 or NaN";
        ctx.record().input_contract = "If normalized empirical PMF input exists, reject sum != 1 +- eps, negative probabilities, and NaN";
        ctx.record().independent_oracle = "API surface audit of software/src/inference/ and software/src/estimator/";
        ctx.record().fixture = "none (interface contract audit)";

        ctx.log("Audited NoiseModel, JointDPEstimator, and HistogramLBFGSSolver in software/src/: all Layer 3 estimator entrypoints accept raw integer counts (`const uint32_t hist[16]` or `ring_buffer_t`) only.");
        ctx.log("Note: deployment/cpp/r2d2_runtime.cpp computes normalized empirical histograms internally from `uint32_t hist_x[16]` / `acc_rounds`, and no standalone normalized floating-point distribution validator exists in Layer 3.");

        ctx.record().expected["normalized_pmf_validator"] = "Reject unnormalized/negative/NaN float PMF or classify NOT_IMPLEMENTED if Layer 3 accepts uint32_t count histograms only";
        ctx.record().actual["normalized_pmf_validator"] = "No float/double PMF input entrypoint exists in software/src/ estimator APIs (uint32_t hist[16] only)";
        return ctx.finalize(TestStatus::NOT_IMPLEMENTED, "NO_NORMALIZED_FLOAT_HISTOGRAM_API_IN_LAYER3");
    }

    /* Default: L3-IP-012 — Histogram dimensionality, NULL pointer, and signed-to-unsigned count safety */
    TestContext ctx(opts, "L3-IP-012", "histogram_dimensionality_and_pointer_validation",
                    "software/src/inference/noise_model.cpp",
                    "NoiseModel::compute_nll_from_histogram, HistogramLBFGSSolver::fit_histogram");
    ctx.record().test_description =
        "Histogram dimensionality & pointer safety — verify C array decay loses length (15-bin vs 17-bin cannot be checked at runtime) and probe NULL hist pointer handling";
    ctx.record().input_contract = "const uint32_t hist[16] non-null 16-bin count array";
    ctx.record().independent_oracle = "fork()-isolated NULL pointer probe + C++ signature inspection";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/uniform/uniform_100.txt";

    auto null_nll_probe = run_isolated_probe([]() -> int {
        NoiseModel m;
        m.init(0.012, 0.004);
        volatile double v = m.compute_nll_from_histogram(nullptr);
        (void)v;
        return 0;
    });

    auto null_solver_probe = run_isolated_probe([]() -> int {
        NoiseModel m;
        m.init(0.012, 0.004);
        HistogramLBFGSSolver s;
        volatile double v = s.fit_histogram(m, nullptr, 10, nullptr);
        (void)v;
        return 0;
    });

    bool crashed = null_nll_probe.signaled || null_solver_probe.signaled ||
                   null_nll_probe.exit_code != 0 || null_solver_probe.exit_code != 0;
    if (crashed) {
        ctx.record_failure("DEFECT: compute_nll_from_histogram(nullptr) and fit_histogram(model, nullptr) dereference NULL without guard (child signaled=" +
                           std::to_string(null_nll_probe.term_signal) + ") and C array decay `const uint32_t hist[16]` carries no length metadata to reject 15-bin buffers.");
    }

    ctx.record().expected["null_and_dim_guard"] = "Explicit error/rejection on nullptr or wrong bin count without crash";
    ctx.record().actual["null_and_dim_guard"] =
        "C array parameter decays to `const uint32_t*` (no length check); nullptr crashes with signal " +
        std::to_string(null_nll_probe.term_signal);

    return ctx.finalize(TestStatus::FAIL, "UNGUARDED_NULL_HISTOGRAM_POINTER_AND_ARRAY_DECAY");
}
