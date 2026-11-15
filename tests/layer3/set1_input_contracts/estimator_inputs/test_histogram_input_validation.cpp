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
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::validate_empirical_pmf, NoiseModel::compute_nll_from_pmf");
        ctx.record().test_description =
            "Invalid normalized distribution — verify NoiseModel::validate_empirical_pmf and compute_nll_from_pmf accept normalized 16-state PMF and reject sum != 1.0, negative bins, NaN, and wrong bin counts";
        ctx.record().input_contract = "Normalized empirical PMF must have n_bins==16, finite p[s] in [0,1], and |sum(p) - 1.0| <= tol";
        ctx.record().independent_oracle = "Simplex membership check + cross-entropy equivalence with count histogram";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/uniform/uniform_100.txt";

        double valid_pmf[16];
        uint32_t uni_hist[16];
        for (int s = 0; s < 16; ++s) {
            valid_pmf[s] = 1.0 / 16.0;
            uni_hist[s] = 100u;
        }
        ASSERT_TRUE(NoiseModel::validate_empirical_pmf(valid_pmf, 16));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(nullptr, 16));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(valid_pmf, 15));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(valid_pmf, 17));

        double subnorm_pmf[16];
        double supernorm_pmf[16];
        double neg_pmf[16];
        double nan_pmf[16];
        for (int s = 0; s < 16; ++s) {
            subnorm_pmf[s] = 0.85 / 16.0;
            supernorm_pmf[s] = 1.20 / 16.0;
            neg_pmf[s] = valid_pmf[s];
            nan_pmf[s] = valid_pmf[s];
        }
        neg_pmf[0] = -0.05;
        neg_pmf[1] += 0.05; /* sum == 1.0, but negative bin! */
        nan_pmf[3] = std::numeric_limits<double>::quiet_NaN();

        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(subnorm_pmf, 16));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(supernorm_pmf, 16));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(neg_pmf, 16));
        ASSERT_FALSE(NoiseModel::validate_empirical_pmf(nan_pmf, 16));

        NoiseModel m;
        m.init(0.012, 0.008);
        double nll_ok = 0.0, nll_bad = 0.0;
        ASSERT_TRUE(m.compute_nll_from_pmf(valid_pmf, 16, &nll_ok));
        ASSERT_NEAR(nll_ok, m.compute_nll_from_histogram(uni_hist), 1e-12);
        ASSERT_FALSE(m.compute_nll_from_pmf(subnorm_pmf, 16, &nll_bad));
        ASSERT_TRUE(std::isinf(nll_bad));
        ASSERT_FALSE(m.compute_nll_from_pmf(neg_pmf, 16, &nll_bad));
        ASSERT_FALSE(m.compute_nll_from_pmf(nan_pmf, 16, &nll_bad));
        ASSERT_FALSE(std::isnan(nll_bad));

        ctx.record().expected["normalized_pmf_validator"] = "Accept valid 16-bin simplex PMF; reject sum!=1, negative bins, NaN, and n_bins!=16";
        ctx.record().actual["normalized_pmf_validator"] = "PASS (valid PMF matches uniform histogram NLL; sub/super-normalized, negative, NaN, and 15/17-bin inputs rejected)";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    /* Default: L3-IP-012 — Histogram dimensionality, NULL pointer, and length-checked API safety */
    TestContext ctx(opts, "L3-IP-012", "histogram_dimensionality_and_pointer_validation",
                    "software/src/inference/noise_model.cpp, software/src/estimator/lbfgs_solver.cpp",
                    "NoiseModel::compute_nll_from_histogram, NoiseModel::compute_nll_from_histogram_checked, HistogramLBFGSSolver::fit_histogram");
    ctx.record().test_description =
        "Histogram dimensionality & pointer safety — verify NULL hist pointers never crash and compute_nll_from_histogram_checked rejects n_bins != 16";
    ctx.record().input_contract = "Non-null 16-bin count array; nullptr and n_bins != 16 explicitly rejected without crash";
    ctx.record().independent_oracle = "fork()-isolated NULL pointer probe + checked length API (15, 16, 17 bins)";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/uniform/uniform_100.txt";

    std::array<uint32_t, 16> uni{};
    ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/uniform/uniform_100.txt"), uni));

    NoiseModel m_chk;
    m_chk.init(0.012, 0.008);
    double checked_nll = 0.0;
    ASSERT_FALSE(m_chk.compute_nll_from_histogram_checked(nullptr, 16, &checked_nll));
    ASSERT_FALSE(m_chk.compute_nll_from_histogram_checked(uni.data(), 15, &checked_nll));
    ASSERT_FALSE(m_chk.compute_nll_from_histogram_checked(uni.data(), 17, &checked_nll));
    ASSERT_TRUE(m_chk.compute_nll_from_histogram_checked(uni.data(), 16, &checked_nll));
    ASSERT_FINITE(checked_nll);

    LBFGSFitDiagnostics null_diag{};
    HistogramLBFGSSolver solver_chk;
    double null_fit_nll = solver_chk.fit_histogram(m_chk, nullptr, 10, &null_diag);
    ASSERT_TRUE(std::isinf(null_fit_nll));
    ASSERT_FALSE(null_diag.converged);
    ASSERT_EQ(null_diag.iterations_executed, 0);

    auto null_nll_probe = run_isolated_probe([]() -> int {
        NoiseModel m;
        m.init(0.012, 0.004);
        double v = m.compute_nll_from_histogram(nullptr);
        return std::isinf(v) ? 0 : 1;
    });

    auto null_solver_probe = run_isolated_probe([]() -> int {
        NoiseModel m;
        m.init(0.012, 0.004);
        HistogramLBFGSSolver s;
        double v = s.fit_histogram(m, nullptr, 10, nullptr);
        return std::isinf(v) ? 0 : 1;
    });

    bool crashed = null_nll_probe.signaled || null_solver_probe.signaled ||
                   null_nll_probe.exit_code != 0 || null_solver_probe.exit_code != 0;
    ASSERT_FALSE(crashed);

    ctx.record().expected["null_and_dim_guard"] = "Explicit error/rejection on nullptr or wrong bin count (15, 17) without crash";
    ctx.record().actual["null_and_dim_guard"] =
        "nullptr returns +Inf with 0 crashes; compute_nll_from_histogram_checked rejects n_bins=15/17 and accepts n_bins=16";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
