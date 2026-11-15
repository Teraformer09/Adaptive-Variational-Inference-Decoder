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
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-025");
    std::array<uint32_t, 16> hist{};
    load_u32_histogram_fixture(
        (opts.repo_root.empty() ? "/tmp/avid_layer3_work" : opts.repo_root) +
        "/tests/layer3/fixtures/histograms/correlated/corr_total1000.txt", hist);

    if (opts.test_id == "L3-IP-026") {
        TestContext ctx(opts, "L3-IP-026", "regularization_lambda_validation_and_override",
                        "software/src/inference/noise_model.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "NoiseModel::prior_lambda_, HistogramLBFGSSolver::fit_histogram");
        ctx.record().test_description =
            "Regularization / prior strength — test lambda=0, lambda>0, lambda<0, NaN, and inspect whether HistogramLBFGSSolver honors caller prior_lambda_";
        ctx.record().input_contract = "prior_lambda_ >= 0 finite; negative/NaN rejected; caller-configured lambda respected or contract discrepancy documented";
        ctx.record().independent_oracle = "architecture/estimator/parameter_schema.json prior_tether_lambda=0.5 vs NoiseModel(0.5) vs fit_histogram(0.04)";
        ctx.record().fixture = "tests/layer3/fixtures/histograms/correlated/corr_total1000.txt";

        NoiseModel m;
        m.init(0.020, 0.006);
        ASSERT_NEAR(m.prior_lambda_, 0.5, 1e-12); /* Constructor default matches parameter_schema.json */

        /* Test lambda = 0 vs lambda = 2.0 in NoiseModel::compute_nll_from_histogram when theta != prior_theta */
        m.theta_(0) = 0.12;
        m.prior_lambda_ = 0.0;
        double nll_lam0 = m.compute_nll_from_histogram(hist.data());
        m.prior_lambda_ = 2.0;
        double nll_lam2 = m.compute_nll_from_histogram(hist.data());
        ASSERT_TRUE(nll_lam2 > nll_lam0);

        /* Test negative lambda = -5.0 and NaN lambda */
        m.prior_lambda_ = -5.0;
        double nll_neg_lam = m.compute_nll_from_histogram(hist.data());
        m.prior_lambda_ = std::numeric_limits<double>::quiet_NaN();
        double nll_nan_lam = m.compute_nll_from_histogram(hist.data());

        ASSERT_TRUE(std::isinf(nll_neg_lam));
        ASSERT_FALSE(std::isnan(nll_neg_lam));
        ASSERT_TRUE(std::isinf(nll_nan_lam));
        ASSERT_FALSE(std::isnan(nll_nan_lam));

        /* Test that HistogramLBFGSSolver::fit_histogram preserves caller prior_lambda_ = 2.5 */
        HistogramLBFGSSolver solver;
        NoiseModel m_fit;
        m_fit.init(0.012, 0.004);
        m_fit.prior_lambda_ = 2.5;
        LBFGSFitDiagnostics diag_fit{};
        double nll_fit = solver.fit_histogram(m_fit, hist.data(), 12, &diag_fit);
        ASSERT_FINITE(nll_fit);
        ASSERT_NEAR(m_fit.prior_lambda_, 2.5, 1e-12);

        /* Also verify solver rejects negative prior_lambda_ */
        NoiseModel m_bad_lam;
        m_bad_lam.init(0.012, 0.004);
        m_bad_lam.prior_lambda_ = -1.0;
        LBFGSFitDiagnostics diag_bad{};
        double nll_bad_fit = solver.fit_histogram(m_bad_lam, hist.data(), 12, &diag_bad);
        ASSERT_TRUE(std::isinf(nll_bad_fit));
        ASSERT_FALSE(diag_bad.converged);

        ctx.record().expected["lambda_contract"] = "Reject negative/NaN lambda and preserve configured prior_lambda_";
        ctx.record().actual["lambda_contract"] =
            "Negative (-5.0, -1.0) and NaN prior_lambda_ rejected (+Inf); caller prior_lambda_=2.5 preserved (" +
            std::to_string(m_fit.prior_lambda_) + ")";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    TestContext ctx(opts, "L3-IP-025", "prior_vector_validation_and_schema_consistency",
                    "software/src/inference/noise_model.cpp",
                    "NoiseModel::init, NoiseModel::prior_theta_");
    ctx.record().test_description =
        "Prior configuration — verify prior_theta_ initialization, check consistency with parameter_schema.json default_prior, and test invalid prior_theta_ values";
    ctx.record().input_contract = "Valid 21-element prior_theta_ in (0, 0.5) consistent with parameter_schema.json";
    ctx.record().independent_oracle = "architecture/estimator/parameter_schema.json default_prior (0.012 single, 0.008 pair)";
    ctx.record().fixture = "tests/layer3/fixtures/parameters/priors/default_prior_21.txt";

    NoiseModel m;
    NoiseModel m_ctor_only;
    double ctor_single_prior = m_ctor_only.prior_theta_(0);
    double ctor_pair_prior = m_ctor_only.prior_theta_(9);
    m.init();
    double init_single_prior = m.prior_theta_(0);
    double init_pair_prior = m.prior_theta_(9);

    ASSERT_NEAR(ctor_single_prior, 0.012, 1e-12);
    ASSERT_NEAR(init_single_prior, 0.012, 1e-12);
    ASSERT_NEAR(ctor_pair_prior, 0.008, 1e-12);
    ASSERT_NEAR(init_pair_prior, 0.008, 1e-12);

    /* Inject invalid NaN and negative values into public prior_theta_ -> must be rejected cleanly */
    m.prior_theta_(0) = std::numeric_limits<double>::quiet_NaN();
    double nll_bad_prior = m.compute_nll_from_histogram(hist.data());
    ASSERT_TRUE(std::isinf(nll_bad_prior));
    ASSERT_FALSE(std::isnan(nll_bad_prior));

    m.init();
    m.prior_theta_(9) = -0.05;
    double nll_neg_prior = m.compute_nll_from_histogram(hist.data());
    ASSERT_TRUE(std::isinf(nll_neg_prior));
    ASSERT_FALSE(std::isnan(nll_neg_prior));

    ctx.record().expected["prior_validation"] = "Consistent default prior (0.012, 0.008) across constructor/init/schema and rejection of NaN/negative prior_theta_";
    ctx.record().actual["prior_validation"] =
        "ctor_pair=" + std::to_string(ctor_pair_prior) + ", init_pair=" + std::to_string(init_pair_prior) +
        " (matches schema 0.008); NaN and negative prior_theta_ rejected cleanly (+Inf)";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
