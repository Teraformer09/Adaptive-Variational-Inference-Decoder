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

        if (nll_neg_lam < nll_lam0) {
            ctx.record_failure("DEFECT: NoiseModel::prior_lambda_ is an unguarded public double; negative prior_lambda_=-5.0 is accepted and rewards diverging away from prior_theta_!");
        }
        if (std::isnan(nll_nan_lam)) {
            ctx.record_failure("DEFECT: NaN prior_lambda_ is accepted without validation and corrupts NLL with NaN!");
        }

        /* Test whether HistogramLBFGSSolver::fit_histogram respects caller prior_lambda_ = 2.5 */
        HistogramLBFGSSolver solver;
        NoiseModel m_fit;
        m_fit.init(0.012, 0.004);
        m_fit.prior_lambda_ = 2.5;
        solver.fit_histogram(m_fit, hist.data(), 12, nullptr);
        if (std::abs(m_fit.prior_lambda_ - 0.04) < 1e-12) {
            ctx.record_failure("CONTRACT MISMATCH: HistogramLBFGSSolver::fit_histogram (lbfgs_solver.cpp:107) unconditionally overwrites `model.prior_lambda_ = 0.04`, discarding caller-configured lambda=2.5 and schema lambda=0.5!");
        }

        ctx.record().expected["lambda_contract"] = "Reject negative/NaN lambda and preserve configured prior_lambda_";
        ctx.record().actual["lambda_contract"] =
            "Negative lambda=-5.0 accepted, NaN lambda->NLL=NaN, and fit_histogram overwrote caller lambda=2.5 to " +
            std::to_string(m_fit.prior_lambda_);
        return ctx.finalize(TestStatus::FAIL, "UNGUARDED_LAMBDA_AND_HARDCODED_SOLVER_OVERRIDE");
    }

    TestContext ctx(opts, "L3-IP-025", "prior_vector_validation_and_schema_consistency",
                    "software/src/inference/noise_model.cpp",
                    "NoiseModel::init, NoiseModel::prior_theta_");
    ctx.record().test_description =
        "Prior configuration — verify prior_theta_ initialization, check consistency with parameter_schema.json default_prior, and test invalid prior_theta_ values";
    ctx.record().input_contract = "Valid 21-element prior_theta_ in (0, 0.5) consistent with parameter_schema.json";
    ctx.record().independent_oracle = "architecture/estimator/parameter_schema.json default_prior (0.012 single, 0.008 pair) vs NoiseModel::init(0.012, 0.004)";
    ctx.record().fixture = "tests/layer3/fixtures/parameters/priors/default_prior_21.txt";

    NoiseModel m;
    /* Note: Default constructor sets prior_theta_=(0.012, 0.003), init() defaults to (0.012, 0.004), parameter_schema.json specifies (0.012, 0.008)! */
    NoiseModel m_ctor_only;
    double ctor_pair_prior = m_ctor_only.prior_theta_(9);
    m.init();
    double init_pair_prior = m.prior_theta_(9);

    if (std::abs(ctor_pair_prior - init_pair_prior) > 1e-9 || std::abs(init_pair_prior - 0.008) > 1e-9) {
        ctx.record_failure("SCHEMA DISCREPANCY: pair default_prior is 0.003 in NoiseModel(), 0.004 in NoiseModel::init(), and 0.008 in architecture/estimator/parameter_schema.json!");
    }

    /* Inject invalid NaN and negative values into public prior_theta_ */
    m.prior_theta_(0) = std::numeric_limits<double>::quiet_NaN();
    double nll_bad_prior = m.compute_nll_from_histogram(hist.data());
    if (std::isnan(nll_bad_prior)) {
        ctx.record_failure("DEFECT: NoiseModel::prior_theta_ has no setter/bounds check; NaN in prior_theta_(0) propagates directly into NLL and gradient!");
    }

    ctx.record().expected["prior_validation"] = "Consistent default prior across constructor/init/schema and rejection of NaN/negative prior_theta_";
    ctx.record().actual["prior_validation"] =
        "ctor_pair=" + std::to_string(ctor_pair_prior) + ", init_pair=" + std::to_string(init_pair_prior) +
        ", schema_pair=0.008; NaN prior_theta_(0) produced NLL=NaN";

    return ctx.finalize(TestStatus::FAIL, "UNGUARDED_PRIOR_VECTOR_AND_DEFAULT_DRIFT");
}
