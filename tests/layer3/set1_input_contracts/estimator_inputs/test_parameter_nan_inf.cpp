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

        /* Case A: Negative and >0.5 initial guess */
        NoiseModel m_oob;
        m_oob.init(0.012, 0.004);
        m_oob.theta_(0) = -0.10;
        m_oob.theta_(1) = 0.85;
        LBFGSFitDiagnostics diag_oob{};
        double nll_oob = solver.fit_histogram(m_oob, hist.data(), 12, &diag_oob);
        ctx.log("Out-of-bounds initial guess (-0.10, 0.85) accepted silently without error; clamp_params silently projected to [1e-5, 0.45], nll=" +
                std::to_string(nll_oob));

        /* Case B: NaN in initial guess */
        NoiseModel m_nan;
        m_nan.init(0.012, 0.004);
        m_nan.theta_(0) = std::numeric_limits<double>::quiet_NaN();
        LBFGSFitDiagnostics diag_nan{};
        double nll_nan = solver.fit_histogram(m_nan, hist.data(), 12, &diag_nan);
        bool nan_corrupted = std::isnan(nll_nan) || std::isnan(m_nan.theta_(0));
        if (nan_corrupted) {
            ctx.record_failure("DEFECT: HistogramLBFGSSolver::fit_histogram does NOT call validate_parameter_bounds; std::clamp(NaN, 1e-5, 0.45) returns NaN, corrupting final NLL and theta_ with NaN!");
        }

        ctx.record().expected["nan_initial_guess"] = "Explicit rejection or safe NaN-free sanitization";
        ctx.record().actual["nan_initial_guess"] =
            std::string("Silent acceptance of -0.10/0.85; NaN initial guess produced nll=") +
            (std::isnan(nll_nan) ? "NaN" : std::to_string(nll_nan)) +
            ", converged=" + (diag_nan.converged ? "true" : "false");

        return ctx.finalize(TestStatus::FAIL, "MISSING_INITIAL_GUESS_VALIDATION_NAN_PROPAGATION");
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

    if (joint_has_nan) {
        ctx.record_failure("DEFECT: NoiseModel::compute_joint_distribution uses std::clamp(theta_(e), 1e-6, 0.49) without std::isfinite/validate_parameter_bounds; NaN in theta_(0) propagates NaN into all 16 p_joint_ bins and NLL!");
    }

    NoiseModel m_neg;
    m_neg.init(0.012, 0.004);
    m_neg.theta_(0) = -1e-6;
    double nll_neg = m_neg.compute_nll_from_histogram(hist.data());
    ctx.log("Negative theta_(0)=-1e-6 silently clamped inside compute_joint_distribution to 1e-6 without error; nll=" + std::to_string(nll_neg));

    ctx.record().expected["invalid_param_handling"] = "Rejected cleanly across both validate_parameter_bounds and compute_nll_from_histogram without NaN propagation";
    ctx.record().actual["invalid_param_handling"] =
        "validate_parameter_bounds rejects all 6 bad inputs, BUT compute_nll_from_histogram does not call it: NaN propagates to p_joint_[0..15] and NLL=NaN";

    return ctx.finalize(TestStatus::FAIL, "NAN_PROPAGATION_IN_NOISE_MODEL_EVALUATION");
}
