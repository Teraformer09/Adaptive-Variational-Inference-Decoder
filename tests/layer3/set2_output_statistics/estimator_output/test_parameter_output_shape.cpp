#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include "estimator/lbfgs_solver.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>

using namespace avid::layer3_test;

namespace {

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const uint32_t valid_hist[16] = {
        790, 22, 34, 18, 28, 12, 48, 6, 18, 4, 9, 3, 4, 2, 1, 1
    };

    if (tid == "L3-OP-001") {
        TestContext ctx(opts, tid, "map_estimator_output_dimensions",
                        "software/src/inference/noise_model.h, software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify fitted parameter vector length == 21 (9 single-qubit [0..8] + 12 crosstalk pair [9..20])";
        ctx.record().independent_oracle = "params().size() == 21, NM_N_QUBIT == 9, NM_N_CROSSTALK == 12";

        NoiseModel m;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        solver.fit_histogram(m, valid_hist, 20, &diag);
        ASSERT_EQ(static_cast<int>(m.params().size()), 21);
        ASSERT_EQ(NM_N_QUBIT, 9);
        ASSERT_EQ(NM_N_CROSSTALK, 12);
        ASSERT_EQ(NM_N_PARAMS, 21);
        ctx.record().expected["param_dim"] = 21;
        ctx.record().actual["param_dim"] = static_cast<int>(m.params().size());
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-002") {
        TestContext ctx(opts, tid, "map_estimator_output_finiteness_20_histograms",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Assert all 21 theta_(i) and 21 grad_(i) are finite across 20 diverse histograms (sparse, uniform, hotspot)";
        ctx.record().independent_oracle = "840/840 finite parameter and gradient entries";

        std::mt19937_64 rng(2026111602ULL);
        int finite_checks = 0;
        HistogramLBFGSSolver solver;
        for (int c = 0; c < 20; ++c) {
            uint32_t h[16] = {0};
            if (c == 0) {
                h[0] = 64; /* pure quiet W_min */
            } else if (c == 1) {
                for (int s = 0; s < 16; ++s) h[s] = 16; /* uniform */
            } else if (c == 2) {
                h[0] = 40; h[6] = 216; /* extreme hotspot */
            } else {
                h[0] = 200 + static_cast<uint32_t>(rng() % 500);
                for (int s = 1; s < 16; ++s) h[s] = static_cast<uint32_t>(rng() % 40);
            }
            NoiseModel m;
            m.init(0.012, 0.008);
            LBFGSFitDiagnostics diag{};
            double nll = solver.fit_histogram(m, h, 20, &diag);
            ASSERT_TRUE(std::isfinite(nll));
            for (int e = 0; e < 21; ++e) {
                ASSERT_TRUE(std::isfinite(m.theta_(e)));
                ASSERT_TRUE(std::isfinite(m.grad()(e)));
                finite_checks += 2;
            }
        }
        ASSERT_EQ(finite_checks, 840);
        ctx.record().expected["finite_entries"] = 840;
        ctx.record().actual["finite_entries"] = finite_checks;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-003") {
        TestContext ctx(opts, tid, "map_estimator_box_bounds_contract",
                        "software/src/inference/noise_model.cpp",
                        "HistogramLBFGSSolver::fit_histogram, NoiseModel::clamp_params", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Assert every fitted theta_(i) lies inside solver box [1e-5, 0.45] across quiet, nominal, and uniform histograms";
        ctx.record().independent_oracle = "1e-5 <= theta_(i) <= 0.45 (< 0.5) for all i in 0..20";

        const uint32_t h_quiet[16] = {512, 0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
        const uint32_t h_unif[16]  = {64,64,64,64, 64,64,64,64, 64,64,64,64, 64,64,64,64};
        HistogramLBFGSSolver solver;
        for (const uint32_t *hp : {valid_hist, h_quiet, h_unif}) {
            NoiseModel m;
            solver.fit_histogram(m, hp, 25, nullptr);
            for (int e = 0; e < 21; ++e) {
                ASSERT_GE(m.theta_(e), 1e-5);
                ASSERT_LE(m.theta_(e), 0.45);
            }
        }
        ctx.record().expected["box_range"] = "[1e-5, 0.45]";
        ctx.record().actual["box_range"] = "Verified across quiet, nominal, and uniform spectra";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-004") {
        TestContext ctx(opts, tid, "map_estimator_objective_output_consistency",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify returned NLL >= 0.0, finite, and matches diag.final_joint_dp_nll";
        ctx.record().independent_oracle = "returned_nll == diag.final_joint_dp_nll >= 0.0";

        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.04;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        double ret_nll = solver.fit_histogram(m, valid_hist, 20, &diag);
        ASSERT_TRUE(std::isfinite(ret_nll));
        ASSERT_GT(ret_nll, 0.0);
        ASSERT_NEAR(ret_nll, diag.final_joint_dp_nll, 1e-14);
        /* Note Pass 2 adjusts coupled prior_theta_; recomputing at current model state also matches */
        double recomputed = m.compute_nll_from_histogram(valid_hist);
        ASSERT_NEAR(ret_nll, recomputed, 1e-12);
        ctx.record().expected["nll_consistency"] = "ret_nll == diag.final_joint_dp_nll == recomputed_nll";
        ctx.record().actual["final_nll"] = ret_nll;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-005") {
        TestContext ctx(opts, tid, "map_estimator_gradient_output_contract",
                        "software/src/inference/lbfgs_optimizer.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Inspect model.grad() after fit_histogram: dimension 21, finite, and projected infinity norm == diag.final_grad_inf_norm";
        ctx.record().independent_oracle = "|proj_grad_inf - diag.final_grad_inf_norm| < 1e-12";

        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.04;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        solver.fit_histogram(m, valid_hist, 20, &diag);
        ASSERT_EQ(static_cast<int>(m.grad().size()), 21);

        double proj_inf = 0.0;
        for (int e = 0; e < 21; ++e) {
            double ge = m.grad()(e);
            ASSERT_TRUE(std::isfinite(ge));
            if ((m.theta_(e) <= 1e-5 + 1e-9 && ge > 0.0) || (m.theta_(e) >= 0.45 - 1e-9 && ge < 0.0)) {
                ge = 0.0;
            }
            proj_inf = std::max(proj_inf, std::abs(ge));
        }
        ASSERT_NEAR(proj_inf, diag.final_grad_inf_norm, 1e-12);
        ctx.record().expected["proj_grad_inf_match"] = "< 1e-12";
        ctx.record().actual["final_grad_inf_norm"] = diag.final_grad_inf_norm;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-006") {
        TestContext ctx(opts, tid, "map_estimator_convergence_flag_semantics",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify LBFGSFitDiagnostics.converged == true on valid convergent fit vs false on max_iters=0 and empty histogram";
        ctx.record().independent_oracle = "valid->true, max_iters=0->false, empty_hist->false";

        HistogramLBFGSSolver solver;
        NoiseModel m1, m2, m3;
        LBFGSFitDiagnostics d1{}, d2{}, d3{};
        solver.fit_histogram(m1, valid_hist, 20, &d1);
        ASSERT_TRUE(d1.converged);

        solver.fit_histogram(m2, valid_hist, 0, &d2);
        ASSERT_FALSE(d2.converged);

        const uint32_t zero_h[16] = {0};
        solver.fit_histogram(m3, zero_h, 20, &d3);
        ASSERT_FALSE(d3.converged);
        ASSERT_EQ(d3.final_joint_dp_nll, 0.0);

        ctx.record().expected["converged_flags"] = "true, false, false";
        ctx.record().actual["converged_flags"] = "true, false, false";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-007") {
        TestContext ctx(opts, tid, "map_estimator_iteration_accounting",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Run fit_histogram with max_iters in {1, 6, 20} and verify 0 <= iterations_executed <= max_iters";
        ctx.record().independent_oracle = "Exact cap enforcement across {1, 6, 20}";

        HistogramLBFGSSolver solver;
        for (int cap : {1, 6, 20}) {
            NoiseModel m;
            m.init(0.15, 0.08);
            LBFGSFitDiagnostics d{};
            solver.fit_histogram(m, valid_hist, cap, &d);
            ASSERT_GE(d.iterations_executed, 1);
            ASSERT_LE(d.iterations_executed, cap);
        }
        ctx.record().expected["bounds"] = "1 <= iterations_executed <= cap";
        ctx.record().actual["bounds"] = "Verified for {1, 6, 20}";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-008") {
        TestContext ctx(opts, tid, "map_estimator_initial_vs_final_objective",
                        "software/src/inference/lbfgs_optimizer.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "LBFGSOptimizer::step_histogram, HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify monotonic objective reduction f_final <= f_initial across single-pass LBFGSOptimizer and two-pass HistogramLBFGSSolver";
        ctx.record().independent_oracle = "f_final < f_initial on mismatched initialization";

        NoiseModel m_opt;
        m_opt.init(0.12, 0.06);
        double f0 = m_opt.compute_nll_from_histogram(valid_hist);
        LBFGSOptimizer opt;
        double f1 = f0, ginf = 0.0;
        opt.step_histogram(m_opt, valid_hist, 20, &f1, &ginf);
        ASSERT_LT(f1, f0);

        NoiseModel m_solv;
        m_solv.init(0.12, 0.06);
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics d{};
        solver.fit_histogram(m_solv, valid_hist, 20, &d);
        ASSERT_LT(d.final_joint_dp_nll, d.initial_nll);

        ctx.record().expected["objective_reduction"] = "final_nll < initial_nll";
        ctx.record().actual["initial_to_final"] = std::to_string(d.initial_nll) + " -> " + std::to_string(d.final_joint_dp_nll);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-009") {
        TestContext ctx(opts, tid, "map_estimator_prior_penalty_contribution",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_nll_from_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Evaluate NLL with prior_lambda_=0.0 vs prior_lambda_=0.04 at identical fitted theta_";
        ctx.record().independent_oracle = "nll(0.04) - nll(0.0) == 0.5 * 0.04 * ||theta - prior_theta||^2 within 1e-13";

        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.04;
        HistogramLBFGSSolver solver;
        solver.fit_histogram(m, valid_hist, 20, nullptr);
        double sq = (m.theta_ - m.prior_theta_).squaredNorm();
        double nll_004 = m.compute_nll_from_histogram(valid_hist);
        m.prior_lambda_ = 0.0;
        double nll_000 = m.compute_nll_from_histogram(valid_hist);
        ASSERT_NEAR(nll_004 - nll_000, 0.5 * 0.04 * sq, 1e-13);
        ctx.record().expected["prior_term"] = 0.5 * 0.04 * sq;
        ctx.record().actual["observed_diff"] = nll_004 - nll_000;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-010") {
        TestContext ctx(opts, tid, "map_estimator_insufficient_evidence_guard",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Feed total=0 empty histogram and total=5 (< W_min=64) sparse histogram and verify solver refuses adaptation cleanly";
        ctx.record().independent_oracle = "total=0 -> nll=0, converged=false; total=5 (<64) -> iterations=0, converged=false, theta preserved";

        HistogramLBFGSSolver solver;
        const uint32_t empty_h[16]  = {0};
        const uint32_t sparse_h[16] = {4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        NoiseModel m0;
        LBFGSFitDiagnostics d0{};
        ASSERT_EQ(solver.fit_histogram(m0, empty_h, 20, &d0), 0.0);
        ASSERT_FALSE(d0.converged);
        ASSERT_EQ(d0.iterations_executed, 0);

        NoiseModel m5;
        m5.init(0.012, 0.008);
        NoiseModel::ParamVec before = m5.theta_;
        LBFGSFitDiagnostics d5{};
        solver.fit_histogram(m5, sparse_h, 20, &d5);
        ASSERT_FALSE(d5.converged);
        ASSERT_EQ(d5.iterations_executed, 0);
        ASSERT_NEAR((m5.theta_ - before).cwiseAbs().maxCoeff(), 0.0, 1e-15);

        ctx.record().expected["w_min_64_guard"] = "Both total=0 and total=5 set converged=false and preserve theta";
        ctx.record().actual["w_min_64_guard"] = "Verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-OP-001", "L3-OP-002", "L3-OP-003", "L3-OP-004", "L3-OP-005",
        "L3-OP-006", "L3-OP-007", "L3-OP-008", "L3-OP-009", "L3-OP-010"
    };
    if (opts.test_id == "ALL") {
        int rc = 0;
        for (const char *id : all_ids) {
            if (run_one(opts, id) != 0) rc = 1;
        }
        return rc;
    }
    return run_one(opts, opts.test_id);
}
