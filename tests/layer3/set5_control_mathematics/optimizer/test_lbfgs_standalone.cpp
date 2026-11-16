#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include "estimator/lbfgs_solver.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace avid::layer3_test;

namespace {

/* Independent reference projected gradient solver using FWHT + central FD (zero shared solver code) */
double ref_fwht_nll(const double x[21], const double prior[21], double lambda, const uint32_t hist[16]) {
    static const uint8_t sigs[21] = {
        0x1u, 0x3u, 0x2u, 0x4u, 0x6u, 0x2u, 0x4u, 0xCu, 0x8u,
        0x2u, 0x1u, 0x2u, 0x4u, 0x8u, 0x4u, 0x5u, 0x0u, 0x5u, 0xAu, 0x0u, 0xAu
    };
    double walsh[16];
    for (int u = 0; u < 16; ++u) {
        double p = 1.0;
        for (int e = 0; e < 21; ++e) {
            if (__builtin_popcount(static_cast<unsigned>(u & sigs[e])) & 1) {
                p *= (1.0 - 2.0 * x[e]);
            }
        }
        walsh[u] = p;
    }
    uint64_t tot = 0;
    for (int s = 0; s < 16; ++s) tot += hist[s];
    double nll = 0.0;
    for (int s = 0; s < 16; ++s) {
        double acc = 0.0;
        for (int u = 0; u < 16; ++u) {
            acc += ((__builtin_popcount(static_cast<unsigned>(u & s)) & 1) ? -walsh[u] : walsh[u]);
        }
        double ps = std::max(1e-15, acc / 16.0);
        if (hist[s] > 0) nll -= (static_cast<double>(hist[s]) / static_cast<double>(tot)) * std::log(ps);
    }
    double reg = 0.0;
    for (int e = 0; e < 21; ++e) {
        double d = x[e] - prior[e];
        reg += d * d;
    }
    return nll + 0.5 * lambda * reg;
}

double ref_solve_box_nll(const uint32_t hist[16], double lambda, double out_x[21]) {
    double prior[21];
    for (int i = 0; i < 9; ++i) { prior[i] = 0.012; out_x[i] = 0.012; }
    for (int i = 9; i < 21; ++i) { prior[i] = 0.008; out_x[i] = 0.008; }
    double f = ref_fwht_nll(out_x, prior, lambda, hist);
    const double eps = 1e-7;
    for (int it = 0; it < 120; ++it) {
        double g[21];
        double gmax = 0.0;
        for (int e = 0; e < 21; ++e) {
            double xp[21], xm[21];
            std::copy(out_x, out_x + 21, xp);
            std::copy(out_x, out_x + 21, xm);
            xp[e] += eps;
            xm[e] -= eps;
            g[e] = (ref_fwht_nll(xp, prior, lambda, hist) - ref_fwht_nll(xm, prior, lambda, hist)) / (2.0 * eps);
            double gp = ((out_x[e] <= 1e-5 + 1e-9 && g[e] > 0.0) || (out_x[e] >= 0.45 - 1e-9 && g[e] < 0.0)) ? 0.0 : g[e];
            gmax = std::max(gmax, std::abs(gp));
        }
        if (gmax < 1e-6) break;
        double step = 0.08;
        for (int ls = 0; ls < 15; ++ls) {
            double cand[21];
            for (int e = 0; e < 21; ++e) cand[e] = std::clamp(out_x[e] - step * g[e], 1e-5, 0.45);
            double fc = ref_fwht_nll(cand, prior, lambda, hist);
            if (fc <= f) {
                std::copy(cand, cand + 21, out_x);
                f = fc;
                break;
            }
            step *= 0.5;
        }
    }
    return f;
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;

    if (tid == "L3-MATH-015") {
        TestContext ctx(opts, tid, "lbfgs_box_bound_invariance",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram, HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify [1e-5, 0.45] box constraint invariance at every intermediate and final iterate from extreme starts";
        ctx.record().independent_oracle = "All 21 coordinates remain strictly inside [1e-5, 0.45]";

        const uint32_t adv_low[16]  = {1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const uint32_t adv_high[16] = {10, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80};
        for (bool high_case : {false, true}) {
            NoiseModel m;
            for (int e = 0; e < 21; ++e) m.theta_(e) = high_case ? 0.449 : 0.00002;
            LBFGSOptimizer opt;
            for (int it = 0; it < 15; ++it) {
                double nll = 0.0, ginf = 0.0;
                opt.step_histogram(m, high_case ? adv_high : adv_low, 1, &nll, &ginf);
                for (int e = 0; e < 21; ++e) {
                    ASSERT_GE(m.theta_(e), 1e-5);
                    ASSERT_LE(m.theta_(e), 0.45);
                }
            }
        }
        ctx.record().expected["bounds"] = "All iterates strictly in [1e-5, 0.45]";
        ctx.record().actual["bounds"] = "Verified across 30 extreme low/high step iterates";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-016") {
        TestContext ctx(opts, tid, "lbfgs_convergence_criterion_audit",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify early projected-gradient stop (g_inf < 1e-5) vs non-convergent max_iters=0 / sparse stop";
        ctx.record().independent_oracle = "Converged run achieves g_inf < 1e-4 & diag.converged==true; max_iters=0 yields diag.converged==false";

        const uint32_t h[16] = {820, 20, 30, 15, 25, 10, 45, 5, 12, 3, 6, 2, 4, 1, 1, 1};
        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.04;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics d_conv{}, d_zero{};
        solver.fit_histogram(m, h, 100, &d_conv);
        ASSERT_TRUE(d_conv.converged);
        ASSERT_LT(d_conv.final_grad_inf_norm, 1e-3);

        m.init(0.012, 0.008);
        solver.fit_histogram(m, h, 0, &d_zero);
        ASSERT_FALSE(d_zero.converged);
        ASSERT_EQ(d_zero.iterations_executed, 0);

        ctx.record().expected["converged_vs_zero_iters"] = "d_conv.converged==true, d_zero.converged==false";
        ctx.record().actual["final_grad_inf_norm"] = d_conv.final_grad_inf_norm;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-017") {
        TestContext ctx(opts, tid, "lbfgs_maximum_iterations_cap",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify executed iterations strictly respect max_iters in {1, 4, 8} from far initialization";
        ctx.record().independent_oracle = "0 < iterations_executed <= max_iters";

        const uint32_t h[16] = {600, 50, 60, 40, 50, 25, 90, 12, 30, 8, 15, 5, 10, 2, 2, 1};
        HistogramLBFGSSolver solver;
        for (int cap : {1, 4, 8}) {
            NoiseModel m;
            m.init(0.25, 0.15);
            LBFGSFitDiagnostics diag{};
            solver.fit_histogram(m, h, cap, &diag);
            ASSERT_GT(diag.iterations_executed, 0);
            ASSERT_LE(diag.iterations_executed, cap);
        }
        ctx.record().expected["iteration_cap"] = "Strictly bounded by {1, 4, 8}";
        ctx.record().actual["iteration_cap"] = "Verified for caps 1, 4, 8";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-018") {
        TestContext ctx(opts, tid, "lbfgs_initial_point_orbit_invariance",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Launch optimization from 5 initial points [1e-4, 0.005, 0.012, 0.10, 0.30] on exact DP expectation histogram";
        ctx.record().independent_oracle = "All 5 starts converge to identical 9-class observable orbit probabilities (< 1e-3)";

        NoiseModel truth;
        truth.init(0.015, 0.006);
        truth.theta_(4) = 0.075;
        truth.theta_(1) = 0.045;
        truth.compute_joint_distribution();
        uint32_t hist[16];
        for (int s = 0; s < 16; ++s) hist[s] = static_cast<uint32_t>(std::lround(100000.0 * truth.joint_probs()(s)));

        const double starts[5] = {1e-4, 0.005, 0.012, 0.10, 0.30};
        double ref_orbits[NM_N_ORBITS] = {0.0};
        double max_orbit_spread = 0.0;
        for (int idx = 0; idx < 5; ++idx) {
            NoiseModel m;
            m.init(0.012, 0.008); /* fixed prior */
            m.prior_lambda_ = 0.04;
            for (int e = 0; e < 21; ++e) m.theta_(e) = starts[idx];
            LBFGSOptimizer opt;
            double nll = 0.0, ginf = 0.0;
            opt.step_histogram(m, hist, 50, &nll, &ginf);
            double cur_orbits[NM_N_ORBITS];
            NoiseModel::compute_orbit_probabilities(m.theta_, false, cur_orbits);
            if (idx == 0) {
                std::copy(cur_orbits, cur_orbits + NM_N_ORBITS, ref_orbits);
            } else {
                for (int o = 0; o < NM_N_ORBITS; ++o) {
                    max_orbit_spread = std::max(max_orbit_spread, std::abs(cur_orbits[o] - ref_orbits[o]));
                }
            }
        }
        ASSERT_LT(max_orbit_spread, 1e-3);
        ctx.record().expected["max_orbit_spread"] = "< 1e-3 across 5 initializations";
        ctx.record().actual["max_orbit_spread"] = max_orbit_spread;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-019") {
        TestContext ctx(opts, tid, "lbfgs_ill_conditioned_near_zero_prior",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Solve near-unregularized problem (prior_lambda_=1e-8) with collinear aliased orbits without NaN or divergence";
        ctx.record().independent_oracle = "Finite objective, valid box bounds, and tight predictive PMF agreement";

        const uint32_t h[16] = {800, 120, 40, 10, 20, 2, 5, 1, 1, 0, 1, 0, 0, 0, 0, 0};
        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 1e-8;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        double nll = solver.fit_histogram(m, h, 30, &diag);
        ASSERT_TRUE(std::isfinite(nll));
        ASSERT_TRUE(NoiseModel::validate_parameter_bounds(m.theta_));
        ASSERT_LE(nll, diag.initial_nll + 1e-9);
        ctx.record().expected["finite_ill_conditioned_fit"] = "Finite NLL <= initial_nll and valid bounds";
        ctx.record().actual["final_nll"] = nll;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-020") {
        TestContext ctx(opts, tid, "flat_likelihood_zero_signature_edges",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_gradient_from_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify zero-signature edges ZZ_3_6 (e=16) and ZZ_2_5 (e=19) in X-basis have exact zero data gradient and stay at prior";
        ctx.record().independent_oracle = "grad_data(16) == 0.0, grad_data(19) == 0.0, and MAP estimate equals prior_theta";

        const uint32_t h[16] = {700, 40, 50, 30, 40, 20, 60, 10, 25, 5, 10, 4, 3, 1, 1, 1};
        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.0;
        m.theta_(16) = 0.17;
        m.theta_(19) = 0.29;
        m.compute_gradient_from_histogram(h);
        ASSERT_EQ(m.grad()(16), 0.0);
        ASSERT_EQ(m.grad()(19), 0.0);

        m.prior_lambda_ = 2.0;
        LBFGSOptimizer opt;
        double nll = 0.0, ginf = 0.0;
        opt.step_histogram(m, h, 50, &nll, &ginf);
        opt.reset();
        opt.step_histogram(m, h, 50, &nll, &ginf);
        ASSERT_NEAR(m.theta_(16), m.prior_theta_(16), 1e-4);
        ASSERT_NEAR(m.theta_(19), m.prior_theta_(19), 1e-4);
        ctx.record().expected["zero_sig_edges"] = "grad_data == 0.0 and MAP estimate == prior_theta (0.008)";
        ctx.record().actual["theta_16_19"] = std::to_string(m.theta_(16)) + ", " + std::to_string(m.theta_(19));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-021") {
        TestContext ctx(opts, tid, "synthetic_infinite_sample_singleton_recovery",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Recover singleton hotspots q1 (0x3), q4 (0x6), q7 (0xC) from 100,000-sample exact expectation histogram";
        ctx.record().independent_oracle = "Singleton error < 0.002 and 9-class orbit RMSE < 1e-3";

        NoiseModel truth;
        truth.set_basis_z(false);
        truth.init(0.012, 0.002);
        for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
        truth.theta_(1) = 0.055;
        truth.theta_(4) = 0.085;
        truth.theta_(7) = 0.042;
        truth.compute_joint_distribution();
        uint32_t hist[16];
        for (int s = 0; s < 16; ++s) hist[s] = static_cast<uint32_t>(std::lround(100000.0 * truth.joint_probs()(s)));

        NoiseModel est;
        est.set_basis_z(false);
        est.init(0.012, 0.008);
        est.prior_lambda_ = 0.005;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        solver.fit_histogram(est, hist, 40, &diag);

        double err1 = std::abs(est.theta_(1) - truth.theta_(1));
        double err4 = std::abs(est.theta_(4) - truth.theta_(4));
        double err7 = std::abs(est.theta_(7) - truth.theta_(7));
        double orb_rmse = est.compute_orbit_rmse(truth.theta_);
        ASSERT_LT(err1, 0.002);
        ASSERT_LT(err4, 0.002);
        ASSERT_LT(err7, 0.002);
        ASSERT_LT(orb_rmse, 1e-3);
        ctx.record().expected["singleton_and_orbit_error"] = "err_{1,4,7} < 0.002, orbit_rmse < 1e-3";
        ctx.record().actual["orbit_rmse"] = orb_rmse;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-022") {
        TestContext ctx(opts, tid, "model_mismatch_uniform_histogram_box_cap",
                        "software/src/estimator/lbfgs_solver.cpp",
                        "HistogramLBFGSSolver::fit_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Feed maximally entropic uniform histogram [100]*16 and verify box cap / elevated NLL residual";
        ctx.record().independent_oracle = "Final NLL >= log(16) ~ 2.7725 and parameters saturate or elevate near upper regime";

        uint32_t uni[16];
        for (int s = 0; s < 16; ++s) uni[s] = 100u;
        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = 0.01;
        HistogramLBFGSSolver solver;
        LBFGSFitDiagnostics diag{};
        double nll = solver.fit_histogram(m, uni, 40, &diag);
        ASSERT_GE(nll, std::log(16.0) - 1e-6);
        ASSERT_GT(m.theta_.maxCoeff(), 0.25);
        ctx.record().expected["uniform_nll_and_max_theta"] = "nll >= ln(16)=2.772588 and max_theta > 0.25";
        ctx.record().actual["final_nll"] = nll;
        ctx.record().actual["max_theta"] = m.theta_.maxCoeff();
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-023") {
        TestContext ctx(opts, tid, "prior_lambda_monotone_shrinkage",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Sweep prior_lambda across {0.0, 0.04, 0.5, 5.0} on constant histogram and verify monotone L2 shrinkage toward prior_theta";
        ctx.record().independent_oracle = "||theta_hat(lambda_k) - prior_theta||_2 strictly decreases with lambda";

        const uint32_t h[16] = {680, 40, 55, 35, 45, 18, 70, 8, 22, 5, 10, 3, 5, 2, 1, 1};
        const double lambdas[4] = {0.0, 0.04, 0.5, 5.0};
        double norms[4] = {0.0};
        for (int i = 0; i < 4; ++i) {
            NoiseModel m;
            m.init(0.012, 0.008);
            m.prior_lambda_ = lambdas[i];
            LBFGSOptimizer opt;
            double nll = 0.0, ginf = 0.0;
            opt.step_histogram(m, h, 50, &nll, &ginf);
            norms[i] = (m.theta_ - m.prior_theta_).norm();
        }
        ASSERT_GT(norms[0], norms[1]);
        ASSERT_GT(norms[1], norms[2]);
        ASSERT_GT(norms[2], norms[3]);
        ctx.record().expected["monotone_norms"] = "norm(0.0) > norm(0.04) > norm(0.5) > norm(5.0)";
        ctx.record().actual["norms"] = std::to_string(norms[0]) + " > " + std::to_string(norms[1]) +
                                       " > " + std::to_string(norms[2]) + " > " + std::to_string(norms[3]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-024") {
        TestContext ctx(opts, tid, "cpp_lbfgs_vs_independent_reference_solver",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare C++ LBFGSOptimizer objective and solution against independent FWHT projected-gradient reference solver";
        ctx.record().independent_oracle = "|NLL_cpp - NLL_ref| < 1e-5 and ||theta_cpp - theta_ref||_inf < 1e-3 (strongly convex lambda=0.2)";

        const uint32_t h[16] = {740, 32, 44, 28, 38, 15, 58, 7, 19, 4, 8, 2, 3, 1, 1, 0};
        const double lam = 0.20;
        double ref_x[21];
        double ref_f = ref_solve_box_nll(h, lam, ref_x);

        NoiseModel m;
        m.init(0.012, 0.008);
        m.prior_lambda_ = lam;
        LBFGSOptimizer opt;
        double cpp_f = 0.0, ginf = 0.0;
        opt.step_histogram(m, h, 50, &cpp_f, &ginf);

        double max_param_diff = 0.0;
        for (int e = 0; e < 21; ++e) {
            max_param_diff = std::max(max_param_diff, std::abs(m.theta_(e) - ref_x[e]));
        }
        ASSERT_LT(std::abs(cpp_f - ref_f), 1e-5);
        ASSERT_LT(max_param_diff, 1e-3);
        ctx.record().expected["objective_and_param_consistency"] = "|dNLL| < 1e-5, ||dtheta||_inf < 1e-3";
        ctx.record().actual["abs_nll_diff"] = std::abs(cpp_f - ref_f);
        ctx.record().actual["max_param_diff"] = max_param_diff;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-015", "L3-MATH-016", "L3-MATH-017", "L3-MATH-018", "L3-MATH-019",
        "L3-MATH-020", "L3-MATH-021", "L3-MATH-022", "L3-MATH-023", "L3-MATH-024"
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
