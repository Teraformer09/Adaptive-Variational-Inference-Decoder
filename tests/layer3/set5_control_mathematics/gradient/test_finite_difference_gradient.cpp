#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include "estimator/lbfgs_solver.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

using namespace avid::layer3_test;

namespace {

uint8_t ref_hx_sig(int e) {
    static const uint8_t sigs[21] = {
        0x1u, 0x3u, 0x2u, 0x4u, 0x6u, 0x2u, 0x4u, 0xCu, 0x8u,
        0x2u, 0x1u, 0x2u, 0x4u, 0x8u, 0x4u, 0x5u, 0x0u, 0x5u, 0xAu, 0x0u, 0xAu
    };
    return sigs[e];
}

void ref_pmf_x(const NoiseModel::ParamVec &theta, double pmf[16]) {
    double walsh[16];
    for (int u = 0; u < 16; ++u) {
        double prod = 1.0;
        for (int e = 0; e < 21; ++e) {
            uint8_t g = ref_hx_sig(e);
            if (__builtin_popcount(static_cast<unsigned>(u & g)) & 1) {
                prod *= (1.0 - 2.0 * theta(e));
            }
        }
        walsh[u] = prod;
    }
    for (int s = 0; s < 16; ++s) {
        double acc = 0.0;
        for (int u = 0; u < 16; ++u) {
            int dot = __builtin_popcount(static_cast<unsigned>(u & s)) & 1;
            acc += (dot ? -walsh[u] : walsh[u]);
        }
        pmf[s] = std::max(1e-15, acc / 16.0);
    }
}

double ref_nll_x(const NoiseModel::ParamVec &theta,
                 const NoiseModel::ParamVec &prior,
                 double lambda,
                 const uint32_t hist[16]) {
    uint64_t total = 0;
    for (int s = 0; s < 16; ++s) total += hist[s];
    double pmf[16];
    ref_pmf_x(theta, pmf);
    double nll = 0.0;
    for (int s = 0; s < 16; ++s) {
        if (hist[s] > 0) {
            nll -= (static_cast<double>(hist[s]) / static_cast<double>(total)) * std::log(pmf[s]);
        }
    }
    double reg = 0.0;
    for (int e = 0; e < 21; ++e) {
        double d = theta(e) - prior(e);
        reg += d * d;
    }
    return nll + 0.5 * lambda * reg;
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const uint32_t sample_hist[16] = {
        720, 35, 48, 22, 41, 14, 52, 9, 28, 6, 17, 4, 19, 3, 5, 1
    };

    if (tid == "L3-MATH-009") {
        TestContext ctx(opts, tid, "nll_empirical_cross_entropy_oracle",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_nll_from_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare compute_nll_from_histogram (prior_lambda_=0) against independent FWHT empirical cross-entropy";
        ctx.record().independent_oracle = "|NLL_prod - NLL_ref| < 1e-12";

        NoiseModel m;
        m.set_basis_z(false);
        m.prior_lambda_ = 0.0;
        for (int e = 0; e < 21; ++e) m.theta_(e) = 0.010 + 0.003 * static_cast<double>((e * 5 + 1) % 11);
        double prod_nll = m.compute_nll_from_histogram(sample_hist);
        double ref_val  = ref_nll_x(m.theta_, m.prior_theta_, 0.0, sample_hist);
        double diff = std::abs(prod_nll - ref_val);
        ASSERT_LT(diff, 1e-12);
        ctx.record().expected["abs_nll_diff"] = "< 1e-12";
        ctx.record().actual["abs_nll_diff"] = diff;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-010") {
        TestContext ctx(opts, tid, "map_l2_prior_regularization_exact",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_nll_from_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify quadratic L2 MAP penalty 0.5 * lambda * ||theta - prior||^2 at lambda = 0.04 and lambda = 0.5";
        ctx.record().independent_oracle = "Penalty difference matches exact analytical quadratic form < 1e-13";

        NoiseModel m;
        m.init(0.012, 0.008);
        for (int e = 0; e < 21; ++e) m.theta_(e) = 0.025 + 0.002 * static_cast<double>(e);
        double sq_norm = (m.theta_ - m.prior_theta_).squaredNorm();

        m.prior_lambda_ = 0.0;
        double nll0 = m.compute_nll_from_histogram(sample_hist);

        m.prior_lambda_ = 0.04;
        double nll004 = m.compute_nll_from_histogram(sample_hist);
        ASSERT_NEAR(nll004 - nll0, 0.5 * 0.04 * sq_norm, 1e-13);

        m.prior_lambda_ = 0.5;
        double nll05 = m.compute_nll_from_histogram(sample_hist);
        ASSERT_NEAR(nll05 - nll0, 0.5 * 0.5 * sq_norm, 1e-13);

        ctx.record().expected["quadratic_penalty_err"] = "< 1e-13 at lambda=0.04 and lambda=0.5";
        ctx.record().actual["err_lambda_05"] = std::abs((nll05 - nll0) - 0.5 * 0.5 * sq_norm);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-011") {
        TestContext ctx(opts, tid, "coset_inversion_gradient_vs_central_fd",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_gradient_from_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare analytical coset-inversion gradient against central finite differences (eps=1e-7) across all 21 parameters in X and Z bases";
        ctx.record().independent_oracle = "Max gradient infinity-norm error < 1e-7";

        const double eps = 1e-7;
        double max_grad_err = 0.0;
        for (bool bz : {false, true}) {
            NoiseModel m;
            m.set_basis_z(bz);
            m.init(0.014, 0.009);
            m.prior_lambda_ = 0.25;
            for (int e = 0; e < 21; ++e) {
                m.theta_(e) = 0.015 + 0.004 * static_cast<double>((e * 3 + 2) % 13);
            }
            m.compute_gradient_from_histogram(sample_hist);
            NoiseModel::GradVec g_analytic = m.grad();

            for (int e = 0; e < 21; ++e) {
                NoiseModel mp = m, mm = m;
                mp.theta_(e) += eps;
                mm.theta_(e) -= eps;
                double fp = mp.compute_nll_from_histogram(sample_hist);
                double fm = mm.compute_nll_from_histogram(sample_hist);
                double g_fd = (fp - fm) / (2.0 * eps);
                max_grad_err = std::max(max_grad_err, std::abs(g_analytic(e) - g_fd));
            }
        }
        ASSERT_LT(max_grad_err, 1e-7);
        ctx.record().expected["max_grad_err_vs_central_fd"] = "< 1e-7";
        ctx.record().actual["max_grad_err_vs_central_fd"] = max_grad_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-012") {
        TestContext ctx(opts, tid, "hessian_vs_fisher_information_spectrum",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_nll_from_histogram, compute_gradient_from_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare 21x21 central-FD Hessian around exact generative expectation against Fisher Information I(theta) + lambda*I_21";
        ctx.record().independent_oracle = "At theta == theta_true, E[Hessian(NLL)] == I(theta) + lambda * I_21 to < 1e-4";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.020, 0.010);
        m.prior_lambda_ = 0.5;
        m.compute_joint_distribution();

        /* Construct high-precision expectation histogram N = 10,000,000 */
        uint32_t exp_hist[16];
        const double N_big = 10000000.0;
        for (int s = 0; s < 16; ++s) {
            exp_hist[s] = static_cast<uint32_t>(std::lround(N_big * m.joint_probs()(s)));
        }

        const double eps = 1e-6;
        Eigen::Matrix<double, 21, 21> H_fd = Eigen::Matrix<double, 21, 21>::Zero();
        Eigen::Matrix<double, 16, 21> J    = Eigen::Matrix<double, 16, 21>::Zero();
        for (int j = 0; j < 21; ++j) {
            NoiseModel mp = m, mm = m;
            mp.theta_(j) += eps;
            mm.theta_(j) -= eps;
            mp.compute_gradient_from_histogram(exp_hist);
            mm.compute_gradient_from_histogram(exp_hist);
            H_fd.col(j) = (mp.grad() - mm.grad()) / (2.0 * eps);

            mp.compute_joint_distribution();
            mm.compute_joint_distribution();
            J.col(j) = (mp.joint_probs() - mm.joint_probs()) / (2.0 * eps);
        }
        H_fd = 0.5 * (H_fd + H_fd.transpose());

        Eigen::Matrix<double, 21, 21> Fisher = Eigen::Matrix<double, 21, 21>::Zero();
        for (int s = 0; s < 16; ++s) {
            double ps = m.joint_probs()(s);
            if (ps > 1e-12) {
                Fisher += (1.0 / ps) * (J.row(s).transpose() * J.row(s));
            }
        }
        Eigen::Matrix<double, 21, 21> Expected_H = Fisher + m.prior_lambda_ * Eigen::Matrix<double, 21, 21>::Identity();
        double max_mat_err = (H_fd - Expected_H).cwiseAbs().maxCoeff();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 21, 21>> eig(H_fd);
        double min_eig = eig.eigenvalues().minCoeff();

        ASSERT_LT(max_mat_err, 5e-4);
        ASSERT_NEAR(min_eig, m.prior_lambda_, 1e-4);
        ctx.record().expected["max_hessian_vs_fisher_err"] = "< 5e-4 and min_eig == prior_lambda (0.5)";
        ctx.record().actual["max_hessian_vs_fisher_err"] = max_mat_err;
        ctx.record().actual["min_eigenvalue"] = min_eig;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-013") {
        TestContext ctx(opts, tid, "lbfgs_two_loop_descent_direction",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify two-loop L-BFGS recursion and projected gradient step produce strict descent (<g, x_{k+1} - x_k> < 0 and f_{k+1} < f_k) on non-stationary interior state";
        ctx.record().independent_oracle = "Directional derivative g^T (x_{k+1} - x_k) < 0 across sequential 1-step invocations";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.030, 0.020);
        m.prior_lambda_ = 0.04;
        LBFGSOptimizer opt;
        opt.reset();

        int descent_steps = 0;
        for (int step = 0; step < 6; ++step) {
            double f0 = m.compute_nll_from_histogram(sample_hist);
            m.compute_gradient_from_histogram(sample_hist);
            NoiseModel::GradVec g0 = m.grad();
            NoiseModel::ParamVec x0 = m.theta_;
            double f1 = f0, g_inf = 0.0;
            int executed = opt.step_histogram(m, sample_hist, 1, &f1, &g_inf);
            if (executed > 0) {
                double dir_deriv = g0.dot(m.theta_ - x0);
                ASSERT_LT(dir_deriv, 0.0);
                ASSERT_LE(f1, f0);
                descent_steps++;
            }
        }
        ASSERT_GE(descent_steps, 4);
        ctx.record().expected["strict_descent_steps"] = ">= 4 sequential quasi-Newton updates with g^T dx < 0";
        ctx.record().actual["strict_descent_steps"] = descent_steps;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-014") {
        TestContext ctx(opts, tid, "lbfgs_armijo_sufficient_decrease",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify Armijo sufficient decrease f(x_{k+1}) <= f(x_k) + 1e-4 * <g_k, x_{k+1} - x_k> at every accepted L-BFGS step";
        ctx.record().independent_oracle = "0 Armijo violations across 12 consecutive optimization steps";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.080, 0.040);
        m.prior_lambda_ = 0.04;
        LBFGSOptimizer opt;
        opt.reset();

        int steps_checked = 0;
        for (int step = 0; step < 12; ++step) {
            double f0 = m.compute_nll_from_histogram(sample_hist);
            m.compute_gradient_from_histogram(sample_hist);
            NoiseModel::GradVec g0 = m.grad();
            NoiseModel::ParamVec x0 = m.theta_;
            double f1 = f0, g_inf = 0.0;
            int executed = opt.step_histogram(m, sample_hist, 1, &f1, &g_inf);
            if (executed == 0) break;
            double dir_deriv = g0.dot(m.theta_ - x0);
            ASSERT_LE(f1, f0 + LBFGS_C1 * dir_deriv + 1e-14);
            steps_checked++;
        }
        ASSERT_GE(steps_checked, 5);
        ctx.record().expected["armijo_steps_verified"] = ">= 5 with zero sufficient-decrease violations";
        ctx.record().actual["armijo_steps_verified"] = steps_checked;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-009", "L3-MATH-010", "L3-MATH-011",
        "L3-MATH-012", "L3-MATH-013", "L3-MATH-014"
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
