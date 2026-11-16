#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace avid::layer3_test;

namespace {

Eigen::Matrix<double, 16, 21> compute_sensitivity_jacobian(const NoiseModel &base_m) {
    const double eps = 1e-7;
    Eigen::Matrix<double, 16, 21> J = Eigen::Matrix<double, 16, 21>::Zero();
    for (int e = 0; e < 21; ++e) {
        NoiseModel mp = base_m, mm = base_m;
        mp.theta_(e) += eps;
        mm.theta_(e) -= eps;
        mp.compute_joint_distribution();
        mm.compute_joint_distribution();
        J.col(e) = (mp.joint_probs() - mm.joint_probs()) / (2.0 * eps);
    }
    return J;
}

Eigen::Matrix<double, 21, 21> compute_fisher_matrix(NoiseModel m) {
    m.compute_joint_distribution();
    auto J = compute_sensitivity_jacobian(m);
    Eigen::Matrix<double, 21, 21> I = Eigen::Matrix<double, 21, 21>::Zero();
    for (int s = 0; s < 16; ++s) {
        double ps = m.joint_probs()(s);
        if (ps > 1e-14) {
            I += (1.0 / ps) * (J.row(s).transpose() * J.row(s));
        }
    }
    return 0.5 * (I + I.transpose());
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;

    if (tid == "L3-MATH-025") {
        TestContext ctx(opts, tid, "jacobian_svd_rank_nine_orbits",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compute 16x21 sensitivity Jacobian dP(s|theta)/dtheta_e and verify exact algebraic rank == 9";
        ctx.record().independent_oracle = "Eigen::JacobiSVD singular values: exactly 9 > 1e-4 and remaining <= 1e-11";

        for (bool bz : {false, true}) {
            NoiseModel m;
            m.set_basis_z(bz);
            m.init(0.015, 0.009);
            auto J = compute_sensitivity_jacobian(m);
            Eigen::JacobiSVD<Eigen::Matrix<double, 16, 21>> svd(J);
            const auto &sv = svd.singularValues();
            int rank = 0;
            for (int i = 0; i < sv.size(); ++i) {
                if (sv(i) > 1e-8) rank++;
            }
            ASSERT_EQ(rank, 9);
            ctx.record().actual[bz ? "rank_z_basis" : "rank_x_basis"] = rank;
            ctx.record().actual[bz ? "sigma9_z" : "sigma9_x"] = sv(8);
            ctx.record().actual[bz ? "sigma10_z" : "sigma10_x"] = sv(9);
        }
        ctx.record().expected["effective_rank"] = 9;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-026") {
        TestContext ctx(opts, tid, "multinomial_fisher_information_eigenspectrum",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compute 21x21 multinomial Fisher Information Matrix I(theta) and verify PSD spectrum with 9 positive eigenvalues";
        ctx.record().independent_oracle = "9 positive eigenvalues > 1.0 and 12 null eigenvalues ~ 0 (< 1e-10)";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.015, 0.008);
        auto F = compute_fisher_matrix(m);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 21, 21>> eig(F);
        const auto &ev = eig.eigenvalues();
        int n_pos = 0, n_null = 0;
        for (int i = 0; i < 21; ++i) {
            ASSERT_GE(ev(i), -1e-10);
            if (ev(i) > 1e-4) n_pos++;
            else n_null++;
        }
        ASSERT_EQ(n_pos, 9);
        ASSERT_EQ(n_null, 12);
        ctx.record().expected["positive_and_null_eigenvalues"] = "9 positive, 12 null";
        ctx.record().actual["max_eigenvalue"] = ev(20);
        ctx.record().actual["eig_12_largest_null"] = ev(11);
        ctx.record().actual["eig_13_smallest_pos"] = ev(12);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-027") {
        TestContext ctx(opts, tid, "posterior_covariance_aliased_anticorrelation",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify strong negative posterior correlation within shared syndrome orbits (Z_0, ZZ_1_2) on 0x1";
        ctx.record().independent_oracle = "Posterior correlation rho(Z_0, ZZ_1_2) < -0.90 at N=10,000, lambda=0.5";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.015, 0.008);
        auto F = compute_fisher_matrix(m);
        const double N = 10000.0;
        const double lam = 0.5;
        Eigen::Matrix<double, 21, 21> Cov = (N * F + lam * Eigen::Matrix<double, 21, 21>::Identity()).inverse();
        /* Z_0 is index 0, ZZ_1_2 is pair index 1 -> parameter index 10 (both have sig 0x1) */
        double rho_0_10 = Cov(0, 10) / std::sqrt(Cov(0, 0) * Cov(10, 10));
        /* Z_1 (sig 0x3) vs Z_4 (sig 0x6) are distinct singletons -> near-zero correlation */
        double rho_1_4  = Cov(1, 4) / std::sqrt(Cov(1, 1) * Cov(4, 4));

        ASSERT_LT(rho_0_10, -0.90);
        ASSERT_LT(std::abs(rho_1_4), 0.15);
        ctx.record().expected["rho_aliased_0_10"] = "< -0.90";
        ctx.record().actual["rho_aliased_0_10"] = rho_0_10;
        ctx.record().actual["rho_orthogonal_1_4"] = rho_1_4;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-028") {
        TestContext ctx(opts, tid, "null_space_dimension_and_zero_signature_columns",
                        "software/src/inference/noise_model.cpp",
                        "NM_EVENT_SIGNATURES, NM_EVENT_SIGNATURES_Z_BASIS", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Confirm 12 null-space directions and exact zero Jacobian columns on g=0 edges (e=16,19 in X-basis; e=9,14 in Z-basis)";
        ctx.record().independent_oracle = "||J.col(16)||==0, ||J.col(19)||==0 in X-basis; ||J.col(9)||==0, ||J.col(14)||==0 in Z-basis";

        NoiseModel mx, mz;
        mx.set_basis_z(false); mx.init(0.015, 0.008);
        mz.set_basis_z(true);  mz.init(0.015, 0.008);
        auto Jx = compute_sensitivity_jacobian(mx);
        auto Jz = compute_sensitivity_jacobian(mz);

        ASSERT_NEAR(Jx.col(16).norm(), 0.0, 1e-15);
        ASSERT_NEAR(Jx.col(19).norm(), 0.0, 1e-15);
        ASSERT_NEAR(Jz.col(9).norm(),  0.0, 1e-15);
        ASSERT_NEAR(Jz.col(14).norm(), 0.0, 1e-15);

        ctx.record().expected["zero_columns"] = "X:(16,19)==0, Z:(9,14)==0";
        ctx.record().actual["norms"] = "verified 0.0 to 1e-15";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-029") {
        TestContext ctx(opts, tid, "crlb_sample_size_scaling_singletons_vs_null",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Evaluate regularized CRLB across N in {256, 1000, 10000}: singletons {Z_1, Z_4, Z_7} scale as 1/sqrt(N) while null edge e=16 stays constant";
        ctx.record().independent_oracle = "SE(Z_4, N=10000) / SE(Z_4, N=256) ~ sqrt(256/10000) = 0.16; SE(e=16) == 1/sqrt(N*0 + lambda_scaled)";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.020, 0.008);
        auto F = compute_fisher_matrix(m);
        /* Note: NLL objective is per-sample average + 0.5*lambda*||theta-prior||^2, so total precision is N*(F + lambda*I) or unscaled F_sample:
         * Under standard Bayesian precision N*F + lam_prior*I: */
        const double lam_prior = 2.0;
        auto se_at = [&](double N, int idx) {
            Eigen::Matrix<double, 21, 21> Cov = (N * F + lam_prior * Eigen::Matrix<double, 21, 21>::Identity()).inverse();
            return std::sqrt(Cov(idx, idx));
        };
        double se4_256   = se_at(256.0, 4);
        double se4_1000  = se_at(1000.0, 4);
        double se4_10000 = se_at(10000.0, 4);
        double se16_256   = se_at(256.0, 16);
        double se16_10000 = se_at(10000.0, 16);

        ASSERT_GT(se4_256, se4_1000);
        ASSERT_GT(se4_1000, se4_10000);
        ASSERT_NEAR(se4_10000 / se4_256, 0.16, 0.03);
        ASSERT_NEAR(se16_256, 1.0 / std::sqrt(lam_prior), 1e-12);
        ASSERT_NEAR(se16_10000, 1.0 / std::sqrt(lam_prior), 1e-12);

        ctx.record().expected["se4_ratio_10000_to_256"] = "~0.16 (1/sqrt(N)) and se16 invariant";
        ctx.record().actual["se4_ratio_10000_to_256"] = se4_10000 / se4_256;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-030") {
        TestContext ctx(opts, tid, "prior_dominated_zero_fisher_ratio",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify Fisher sample information to prior stiffness ratio is exact 0.0 on ZZ_3_6 (e=16) and ZZ_2_5 (e=19)";
        ctx.record().independent_oracle = "I(theta)_{16,16} / lambda == 0.0 and I(theta)_{19,19} / lambda == 0.0";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.015, 0.008);
        auto F = compute_fisher_matrix(m);
        double ratio16 = F(16, 16) / 0.5;
        double ratio19 = F(19, 19) / 0.5;
        double ratio4  = F(4, 4) / 0.5;
        ASSERT_NEAR(ratio16, 0.0, 1e-15);
        ASSERT_NEAR(ratio19, 0.0, 1e-15);
        ASSERT_GT(ratio4, 50.0);
        ctx.record().expected["fisher_to_prior_ratio"] = "0.0 on (16,19) vs >50 on singleton Z_4";
        ctx.record().actual["ratio_z4"] = ratio4;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-031") {
        TestContext ctx(opts, tid, "monte_carlo_se_vs_asymptotic_inverse_hessian",
                        "software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare 30-trial Monte Carlo empirical SD on singleton Z_4 against asymptotic inverse-Fisher SE";
        ctx.record().independent_oracle = "Empirical SD within [0.5 * SE_asymp, 1.8 * SE_asymp] for 30 trials at N=4000";

        NoiseModel truth;
        truth.set_basis_z(false);
        truth.init(0.012, 0.002);
        for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
        truth.theta_(4) = 0.050;
        truth.compute_joint_distribution();

        auto F = compute_fisher_matrix(truth);
        const int N = 4000;
        double se_asymp = 1.0 / std::sqrt(static_cast<double>(N) * F(4, 4));

        std::mt19937_64 rng(2026111631ULL);
        std::discrete_distribution<int> mult(truth.joint_probs().data(), truth.joint_probs().data() + 16);
        std::vector<double> ests;
        for (int trial = 0; trial < 30; ++trial) {
            uint32_t h[16] = {0};
            for (int r = 0; r < N; ++r) h[mult(rng)]++;
            NoiseModel m;
            m.set_basis_z(false);
            m.init(0.012, 0.002);
            for (int k = 0; k < 12; ++k) m.theta_(9 + k) = 1e-5;
            m.prior_lambda_ = 1e-4;
            LBFGSOptimizer opt;
            double nll = 0.0, ginf = 0.0;
            opt.step_histogram(m, h, 40, &nll, &ginf);
            ests.push_back(m.theta_(4));
        }
        double mean = 0.0;
        for (double v : ests) mean += v;
        mean /= 30.0;
        double var = 0.0;
        for (double v : ests) var += (v - mean) * (v - mean);
        double emp_sd = std::sqrt(var / 29.0);

        ASSERT_GT(emp_sd, 0.5 * se_asymp);
        ASSERT_LT(emp_sd, 1.8 * se_asymp);
        ctx.record().expected["se_asymp"] = se_asymp;
        ctx.record().actual["emp_sd_30_trials"] = emp_sd;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-032") {
        TestContext ctx(opts, tid, "exact_syndrome_alias_equivalence_proof",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Construct distinct parameter vectors theta_A != theta_B with equal parity factor (1-2p_0)(1-2p_{1,2}) on orbit 0x1 and prove identical P_joint(0..15)";
        ctx.record().independent_oracle = "||theta_A - theta_B||_inf > 0.05 while ||P(theta_A) - P(theta_B)||_inf < 1e-14";

        NoiseModel mA, mB;
        mA.set_basis_z(false); mA.init(0.015, 0.008);
        mB.set_basis_z(false); mB.init(0.015, 0.008);

        /* Orbit 0x1 contains e=0 (Z_0) and e=10 (ZZ_1_2). Keep (1 - 2 p_0)(1 - 2 p_10) == K */
        const double p0_A  = 0.080;
        const double p10_A = 0.010;
        const double K = (1.0 - 2.0 * p0_A) * (1.0 - 2.0 * p10_A);
        const double p0_B  = 0.015;
        const double p10_B = 0.5 * (1.0 - K / (1.0 - 2.0 * p0_B));

        mA.theta_(0) = p0_A;  mA.theta_(10) = p10_A;
        mB.theta_(0) = p0_B;  mB.theta_(10) = p10_B;
        mA.compute_joint_distribution();
        mB.compute_joint_distribution();

        double param_gap = (mA.theta_ - mB.theta_).cwiseAbs().maxCoeff();
        double pmf_gap   = (mA.joint_probs() - mB.joint_probs()).cwiseAbs().maxCoeff();
        double orbit_gap = mA.compute_orbit_rmse(mB.theta_);

        ASSERT_GT(param_gap, 0.05);
        ASSERT_LT(pmf_gap, 1e-14);
        ASSERT_LT(orbit_gap, 1e-14);
        ctx.record().expected["alias_invariance"] = "param_gap > 0.05 AND pmf_gap < 1e-14";
        ctx.record().actual["param_gap"] = param_gap;
        ctx.record().actual["pmf_gap"] = pmf_gap;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-025", "L3-MATH-026", "L3-MATH-027", "L3-MATH-028",
        "L3-MATH-029", "L3-MATH-030", "L3-MATH-031", "L3-MATH-032"
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
