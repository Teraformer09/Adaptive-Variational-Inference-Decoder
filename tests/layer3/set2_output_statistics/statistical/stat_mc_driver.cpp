/* ===========================================================================
 * stat_mc_driver.cpp — Native C++ >=30-Seed Monte Carlo Driver for L3-STAT-001..015
 * Executes genuine C++ NoiseModel, HistogramLBFGSSolver, LBFGSOptimizer, and
 * avid_fast_burst_detector_t across >=30 independent seeds and emits JSON to stdout.
 * =========================================================================*/
#include "inference/noise_model.h"
#include "inference/lbfgs_optimizer.h"
#include "estimator/lbfgs_solver.h"
#include "control/causal_scheduler.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

void sample_multinomial_hist(std::mt19937_64 &rng,
                             const NoiseModel::StateVec &pmf,
                             uint32_t n_samples,
                             uint32_t out_hist[16],
                             double p_meas_bitflip = 0.0) {
    std::memset(out_hist, 0, 16 * sizeof(uint32_t));
    double cdf[16];
    double acc = 0.0;
    for (int s = 0; s < 16; ++s) {
        acc += std::max(0.0, pmf(s));
        cdf[s] = acc;
    }
    if (acc <= 0.0) {
        out_hist[0] = n_samples;
        return;
    }
    for (int s = 0; s < 16; ++s) cdf[s] /= acc;
    cdf[15] = 1.0;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (uint32_t i = 0; i < n_samples; ++i) {
        double u = u01(rng);
        uint8_t s = 0;
        while (s < 15 && u > cdf[s]) ++s;
        if (p_meas_bitflip > 0.0) {
            for (int b = 0; b < 4; ++b) {
                if (u01(rng) < p_meas_bitflip) s ^= static_cast<uint8_t>(1u << b);
            }
        }
        out_hist[s & 0xFu]++;
    }
}

double compute_kl_emp_to_model(const uint32_t hist[16], const NoiseModel::StateVec &p_hat) {
    uint64_t tot = 0;
    for (int s = 0; s < 16; ++s) tot += hist[s];
    if (tot == 0) return 0.0;
    double kl = 0.0;
    for (int s = 0; s < 16; ++s) {
        if (hist[s] == 0) continue;
        double pe = static_cast<double>(hist[s]) / static_cast<double>(tot);
        double pm = std::max(1e-12, p_hat(s));
        kl += pe * std::log(pe / pm);
    }
    return std::max(0.0, kl);
}

double compute_tvd_states(const NoiseModel::StateVec &a, const NoiseModel::StateVec &b) {
    double s = 0.0;
    for (int i = 0; i < 16; ++i) s += std::abs(a(i) - b(i));
    return 0.5 * s;
}

void print_vec_json(const std::vector<double> &v) {
    std::printf("[");
    for (size_t i = 0; i < v.size(); ++i) {
        std::printf("%s%.10g", (i ? ", " : ""), v[i]);
    }
    std::printf("]");
}

} // namespace

int main(int argc, char **argv) {
    std::string tid = "L3-STAT-001";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test-id") == 0 && i + 1 < argc) {
            tid = argv[++i];
        }
    }

    constexpr int N_SEEDS = 32; /* >= 30 independent seeds per spec */
    HistogramLBFGSSolver solver;

    if (tid == "L3-STAT-001") {
        const double levels[3] = {0.015, 0.040, 0.080};
        std::vector<double> errors, tvds;
        for (int li = 0; li < 3; ++li) {
            NoiseModel truth;
            truth.init(0.012, 0.0);
            for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
            truth.theta_(4) = levels[li];
            truth.compute_joint_distribution();
            for (int s = 0; s < N_SEEDS; ++s) {
                std::mt19937_64 rng(20260100ULL + li * 1000ULL + s);
                uint32_t hist[16];
                sample_multinomial_hist(rng, truth.joint_probs(), 2000, hist);
                NoiseModel est;
                est.init(0.012, 0.008);
                est.prior_lambda_ = 0.005;
                solver.fit_histogram(est, hist, 30);
                est.compute_joint_distribution();
                errors.push_back(est.theta_(4) - levels[li]);
                tvds.push_back(compute_tvd_states(truth.joint_probs(), est.joint_probs()));
            }
        }
        std::printf("{\"test_id\":\"%s\",\"seeds_per_level\":%d,\"errors\":", tid.c_str(), N_SEEDS);
        print_vec_json(errors);
        std::printf(",\"tvds\":");
        print_vec_json(tvds);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-002") {
        std::vector<double> param_rmses, orbit_rmses, kls, singleton_errs;
        NoiseModel truth;
        truth.init(0.010, 0.0);
        for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
        truth.theta_(1) = 0.032;
        truth.theta_(4) = 0.055;
        truth.theta_(7) = 0.042;
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20260200ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 2500, hist);
            NoiseModel est;
            est.init(0.012, 0.004);
            est.prior_lambda_ = 0.005;
            solver.fit_histogram(est, hist, 30);
            est.compute_joint_distribution();
            param_rmses.push_back(est.compute_param_21_rmse(truth.theta_));
            orbit_rmses.push_back(est.compute_orbit_rmse(truth.theta_));
            kls.push_back(compute_kl_emp_to_model(hist, est.joint_probs()));
            double se = std::sqrt(((est.theta_(1) - 0.032) * (est.theta_(1) - 0.032) +
                                   (est.theta_(4) - 0.055) * (est.theta_(4) - 0.055) +
                                   (est.theta_(7) - 0.042) * (est.theta_(7) - 0.042)) / 3.0);
            singleton_errs.push_back(se);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"param_rmses\":", tid.c_str(), N_SEEDS);
        print_vec_json(param_rmses);
        std::printf(",\"orbit_rmses\":");
        print_vec_json(orbit_rmses);
        std::printf(",\"kls\":");
        print_vec_json(kls);
        std::printf(",\"singleton_rmses\":");
        print_vec_json(singleton_errs);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-003") {
        std::vector<double> orbit_rmses, singleton_mae, aliased_mae;
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20260300ULL + s);
            std::uniform_real_distribution<double> up(0.008, 0.030);
            NoiseModel truth;
            for (int q = 0; q < 9; ++q) truth.theta_(q) = up(rng);
            for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
            truth.compute_joint_distribution();
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 3000, hist);
            NoiseModel est;
            est.init(0.012, 0.002);
            est.prior_lambda_ = 0.005;
            solver.fit_histogram(est, hist, 30);
            orbit_rmses.push_back(est.compute_orbit_rmse(truth.theta_));
            double sm = (std::abs(est.theta_(1) - truth.theta_(1)) +
                         std::abs(est.theta_(4) - truth.theta_(4)) +
                         std::abs(est.theta_(7) - truth.theta_(7))) / 3.0;
            double am = (std::abs(est.theta_(2) - truth.theta_(2)) +
                         std::abs(est.theta_(5) - truth.theta_(5))) / 2.0;
            singleton_mae.push_back(sm);
            aliased_mae.push_back(am);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"orbit_rmses\":", tid.c_str(), N_SEEDS);
        print_vec_json(orbit_rmses);
        std::printf(",\"singleton_mae\":");
        print_vec_json(singleton_mae);
        std::printf(",\"aliased_mae\":");
        print_vec_json(aliased_mae);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-004") {
        std::vector<double> orbit_rmses, pair_45_hat, inactive_pair_hat;
        NoiseModel truth;
        truth.init(0.012, 0.004);
        truth.theta_(4) = 0.055;
        truth.theta_(5) = 0.045;
        truth.theta_(9 + 3) = 0.080; /* pair (4,5) */
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20260400ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 3000, hist);
            NoiseModel est;
            est.init(0.012, 0.008);
            solver.fit_histogram(est, hist, 30);
            orbit_rmses.push_back(est.compute_orbit_rmse(truth.theta_));
            pair_45_hat.push_back(est.theta_(9 + 3));
            inactive_pair_hat.push_back(est.theta_(9 + 0));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"orbit_rmses\":", tid.c_str(), N_SEEDS);
        print_vec_json(orbit_rmses);
        std::printf(",\"pair_45_hat\":");
        print_vec_json(pair_45_hat);
        std::printf(",\"inactive_pair_hat\":");
        print_vec_json(inactive_pair_hat);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-005") {
        std::vector<double> clean_bias, meas_bias;
        NoiseModel truth;
        truth.init(0.012, 0.004);
        truth.theta_(4) = 0.035;
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng1(20260500ULL + s);
            std::mt19937_64 rng2(20260500ULL + s);
            uint32_t h_clean[16], h_meas[16];
            sample_multinomial_hist(rng1, truth.joint_probs(), 2500, h_clean, 0.0);
            sample_multinomial_hist(rng2, truth.joint_probs(), 2500, h_meas, 0.012);
            NoiseModel e1, e2;
            e1.init(0.012, 0.008);
            e2.init(0.012, 0.008);
            solver.fit_histogram(e1, h_clean, 25);
            solver.fit_histogram(e2, h_meas, 25);
            double err_clean_mean = 0.0, err_meas_mean = 0.0;
            for (int q = 0; q < 9; ++q) {
                err_clean_mean += (e1.theta_(q) - truth.theta_(q));
                err_meas_mean += (e2.theta_(q) - truth.theta_(q));
            }
            clean_bias.push_back(err_clean_mean / 9.0);
            meas_bias.push_back(err_meas_mean / 9.0);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"clean_err\":", tid.c_str(), N_SEEDS);
        print_vec_json(clean_bias);
        std::printf(",\"meas_err\":");
        print_vec_json(meas_bias);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-006") {
        const uint32_t ws[4] = {256, 1000, 5000, 20000};
        NoiseModel truth;
        truth.init(0.012, 0.0);
        for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
        truth.theta_(4) = 0.050;
        truth.compute_joint_distribution();
        std::vector<double> mean_orbit_rmse(4, 0.0), var_z4(4, 0.0);
        for (int wi = 0; wi < 4; ++wi) {
            std::vector<double> z4s;
            double sum_orb = 0.0;
            for (int s = 0; s < N_SEEDS; ++s) {
                std::mt19937_64 rng(20260600ULL + wi * 500ULL + s);
                uint32_t hist[16];
                sample_multinomial_hist(rng, truth.joint_probs(), ws[wi], hist);
                NoiseModel est;
                est.init(0.012, 0.004);
                est.prior_lambda_ = 0.001;
                solver.fit_histogram(est, hist, 30);
                z4s.push_back(est.theta_(4));
                sum_orb += est.compute_orbit_rmse(truth.theta_);
            }
            mean_orbit_rmse[wi] = sum_orb / N_SEEDS;
            double m = std::accumulate(z4s.begin(), z4s.end(), 0.0) / N_SEEDS;
            double sq = 0.0;
            for (double v : z4s) sq += (v - m) * (v - m);
            var_z4[wi] = sq / (N_SEEDS - 1);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"windows\":[256,1000,5000,20000],\"mean_orbit_rmse\":", tid.c_str(), N_SEEDS);
        print_vec_json(mean_orbit_rmse);
        std::printf(",\"var_z4\":");
        print_vec_json(var_z4);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-007") {
        std::vector<double> nlls, orbit_rmses, param_rmses;
        NoiseModel truth;
        truth.init(0.014, 0.006);
        truth.theta_(4) = 0.060;
        truth.theta_(9 + 3) = 0.065;
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20260700ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 1024, hist);
            NoiseModel est;
            est.init(0.012, 0.008);
            double nll = solver.fit_histogram(est, hist, 25);
            nlls.push_back(nll);
            orbit_rmses.push_back(est.compute_orbit_rmse(truth.theta_));
            param_rmses.push_back(est.compute_param_21_rmse(truth.theta_));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"nlls\":", tid.c_str(), N_SEEDS);
        print_vec_json(nlls);
        std::printf(",\"orbit_rmses\":");
        print_vec_json(orbit_rmses);
        std::printf(",\"param_rmses\":");
        print_vec_json(param_rmses);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-008") {
        const double lambdas[4] = {0.0, 0.04, 0.5, 5.0};
        NoiseModel truth;
        truth.init(0.012, 0.008);
        truth.theta_(4) = 0.095;
        truth.compute_joint_distribution();
        std::vector<double> mean_dist_to_prior(4, 0.0), var_z4(4, 0.0);
        for (int li = 0; li < 4; ++li) {
            std::vector<double> z4s;
            double sum_dist = 0.0;
            for (int s = 0; s < N_SEEDS; ++s) {
                std::mt19937_64 rng(20260800ULL + s); /* shared histograms across lambdas */
                uint32_t hist[16];
                sample_multinomial_hist(rng, truth.joint_probs(), 500, hist);
                NoiseModel est;
                est.init(0.012, 0.008);
                est.prior_lambda_ = lambdas[li];
                LBFGSOptimizer opt;
                double nll = 0.0, ginf = 0.0;
                opt.step_histogram(est, hist, 35, &nll, &ginf);
                z4s.push_back(est.theta_(4));
                sum_dist += (est.theta_ - est.prior_theta_).norm();
            }
            mean_dist_to_prior[li] = sum_dist / N_SEEDS;
            double m = std::accumulate(z4s.begin(), z4s.end(), 0.0) / N_SEEDS;
            double sq = 0.0;
            for (double v : z4s) sq += (v - m) * (v - m);
            var_z4[li] = sq / (N_SEEDS - 1);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"lambdas\":[0.0,0.04,0.5,5.0],\"mean_dist_to_prior\":", tid.c_str(), N_SEEDS);
        print_vec_json(mean_dist_to_prior);
        std::printf(",\"var_z4\":");
        print_vec_json(var_z4);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-009") {
        const double p0s[4] = {0.002, 0.012, 0.050, 0.150};
        std::vector<double> orbit_spreads, null_spreads;
        NoiseModel truth;
        truth.init(0.012, 0.006);
        truth.theta_(4) = 0.055;
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20260900ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 2000, hist);
            double z4_min = 1.0, z4_max = 0.0;
            double null_min = 1.0, null_max = 0.0;
            for (int pi = 0; pi < 4; ++pi) {
                NoiseModel m;
                m.init(p0s[pi], p0s[pi]);
                m.prior_lambda_ = 0.0; /* pure MLE to expose null-space invariance vs singleton uniqueness */
                LBFGSOptimizer opt;
                double nll = 0.0, ginf = 0.0;
                opt.step_histogram(m, hist, 45, &nll, &ginf);
                z4_min = std::min(z4_min, m.theta_(4));
                z4_max = std::max(z4_max, m.theta_(4));
                null_min = std::min(null_min, m.theta_(9 + 7)); /* (3,6) has g=0 in X-basis! */
                null_max = std::max(null_max, m.theta_(9 + 7));
            }
            orbit_spreads.push_back(z4_max - z4_min);
            null_spreads.push_back(null_max - null_min);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"singleton_spreads\":", tid.c_str(), N_SEEDS);
        print_vec_json(orbit_spreads);
        std::printf(",\"null_spreads\":");
        print_vec_json(null_spreads);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-010") {
        std::vector<double> kl_in_model, kl_mismatched;
        NoiseModel truth;
        truth.init(0.012, 0.004);
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20261000ULL + s);
            uint32_t h_in[16], h_mis[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 2000, h_in);
            /* Inject out-of-model high-weight syndrome concentration (e.g. 3-check streak s=0xF & 0x7) */
            std::memcpy(h_mis, h_in, sizeof(h_in));
            uint32_t move = std::min<uint32_t>(500, h_mis[0]);
            h_mis[0] -= move;
            h_mis[15] += move / 2;
            h_mis[7] += (move - move / 2);

            NoiseModel e1, e2;
            e1.init(0.012, 0.008);
            e2.init(0.012, 0.008);
            solver.fit_histogram(e1, h_in, 30);
            solver.fit_histogram(e2, h_mis, 30);
            e1.compute_joint_distribution();
            e2.compute_joint_distribution();
            kl_in_model.push_back(compute_kl_emp_to_model(h_in, e1.joint_probs()));
            kl_mismatched.push_back(compute_kl_emp_to_model(h_mis, e2.joint_probs()));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"kl_in_model\":", tid.c_str(), N_SEEDS);
        print_vec_json(kl_in_model);
        std::printf(",\"kl_mismatched\":");
        print_vec_json(kl_mismatched);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-011") {
        /* Aliased pair (Z_0 [e=0], ZZ_1_2 [e=10]) both share signature 0x1; ZZ_3_6 [e=16] has signature 0x0 */
        std::vector<double> orbit_rmses, coord_z0_errs;
        NoiseModel truth;
        truth.init(0.012, 0.004);
        truth.theta_(0) = 0.065;      /* high Z_0 */
        truth.theta_(10) = 0.002;     /* low ZZ_1_2 (aliased with Z_0 on 0x1!) */
        truth.theta_(16) = 0.090;     /* invisible stabilizer-like ZZ_3_6 (g=0) */
        truth.compute_joint_distribution();
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20261100ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), 3000, hist);
            NoiseModel est;
            est.init(0.012, 0.008);
            solver.fit_histogram(est, hist, 30);
            orbit_rmses.push_back(est.compute_orbit_rmse(truth.theta_));
            coord_z0_errs.push_back(std::abs(est.theta_(16) - truth.theta_(16)));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"orbit_rmses\":", tid.c_str(), N_SEEDS);
        print_vec_json(orbit_rmses);
        std::printf(",\"null_coord_abs_err\":");
        print_vec_json(coord_z0_errs);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-012") {
        std::vector<double> dynamic_rmses, static_rmses, corrs;
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20261200ULL + s);
            NoiseModel est;
            est.init(0.012, 0.008);
            double sq_dyn = 0.0, sq_stat = 0.0;
            std::vector<double> trues, hats;
            for (int w = 0; w < 20; ++w) {
                double p4 = 0.015 + 0.070 * (static_cast<double>(w) / 19.0);
                NoiseModel truth;
                truth.init(0.012, 0.004);
                truth.theta_(4) = p4;
                truth.compute_joint_distribution();
                uint32_t hist[16];
                sample_multinomial_hist(rng, truth.joint_probs(), 512, hist);
                solver.fit_histogram(est, hist, 25);
                sq_dyn += (est.theta_(4) - p4) * (est.theta_(4) - p4);
                sq_stat += (0.012 - p4) * (0.012 - p4);
                trues.push_back(p4);
                hats.push_back(est.theta_(4));
            }
            dynamic_rmses.push_back(std::sqrt(sq_dyn / 20.0));
            static_rmses.push_back(std::sqrt(sq_stat / 20.0));
            double mt = std::accumulate(trues.begin(), trues.end(), 0.0) / 20.0;
            double mh = std::accumulate(hats.begin(), hats.end(), 0.0) / 20.0;
            double cov = 0.0, vt = 0.0, vh = 0.0;
            for (int w = 0; w < 20; ++w) {
                cov += (trues[w] - mt) * (hats[w] - mh);
                vt += (trues[w] - mt) * (trues[w] - mt);
                vh += (hats[w] - mh) * (hats[w] - mh);
            }
            corrs.push_back(cov / std::max(1e-12, std::sqrt(vt * vh)));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"dynamic_rmses\":", tid.c_str(), N_SEEDS);
        print_vec_json(dynamic_rmses);
        std::printf(",\"static_rmses\":");
        print_vec_json(static_rmses);
        std::printf(",\"tracking_corrs\":");
        print_vec_json(corrs);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-013") {
        /* 60-round burst inside 10,000-round stationary window */
        std::vector<double> slow_z4_shifts, fast_trigger_lags;
        NoiseModel stat_m, burst_m;
        stat_m.init(0.012, 0.004);
        stat_m.compute_joint_distribution();
        burst_m.init(0.012, 0.004);
        burst_m.theta_(4) = 0.28;
        burst_m.theta_(9 + 3) = 0.25;
        burst_m.compute_joint_distribution();

        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20261300ULL + s);
            uint32_t h_stat[16], h_burst[16], h_comb[16];
            sample_multinomial_hist(rng, stat_m.joint_probs(), 9940, h_stat);
            sample_multinomial_hist(rng, burst_m.joint_probs(), 60, h_burst);
            for (int i = 0; i < 16; ++i) h_comb[i] = h_stat[i] + h_burst[i];

            NoiseModel e_clean, e_comb;
            e_clean.init(0.012, 0.008);
            e_comb.init(0.012, 0.008);
            solver.fit_histogram(e_clean, h_stat, 25);
            solver.fit_histogram(e_comb, h_comb, 25);
            slow_z4_shifts.push_back(std::abs(e_comb.theta_(4) - e_clean.theta_(4)));

            avid_fast_burst_detector_t det;
            avid_fast_detector_init(&det, 0.35, 4);
            int trig_lag = -1;
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            for (uint64_t r = 0; r < 200; ++r) {
                bool in_b = (r >= 100 && r < 160);
                uint8_t sx = 0;
                if (in_b) {
                    if (u01(rng) < 0.72) sx = 0x6u;
                } else {
                    if (u01(rng) < 0.04) sx = 0x1u;
                }
                if (avid_fast_detector_step(&det, r, sx, 0) && trig_lag < 0 && r >= 100) {
                    trig_lag = static_cast<int>(r - 100 + 1);
                }
            }
            fast_trigger_lags.push_back(static_cast<double>(trig_lag));
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"slow_z4_shifts\":", tid.c_str(), N_SEEDS);
        print_vec_json(slow_z4_shifts);
        std::printf(",\"fast_trigger_lags\":");
        print_vec_json(fast_trigger_lags);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-014") {
        std::vector<double> trigger_counts, trigger_rates;
        NoiseModel stat_m;
        stat_m.init(0.006, 0.001); /* low stationary syndrome density p_nonzero ~ 0.055 */
        stat_m.compute_joint_distribution();
        const double p_nonzero = 1.0 - stat_m.joint_probs()(0);
        for (int s = 0; s < N_SEEDS; ++s) {
            std::mt19937_64 rng(20261400ULL + s);
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            avid_fast_burst_detector_t det;
            avid_fast_detector_init(&det, 0.35, 4);
            for (uint64_t r = 0; r < 50000; ++r) {
                uint8_t sx = (u01(rng) < p_nonzero) ? 0x1u : 0x0u;
                avid_fast_detector_step(&det, r, sx, 0u);
            }
            trigger_counts.push_back(static_cast<double>(det.trigger_count));
            trigger_rates.push_back(static_cast<double>(det.trigger_count) / 50000.0);
        }
        std::printf("{\"test_id\":\"%s\",\"seeds\":%d,\"p_nonzero\":%.8f,\"trigger_counts\":",
                    tid.c_str(), N_SEEDS, p_nonzero);
        print_vec_json(trigger_counts);
        std::printf(",\"trigger_rates\":");
        print_vec_json(trigger_rates);
        std::printf("}\n");
        return 0;
    }

    if (tid == "L3-STAT-015") {
        /* Empirical 95% CI coverage on identifiable singletons Z_1, Z_4, Z_7 across 60 MC trials */
        constexpr int N_CAL = 60;
        constexpr uint32_t W = 4000;
        NoiseModel truth;
        truth.init(0.012, 0.0);
        for (int q = 0; q < 9; ++q) truth.theta_(q) = 1e-5;
        for (int k = 0; k < 12; ++k) truth.theta_(9 + k) = 1e-5;
        truth.theta_(1) = 0.035; /* Z_1 -> unique orbit 0x3 */
        truth.theta_(4) = 0.055; /* Z_4 -> unique orbit 0x6 */
        truth.theta_(7) = 0.042; /* Z_7 -> unique orbit 0xC */
        truth.compute_joint_distribution();

        const int idxs[3] = {1, 4, 7};
        /* Compute asymptotic Fisher SE for each singleton via central FD dP/dtheta_e */
        double fisher_se[3] = {0};
        const double eps = 1e-6;
        for (int j = 0; j < 3; ++j) {
            int e = idxs[j];
            NoiseModel mp = truth, mm = truth;
            mp.theta_(e) += eps;
            mm.theta_(e) -= eps;
            mp.compute_joint_distribution();
            mm.compute_joint_distribution();
            double info = 0.0;
            for (int s = 0; s < 16; ++s) {
                double dp = (mp.joint_probs()(s) - mm.joint_probs()(s)) / (2.0 * eps);
                info += (dp * dp) / std::max(1e-12, truth.joint_probs()(s));
            }
            fisher_se[j] = 1.0 / std::sqrt(static_cast<double>(W) * info);
        }

        std::vector<double> covered_flags, z_scores;
        for (int s = 0; s < N_CAL; ++s) {
            std::mt19937_64 rng(20261500ULL + s);
            uint32_t hist[16];
            sample_multinomial_hist(rng, truth.joint_probs(), W, hist);
            /* Pure MLE over active singletons via LBFGSOptimizer with tiny prior */
            NoiseModel est = truth;
            est.theta_(1) = 0.020;
            est.theta_(4) = 0.020;
            est.theta_(7) = 0.020;
            est.prior_lambda_ = 0.0;
            LBFGSOptimizer opt;
            double nll = 0.0, ginf = 0.0;
            opt.step_histogram(est, hist, 50, &nll, &ginf);
            for (int j = 0; j < 3; ++j) {
                int e = idxs[j];
                double z = (est.theta_(e) - truth.theta_(e)) / fisher_se[j];
                z_scores.push_back(z);
                covered_flags.push_back(std::abs(z) <= 1.959964 ? 1.0 : 0.0);
            }
        }
        std::printf("{\"test_id\":\"%s\",\"trials\":%d,\"fisher_se\":[%.8f,%.8f,%.8f],\"covered\":",
                    tid.c_str(), N_CAL, fisher_se[0], fisher_se[1], fisher_se[2]);
        print_vec_json(covered_flags);
        std::printf(",\"z_scores\":");
        print_vec_json(z_scores);
        std::printf("}\n");
        return 0;
    }

    std::fprintf(stderr, "Unknown L3-STAT test ID: %s\n", tid.c_str());
    return 2;
}
