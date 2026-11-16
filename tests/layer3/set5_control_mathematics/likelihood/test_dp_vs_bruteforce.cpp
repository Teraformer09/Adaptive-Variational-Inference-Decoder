#include "test_assertions.h"
#include "test_common.h"
#include "inference/noise_model.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace avid::layer3_test;

namespace {

/* Independent first-principles parity-check & XOR subset oracle (zero shared NoiseModel code) */
uint8_t ref_hx_sig_single(int q) {
    /* X-checks: c0={0,1}, c1={1,2,4,5}, c2={3,4,6,7}, c3={7,8} */
    uint8_t s = 0;
    if (q == 0 || q == 1) s |= 0x1u;
    if (q == 1 || q == 2 || q == 4 || q == 5) s |= 0x2u;
    if (q == 3 || q == 4 || q == 6 || q == 7) s |= 0x4u;
    if (q == 7 || q == 8) s |= 0x8u;
    return s;
}

uint8_t ref_hz_sig_single(int q) {
    /* Z-checks: z0={0,1,3,4}, z1={2,5}, z2={3,6}, z3={4,5,7,8} */
    uint8_t s = 0;
    if (q == 0 || q == 1 || q == 3 || q == 4) s |= 0x1u;
    if (q == 2 || q == 5) s |= 0x2u;
    if (q == 3 || q == 6) s |= 0x4u;
    if (q == 4 || q == 5 || q == 7 || q == 8) s |= 0x8u;
    return s;
}

static const int REF_PAIRS[12][2] = {
    {0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8},
    {0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}
};

uint8_t ref_event_sig(int e, bool basis_z) {
    if (e < 9) {
        return basis_z ? ref_hz_sig_single(e) : ref_hx_sig_single(e);
    }
    int k = e - 9;
    uint8_t gi = basis_z ? ref_hz_sig_single(REF_PAIRS[k][0]) : ref_hx_sig_single(REF_PAIRS[k][0]);
    uint8_t gj = basis_z ? ref_hz_sig_single(REF_PAIRS[k][1]) : ref_hx_sig_single(REF_PAIRS[k][1]);
    return static_cast<uint8_t>(gi ^ gj);
}

/* Independent Fast Walsh-Hadamard Transform oracle on GF(2)^4 */
void ref_fwht_pmf(const double p[21], bool basis_z, double out_pmf[16]) {
    double walsh[16];
    for (int u = 0; u < 16; ++u) {
        double prod = 1.0;
        for (int e = 0; e < 21; ++e) {
            uint8_t g = ref_event_sig(e, basis_z);
            int dot = __builtin_popcount(static_cast<unsigned>(u & g)) & 1;
            if (dot == 1) {
                prod *= (1.0 - 2.0 * p[e]);
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
        out_pmf[s] = acc / 16.0;
    }
}

/* Brute-force 2^k subset enumeration over active indices (exact, zero DP) */
void ref_bruteforce_subset_pmf(const std::vector<int> &active_idx,
                               const double p_all[21],
                               bool basis_z,
                               double out_pmf[16]) {
    for (int s = 0; s < 16; ++s) out_pmf[s] = 0.0;
    const size_t k = active_idx.size();
    const uint64_t total_masks = (1ULL << k);
    bool is_active[21] = {false};
    for (int idx : active_idx) is_active[idx] = true;

    /* First enumerate active k-subset */
    double base_pmf[16] = {0.0};
    for (uint64_t mask = 0; mask < total_masks; ++mask) {
        uint8_t synd = 0;
        double prob = 1.0;
        for (size_t bit = 0; bit < k; ++bit) {
            int e = active_idx[bit];
            if ((mask >> bit) & 1ULL) {
                synd ^= ref_event_sig(e, basis_z);
                prob *= p_all[e];
            } else {
                prob *= (1.0 - p_all[e]);
            }
        }
        base_pmf[synd] += prob;
    }
    /* Account for inactive channels at p_all[e] (e.g. 1e-6 floor in NoiseModel) */
    for (int s = 0; s < 16; ++s) out_pmf[s] = base_pmf[s];
    for (int e = 0; e < 21; ++e) {
        if (is_active[e]) continue;
        double pe = p_all[e];
        uint8_t ge = ref_event_sig(e, basis_z);
        if (ge == 0u || pe == 0.0) continue;
        double tmp[16];
        for (int s = 0; s < 16; ++s) {
            tmp[s] = (1.0 - pe) * out_pmf[s] + pe * out_pmf[s ^ ge];
        }
        for (int s = 0; s < 16; ++s) out_pmf[s] = tmp[s];
    }
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;

    if (tid == "L3-MATH-001") {
        TestContext ctx(opts, tid, "joint_dp_probability_normalization",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Probability normalization across 20 random 21-parameter vectors in both X and Z bases";
        ctx.record().independent_oracle = "sum_{s=0..15} P_joint(s | theta) == 1.0 within 1e-14";

        std::mt19937_64 rng(2026111601ULL);
        std::uniform_real_distribution<double> udist(0.0005, 0.35);
        double max_err = 0.0;
        for (int trial = 0; trial < 20; ++trial) {
            for (bool bz : {false, true}) {
                NoiseModel m;
                m.set_basis_z(bz);
                for (int e = 0; e < 21; ++e) m.theta_(e) = udist(rng);
                m.compute_joint_distribution();
                double sum_p = m.joint_probs().sum();
                double err = std::abs(sum_p - 1.0);
                max_err = std::max(max_err, err);
                ASSERT_NEAR(sum_p, 1.0, 1e-14);
            }
        }
        ctx.record().expected["sum_p"] = "1.0 +/- 1e-14 across 40 (theta, basis) evaluations";
        ctx.record().actual["max_abs_sum_err"] = max_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-002") {
        TestContext ctx(opts, tid, "joint_dp_nonnegative_probabilities",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Non-negativity of all 16 joint syndrome probabilities including boundary parameters 1e-6 and 0.49";
        ctx.record().independent_oracle = "All P_joint(s | theta) >= 0.0 and <= 1.0";

        double min_prob = 1.0;
        const double test_vals[] = {1e-6, 1e-5, 0.012, 0.25, 0.45, 0.49};
        for (double v : test_vals) {
            for (bool bz : {false, true}) {
                NoiseModel m;
                m.set_basis_z(bz);
                for (int e = 0; e < 21; ++e) m.theta_(e) = v;
                m.compute_joint_distribution();
                for (int s = 0; s < 16; ++s) {
                    double ps = m.joint_probs()(s);
                    min_prob = std::min(min_prob, ps);
                    ASSERT_TRUE(std::isfinite(ps));
                    ASSERT_GE(ps, 0.0);
                    ASSERT_LE(ps, 1.0 + 1e-14);
                }
            }
        }
        ctx.record().expected["min_prob"] = ">= 0.0 across boundary sweep [1e-6 .. 0.49]";
        ctx.record().actual["min_prob"] = min_prob;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-003") {
        TestContext ctx(opts, tid, "joint_dp_zero_event_asymptotic",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Zero-event limit at sanitize floor theta = [1e-6]*21 vs exact first-order expansion";
        ctx.record().independent_oracle = "1 - P(0) matches 19 * 1e-6 (19 non-zero-signature mechanisms out of 21) within O(p^2)";

        double p_ref[21];
        for (int e = 0; e < 21; ++e) p_ref[e] = 0.0;
        double exact_zero_pmf[16];
        ref_fwht_pmf(p_ref, false, exact_zero_pmf);
        ASSERT_NEAR(exact_zero_pmf[0], 1.0, 1e-15);

        NoiseModel m;
        m.set_basis_z(false);
        for (int e = 0; e < 21; ++e) m.theta_(e) = 1e-6;
        m.compute_joint_distribution();
        int active_sigs = 0;
        for (int e = 0; e < 21; ++e) {
            if (ref_event_sig(e, false) != 0u) active_sigs++;
        }
        ASSERT_EQ(active_sigs, 19);
        double non_zero_mass = 1.0 - m.joint_probs()(0);
        double first_order = static_cast<double>(active_sigs) * 1e-6;
        ASSERT_NEAR(non_zero_mass, first_order, 1e-9);

        ctx.record().expected["non_zero_mass"] = first_order;
        ctx.record().actual["non_zero_mass"] = non_zero_mass;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-004") {
        TestContext ctx(opts, tid, "joint_dp_upper_bound_and_xor_swap",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Upper sanitized bound p_e=0.49 convolution vs exact GF(2)^4 XOR swap P'(s)=P(s^g_e) at p_e=1.0";
        ctx.record().independent_oracle = "Exact FWHT convolution matches NoiseModel at p_e=0.49 (<1e-14) and pure XOR swap at p_e=1.0";

        double p_base[21];
        for (int e = 0; e < 21; ++e) p_base[e] = 0.015;
        double pmf_base[16], pmf_swap[16];
        ref_fwht_pmf(p_base, false, pmf_base);
        double p_one[21];
        std::copy(p_base, p_base + 21, p_one);
        p_one[4] = 1.0; /* Z_4 has signature 0x6 */
        /* Recompute without e=4 then swap by 0x6 */
        double p_no4[21];
        std::copy(p_base, p_base + 21, p_no4);
        p_no4[4] = 0.0;
        double pmf_no4[16];
        ref_fwht_pmf(p_no4, false, pmf_no4);
        ref_fwht_pmf(p_one, false, pmf_swap);
        for (int s = 0; s < 16; ++s) {
            ASSERT_NEAR(pmf_swap[s], pmf_no4[s ^ 0x6], 1e-14);
        }

        NoiseModel m;
        m.set_basis_z(false);
        double p_clamped[21];
        for (int e = 0; e < 21; ++e) {
            p_clamped[e] = (e == 4) ? 0.49 : 0.015;
            m.theta_(e) = p_clamped[e];
        }
        m.compute_joint_distribution();
        double pmf_ref_049[16];
        ref_fwht_pmf(p_clamped, false, pmf_ref_049);
        double max_diff = 0.0;
        for (int s = 0; s < 16; ++s) {
            max_diff = std::max(max_diff, std::abs(m.joint_probs()(s) - pmf_ref_049[s]));
        }
        ASSERT_LT(max_diff, 1e-14);
        ctx.record().expected["max_diff_vs_fwht_at_0_49"] = "< 1e-14";
        ctx.record().actual["max_diff"] = max_diff;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-005") {
        TestContext ctx(opts, tid, "joint_dp_single_mechanism_likelihood",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Single active mechanism e at p_e = 0.08 (others at 1e-6 floor) across all 21 mechanisms";
        ctx.record().independent_oracle = "P(g_e) == 0.08 +/- 2e-5 (floor perturbation) and exact match (<1e-14) against FWHT with 1e-6 floor";

        double max_exact_diff = 0.0;
        for (int target_e = 0; target_e < 21; ++target_e) {
            NoiseModel m;
            m.set_basis_z(false);
            double pref[21];
            for (int e = 0; e < 21; ++e) {
                pref[e] = (e == target_e) ? 0.08 : 1e-6;
                m.theta_(e) = pref[e];
            }
            m.compute_joint_distribution();
            double pmf_ref[16];
            ref_fwht_pmf(pref, false, pmf_ref);
            for (int s = 0; s < 16; ++s) {
                max_exact_diff = std::max(max_exact_diff, std::abs(m.joint_probs()(s) - pmf_ref[s]));
            }
            uint8_t ge = ref_event_sig(target_e, false);
            if (ge != 0u) {
                ASSERT_NEAR(m.joint_probs()(ge), 0.08, 3e-5);
                ASSERT_NEAR(m.joint_probs()(0), 0.92, 3e-5);
            } else {
                ASSERT_NEAR(m.joint_probs()(0), 1.0, 3e-5);
            }
        }
        ASSERT_LT(max_exact_diff, 1e-14);
        ctx.record().expected["max_exact_diff"] = "< 1e-14 across all 21 mechanisms";
        ctx.record().actual["max_exact_diff"] = max_exact_diff;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-006") {
        TestContext ctx(opts, tid, "joint_dp_multi_mechanism_bruteforce_256",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Exact 2^8 = 256 brute-force subset enumeration over k=8 active mechanisms across X and Z bases";
        ctx.record().independent_oracle = "Max absolute PMF error < 1e-12 vs brute-force subset enumeration";

        std::vector<int> active = {0, 1, 4, 7, 9, 12, 15, 18};
        const double active_probs[8] = {0.031, 0.045, 0.082, 0.027, 0.064, 0.051, 0.038, 0.073};
        double p_all[21];
        for (int e = 0; e < 21; ++e) p_all[e] = 1e-6;
        for (size_t i = 0; i < active.size(); ++i) p_all[active[i]] = active_probs[i];

        double max_err = 0.0;
        for (bool bz : {false, true}) {
            NoiseModel m;
            m.set_basis_z(bz);
            for (int e = 0; e < 21; ++e) m.theta_(e) = p_all[e];
            m.compute_joint_distribution();
            double bf_pmf[16];
            ref_bruteforce_subset_pmf(active, p_all, bz, bf_pmf);
            for (int s = 0; s < 16; ++s) {
                max_err = std::max(max_err, std::abs(m.joint_probs()(s) - bf_pmf[s]));
            }
        }
        ASSERT_LT(max_err, 1e-12);
        ctx.record().expected["max_bruteforce_err"] = "< 1e-12 across X and Z bases";
        ctx.record().actual["max_bruteforce_err"] = max_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-007") {
        TestContext ctx(opts, tid, "joint_dp_walsh_hadamard_identity",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::compute_joint_distribution", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Walsh-Hadamard identity E[(-1)^<u,s>] == prod_e (1 - 2 p_e)^{I[<u,g_e>=1]} across all u in 0..15";
        ctx.record().independent_oracle = "Max spectral discrepancy < 1e-13 across all 16 Dual characters u";

        NoiseModel m;
        m.set_basis_z(false);
        for (int e = 0; e < 21; ++e) {
            m.theta_(e) = 0.008 + 0.004 * static_cast<double>((e * 7 + 3) % 19);
        }
        m.compute_joint_distribution();

        double max_spec_err = 0.0;
        for (int u = 0; u < 16; ++u) {
            double emp_char = 0.0;
            for (int s = 0; s < 16; ++s) {
                int dot = __builtin_popcount(static_cast<unsigned>(u & s)) & 1;
                emp_char += (dot ? -1.0 : 1.0) * m.joint_probs()(s);
            }
            double analytic_char = 1.0;
            for (int e = 0; e < 21; ++e) {
                uint8_t ge = ref_event_sig(e, false);
                int dot = __builtin_popcount(static_cast<unsigned>(u & ge)) & 1;
                if (dot == 1) analytic_char *= (1.0 - 2.0 * m.theta_(e));
            }
            max_spec_err = std::max(max_spec_err, std::abs(emp_char - analytic_char));
        }
        ASSERT_LT(max_spec_err, 1e-13);
        ctx.record().expected["max_walsh_character_err"] = "< 1e-13";
        ctx.record().actual["max_walsh_character_err"] = max_spec_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-008") {
        TestContext ctx(opts, tid, "joint_parameter_signature_algebra",
                        "software/src/inference/noise_model.cpp",
                        "NM_EVENT_SIGNATURES, NM_EVENT_SIGNATURES_Z_BASIS, NM_CROSSTALK_PAIRS", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify all 21 Z-fault and 21 X-fault syndrome signatures match H_X*e_Z, H_Z*e_X, and g_{i,j} == g_i ^ g_j";
        ctx.record().independent_oracle = "42/42 bit-exact signature matches against rotated d=3 stabilizer matrices";

        int matches = 0;
        for (int k = 0; k < 12; ++k) {
            ASSERT_EQ(NM_CROSSTALK_PAIRS[k].i, REF_PAIRS[k][0]);
            ASSERT_EQ(NM_CROSSTALK_PAIRS[k].j, REF_PAIRS[k][1]);
        }
        for (bool bz : {false, true}) {
            NoiseModel m;
            m.set_basis_z(bz);
            for (int e = 0; e < 21; ++e) {
                uint8_t prod_sig = m.event_signature(e);
                uint8_t ref_sig  = ref_event_sig(e, bz);
                ASSERT_EQ(prod_sig, ref_sig);
                if (prod_sig == ref_sig) matches++;
            }
            for (int k = 0; k < 12; ++k) {
                uint8_t gi  = m.event_signature(NM_CROSSTALK_PAIRS[k].i);
                uint8_t gj  = m.event_signature(NM_CROSSTALK_PAIRS[k].j);
                uint8_t gij = m.event_signature(9 + k);
                ASSERT_EQ(gij, static_cast<uint8_t>(gi ^ gj));
            }
        }
        ASSERT_EQ(matches, 42);
        ctx.record().expected["signature_matches"] = 42;
        ctx.record().actual["signature_matches"] = matches;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-001", "L3-MATH-002", "L3-MATH-003", "L3-MATH-004",
        "L3-MATH-005", "L3-MATH-006", "L3-MATH-007", "L3-MATH-008"
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
