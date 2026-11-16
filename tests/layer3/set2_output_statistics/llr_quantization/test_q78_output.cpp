#include "test_assertions.h"
#include "test_common.h"
#include "estimator/llr_quantizer.h"
#include "inference/noise_model.h"
#include "driver/weight_reg_driver.h"
#include "control/fallback_fsm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using namespace avid::layer3_test;

namespace {

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;

    if (tid == "L3-OP-011") {
        TestContext ctx(opts, tid, "llr_bank_dimension_nine",
                        "software/src/estimator/llr_quantizer.cpp, software/src/inference/noise_model.cpp",
                        "avid_quantize_and_validate_bank, NoiseModel::to_q7_8_llr", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify exact int16_t[9] output bank population without out-of-bounds canary writes";
        ctx.record().independent_oracle = "All 9 slots populated; guard canaries before/after array untouched";

        double p_eff[9] = {0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05};
        int16_t padded_a[11];
        int16_t padded_b[11];
        std::fill(padded_a, padded_a + 11, static_cast<int16_t>(-12345));
        std::fill(padded_b, padded_b + 11, static_cast<int16_t>(-12345));

        ASSERT_TRUE(avid_quantize_and_validate_bank(p_eff, &padded_a[1]));
        NoiseModel m;
        m.init(0.012, 0.008);
        m.to_q7_8_llr(&padded_b[1]);

        ASSERT_EQ(padded_a[0], static_cast<int16_t>(-12345));
        ASSERT_EQ(padded_a[10], static_cast<int16_t>(-12345));
        ASSERT_EQ(padded_b[0], static_cast<int16_t>(-12345));
        ASSERT_EQ(padded_b[10], static_cast<int16_t>(-12345));
        for (int q = 1; q <= 9; ++q) {
            ASSERT_GT(padded_a[q], static_cast<int16_t>(0));
            ASSERT_GT(padded_b[q], static_cast<int16_t>(0));
        }
        ctx.record().expected["bank_len_and_canaries"] = "9 elements written, 0 canary overwrites";
        ctx.record().actual["bank_len_and_canaries"] = "Verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-012") {
        TestContext ctx(opts, tid, "llr_q78_golden_reference_table",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Test golden reference probabilities {0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30} against lround(ln((1-p)/p)*256)";
        ctx.record().independent_oracle = "0 LSB difference across all 7 golden points";

        const double probs[7] = {0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30};
        for (double p : probs) {
            int16_t exp_q = static_cast<int16_t>(std::lround(std::log((1.0 - p) / p) * 256.0));
            int16_t obs_q = avid_prob_to_q78_llr(p);
            ASSERT_EQ(obs_q, exp_q);
        }
        ctx.record().expected["golden_lsb_diff"] = 0;
        ctx.record().actual["golden_lsb_diff"] = 0;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-013") {
        TestContext ctx(opts, tid, "llr_q78_rounding_dense_grid",
                        "software/src/estimator/llr_quantizer.cpp, software/src/driver/weight_reg_driver.h",
                        "avid_prob_to_q78_llr vs wrd_float_to_q78", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Compare nearest-integer rounding in avid_prob_to_q78_llr vs truncation in wrd_float_to_q78 over 50 dense LLR points";
        ctx.record().independent_oracle = "0 <= (q_round - q_trunc) <= 1 for all positive LLRs";

        int max_gap = 0;
        for (int i = 0; i < 50; ++i) {
            double llr = 0.5 + 0.15 * static_cast<double>(i);
            double p = 1.0 / (1.0 + std::exp(llr));
            int q_r = avid_prob_to_q78_llr(p);
            int q_t = wrd_float_to_q78(static_cast<float>(llr));
            int diff = q_r - q_t;
            ASSERT_GE(diff, 0);
            ASSERT_LE(diff, 1);
            max_gap = std::max(max_gap, diff);
        }
        ASSERT_EQ(max_gap, 1);
        ctx.record().expected["round_minus_trunc"] = "in {0, 1}";
        ctx.record().actual["max_gap"] = max_gap;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-014") {
        TestContext ctx(opts, tid, "llr_q78_safe_bank_saturation",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_quantize_and_validate_bank", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Test p=1e-5 (0x0B83 in-bounds) vs p=0.45 (0x0033 < 0x0040 -> clamps to AVID_Q78_MIN_SAFE and returns false)";
        ctx.record().independent_oracle = "p=1e-5 safe; p=0.45 clamped to 0x0040 with return false";

        double ok_p[9] = {1e-5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
        int16_t ok_q[9] = {0};
        ASSERT_TRUE(avid_quantize_and_validate_bank(ok_p, ok_q));
        ASSERT_EQ(ok_q[0], static_cast<int16_t>(2947)); /* 0x0B83 */

        double hi_p[9] = {0.45, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
        int16_t hi_q[9] = {0};
        ASSERT_FALSE(avid_quantize_and_validate_bank(hi_p, hi_q));
        ASSERT_EQ(hi_q[0], AVID_Q78_MIN_SAFE);
        ctx.record().expected["q78_1e5_and_045"] = "2947 (safe), 64 (clamped unsafe)";
        ctx.record().actual["q78_1e5_and_045"] = std::to_string(ok_q[0]) + ", " + std::to_string(hi_q[0]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-015") {
        TestContext ctx(opts, tid, "llr_zero_probability_handling",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr, avid_quantize_and_validate_bank", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Pass p=0.0 into scalar and bank quantizer and verify finite safe fallback";
        ctx.record().independent_oracle = "Scalar clamps to 1e-5 (2947); bank returns false and writes AVID_Q78_DEFAULT";

        ASSERT_EQ(avid_prob_to_q78_llr(0.0), static_cast<int16_t>(2947));
        double p_zero[9] = {0.0, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012};
        int16_t out[9] = {0};
        ASSERT_FALSE(avid_quantize_and_validate_bank(p_zero, out));
        ASSERT_EQ(out[0], AVID_Q78_DEFAULT);
        ctx.record().expected["zero_prob_bank_slot"] = AVID_Q78_DEFAULT;
        ctx.record().actual["zero_prob_bank_slot"] = out[0];
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-016") {
        TestContext ctx(opts, tid, "llr_unit_and_half_probability_handling",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr, avid_quantize_and_validate_bank", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Pass p=0.5 and p=1.0 into scalar and bank quantizer";
        ctx.record().independent_oracle = "Scalar clamps to 0.49 (10); bank rejects >=0.5 returning false and AVID_Q78_DEFAULT";

        ASSERT_EQ(avid_prob_to_q78_llr(0.5), avid_prob_to_q78_llr(0.49));
        ASSERT_EQ(avid_prob_to_q78_llr(1.0), avid_prob_to_q78_llr(0.49));
        double p_bad[9] = {0.5, 1.0, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012};
        int16_t out[9] = {0};
        ASSERT_FALSE(avid_quantize_and_validate_bank(p_bad, out));
        ASSERT_EQ(out[0], AVID_Q78_DEFAULT);
        ASSERT_EQ(out[1], AVID_Q78_DEFAULT);
        ctx.record().expected["unit_and_half_bank"] = "false, AVID_Q78_DEFAULT";
        ctx.record().actual["out_0_1"] = std::to_string(out[0]) + ", " + std::to_string(out[1]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-017") {
        TestContext ctx(opts, tid, "llr_positive_sign_monotonicity",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify q78(p) > 0 and non-increasing across 200 samples on (0, 0.49)";
        ctx.record().independent_oracle = "0 sign or monotonicity inversions";

        int16_t prev = 32767;
        for (int i = 0; i < 200; ++i) {
            double p = 1e-5 + (0.489 - 1e-5) * (static_cast<double>(i) / 199.0);
            int16_t q = avid_prob_to_q78_llr(p);
            ASSERT_GT(q, static_cast<int16_t>(0));
            ASSERT_LE(q, prev);
            prev = q;
        }
        ctx.record().expected["inversions"] = 0;
        ctx.record().actual["min_positive_q78"] = prev;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-018") {
        TestContext ctx(opts, tid, "llr_roundtrip_conversion_accuracy",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify round-trip LLR error |q78/256.0 - LLR| <= 1/512 + eps over [1e-4, 0.42]";
        ctx.record().independent_oracle = "Max LLR round-trip error <= 1/512 + 1e-12";

        double max_err = 0.0;
        for (int i = 0; i < 100; ++i) {
            double p = 1e-4 + (0.42 - 1e-4) * (static_cast<double>(i) / 99.0);
            double llr = std::log((1.0 - p) / p);
            int16_t q = avid_prob_to_q78_llr(p);
            double err = std::abs(static_cast<double>(q) / 256.0 - llr);
            max_err = std::max(max_err, err);
            ASSERT_LE(err, (1.0 / 512.0) + 1e-12);
        }
        ctx.record().expected["max_roundtrip_llr_err"] = "<= 1/512";
        ctx.record().actual["max_roundtrip_llr_err"] = max_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-019") {
        TestContext ctx(opts, tid, "llr_spatial_hotspot_parameter_mapping",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::to_q7_8_llr, NoiseModel::effective_p", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Elevate single qubit q=4 (theta[4]=0.12, pairs=1e-5) and verify ONLY q=4 LLR drops relative to non-adjacent qubits";
        ctx.record().independent_oracle = "q78[4] < q78[0] - 500 and all non-hot qubits equal";

        NoiseModel m;
        m.init(0.012, 1e-5);
        for (int k = 0; k < 12; ++k) m.theta_(9 + k) = 1e-5;
        m.theta_(4) = 0.120;
        m.compute_joint_distribution();
        int16_t q78[9] = {0};
        m.to_q7_8_llr(q78);
        ASSERT_LT(static_cast<int>(q78[4]), static_cast<int>(q78[0]) - 500);
        for (int q = 0; q < 9; ++q) {
            if (q != 4) ASSERT_EQ(q78[q], q78[0]);
        }
        ctx.record().expected["spatial_isolation"] = "q78[4] sharply reduced; q != 4 unchanged";
        ctx.record().actual["q78_4_vs_0"] = std::to_string(q78[4]) + " vs " + std::to_string(q78[0]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-020") {
        TestContext ctx(opts, tid, "llr_invalid_estimate_blocks_fsm_commit",
                        "software/src/estimator/llr_quantizer.cpp, software/src/control/fallback_fsm.cpp",
                        "avid_quantize_and_validate_bank -> avid_fallback_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Inject NaN, -0.1, and 0.6 into p_eff and verify bounds_ok==false routes FSM to SAFE_STATIC_PRIOR";
        ctx.record().independent_oracle = "bounds_ok==false and FSM selects static_prior_q78 (never corrupted candidate)";

        const double bad_vals[3] = {std::numeric_limits<double>::quiet_NaN(), -0.1, 0.6};
        for (double bv : bad_vals) {
            double p_eff[9] = {0.012, 0.012, 0.012, bv, 0.012, 0.012, 0.012, 0.012, 0.012};
            int16_t cand[9] = {0}, sel[9] = {0};
            bool bounds_ok = avid_quantize_and_validate_bank(p_eff, cand);
            ASSERT_FALSE(bounds_ok);
            cand[3] = -1234; /* tamper candidate to prove FSM ignores it */
            avid_fallback_fsm_t fsm{};
            avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
            bool should_commit = false;
            auto st = avid_fallback_step(&fsm, true, true, bounds_ok, true, cand, sel, &should_commit);
            ASSERT_EQ(st, AVID_FSM_SAFE_STATIC_PRIOR);
            ASSERT_EQ(sel[3], AVID_Q78_DEFAULT);
        }
        ctx.record().expected["fsm_state_on_invalid_peff"] = "AVID_FSM_SAFE_STATIC_PRIOR";
        ctx.record().actual["fsm_state_on_invalid_peff"] = "AVID_FSM_SAFE_STATIC_PRIOR";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-OP-011", "L3-OP-012", "L3-OP-013", "L3-OP-014", "L3-OP-015",
        "L3-OP-016", "L3-OP-017", "L3-OP-018", "L3-OP-019", "L3-OP-020"
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
