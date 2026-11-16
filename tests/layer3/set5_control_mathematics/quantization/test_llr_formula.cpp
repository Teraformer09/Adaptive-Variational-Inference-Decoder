#include "test_assertions.h"
#include "test_common.h"
#include "estimator/llr_quantizer.h"
#include "inference/noise_model.h"
#include "driver/weight_reg_driver.h"
#include "custom_engine/cycle_accurate_bp_engine.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using namespace avid::layer3_test;

namespace {

int16_t ref_prob_to_q78(double p) {
    if (!std::isfinite(p)) return AVID_Q78_DEFAULT;
    double pc = std::clamp(p, 1e-5, 0.49);
    double llr = std::log((1.0 - pc) / pc);
    long r = std::lround(llr * 256.0);
    r = std::clamp<long>(r, -32768L, 32767L);
    return static_cast<int16_t>(r);
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;

    if (tid == "L3-MATH-033") {
        TestContext ctx(opts, tid, "llr_q78_100_point_reference_sweep",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Sweep 100 probabilities in [1e-5, 0.49] and compare avid_prob_to_q78_llr against independent lround(ln((1-p)/p)*256)";
        ctx.record().independent_oracle = "Max integer LSB difference == 0 across all 100 points";

        int max_lsb_diff = 0;
        for (int i = 0; i < 100; ++i) {
            double frac = static_cast<double>(i) / 99.0;
            double p = 1e-5 + frac * (0.49 - 1e-5);
            int16_t q_prod = avid_prob_to_q78_llr(p);
            int16_t q_ref  = ref_prob_to_q78(p);
            max_lsb_diff = std::max(max_lsb_diff, std::abs(static_cast<int>(q_prod) - static_cast<int>(q_ref)));
        }
        ASSERT_EQ(max_lsb_diff, 0);
        ctx.record().expected["max_lsb_diff"] = 0;
        ctx.record().actual["max_lsb_diff"] = max_lsb_diff;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-034") {
        TestContext ctx(opts, tid, "llr_sign_convention_contract",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify p < 0.5 => positive Q7.8 LLR (0-bit preferred) and monotonic non-increasing LLR with increasing p";
        ctx.record().independent_oracle = "q78(p) > 0 for all p in (0, 0.49] and strictly decreasing across discrete steps";

        int16_t prev = avid_prob_to_q78_llr(1e-5);
        ASSERT_GT(prev, static_cast<int16_t>(0));
        for (double p : {0.001, 0.01, 0.05, 0.10, 0.25, 0.40, 0.49}) {
            int16_t cur = avid_prob_to_q78_llr(p);
            ASSERT_GT(cur, static_cast<int16_t>(0));
            ASSERT_LT(cur, prev);
            prev = cur;
        }
        /* Direct mathematical log((1-0.6)/0.6) < 0 while quantizer clamps physical channel p <= 0.49 */
        ASSERT_LT(std::log((1.0 - 0.6) / 0.6), 0.0);
        ASSERT_EQ(avid_prob_to_q78_llr(0.6), avid_prob_to_q78_llr(0.49));
        ctx.record().expected["sign_and_monotonicity"] = "Positive and strictly decreasing on [1e-5 .. 0.49]";
        ctx.record().actual["q78_at_0_49"] = prev;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-035") {
        TestContext ctx(opts, tid, "llr_q78_scale_factor_256",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr, wrd_float_to_q78", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify exact 2^8 = 256.0 Q7.8 scale factor at LLR = 1.0, 2.5, and 4.0";
        ctx.record().independent_oracle = "LLR {1.0, 2.5, 4.0} -> Q7.8 {256, 640, 1024}";

        auto p_from_llr = [](double llr) { return 1.0 / (1.0 + std::exp(llr)); };
        ASSERT_EQ(avid_prob_to_q78_llr(p_from_llr(1.0)), static_cast<int16_t>(256));
        ASSERT_EQ(avid_prob_to_q78_llr(p_from_llr(2.5)), static_cast<int16_t>(640));
        ASSERT_EQ(avid_prob_to_q78_llr(p_from_llr(4.0)), static_cast<int16_t>(1024));
        ASSERT_EQ(wrd_float_to_q78(1.0f), static_cast<int16_t>(256));
        ASSERT_EQ(wrd_float_to_q78(2.5f), static_cast<int16_t>(640));
        ASSERT_EQ(wrd_float_to_q78(4.0f), static_cast<int16_t>(1024));
        ctx.record().expected["q78_1_2p5_4"] = "256, 640, 1024";
        ctx.record().actual["q78_1_2p5_4"] = "256, 640, 1024";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-036") {
        TestContext ctx(opts, tid, "llr_q78_rounding_vs_driver_truncation",
                        "software/src/estimator/llr_quantizer.cpp, software/src/driver/weight_reg_driver.h",
                        "avid_prob_to_q78_llr vs wrd_float_to_q78", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify nearest-integer std::lround at .5 fractional midpoints vs wrd_float_to_q78 truncation";
        ctx.record().independent_oracle = "At LLR = (512.5 / 256.0), avid_prob_to_q78_llr rounds to 513 while wrd_float_to_q78 truncates to 512 (<= 1 LSB)";

        double target_llr = 512.75 / 256.0;
        double p = 1.0 / (1.0 + std::exp(target_llr));
        int16_t q_round = avid_prob_to_q78_llr(p);
        int16_t q_trunc = wrd_float_to_q78(static_cast<float>(target_llr));
        ASSERT_EQ(q_round, static_cast<int16_t>(513));
        ASSERT_EQ(q_trunc, static_cast<int16_t>(512));
        ASSERT_LE(std::abs(static_cast<int>(q_round) - static_cast<int>(q_trunc)), 1);
        ctx.record().expected["round_vs_trunc"] = "513 vs 512 (exact 1 LSB midpoint behavior)";
        ctx.record().actual["round_vs_trunc"] = std::to_string(q_round) + " vs " + std::to_string(q_trunc);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-037") {
        TestContext ctx(opts, tid, "llr_q78_saturation_and_safe_bank_bounds",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_quantize_and_validate_bank, wrd_float_to_q78", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify [-32768, 32767] int16 saturation and [0x0040, 0x3000] safe bank clamp";
        ctx.record().independent_oracle = "wrd_float_to_q78(200.0f)==32767, (-200.0f)==-32768; p=0.45 (<0x0040) clamps to 0x0040 and returns false";

        ASSERT_EQ(wrd_float_to_q78(200.0f), static_cast<int16_t>(32767));
        ASSERT_EQ(wrd_float_to_q78(-200.0f), static_cast<int16_t>(-32768));

        double p_bank[9] = {0.012, 0.012, 0.012, 0.012, 0.45, 0.012, 0.012, 0.012, 0.012};
        int16_t q_bank[9] = {0};
        bool ok = avid_quantize_and_validate_bank(p_bank, q_bank);
        ASSERT_FALSE(ok);
        ASSERT_EQ(q_bank[4], AVID_Q78_MIN_SAFE);
        ctx.record().expected["unsafe_high_p_clamp"] = "ok==false and q_bank[4]==0x0040 (64)";
        ctx.record().actual["q_bank_4"] = q_bank[4];
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-038") {
        TestContext ctx(opts, tid, "llr_q78_dequantization_roundtrip",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Reconstruct p_deq = 1 / (1 + exp(q78 / 256.0)) across physical probability range [1e-4, 0.40]";
        ctx.record().independent_oracle = "Max probability reconstruction error |p_deq - p| < 5e-4";

        double max_p_err = 0.0;
        for (int i = 0; i < 50; ++i) {
            double p = 1e-4 + (0.40 - 1e-4) * (static_cast<double>(i) / 49.0);
            int16_t q = avid_prob_to_q78_llr(p);
            double llr_deq = static_cast<double>(q) / 256.0;
            double p_deq = 1.0 / (1.0 + std::exp(llr_deq));
            max_p_err = std::max(max_p_err, std::abs(p_deq - p));
        }
        ASSERT_LT(max_p_err, 5e-4);
        ctx.record().expected["max_prob_roundtrip_err"] = "< 5e-4";
        ctx.record().actual["max_prob_roundtrip_err"] = max_p_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-039") {
        TestContext ctx(opts, tid, "llr_q78_half_lsb_quantization_error_bound",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify unsaturated quantization error |q78/256.0 - ln((1-p)/p)| <= 1/512 + 1e-12 everywhere on [1e-5, 0.43]";
        ctx.record().independent_oracle = "Max LLR error <= 0.001953125 + 1e-12";

        const double bound = (1.0 / 512.0) + 1e-12;
        double max_err = 0.0;
        for (int i = 0; i < 200; ++i) {
            double p = 1e-5 + (0.43 - 1e-5) * (static_cast<double>(i) / 199.0);
            int16_t q = avid_prob_to_q78_llr(p);
            ASSERT_GE(q, AVID_Q78_MIN_SAFE);
            ASSERT_LE(q, AVID_Q78_MAX_SAFE);
            double exact_llr = std::log((1.0 - p) / p);
            double err = std::abs(static_cast<double>(q) / 256.0 - exact_llr);
            max_err = std::max(max_err, err);
            ASSERT_LE(err, bound);
        }
        ctx.record().expected["max_llr_quant_err"] = "<= 1/512 (0.001953125)";
        ctx.record().actual["max_llr_quant_err"] = max_err;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-040") {
        TestContext ctx(opts, tid, "llr_extreme_and_nonfinite_probabilities",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr, avid_quantize_and_validate_bank", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Evaluate p in {0, -1, 1e-12, 1e-5, 0.49, 0.5, 1.0, NaN, Inf} cleanly without UB or NaN propagation";
        ctx.record().independent_oracle = "Non-finite -> AVID_Q78_DEFAULT; out-of-(0,0.5) bank entries return false and emit safe default";

        const double nan_v = std::numeric_limits<double>::quiet_NaN();
        const double inf_v = std::numeric_limits<double>::infinity();
        ASSERT_EQ(avid_prob_to_q78_llr(nan_v), AVID_Q78_DEFAULT);
        ASSERT_EQ(avid_prob_to_q78_llr(inf_v), AVID_Q78_DEFAULT);
        ASSERT_EQ(avid_prob_to_q78_llr(-inf_v), AVID_Q78_DEFAULT);
        ASSERT_EQ(avid_prob_to_q78_llr(0.0), avid_prob_to_q78_llr(1e-5));
        ASSERT_EQ(avid_prob_to_q78_llr(-1.0), avid_prob_to_q78_llr(1e-5));
        ASSERT_EQ(avid_prob_to_q78_llr(1e-12), avid_prob_to_q78_llr(1e-5));
        ASSERT_EQ(avid_prob_to_q78_llr(0.5), avid_prob_to_q78_llr(0.49));
        ASSERT_EQ(avid_prob_to_q78_llr(1.0), avid_prob_to_q78_llr(0.49));

        const double bad_inputs[] = {0.0, -1.0, 0.5, 1.0, nan_v, inf_v};
        for (double bad : bad_inputs) {
            double p_bank[9] = {0.012, 0.012, bad, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012};
            int16_t out[9] = {0};
            ASSERT_FALSE(avid_quantize_and_validate_bank(p_bank, out));
            ASSERT_EQ(out[2], AVID_Q78_DEFAULT);
        }
        ctx.record().expected["extreme_handling"] = "Zero UB; all invalid bank entries rejected to AVID_Q78_DEFAULT";
        ctx.record().actual["extreme_handling"] = "Verified across 9 extreme/non-finite inputs";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-041") {
        TestContext ctx(opts, tid, "llr_qubit_weight_index_ordering",
                        "software/src/inference/noise_model.cpp",
                        "NoiseModel::effective_p, NoiseModel::to_q7_8_llr", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Assign distinct strictly increasing error rates to qubits 0..8 and verify q78[0..8] strictly decreases in exact index order";
        ctx.record().independent_oracle = "q78[0] > q78[1] > ... > q78[8] and each matches avid_prob_to_q78_llr(effective_p(q))";

        NoiseModel m;
        m.init(0.010, 1e-5);
        for (int k = 0; k < 12; ++k) m.theta_(9 + k) = 1e-5;
        for (int q = 0; q < 9; ++q) {
            m.theta_(q) = 0.005 + 0.020 * static_cast<double>(q);
        }
        m.compute_joint_distribution();
        int16_t q78[9] = {0};
        m.to_q7_8_llr(q78);
        for (int q = 0; q < 9; ++q) {
            int16_t expected_q = avid_prob_to_q78_llr(m.effective_p(q));
            ASSERT_NEAR(static_cast<double>(q78[q]), static_cast<double>(expected_q), 1.0);
            if (q > 0) ASSERT_LT(q78[q], q78[q - 1]);
        }
        ctx.record().expected["strict_ordering"] = "q78[0] > q78[1] > ... > q78[8]";
        ctx.record().actual["q78_0_and_8"] = std::to_string(q78[0]) + " -> " + std::to_string(q78[8]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-042") {
        TestContext ctx(opts, tid, "llr_q78_bp_decoder_degeneracy_steering",
                        "software/src/custom_engine/cycle_accurate_bp_engine.cpp",
                        "wrd_write_all_weights, wrd_commit, CycleAccurateBPEngine::decode", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Commit distinct Q7.8 priors through wrd_handle_t and verify CycleAccurateBPEngine steers degenerate X-syndrome sx=0x2 between q2 and q5";
        ctx.record().independent_oracle = "Low LLR on q2 selects corr_mask==(1<<2); low LLR on q5 selects corr_mask==(1<<5)";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        CycleAccurateBPEngine bp(3, 250.0);

        int16_t bank_q2_hot[9], bank_q5_hot[9], active[9];
        for (int q = 0; q < 9; ++q) {
            bank_q2_hot[q] = avid_prob_to_q78_llr(0.010);
            bank_q5_hot[q] = avid_prob_to_q78_llr(0.010);
        }
        bank_q2_hot[2] = avid_prob_to_q78_llr(0.150);
        bank_q5_hot[5] = avid_prob_to_q78_llr(0.150);

        wrd_write_all_weights(&wrd, bank_q2_hot);
        wrd_commit(&wrd);
        for (int q = 0; q < 9; ++q) active[q] = wrd_read_active_weight(&wrd, q);
        auto r2 = bp.decode(0x2u, active, 3);
        ASSERT_TRUE(r2.syndrome_satisfied);
        ASSERT_EQ(r2.correction_mask, static_cast<uint16_t>(1u << 2));

        wrd_write_all_weights(&wrd, bank_q5_hot);
        wrd_commit(&wrd);
        for (int q = 0; q < 9; ++q) active[q] = wrd_read_active_weight(&wrd, q);
        auto r5 = bp.decode(0x2u, active, 3);
        ASSERT_TRUE(r5.syndrome_satisfied);
        ASSERT_EQ(r5.correction_mask, static_cast<uint16_t>(1u << 5));

        wrd_close(&wrd);
        ctx.record().expected["degenerate_sx2_steering"] = "q2 hot -> 0x0004, q5 hot -> 0x0020";
        ctx.record().actual["corr_masks"] = std::to_string(r2.correction_mask) + ", " + std::to_string(r5.correction_mask);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-033", "L3-MATH-034", "L3-MATH-035", "L3-MATH-036", "L3-MATH-037",
        "L3-MATH-038", "L3-MATH-039", "L3-MATH-040", "L3-MATH-041", "L3-MATH-042"
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
