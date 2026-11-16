#include "test_assertions.h"
#include "test_common.h"
#include "estimator/llr_quantizer.h"
#include "control/causal_scheduler.h"
#include "control/r2d2_lstm_controller.h"

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-007");
    const std::string tid = opts.test_id;

    if (tid == "L3-FLOW-007") {
        TestContext ctx(opts, tid, "llr_converter_to_causal_scheduler_continuity",
                        "software/src/estimator/llr_quantizer.cpp, software/src/control/causal_scheduler.cpp",
                        "avid_quantize_and_validate_bank -> avid_scheduler_schedule_commit -> poll_commit_ready", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Pipe quantized Q7.8 bank into causal scheduler and verify exact 9-word preservation at pending and ready poll";
        ctx.record().independent_oracle = "pending_weights_q78[0..8] == out_weights_q78[0..8] == cand_q78[0..8]";

        const double p_eff[9] = {0.010, 0.014, 0.018, 0.022, 0.065, 0.030, 0.012, 0.016, 0.009};
        int16_t cand_q78[9] = {0};
        ASSERT_TRUE(avid_quantize_and_validate_bank(p_eff, cand_q78));

        avid_causal_scheduler_t sched;
        avid_scheduler_init(&sched, 16, 256, 24);
        avid_scheduler_schedule_commit(&sched, 256, cand_q78);
        ASSERT_TRUE(sched.has_pending_commit);
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(sched.pending_weights_q78[q], cand_q78[q]);
        }

        int16_t out_q78[9] = {0};
        ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, 279, out_q78));
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 280, out_q78));
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(out_q78[q], cand_q78[q]);
        }
        ASSERT_EQ(sched.active_epoch_id, 1u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-008") {
        TestContext ctx(opts, tid, "burst_detector_to_scheduler_and_r2d2_score_flow",
                        "software/src/control/causal_scheduler.cpp, software/src/control/r2d2_lstm_controller.h",
                        "avid_fast_detector_step -> AvidR2D2LSTMController::step_causal", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Drive 60-round burst through avid_fast_detector_step, verify <=20-round trigger and burst_score delivery to R2D2";
        ctx.record().independent_oracle = "100 <= trig_round <= 119, trigger_count == 1, peak_density >= 0.35";

        avid_fast_burst_detector_t det;
        avid_fast_detector_init(&det, 0.35, 4);
        int trig_round = -1;
        for (uint64_t r = 0; r < 180; ++r) {
            uint8_t sx = (r >= 100 && r < 160) ? 0x6u : 0x0u;
            if (avid_fast_detector_step(&det, r, sx, 0u) && trig_round < 0) {
                trig_round = static_cast<int>(r);
            }
        }
        ASSERT_TRUE(trig_round >= 103 && trig_round <= 119);
        ASSERT_EQ(det.trigger_count, 1u);
        ASSERT_TRUE(det.peak_density >= 0.35);

        AvidR2D2LSTMController r2d2;
        ASSERT_TRUE(r2d2.load_checkpoint(opts.repo_root + "/checkpoints/r2d2_v1_trained.bin"));
        uint32_t hx[16] = {40, 2, 2, 2, 2, 2, 48, 0, 2, 0, 0, 0, 0, 0, 0, 0};
        uint32_t hz[16] = {96, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double theta[21];
        for (int i = 0; i < 21; ++i) theta[i] = 0.015;
        theta[4] = 0.085;
        theta[12] = 0.090;
        double gains[9] = {0};
        uint32_t rec_k = 0;
        r2d2.step_causal(hx, hz, 100, theta, 0.95, gains, &rec_k, det.peak_density);
        ASSERT_TRUE(rec_k >= 3 && rec_k <= 5);

        ctx.record().expected["trigger_lag_max_rounds"] = "20";
        ctx.record().actual["trigger_lag_rounds"] = std::to_string(trig_round - 100 + 1);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
