#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include <cstdint>
#include <string>

using namespace avid::layer3_test;

namespace {

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const int16_t w_a[9] = {800, 810, 820, 830, 840, 850, 860, 870, 880};
    const int16_t w_b[9] = {600, 610, 620, 630, 640, 650, 660, 670, 680};

    if (tid == "L3-OP-021") {
        TestContext ctx(opts, tid, "burst_detector_boolean_indication",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Drive avid_fast_burst_detector_t with calm sx=0 then dense sx=0x2 and verify burst_active & step return";
        ctx.record().independent_oracle = "Calm: burst_active==false; dense: step returns true once and burst_active==true";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        for (uint64_t r = 0; r < 32; ++r) {
            ASSERT_FALSE(avid_fast_detector_step(&det, r, 0u, 0u));
            ASSERT_FALSE(det.burst_active);
        }
        bool fired = false;
        for (uint64_t r = 32; r < 52; ++r) {
            if (avid_fast_detector_step(&det, r, 0x2u, 0u)) fired = true;
        }
        ASSERT_TRUE(fired);
        ASSERT_TRUE(det.burst_active);
        ctx.record().expected["fired_and_active"] = "true, true";
        ctx.record().actual["fired_and_active"] = "true, true";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-022") {
        TestContext ctx(opts, tid, "burst_detector_trigger_timestamp_bounds",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Burst starts at round=100 (density=1.0, threshold=0.35, debounce=4) -> verify trigger round in [100, 119]";
        ctx.record().independent_oracle = "r_trig == 108 in [100, 119]";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        int64_t r_trig = -1;
        for (uint64_t r = 0; r < 140; ++r) {
            uint8_t sx = (r >= 100) ? 0x6u : 0x0u;
            if (avid_fast_detector_step(&det, r, sx, 0u) && r_trig < 0) {
                r_trig = static_cast<int64_t>(r);
            }
        }
        ASSERT_GE(r_trig, 100);
        ASSERT_LE(r_trig, 119);
        ASSERT_EQ(r_trig, 108);
        ctx.record().expected["r_trig"] = 108;
        ctx.record().actual["r_trig"] = r_trig;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-023") {
        TestContext ctx(opts, tid, "burst_detector_causal_trigger_latency",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Measure trigger latency r_trig - burst_start + 1 across burst densities {0.5, 0.75, 1.0}";
        ctx.record().independent_oracle = "Latency strictly in [4, 20] rounds (never <= 0 retrospective)";

        /* Pattern generators over 16-round cycles >= 0.35 */
        for (int mode = 0; mode < 3; ++mode) {
            avid_fast_burst_detector_t det{};
            avid_fast_detector_init(&det, 0.35, 4u);
            int64_t r_trig = -1;
            for (uint64_t r = 0; r < 150; ++r) {
                bool hit = false;
                if (r >= 100) {
                    if (mode == 0) hit = ((r % 2) == 0);        /* 0.50 */
                    else if (mode == 1) hit = ((r % 4) != 3);   /* 0.75 */
                    else hit = true;                            /* 1.00 */
                }
                if (avid_fast_detector_step(&det, r, hit ? 0x1u : 0x0u, 0u) && r_trig < 0) {
                    r_trig = static_cast<int64_t>(r);
                }
            }
            int64_t lat = r_trig - 100 + 1;
            ASSERT_GE(lat, 4);
            ASSERT_LE(lat, 20);
        }
        ctx.record().expected["latency_range"] = "[4, 20] rounds";
        ctx.record().actual["latency_range"] = "Verified across {0.50, 0.75, 1.00} burst densities";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-024") {
        TestContext ctx(opts, tid, "scheduler_delayed_commit_threshold",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_schedule_commit, avid_scheduler_poll_commit_ready", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Schedule commit at window_end=256 with commit_delay=32 -> false at 287, true at 288";
        ctx.record().independent_oracle = "poll(287)==false, poll(288)==true";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 32u);
        avid_scheduler_schedule_commit(&sched, 256ULL, w_a);
        int16_t out[9] = {0};
        ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, 287ULL, out));
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 288ULL, out));
        ASSERT_EQ(out[0], w_a[0]);
        ctx.record().expected["poll_287_288"] = "false, true";
        ctx.record().actual["poll_287_288"] = "false, true";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-025") {
        TestContext ctx(opts, tid, "scheduler_epoch_id_monotonic_assignment",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify active_epoch_id increments 0 -> 1 -> 2 across sequential committed weight banks";
        ctx.record().independent_oracle = "active_epoch_id == {0, 1, 2} with matching bank payloads";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 0u);
        ASSERT_EQ(sched.active_epoch_id, 0u);
        int16_t out[9] = {0};
        avid_scheduler_schedule_commit(&sched, 256ULL, w_a);
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 256ULL, out));
        ASSERT_EQ(sched.active_epoch_id, 1u);
        ASSERT_EQ(out[0], w_a[0]);

        avid_scheduler_schedule_commit(&sched, 512ULL, w_b);
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 512ULL, out));
        ASSERT_EQ(sched.active_epoch_id, 2u);
        ASSERT_EQ(out[0], w_b[0]);
        ctx.record().expected["epoch_progression"] = "0 -> 1 -> 2";
        ctx.record().actual["final_epoch"] = sched.active_epoch_id;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-026") {
        TestContext ctx(opts, tid, "scheduler_stale_delayed_evidence_guard",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_schedule_commit", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Schedule commit for window_end=500 then attempt out-of-order older window_end=300";
        ctx.record().independent_oracle = "Older evidence dropped (stale_schedule_drops==1); newer bank w_b preserved";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 0u);
        avid_scheduler_schedule_commit(&sched, 500ULL, w_b);
        avid_scheduler_schedule_commit(&sched, 300ULL, w_a);
        ASSERT_EQ(sched.stale_schedule_drops, 1ULL);
        int16_t out[9] = {0};
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 500ULL, out));
        ASSERT_EQ(out[0], w_b[0]);
        ctx.record().expected["stale_drops_and_winner"] = "1 drop, w_b (600) committed";
        ctx.record().actual["committed_w0"] = out[0];
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-027") {
        TestContext ctx(opts, tid, "burst_detector_single_trigger_debounce",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Feed sustained burst for 50 consecutive rounds and verify trigger_count == 1 (no duplicate trigger flood)";
        ctx.record().independent_oracle = "trigger_count == 1 across 50 sustained burst rounds";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        for (uint64_t r = 0; r < 70; ++r) {
            avid_fast_detector_step(&det, r, (r >= 20) ? 0xFu : 0x0u, 0u);
        }
        ASSERT_EQ(det.trigger_count, 1u);
        ctx.record().expected["trigger_count"] = 1;
        ctx.record().actual["trigger_count"] = det.trigger_count;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-028") {
        TestContext ctx(opts, tid, "burst_detector_subthreshold_zero_triggers",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Feed 10,000 sub-threshold rounds (5/16 = 0.3125 < 0.35) and verify exact 0 triggers";
        ctx.record().independent_oracle = "trigger_count == 0 and peak_density == 0.3125";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        for (uint64_t r = 0; r < 10000; ++r) {
            bool hit = ((r % 16ULL) >= 11ULL); /* 5 out of every 16 == 0.3125 (post-warmup) */
            ASSERT_FALSE(avid_fast_detector_step(&det, r, hit ? 0x1u : 0x0u, 0u));
        }
        ASSERT_EQ(det.trigger_count, 0u);
        ASSERT_NEAR(det.peak_density, 0.3125, 1e-12);
        ctx.record().expected["trigger_count"] = 0;
        ctx.record().actual["peak_density"] = det.peak_density;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-OP-021", "L3-OP-022", "L3-OP-023", "L3-OP-024",
        "L3-OP-025", "L3-OP-026", "L3-OP-027", "L3-OP-028"
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
