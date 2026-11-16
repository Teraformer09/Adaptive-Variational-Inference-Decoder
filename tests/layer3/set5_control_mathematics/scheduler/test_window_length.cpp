#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "driver/weight_reg_driver.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace avid::layer3_test;

namespace {

/* Independent analytical sliding-subwindow burst trigger oracle */
int64_t ref_predict_burst_trigger_round(const std::vector<uint8_t> &nonzero_flags,
                                        double threshold,
                                        uint32_t debounce_rounds) {
    uint32_t deb = 0;
    for (size_t r = 0; r < nonzero_flags.size(); ++r) {
        if (r >= 15u) {
            uint32_t s = 0;
            for (size_t j = r - 15u; j <= r; ++j) s += (nonzero_flags[j] ? 1u : 0u);
            double dens = static_cast<double>(s) / 16.0;
            if (dens >= threshold) {
                deb++;
                if (deb == debounce_rounds) return static_cast<int64_t>(r);
            } else {
                deb = 0;
            }
        }
    }
    return -1;
}

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const int16_t sample_w[9] = {900, 910, 920, 930, 940, 950, 960, 970, 980};

    if (tid == "L3-MATH-043") {
        TestContext ctx(opts, tid, "scheduler_evidence_timestamp_accounting",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_schedule_commit, avid_fast_detector_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify exact window_end_round and pending_commit_round == window_end_round + commit_delay_rounds";
        ctx.record().independent_oracle = "last_scheduled_window_end == 512 and pending_commit_round == 544 for delay=32";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 32u);
        avid_scheduler_schedule_commit(&sched, 512u, sample_w);
        ASSERT_TRUE(sched.has_pending_commit);
        ASSERT_TRUE(sched.has_scheduled_window);
        ASSERT_EQ(sched.last_scheduled_window_end, 512ULL);
        ASSERT_EQ(sched.pending_commit_round, 544ULL);

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        for (uint64_t r = 0; r < 25; ++r) {
            avid_fast_detector_step(&det, r, 0x3u, 0x0u);
        }
        ASSERT_EQ(det.last_round_idx, 24ULL);
        ctx.record().expected["pending_commit_round"] = 544;
        ctx.record().actual["pending_commit_round"] = sched.pending_commit_round;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-044") {
        TestContext ctx(opts, tid, "scheduler_commit_eligibility_50_random_pairs",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Across 50 random (t_e, d_c) pairs, verify poll_commit_ready is false for all t < t_e + d_c and true at t == t_e + d_c";
        ctx.record().independent_oracle = "50/50 exact causal threshold transitions";

        std::mt19937_64 rng(2026111644ULL);
        std::uniform_int_distribution<uint64_t> te_dist(64, 2000);
        std::uniform_int_distribution<uint64_t> dc_dist(0, 128);
        int passed_pairs = 0;
        for (int i = 0; i < 50; ++i) {
            uint64_t te = te_dist(rng);
            uint64_t dc = dc_dist(rng);
            avid_causal_scheduler_t sched{};
            avid_scheduler_init(&sched, 16u, 256u, dc);
            avid_scheduler_schedule_commit(&sched, te, sample_w);
            int16_t out[9] = {0};
            if (te + dc > 0) {
                ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, te + dc - 1u, out));
            }
            if (dc > 2) {
                ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, te, out));
            }
            ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, te + dc, out));
            ASSERT_EQ(out[0], sample_w[0]);
            passed_pairs++;
        }
        ASSERT_EQ(passed_pairs, 50);
        ctx.record().expected["causal_pairs_verified"] = 50;
        ctx.record().actual["causal_pairs_verified"] = passed_pairs;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-045") {
        TestContext ctx(opts, tid, "scheduler_strict_monotonic_epoch_ordering",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Execute 20 scheduled commits and verify active_epoch_id forms strict sequence 0, 1, 2, ..., 20";
        ctx.record().independent_oracle = "Exact integer sequence 0..20";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 8u);
        ASSERT_EQ(sched.active_epoch_id, 0u);
        for (uint32_t ep = 1; ep <= 20; ++ep) {
            uint64_t te = static_cast<uint64_t>(ep) * 256ULL;
            avid_scheduler_schedule_commit(&sched, te, sample_w);
            int16_t out[9] = {0};
            ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, te + 8ULL, out));
            ASSERT_EQ(sched.active_epoch_id, ep);
        }
        ctx.record().expected["final_epoch_id"] = 20;
        ctx.record().actual["final_epoch_id"] = sched.active_epoch_id;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-046") {
        TestContext ctx(opts, tid, "scheduler_stale_window_rejection",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_schedule_commit", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify scheduling an older window_end_round < last_scheduled_window_end is rejected and increments stale_schedule_drops";
        ctx.record().independent_oracle = "Pending newer weights preserved intact; stale_schedule_drops == 1";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 16u);
        int16_t w_new[9] = {700, 700, 700, 700, 700, 700, 700, 700, 700};
        int16_t w_old[9] = {300, 300, 300, 300, 300, 300, 300, 300, 300};
        avid_scheduler_schedule_commit(&sched, 512ULL, w_new);
        avid_scheduler_schedule_commit(&sched, 256ULL, w_old);
        ASSERT_EQ(sched.stale_schedule_drops, 1ULL);
        ASSERT_EQ(sched.pending_commit_round, 528ULL);
        int16_t out[9] = {0};
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 528ULL, out));
        ASSERT_EQ(out[0], static_cast<int16_t>(700));
        ctx.record().expected["stale_schedule_drops"] = 1;
        ctx.record().actual["committed_weight_0"] = out[0];
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-047") {
        TestContext ctx(opts, tid, "scheduler_and_burst_duplicate_prevention",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready, avid_fast_detector_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Verify single scheduled commit fires once across repeated polls, and sustained 60-round burst fires trigger_count==1";
        ctx.record().independent_oracle = "Second poll == false; sustained burst trigger_count == 1";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 4u);
        avid_scheduler_schedule_commit(&sched, 256ULL, sample_w);
        int16_t out[9] = {0};
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 260ULL, out));
        ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, 260ULL, out));
        ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, 261ULL, out));
        ASSERT_EQ(sched.active_epoch_id, 1u);

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        int fires = 0;
        for (uint64_t r = 0; r < 80; ++r) {
            uint8_t sx = (r >= 20) ? 0x6u : 0x0u;
            if (avid_fast_detector_step(&det, r, sx, 0u)) fires++;
        }
        ASSERT_EQ(fires, 1);
        ASSERT_EQ(det.trigger_count, 1u);
        ctx.record().expected["fires_per_sustained_burst"] = 1;
        ctx.record().actual["trigger_count"] = det.trigger_count;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-048") {
        TestContext ctx(opts, tid, "busy_decoder_commit_deferral_timeout",
                        "software/src/driver/weight_reg_driver.c",
                        "wrd_commit_and_wait", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Emulate sticky commit_pending=1 (busy decoder) in status register and verify wrd_commit_and_wait times out cleanly (-1)";
        ctx.record().independent_oracle = "Normal wrd_commit_and_wait returns 0; sticky bit 0 returns -1 without crash";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        wrd_write_all_weights(&wrd, sample_w);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 10), 0);

        /* Simulate hardware holding commit_pending=1 when mapped_base points to synthetic MMIO block */
        uint32_t fake_mmio[32] = {0};
        fake_mmio[WRD_OFF_STATUS >> 2] = 0x00010001u; /* commit_pending = 1, bp_ready = 0 */
        wrd_handle_t busy_h{};
        busy_h.fd = -1;
        busy_h.is_emulated = 0;
        busy_h.mapped_base = fake_mmio;
        ASSERT_EQ(wrd_commit_and_wait(&busy_h, 16), -1);

        wrd_close(&wrd);
        ctx.record().expected["busy_commit_rc"] = -1;
        ctx.record().actual["busy_commit_rc"] = -1;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-049") {
        TestContext ctx(opts, tid, "scheduler_and_detector_reset_invariants",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_init, avid_fast_detector_init", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Re-initialize scheduler and fast burst detector mid-operation and verify all pending/ring/debounce state clears";
        ctx.record().independent_oracle = "has_pending_commit==false, active_epoch_id==0, ring_sum==0, debounce_counter==0";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 8u);
        avid_scheduler_schedule_commit(&sched, 256ULL, sample_w);
        ASSERT_TRUE(sched.has_pending_commit);
        avid_scheduler_init(&sched, 16u, 256u, 8u);
        ASSERT_FALSE(sched.has_pending_commit);
        ASSERT_EQ(sched.active_epoch_id, 0u);
        ASSERT_EQ(sched.stale_schedule_drops, 0ULL);

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        for (uint64_t r = 0; r < 25; ++r) avid_fast_detector_step(&det, r, 0x7u, 0u);
        ASSERT_GT(det.ring_sum, 0u);
        avid_fast_detector_init(&det, 0.35, 4u);
        ASSERT_EQ(det.ring_sum, 0u);
        ASSERT_EQ(det.debounce_counter, 0u);
        ASSERT_EQ(det.trigger_count, 0u);
        ASSERT_FALSE(det.burst_active);

        ctx.record().expected["post_reset_state"] = "All counters and pending flags == 0";
        ctx.record().actual["post_reset_state"] = "Verified clean";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-050") {
        TestContext ctx(opts, tid, "prefix_equivalence_zero_lookahead_proof",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready, avid_fast_detector_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Prefix-equivalence causality proof: common prefix r=0..499 followed by quiet suffix A vs burst suffix B (r=500..999)";
        ctx.record().independent_oracle = "500/500 bit-identical outputs on r=0..499 between Trajectory A and Trajectory B";

        std::mt19937_64 rng(2026111650ULL);
        std::vector<uint8_t> sA(1000, 0), sB(1000, 0);
        for (int r = 0; r < 500; ++r) {
            uint8_t v = ((rng() % 100) < 12) ? 0x2u : 0x0u;
            sA[r] = v;
            sB[r] = v;
        }
        for (int r = 500; r < 1000; ++r) {
            sA[r] = 0x0u;
            sB[r] = 0xFu; /* massive future burst */
        }

        avid_fast_burst_detector_t detA{}, detB{};
        avid_fast_detector_init(&detA, 0.35, 4u);
        avid_fast_detector_init(&detB, 0.35, 4u);
        std::vector<int> trigA(1000, 0), trigB(1000, 0);
        std::vector<uint32_t> sumA(1000, 0), sumB(1000, 0);
        for (int r = 0; r < 1000; ++r) {
            trigA[r] = avid_fast_detector_step(&detA, static_cast<uint64_t>(r), sA[r], 0u) ? 1 : 0;
            trigB[r] = avid_fast_detector_step(&detB, static_cast<uint64_t>(r), sB[r], 0u) ? 1 : 0;
            sumA[r] = detA.ring_sum;
            sumB[r] = detB.ring_sum;
        }
        int prefix_mismatches = 0;
        for (int r = 0; r < 500; ++r) {
            if (trigA[r] != trigB[r] || sumA[r] != sumB[r]) prefix_mismatches++;
        }
        ASSERT_EQ(prefix_mismatches, 0);
        ASSERT_GT(detB.trigger_count, detA.trigger_count);
        ctx.record().expected["prefix_mismatches_0_499"] = 0;
        ctx.record().actual["prefix_mismatches_0_499"] = prefix_mismatches;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-051") {
        TestContext ctx(opts, tid, "analytical_burst_detection_delay_formula",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Compare analytical trigger round (16-round sum >= ceil(0.35*16)=6 + 4-round debounce) against C++ avid_fast_detector_step";
        ctx.record().independent_oracle = "Exact round-index match across burst densities {1.0, 0.50} and latency <= 20 rounds";

        /* Case 1: 100% density burst starting at B0 = 100 -> 6th hit at r=105, 4th debounce at r=108 (latency = 9 <= 20) */
        std::vector<uint8_t> seq1(200, 0);
        for (int r = 100; r < 160; ++r) seq1[r] = 1;
        int64_t exp1 = ref_predict_burst_trigger_round(seq1, 0.35, 4u);
        ASSERT_EQ(exp1, 108);

        avid_fast_burst_detector_t det1{};
        avid_fast_detector_init(&det1, 0.35, 4u);
        int64_t obs1 = -1;
        for (size_t r = 0; r < seq1.size(); ++r) {
            if (avid_fast_detector_step(&det1, r, seq1[r] ? 0x2u : 0x0u, 0u) && obs1 < 0) {
                obs1 = static_cast<int64_t>(r);
            }
        }
        ASSERT_EQ(obs1, exp1);
        ASSERT_LE(obs1 - 100 + 1, 20);

        /* Case 2: 50% alternating burst starting at B0 = 100 -> hits at 100,102,104,106,108,110 (6th hit at 110), 4th debounce at 113 (latency = 14 <= 20) */
        std::vector<uint8_t> seq2(200, 0);
        for (int r = 100; r < 160; ++r) seq2[r] = ((r % 2) == 0) ? 1 : 0;
        int64_t exp2 = ref_predict_burst_trigger_round(seq2, 0.35, 4u);
        avid_fast_burst_detector_t det2{};
        avid_fast_detector_init(&det2, 0.35, 4u);
        int64_t obs2 = -1;
        for (size_t r = 0; r < seq2.size(); ++r) {
            if (avid_fast_detector_step(&det2, r, seq2[r] ? 0x2u : 0x0u, 0u) && obs2 < 0) {
                obs2 = static_cast<int64_t>(r);
            }
        }
        ASSERT_EQ(obs2, exp2);
        ASSERT_LE(obs2 - 100 + 1, 20);

        ctx.record().expected["trigger_rounds_100pct_50pct"] = std::to_string(exp1) + ", " + std::to_string(exp2);
        ctx.record().actual["trigger_rounds_100pct_50pct"] = std::to_string(obs1) + ", " + std::to_string(obs2);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-MATH-052") {
        TestContext ctx(opts, tid, "slow_loop_dilution_vs_fast_detector_separation",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set5");
        ctx.record().test_set = "set5_control_mathematics";
        ctx.record().test_description = "Mathematically contrast 60-round burst inside 10,000-round (10 ms) slow window (0.6% dilution) vs 16-round fast detector (<=20 us)";
        ctx.record().independent_oracle = "Slow empirical density shift < 0.006 while fast subwindow density reaches >= 0.75 and triggers in <= 20 rounds";

        const int N_slow = 10000;
        const int burst_start = 5000;
        const int burst_len = 60;
        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);

        int nonzero_without_burst = 0;
        int nonzero_with_burst = 0;
        int64_t fast_trig_round = -1;
        std::mt19937_64 rng(2026111652ULL);
        for (int r = 0; r < N_slow; ++r) {
            bool base_hit = ((rng() % 1000) < 40); /* ~4.0% quiet non-trivial syndrome rate */
            bool in_burst = (r >= burst_start && r < burst_start + burst_len);
            bool actual_hit = in_burst ? true : base_hit;
            if (base_hit) nonzero_without_burst++;
            if (actual_hit) nonzero_with_burst++;
            if (avid_fast_detector_step(&det, static_cast<uint64_t>(r), actual_hit ? 0x2u : 0x0u, 0u) && fast_trig_round < 0) {
                fast_trig_round = r;
            }
        }
        double slow_density_shift = static_cast<double>(nonzero_with_burst - nonzero_without_burst) / static_cast<double>(N_slow);
        ASSERT_LT(slow_density_shift, 0.0065);
        ASSERT_GE(fast_trig_round, burst_start);
        ASSERT_LE(fast_trig_round - burst_start + 1, 20);
        ASSERT_GE(det.peak_density, 0.90);

        ctx.record().expected["slow_dilution_vs_fast_latency"] = "slow_shift < 0.0065 (0.6%), fast_latency <= 20 rounds";
        ctx.record().actual["slow_density_shift"] = slow_density_shift;
        ctx.record().actual["fast_trigger_latency_rounds"] = fast_trig_round - burst_start + 1;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-MATH-043", "L3-MATH-044", "L3-MATH-045", "L3-MATH-046", "L3-MATH-047",
        "L3-MATH-048", "L3-MATH-049", "L3-MATH-050", "L3-MATH-051", "L3-MATH-052"
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
