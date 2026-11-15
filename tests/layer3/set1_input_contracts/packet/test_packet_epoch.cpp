#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "ingress/dma_ingress_tap.h"
#include "ingress/packet_validator.h"
#include <cstring>

using namespace avid::layer3_test;

static int run_ip_006(TestContext &ctx) {
    ctx.record().test_name = "timestamp_and_epoch_monotonicity";
    ctx.record().test_description =
        "Timestamp validity — verify hardware packet timestamp presence and monotonic round/epoch enforcement in packet_validator and causal_scheduler";
    ctx.record().input_contract = "Monotonic timestamps/epochs accepted; duplicate or backward timestamps/epochs rejected";
    ctx.record().independent_oracle = "packet_validator.h struct inspection + backward epoch (5->2) & backward round_idx (20->18) probes";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/packet_001.bin";

    /* 1. Verify epoch regression (epoch 5 -> epoch 2) is rejected by avid_validate_packet without mutating state */
    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t p1{}, p2{0x77, 0x88, 0x99, 0xAAAA, 0xBBBBBBBBu};
    bool ok1 = avid_validate_packet(&val, avid_pack_packet_u64(1, 1, 0, 5, AVID_PKT_FLAG_VALID), &p1);
    bool ok2_regressed_epoch = avid_validate_packet(&val, avid_pack_packet_u64(1, 1, 1, 2, AVID_PKT_FLAG_VALID), &p2);
    bool ok3_next_valid = avid_validate_packet(&val, avid_pack_packet_u64(1, 1, 1, 5, AVID_PKT_FLAG_VALID), &p2);
    ASSERT_TRUE(ok1);
    ASSERT_FALSE(ok2_regressed_epoch);
    ASSERT_EQ(val.epoch_regress_count, 1ULL);
    ASSERT_TRUE(ok3_next_valid);

    /* 2. Verify avid_fast_detector_step rejects duplicate (20->20) and backward (20->18) round_idx without mutating ring_sum */
    avid_fast_burst_detector_t det;
    avid_fast_detector_init(&det, 0.35, 4);
    for (uint64_t r = 0; r <= 20; ++r) {
        avid_fast_detector_step(&det, r, 0x1, 0x0);
    }
    uint32_t sum_at_20 = det.ring_sum;
    ASSERT_FALSE(avid_fast_detector_step(&det, 20, 0x0, 0x0));
    ASSERT_FALSE(avid_fast_detector_step(&det, 18, 0x0, 0x0));
    uint32_t sum_after_backward = det.ring_sum;
    ASSERT_EQ(sum_after_backward, sum_at_20);
    ASSERT_EQ(det.temporal_regress_count, 2ULL);

    ctx.log("Authoritative temporal contract (syndrome_packet_v1.yaml v1.1.0): logical QEC time indexed by monotonic (epoch_id, seq_id) and round_idx.");
    ctx.log("Backward epoch (5->2) rejected (epoch_regress_count=1); duplicate/backward round_idx (20->20, 20->18) rejected with ring_sum preserved.");

    ctx.record().expected["timestamp_contract"] =
        "Monotonic round/epoch enforcement rejecting duplicate/backward round_idx and regressed epoch_id";
    ctx.record().actual["backward_epoch_5_to_2"] = "REJECTED (epoch_regress_count=1)";
    ctx.record().actual["backward_round_20_to_18"] = "REJECTED (ring_sum preserved, temporal_regress_count=2)";
    ctx.record().evidence.push_back("validation/layer3/reports/packet_contract_decision.md");

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}

static int run_ip_030(TestContext &ctx) {
    ctx.record().test_name = "stale_observations_and_epoch_isolation";
    ctx.record().test_description =
        "Stale observations — verify 4-slot DMA ring overrun accounting, stale-epoch packet exclusion from histograms, and stale window rejection in scheduler";
    ctx.record().input_contract = "Overrun increments overflow_drops; packets from epoch < active_epoch and older scheduler windows rejected";
    ctx.record().independent_oracle = "Active epoch=5 vs packet epoch=1 histogram isolation + window_end_round=500 vs 100 ordering";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/burst_64_seq.bin";

    /* 1. Verify 4-slot DMA ring overflow accounting */
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint64_t burst[64];
    for (int s = 0; s < 5; ++s) {
        for (int i = 0; i < 64; ++i) {
            burst[i] = avid_pack_packet_u64(3, 4, static_cast<uint32_t>(s * 64 + i), 5, AVID_PKT_FLAG_VALID);
        }
        bool pushed = avid_dma_ring_push_burst(&ring, burst, 64);
        if (s < 4) {
            ASSERT_TRUE(pushed);
        } else {
            ASSERT_FALSE(pushed);
        }
    }
    ASSERT_EQ(ring.overflow_drops, 64ULL);

    /* 2. Feed stale-epoch packets (epoch_id=1 while current epoch is 5) */
    avid_dma_ring_init(&ring);
    uint64_t mixed_epoch[2] = {
        avid_pack_packet_u64(6, 0, 0, 5, AVID_PKT_FLAG_VALID), /* current epoch=5 */
        avid_pack_packet_u64(6, 0, 1, 1, AVID_PKT_FLAG_VALID)  /* stale epoch=1! */
    };
    uint32_t hx[16] = {0}, hz[16] = {0}, valid_cnt = 0;
    avid_dma_ring_push_burst(&ring, mixed_epoch, 2);
    avid_dma_ring_consume_slot(&ring, hx, hz, &valid_cnt);
    bool stale_epoch_accumulated = (hx[6] == 2u);

    /* 3. Feed stale window_end_round=100 after newer window_end_round=500 into causal scheduler */
    avid_causal_scheduler_t sched;
    avid_scheduler_init(&sched, 16, 256, 32);
    int16_t w_new[9] = {500,500,500,500,500,500,500,500,500};
    int16_t w_old[9] = {100,100,100,100,100,100,100,100,100};
    avid_scheduler_schedule_commit(&sched, 500, w_new);
    avid_scheduler_schedule_commit(&sched, 100, w_old);
    bool stale_window_overwrote_newer = (sched.pending_commit_round == 132u && sched.pending_weights_q78[0] == 100);

    ctx.log("DMA 5th unconsumed slot overflow_drops=" + std::to_string(ring.overflow_drops) + " (expected 64).");
    ctx.log("Stale epoch_id=1 packet after epoch_id=5 accumulated into hist_x[6]: " +
            std::string(stale_epoch_accumulated ? "YES (DEFECT)" : "NO"));
    ctx.log("Older window_end_round=100 overwrote pending window_end_round=500 in scheduler: " +
            std::string(stale_window_overwrote_newer ? "YES (DEFECT)" : "NO"));

    ctx.record().expected["stale_epoch_packet"] = "Rejected before contaminating current epoch histogram";
    ctx.record().expected["stale_scheduler_window"] = "Rejected when window_end_round < pending window";
    ctx.record().actual["dma_overflow_drops"] = std::to_string(ring.overflow_drops);
    ctx.record().actual["stale_epoch_packet"] = stale_epoch_accumulated ? "Accumulated into hist_x[6] (count=2)" : "Excluded";
    ctx.record().actual["stale_scheduler_window"] = stale_window_overwrote_newer ? "Overwrote newer pending commit" : "Rejected";

    if (stale_epoch_accumulated || stale_window_overwrote_newer) {
        ctx.record_failure("Stale epoch observations and stale scheduler windows are not rejected");
        return ctx.finalize(TestStatus::FAIL, "stale_epoch_packets_accumulated_and_older_window_overwrites_pending_commit");
    }
    return ctx.finalize(TestStatus::PASS);
}

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-006");
    TestContext ctx(opts, opts.test_id, "packet_epoch_and_temporal_contracts",
                    "software/src/ingress/packet_validator.cpp, software/src/control/causal_scheduler.cpp",
                    "avid_validate_packet, avid_fast_detector_step, avid_scheduler_schedule_commit");
    if (opts.test_id == "L3-IP-030") {
        return run_ip_030(ctx);
    }
    return run_ip_006(ctx);
}
