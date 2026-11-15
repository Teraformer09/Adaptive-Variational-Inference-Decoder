#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/dma_ingress_tap.h"
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-010");
    TestContext ctx(opts, opts.test_id, "dma_ring_packet_burst_and_overflow",
                    "software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_init, avid_dma_ring_push_burst, avid_dma_ring_consume_slot");

    ctx.record().test_description =
        "Packet burst — test 4x64 sequential burst cadence, 5th unconsumed burst backpressure (overflow_drops==64), and oversized burst (n_pkts=100 > 64) handling";
    ctx.record().input_contract = "4 slots x 64 packets sustained with 0 drops when consumed; full ring returns false & records drops; n_pkts > 64 must not silently drop packets";
    ctx.record().independent_oracle = "Slot FSM state transitions (FREE->DMA_OWNED->CPU_READY->CPU_OWNED->FREE) + packet conservation count";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/burst_64_seq.bin";

    /* 1. Sustained 4-burst sequential push + consume */
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0};
    uint32_t total_consumed = 0;

    for (int b = 0; b < 4; ++b) {
        uint64_t burst[64];
        for (int i = 0; i < 64; ++i) {
            burst[i] = avid_pack_packet_u64(static_cast<uint8_t>(i & 0xF), static_cast<uint8_t>((i + 3) & 0xF),
                                            static_cast<uint32_t>(b * 64 + i), 1, AVID_PKT_FLAG_VALID);
        }
        ASSERT_TRUE(avid_dma_ring_push_burst(&ring, burst, 64));
        uint32_t vcnt = 0;
        ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &vcnt));
        ASSERT_EQ(vcnt, 64u);
        total_consumed += vcnt;
    }
    ASSERT_EQ(total_consumed, 256u);
    ASSERT_EQ(ring.overflow_drops, 0ULL);

    /* 2. Fill all 4 slots without consuming, then push 5th burst */
    avid_dma_ring_init(&ring);
    uint64_t burst64[64];
    for (int s = 0; s < 5; ++s) {
        for (int i = 0; i < 64; ++i) {
            burst64[i] = avid_pack_packet_u64(1, 2, static_cast<uint32_t>(s * 64 + i), 1, AVID_PKT_FLAG_VALID);
        }
        bool pushed = avid_dma_ring_push_burst(&ring, burst64, 64);
        if (s < 4) {
            ASSERT_TRUE(pushed);
        } else {
            ASSERT_FALSE(pushed);
        }
    }
    ASSERT_EQ(ring.overflow_drops, 64ULL);

    /* 3. Oversized single burst (n_pkts = 100 > AVID_DMA_SLOT_PACKETS = 64) */
    avid_dma_ring_init(&ring);
    std::vector<uint64_t> burst100(100);
    for (int i = 0; i < 100; ++i) {
        burst100[i] = avid_pack_packet_u64(5, 7, static_cast<uint32_t>(i), 1, AVID_PKT_FLAG_VALID);
    }
    bool pushed_oversized = avid_dma_ring_push_burst(&ring, burst100.data(), 100u);
    ASSERT_FALSE(pushed_oversized);
    ASSERT_EQ(ring.overflow_drops, 100ULL);
    uint32_t hx2[16] = {0}, hz2[16] = {0}, vcnt2 = 0;
    ASSERT_FALSE(avid_dma_ring_consume_slot(&ring, hx2, hz2, &vcnt2));
    ASSERT_EQ(vcnt2, 0u);
    bool silent_truncation = (pushed_oversized && vcnt2 == 64u && ring.overflow_drops == 0ULL);

    ctx.log("4x64 sequential cadence: total_consumed=256, overflow_drops=0 (PASS).");
    ctx.log("5th unconsumed slot burst: returned false, overflow_drops=64 (PASS).");
    ctx.log("Oversized n_pkts=100 burst: returned true, consumed=" + std::to_string(vcnt2) +
            ", overflow_drops=" + std::to_string(ring.overflow_drops) +
            " -> silently dropped 36 packets without incrementing overflow_drops!");

    ctx.record().expected["4_slot_cadence_and_5th_backpressure"] = "256 consumed with 0 drops; 5th unconsumed returns false with overflow_drops=64";
    ctx.record().expected["oversized_burst_100_pkts"] = "Reject or record 36 excess packets in overflow_drops";
    ctx.record().actual["4_slot_cadence_and_5th_backpressure"] = "PASS (256 consumed, 5th rejected with overflow_drops=64)";
    ctx.record().actual["oversized_burst_100_pkts"] =
        silent_truncation ? "SILENT TRUNCATION: returned true, stored 64, dropped 36 with overflow_drops=0" : "Bounded";

    if (silent_truncation) {
        ctx.record_failure("avid_dma_ring_push_burst(n_pkts=100) silently truncated 36 packets with overflow_drops=0");
        return ctx.finalize(TestStatus::FAIL, "oversized_dma_burst_silently_truncated_without_drop_accounting");
    }
    return ctx.finalize(TestStatus::PASS);
}
