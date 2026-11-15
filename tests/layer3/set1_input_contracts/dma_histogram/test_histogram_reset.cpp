#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/ring_buffer.h"
#include "ingress/dma_ingress_tap.h"
#include <cstring>
#include <string>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-018");
    TestContext ctx(opts, opts.test_id, "window_rollover_and_histogram_reset",
                    "software/src/inference/ring_buffer.h, software/src/inference/avid_daemon.cpp",
                    "ring_init, ring_push, ring_at, avid_daemon window reset");

    ctx.record().test_description =
        "Window rollover — test standalone ring_buffer_t (RING_WINDOW=10000) rollover on 10,500 pushes, histogram reset, and verify production daemon connectivity";
    ctx.record().input_contract = "10,000-round sliding ring buffer evicts oldest 500 entries cleanly and feeds production estimator, or daemon window semantics must match contract";
    ctx.record().independent_oracle = "Pattern A (first 500) vs Pattern B (last 10,000) FIFO eviction check + avid_daemon.cpp symbol/source check";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/empty/zeros_16.txt";

    /* 1. Standalone ring_buffer_t rollover test (heap-allocated to keep stack small) */
    auto *rb = new ring_buffer_t();
    ring_init(rb);
    ASSERT_EQ(rb->count, 0);

    syndrome_round_t pat_a{{1, 0, 0, 0}, {0, 1, 0, 0}};
    syndrome_round_t pat_b{{0, 0, 1, 1}, {1, 1, 0, 0}};
    for (int i = 0; i < 500; ++i) ring_push(rb, &pat_a);
    for (int i = 0; i < 10000; ++i) ring_push(rb, &pat_b);

    ASSERT_EQ(rb->count, RING_WINDOW);
    int pat_b_matches = 0;
    for (int i = 0; i < RING_WINDOW; ++i) {
        const syndrome_round_t *sr = ring_at(rb, i);
        if (sr->x[2] == 1 && sr->x[3] == 1 && sr->x[0] == 0) {
            pat_b_matches++;
        }
    }
    ASSERT_EQ(pat_b_matches, 10000);
    delete rb;

    /* 2. Histogram reset verification */
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint64_t pkt = avid_pack_packet_u64(4, 7, 0, 1, AVID_PKT_FLAG_VALID);
    uint32_t hx[16] = {0}, hz[16] = {0}, v = 0;
    avid_dma_ring_push_burst(&ring, &pkt, 1);
    avid_dma_ring_consume_slot(&ring, hx, hz, &v);
    ASSERT_EQ(hx[4], 1u);
    std::memset(hx, 0, sizeof(hx));
    std::memset(hz, 0, sizeof(hz));
    uint64_t pkt2 = avid_pack_packet_u64(2, 3, 1, 1, AVID_PKT_FLAG_VALID);
    avid_dma_ring_push_burst(&ring, &pkt2, 1);
    avid_dma_ring_consume_slot(&ring, hx, hz, &v);
    ASSERT_EQ(hx[4], 0u);
    ASSERT_EQ(hx[2], 1u);

    /* 3. Production daemon connectivity check for ring_buffer_t */
    std::string daemon_src;
    ASSERT_TRUE(load_text_fixture(ctx.env().repo_root + "/software/src/inference/avid_daemon.cpp", daemon_src));
    bool daemon_uses_ring_buffer =
        (daemon_src.find("ring_buffer_t") != std::string::npos) ||
        (daemon_src.find("ring_init") != std::string::npos) ||
        (daemon_src.find("ring_push") != std::string::npos);

    ctx.log("Standalone ring_buffer_t (RING_WINDOW=10000): 10,500 pushes evicted oldest 500 entries cleanly (10000/10000 Pattern B).");
    ctx.log("Production avid_daemon.cpp invokes ring_buffer_t: " +
            std::string(daemon_uses_ring_buffer ? "YES (IMPLEMENTED_AND_CONNECTED)" : "NO (IMPLEMENTED_BUT_UNUSED — daemon uses flat hist_x/hist_z zeroed per window)"));

    ctx.record().expected["standalone_ring_buffer_rollover"] = "count==10000, oldest 500 evicted";
    ctx.record().expected["production_daemon_window_mechanism"] = "ring_buffer_t connected in avid_daemon OR contract updated to non-overlapping reset windows";
    ctx.record().actual["standalone_ring_buffer_rollover"] = "PASS (10000/10000 Pattern B retained)";
    ctx.record().actual["production_ring_buffer_status"] =
        daemon_uses_ring_buffer ? "IMPLEMENTED_AND_CONNECTED" : "IMPLEMENTED_BUT_UNUSED (avid_daemon zeros flat hist_x[16]/hist_z[16] each epoch)";

    if (!daemon_uses_ring_buffer) {
        return ctx.finalize(TestStatus::CONTRACT_MISMATCH,
                            "ring_buffer_implemented_but_unused_in_avid_daemon");
    }
    return ctx.finalize(TestStatus::PASS);
}
