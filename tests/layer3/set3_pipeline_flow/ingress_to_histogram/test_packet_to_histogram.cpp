#include "test_assertions.h"
#include "test_common.h"
#include "ingress/dma_ingress_tap.h"
#include "ingress/packet_validator.h"
#include <cstring>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-001");
    TestContext ctx(opts, "L3-FLOW-001", "packet_ingress_to_histogram_continuity",
                    "software/src/ingress/packet_validator.cpp, software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_push_burst -> avid_dma_ring_consume_slot", "set3");
    ctx.record().test_set = "set3_pipeline_flow";
    ctx.record().test_description = "Pack known 64-bit sequence (s_x in {0,6}, s_z in {0,9}), push into DMA ring, consume slot, verify exact bin deltas";
    ctx.record().independent_oracle = "hist_x[0]==40, hist_x[6]==24, hist_z[0]==45, hist_z[9]==19, valid_packets==64";

    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);

    uint64_t burst[64];
    uint32_t exp_hx[16] = {0}, exp_hz[16] = {0};
    for (uint32_t i = 0; i < 64; ++i) {
        uint8_t sx = (i < 40) ? 0u : 6u;
        uint8_t sz = (i < 45) ? 0u : 9u;
        burst[i] = avid_pack_packet_u64(sx, sz, i, 1u, AVID_PKT_FLAG_VALID);
        exp_hx[sx]++;
        exp_hz[sz]++;
    }

    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, burst, 64));
    uint32_t hx[16] = {0}, hz[16] = {0}, n_valid = 0;
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &n_valid));
    ASSERT_EQ(n_valid, 64u);
    for (int s = 0; s < 16; ++s) {
        ASSERT_EQ(hx[s], exp_hx[s]);
        ASSERT_EQ(hz[s], exp_hz[s]);
    }
    ctx.record().expected["hist_x_6"] = "24";
    ctx.record().expected["hist_z_9"] = "19";
    ctx.record().actual["hist_x_6"] = std::to_string(hx[6]);
    ctx.record().actual["hist_z_9"] = std::to_string(hz[9]);
    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
