#include "test_assertions.h"
#include "test_common.h"
#include "ingress/dma_ingress_tap.h"

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-020");
    TestContext ctx(opts, opts.test_id, "histogram_xz_channel_isolation",
                    "software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_consume_slot");

    ctx.record().test_description =
        "X/Z independence — verify hist_x is strictly invariant to Z-syndrome stream variations and hist_z is strictly invariant to X-syndrome stream variations";
    ctx.record().input_contract = "syndrome_packet_v1.yaml invariant #1: independent X and Z channel accumulation";
    ctx.record().independent_oracle = "Paired differential stream experiment holding one channel fixed while permuting the other";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/burst_64_seq.bin";

    uint64_t stream_a[64], stream_b[64];
    for (uint32_t i = 0; i < 64; ++i) {
        uint8_t sx = static_cast<uint8_t>((i * 7 + 3) & 0xFu);
        stream_a[i] = avid_pack_packet_u64(sx, 0u, i, 1u, AVID_PKT_FLAG_VALID);
        stream_b[i] = avid_pack_packet_u64(sx, static_cast<uint8_t>((i * 11 + 5) & 0xFu), i, 1u, AVID_PKT_FLAG_VALID);
    }

    avid_dma_ring_t ra, rb;
    avid_dma_ring_init(&ra);
    avid_dma_ring_init(&rb);
    uint32_t hxa[16] = {0}, hza[16] = {0}, va = 0;
    uint32_t hxb[16] = {0}, hzb[16] = {0}, vb = 0;

    ASSERT_TRUE(avid_dma_ring_push_burst(&ra, stream_a, 64));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ra, hxa, hza, &va));
    ASSERT_TRUE(avid_dma_ring_push_burst(&rb, stream_b, 64));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&rb, hxb, hzb, &vb));

    ASSERT_ARRAY_EQ(hxa, hxb, 16);
    ASSERT_EQ(hza[0], 64u);

    /* Symmetric check: hold s_z pattern fixed while varying s_x */
    uint64_t stream_c[64], stream_d[64];
    for (uint32_t i = 0; i < 64; ++i) {
        uint8_t sz = static_cast<uint8_t>((i * 5 + 9) & 0xFu);
        stream_c[i] = avid_pack_packet_u64(0u, sz, i, 1u, AVID_PKT_FLAG_VALID);
        stream_d[i] = avid_pack_packet_u64(static_cast<uint8_t>((i * 13 + 1) & 0xFu), sz, i, 1u, AVID_PKT_FLAG_VALID);
    }
    avid_dma_ring_init(&ra);
    avid_dma_ring_init(&rb);
    uint32_t hxc[16] = {0}, hzc[16] = {0}, vc = 0;
    uint32_t hxd[16] = {0}, hzd[16] = {0}, vd = 0;
    ASSERT_TRUE(avid_dma_ring_push_burst(&ra, stream_c, 64));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ra, hxc, hzc, &vc));
    ASSERT_TRUE(avid_dma_ring_push_burst(&rb, stream_d, 64));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&rb, hxd, hzd, &vd));

    ASSERT_ARRAY_EQ(hzc, hzd, 16);
    ASSERT_EQ(hxc[0], 64u);

    ctx.record().expected["channel_isolation"] = "hist_x(A)==hist_x(B) and hist_z(C)==hist_z(D) bit-for-bit across all 16 bins";
    ctx.record().actual["channel_isolation"] = "Exact 16/16 bin invariance verified in both X and Z directions";
    return ctx.finalize(TestStatus::PASS);
}
