#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "histogram_builder.h"
#include "ingress/dma_ingress_tap.h"
#include "inference/noise_model.h"
#include <vector>

using namespace avid::layer3_test;

static int run_ip_011(TestContext &ctx) {
    ctx.record().test_name = "histogram_dimensions_and_accumulation";
    ctx.record().test_description =
        "Histogram dimensions — verify NM_N_STATES==16, all 16 syndrome codes 0x0..0xF addressable, and exact counts against independent histogram oracle";
    ctx.record().input_contract = "16-bin uint32_t hist_x[16] and hist_z[16] covering 0x0..0xF";
    ctx.record().independent_oracle = "histogram_builder::compute_reference_histogram_from_u64_stream";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/histogram_5pkt_stream.bin";

    ASSERT_EQ(NM_N_STATES, 16);

    /* 1. Controlled 5-packet fixture: X={3,3,4,3,7}, Z={1,1,2,1,5} */
    std::vector<uint64_t> words5;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/valid/histogram_5pkt_stream.bin"), words5));
    HistogramPair ref5 = compute_reference_histogram_from_u64_stream(words5);

    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0}, vcnt = 0;
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, words5.data(), static_cast<uint32_t>(words5.size())));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &vcnt));
    ASSERT_EQ(vcnt, 5u);
    ASSERT_ARRAY_EQ(hx, ref5.hist_x.data(), 16);
    ASSERT_ARRAY_EQ(hz, ref5.hist_z.data(), 16);
    ASSERT_EQ(hx[3], 3u);
    ASSERT_EQ(hx[4], 1u);
    ASSERT_EQ(hx[7], 1u);
    ASSERT_EQ(hz[1], 3u);
    ASSERT_EQ(hz[2], 1u);
    ASSERT_EQ(hz[5], 1u);

    /* 2. Full 16-state addressability check */
    avid_dma_ring_init(&ring);
    uint64_t sweep16[16];
    for (uint8_t s = 0; s < 16; ++s) {
        sweep16[s] = avid_pack_packet_u64(s, static_cast<uint8_t>(15u - s), s, 1, AVID_PKT_FLAG_VALID);
    }
    uint32_t sx16[16] = {0}, sz16[16] = {0}, v16 = 0;
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, sweep16, 16));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, sx16, sz16, &v16));
    ASSERT_EQ(v16, 16u);
    for (int s = 0; s < 16; ++s) {
        ASSERT_EQ(sx16[s], 1u);
        ASSERT_EQ(sz16[s], 1u);
    }

    ctx.record().expected["16_bin_oracle_match"] = "Exact match on 5-pkt fixture and 16-state sweep (all bins=1)";
    ctx.record().actual["5pkt_hist_x_bins_3_4_7"] =
        std::to_string(hx[3]) + "," + std::to_string(hx[4]) + "," + std::to_string(hx[7]);
    ctx.record().actual["16_state_sweep_valid"] = std::to_string(v16) + "/16";
    return ctx.finalize(TestStatus::PASS);
}

static int run_ip_014(TestContext &ctx) {
    ctx.record().test_name = "single_state_histogram";
    ctx.record().test_description =
        "Single-state histogram — feed 64 valid packets all with s_x=6, s_z=9 and verify hist_x[6]==64, hist_z[9]==64, all other bins==0";
    ctx.record().input_contract = "Constant-syndrome burst increments only the matching X and Z bin";
    ctx.record().independent_oracle = "bin6_64.txt fixture & single-bin Kronecker delta oracle";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/single_bin/bin6_64.txt";

    std::array<uint32_t, 16> expected_hx{};
    ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/single_bin/bin6_64.txt"), expected_hx));

    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint64_t pkts[64];
    for (uint32_t i = 0; i < 64; ++i) {
        pkts[i] = avid_pack_packet_u64(6u, 9u, i, 1u, AVID_PKT_FLAG_VALID);
    }
    uint32_t hx[16] = {0}, hz[16] = {0}, vcnt = 0;
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, pkts, 64));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &vcnt));

    ASSERT_EQ(vcnt, 64u);
    ASSERT_ARRAY_EQ(hx, expected_hx.data(), 16);
    for (int s = 0; s < 16; ++s) {
        ASSERT_EQ(hz[s], (s == 9) ? 64u : 0u);
    }

    ctx.record().expected["hist_counts"] = "hist_x[6]==64 (others 0); hist_z[9]==64 (others 0)";
    ctx.record().actual["hist_counts"] =
        "hist_x[6]=" + std::to_string(hx[6]) + ", hist_z[9]=" + std::to_string(hz[9]);
    return ctx.finalize(TestStatus::PASS);
}

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-011");
    TestContext ctx(opts, opts.test_id, "histogram_accumulation",
                    "software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_consume_slot");
    if (opts.test_id == "L3-IP-014") {
        return run_ip_014(ctx);
    }
    return run_ip_011(ctx);
}
