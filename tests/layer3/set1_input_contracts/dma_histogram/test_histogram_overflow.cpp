#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "inference/noise_model.h"
#include "ingress/dma_ingress_tap.h"
#include <cstdint>
#include <limits>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-017");
    TestContext ctx(opts, opts.test_id, "histogram_counter_overflow",
                    "software/src/ingress/dma_ingress_tap.cpp, software/src/inference/noise_model.cpp",
                    "avid_dma_ring_consume_slot, NoiseModel::compute_nll_from_histogram");

    ctx.record().test_description =
        "Counter overflow — verify uint64_t total sum > 2^32 in NoiseModel::compute_nll_from_histogram and test UINT32_MAX bin increment behavior in avid_dma_ring_consume_slot";
    ctx.record().input_contract = "Total sum > 2^32 must not wrap; individual bin at UINT32_MAX must saturate or report overflow instead of wrapping to 0";
    ctx.record().independent_oracle = "16 * 0x7FFFFFFF = 34,359,738,352 > 2^32 uniform NLL oracle + UINT32_MAX + 1 saturation check";
    ctx.record().fixture = "tests/layer3/fixtures/histograms/overflow/half_u32_16.txt";

    /* 1. NoiseModel::compute_nll_from_histogram with sum > 2^32 */
    std::array<uint32_t, 16> big_hist{};
    ASSERT_TRUE(load_u32_histogram_fixture(ctx.fixture_path("histograms/overflow/half_u32_16.txt"), big_hist));
    uint32_t small_uniform[16];
    for (int i = 0; i < 16; ++i) small_uniform[i] = 100u;

    NoiseModel model;
    model.init(0.012, 0.004);
    double nll_big = model.compute_nll_from_histogram(big_hist.data());
    double nll_small = model.compute_nll_from_histogram(small_uniform);
    ASSERT_FINITE(nll_big);
    ASSERT_NEAR(nll_big, nll_small, 1e-12);

    /* 2. avid_dma_ring_consume_slot bin increment at UINT32_MAX */
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0}, vcnt = 0;
    hx[6] = std::numeric_limits<uint32_t>::max();
    uint64_t one_pkt = avid_pack_packet_u64(6u, 1u, 0u, 1u, AVID_PKT_FLAG_VALID);
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, &one_pkt, 1));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &vcnt));
    bool wrapped_to_zero = (hx[6] == 0u);

    ctx.log("NoiseModel::compute_nll_from_histogram with sum=34,359,738,352 (>2^32): nll=" +
            std::to_string(nll_big) + " matches uniform 100-per-bin nll=" + std::to_string(nll_small) + " (PASS).");
    ctx.log("avid_dma_ring_consume_slot with hist_x[6]=UINT32_MAX + 1 packet: resulting hist_x[6]=" +
            std::to_string(hx[6]) + (wrapped_to_zero ? " (SILENT WRAPAROUND TO ZERO!)" : " (saturated)"));

    ctx.record().expected["noise_model_sum_gt_2pow32"] = "Finite, matches normalized uniform NLL without 32-bit wrap";
    ctx.record().expected["dma_consume_bin_at_uint32_max"] = "Saturate at UINT32_MAX or return explicit overflow error";
    ctx.record().actual["noise_model_sum_gt_2pow32"] = "PASS (nll=" + std::to_string(nll_big) + ")";
    ctx.record().actual["dma_consume_bin_at_uint32_max"] =
        wrapped_to_zero ? "WRAPPED TO 0 (0xFFFFFFFFu + 1 -> 0)" : std::to_string(hx[6]);

    if (wrapped_to_zero) {
        ctx.record_failure("dma_ingress_tap.cpp:52 hist_x[s]++ wrapped UINT32_MAX silently to 0");
        return ctx.finalize(TestStatus::FAIL, "histogram_bin_uint32_wraparound_in_dma_ingress_tap");
    }
    return ctx.finalize(TestStatus::PASS);
}
