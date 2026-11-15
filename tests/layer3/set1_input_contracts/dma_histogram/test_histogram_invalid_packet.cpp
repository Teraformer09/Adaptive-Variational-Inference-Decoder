#include "test_assertions.h"
#include "test_common.h"
#include "histogram_builder.h"
#include "ingress/dma_ingress_tap.h"
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-DMA-INVALID-PKT");
    TestContext ctx(opts, opts.test_id, "histogram_invalid_packet_exclusion",
                    "software/src/ingress/dma_ingress_tap.cpp",
                    "avid_dma_ring_consume_slot");

    ctx.record().test_description =
        "Histogram invalid-packet isolation — verify missing VALID, PARITY_ERR, and sequence-gap packets never increment hist_x or hist_z in avid_dma_ring_consume_slot";
    ctx.record().input_contract = "Invalid packets in DMA slot must contribute 0 counts to both X and Z histograms";
    ctx.record().independent_oracle = "compute_reference_histogram_from_u64_stream";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/parity_err_flag.bin";

    std::vector<uint64_t> mixed = {
        avid_pack_packet_u64(2, 3, 0, 1, AVID_PKT_FLAG_VALID),                          /* valid seq=0 */
        avid_pack_packet_u64(8, 9, 1, 1, 0x00u),                                        /* invalid: VALID=0 */
        avid_pack_packet_u64(10, 11, 1, 1, AVID_PKT_FLAG_VALID | AVID_PKT_FLAG_PARITY_ERR), /* invalid: PARITY_ERR */
        avid_pack_packet_u64(4, 5, 1, 1, AVID_PKT_FLAG_VALID),                          /* valid seq=1 (since rejected flag pkts did not advance last_seq_id) */
        avid_pack_packet_u64(12, 13, 5, 1, AVID_PKT_FLAG_VALID),                        /* invalid: seq gap 1->5 */
        avid_pack_packet_u64(7, 14, 6, 1, AVID_PKT_FLAG_VALID)                          /* valid after last_seq_id became 5 -> 6 */
    };

    HistogramPair ref = compute_reference_histogram_from_u64_stream(mixed);
    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0}, vcnt = 0;
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, mixed.data(), static_cast<uint32_t>(mixed.size())));
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &vcnt));

    ASSERT_EQ(vcnt, ref.accepted_packets);
    ASSERT_ARRAY_EQ(hx, ref.hist_x.data(), 16);
    ASSERT_ARRAY_EQ(hz, ref.hist_z.data(), 16);
    ASSERT_EQ(hx[8], 0u);
    ASSERT_EQ(hx[10], 0u);
    ASSERT_EQ(hx[12], 0u);

    ctx.record().expected["invalid_packet_contributions"] = "0 counts in bins 8, 10, 12; exact match with reference oracle";
    ctx.record().actual["valid_packets_accepted"] = std::to_string(vcnt) + "/6";
    return ctx.finalize(TestStatus::PASS);
}
