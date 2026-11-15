#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/packet_validator.h"
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-007");
    TestContext ctx(opts, opts.test_id, "packet_sequence_continuity",
                    "software/src/ingress/packet_validator.cpp",
                    "avid_validate_packet");

    ctx.record().test_description =
        "Sequence continuity — test monotonic seq (10->11), gap (11->13), duplicate (13->13), UINT32_MAX->0 rollover, and out-of-order recovery (11->999->12)";
    ctx.record().input_contract = "Monotonic +1 (mod 2^32) accepted; gaps and duplicates counted distinctly without corrupting expected sequence on spoofed/out-of-order packet";
    ctx.record().independent_oracle = "Monotonic sequence counter state machine";
    ctx.record().fixture = "tests/layer3/fixtures/packets/boundary/seq_rollover_pair.bin";

    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t pkt{};

    bool ok10 = avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 10, 1, AVID_PKT_FLAG_VALID), &pkt);
    bool ok11 = avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 11, 1, AVID_PKT_FLAG_VALID), &pkt);
    bool ok13_gap = avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 13, 1, AVID_PKT_FLAG_VALID), &pkt);
    uint64_t gaps_after_13 = val.seq_gap_count;
    bool ok13_dup = avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 13, 1, AVID_PKT_FLAG_VALID), &pkt);
    uint64_t gaps_after_dup = val.seq_gap_count;

    ASSERT_TRUE(ok10);
    ASSERT_TRUE(ok11);
    ASSERT_FALSE(ok13_gap);
    ASSERT_EQ(gaps_after_13, 1ULL);
    ASSERT_FALSE(ok13_dup);
    ASSERT_EQ(gaps_after_dup, 2ULL);

    /* Rollover UINT32_MAX -> 0 */
    std::vector<uint64_t> roll;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/boundary/seq_rollover_pair.bin"), roll));
    avid_packet_validator_t val_roll;
    avid_validator_init(&val_roll);
    ASSERT_TRUE(avid_validate_packet(&val_roll, roll[0], &pkt));
    ASSERT_TRUE(avid_validate_packet(&val_roll, roll[1], &pkt));

    /* Out-of-order / corrupted seq injection: 10 -> 11 -> 999 (rejected) -> 12 (legitimate next packet!) */
    avid_packet_validator_t val_ooo;
    avid_validator_init(&val_ooo);
    avid_validate_packet(&val_ooo, avid_pack_packet_u64(1, 1, 10, 1, AVID_PKT_FLAG_VALID), &pkt);
    avid_validate_packet(&val_ooo, avid_pack_packet_u64(1, 1, 11, 1, AVID_PKT_FLAG_VALID), &pkt);
    bool ok_corrupt_999 = avid_validate_packet(&val_ooo, avid_pack_packet_u64(1, 1, 999, 1, AVID_PKT_FLAG_VALID), &pkt);
    bool ok_legit_12 = avid_validate_packet(&val_ooo, avid_pack_packet_u64(1, 1, 12, 1, AVID_PKT_FLAG_VALID), &pkt);

    ctx.log("seq 10->11 accepted; gap 13 rejected (seq_gap_count=1); duplicate 13 rejected (seq_gap_count=2, no separate duplicate counter); UINT32_MAX->0 rollover accepted.");
    ctx.log("Spoofed/out-of-order seq=999 after seq=11 rejected=" + std::string(!ok_corrupt_999 ? "true" : "false") +
            ", but mutated v->last_seq_id=999 so legitimate next packet seq=12 was ALSO rejected (ok_legit_12=" +
            std::string(ok_legit_12 ? "true" : "false") + ").");

    ctx.record().expected["seq_10_11_gap13_dup13_rollover"] = "10,11,rollover ACCEPT; gap 13 & dup 13 REJECT with distinct gap vs duplicate tracking and without poisoning last_seq_id on single corrupt seq";
    ctx.record().actual["seq_10_11_gap13_dup13_rollover"] = "10,11,rollover ACCEPT; gap 13 & dup 13 REJECT (both increment seq_gap_count)";
    ctx.record().actual["legit_seq12_after_rejected_999"] = ok_legit_12 ? "ACCEPT" : "REJECT (last_seq_id overwritten by rejected packet 999)";

    if (!ok_legit_12) {
        ctx.record_failure("Rejected packet seq=999 mutated v->last_seq_id=999, causing legitimate next packet seq=12 to fail");
        return ctx.finalize(TestStatus::FAIL, "rejected_sequence_packet_mutates_validator_last_seq_id_and_lacks_duplicate_counter");
    }
    return ctx.finalize(TestStatus::PASS);
}
