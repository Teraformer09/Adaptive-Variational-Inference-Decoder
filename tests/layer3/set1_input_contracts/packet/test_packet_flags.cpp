#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/packet_validator.h"
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-004");
    TestContext ctx(opts, opts.test_id, "packet_field_encoding_and_flags",
                    "software/src/ingress/packet_validator.cpp",
                    "avid_validate_packet");

    ctx.record().test_description =
        "Field encoding — test VALID, OVERFLOW, PARITY_ERR, EPOCH_MARK combinations vs flags=0 and reserved upper flag bits [7:4]";
    ctx.record().input_contract = "Only defined flags [3:0] with VALID=1 and PARITY_ERR=0 accepted; flags=0 or reserved bits [7:4] rejected";
    ctx.record().independent_oracle = "Flag bitmask specification (VALID=0x01, OVERFLOW=0x02, PARITY_ERR=0x04, EPOCH_MARK=0x08; reserved[7:4]==0)";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/reserved_upper_flags.bin";

    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t pkt{};

    /* 1. Missing VALID (flags=0x00) -> must reject */
    std::vector<uint64_t> missing_valid;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/invalid/missing_valid_flag.bin"), missing_valid));
    bool ok_zero_flags = avid_validate_packet(&val, missing_valid[0], &pkt);
    ASSERT_FALSE(ok_zero_flags);
    ASSERT_EQ(val.invalid_flag_count, 1ULL);

    /* 2. Defined valid combos: VALID, VALID|OVERFLOW, VALID|EPOCH_MARK */
    ASSERT_TRUE(avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 10, 1, AVID_PKT_FLAG_VALID), &pkt));
    ASSERT_TRUE(avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 11, 1, AVID_PKT_FLAG_VALID | AVID_PKT_FLAG_OVERFLOW), &pkt));
    ASSERT_EQ(val.overflow_warn_count, 1ULL);
    ASSERT_TRUE(avid_validate_packet(&val, avid_pack_packet_u64(1, 2, 12, 1, AVID_PKT_FLAG_VALID | AVID_PKT_FLAG_EPOCH_MARK), &pkt));

    /* 3. Reserved upper flag bits [7:4] set (flags = 0xF1 = VALID | 0xF0) */
    std::vector<uint64_t> reserved_flags;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/invalid/reserved_upper_flags.bin"), reserved_flags));
    avid_packet_validator_t val_res;
    avid_validator_init(&val_res);
    bool accepted_reserved = avid_validate_packet(&val_res, reserved_flags[0], &pkt);

    ctx.log("flags=0x00 rejected=" + std::string(!ok_zero_flags ? "true" : "false") +
            "; reserved flags=0xF1 accepted=" + std::string(accepted_reserved ? "true (DEFECT)" : "false"));

    ctx.record().expected["flags_0x00"] = "REJECT (false), invalid_flag_count++";
    ctx.record().expected["flags_0xF1_reserved_bits"] = "REJECT (false), invalid_flag_count++";
    ctx.record().actual["flags_0x00"] = !ok_zero_flags ? "REJECT (false)" : "ACCEPT (true)";
    ctx.record().actual["flags_0xF1_reserved_bits"] = accepted_reserved ? "ACCEPT (true) — reserved bits [7:4] ignored" : "REJECT (false)";

    if (accepted_reserved) {
        ctx.record_failure("avid_validate_packet accepted packet with reserved flag bits 0xF0 set (flags=0xF1)");
        return ctx.finalize(TestStatus::FAIL, "reserved_upper_flag_bits_accepted_without_validation");
    }
    return ctx.finalize(TestStatus::PASS);
}
