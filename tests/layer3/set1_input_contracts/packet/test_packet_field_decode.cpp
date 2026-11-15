#include "test_assertions.h"
#include "test_common.h"
#include "ingress/packet_validator.h"

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-005");
    TestContext ctx(opts, opts.test_id, "syndrome_bit_range_and_field_decode",
                    "software/src/ingress/packet_validator.cpp",
                    "avid_pack_packet_u64, avid_unpack_packet_u64");

    ctx.record().test_description =
        "Syndrome bit range — verify all 256 valid (s_x in 0..15, s_z in 0..15) combinations and inspect out-of-range s_x/s_z >= 16 handling";
    ctx.record().input_contract = "4-bit s_x in [0,15] and 4-bit s_z in [0,15]; out-of-range uint8_t >= 16 must be rejected rather than silently aliased";
    ctx.record().independent_oracle = "Exhaustive 256-state bitfield oracle + out-of-domain injection (16, 255)";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/packet_001.bin";

    int valid_pairs_ok = 0;
    for (uint16_t sx = 0; sx < 16; ++sx) {
        for (uint16_t sz = 0; sz < 16; ++sz) {
            uint32_t seq = static_cast<uint32_t>(sx * 16 + sz);
            uint16_t ep = static_cast<uint16_t>(100 + sx);
            uint64_t w = avid_pack_packet_u64(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz),
                                              seq, ep, AVID_PKT_FLAG_VALID);
            avid_syndrome_packet_t out{};
            avid_unpack_packet_u64(w, &out);
            if (out.s_x == sx && out.s_z == sz && out.seq_id == seq && out.epoch_id == ep) {
                valid_pairs_ok++;
            }
        }
    }
    ASSERT_EQ(valid_pairs_ok, 256);

    /* Out-of-range uint8_t syndrome inputs: s_x=16 (0x10), s_z=255 (0xFF) */
    uint64_t w_oob = avid_pack_packet_u64(16u, 255u, 0u, 1u, AVID_PKT_FLAG_VALID);
    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t out_oob{};
    bool oob_accepted = avid_validate_packet(&val, w_oob, &out_oob);

    ctx.log("All 256 valid (s_x, s_z) pairs round-tripped Losslessly.");
    ctx.log("Passing out-of-range s_x=16, s_z=255 to avid_pack_packet_u64 produced s_x=" +
            std::to_string(out_oob.s_x) + ", s_z=" + std::to_string(out_oob.s_z) +
            " via silent '& 0x0F' masking and avid_validate_packet returned " +
            (oob_accepted ? "true" : "false") + ".");

    ctx.record().expected["valid_256_domain"] = "256/256 exact round-trip";
    ctx.record().expected["out_of_range_sx16_sz255"] = "Explicit error / rejection before silent 4-bit aliasing";
    ctx.record().actual["valid_256_domain"] = std::to_string(valid_pairs_ok) + "/256 exact round-trip";
    ctx.record().actual["out_of_range_sx16_sz255"] =
        "Silently masked to s_x=" + std::to_string(out_oob.s_x) + ", s_z=" + std::to_string(out_oob.s_z) + " and accepted";

    if (oob_accepted && out_oob.s_x == 0u) {
        ctx.record_failure("avid_pack_packet_u64 silently aliases out-of-range s_x=16 to s_x=0 (no-error syndrome)");
        return ctx.finalize(TestStatus::FAIL, "out_of_range_syndrome_silently_masked_to_valid_4bit_domain");
    }
    return ctx.finalize(TestStatus::PASS);
}
