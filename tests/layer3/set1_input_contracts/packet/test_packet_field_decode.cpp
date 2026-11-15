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
            uint64_t w = 0;
            bool packed_ok = avid_pack_packet_u64_checked(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz),
                                                          seq, ep, AVID_PKT_FLAG_VALID, &w);
            avid_syndrome_packet_t out{};
            avid_unpack_packet_u64(w, &out);
            if (packed_ok && out.s_x == sx && out.s_z == sz && out.seq_id == seq && out.epoch_id == ep) {
                valid_pairs_ok++;
            }
        }
    }
    ASSERT_EQ(valid_pairs_ok, 256);

    /* Out-of-range uint8_t syndrome & reserved flag inputs must be explicitly rejected by checked packer */
    uint64_t w_oob = 0xDEADBEEFCAFEBABEULL;
    ASSERT_FALSE(avid_pack_packet_u64_checked(16u, 0u, 0u, 1u, AVID_PKT_FLAG_VALID, &w_oob));
    ASSERT_FALSE(avid_pack_packet_u64_checked(0u, 255u, 0u, 1u, AVID_PKT_FLAG_VALID, &w_oob));
    ASSERT_FALSE(avid_pack_packet_u64_checked(3u, 5u, 0u, 1u, 0xF1u, &w_oob));
    ASSERT_FALSE(avid_pack_packet_u64_checked(3u, 5u, 0u, 1u, AVID_PKT_FLAG_VALID, nullptr));
    ASSERT_EQ(w_oob, 0xDEADBEEFCAFEBABEULL);

    ctx.log("All 256 valid (s_x, s_z) pairs round-tripped losslessly; avid_pack_packet_u64_checked rejected s_x=16, s_z=255, flags=0xF1, and nullptr without mutating out_raw_word.");
    ctx.record().expected["valid_256_domain"] = "256/256 exact round-trip";
    ctx.record().expected["out_of_range_sx16_sz255"] = "Explicit false return from avid_pack_packet_u64_checked without silent aliasing";
    ctx.record().actual["valid_256_domain"] = std::to_string(valid_pairs_ok) + "/256 exact round-trip";
    ctx.record().actual["out_of_range_sx16_sz255"] = "Rejected cleanly (false, out_raw_word untouched)";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
