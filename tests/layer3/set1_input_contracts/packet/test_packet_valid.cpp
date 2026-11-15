#include "test_assertions.h"
#include "test_common.h"
#include "binary_packet_builder.h"
#include "fixture_loader.h"
#include "ingress/packet_validator.h"
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-001");
    TestContext ctx(opts, opts.test_id, "valid_packet_ingress",
                    "software/src/ingress/packet_validator.cpp",
                    "avid_pack_packet_u64, avid_unpack_packet_u64, avid_validate_packet");

    ctx.record().test_description =
        "Valid packet ingress — pack (s_x=5, s_z=11, seq=0, epoch=3, flags=AVID_PKT_FLAG_VALID), unpack & validate, plus 64-packet sequential stream";
    ctx.record().input_contract = "64-bit word with VALID=1, PARITY_ERR=0, monotonic seq_id";
    ctx.record().independent_oracle = "binary_packet_builder::build_raw_packet_u64_oracle & packet_001_expected.json";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/packet_001.bin";

    std::vector<uint64_t> words;
    bool loaded = load_u64_stream_fixture(ctx.fixture_path("packets/valid/packet_001.bin"), words);
    ASSERT_TRUE(loaded);
    ASSERT_EQ(words.size(), static_cast<size_t>(1));

    uint64_t oracle_word = build_raw_packet_u64_oracle(5, 11, AVID_PKT_FLAG_VALID, 3, 0);
    uint64_t prod_word = avid_pack_packet_u64(5, 11, 0, 3, AVID_PKT_FLAG_VALID);
    ASSERT_EQ(prod_word, oracle_word);
    if (!words.empty()) {
        ASSERT_EQ(words[0], oracle_word);
    }

    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t decoded{};
    bool ok = avid_validate_packet(&val, prod_word, &decoded);

    ASSERT_TRUE(ok);
    ASSERT_EQ(static_cast<int>(decoded.s_x), 5);
    ASSERT_EQ(static_cast<int>(decoded.s_z), 11);
    ASSERT_EQ(static_cast<int>(decoded.flags), static_cast<int>(AVID_PKT_FLAG_VALID));
    ASSERT_EQ(static_cast<int>(decoded.epoch_id), 3);
    ASSERT_EQ(decoded.seq_id, 0u);
    ASSERT_EQ(val.valid_count, 1ULL);
    ASSERT_EQ(val.seq_gap_count, 0ULL);
    ASSERT_EQ(val.invalid_flag_count, 0ULL);

    /* Validate 64-packet sequential fixture stream */
    std::vector<uint64_t> stream64;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/valid/burst_64_seq.bin"), stream64));
    ASSERT_EQ(stream64.size(), static_cast<size_t>(64));

    avid_packet_validator_t val_stream;
    avid_validator_init(&val_stream);
    for (size_t i = 0; i < stream64.size(); ++i) {
        avid_syndrome_packet_t p{};
        bool step_ok = avid_validate_packet(&val_stream, stream64[i], &p);
        ASSERT_TRUE(step_ok);
        ASSERT_EQ(p.seq_id, static_cast<uint32_t>(i));
    }
    ASSERT_EQ(val_stream.valid_count, 64ULL);

    ctx.record().expected["validation_status"] = "ACCEPT (true), s_x=5, s_z=11, epoch_id=3, seq_id=0, valid_count=1";
    ctx.record().actual["validation_status"] = ok ? "ACCEPT (true)" : "REJECT (false)";
    ctx.record().actual["decoded_fields"] =
        "s_x=" + std::to_string(decoded.s_x) + ", s_z=" + std::to_string(decoded.s_z) +
        ", epoch_id=" + std::to_string(decoded.epoch_id) + ", seq_id=" + std::to_string(decoded.seq_id);
    ctx.record().actual["stream64_valid_count"] = std::to_string(val_stream.valid_count);

    return ctx.finalize(TestStatus::PASS);
}
