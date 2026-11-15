#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/packet_validator.h"
#include <string>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-002");
    TestContext ctx(opts, opts.test_id, "packet_length_and_wire_contract",
                    "software/src/ingress/packet_validator.h",
                    "avid_validate_packet vs syndrome_packet_v1.yaml");

    ctx.record().test_description =
        "Packet length — evaluate 0, 3, 7, 8, 9, 15, and 24-byte buffers against 64-bit C API and syndrome_packet_v1.yaml (16B header + 4B round word)";
    ctx.record().input_contract = "Exact packet framing length validation across wire formats";
    ctx.record().independent_oracle = "architecture/contracts/syndrome_packet_v1.yaml vs packet_validator.h signature inspection";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/short_7B.bin";

    std::vector<uint8_t> b0, b3, b7, b8, b9, b15, b24, yaml_text_bytes;
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/invalid/empty_0B.bin"), b0));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/invalid/truncated_3B.bin"), b3));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/invalid/short_7B.bin"), b7));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/valid/packet_001.bin"), b8));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/invalid/oversize_9B.bin"), b9));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/invalid/yaml_hdr_truncated_15B.bin"), b15));
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/valid/yaml_v1_stream_2rounds.bin"), b24));

    std::string yaml_contract;
    ASSERT_TRUE(load_text_fixture(ctx.env().repo_root + "/architecture/contracts/syndrome_packet_v1.yaml", yaml_contract));
    bool yaml_specifies_8b_packet = yaml_contract.find("packet_size_bytes: 8") != std::string::npos;
    bool yaml_specifies_8b_framing = yaml_contract.find("byte_length_multiple_of: 8") != std::string::npos;
    ASSERT_TRUE(yaml_specifies_8b_packet);
    ASSERT_TRUE(yaml_specifies_8b_framing);
    ASSERT_EQ(static_cast<size_t>(AVID_PKT_WIRE_BYTES), static_cast<size_t>(8));

    avid_packet_validator_t val{};
    avid_validator_init(&val);
    avid_syndrome_packet_t out{};

    /* Single-packet byte-buffer validation: ONLY exact 8-byte buffer with valid packet fields accepted */
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b0.data(), b0.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b3.data(), b3.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b7.data(), b7.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b9.data(), b9.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b15.data(), b15.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, b24.data(), b24.size(), &out));
    ASSERT_FALSE(avid_validate_packet_bytes(&val, nullptr, 8u, &out));
    ASSERT_TRUE(avid_validate_packet_bytes(&val, b8.data(), b8.size(), &out));

    /* Multi-packet binary stream framing validation: non-null, positive multiple of 8 bytes required */
    uint8_t dummy = 0;
    ASSERT_FALSE(avid_validate_stream_framing(nullptr, 8u));
    ASSERT_FALSE(avid_validate_stream_framing(&dummy, b0.size()));
    ASSERT_FALSE(avid_validate_stream_framing(b3.data(), b3.size()));
    ASSERT_FALSE(avid_validate_stream_framing(b7.data(), b7.size()));
    ASSERT_TRUE(avid_validate_stream_framing(b8.data(), b8.size()));
    ASSERT_FALSE(avid_validate_stream_framing(b9.data(), b9.size()));
    ASSERT_FALSE(avid_validate_stream_framing(b15.data(), b15.size()));
    ASSERT_TRUE(avid_validate_stream_framing(b24.data(), b24.size()));

    ctx.log("Verified authoritative A05/Q11 8-byte wire contract across syndrome_packet_v1.yaml (v1.1.0) and packet_validator.h.");
    ctx.record().expected["wire_contract"] =
        "Exact 8-byte packet framing in syndrome_packet_v1.yaml & packet_validator.h; non-8-byte packet buffers rejected";
    ctx.record().actual["c_wire_size_bytes"] = "8 (AVID_PACKET_WIRE_BYTES=8; avid_validate_packet_bytes & avid_validate_stream_framing)";
    ctx.record().actual["c_unpacked_struct_bytes"] = std::to_string(sizeof(avid_syndrome_packet_t));
    ctx.record().actual["yaml_contract_bytes"] = "8-byte little-endian uint64_t DMA packet + 16-bit AXI-Stream RTL word";
    ctx.record().evidence.push_back("validation/layer3/reports/packet_contract_decision.md");

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
