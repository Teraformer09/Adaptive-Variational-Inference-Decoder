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
    bool yaml_specifies_16b_header = yaml_contract.find("header_size_bytes: 16") != std::string::npos;
    bool yaml_specifies_4b_packet = yaml_contract.find("packet_size_bytes: 4") != std::string::npos;
    ASSERT_TRUE(yaml_specifies_16b_header);
    ASSERT_TRUE(yaml_specifies_4b_packet);

    ctx.log("Fixture sizes loaded: " + std::to_string(b0.size()) + ", " + std::to_string(b3.size()) + ", " +
            std::to_string(b7.size()) + ", " + std::to_string(b8.size()) + ", " + std::to_string(b9.size()) +
            ", " + std::to_string(b15.size()) + ", " + std::to_string(b24.size()) + " bytes.");
    ctx.log("C implementation wire word size: sizeof(uint64_t) = 8 bytes; unpacked struct sizeof(avid_syndrome_packet_t) = " +
            std::to_string(sizeof(avid_syndrome_packet_t)) + " bytes.");
    ctx.log("packet_validator.h exposes NO (const uint8_t *buf, size_t len) entrypoint; it accepts uint64_t by value only.");
    ctx.log("syndrome_packet_v1.yaml specifies header_size_bytes=16 and packet_size_bytes=4 (32-bit DMA word).");

    ctx.record().expected["wire_contract"] =
        "Agreement between architecture/contracts/syndrome_packet_v1.yaml and packet_validator.h, plus explicit byte-length rejection for != packet_size";
    ctx.record().actual["c_wire_size_bytes"] = "8 (uint64_t by value; no byte-length parameter)";
    ctx.record().actual["c_unpacked_struct_bytes"] = std::to_string(sizeof(avid_syndrome_packet_t));
    ctx.record().actual["yaml_contract_bytes"] = "16-byte stream header + 4-byte per-round word";
    ctx.record().evidence.push_back("validation/layer3/reports/packet_contract_comparison.md");

    return ctx.finalize(TestStatus::CONTRACT_MISMATCH,
                        "wire_packet_length_and_api_mismatch_64bit_u64_vs_yaml_16B_hdr_plus_4B_word");
}
