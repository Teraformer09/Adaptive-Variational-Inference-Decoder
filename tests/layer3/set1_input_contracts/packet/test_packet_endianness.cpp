#include "test_assertions.h"
#include "test_common.h"
#include "binary_packet_builder.h"
#include "fixture_loader.h"
#include "hal/avid_hal.h"
#include "ingress/packet_validator.h"
#include <cstring>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-003");
    TestContext ctx(opts, opts.test_id, "packet_version_and_endianness",
                    "software/src/ingress/packet_validator.cpp, software/src/hal/simulation_backend.cpp",
                    "avid_pack_packet_u64, avid_unpack_packet_u64, send_bridge_request");

    ctx.record().test_description =
        "Packet version & endianness — verify little-endian byte order, HAL protocol_version==1 check, and YAML magic 0xAAEC0002 / version=1 stream header support";
    ctx.record().input_contract = "Versioned little-endian packet framing (protocol_version=1, magic=0xAAEC0002/0xAAEC0001)";
    ctx.record().independent_oracle = "Byte-level little-endian oracle + AF_UNIX mock peer + syndrome_packet_v1.yaml";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/yaml_v1_stream_2rounds.bin";

    /* 1. Verify exact little-endian byte layout of 64-bit word */
    uint64_t w = avid_pack_packet_u64(0x5u, 0xBu, 0x01020304u, 0xABCDu, 0x09u);
    uint8_t raw_le[8];
    std::memcpy(raw_le, &w, 8);
    ASSERT_EQ(static_cast<int>(raw_le[0]), 0xB5);
    ASSERT_EQ(static_cast<int>(raw_le[1]), 0x09);
    ASSERT_EQ(static_cast<int>(raw_le[2]), 0xCD);
    ASSERT_EQ(static_cast<int>(raw_le[3]), 0xAB);
    ASSERT_EQ(static_cast<int>(raw_le[4]), 0x04);
    ASSERT_EQ(static_cast<int>(raw_le[5]), 0x03);
    ASSERT_EQ(static_cast<int>(raw_le[6]), 0x02);
    ASSERT_EQ(static_cast<int>(raw_le[7]), 0x01);

    /* 2. Verify HAL simulation_backend protocol_version enforcement via socketpair */
    int sv[2] = {-1, -1};
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
    std::thread mock_server([fd = sv[1]]() {
        char buf[512];
        /* Request 1 (tx 1000): reply with valid protocol_version=1 */
        ssize_t n1 = read(fd, buf, sizeof(buf));
        (void)n1;
        const char *ok_resp = "{\"protocol_version\":1,\"transaction_id\":1000,\"status\":\"OK\",\"rdata\":\"0x00000001\",\"accepted_cycle\":2}\n";
        ssize_t w1 = write(fd, ok_resp, std::strlen(ok_resp));
        (void)w1;
        /* Request 2 (tx 1001): reply with unsupported protocol_version=99 */
        ssize_t n2 = read(fd, buf, sizeof(buf));
        (void)n2;
        const char *bad_ver = "{\"protocol_version\":99,\"transaction_id\":1001,\"status\":\"OK\",\"rdata\":\"0x00000001\",\"accepted_cycle\":4}\n";
        ssize_t w2 = write(fd, bad_ver, std::strlen(bad_ver));
        (void)w2;
        close(fd);
    });

    avid_hal_t hal{};
    hal.backend = AVID_HAL_BACKEND_SIMULATION;
    hal.sock_fd = sv[0];
    hal.next_tx_id = 1000;
    hal.timeout_ms = 1000;
    uint32_t rdata = 0;
    uint64_t cyc = 0;
    int rc_v1 = avid_hal_mmio_read(&hal, 0x050u, &rdata, &cyc);
    int rc_v99 = avid_hal_mmio_read(&hal, 0x050u, &rdata, &cyc);
    close(sv[0]);
    mock_server.join();

    ASSERT_EQ(rc_v1, 0);
    ASSERT_EQ(rc_v99, -1);
    ASSERT_EQ(hal.stats.rejected_transactions, 1ULL);

    /* 3. Inspect YAML stream header (0xAAEC0002, version=1) handling in packet_validator.cpp */
    std::vector<uint8_t> yaml_bytes;
    ASSERT_TRUE(load_binary_fixture(ctx.fixture_path("packets/valid/yaml_v1_stream_2rounds.bin"), yaml_bytes));
    uint64_t first8 = le_bytes_to_u64(yaml_bytes, 0);
    avid_syndrome_packet_t misdecoded{};
    avid_unpack_packet_u64(first8, &misdecoded);

    ctx.log("HAL JSON RPC protocol_version=1 accepted (rc=0) and protocol_version=99 rejected (rc=-1, rejected_transactions=1).");
    ctx.log("However, packet_validator.cpp has NO version field and NO parser for syndrome_packet_v1.yaml magic 0xAAEC0002 / 0xAAEC0001.");
    ctx.log("Feeding YAML header bytes [0..7] into avid_unpack_packet_u64 misinterprets magic 0xAAEC0002 as s_x=" +
            std::to_string(misdecoded.s_x) + ", epoch_id=" + std::to_string(misdecoded.epoch_id) +
            ", seq_id=" + std::to_string(misdecoded.seq_id));

    ctx.record().expected["version_validation"] =
        "Both HAL RPC protocol_version==1 and syndrome_packet_v1.yaml magic (0xAAEC0002/0xAAEC0001) + version==1 validated";
    ctx.record().actual["hal_rpc_version_check"] = "PASS (v1 accepted, v99 rejected)";
    ctx.record().actual["syndrome_packet_version_check"] =
        "MISSING in packet_validator.cpp (no version field in 64-bit word; YAML 0xAAEC0002 header not parsed)";
    ctx.record().evidence.push_back("validation/layer3/reports/packet_contract_comparison.md");

    return ctx.finalize(TestStatus::CONTRACT_MISMATCH,
                        "missing_syndrome_stream_version_and_magic_header_parser");
}
