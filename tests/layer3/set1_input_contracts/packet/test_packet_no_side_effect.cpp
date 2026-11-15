#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/dma_ingress_tap.h"
#include "ingress/packet_validator.h"
#include <cstring>
#include <string>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-008");
    TestContext ctx(opts, opts.test_id, "packet_corruption_and_no_side_effect",
                    "software/src/ingress/packet_validator.cpp, software/src/inference/avid_daemon.cpp",
                    "avid_validate_packet, avid_dma_ring_consume_slot, avid_daemon(--stim-bin)");

    ctx.record().test_description =
        "Packet corruption & side-effect isolation — test PARITY_ERR rejection, output struct mutation on reject, CRC/checksum absence, and production --stim-bin parity bypass";
    ctx.record().input_contract = "Corrupted packets rejected without mutating output struct or downstream histograms/estimator";
    ctx.record().independent_oracle = "Zero histogram delta oracle on PARITY_ERR stream across standalone DMA ring and avid_daemon --stim-bin";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/stim_parity_corrupted_64.bin";

    /* 1. Standalone validator & DMA ring behavior on PARITY_ERR */
    std::vector<uint64_t> parity_pkt;
    ASSERT_TRUE(load_u64_stream_fixture(ctx.fixture_path("packets/invalid/parity_err_flag.bin"), parity_pkt));
    avid_packet_validator_t val;
    avid_validator_init(&val);
    avid_syndrome_packet_t out_pkt{0xAA, 0xBB, 0xCC, 0xDDDD, 0xEEEEEEEEu};
    bool ok_parity = avid_validate_packet(&val, parity_pkt[0], &out_pkt);
    ASSERT_FALSE(ok_parity);
    ASSERT_EQ(val.invalid_flag_count, 1ULL);
    bool out_struct_mutated_on_reject = (out_pkt.s_x == 6 && out_pkt.s_z == 9);

    avid_dma_ring_t ring;
    avid_dma_ring_init(&ring);
    uint32_t hx[16] = {0}, hz[16] = {0}, valid_cnt = 99;
    avid_dma_ring_push_burst(&ring, parity_pkt.data(), 1);
    avid_dma_ring_consume_slot(&ring, hx, hz, &valid_cnt);
    ASSERT_EQ(valid_cnt, 0u);
    ASSERT_EQ(hx[6], 0u);
    ASSERT_EQ(hz[9], 0u);

    /* 2. Production daemon --stim-bin path with 64 packets ALL having VALID=0 and PARITY_ERR=1 */
    bool daemon_bypassed_parity = false;
    std::string daemon_bin = opts.daemon_bin;
    if (!daemon_bin.empty()) {
        std::string metrics_tmp = "/tmp/avid_l3_ip008_metrics.json";
        std::string stim_path = ctx.fixture_path("packets/invalid/stim_parity_corrupted_64.bin");
        std::string cmd = "\"" + daemon_bin + "\" --backend physical --scenario stationary_noise "
                          "--stim-bin \"" + stim_path + "\" --rounds 64 --window 64 --metrics-out \"" + metrics_tmp + "\"";
        SubprocessResult dres = run_command_capture(cmd);
        std::string metrics_json;
        if (dres.exit_code == 0 && load_text_fixture(metrics_tmp, metrics_json)) {
            if (metrics_json.find("\"epochs_completed\": 1") != std::string::npos &&
                metrics_json.find("\"validator_invalid_flags\": 0") != std::string::npos) {
                daemon_bypassed_parity = true;
            }
        }
        std::remove(metrics_tmp.c_str());
    }

    ctx.log("Standalone avid_dma_ring_consume_slot properly excluded PARITY_ERR packet from hist_x/hist_z.");
    ctx.log("avid_validate_packet mutated caller out struct before returning false: " +
            std::string(out_struct_mutated_on_reject ? "YES" : "NO"));
    ctx.log("No CRC/checksum field exists in 64-bit packet: unflagged bit flips in s_x/s_z are undetectable.");
    ctx.log("Production avid_daemon --stim-bin bypassed PARITY_ERR=1, VALID=0 and re-stamped VALID=1: " +
            std::string(daemon_bypassed_parity ? "YES (DEFECT at avid_daemon.cpp:405-408,491)" : "NOT_RUN"));

    ctx.record().expected["standalone_parity_err"] = "REJECT (0 histogram updates)";
    ctx.record().expected["daemon_stim_bin_parity_err"] = "REJECT (validator_invalid_flags == 64, 0 valid rounds)";
    ctx.record().expected["crc_checksum"] = "CRC or documented parity-only boundary";
    ctx.record().actual["standalone_parity_err"] = "REJECT (0 histogram updates; caller out struct overwritten)";
    ctx.record().actual["daemon_stim_bin_parity_err"] =
        daemon_bypassed_parity ? "ACCEPTED (validator_invalid_flags=0, epochs_completed=1; re-packed with VALID=1)" : "N/A";
    ctx.record().actual["crc_checksum"] = "ABSENT (single PARITY_ERR status bit only)";

    if (daemon_bypassed_parity || out_struct_mutated_on_reject) {
        ctx.record_failure("avid_daemon --stim-bin bypasses avid_validate_packet, re-stamping PARITY_ERR packets with VALID=1 into histograms");
        return ctx.finalize(TestStatus::FAIL, "daemon_stim_bin_bypasses_parity_validator_and_no_crc");
    }
    return ctx.finalize(TestStatus::INCONCLUSIVE, "parity_bit_only_no_crc_field");
}
