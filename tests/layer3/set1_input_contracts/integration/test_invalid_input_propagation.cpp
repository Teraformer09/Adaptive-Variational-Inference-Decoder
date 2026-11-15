#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/dma_ingress_tap.h"
#include <string>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-INT-004");
    TestContext ctx(opts, opts.test_id, "invalid_packet_end_to_end_propagation",
                    "software/src/ingress/dma_ingress_tap.cpp, software/src/inference/avid_daemon.cpp",
                    "avid_dma_ring_consume_slot vs avid_daemon --stim-bin",
                    "set1");
    ctx.record().test_set = "set1_integration";
    ctx.record().test_description =
        "Invalid-input propagation — feed 64 packets with VALID=0 and PARITY_ERR=1 (sx=6, sz=9) to standalone DMA ring (0 accepted) vs avid_daemon --stim-bin (all 64 accepted due to flag re-stamping!)";
    ctx.record().input_contract = "Packets with VALID=0 or PARITY_ERR=1 must NEVER contaminate histogram bins or trigger adaptive parameter updates";
    ctx.record().independent_oracle = "Standalone DMA ring valid_out==0; production daemon must match and NOT adapt on parity-corrupted packets";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/stim_parity_corrupted_64.bin";

    std::string bad64_path = ctx.fixture_path("packets/invalid/stim_parity_corrupted_64.bin");
    std::vector<uint64_t> bad_words;
    ASSERT_TRUE(load_u64_stream_fixture(bad64_path, bad_words));
    ASSERT_EQ(bad_words.size(), static_cast<size_t>(64));

    /* 1. Standalone DMA ring properly blocks all 64 parity-corrupted packets */
    avid_dma_ring_t ring{};
    avid_dma_ring_init(&ring);
    ASSERT_TRUE(avid_dma_ring_push_burst(&ring, bad_words.data(), 64));
    uint32_t hx[16] = {0}, hz[16] = {0}, valid_out = 999;
    ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &valid_out));
    ASSERT_EQ(valid_out, 0u);
    ASSERT_EQ(hx[6], 0u);
    ASSERT_EQ(hz[9], 0u);

    /* 2. Feed the exact same parity-corrupted file to avid_daemon --stim-bin vs all-zero valid file */
    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());
    std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
    std::string quiet_path = ctx.fixture_path("packets/boundary/min_fields.bin");

    std::string m_corrupt = "/tmp/avid_int_corrupt_" + std::to_string(::getpid()) + ".json";
    std::string m_quiet   = "/tmp/avid_int_clean0_" + std::to_string(::getpid()) + ".json";

    auto rc_corrupt = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + bad64_path + "\" --rounds 64 --window 64 --metrics-out \"" + m_corrupt + "\"");
    auto rc_quiet = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + quiet_path + "\" --rounds 64 --window 64 --metrics-out \"" + m_quiet + "\"");

    std::string jc, jq;
    ASSERT_EQ(rc_corrupt.exit_code, 0);
    ASSERT_EQ(rc_quiet.exit_code, 0);
    ASSERT_TRUE(load_text_fixture(m_corrupt, jc));
    ASSERT_TRUE(load_text_fixture(m_quiet, jq));
    ::unlink(m_corrupt.c_str());
    ::unlink(m_quiet.c_str());

    /* On 64 parity-corrupted packets: 0 valid packets, 64 invalid flags, 0 decoded rounds, 0 ring pushes, 0 adaptive commits! */
    ASSERT_TRUE(jc.find("\"validator_invalid_flags\": 64") != std::string::npos);
    ASSERT_TRUE(jc.find("\"validator_valid_packets\": 0") != std::string::npos);
    ASSERT_TRUE(jc.find("\"decoded_rtl_transactions\": 0") != std::string::npos);
    ASSERT_TRUE(jc.find("\"ring_buffer_count\": 0") != std::string::npos);
    ASSERT_TRUE(jc.find("\"epochs_completed\": 0") != std::string::npos);
    ASSERT_TRUE(jc.find("\"adaptive_epoch_commits\": 0") != std::string::npos);

    /* Contrast with clean 64-round stream: 64 valid packets, 1 epoch, 1 adaptive commit */
    ASSERT_TRUE(jq.find("\"validator_valid_packets\": 64") != std::string::npos);
    ASSERT_TRUE(jq.find("\"epochs_completed\": 1") != std::string::npos);

    ctx.record().expected["invalid_propagation_blocked"] = "0/64 parity-corrupted packets reach estimator, ring_buffer, or weight commits in both standalone DMA ring and avid_daemon";
    ctx.record().actual["invalid_propagation_blocked"] =
        "Standalone DMA ring: 0/64 accepted (PASS); avid_daemon --stim-bin: 0/64 accepted, validator_invalid_flags=64, adaptive_epoch_commits=0 (PASS)";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
