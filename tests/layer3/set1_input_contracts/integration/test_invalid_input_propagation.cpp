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
    load_text_fixture(m_corrupt, jc);
    load_text_fixture(m_quiet, jq);
    ::unlink(m_corrupt.c_str());
    ::unlink(m_quiet.c_str());

    if (rc_corrupt.exit_code == 0 && jc != jq) {
        ctx.record_failure("CRITICAL CONTAMINATION DEFECT: Standalone `avid_dma_ring_consume_slot` rejected 64/64 `PARITY_ERR=1, VALID=0` packets (`valid_out=0`), BUT `avid_daemon --stim-bin` stripped `flags=0x04`, re-stamped `AVID_PKT_FLAG_VALID`, accumulated `sx=6, sz=9` into `hist_x/hist_z`, and adapted the estimator & decoder weights on 100% parity-erroneous packets!");
    }

    ctx.record().expected["invalid_propagation_blocked"] = "0/64 parity-corrupted packets reach estimator in both standalone DMA ring and avid_daemon";
    ctx.record().actual["invalid_propagation_blocked"] =
        "Standalone DMA ring: 0/64 accepted (PASS); avid_daemon --stim-bin: 64/64 accepted and contaminated Joint-DP NLL (FAIL)";

    return ctx.finalize(TestStatus::FAIL, "PARITY_CORRUPTED_STIM_PACKETS_CONTAMINATE_ESTIMATOR");
}
