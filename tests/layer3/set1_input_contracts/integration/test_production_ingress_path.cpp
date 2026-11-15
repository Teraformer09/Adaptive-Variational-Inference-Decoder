#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <string>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-INT-001");
    TestContext ctx(opts, opts.test_id, "production_daemon_ingress_path_audit",
                    "software/src/inference/avid_daemon.cpp",
                    "main (--stim-bin fread -> avid_unpack_packet_u64 -> avid_pack_packet_u64 -> avid_dma_ring_push_burst)",
                    "set1");
    ctx.record().test_set = "set1_integration";
    ctx.record().test_description =
        "Production ingress path connectivity — verify actual path taken by packets entering avid_daemon via --stim-bin vs avid_validate_packet";
    ctx.record().input_contract = "Production daemon must validate external --stim-bin packets through avid_validate_packet rather than overwriting flags/seq_id";
    ctx.record().independent_oracle = "Direct source + runtime verification on avid_daemon_under_test with truncated 3-byte and invalid-flag --stim-bin files";
    ctx.record().fixture = "tests/layer3/fixtures/packets/invalid/truncated_3B.bin";

    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());

    /* Feed a 3-byte truncated --stim-bin file to avid_daemon_under_test:
     * avid_daemon.cpp:399-407 calls fread(&raw, sizeof(uint64_t), 1, stim_fp); when fread != 1 on 3-byte file,
     * it seeks to 0, reads 0 items again, leaves sx=0, sz=0, and runs 64 rounds returning exit code 0! */
    std::string trunc_bin = ctx.fixture_path("packets/invalid/truncated_3B.bin");
    auto r_trunc = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode static --stim-bin \"" + trunc_bin + "\" --rounds 64 --window 64");

    std::string daemon_src;
    ASSERT_TRUE(load_text_fixture(opts.repo_root + "/software/src/inference/avid_daemon.cpp", daemon_src));
    bool calls_unpack = (daemon_src.find("avid_unpack_packet_u64(raw, &pkt)") != std::string::npos);
    bool repacks_valid = (daemon_src.find("avid_pack_packet_u64(sx, sz, seq_gen++, (uint16_t)hw_epoch, AVID_PKT_FLAG_VALID)") != std::string::npos);

    if (r_trunc.exit_code == 0 || (calls_unpack && repacks_valid)) {
        ctx.record_failure("PRODUCTION INGRESS BYPASS: `avid_daemon.cpp:398-408,491` accepts a 3-byte truncated `--stim-bin` with rc=0 (synthesizing sx=0, sz=0 on short `fread`) AND unpacks `--stim-bin` words via `avid_unpack_packet_u64` without calling `avid_validate_packet`, then re-packs every word with fresh `AVID_PKT_FLAG_VALID` and `seq_gen++` before `avid_dma_ring_push_burst`!");
    }

    ctx.record().expected["daemon_ingress_validation"] = "Reject truncated 3B --stim-bin and preserve/validate raw packet flags & seq_id";
    ctx.record().actual["daemon_ingress_validation"] =
        "truncated_3B --stim-bin exited rc=" + std::to_string(r_trunc.exit_code) +
        "; unpacks raw without avid_validate_packet and re-stamps AVID_PKT_FLAG_VALID";

    return ctx.finalize(TestStatus::FAIL, "DAEMON_STIM_BIN_BYPASSES_PACKET_VALIDATOR");
}
