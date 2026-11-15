#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <cstdio>
#include <string>
#include <unistd.h>

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

    std::string trunc_bin = ctx.fixture_path("packets/invalid/truncated_3B.bin");
    auto r_trunc = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode static --stim-bin \"" + trunc_bin + "\" --rounds 64 --window 64");
    ASSERT_EQ(r_trunc.exit_code, 2);

    /* Also verify raw flags in --stim-bin are preserved into g_dma_ring without re-stamping AVID_PKT_FLAG_VALID */
    std::string bad_flags_bin = ctx.fixture_path("packets/invalid/stim_parity_corrupted_64.bin");
    std::string m_path = "/tmp/avid_l3_int001_" + std::to_string(::getpid()) + ".json";
    auto r_flags = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode static --stim-bin \"" + bad_flags_bin +
        "\" --rounds 64 --window 64 --metrics-out \"" + m_path + "\"");
    ASSERT_EQ(r_flags.exit_code, 0);
    std::string mjson;
    ASSERT_TRUE(load_text_fixture(m_path, mjson));
    ::unlink(m_path.c_str());
    ASSERT_TRUE(mjson.find("\"validator_invalid_flags\": 64") != std::string::npos);
    ASSERT_TRUE(mjson.find("\"validator_valid_packets\": 0") != std::string::npos);

    ctx.record().expected["daemon_ingress_validation"] = "Reject truncated 3B --stim-bin (rc=2) and preserve/validate raw packet flags & seq_id";
    ctx.record().actual["daemon_ingress_validation"] =
        "truncated_3B --stim-bin rejected with rc=" + std::to_string(r_trunc.exit_code) +
        "; raw --stim-bin flags validated end-to-end (validator_invalid_flags=64, validator_valid_packets=0)";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
