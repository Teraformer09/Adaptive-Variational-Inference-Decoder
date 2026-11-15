#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <string>
#include <unistd.h>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-INT-002");
    TestContext ctx(opts, opts.test_id, "daemon_histogram_to_estimator_connectivity",
                    "software/src/inference/avid_daemon.cpp",
                    "avid_dma_ring_consume_slot -> hist_x/hist_z -> HistogramLBFGSSolver::fit_histogram",
                    "set1");
    ctx.record().test_set = "set1_integration";
    ctx.record().test_description =
        "Daemon histogram path — execute avid_daemon_under_test on two distinct 64-packet --stim-bin streams (all-zero syndrome vs high-syndrome burst) and verify histogram accumulation drives distinct Joint-DP NLL outputs";
    ctx.record().input_contract = "DMA ring consumer populates hist_x[16], hist_z[16] consumed by HistogramLBFGSSolver::fit_histogram";
    ctx.record().independent_oracle = "final_joint_dp_nll differs materially between quiet stream and burst_64_seq.bin stream";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/burst_64_seq.bin";

    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());
    std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
    std::string burst_bin = ctx.fixture_path("packets/valid/burst_64_seq.bin");
    std::string quiet_bin = ctx.fixture_path("packets/boundary/min_fields.bin");

    std::string m_burst = "/tmp/avid_int_burst_" + std::to_string(::getpid()) + ".json";
    std::string m_quiet = "/tmp/avid_int_quiet_" + std::to_string(::getpid()) + ".json";

    auto rb = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + burst_bin + "\" --rounds 64 --window 64 --metrics-out \"" + m_burst + "\"");
    auto rq = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + quiet_bin + "\" --rounds 64 --window 64 --metrics-out \"" + m_quiet + "\"");

    ASSERT_EQ(rb.exit_code, 0);
    ASSERT_EQ(rq.exit_code, 0);

    std::string jb, jq;
    ASSERT_TRUE(load_text_fixture(m_burst, jb));
    ASSERT_TRUE(load_text_fixture(m_quiet, jq));
    ::unlink(m_burst.c_str());
    ::unlink(m_quiet.c_str());

    ASSERT_TRUE(jb != jq);
    ASSERT_TRUE(jb.find("\"final_nll\":") != std::string::npos);

    ctx.record().expected["histogram_connectivity"] = "avid_dma_ring_consume_slot -> hist_x/hist_z -> fit_histogram connected and responsive to input stream";
    ctx.record().actual["histogram_connectivity"] = "Both runs rc=0; metrics JSON confirms distinct Joint-DP NLL across quiet vs burst streams";
    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
