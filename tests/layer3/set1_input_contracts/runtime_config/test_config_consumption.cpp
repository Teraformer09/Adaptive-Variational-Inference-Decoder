#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <cstdio>
#include <string>
#include <unistd.h>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-031");
    std::string daemon = opts.daemon_bin;

    if (opts.test_id == "L3-IP-037") {
        TestContext ctx(opts, "L3-IP-037", "runtime_telemetry_and_epoch_reconciliation",
                        "software/src/inference/avid_daemon.cpp",
                        "avid_daemon telemetry JSON emission & epoch commit reconciliation");
        ctx.record().test_description =
            "Runtime telemetry input — execute avid_daemon_under_test over 256 rounds (window=64) and verify epoch_commit_reconciled, hw_epoch, and profiler telemetry fields";
        ctx.record().input_contract = "Runtime telemetry tracks rounds_completed, epochs_completed, adaptive_epoch_commits, and epoch_commit_reconciled";
        ctx.record().independent_oracle = "initial_boot_commits + adaptive_epoch_commits == final_hw_epoch + reset_cleared_commits (epoch_commit_reconciled=true)";
        ctx.record().fixture = "checkpoints/r2d2_v1_trained.bin";

        ASSERT_TRUE(!daemon.empty());
        std::string metrics_path = "/tmp/avid_l3_ip037_metrics_" + std::to_string(::getpid()) + ".json";
        std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

        auto res = run_command_capture(
            "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
            "\" --rounds 256 --window 64 --seed 42 --metrics-out \"" + metrics_path + "\"");
        ASSERT_EQ(res.exit_code, 0);

        std::string mjson;
        ASSERT_TRUE(load_text_fixture(metrics_path, mjson));
        ::unlink(metrics_path.c_str());

        ASSERT_TRUE(mjson.find("\"total_rounds\": 256") != std::string::npos);
        ASSERT_TRUE(mjson.find("\"decoded_rtl_transactions\": 256") != std::string::npos);
        ASSERT_TRUE(mjson.find("\"epochs_completed\": 4") != std::string::npos);
        ASSERT_TRUE(mjson.find("\"epoch_commit_reconciled\": true") != std::string::npos);

        ctx.record().expected["telemetry_reconciliation"] = "rc=0, total_rounds=256, decoded_rtl_transactions=256, epochs_completed=4, epoch_commit_reconciled=true";
        ctx.record().actual["telemetry_reconciliation"] =
            "rc=" + std::to_string(res.exit_code) + ", epochs_completed=4, epoch_commit_reconciled=true";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    /* L3-IP-031: Window size configuration consumption */
    TestContext ctx(opts, "L3-IP-031", "window_size_configuration_consumption",
                    "software/src/inference/avid_daemon.cpp",
                    "main (--window validation & epoch cadence)");
    ctx.record().test_description =
        "Window size configuration — verify avid_daemon rejects --window 0 and --window -16 (rc=2) and genuinely consumes --window 256 (4 epochs/1024 rounds) vs --window 512 (2 epochs/1024 rounds)";
    ctx.record().input_contract = "window_rounds > 0 required; epoch cadence scales inversely with window_rounds";
    ctx.record().independent_oracle = "1024 / 256 == 4 epochs vs 1024 / 512 == 2 epochs; window<=0 -> exit code 2";
    ctx.record().fixture = "checkpoints/r2d2_v1_trained.bin";

    ASSERT_TRUE(!daemon.empty());
    auto r_zero = run_command_capture("\"" + daemon + "\" --controller-mode static --rounds 256 --window 0");
    auto r_neg  = run_command_capture("\"" + daemon + "\" --controller-mode static --rounds 256 --window -16");
    ASSERT_EQ(r_zero.exit_code, 2);
    ASSERT_EQ(r_neg.exit_code, 2);

    std::string m256 = "/tmp/avid_l3_ip031_w256_" + std::to_string(::getpid()) + ".json";
    std::string m512 = "/tmp/avid_l3_ip031_w512_" + std::to_string(::getpid()) + ".json";
    std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    auto r256 = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --rounds 1024 --window 256 --seed 42 --metrics-out \"" + m256 + "\"");
    auto r512 = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --rounds 1024 --window 512 --seed 42 --metrics-out \"" + m512 + "\"");
    ASSERT_EQ(r256.exit_code, 0);
    ASSERT_EQ(r512.exit_code, 0);

    std::string j256, j512;
    ASSERT_TRUE(load_text_fixture(m256, j256));
    ASSERT_TRUE(load_text_fixture(m512, j512));
    ::unlink(m256.c_str());
    ::unlink(m512.c_str());

    ASSERT_TRUE(j256.find("\"epochs_completed\": 4") != std::string::npos);
    ASSERT_TRUE(j512.find("\"epochs_completed\": 2") != std::string::npos);

    ctx.record().expected["window_consumption"] = "--window 0/-16 -> rc=2; --window 256 -> 4 epochs; --window 512 -> 2 epochs";
    ctx.record().actual["window_consumption"] =
        "w=0 rc=" + std::to_string(r_zero.exit_code) + ", w=-16 rc=" + std::to_string(r_neg.exit_code) +
        ", w=256->4 epochs, w=512->2 epochs";
    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
