#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include <string>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-CFG-MISSING");
    TestContext ctx(opts, opts.test_id, "missing_required_runtime_inputs",
                    "software/src/inference/avid_daemon.cpp",
                    "main CLI & required resource validation (--checkpoint, --stim-bin)");

    ctx.record().test_description =
        "Missing required runtime inputs — verify avid_daemon safely rejects controller_mode=avid_r2d2 without --checkpoint, nonexistent --checkpoint, and nonexistent --stim-bin with exit code 2";
    ctx.record().input_contract = "Missing required checkpoint or stimulus file must fail fast with non-zero exit code 2";
    ctx.record().independent_oracle = "avid_daemon.cpp lines 279-305 exit code == 2 and explicit error message on stderr";
    ctx.record().fixture = "tests/layer3/fixtures/configs/invalid/missing_required.yaml";

    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());

    /* 1. Default controller_mode=avid_r2d2 without --checkpoint */
    auto r_no_ckpt = run_command_capture("\"" + daemon + "\" --rounds 64 --window 64");
    ASSERT_EQ(r_no_ckpt.exit_code, 2);
    ASSERT_TRUE((r_no_ckpt.stdout_text + r_no_ckpt.stderr_text).find("requires a valid R2D2 checkpoint") != std::string::npos);

    /* 2. Missing/nonexistent --checkpoint file */
    auto r_bad_ckpt = run_command_capture("\"" + daemon + "\" --checkpoint /nonexistent/r2d2_missing.bin --rounds 64 --window 64");
    ASSERT_EQ(r_bad_ckpt.exit_code, 2);
    ASSERT_TRUE((r_bad_ckpt.stdout_text + r_bad_ckpt.stderr_text).find("Failed to load R2D2 checkpoint") != std::string::npos);

    /* 3. Missing/nonexistent --stim-bin file */
    auto r_bad_stim = run_command_capture("\"" + daemon + "\" --controller-mode static --stim-bin /nonexistent/missing_stim.bin --rounds 64 --window 64");
    ASSERT_EQ(r_bad_stim.exit_code, 2);
    ASSERT_TRUE((r_bad_stim.stdout_text + r_bad_stim.stderr_text).find("Failed to open --stim-bin") != std::string::npos);

    /* 4. Flag with missing trailing value (--window with no argument at end of argv) */
    auto r_dangling_flag = run_command_capture("\"" + daemon + "\" --controller-mode static --rounds 64 --window");
    ASSERT_EQ(r_dangling_flag.exit_code, 2);

    ctx.record().expected["missing_config"] = "All missing required resources and dangling value-flags rejected with non-zero exit code";
    ctx.record().actual["missing_config"] =
        "missing_ckpt rc=" + std::to_string(r_no_ckpt.exit_code) +
        ", bad_ckpt rc=" + std::to_string(r_bad_ckpt.exit_code) +
        ", bad_stim rc=" + std::to_string(r_bad_stim.exit_code) +
        ", dangling_window_flag rc=" + std::to_string(r_dangling_flag.exit_code);

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS,
                        ctx.has_assertion_failures() ? "DANGLING_CLI_ARGUMENT_SILENTLY_IGNORED" : "");
}
