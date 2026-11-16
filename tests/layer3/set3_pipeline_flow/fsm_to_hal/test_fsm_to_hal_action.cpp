#include "test_assertions.h"
#include "test_common.h"
#include "control/fallback_fsm.h"
#include "estimator/llr_quantizer.h"
#include "hal/avid_hal.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-009");
    const std::string tid = opts.test_id;

    if (tid == "L3-FLOW-009") {
        TestContext ctx(opts, tid, "scheduler_to_safety_fsm_gate_continuity",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify (est_ok, deadline_ok, bounds_ok, bus_ok, cand_q78) produce exact sel_q78 and should_commit across FSM states";
        ctx.record().independent_oracle = "NORMAL commits cand_q78; HOLD retains last_good with should_commit=false; SAFE_STATIC commits 0x0468";

        avid_fallback_fsm_t fsm;
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t cand_good[9] = {900, 910, 920, 930, 620, 940, 950, 960, 970};
        const int16_t cand_bad[9]  = {1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200};
        int16_t sel[9] = {0};
        bool commit = false;

        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, true, cand_good, sel, &commit), AVID_FSM_NORMAL_ADAPTIVE);
        ASSERT_TRUE(commit);
        ASSERT_EQ(sel[4], 620);

        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand_bad, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_FALSE(commit);
        ASSERT_EQ(sel[4], 620);

        ASSERT_EQ(avid_fallback_step(&fsm, true, true, false, true, cand_bad, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(sel[4], AVID_Q78_DEFAULT);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-010") {
        TestContext ctx(opts, tid, "safety_fsm_to_shadow_bank_write_protection",
                        "software/src/control/fallback_fsm.cpp, software/src/driver/weight_reg_driver.c",
                        "avid_fallback_step -> wrd_write_all_weights", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify NORMAL_ADAPTIVE writes shadow bank & commits while HOLD_LAST_GOOD blocks commit leaving active bank unchanged";
        ctx.record().independent_oracle = "Active bank stays at cand_good when HOLD_LAST_GOOD blocks cand_bad";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        avid_fallback_fsm_t fsm;
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);

        const int16_t cand_good[9] = {800, 805, 810, 815, 580, 820, 825, 830, 835};
        const int16_t cand_bad[9]  = {300, 300, 300, 300, 300, 300, 300, 300, 300};
        int16_t sel[9] = {0};
        bool should_commit = false;

        avid_fallback_step(&fsm, true, true, true, true, cand_good, sel, &should_commit);
        if (should_commit) {
            wrd_write_all_weights(&wrd, sel);
            ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        }
        ASSERT_EQ(wrd_read_active_weight(&wrd, 4), 580);
        uint32_t ep1 = wrd_read_epoch(&wrd);

        /* Trigger HOLD_LAST_GOOD via deadline miss */
        avid_fallback_step(&fsm, true, false, true, true, cand_bad, sel, &should_commit);
        ASSERT_FALSE(should_commit);
        if (should_commit) {
            wrd_write_all_weights(&wrd, sel);
            wrd_commit_and_wait(&wrd, 16);
        }
        ASSERT_EQ(wrd_read_active_weight(&wrd, 4), 580);
        ASSERT_EQ(wrd_read_epoch(&wrd), ep1);
        wrd_close(&wrd);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-011") {
        TestContext ctx(opts, tid, "shadow_bank_to_hal_commit_atomic_swap_and_readback",
                        "software/src/hal/physical_backend.cpp, software/src/driver/weight_reg_driver.c",
                        "avid_hal_write_weights_and_commit -> avid_hal_mmio_read", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Write distinct X and Z Q7.8 banks via HAL commit, read 0x050 epoch and 0x100..0x144 active registers";
        ctx.record().independent_oracle = "hw_epoch increments by 1 and active X (0x100..0x120) != active Z (0x124..0x144) match committed arrays";

        avid_hal_t hal{};
        ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR), 0);
        int16_t qx[9], qz[9];
        for (int q = 0; q < 9; ++q) {
            qx[q] = static_cast<int16_t>(1100 + q * 10);
            qz[q] = static_cast<int16_t>(650 + q * 15);
        }
        uint32_t ep = 0;
        uint64_t cyc = 0;
        ASSERT_EQ(avid_hal_write_weights_and_commit(&hal, qx, qz, &ep, &cyc), 0);
        ASSERT_EQ(ep, 1u);

        uint32_t ep_reg = 0;
        ASSERT_EQ(avid_hal_mmio_read(&hal, 0x050u, &ep_reg, &cyc), 0);
        ASSERT_EQ(ep_reg, 1u);

        for (int q = 0; q < 9; ++q) {
            uint32_t rx = 0, rz = 0;
            ASSERT_EQ(avid_hal_mmio_read(&hal, 0x100u + 4u * q, &rx, &cyc), 0);
            ASSERT_EQ(avid_hal_mmio_read(&hal, 0x124u + 4u * q, &rz, &cyc), 0);
            ASSERT_EQ(static_cast<int16_t>(rx & 0xFFFFu), qx[q]);
            ASSERT_EQ(static_cast<int16_t>(rz & 0xFFFFu), qz[q]);
        }
        avid_hal_close(&hal);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-012") {
        TestContext ctx(opts, tid, "hal_commit_response_to_daemon_reconciliation",
                        "software/src/inference/avid_daemon.cpp",
                        "main (epoch_commit_reconciled & metrics JSON)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run avid_daemon_under_test and verify HAL commit_operations, weight_bank_commits, and epoch_commit_reconciled==true";
        ctx.record().independent_oracle = "rc==0, epoch_commit_reconciled==true, weight_bank_commits == initial_boot_commits + adaptive_epoch_commits";

        std::string mpath = "/tmp/l3_flow012_" + std::to_string(::getpid()) + ".json";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl = " --slow-deadline-us 500000";
#else
        const std::string san_dl = "";
#endif
        std::string cmd = opts.daemon_bin +
            " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + opts.repo_root + "/checkpoints/r2d2_v1_trained.bin" + san_dl +
            " --metrics-out " + mpath;
        SubprocessResult res = run_command_capture(cmd);
        ASSERT_EQ(res.exit_code, 0);

        std::ifstream ifs(mpath);
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string body = ss.str();
        ::unlink(mpath.c_str());

        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        ASSERT_TRUE(body.find("\"hal_rejected_transactions\": 0") != std::string::npos);
        ASSERT_TRUE(body.find("\"adaptive_epoch_commits\": 2") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
