#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "control/fallback_fsm.h"
#include "estimator/llr_quantizer.h"
#include "ingress/packet_validator.h"
#include "driver/weight_reg_driver.h"
#include <cstdint>
#include <string>
#include <unistd.h>

using namespace avid::layer3_test;

namespace {

int run_one(const CliOptions &base_opts, const std::string &tid) {
    CliOptions opts = base_opts;
    opts.test_id = tid;
    const int16_t cand[9] = {750, 760, 770, 780, 790, 800, 810, 820, 830};

    if (tid == "L3-OP-029") {
        TestContext ctx(opts, tid, "fsm_valid_enum_output_domain",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify fsm.state stays strictly in {0, 1, 2, 3} across 32 sequential mixed health steps";
        ctx.record().independent_oracle = "All observed states in {0..3}";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        for (int i = 0; i < 32; ++i) {
            int16_t sel[9] = {0};
            bool commit = false;
            auto st = avid_fallback_step(&fsm, (i % 3) != 0, (i % 5) != 0, (i % 7) != 0, (i % 11) != 0, cand, sel, &commit);
            ASSERT_GE(static_cast<int>(st), 0);
            ASSERT_LE(static_cast<int>(st), 3);
        }
        ctx.record().expected["state_domain"] = "{0, 1, 2, 3}";
        ctx.record().actual["state_domain"] = "Verified over 32 mixed transitions";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-030") {
        TestContext ctx(opts, tid, "fsm_canonical_transition_sequence",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Drive exact sequence NORMAL -> HOLD -> HOLD -> SAFE_STATIC -> NORMAL -> DEGRADED -> NORMAL";
        ctx.record().independent_oracle = "Observed states match [0, 1, 1, 2, 0, 3, 0]";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool c = false;
        ASSERT_EQ(fsm.state, AVID_FSM_NORMAL_ADAPTIVE);
        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand, sel, &c), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(avid_fallback_step(&fsm, true, false, true, true, cand, sel, &c), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand, sel, &c), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, true, cand, sel, &c), AVID_FSM_NORMAL_ADAPTIVE);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, false, cand, sel, &c), AVID_FSM_DEGRADED_BYPASS);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, true, cand, sel, &c), AVID_FSM_NORMAL_ADAPTIVE);
        ctx.record().expected["sequence"] = "0->1->1->2->0->3->0";
        ctx.record().actual["sequence"] = "0->1->1->2->0->3->0";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-031") {
        TestContext ctx(opts, tid, "fsm_independent_fault_predicate_triggers",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Independently violate estimator_ok, deadline_ok, bounds_ok, and bus_ok and verify exact fallback state and trigger counter";
        ctx.record().independent_oracle = "!est->HOLD, !ddl->HOLD, !bnd->SAFE_STATIC, !bus->DEGRADED, each incrementing fallback_triggers";

        int16_t sel[9] = {0};
        bool c = false;
        avid_fallback_fsm_t f1{}, f2{}, f3{}, f4{};
        avid_fallback_init(&f1, AVID_Q78_DEFAULT);
        avid_fallback_init(&f2, AVID_Q78_DEFAULT);
        avid_fallback_init(&f3, AVID_Q78_DEFAULT);
        avid_fallback_init(&f4, AVID_Q78_DEFAULT);

        ASSERT_EQ(avid_fallback_step(&f1, false, true, true, true, cand, sel, &c), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(f1.fallback_triggers, 1ULL);

        ASSERT_EQ(avid_fallback_step(&f2, true, false, true, true, cand, sel, &c), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(f2.fallback_triggers, 1ULL);

        ASSERT_EQ(avid_fallback_step(&f3, true, true, false, true, cand, sel, &c), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(f3.fallback_triggers, 1ULL);

        ASSERT_EQ(avid_fallback_step(&f4, true, true, true, false, cand, sel, &c), AVID_FSM_DEGRADED_BYPASS);
        ASSERT_EQ(f4.fallback_triggers, 1ULL);

        ctx.record().expected["independent_triggers"] = "4/4 predicates independently increment fallback_triggers";
        ctx.record().actual["independent_triggers"] = "4/4 verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-032") {
        TestContext ctx(opts, tid, "fsm_recovery_eligibility_gating",
                        "software/src/control/fallback_fsm.cpp",
                        "avid_fallback_step", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify faulty steps hold FSM out of NORMAL_ADAPTIVE while 1 clean contiguous window restores NORMAL_ADAPTIVE";
        ctx.record().independent_oracle = "Faulty steps stay in HOLD/SAFE_STATIC; all-4-true step restores NORMAL_ADAPTIVE";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool c = false;
        ASSERT_EQ(avid_fallback_step(&fsm, false, true, true, true, cand, sel, &c), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_NE(avid_fallback_step(&fsm, true, false, true, true, cand, sel, &c), AVID_FSM_NORMAL_ADAPTIVE);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, true, cand, sel, &c), AVID_FSM_NORMAL_ADAPTIVE);
        ctx.record().expected["recovery"] = "Blocked while faulty, restored on clean window";
        ctx.record().actual["recovery"] = "Verified";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-033") {
        TestContext ctx(opts, tid, "daemon_health_exit_code_contract",
                        "software/src/inference/avid_daemon.cpp",
                        "avid_daemon main(--self-test / bad CLI)", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Run avid_daemon --self-test vs invalid CLI and verify exit code 0 vs 2";
        ctx.record().independent_oracle = "self-test rc==0; bad CLI rc==2";

        ASSERT_TRUE(!opts.daemon_bin.empty());
        std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
        auto r_ok = run_command_capture("\"" + opts.daemon_bin + "\" --self-test --checkpoint \"" + ckpt + "\"");
        auto r_bad = run_command_capture("\"" + opts.daemon_bin + "\" --scenario invalid_configuration");
        ASSERT_EQ(r_ok.exit_code, 0);
        ASSERT_EQ(r_bad.exit_code, 2);
        ctx.record().expected["rc_ok_and_bad"] = "0, 2";
        ctx.record().actual["rc_ok_and_bad"] = std::to_string(r_ok.exit_code) + ", " + std::to_string(r_bad.exit_code);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-034") {
        TestContext ctx(opts, tid, "error_reporting_counters_and_diagnostics",
                        "software/src/ingress/packet_validator.cpp, software/src/inference/avid_daemon.cpp",
                        "avid_validate_and_unpack_packet, avid_daemon", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify PARITY_ERR increments invalid_flag_count and invalid daemon scenario emits explicit stderr diagnostic";
        ctx.record().independent_oracle = "invalid_flag_count == 1 and stderr non-empty on rc=2";

        avid_packet_validator_t val{};
        avid_validator_init(&val);
        avid_syndrome_packet_t pkt{};
        uint64_t bad_word = avid_pack_packet_u64(
            0x2u, 0x1u, 0u, 0u,
            static_cast<uint8_t>(AVID_PKT_FLAG_VALID | AVID_PKT_FLAG_PARITY_ERR));
        ASSERT_FALSE(avid_validate_packet(&val, bad_word, &pkt));
        ASSERT_EQ(val.invalid_flag_count, 1ULL);

        auto r_bad = run_command_capture("\"" + opts.daemon_bin + "\" --scenario invalid_configuration");
        ASSERT_EQ(r_bad.exit_code, 2);
        ASSERT_TRUE((r_bad.stdout_text + r_bad.stderr_text).find("Invalid") != std::string::npos);
        ctx.record().expected["invalid_flag_and_stderr"] = "1 and explicit stderr error";
        ctx.record().actual["invalid_flag_count"] = val.invalid_flag_count;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-035") {
        TestContext ctx(opts, tid, "commit_result_and_epoch_reconciliation_output",
                        "software/src/driver/weight_reg_driver.c, software/src/inference/avid_daemon.cpp",
                        "wrd_commit_and_wait, avid_daemon(--metrics-out)", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Verify wrd_commit_and_wait status/epoch bump vs busy timeout, and daemon epoch_commit_reconciled == true";
        ctx.record().independent_oracle = "Emulated commit increments swap_count; daemon JSON epoch_commit_reconciled == true";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        ASSERT_EQ(wrd_read_epoch(&wrd), 0u);
        wrd_write_all_weights(&wrd, cand);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 8), 0);
        ASSERT_EQ(wrd_read_epoch(&wrd), 1u);
        wrd_close(&wrd);

        std::string mpath = "/tmp/avid_op035_" + std::to_string(::getpid()) + ".json";
        std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
        auto r = run_command_capture("\"" + opts.daemon_bin + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" +
                                     ckpt + "\" --rounds 512 --window 256 --metrics-out \"" + mpath + "\"");
        ASSERT_EQ(r.exit_code, 0);
        std::string js;
        ASSERT_TRUE(load_text_fixture(mpath, js));
        ::unlink(mpath.c_str());
        ASSERT_TRUE(js.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        ctx.record().expected["epoch_commit_reconciled"] = true;
        ctx.record().actual["epoch_commit_reconciled"] = true;
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-OP-036") {
        TestContext ctx(opts, tid, "daemon_runtime_counters_ledger_parity",
                        "software/src/inference/avid_daemon.cpp",
                        "avid_daemon telemetry ledger", "set2");
        ctx.record().test_set = "set2_output_statistics";
        ctx.record().test_description = "Compare validator_valid_packets, decoded_rtl_transactions, epochs_completed, and weight_bank_commits against exact 512-round ledger";
        ctx.record().independent_oracle = "valid==512, decoded==512, epochs==2, weight_bank_commits==3 (1 boot + 2 adaptive)";

        std::string mpath = "/tmp/avid_op036_" + std::to_string(::getpid()) + ".json";
        std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl = " --slow-deadline-us 500000";
#else
        const std::string san_dl = "";
#endif
        auto r = run_command_capture("\"" + opts.daemon_bin + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" +
                                     ckpt + "\" --scenario stationary_noise --rounds 512 --window 256" + san_dl + " --metrics-out \"" + mpath + "\"");
        ASSERT_EQ(r.exit_code, 0);
        std::string js;
        ASSERT_TRUE(load_text_fixture(mpath, js));
        ::unlink(mpath.c_str());

        ASSERT_TRUE(js.find("\"validator_valid_packets\": 512") != std::string::npos);
        ASSERT_TRUE(js.find("\"decoded_rtl_transactions\": 512") != std::string::npos);
        ASSERT_TRUE(js.find("\"epochs_completed\": 2") != std::string::npos);
        ASSERT_TRUE(js.find("\"weight_bank_commits\": 3") != std::string::npos);
        ctx.record().expected["ledger_512_256"] = "valid=512, decoded=512, epochs=2, commits=3";
        ctx.record().actual["ledger_512_256"] = "Exact match";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "ALL");
    const char *all_ids[] = {
        "L3-OP-029", "L3-OP-030", "L3-OP-031", "L3-OP-032",
        "L3-OP-033", "L3-OP-034", "L3-OP-035", "L3-OP-036"
    };
    if (opts.test_id == "ALL") {
        int rc = 0;
        for (const char *id : all_ids) {
            if (run_one(opts, id) != 0) rc = 1;
        }
        return rc;
    }
    return run_one(opts, opts.test_id);
}
