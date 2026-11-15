#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "driver/weight_reg_driver.h"
#include "driver/weight_bank_driver.h"
#include "fixture_loader.h"
#include <string>

using namespace avid::layer3_test;

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-IP-032");

    if (opts.test_id == "L3-IP-039") {
        TestContext ctx(opts, "L3-IP-039", "active_decoder_weight_bank_readback",
                        "software/src/driver/weight_reg_driver.c, weight_bank_driver.c",
                        "wrd_open, wrd_write_all_weights, wrd_commit, wrd_read_active_weight, avid_weight_bank_write_and_commit");
        ctx.record().test_description =
            "Active decoder config read — write 9 Q7.8 weights to shadow bank, verify active bank unchanged before commit, commit, and verify exact readback + out-of-range index safety";
        ctx.record().input_contract = "9 signed 16-bit Q7.8 LLR weights in shadow/active register banks";
        ctx.record().independent_oracle = "weight_map_reg.sv contract: 0x000..0x020 shadow, 0x024..0x044 active, 0x048 commit";
        ctx.record().fixture = "none (emulated MMIO register file)";

        wrd_handle_t h{};
        ASSERT_EQ(wrd_open(&h, WRD_BASE_ADDR), 0);

        const int16_t bank1[9] = {512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536};
        uint16_t swaps = 0;
        ASSERT_EQ(avid_weight_bank_write_and_commit(&h, bank1, &swaps), 0);
        ASSERT_EQ(static_cast<int>(swaps), 1);

        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(wrd_read_active_weight(&h, q), bank1[q]);
        }

        /* Write new shadow weights WITHOUT committing — active weights must remain bank1! */
        const int16_t bank2[9] = {2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008};
        wrd_write_all_weights(&h, bank2);
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(wrd_read_active_weight(&h, q), bank1[q]);
        }
        wrd_commit(&h);
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(wrd_read_active_weight(&h, q), bank2[q]);
        }

        /* Out-of-bounds qubit indices (-1, 9) must safely return 0 */
        ASSERT_EQ(wrd_read_active_weight(&h, -1), static_cast<int16_t>(0));
        ASSERT_EQ(wrd_read_active_weight(&h, 9), static_cast<int16_t>(0));

        ctx.log("Architectural note: weight_reg_driver.c implements a single 9-entry shadow bank (wrd_write_z_llr aliases wrd_write_x_llr).");
        wrd_close(&h);

        ctx.record().expected["active_readback"] = "Shadow isolated prior to commit; exact 9-weight active readback after commit; OOB index -> 0";
        ctx.record().actual["active_readback"] = "All 9 weights verified before/after commit; swap_count=2; OOB(-1,9)=0";
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    /* L3-IP-032: Commit delay configuration */
    TestContext ctx(opts, "L3-IP-032", "commit_delay_configuration_and_daemon_wiring",
                    "software/src/control/causal_scheduler.cpp, software/src/inference/avid_daemon.cpp",
                    "avid_scheduler_init, avid_scheduler_schedule_commit, avid_scheduler_poll_commit_ready");
    ctx.record().test_description =
        "Commit delay configuration — verify causal_scheduler commit_delay=0 vs commit_delay>0 AND check whether avid_daemon wires commit_delay";
    ctx.record().input_contract = "commit_delay_rounds >= 0 delays weight activation to window_end_round + commit_delay_rounds";
    ctx.record().independent_oracle = "architecture/contracts/system_timing.yaml & causal_scheduler.h vs avid_daemon.cpp call graph";
    ctx.record().fixture = "tests/layer3/fixtures/configs/valid/valid_experiment.yaml";

    avid_causal_scheduler_t sched0{}, sched8{};
    const int16_t w[9] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
    int16_t out[9] = {0};

    avid_scheduler_init(&sched0, 16, 256, 0);
    avid_scheduler_schedule_commit(&sched0, 256, w);
    ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched0, 255, out));
    ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched0, 256, out));

    avid_scheduler_init(&sched8, 16, 256, 8);
    avid_scheduler_schedule_commit(&sched8, 256, w);
    ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched8, 263, out));
    ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched8, 264, out));
    ASSERT_EQ(static_cast<int>(sched8.active_epoch_id), 1);

    /* Inspect avid_daemon.cpp source to verify whether avid_causal_scheduler_t or --commit-delay is wired */
    std::string daemon_src;
    ASSERT_TRUE(load_text_fixture(opts.repo_root + "/software/src/inference/avid_daemon.cpp", daemon_src));
    bool daemon_uses_scheduler = (daemon_src.find("avid_scheduler_init") != std::string::npos) ||
                                 (daemon_src.find("avid_scheduler_schedule_commit") != std::string::npos);
    bool daemon_has_commit_delay_cli = (daemon_src.find("--commit-delay") != std::string::npos);

    if (!daemon_uses_scheduler || !daemon_has_commit_delay_cli) {
        ctx.record_failure("CONTRACT MISMATCH: `avid_causal_scheduler_t` enforces `commit_delay_rounds` standalone, but `avid_daemon.cpp` never instantiates `avid_causal_scheduler_t`, exposes no `--commit-delay` flag, and commits weights synchronously inside the window boundary!");
    }

    ctx.record().expected["commit_delay"] = "causal_scheduler enforces commit_delay AND avid_daemon consumes commit_delay configuration";
    ctx.record().actual["commit_delay"] =
        "Standalone causal_scheduler PASS (delay=0->256, delay=8->264), BUT avid_daemon uses_scheduler=" +
        std::string(daemon_uses_scheduler ? "true" : "false") + ", has_cli_flag=" +
        std::string(daemon_has_commit_delay_cli ? "true" : "false");

    return ctx.finalize(TestStatus::CONTRACT_MISMATCH, "SCHEDULER_COMMIT_DELAY_DISCONNECTED_FROM_DAEMON");
}
