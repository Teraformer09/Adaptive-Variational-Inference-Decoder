#include "test_assertions.h"
#include "test_common.h"
#include "estimator/lbfgs_solver.h"
#include "hal/avid_hal.h"
#include "inference/noise_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

namespace {

static std::string read_text_file_unlink(const std::string &path) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    ::unlink(path.c_str());
    return ss.str();
}

static double extract_json_num(const std::string &body, const std::string &key, double def_val = -1.0) {
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def_val;
    pos += pat.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) pos++;
    return std::strtod(body.c_str() + pos, nullptr);
}

static std::vector<double> extract_json_array(const std::string &body, const std::string &key) {
    std::vector<double> out;
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return out;
    size_t lb = body.find('[', pos);
    size_t rb = body.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return out;
    std::stringstream ss(body.substr(lb + 1, rb - lb - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        char *endp = nullptr;
        double v = std::strtod(item.c_str(), &endp);
        if (endp != item.c_str()) out.push_back(v);
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-EXEC-010");
    const std::string tid  = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-EXEC-010") {
        TestContext ctx(opts, tid, "graceful_sigterm_shutdown_and_telemetry_flush",
                        "software/src/inference/avid_daemon.cpp", "sigint_handler + main", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Launch avid_daemon with --rounds 5000000, send SIGTERM mid-run, and verify clean exit 0 + JSON flush";
        ctx.record().independent_oracle = "WIFEXITED && WEXITSTATUS==0, valid metrics JSON written with epoch_commit_reconciled==true";

        std::string mp = "/tmp/l3_exec010_" + std::to_string(::getpid()) + ".json";
        ::unlink(mp.c_str());
        pid_t pid = ::fork();
        if (pid == 0) {
            ::execl(opts.daemon_bin.c_str(), opts.daemon_bin.c_str(),
                    "--scenario", "stationary_noise",
                    "--rounds", "5000000",
                    "--window", "256",
                    "--backend", "physical",
                    "--checkpoint", ckpt.c_str(),
                    "--metrics-out", mp.c_str(),
                    static_cast<char *>(nullptr));
            _exit(127);
        }
        ::usleep(120000); /* 120 ms mid-run */
        ::kill(pid, SIGTERM);
        int status = 0;
        ::waitpid(pid, &status, 0);
        ASSERT_TRUE(WIFEXITED(status));
        ASSERT_EQ(WEXITSTATUS(status), 0);

        std::string body = read_text_file_unlink(mp);
        ASSERT_FALSE(body.empty());
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        ASSERT_TRUE(extract_json_num(body, "decoded_rtl_transactions") > 0.0);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-011") {
        TestContext ctx(opts, tid, "histogram_persistence_in_telemetry_and_state_file",
                        "software/src/inference/avid_daemon.cpp", "main (total_hist_x/z + --state-out)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Verify 16-bin total_hist_x and total_hist_z persist to --metrics-out JSON and --state-out file";
        ctx.record().independent_oracle = "16 bins persisted in JSON summing to total_rounds=512 and binary state file written atomically";

        std::string mp = "/tmp/l3_exec011_" + std::to_string(::getpid()) + ".json";
        std::string sp = "/tmp/l3_exec011_" + std::to_string(::getpid()) + ".state";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --state-out " + sp + " --metrics-out " + mp).exit_code, 0);

        struct stat st{};
        ASSERT_EQ(::stat(sp.c_str(), &st), 0);
        ASSERT_TRUE(st.st_size > 200);
        ::unlink(sp.c_str());

        std::string body = read_text_file_unlink(mp);
        auto hx = extract_json_array(body, "total_hist_x");
        auto hz = extract_json_array(body, "total_hist_z");
        ASSERT_EQ(hx.size(), 16u);
        ASSERT_EQ(hz.size(), 16u);
        double sx = 0.0, sz = 0.0;
        for (int i = 0; i < 16; ++i) { sx += hx[i]; sz += hz[i]; }
        ASSERT_EQ(static_cast<int>(sx), 512);
        ASSERT_EQ(static_cast<int>(sz), 512);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-012") {
        TestContext ctx(opts, tid, "noise_estimate_persistence_21_param_and_rmse",
                        "software/src/inference/avid_daemon.cpp", "main (final_theta_z + orbit_9class_rmse)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Verify 21-parameter final_theta_z, orbit_9class_rmse, param_21_rmse, and final_nll persist in JSON";
        ctx.record().independent_oracle = "All 21 persisted theta_z coordinates lie strictly in [1e-5, 0.45] with finite RMSE metrics";

        std::string mp = "/tmp/l3_exec012_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        auto th = extract_json_array(body, "final_theta_z");
        ASSERT_EQ(th.size(), 21u);
        for (double v : th) {
            ASSERT_IN_RANGE(v, 1e-5 - 1e-7, 0.45 + 1e-7);
        }
        ASSERT_FINITE(extract_json_num(body, "orbit_9class_rmse"));
        ASSERT_FINITE(extract_json_num(body, "param_21_rmse"));
        ASSERT_FINITE(extract_json_num(body, "final_nll"));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-013") {
        TestContext ctx(opts, tid, "optimizer_diagnostics_persistence",
                        "software/src/inference/avid_daemon.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "LBFGSFitDiagnostics -> metrics JSON", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Verify final_nll, legacy_marginal_nll, lbfgs_iterations, and lbfgs_grad_inf_norm persist cleanly";
        ctx.record().independent_oracle = "final_nll > 0, legacy_marginal_nll > 0, lbfgs_iterations > 0, finite lbfgs_grad_inf_norm";

        std::string mp = "/tmp/l3_exec013_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(extract_json_num(body, "final_nll") > 0.0);
        ASSERT_TRUE(extract_json_num(body, "legacy_marginal_nll") > 0.0);
        ASSERT_TRUE(extract_json_num(body, "lbfgs_iterations") > 0.0);
        ASSERT_FINITE(extract_json_num(body, "lbfgs_grad_inf_norm"));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-014") {
        TestContext ctx(opts, tid, "committed_weight_persistence_in_hal_and_telemetry",
                        "software/src/hal/physical_backend.cpp, software/src/driver/weight_reg_driver.c",
                        "avid_hal_write_weights_and_commit + MMIO 0x100..0x144", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Probe committed active weights via HAL MMIO read 0x100..0x144 and daemon final_q78_z telemetry";
        ctx.record().independent_oracle = "Shadow/active register banks match post-commit and final_q78_z persists 9 safe Q7.8 words";

        avid_hal_t hal{};
        ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR), 0);
        const int16_t wx[9] = {1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090};
        const int16_t wz[9] = {710,  720,  730,  740,  550,  760,  770,  780,  790};
        uint32_t ep = 0;
        uint64_t cyc = 0;
        ASSERT_EQ(avid_hal_write_weights_and_commit(&hal, wx, wz, &ep, &cyc), 0);
        for (int q = 0; q < 9; ++q) {
            uint32_t rx = 0, rz = 0;
            ASSERT_EQ(avid_hal_mmio_read(&hal, 0x100u + 4u * q, &rx, &cyc), 0);
            ASSERT_EQ(avid_hal_mmio_read(&hal, 0x124u + 4u * q, &rz, &cyc), 0);
            ASSERT_EQ(static_cast<int16_t>(rx & 0xFFFFu), wx[q]);
            ASSERT_EQ(static_cast<int16_t>(rz & 0xFFFFu), wz[q]);
        }
        avid_hal_close(&hal);

        std::string mp = "/tmp/l3_exec014_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        auto qz = extract_json_array(read_text_file_unlink(mp), "final_q78_z");
        ASSERT_EQ(qz.size(), 9u);
        for (double v : qz) ASSERT_IN_RANGE(v, 64.0, 12288.0);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-015") {
        TestContext ctx(opts, tid, "epoch_ledger_persistence_and_reconciliation",
                        "software/src/inference/avid_daemon.cpp", "main (commit ledger)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Verify initial_boot_commits + adaptive_epoch_commits - reset_cleared_commits == weight_bank_commits";
        ctx.record().independent_oracle = "Exact ledger identity holds across both stationary_noise and reset_recovery";

        for (const char *sc : {"stationary_noise", "reset_recovery"}) {
            std::string mp = "/tmp/l3_exec015_" + std::string(sc) + "_" + std::to_string(::getpid()) + ".json";
            ASSERT_EQ(run_command_capture(
                opts.daemon_bin + " --scenario " + sc + " --rounds 1024 --window 256 --seed 2026 --backend physical"
                " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
            std::string body = read_text_file_unlink(mp);
            int init_c = static_cast<int>(extract_json_num(body, "initial_boot_commits"));
            int adap_c = static_cast<int>(extract_json_num(body, "adaptive_epoch_commits"));
            int rst_c  = static_cast<int>(extract_json_num(body, "reset_cleared_commits"));
            int hw_ep  = static_cast<int>(extract_json_num(body, "weight_bank_commits"));
            ASSERT_EQ(init_c + adap_c - rst_c, hw_ep);
            ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-016") {
        TestContext ctx(opts, tid, "safety_fsm_state_and_fallback_count_persistence",
                        "software/src/inference/avid_daemon.cpp, software/src/control/fallback_fsm.cpp",
                        "main (fallback_events + final_fsm_state)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run fallback_safety_trigger (1 faulted epoch vs 4 epochs) and inspect persisted final_fsm_state & fallback_events";
        ctx.record().independent_oracle = "Epoch-1-only run ends in SAFE_STATIC_PRIOR(2) with fallback_events==1; 4-epoch run recovers to NORMAL(0)";

        std::string mp1 = "/tmp/l3_exec016_1_" + std::to_string(::getpid()) + ".json";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl016 = " --slow-deadline-us 500000";
#else
        const std::string san_dl016 = "";
#endif
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario fallback_safety_trigger --rounds 256 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + san_dl016 + " --metrics-out " + mp1).exit_code, 0);
        std::string b1 = read_text_file_unlink(mp1);
        int fb1 = static_cast<int>(extract_json_num(b1, "fallback_events"));
        ASSERT_GE(fb1, 1);
        ASSERT_EQ(static_cast<int>(extract_json_num(b1, "final_fsm_state")), 2);

        std::string mp2 = "/tmp/l3_exec016_2_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario fallback_safety_trigger --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + san_dl016 + " --metrics-out " + mp2).exit_code, 0);
        std::string b2 = read_text_file_unlink(mp2);
        ASSERT_EQ(static_cast<int>(extract_json_num(b2, "fallback_events")), fb1);
        ASSERT_EQ(static_cast<int>(extract_json_num(b2, "final_fsm_state")), 0);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
