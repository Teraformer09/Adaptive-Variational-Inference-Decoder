#include "test_assertions.h"
#include "test_common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/resource.h>
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

static std::string strip_wallclock_fields(const std::string &json_text) {
    std::istringstream iss(json_text);
    std::ostringstream oss;
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("\"mean_estimator_us\":") != std::string::npos ||
            line.find("\"max_estimator_us\":") != std::string::npos ||
            line.find("\"slow_loop_cpu_utilization_10ms_pct\":") != std::string::npos) {
            continue;
        }
        oss << line << "\n";
    }
    return oss.str();
}

static double extract_json_num(const std::string &body, const std::string &key, double def_val = -1.0) {
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def_val;
    pos += pat.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) pos++;
    return std::strtod(body.c_str() + pos, nullptr);
}

static long measure_maxrss_kb(const std::string &cmd) {
    std::fflush(nullptr);
    pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        ::execl("/bin/sh", "sh", "-c", (cmd + " >/dev/null 2>&1").c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    int wstatus = 0;
    struct rusage ru {};
    if (::wait4(pid, &wstatus, 0, &ru) < 0) return -1;
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) return -1;
    return ru.ru_maxrss;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-EXEC-001");
    const std::string tid  = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-EXEC-001") {
        TestContext ctx(opts, tid, "clean_cpp_cmake_build_from_scratch",
                        "software/CMakeLists.txt", "cmake configure + build", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Configure and build software/CMakeLists.txt in a clean temporary directory";
        ctx.record().independent_oracle = "CMake exits 0 and emits executable avid_daemon and test_baseline_suite binaries";

        std::string bdir = "/tmp/l3_exec001_build_" + std::to_string(::getpid());
        run_command_capture("rm -rf " + bdir);
        SubprocessResult cfg = run_command_capture(
            "cmake -S " + opts.repo_root + "/software -B " + bdir + " -DCMAKE_BUILD_TYPE=Release");
        ASSERT_EQ(cfg.exit_code, 0);
        SubprocessResult bld = run_command_capture("cmake --build " + bdir + " -j4");
        ASSERT_EQ(bld.exit_code, 0);

        struct stat st_d{}, st_t{};
        ASSERT_EQ(::stat((bdir + "/avid_daemon").c_str(), &st_d), 0);
        ASSERT_EQ(::stat((bdir + "/test_baseline_suite").c_str(), &st_t), 0);
        run_command_capture("rm -rf " + bdir);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-002") {
        TestContext ctx(opts, tid, "daemon_initialization_and_boot_commit",
                        "software/src/inference/avid_daemon.cpp", "main (--self-test)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Launch avid_daemon --self-test --checkpoint ... --metrics-out ...";
        ctx.record().independent_oracle = "Exit 0, r2d2_checkpoint_loaded==true, initial_boot_commits==1, epoch_commit_reconciled==true";

        std::string mp = "/tmp/l3_exec002_" + std::to_string(::getpid()) + ".json";
        SubprocessResult res = run_command_capture(
            opts.daemon_bin + " --self-test --checkpoint " + ckpt + " --metrics-out " + mp);
        ASSERT_EQ(res.exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(body.find("\"r2d2_checkpoint_loaded\": true") != std::string::npos);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "initial_boot_commits")), 1);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-003") {
        TestContext ctx(opts, tid, "daemon_self_test_and_baseline_suite",
                        "software/src/inference/avid_daemon.cpp, software/tests/test_baseline_suite.cpp",
                        "main (--self-test) + test_baseline_suite", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Execute avid_daemon --self-test and build+run test_baseline_suite from repo root";
        ctx.record().independent_oracle = "Both exit 0 and print Self-test passed and [ALL C++ BASELINE TESTS PASSED]";

        SubprocessResult r_self = run_command_capture(opts.daemon_bin + " --self-test --checkpoint " + ckpt);
        ASSERT_EQ(r_self.exit_code, 0);
        ASSERT_TRUE(r_self.stdout_text.find("Self-test passed") != std::string::npos);

        std::string bdir = "/tmp/l3_exec003_build_" + std::to_string(::getpid());
        run_command_capture("rm -rf " + bdir);
        ASSERT_EQ(run_command_capture(
            "cmake -S " + opts.repo_root + "/software -B " + bdir +
            " -DCMAKE_BUILD_TYPE=Release && cmake --build " + bdir + " --target test_baseline_suite -j4").exit_code, 0);
        SubprocessResult r_base = run_command_capture("cd " + opts.repo_root + " && " + bdir + "/test_baseline_suite");
        run_command_capture("rm -rf " + bdir);
        ASSERT_EQ(r_base.exit_code, 0);
        ASSERT_TRUE(r_base.stdout_text.find("[ALL C++ BASELINE TESTS PASSED]") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-004") {
        TestContext ctx(opts, tid, "runtime_cli_configuration_consumption",
                        "software/src/inference/avid_daemon.cpp", "main (CLI options)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Execute avid_daemon across distinct --rounds, --window, --seed, and --controller-mode static|avid_r2d2|b2_pid";
        ctx.record().independent_oracle = "Metrics JSON reflects exact CLI parameters and distinct controller trajectories";

        const char *modes[3] = {"static", "avid_r2d2", "b2_pid"};
        double lers[3] = {0.0};
        for (int i = 0; i < 3; ++i) {
            std::string mp = "/tmp/l3_exec004_" + std::to_string(i) + "_" + std::to_string(::getpid()) + ".json";
            int rounds = 512 + 256 * i;
            int seed = 1000 + i;
            std::string cmd = opts.daemon_bin +
                " --scenario correlated_noise --rounds " + std::to_string(rounds) +
                " --window 256 --seed " + std::to_string(seed) +
                " --controller-mode " + modes[i] +
                " --checkpoint " + ckpt + " --metrics-out " + mp;
            ASSERT_EQ(run_command_capture(cmd).exit_code, 0);
            std::string body = read_text_file_unlink(mp);
            ASSERT_TRUE(body.find(std::string("\"controller_mode\": \"") + modes[i] + "\"") != std::string::npos);
            ASSERT_EQ(static_cast<int>(extract_json_num(body, "total_rounds")), rounds);
            ASSERT_EQ(static_cast<int>(extract_json_num(body, "seed")), seed);
            lers[i] = extract_json_num(body, "logical_error_rate");
        }
        ASSERT_TRUE(lers[0] != lers[1] || lers[1] != lers[2]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-005") {
        TestContext ctx(opts, tid, "invalid_configuration_rejection",
                        "software/src/inference/avid_daemon.cpp", "main (config validation)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Pass --scenario invalid_configuration, --window 0, --rounds -5, --osd-mode 9";
        ctx.record().independent_oracle = "Every invalid invocation exits cleanly with code 2 and diagnostic stderr";

        const std::vector<std::string> bad_flags = {
            "--scenario invalid_configuration",
            "--window 0 --checkpoint " + ckpt,
            "--rounds -5 --checkpoint " + ckpt,
            "--osd-mode 9 --checkpoint " + ckpt,
            "--unknown-nonsense-flag"
        };
        for (const auto &fl : bad_flags) {
            SubprocessResult r = run_command_capture(opts.daemon_bin + " " + fl);
            ASSERT_FALSE(r.signaled);
            ASSERT_EQ(r.exit_code, 2);
            ASSERT_FALSE((r.stdout_text + r.stderr_text).empty());
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-006") {
        TestContext ctx(opts, tid, "missing_input_safe_startup_failure",
                        "software/src/inference/avid_daemon.cpp", "main (missing file/socket guards)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Launch avid_daemon with nonexistent --checkpoint, --stim-bin, and --socket";
        ctx.record().independent_oracle = "Non-zero exit code in {1, 2} with signaled==false (zero segfault)";

        SubprocessResult r1 = run_command_capture(opts.daemon_bin + " --checkpoint /tmp/no_such_ckpt_99999.bin");
        ASSERT_FALSE(r1.signaled);
        ASSERT_EQ(r1.exit_code, 2);

        SubprocessResult r2 = run_command_capture(opts.daemon_bin + " --checkpoint " + ckpt + " --stim-bin /tmp/no_such_stim_99999.bin");
        ASSERT_FALSE(r2.signaled);
        ASSERT_EQ(r2.exit_code, 2);

        SubprocessResult r3 = run_command_capture(opts.daemon_bin + " --checkpoint " + ckpt + " --backend sim --socket /tmp/no_such_sock_99999.sock");
        ASSERT_FALSE(r3.signaled);
        ASSERT_EQ(r3.exit_code, 1);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-007") {
        TestContext ctx(opts, tid, "repeated_start_stop_determinism",
                        "software/src/inference/avid_daemon.cpp", "main (10x sequential restarts)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Spawn avid_daemon 10 times sequentially with identical CLI arguments and seed";
        ctx.record().independent_oracle = "All 10 runs exit 0 and produce bit-identical JSON output (zero cross-run state leakage)";

        std::string ref_json;
        for (int run = 0; run < 10; ++run) {
            std::string mp = "/tmp/l3_exec007_" + std::to_string(::getpid()) + ".json";
            SubprocessResult r = run_command_capture(
                opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
                " --checkpoint " + ckpt + " --metrics-out " + mp);
            ASSERT_EQ(r.exit_code, 0);
            std::string body = strip_wallclock_fields(read_text_file_unlink(mp));
            if (run == 0) {
                ref_json = body;
            } else {
                ASSERT_EQ(body, ref_json);
            }
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-008") {
        TestContext ctx(opts, tid, "long_running_daemon_50k_rounds_rss_stability",
                        "software/src/inference/avid_daemon.cpp", "main (50,000 rounds)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run avid_daemon for 50,000 rounds (~195 epochs) vs 1,024 rounds and compare peak RSS";
        ctx.record().independent_oracle = "Exit 0, all 50,000 rounds decoded, and RSS growth < 16 MB (zero unbounded memory leaks)";

        std::string mp_short = "/tmp/l3_exec008_s_" + std::to_string(::getpid()) + ".json";
        std::string mp_long  = "/tmp/l3_exec008_l_" + std::to_string(::getpid()) + ".json";
        long rss_short = measure_maxrss_kb(
            opts.daemon_bin + " --scenario stationary_noise --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp_short);
        long rss_long = measure_maxrss_kb(
            opts.daemon_bin + " --scenario stationary_noise --rounds 50000 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp_long);
        std::string bs = read_text_file_unlink(mp_short);
        std::string bl = read_text_file_unlink(mp_long);

        ctx.log("rss_1024_kb=" + std::to_string(rss_short) + " rss_50000_kb=" + std::to_string(rss_long));
        ASSERT_TRUE(rss_short > 0 && rss_long > 0);
        ASSERT_TRUE((rss_long - rss_short) < 16384L);
        ASSERT_EQ(static_cast<int>(extract_json_num(bl, "decoded_rtl_transactions")), 50000);
        ASSERT_TRUE(bl.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-009") {
        TestContext ctx(opts, tid, "multiple_runtime_cycles_20_epochs_ledger",
                        "software/src/inference/avid_daemon.cpp", "main (20 epochs)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run 20 slow adaptation epochs (--rounds 5120 --window 256) checking commit ledger and RMSE";
        ctx.record().independent_oracle = "epochs_completed==20, weight_bank_commits == 1 + adaptive_epoch_commits, finite param_21_rmse";

        std::string mp = "/tmp/l3_exec009_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario slowly_drifting_noise --rounds 5120 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        int ep = static_cast<int>(extract_json_num(body, "epochs_completed"));
        int init_c = static_cast<int>(extract_json_num(body, "initial_boot_commits"));
        int adap_c = static_cast<int>(extract_json_num(body, "adaptive_epoch_commits"));
        int tot_c  = static_cast<int>(extract_json_num(body, "weight_bank_commits"));
        ASSERT_EQ(ep, 20);
        ASSERT_EQ(tot_c, init_c + adap_c);
        ASSERT_FINITE(extract_json_num(body, "param_21_rmse"));
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
