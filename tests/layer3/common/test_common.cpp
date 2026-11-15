#include "test_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace avid::layer3_test {

CliOptions parse_cli_options(int argc, char **argv, const std::string &default_test_id) {
    CliOptions o;
    o.test_id = default_test_id;
    if (const char *db = std::getenv("AVID_DAEMON_BIN")) {
        o.daemon_bin = db;
    }
    if (const char *se = std::getenv("AVID_STRICT_EXIT")) {
        o.ctest_strict_exit = (std::strcmp(se, "1") == 0);
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test-id") == 0 && i + 1 < argc) {
            o.test_id = argv[++i];
        } else if (std::strcmp(argv[i], "--repo-root") == 0 && i + 1 < argc) {
            o.repo_root = argv[++i];
        } else if (std::strcmp(argv[i], "--validation-dir") == 0 && i + 1 < argc) {
            o.validation_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--daemon-bin") == 0 && i + 1 < argc) {
            o.daemon_bin = argv[++i];
        } else if (std::strcmp(argv[i], "--strict-exit") == 0) {
            o.ctest_strict_exit = true;
        }
    }
    return o;
}

SubprocessResult run_isolated_probe(const std::function<int()> &fn) {
    SubprocessResult res;
    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        res.exit_code = -1;
        return res;
    }

    std::fflush(nullptr);
    pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        res.exit_code = -1;
        return res;
    }
    if (pid == 0) {
        close(out_pipe[0]);
        close(err_pipe[0]);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(out_pipe[1]);
        close(err_pipe[1]);
        /* Prevent ASan from generating huge core dumps on intentional null-deref probe */
        setenv("ASAN_OPTIONS", "abort_on_error=1:disable_coredump=1:handle_segv=0", 1);
        int rc = fn();
        std::fflush(nullptr);
        _exit(rc & 0xFF);
    }

    close(out_pipe[1]);
    close(err_pipe[1]);
    char buf[512];
    ssize_t n = 0;
    while ((n = read(out_pipe[0], buf, sizeof(buf))) > 0) {
        res.stdout_text.append(buf, (size_t)n);
    }
    while ((n = read(err_pipe[0], buf, sizeof(buf))) > 0) {
        res.stderr_text.append(buf, (size_t)n);
    }
    close(out_pipe[0]);
    close(err_pipe[0]);

    int wstatus = 0;
    if (waitpid(pid, &wstatus, 0) > 0) {
        if (WIFSIGNALED(wstatus)) {
            res.signaled = true;
            res.term_signal = WTERMSIG(wstatus);
            res.exit_code = 128 + res.term_signal;
        } else if (WIFEXITED(wstatus)) {
            res.exit_code = WEXITSTATUS(wstatus);
        }
    }
    return res;
}

SubprocessResult run_command_capture(const std::string &cmd) {
    SubprocessResult res;
    std::string full_cmd = cmd + " 2>&1";
    FILE *fp = popen(full_cmd.c_str(), "r");
    if (!fp) {
        res.exit_code = -1;
        return res;
    }
    char buf[512];
    while (fgets(buf, sizeof(buf), fp)) {
        res.stdout_text += buf;
    }
    int wstatus = pclose(fp);
    if (WIFSIGNALED(wstatus)) {
        res.signaled = true;
        res.term_signal = WTERMSIG(wstatus);
        res.exit_code = 128 + res.term_signal;
    } else if (WIFEXITED(wstatus)) {
        res.exit_code = WEXITSTATUS(wstatus);
    } else {
        res.exit_code = wstatus;
    }
    return res;
}

TestContext::TestContext(const CliOptions &opts,
                         const std::string &test_id,
                         const std::string &test_name,
                         const std::string &impl_file,
                         const std::string &impl_func,
                         const std::string &set_subdir)
    : opts_(opts),
      env_(collect_environment(opts.repo_root, opts.validation_dir)),
      set_subdir_(set_subdir),
      t0_(std::chrono::steady_clock::now()) {
    rec_.test_id = test_id;
    rec_.test_name = test_name;
    rec_.test_set = (set_subdir == "set1") ? "set1_input_contracts" : set_subdir;
    rec_.source_commit = env_.source_commit;
    rec_.implementation_file = impl_file;
    rec_.implementation_function = impl_func;
    rec_.execution_command = "ctest --test-dir build/layer3 -R \"^" + test_id + "$\" --output-on-failure";
    rec_.evidence.push_back("validation/layer3/logs/" + set_subdir + "/" + test_id + ".log");
    rec_.evidence.push_back("validation/layer3/raw_results/" + set_subdir + "/" + test_id + ".json");
    log("Initialized test context: build_config=" + env_.build_config +
        " compiler=" + env_.compiler_id + " os=" + env_.os_info);
}

std::string TestContext::fixture_path(const std::string &rel) const {
    return env_.repo_root + "/tests/layer3/fixtures/" + rel;
}

void TestContext::log(const std::string &line) {
    log_stream_ << line << "\n";
}

void TestContext::record_failure(const std::string &msg) {
    failed_assertions_++;
    rec_.assertion_notes.push_back(msg);
    log_stream_ << msg << "\n";
}

int TestContext::finalize(TestStatus explicit_status, const std::string &failure_class) {
    auto t1 = std::chrono::steady_clock::now();
    rec_.duration_ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();

    if (explicit_status == TestStatus::PASS && failed_assertions_ > 0) {
        rec_.status = TestStatus::FAIL;
        rec_.failure_classification = failure_class.empty() ? "assertion_violation" : failure_class;
    } else {
        rec_.status = explicit_status;
        rec_.failure_classification = failure_class;
    }

    write_result_and_log(rec_, env_.validation_root, set_subdir_, log_stream_.str());

    std::cout << "[" << rec_.test_id << "] status=" << status_to_string(rec_.status)
              << " (" << rec_.test_name << ") duration_ms=" << rec_.duration_ms;
    if (!rec_.failure_classification.empty()) {
        std::cout << " classification=" << rec_.failure_classification;
    }
    std::cout << std::endl;

    if (opts_.ctest_strict_exit && rec_.status != TestStatus::PASS) {
        return 1;
    }
    return 0;
}

} // namespace avid::layer3_test
