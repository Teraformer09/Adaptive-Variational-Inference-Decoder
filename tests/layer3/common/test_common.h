#pragma once
#include "test_environment.h"
#include "test_result.h"
#include <chrono>
#include <functional>
#include <sstream>
#include <string>

namespace avid::layer3_test {

struct CliOptions {
    std::string test_id;
    std::string repo_root;
    std::string validation_dir;
    std::string daemon_bin;
    bool ctest_strict_exit = false;
};

CliOptions parse_cli_options(int argc, char **argv, const std::string &default_test_id = "");

struct SubprocessResult {
    int exit_code = -1;
    bool signaled = false;
    int term_signal = 0;
    std::string stdout_text;
    std::string stderr_text;
};

/* Execute a risky C/C++ function inside a forked child process so null-pointer
 * dereferences / SIGSEGV / SIGABRT are observed cleanly without aborting the parent runner. */
SubprocessResult run_isolated_probe(const std::function<int()> &fn);

/* Execute an external binary (such as avid_daemon_under_test) and capture exit code + output. */
SubprocessResult run_command_capture(const std::string &cmd);

class TestContext {
public:
    TestContext(const CliOptions &opts,
                const std::string &test_id,
                const std::string &test_name,
                const std::string &impl_file,
                const std::string &impl_func,
                const std::string &set_subdir = "set1");

    TestResultRecord &record() { return rec_; }
    const EnvironmentMetadata &env() const { return env_; }
    const CliOptions &opts() const { return opts_; }

    std::string fixture_path(const std::string &rel) const;
    void log(const std::string &line);
    void record_failure(const std::string &msg);
    bool has_assertion_failures() const { return failed_assertions_ > 0; }

    /* Finalizes status, writes raw_results/<set>/<test_id>.json & logs/<set>/<test_id>.log,
     * prints summary to stdout, and returns process exit code. */
    int finalize(TestStatus explicit_status, const std::string &failure_class = "");

private:
    CliOptions opts_;
    EnvironmentMetadata env_;
    TestResultRecord rec_;
    std::string set_subdir_;
    std::ostringstream log_stream_;
    int failed_assertions_ = 0;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace avid::layer3_test
