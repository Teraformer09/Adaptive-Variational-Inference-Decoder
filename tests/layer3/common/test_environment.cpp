#include "test_environment.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/utsname.h>

namespace avid::layer3_test {

static std::string run_git_query(const std::string &repo_root, const std::string &args) {
    std::string cmd = "git -C \"" + repo_root + "\" " + args + " 2>/dev/null";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return "unknown";
    char buf[256] = {0};
    std::string out;
    if (fgets(buf, sizeof(buf), fp)) {
        out = buf;
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
            out.pop_back();
        }
    }
    pclose(fp);
    return out.empty() ? "unknown" : out;
}

EnvironmentMetadata collect_environment(const std::string &repo_root_override,
                                        const std::string &validation_root_override) {
    EnvironmentMetadata env;
    const char *env_repo = std::getenv("AVID_REPO_ROOT");
    if (!repo_root_override.empty()) {
        env.repo_root = repo_root_override;
    } else if (env_repo && env_repo[0] != '\0') {
        env.repo_root = env_repo;
    } else {
#ifdef AVID_DEFAULT_REPO_ROOT
        env.repo_root = AVID_DEFAULT_REPO_ROOT;
#else
        env.repo_root = ".";
#endif
    }

    const char *env_val = std::getenv("AVID_VALIDATION_DIR");
    if (!validation_root_override.empty()) {
        env.validation_root = validation_root_override;
    } else if (env_val && env_val[0] != '\0') {
        env.validation_root = env_val;
    } else {
        env.validation_root = env.repo_root + "/validation/layer3";
    }

    const char *baseline_override = std::getenv("AVID_SOURCE_COMMIT");
    if (baseline_override && baseline_override[0] != '\0') {
        env.source_commit = baseline_override;
    } else {
        env.source_commit = run_git_query(env.repo_root, "rev-parse --short=7 HEAD");
    }
    env.frozen_baseline_commit = run_git_query(env.repo_root, "rev-parse --short=7 origin/AVID_EPJ_Patches");

#ifdef __VERSION__
    env.compiler_id = std::string("GCC/Clang ") + __VERSION__;
#else
    env.compiler_id = "C++17 compiler";
#endif

    struct utsname un {};
    if (uname(&un) == 0) {
        env.os_info = std::string(un.sysname) + " " + un.release + " " + un.machine;
    } else {
        env.os_info = "Linux";
    }

#if defined(AVID_BUILD_ASAN) && AVID_BUILD_ASAN
    env.asan_enabled = true;
#endif
#if defined(AVID_BUILD_UBSAN) && AVID_BUILD_UBSAN
    env.ubsan_enabled = true;
#endif

    if (env.asan_enabled && env.ubsan_enabled) {
        env.build_config = "ASan+UBSan";
    } else if (env.asan_enabled) {
        env.build_config = "ASan";
    } else if (env.ubsan_enabled) {
        env.build_config = "UBSan";
    } else {
        env.build_config = "Standard";
    }
    return env;
}

} // namespace avid::layer3_test
