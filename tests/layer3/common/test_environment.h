#pragma once
#include <string>

namespace avid::layer3_test {

struct EnvironmentMetadata {
    std::string repo_root;
    std::string validation_root;
    std::string source_commit;
    std::string frozen_baseline_commit;
    std::string compiler_id;
    std::string os_info;
    std::string build_config;
    bool asan_enabled = false;
    bool ubsan_enabled = false;
};

EnvironmentMetadata collect_environment(const std::string &repo_root_override = "",
                                        const std::string &validation_root_override = "");

} // namespace avid::layer3_test
