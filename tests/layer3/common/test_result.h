#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace avid::layer3_test {

enum class TestStatus {
    PASS,
    FAIL,
    BLOCKED,
    INCONCLUSIVE,
    NOT_IMPLEMENTED,
    CONTRACT_MISMATCH
};

std::string status_to_string(TestStatus st);
bool is_valid_status_string(const std::string &s);

struct TestResultRecord {
    std::string test_id;
    std::string test_name;
    std::string test_description;
    std::string test_set = "set1_input_contracts";
    TestStatus status = TestStatus::INCONCLUSIVE;
    std::string source_commit;
    std::string implementation_file;
    std::string implementation_function;
    std::string input_contract;
    std::string independent_oracle;
    std::string execution_command;
    std::string fixture;
    std::map<std::string, std::string> expected;
    std::map<std::string, std::string> actual;
    std::vector<std::string> assertion_notes;
    std::vector<std::string> evidence;
    std::string failure_classification;
    double duration_ms = 0.0;
};

std::string json_escape(const std::string &s);
std::string serialize_result_json(const TestResultRecord &rec);
bool write_result_and_log(const TestResultRecord &rec,
                          const std::string &validation_root,
                          const std::string &set_subdir,
                          const std::string &log_body);

} // namespace avid::layer3_test
