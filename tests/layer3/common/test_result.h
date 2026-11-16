#pragma once
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace avid::layer3_test {

struct RecordString : public std::string {
    RecordString() = default;
    RecordString(const RecordString &) = default;
    RecordString(RecordString &&) noexcept = default;
    RecordString &operator=(const RecordString &) = default;
    RecordString &operator=(RecordString &&) noexcept = default;

    RecordString(const std::string &s) : std::string(s) {}
    RecordString(std::string &&s) noexcept : std::string(std::move(s)) {}
    RecordString(bool b) : std::string(b ? "true" : "false") {}
    RecordString(char c) : std::string(1, c) {}

    template <typename T,
              typename std::enable_if_t<std::is_same_v<std::decay_t<T>, const char *> ||
                                            std::is_same_v<std::decay_t<T>, char *>,
                                        int> = 0>
    RecordString(T s) : std::string(s ? s : "") {}

    template <typename T,
              typename std::enable_if_t<std::is_integral_v<std::decay_t<T>> &&
                                            !std::is_same_v<std::decay_t<T>, bool> &&
                                            !std::is_same_v<std::decay_t<T>, char>,
                                        int> = 0>
    RecordString(T v) : std::string(std::to_string(v)) {}

    template <typename T,
              typename std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, int> = 0>
    RecordString(T v) {
        std::ostringstream oss;
        oss << v;
        assign(oss.str());
    }

    RecordString &operator=(const std::string &s) {
        assign(s);
        return *this;
    }
    RecordString &operator=(std::string &&s) noexcept {
        assign(std::move(s));
        return *this;
    }
    RecordString &operator=(bool b) {
        assign(b ? "true" : "false");
        return *this;
    }
    RecordString &operator=(char c) {
        assign(1, c);
        return *this;
    }

    template <typename T,
              typename std::enable_if_t<std::is_same_v<std::decay_t<T>, const char *> ||
                                            std::is_same_v<std::decay_t<T>, char *>,
                                        int> = 0>
    RecordString &operator=(T s) {
        assign(s ? s : "");
        return *this;
    }

    template <typename T,
              typename std::enable_if_t<std::is_integral_v<std::decay_t<T>> &&
                                            !std::is_same_v<std::decay_t<T>, bool> &&
                                            !std::is_same_v<std::decay_t<T>, char>,
                                        int> = 0>
    RecordString &operator=(T v) {
        assign(std::to_string(v));
        return *this;
    }

    template <typename T,
              typename std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, int> = 0>
    RecordString &operator=(T v) {
        std::ostringstream oss;
        oss << v;
        assign(oss.str());
        return *this;
    }
};

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
    std::map<std::string, RecordString> expected;
    std::map<std::string, RecordString> actual;
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
