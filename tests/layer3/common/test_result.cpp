#include "test_result.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace avid::layer3_test {

std::string status_to_string(TestStatus st) {
    switch (st) {
        case TestStatus::PASS:              return "PASS";
        case TestStatus::FAIL:              return "FAIL";
        case TestStatus::BLOCKED:           return "BLOCKED";
        case TestStatus::INCONCLUSIVE:      return "INCONCLUSIVE";
        case TestStatus::NOT_IMPLEMENTED:   return "NOT_IMPLEMENTED";
        case TestStatus::CONTRACT_MISMATCH: return "CONTRACT_MISMATCH";
    }
    return "INCONCLUSIVE";
}

bool is_valid_status_string(const std::string &s) {
    return s == "PASS" || s == "FAIL" || s == "BLOCKED" ||
           s == "INCONCLUSIVE" || s == "NOT_IMPLEMENTED" ||
           s == "CONTRACT_MISMATCH";
}

std::string json_escape(const std::string &s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b";  break;
            case '\f': oss << "\\f";  break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c << std::dec;
                } else {
                    oss << (char)c;
                }
        }
    }
    return oss.str();
}

static void write_kv_map(std::ostringstream &oss, const std::map<std::string, RecordString> &m, int indent) {
    std::string pad(indent, ' ');
    std::string pad_inner(indent + 2, ' ');
    oss << "{\n";
    size_t idx = 0;
    for (const auto &kv : m) {
        oss << pad_inner << "\"" << json_escape(kv.first) << "\": \"" << json_escape(kv.second) << "\"";
        if (++idx < m.size()) oss << ",";
        oss << "\n";
    }
    oss << pad << "}";
}

std::string serialize_result_json(const TestResultRecord &rec) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"test_id\": \"" << json_escape(rec.test_id) << "\",\n";
    oss << "  \"test_name\": \"" << json_escape(rec.test_name) << "\",\n";
    oss << "  \"test_description\": \"" << json_escape(rec.test_description) << "\",\n";
    oss << "  \"test_set\": \"" << json_escape(rec.test_set) << "\",\n";
    oss << "  \"status\": \"" << status_to_string(rec.status) << "\",\n";
    oss << "  \"source_commit\": \"" << json_escape(rec.source_commit) << "\",\n";
    oss << "  \"implementation_file\": \"" << json_escape(rec.implementation_file) << "\",\n";
    oss << "  \"implementation_function\": \"" << json_escape(rec.implementation_function) << "\",\n";
    oss << "  \"input_contract\": \"" << json_escape(rec.input_contract) << "\",\n";
    oss << "  \"independent_oracle\": \"" << json_escape(rec.independent_oracle) << "\",\n";
    oss << "  \"execution_command\": \"" << json_escape(rec.execution_command) << "\",\n";
    oss << "  \"fixture\": \"" << json_escape(rec.fixture) << "\",\n";
    oss << "  \"expected\": ";
    write_kv_map(oss, rec.expected, 2);
    oss << ",\n";
    oss << "  \"actual\": ";
    write_kv_map(oss, rec.actual, 2);
    oss << ",\n";
    oss << "  \"assertion_notes\": [";
    for (size_t i = 0; i < rec.assertion_notes.size(); ++i) {
        if (i == 0) oss << "\n";
        oss << "    \"" << json_escape(rec.assertion_notes[i]) << "\"";
        if (i + 1 < rec.assertion_notes.size()) oss << ",";
        oss << "\n";
    }
    if (!rec.assertion_notes.empty()) oss << "  ";
    oss << "],\n";
    oss << "  \"evidence\": [";
    for (size_t i = 0; i < rec.evidence.size(); ++i) {
        if (i == 0) oss << "\n";
        oss << "    \"" << json_escape(rec.evidence[i]) << "\"";
        if (i + 1 < rec.evidence.size()) oss << ",";
        oss << "\n";
    }
    if (!rec.evidence.empty()) oss << "  ";
    oss << "],\n";
    oss << "  \"duration_ms\": " << std::fixed << std::setprecision(3) << rec.duration_ms << ",\n";
    if (rec.failure_classification.empty()) {
        oss << "  \"failure_classification\": null\n";
    } else {
        oss << "  \"failure_classification\": \"" << json_escape(rec.failure_classification) << "\"\n";
    }
    oss << "}\n";
    return oss.str();
}

bool write_result_and_log(const TestResultRecord &rec,
                          const std::string &validation_root,
                          const std::string &set_subdir,
                          const std::string &log_body) {
    namespace fs = std::filesystem;
    fs::path raw_dir = fs::path(validation_root) / "raw_results" / set_subdir;
    fs::path log_dir = fs::path(validation_root) / "logs" / set_subdir;
    std::error_code ec;
    fs::create_directories(raw_dir, ec);
    fs::create_directories(log_dir, ec);

    std::string suffix;
    if (const char *sfx = std::getenv("AVID_RESULT_SUFFIX")) {
        if (sfx[0] != '\0') suffix = std::string("_") + sfx;
    }

    fs::path log_path = log_dir / (rec.test_id + suffix + ".log");
    {
        std::ofstream lf(log_path);
        if (!lf) return false;
        lf << "=== AVID Layer 3 Test Execution Log ===\n"
           << "Test ID: " << rec.test_id << "\n"
           << "Test Name: " << rec.test_name << "\n"
           << "Status: " << status_to_string(rec.status) << "\n"
           << "Source Commit: " << rec.source_commit << "\n"
           << "Implementation File: " << rec.implementation_file << "\n"
           << "Implementation Function: " << rec.implementation_function << "\n"
           << "Execution Command: " << rec.execution_command << "\n"
           << "Fixture: " << rec.fixture << "\n"
           << "Failure Classification: " << (rec.failure_classification.empty() ? "NONE" : rec.failure_classification) << "\n"
           << "----------------------------------------\n"
           << log_body << "\n";
    }

    if (suffix.empty()) {
        fs::path json_path = raw_dir / (rec.test_id + ".json");
        std::ofstream jf(json_path);
        if (!jf) return false;
        jf << serialize_result_json(rec);
    } else {
        fs::path json_path = raw_dir / (rec.test_id + suffix + ".json");
        std::ofstream jf(json_path);
        if (!jf) return false;
        jf << serialize_result_json(rec);
    }
    return true;
}

} // namespace avid::layer3_test
