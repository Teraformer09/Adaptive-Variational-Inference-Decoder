#pragma once
/* ===========================================================================
 * l5_test_harness.hpp — Audit-Grade Layer 5 FPGA RTL Validation Harness
 * Executes canonical L5-* test cases against compiled native Verilator
 * Vaa_qec_top (--assert enabled) and emits structured JSONL records:
 *   {test_id, suite, set_id, gate_ids, gates, title, status,
 *    rtl_cycles, wall_time_ms, simulator_mode, details, evidence}
 * =========================================================================*/
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace avid_l5 {

inline std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

inline std::string gates_to_json_array(const std::string &gates_csv) {
    std::string out = "[";
    bool first = true;
    size_t start = 0;
    while (start < gates_csv.size()) {
        size_t comma = gates_csv.find(',', start);
        std::string tok = (comma == std::string::npos)
                              ? gates_csv.substr(start)
                              : gates_csv.substr(start, comma - start);
        while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
        while (!tok.empty() && tok.back() == ' ') tok.pop_back();
        if (!tok.empty()) {
            if (!first) out += ",";
            out += "\"" + json_escape(tok) + "\"";
            first = false;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    out += "]";
    return out;
}

struct L5CaseRecord {
    std::string test_id;
    std::string set_id;
    std::string gates;
    std::string title;
    std::string status;
    double      wall_time_ms = 0.0;
    uint64_t    rtl_cycles   = 0ULL;
    std::string simulator_mode;
    std::string details;
};

class L5Suite {
public:
    std::string suite_name;
    std::string set_id;
    std::string out_jsonl_path;
    std::string repo_root;
    std::vector<L5CaseRecord> records;

    L5Suite(const std::string &name, const std::string &sid, int argc, char **argv)
        : suite_name(name), set_id(sid) {
        const char *env_root = std::getenv("AVID_L5_REPO_ROOT");
        repo_root = env_root ? env_root : ".";
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "--out-jsonl") == 0 && i + 1 < argc) {
                out_jsonl_path = argv[++i];
            } else if (std::strcmp(argv[i], "--repo-root") == 0 && i + 1 < argc) {
                repo_root = argv[++i];
            }
        }
    }

    template <typename Fn>
    void run_case(const std::string &id,
                  const std::string &gates,
                  const std::string &title,
                  const std::string &sim_mode,
                  Fn &&fn) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t rtl_cyc = 0ULL;
        std::string det;
        std::string status = "PASS";
        try {
            bool ok = fn(rtl_cyc, det);
            status = ok ? "PASS" : "FAIL";
        } catch (const std::exception &ex) {
            status = "FAIL";
            det = std::string("Unhandled exception: ") + ex.what();
        } catch (...) {
            status = "FAIL";
            det = "Unhandled non-std exception";
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        records.push_back({id, set_id, gates, title, status, ms, rtl_cyc, sim_mode, det});
        std::printf("[%s] %-14s %-6s (%.2f ms, rtl_cyc=%llu) %s\n",
                    suite_name.c_str(), id.c_str(), status.c_str(), ms,
                    static_cast<unsigned long long>(rtl_cyc), det.c_str());
        std::fflush(stdout);
    }

    int finalize() const {
        int fails = 0;
        for (const auto &r : records) {
            if (r.status != "PASS") fails++;
        }
        if (!out_jsonl_path.empty()) {
            FILE *fp = std::fopen(out_jsonl_path.c_str(), "a");
            if (fp) {
                for (const auto &r : records) {
                    std::string gate_arr = gates_to_json_array(r.gates);
                    std::fprintf(fp,
                                 "{\"test_id\":\"%s\",\"suite\":\"%s\",\"set_id\":\"%s\","
                                 "\"gate_ids\":%s,\"gates\":\"%s\",\"title\":\"%s\","
                                 "\"status\":\"%s\",\"rtl_cycles\":%llu,\"wall_time_ms\":%.4f,"
                                 "\"simulator_mode\":\"%s\",\"details\":\"%s\",\"evidence\":\"%s\"}\n",
                                 json_escape(r.test_id).c_str(),
                                 json_escape(suite_name).c_str(),
                                 json_escape(r.set_id).c_str(),
                                 gate_arr.c_str(),
                                 json_escape(r.gates).c_str(),
                                 json_escape(r.title).c_str(),
                                 json_escape(r.status).c_str(),
                                 static_cast<unsigned long long>(r.rtl_cycles),
                                 r.wall_time_ms,
                                 json_escape(r.simulator_mode).c_str(),
                                 json_escape(r.details).c_str(),
                                 json_escape(r.details).c_str());
                }
                std::fflush(fp);
                std::fclose(fp);
            }
        }
        return (fails == 0) ? 0 : 1;
    }
};

/* Register byte-offset helpers matching weight_map_reg.sv */
static constexpr uint32_t L5_OFF_SHADOW_X(int q) { return static_cast<uint32_t>(4 * q); }
static constexpr uint32_t L5_OFF_SHADOW_Z(int q) { return static_cast<uint32_t>(0x024 + 4 * q); }
static constexpr uint32_t L5_OFF_COMMIT       = 0x048u;
static constexpr uint32_t L5_OFF_STATUS       = 0x04Cu;
static constexpr uint32_t L5_OFF_EPOCH_COUNT  = 0x050u;
static constexpr uint32_t L5_OFF_CFG_MAX_ITER = 0x054u;
static constexpr uint32_t L5_OFF_OSD_MODE     = 0x058u;
static constexpr uint32_t L5_OFF_ACTIVE_X(int q) { return static_cast<uint32_t>(0x100 + 4 * q); }
static constexpr uint32_t L5_OFF_ACTIVE_Z(int q) { return static_cast<uint32_t>(0x124 + 4 * q); }

/* GF(2) CSS parity-check matrices for d=3 rotated surface code:
 *   H_X checks Z errors: c0={0,1}, c1={1,2,4,5}, c2={3,4,6,7}, c3={7,8}
 *   H_Z checks X errors: c0={0,1,3,4}, c1={2,5}, c2={3,6}, c3={4,5,7,8}
 */
inline uint8_t gf2_hx_mul_ez(uint16_t corr_z) {
    uint8_t s0 = ((corr_z >> 0) ^ (corr_z >> 1)) & 1u;
    uint8_t s1 = ((corr_z >> 1) ^ (corr_z >> 2) ^ (corr_z >> 4) ^ (corr_z >> 5)) & 1u;
    uint8_t s2 = ((corr_z >> 3) ^ (corr_z >> 4) ^ (corr_z >> 6) ^ (corr_z >> 7)) & 1u;
    uint8_t s3 = ((corr_z >> 7) ^ (corr_z >> 8)) & 1u;
    return static_cast<uint8_t>(s0 | (s1 << 1) | (s2 << 2) | (s3 << 3));
}

inline uint8_t gf2_hz_mul_ex(uint16_t corr_x) {
    uint8_t s0 = ((corr_x >> 0) ^ (corr_x >> 1) ^ (corr_x >> 3) ^ (corr_x >> 4)) & 1u;
    uint8_t s1 = ((corr_x >> 2) ^ (corr_x >> 5)) & 1u;
    uint8_t s2 = ((corr_x >> 3) ^ (corr_x >> 6)) & 1u;
    uint8_t s3 = ((corr_x >> 4) ^ (corr_x >> 5) ^ (corr_x >> 7) ^ (corr_x >> 8)) & 1u;
    return static_cast<uint8_t>(s0 | (s1 << 1) | (s2 << 2) | (s3 << 3));
}

inline int popcount9(uint16_t mask) {
    int c = 0;
    for (int i = 0; i < 9; i++) c += (mask >> i) & 1;
    return c;
}

inline std::string make_temp_path(const std::string &tag, const std::string &ext) {
    static unsigned seq = 0;
    seq++;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/avid_l5_%s_%d_%u%s",
                  tag.c_str(), static_cast<int>(getpid()), seq, ext.c_str());
    return std::string(buf);
}

} /* namespace avid_l5 */
