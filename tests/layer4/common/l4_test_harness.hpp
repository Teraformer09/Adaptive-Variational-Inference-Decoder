#pragma once
/* ===========================================================================
 * l4_test_harness.hpp — Audit-Grade Layer 4 Test Execution & Evidence Harness
 * Records machine-readable per-ID JSONL results with strict separation of
 * host wall-clock timing (wall_time_ms) vs simulated RTL cycles.
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

#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace avid_l4 {

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

struct L4CaseRecord {
    std::string test_id;
    std::string set_id;
    std::string gates;
    std::string title;
    std::string status; /* PASS | FAIL | BLOCKED | NOT_IMPLEMENTED | CONTRACT_MISMATCH */
    double      wall_time_ms = 0.0;
    uint64_t    rtl_cycles   = 0ULL;
    std::string simulator_mode;
    std::string evidence;
};

class L4Suite {
public:
    std::string suite_name;
    std::string set_id;
    std::string out_jsonl_path;
    std::string repo_root;
    std::vector<L4CaseRecord> records;

    L4Suite(const std::string &name, const std::string &sid, int argc, char **argv)
        : suite_name(name), set_id(sid) {
        signal(SIGPIPE, SIG_IGN);
        const char *env_root = std::getenv("AVID_L4_REPO_ROOT");
        repo_root = env_root ? env_root : ".";
        for (int i = 1; i < argc; i++) {
            if (std::strcmp(argv[i], "--out-jsonl") == 0 && i + 1 < argc) {
                out_jsonl_path = argv[++i];
            } else if (std::strcmp(argv[i], "--repo-root") == 0 && i + 1 < argc) {
                repo_root = argv[++i];
            }
        }
    }

    std::string bridge_bin_path() const {
        const char *env_b = std::getenv("AVID_L4_BRIDGE_BIN");
        if (env_b && env_b[0] != '\0') return env_b;
        return repo_root + "/emulator/verilator/models/avid_cosim_bridge";
    }

    template <typename Fn>
    void run_case(const std::string &id,
                  const std::string &gates,
                  const std::string &title,
                  const std::string &sim_mode,
                  Fn &&fn) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t rtl_cyc = 0ULL;
        std::string ev;
        std::string status = "PASS";
        try {
            bool ok = fn(rtl_cyc, ev);
            status = ok ? "PASS" : "FAIL";
        } catch (const std::exception &ex) {
            status = "FAIL";
            ev = std::string("Unhandled exception: ") + ex.what();
        } catch (...) {
            status = "FAIL";
            ev = "Unhandled non-std exception";
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        records.push_back({id, set_id, gates, title, status, ms, rtl_cyc, sim_mode, ev});
        std::printf("[%s] %-14s %-6s (%.2f ms, rtl_cyc=%llu) %s\n",
                    suite_name.c_str(), id.c_str(), status.c_str(), ms,
                    static_cast<unsigned long long>(rtl_cyc), ev.c_str());
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
                    std::fprintf(fp,
                                 "{\"test_id\":\"%s\",\"suite\":\"%s\",\"set_id\":\"%s\","
                                 "\"gates\":\"%s\",\"title\":\"%s\",\"status\":\"%s\","
                                 "\"wall_time_ms\":%.4f,\"rtl_cycles\":%llu,"
                                 "\"simulator_mode\":\"%s\",\"evidence\":\"%s\"}\n",
                                 json_escape(r.test_id).c_str(),
                                 json_escape(suite_name).c_str(),
                                 json_escape(r.set_id).c_str(),
                                 json_escape(r.gates).c_str(),
                                 json_escape(r.title).c_str(),
                                 json_escape(r.status).c_str(),
                                 r.wall_time_ms,
                                 static_cast<unsigned long long>(r.rtl_cycles),
                                 json_escape(r.simulator_mode).c_str(),
                                 json_escape(r.evidence).c_str());
                }
                std::fflush(fp);
                std::fclose(fp);
            }
        }
        return (fails == 0) ? 0 : 1;
    }
};

inline std::string make_temp_path(const std::string &tag, const std::string &ext) {
    static unsigned counter = 0;
    counter++;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/avid_l4_%s_%d_%u%s",
                  tag.c_str(), static_cast<int>(getpid()), counter, ext.c_str());
    return std::string(buf);
}

inline int raw_unix_connect(const std::string &sock_path, int timeout_ms = 2000) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path.c_str());
    if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

inline bool raw_write_all(int fd, const char *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

inline bool raw_read_line(int fd, std::string &out_line, size_t max_bytes = 8192) {
    out_line.clear();
    char c = 0;
    while (out_line.size() < max_bytes) {
        ssize_t n = read(fd, &c, 1);
        if (n <= 0) break;
        out_line.push_back(c);
        if (c == '\n') return true;
    }
    return !out_line.empty();
}

inline bool raw_rpc(int fd, const std::string &req_line, std::string &resp_line) {
    if (!raw_write_all(fd, req_line.data(), req_line.size())) return false;
    return raw_read_line(fd, resp_line);
}

class BridgeProcessGuard {
public:
    pid_t       pid = -1;
    std::string sock_path;
    std::string tx_log_path;
    std::string metrics_path;

    BridgeProcessGuard() = default;

    bool start(const std::string &bridge_bin,
               const std::string &tag,
               int max_clients = 32,
               bool require_native = true) {
        sock_path    = make_temp_path(tag, ".sock");
        tx_log_path  = make_temp_path(tag, "_m3.jsonl");
        metrics_path = make_temp_path(tag, "_metrics.json");
        unlink(sock_path.c_str());
        unlink(tx_log_path.c_str());
        unlink(metrics_path.c_str());

        std::string max_c_str = std::to_string(max_clients);
        pid = fork();
        if (pid < 0) return false;
        if (pid == 0) {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
            std::vector<const char *> args;
            args.push_back(bridge_bin.c_str());
            if (require_native) args.push_back("--require-native-rtl");
            args.push_back("--socket");
            args.push_back(sock_path.c_str());
            args.push_back("--tx-log");
            args.push_back(tx_log_path.c_str());
            args.push_back("--metrics-out");
            args.push_back(metrics_path.c_str());
            args.push_back("--max-clients");
            args.push_back(max_c_str.c_str());
            args.push_back(nullptr);
            execv(bridge_bin.c_str(), const_cast<char *const *>(args.data()));
            _exit(127);
        }

        for (int i = 0; i < 150; i++) {
            struct stat st;
            if (stat(sock_path.c_str(), &st) == 0) {
                int probe_fd = raw_unix_connect(sock_path, 500);
                if (probe_fd >= 0) {
                    close(probe_fd);
                    return true;
                }
            }
            int wstatus = 0;
            if (waitpid(pid, &wstatus, WNOHANG) == pid) {
                pid = -1;
                return false;
            }
            usleep(15000);
        }
        stop();
        return false;
    }

    void graceful_shutdown(uint64_t tx_id = 9999999ULL) {
        if (pid <= 0) return;
        int fd = raw_unix_connect(sock_path, 500);
        if (fd >= 0) {
            char req[160];
            std::snprintf(req, sizeof(req),
                          "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"SHUTDOWN\"}\n",
                          static_cast<unsigned long long>(tx_id));
            std::string resp;
            raw_rpc(fd, req, resp);
            close(fd);
        }
        for (int i = 0; i < 50; i++) {
            int st = 0;
            if (waitpid(pid, &st, WNOHANG) == pid) {
                pid = -1;
                return;
            }
            usleep(10000);
        }
        stop();
    }

    void stop(int sig = SIGTERM) {
        if (pid > 0) {
            kill(pid, sig);
            int st = 0;
            waitpid(pid, &st, 0);
            pid = -1;
        }
        if (!sock_path.empty()) unlink(sock_path.c_str());
    }

    ~BridgeProcessGuard() {
        stop();
    }
};

inline bool extract_json_u64_simple(const std::string &line, const char *key, uint64_t *out) {
    std::string pat = std::string("\"") + key + "\":";
    size_t pos = line.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos < line.size() && line[pos] == '"') {
        pos++;
        char *end = nullptr;
        unsigned long long v = std::strtoull(line.c_str() + pos, &end, 0);
        if (end == line.c_str() + pos) return false;
        *out = static_cast<uint64_t>(v);
        return true;
    }
    char *end = nullptr;
    unsigned long long v = std::strtoull(line.c_str() + pos, &end, 10);
    if (end == line.c_str() + pos) return false;
    *out = static_cast<uint64_t>(v);
    return true;
}

inline std::string extract_json_str_simple(const std::string &line, const char *key) {
    std::string pat = std::string("\"") + key + "\":\"";
    size_t pos = line.find(pat);
    if (pos == std::string::npos) return "";
    pos += pat.size();
    size_t end = line.find('"', pos);
    if (end == std::string::npos) return "";
    return line.substr(pos, end - pos);
}

} /* namespace avid_l4 */
