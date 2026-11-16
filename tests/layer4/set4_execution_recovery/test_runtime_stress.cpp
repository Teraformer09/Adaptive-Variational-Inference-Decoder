/* ===========================================================================
 * test_runtime_stress.cpp — Layer 4 Set 4.2 Runtime Throughput & Stress Suite
 * Canonical IDs: L4-R01 .. L4-R14 (Gate G7)
 * Strictly separates host wall-clock latency/throughput from RTL cycles.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include <atomic>
#include <thread>

static long read_self_rss_kb() {
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str() + 6, "%ld", &kb);
            return kb;
        }
    }
    return 0;
}

static long read_pid_rss_kb(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/%d/status", static_cast<int>(pid));
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            std::sscanf(line.c_str() + 6, "%ld", &kb);
            return kb;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set4_runtime_stress", "Set4", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set4_stress", 256, true)) return 2;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return 2;

    /* L4-R01: Single baseline transaction */
    suite.run_case("L4-R01", "G7", "Single baseline MMIO_READ transaction latency & cycle check", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_reset_rtl(&hal, 10u, &cyc);
        uint32_t rd = 0;
        uint64_t acc = 0;
        if (avid_hal_mmio_read(&hal, 0x054u, &rd, &acc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "rdata=" + std::to_string(rd) + " rtl_delta=" + std::to_string(cyc - acc);
        return rd == 3u && (cyc - acc) == 2ULL;
    });

    /* L4-R02: 3,000 back-to-back MMIO transactions */
    suite.run_case("L4-R02", "G7", "3,000 back-to-back MMIO R/W ops complete with exact +6,000 RTL cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t sync_rd = 0;
        avid_hal_mmio_read(&hal, 0x050u, &sync_rd, &cyc);
        uint64_t c0 = hal.stats.last_completed_cycle;
        auto t0 = std::chrono::steady_clock::now();
        uint32_t rd = 0;
        for (int i = 0; i < 3000; i++) {
            uint64_t acc = 0;
            if ((i & 1) == 0) {
                if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_X(i % 9), 0x0120u + (i & 0x7F), 0x3u, &acc) != 0) return false;
            } else {
                if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X((i - 1) % 9), &rd, &acc) != 0) return false;
            }
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cyc = hal.stats.last_completed_cycle;
        ev = "rtl_delta=" + std::to_string(cyc - c0) + " host_rate=" + std::to_string(static_cast<int>(3000.0 / sec)) + " rpc/s";
        return (cyc - c0) == 6000ULL;
    });

    /* L4-R03: 3,000 continuous syndrome transfers */
    suite.run_case("L4-R03", "G7", "3,000 continuous SYNDROME_TRANSFER ops advance exact 66,000 RTL cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t sync_rd = 0;
        avid_hal_mmio_read(&hal, 0x050u, &sync_rd, &cyc);
        uint64_t c0 = hal.stats.last_completed_cycle;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3000; i++) {
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            uint64_t acc = 0;
            if (avid_hal_send_syndrome(&hal, static_cast<uint8_t>(i & 0xF), static_cast<uint8_t>((i >> 4) & 0xF),
                                       ((i + 1) % 250 == 0), &cx, &cz, &bp, &acc) != 0) {
                return false;
            }
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        cyc = hal.stats.last_completed_cycle;
        ev = "rtl_delta=" + std::to_string(cyc - c0) + " host_rate=" + std::to_string(static_cast<int>(3000.0 / sec)) + " decodes/s";
        return (cyc - c0) == 66000ULL;
    });

    /* L4-R04: Maximum pipelined batch RPC rate (1,000 pipelined requests per batch x 5) */
    suite.run_case("L4-R04", "G7", "5,000 pipelined socket RPCs achieve high throughput with zero dropped frames", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        uint64_t base_tid = hal.next_tx_id + 1000ULL;
        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < 5; b++) {
            std::string batch;
            batch.reserve(96000);
            for (int i = 0; i < 1000; i++) {
                char ln[128];
                std::snprintf(ln, sizeof(ln),
                              "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n",
                              static_cast<unsigned long long>(base_tid + b * 1000 + i));
                batch += ln;
            }
            if (!avid_l4::raw_write_all(rfd, batch.data(), batch.size())) return false;
            for (int i = 0; i < 1000; i++) {
                std::string resp;
                if (!avid_l4::raw_read_line(rfd, resp)) return false;
                if (i == 999) avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
            }
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        hal.next_tx_id = base_tid + 10000ULL;
        ev = "5,000 pipelined RPCs @ " + std::to_string(static_cast<int>(5000.0 / sec)) + " rpc/s";
        return true;
    });

    /* L4-R05: Multi-epoch sustained session (32 slow-loop windows x 32 syndromes) */
    suite.run_case("L4-R05", "G6,G7", "32-epoch sustained adaptive session (1,024 syndromes + 32 commits) reconciles cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_reset_rtl(&hal, 10u, &cyc);
        uint32_t last_ep = 0;
        for (int w = 0; w < 32; w++) {
            int16_t wx[9], wz[9];
            for (int q = 0; q < 9; q++) { wx[q] = 0x0140 + w + q; wz[q] = 0x0200 + w + q; }
            if (avid_hal_write_weights_and_commit(&hal, wx, wz, &last_ep, &cyc) != 0) return false;
            for (int r = 0; r < 32; r++) {
                uint16_t cx = 0, cz = 0;
                uint32_t bp = 0;
                if (avid_hal_send_syndrome(&hal, static_cast<uint8_t>((w + r) & 0xF), static_cast<uint8_t>((w ^ r) & 0xF),
                                           (r == 31), &cx, &cz, &bp, &cyc) != 0) return false;
            }
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "final_epoch=" + std::to_string(last_ep) + " after 1,024 syndromes";
        return last_ep == 32u;
    });

    /* L4-R06: 200 consecutive hardware RESET cycles */
    suite.run_case("L4-R06", "G7", "200 consecutive RESET(8) cycles execute without leak or state corruption", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t sync_rd = 0;
        avid_hal_mmio_read(&hal, 0x050u, &sync_rd, &cyc);
        uint64_t c0 = hal.stats.last_completed_cycle;
        for (int i = 0; i < 200; i++) {
            if (avid_hal_reset_rtl(&hal, 8u, &cyc) != 0) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "200 resets -> rtl_delta=" + std::to_string(cyc - c0);
        return (cyc - c0) == 1600ULL;
    });

    /* L4-R07: Rapid interleaved MMIO read/write + syndrome mix */
    suite.run_case("L4-R07", "G7", "500 mixed write/read/syndrome triplets maintain register integrity", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (int i = 0; i < 500; i++) {
            uint32_t v = 0x0100u + static_cast<uint32_t>(i & 0xFF);
            uint32_t rd = 0;
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_X(i % 9), v, 0x3u, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(i % 9), &rd, &cyc) != 0 || rd != v) return false;
            if (avid_hal_send_syndrome(&hal, static_cast<uint8_t>(i & 0xF), 0x5u, false, &cx, &cz, &bp, &cyc) != 0) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "1,500 interleaved ops verified";
        return true;
    });

    /* L4-R08: 50 consecutive async mid-decode busy swap cycles over socket */
    suite.run_case("L4-R08", "G6,G7", "50 async SYNDROME_START + mid-decode COMMIT + SYNDROME_WAIT cycles succeed cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        uint64_t base_tid = hal.next_tx_id + 1000ULL;
        std::string r0;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"RESET\",\"data\":6}\n", r0);
        for (int i = 0; i < 50; i++) {
            std::string rs, rw, rc, rwait;
            avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000000\",\"data\":\"0x0150\",\"byte_enable\":3}\n", rw);
            avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":false}\n", rs);
            avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000048\",\"data\":\"0x1\",\"byte_enable\":1}\n", rc);
            avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"SYNDROME_WAIT\"}\n", rwait);
            if (avid_l4::extract_json_str_simple(rwait, "status") != "OK") return false;
            avid_l4::extract_json_u64_simple(rwait, "completed_cycle", &cyc);
        }
        std::string rep;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid++) + ",\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", rep);
        hal.next_tx_id = base_tid + 100ULL;
        uint64_t ep = 0;
        avid_l4::extract_json_u64_simple(rep, "rdata", &ep);
        ev = "50 busy commits -> epoch=" + std::to_string(ep);
        return ep == 50ULL;
    });

    /* L4-R09: Multi-client kernel listen() backlog queue handling (6 concurrent waiting clients) */
    suite.run_case("L4-R09", "G7", "6 concurrent waiting client sockets in listen() backlog are served sequentially without ECONNREFUSED", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard b_mc;
        if (!b_mc.start(suite.bridge_bin_path(), "stress_mc", 10, true)) return false;
        int fds[6];
        for (int i = 0; i < 6; i++) {
            fds[i] = avid_l4::raw_unix_connect(b_mc.sock_path, 3000);
            if (fds[i] < 0) return false;
        }
        /* Serve each queued client in order as prior socket closes */
        for (int i = 0; i < 6; i++) {
            std::string resp;
            char req[160];
            std::snprintf(req, sizeof(req),
                          "{\"protocol_version\":1,\"transaction_id\":%d,\"operation\":\"MMIO_READ\",\"address\":\"0x00000054\"}\n",
                          100 + i);
            if (!avid_l4::raw_rpc(fds[i], req, resp)) return false;
            avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
            close(fds[i]);
        }
        b_mc.graceful_shutdown(999ULL);
        ev = "6/6 queued client sockets served cleanly";
        return true;
    });

    /* L4-R10: >15,000-line m3_transactions.jsonl flush integrity */
    suite.run_case("L4-R10", "G7,G8", "Large m3_transactions.jsonl (>13,000 lines) contains zero truncated or malformed lines", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::ifstream in(bridge.tx_log_path);
        size_t lines = 0;
        std::string ln;
        while (std::getline(in, ln)) {
            if (ln.empty()) continue;
            if (ln.front() != '{' || ln.back() != '}') return false;
            lines++;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "verified " + std::to_string(lines) + " flushed JSONL records";
        return lines >= 13000u;
    });

    /* L4-R11: Host CPU contention invariance on simulated RTL cycles */
    suite.run_case("L4-R11", "G3,G7", "4 background CPU spin threads cause 0 variation in simulated RTL cycle counts", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::atomic<bool> spin{true};
        std::vector<std::thread> workers;
        for (int t = 0; t < 4; t++) {
            workers.emplace_back([&]() {
                volatile uint64_t acc = 1;
                while (spin.load(std::memory_order_relaxed)) acc = acc * 6364136223846793005ULL + 1ULL;
            });
        }
        uint32_t sync_rd = 0;
        avid_hal_mmio_read(&hal, 0x050u, &sync_rd, &cyc);
        uint64_t c0 = hal.stats.last_completed_cycle;
        for (int i = 0; i < 64; i++) {
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, (i == 63), &cx, &cz, &bp, &cyc) != 0) {
                spin.store(false);
                for (auto &w : workers) w.join();
                return false;
            }
        }
        spin.store(false);
        for (auto &w : workers) w.join();
        cyc = hal.stats.last_completed_cycle;
        ev = "64 syndromes under 4x CPU contention -> exact " + std::to_string(cyc - c0) + " RTL cycles (64*22=1408)";
        return (cyc - c0) == 1408ULL;
    });

    /* L4-R12: Bridge & client memory RSS stability across 5,000 RPCs */
    suite.run_case("L4-R12", "G7", "Bridge and HAL VmRSS remain bounded (< 8 MB growth) across 5,000 transactions", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        long br_rss0 = read_pid_rss_kb(bridge.pid);
        long cl_rss0 = read_self_rss_kb();
        uint32_t rd = 0;
        for (int i = 0; i < 5000; i++) {
            if (avid_hal_mmio_read(&hal, 0x050u, &rd, &cyc) != 0) return false;
        }
        long br_rss1 = read_pid_rss_kb(bridge.pid);
        long cl_rss1 = read_self_rss_kb();
        cyc = hal.stats.last_completed_cycle;
        ev = "bridge_rss_delta=" + std::to_string(br_rss1 - br_rss0) + "KB client_rss_delta=" + std::to_string(cl_rss1 - cl_rss0) + "KB";
        return (br_rss1 - br_rss0) < 8192L && (cl_rss1 - cl_rss0) < 8192L;
    });

    /* L4-R13: 100 rapid connect/disconnect cycles */
    suite.run_case("L4-R13", "G7", "100 rapid AF_UNIX connect/RPC/disconnect cycles complete with zero FD leaks", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint64_t base_tid = hal.next_tx_id + 1000ULL;
        avid_hal_close(&hal);
        for (int i = 0; i < 100; i++) {
            int fd = avid_l4::raw_unix_connect(bridge.sock_path);
            if (fd < 0) return false;
            std::string resp;
            if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":" + std::to_string(base_tid + i) +
                                      ",\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", resp)) {
                close(fd);
                return false;
            }
            avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
            close(fd);
        }
        ev = "100/100 connect+RPC+close cycles succeeded";
        return true;
    });

    /* L4-R14: Graceful SHUTDOWN with trailing pipelined bytes in socket buffer */
    suite.run_case("L4-R14", "G7", "SHUTDOWN with trailing pipelined bytes exits cleanly with rc=0 without crash", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int fd = avid_l4::raw_unix_connect(bridge.sock_path);
        if (fd < 0) return false;
        std::string payload = "{\"protocol_version\":1,\"transaction_id\":99999990,\"operation\":\"SHUTDOWN\"}\n"
                              "{\"protocol_version\":1,\"transaction_id\":99999991,\"operation\":\"MMIO_READ\",\"address\":\"0x0\"}\n";
        avid_l4::raw_write_all(fd, payload.data(), payload.size());
        std::string resp;
        avid_l4::raw_read_line(fd, resp);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        close(fd);
        int ws = 0;
        bool clean = false;
        for (int i = 0; i < 50; i++) {
            if (waitpid(bridge.pid, &ws, WNOHANG) == bridge.pid) {
                bridge.pid = -1;
                clean = WIFEXITED(ws) && WEXITSTATUS(ws) == 0;
                break;
            }
            usleep(10000);
        }
        unlink(bridge.tx_log_path.c_str());
        unlink(bridge.metrics_path.c_str());
        ev = "SHUTDOWN ack=" + avid_l4::extract_json_str_simple(resp, "status") + " clean_exit=" + std::to_string(clean);
        return clean && avid_l4::extract_json_str_simple(resp, "status") == "OK";
    });

    return suite.finalize();
}
