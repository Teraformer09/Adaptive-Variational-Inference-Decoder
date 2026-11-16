/* ===========================================================================
 * test_fault_injection.cpp — Layer 4 Set 4.3 Fault Injection & Non-Idempotency
 * Canonical IDs: L4-FLT-01 .. L4-FLT-14 (Gate G7)
 * Verifies NO blind retry of non-idempotent SYNDROME_TRANSFER or COMMIT.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set4_fault_injection", "Set4", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set4_flt", 64, true)) return 2;

    /* L4-FLT-01: Mid-frame client disconnect (truncated JSON without '\n') */
    suite.run_case("L4-FLT-01", "G7", "Client disconnecting mid-JSON frame without newline does not hang or corrupt bridge", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f1 = avid_l4::raw_unix_connect(bridge.sock_path);
        const char *half = "{\"protocol_version\":1,\"transaction_id\":101,\"operation\":\"MMIO_WR";
        avid_l4::raw_write_all(f1, half, std::strlen(half));
        close(f1); /* abrupt EOF mid-line */
        int f2 = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string resp;
        bool ok = avid_l4::raw_rpc(f2, "{\"protocol_version\":1,\"transaction_id\":102,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", resp);
        close(f2);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        ev = "next client status=" + avid_l4::extract_json_str_simple(resp, "status");
        return ok && avid_l4::extract_json_str_simple(resp, "status") == "OK";
    });

    /* L4-FLT-02: Post-accept immediate zero-byte disconnect */
    suite.run_case("L4-FLT-02", "G7", "10 immediate zero-byte client disconnects do not wedge bridge accept loop", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (int i = 0; i < 10; i++) {
            int f = avid_l4::raw_unix_connect(bridge.sock_path);
            if (f < 0) return false;
            close(f);
        }
        int f2 = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string resp;
        bool ok = avid_l4::raw_rpc(f2, "{\"protocol_version\":1,\"transaction_id\":103,\"operation\":\"MMIO_READ\",\"address\":\"0x00000054\"}\n", resp);
        close(f2);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        ev = "survived 10 zero-byte disconnects";
        return ok && avid_l4::extract_json_str_simple(resp, "status") == "OK";
    });

    /* L4-FLT-03: Malformed JSON storm (25 corrupted frames) */
    suite.run_case("L4-FLT-03", "G7", "25 consecutive malformed JSON frames rejected with +0 RTL cycle advance", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int fd = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r0, r_end;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":110,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r0);
        uint64_t c0 = 0;
        avid_l4::extract_json_u64_simple(r0, "completed_cycle", &c0);
        for (int i = 0; i < 25; i++) {
            std::string rb;
            avid_l4::raw_rpc(fd, "{\"corrupt_frame_" + std::to_string(i) + "\n", rb);
            if (avid_l4::extract_json_str_simple(rb, "status") != "ERR_MALFORMED_JSON") { close(fd); return false; }
        }
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":111,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r_end);
        close(fd);
        uint64_t acc_end = 0;
        avid_l4::extract_json_u64_simple(r_end, "accepted_cycle", &acc_end);
        avid_l4::extract_json_u64_simple(r_end, "completed_cycle", &cyc);
        ev = "25 malformed frames advanced 0 cycles (c0=" + std::to_string(c0) + ", acc_end=" + std::to_string(acc_end) + ")";
        return acc_end == c0;
    });

    /* L4-FLT-04: Socket SO_RCVTIMEO timeout handling on stalled server (NO blind retry!) */
    suite.run_case("L4-FLT-04", "G7", "HAL enforces SO_RCVTIMEO (120 ms) against stalled server, increments timeout_transactions, and does NOT blind-retry", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string stall_sock = avid_l4::make_temp_path("flt_stall", ".sock");
        unlink(stall_sock.c_str());
        int sfd = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", stall_sock.c_str());
        bind(sfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        listen(sfd, 2);

        pid_t mpid = fork();
        if (mpid == 0) {
            int cfd = accept(sfd, nullptr, nullptr);
            if (cfd >= 0) {
                /* Count how many newline-terminated requests arrive during timeout window */
                int req_count = 0;
                struct timeval tv{0, 350000}; /* 350 ms */
                setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                std::string line;
                while (avid_l4::raw_read_line(cfd, line)) req_count++;
                close(cfd);
                close(sfd);
                _exit(req_count); /* exit code == number of requests received! */
            }
            _exit(99);
        }
        close(sfd);

        avid_hal_t mhal;
        avid_hal_open(&mhal, AVID_HAL_BACKEND_SIMULATION, stall_sock.c_str(), WRD_BASE_ADDR);
        avid_hal_set_timeout_ms(&mhal, 120);
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        int rc = avid_hal_send_syndrome(&mhal, 0x6u, 0x0u, true, &cx, &cz, &bp, &cyc);
        uint64_t tmo = mhal.stats.timeout_transactions;
        avid_hal_close(&mhal);
        int ws = 0;
        waitpid(mpid, &ws, 0);
        int server_reqs = WIFEXITED(ws) ? WEXITSTATUS(ws) : -1;
        unlink(stall_sock.c_str());
        cyc = 0;
        ev = "rc=" + std::to_string(rc) + " timeout_tx=" + std::to_string(tmo) + " server_reqs_seen=" + std::to_string(server_reqs);
        /* Crucial: server_reqs MUST be 1 (zero blind retries of non-idempotent SYNDROME_TRANSFER!) */
        return rc != 0 && tmo == 1ULL && server_reqs == 1;
    });

    /* L4-FLT-05: No blind retry of non-idempotent COMMIT (0x048=1) on timeout */
    suite.run_case("L4-FLT-05", "G6,G7", "COMMIT timeout sends exactly 1 COMMIT frame (never double-incrementing epoch via blind retry)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string stall_sock = avid_l4::make_temp_path("flt_com_stall", ".sock");
        unlink(stall_sock.c_str());
        int sfd = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", stall_sock.c_str());
        bind(sfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        listen(sfd, 2);

        pid_t mpid = fork();
        if (mpid == 0) {
            int cfd = accept(sfd, nullptr, nullptr);
            int commits_seen = 0;
            if (cfd >= 0) {
                struct timeval tv{0, 300000};
                setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
                std::string line;
                while (avid_l4::raw_read_line(cfd, line)) {
                    if (line.find("0x00000048") != std::string::npos) commits_seen++;
                }
                close(cfd);
            }
            close(sfd);
            _exit(commits_seen);
        }
        close(sfd);

        avid_hal_t mhal;
        avid_hal_open(&mhal, AVID_HAL_BACKEND_SIMULATION, stall_sock.c_str(), WRD_BASE_ADDR);
        avid_hal_set_timeout_ms(&mhal, 100);
        int rc = avid_hal_mmio_write(&mhal, 0x048u, 1u, 0x1u, &cyc);
        avid_hal_close(&mhal);
        int ws = 0;
        waitpid(mpid, &ws, 0);
        int commits_seen = WIFEXITED(ws) ? WEXITSTATUS(ws) : -1;
        unlink(stall_sock.c_str());
        cyc = 0;
        ev = "rc=" + std::to_string(rc) + " commits_seen=" + std::to_string(commits_seen);
        return rc != 0 && commits_seen == 1;
    });

    /* L4-FLT-06: Duplicate tx_id across distinct client connections still rejected */
    suite.run_case("L4-FLT-06", "G1,G7", "Replaying an old transaction_id from a new client socket is rejected with ERR_DUPLICATE_TX_ID", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f1 = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r1, r2;
        avid_l4::raw_rpc(f1, "{\"protocol_version\":1,\"transaction_id\":500,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r1);
        close(f1);
        int f2 = avid_l4::raw_unix_connect(bridge.sock_path);
        avid_l4::raw_rpc(f2, "{\"protocol_version\":1,\"transaction_id\":500,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r2);
        close(f2);
        avid_l4::extract_json_u64_simple(r1, "completed_cycle", &cyc);
        ev = "cross-session dup tx_id=500 -> " + avid_l4::extract_json_str_simple(r2, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_DUPLICATE_TX_ID";
    });

    /* L4-FLT-07: Out-of-order tx_id across sessions rejected */
    suite.run_case("L4-FLT-07", "G1,G7", "Decreasing unseen tx_id across reconnect rejected with ERR_OUT_OF_ORDER_TX_ID", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r1, r2;
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":600,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r1);
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":550,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r2);
        close(f);
        avid_l4::extract_json_u64_simple(r1, "completed_cycle", &cyc);
        ev = "tx 600 -> 550 => " + avid_l4::extract_json_str_simple(r2, "status");
        return avid_l4::extract_json_str_simple(r2, "status") == "ERR_OUT_OF_ORDER_TX_ID";
    });

    /* L4-FLT-08: Oversized payload storm (10 x 4KB frames) */
    suite.run_case("L4-FLT-08", "G1,G7", "10 consecutive 4KB oversized payloads drained cleanly without buffer overflow", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string huge(4096, 'Z');
        huge.push_back('\n');
        for (int i = 0; i < 10; i++) {
            std::string r;
            if (!avid_l4::raw_rpc(f, huge, r) || avid_l4::extract_json_str_simple(r, "status") != "ERR_OVERSIZED") {
                close(f);
                return false;
            }
        }
        std::string ok_r;
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":700,\"operation\":\"MMIO_READ\",\"address\":\"0x00000054\"}\n", ok_r);
        close(f);
        avid_l4::extract_json_u64_simple(ok_r, "completed_cycle", &cyc);
        ev = "10x 4KB drained + valid follow-up OK";
        return avid_l4::extract_json_str_simple(ok_r, "status") == "OK";
    });

    /* L4-FLT-09: SYNDROME_WAIT without prior SYNDROME_START -> ERR_NOT_INFLIGHT */
    suite.run_case("L4-FLT-09", "G1,G7", "Orphan SYNDROME_WAIT with no inflight syndrome returns ERR_NOT_INFLIGHT", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r;
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":710,\"operation\":\"SYNDROME_WAIT\"}\n", r);
        close(f);
        avid_l4::extract_json_u64_simple(r, "completed_cycle", &cyc);
        ev = "orphan SYNDROME_WAIT -> " + avid_l4::extract_json_str_simple(r, "status");
        return avid_l4::extract_json_str_simple(r, "status") == "ERR_NOT_INFLIGHT";
    });

    /* L4-FLT-10: Blocking SYNDROME_TRANSFER while async SYNDROME_START inflight -> ERR_BUSY */
    suite.run_case("L4-FLT-10", "G1,G7", "SYNDROME_TRANSFER while SYNDROME_START is inflight returns ERR_BUSY without corrupting inflight decode", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r1, r2, r3;
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":720,\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":true}\n", r1);
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":721,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0001\"}\n", r2);
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":722,\"operation\":\"SYNDROME_WAIT\"}\n", r3);
        close(f);
        avid_l4::extract_json_u64_simple(r3, "completed_cycle", &cyc);
        ev = "concurrent SYNDROME_TRANSFER -> " + avid_l4::extract_json_str_simple(r2, "status") +
             ", inflight WAIT -> " + avid_l4::extract_json_str_simple(r3, "status");
        return avid_l4::extract_json_str_simple(r2, "status") == "ERR_BUSY" &&
               avid_l4::extract_json_str_simple(r3, "status") == "OK";
    });

    /* L4-FLT-11: STEP_CYCLES out-of-range (> 100000 or 0) rejected safely */
    suite.run_case("L4-FLT-11", "G1,G7", "STEP_CYCLES with data=0 or data=999999 rejected with ERR_INVALID_REGISTER (+0 cycles)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int f = avid_l4::raw_unix_connect(bridge.sock_path);
        std::string r1, r2;
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":730,\"operation\":\"STEP_CYCLES\",\"data\":0}\n", r1);
        avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":731,\"operation\":\"STEP_CYCLES\",\"data\":999999}\n", r2);
        close(f);
        avid_l4::extract_json_u64_simple(r2, "completed_cycle", &cyc);
        ev = "0->" + avid_l4::extract_json_str_simple(r1, "status") + " 999999->" + avid_l4::extract_json_str_simple(r2, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "ERR_INVALID_REGISTER" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_INVALID_REGISTER";
    });

    /* L4-FLT-12: RTL SLVERR rejection increments HAL rejected_transactions without closing socket */
    suite.run_case("L4-FLT-12", "G7", "SLVERR via HAL increments rejected_transactions while keeping HAL connection healthy", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_t hal;
        avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR);
        hal.next_tx_id = 800ULL;
        int rc_bad = avid_hal_mmio_write(&hal, 0x054u, 99u, 0x1u, &cyc);
        uint32_t rd = 0;
        int rc_ok = avid_hal_mmio_read(&hal, 0x054u, &rd, &cyc);
        uint64_t rej = hal.stats.rejected_transactions;
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        ev = "rc_bad=" + std::to_string(rc_bad) + " rejected=" + std::to_string(rej) + " rc_ok=" + std::to_string(rc_ok);
        return rc_bad != 0 && rej == 1ULL && rc_ok == 0 && rd == 3u;
    });

    /* L4-FLT-13: RTL mid-decode hardware reset recovery on Vaa_qec_top */
    suite.run_case("L4-FLT-13", "G7", "Direct Vaa_qec_top verify_reset_during_active_decode() recovers cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(10);
        bool ok = rtl.verify_reset_during_active_decode();
        cyc = rtl.sim_cycle;
        ev = "mid-decode reset + clean 17-cycle recovery verified";
        return ok;
    });

    /* L4-FLT-14: HAL null-pointer & out-of-range argument defensive guards */
    suite.run_case("L4-FLT-14", "G7", "HAL functions reject nullptr handles, nullptr outputs, and invalid K/osd_mode safely", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        avid_hal_t hal;
        avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR);
        bool ok = (avid_hal_open(nullptr, AVID_HAL_BACKEND_PHYSICAL, nullptr, 0) < 0) &&
                  (avid_hal_mmio_read(&hal, 0x000u, nullptr, &cyc) < 0) &&
                  (avid_hal_write_weights_and_commit(&hal, nullptr, nullptr, nullptr, &cyc) < 0) &&
                  (avid_hal_set_bp_iterations(&hal, 0u, &cyc) < 0) &&
                  (avid_hal_set_bp_iterations(&hal, 6u, &cyc) < 0) &&
                  (avid_hal_set_osd_mode(&hal, 3u, &cyc) < 0);
        avid_hal_close(&hal);
        ev = "all 6 defensive parameter guards returned negative error without crash";
        return ok;
    });

    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
