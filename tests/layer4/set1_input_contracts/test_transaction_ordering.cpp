/* ===========================================================================
 * test_transaction_ordering.cpp — Layer 4 Set 1.4 Sequencing & Ordering Tests
 * Canonical IDs: L4-SEQ-01 .. L4-SEQ-07 (Gates G1, G2, G7)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set1_tx_ordering", "Set1", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set1_seq", 16, true)) return 2;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return 2;

    /* L4-SEQ-01: MMIO config + COMMIT before syndrome transfer */
    suite.run_case("L4-SEQ-01", "G1,G2", "MMIO config (K=5, inhomogeneous weights) before syndrome takes effect immediately", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_reset_rtl(&hal, 10u, &cyc);
        avid_hal_set_bp_iterations(&hal, 5u, &cyc);
        int16_t wx[9] = {0x0260, 0x0260, 0x0260, 0x0260, 0x00C0, 0x0260, 0x0260, 0x0260, 0x0260};
        int16_t wz[9] = {0x0260, 0x0260, 0x0260, 0x0260, 0x00C0, 0x0260, 0x0260, 0x0260, 0x0260};
        uint32_t ep = 0;
        if (avid_hal_write_weights_and_commit(&hal, wx, wz, &ep, &cyc) != 0) return false;
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        if (avid_hal_send_syndrome(&hal, 0x6u, 0x0u, true, &cx, &cz, &bp, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "K=5 -> bp_cycles=" + std::to_string(bp) + " corr_z=" + std::to_string(cz);
        return ep == 1u && bp == 27u && cz == (1u << 4);
    });

    /* L4-SEQ-02: Syndrome transfer before config uses reset defaults */
    suite.run_case("L4-SEQ-02", "G1,G2", "Syndrome before config uses reset defaults (K=3 -> 17 cyc, weights=0x0232)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_reset_rtl(&hal, 10u, &cyc);
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        if (avid_hal_send_syndrome(&hal, 0x1u, 0x0u, false, &cx, &cz, &bp, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "pre-config bp_cycles=" + std::to_string(bp);
        return bp == 17u && cz != 0u;
    });

    /* L4-SEQ-03: RESET during outstanding async decode transaction */
    suite.run_case("L4-SEQ-03", "G1,G7", "Mid-decode RESET aborts inflight decode cleanly and recovers on next syndrome", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r_st, r_step, r_rst, r_syn;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":81001,\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":true}\n", r_st);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":81002,\"operation\":\"STEP_CYCLES\",\"data\":5}\n", r_step);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":81003,\"operation\":\"RESET\",\"data\":8}\n", r_rst);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":81004,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0001\",\"tlast\":true}\n", r_syn);
        hal.next_tx_id = 82000ULL;
        uint64_t bp = 0;
        avid_l4::extract_json_u64_simple(r_syn, "bp_cycles", &bp);
        avid_l4::extract_json_u64_simple(r_syn, "completed_cycle", &cyc);
        ev = "mid-decode RESET -> post-reset SYNDROME_TRANSFER OK (bp=" + std::to_string(bp) + ")";
        return avid_l4::extract_json_str_simple(r_rst, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r_syn, "status") == "OK" && bp == 17u;
    });

    /* L4-SEQ-04: Consecutive AND non-consecutive duplicate transaction_id rejection (DEF-L4-06) */
    suite.run_case("L4-SEQ-04", "G1,G2", "Both adjacent and non-adjacent duplicate transaction_ids rejected with ERR_DUPLICATE_TX_ID", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r1, r2, r3, r4;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":83001,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r1);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":83001,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r2);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":83002,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r3);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":83001,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r4);
        hal.next_tx_id = 84000ULL;
        avid_l4::extract_json_u64_simple(r3, "completed_cycle", &cyc);
        ev = "adj=" + avid_l4::extract_json_str_simple(r2, "status") +
             " non_adj=" + avid_l4::extract_json_str_simple(r4, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_DUPLICATE_TX_ID" &&
               avid_l4::extract_json_str_simple(r3, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r4, "status") == "ERR_DUPLICATE_TX_ID";
    });

    /* L4-SEQ-05: Decreasing / out-of-order transaction_id rejection (DEF-L4-06) */
    suite.run_case("L4-SEQ-05", "G1,G2", "Decreasing unseen transaction_id rejected with ERR_OUT_OF_ORDER_TX_ID", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r1, r2;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":85010,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r1);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":85005,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r2);
        hal.next_tx_id = 86000ULL;
        avid_l4::extract_json_u64_simple(r1, "completed_cycle", &cyc);
        ev = "tx 85010 -> 85005 => " + avid_l4::extract_json_str_simple(r2, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_OUT_OF_ORDER_TX_ID";
    });

    /* L4-SEQ-06: Missing / mismatched ack detection in HAL client */
    suite.run_case("L4-SEQ-06", "G1,G2", "HAL client detects missing ack (EOF) and mismatched response transaction_id", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string mock_sock = avid_l4::make_temp_path("mock_ack", ".sock");
        unlink(mock_sock.c_str());
        int sfd = socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", mock_sock.c_str());
        bind(sfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        listen(sfd, 2);

        pid_t mpid = fork();
        if (mpid == 0) {
            /* Serve 1 wrong-tx_id response, then 1 immediate-EOF response */
            int c1 = accept(sfd, nullptr, nullptr);
            if (c1 >= 0) {
                std::string req;
                avid_l4::raw_read_line(c1, req);
                const char *wrong = "{\"protocol_version\":1,\"transaction_id\":99999,\"status\":\"OK\",\"accepted_cycle\":2,\"completed_cycle\":4,\"rdata\":\"0x0\"}\n";
                avid_l4::raw_write_all(c1, wrong, std::strlen(wrong));
                std::string req2;
                avid_l4::raw_read_line(c1, req2);
                close(c1); /* Missing ack (EOF) */
            }
            close(sfd);
            _exit(0);
        }
        close(sfd);

        avid_hal_t mhal;
        avid_hal_open(&mhal, AVID_HAL_BACKEND_SIMULATION, mock_sock.c_str(), WRD_BASE_ADDR);
        avid_hal_set_timeout_ms(&mhal, 500);
        uint32_t rd = 0;
        int rc1 = avid_hal_mmio_read(&mhal, 0x050u, &rd, &cyc);
        int rc2 = avid_hal_mmio_read(&mhal, 0x050u, &rd, &cyc);
        uint64_t id_err = mhal.stats.id_mismatch_errors;
        uint64_t rej = mhal.stats.rejected_transactions;
        avid_hal_close(&mhal);
        int st = 0;
        waitpid(mpid, &st, 0);
        unlink(mock_sock.c_str());
        cyc = 0;
        ev = "rc1=" + std::to_string(rc1) + " id_mismatch=" + std::to_string(id_err) +
             " rc2=" + std::to_string(rc2) + " rejected=" + std::to_string(rej);
        return rc1 != 0 && id_err == 1ULL && rc2 != 0 && rej >= 2ULL;
    });

    /* L4-SEQ-07: Queued requests arriving while bridge is processing 22-cycle decode */
    suite.run_case("L4-SEQ-07", "G1,G7", "Burst of 8 queued SYNDROME_TRANSFER requests sent in single write() execute in strict FIFO order", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string burst;
        for (int i = 0; i < 8; i++) {
            char line[200];
            std::snprintf(line, sizeof(line),
                          "{\"protocol_version\":1,\"transaction_id\":%d,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x%04x\",\"tlast\":%s}\n",
                          87000 + i, (i + 1) & 0xF, (i == 7) ? "true" : "false");
            burst += line;
        }
        if (!avid_l4::raw_write_all(rfd, burst.data(), burst.size())) return false;
        uint64_t prev_comp = 0;
        for (int i = 0; i < 8; i++) {
            std::string resp;
            if (!avid_l4::raw_read_line(rfd, resp)) return false;
            uint64_t tid = 0, acc = 0, comp = 0;
            avid_l4::extract_json_u64_simple(resp, "transaction_id", &tid);
            avid_l4::extract_json_u64_simple(resp, "accepted_cycle", &acc);
            avid_l4::extract_json_u64_simple(resp, "completed_cycle", &comp);
            if (tid != static_cast<uint64_t>(87000 + i)) return false;
            if (i > 0 && acc != prev_comp) return false;
            if ((comp - acc) != 22ULL) return false;
            prev_comp = comp;
        }
        hal.next_tx_id = 88000ULL;
        cyc = prev_comp;
        ev = "8/8 queued decodes executed in FIFO order with 22-cycle contiguous timeline";
        return true;
    });

    avid_hal_close(&hal);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
