/* ===========================================================================
 * test_rpc_schema.cpp — Layer 4 Set 1.1 RPC Protocol Contract Tests
 * Canonical IDs: L4-I01 .. L4-I12 (Gate G1)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set1_rpc_schema", "Set1", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set1_rpc", 32, true)) {
        std::fprintf(stderr, "[FATAL] Failed to start Verilated avid_cosim_bridge\n");
        return 2;
    }

    int fd = avid_l4::raw_unix_connect(bridge.sock_path);
    if (fd < 0) return 2;

    /* L4-I01: Valid MMIO_WRITE */
    suite.run_case("L4-I01", "G1", "Valid MMIO_WRITE reaches RTL shadow register", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r1, r2, r3;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":101,\"operation\":\"RESET\",\"data\":10}\n", r1)) return false;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":102,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000000\",\"data\":\"0x0140\",\"byte_enable\":3}\n", r2)) return false;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":103,\"operation\":\"MMIO_READ\",\"address\":\"0x00000000\",\"byte_enable\":15}\n", r3)) return false;
        uint64_t rdata = 0;
        avid_l4::extract_json_u64_simple(r3, "rdata", &rdata);
        avid_l4::extract_json_u64_simple(r2, "completed_cycle", &cyc);
        ev = "shadow_x[0]=" + std::to_string(rdata & 0xFFFFu);
        return avid_l4::extract_json_str_simple(r2, "status") == "OK" && (rdata & 0xFFFFu) == 0x0140u;
    });

    /* L4-I02: Valid MMIO_READ across register classes */
    suite.run_case("L4-I02", "G1", "Valid MMIO_READ across shadow, status, epoch, config, active", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        const uint32_t addrs[6] = {0x004u, 0x04Cu, 0x050u, 0x054u, 0x058u, 0x104u};
        const uint32_t expv[6]  = {0x0232u, 0x0u, 0x0u, 3u, 1u, 0x0232u};
        for (int i = 0; i < 6; i++) {
            char req[192];
            std::snprintf(req, sizeof(req),
                          "{\"protocol_version\":1,\"transaction_id\":%d,\"operation\":\"MMIO_READ\",\"address\":\"0x%08x\"}\n",
                          110 + i, addrs[i]);
            std::string resp;
            if (!avid_l4::raw_rpc(fd, req, resp)) return false;
            uint64_t rd = 0;
            avid_l4::extract_json_u64_simple(resp, "rdata", &rd);
            avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
            if (avid_l4::extract_json_str_simple(resp, "status") != "OK" || (rd & 0xFFFFu) != expv[i]) return false;
        }
        ev = "6 register classes matched reset/configured values";
        return true;
    });

    /* L4-I03: Valid SYNDROME_TRANSFER */
    suite.run_case("L4-I03", "G1", "Valid SYNDROME_TRANSFER dual-basis sx=0x6 sz=0x9", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string resp;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":120,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0096\",\"tlast\":true}\n", resp)) return false;
        uint64_t cx = 0, cz = 0, bp = 0, irq = 0;
        avid_l4::extract_json_u64_simple(resp, "corr_x", &cx);
        avid_l4::extract_json_u64_simple(resp, "corr_z", &cz);
        avid_l4::extract_json_u64_simple(resp, "bp_cycles", &bp);
        avid_l4::extract_json_u64_simple(resp, "epoch_irq", &irq);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        ev = "corr_x=" + std::to_string(cx) + " corr_z=" + std::to_string(cz) + " bp=" + std::to_string(bp);
        return avid_l4::extract_json_str_simple(resp, "status") == "OK" &&
               bp == 17u && irq == 1u && cx != 0u && cz != 0u;
    });

    /* L4-I04: Valid RESET */
    suite.run_case("L4-I04", "G1", "Valid RESET restores RTL registers and advances exact reset cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r_wr, r_com, r_rst, r_ep, r_sh;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":130,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000000\",\"data\":\"0x0111\",\"byte_enable\":3}\n", r_wr);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":131,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000048\",\"data\":\"0x1\",\"byte_enable\":1}\n", r_com);
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":132,\"operation\":\"RESET\",\"data\":16}\n", r_rst)) return false;
        uint64_t acc = 0, comp = 0;
        avid_l4::extract_json_u64_simple(r_rst, "accepted_cycle", &acc);
        avid_l4::extract_json_u64_simple(r_rst, "completed_cycle", &comp);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":133,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r_ep);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":134,\"operation\":\"MMIO_READ\",\"address\":\"0x00000000\"}\n", r_sh);
        uint64_t ep = 99, sh = 0;
        avid_l4::extract_json_u64_simple(r_ep, "rdata", &ep);
        avid_l4::extract_json_u64_simple(r_sh, "rdata", &sh);
        cyc = comp;
        ev = "delta_cyc=" + std::to_string(comp - acc) + " epoch=" + std::to_string(ep);
        return (comp - acc) == 16u && ep == 0u && (sh & 0xFFFFu) == 0x0232u;
    });

    /* L4-I05: Unknown RPC operation */
    suite.run_case("L4-I05", "G1", "Unknown RPC operation rejected with ERR_UNKNOWN_OP and +0 cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r_pre, r_unk;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":140,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r_pre);
        uint64_t c_before = 0, acc = 0, comp = 0;
        avid_l4::extract_json_u64_simple(r_pre, "completed_cycle", &c_before);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":141,\"operation\":\"DMA_BURST\"}\n", r_unk);
        avid_l4::extract_json_u64_simple(r_unk, "accepted_cycle", &acc);
        avid_l4::extract_json_u64_simple(r_unk, "completed_cycle", &comp);
        cyc = comp;
        ev = "status=" + avid_l4::extract_json_str_simple(r_unk, "status");
        return avid_l4::extract_json_str_simple(r_unk, "status") == "ERR_UNKNOWN_OP" &&
               acc == c_before && comp == c_before;
    });

    /* L4-I06: Missing required JSON fields */
    suite.run_case("L4-I06", "G1", "Missing protocol_version, transaction_id, operation, or address/data rejected", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r1, r2, r3, r4, r5;
        avid_l4::raw_rpc(fd, "{\"transaction_id\":150,\"operation\":\"MMIO_READ\",\"address\":\"0x0\"}\n", r1);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"operation\":\"MMIO_READ\",\"address\":\"0x0\"}\n", r2);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":151}\n", r3);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":152,\"operation\":\"MMIO_READ\"}\n", r4);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":153,\"operation\":\"MMIO_WRITE\",\"address\":\"0x0\"}\n", r5);
        avid_l4::extract_json_u64_simple(r5, "completed_cycle", &cyc);
        ev = avid_l4::extract_json_str_simple(r1, "status") + "," +
             avid_l4::extract_json_str_simple(r2, "status") + "," +
             avid_l4::extract_json_str_simple(r3, "status") + "," +
             avid_l4::extract_json_str_simple(r4, "status") + "," +
             avid_l4::extract_json_str_simple(r5, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "ERR_MISSING_VERSION" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_MISSING_TX_ID" &&
               avid_l4::extract_json_str_simple(r3, "status") == "ERR_MISSING_OPERATION" &&
               avid_l4::extract_json_str_simple(r4, "status") == "ERR_MISSING_FIELD" &&
               avid_l4::extract_json_str_simple(r5, "status") == "ERR_MISSING_FIELD";
    });

    /* L4-I07: Invalid field type (DEF-L4-04 remediation check) */
    suite.run_case("L4-I07", "G1", "Invalid JSON field types rejected with ERR_INVALID_TYPE while boolean tlast works", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r1, r2, r3, r4;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":\"not_num\",\"operation\":\"MMIO_READ\",\"address\":\"0x0\"}\n", r1);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":161,\"operation\":\"MMIO_READ\",\"address\":\"not_hex\"}\n", r2);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":162,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x06\",\"tlast\":\"yes\"}\n", r3);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":163,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x06\",\"tlast\":true}\n", r4);
        uint64_t irq = 0;
        avid_l4::extract_json_u64_simple(r4, "epoch_irq", &irq);
        avid_l4::extract_json_u64_simple(r4, "completed_cycle", &cyc);
        ev = avid_l4::extract_json_str_simple(r1, "status") + "/" +
             avid_l4::extract_json_str_simple(r2, "status") + "/" +
             avid_l4::extract_json_str_simple(r3, "status") + " bool_tlast_irq=" + std::to_string(irq);
        return avid_l4::extract_json_str_simple(r1, "status") == "ERR_INVALID_TYPE" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_INVALID_TYPE" &&
               avid_l4::extract_json_str_simple(r3, "status") == "ERR_INVALID_TYPE" &&
               avid_l4::extract_json_str_simple(r4, "status") == "OK" && irq == 1u;
    });

    /* L4-I08: Invalid JSON syntax */
    suite.run_case("L4-I08", "G1", "Truncated, trailing-comma, and unquoted-key JSON rejected without RTL mutation", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r0, r1, r2, r3, r4;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":170,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r0);
        uint64_t c0 = 0, c4 = 0;
        avid_l4::extract_json_u64_simple(r0, "completed_cycle", &c0);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":171\n", r1);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":172,\"operation\":\"RESET\",}\n", r2);
        avid_l4::raw_rpc(fd, "{protocol_version:1,\"transaction_id\":173,\"operation\":\"RESET\"}\n", r3);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":174,\"operation\":\"RESET\"} trailing\n", r4);
        avid_l4::extract_json_u64_simple(r4, "completed_cycle", &c4);
        cyc = c4;
        ev = "all 4 malformed variants -> ERR_MALFORMED_JSON, +0 cycles";
        return avid_l4::extract_json_str_simple(r1, "status") == "ERR_MALFORMED_JSON" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_MALFORMED_JSON" &&
               avid_l4::extract_json_str_simple(r3, "status") == "ERR_MALFORMED_JSON" &&
               avid_l4::extract_json_str_simple(r4, "status") == "ERR_MALFORMED_JSON" &&
               c4 == c0;
    });

    /* L4-I09: Empty RPC payload */
    suite.run_case("L4-I09", "G1", "Empty newline and immediate-EOF socket handled cleanly without hang", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        /* Send bare newline followed by valid request on same connection */
        std::string resp;
        if (!avid_l4::raw_write_all(fd, "\n", 1)) return false;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":180,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", resp)) return false;
        /* Also test immediate EOF on separate client socket */
        int eof_fd = avid_l4::raw_unix_connect(bridge.sock_path);
        if (eof_fd < 0) return false;
        close(eof_fd);
        std::string resp2;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":181,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", resp2)) return false;
        avid_l4::extract_json_u64_simple(resp2, "completed_cycle", &cyc);
        ev = "bare '\\n' + immediate-EOF client handled cleanly";
        return avid_l4::extract_json_str_simple(resp, "status") == "OK" &&
               avid_l4::extract_json_str_simple(resp2, "status") == "OK";
    });

    /* L4-I10: Oversized RPC payload (> 2047 B) */
    suite.run_case("L4-I10", "G1", "Oversized 3200-byte line drained safely and rejected with ERR_OVERSIZED", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string big = "{\"protocol_version\":1,\"transaction_id\":190,\"operation\":\"MMIO_READ\",\"pad\":\"";
        big.append(3100, 'A');
        big += "\"}\n";
        std::string r_over, r_ok;
        if (!avid_l4::raw_rpc(fd, big, r_over)) return false;
        if (!avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":191,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", r_ok)) return false;
        avid_l4::extract_json_u64_simple(r_ok, "completed_cycle", &cyc);
        ev = "oversized=" + avid_l4::extract_json_str_simple(r_over, "status") +
             " follow_up=" + avid_l4::extract_json_str_simple(r_ok, "status");
        return avid_l4::extract_json_str_simple(r_over, "status") == "ERR_OVERSIZED" &&
               avid_l4::extract_json_str_simple(r_ok, "status") == "OK";
    });

    /* L4-I11: Multiple requests on one connection (pipelined + chunked) */
    suite.run_case("L4-I11", "G1", "50 pipelined and chunked RPCs on one connection preserve 1-to-1 tx_id association", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string batch;
        for (int i = 0; i < 50; i++) {
            char line[192];
            std::snprintf(line, sizeof(line),
                          "{\"protocol_version\":1,\"transaction_id\":%d,\"operation\":\"MMIO_READ\",\"address\":\"0x00000054\"}\n",
                          200 + i);
            batch += line;
        }
        /* Write in 17-byte irregular chunks to exercise stream reassembly */
        size_t pos = 0;
        while (pos < batch.size()) {
            size_t n = (batch.size() - pos > 17) ? 17 : (batch.size() - pos);
            if (!avid_l4::raw_write_all(fd, batch.data() + pos, n)) return false;
            pos += n;
        }
        for (int i = 0; i < 50; i++) {
            std::string resp;
            if (!avid_l4::raw_read_line(fd, resp)) return false;
            uint64_t tid = 0;
            avid_l4::extract_json_u64_simple(resp, "transaction_id", &tid);
            avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
            if (tid != static_cast<uint64_t>(200 + i) ||
                avid_l4::extract_json_str_simple(resp, "status") != "OK") {
                return false;
            }
        }
        ev = "50/50 pipelined 17B-chunked responses matched exact tx_id";
        return true;
    });

    /* L4-I12: Malformed request followed by valid request */
    suite.run_case("L4-I12", "G1", "Sequence of 4 distinct error classes followed immediately by valid decode", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string e1, e2, e3, e4, ok_resp;
        avid_l4::raw_rpc(fd, "{broken_json\n", e1);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":9,\"transaction_id\":301,\"operation\":\"RESET\"}\n", e2);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":302,\"operation\":\"NO_SUCH_OP\"}\n", e3);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":303,\"operation\":\"MMIO_READ\",\"address\":\"0x00000990\"}\n", e4);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":304,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0006\",\"tlast\":true}\n", ok_resp);
        uint64_t cz = 0;
        avid_l4::extract_json_u64_simple(ok_resp, "corr_z", &cz);
        avid_l4::extract_json_u64_simple(ok_resp, "completed_cycle", &cyc);
        ev = "recovered cleanly after 4 errors; corr_z=" + std::to_string(cz);
        return avid_l4::extract_json_str_simple(e1, "status") == "ERR_MALFORMED_JSON" &&
               avid_l4::extract_json_str_simple(e2, "status") == "ERR_UNSUPPORTED_VERSION" &&
               avid_l4::extract_json_str_simple(e3, "status") == "ERR_UNKNOWN_OP" &&
               avid_l4::extract_json_str_simple(e4, "status") == "ERR_INVALID_REGISTER" &&
               avid_l4::extract_json_str_simple(ok_resp, "status") == "OK";
    });

    close(fd);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
