/* ===========================================================================
 * test_acknowledgements.cpp — Layer 4 Set 2.1 Acknowledgement Schema & Semantics
 * Canonical IDs: L4-ACK-01 .. L4-ACK-07 (Gate G2)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set2_acknowledgements", "Set2", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set2_ack", 16, true)) return 2;

    int fd = avid_l4::raw_unix_connect(bridge.sock_path);
    if (fd < 0) return 2;

    /* L4-ACK-01: MMIO_WRITE & MMIO_READ response schema & cycle separation */
    suite.run_case("L4-ACK-01", "G2", "MMIO_WRITE and MMIO_READ responses include accepted_cycle, completed_cycle, rdata", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r0, rw, rr;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":1,\"operation\":\"RESET\",\"data\":10}\n", r0);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":2,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000004\",\"data\":\"0x01AB\",\"byte_enable\":3}\n", rw);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":3,\"operation\":\"MMIO_READ\",\"address\":\"0x00000004\"}\n", rr);
        uint64_t aw = 0, cw = 0, ar = 0, cr = 0, rd = 0;
        avid_l4::extract_json_u64_simple(rw, "accepted_cycle", &aw);
        avid_l4::extract_json_u64_simple(rw, "completed_cycle", &cw);
        avid_l4::extract_json_u64_simple(rr, "accepted_cycle", &ar);
        avid_l4::extract_json_u64_simple(rr, "completed_cycle", &cr);
        avid_l4::extract_json_u64_simple(rr, "rdata", &rd);
        cyc = cr;
        ev = "wr=[" + std::to_string(aw) + "," + std::to_string(cw) + "] rd=[" + std::to_string(ar) + "," + std::to_string(cr) + "]";
        return (cw - aw) == 2ULL && ar == cw && (cr - ar) == 2ULL && (rd & 0xFFFFu) == 0x01ABu;
    });

    /* L4-ACK-02: SYNDROME_TRANSFER response schema completeness */
    suite.run_case("L4-ACK-02", "G2", "SYNDROME_TRANSFER response reports distinct accepted_cycle vs completed_cycle, corr_x/z, bp_cycles, axis_total_cycles, epoch_irq", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string rs;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":4,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0096\",\"tlast\":true}\n", rs);
        uint64_t acc = 0, comp = 0, bp = 0, ax = 0, irq = 0, cx = 0, cz = 0;
        bool ok = avid_l4::extract_json_u64_simple(rs, "accepted_cycle", &acc) &&
                  avid_l4::extract_json_u64_simple(rs, "completed_cycle", &comp) &&
                  avid_l4::extract_json_u64_simple(rs, "bp_cycles", &bp) &&
                  avid_l4::extract_json_u64_simple(rs, "axis_total_cycles", &ax) &&
                  avid_l4::extract_json_u64_simple(rs, "epoch_irq", &irq) &&
                  avid_l4::extract_json_u64_simple(rs, "corr_x", &cx) &&
                  avid_l4::extract_json_u64_simple(rs, "corr_z", &cz);
        cyc = comp;
        ev = "acc=" + std::to_string(acc) + " comp=" + std::to_string(comp) + " bp=" + std::to_string(bp) + " axis=" + std::to_string(ax);
        return ok && (comp - acc) == 22ULL && ax == 22ULL && bp == 17ULL && irq == 1ULL && cx != 0ULL && cz != 0ULL;
    });

    /* L4-ACK-03: Acceptance != Completion via SYNDROME_START -> SYNDROME_WAIT */
    suite.run_case("L4-ACK-03", "G2", "SYNDROME_START acknowledges pin acceptance (inflight=1) 19 cycles before SYNDROME_WAIT completion", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r_start, r_wait;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":5,\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":true}\n", r_start);
        uint64_t acc_s = 0, comp_s = 0, inf_s = 0;
        avid_l4::extract_json_u64_simple(r_start, "accepted_cycle", &acc_s);
        avid_l4::extract_json_u64_simple(r_start, "completed_cycle", &comp_s);
        avid_l4::extract_json_u64_simple(r_start, "inflight", &inf_s);
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":6,\"operation\":\"SYNDROME_WAIT\"}\n", r_wait);
        uint64_t acc_w = 0, comp_w = 0, bp_w = 0;
        avid_l4::extract_json_u64_simple(r_wait, "accepted_cycle", &acc_w);
        avid_l4::extract_json_u64_simple(r_wait, "completed_cycle", &comp_w);
        avid_l4::extract_json_u64_simple(r_wait, "bp_cycles", &bp_w);
        cyc = comp_w;
        ev = "start_comp=" + std::to_string(comp_s) + " wait_comp=" + std::to_string(comp_w) + " inflight=" + std::to_string(inf_s);
        return inf_s == 1ULL && acc_s == acc_w && (comp_s - acc_s) == 3ULL && (comp_w - acc_w) == 22ULL && bp_w == 17ULL;
    });

    /* L4-ACK-04: Protocol-rejected RPCs report accepted_cycle == completed_cycle (+0 cycles) */
    suite.run_case("L4-ACK-04", "G2", "Protocol-rejected requests return exact ERR_* status with zero RTL cycle advance", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r_bad;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":7,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0999\"}\n", r_bad);
        uint64_t acc = 0, comp = 1;
        avid_l4::extract_json_u64_simple(r_bad, "accepted_cycle", &acc);
        avid_l4::extract_json_u64_simple(r_bad, "completed_cycle", &comp);
        cyc = comp;
        ev = "status=" + avid_l4::extract_json_str_simple(r_bad, "status") + " delta=" + std::to_string(comp - acc);
        return avid_l4::extract_json_str_simple(r_bad, "status") == "ERR_INVALID_SYNDROME" && acc == comp;
    });

    /* L4-ACK-05: Hardware SLVERR MMIO rejection acknowledgment */
    suite.run_case("L4-ACK-05", "G2", "RTL SLVERR read/write returns ERR_INVALID_REGISTER, rdata=0xdeadbeef, and +2 bus handshake cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string r_slv;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":8,\"operation\":\"MMIO_READ\",\"address\":\"0x00000990\"}\n", r_slv);
        uint64_t acc = 0, comp = 0, rd = 0;
        avid_l4::extract_json_u64_simple(r_slv, "accepted_cycle", &acc);
        avid_l4::extract_json_u64_simple(r_slv, "completed_cycle", &comp);
        avid_l4::extract_json_u64_simple(r_slv, "rdata", &rd);
        cyc = comp;
        ev = "rdata=" + avid_l4::extract_json_str_simple(r_slv, "rdata") + " delta=" + std::to_string(comp - acc);
        return avid_l4::extract_json_str_simple(r_slv, "status") == "ERR_INVALID_REGISTER" &&
               rd == 0xDEADBEEFULL && (comp - acc) == 2ULL;
    });

    /* L4-ACK-06: HAL client stats capture accepted_cycle, completed_cycle, axis_total_cycles, epoch_irq */
    suite.run_case("L4-ACK-06", "G2", "avid_hal_t stats capture last_accepted_cycle, last_completed_cycle, last_axis_total_cycles, irq_events", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        close(fd);
        fd = -1;
        avid_hal_t hal;
        if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return false;
        hal.next_tx_id = 1000ULL;
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        uint64_t acc = 0;
        int rc = avid_hal_send_syndrome(&hal, 0x6u, 0x0u, true, &cx, &cz, &bp, &acc);
        bool ok = (rc == 0) &&
                  (hal.stats.last_accepted_cycle == acc) &&
                  (hal.stats.last_completed_cycle == acc + 22ULL) &&
                  (hal.stats.last_axis_total_cycles == 22u) &&
                  (hal.stats.last_epoch_irq == 1u) &&
                  (hal.stats.irq_events == 1ULL);
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        fd = avid_l4::raw_unix_connect(bridge.sock_path);
        ev = "acc=" + std::to_string(acc) + " comp=" + std::to_string(cyc) + " irq_events=1";
        return ok && (fd >= 0);
    });

    /* L4-ACK-07: ERR_OVERSIZED acknowledgment structure */
    suite.run_case("L4-ACK-07", "G2", "ERR_OVERSIZED returns valid JSON ack with transaction_id=0 and +0 cycle delta", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string big(2600, 'X');
        big.push_back('\n');
        std::string resp;
        if (!avid_l4::raw_rpc(fd, big, resp)) return false;
        uint64_t tid = 99, acc = 0, comp = 1;
        avid_l4::extract_json_u64_simple(resp, "transaction_id", &tid);
        avid_l4::extract_json_u64_simple(resp, "accepted_cycle", &acc);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &comp);
        cyc = comp;
        ev = "tid=" + std::to_string(tid) + " status=" + avid_l4::extract_json_str_simple(resp, "status");
        return tid == 0ULL && acc == comp && avid_l4::extract_json_str_simple(resp, "status") == "ERR_OVERSIZED";
    });

    close(fd);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
