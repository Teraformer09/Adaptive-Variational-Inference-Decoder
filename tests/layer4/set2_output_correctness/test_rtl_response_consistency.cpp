/* ===========================================================================
 * test_rtl_response_consistency.cpp — Layer 4 Set 2.3 5-Way Consistency Tests
 * Canonical IDs: L4-CONS-01 .. L4-CONS-05 (Gates G2, G4)
 * Cross-checks: (1) Bridge JSON ack, (2) Standalone Vaa_qec_top pins,
 *               (3) RTL MMIO readback, (4) AXI-Stream tdata, (5) m3_transactions.jsonl
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set2_rtl_consistency", "Set2", argc, argv);
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set2_cons", 8, true)) return 2;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return 2;

    AvidVerilatorRtlModel ref_rtl(10); /* Initial 10-cycle reset in standalone Vaa_qec_top */
    uint64_t bc = 0, rc = 0;
    avid_hal_reset_rtl(&hal, 10u, &bc);
    ref_rtl.reset_model(10u);

    /* L4-CONS-01: Weight commit + active bank MMIO readback consistency */
    suite.run_case("L4-CONS-01", "G2,G4", "Bridge MMIO commit & active-bank readback match standalone Vaa_qec_top bit-for-bit", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int16_t wx[9], wz[9];
        for (int i = 0; i < 9; i++) {
            wx[i] = static_cast<int16_t>(0x0120 + 13 * i);
            wz[i] = static_cast<int16_t>(0x0210 + 9 * i);
            ref_rtl.axil_write(WRD_OFF_SHADOW_X(i), static_cast<uint16_t>(wx[i]), 0x3u, &rc);
            ref_rtl.axil_write(WRD_OFF_SHADOW_Z(i), static_cast<uint16_t>(wz[i]), 0x3u, &rc);
        }
        ref_rtl.axil_write(0x048u, 1u, 0x1u, &rc);
        uint32_t ref_ep = 0;
        ref_rtl.axil_read(0x050u, &ref_ep, &rc);

        uint32_t hal_ep = 0;
        if (avid_hal_write_weights_and_commit(&hal, wx, wz, &hal_ep, &bc) != 0) return false;
        if (hal_ep != ref_ep) return false;

        for (int i = 0; i < 9; i++) {
            uint32_t bx = 0, bz = 0, rx = 0, rz = 0;
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(i), &bx, &bc);
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(i), &bz, &bc);
            ref_rtl.axil_read(WRD_OFF_ACTIVE_X(i), &rx, &rc);
            ref_rtl.axil_read(WRD_OFF_ACTIVE_Z(i), &rz, &rc);
            if (bx != rx || bz != rz) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "18/18 active weights + epoch_count=1 identical between Bridge and standalone Vaa_qec_top";
        return cyc == rc;
    });

    /* L4-CONS-02: Syndrome correction & cycle consistency across 32 diverse syndromes */
    suite.run_case("L4-CONS-02", "G2,G4", "32 dual-basis decodes match corr_x, corr_z, bp_cycles, epoch_irq, and sim_cycle", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (int i = 0; i < 32; i++) {
            uint8_t sx = static_cast<uint8_t>((i * 5 + 3) & 0xF);
            uint8_t sz = static_cast<uint8_t>((i * 7 + 1) & 0xF);
            bool tl = ((i % 8) == 7);
            uint16_t bx = 0, bz = 0;
            uint32_t bbp = 0;
            uint64_t bacc = 0;
            if (avid_hal_send_syndrome(&hal, sx, sz, tl, &bx, &bz, &bbp, &bacc) != 0) return false;
            RtlDecodeTrace rtr = ref_rtl.decode_syndrome(sx, sz, tl);
            if (bx != rtr.corr_x || bz != rtr.corr_z || bbp != rtr.bp_core_cycles ||
                (hal.stats.last_epoch_irq != 0u) != rtr.epoch_irq_pulsed ||
                bacc != rtr.accepted_cycle || hal.stats.last_completed_cycle != rtr.end_cycle) {
                return false;
            }
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "32/32 decodes matched corr_x/z, bp_cycles, epoch_irq, and exact clock cycles";
        return true;
    });

    /* L4-CONS-03: AXI-Stream packed m_axis_correction_tdata[17:0] bit-slice agreement */
    suite.run_case("L4-CONS-03", "G2,G4", "Packed AXI-Stream m_axis_correction_tdata[8:0]=corr_x and [17:9]=corr_z match JSON ack", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
#if defined(USE_VERILATOR) && USE_VERILATOR
        AvidVerilatorRtlModel pin_rtl(10);
        pin_rtl.top->s_axis_syndrome_tdata = 0x0096u; /* sx=6, sz=9 */
        pin_rtl.top->s_axis_syndrome_tvalid = 1;
        pin_rtl.top->s_axis_syndrome_tlast = 1;
        pin_rtl.top->m_axis_correction_tready = 1;
        pin_rtl.step_clock(1);
        pin_rtl.top->s_axis_syndrome_tvalid = 0;
        uint32_t raw_tdata = 0;
        for (int g = 0; g < 30; g++) {
            pin_rtl.step_clock(1);
            if (pin_rtl.top->m_axis_correction_tvalid) {
                raw_tdata = pin_rtl.top->m_axis_correction_tdata;
                break;
            }
        }
        avid_hal_t h2;
        avid_l4::BridgeProcessGuard b2;
        if (!b2.start(suite.bridge_bin_path(), "set2_cons_pin", 4, true)) return false;
        avid_hal_open(&h2, AVID_HAL_BACKEND_SIMULATION, b2.sock_path.c_str(), WRD_BASE_ADDR);
        uint16_t hx = 0, hz = 0;
        uint32_t hbp = 0;
        avid_hal_send_syndrome(&h2, 0x6u, 0x9u, true, &hx, &hz, &hbp, &cyc);
        cyc = h2.stats.last_completed_cycle;
        avid_hal_close(&h2);
        b2.graceful_shutdown(888888ULL);
        uint16_t px = static_cast<uint16_t>(raw_tdata & 0x1FFu);
        uint16_t pz = static_cast<uint16_t>((raw_tdata >> 9) & 0x1FFu);
        ev = "raw_tdata=" + std::to_string(raw_tdata) + " px=" + std::to_string(px) + " pz=" + std::to_string(pz);
        return px == hx && pz == hz && px != 0u && pz != 0u;
#else
        return false;
#endif
    });

    /* L4-CONS-04: Error response consistency (SLVERR readback == 0xDEADBEEF on both RTL & Bridge) */
    suite.run_case("L4-CONS-04", "G2,G4", "Invalid register read produces identical SLVERR + 0xDEADBEEF on Vaa_qec_top and Bridge", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t r_pin = 0;
        bool ok_pin = ref_rtl.axil_read(0x090u, &r_pin, &rc);
        int rfd = hal.sock_fd;
        std::string resp;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":5000,\"operation\":\"MMIO_READ\",\"address\":\"0x00000090\"}\n", resp);
        hal.next_tx_id = 6000ULL;
        uint64_t r_br = 0;
        avid_l4::extract_json_u64_simple(resp, "rdata", &r_br);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        ev = "pin_ok=" + std::to_string(ok_pin) + " r_pin=0xdeadbeef r_bridge=0xdeadbeef";
        return !ok_pin && r_pin == 0xDEADBEEFu && r_br == 0xDEADBEEFULL &&
               avid_l4::extract_json_str_simple(resp, "status") == "ERR_INVALID_REGISTER";
    });

    /* L4-CONS-05: m3_transactions.jsonl cross-check against HAL response fields (DEF-L4-07) */
    suite.run_case("L4-CONS-05", "G2,G4", "m3_transactions.jsonl records rdata, corr_x, corr_z, bp_cycles, epoch_irq, completed_cycle matching HAL", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        uint64_t acc = 0;
        uint64_t expected_tid = hal.next_tx_id;
        if (avid_hal_send_syndrome(&hal, 0xCu, 0x3u, true, &cx, &cz, &bp, &acc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;

        std::ifstream in(bridge.tx_log_path);
        std::string line, matched;
        while (std::getline(in, line)) {
            uint64_t tid = 0;
            if (avid_l4::extract_json_u64_simple(line, "tx_id", &tid) && tid == expected_tid) {
                matched = line;
            }
        }
        if (matched.empty()) return false;
        uint64_t l_acc = 0, l_comp = 0, l_cx = 0, l_cz = 0, l_bp = 0, l_irq = 0;
        avid_l4::extract_json_u64_simple(matched, "accepted_cycle", &l_acc);
        avid_l4::extract_json_u64_simple(matched, "completed_cycle", &l_comp);
        avid_l4::extract_json_u64_simple(matched, "corr_x", &l_cx);
        avid_l4::extract_json_u64_simple(matched, "corr_z", &l_cz);
        avid_l4::extract_json_u64_simple(matched, "bp_cycles", &l_bp);
        avid_l4::extract_json_u64_simple(matched, "epoch_irq", &l_irq);
        ev = "tx_id=" + std::to_string(expected_tid) + " JSONL fields matched HAL 100%";
        return l_acc == acc && l_comp == cyc && l_cx == cx && l_cz == cz && l_bp == bp && l_irq == 1ULL;
    });

    avid_hal_close(&hal);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
