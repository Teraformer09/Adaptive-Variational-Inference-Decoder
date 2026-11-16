/* ===========================================================================
 * test_axil_register_contract.cpp — Layer 5 Set 1.2 AXI4-Lite Register Map Tests
 * Canonical IDs: L5-AXIL-01 .. L5-AXIL-30 (Gates G1, G4)
 * Exhaustively verifies AXI4-Lite handshakes, register map, WSTRB, SLVERR,
 * CFG_MAX_ITER (1..5), OSD_MODE (0..2), and Q7.8 signed boundaries on Vaa_qec_top.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set1_axil_register_contract", "Set1", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] test_axil_register_contract requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-AXIL-01: Simultaneous AW+W write handshake (2 cycles, BRESP=OKAY) */
    suite.run_case("L5-AXIL-01", "G1", "Simultaneous AW+W write completes in 2 clock cycles with BRESP=2'b00 (OKAY)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axil_awaddr = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x0155u;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_bready = 1;
        bool rdy = rtl.top->s_axil_awready && rtl.top->s_axil_wready;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        bool bv1 = (rtl.top->s_axil_bvalid == 1) && (rtl.top->s_axil_bresp == 0u);
        rtl.step_clock(1);
        bool bv2 = (rtl.top->s_axil_bvalid == 0);
        uint32_t rd = 0;
        rtl.axil_read(0x000u, &rd, &cyc);
        det = "aw/wready=1, bvalid=1->0, bresp=0, rdata=0x0155";
        return rdy && bv1 && bv2 && rd == 0x0155u;
    });

    /* L5-AXIL-02: Split AW-before-W handshake across 1, 4, and 8 cycle delays */
    suite.run_case("L5-AXIL-02", "G1", "Split AW-before-W latches aw_addr_lat cleanly across 1, 4, and 8 idle cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint32_t delays[3] = {1u, 4u, 8u};
        for (int i = 0; i < 3; i++) {
            uint32_t val = 0x01A0u + i;
            if (!rtl.axil_write_split_aw_w(4u * i, val, 0x3u, delays[i], &cyc)) return false;
            uint32_t rd = 0;
            if (!rtl.axil_read(4u * i, &rd, &cyc) || rd != val) return false;
        }
        det = "AW-before-W delays {1,4,8} latched & committed accurately";
        return true;
    });

    /* L5-AXIL-03: Split W-before-AW handshake across 1, 4, and 8 cycle delays */
    suite.run_case("L5-AXIL-03", "G1", "Split W-before-AW latches w_data_lat & w_strb_lat cleanly across 1, 4, and 8 idle cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint32_t delays[3] = {1u, 4u, 8u};
        for (int i = 0; i < 3; i++) {
            uint32_t val = 0x01B0u + i;
            if (!rtl.axil_write_split_w_aw(0x024u + 4u * i, val, 0x3u, delays[i], &cyc)) return false;
            uint32_t rd = 0;
            if (!rtl.axil_read(0x024u + 4u * i, &rd, &cyc) || rd != val) return false;
        }
        det = "W-before-AW delays {1,4,8} latched & committed accurately";
        return true;
    });

    /* L5-AXIL-04: BREADY=0 write-response stall stability */
    suite.run_case("L5-AXIL-04", "G1", "Holding s_axil_bready=0 holds s_axil_bvalid=1 & bresp stable and gates awready=wready=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axil_awaddr = 0x008u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x01CCu;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_bready = 0;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        for (int s = 0; s < 8; s++) {
            if (rtl.top->s_axil_bvalid != 1 || rtl.top->s_axil_bresp != 0u) return false;
            if (rtl.top->s_axil_awready != 0 || rtl.top->s_axil_wready != 0) return false;
            rtl.step_clock(1);
        }
        rtl.top->s_axil_bready = 1;
        rtl.step_clock(1);
        cyc = rtl.sim_cycle;
        det = "8-cycle BREADY=0 stall held BVALID=1 and gated AWREADY/WREADY=0";
        return rtl.top->s_axil_bvalid == 0 && rtl.top->s_axil_awready == 1;
    });

    /* L5-AXIL-05: RREADY=0 read-response stall stability */
    suite.run_case("L5-AXIL-05", "G1", "Holding s_axil_rready=0 holds s_axil_rvalid=1, rdata, and rresp stable and gates arready=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x00Cu, 0x01DDu, 0x3u, &cyc);
        rtl.top->s_axil_araddr = 0x00Cu;
        rtl.top->s_axil_arvalid = 1;
        rtl.top->s_axil_rready = 0;
        rtl.step_clock(1);
        rtl.top->s_axil_arvalid = 0;
        for (int s = 0; s < 8; s++) {
            if (rtl.top->s_axil_rvalid != 1 || rtl.top->s_axil_rdata != 0x01DDu) return false;
            if (rtl.top->s_axil_rresp != 0u || rtl.top->s_axil_arready != 0) return false;
            rtl.step_clock(1);
        }
        rtl.top->s_axil_rready = 1;
        rtl.step_clock(1);
        cyc = rtl.sim_cycle;
        det = "8-cycle RREADY=0 stall held RVALID=1, RDATA=0x01DD, and ARREADY=0";
        return rtl.top->s_axil_rvalid == 0 && rtl.top->s_axil_arready == 1;
    });

    /* L5-AXIL-06: Simultaneous AW+W write and AR read on identical clock edge */
    suite.run_case("L5-AXIL-06", "G1", "Simultaneous AXI4-Lite AW+W write (0x000) and AR read (0x024) complete on the same clock cycle", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x024u, 0x0277u, 0x3u, &cyc);
        rtl.top->s_axil_awaddr = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x0188u;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_bready = 1;
        rtl.top->s_axil_araddr = 0x024u;
        rtl.top->s_axil_arvalid = 1;
        rtl.top->s_axil_rready = 1;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        rtl.top->s_axil_arvalid = 0;
        bool ok_simul = (rtl.top->s_axil_bvalid == 1) && (rtl.top->s_axil_rvalid == 1) && (rtl.top->s_axil_rdata == 0x0277u);
        rtl.step_clock(1);
        uint32_t wr_rd = 0;
        rtl.axil_read(0x000u, &wr_rd, &cyc);
        det = "simultaneous BVALID=1 & RVALID=1 (0x0277), post-read 0x000=0x0188";
        return ok_simul && wr_rd == 0x0188u;
    });

    /* L5-AXIL-07: AMBA reset gating of awready/wready/arready during aresetn==0 (DEF-L5-01) */
    suite.run_case("L5-AXIL-07", "G1,G5", "s_axil_awready, s_axil_wready, and s_axil_arready held 0 while aresetn==0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->aresetn = 0;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_arvalid = 1;
        rtl.top->eval();
        bool rdy_low = (rtl.top->s_axil_awready == 0) && (rtl.top->s_axil_wready == 0) && (rtl.top->s_axil_arready == 0);
        rtl.step_clock(3);
        if (rtl.top->s_axil_awready || rtl.top->s_axil_wready || rtl.top->s_axil_arready) rdy_low = false;
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        rtl.top->s_axil_arvalid = 0;
        rtl.top->aresetn = 1;
        rtl.step_clock(1);
        bool rdy_high = (rtl.top->s_axil_awready == 1) && (rtl.top->s_axil_wready == 1) && (rtl.top->s_axil_arready == 1);
        cyc = rtl.sim_cycle;
        det = "reset *ready==0: " + std::to_string(rdy_low) + ", post-reset *ready==1: " + std::to_string(rdy_high);
        return rdy_low && rdy_high;
    });

    /* L5-AXIL-08: Valid RW access across all 9 shadow_x[0..8] offsets (0x000..0x020) */
    suite.run_case("L5-AXIL-08", "G1", "All 9 shadow_x[0..8] registers (0x000..0x020) accept distinct 16-bit writes and read back cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) {
            if (!rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), 0x0110u + 3u * i, 0x3u, &cyc)) return false;
        }
        for (int i = 0; i < 9; i++) {
            uint32_t rd = 0;
            if (!rtl.axil_read(avid_l5::L5_OFF_SHADOW_X(i), &rd, &cyc) || rd != (0x0110u + 3u * i)) return false;
        }
        det = "9/9 shadow_x offsets (0x000..0x020) verified";
        return true;
    });

    /* L5-AXIL-09: Valid RW access across all 9 shadow_z[0..8] offsets (0x024..0x044) */
    suite.run_case("L5-AXIL-09", "G1", "All 9 shadow_z[0..8] registers (0x024..0x044) accept distinct 16-bit writes and read back cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) {
            if (!rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), 0x0210u + 5u * i, 0x3u, &cyc)) return false;
        }
        for (int i = 0; i < 9; i++) {
            uint32_t rd = 0;
            if (!rtl.axil_read(avid_l5::L5_OFF_SHADOW_Z(i), &rd, &cyc) || rd != (0x0210u + 5u * i)) return false;
        }
        det = "9/9 shadow_z offsets (0x024..0x044) verified";
        return true;
    });

    /* L5-AXIL-10: Valid control/status reads at 0x048, 0x04C, 0x050, 0x054, 0x058 */
    suite.run_case("L5-AXIL-10", "G1", "Control & status registers (0x048 COMMIT, 0x04C STATUS, 0x050 EPOCH, 0x054 K, 0x058 OSD) return OKAY", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t c = 9, s = 9, e = 9, k = 0, m = 9;
        if (!rtl.axil_read(0x048u, &c, &cyc) || !rtl.axil_read(0x04Cu, &s, &cyc)) return false;
        if (!rtl.axil_read(0x050u, &e, &cyc) || !rtl.axil_read(0x054u, &k, &cyc)) return false;
        if (!rtl.axil_read(0x058u, &m, &cyc)) return false;
        det = "COMMIT=" + std::to_string(c) + " STATUS=" + std::to_string(s) +
              " EPOCH=" + std::to_string(e) + " K=" + std::to_string(k) + " OSD=" + std::to_string(m);
        return c == 0u && s == 0u && e == 0u && k == 3u && m == 1u;
    });

    /* L5-AXIL-11: Valid RO readback across all 18 active_x/z registers (0x100..0x144) */
    suite.run_case("L5-AXIL-11", "G1,G4", "All 18 active_x/z read-only registers (0x100..0x144) return committed Q7.8 weights with RRESP=OKAY", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), 0x0130u + i, 0x3u, &cyc);
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), 0x0230u + i, 0x3u, &cyc);
        }
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 9; i++) {
            uint32_t rx = 0, rz = 0;
            if (!rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(i), &rx, &cyc) || rx != (0x0130u + i)) return false;
            if (!rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &rz, &cyc) || rz != (0x0230u + i)) return false;
        }
        det = "18/18 active_x/z registers (0x100..0x144) matched committed values";
        return true;
    });

    /* L5-AXIL-12: WSTRB=4'b0011 full halfword write & WSTRB=4'b1100 upper-halfword ignore */
    suite.run_case("L5-AXIL-12", "G1", "WSTRB=0x3 writes lower 16 bits while WSTRB=0xC ignores upper 16 bits [31:16]", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t rd = 0;
        rtl.axil_write(0x000u, 0xDEAD1234u, 0x3u, &cyc);
        rtl.axil_write(0x000u, 0xFFFFFFFFu, 0xCu, &cyc);
        rtl.axil_read(0x000u, &rd, &cyc);
        det = "rdata after WSTRB=0xC write of 0xFFFFFFFF -> 0x" + std::to_string(rd);
        return rd == 0x1234u;
    });

    /* L5-AXIL-13: WSTRB=4'b0001 low-byte [7:0] selective update */
    suite.run_case("L5-AXIL-13", "G1", "WSTRB=0x1 updates byte [7:0] while preserving byte [15:8]", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t rd = 0;
        rtl.axil_write(0x004u, 0x1234u, 0x3u, &cyc);
        rtl.axil_write(0x004u, 0x99AAu, 0x1u, &cyc);
        rtl.axil_read(0x004u, &rd, &cyc);
        det = "0x1234 + WSTRB=0x1(0x99AA) -> 0x12AA (got " + std::to_string(rd) + ")";
        return rd == 0x12AAu;
    });

    /* L5-AXIL-14: WSTRB=4'b0010 high-byte [15:8] selective update */
    suite.run_case("L5-AXIL-14", "G1", "WSTRB=0x2 updates byte [15:8] while preserving byte [7:0]", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t rd = 0;
        rtl.axil_write(0x004u, 0x12AAu, 0x3u, &cyc);
        rtl.axil_write(0x004u, 0x56FFu, 0x2u, &cyc);
        rtl.axil_read(0x004u, &rd, &cyc);
        det = "0x12AA + WSTRB=0x2(0x56FF) -> 0x56AA (got " + std::to_string(rd) + ")";
        return rd == 0x56AAu;
    });

    /* L5-AXIL-15: WSTRB=4'b0000 zero-strobe behavior across shadow vs config registers */
    suite.run_case("L5-AXIL-15", "G1", "WSTRB=0x0 leaves shadow registers unchanged and rejects 0x054/0x058 config writes with SLVERR", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t rd = 0;
        rtl.axil_write(0x000u, 0x0345u, 0x3u, &cyc);
        bool sh_ok = rtl.axil_write(0x000u, 0x9999u, 0x0u, &cyc);
        rtl.axil_read(0x000u, &rd, &cyc);
        bool k_rej = !rtl.axil_write(0x054u, 4u, 0x0u, &cyc) && (rtl.last_bresp == 2u);
        bool m_rej = !rtl.axil_write(0x058u, 2u, 0x0u, &cyc) && (rtl.last_bresp == 2u);
        det = "shadow_unchanged=" + std::to_string(rd == 0x0345u) + " k_rej=" + std::to_string(k_rej) + " m_rej=" + std::to_string(m_rej);
        return sh_ok && rd == 0x0345u && k_rej && m_rej;
    });

    /* L5-AXIL-16: Halfword/byte torn-write SLVERR rejection when swap_pending && !bp_ready_in */
    suite.run_case("L5-AXIL-16", "G1,G4", "Halfword & byte writes (WSTRB=0x1, 0x2, 0x3) to shadow banks rejected with SLVERR during pending commit", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x000u, 0x0180u, 0x3u, &cyc);
        rtl.top->s_axis_syndrome_tdata = 0x06u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(2); /* bp_core_busy_mon == 1 */
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc); /* swap_pending == 1 */
        rtl.bp_busy = true;
        bool rej_b0 = !rtl.axil_write(0x000u, 0x00FFu, 0x1u, &cyc) && (rtl.last_bresp == 2u);
        bool rej_b1 = !rtl.axil_write(0x000u, 0x7F00u, 0x2u, &cyc) && (rtl.last_bresp == 2u);
        bool rej_hw = !rtl.axil_write(0x024u, 0x03FFu, 0x3u, &cyc) && (rtl.last_bresp == 2u);
        rtl.step_clock(24);
        uint32_t ax0 = 0;
        rtl.axil_read(0x100u, &ax0, &cyc);
        det = "SLVERR on WSTRB {0x1,0x2,0x3} torn attempts; committed active_x[0]=0x0180";
        return rej_b0 && rej_b1 && rej_hw && ax0 == 0x0180u;
    });

    /* L5-AXIL-17: Unaligned write addresses (+1, +2, +3) SLVERR rejection */
    suite.run_case("L5-AXIL-17", "G1", "Unaligned write addresses (+1, +2, +3 across register space) return SLVERR (2'b10) with zero mutation", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint32_t bad_addrs[6] = {0x001u, 0x002u, 0x003u, 0x025u, 0x049u, 0x056u};
        for (uint32_t a : bad_addrs) {
            if (rtl.axil_write(a, 0x0111u, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
        }
        uint32_t rd0 = 0;
        rtl.axil_read(0x000u, &rd0, &cyc);
        det = "6/6 unaligned writes returned BRESP=2'b10; shadow_x[0]=0x0232";
        return rd0 == 0x0232u;
    });

    /* L5-AXIL-18: Unaligned read addresses (+1, +2, +3) SLVERR & 0xDEADBEEF */
    suite.run_case("L5-AXIL-18", "G1", "Unaligned read addresses (+1, +2, +3) return RRESP=2'b10 (SLVERR) and RDATA=0xDEADBEEF", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint32_t bad_addrs[5] = {0x001u, 0x002u, 0x003u, 0x04Du, 0x102u};
        for (uint32_t a : bad_addrs) {
            uint32_t rd = 0;
            if (rtl.axil_read(a, &rd, &cyc) || rtl.last_rresp != 2u || rd != 0xDEADBEEFu) return false;
        }
        det = "5/5 unaligned reads returned RRESP=2'b10 & RDATA=0xDEADBEEF";
        return true;
    });

    /* L5-AXIL-19: Unmapped mid-map gap (0x05C..0x0FC) SLVERR rejection */
    suite.run_case("L5-AXIL-19", "G1", "Unmapped gap offsets 0x05C..0x0FC rejected on read and write with SLVERR", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t a : {0x05Cu, 0x060u, 0x080u, 0x0C0u, 0x0FCu}) {
            uint32_t rd = 0;
            if (rtl.axil_write(a, 0x1u, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
            if (rtl.axil_read(a, &rd, &cyc) || rtl.last_rresp != 2u || rd != 0xDEADBEEFu) return false;
        }
        det = "5/5 gap offsets (0x05C..0x0FC) rejected with SLVERR";
        return true;
    });

    /* L5-AXIL-20: Upper unmapped addresses (>= 0x148) SLVERR rejection */
    suite.run_case("L5-AXIL-20", "G1", "Upper unmapped offsets (0x148, 0x200, 0xFFC, 0xFFFFFFFC) rejected with SLVERR", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t a : {0x148u, 0x200u, 0xFFCu, 0xFFFFFFFCu}) {
            uint32_t rd = 0;
            if (rtl.axil_write(a, 0x1u, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
            if (rtl.axil_read(a, &rd, &cyc) || rtl.last_rresp != 2u || rd != 0xDEADBEEFu) return false;
        }
        det = "4/4 upper unmapped addresses rejected with SLVERR";
        return true;
    });

    /* L5-AXIL-21: Read-only STATUS (0x04C) and EPOCH_COUNT (0x050) write rejection */
    suite.run_case("L5-AXIL-21", "G1,G4", "Writes to read-only STATUS (0x04C) and EPOCH_COUNT (0x050) rejected with SLVERR without mutation", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        bool w1 = !rtl.axil_write(0x04Cu, 0x3u, 0xFu, &cyc) && (rtl.last_bresp == 2u);
        bool w2 = !rtl.axil_write(0x050u, 0x99u, 0xFu, &cyc) && (rtl.last_bresp == 2u);
        uint32_t st = 9, ep = 9;
        rtl.axil_read(0x04Cu, &st, &cyc);
        rtl.axil_read(0x050u, &ep, &cyc);
        det = "STATUS=" + std::to_string(st) + " EPOCH_COUNT=" + std::to_string(ep) + " intact";
        return w1 && w2 && st == 0u && ep == 0u;
    });

    /* L5-AXIL-22: Read-only active_x/z (0x100..0x144) direct write prohibition across all 18 regs */
    suite.run_case("L5-AXIL-22", "G1,G4", "Direct AXI4-Lite writes to all 18 active_x/z registers (0x100..0x144) return SLVERR", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 9; i++) {
            if (rtl.axil_write(avid_l5::L5_OFF_ACTIVE_X(i), 0x0100u, 0x3u, &cyc) || rtl.last_bresp != 2u) return false;
            if (rtl.axil_write(avid_l5::L5_OFF_ACTIVE_Z(i), 0x0100u, 0x3u, &cyc) || rtl.last_bresp != 2u) return false;
        }
        uint32_t ax4 = 0;
        rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(4), &ax4, &cyc);
        det = "18/18 active_x/z write attempts returned SLVERR; active_x[4]=0x0232";
        return ax4 == 0x0232u;
    });

    /* L5-AXIL-23: Valid CFG_MAX_ITER (0x054) domain K in {1, 2, 3, 4, 5} */
    suite.run_case("L5-AXIL-23", "G1,G2", "CFG_MAX_ITER (0x054) accepts all supported K in {1, 2, 3, 4, 5} with exact readback", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t k = 1; k <= 5; k++) {
            uint32_t rd = 0;
            if (!rtl.axil_write(0x054u, k, 0x1u, &cyc)) return false;
            if (!rtl.axil_read(0x054u, &rd, &cyc) || rd != k) return false;
        }
        det = "K in {1,2,3,4,5} accepted and read back cleanly";
        return true;
    });

    /* L5-AXIL-24: Out-of-range CFG_MAX_ITER K in {0, 6, 7, 8, 15, 255} SLVERR rejection (DEF-L5-02) */
    suite.run_case("L5-AXIL-24", "G1", "Out-of-range CFG_MAX_ITER values {0, 6, 7, 8, 15, 255} rejected with SLVERR preserving prior K", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x054u, 4u, 0x1u, &cyc);
        for (uint32_t bad_k : {0u, 6u, 7u, 8u, 15u, 255u}) {
            if (rtl.axil_write(0x054u, bad_k, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
        }
        uint32_t rd = 0;
        rtl.axil_read(0x054u, &rd, &cyc);
        det = "rejected {0,6,7,8,15,255} with SLVERR; preserved K=" + std::to_string(rd);
        return rd == 4u;
    });

    /* L5-AXIL-25: Non-zero upper bits on CFG_MAX_ITER (0x13, 0x103, 0xFFFF0003) SLVERR rejection */
    suite.run_case("L5-AXIL-25", "G1", "CFG_MAX_ITER rejects upper-bit aliasing patterns {0x13, 0x103, 0xFFFF0003} with SLVERR", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t bad_k : {0x13u, 0x103u, 0xFFFF0003u}) {
            if (rtl.axil_write(0x054u, bad_k, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
        }
        uint32_t rd = 0;
        rtl.axil_read(0x054u, &rd, &cyc);
        det = "upper-bit garbage rejected with SLVERR; K remained " + std::to_string(rd);
        return rd == 3u;
    });

    /* L5-AXIL-26: Valid OSD_MODE (0x058) domain {0, 1, 2} */
    suite.run_case("L5-AXIL-26", "G1,G2", "OSD_MODE (0x058) accepts {0 (Pure BP), 1 (Inline OSD-0), 2 (Post-BP OSD-0)} with exact readback", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t m = 0; m <= 2; m++) {
            uint32_t rd = 99;
            if (!rtl.axil_write(0x058u, m, 0x1u, &cyc)) return false;
            if (!rtl.axil_read(0x058u, &rd, &cyc) || rd != m) return false;
        }
        det = "OSD_MODE {0,1,2} accepted and read back cleanly";
        return true;
    });

    /* L5-AXIL-27: Out-of-range OSD_MODE {3, 4, 7, 16, 255, 0x101} SLVERR rejection */
    suite.run_case("L5-AXIL-27", "G1", "Out-of-range OSD_MODE values {3, 4, 7, 16, 255, 0x101} rejected with SLVERR preserving prior mode", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x058u, 2u, 0x1u, &cyc);
        for (uint32_t bad_m : {3u, 4u, 7u, 16u, 255u, 0x101u}) {
            if (rtl.axil_write(0x058u, bad_m, 0xFu, &cyc) || rtl.last_bresp != 2u) return false;
        }
        uint32_t rd = 0;
        rtl.axil_read(0x058u, &rd, &cyc);
        det = "rejected {3,4,7,16,255,0x101} with SLVERR; preserved OSD_MODE=" + std::to_string(rd);
        return rd == 2u;
    });

    /* L5-AXIL-28: Q7.8 non-negative boundaries (0, +1, +562, +5120, +32767) */
    suite.run_case("L5-AXIL-28", "G1,G6", "Q7.8 non-negative boundaries {0, +1, +562 (0x0232), +5120 (0x1400), +32767 (0x7FFF)} round-trip cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int16_t vals[5] = {0, 1, 562, 5120, 32767};
        for (int i = 0; i < 5; i++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), static_cast<uint16_t>(vals[i]), 0x3u, &cyc);
        }
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 5; i++) {
            uint32_t rd = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(i), &rd, &cyc);
            if (static_cast<int16_t>(rd & 0xFFFFu) != vals[i]) return false;
        }
        det = "verified Q7.8 boundaries {0, +1, +562, +5120, +32767} in active bank";
        return true;
    });

    /* L5-AXIL-29: Q7.8 negative two's-complement boundaries (-32768, -5120, -1) & sign-extension */
    suite.run_case("L5-AXIL-29", "G1,G6", "Q7.8 negative boundaries {-32768 (0x8000), -5120 (0xEC00), -1 (0xFFFF)} preserve two's-complement bits & sign", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int16_t negs[3] = {-32768, -5120, -1};
        for (int i = 0; i < 3; i++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), static_cast<uint16_t>(negs[i]), 0x3u, &cyc);
        }
        rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
        for (int i = 0; i < 3; i++) {
            uint32_t rd = 0;
            rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(i), &rd, &cyc);
            if ((rd >> 16) != 0u || static_cast<int16_t>(rd & 0xFFFFu) != negs[i]) return false;
        }
        /* Decode under negative weights (OSD_MODE=1 clamps post_w<=0 to w_c=1 without overflow) */
        RtlDecodeTrace tr = rtl.decode_syndrome(0x1u, 0x0u, false);
        uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x1u, rtl.active_z, 3, false, 1);
        det = "{-32768, -5120, -1} stored as exact 16-bit signed Q7.8; decode corr_z matched golden=" + std::to_string(exp_cz);
        return tr.corr_z == exp_cz;
    });

    /* L5-AXIL-30: COMMIT (0x048) write=0 / wstrb=0 no-op vs write=1 pulse */
    suite.run_case("L5-AXIL-30", "G1,G4", "COMMIT (0x048) treats wdata=0 or wstrb=0 as safe OKAY no-op and triggers swap strictly on wdata[0]&wstrb[0]", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.axil_write(0x000u, 0x0199u, 0x3u, &cyc);
        if (!rtl.axil_write(0x048u, 0u, 0x1u, &cyc)) return false;
        if (!rtl.axil_write(0x048u, 1u, 0x0u, &cyc)) return false;
        uint32_t ep0 = 99, ax0 = 0;
        rtl.axil_read(0x050u, &ep0, &cyc);
        rtl.axil_read(0x100u, &ax0, &cyc);
        if (ep0 != 0u || ax0 != 0x0232u) return false;
        if (!rtl.axil_write(0x048u, 1u, 0x1u, &cyc)) return false;
        uint32_t ep1 = 0, ax1 = 0;
        rtl.axil_read(0x050u, &ep1, &cyc);
        rtl.axil_read(0x100u, &ax1, &cyc);
        det = "no-op ep=" + std::to_string(ep0) + ", after wdata=1&wstrb=1 ep=" + std::to_string(ep1) + " active_x[0]=0x0199";
        return ep1 == 1u && ax1 == 0x0199u;
    });

    return suite.finalize();
}
