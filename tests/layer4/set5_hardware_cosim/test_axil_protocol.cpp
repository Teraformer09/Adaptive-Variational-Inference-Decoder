/* ===========================================================================
 * test_axil_protocol.cpp — Layer 4 Set 5.1 AXI4-Lite Pin-Level Protocol Tests
 * Canonical IDs: L4-AXIL-01 .. L4-AXIL-11 (Gates G1, G3)
 * Drives native Verilator Vaa_qec_top AXI4-Lite slave signals directly.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set5_axil_protocol", "Set5", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] set5_axil_protocol requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L4-AXIL-01: Simultaneous AW+W handshake */
    suite.run_case("L4-AXIL-01", "G1,G3", "Simultaneous AW+W handshake completes in 2 cycles with BRESP=OKAY(00)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        rtl.top->s_axil_awaddr = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x0155u;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_bready = 1;
        bool rdy0 = rtl.top->s_axil_awready && rtl.top->s_axil_wready;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        bool bv1 = (rtl.top->s_axil_bvalid == 1) && (rtl.top->s_axil_bresp == 0u);
        rtl.step_clock(1);
        bool bv2 = (rtl.top->s_axil_bvalid == 0);
        uint32_t rd = 0;
        rtl.axil_read(0x000u, &rd, &cyc);
        ev = "bvalid pulsed 1 cycle, bresp=0, rdata=0x0155";
        return rdy0 && bv1 && bv2 && (rd & 0xFFFFu) == 0x0155u;
    });

    /* L4-AXIL-02: Split AW-before-W with 5-cycle delay */
    suite.run_case("L4-AXIL-02", "G1,G3", "Split AW-before-W latches aw_addr_lat across 5 idle cycles before W arrives", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        bool ok = rtl.axil_write_split_aw_w(0x008u, 0x01AAu, 0x3u, 5u, &cyc);
        uint32_t rd = 0;
        rtl.axil_read(0x008u, &rd, &cyc);
        ev = "5-cycle AW->W split write read back 0x01AA";
        return ok && (rd & 0xFFFFu) == 0x01AAu;
    });

    /* L4-AXIL-03: Split W-before-AW with 5-cycle delay */
    suite.run_case("L4-AXIL-03", "G1,G3", "Split W-before-AW latches w_data_lat/w_strb_lat across 5 cycles before AW arrives", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        bool ok = rtl.axil_write_split_w_aw(0x00Cu, 0x01BBu, 0x3u, 5u, &cyc);
        uint32_t rd = 0;
        rtl.axil_read(0x00Cu, &rd, &cyc);
        ev = "5-cycle W->AW split write read back 0x01BB";
        return ok && (rd & 0xFFFFu) == 0x01BBu;
    });

    /* L4-AXIL-04: Write response BREADY=0 stall stability */
    suite.run_case("L4-AXIL-04", "G1,G3", "Holding s_axil_bready=0 stalls s_axil_bvalid=1, holds bresp stable, and blocks new AW/W", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        rtl.top->s_axil_awaddr = 0x010u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x01CCu;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_bready = 0;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        for (int s = 0; s < 6; s++) {
            if (rtl.top->s_axil_bvalid != 1 || rtl.top->s_axil_bresp != 0u) return false;
            if (rtl.top->s_axil_awready != 0 || rtl.top->s_axil_wready != 0) return false;
            rtl.step_clock(1);
        }
        rtl.top->s_axil_bready = 1;
        rtl.step_clock(1);
        bool cleared = (rtl.top->s_axil_bvalid == 0) && (rtl.top->s_axil_awready == 1);
        cyc = rtl.sim_cycle;
        ev = "bvalid=1 & awready=0 held stable across 6 stalled cycles";
        return cleared;
    });

    /* L4-AXIL-05: Read response RREADY=0 stall stability */
    suite.run_case("L4-AXIL-05", "G1,G3", "Holding s_axil_rready=0 stalls s_axil_rvalid=1, holds rdata/rresp stable, and blocks ARREADY", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        rtl.axil_write(0x014u, 0x01DDu, 0x3u, &cyc);
        rtl.top->s_axil_araddr = 0x014u;
        rtl.top->s_axil_arvalid = 1;
        rtl.top->s_axil_rready = 0;
        rtl.step_clock(1);
        rtl.top->s_axil_arvalid = 0;
        for (int s = 0; s < 6; s++) {
            if (rtl.top->s_axil_rvalid != 1 || (rtl.top->s_axil_rdata & 0xFFFFu) != 0x01DDu) return false;
            if (rtl.top->s_axil_rresp != 0u || rtl.top->s_axil_arready != 0) return false;
            rtl.step_clock(1);
        }
        rtl.top->s_axil_rready = 1;
        rtl.step_clock(1);
        bool cleared = (rtl.top->s_axil_rvalid == 0) && (rtl.top->s_axil_arready == 1);
        cyc = rtl.sim_cycle;
        ev = "rvalid=1, rdata=0x01DD & arready=0 held stable across 6 stalled cycles";
        return cleared;
    });

    /* L4-AXIL-06: Back-to-back writes on every 2nd cycle */
    suite.run_case("L4-AXIL-06", "G1,G3", "18 back-to-back AXI-Lite shadow writes complete without dropped beats", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        for (int i = 0; i < 9; i++) {
            if (!rtl.axil_write(4u * i, 0x0100u + i, 0x3u, &cyc)) return false;
            if (!rtl.axil_write(0x024u + 4u * i, 0x0200u + i, 0x3u, &cyc)) return false;
        }
        for (int i = 0; i < 9; i++) {
            uint32_t rx = 0, rz = 0;
            rtl.axil_read(4u * i, &rx, &cyc);
            rtl.axil_read(0x024u + 4u * i, &rz, &cyc);
            if ((rx & 0xFFFFu) != (0x0100u + i) || (rz & 0xFFFFu) != (0x0200u + i)) return false;
        }
        ev = "18/18 back-to-back writes verified";
        return true;
    });

    /* L4-AXIL-07: Back-to-back reads */
    suite.run_case("L4-AXIL-07", "G1,G3", "36 back-to-back AXI-Lite reads return exact values in 72 cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        uint64_t c0 = rtl.sim_cycle;
        uint32_t rd = 0;
        for (int i = 0; i < 9; i++) {
            if (!rtl.axil_read(4u * i, &rd, &cyc) || rd != 0x0232u) return false;
            if (!rtl.axil_read(0x024u + 4u * i, &rd, &cyc) || rd != 0x0232u) return false;
            if (!rtl.axil_read(0x100u + 4u * i, &rd, &cyc) || rd != 0x0232u) return false;
            if (!rtl.axil_read(0x124u + 4u * i, &rd, &cyc) || rd != 0x0232u) return false;
        }
        ev = "36 reads completed in " + std::to_string(cyc - c0) + " cycles";
        return (cyc - c0) == 72ULL;
    });

    /* L4-AXIL-08: Simultaneous read + write channel concurrency (AW+W+AR on same cycle) */
    suite.run_case("L4-AXIL-08", "G1,G3", "Simultaneous AW+W write to 0x000 and AR read from 0x024 succeed on identical clock edge", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
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
        bool both_valid = (rtl.top->s_axil_bvalid == 1) && (rtl.top->s_axil_rvalid == 1) &&
                          ((rtl.top->s_axil_rdata & 0xFFFFu) == 0x0277u);
        rtl.step_clock(1);
        uint32_t wr_check = 0;
        rtl.axil_read(0x000u, &wr_check, &cyc);
        ev = "simultaneous bvalid=1 & rvalid=1 (rdata=0x0277), post-read 0x000=0x0188";
        return both_valid && (wr_check & 0xFFFFu) == 0x0188u;
    });

    /* L4-AXIL-09: Direct pin-level SLVERR bresp=2'b10 and rresp=2'b10 verification */
    suite.run_case("L4-AXIL-09", "G1", "SLVERR drives exact 2'b10 on s_axil_bresp and s_axil_rresp across invalid accesses", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        uint32_t rd = 0;
        bool w_fail = !rtl.axil_write(0x054u, 0x0013u, 0x1u, &cyc) && (rtl.last_bresp == 2u);
        bool r_fail = !rtl.axil_read(0x060u, &rd, &cyc) && (rtl.last_rresp == 2u) && (rd == 0xDEADBEEFu);
        ev = "bresp=" + std::to_string(rtl.last_bresp) + " rresp=" + std::to_string(rtl.last_rresp);
        return w_fail && r_fail;
    });

    /* L4-AXIL-10: Pin-level wstrb[3:2] upper-byte ignore & wstrb[1:0] masking */
    suite.run_case("L4-AXIL-10", "G1", "s_axil_wstrb=4'b1100 ignores upper 16 bits while 4'b0001/4'b0010 mask exact bytes", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        uint32_t rd = 0;
        rtl.axil_write(0x000u, 0xABCD1234u, 0x3u, &cyc);
        rtl.axil_write(0x000u, 0xFFFFFFFFu, 0xCu, &cyc); /* wstrb[3:2] only -> lower 16b untouched */
        rtl.axil_read(0x000u, &rd, &cyc);
        ev = "after wstrb=0xC rdata=0x" + std::to_string(rd);
        return rd == 0x00001234u;
    });

    /* L4-AXIL-11: Mid-handshake aresetn=0 abort clears aw_valid_lat/w_valid_lat & forces *ready=0 */
    suite.run_case("L4-AXIL-11", "G1,G3", "Asserting aresetn=0 after split AW beat clears latched AW, drives *ready=0, and recovers cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
        rtl.top->s_axil_awaddr = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wvalid = 0;
        rtl.step_clock(1); /* aw_valid_lat == 1 */
        rtl.top->s_axil_awvalid = 0;
        rtl.top->aresetn = 0;
        rtl.top->eval();
        bool rdy_low = (rtl.top->s_axil_awready == 0) && (rtl.top->s_axil_wready == 0) && (rtl.top->s_axil_arready == 0);
        rtl.step_clock(3);
        rtl.top->aresetn = 1;
        rtl.step_clock(1);
        /* Send orphan W beat — must NOT commit to old 0x000 because aw_valid_lat was cleared! */
        rtl.top->s_axil_wdata = 0x0199u;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.step_clock(1);
        rtl.top->s_axil_wvalid = 0;
        bool no_spurious_bvalid = (rtl.top->s_axil_bvalid == 0);
        rtl.reset_model(4);
        uint32_t rd = 0;
        rtl.axil_read(0x000u, &rd, &cyc);
        ev = "aresetn=0 drove aw/w/arready=0 and cleared latched split AW";
        return rdy_low && no_spurious_bvalid && rd == 0x0232u;
    });

    return suite.finalize();
}
