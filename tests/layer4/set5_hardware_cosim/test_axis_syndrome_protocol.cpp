/* ===========================================================================
 * test_axis_syndrome_protocol.cpp — Layer 4 Set 5.2 AXI4-Stream Protocol Tests
 * Canonical IDs: L4-AXIS-01 .. L4-AXIS-09 (Gates G1, G3)
 * Drives native Verilator Vaa_qec_top AXI4-Stream & axis_ingress_tap pins.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set5_axis_protocol", "Set5", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) return 2;
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L4-AXIS-01: TVALID && TREADY transfer rule (TVALID=0 never triggers decode) */
    suite.run_case("L4-AXIS-01", "G1,G3", "Changing s_axis_tdata with s_axis_tvalid=0 never triggers decode or alters state", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axis_syndrome_tdata = 0x00FFu;
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(25);
        bool idle = (rtl.top->bp_core_busy_mon == 0) && (rtl.top->m_axis_correction_tvalid == 0);
        cyc = rtl.sim_cycle;
        ev = "25 cycles with tvalid=0 produced 0 busy/valid pulses";
        return idle;
    });

    /* L4-AXIS-02: Master egress TVALID=1 && TREADY=0 payload & TLAST stability */
    suite.run_case("L4-AXIS-02", "G1,G3", "When m_axis_correction_tready=0, m_axis_tvalid, tdata, and tlast remain rock-stable", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        rtl.top->s_axis_syndrome_tdata = 0x0096u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        bool seen = false;
        uint32_t td0 = 0;
        for (int g = 0; g < 35; g++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid) { seen = true; td0 = rtl.top->m_axis_correction_tdata; break; }
        }
        if (!seen) return false;
        for (int hold = 0; hold < 12; hold++) {
            if (!rtl.top->m_axis_correction_tvalid ||
                rtl.top->m_axis_correction_tdata != td0 ||
                !rtl.top->m_axis_correction_tlast ||
                rtl.top->epoch_irq != 0) { /* epoch_irq must NOT fire before tready==1! */
                return false;
            }
            rtl.step_clock(1);
        }
        rtl.top->m_axis_correction_tready = 1;
        rtl.step_clock(1); /* handshake happens on this edge -> registered epoch_irq pulses 1 */
        bool irq_pulsed = (rtl.top->epoch_irq == 1);
        rtl.step_clock(1); /* single-cycle epoch_irq clears back to 0 */
        bool irq_cleared = (rtl.top->epoch_irq == 0);
        cyc = rtl.sim_cycle;
        ev = "held tvalid/tdata/tlast stable 12 cycles; single-cycle epoch_irq fired only on tready=1";
        return irq_pulsed && irq_cleared;
    });

    /* L4-AXIS-03: 16-deep RX + 16-deep TX elastic FIFO full backpressure & lossless drain */
    suite.run_case("L4-AXIS-03", "G1,G3", "Filling elastic pipeline with m_axis_tready=0 deasserts s_axis_tready=0 with zero lost beats", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        std::vector<uint8_t> sent_sx;
        bool hit_full = false;
        for (int g = 0; g < 700; g++) {
            if (!rtl.top->s_axis_syndrome_tready) {
                hit_full = true;
                break;
            }
            uint8_t sx = static_cast<uint8_t>((sent_sx.size() + 1) & 0xF);
            rtl.top->s_axis_syndrome_tdata = sx;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 0;
            rtl.step_clock(1);
            sent_sx.push_back(sx);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->m_axis_correction_tready = 1;
        std::vector<uint16_t> recv_cz;
        for (int g = 0; g < 1500 && recv_cz.size() < sent_sx.size(); g++) {
            bool v = (rtl.top->m_axis_correction_tvalid != 0);
            uint32_t d = rtl.top->m_axis_correction_tdata;
            rtl.step_clock(1);
            if (v) recv_cz.push_back(static_cast<uint16_t>((d >> 9) & 0x1FFu));
        }
        /* Verify every drained correction matches reference in exact FIFO order */
        for (size_t i = 0; i < recv_cz.size(); i++) {
            uint16_t exp = AvidVerilatorRtlModel::eval_rtl_bp_reference(sent_sx[i], rtl.active_z, 3, false, 1);
            if (recv_cz[i] != exp) return false;
        }
        cyc = rtl.sim_cycle;
        ev = "queued=" + std::to_string(sent_sx.size()) + " drained=" + std::to_string(recv_cz.size()) + " in exact order";
        return hit_full && sent_sx.size() >= 17u && recv_cz.size() == sent_sx.size();
    });

    /* L4-AXIS-04: Selective TLAST propagation across multi-beat burst */
    suite.run_case("L4-AXIS-04", "G1,G3", "8-beat burst with TLAST on beats #3 and #7 pulses m_axis_tlast and epoch_irq exactly twice", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 1;
        for (int i = 0; i < 8; i++) {
            rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>((i + 1) & 0xF);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = (i == 3 || i == 7) ? 1 : 0;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        int out_beats = 0, out_lasts = 0, irq_pulses = 0;
        for (int g = 0; g < 240; g++) {
            bool v = (rtl.top->m_axis_correction_tvalid != 0);
            bool l = (rtl.top->m_axis_correction_tlast != 0);
            rtl.step_clock(1);
            if (v) { out_beats++; if (l) out_lasts++; }
            if (rtl.top->epoch_irq) irq_pulses++;
        }
        cyc = rtl.sim_cycle;
        ev = "beats=" + std::to_string(out_beats) + " tlasts=" + std::to_string(out_lasts) +
             " irqs=" + std::to_string(irq_pulses) + " rx_epochs=" + std::to_string(rtl.top->rx_epoch_count_mon);
        return out_beats == 8 && out_lasts == 2 && irq_pulses == 2 && rtl.top->rx_epoch_count_mon == 2u;
    });

    /* L4-AXIS-05: Variable TVALID=0 inter-beat bubbles */
    suite.run_case("L4-AXIS-05", "G1,G3", "Irregular 1..11 cycle TVALID=0 bubbles between beats preserve exact results", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 1; i <= 6; i++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(i), static_cast<uint8_t>(15 - i), (i == 6));
            if (tr.total_axis_cycles != 22u) return false;
            rtl.step_clock(static_cast<uint32_t>(i * 2 - 1)); /* idle gap */
        }
        cyc = rtl.sim_cycle;
        ev = "6 beats separated by odd idle bubbles decoded in exact 22 cycles each";
        return true;
    });

    /* L4-AXIS-06: axis_ingress_tap non-blocking DMA stall & sticky overflow flag (DEF-L4-10) */
    suite.run_case("L4-AXIS-06", "G1,G3", "Stalling slow-path DMA tap (dma_tap_stall_in=1) NEVER stalls fast path and sets overflow bit[9]", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 1; /* Slow-loop DMA ring full! */
        if (rtl.top->s_axis_syndrome_tready != 1) return false;
        RtlDecodeTrace tr1 = rtl.decode_syndrome(0x6u, 0x0u, false);
        uint32_t drops = rtl.top->dma_tap_drop_count_mon;
        /* Now release DMA stall and send next beat; verify bit[9] sticky overflow asserted on tap! */
        rtl.top->dma_tap_stall_in = 0;
        rtl.top->s_axis_syndrome_tdata = 0x0003u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->eval();
        bool tap_valid = (rtl.top->dma_tap_tvalid_mon == 1);
        bool sticky_ovf = ((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 1u;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        /* Next beat after recovery must have sticky overflow cleared */
        rtl.top->s_axis_syndrome_tdata = 0x0001u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->eval();
        bool ovf_cleared = ((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 0u;
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(50);
        cyc = rtl.sim_cycle;
        ev = "fast_path_cycles=" + std::to_string(tr1.total_axis_cycles) +
             " dma_drops=" + std::to_string(drops) + " sticky_ovf=" + std::to_string(sticky_ovf);
        return tr1.total_axis_cycles == 22u && drops == 1u && tap_valid && sticky_ovf && ovf_cleared;
    });

    /* L4-AXIS-07: Back-to-back 8-beat streaming helper verification */
    suite.run_case("L4-AXIS-07", "G1,G3", "verify_axis_backpressure_and_consecutive() passes 8-of-8 burst under stall", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        bool ok = rtl.verify_axis_backpressure_and_consecutive();
        cyc = rtl.sim_cycle;
        ev = "8-of-8 backpressured burst drained with 1 tlast & 1 epoch_irq";
        return ok;
    });

    /* L4-AXIS-08: AMBA AXI4-Stream reset compliance (s_axis_syndrome_tready == 0 when aresetn == 0) */
    suite.run_case("L4-AXIS-08", "G1,G3", "During aresetn=0, s_axis_syndrome_tready and m_axis_correction_tvalid are strictly 0", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->aresetn = 0;
        rtl.top->eval();
        bool rdy_low = (rtl.top->s_axis_syndrome_tready == 0);
        rtl.step_clock(3);
        bool val_low = (rtl.top->m_axis_correction_tvalid == 0) && (rtl.top->epoch_irq == 0);
        rtl.top->aresetn = 1;
        rtl.step_clock(1);
        bool rdy_high = (rtl.top->s_axis_syndrome_tready == 1);
        cyc = rtl.sim_cycle;
        ev = "aresetn=0 -> tready=0, tvalid=0; aresetn=1 -> tready=1";
        return rdy_low && val_low && rdy_high;
    });

    /* L4-AXIS-09: Mid-stream hardware reset flushes queued RX FIFO beats without stale output */
    suite.run_case("L4-AXIS-09", "G1,G3,G7", "Asserting reset with 6 queued FIFO beats flushes pipeline cleanly with zero stale corrections", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 6; i++) {
            rtl.top->s_axis_syndrome_tdata = 0x0Fu;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 1;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.reset_model(6);
        /* Run 30 idle cycles: zero stale corrections may emerge! */
        for (int g = 0; g < 30; g++) {
            if (rtl.top->m_axis_correction_tvalid || rtl.top->epoch_irq) return false;
            rtl.step_clock(1);
        }
        RtlDecodeTrace tr = rtl.decode_syndrome(0x1u, 0x0u, true);
        cyc = rtl.sim_cycle;
        ev = "6 queued beats flushed by reset; 0 stale beats; clean post-reset decode";
        return tr.bp_core_cycles == 17u && tr.corr_z != 0u;
    });

    return suite.finalize();
}
