/* ===========================================================================
 * test_reset_and_backpressure.cpp — Layer 5 Set 4.1 Reset, Backpressure & FIFOs
 * Canonical IDs: L5-EXEC-01 .. L5-EXEC-22 (Gates G1, G2, G3, G4, G5, G8)
 * Exercises synchronous reset across every BP FSM state, AXI4-Stream egress
 * backpressure, 16-entry RX/TX elastic FIFO watermarks, and VCD waveform dump
 * on the compiled native Verilator Vaa_qec_top RTL model (--assert).
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set4_reset_and_backpressure", "Set4", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] test_reset_and_backpressure requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-EXEC-01: Power-on reset defaults across all 41 mapped registers & top outputs */
    suite.run_case("L5-EXEC-01", "G5", "Power-on reset defaults verified across all 41 AXI4-Lite registers and top-level outputs", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 10);
        bool top_ok = (rtl.top->m_axis_correction_tvalid == 0) &&
                      (rtl.top->m_axis_correction_tdata == 0u) &&
                      (rtl.top->m_axis_correction_tlast == 0) &&
                      (rtl.top->epoch_irq == 0) &&
                      (rtl.top->bp_core_busy_mon == 0) &&
                      (rtl.top->dma_tap_drop_count_mon == 0u) &&
                      (rtl.top->rx_epoch_count_mon == 0u) &&
                      (rtl.top->s_axis_syndrome_tready == 1);
        bool regs_ok = true;
        uint64_t c = 0;
        uint32_t v = 0;
        for (int q = 0; q < 9; q++) {
            regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_SHADOW_X(q), &v, &c) && (v == 0x0232u);
            regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_SHADOW_Z(q), &v, &c) && (v == 0x0232u);
            regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_ACTIVE_X(q), &v, &c) && (v == 0x0232u);
            regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_ACTIVE_Z(q), &v, &c) && (v == 0x0232u);
        }
        uint32_t commit_v = 99, status_v = 99, epoch_v = 99, k_v = 99, osd_v = 99;
        regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_COMMIT, &commit_v, &c) && (commit_v == 0u);
        regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_STATUS, &status_v, &c) && (status_v == 0u);
        regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_EPOCH_COUNT, &epoch_v, &c) && (epoch_v == 0u);
        regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_CFG_MAX_ITER, &k_v, &c) && (k_v == 3u);
        regs_ok = regs_ok && rtl.axil_read(avid_l5::L5_OFF_OSD_MODE, &osd_v, &c) && (osd_v == 1u);
        cyc = rtl.sim_cycle;
        det = "top_ok=" + std::to_string(top_ok) + " regs_41_ok=" + std::to_string(regs_ok) +
              " K=" + std::to_string(k_v) + " OSD=" + std::to_string(osd_v);
        return top_ok && regs_ok;
    });

    /* L5-EXEC-02: Short (2-cycle) vs long (32-cycle) synchronous reset parity */
    suite.run_case("L5-EXEC-02", "G5", "Short (2-cycle) and extended (32-cycle) synchronous resets restore identical state and latency", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl_short(4, 2);
        AvidVerilatorRtlModel rtl_long(4, 32);
        RtlDecodeTrace t1 = rtl_short.decode_syndrome(0x6u, 0x9u, true);
        RtlDecodeTrace t2 = rtl_long.decode_syndrome(0x6u, 0x9u, true);
        cyc = rtl_short.sim_cycle + rtl_long.sim_cycle;
        det = "short_axis=" + std::to_string(t1.total_axis_cycles) +
              " long_axis=" + std::to_string(t2.total_axis_cycles);
        return t1.total_axis_cycles == 22u && t2.total_axis_cycles == 22u &&
               t1.corr_x == t2.corr_x && t1.corr_z == t2.corr_z;
    });

    /* Helper lambda to inject syndrome, step to exact core busy cycle target, assert aresetn=0, and verify clean recovery */
    auto test_reset_at_busy_cycle = [](uint32_t osd_mode, uint32_t busy_step_after_accept,
                                       uint64_t &cyc_out, std::string &det_out) -> bool {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        if (osd_mode != 1u) {
            rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, osd_mode, 0x1u, &c);
        }
        rtl.top->s_axis_syndrome_tdata  = 0x96u; /* sx=0x6, sz=0x9 */
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast  = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast  = 0;
        rtl.step_clock(busy_step_after_accept);
        bool was_busy = (rtl.top->bp_core_busy_mon == 1);

        rtl.reset_model(6);
        bool idle_after = (rtl.top->bp_core_busy_mon == 0) &&
                          (rtl.top->m_axis_correction_tvalid == 0) &&
                          (rtl.top->epoch_irq == 0);
        /* Ensure 24 idle clocks produce zero ghost outputs */
        bool ghost = false;
        for (int i = 0; i < 24; i++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid || rtl.top->epoch_irq) ghost = true;
        }
        RtlDecodeTrace rec = rtl.decode_syndrome(0x6u, 0x9u, true);
        cyc_out = rtl.sim_cycle;
        det_out = "was_busy=" + std::to_string(was_busy) + " ghost=" + std::to_string(ghost) +
                  " rec_core=" + std::to_string(rec.bp_core_cycles) +
                  " corr_x=0x" + std::to_string(rec.corr_x) + " corr_z=0x" + std::to_string(rec.corr_z);
        return was_busy && idle_after && !ghost &&
               rec.bp_core_cycles == 17u && rec.total_axis_cycles == 22u &&
               avid_l5::gf2_hz_mul_ex(rec.corr_x) == 0x9u &&
               avid_l5::gf2_hx_mul_ez(rec.corr_z) == 0x6u;
    };

    /* L5-EXEC-03: Mid-decode reset in ST_INIT (step 2 after accept = core busy cycle 1) */
    suite.run_case("L5-EXEC-03", "G5", "Synchronous mid-decode aresetn=0 in ST_INIT aborts cleanly and recovers 17-cycle decode", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        return test_reset_at_busy_cycle(1u, 2u, cyc, det);
    });

    /* L5-EXEC-04: Mid-decode reset in ST_V2C (step 3 after accept = core busy cycle 2) */
    suite.run_case("L5-EXEC-04", "G5", "Synchronous mid-decode aresetn=0 in ST_V2C aborts cleanly with zero ghost outputs", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        return test_reset_at_busy_cycle(1u, 3u, cyc, det);
    });

    /* L5-EXEC-05: Mid-decode reset across all 3 ST_WAIT_C2V pipeline stages (steps 4, 5, 6) */
    suite.run_case("L5-EXEC-05", "G5", "Synchronous mid-decode aresetn=0 across all 3 ST_WAIT_C2V check_node_minsum stages flushes cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint64_t c4 = 0, c5 = 0, c6 = 0;
        std::string d4, d5, d6;
        bool ok4 = test_reset_at_busy_cycle(1u, 4u, c4, d4);
        bool ok5 = test_reset_at_busy_cycle(1u, 5u, c5, d5);
        bool ok6 = test_reset_at_busy_cycle(1u, 6u, c6, d6);
        cyc = c4 + c5 + c6;
        det = "wait_s0=" + std::to_string(ok4) + " wait_s1=" + std::to_string(ok5) + " wait_s2=" + std::to_string(ok6);
        return ok4 && ok5 && ok6;
    });

    /* L5-EXEC-06: Mid-decode reset in ST_BELIEF_UPD (step 7 after accept = core busy cycle 6) */
    suite.run_case("L5-EXEC-06", "G5", "Synchronous mid-decode aresetn=0 in ST_BELIEF_UPD clears beliefs and recovers cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        return test_reset_at_busy_cycle(1u, 7u, cyc, det);
    });

    /* L5-EXEC-07: Mid-decode reset in ST_HARD_DECIDE (step 18 after accept = core busy cycle 17) */
    suite.run_case("L5-EXEC-07", "G5", "Synchronous mid-decode aresetn=0 in ST_HARD_DECIDE suppresses correction_valid and recovers", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        return test_reset_at_busy_cycle(1u, 18u, cyc, det);
    });

    /* L5-EXEC-08: Mid-decode reset in ST_OSD_STEP1 & ST_OSD_STEP2 (OSD_MODE=2, steps 19 & 20) */
    suite.run_case("L5-EXEC-08", "G5", "Synchronous mid-decode aresetn=0 in ST_OSD_STEP1 and ST_OSD_STEP2 (Mode 2) aborts and restores Mode 1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint64_t c19 = 0, c20 = 0;
        std::string d19, d20;
        bool ok19 = test_reset_at_busy_cycle(2u, 19u, c19, d19);
        bool ok20 = test_reset_at_busy_cycle(2u, 20u, c20, d20);
        cyc = c19 + c20;
        det = "osd_step1=" + std::to_string(ok19) + " osd_step2=" + std::to_string(ok20);
        return ok19 && ok20;
    });

    /* L5-EXEC-09: Mid-handshake AXI4-Lite reset (split AW-only, split W-only, stalled BVALID) */
    suite.run_case("L5-EXEC-09", "G1,G5", "Asserting aresetn=0 mid-AXI4-Lite split AW/W or stalled BVALID clears latches without lockup", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        /* Latch AW without W, then reset */
        rtl.top->s_axil_awaddr  = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wvalid  = 0;
        rtl.step_clock(1);
        rtl.top->s_axil_awvalid = 0;
        bool aw_latched = (rtl.top->s_axil_awready == 0);
        rtl.reset_model(4);
        bool aw_cleared = (rtl.top->s_axil_awready == 1);

        /* Latch W without AW, then reset */
        rtl.top->s_axil_wdata  = 0x0456u;
        rtl.top->s_axil_wstrb  = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.step_clock(1);
        rtl.top->s_axil_wvalid = 0;
        bool w_latched = (rtl.top->s_axil_wready == 0);
        rtl.reset_model(4);
        bool w_cleared = (rtl.top->s_axil_wready == 1);

        uint64_t c = 0;
        uint32_t rd = 0;
        bool post_ok = rtl.axil_write(0x000u, 0x0345u, 0x3u, &c) &&
                       rtl.axil_read(0x000u, &rd, &c) && (rd == 0x0345u);
        cyc = rtl.sim_cycle;
        det = "aw_lat=" + std::to_string(aw_latched) + "->" + std::to_string(aw_cleared) +
              " w_lat=" + std::to_string(w_latched) + "->" + std::to_string(w_cleared) +
              " post_ok=" + std::to_string(post_ok);
        return aw_latched && aw_cleared && w_latched && w_cleared && post_ok;
    });

    /* L5-EXEC-10: m_axis_correction_tready=0 holds tvalid, tdata, and tlast stable across 32 cycles */
    suite.run_case("L5-EXEC-10", "G2,G5", "m_axis_correction_tready=0 holds tvalid=1, tdata, and tlast stable across 32 stalled cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        rtl.top->s_axis_syndrome_tdata    = 0x96u;
        rtl.top->s_axis_syndrome_tvalid   = 1;
        rtl.top->s_axis_syndrome_tlast    = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid   = 0;
        rtl.top->s_axis_syndrome_tlast    = 0;

        bool seen = false;
        uint32_t held_data = 0;
        uint8_t  held_last = 0;
        for (int g = 0; g < 35; g++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid) {
                seen      = true;
                held_data = rtl.top->m_axis_correction_tdata;
                held_last = rtl.top->m_axis_correction_tlast;
                break;
            }
        }
        bool stable = seen && (held_last == 1);
        for (int s = 0; s < 32; s++) {
            rtl.step_clock(1);
            if (!rtl.top->m_axis_correction_tvalid ||
                rtl.top->m_axis_correction_tdata != held_data ||
                rtl.top->m_axis_correction_tlast != held_last ||
                rtl.top->epoch_irq != 0) {
                stable = false;
            }
        }
        rtl.top->m_axis_correction_tready = 1;
        rtl.step_clock(1); /* Handshake completes */
        bool irq_fired = (rtl.top->epoch_irq == 1);
        bool cleared   = (rtl.top->m_axis_correction_tvalid == 0);
        cyc = rtl.sim_cycle;
        det = "stable_32cyc=" + std::to_string(stable) + " tdata=0x" + std::to_string(held_data) +
              " irq_on_release=" + std::to_string(irq_fired);
        return stable && irq_fired && cleared;
    });

    /* L5-EXEC-11: 8-packet burst under egress stall drains in exact FIFO order under pulsed tready */
    suite.run_case("L5-EXEC-11", "G5", "8-packet burst stalled at egress drains in exact FIFO order under pulsed m_axis_correction_tready", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        bool ok = rtl.verify_axis_backpressure_and_consecutive();
        cyc = rtl.sim_cycle;
        det = "verify_axis_backpressure_and_consecutive=" + std::to_string(ok);
        return ok;
    });

    /* L5-EXEC-12: Filling axis_correction_tx to 14-entry watermark cleanly gates sbu_ready without drops */
    suite.run_case("L5-EXEC-12", "G3,G5", "Filling axis_correction_tx to 14-entry watermark under tready=0 backpressures BP core with 0 drops", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        /* Feed 16 syndromes spaced out so TX FIFO fills to 14 entries and holds 2 in RX FIFO */
        std::vector<uint8_t> sxs(16);
        for (int i = 0; i < 16; i++) {
            sxs[i] = static_cast<uint8_t>((i % 15) + 1);
            rtl.top->s_axis_syndrome_tdata  = sxs[i];
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = (i == 15) ? 1 : 0;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast  = 0;
            rtl.step_clock(20);
        }
        /* All 14 early packets sit in TX FIFO; packets 14 & 15 sit safely in RX FIFO while sbu_ready==0 */
        bool tx_waiting = (rtl.top->m_axis_correction_tvalid == 1);
        rtl.top->m_axis_correction_tready = 1;
        std::vector<uint16_t> got_cz;
        int irq_cnt = 0;
        for (int g = 0; g < 160 && got_cz.size() < 16u; g++) {
            bool v = (rtl.top->m_axis_correction_tvalid == 1);
            uint32_t d = rtl.top->m_axis_correction_tdata;
            rtl.step_clock(1);
            if (v) got_cz.push_back(static_cast<uint16_t>((d >> 9) & 0x1FFu));
            if (rtl.top->epoch_irq) irq_cnt++;
        }
        rtl.step_clock(2);
        if (rtl.top->epoch_irq) irq_cnt++;
        bool all_match = (got_cz.size() == 16u);
        for (size_t i = 0; i < got_cz.size(); i++) {
            uint16_t exp = AvidVerilatorRtlModel::eval_rtl_bp_reference(sxs[i], rtl.active_z, 3, false, 1u);
            if (got_cz[i] != exp) all_match = false;
        }
        cyc = rtl.sim_cycle;
        det = "tx_waiting=" + std::to_string(tx_waiting) + " drained=" + std::to_string(got_cz.size()) +
              "/16 match=" + std::to_string(all_match) + " irq=" + std::to_string(irq_cnt);
        return tx_waiting && all_match && (irq_cnt == 1);
    });

    /* L5-EXEC-13: Filling 16-entry axis_syndrome_rx FIFO drives s_axis_syndrome_tready strictly to 0 */
    suite.run_case("L5-EXEC-13", "G1,G5", "Filling 16-entry axis_syndrome_rx FIFO (1 active in K=5 BP + 16 queued) drives s_axis_syndrome_tready=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, 5u, 0x1u, &c); /* 27 core cycles keeps BP busy */
        int accepted = 0;
        for (int beat = 0; beat < 20; beat++) {
            if (rtl.top->s_axis_syndrome_tready == 0) break;
            rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>((beat % 15) + 1);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 0;
            rtl.step_clock(1);
            accepted++;
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        bool full_deasserted_ready = (rtl.top->s_axis_syndrome_tready == 0);
        cyc = rtl.sim_cycle;
        det = "accepted_before_full=" + std::to_string(accepted) +
              " (1_in_BP+16_in_FIFO) tready_at_full=" + std::to_string(rtl.top->s_axis_syndrome_tready);
        return (accepted == 17) && full_deasserted_ready;
    });

    /* L5-EXEC-14: Holding tvalid=1 against full RX FIFO (tready=0) never overwrites buffered entries & drains all 17 losslessly */
    suite.run_case("L5-EXEC-14", "G1,G5", "Full 16-entry RX FIFO blocks unready beats without overwrite and drains all 17 accepted packets losslessly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, 5u, 0x1u, &c);
        std::vector<uint8_t> sent_sx;
        for (int i = 0; i < 17; i++) {
            if (!rtl.top->s_axis_syndrome_tready) return false;
            uint8_t sx = static_cast<uint8_t>((i % 15) + 1);
            sent_sx.push_back(sx);
            rtl.top->s_axis_syndrome_tdata  = sx;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = (i == 16) ? 1 : 0;
            rtl.step_clock(1);
        }
        bool was_full = (rtl.top->s_axis_syndrome_tready == 0);
        /* Drive poison pattern 0x0F while tready==0 for 5 cycles (BP still busy at K=5) */
        for (int k = 0; k < 5; k++) {
            if (rtl.top->s_axis_syndrome_tready != 0) was_full = false;
            rtl.top->s_axis_syndrome_tdata  = 0xFFu;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 1;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast  = 0;

        std::vector<uint16_t> got_cz;
        for (int g = 0; g < 700; g++) {
            bool v = (rtl.top->m_axis_correction_tvalid == 1);
            uint32_t d = rtl.top->m_axis_correction_tdata;
            rtl.step_clock(1);
            if (v) got_cz.push_back(static_cast<uint16_t>((d >> 9) & 0x1FFu));
        }
        bool exact = was_full && (got_cz.size() == 17u);
        for (size_t i = 0; i < sent_sx.size() && exact; i++) {
            uint16_t exp = AvidVerilatorRtlModel::eval_rtl_bp_reference(sent_sx[i], rtl.active_z, 5, false, 1u);
            if (got_cz[i] != exp) exact = false;
        }
        cyc = rtl.sim_cycle;
        det = "was_full=" + std::to_string(was_full) + " drained=" + std::to_string(got_cz.size()) +
              "/17 exact_order=" + std::to_string(exact);
        return exact;
    });

    /* L5-EXEC-15: 4x16-packet RX FIFO fill-and-drain pointer wrap-around (64 syndromes) */
    suite.run_case("L5-EXEC-15", "G5", "4 consecutive 16-beat RX FIFO bursts wrap wr_ptr/rd_ptr cleanly across 64 lossless decodes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int total_ok = 0;
        for (int b = 0; b < 4; b++) {
            std::vector<uint8_t> burst_sx(16);
            for (int i = 0; i < 16; i++) {
                burst_sx[i] = static_cast<uint8_t>(((b * 7 + i) % 15) + 1);
                rtl.top->s_axis_syndrome_tdata  = burst_sx[i];
                rtl.top->s_axis_syndrome_tvalid = 1;
                rtl.top->s_axis_syndrome_tlast  = (i == 15) ? 1 : 0;
                rtl.step_clock(1);
            }
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast  = 0;

            std::vector<uint16_t> out_cz;
            for (int g = 0; g < 420 && out_cz.size() < 16u; g++) {
                bool v = (rtl.top->m_axis_correction_tvalid == 1);
                uint32_t d = rtl.top->m_axis_correction_tdata;
                rtl.step_clock(1);
                if (v) out_cz.push_back(static_cast<uint16_t>((d >> 9) & 0x1FFu));
            }
            rtl.step_clock(4);
            if (out_cz.size() == 16u) {
                for (int i = 0; i < 16; i++) {
                    uint16_t exp = AvidVerilatorRtlModel::eval_rtl_bp_reference(burst_sx[i], rtl.active_z, 3, false, 1u);
                    if (out_cz[i] == exp) total_ok++;
                }
            }
        }
        cyc = rtl.sim_cycle;
        det = "matched=" + std::to_string(total_ok) + "/64 rx_epochs=" + std::to_string(rtl.top->rx_epoch_count_mon);
        return (total_ok == 64) && (rtl.top->rx_epoch_count_mon == 4u);
    });

    /* L5-EXEC-16: Simultaneous 3-way randomized handshake stress (s_axis_tvalid + m_axis_tready + dma_tap_stall_in) */
    suite.run_case("L5-EXEC-16", "G3,G5", "3-way pseudo-random s_axis_tvalid, m_axis_tready, and dma_tap_stall_in stress delivers 256/256 ordered decodes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint32_t lfsr = 0xACE1u;
        auto next_u32 = [&]() {
            lfsr ^= lfsr << 13;
            lfsr ^= lfsr >> 17;
            lfsr ^= lfsr << 5;
            return lfsr;
        };
        const size_t N = 256;
        std::vector<uint8_t> in_sx(N), in_sz(N);
        for (size_t i = 0; i < N; i++) {
            uint32_t r = next_u32();
            in_sx[i] = static_cast<uint8_t>(r & 0xFu);
            in_sz[i] = static_cast<uint8_t>((r >> 4) & 0xFu);
        }
        size_t tx_idx = 0;
        std::vector<std::pair<uint16_t, uint16_t>> rx_out;
        rx_out.reserve(N);
        bool holding_in = false;

        for (int step = 0; step < 20000 && rx_out.size() < N; step++) {
            uint32_t r = next_u32();
            rtl.top->dma_tap_stall_in         = (r & 1u) ? 1 : 0;
            rtl.top->m_axis_correction_tready = ((r >> 1) & 3u) != 0 ? 1 : 0; /* 75% ready */

            if (!holding_in && tx_idx < N && ((r >> 3) & 1u)) {
                rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>(in_sx[tx_idx] | (in_sz[tx_idx] << 4));
                rtl.top->s_axis_syndrome_tvalid = 1;
                rtl.top->s_axis_syndrome_tlast  = (tx_idx + 1 == N) ? 1 : 0;
                holding_in = true;
            } else if (!holding_in) {
                rtl.top->s_axis_syndrome_tvalid = 0;
                rtl.top->s_axis_syndrome_tlast  = 0;
            }

            bool in_hs  = (rtl.top->s_axis_syndrome_tvalid && rtl.top->s_axis_syndrome_tready);
            bool out_hs = (rtl.top->m_axis_correction_tvalid && rtl.top->m_axis_correction_tready);
            uint32_t out_w = rtl.top->m_axis_correction_tdata;

            rtl.step_clock(1);

            if (in_hs) {
                tx_idx++;
                holding_in = false;
                rtl.top->s_axis_syndrome_tvalid = 0;
                rtl.top->s_axis_syndrome_tlast  = 0;
            }
            if (out_hs) {
                rx_out.push_back({static_cast<uint16_t>(out_w & 0x1FFu),
                                  static_cast<uint16_t>((out_w >> 9) & 0x1FFu)});
            }
        }
        bool match = (rx_out.size() == N);
        for (size_t i = 0; i < N && match; i++) {
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(in_sz[i], rtl.active_x, 3, true, 1u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(in_sx[i], rtl.active_z, 3, false, 1u);
            if (rx_out[i].first != ex || rx_out[i].second != ez) match = false;
        }
        cyc = rtl.sim_cycle;
        det = "received=" + std::to_string(rx_out.size()) + "/256 match=" + std::to_string(match) +
              " dma_drops=" + std::to_string(rtl.top->dma_tap_drop_count_mon);
        return match;
    });

    /* L5-EXEC-17: Reset flush when both RX and TX FIFOs hold buffered entries */
    suite.run_case("L5-EXEC-17", "G5", "aresetn=0 with non-empty RX and TX FIFOs flushes both wr_ptr/rd_ptr=0 with zero post-reset phantom beats", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        for (int i = 0; i < 6; i++) {
            rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>(i + 1);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 1;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(30); /* First packet sits in TX FIFO; remaining 5 in BP/RX FIFO */
        bool had_tx = (rtl.top->m_axis_correction_tvalid == 1);

        rtl.reset_model(6);
        bool phantom = false;
        for (int i = 0; i < 32; i++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid || rtl.top->epoch_irq) phantom = true;
        }
        RtlDecodeTrace tr = rtl.decode_syndrome(0x3u, 0x5u, true);
        cyc = rtl.sim_cycle;
        det = "had_tx=" + std::to_string(had_tx) + " phantom=" + std::to_string(phantom) +
              " post_axis=" + std::to_string(tr.total_axis_cycles);
        return had_tx && !phantom && tr.total_axis_cycles == 22u;
    });

    /* L5-EXEC-18: AXI4-Lite SLVERR storm during active AXI4-Stream decoding */
    suite.run_case("L5-EXEC-18", "G1,G5", "64 invalid AXI4-Lite SLVERR writes injected mid-decode cause zero perturbation to 64 fast-path decodes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int slverr_cnt = 0;
        int ok_decodes = 0;
        uint32_t bad_addrs[4] = {0x001u, 0x060u, 0x100u, 0x054u};
        uint32_t bad_vals[4]  = {0x123u, 0x456u, 0x789u, 99u};
        for (int i = 0; i < 64; i++) {
            uint8_t sx = static_cast<uint8_t>(i & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i * 3) & 0xFu);
            uint64_t acc = 0, wc = 0;
            rtl.start_async_syndrome(sx, sz, (i == 63), &acc);
            bool wr_ok = rtl.axil_write(bad_addrs[i & 3], bad_vals[i & 3], 0x3u, &wc);
            if (!wr_ok && rtl.last_bresp == 2u) slverr_cnt++;
            RtlDecodeTrace tr{};
            rtl.finish_async_syndrome(&tr);
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, rtl.active_x, 3, true, 1u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, rtl.active_z, 3, false, 1u);
            if (tr.corr_x == ex && tr.corr_z == ez && tr.bp_core_cycles == 17u) ok_decodes++;
        }
        cyc = rtl.sim_cycle;
        det = "slverr_cnt=" + std::to_string(slverr_cnt) + "/64 ok_decodes=" + std::to_string(ok_decodes) + "/64";
        return (slverr_cnt == 64) && (ok_decodes == 64);
    });

    /* L5-EXEC-19: Native VerilatedVcdC waveform dump creation & signal trace verification */
    suite.run_case("L5-EXEC-19", "G5,G8", "Native VerilatedVcdC waveform dump captures aa_qec_top hierarchy, clocks, and signal transitions", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        std::string vcd_path = avid_l5::make_temp_path("trace_exec19", ".vcd");
        {
            AvidVerilatorRtlModel rtl(4, 6);
            rtl.open_waveform(vcd_path.c_str());
            rtl.reset_model(4);
            (void)rtl.decode_syndrome(0x6u, 0x9u, true);
            rtl.close_waveform();
            cyc = rtl.sim_cycle;
        }
        std::ifstream ifs(vcd_path);
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        std::remove(vcd_path.c_str());
        bool has_hdr = content.find("$timescale") != std::string::npos &&
                       content.find("aa_qec_top") != std::string::npos &&
                       content.find("aclk") != std::string::npos &&
                       content.find("bp_core_busy_mon") != std::string::npos &&
                       content.find("#") != std::string::npos;
        det = "vcd_bytes=" + std::to_string(content.size()) + " has_signals=" + std::to_string(has_hdr);
        return (content.size() > 1024u) && has_hdr;
    });

    /* L5-EXEC-20: Native Verilator linkage verification (is_native_verilator == true) */
    suite.run_case("L5-EXEC-20", "G5,G8", "Compiled binary links directly against native Verilator Vaa_qec_top (verilator_c++_rtl)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 4);
        cyc = rtl.sim_cycle;
        det = std::string("simulator=") + AvidVerilatorRtlModel::simulator_name() +
              " native=" + std::to_string(AvidVerilatorRtlModel::is_native_verilator());
        return AvidVerilatorRtlModel::is_native_verilator() &&
               std::string(AvidVerilatorRtlModel::simulator_name()) == "verilator_c++_rtl" &&
               (rtl.top != nullptr);
    });

    /* L5-EXEC-21: Reset recovery from simultaneous swap_pending=1 busy-lock + full RX FIFO */
    suite.run_case("L5-EXEC-21", "G4,G5", "Synchronous reset clears simultaneous swap_pending=1 shadow lock and full RX FIFO cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, 5u, 0x1u, &c);
        rtl.axil_write(0x000u, 0x0123u, 0x3u, &c);
        for (int i = 0; i < 17; i++) {
            rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>((i % 15) + 1);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 0;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        bool full_before = (rtl.top->s_axis_syndrome_tready == 0) && (rtl.top->bp_core_busy_mon == 1);
        rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
        uint32_t st = 0;
        rtl.axil_read(avid_l5::L5_OFF_STATUS, &st, &c);

        rtl.reset_model(6);
        uint32_t st_after = 99, ep_after = 99;
        bool wr_after = rtl.axil_write(0x000u, 0x0280u, 0x3u, &c);
        rtl.axil_read(avid_l5::L5_OFF_STATUS, &st_after, &c);
        rtl.axil_read(avid_l5::L5_OFF_EPOCH_COUNT, &ep_after, &c);
        cyc = rtl.sim_cycle;
        det = "full_before=" + std::to_string(full_before) + " status_mid=0x" + std::to_string(st) +
              " wr_after=" + std::to_string(wr_after) + " status_after=0x" + std::to_string(st_after);
        return full_before && (st == 3u) && wr_after && (st_after == 0u) &&
               (ep_after == 0u) && (rtl.top->s_axis_syndrome_tready == 1);
    });

    /* L5-EXEC-22: 1-beat-per-cycle cut-through drain rate from TX FIFO once tready asserts */
    suite.run_case("L5-EXEC-22", "G2,G5", "Buffered corrections in axis_correction_tx stream out at 1 beat/cycle (250 Mwords/s peak) upon tready=1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        for (int i = 0; i < 6; i++) {
            rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>(i + 1);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = (i == 5) ? 1 : 0;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast  = 0;
            rtl.step_clock(20);
        }
        /* All 6 corrections now queued inside axis_correction_tx FIFO */
        rtl.top->m_axis_correction_tready = 1;
        int consecutive_valid = 0;
        for (int c_idx = 0; c_idx < 6; c_idx++) {
            if (rtl.top->m_axis_correction_tvalid == 1) consecutive_valid++;
            rtl.step_clock(1);
        }
        bool empty_after = (rtl.top->m_axis_correction_tvalid == 0);
        cyc = rtl.sim_cycle;
        det = "consecutive_1cyc_beats=" + std::to_string(consecutive_valid) + "/6 empty_after=" + std::to_string(empty_after);
        return (consecutive_valid == 6) && empty_after;
    });

    return suite.finalize();
}
