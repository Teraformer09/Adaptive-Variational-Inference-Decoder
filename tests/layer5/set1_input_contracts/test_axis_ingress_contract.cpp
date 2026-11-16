/* ===========================================================================
 * test_axis_ingress_contract.cpp — Layer 5 Set 1.1 AXI4-Stream Ingress Tests
 * Canonical IDs: L5-I01 .. L5-I14 (Gates G1, G3)
 * Drives native Verilator Vaa_qec_top s_axis_syndrome_* pins directly.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set1_axis_ingress_contract", "Set1", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] test_axis_ingress_contract requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-I01: Valid single-beat s_axis_syndrome handshake (tvalid=1 && tready=1, tlast=1) */
    suite.run_case("L5-I01", "G1", "Single-beat s_axis_syndrome handshake accepted in 1 cycle with tready=1 and tlast=1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        bool rdy_before = (rtl.top->s_axis_syndrome_tready == 1);
        RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x0u, true);
        cyc = rtl.sim_cycle;
        det = "tready=" + std::to_string(rdy_before) + " bp_cyc=" + std::to_string(tr.bp_core_cycles) +
              " corr_z=0x" + std::to_string(tr.corr_z) + " irq=" + std::to_string(tr.epoch_irq_pulsed);
        return rdy_before && tr.bp_core_cycles == 17u && tr.corr_z == 0x10u && tr.epoch_irq_pulsed;
    });

    /* L5-I02: Non-epoch beat handshake with s_axis_syndrome_tlast=0 */
    suite.run_case("L5-I02", "G1", "Non-epoch s_axis_syndrome beat (tlast=0) accepted cleanly without asserting epoch_irq", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, false);
        cyc = rtl.sim_cycle;
        det = "tlast=0 -> corr_x=" + std::to_string(tr.corr_x) + " corr_z=" + std::to_string(tr.corr_z) +
              " epoch_irq=" + std::to_string(tr.epoch_irq_pulsed);
        return tr.corr_x == 0x10u && tr.corr_z == 0x10u && !tr.epoch_irq_pulsed;
    });

    /* L5-I03: Idle s_axis_syndrome_tvalid=0 across 64 cycles causes zero spurious decodes */
    suite.run_case("L5-I03", "G1", "Holding s_axis_syndrome_tvalid=0 across 64 clock cycles keeps decoder idle with zero spurious outputs", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axis_syndrome_tdata = 0x00FFu;
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 1;
        int spurious = 0;
        for (int i = 0; i < 64; i++) {
            rtl.step_clock(1);
            if (rtl.top->bp_core_busy_mon || rtl.top->m_axis_correction_tvalid || rtl.top->epoch_irq) {
                spurious++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "64 idle cycles with tdata=0xFF & tvalid=0 -> spurious_events=" + std::to_string(spurious);
        return spurious == 0;
    });

    /* L5-I04: 4-bit s_x = tdata[3:0] bit-packing isolation across all 16 values */
    suite.run_case("L5-I04", "G1,G6", "tdata[3:0] routes s_x exclusively to Z-error BP engine (corr_x==0 across all 16 sx values)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int sx = 0; sx < 16; sx++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), 0u, false);
            if (tr.corr_x != 0u) return false;
            if (sx != 0 && tr.corr_z == 0u) return false;
            if (avid_l5::gf2_hx_mul_ez(tr.corr_z) != static_cast<uint8_t>(sx)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "16/16 sx values isolated to corr_z with corr_x==0 and Hx*corr_z==sx";
        return true;
    });

    /* L5-I05: 4-bit s_z = tdata[7:4] bit-packing isolation across all 16 values */
    suite.run_case("L5-I05", "G1,G6", "tdata[7:4] routes s_z exclusively to X-error BP engine (corr_z==0 across all 16 sz values)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int sz = 0; sz < 16; sz++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(0u, static_cast<uint8_t>(sz), false);
            if (tr.corr_z != 0u) return false;
            if (sz != 0 && tr.corr_x == 0u) return false;
            if (avid_l5::gf2_hz_mul_ex(tr.corr_x) != static_cast<uint8_t>(sz)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "16/16 sz values isolated to corr_x with corr_z==0 and Hz*corr_x==sz";
        return true;
    });

    /* L5-I06: Simultaneous dual-basis (sx, sz) bit-packing orthogonality */
    suite.run_case("L5-I06", "G1", "Combined tdata[7:0]=(sz<<4)|sx produces identical corr_x/corr_z to separate basis decodes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint8_t pairs[8][2] = {{0x1,0x8},{0x6,0x9},{0x3,0xC},{0xF,0x5},{0xA,0x2},{0x4,0xE},{0xD,0x7},{0xB,0xB}};
        for (int i = 0; i < 8; i++) {
            uint8_t sx = pairs[i][0], sz = pairs[i][1];
            RtlDecodeTrace tx_only = rtl.decode_syndrome(sx, 0u, false);
            RtlDecodeTrace tz_only = rtl.decode_syndrome(0u, sz, false);
            RtlDecodeTrace t_joint = rtl.decode_syndrome(sx, sz, false);
            if (t_joint.corr_z != tx_only.corr_z || t_joint.corr_x != tz_only.corr_x) return false;
        }
        cyc = rtl.sim_cycle;
        det = "8/8 joint (sx, sz) decodes matched orthogonal basis superpositions";
        return true;
    });

    /* L5-I07: Upper-byte flag/metadata bits tdata[15:8] ignored by fast-path decoder */
    suite.run_case("L5-I07", "G1", "Non-zero upper flag bits tdata[15:8] (valid_mask, meas_flip, reserved) do not alter fast-path BP output", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace ref_tr = rtl.decode_syndrome(0x6u, 0x9u, false);
        const uint16_t upper_masks[4] = {0x0100u, 0x0200u, 0x5500u, 0xFF00u};
        for (uint16_t um : upper_masks) {
            rtl.top->s_axis_syndrome_tdata = um | 0x0096u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 0;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            uint32_t out_word = 0;
            bool got = false;
            for (int g = 0; g < 40; g++) {
                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid) {
                    out_word = rtl.top->m_axis_correction_tdata;
                    got = true;
                    rtl.step_clock(1);
                    break;
                }
            }
            if (!got) return false;
            uint16_t cx = static_cast<uint16_t>(out_word & 0x1FFu);
            uint16_t cz = static_cast<uint16_t>((out_word >> 9) & 0x1FFu);
            if (cx != ref_tr.corr_x || cz != ref_tr.corr_z) return false;
        }
        cyc = rtl.sim_cycle;
        det = "4/4 upper-byte masks (0x0100..0xFF00) produced identical corr_x/corr_z";
        return true;
    });

    /* L5-I08: Back-to-back single-cycle s_axis_syndrome beat burst ingestion */
    suite.run_case("L5-I08", "G1,G5", "8 consecutive single-cycle s_axis_syndrome beats ingested at 1 beat/cycle into RX FIFO", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint8_t sxs[8] = {0x1, 0x2, 0x4, 0x8, 0x3, 0x6, 0xC, 0x9};
        for (int i = 0; i < 8; i++) {
            if (!rtl.top->s_axis_syndrome_tready) return false;
            rtl.top->s_axis_syndrome_tdata = sxs[i];
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = (i == 7) ? 1 : 0;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        std::vector<uint16_t> got_cz;
        for (int g = 0; g < 220 && got_cz.size() < 8; g++) {
            if (rtl.top->m_axis_correction_tvalid && rtl.top->m_axis_correction_tready) {
                got_cz.push_back(static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu));
            }
            rtl.step_clock(1);
        }
        for (size_t i = 0; i < got_cz.size(); i++) {
            if (avid_l5::gf2_hx_mul_ez(got_cz[i]) != sxs[i]) return false;
        }
        cyc = rtl.sim_cycle;
        det = "ingested & decoded " + std::to_string(got_cz.size()) + "/8 back-to-back beats in order";
        return got_cz.size() == 8;
    });

    /* L5-I09: Intermittent idle gaps (1, 3, 7 cycles) between s_axis_syndrome beats */
    suite.run_case("L5-I09", "G1", "Variable inter-packet idle gaps (1, 3, 7 cycles) preserve exact per-packet decode determinism", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int gaps[4] = {1, 3, 7, 2};
        for (int i = 0; i < 4; i++) {
            rtl.step_clock(gaps[i]);
            RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(i + 1), static_cast<uint8_t>(8 - i), false);
            if (tr.bp_core_cycles != 17u || tr.total_axis_cycles != 22u) return false;
        }
        cyc = rtl.sim_cycle;
        det = "4/4 gapped syndrome packets completed in exact 17 core / 22 AXI cycles";
        return true;
    });

    /* L5-I10: s_axis_syndrome_tready strictly 0 during active-low reset (DEF-L5-01) */
    suite.run_case("L5-I10", "G1,G5", "s_axis_syndrome_tready gated to 0 throughout aresetn==0 per AMBA AXI4-Stream specification", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->aresetn = 0;
        rtl.top->eval();
        bool rdy_low = (rtl.top->s_axis_syndrome_tready == 0);
        for (int i = 0; i < 4; i++) {
            rtl.step_clock(1);
            if (rtl.top->s_axis_syndrome_tready != 0) rdy_low = false;
        }
        rtl.top->aresetn = 1;
        rtl.step_clock(1);
        bool rdy_high = (rtl.top->s_axis_syndrome_tready == 1);
        cyc = rtl.sim_cycle;
        det = "tready_during_reset=" + std::to_string(!rdy_low) + " tready_after_reset=" + std::to_string(rdy_high);
        return rdy_low && rdy_high;
    });

    /* L5-I11: Asserting s_axis_syndrome_tvalid=1 while aresetn==0 queues zero FIFO entries */
    suite.run_case("L5-I11", "G1,G5", "Driving s_axis_syndrome_tvalid=1 while aresetn==0 is ignored and leaves RX FIFO empty post-reset", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->aresetn = 0;
        rtl.top->s_axis_syndrome_tdata = 0x0066u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.step_clock(5);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        rtl.top->aresetn = 1;
        int ghost_outputs = 0;
        for (int i = 0; i < 35; i++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid || rtl.top->bp_core_busy_mon) ghost_outputs++;
        }
        cyc = rtl.sim_cycle;
        det = "ghost_outputs_post_reset=" + std::to_string(ghost_outputs);
        return ghost_outputs == 0;
    });

    /* L5-I12: rx_epoch_count_mon increments strictly on s_axis_syndrome_tlast=1 pops */
    suite.run_case("L5-I12", "G1,G3", "rx_epoch_count_mon increments strictly on tlast=1 syndrome pops and holds steady on tlast=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        if (rtl.top->rx_epoch_count_mon != 0u) return false;
        for (int i = 0; i < 5; i++) rtl.decode_syndrome(0x1u, 0x0u, false);
        uint32_t after_nonlast = rtl.top->rx_epoch_count_mon;
        rtl.decode_syndrome(0x1u, 0x0u, true);
        uint32_t after_ep1 = rtl.top->rx_epoch_count_mon;
        rtl.decode_syndrome(0x2u, 0x0u, false);
        rtl.decode_syndrome(0x4u, 0x0u, true);
        uint32_t after_ep2 = rtl.top->rx_epoch_count_mon;
        cyc = rtl.sim_cycle;
        det = "rx_epoch_count: 5x(tlast=0)->" + std::to_string(after_nonlast) +
              ", ep1->" + std::to_string(after_ep1) + ", ep2->" + std::to_string(after_ep2);
        return after_nonlast == 0u && after_ep1 == 1u && after_ep2 == 2u;
    });

    /* L5-I13: Zero-syndrome (sx=0, sz=0) traverses full FSM without early-exit timing leak */
    suite.run_case("L5-I13", "G1,G2", "Zero-syndrome (0, 0) executes full 17-cycle core / 22-cycle AXI pipeline and outputs (0, 0)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace tr = rtl.decode_syndrome(0u, 0u, true);
        cyc = rtl.sim_cycle;
        det = "core_cyc=" + std::to_string(tr.bp_core_cycles) + " axis_cyc=" + std::to_string(tr.total_axis_cycles) +
              " corr=(" + std::to_string(tr.corr_x) + "," + std::to_string(tr.corr_z) + ")";
        return tr.bp_core_cycles == 17u && tr.total_axis_cycles == 22u && tr.corr_x == 0u && tr.corr_z == 0u;
    });

    /* L5-I14: Maximum popcount all-ones syndrome (sx=0xF, sz=0xF) ingestion & satisfaction */
    suite.run_case("L5-I14", "G1,G6", "All-ones syndrome (sx=0xF, sz=0xF) decodes in 17 core cycles and satisfies both Hx and Hz", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace tr = rtl.decode_syndrome(0xFu, 0xFu, true);
        bool sat_x = (avid_l5::gf2_hx_mul_ez(tr.corr_z) == 0xFu);
        bool sat_z = (avid_l5::gf2_hz_mul_ex(tr.corr_x) == 0xFu);
        cyc = rtl.sim_cycle;
        det = "corr_x=0x" + std::to_string(tr.corr_x) + " corr_z=0x" + std::to_string(tr.corr_z) +
              " sat_x=" + std::to_string(sat_x) + " sat_z=" + std::to_string(sat_z);
        return tr.bp_core_cycles == 17u && sat_x && sat_z;
    });

    return suite.finalize();
}
