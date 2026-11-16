/* ===========================================================================
 * test_ingress_tap_pipeline.cpp — Layer 5 Set 3.1 Ingress Tap & Pipeline Flow
 * Canonical IDs: L5-FLOW-01 .. L5-FLOW-16 (Gates G2, G3)
 * Verifies axis_ingress_tap -> axis_syndrome_rx -> bp_engine -> axis_correction_tx
 * and proves dma_tap_stall_in=1 never stalls or slows the fast BP path.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set3_ingress_tap_pipeline", "Set3", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) return 2;
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-FLOW-01: Zero-cycle cut-through of axis_ingress_tap on fast BP path */
    suite.run_case("L5-FLOW-01", "G2,G3", "axis_ingress_tap adds 0 clock cycles to fast BP path (22 total AXI cycles @ K=3)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 0;
        RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, true);
        cyc = rtl.sim_cycle;
        det = "ingress=" + std::to_string(tr.axi_ingress_cycles) + " core=" + std::to_string(tr.bp_core_cycles) +
              " egress=" + std::to_string(tr.axi_egress_cycles) + " total=" + std::to_string(tr.total_axis_cycles);
        return tr.axi_ingress_cycles == 2u && tr.bp_core_cycles == 17u && tr.total_axis_cycles == 22u;
    });

    /* L5-FLOW-02: Normal slow-DMA tap mirroring when dma_tap_stall_in == 0 */
    suite.run_case("L5-FLOW-02", "G3", "With dma_tap_stall_in=0, dma_tap_tvalid_mon mirrors accepted beats with drop_count=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 0;
        rtl.top->s_axis_syndrome_tdata = 0x00A5u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->eval();
        bool valid_mirror = (rtl.top->dma_tap_tvalid_mon == 1) && (rtl.top->dma_tap_tdata_mon == 0x00A5u);
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(24);
        cyc = rtl.sim_cycle;
        det = "dma_tvalid=" + std::to_string(valid_mirror) + " drop_count=" + std::to_string(rtl.top->dma_tap_drop_count_mon);
        return valid_mirror && rtl.top->dma_tap_drop_count_mon == 0u;
    });

    /* L5-FLOW-03: dma_tap_stall_in=1 increments dma_tap_drop_count_mon per accepted beat */
    suite.run_case("L5-FLOW-03", "G3", "Asserting dma_tap_stall_in=1 increments dma_tap_drop_count_mon by +1 per accepted syndrome", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 1;
        for (int i = 1; i <= 5; i++) {
            rtl.decode_syndrome(static_cast<uint8_t>(i), 0u, false);
            if (rtl.top->dma_tap_drop_count_mon != static_cast<uint32_t>(i)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "5 syndromes under dma_tap_stall_in=1 -> dma_tap_drop_count_mon=" + std::to_string(rtl.top->dma_tap_drop_count_mon);
        return true;
    });

    /* L5-FLOW-04: Zero fast-path stall or cycle increase under dma_tap_stall_in=1 */
    suite.run_case("L5-FLOW-04", "G2,G3", "dma_tap_stall_in=1 causes zero stall (s_axis_tready=1) and zero cycle increase (22 cycles) on fast BP path", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace clean_tr = rtl.decode_syndrome(0x6u, 0x9u, true);
        rtl.top->dma_tap_stall_in = 1;
        rtl.top->eval();
        bool rdy_unstalled = (rtl.top->s_axis_syndrome_tready == 1);
        RtlDecodeTrace stalled_tr = rtl.decode_syndrome(0x6u, 0x9u, true);
        cyc = rtl.sim_cycle;
        det = "clean_axis=" + std::to_string(clean_tr.total_axis_cycles) +
              " dma_stalled_axis=" + std::to_string(stalled_tr.total_axis_cycles) +
              " s_tready=" + std::to_string(rdy_unstalled);
        return rdy_unstalled &&
               stalled_tr.bp_core_cycles == clean_tr.bp_core_cycles &&
               stalled_tr.total_axis_cycles == clean_tr.total_axis_cycles &&
               stalled_tr.corr_x == clean_tr.corr_x &&
               stalled_tr.corr_z == clean_tr.corr_z;
    });

    /* L5-FLOW-05: Sticky overflow_warn flag (tdata[9]) injection & auto-clear on DMA recovery */
    suite.run_case("L5-FLOW-05", "G3", "After DMA stall drop, next unstalled DMA beat asserts overflow_warn (tdata[9]=1) and clears on subsequent beat", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        /* Beat 1: stall DMA -> sticky_overflow becomes 1 */
        rtl.top->dma_tap_stall_in = 1;
        rtl.top->s_axis_syndrome_tdata = 0x0006u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.step_clock(1);
        /* Beat 2: release DMA stall -> dma_tap_tdata_mon[9] MUST be 1! */
        rtl.top->dma_tap_stall_in = 0;
        rtl.top->s_axis_syndrome_tdata = 0x0003u;
        rtl.top->eval();
        bool flag_set = ((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 1u;
        rtl.step_clock(1);
        /* Beat 3: next unstalled beat -> sticky_overflow cleared to 0! */
        rtl.top->s_axis_syndrome_tdata = 0x0001u;
        rtl.top->eval();
        bool flag_cleared = ((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 0u;
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(70);
        cyc = rtl.sim_cycle;
        det = "beat2_overflow_warn=" + std::to_string(flag_set) + " beat3_cleared=" + std::to_string(flag_cleared);
        return flag_set && flag_cleared && rtl.top->dma_tap_drop_count_mon == 1u;
    });

    /* L5-FLOW-06: Preservation of incoming s_axis_tdata[9]=1 (meas_flip_flag) when unstalled */
    suite.run_case("L5-FLOW-06", "G3", "Incoming s_axis_tdata[9]=1 passes transparently through dma_tap_tdata_mon[9] when unstalled", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 0;
        rtl.top->s_axis_syndrome_tdata = 0x0206u; /* bit 9 set */
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->eval();
        bool bit9_kept = ((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 1u;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.step_clock(24);
        cyc = rtl.sim_cycle;
        det = "dma_tap_tdata_mon=0x" + std::to_string(rtl.top->dma_tap_tdata_mon) + " bit9_kept=" + std::to_string(bit9_kept);
        return bit9_kept && rtl.top->dma_tap_drop_count_mon == 0u;
    });

    /* L5-FLOW-07: Sustained 256-beat syndrome stream under 100% permanent dma_tap_stall_in=1 */
    suite.run_case("L5-FLOW-07", "G3,G5", "256-syndrome stream under permanent dma_tap_stall_in=1 delivers 256/256 fast corrections with drop_count=256", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 1;
        for (int i = 0; i < 256; i++) {
            uint8_t sx = static_cast<uint8_t>(i & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i >> 4) & 0xFu);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (i == 255));
            if (tr.bp_core_cycles != 17u || tr.total_axis_cycles != 22u) return false;
            if (avid_l5::gf2_hx_mul_ez(tr.corr_z) != sx || avid_l5::gf2_hz_mul_ex(tr.corr_x) != sz) return false;
        }
        cyc = rtl.sim_cycle;
        det = "256/256 fast corrections exact; dma_tap_drop_count_mon=" + std::to_string(rtl.top->dma_tap_drop_count_mon);
        return rtl.top->dma_tap_drop_count_mon == 256u;
    });

    /* L5-FLOW-08: Intermittent bursty dma_tap_stall_in (stall 3, accept 2 x 20 = 60 drops) */
    suite.run_case("L5-FLOW-08", "G3", "Alternating DMA stall pattern (3 stalled, 2 unstalled x 20) counts exact 60 drops", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 100; i++) {
            rtl.top->dma_tap_stall_in = ((i % 5) < 3) ? 1 : 0;
            rtl.decode_syndrome(static_cast<uint8_t>((i % 15) + 1), 0u, false);
        }
        cyc = rtl.sim_cycle;
        det = "100 packets (60% DMA stalled) -> dma_tap_drop_count_mon=" + std::to_string(rtl.top->dma_tap_drop_count_mon);
        return rtl.top->dma_tap_drop_count_mon == 60u;
    });

    /* L5-FLOW-09: Hardware reset clears dma_tap_drop_count_mon and sticky_overflow */
    suite.run_case("L5-FLOW-09", "G3,G5", "aresetn=0 resets dma_tap_drop_count_mon=0 and clears sticky_overflow", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->dma_tap_stall_in = 1;
        rtl.decode_syndrome(0x6u, 0x0u, false);
        if (rtl.top->dma_tap_drop_count_mon != 1u) return false;
        rtl.reset_model(6);
        rtl.top->s_axis_syndrome_tdata = 0x0001u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->eval();
        bool clean_after = (rtl.top->dma_tap_drop_count_mon == 0u) && (((rtl.top->dma_tap_tdata_mon >> 9) & 1u) == 0u);
        rtl.top->s_axis_syndrome_tvalid = 0;
        cyc = rtl.sim_cycle;
        det = "post-reset drop_count=0 & bit9=0: " + std::to_string(clean_after);
        return clean_after;
    });

    /* L5-FLOW-10: Unaccepted beat (RX FIFO full -> m_axis_bp_tready=0) does NOT increment dma_drop_count */
    suite.run_case("L5-FLOW-10", "G3,G5", "When RX FIFO is full (s_axis_tready=0), dma_tap_stall_in=1 does not over-count unaccepted beats", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        rtl.top->dma_tap_stall_in = 1;
        int accepted = 0;
        /* Push until s_axis_syndrome_tready drops to 0, then hold tvalid=1 for 10 extra cycles */
        for (int c = 0; c < 45; c++) {
            rtl.top->s_axis_syndrome_tdata = 0x06u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            bool will_accept = (rtl.top->s_axis_syndrome_tready != 0);
            rtl.step_clock(1);
            if (will_accept) accepted++;
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        uint32_t drops = rtl.top->dma_tap_drop_count_mon;
        cyc = rtl.sim_cycle;
        det = "accepted=" + std::to_string(accepted) + " dma_drop_count=" + std::to_string(drops);
        return accepted >= 17 && drops == static_cast<uint32_t>(accepted);
    });

    /* L5-FLOW-11: 4-epoch streaming window (4 x 64 = 256 syndromes) epoch_irq & rx_epoch_count parity */
    suite.run_case("L5-FLOW-11", "G2,G3", "4x64-round streaming window triggers exactly 4 epoch_irq pulses and rx_epoch_count_mon==4", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int irqs = 0;
        for (int r = 0; r < 256; r++) {
            bool is_end = ((r % 64) == 63);
            RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(r & 0xFu), static_cast<uint8_t>((r >> 4) & 0xFu), is_end);
            if (tr.epoch_irq_pulsed != is_end) return false;
            if (tr.epoch_irq_pulsed) irqs++;
        }
        cyc = rtl.sim_cycle;
        det = "256 rounds (4 epochs) -> irqs=" + std::to_string(irqs) + " rx_epoch_count=" + std::to_string(rtl.top->rx_epoch_count_mon);
        return irqs == 4 && rtl.top->rx_epoch_count_mon == 4u;
    });

    /* L5-FLOW-12: Dynamic runtime CFG_MAX_ITER transitions (1->2->3->4->5->3) without reset */
    suite.run_case("L5-FLOW-12", "G2,G3", "Runtime CFG_MAX_ITER transitions K=1->2->3->4->5->3 switch core latency cleanly without reset", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t k : {1u, 2u, 3u, 4u, 5u, 3u}) {
            rtl.axil_write(0x054u, k, 0x1u, &cyc);
            RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, false);
            if (tr.bp_core_cycles != (2u + 5u * k)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "6 sequential runtime K reconfigurations matched 2+5*K";
        return true;
    });

    /* L5-FLOW-13: Dynamic runtime OSD_MODE transitions (0->1->2->0->1) without reset */
    suite.run_case("L5-FLOW-13", "G2,G3", "Runtime OSD_MODE transitions 0->1->2->0->1 switch between 17-cycle and 19-cycle paths cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t m : {0u, 1u, 2u, 0u, 1u}) {
            rtl.axil_write(0x058u, m, 0x1u, &cyc);
            RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, false);
            uint32_t exp_c = (m == 2u) ? 19u : 17u;
            if (tr.bp_core_cycles != exp_c) return false;
        }
        cyc = rtl.sim_cycle;
        det = "5 sequential runtime OSD_MODE transitions verified";
        return true;
    });

    /* L5-FLOW-14: Mid-decode CFG_MAX_ITER & OSD_MODE latch immunity (lat_max_iter / lat_osd_mode) */
    suite.run_case("L5-FLOW-14", "G2,G3", "Writing 0x054 (K=5) and 0x058 (Mode=2) mid-decode does not alter in-flight K=3 Mode=1 decode", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t acc = 0;
        if (!rtl.start_async_syndrome(0x6u, 0x9u, true, &acc)) return false;
        rtl.axil_write(0x054u, 5u, 0x1u, &cyc);
        rtl.axil_write(0x058u, 2u, 0x1u, &cyc);
        RtlDecodeTrace inflight;
        rtl.finish_async_syndrome(&inflight);
        /* Next decode now uses K=5, Mode=2 -> 4 + 25 = 29 core cycles! */
        RtlDecodeTrace next_tr = rtl.decode_syndrome(0x6u, 0x9u, true);
        cyc = rtl.sim_cycle;
        det = "inflight_core=" + std::to_string(inflight.bp_core_cycles) + " next_core=" + std::to_string(next_tr.bp_core_cycles);
        return inflight.bp_core_cycles == 17u && next_tr.bp_core_cycles == 29u;
    });

    /* L5-FLOW-15: Pipelined back-to-back RX FIFO queueing & continuous back-to-back core execution */
    suite.run_case("L5-FLOW-15", "G3,G5", "12 back-to-back queued syndromes drain continuously through bp_engine without dropped or reordered packets", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int i = 0; i < 12; i++) {
            rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>((i + 1) & 0xFu);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = (i == 11) ? 1 : 0;
            rtl.step_clock(1);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        std::vector<uint8_t> rec_sx;
        for (int g = 0; g < 300 && rec_sx.size() < 12; g++) {
            if (rtl.top->m_axis_correction_tvalid && rtl.top->m_axis_correction_tready) {
                uint16_t cz = static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu);
                rec_sx.push_back(avid_l5::gf2_hx_mul_ez(cz));
            }
            rtl.step_clock(1);
        }
        for (int i = 0; i < 12; i++) {
            if (i >= static_cast<int>(rec_sx.size()) || rec_sx[i] != ((i + 1) & 0xFu)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "12/12 back-to-back queued syndromes drained in exact FIFO order";
        return true;
    });

    /* L5-FLOW-16: Concurrent AXI4-Stream decoding + AXI4-Lite STATUS/EPOCH polling */
    suite.run_case("L5-FLOW-16", "G1,G3", "Continuous AXI4-Lite STATUS/EPOCH polling during active AXI4-Stream decoding causes zero interference", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t acc = 0;
        rtl.start_async_syndrome(0x6u, 0x9u, true, &acc);
        uint32_t st = 0, ep = 99;
        rtl.axil_read(0x04Cu, &st, &cyc);
        rtl.axil_read(0x050u, &ep, &cyc);
        bool saw_busy = ((st >> 1) & 1u) == 1u;
        RtlDecodeTrace tr;
        rtl.finish_async_syndrome(&tr);
        cyc = rtl.sim_cycle;
        det = "mid_decode STATUS=0x" + std::to_string(st) + " core_cyc=" + std::to_string(tr.bp_core_cycles);
        return saw_busy && ep == 0u && tr.bp_core_cycles == 17u && tr.corr_x == 0x10u && tr.corr_z == 0x10u;
    });

    return suite.finalize();
}
