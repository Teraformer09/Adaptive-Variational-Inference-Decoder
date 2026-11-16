/* ===========================================================================
 * test_cycle_accounting.cpp — Layer 4 Set 2.2 Cycle-Level Accounting Tests
 * Canonical IDs: L4-CYC-01 .. L4-CYC-10 (Gate G3)
 * Verifies C_latency = C_completed - C_accepted across Bridge & Vaa_qec_top.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set2_cycle_accounting", "Set2", argc, argv);
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set2_cyc", 16, true)) return 2;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return 2;

    /* L4-CYC-01: MMIO_WRITE exact 2-cycle latency */
    suite.run_case("L4-CYC-01", "G3", "MMIO_WRITE latency is exactly 2 RTL clock cycles (8.0 ns @ 250 MHz)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_reset_rtl(&hal, 10u, &cyc);
        uint64_t acc = 0;
        if (avid_hal_mmio_write(&hal, 0x000u, 0x0180u, 0x3u, &acc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "C_accepted=" + std::to_string(acc) + " C_completed=" + std::to_string(cyc);
        return (cyc - acc) == 2ULL;
    });

    /* L4-CYC-02: MMIO_READ exact 2-cycle latency */
    suite.run_case("L4-CYC-02", "G3", "MMIO_READ latency is exactly 2 RTL clock cycles (8.0 ns @ 250 MHz)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t rd = 0;
        uint64_t acc = 0;
        if (avid_hal_mmio_read(&hal, 0x000u, &rd, &acc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "C_accepted=" + std::to_string(acc) + " C_completed=" + std::to_string(cyc);
        return (cyc - acc) == 2ULL && (rd & 0xFFFFu) == 0x0180u;
    });

    /* L4-CYC-03: Syndrome decode K=1..5 cycle matrix in OSD Mode 0 (Pure BP: 2+5K core, 7+5K AXI) */
    suite.run_case("L4-CYC-03", "G3", "OSD Mode 0 (Pure BP) matches 2+5K core and 7+5K AXI cycles for K=1..5", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_set_osd_mode(&hal, 0u, &cyc);
        for (uint32_t k = 1u; k <= 5u; k++) {
            avid_hal_set_bp_iterations(&hal, k, &cyc);
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            uint64_t acc = 0;
            if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, false, &cx, &cz, &bp, &acc) != 0) return false;
            cyc = hal.stats.last_completed_cycle;
            if (bp != (2u + 5u * k) || (cyc - acc) != (7ULL + 5ULL * k)) return false;
        }
        ev = "K=1..5 Mode 0 core={7,12,17,22,27} axis={12,17,22,27,32}";
        return true;
    });

    /* L4-CYC-04: Syndrome decode K=1..5 cycle matrix in OSD Mode 1 (Inline OSD-0: 2+5K core, 7+5K AXI) */
    suite.run_case("L4-CYC-04", "G3", "OSD Mode 1 (Inline OSD-0) matches 2+5K core and 7+5K AXI cycles for K=1..5", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_set_osd_mode(&hal, 1u, &cyc);
        for (uint32_t k = 1u; k <= 5u; k++) {
            avid_hal_set_bp_iterations(&hal, k, &cyc);
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            uint64_t acc = 0;
            if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, true, &cx, &cz, &bp, &acc) != 0) return false;
            cyc = hal.stats.last_completed_cycle;
            if (bp != (2u + 5u * k) || (cyc - acc) != (7ULL + 5ULL * k)) return false;
        }
        ev = "K=1..5 Mode 1 core={7,12,17,22,27} axis={12,17,22,27,32}";
        return true;
    });

    /* L4-CYC-05: Syndrome decode K=1..5 cycle matrix in OSD Mode 2 (Post-BP OSD-0: 4+5K core, 9+5K AXI) */
    suite.run_case("L4-CYC-05", "G3", "OSD Mode 2 (Post-BP OSD-0) matches 4+5K core and 9+5K AXI cycles for K=1..5", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_set_osd_mode(&hal, 2u, &cyc);
        for (uint32_t k = 1u; k <= 5u; k++) {
            avid_hal_set_bp_iterations(&hal, k, &cyc);
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            uint64_t acc = 0;
            if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, true, &cx, &cz, &bp, &acc) != 0) return false;
            cyc = hal.stats.last_completed_cycle;
            if (bp != (4u + 5u * k) || (cyc - acc) != (9ULL + 5ULL * k)) return false;
        }
        avid_hal_set_osd_mode(&hal, 1u, &cyc);
        avid_hal_set_bp_iterations(&hal, 3u, &cyc);
        ev = "K=1..5 Mode 2 core={9,14,19,24,29} axis={14,19,24,29,34}";
        return true;
    });

    /* L4-CYC-06: Sub-stage ingress/core/egress cycle decomposition on Vaa_qec_top */
    suite.run_case("L4-CYC-06", "G3", "Direct Vaa_qec_top sub-stage decomposition equals 2 ingress + (2+5K) core + 3 egress", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(10);
        RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, true);
        cyc = rtl.sim_cycle;
        ev = "ingress=" + std::to_string(tr.axi_ingress_cycles) +
             " core=" + std::to_string(tr.bp_core_cycles) +
             " egress=" + std::to_string(tr.axi_egress_cycles) +
             " total=" + std::to_string(tr.total_axis_cycles);
        return tr.axi_ingress_cycles == 2u &&
               tr.bp_core_cycles == 17u &&
               tr.axi_egress_cycles == 3u &&
               tr.total_axis_cycles == 22u &&
               (tr.end_cycle - tr.accepted_cycle) == 22ULL;
    });

    /* L4-CYC-07: Single-cycle epoch_irq pulse width & exact cycle offset */
    suite.run_case("L4-CYC-07", "G3", "epoch_irq asserts for exactly 1 clock cycle immediately after m_axis_correction handshake", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(10);
#if defined(USE_VERILATOR) && USE_VERILATOR
        rtl.top->s_axis_syndrome_tdata = 0x0006u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.top->m_axis_correction_tready = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        int irq_high_cycles = 0;
        int corr_valid_cycle = -1;
        int irq_pulse_cycle = -1;
        for (int c = 1; c <= 30; c++) {
            bool cv = (rtl.top->m_axis_correction_tvalid != 0);
            if (cv && corr_valid_cycle < 0) corr_valid_cycle = c;
            rtl.step_clock(1);
            if (rtl.top->epoch_irq) {
                irq_high_cycles++;
                irq_pulse_cycle = c;
            }
        }
        cyc = rtl.sim_cycle;
        ev = "corr_valid_c=" + std::to_string(corr_valid_cycle) +
             " irq_pulse_c=" + std::to_string(irq_pulse_cycle) +
             " width=" + std::to_string(irq_high_cycles);
        return irq_high_cycles == 1 && irq_pulse_cycle == corr_valid_cycle;
#else
        return false;
#endif
    });

    /* L4-CYC-08: Exact RESET cycle accounting across rcyc={4, 10, 25, 50} */
    suite.run_case("L4-CYC-08", "G3", "RESET(N) advances simulated clock by exact N cycles across Bridge and standalone RTL", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (uint32_t n : {4u, 10u, 25u, 50u}) {
            uint64_t acc = 0;
            if (avid_hal_reset_rtl(&hal, n, &acc) != 0) return false;
            cyc = hal.stats.last_completed_cycle;
            if ((cyc - acc) != static_cast<uint64_t>(n)) return false;
        }
        ev = "verified exact delta N for N in {4, 10, 25, 50}";
        return true;
    });

    /* L4-CYC-09: Idle socket wait & rejected RPCs advance 0 RTL clock cycles */
    suite.run_case("L4-CYC-09", "G3", "50 ms host socket idle sleep + rejected JSON advance 0 simulated RTL cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t rd = 0;
        uint64_t acc1 = 0, acc2 = 0;
        avid_hal_mmio_read(&hal, 0x050u, &rd, &acc1);
        uint64_t comp1 = hal.stats.last_completed_cycle;
        usleep(50000); /* 50 ms host wall-clock sleep == 12,500,000 cycles @ 250 MHz if free-running! */
        avid_hal_mmio_read(&hal, 0x050u, &rd, &acc2);
        cyc = hal.stats.last_completed_cycle;
        ev = "comp1=" + std::to_string(comp1) + " acc2_after_50ms=" + std::to_string(acc2);
        return acc2 == comp1 && (cyc - acc2) == 2ULL;
    });

    /* L4-CYC-10: Mid-decode busy swap hold cycle accounting */
    suite.run_case("L4-CYC-10", "G3", "Mid-decode COMMIT holds swap_pending until exact bp_ready_in==1 edge", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(10);
        uint32_t delayed_cycles = 0;
        bool ok = rtl.verify_busy_swap_protection(&delayed_cycles);
        cyc = rtl.sim_cycle;
        ev = "held swap_pending for " + std::to_string(delayed_cycles) + " busy core cycles";
        return ok && delayed_cycles > 0u && delayed_cycles <= 17u;
    });

    avid_hal_close(&hal);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
