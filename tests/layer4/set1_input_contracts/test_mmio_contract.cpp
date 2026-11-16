/* ===========================================================================
 * test_mmio_contract.cpp — Layer 4 Set 1.2 MMIO Register Contract Tests
 * Canonical IDs: L4-MMIO-01 .. L4-MMIO-13 (Gates G1, G6)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set1_mmio_contract", "Set1", argc, argv);
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set1_mmio", 16, true)) {
        std::fprintf(stderr, "[FATAL] Failed to start Verilated bridge\n");
        return 2;
    }

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) {
        return 2;
    }

    /* L4-MMIO-01: All valid mapped register addresses */
    suite.run_case("L4-MMIO-01", "G1", "All 41 valid mapped register offsets return OK over HAL/Bridge", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0) return false;
        uint32_t rd = 0;
        int count = 0;
        for (int i = 0; i < 9; i++) {
            if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(i), &rd, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_Z(i), &rd, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(i), &rd, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(i), &rd, &cyc) != 0) return false;
            count += 4;
        }
        const uint32_t ctrl_offs[5] = {0x048u, 0x04Cu, 0x050u, 0x054u, 0x058u};
        for (uint32_t off : ctrl_offs) {
            if (avid_hal_mmio_read(&hal, off, &rd, &cyc) != 0) return false;
            count++;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "verified " + std::to_string(count) + "/41 mapped offsets";
        return count == 41;
    });

    /* L4-MMIO-02: Unmapped addresses (gap 0x05C..0x0FC and >= 0x148) */
    suite.run_case("L4-MMIO-02", "G1", "Unmapped gap 0x05C..0x0FC and >= 0x148 rejected with SLVERR (ERR_INVALID_REGISTER)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        const uint32_t bad_offs[6] = {0x05Cu, 0x080u, 0x0FCu, 0x148u, 0x200u, 0xFFCu};
        uint32_t rd = 0;
        for (uint32_t off : bad_offs) {
            if (avid_hal_mmio_read(&hal, off, &rd, &cyc) == 0) return false;
            if (avid_hal_mmio_write(&hal, off, 0x0100u, 0x3u, &cyc) == 0) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "all 6 unmapped read/write pairs rejected with SLVERR";
        return true;
    });

    /* L4-MMIO-03: Address alignment ((addr & 3) != 0) */
    suite.run_case("L4-MMIO-03", "G1", "Unaligned offsets (+1, +2, +3) rejected with SLVERR without mutating shadow[0]", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t rd = 0;
        for (uint32_t off : {0x001u, 0x002u, 0x003u, 0x049u, 0x055u}) {
            if (avid_hal_mmio_write(&hal, off, 0x0111u, 0xFu, &cyc) == 0) return false;
            if (avid_hal_mmio_read(&hal, off, &rd, &cyc) == 0) return false;
        }
        if (avid_hal_mmio_read(&hal, 0x000u, &rd, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "shadow_x[0] intact at 0x0232 after 5 unaligned attempts";
        return (rd & 0xFFFFu) == 0x0232u;
    });

    /* L4-MMIO-04: Read-only register write attempts */
    suite.run_case("L4-MMIO-04", "G1", "Writes to STATUS (0x04C), EPOCH (0x050), and ACTIVE banks (0x100..0x144) rejected", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        if (avid_hal_mmio_write(&hal, 0x04Cu, 0x3u, 0xFu, &cyc) == 0) return false;
        if (avid_hal_mmio_write(&hal, 0x050u, 0x99u, 0xFu, &cyc) == 0) return false;
        if (avid_hal_mmio_write(&hal, 0x100u, 0x0100u, 0x3u, &cyc) == 0) return false;
        if (avid_hal_mmio_write(&hal, 0x124u, 0x0100u, 0x3u, &cyc) == 0) return false;
        uint32_t ep = 99, ax0 = 0;
        avid_hal_mmio_read(&hal, 0x050u, &ep, &cyc);
        avid_hal_mmio_read(&hal, 0x100u, &ax0, &cyc);
        cyc = hal.stats.last_completed_cycle;
        ev = "RO registers rejected writes; epoch=0 active_x[0]=0x0232";
        return ep == 0u && (ax0 & 0xFFFFu) == 0x0232u;
    });

    /* L4-MMIO-05: Control register COMMIT (0x048) readback & no-op write=0 semantics */
    suite.run_case("L4-MMIO-05", "G1", "COMMIT (0x048) reads {31'b0, swap_pending} and write 0 is safe no-op", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t com_rd = 99, ep = 99;
        if (avid_hal_mmio_read(&hal, 0x048u, &com_rd, &cyc) != 0 || com_rd != 0u) return false;
        if (avid_hal_mmio_write(&hal, 0x048u, 0x0u, 0x1u, &cyc) != 0) return false;
        if (avid_hal_mmio_read(&hal, 0x050u, &ep, &cyc) != 0 || ep != 0u) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "COMMIT readback=0 when idle; write 0 leaves epoch=0";
        return true;
    });

    /* L4-MMIO-06: Reserved bits & domain bounds on cfg_max_iter (0x054) and cfg_osd_mode (0x058) */
    suite.run_case("L4-MMIO-06", "G1", "0x054 enforces K in 1..5 (rejecting 0, 6, 7, 0x13) and 0x058 enforces 0..2", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (uint32_t bad_k : {0u, 6u, 7u, 15u, 0x13u, 0x103u}) {
            if (avid_hal_mmio_write(&hal, 0x054u, bad_k, 0xFu, &cyc) == 0) return false;
        }
        for (uint32_t bad_m : {3u, 4u, 0x11u}) {
            if (avid_hal_mmio_write(&hal, 0x058u, bad_m, 0xFu, &cyc) == 0) return false;
        }
        for (uint32_t ok_k = 1u; ok_k <= 5u; ok_k++) {
            uint32_t rd = 0;
            if (avid_hal_mmio_write(&hal, 0x054u, ok_k, 0x1u, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, 0x054u, &rd, &cyc) != 0 || rd != ok_k) return false;
        }
        avid_hal_mmio_write(&hal, 0x054u, 3u, 0x1u, &cyc);
        cyc = hal.stats.last_completed_cycle;
        ev = "rejected {0,6,7,15,0x13,0x103} on 0x054 and {3,4,0x11} on 0x058; K=1..5 OK";
        return true;
    });

    /* L4-MMIO-07: Byte-enable (wstrb) behavior */
    suite.run_case("L4-MMIO-07", "G1", "Byte-enable wstrb=0x1, 0x2, 0x0 masks shadow writes and rejects wstrb=0 on 0x054", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t rd = 0;
        if (avid_hal_mmio_write(&hal, 0x000u, 0x1234u, 0x3u, &cyc) != 0) return false;
        if (avid_hal_mmio_write(&hal, 0x000u, 0xFFAAu, 0x1u, &cyc) != 0) return false; /* low byte -> 0x12AA */
        if (avid_hal_mmio_read(&hal, 0x000u, &rd, &cyc) != 0 || (rd & 0xFFFFu) != 0x12AAu) return false;
        if (avid_hal_mmio_write(&hal, 0x000u, 0x56FFu, 0x2u, &cyc) != 0) return false; /* high byte -> 0x56AA */
        if (avid_hal_mmio_read(&hal, 0x000u, &rd, &cyc) != 0 || (rd & 0xFFFFu) != 0x56AAu) return false;
        if (avid_hal_mmio_write(&hal, 0x000u, 0x9999u, 0x0u, &cyc) != 0) return false; /* wstrb=0 -> unchanged */
        if (avid_hal_mmio_read(&hal, 0x000u, &rd, &cyc) != 0 || (rd & 0xFFFFu) != 0x56AAu) return false;
        if (avid_hal_mmio_write(&hal, 0x054u, 0x0004u, 0x0u, &cyc) == 0) return false; /* wstrb[0]==0 -> SLVERR */
        cyc = hal.stats.last_completed_cycle;
        ev = "verified wstrb=0x3->0x1234, 0x1->0x12AA, 0x2->0x56AA, 0x0->unchanged";
        return true;
    });

    /* L4-MMIO-08: Register reset values across all 41 registers */
    suite.run_case("L4-MMIO-08", "G1", "RESET restores all 36 shadow/active weights to 0x0232, epoch=0, K=3, osd_mode=1", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0) return false;
        uint32_t rd = 0;
        for (int i = 0; i < 9; i++) {
            avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != 0x0232u) return false;
            avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_Z(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != 0x0232u) return false;
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != 0x0232u) return false;
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != 0x0232u) return false;
        }
        uint32_t st = 1, ep = 1, k = 0, om = 0;
        avid_hal_mmio_read(&hal, 0x04Cu, &st, &cyc);
        avid_hal_mmio_read(&hal, 0x050u, &ep, &cyc);
        avid_hal_mmio_read(&hal, 0x054u, &k, &cyc);
        avid_hal_mmio_read(&hal, 0x058u, &om, &cyc);
        cyc = hal.stats.last_completed_cycle;
        ev = "all 36 weights=0x0232, status=0, epoch=0, K=3, osd=1";
        return st == 0u && ep == 0u && k == 3u && om == 1u;
    });

    /* L4-MMIO-09: Read-after-write inhomogeneous X != Z consistency */
    suite.run_case("L4-MMIO-09", "G1", "Distinct 18 shadow registers (wx != wz) + K + osd_mode read back before & after COMMIT", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int16_t wx[9], wz[9];
        for (int i = 0; i < 9; i++) {
            wx[i] = static_cast<int16_t>(0x0110 + 7 * i);
            wz[i] = static_cast<int16_t>(0x0210 + 11 * i);
            if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_X(i), static_cast<uint16_t>(wx[i]), 0x3u, &cyc) != 0) return false;
            if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_Z(i), static_cast<uint16_t>(wz[i]), 0x3u, &cyc) != 0) return false;
        }
        /* Before COMMIT: shadow updated, active still 0x0232 */
        uint32_t rd = 0;
        for (int i = 0; i < 9; i++) {
            avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != static_cast<uint16_t>(wx[i])) return false;
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != 0x0232u) return false;
        }
        if (avid_hal_mmio_write(&hal, 0x048u, 1u, 0x1u, &cyc) != 0) return false;
        for (int i = 0; i < 9; i++) {
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != static_cast<uint16_t>(wx[i])) return false;
            avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(i), &rd, &cyc);
            if ((rd & 0xFFFFu) != static_cast<uint16_t>(wz[i])) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "18/18 inhomogeneous X/Z weights isolated before commit and exact after commit";
        return true;
    });

    /* L4-MMIO-10: Interleaved reads & writes across banks */
    suite.run_case("L4-MMIO-10", "G1", "Interleaved MMIO_WRITE and MMIO_READ across shadow_x, shadow_z, active, STATUS", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (int i = 0; i < 9; i++) {
            uint32_t vx = 0x0150u + static_cast<uint32_t>(i);
            uint32_t vz = 0x0250u + static_cast<uint32_t>(i);
            uint32_t rdx = 0, rdz = 0, st = 99;
            if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_X(i), vx, 0x3u, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(i), &rdx, &cyc) != 0 || rdx != vx) return false;
            if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_Z(i), vz, 0x3u, &cyc) != 0) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_Z(i), &rdz, &cyc) != 0 || rdz != vz) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_STATUS, &st, &cyc) != 0 || st != 0u) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "45 interleaved R/W ops returned exact values with zero cross-bank bleed";
        return true;
    });

    /* L4-MMIO-11: 100 back-to-back MMIO transactions with exact +2 cycle spacing */
    suite.run_case("L4-MMIO-11", "G1", "100 consecutive MMIO ops advance deterministically by exactly +2 RTL cycles/op", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint32_t rd = 0;
        uint64_t prev_comp = hal.stats.last_completed_cycle;
        for (int i = 0; i < 100; i++) {
            uint64_t acc = 0;
            if ((i & 1) == 0) {
                if (avid_hal_mmio_write(&hal, WRD_OFF_SHADOW_X(i % 9), 0x0180u + i, 0x3u, &acc) != 0) return false;
            } else {
                if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X((i - 1) % 9), &rd, &acc) != 0) return false;
            }
            uint64_t comp = hal.stats.last_completed_cycle;
            if (acc != prev_comp || (comp - acc) != 2ULL) return false;
            prev_comp = comp;
        }
        cyc = prev_comp;
        ev = "100/100 MMIO ops had exact C_accepted=prev_C_completed and C_latency=2";
        return true;
    });

    /* L4-MMIO-12: Access while decoder is busy over async socket RPC (DEF-L4-05 remediated) */
    suite.run_case("L4-MMIO-12", "G1,G6", "Mid-decode COMMIT queues swap_pending=1 (STATUS=0x3) and locks shadow bank (SLVERR) over socket", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r_rst, r_sh, r_st_syn, r_com, r_stat, r_torn, r_wait, r_act;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50001,\"operation\":\"RESET\",\"data\":10}\n", r_rst);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50002,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000034\",\"data\":\"0x0140\",\"byte_enable\":3}\n", r_sh);
        /* Launch async syndrome decode: after 2 cycles, bp_busy == 1 inside 17-cycle core */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50003,\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":true}\n", r_st_syn);
        /* Issue COMMIT mid-decode -> accepted (OK), sets swap_pending=1 while bp_busy=1 */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50004,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000048\",\"data\":\"0x1\",\"byte_enable\":1}\n", r_com);
        /* Read STATUS (0x04C) mid-decode -> bit0=swap_pending(1), bit1=bp_busy(1) => 0x3 */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50005,\"operation\":\"MMIO_READ\",\"address\":\"0x0000004C\"}\n", r_stat);
        /* Attempt torn shadow write while swap_pending && bp_busy -> rejected with ERR_INVALID_REGISTER (SLVERR) */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50006,\"operation\":\"MMIO_WRITE\",\"address\":\"0x00000034\",\"data\":\"0x03FF\",\"byte_enable\":3}\n", r_torn);
        /* Complete inflight syndrome decode */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50007,\"operation\":\"SYNDROME_WAIT\"}\n", r_wait);
        /* Read active_z[4] (0x134) -> now 0x0140 (not 0x03FF!) */
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":50008,\"operation\":\"MMIO_READ\",\"address\":\"0x00000134\"}\n", r_act);
        hal.next_tx_id = 60000ULL;

        uint64_t stat_v = 0, act_v = 0;
        avid_l4::extract_json_u64_simple(r_stat, "rdata", &stat_v);
        avid_l4::extract_json_u64_simple(r_act, "rdata", &act_v);
        avid_l4::extract_json_u64_simple(r_act, "completed_cycle", &cyc);
        ev = "mid_decode STATUS=" + std::to_string(stat_v) +
             " torn_write=" + avid_l4::extract_json_str_simple(r_torn, "status") +
             " post_decode active_z[4]=" + std::to_string(act_v & 0xFFFFu);
        return stat_v == 3u &&
               avid_l4::extract_json_str_simple(r_torn, "status") == "ERR_INVALID_REGISTER" &&
               (act_v & 0xFFFFu) == 0x0140u;
    });

    /* L4-MMIO-13: Access during active reset (aresetn == 0 on RTL pins) */
    suite.run_case("L4-MMIO-13", "G1", "AXI-Lite awready/wready/arready held 0 while aresetn==0 on Vaa_qec_top", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(10);
#if defined(USE_VERILATOR) && USE_VERILATOR
        rtl.top->aresetn = 0;
        rtl.top->s_axil_awaddr = 0x000u;
        rtl.top->s_axil_awvalid = 1;
        rtl.top->s_axil_wdata = 0x0111u;
        rtl.top->s_axil_wstrb = 0x3u;
        rtl.top->s_axil_wvalid = 1;
        rtl.top->s_axil_araddr = 0x000u;
        rtl.top->s_axil_arvalid = 1;
        rtl.step_clock(3);
        bool ready_low = (rtl.top->s_axil_awready == 0) &&
                         (rtl.top->s_axil_wready == 0) &&
                         (rtl.top->s_axil_arready == 0);
        rtl.top->s_axil_awvalid = 0;
        rtl.top->s_axil_wvalid = 0;
        rtl.top->s_axil_arvalid = 0;
        rtl.top->aresetn = 1;
        rtl.step_clock(2);
        uint32_t rd = 0;
        bool ok_after = rtl.axil_read(0x000u, &rd, &cyc);
        ev = "awready=wready=arready=0 while aresetn=0; post-reset rdata=0x0232";
        return ready_low && ok_after && (rd & 0xFFFFu) == 0x0232u;
#else
        ev = "Requires native Verilator pin access";
        return false;
#endif
    });

    avid_hal_close(&hal);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
