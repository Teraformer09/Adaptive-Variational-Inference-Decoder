/* ===========================================================================
 * test_syndrome_transfer.cpp — Layer 4 Set 1.3 Syndrome-Transfer Contract Tests
 * Canonical IDs: L4-SYND-01 .. L4-SYND-14 (Gates G1, G4)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

static uint8_t synd_x_of(uint16_t cz) {
    uint8_t b0 = ((cz >> 0) ^ (cz >> 1)) & 1u;
    uint8_t b1 = ((cz >> 1) ^ (cz >> 2) ^ (cz >> 4) ^ (cz >> 5)) & 1u;
    uint8_t b2 = ((cz >> 3) ^ (cz >> 4) ^ (cz >> 6) ^ (cz >> 7)) & 1u;
    uint8_t b3 = ((cz >> 7) ^ (cz >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

static uint8_t synd_z_of(uint16_t cx) {
    uint8_t b0 = ((cx >> 0) ^ (cx >> 1) ^ (cx >> 3) ^ (cx >> 4)) & 1u;
    uint8_t b1 = ((cx >> 2) ^ (cx >> 5)) & 1u;
    uint8_t b2 = ((cx >> 3) ^ (cx >> 6)) & 1u;
    uint8_t b3 = ((cx >> 4) ^ (cx >> 5) ^ (cx >> 7) ^ (cx >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set1_syndrome_transfer", "Set1", argc, argv);
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set1_synd", 16, true)) return 2;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return 2;

    uint64_t init_c = 0;
    avid_hal_reset_rtl(&hal, 10u, &init_c);

    /* L4-SYND-01: Payload width enforcement (<= 0xFF valid, > 0xFF ERR_INVALID_SYNDROME) */
    suite.run_case("L4-SYND-01", "G1", "Syndrome payload width enforces 8-bit packed (sx,sz) <= 0xFF", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r_ok, r_b1, r_b2;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":70001,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x00FF\",\"tlast\":false}\n", r_ok);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":70002,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0100\",\"tlast\":false}\n", r_b1);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":70003,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0xFFFF\",\"tlast\":false}\n", r_b2);
        hal.next_tx_id = 71000ULL;
        avid_l4::extract_json_u64_simple(r_b2, "completed_cycle", &cyc);
        ev = "0xFF->OK, 0x100->ERR_INVALID_SYNDROME, 0xFFFF->ERR_INVALID_SYNDROME";
        return avid_l4::extract_json_str_simple(r_ok, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r_b1, "status") == "ERR_INVALID_SYNDROME" &&
               avid_l4::extract_json_str_simple(r_b2, "status") == "ERR_INVALID_SYNDROME";
    });

    /* L4-SYND-02: Bit ordering within each 4-bit stabilizer nibble */
    suite.run_case("L4-SYND-02", "G1,G4", "Single-bit X and Z stabilizer triggers match exact check-node bit ordering", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        for (int b = 0; b < 4; b++) {
            uint8_t s = static_cast<uint8_t>(1u << b);
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            if (avid_hal_send_syndrome(&hal, s, 0u, false, &cx, &cz, &bp, &cyc) != 0) return false;
            if (synd_x_of(cz) != s || cx != 0u) return false;
            if (avid_hal_send_syndrome(&hal, 0u, s, false, &cx, &cz, &bp, &cyc) != 0) return false;
            if (synd_z_of(cx) != s || cz != 0u) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "bits 0..3 of sx and sz satisfy H_X*corr_z == sx and H_Z*corr_x == sz";
        return true;
    });

    /* L4-SYND-03: Byte/nibble packing (tdata[3:0]=sx, tdata[7:4]=sz) across all 256 combinations */
    suite.run_case("L4-SYND-03", "G1,G4", "Exhaustive 256 (sx, sz) packed beats satisfy CSS syndromes", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                uint16_t cx = 0, cz = 0;
                uint32_t bp = 0;
                if (avid_hal_send_syndrome(&hal, static_cast<uint8_t>(sx), static_cast<uint8_t>(sz),
                                           false, &cx, &cz, &bp, &cyc) != 0) {
                    return false;
                }
                if (synd_x_of(cz) == sx && synd_z_of(cx) == sz) sat++;
            }
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "satisfied " + std::to_string(sat) + "/256 packed (sx,sz) combinations";
        return sat == 256;
    });

    /* L4-SYND-04: Strict X/Z channel separation */
    suite.run_case("L4-SYND-04", "G1,G4", "Varying sz never alters corr_z and varying sx never alters corr_x", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t ref_cz = 0, ref_cx = 0, cx = 0, cz = 0;
        uint32_t bp = 0;
        avid_hal_send_syndrome(&hal, 0x6u, 0x0u, false, &cx, &ref_cz, &bp, &cyc);
        avid_hal_send_syndrome(&hal, 0x0u, 0x9u, false, &ref_cx, &cz, &bp, &cyc);
        for (uint8_t other = 0; other < 16; other++) {
            avid_hal_send_syndrome(&hal, 0x6u, other, false, &cx, &cz, &bp, &cyc);
            if (cz != ref_cz) return false;
            avid_hal_send_syndrome(&hal, other, 0x9u, false, &cx, &cz, &bp, &cyc);
            if (cx != ref_cx) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        ev = "zero cross-talk across 16 opponent-basis sweeps";
        return true;
    });

    /* L4-SYND-05: Single-beat transfer length invariant */
    suite.run_case("L4-SYND-05", "G1", "Each SYNDROME_TRANSFER issues exactly 1 AXI-Stream beat (22 RTL cycles at K=3)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        uint64_t acc = 0;
        if (avid_hal_send_syndrome(&hal, 0x3u, 0x5u, true, &cx, &cz, &bp, &acc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "latency=" + std::to_string(cyc - acc) + " cycles (bp=" + std::to_string(bp) + ")";
        return bp == 17u && (cyc - acc) == 22ULL;
    });

    /* L4-SYND-06: Architectural fixed-width 16-bit beat contract (no sparse TKEEP port) */
    suite.run_case("L4-SYND-06", "G1", "Verify fixed 16-bit atomic beat contract (no partial-byte TKEEP port in aa_qec_top)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        /* aa_qec_top.sv defines fixed 16-bit s_axis_syndrome_tdata where [7:0] is atomic;
         * verify axis_ingress_tap & axis_syndrome_rx transfer full 16-bit beat on tvalid && tready. */
        AvidVerilatorRtlModel rtl(6);
        RtlDecodeTrace tr = rtl.decode_syndrome(0xAu, 0x5u, true);
        cyc = rtl.sim_cycle;
        ev = "atomic 16b beat verified (TKEEP intentionally absent per fixed-beat RTL spec)";
        return tr.total_axis_cycles == 22u && synd_x_of(tr.corr_z) == 0xAu && synd_z_of(tr.corr_x) == 0x5u;
    });

    /* L4-SYND-07: tlast=0 vs tlast=1 selective epoch_irq generation */
    suite.run_case("L4-SYND-07", "G1", "tlast=false yields epoch_irq=0 and tlast=true yields epoch_irq=1", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        uint64_t irq_before = hal.stats.irq_events;
        if (avid_hal_send_syndrome(&hal, 0x1u, 0x0u, false, &cx, &cz, &bp, &cyc) != 0) return false;
        if (hal.stats.last_epoch_irq != 0u || hal.stats.irq_events != irq_before) return false;
        if (avid_hal_send_syndrome(&hal, 0x1u, 0x0u, true, &cx, &cz, &bp, &cyc) != 0) return false;
        if (hal.stats.last_epoch_irq != 1u || hal.stats.irq_events != irq_before + 1ULL) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "tlast=0 -> irq=0; tlast=1 -> irq=1 propagated to HAL stats";
        return true;
    });

    /* L4-SYND-08: AXI-Stream backpressure (16-deep FIFO full -> s_axis_syndrome_tready == 0) */
    suite.run_case("L4-SYND-08", "G1", "Stalling m_axis_correction_tready=0 fills RX+BP+TX pipeline until s_axis_tready deasserts", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        AvidVerilatorRtlModel rtl(8);
#if defined(USE_VERILATOR) && USE_VERILATOR
        rtl.top->m_axis_correction_tready = 0;
        int accepted = 0;
        bool saw_tready_low = false;
        for (int g = 0; g < 600; g++) {
            rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>((accepted & 0xF) | 0x10u);
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 0;
            bool rdy_pre = (rtl.top->s_axis_syndrome_tready != 0);
            if (!rdy_pre) {
                saw_tready_low = true;
                rtl.top->s_axis_syndrome_tvalid = 0;
                break;
            }
            rtl.step_clock(1);
            accepted++;
        }
        /* Now drain all accepted beats by asserting m_axis_correction_tready = 1 */
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->m_axis_correction_tready = 1;
        int drained = 0;
        for (int g = 0; g < 1200 && drained < accepted; g++) {
            bool v = (rtl.top->m_axis_correction_tvalid != 0);
            rtl.step_clock(1);
            if (v) drained++;
        }
        cyc = rtl.sim_cycle;
        ev = "accepted=" + std::to_string(accepted) + " before s_axis_tready=0, drained=" + std::to_string(drained);
        return saw_tready_low && accepted >= 16 && drained == accepted;
#else
        return false;
#endif
    });

    /* L4-SYND-09: Partial socket frame syndrome reassembly */
    suite.run_case("L4-SYND-09", "G1", "Byte-by-byte partial socket write of SYNDROME_TRANSFER reassembles cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string req = "{\"protocol_version\":1,\"transaction_id\":75001,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x0006\",\"tlast\":true}\n";
        for (char c : req) {
            if (!avid_l4::raw_write_all(rfd, &c, 1)) return false;
        }
        std::string resp;
        bool ok = avid_l4::raw_read_line(rfd, resp);
        hal.next_tx_id = 76000ULL;
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        ev = "1-byte chunked SYNDROME_TRANSFER -> status=" + avid_l4::extract_json_str_simple(resp, "status");
        return ok && avid_l4::extract_json_str_simple(resp, "status") == "OK";
    });

    /* L4-SYND-10: Zero syndrome (sx=0, sz=0) trivial noiseless transfer */
    suite.run_case("L4-SYND-10", "G1,G4", "Zero syndrome (0x00) produces corr_x=0, corr_z=0 in exact 17/22 cycles", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t cx = 99, cz = 99;
        uint32_t bp = 0;
        if (avid_hal_send_syndrome(&hal, 0u, 0u, false, &cx, &cz, &bp, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "corr_x=0 corr_z=0 bp=17";
        return cx == 0u && cz == 0u && bp == 17u;
    });

    /* L4-SYND-11: Maximum all-ones syndrome (sx=0xF, sz=0xF -> 0xFF) */
    suite.run_case("L4-SYND-11", "G1,G4", "Maximum weight-4 syndrome (sx=0xF, sz=0xF) satisfies both parity matrices", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        if (avid_hal_send_syndrome(&hal, 0xFu, 0xFu, true, &cx, &cz, &bp, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "sx(cz)=" + std::to_string(synd_x_of(cz)) + " sz(cx)=" + std::to_string(synd_z_of(cx));
        return synd_x_of(cz) == 0xFu && synd_z_of(cx) == 0xFu && bp == 17u;
    });

    /* L4-SYND-12: Repeated transfers determinism (32 identical transfers) */
    suite.run_case("L4-SYND-12", "G1,G4", "32 repeated SYNDROME_TRANSFER beats return identical corrections and +22 cycle deltas", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint16_t c0x = 0, c0z = 0;
        uint32_t bp0 = 0;
        uint64_t prev_comp = hal.stats.last_completed_cycle;
        for (int i = 0; i < 32; i++) {
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            uint64_t acc = 0;
            if (avid_hal_send_syndrome(&hal, 0xBu, 0xDu, (i == 31), &cx, &cz, &bp, &acc) != 0) return false;
            if (i == 0) { c0x = cx; c0z = cz; bp0 = bp; }
            if (cx != c0x || cz != c0z || bp != bp0) return false;
            uint64_t comp = hal.stats.last_completed_cycle;
            if (acc != prev_comp || (comp - acc) != 22ULL) return false;
            prev_comp = comp;
        }
        cyc = prev_comp;
        ev = "32/32 beats bit-exact with exact +22 cycle progression";
        return true;
    });

    /* L4-SYND-13: Busy rejection when async SYNDROME_START already inflight */
    suite.run_case("L4-SYND-13", "G1", "Second SYNDROME_START while inflight rejected cleanly with ERR_BUSY", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        int rfd = hal.sock_fd;
        if (rfd < 0) return false;
        std::string r1, r2, r3;
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":77001,\"operation\":\"SYNDROME_START\",\"data\":\"0x0006\",\"tlast\":false}\n", r1);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":77002,\"operation\":\"SYNDROME_START\",\"data\":\"0x0003\",\"tlast\":false}\n", r2);
        avid_l4::raw_rpc(rfd, "{\"protocol_version\":1,\"transaction_id\":77003,\"operation\":\"SYNDROME_WAIT\"}\n", r3);
        hal.next_tx_id = 78000ULL;
        avid_l4::extract_json_u64_simple(r3, "completed_cycle", &cyc);
        ev = "second SYNDROME_START -> " + avid_l4::extract_json_str_simple(r2, "status");
        return avid_l4::extract_json_str_simple(r1, "status") == "OK" &&
               avid_l4::extract_json_str_simple(r2, "status") == "ERR_BUSY" &&
               avid_l4::extract_json_str_simple(r3, "status") == "OK";
    });

    /* L4-SYND-14: Post-reset syndrome transfer */
    suite.run_case("L4-SYND-14", "G1,G4", "Immediate SYNDROME_TRANSFER after RESET decodes cleanly with default weights", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        if (avid_hal_reset_rtl(&hal, 12u, &cyc) != 0) return false;
        uint16_t cx = 0, cz = 0;
        uint32_t bp = 0;
        if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, true, &cx, &cz, &bp, &cyc) != 0) return false;
        cyc = hal.stats.last_completed_cycle;
        ev = "post-reset decode satisfied X/Z syndromes in 17 core cycles";
        return bp == 17u && synd_x_of(cz) == 0x6u && synd_z_of(cx) == 0x9u;
    });

    avid_hal_close(&hal);
    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
