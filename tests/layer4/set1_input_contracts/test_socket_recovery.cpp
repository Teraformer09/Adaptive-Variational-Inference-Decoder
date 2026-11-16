/* ===========================================================================
 * test_socket_recovery.cpp — Layer 4 Set 1.4 Socket Reconnect State Recovery
 * Canonical ID: L4-SEQ-08 (Gates G1, G7)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set1_socket_recovery", "Set1", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    avid_l4::BridgeProcessGuard bridge;
    if (!bridge.start(suite.bridge_bin_path(), "set1_rec", 32, true)) return 2;

    suite.run_case("L4-SEQ-08", "G1,G7", "10 sequential socket disconnects & avid_hal_reconnects preserve RTL state & epoch_count", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_hal_t hal;
        if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, bridge.sock_path.c_str(), WRD_BASE_ADDR) != 0) return false;
        if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0) return false;
        if (avid_hal_set_bp_iterations(&hal, 4u, &cyc) != 0) return false;
        if (avid_hal_set_osd_mode(&hal, 2u, &cyc) != 0) return false;

        for (uint32_t s = 1; s <= 10; s++) {
            int16_t wx[9], wz[9];
            for (int q = 0; q < 9; q++) {
                wx[q] = static_cast<int16_t>(0x0130 + s * 4 + q);
                wz[q] = static_cast<int16_t>(0x0210 + s * 4 + q);
            }
            uint32_t ep = 0;
            if (avid_hal_write_weights_and_commit(&hal, wx, wz, &ep, &cyc) != 0 || ep != s) {
                avid_hal_close(&hal);
                return false;
            }
            /* Disconnect and reconnect */
            if (avid_hal_reconnect(&hal) != 0) {
                avid_hal_close(&hal);
                return false;
            }
            uint32_t ep_rd = 0, k_rd = 0, om_rd = 0, ax0 = 0, az8 = 0;
            if (avid_hal_mmio_read(&hal, WRD_OFF_EPOCH, &ep_rd, &cyc) != 0 || ep_rd != s) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_MAX_ITER, &k_rd, &cyc) != 0 || k_rd != 4u) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_OSD_MODE, &om_rd, &cyc) != 0 || om_rd != 2u) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(0), &ax0, &cyc) != 0 ||
                (ax0 & 0xFFFFu) != static_cast<uint16_t>(wx[0])) return false;
            if (avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(8), &az8, &cyc) != 0 ||
                (az8 & 0xFFFFu) != static_cast<uint16_t>(wz[8])) return false;
        }
        cyc = hal.stats.last_completed_cycle;
        uint64_t reconns = hal.stats.reconnect_events;
        avid_hal_close(&hal);
        ev = "reconnect_events=" + std::to_string(reconns) + " final_epoch=10 K=4 osd=2 intact";
        return reconns == 10ULL;
    });

    bridge.graceful_shutdown(999999ULL);
    return suite.finalize();
}
