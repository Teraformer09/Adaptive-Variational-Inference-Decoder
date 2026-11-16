/* ===========================================================================
 * test_restart_recovery.cpp — Layer 4 Set 4.3 Process Restart & Guest Recovery
 * Canonical IDs: L4-FLT-15 .. L4-FLT-22 (Gate G7)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set4_restart_recovery", "Set4", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    /* L4-FLT-15: Bridge graceful SHUTDOWN + new bridge start + avid_hal_reconnect */
    suite.run_case("L4-FLT-15", "G7", "HAL client detects bridge restart and recovers via avid_hal_reconnect()", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard b1;
        if (!b1.start(suite.bridge_bin_path(), "rec15", 8, true)) return false;
        std::string sp = b1.sock_path;
        avid_hal_t hal;
        if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, sp.c_str(), WRD_BASE_ADDR) != 0) return false;
        uint32_t rd = 0;
        avid_hal_mmio_read(&hal, 0x054u, &rd, &cyc);
        b1.stop(SIGTERM);

        /* While bridge is down, MMIO read fails cleanly */
        int rc_down = avid_hal_mmio_read(&hal, 0x054u, &rd, &cyc);

        /* Spawn replacement bridge on exact same socket path */
        pid_t p2 = fork();
        if (p2 == 0) {
            execl(suite.bridge_bin_path().c_str(), suite.bridge_bin_path().c_str(),
                  "--require-native-rtl", "--socket", sp.c_str(), "--max-clients", "4", nullptr);
            _exit(127);
        }
        for (int i = 0; i < 80; i++) {
            int f = avid_l4::raw_unix_connect(sp, 200);
            if (f >= 0) { close(f); break; }
            usleep(15000);
        }
        int rc_rec = avid_hal_reconnect(&hal);
        int rc_ok = avid_hal_mmio_read(&hal, 0x054u, &rd, &cyc);
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        kill(p2, SIGTERM);
        int ws = 0;
        waitpid(p2, &ws, 0);
        unlink(sp.c_str());
        ev = "rc_down=" + std::to_string(rc_down) + " rc_rec=" + std::to_string(rc_rec) + " rc_ok=" + std::to_string(rc_ok);
        return rc_down != 0 && rc_rec == 0 && rc_ok == 0 && rd == 3u;
    });

    /* L4-FLT-16: Bridge abrupt SIGKILL crash + restart + fresh epoch re-sync from 0 */
    suite.run_case("L4-FLT-16", "G6,G7", "After abrupt bridge SIGKILL & restart, HAL re-syncs fresh RTL epoch=0 cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard b1;
        if (!b1.start(suite.bridge_bin_path(), "rec16", 8, true)) return false;
        std::string sp = b1.sock_path;
        avid_hal_t hal;
        avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, sp.c_str(), WRD_BASE_ADDR);
        int16_t w[9] = {0x0150,0x0150,0x0150,0x0150,0x0150,0x0150,0x0150,0x0150,0x0150};
        uint32_t ep_before = 0;
        avid_hal_write_weights_and_commit(&hal, w, w, &ep_before, &cyc);
        b1.stop(SIGKILL);

        pid_t p2 = fork();
        if (p2 == 0) {
            execl(suite.bridge_bin_path().c_str(), suite.bridge_bin_path().c_str(),
                  "--require-native-rtl", "--socket", sp.c_str(), "--max-clients", "4", nullptr);
            _exit(127);
        }
        for (int i = 0; i < 80; i++) {
            int f = avid_l4::raw_unix_connect(sp, 200);
            if (f >= 0) { close(f); break; }
            usleep(15000);
        }
        avid_hal_reconnect(&hal);
        uint32_t ep_after_crash = 99;
        avid_hal_mmio_read(&hal, 0x050u, &ep_after_crash, &cyc);
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        kill(p2, SIGTERM);
        int ws = 0;
        waitpid(p2, &ws, 0);
        unlink(sp.c_str());
        ev = "ep_before=" + std::to_string(ep_before) + " ep_after_restart=" + std::to_string(ep_after_crash);
        return ep_before == 1u && ep_after_crash == 0u;
    });

    /* L4-FLT-17: HAL open failure on missing socket returns -1 cleanly without leak */
    suite.run_case("L4-FLT-17", "G7", "avid_hal_open on non-existent socket fails cleanly with rc=-1 and sock_fd=-1", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        avid_hal_t hal;
        int rc = avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, "/tmp/no_such_avid_sock_9999.sock", WRD_BASE_ADDR);
        ev = "rc=" + std::to_string(rc) + " sock_fd=" + std::to_string(hal.sock_fd);
        return rc < 0 && hal.sock_fd == -1;
    });

    /* L4-FLT-18: QEMU ARM Cortex-A9 guest fails cleanly (rc=1) when bridge socket is absent */
    suite.run_case("L4-FLT-18", "G5,G7", "QEMU ARM Cortex-A9 guest exits cleanly with rc=1 when bridge socket is absent", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string cmd = "AVID_QEMU_ARCH=cortex-a9 bash " + suite.repo_root +
                          "/emulator/qemu/run.sh --backend sim --socket /tmp/missing_l4.sock >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
        int ec = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
        ev = "cortex-a9 missing-socket exit_code=" + std::to_string(ec);
        return ec == 1;
    });

    /* L4-FLT-19: QEMU ARM64 Cortex-A53 guest fails cleanly (rc=1) when bridge socket is absent */
    suite.run_case("L4-FLT-19", "G5,G7", "QEMU ARM64 Cortex-A53 guest exits cleanly with rc=1 when bridge socket is absent", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string cmd = "AVID_QEMU_ARCH=aarch64 bash " + suite.repo_root +
                          "/emulator/qemu/run.sh --backend sim --socket /tmp/missing_l4.sock >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
        int ec = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
        ev = "aarch64 missing-socket exit_code=" + std::to_string(ec);
        return ec == 1;
    });

    /* L4-FLT-20: QEMU ARM guest rejects invalid CLI arguments with rc=2 */
    suite.run_case("L4-FLT-20", "G5,G7", "QEMU ARM guest rejects out-of-range --k-iters 99 and --osd-mode 9 with rc=2", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string c1 = "AVID_QEMU_ARCH=cortex-a9 bash " + suite.repo_root +
                         "/emulator/qemu/run.sh --backend physical --k-iters 99 >/dev/null 2>&1";
        std::string c2 = "AVID_QEMU_ARCH=aarch64 bash " + suite.repo_root +
                         "/emulator/qemu/run.sh --backend physical --osd-mode 9 >/dev/null 2>&1";
        int r1 = std::system(c1.c_str());
        int r2 = std::system(c2.c_str());
        int e1 = WIFEXITED(r1) ? WEXITSTATUS(r1) : -1;
        int e2 = WIFEXITED(r2) ? WEXITSTATUS(r2) : -1;
        ev = "bad_k_rc=" + std::to_string(e1) + " bad_osd_rc=" + std::to_string(e2);
        return e1 == 2 && e2 == 2;
    });

    /* L4-FLT-21: QEMU guest abrupt kill mid-run leaves bridge alive to serve next QEMU guest */
    suite.run_case("L4-FLT-21", "G5,G7", "Killing QEMU guest mid-session leaves Verilator bridge healthy for next QEMU guest", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        if (!br.start(suite.bridge_bin_path(), "rec21", 8, true)) return false;
        pid_t gpid = fork();
        if (gpid == 0) {
            setenv("AVID_QEMU_ARCH", "cortex-a9", 1);
            std::string run_sh = suite.repo_root + "/emulator/qemu/run.sh";
            execlp("bash", "bash", run_sh.c_str(), "--backend", "sim", "--socket", br.sock_path.c_str(),
                   "--rounds", "5000", "--window", "64", nullptr);
            _exit(127);
        }
        usleep(40000); /* Let guest execute some transactions, then SIGKILL it */
        kill(gpid, SIGKILL);
        int ws = 0;
        waitpid(gpid, &ws, 0);

        /* Now run a complete ARM64 guest session on the same surviving bridge! */
        std::string gm_path = avid_l4::make_temp_path("rec21_gm", ".json");
        std::string cmd = "AVID_QEMU_ARCH=aarch64 bash " + suite.repo_root +
                          "/emulator/qemu/run.sh --backend sim --socket " + br.sock_path +
                          " --rounds 16 --window 16 --metrics-out " + gm_path + " >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
        int ec = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
        std::ifstream mf(gm_path);
        std::string mjson((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
        unlink(gm_path.c_str());
        avid_l4::extract_json_u64_simple(mjson, "final_completed_cycle", &cyc);
        br.graceful_shutdown(9999999ULL);
        ev = "follow-up aarch64 guest rc=" + std::to_string(ec) + " cyc=" + std::to_string(cyc);
        return ec == 0 && cyc > 0ULL;
    });

    /* L4-FLT-22: Repeated 3x full bridge lifecycle restart + guest execution */
    suite.run_case("L4-FLT-22", "G7", "3 consecutive Bridge start -> QEMU guest run -> Bridge SHUTDOWN lifecycles succeed identically", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        uint64_t ref_cyc = 0;
        for (int i = 0; i < 3; i++) {
            avid_l4::BridgeProcessGuard br;
            if (!br.start(suite.bridge_bin_path(), "rec22_" + std::to_string(i), 4, true)) return false;
            std::string gm = avid_l4::make_temp_path("rec22_gm", ".json");
            std::string cmd = "AVID_QEMU_ARCH=cortex-a9 bash " + suite.repo_root +
                              "/emulator/qemu/run.sh --backend sim --socket " + br.sock_path +
                              " --rounds 16 --window 16 --metrics-out " + gm + " >/dev/null 2>&1";
            int rc = std::system(cmd.c_str());
            if (!WIFEXITED(rc) || WEXITSTATUS(rc) != 0) return false;
            std::ifstream mf(gm);
            std::string s((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
            unlink(gm.c_str());
            uint64_t c = 0;
            avid_l4::extract_json_u64_simple(s, "final_completed_cycle", &c);
            if (i == 0) ref_cyc = c;
            if (c != ref_cyc) return false;
            cyc = c;
            br.graceful_shutdown(999999ULL);
        }
        ev = "3/3 full lifecycles finished at exact cycle " + std::to_string(cyc);
        return cyc > 0ULL;
    });

    return suite.finalize();
}
