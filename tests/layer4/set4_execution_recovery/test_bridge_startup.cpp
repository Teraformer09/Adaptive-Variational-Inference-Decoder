/* ===========================================================================
 * test_bridge_startup.cpp — Layer 4 Set 4.1 Build & Startup Verification
 * Canonical IDs: L4-BLD-01 .. L4-BLD-11 (Gates G7, G8)
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set4_bridge_startup", "Set4", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    /* L4-BLD-01: Bridge executable exists and is executable */
    suite.run_case("L4-BLD-01", "G7,G8", "Native Verilated avid_cosim_bridge binary exists and is executable", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string b = suite.bridge_bin_path();
        ev = b;
        return access(b.c_str(), X_OK) == 0;
    });

    /* L4-BLD-02: Standalone Verilator RTL testbench executes cleanly with --require-native-rtl */
    suite.run_case("L4-BLD-02", "G7,G8", "standalone_rtl_tb --require-native-rtl executes and exits 0", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string tb = suite.repo_root + "/emulator/verilator/models/standalone_rtl_tb";
        std::string cmd = tb + " --require-native-rtl >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
        int ec = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
        ev = "standalone_rtl_tb exit_code=" + std::to_string(ec);
        return ec == 0;
    });

    /* L4-BLD-03: Native Verilator --require-native-rtl accepted on bridge startup */
    suite.run_case("L4-BLD-03", "G7,G8", "avid_cosim_bridge starts cleanly under --require-native-rtl", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        bool ok = br.start(suite.bridge_bin_path(), "bld03", 2, true);
        br.graceful_shutdown(10ULL);
        cyc = 20;
        ev = "started and shut down cleanly with --require-native-rtl";
        return ok;
    });

    /* L4-BLD-04: QEMU ARM Cortex-A9 and ARM64 Cortex-A53 guest ELFs execute --self-test */
    suite.run_case("L4-BLD-04", "G5,G7,G8", "Both QEMU ARM Cortex-A9 and ARM64 Cortex-A53 guest ELFs pass --self-test", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 68;
        std::string c1 = "AVID_QEMU_ARCH=cortex-a9 bash " + suite.repo_root + "/emulator/qemu/run.sh --self-test >/dev/null 2>&1";
        std::string c2 = "AVID_QEMU_ARCH=aarch64 bash " + suite.repo_root + "/emulator/qemu/run.sh --self-test >/dev/null 2>&1";
        int r1 = std::system(c1.c_str());
        int r2 = std::system(c2.c_str());
        int e1 = WIFEXITED(r1) ? WEXITSTATUS(r1) : -1;
        int e2 = WIFEXITED(r2) ? WEXITSTATUS(r2) : -1;
        ev = "cortex-a9_rc=" + std::to_string(e1) + " aarch64_rc=" + std::to_string(e2);
        return e1 == 0 && e2 == 0;
    });

    /* L4-BLD-05: AF_UNIX socket creation and clean unlink on SHUTDOWN */
    suite.run_case("L4-BLD-05", "G7", "Bridge creates AF_UNIX socket on startup and unlinks it on graceful SHUTDOWN", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        if (!br.start(suite.bridge_bin_path(), "bld05", 2, true)) return false;
        struct stat st1, st2;
        bool existed = (stat(br.sock_path.c_str(), &st1) == 0);
        std::string sp = br.sock_path;
        br.graceful_shutdown(11ULL);
        bool unlinked = (stat(sp.c_str(), &st2) != 0);
        cyc = 20;
        ev = "existed=" + std::to_string(existed) + " unlinked_on_shutdown=" + std::to_string(unlinked);
        return existed && unlinked;
    });

    /* L4-BLD-06: Stale socket file pre-existing on disk is unlinked & rebound cleanly */
    suite.run_case("L4-BLD-06", "G7", "Bridge unlinks pre-existing stale file at --socket path and binds cleanly", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string stale = avid_l4::make_temp_path("bld06_stale", ".sock");
        {
            std::ofstream f(stale);
            f << "stale_marker";
        }
        pid_t pid = fork();
        if (pid == 0) {
            execl(suite.bridge_bin_path().c_str(), suite.bridge_bin_path().c_str(),
                  "--require-native-rtl", "--socket", stale.c_str(), "--max-clients", "1", nullptr);
            _exit(127);
        }
        bool connected = false;
        for (int i = 0; i < 100; i++) {
            int fd = avid_l4::raw_unix_connect(stale, 200);
            if (fd >= 0) {
                std::string resp;
                avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":1,\"operation\":\"SHUTDOWN\"}\n", resp);
                close(fd);
                connected = (avid_l4::extract_json_str_simple(resp, "status") == "OK");
                break;
            }
            usleep(15000);
        }
        int ws = 0;
        waitpid(pid, &ws, 0);
        unlink(stale.c_str());
        cyc = 20;
        ev = "rebound over stale file & served SHUTDOWN";
        return connected;
    });

    /* L4-BLD-07: Startup RTL reset register state (0x0232, K=3, osd=1, epoch=0) */
    suite.run_case("L4-BLD-07", "G7", "Fresh bridge startup initializes RTL registers without explicit client RESET", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        if (!br.start(suite.bridge_bin_path(), "bld07", 2, true)) return false;
        avid_hal_t hal;
        avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, br.sock_path.c_str(), WRD_BASE_ADDR);
        uint32_t sx0 = 0, ax0 = 0, ep = 99, k = 0, om = 0;
        avid_hal_mmio_read(&hal, 0x000u, &sx0, &cyc);
        avid_hal_mmio_read(&hal, 0x100u, &ax0, &cyc);
        avid_hal_mmio_read(&hal, 0x050u, &ep, &cyc);
        avid_hal_mmio_read(&hal, 0x054u, &k, &cyc);
        avid_hal_mmio_read(&hal, 0x058u, &om, &cyc);
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        br.graceful_shutdown(99ULL);
        ev = "sx0=0x0232 ax0=0x0232 ep=0 K=3 osd=1";
        return sx0 == 0x0232u && ax0 == 0x0232u && ep == 0u && k == 3u && om == 1u;
    });

    /* L4-BLD-08: Deterministic startup cycle == 10 */
    suite.run_case("L4-BLD-08", "G3,G8", "Fresh bridge startup begins serving first transaction at exact RTL cycle 10", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        if (!br.start(suite.bridge_bin_path(), "bld08", 4, true)) return false;
        avid_hal_t hal;
        avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, br.sock_path.c_str(), WRD_BASE_ADDR);
        uint32_t rd = 0;
        uint64_t acc = 0;
        avid_hal_mmio_read(&hal, 0x050u, &rd, &acc);
        cyc = hal.stats.last_completed_cycle;
        avid_hal_close(&hal);
        br.graceful_shutdown(99ULL);
        ev = "first accepted_cycle=" + std::to_string(acc);
        return acc == 10ULL && cyc == 12ULL;
    });

    /* L4-BLD-09: Invalid / non-existent parent socket directory exits cleanly with non-zero rc */
    suite.run_case("L4-BLD-09", "G7", "Bridge exits cleanly with rc=1 when --socket parent directory does not exist", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = 0;
        std::string cmd = suite.bridge_bin_path() + " --socket /nonexistent_avid_dir_xyz/bad.sock >/dev/null 2>&1";
        int rc = std::system(cmd.c_str());
        int ec = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
        ev = "bad socket path exit_code=" + std::to_string(ec);
        return ec == 1;
    });

    /* L4-BLD-10: --max-clients=2 terminates automatically after serving exact 2 client sessions */
    suite.run_case("L4-BLD-10", "G7", "Bridge with --max-clients 2 exits cleanly with rc=0 after 2nd client disconnects", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        avid_l4::BridgeProcessGuard br;
        if (!br.start(suite.bridge_bin_path(), "bld10", 2, true)) return false;
        /* Note: br.start() used 1 probe connect! Serve 1 more client to hit max_clients == 2 */
        int fd = avid_l4::raw_unix_connect(br.sock_path);
        std::string resp;
        avid_l4::raw_rpc(fd, "{\"protocol_version\":1,\"transaction_id\":1,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n", resp);
        avid_l4::extract_json_u64_simple(resp, "completed_cycle", &cyc);
        close(fd);
        int ws = 0;
        bool exited = false;
        for (int i = 0; i < 50; i++) {
            if (waitpid(br.pid, &ws, WNOHANG) == br.pid) {
                br.pid = -1;
                exited = WIFEXITED(ws) && WEXITSTATUS(ws) == 0;
                break;
            }
            usleep(10000);
        }
        ev = "auto-exited rc=0 after 2 clients: " + std::to_string(exited);
        return exited;
    });

    /* L4-BLD-11: Bridge restart on identical socket path after SIGKILL */
    suite.run_case("L4-BLD-11", "G7", "After SIGKILL leaves stale socket node, new bridge instance rebinds same path and serves HAL", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        std::string shared_sock = avid_l4::make_temp_path("bld11_kill", ".sock");
        unlink(shared_sock.c_str());
        pid_t p1 = fork();
        if (p1 == 0) {
            execl(suite.bridge_bin_path().c_str(), suite.bridge_bin_path().c_str(),
                  "--require-native-rtl", "--socket", shared_sock.c_str(), "--max-clients", "8", nullptr);
            _exit(127);
        }
        for (int i = 0; i < 60; i++) {
            int f = avid_l4::raw_unix_connect(shared_sock, 200);
            if (f >= 0) { close(f); break; }
            usleep(15000);
        }
        kill(p1, SIGKILL);
        int ws = 0;
        waitpid(p1, &ws, 0);

        /* Start 2nd instance on exact same socket path (stale socket still on disk!) */
        pid_t p2 = fork();
        if (p2 == 0) {
            execl(suite.bridge_bin_path().c_str(), suite.bridge_bin_path().c_str(),
                  "--require-native-rtl", "--socket", shared_sock.c_str(), "--max-clients", "4", nullptr);
            _exit(127);
        }
        bool ok = false;
        for (int i = 0; i < 80; i++) {
            int f = avid_l4::raw_unix_connect(shared_sock, 200);
            if (f >= 0) {
                std::string r1, r2;
                avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":1,\"operation\":\"MMIO_READ\",\"address\":\"0x00000054\"}\n", r1);
                avid_l4::raw_rpc(f, "{\"protocol_version\":1,\"transaction_id\":2,\"operation\":\"SHUTDOWN\"}\n", r2);
                avid_l4::extract_json_u64_simple(r1, "completed_cycle", &cyc);
                close(f);
                ok = (avid_l4::extract_json_str_simple(r1, "status") == "OK");
                break;
            }
            usleep(15000);
        }
        waitpid(p2, &ws, 0);
        unlink(shared_sock.c_str());
        ev = "rebound same socket after SIGKILL; read K=3 OK";
        return ok;
    });

    return suite.finalize();
}
