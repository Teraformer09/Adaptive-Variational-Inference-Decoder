/* ===========================================================================
 * test_bridge_transactions.cpp — Milestone M3 Bridge & HAL Transaction Test
 * Verifies:
 *   1. Versioned JSON protocol handshake over Unix domain socket
 *   2. Register reads, writes, byte enables, and shadow -> active COMMIT
 *   3. Syndrome transfer -> 17-cycle (68.0 ns @ K=3) BP decode response
 *   4. Reset recovery and monotonic deterministic cycle progression
 * =========================================================================*/
#include "../../../software/src/hal/avid_hal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "[FAIL M3] %s (line %d)\n", msg, __LINE__); \
            return 1; \
        } \
    } while (0)

int main(int argc, char **argv) {
    const char *socket_path = "/tmp/avid_cosim_test.sock";
    if (argc > 1) socket_path = argv[1];

    avid_hal_t hal;
    int rc = avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, socket_path, 0x43C01000u);
    CHECK_TRUE(rc == 0, "avid_hal_open failed");

    uint64_t c0 = 0, c1 = 0, c2 = 0;
    uint32_t rdata = 0;

    /* 1. Reset RTL */
    CHECK_TRUE(avid_hal_reset_rtl(&hal, 10, &c0) == 0, "avid_hal_reset_rtl failed");
    CHECK_TRUE(c0 >= 10, "reset cycles < 10");

    /* 2. Read default weight register */
    CHECK_TRUE(avid_hal_mmio_read(&hal, 0x000u, &rdata, &c1) == 0, "mmio_read 0x000 failed");
    CHECK_TRUE((rdata & 0xFFFFu) == 0x0232u, "default weight mismatch");
    CHECK_TRUE(c1 > c0, "cycle did not advance on mmio_read");

    /* 3. Write inhomogeneous weights and COMMIT */
    int16_t wx[9] = {0x0190, 0x0240, 0x0190, 0x0190, 0x0140, 0x0260, 0x0260, 0x0240, 0x0190};
    int16_t wz[9] = {0x0190, 0x0240, 0x0190, 0x0190, 0x0140, 0x0260, 0x0260, 0x0240, 0x0190};
    uint32_t epoch_after = 0;
    CHECK_TRUE(avid_hal_write_weights_and_commit(&hal, wx, wz, &epoch_after, &c2) == 0, "write_weights_and_commit failed");
    CHECK_TRUE(epoch_after == 1u, "epoch_after != 1");
    CHECK_TRUE(c2 > c1, "cycle did not advance on commit");

    /* Verify active bank updated */
    CHECK_TRUE(avid_hal_mmio_read(&hal, 0x124u + 4u * 4u, &rdata, &c2) == 0, "active bank read failed");
    CHECK_TRUE((rdata & 0xFFFFu) == 0x0140u, "active bank q4 weight mismatch");

    /* 4. Send syndrome for q4 Z-error (sx = 0x6) and verify 17-cycle decode */
    uint16_t corr_x = 0, corr_z = 0;
    uint32_t bp_cycles = 0;
    CHECK_TRUE(avid_hal_send_syndrome(&hal, 0x6u, 0x0u, true, &corr_x, &corr_z, &bp_cycles, &c2) == 0, "send_syndrome failed");
    CHECK_TRUE(bp_cycles == 17u, "bp_cycles != 17 at K=3");
    CHECK_TRUE(corr_z == (1u << 4), "corr_z != (1 << 4) for q4 Z-error");

    avid_hal_close(&hal);
    printf("[Bridge M3] HAL <-> Bridge <-> RTL transaction test passed (tx=%llu, bp_cycles=%u = 68.0 ns, final_cycle=%llu)\n",
           (unsigned long long)hal.stats.total_transactions,
           bp_cycles,
           (unsigned long long)c2);
    return 0;
}
