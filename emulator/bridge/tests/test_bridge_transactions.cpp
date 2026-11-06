/* ===========================================================================
 * test_bridge_transactions.cpp — Milestone M3 Bridge & HAL Transaction Test
 * Verifies all Section 2.3 HAL/Bridge Acceptance Criteria:
 *   1. Versioned JSON protocol handshake & request-response ID matching
 *   2. Register reads, writes, byte enables, and shadow -> active COMMIT
 *   3. Syndrome transfer -> 17-cycle core / 22-cycle AXI BP decode response
 *   4. Partial socket write reassembly (chunked write across 4 syscalls)
 *   5. Negative tests:
 *        - Unsupported protocol_version (99) -> ERR_UNSUPPORTED_VERSION
 *        - Malformed / truncated JSON        -> ERR_MALFORMED_JSON
 *        - Duplicate transaction_id          -> ERR_DUPLICATE_TX_ID
 *        - Invalid MMIO register / K=0       -> ERR_INVALID_REGISTER
 *        - Invalid syndrome (> 0xFF)         -> ERR_INVALID_SYNDROME
 *   6. Disconnect & reconnect recovery without state loss
 * =========================================================================*/
#include "../../../software/src/hal/avid_hal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#define CHECK_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "[FAIL M3] %s (line %d)\n", msg, __LINE__); \
            return 1; \
        } \
    } while (0)

static int send_raw_and_read_line(int fd, const char *raw, char *resp, size_t cap, bool chunked = false) {
    size_t len = std::strlen(raw);
    if (!chunked) {
        if (write(fd, raw, len) != (ssize_t)len) return -1;
    } else {
        size_t pos = 0;
        while (pos < len) {
            size_t step = (len - pos > 12) ? 12 : (len - pos);
            if (write(fd, raw + pos, step) != (ssize_t)step) return -1;
            pos += step;
            usleep(1000);
        }
    }
    size_t rpos = 0;
    while (rpos + 1 < cap) {
        char c = 0;
        if (read(fd, &c, 1) <= 0) break;
        resp[rpos++] = c;
        if (c == '\n') break;
    }
    resp[rpos] = '\0';
    return (rpos > 0) ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *socket_path = "/tmp/avid_cosim_test.sock";
    if (argc > 1) socket_path = argv[1];

    avid_hal_t hal;
    int rc = avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, socket_path, 0x43C01000u);
    CHECK_TRUE(rc == 0, "avid_hal_open failed");
    CHECK_TRUE(avid_hal_set_timeout_ms(&hal, 2000) == 0, "avid_hal_set_timeout_ms failed");

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

    /* 4. Send syndrome for q4 Z-error (sx = 0x6) and verify 17-cycle core / 22-cycle AXI decode */
    uint16_t corr_x = 0, corr_z = 0;
    uint32_t bp_cycles = 0;
    CHECK_TRUE(avid_hal_send_syndrome(&hal, 0x6u, 0x0u, true, &corr_x, &corr_z, &bp_cycles, &c2) == 0, "send_syndrome failed");
    CHECK_TRUE(bp_cycles == 17u, "bp_cycles != 17 at K=3");
    CHECK_TRUE(hal.stats.last_axis_total_cycles == 22u, "axis_total_cycles != 22 at K=3");
    CHECK_TRUE(corr_z == (1u << 4), "corr_z != (1 << 4) for q4 Z-error");

    /* 5. Test chunked / partial socket write reassembly (Section 2.3 I/O reliability) */
    char resp[512];
    const char *chunked_req = "{\"protocol_version\":1,\"transaction_id\":5001,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\",\"byte_enable\":15}\n";
    CHECK_TRUE(send_raw_and_read_line(hal.sock_fd, chunked_req, resp, sizeof(resp), true) == 0, "chunked write failed");
    CHECK_TRUE(std::strstr(resp, "\"status\":\"OK\"") != nullptr, "chunked write did not return OK");
    CHECK_TRUE(std::strstr(resp, "\"transaction_id\":5001") != nullptr, "chunked write tx_id mismatch");

    /* 6. Section 2.3 Negative Tests */
    /* 6a. Duplicate transaction_id (5001 repeated) */
    CHECK_TRUE(send_raw_and_read_line(hal.sock_fd, chunked_req, resp, sizeof(resp), false) == 0, "dup tx write failed");
    CHECK_TRUE(std::strstr(resp, "ERR_DUPLICATE_TX_ID") != nullptr, "duplicate tx_id was not rejected");

    /* 6b. Unsupported protocol_version: 99 */
    const char *bad_ver = "{\"protocol_version\":99,\"transaction_id\":5002,\"operation\":\"MMIO_READ\",\"address\":\"0x00000050\"}\n";
    CHECK_TRUE(send_raw_and_read_line(hal.sock_fd, bad_ver, resp, sizeof(resp), false) == 0, "bad_ver write failed");
    CHECK_TRUE(std::strstr(resp, "ERR_UNSUPPORTED_VERSION") != nullptr, "unsupported version was not rejected");

    /* 6c. Malformed / truncated JSON */
    const char *bad_json = "{\"protocol_version\":1,\"transaction_id\":5003,\"operation\":\"MMIO_READ\"\n";
    CHECK_TRUE(send_raw_and_read_line(hal.sock_fd, bad_json, resp, sizeof(resp), false) == 0, "bad_json write failed");
    CHECK_TRUE(std::strstr(resp, "ERR_MALFORMED_JSON") != nullptr, "malformed JSON was not rejected");

    /* 6d. Invalid MMIO register / invalid K=0 via HAL */
    CHECK_TRUE(avid_hal_mmio_write(&hal, 0x054u, 0x0u, 0x1u, &c2) != 0, "invalid K=0 write was not rejected by HAL");
    CHECK_TRUE(avid_hal_mmio_read(&hal, 0x990u, &rdata, &c2) != 0, "unmapped read 0x990 was not rejected by HAL");

    /* 6e. Invalid syndrome (> 0xFF) */
    const char *bad_synd = "{\"protocol_version\":1,\"transaction_id\":5004,\"operation\":\"SYNDROME_TRANSFER\",\"address\":\"0x0\",\"data\":\"0x1FFFF\"}\n";
    CHECK_TRUE(send_raw_and_read_line(hal.sock_fd, bad_synd, resp, sizeof(resp), false) == 0, "bad_synd write failed");
    CHECK_TRUE(std::strstr(resp, "ERR_INVALID_SYNDROME") != nullptr, "invalid syndrome was not rejected");

    /* 7. Section 2.3 Disconnect & Reconnect Recovery Test */
    hal.next_tx_id = 6000;
    CHECK_TRUE(avid_hal_reconnect(&hal) == 0, "avid_hal_reconnect failed");
    CHECK_TRUE(hal.stats.reconnect_events == 1u, "reconnect_events != 1");
    CHECK_TRUE(avid_hal_mmio_read(&hal, 0x050u, &rdata, &c2) == 0, "mmio_read after reconnect failed");
    CHECK_TRUE(rdata == 1u, "epoch_count state lost across reconnect");

    avid_hal_close(&hal);
    printf("[Bridge M3] HAL <-> Bridge <-> RTL positive, chunked-I/O, negative & reconnect tests passed (bp_cycles=%u, axis_cycles=22, final_cycle=%llu)\n",
           bp_cycles,
           (unsigned long long)c2);
    return 0;
}
