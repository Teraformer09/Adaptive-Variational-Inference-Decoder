/* ===========================================================================
 * simulation_backend.cpp — Unix Domain Socket Transport to Co-Sim Bridge
 * Implements the versioned fixed-schema JSON transaction protocol (v1)
 * between the QEMU ARM64 guest (avid_daemon) and the C++ Co-Sim Bridge.
 * =========================================================================*/
#include "avid_hal.h"
#include "../custom_engine/cycle_accurate_bp_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

extern int  avid_hal_open_physical(avid_hal_t *hal, uint32_t phys_base_addr);
extern void avid_hal_close_physical(avid_hal_t *hal);
extern int  avid_hal_mmio_write_physical(avid_hal_t *hal, uint32_t offset, uint32_t data, uint8_t byte_enable, uint64_t *accepted_cycle);
extern int  avid_hal_mmio_read_physical(avid_hal_t *hal, uint32_t offset, uint32_t *rdata, uint64_t *accepted_cycle);

static bool extract_json_str(const char *buf, const char *key, char *out, size_t max_len) {
    char pat[64];
    std::snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = std::strstr(buf, pat);
    if (!p) return false;
    p = std::strchr(p + std::strlen(pat), ':');
    if (!p) return false;
    while (*p == ':' || *p == ' ' || *p == '\t') p++;
    if (*p == '"') {
        p++;
        size_t i = 0;
        while (*p && *p != '"' && i + 1 < max_len) out[i++] = *p++;
        out[i] = '\0';
        return true;
    }
    size_t i = 0;
    while (*p && *p != ',' && *p != '}' && *p != ' ' && *p != '\n' && *p != '\r' && i + 1 < max_len) {
        out[i++] = *p++;
    }
    out[i] = '\0';
    return true;
}

static uint64_t extract_json_u64(const char *buf, const char *key, uint64_t def_val) {
    char tmp[64];
    if (!extract_json_str(buf, key, tmp, sizeof(tmp))) return def_val;
    return (uint64_t)std::strtoull(tmp, nullptr, 0);
}

static int send_bridge_request(avid_hal_t *hal, const char *json_line, char *resp_buf, size_t resp_cap) {
    if (!hal || hal->sock_fd < 0) return -1;
    size_t len = std::strlen(json_line);
    ssize_t w = write(hal->sock_fd, json_line, len);
    if (w != (ssize_t)len) {
        hal->stats.rejected_transactions++;
        return -1;
    }
    size_t pos = 0;
    while (pos + 1 < resp_cap) {
        char c = 0;
        ssize_t r = read(hal->sock_fd, &c, 1);
        if (r <= 0) break;
        resp_buf[pos++] = c;
        if (c == '\n') break;
    }
    resp_buf[pos] = '\0';
    if (pos == 0) {
        hal->stats.rejected_transactions++;
        return -1;
    }
    char status[32] = {0};
    extract_json_str(resp_buf, "status", status, sizeof(status));
    uint64_t cyc = extract_json_u64(resp_buf, "accepted_cycle", hal->stats.last_accepted_cycle);
    hal->stats.last_accepted_cycle = cyc;
    hal->stats.total_transactions++;
    if (std::strcmp(status, "OK") != 0) {
        hal->stats.rejected_transactions++;
        return -1;
    }
    return 0;
}

int avid_hal_open(avid_hal_t *hal,
                  avid_hal_backend_type_t backend,
                  const char *socket_path,
                  uint32_t phys_base_addr) {
    if (!hal) return -1;
    std::memset(hal, 0, sizeof(*hal));
    hal->backend = backend;
    hal->sock_fd = -1;
    hal->next_tx_id = 1000;

    if (backend == AVID_HAL_BACKEND_PHYSICAL) {
        return avid_hal_open_physical(hal, phys_base_addr);
    }

    if (!socket_path || socket_path[0] == '\0') return -1;
    std::snprintf(hal->socket_path, sizeof(hal->socket_path), "%s", socket_path);

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path);

    int retries = 50;
    while (retries-- > 0) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            hal->sock_fd = fd;
            return 0;
        }
        usleep(20000);
    }
    close(fd);
    return -1;
}

void avid_hal_close(avid_hal_t *hal) {
    if (!hal) return;
    if (hal->backend == AVID_HAL_BACKEND_PHYSICAL) {
        avid_hal_close_physical(hal);
        return;
    }
    if (hal->sock_fd >= 0) {
        close(hal->sock_fd);
        hal->sock_fd = -1;
    }
}

int avid_hal_mmio_write(avid_hal_t *hal,
                        uint32_t offset,
                        uint32_t data,
                        uint8_t byte_enable,
                        uint64_t *accepted_cycle) {
    if (!hal) return -1;
    if (hal->backend == AVID_HAL_BACKEND_PHYSICAL) {
        return avid_hal_mmio_write_physical(hal, offset, data, byte_enable, accepted_cycle);
    }
    char req[512], resp[512];
    uint64_t tx_id = hal->next_tx_id++;
    std::snprintf(req, sizeof(req),
        "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"MMIO_WRITE\","
        "\"address\":\"0x%08X\",\"data\":\"0x%08X\",\"byte_enable\":%u}\n",
        (unsigned long long)tx_id, offset, data, (unsigned)(byte_enable & 0xFu));
    if (send_bridge_request(hal, req, resp, sizeof(resp)) != 0) return -1;
    hal->stats.mmio_writes++;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_mmio_read(avid_hal_t *hal,
                       uint32_t offset,
                       uint32_t *rdata,
                       uint64_t *accepted_cycle) {
    if (!hal || !rdata) return -1;
    if (hal->backend == AVID_HAL_BACKEND_PHYSICAL) {
        return avid_hal_mmio_read_physical(hal, offset, rdata, accepted_cycle);
    }
    char req[512], resp[512];
    uint64_t tx_id = hal->next_tx_id++;
    std::snprintf(req, sizeof(req),
        "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"MMIO_READ\","
        "\"address\":\"0x%08X\",\"byte_enable\":15}\n",
        (unsigned long long)tx_id, offset);
    if (send_bridge_request(hal, req, resp, sizeof(resp)) != 0) return -1;
    *rdata = (uint32_t)extract_json_u64(resp, "rdata", 0u);
    hal->stats.mmio_reads++;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_write_weights_and_commit(avid_hal_t *hal,
                                      const int16_t q78_x[9],
                                      const int16_t q78_z[9],
                                      uint32_t *epoch_out,
                                      uint64_t *accepted_cycle) {
    if (!hal || !q78_x || !q78_z) return -1;
    for (int i = 0; i < 9; i++) {
        if (avid_hal_mmio_write(hal, (uint32_t)(0x000 + 4 * i), (uint16_t)q78_x[i], 0x3, accepted_cycle) != 0)
            return -1;
    }
    for (int i = 0; i < 9; i++) {
        if (avid_hal_mmio_write(hal, (uint32_t)(0x024 + 4 * i), (uint16_t)q78_z[i], 0x3, accepted_cycle) != 0)
            return -1;
    }
    if (avid_hal_mmio_write(hal, 0x048u, 0x1u, 0x1, accepted_cycle) != 0) return -1;
    hal->stats.commit_operations++;
    uint32_t ep = 0;
    if (avid_hal_mmio_read(hal, 0x050u, &ep, accepted_cycle) != 0) return -1;
    if (epoch_out) *epoch_out = ep;
    return 0;
}

int avid_hal_send_syndrome(avid_hal_t *hal,
                           uint8_t sx,
                           uint8_t sz,
                           bool tlast,
                           uint16_t *corr_x,
                           uint16_t *corr_z,
                           uint32_t *bp_core_cycles,
                           uint64_t *accepted_cycle) {
    if (!hal) return -1;
    if (hal->backend == AVID_HAL_BACKEND_PHYSICAL) {
        int16_t wx[9], wz[9];
        for (int i = 0; i < 9; i++) {
            wx[i] = wrd_read_active_weight(&hal->phys_wrd, i);
            wz[i] = wx[i];
        }
        CycleAccurateBPEngine bp(3, 250.0);
        avid_bp_result_t r = bp.decode((uint8_t)(sx & 0xFu), wz, 3);
        if (corr_x) *corr_x = 0;
        if (corr_z) *corr_z = r.correction_mask;
        if (bp_core_cycles) *bp_core_cycles = (uint32_t)r.cycles_fused;
        hal->stats.syndrome_packets++;
        hal->stats.last_accepted_cycle += (uint64_t)r.cycles_fused;
        if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
        return 0;
    }

    uint16_t packed_synd = (uint16_t)((sx & 0xFu) | ((sz & 0xFu) << 4));
    char req[512], resp[512];
    uint64_t tx_id = hal->next_tx_id++;
    std::snprintf(req, sizeof(req),
        "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"SYNDROME_TRANSFER\","
        "\"address\":\"0x00000000\",\"data\":\"0x%04X\",\"tlast\":%d,\"byte_enable\":3}\n",
        (unsigned long long)tx_id, (unsigned)packed_synd, tlast ? 1 : 0);
    if (send_bridge_request(hal, req, resp, sizeof(resp)) != 0) return -1;
    if (corr_x) *corr_x = (uint16_t)extract_json_u64(resp, "corr_x", 0u);
    if (corr_z) *corr_z = (uint16_t)extract_json_u64(resp, "corr_z", 0u);
    if (bp_core_cycles) *bp_core_cycles = (uint32_t)extract_json_u64(resp, "bp_cycles", 17u);
    hal->stats.syndrome_packets++;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_reset_rtl(avid_hal_t *hal,
                       uint32_t reset_cycles,
                       uint64_t *accepted_cycle) {
    if (!hal) return -1;
    if (hal->backend == AVID_HAL_BACKEND_PHYSICAL) {
        uint32_t baddr = hal->phys_wrd.base_addr ? hal->phys_wrd.base_addr : WRD_BASE_ADDR;
        wrd_close(&hal->phys_wrd);
        wrd_open(&hal->phys_wrd, baddr);
        hal->stats.reset_operations++;
        hal->stats.last_accepted_cycle += reset_cycles;
        if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
        return 0;
    }
    char req[512], resp[512];
    uint64_t tx_id = hal->next_tx_id++;
    std::snprintf(req, sizeof(req),
        "{\"protocol_version\":1,\"transaction_id\":%llu,\"operation\":\"RESET\","
        "\"address\":\"0x00000000\",\"data\":\"0x%08X\",\"byte_enable\":15}\n",
        (unsigned long long)tx_id, reset_cycles);
    if (send_bridge_request(hal, req, resp, sizeof(resp)) != 0) return -1;
    hal->stats.reset_operations++;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_set_bp_iterations(avid_hal_t *hal,
                               uint32_t k_iter,
                               uint64_t *accepted_cycle) {
    return avid_hal_mmio_write(hal, 0x054u, k_iter, 0x1, accepted_cycle);
}
