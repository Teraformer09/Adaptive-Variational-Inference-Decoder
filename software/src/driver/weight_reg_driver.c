/* ===========================================================================
 * weight_reg_driver.c — RTL-Aligned AXI-Lite Driver for `weight_map_reg.sv`
 * Matches `firmware/rtl/config/weight_map_reg.sv` bit-for-bit across all
 * shadow_x/z, commit_snap_x/z, active_x/z, COMMIT, STATUS, epoch_count,
 * cfg_max_iter, cfg_osd_mode, byte_enable, and SLVERR rules.
 * =========================================================================*/
#include "weight_reg_driver.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

static inline void avid_memory_barrier(void) {
#if defined(__arm__) || defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static void wrd_init_defaults(wrd_handle_t *h) {
    for (int i = 0; i < WRD_N_DATA; i++) {
        h->shadow_x[i]      = (int16_t)WRD_DEFAULT_Q78;
        h->shadow_z[i]      = (int16_t)WRD_DEFAULT_Q78;
        h->commit_snap_x[i] = (int16_t)WRD_DEFAULT_Q78;
        h->commit_snap_z[i] = (int16_t)WRD_DEFAULT_Q78;
        h->active_x[i]      = (int16_t)WRD_DEFAULT_Q78;
        h->active_z[i]      = (int16_t)WRD_DEFAULT_Q78;
    }
    h->swap_pending = 0;
    h->bp_busy      = 0;
    h->epoch_count  = 0;
    h->cfg_max_iter = 3;
    h->cfg_osd_mode = 1;
}

int wrd_open(wrd_handle_t *h, uint32_t base_addr) {
    if (!h) return -1;
    memset(h, 0, sizeof(*h));
    h->base_addr = base_addr ? base_addr : (uint32_t)WRD_BASE_ADDR;
    h->fd = -1;
    wrd_init_defaults(h);

    /* User-space emulation mode unless running with mapped Zynq /dev/mem */
    if (base_addr == (uint32_t)WRD_BASE_ADDR) {
        h->fd = open("/dev/mem", O_RDWR | O_SYNC);
    }
    if (h->fd < 0) {
        h->is_emulated = 1;
        return 0;
    }
    h->mapped_base = mmap(NULL, WRD_MAP_SIZE,
                          PROT_READ | PROT_WRITE, MAP_SHARED,
                          h->fd, (off_t)h->base_addr);
    if (h->mapped_base == MAP_FAILED) {
        close(h->fd);
        h->fd = -1;
        h->mapped_base = NULL;
        h->is_emulated = 1;
        return 0;
    }
    h->is_emulated = 0;
    return 0;
}

void wrd_close(wrd_handle_t *h) {
    if (!h) return;
    if (!h->is_emulated && h->mapped_base && h->mapped_base != MAP_FAILED) {
        munmap(h->mapped_base, WRD_MAP_SIZE);
    }
    if (h->fd >= 0) {
        close(h->fd);
    }
    memset(h, 0, sizeof(*h));
}

int wrd_mmio_write_checked(wrd_handle_t *h, uint32_t offset, uint32_t data, uint8_t byte_enable) {
    if (!h) return -1;
    if ((offset & 0x3U) != 0U) return -1; /* SLVERR on unaligned address */

    uint16_t wmask = 0;
    if (byte_enable & 0x1U) wmask |= 0x00FFU;
    if (byte_enable & 0x2U) wmask |= 0xFF00U;

    bool shadow_locked = (h->swap_pending != 0) && (h->bp_busy != 0);

    if (offset < 0x024U) {
        if (shadow_locked) return -1;
        uint32_t idx = offset >> 2;
        if (h->is_emulated) {
            uint16_t cur = (uint16_t)h->shadow_x[idx];
            h->shadow_x[idx] = (int16_t)((cur & ~wmask) | ((uint16_t)data & wmask));
        } else if (h->mapped_base) {
            volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)h->mapped_base + offset);
            *reg = (*reg & ~(uint32_t)wmask) | (data & (uint32_t)wmask);
            avid_memory_barrier();
        }
        return 0;
    }

    if (offset >= 0x024U && offset < 0x048U) {
        if (shadow_locked) return -1;
        uint32_t idx = (offset - 0x024U) >> 2;
        if (h->is_emulated) {
            uint16_t cur = (uint16_t)h->shadow_z[idx];
            h->shadow_z[idx] = (int16_t)((cur & ~wmask) | ((uint16_t)data & wmask));
        } else if (h->mapped_base) {
            volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)h->mapped_base + offset);
            *reg = (*reg & ~(uint32_t)wmask) | (data & (uint32_t)wmask);
            avid_memory_barrier();
        }
        return 0;
    }

    if (offset == WRD_OFF_COMMIT) {
        if (shadow_locked) return -1;
        if ((data & 0x1U) && (byte_enable & 0x1U)) {
            if (h->is_emulated) {
                for (int i = 0; i < WRD_N_DATA; i++) {
                    h->commit_snap_x[i] = h->shadow_x[i];
                    h->commit_snap_z[i] = h->shadow_z[i];
                }
                if (!h->bp_busy) {
                    for (int i = 0; i < WRD_N_DATA; i++) {
                        h->active_x[i] = h->commit_snap_x[i];
                        h->active_z[i] = h->commit_snap_z[i];
                    }
                    h->swap_pending = 0;
                    h->epoch_count += 1U;
                } else {
                    h->swap_pending = 1;
                }
            } else if (h->mapped_base) {
                volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)h->mapped_base + offset);
                *reg = 1U;
                avid_memory_barrier();
            }
        }
        return 0;
    }

    if (offset == WRD_OFF_MAX_ITER) {
        if (!(byte_enable & 0x1U) || data < 1U || data > 5U) return -1;
        if (h->is_emulated) {
            h->cfg_max_iter = data;
        } else if (h->mapped_base) {
            volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)h->mapped_base + offset);
            *reg = data;
            avid_memory_barrier();
        }
        return 0;
    }

    if (offset == WRD_OFF_OSD_MODE) {
        if (!(byte_enable & 0x1U) || data > 2U) return -1;
        if (h->is_emulated) {
            h->cfg_osd_mode = data;
        } else if (h->mapped_base) {
            volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)h->mapped_base + offset);
            *reg = data;
            avid_memory_barrier();
        }
        return 0;
    }

    /* Read-only (0x04C, 0x050, 0x100..0x144) or unmapped -> SLVERR */
    return -1;
}

int wrd_mmio_read_checked(const wrd_handle_t *h, uint32_t offset, uint32_t *rdata) {
    if (!h || !rdata) return -1;
    if ((offset & 0x3U) != 0U) {
        *rdata = 0xDEADBEEFU;
        return -1;
    }

    if (!h->is_emulated && h->mapped_base) {
        if (offset <= 0x058U || (offset >= 0x100U && offset < 0x148U)) {
            volatile uint32_t *reg = (volatile uint32_t *)((const uint8_t *)h->mapped_base + offset);
            *rdata = *reg;
            avid_memory_barrier();
            return 0;
        }
        *rdata = 0xDEADBEEFU;
        return -1;
    }

    if (offset < 0x024U) {
        *rdata = (uint32_t)(uint16_t)h->shadow_x[offset >> 2];
        return 0;
    }
    if (offset >= 0x024U && offset < 0x048U) {
        *rdata = (uint32_t)(uint16_t)h->shadow_z[(offset - 0x024U) >> 2];
        return 0;
    }
    if (offset == WRD_OFF_COMMIT) {
        *rdata = (uint32_t)(h->swap_pending & 0x1U);
        return 0;
    }
    if (offset == WRD_OFF_STATUS) {
        *rdata = ((uint32_t)(h->bp_busy & 0x1U) << 1) | (uint32_t)(h->swap_pending & 0x1U);
        return 0;
    }
    if (offset == WRD_OFF_EPOCH) {
        *rdata = h->epoch_count;
        return 0;
    }
    if (offset == WRD_OFF_MAX_ITER) {
        *rdata = h->cfg_max_iter;
        return 0;
    }
    if (offset == WRD_OFF_OSD_MODE) {
        *rdata = h->cfg_osd_mode;
        return 0;
    }
    if (offset >= 0x100U && offset < 0x124U) {
        *rdata = (uint32_t)(uint16_t)h->active_x[(offset - 0x100U) >> 2];
        return 0;
    }
    if (offset >= 0x124U && offset < 0x148U) {
        *rdata = (uint32_t)(uint16_t)h->active_z[(offset - 0x124U) >> 2];
        return 0;
    }

    *rdata = 0xDEADBEEFU;
    return -1;
}

void wrd_write_x_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val) {
    if (!h || qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return;
    (void)wrd_mmio_write_checked(h, WRD_OFF_SHADOW_X(qubit_idx), (uint32_t)(uint16_t)q78_val, 0x3U);
}

void wrd_write_z_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val) {
    if (!h || qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return;
    (void)wrd_mmio_write_checked(h, WRD_OFF_SHADOW_Z(qubit_idx), (uint32_t)(uint16_t)q78_val, 0x3U);
}

void wrd_write_all(wrd_handle_t *h, const float *llr_x, const float *llr_z) {
    if (!h || !llr_x || !llr_z) return;
    for (int i = 0; i < WRD_N_DATA; i++) {
        wrd_write_x_llr(h, i, wrd_float_to_q78(llr_x[i]));
        wrd_write_z_llr(h, i, wrd_float_to_q78(llr_z[i]));
    }
}

void wrd_write_all_weights(wrd_handle_t *h, const int16_t weights_q78[WRD_NUM_QUBITS]) {
    if (!h || !weights_q78) return;
    for (int i = 0; i < WRD_NUM_QUBITS; i++) {
        wrd_write_x_llr(h, i, weights_q78[i]);
        wrd_write_z_llr(h, i, weights_q78[i]);
    }
}

int16_t wrd_read_active_x_weight(const wrd_handle_t *h, int qubit_idx) {
    if (!h || qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return 0;
    uint32_t rdata = 0;
    if (wrd_mmio_read_checked(h, WRD_OFF_ACTIVE_X(qubit_idx), &rdata) != 0) return 0;
    return (int16_t)(rdata & 0xFFFFU);
}

int16_t wrd_read_active_z_weight(const wrd_handle_t *h, int qubit_idx) {
    if (!h || qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return 0;
    uint32_t rdata = 0;
    if (wrd_mmio_read_checked(h, WRD_OFF_ACTIVE_Z(qubit_idx), &rdata) != 0) return 0;
    return (int16_t)(rdata & 0xFFFFU);
}

int16_t wrd_read_active_weight(const wrd_handle_t *h, int qubit_idx) {
    return wrd_read_active_x_weight(h, qubit_idx);
}

void wrd_commit(wrd_handle_t *h) {
    if (!h) return;
    (void)wrd_mmio_write_checked(h, WRD_OFF_COMMIT, 1U, 0x1U);
}

int wrd_commit_and_wait(wrd_handle_t *h, int max_polls) {
    if (!h || max_polls <= 0) return -1;
    if (wrd_mmio_write_checked(h, WRD_OFF_COMMIT, 1U, 0x1U) != 0) return -1;
    for (int p = 0; p < max_polls; p++) {
        uint32_t st = wrd_read_status(h);
        if ((st & 0x1U) == 0U) {
            return 0;
        }
    }
    return -1;
}

uint32_t wrd_read_status(const wrd_handle_t *h) {
    uint32_t rdata = 0;
    if (wrd_mmio_read_checked(h, WRD_OFF_STATUS, &rdata) != 0) return 0;
    return rdata;
}

uint32_t wrd_read_epoch(const wrd_handle_t *h) {
    uint32_t rdata = 0;
    if (wrd_mmio_read_checked(h, WRD_OFF_EPOCH, &rdata) != 0) return 0;
    return rdata;
}

uint16_t wrd_get_swap_count(const wrd_handle_t *h) {
    return (uint16_t)(wrd_read_epoch(h) & 0xFFFFU);
}
