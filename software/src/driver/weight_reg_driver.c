/* ===========================================================================
 * weight_reg_driver.c — AVID Phase 2.1
 * User-space /dev/mem driver for weight_map_reg AXI-Lite registers.
 *
 * Build (on Zynq ARM Linux):
 *   gcc -O2 -o weight_reg_driver.o -c weight_reg_driver.c
 *
 * Cross-compile (from x86 host):
 *   arm-linux-gnueabihf-gcc -O2 -o weight_reg_driver.o -c weight_reg_driver.c
 * =========================================================================*/
#include "weight_reg_driver.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>

/* Internal helper: 32-bit MMIO write */
static inline void mmio_write32(void *base, uint32_t offset, uint32_t val) {
    volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)base + offset);
    *reg = val;
    __asm__ volatile("dsb sy" ::: "memory");  /* Data Synchronisation Barrier */
}

/* Internal helper: 32-bit MMIO read */
static inline uint32_t mmio_read32(void *base, uint32_t offset) {
    volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)base + offset);
    uint32_t val = *reg;
    __asm__ volatile("dsb sy" ::: "memory");
    return val;
}

/* -------------------------------------------------------------------------
 * wrd_open
 * -------------------------------------------------------------------------*/
int wrd_open(wrd_handle_t *h, uint32_t base_addr) {
    h->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (h->fd < 0) {
        perror("[wrd] open /dev/mem");
        return -1;
    }
    h->base_addr = base_addr;
    h->mapped_base = mmap(NULL, WRD_MAP_SIZE,
                          PROT_READ | PROT_WRITE, MAP_SHARED,
                          h->fd, (off_t)base_addr);
    if (h->mapped_base == MAP_FAILED) {
        perror("[wrd] mmap");
        close(h->fd);
        return -1;
    }
    printf("[wrd] Mapped weight registry at 0x%08X (fd=%d)\n", base_addr, h->fd);
    return 0;
}

/* -------------------------------------------------------------------------
 * wrd_close
 * -------------------------------------------------------------------------*/
void wrd_close(wrd_handle_t *h) {
    if (h->mapped_base && h->mapped_base != MAP_FAILED)
        munmap(h->mapped_base, WRD_MAP_SIZE);
    if (h->fd >= 0)
        close(h->fd);
    memset(h, 0, sizeof(*h));
}

/* -------------------------------------------------------------------------
 * wrd_write_x_llr / wrd_write_z_llr
 * -------------------------------------------------------------------------*/
void wrd_write_x_llr(const wrd_handle_t *h, int qubit_idx, int16_t q78_val) {
    if (qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return;
    mmio_write32(h->mapped_base, WRD_OFF_X(qubit_idx), (uint32_t)(uint16_t)q78_val);
}

void wrd_write_z_llr(const wrd_handle_t *h, int qubit_idx, int16_t q78_val) {
    if (qubit_idx < 0 || qubit_idx >= WRD_N_DATA) return;
    mmio_write32(h->mapped_base, WRD_OFF_Z(qubit_idx), (uint32_t)(uint16_t)q78_val);
}

/* -------------------------------------------------------------------------
 * wrd_write_all
 * Write all 9+9 LLRs in one sequential burst.
 * The ARM bus handles this as ~18 single-word stores; no DMA needed.
 * -------------------------------------------------------------------------*/
void wrd_write_all(const wrd_handle_t *h, const float *llr_x, const float *llr_z) {
    for (int i = 0; i < WRD_N_DATA; i++) {
        int16_t q78x = wrd_float_to_q78(llr_x[i]);
        int16_t q78z = wrd_float_to_q78(llr_z[i]);
        mmio_write32(h->mapped_base, WRD_OFF_X(i), (uint32_t)(uint16_t)q78x);
        mmio_write32(h->mapped_base, WRD_OFF_Z(i), (uint32_t)(uint16_t)q78z);
    }
}

/* -------------------------------------------------------------------------
 * wrd_commit
 * Triggers the atomic shadow → active swap in the FPGA.
 * The FPGA will execute the copy at the next bp_ready=1 clock edge.
 * No polling required — the swap is deterministic on the FPGA side.
 * -------------------------------------------------------------------------*/
void wrd_commit(const wrd_handle_t *h) {
    mmio_write32(h->mapped_base, WRD_OFF_COMMIT, 1U);
}

/* -------------------------------------------------------------------------
 * wrd_read_epoch / wrd_read_status
 * -------------------------------------------------------------------------*/
uint32_t wrd_read_epoch(const wrd_handle_t *h) {
    return mmio_read32(h->mapped_base, WRD_OFF_EPOCH);
}

uint32_t wrd_read_status(const wrd_handle_t *h) {
    return mmio_read32(h->mapped_base, WRD_OFF_STATUS);
}
