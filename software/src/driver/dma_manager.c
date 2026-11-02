/* ===========================================================================
 * dma_manager.c — Xilinx AXI DMA Simple Mode User-Space Driver (Portable)
 * =========================================================================*/
#include "dma_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>

static inline void dma_memory_barrier(void) {
#if defined(__arm__) || defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static inline void dma_reg_write(void *base, uint32_t off, uint32_t val) {
    volatile uint32_t *r = (volatile uint32_t *)((uint8_t *)base + off);
    *r = val;
    dma_memory_barrier();
}

static inline uint32_t dma_reg_read(void *base, uint32_t off) {
    volatile uint32_t *r = (volatile uint32_t *)((uint8_t *)base + off);
    uint32_t v = *r;
    dma_memory_barrier();
    return v;
}

int dma_open(dma_handle_t *h, uint32_t dma_base_addr) {
    memset(h, 0, sizeof(*h));
    h->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (h->fd_mem < 0) {
        return -1;
    }

    h->dma_base_addr = dma_base_addr;
    h->dma_base = mmap(NULL, DMA_MAP_SIZE, PROT_READ | PROT_WRITE,
                       MAP_SHARED, h->fd_mem, (off_t)dma_base_addr);
    if (h->dma_base == MAP_FAILED) goto err;

    h->syndrome_buf = mmap(NULL, DDR_BUFFER_SIZE, PROT_READ | PROT_WRITE,
                           MAP_SHARED, h->fd_mem, (off_t)DDR_SYNDROME_PHYS);
    if (h->syndrome_buf == MAP_FAILED) goto err;

    h->correction_buf = mmap(NULL, DDR_BUFFER_SIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, h->fd_mem, (off_t)DDR_CORRECT_PHYS);
    if (h->correction_buf == MAP_FAILED) goto err;

    h->max_syndrome_bytes = DDR_BUFFER_SIZE;
    dma_reg_write(h->dma_base, DMA_MM2S_CONTROL, DMA_CTRL_RESET);
    dma_reg_write(h->dma_base, DMA_S2MM_CONTROL, DMA_CTRL_RESET);
    usleep(10);
    dma_reg_write(h->dma_base, DMA_MM2S_CONTROL, DMA_CTRL_RUN | DMA_CTRL_IOC_IRQEN);
    dma_reg_write(h->dma_base, DMA_S2MM_CONTROL, DMA_CTRL_RUN | DMA_CTRL_IOC_IRQEN);
    return 0;

err:
    dma_close(h);
    return -1;
}

void dma_close(dma_handle_t *h) {
    if (h->syndrome_buf   && h->syndrome_buf   != MAP_FAILED) munmap(h->syndrome_buf,   DDR_BUFFER_SIZE);
    if (h->correction_buf && h->correction_buf != MAP_FAILED) munmap(h->correction_buf, DDR_BUFFER_SIZE);
    if (h->dma_base       && h->dma_base       != MAP_FAILED) munmap(h->dma_base,       DMA_MAP_SIZE);
    if (h->fd_mem >= 0) close(h->fd_mem);
    memset(h, 0, sizeof(*h));
}

int dma_load_syndromes(dma_handle_t *h, const void *data, size_t n_bytes) {
    if (n_bytes > h->max_syndrome_bytes) return -1;
    memcpy(h->syndrome_buf, data, n_bytes);
    dma_memory_barrier();
    return 0;
}

int dma_start_stream(dma_handle_t *h, size_t syndrome_bytes, size_t correction_bytes) {
    dma_reg_write(h->dma_base, DMA_S2MM_DST_ADDR, DDR_CORRECT_PHYS);
    dma_reg_write(h->dma_base, DMA_S2MM_LENGTH,   (uint32_t)correction_bytes);
    dma_reg_write(h->dma_base, DMA_MM2S_SRC_ADDR, DDR_SYNDROME_PHYS);
    dma_reg_write(h->dma_base, DMA_MM2S_LENGTH,   (uint32_t)syndrome_bytes);
    return 0;
}

int dma_wait_complete(dma_handle_t *h, int timeout_us) {
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    while (1) {
        uint32_t mm2s_stat = dma_reg_read(h->dma_base, DMA_MM2S_STATUS);
        uint32_t s2mm_stat = dma_reg_read(h->dma_base, DMA_S2MM_STATUS);
        if ((mm2s_stat & 0x2) && (s2mm_stat & 0x2)) return 0;

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        long elapsed_us = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000L
                        + (ts_now.tv_nsec - ts_start.tv_nsec) / 1000L;
        if (elapsed_us > timeout_us) return -1;
    }
}

int dma_read_corrections(dma_handle_t *h, void *out_buf, size_t n_bytes) {
    dma_memory_barrier();
    memcpy(out_buf, h->correction_buf, n_bytes);
    return 0;
}
