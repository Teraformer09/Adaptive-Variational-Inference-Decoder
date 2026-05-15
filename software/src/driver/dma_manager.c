/* ===========================================================================
 * dma_manager.c — AVID Phase 2.2
 * Xilinx AXI DMA Simple Mode user-space driver.
 * =========================================================================*/
#include "dma_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>

static inline void dma_reg_write(void *base, uint32_t off, uint32_t val) {
    volatile uint32_t *r = (volatile uint32_t *)((uint8_t *)base + off);
    *r = val;
    __asm__ volatile("dsb sy" ::: "memory");
}
static inline uint32_t dma_reg_read(void *base, uint32_t off) {
    volatile uint32_t *r = (volatile uint32_t *)((uint8_t *)base + off);
    uint32_t v = *r;
    __asm__ volatile("dsb sy" ::: "memory");
    return v;
}

int dma_open(dma_handle_t *h, uint32_t dma_base_addr) {
    h->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (h->fd_mem < 0) { perror("[dma] open /dev/mem"); return -1; }

    h->dma_base_addr = dma_base_addr;
    h->dma_base = mmap(NULL, DMA_MAP_SIZE, PROT_READ|PROT_WRITE,
                       MAP_SHARED, h->fd_mem, (off_t)dma_base_addr);
    if (h->dma_base == MAP_FAILED) { perror("[dma] mmap DMA regs"); goto err; }

    /* Map coherent syndrome buffer (DDR) */
    h->syndrome_buf = mmap(NULL, DDR_BUFFER_SIZE, PROT_READ|PROT_WRITE,
                           MAP_SHARED, h->fd_mem, (off_t)DDR_SYNDROME_PHYS);
    if (h->syndrome_buf == MAP_FAILED) { perror("[dma] mmap syndrome buf"); goto err; }

    /* Map coherent correction buffer */
    h->correction_buf = mmap(NULL, DDR_BUFFER_SIZE, PROT_READ|PROT_WRITE,
                              MAP_SHARED, h->fd_mem, (off_t)DDR_CORRECT_PHYS);
    if (h->correction_buf == MAP_FAILED) { perror("[dma] mmap correction buf"); goto err; }

    h->max_syndrome_bytes = DDR_BUFFER_SIZE;

    /* Reset DMA engines */
    dma_reg_write(h->dma_base, DMA_MM2S_CONTROL, DMA_CTRL_RESET);
    dma_reg_write(h->dma_base, DMA_S2MM_CONTROL, DMA_CTRL_RESET);
    usleep(10);
    dma_reg_write(h->dma_base, DMA_MM2S_CONTROL, DMA_CTRL_RUN | DMA_CTRL_IOC_IRQEN);
    dma_reg_write(h->dma_base, DMA_S2MM_CONTROL, DMA_CTRL_RUN | DMA_CTRL_IOC_IRQEN);

    printf("[dma] Initialised @ 0x%08X, syndrome buf @ 0x%08X\n",
           dma_base_addr, DDR_SYNDROME_PHYS);
    return 0;

err:
    dma_close(h);
    return -1;
}

void dma_close(dma_handle_t *h) {
    if (h->syndrome_buf  && h->syndrome_buf  != MAP_FAILED) munmap(h->syndrome_buf,  DDR_BUFFER_SIZE);
    if (h->correction_buf && h->correction_buf!= MAP_FAILED) munmap(h->correction_buf,DDR_BUFFER_SIZE);
    if (h->dma_base      && h->dma_base      != MAP_FAILED) munmap(h->dma_base,      DMA_MAP_SIZE);
    if (h->fd_mem >= 0) close(h->fd_mem);
    memset(h, 0, sizeof(*h));
}

int dma_load_syndromes(dma_handle_t *h, const void *data, size_t n_bytes) {
    if (n_bytes > h->max_syndrome_bytes) {
        fprintf(stderr, "[dma] syndrome data %zu > buffer %zu\n", n_bytes, h->max_syndrome_bytes);
        return -1;
    }
    /* ARM CPU copies syndrome data into the coherent DDR buffer.
     * After this point, the DMA engine owns the buffer — do not touch. */
    memcpy(h->syndrome_buf, data, n_bytes);
    __asm__ volatile("dsb sy" ::: "memory");  /* Flush to DRAM before DMA */
    return 0;
}

int dma_start_stream(dma_handle_t *h, size_t syndrome_bytes, size_t correction_bytes) {
    /* S2MM: arm receive first (DMA needs a destination before MM2S sends) */
    dma_reg_write(h->dma_base, DMA_S2MM_DST_ADDR, DDR_CORRECT_PHYS);
    dma_reg_write(h->dma_base, DMA_S2MM_LENGTH,   (uint32_t)correction_bytes);

    /* MM2S: start transmission — syndromes flow to FPGA via S_AXI_HP0 */
    dma_reg_write(h->dma_base, DMA_MM2S_SRC_ADDR, DDR_SYNDROME_PHYS);
    dma_reg_write(h->dma_base, DMA_MM2S_LENGTH,   (uint32_t)syndrome_bytes);
    /* Writing LENGTH kicks off the transfer */
    return 0;
}

int dma_wait_complete(dma_handle_t *h, int timeout_us) {
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    while (1) {
        uint32_t mm2s_stat = dma_reg_read(h->dma_base, DMA_MM2S_STATUS);
        uint32_t s2mm_stat = dma_reg_read(h->dma_base, DMA_S2MM_STATUS);

        /* Bit 1 = Idle in both channels */
        if ((mm2s_stat & 0x2) && (s2mm_stat & 0x2)) return 0;

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        long elapsed_us = (ts_now.tv_sec  - ts_start.tv_sec)  * 1000000L
                        + (ts_now.tv_nsec - ts_start.tv_nsec) / 1000L;
        if (elapsed_us > timeout_us) {
            fprintf(stderr, "[dma] timeout after %d us (MM2S=0x%08X S2MM=0x%08X)\n",
                    timeout_us, mm2s_stat, s2mm_stat);
            return -1;
        }
    }
}

int dma_read_corrections(dma_handle_t *h, void *out_buf, size_t n_bytes) {
    __asm__ volatile("dsb sy" ::: "memory");  /* Invalidate cache line */
    memcpy(out_buf, h->correction_buf, n_bytes);
    return 0;
}
