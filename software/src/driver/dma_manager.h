/* ===========================================================================
 * dma_manager.h — AVID Phase 2.2
 * User-space DMA descriptor management for the Xilinx AXI DMA IP.
 * Feeds pre-generated syndrome binary blobs from DDR to the FPGA S_AXI_HP0
 * port (syndrome stream), and receives correction data back (S2MM).
 *
 * The ARM CPU NEVER touches the raw syndrome stream data after DMA starts.
 * This satisfies COMMANDMENT #4.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stddef.h>

/* Xilinx AXI DMA register offsets (Simple DMA, no Scatter-Gather) */
#define DMA_MM2S_CONTROL   0x00U
#define DMA_MM2S_STATUS    0x04U
#define DMA_MM2S_SRC_ADDR  0x18U
#define DMA_MM2S_LENGTH    0x28U
#define DMA_S2MM_CONTROL   0x30U
#define DMA_S2MM_STATUS    0x34U
#define DMA_S2MM_DST_ADDR  0x48U
#define DMA_S2MM_LENGTH    0x58U

#define DMA_CTRL_RUN       0x01U
#define DMA_CTRL_RESET     0x04U
#define DMA_CTRL_IOC_IRQEN 0x1000U

/* Physical addresses — adjust to your block design */
#define DMA_BASE_ADDR      0x40000000UL
#define DMA_MAP_SIZE       0x10000UL
#define DDR_SYNDROME_PHYS  0x10000000UL   /* Pre-allocated coherent buffer */
#define DDR_CORRECT_PHYS   0x10100000UL
#define DDR_BUFFER_SIZE    0x100000UL     /* 1 MB per buffer */

typedef struct {
    int       fd_mem;
    void     *dma_base;        /* mmap of DMA registers */
    void     *syndrome_buf;    /* mmap of DDR syndrome buffer (write-only) */
    void     *correction_buf;  /* mmap of DDR correction buffer (read-only) */
    uint32_t  dma_base_addr;
    size_t    max_syndrome_bytes;
} dma_handle_t;

int  dma_open(dma_handle_t *h, uint32_t dma_base_addr);
void dma_close(dma_handle_t *h);

/**
 * dma_load_syndromes() — Copy syndrome binary blob into the DMA source buffer.
 * @data: pointer to the syndrome binary (from stim_to_bin.py output)
 * @n_bytes: number of bytes (2 bytes per syndrome round)
 */
int  dma_load_syndromes(dma_handle_t *h, const void *data, size_t n_bytes);

/**
 * dma_start_stream() — Arm both MM2S (syndrome → FPGA) and S2MM (FPGA → correction).
 * Non-blocking: returns immediately. Use dma_wait_complete() to poll.
 */
int  dma_start_stream(dma_handle_t *h, size_t syndrome_bytes, size_t correction_bytes);

/**
 * dma_wait_complete() — Poll DMA status until both channels complete.
 * @timeout_us: maximum wait in microseconds. Returns 0 on success, -1 on timeout.
 */
int  dma_wait_complete(dma_handle_t *h, int timeout_us);

/**
 * dma_read_corrections() — Copy correction data out of the S2MM receive buffer.
 */
int  dma_read_corrections(dma_handle_t *h, void *out_buf, size_t n_bytes);
