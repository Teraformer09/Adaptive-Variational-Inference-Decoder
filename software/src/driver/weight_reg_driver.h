/* ===========================================================================
 * weight_reg_driver.h — AVID Phase 2.1
 * User-space AXI-Lite driver for the weight_map_reg module.
 * Maps /dev/mem → physical base address of the weight register block.
 *
 * Physical base address (from Vivado block design address editor):
 *   #define WRD_BASE_ADDR  0x43C01000UL   (adjust to your block design)
 *
 * Register layout (matches weight_map_reg.sv):
 *   0x000 + 4*i : Shadow X-LLR[i]  Q7.8 fixed-point  (i = 0..N_DATA-1)
 *   0x024 + 4*i : Shadow Z-LLR[i]
 *   0x048       : COMMIT register  (write 1 to schedule swap)
 *   0x04C       : STATUS register  (read-only: [0]=swap_pending, [1]=bp_busy)
 *   0x050       : EPOCH_COUNT      (read-only: # completed weight epochs)
 * =========================================================================*/
#pragma once
#include <stdint.h>

#define WRD_BASE_ADDR   0x43C01000UL   /* FPGA AXI-Lite base — override per board */
#define WRD_MAP_SIZE    0x1000UL       /* 4 KB mapping */

#define WRD_N_DATA      9              /* d=3: 9 qubits per Pauli type */

#define WRD_OFF_X(i)    (0x000U + (uint32_t)(i) * 4U)
#define WRD_OFF_Z(i)    (0x024U + (uint32_t)(i) * 4U)
#define WRD_OFF_COMMIT  0x048U
#define WRD_OFF_STATUS  0x04CU
#define WRD_OFF_EPOCH   0x050U

/* Q7.8 conversion: float LLR → int16_t */
static inline int16_t wrd_float_to_q78(float llr) {
    float scaled = llr * 256.0f;
    if (scaled >  32767.0f) return  32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)scaled;
}

/* Opaque driver handle */
typedef struct {
    int      fd;
    void    *mapped_base;
    uint32_t base_addr;
} wrd_handle_t;

/* ---------- API ---------- */

/**
 * wrd_open() — Map the weight register block into user-space.
 * @base_addr: Physical address of AXI-Lite base (WRD_BASE_ADDR).
 * Returns 0 on success, -1 on error (check errno).
 */
int wrd_open(wrd_handle_t *h, uint32_t base_addr);

/**
 * wrd_close() — Unmap and close /dev/mem.
 */
void wrd_close(wrd_handle_t *h);

/**
 * wrd_write_x_llr() — Write one X-type channel LLR to the shadow bank.
 * @qubit_idx: 0..WRD_N_DATA-1
 * @q78_val:   Q7.8 fixed-point LLR value
 */
void wrd_write_x_llr(const wrd_handle_t *h, int qubit_idx, int16_t q78_val);

/**
 * wrd_write_z_llr() — Write one Z-type channel LLR to the shadow bank.
 */
void wrd_write_z_llr(const wrd_handle_t *h, int qubit_idx, int16_t q78_val);

/**
 * wrd_write_all() — Write all N_DATA X and Z LLRs in one burst.
 * @llr_x: array of WRD_N_DATA float LLRs for X-type
 * @llr_z: array of WRD_N_DATA float LLRs for Z-type
 */
void wrd_write_all(const wrd_handle_t *h, const float *llr_x, const float *llr_z);

/**
 * wrd_commit() — Schedule atomic swap of shadow → active bank.
 * The FPGA will execute the swap at the next bp_ready=1 boundary.
 * MUST be called after wrd_write_all() to activate new weights.
 */
void wrd_commit(const wrd_handle_t *h);

/**
 * wrd_read_epoch() — Read the epoch counter (# completed weight swaps).
 */
uint32_t wrd_read_epoch(const wrd_handle_t *h);

/**
 * wrd_read_status() — Read status register.
 * Returns bitfield: [0]=swap_pending, [1]=bp_busy
 */
uint32_t wrd_read_status(const wrd_handle_t *h);
