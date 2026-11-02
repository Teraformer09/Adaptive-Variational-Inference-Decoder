/* ===========================================================================
 * weight_reg_driver.h — Repaired AXI-Lite Driver for `weight_map_reg.sv`
 * Policy Items: SF5, A09 / Q18
 *
 * Register layout (matches `weight_map_reg.sv` bit-for-bit):
 *   0x000 + 4*i : shadow_mem[i]   (R/W, signed 16-bit Q7.8 LLR, i = 0..8)
 *   0x024 + 4*i : active_mem[i]   (RO,  signed 16-bit Q7.8 LLR, i = 0..8)
 *   0x048       : COMMIT register (WO,  bit 0 = 1 schedules atomic swap)
 *   0x04C       : STATUS register (RO,  [0]=commit_pending, [1]=bp_ready_in, [31:16]=swap_count)
 * =========================================================================*/
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WRD_BASE_ADDR    0xA0000000UL
#define WRD_MAP_SIZE     0x1000UL
#define WRD_N_DATA       9
#define WRD_NUM_QUBITS   9

#define WRD_OFF_SHADOW(i) (0x000U + (uint32_t)(i) * 4U)
#define WRD_OFF_ACTIVE(i) (0x024U + (uint32_t)(i) * 4U)
#define WRD_OFF_X(i)      WRD_OFF_SHADOW(i)
#define WRD_OFF_Z(i)      WRD_OFF_ACTIVE(i)
#define WRD_OFF_COMMIT    0x048U
#define WRD_OFF_STATUS    0x04CU

static inline int16_t wrd_float_to_q78(float llr) {
    float scaled = llr * 256.0f;
    if (scaled >  32767.0f) return  32767;
    if (scaled < -32768.0f) return -32768;
    return (int16_t)scaled;
}

typedef struct {
    int      fd;
    void    *mapped_base;
    uint32_t base_addr;
    uint32_t sim_regs[32]; /* Emulated register file when /dev/mem is not mapped */
    int      is_emulated;
} wrd_handle_t;

int      wrd_open(wrd_handle_t *h, uint32_t base_addr);
void     wrd_close(wrd_handle_t *h);
void     wrd_write_x_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val);
void     wrd_write_z_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val);
void     wrd_write_all(wrd_handle_t *h, const float *llr_x, const float *llr_z);
void     wrd_write_all_weights(wrd_handle_t *h, const int16_t weights_q78[WRD_NUM_QUBITS]);
int16_t  wrd_read_active_weight(const wrd_handle_t *h, int qubit_idx);
void     wrd_commit(wrd_handle_t *h);
int      wrd_commit_and_wait(wrd_handle_t *h, int max_polls);
uint32_t wrd_read_epoch(const wrd_handle_t *h);
uint16_t wrd_get_swap_count(const wrd_handle_t *h);
uint32_t wrd_read_status(const wrd_handle_t *h);

#ifdef __cplusplus
}
#endif
