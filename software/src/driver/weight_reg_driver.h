/* ===========================================================================
 * weight_reg_driver.h — RTL-Aligned AXI-Lite Driver for `weight_map_reg.sv`
 * Policy Items: SF5, A09 / Q18, DEF-L4-03
 *
 * Authoritative Register Layout (matches `weight_map_reg.sv` bit-for-bit):
 *   0x000 + 4*i : shadow_x[i]     (R/W, signed 16-bit Q7.8 LLR, i = 0..8)
 *   0x024 + 4*i : shadow_z[i]     (R/W, signed 16-bit Q7.8 LLR, i = 0..8)
 *   0x048       : COMMIT          (R/W, bit 0 = swap_pending; write 1 commits)
 *   0x04C       : STATUS          (RO,  [0]=swap_pending, [1]=bp_busy)
 *   0x050       : EPOCH_COUNT     (RO,  32-bit committed epoch counter)
 *   0x054       : CFG_MAX_ITER    (R/W, supported K in 1..5, default 3)
 *   0x058       : CFG_OSD_MODE    (R/W, supported mode in 0..2, default 1)
 *   0x100 + 4*i : active_x[i]     (RO,  signed 16-bit Q7.8 LLR, i = 0..8)
 *   0x124 + 4*i : active_z[i]     (RO,  signed 16-bit Q7.8 LLR, i = 0..8)
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WRD_BASE_ADDR       0xA0000000UL
#define WRD_MAP_SIZE        0x1000UL
#define WRD_N_DATA          9
#define WRD_NUM_QUBITS      9
#define WRD_DEFAULT_Q78     0x0232

#define WRD_OFF_SHADOW_X(i) (0x000U + (uint32_t)(i) * 4U)
#define WRD_OFF_SHADOW_Z(i) (0x024U + (uint32_t)(i) * 4U)
#define WRD_OFF_SHADOW(i)   WRD_OFF_SHADOW_X(i)
#define WRD_OFF_X(i)        WRD_OFF_SHADOW_X(i)
#define WRD_OFF_Z(i)        WRD_OFF_SHADOW_Z(i)
#define WRD_OFF_COMMIT      0x048U
#define WRD_OFF_STATUS      0x04CU
#define WRD_OFF_EPOCH       0x050U
#define WRD_OFF_MAX_ITER    0x054U
#define WRD_OFF_OSD_MODE    0x058U
#define WRD_OFF_ACTIVE_X(i) (0x100U + (uint32_t)(i) * 4U)
#define WRD_OFF_ACTIVE_Z(i) (0x124U + (uint32_t)(i) * 4U)
#define WRD_OFF_ACTIVE(i)   WRD_OFF_ACTIVE_X(i)

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
    int16_t  shadow_x[WRD_N_DATA];
    int16_t  shadow_z[WRD_N_DATA];
    int16_t  commit_snap_x[WRD_N_DATA];
    int16_t  commit_snap_z[WRD_N_DATA];
    int16_t  active_x[WRD_N_DATA];
    int16_t  active_z[WRD_N_DATA];
    uint8_t  swap_pending;
    uint8_t  bp_busy;
    uint32_t epoch_count;
    uint32_t cfg_max_iter;
    uint32_t cfg_osd_mode;
    int      is_emulated;
} wrd_handle_t;

int      wrd_open(wrd_handle_t *h, uint32_t base_addr);
void     wrd_close(wrd_handle_t *h);
int      wrd_mmio_write_checked(wrd_handle_t *h, uint32_t offset, uint32_t data, uint8_t byte_enable);
int      wrd_mmio_read_checked(const wrd_handle_t *h, uint32_t offset, uint32_t *rdata);
void     wrd_write_x_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val);
void     wrd_write_z_llr(wrd_handle_t *h, int qubit_idx, int16_t q78_val);
void     wrd_write_all(wrd_handle_t *h, const float *llr_x, const float *llr_z);
void     wrd_write_all_weights(wrd_handle_t *h, const int16_t weights_q78[WRD_NUM_QUBITS]);
int16_t  wrd_read_active_weight(const wrd_handle_t *h, int qubit_idx);
int16_t  wrd_read_active_x_weight(const wrd_handle_t *h, int qubit_idx);
int16_t  wrd_read_active_z_weight(const wrd_handle_t *h, int qubit_idx);
void     wrd_commit(wrd_handle_t *h);
int      wrd_commit_and_wait(wrd_handle_t *h, int max_polls);
uint32_t wrd_read_epoch(const wrd_handle_t *h);
uint16_t wrd_get_swap_count(const wrd_handle_t *h);
uint32_t wrd_read_status(const wrd_handle_t *h);

#ifdef __cplusplus
}
#endif
