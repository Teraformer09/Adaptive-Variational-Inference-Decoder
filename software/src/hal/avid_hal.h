/* ===========================================================================
 * avid_hal.h — AVID Hardware Abstraction Layer (HAL)
 * Separates the AVID daemon & controller from direct /dev/mem access:
 *   - physical_backend   : Hardware MMIO register interface (reserved for V2)
 *   - simulation_backend : Hardened Unix-domain socket transport to C++ Bridge
 * =========================================================================*/
#pragma once

#include <cstdint>
#include <cstddef>
#include "../driver/weight_reg_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AVID_HAL_BACKEND_PHYSICAL   = 0,
    AVID_HAL_BACKEND_SIMULATION = 1
} avid_hal_backend_type_t;

typedef struct {
    uint64_t total_transactions;
    uint64_t mmio_writes;
    uint64_t mmio_reads;
    uint64_t syndrome_packets;
    uint64_t commit_operations;
    uint64_t reset_operations;
    uint64_t rejected_transactions;
    uint64_t timeout_transactions;
    uint64_t id_mismatch_errors;
    uint64_t reconnect_events;
    uint64_t last_accepted_cycle;
    uint32_t last_axis_total_cycles;
} avid_hal_stats_t;

typedef struct {
    avid_hal_backend_type_t backend;
    int                     sock_fd;
    char                    socket_path[256];
    uint64_t                next_tx_id;
    uint32_t                timeout_ms;
    wrd_handle_t            phys_wrd;
    avid_hal_stats_t        stats;
} avid_hal_t;

int  avid_hal_open(avid_hal_t *hal,
                   avid_hal_backend_type_t backend,
                   const char *socket_path,
                   uint32_t phys_base_addr);

void avid_hal_close(avid_hal_t *hal);

int  avid_hal_reconnect(avid_hal_t *hal);

int  avid_hal_set_timeout_ms(avid_hal_t *hal, uint32_t timeout_ms);

int  avid_hal_mmio_write(avid_hal_t *hal,
                         uint32_t offset,
                         uint32_t data,
                         uint8_t byte_enable,
                         uint64_t *accepted_cycle);

int  avid_hal_mmio_read(avid_hal_t *hal,
                        uint32_t offset,
                        uint32_t *rdata,
                        uint64_t *accepted_cycle);

int  avid_hal_write_weights_and_commit(avid_hal_t *hal,
                                       const int16_t q78_x[9],
                                       const int16_t q78_z[9],
                                       uint32_t *epoch_out,
                                       uint64_t *accepted_cycle);

int  avid_hal_send_syndrome(avid_hal_t *hal,
                            uint8_t sx,
                            uint8_t sz,
                            bool tlast,
                            uint16_t *corr_x,
                            uint16_t *corr_z,
                            uint32_t *bp_core_cycles,
                            uint64_t *accepted_cycle);

int  avid_hal_reset_rtl(avid_hal_t *hal,
                        uint32_t reset_cycles,
                        uint64_t *accepted_cycle);

int  avid_hal_set_bp_iterations(avid_hal_t *hal,
                                uint32_t k_iter,
                                uint64_t *accepted_cycle);

int  avid_hal_set_osd_mode(avid_hal_t *hal,
                           uint32_t osd_mode,
                           uint64_t *accepted_cycle);

#ifdef __cplusplus
}
#endif
