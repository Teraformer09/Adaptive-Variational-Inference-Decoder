/* ===========================================================================
 * dma_ingress_tap.h — 4-Slot DMA Ring Ownership FSM & Syndrome Tap (A05 / Q11)
 * Enforces `FREE -> DMA_OWNED -> CPU_READY -> CPU_OWNED -> FREE`.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "packet_validator.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AVID_DMA_NUM_SLOTS       4
#define AVID_DMA_SLOT_PACKETS    64

typedef enum {
    AVID_SLOT_FREE      = 0,
    AVID_SLOT_DMA_OWNED = 1,
    AVID_SLOT_CPU_READY = 2,
    AVID_SLOT_CPU_OWNED = 3
} avid_slot_state_t;

typedef struct {
    uint64_t          packets[AVID_DMA_SLOT_PACKETS];
    uint32_t          count;
    avid_slot_state_t state;
} avid_dma_slot_t;

typedef struct {
    avid_dma_slot_t         slots[AVID_DMA_NUM_SLOTS];
    int                     write_slot_idx;
    int                     read_slot_idx;
    uint64_t                overflow_drops;
    avid_packet_validator_t validator;
    /* Synchronized epoch-fencing & atomic snapshot extensions (C11/C++17 layout-identical) */
    uint8_t                 spinlock_flag;
    bool                    has_active_epoch;
    uint16_t                active_epoch_id;
    uint64_t                stale_epoch_drops;
    uint32_t                accum_hist_x[16];
    uint32_t                accum_hist_z[16];
} avid_dma_ring_t;

void avid_dma_ring_init(avid_dma_ring_t *ring);

/* Optional epoch fence: rejects any packet with pkt.epoch_id < epoch_id */
void avid_dma_ring_set_active_epoch(avid_dma_ring_t *ring, uint16_t epoch_id);

/* Producer (PL DMA emulation / hardware tap): acquire FREE slot, write burst, mark CPU_READY */
bool avid_dma_ring_push_burst(avid_dma_ring_t *ring, const uint64_t *pkts, uint32_t n_pkts);

/* Consumer (ARM estimator): claim CPU_READY slot -> CPU_OWNED, accumulate 16-bin X/Z histograms, release -> FREE */
bool avid_dma_ring_consume_slot(avid_dma_ring_t *ring,
                                uint32_t hist_x[16],
                                uint32_t hist_z[16],
                                uint32_t *out_valid_packets);

/* Thread-safe synchronized snapshot and zero-reset of internal 16-bin X/Z histograms */
bool avid_dma_ring_snapshot_and_reset(avid_dma_ring_t *ring,
                                      uint32_t out_hist_x[16],
                                      uint32_t out_hist_z[16]);

#ifdef __cplusplus
}
#endif
