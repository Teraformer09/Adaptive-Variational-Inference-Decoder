/* ===========================================================================
 * dma_ingress_tap.cpp — 4-Slot DMA Ring Ownership FSM Implementation (A05 / Q11)
 * =========================================================================*/
#include "dma_ingress_tap.h"
#include <limits.h>
#include <string.h>

static inline void avid_dma_lock(avid_dma_ring_t *ring) {
    while (__atomic_test_and_set(&ring->spinlock_flag, __ATOMIC_ACQUIRE)) {
        /* spin */
    }
}

static inline void avid_dma_unlock(avid_dma_ring_t *ring) {
    __atomic_clear(&ring->spinlock_flag, __ATOMIC_RELEASE);
}

void avid_dma_ring_init(avid_dma_ring_t *ring) {
    if (!ring) {
        return;
    }
    memset(ring, 0, sizeof(*ring));
    for (int i = 0; i < AVID_DMA_NUM_SLOTS; i++) {
        __atomic_store_n(&ring->slots[i].state, AVID_SLOT_FREE, __ATOMIC_RELEASE);
        ring->slots[i].count = 0;
    }
    avid_validator_init(&ring->validator);
}

void avid_dma_ring_set_active_epoch(avid_dma_ring_t *ring, uint16_t epoch_id) {
    if (!ring) {
        return;
    }
    avid_dma_lock(ring);
    ring->has_active_epoch = true;
    ring->active_epoch_id  = epoch_id;
    avid_dma_unlock(ring);
}

bool avid_dma_ring_push_burst(avid_dma_ring_t *ring, const uint64_t *pkts, uint32_t n_pkts) {
    if (!ring || !pkts || n_pkts == 0u) {
        return false;
    }

    avid_dma_lock(ring);

    if (n_pkts > AVID_DMA_SLOT_PACKETS) {
        ring->overflow_drops += n_pkts;
        avid_dma_unlock(ring);
        return false;
    }

    avid_dma_slot_t *slot = &ring->slots[ring->write_slot_idx];
    avid_slot_state_t cur_state = __atomic_load_n(&slot->state, __ATOMIC_ACQUIRE);
    if (cur_state != AVID_SLOT_FREE) {
        ring->overflow_drops += n_pkts;
        avid_dma_unlock(ring);
        return false;
    }

    /* Transition FREE -> DMA_OWNED */
    __atomic_store_n(&slot->state, AVID_SLOT_DMA_OWNED, __ATOMIC_RELEASE);
    memcpy(slot->packets, pkts, n_pkts * sizeof(uint64_t));
    slot->count = n_pkts;

    /* Transition DMA_OWNED -> CPU_READY */
    __atomic_store_n(&slot->state, AVID_SLOT_CPU_READY, __ATOMIC_RELEASE);
    ring->write_slot_idx = (ring->write_slot_idx + 1) % AVID_DMA_NUM_SLOTS;
    avid_dma_unlock(ring);
    return true;
}

bool avid_dma_ring_consume_slot(avid_dma_ring_t *ring,
                                uint32_t hist_x[16],
                                uint32_t hist_z[16],
                                uint32_t *out_valid_packets) {
    if (!ring) {
        if (out_valid_packets) {
            *out_valid_packets = 0;
        }
        return false;
    }

    avid_dma_lock(ring);

    avid_dma_slot_t *slot = &ring->slots[ring->read_slot_idx];
    avid_slot_state_t cur_state = __atomic_load_n(&slot->state, __ATOMIC_ACQUIRE);
    if (cur_state != AVID_SLOT_CPU_READY) {
        if (out_valid_packets) {
            *out_valid_packets = 0;
        }
        avid_dma_unlock(ring);
        return false;
    }

    /* Transition CPU_READY -> CPU_OWNED */
    __atomic_store_n(&slot->state, AVID_SLOT_CPU_OWNED, __ATOMIC_RELEASE);

    uint32_t valid_cnt = 0;
    for (uint32_t i = 0; i < slot->count; i++) {
        avid_syndrome_packet_t pkt;
        if (avid_validate_packet(&ring->validator, slot->packets[i], &pkt)) {
            if (ring->has_active_epoch && pkt.epoch_id < ring->active_epoch_id) {
                ring->stale_epoch_drops++;
                continue;
            }
            uint8_t bx = (uint8_t)(pkt.s_x & 0x0Fu);
            uint8_t bz = (uint8_t)(pkt.s_z & 0x0Fu);
            if (hist_x && hist_x[bx] < UINT32_MAX) {
                hist_x[bx]++;
            }
            if (hist_z && hist_z[bz] < UINT32_MAX) {
                hist_z[bz]++;
            }
            if (ring->accum_hist_x[bx] < UINT32_MAX) {
                ring->accum_hist_x[bx]++;
            }
            if (ring->accum_hist_z[bz] < UINT32_MAX) {
                ring->accum_hist_z[bz]++;
            }
            valid_cnt++;
        }
    }

    /* Transition CPU_OWNED -> FREE */
    slot->count = 0;
    __atomic_store_n(&slot->state, AVID_SLOT_FREE, __ATOMIC_RELEASE);
    ring->read_slot_idx = (ring->read_slot_idx + 1) % AVID_DMA_NUM_SLOTS;
    if (out_valid_packets) {
        *out_valid_packets = valid_cnt;
    }
    avid_dma_unlock(ring);
    return true;
}

bool avid_dma_ring_snapshot_and_reset(avid_dma_ring_t *ring,
                                      uint32_t out_hist_x[16],
                                      uint32_t out_hist_z[16]) {
    if (!ring || !out_hist_x || !out_hist_z) {
        return false;
    }
    avid_dma_lock(ring);
    memcpy(out_hist_x, ring->accum_hist_x, 16u * sizeof(uint32_t));
    memcpy(out_hist_z, ring->accum_hist_z, 16u * sizeof(uint32_t));
    memset(ring->accum_hist_x, 0, 16u * sizeof(uint32_t));
    memset(ring->accum_hist_z, 0, 16u * sizeof(uint32_t));
    avid_dma_unlock(ring);
    return true;
}
