/* ===========================================================================
 * dma_ingress_tap.cpp — 4-Slot DMA Ring Ownership FSM Implementation (A05 / Q11)
 * =========================================================================*/
#include "dma_ingress_tap.h"
#include <string.h>

void avid_dma_ring_init(avid_dma_ring_t *ring) {
    memset(ring, 0, sizeof(*ring));
    for (int i = 0; i < AVID_DMA_NUM_SLOTS; i++) {
        ring->slots[i].state = AVID_SLOT_FREE;
        ring->slots[i].count = 0;
    }
    avid_validator_init(&ring->validator);
}

bool avid_dma_ring_push_burst(avid_dma_ring_t *ring, const uint64_t *pkts, uint32_t n_pkts) {
    if (n_pkts > AVID_DMA_SLOT_PACKETS) {
        n_pkts = AVID_DMA_SLOT_PACKETS;
    }
    avid_dma_slot_t *slot = &ring->slots[ring->write_slot_idx];
    if (slot->state != AVID_SLOT_FREE) {
        ring->overflow_drops += n_pkts;
        return false;
    }
    /* Transition FREE -> DMA_OWNED */
    slot->state = AVID_SLOT_DMA_OWNED;
    memcpy(slot->packets, pkts, n_pkts * sizeof(uint64_t));
    slot->count = n_pkts;

    /* Transition DMA_OWNED -> CPU_READY */
    slot->state = AVID_SLOT_CPU_READY;
    ring->write_slot_idx = (ring->write_slot_idx + 1) % AVID_DMA_NUM_SLOTS;
    return true;
}

bool avid_dma_ring_consume_slot(avid_dma_ring_t *ring,
                                uint32_t hist_x[16],
                                uint32_t hist_z[16],
                                uint32_t *out_valid_packets) {
    avid_dma_slot_t *slot = &ring->slots[ring->read_slot_idx];
    if (slot->state != AVID_SLOT_CPU_READY) {
        if (out_valid_packets) *out_valid_packets = 0;
        return false;
    }
    /* Transition CPU_READY -> CPU_OWNED */
    slot->state = AVID_SLOT_CPU_OWNED;

    uint32_t valid_cnt = 0;
    for (uint32_t i = 0; i < slot->count; i++) {
        avid_syndrome_packet_t pkt;
        if (avid_validate_packet(&ring->validator, slot->packets[i], &pkt)) {
            hist_x[pkt.s_x & 0x0Fu]++;
            hist_z[pkt.s_z & 0x0Fu]++;
            valid_cnt++;
        }
    }

    /* Transition CPU_OWNED -> FREE */
    slot->count = 0;
    slot->state = AVID_SLOT_FREE;
    ring->read_slot_idx = (ring->read_slot_idx + 1) % AVID_DMA_NUM_SLOTS;
    if (out_valid_packets) *out_valid_packets = valid_cnt;
    return true;
}
