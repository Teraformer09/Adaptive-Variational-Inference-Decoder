/* ===========================================================================
 * packet_validator.cpp — Canonical 64-Bit Syndrome Packet Validator (A05 / Q11)
 * =========================================================================*/
#include "packet_validator.h"
#include <string.h>

void avid_validator_init(avid_packet_validator_t *v) {
    memset(v, 0, sizeof(*v));
}

uint64_t avid_pack_packet_u64(uint8_t s_x, uint8_t s_z, uint32_t seq_id,
                              uint16_t epoch_id, uint8_t flags) {
    uint64_t w = 0;
    w |= (uint64_t)(s_x & 0x0Fu);
    w |= (uint64_t)(s_z & 0x0Fu) << 4;
    w |= (uint64_t)(flags & 0xFFu) << 8;
    w |= (uint64_t)(epoch_id & 0xFFFFu) << 16;
    w |= (uint64_t)seq_id << 32;
    return w;
}

void avid_unpack_packet_u64(uint64_t raw_word, avid_syndrome_packet_t *out) {
    out->s_x      = (uint8_t)(raw_word & 0x0Fu);
    out->s_z      = (uint8_t)((raw_word >> 4) & 0x0Fu);
    out->flags    = (uint8_t)((raw_word >> 8) & 0xFFu);
    out->epoch_id = (uint16_t)((raw_word >> 16) & 0xFFFFu);
    out->seq_id   = (uint32_t)((raw_word >> 32) & 0xFFFFFFFFu);
}

bool avid_validate_packet(avid_packet_validator_t *v, uint64_t raw_word,
                          avid_syndrome_packet_t *out) {
    avid_unpack_packet_u64(raw_word, out);

    if ((out->flags & AVID_PKT_FLAG_VALID) == 0 ||
        (out->flags & AVID_PKT_FLAG_PARITY_ERR) != 0) {
        v->invalid_flag_count++;
        return false;
    }

    if (out->flags & AVID_PKT_FLAG_OVERFLOW) {
        v->overflow_warn_count++;
    }

    if (v->has_seen_first) {
        uint32_t expected = v->last_seq_id + 1u;
        if (out->seq_id != expected) {
            v->seq_gap_count++;
            v->last_seq_id = out->seq_id;
            return false;
        }
    }
    v->has_seen_first = true;
    v->last_seq_id = out->seq_id;
    v->valid_count++;
    return true;
}
