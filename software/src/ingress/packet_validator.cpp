/* ===========================================================================
 * packet_validator.cpp — Canonical 64-Bit Syndrome Packet Validator (A05 / Q11)
 * =========================================================================*/
#include "packet_validator.h"
#include <string.h>

void avid_validator_init(avid_packet_validator_t *v) {
    if (!v) {
        return;
    }
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

bool avid_pack_packet_u64_checked(uint8_t s_x, uint8_t s_z, uint32_t seq_id,
                                  uint16_t epoch_id, uint8_t flags,
                                  uint64_t *out_word) {
    if (!out_word) {
        return false;
    }
    if (s_x > AVID_PKT_MAX_SYNDROME || s_z > AVID_PKT_MAX_SYNDROME) {
        return false;
    }
    if ((flags & AVID_PKT_FLAG_RESERVED_MASK) != 0u) {
        return false;
    }
    *out_word = avid_pack_packet_u64(s_x, s_z, seq_id, epoch_id, flags);
    return true;
}

bool avid_encode_packet_le64(uint64_t raw_word, uint8_t *out_bytes, size_t out_len) {
    if (!out_bytes || out_len != AVID_PKT_WIRE_BYTES) {
        return false;
    }
    for (size_t i = 0; i < AVID_PKT_WIRE_BYTES; ++i) {
        out_bytes[i] = (uint8_t)((raw_word >> (8u * i)) & 0xFFu);
    }
    return true;
}

bool avid_decode_packet_le64(const uint8_t *bytes, size_t len, uint64_t *out_word) {
    if (!bytes || !out_word || len != AVID_PKT_WIRE_BYTES) {
        return false;
    }
    uint64_t w = 0;
    for (size_t i = 0; i < AVID_PKT_WIRE_BYTES; ++i) {
        w |= ((uint64_t)bytes[i]) << (8u * i);
    }
    *out_word = w;
    return true;
}

void avid_unpack_packet_u64(uint64_t raw_word, avid_syndrome_packet_t *out) {
    if (!out) {
        return;
    }
    out->s_x      = (uint8_t)(raw_word & 0x0Fu);
    out->s_z      = (uint8_t)((raw_word >> 4) & 0x0Fu);
    out->flags    = (uint8_t)((raw_word >> 8) & 0xFFu);
    out->epoch_id = (uint16_t)((raw_word >> 16) & 0xFFFFu);
    out->seq_id   = (uint32_t)((raw_word >> 32) & 0xFFFFFFFFu);
}

bool avid_validate_packet(avid_packet_validator_t *v, uint64_t raw_word,
                          avid_syndrome_packet_t *out) {
    if (!v || !out) {
        return false;
    }

    avid_syndrome_packet_t tmp;
    avid_unpack_packet_u64(raw_word, &tmp);

    if ((tmp.flags & AVID_PKT_FLAG_VALID) == 0u ||
        (tmp.flags & AVID_PKT_FLAG_PARITY_ERR) != 0u ||
        (tmp.flags & AVID_PKT_FLAG_RESERVED_MASK) != 0u) {
        if ((tmp.flags & AVID_PKT_FLAG_VALID) == 0u &&
            (tmp.flags & (AVID_PKT_FLAG_PARITY_ERR | AVID_PKT_FLAG_RESERVED_MASK)) == 0u &&
            v->has_seen_first &&
            tmp.epoch_id >= v->last_epoch_id &&
            tmp.seq_id == (uint32_t)(v->last_seq_id + 1u + (uint32_t)v->pending_dropout_span)) {
            if (v->pending_dropout_span < 0xFFFFu) {
                v->pending_dropout_span++;
            }
        }
        v->invalid_flag_count++;
        return false;
    }

    if (v->has_seen_first) {
        if (tmp.epoch_id < v->last_epoch_id) {
            v->epoch_regress_count++;
            return false;
        }
        uint32_t expected_next = v->last_seq_id + 1u;
        uint32_t expected_after_dropout = v->last_seq_id + 1u + (uint32_t)v->pending_dropout_span;
        bool seq_ok = (tmp.seq_id == expected_next) ||
                      (v->pending_dropout_span > 0u && tmp.seq_id == expected_after_dropout);
        if (!seq_ok) {
            v->seq_gap_count++;
            if (tmp.seq_id == v->last_seq_id) {
                v->seq_dup_count++;
            }
            return false;
        }
    }

    if (tmp.flags & AVID_PKT_FLAG_OVERFLOW) {
        v->overflow_warn_count++;
    }

    v->has_seen_first       = true;
    v->last_seq_id          = tmp.seq_id;
    v->last_epoch_id        = tmp.epoch_id;
    v->pending_dropout_span = 0u;
    v->valid_count++;
    *out = tmp;
    return true;
}

bool avid_validate_packet_bytes(avid_packet_validator_t *v,
                                const uint8_t *buf, size_t len,
                                avid_syndrome_packet_t *out) {
    if (!v || !buf || !out || len != AVID_PKT_WIRE_BYTES) {
        return false;
    }
    uint64_t word = 0;
    if (!avid_decode_packet_le64(buf, len, &word)) {
        return false;
    }
    return avid_validate_packet(v, word, out);
}

bool avid_validate_stream_framing(const uint8_t *buf, size_t len) {
    if (!buf || len == 0u || (len % AVID_PKT_WIRE_BYTES) != 0u) {
        return false;
    }
    return true;
}

static uint32_t avid_decode_le32(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           (((uint32_t)p[1]) << 8) |
           (((uint32_t)p[2]) << 16) |
           (((uint32_t)p[3]) << 24);
}

bool avid_inspect_stream_container(const uint8_t *probe_bytes, size_t probe_len,
                                   size_t total_stream_bytes,
                                   bool *out_has_header,
                                   size_t *out_payload_offset,
                                   uint64_t *out_n_packets) {
    if (!probe_bytes || total_stream_bytes == 0u) {
        return false;
    }

    if (probe_len >= 4u && avid_decode_le32(probe_bytes) == AVID_STREAM_HEADER_MAGIC) {
        if (probe_len < AVID_STREAM_HEADER_BYTES || total_stream_bytes <= AVID_STREAM_HEADER_BYTES) {
            return false;
        }
        uint32_t n_rounds = avid_decode_le32(probe_bytes + 4u);
        uint32_t is_u64   = avid_decode_le32(probe_bytes + 8u);
        uint8_t  n_cx     = probe_bytes[12];
        uint8_t  n_cz     = probe_bytes[13];
        uint8_t  dist     = probe_bytes[14];
        if (n_rounds == 0u || is_u64 != 1u || n_cx != 4u || n_cz != 4u || dist != 3u) {
            return false;
        }
        size_t expected_total = (size_t)AVID_STREAM_HEADER_BYTES +
                                ((size_t)n_rounds * (size_t)AVID_PKT_WIRE_BYTES);
        if (total_stream_bytes != expected_total) {
            return false;
        }
        if (out_has_header)     *out_has_header     = true;
        if (out_payload_offset) *out_payload_offset = (size_t)AVID_STREAM_HEADER_BYTES;
        if (out_n_packets)      *out_n_packets      = (uint64_t)n_rounds;
        return true;
    }

    if (!avid_validate_stream_framing(probe_bytes, total_stream_bytes)) {
        return false;
    }
    if (out_has_header)     *out_has_header     = false;
    if (out_payload_offset) *out_payload_offset = 0u;
    if (out_n_packets)      *out_n_packets      = (uint64_t)(total_stream_bytes / AVID_PKT_WIRE_BYTES);
    return true;
}
