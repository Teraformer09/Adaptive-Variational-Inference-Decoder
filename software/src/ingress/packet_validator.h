/* ===========================================================================
 * packet_validator.h — Canonical 64-Bit Syndrome Packet Validator (A05 / Q11)
 * Enforces `architecture/contracts/syndrome_packet_v1.yaml` (v1.1.0).
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AVID_PKT_WIRE_BYTES        8u
#define AVID_PKT_MAX_SYNDROME      15u
#define AVID_STREAM_HEADER_MAGIC   0xAAEC0001u
#define AVID_STREAM_HEADER_BYTES   16u

#define AVID_PKT_FLAG_VALID        (1u << 0)
#define AVID_PKT_FLAG_OVERFLOW     (1u << 1)
#define AVID_PKT_FLAG_PARITY_ERR   (1u << 2)
#define AVID_PKT_FLAG_EPOCH_MARK   (1u << 3)
#define AVID_PKT_FLAG_KNOWN_MASK   0x0Fu
#define AVID_PKT_FLAG_RESERVED_MASK 0xF0u

typedef struct {
    uint8_t  s_x;       /* bits [3:0]   : 4-bit X-stabilizer syndrome */
    uint8_t  s_z;       /* bits [7:4]   : 4-bit Z-stabilizer syndrome */
    uint8_t  flags;     /* bits [15:8]  : status flags */
    uint16_t epoch_id;  /* bits [31:16] : weight bank epoch ID */
    uint32_t seq_id;    /* bits [63:32] : monotonic sequence ID */
} avid_syndrome_packet_t;

typedef struct {
    uint32_t last_seq_id;
    bool     has_seen_first;
    uint64_t valid_count;
    uint64_t seq_gap_count;
    uint64_t invalid_flag_count;
    uint64_t overflow_warn_count;
    uint16_t last_epoch_id;
    uint16_t pending_dropout_span;
    uint64_t seq_dup_count;
    uint64_t epoch_regress_count;
} avid_packet_validator_t;

typedef struct {
    uint32_t magic;
    uint32_t n_rounds;
    uint32_t is_u64;
    uint8_t  n_check_x;
    uint8_t  n_check_z;
    uint8_t  distance;
    uint8_t  reserved;
} avid_stream_header_t;

void avid_validator_init(avid_packet_validator_t *v);

uint64_t avid_pack_packet_u64(uint8_t s_x, uint8_t s_z, uint32_t seq_id,
                              uint16_t epoch_id, uint8_t flags);

/* Strict domain-checked packer: rejects s_x > 15, s_z > 15, reserved flags, or NULL out_word */
bool avid_pack_packet_u64_checked(uint8_t s_x, uint8_t s_z, uint32_t seq_id,
                                  uint16_t epoch_id, uint8_t flags,
                                  uint64_t *out_word);

/* Explicit little-endian 8-byte wire encoder/decoder */
bool avid_encode_packet_le64(uint64_t raw_word, uint8_t *out_bytes, size_t out_len);
bool avid_decode_packet_le64(const uint8_t *bytes, size_t len, uint64_t *out_word);

void avid_unpack_packet_u64(uint64_t raw_word, avid_syndrome_packet_t *out);

bool avid_validate_packet(avid_packet_validator_t *v, uint64_t raw_word,
                          avid_syndrome_packet_t *out);

/* Validates an exact 8-byte little-endian packet buffer */
bool avid_validate_packet_bytes(avid_packet_validator_t *v,
                                const uint8_t *buf, size_t len,
                                avid_syndrome_packet_t *out);

/* Validates binary stream framing (non-empty, exact multiple of 8 bytes) */
bool avid_validate_stream_framing(const uint8_t *buf, size_t len);

/* Inspects stream container header (supports both 16-byte 0xAAEC0001 Stage 6 streams
 * and headless raw 8-byte A05/Q11 DMA packet streams) */
bool avid_inspect_stream_container(const uint8_t *probe_bytes, size_t probe_len,
                                   size_t total_stream_bytes,
                                   bool *out_has_header,
                                   size_t *out_payload_offset,
                                   uint64_t *out_n_packets);

#ifdef __cplusplus
}
#endif
