/* ===========================================================================
 * packet_validator.h — Canonical 64-Bit Syndrome Packet Validator (A05 / Q11)
 * Enforces `architecture/contracts/syndrome_packet_v1.yaml`.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AVID_PKT_FLAG_VALID        (1u << 0)
#define AVID_PKT_FLAG_OVERFLOW     (1u << 1)
#define AVID_PKT_FLAG_PARITY_ERR   (1u << 2)
#define AVID_PKT_FLAG_EPOCH_MARK   (1u << 3)

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
} avid_packet_validator_t;

void avid_validator_init(avid_packet_validator_t *v);

uint64_t avid_pack_packet_u64(uint8_t s_x, uint8_t s_z, uint32_t seq_id,
                              uint16_t epoch_id, uint8_t flags);

void avid_unpack_packet_u64(uint64_t raw_word, avid_syndrome_packet_t *out);

bool avid_validate_packet(avid_packet_validator_t *v, uint64_t raw_word,
                          avid_syndrome_packet_t *out);

#ifdef __cplusplus
}
#endif
