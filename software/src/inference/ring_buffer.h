/* ===========================================================================
 * ring_buffer.h — AVID Phase 3.1
 * Fixed-capacity ring buffer for syndrome history.
 * Stores the last W syndrome rounds (no dynamic allocation).
 *
 * COMMANDMENT #3 (No std::vector in hot loop):
 *   All storage is pre-allocated as a flat array.  Pop and push are O(1)
 *   pointer arithmetic with no heap interaction.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <string.h>

#define RING_WINDOW    10000     /* W: syndrome history window */
#define RING_N_CHECK_X     4    /* d=3: 4 X-stabilizers */
#define RING_N_CHECK_Z     4    /* d=3: 4 Z-stabilizers */
#define RING_N_CHECK   (RING_N_CHECK_X + RING_N_CHECK_Z)

/* One syndrome round: N_CHECK bits packed into a uint8 each for fast access */
typedef struct {
    uint8_t x[RING_N_CHECK_X];   /* X-stabilizer syndrome bits */
    uint8_t z[RING_N_CHECK_Z];   /* Z-stabilizer syndrome bits */
} syndrome_round_t;

typedef struct {
    syndrome_round_t buf[RING_WINDOW];
    int              head;     /* Next write index (modulo RING_WINDOW) */
    int              count;    /* Valid entries (0..RING_WINDOW) */
} ring_buffer_t;

static inline void ring_init(ring_buffer_t *rb) {
    memset(rb, 0, sizeof(*rb));
}

/* Push one syndrome round.  Overwrites oldest entry once full. */
static inline void ring_push(ring_buffer_t *rb, const syndrome_round_t *s) {
    rb->buf[rb->head] = *s;
    rb->head = (rb->head + 1) % RING_WINDOW;
    if (rb->count < RING_WINDOW) rb->count++;
}

/* Read-only access: index 0 = oldest, count-1 = newest */
static inline const syndrome_round_t *ring_at(const ring_buffer_t *rb, int idx) {
    int real = (rb->head - rb->count + idx + RING_WINDOW * 2) % RING_WINDOW;
    return &rb->buf[real];
}
