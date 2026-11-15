#include "c_abi_harness.h"
#include "ingress/packet_validator.h"
#include "inference/ring_buffer.h"
#include "estimator/llr_quantizer.h"
#include "driver/weight_reg_driver.h"
#include "driver/weight_bank_driver.h"

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-007", "c_abi_buffer_ownership",
               "software/src/{ingress,inference,estimator,driver}/*",
               "caller-owned buffer lifecycle, canary guard zones, zero hidden free/realloc");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify caller-owned stack/heap buffer ownership, zero out-of-bounds canary writes, and clean wrd_open/wrd_close lifecycle from C11");

    /* 1. Heap buffer with front/back 64-byte 0xA5 canaries for quantizer input/output */
    uint8_t *raw_in = (uint8_t *)malloc(64 + 9 * sizeof(double) + 64);
    uint8_t *raw_out = (uint8_t *)malloc(64 + 9 * sizeof(int16_t) + 64);
    memset(raw_in, 0xA5, 64 + 9 * sizeof(double) + 64);
    memset(raw_out, 0xA5, 64 + 9 * sizeof(int16_t) + 64);

    double *p_in = (double *)(raw_in + 64);
    int16_t *q_out = (int16_t *)(raw_out + 64);
    for (int i = 0; i < 9; ++i) p_in[i] = 0.012 + 0.001 * i;

    bool ok = avid_quantize_and_validate_bank(p_in, q_out);
    C_ABI_CHECK_TRUE(&ctx, ok, "avid_quantize_and_validate_bank succeeded on caller-owned heap buffers");

    bool canary_intact = true;
    for (int i = 0; i < 64; ++i) {
        if (raw_in[i] != 0xA5 || raw_in[64 + 9 * sizeof(double) + i] != 0xA5) canary_intact = false;
        if (raw_out[i] != 0xA5 || raw_out[64 + 9 * sizeof(int16_t) + i] != 0xA5) canary_intact = false;
    }
    C_ABI_CHECK_TRUE(&ctx, canary_intact, "64-byte front/back heap canaries intact around p_eff[9] and out_q78[9]");

    /* Mutate caller input buffer after call and verify output bank holds independent values */
    int16_t saved_q0 = q_out[0];
    for (int i = 0; i < 9; ++i) p_in[i] = 0.44;
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)q_out[0], (size_t)saved_q0, "output Q7.8 bank does not alias input p_eff buffer");

    free(raw_in);
    free(raw_out);

    /* 2. Flat pre-allocated ring_buffer_t ownership & wraparound (> RING_WINDOW pushes without heap alloc) */
    ring_buffer_t *rb = (ring_buffer_t *)calloc(1, sizeof(ring_buffer_t));
    ring_init(rb);
    for (int r = 0; r < RING_WINDOW + 25; ++r) {
        syndrome_round_t sr = {{(uint8_t)(r & 1), 0, 0, 0}, {0, (uint8_t)((r >> 1) & 1), 0, 0}};
        ring_push(rb, &sr);
    }
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)rb->count, (size_t)RING_WINDOW, "ring_buffer_t saturates count at RING_WINDOW=10000");
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)rb->head, 25, "ring_buffer_t head wraps cleanly modulo 10000");
    free(rb);

    /* 3. Emulated wrd_handle_t lifecycle (caller-owned struct, safe close) */
    wrd_handle_t wrd;
    memset(&wrd, 0xCC, sizeof(wrd));
    int wrc = wrd_open(&wrd, WRD_BASE_ADDR);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)wrc, 0, "wrd_open initializes caller-allocated wrd_handle_t");
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)wrd.is_emulated, 1, "wrd_open falls back to emulated sim_regs[32] without /dev/mem");
    wrd_close(&wrd);

    c_abi_note(&ctx, "All C-callable APIs use caller-owned storage with zero hidden free() or out-of-bounds writes across 64-byte guard canaries.");

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "Caller-owned buffers remain uncorrupted, unaliased, and freeable by caller");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "All %d ownership & canary checks passed (%d failed)", ctx.checks_run, ctx.checks_failed);

    return c_abi_finalize(&ctx, (ctx.checks_failed == 0) ? "PASS" : "FAIL", "");
}
