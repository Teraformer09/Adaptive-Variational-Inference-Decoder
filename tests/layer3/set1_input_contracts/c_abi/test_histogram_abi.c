#include "c_abi_harness.h"
#include "c_abi_cxx_layout_oracle.h"
#include "ingress/dma_ingress_tap.h"
#include "inference/ring_buffer.h"
#include <dlfcn.h>

_Static_assert(sizeof(syndrome_round_t) == 8, "syndrome_round_t must be 8 bytes");
_Static_assert(RING_WINDOW == 10000, "RING_WINDOW must be 10000");

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-002", "c_abi_histogram_dma_ring",
               "software/src/ingress/dma_ingress_tap.h, software/src/inference/ring_buffer.h",
               "avid_dma_ring_init, avid_dma_ring_push_burst, avid_dma_ring_consume_slot, ring_init, ring_push, ring_at");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify C11 struct layout of avid_dma_ring_t & ring_buffer_t and test C symbol linkage of dma_ingress_tap.h");

    c_abi_cxx_retain_prod_symbols();

    c_abi_struct_metrics_t cxx_slot, cxx_ring, cxx_round, cxx_rb;
    c_abi_cxx_get_dma_slot_layout(&cxx_slot);
    c_abi_cxx_get_dma_ring_layout(&cxx_ring);
    c_abi_cxx_get_syndrome_round_layout(&cxx_round);
    c_abi_cxx_get_ring_buffer_layout(&cxx_rb);

    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(avid_dma_slot_t), cxx_slot.size, "sizeof(avid_dma_slot_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, _Alignof(avid_dma_slot_t), cxx_slot.align, "alignof(avid_dma_slot_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(avid_dma_ring_t), cxx_ring.size, "sizeof(avid_dma_ring_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, offsetof(avid_dma_ring_t, validator), cxx_ring.off4, "offsetof(validator) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(syndrome_round_t), cxx_round.size, "sizeof(syndrome_round_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(ring_buffer_t), cxx_rb.size, "sizeof(ring_buffer_t) C11==C++17");

    /* Header-only C11 ring_buffer.h calls work directly in C11 */
    static ring_buffer_t rb;
    ring_init(&rb);
    syndrome_round_t sr = {{1, 0, 1, 0}, {0, 1, 0, 1}};
    ring_push(&rb, &sr);
    const syndrome_round_t *got = ring_at(&rb, 0);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)rb.count, 1, "ring_buffer_t count == 1 in C11");
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)got->x[0], 1, "ring_buffer_t x[0] == 1 in C11");

    /* Direct C11 extern "C" call into dma_ingress_tap.cpp + dynamic symbol verification */
    static avid_dma_ring_t dma_ring;
    avid_dma_ring_init(&dma_ring);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)dma_ring.write_slot_idx, 0, "avid_dma_ring_init C11 call sets write_slot_idx == 0");

    void *c_sym_dma_init   = dlsym(NULL, "avid_dma_ring_init");
    void *cxx_sym_dma_init = dlsym(NULL, "_Z18avid_dma_ring_initP15avid_dma_ring_t");

    c_abi_note(&ctx, c_sym_dma_init
        ? "dlsym(\"avid_dma_ring_init\") resolved and direct C11 call succeeded"
        : "ABI DEFECT: dma_ingress_tap.h lacks extern \"C\"; dlsym(\"avid_dma_ring_init\") == NULL!");

    C_ABI_CHECK_TRUE(&ctx, c_sym_dma_init != NULL,
                     "dma_ingress_tap.h must export unmangled C symbol avid_dma_ring_init for C11 callers");

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "C11/C++17 struct layout parity AND unmangled extern \"C\" symbol linkage for dma_ingress_tap.h");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "Layouts match (avid_dma_ring_t=%zu B, ring_buffer_t=%zu B); dlsym(\"avid_dma_ring_init\")=%s, mangled=%s",
             sizeof(avid_dma_ring_t), sizeof(ring_buffer_t),
             c_sym_dma_init ? "FOUND" : "NULL (missing extern \"C\")",
             cxx_sym_dma_init ? "FOUND" : "NULL");

    return c_abi_finalize(&ctx, (c_sym_dma_init != NULL && ctx.checks_failed == 0) ? "PASS" : "FAIL",
                          (c_sym_dma_init == NULL) ? "C_ABI_MISSING_EXTERN_C_LINKAGE" : "");
}
