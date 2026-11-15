#include "c_abi_harness.h"
#include "ingress/packet_validator.h"
#include "estimator/llr_quantizer.h"
#include "driver/weight_bank_driver.h"

static int probe_validator_init_null(void *arg) {
    (void)arg;
    avid_validator_init(NULL);
    return 0;
}

static int probe_unpack_null(void *arg) {
    (void)arg;
    avid_unpack_packet_u64(0x00000000000301B5ULL, NULL);
    return 0;
}

static int probe_validate_null_v(void *arg) {
    (void)arg;
    avid_syndrome_packet_t pkt;
    bool ok = avid_validate_packet(NULL, 0x00000000000301B5ULL, &pkt);
    return ok ? 0 : 1;
}

static int probe_validate_null_out(void *arg) {
    (void)arg;
    avid_packet_validator_t v;
    avid_validator_init(&v);
    bool ok = avid_validate_packet(&v, 0x00000000000301B5ULL, NULL);
    return ok ? 0 : 1;
}

static int probe_quantize_null_in(void *arg) {
    (void)arg;
    int16_t out[9] = {0};
    bool ok = avid_quantize_and_validate_bank(NULL, out);
    return ok ? 0 : 1;
}

static int probe_quantize_null_out(void *arg) {
    (void)arg;
    double p[9] = {0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012};
    bool ok = avid_quantize_and_validate_bank(p, NULL);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-006", "c_abi_null_pointer_safety",
               "software/src/ingress/packet_validator.cpp, estimator/llr_quantizer.cpp, driver/weight_bank_driver.c",
               "avid_validator_init, avid_unpack_packet_u64, avid_validate_packet, avid_quantize_and_validate_bank, avid_weight_bank_write_and_commit");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify C11 null-pointer handling across public extern \"C\" entrypoints using fork()-isolated probes");

    /* 1. Safe null check in weight_bank_driver.c */
    int16_t dummy_w[9] = {1024,1024,1024,1024,1024,1024,1024,1024,1024};
    int rc_wbd_null_h = avid_weight_bank_write_and_commit(NULL, dummy_w, NULL);
    C_ABI_CHECK_TRUE(&ctx, rc_wbd_null_h == -1, "avid_weight_bank_write_and_commit(NULL, ...) safely returns -1");

    /* 2. Fork-isolated probes for unguarded C entrypoints */
    struct {
        const char *name;
        int (*fn)(void *);
    } probes[] = {
        {"avid_validator_init(NULL)", probe_validator_init_null},
        {"avid_unpack_packet_u64(..., NULL)", probe_unpack_null},
        {"avid_validate_packet(NULL, ...)", probe_validate_null_v},
        {"avid_validate_packet(..., NULL)", probe_validate_null_out},
        {"avid_quantize_and_validate_bank(NULL, out)", probe_quantize_null_in},
        {"avid_quantize_and_validate_bank(p, NULL)", probe_quantize_null_out},
    };

    int crash_count = 0;
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
        int signaled = 0, sig = 0;
        int rc = c_abi_run_isolated_probe(probes[i].fn, NULL, &signaled, &sig);
        if (signaled || rc >= 128) {
            crash_count++;
            char msg[256];
            snprintf(msg, sizeof(msg), "NULL DEREF CRASH: %s terminated child with signal %d (rc=%d)",
                     probes[i].name, sig, rc);
            c_abi_note(&ctx, msg);
        }
        C_ABI_CHECK_TRUE(&ctx, !signaled, probes[i].name);
    }

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "Zero SIGSEGV/SIGABRT crashes when NULL pointers are passed across C ABI entrypoints");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "avid_weight_bank_write_and_commit(NULL)=-1 (safe), but %d/6 packet_validator & llr_quantizer C entrypoints crashed on NULL pointer",
             crash_count);

    return c_abi_finalize(&ctx, (crash_count == 0 && ctx.checks_failed == 0) ? "PASS" : "FAIL",
                          (crash_count > 0) ? "C_ABI_UNGUARDED_NULL_DEREFERENCE" : "");
}
