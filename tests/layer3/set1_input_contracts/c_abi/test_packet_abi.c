#include "c_abi_harness.h"
#include "c_abi_cxx_layout_oracle.h"
#include "ingress/packet_validator.h"

_Static_assert(sizeof(avid_syndrome_packet_t) == 12, "avid_syndrome_packet_t must be 12 bytes on LP64/ILP32");
_Static_assert(_Alignof(avid_syndrome_packet_t) == 4, "avid_syndrome_packet_t alignment must be 4");
_Static_assert(offsetof(avid_syndrome_packet_t, s_x) == 0, "s_x offset must be 0");
_Static_assert(offsetof(avid_syndrome_packet_t, s_z) == 1, "s_z offset must be 1");
_Static_assert(offsetof(avid_syndrome_packet_t, flags) == 2, "flags offset must be 2");
_Static_assert(offsetof(avid_syndrome_packet_t, epoch_id) == 4, "epoch_id offset must be 4");
_Static_assert(offsetof(avid_syndrome_packet_t, seq_id) == 8, "seq_id offset must be 8");

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-001", "c_abi_packet_validator",
               "software/src/ingress/packet_validator.h",
               "avid_validator_init, avid_pack_packet_u64, avid_unpack_packet_u64, avid_validate_packet");
    snprintf(ctx.description, sizeof(ctx.description),
             "Pure C11 compilation, struct layout parity vs C++17, and extern \"C\" linkage call into packet_validator.cpp");

    c_abi_struct_metrics_t cxx_pkt, cxx_val;
    c_abi_cxx_get_packet_layout(&cxx_pkt);
    c_abi_cxx_get_validator_layout(&cxx_val);

    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(avid_syndrome_packet_t), cxx_pkt.size, "sizeof(avid_syndrome_packet_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, _Alignof(avid_syndrome_packet_t), cxx_pkt.align, "alignof(avid_syndrome_packet_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, offsetof(avid_syndrome_packet_t, epoch_id), cxx_pkt.off3, "offsetof(epoch_id) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, offsetof(avid_syndrome_packet_t, seq_id), cxx_pkt.off4, "offsetof(seq_id) C11==C++17");

    C_ABI_CHECK_EQ_ZU(&ctx, sizeof(avid_packet_validator_t), cxx_val.size, "sizeof(avid_packet_validator_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, _Alignof(avid_packet_validator_t), cxx_val.align, "alignof(avid_packet_validator_t) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, offsetof(avid_packet_validator_t, has_seen_first), cxx_val.off1, "offsetof(has_seen_first) C11==C++17");
    C_ABI_CHECK_EQ_ZU(&ctx, offsetof(avid_packet_validator_t, valid_count), cxx_val.off2, "offsetof(valid_count) C11==C++17");

    /* Direct C11 call into C++ production translation unit via extern "C" */
    avid_packet_validator_t v;
    avid_validator_init(&v);
    uint64_t w = avid_pack_packet_u64(5, 11, 0, 3, AVID_PKT_FLAG_VALID);
    avid_syndrome_packet_t out;
    memset(&out, 0, sizeof(out));
    bool ok = avid_validate_packet(&v, w, &out);

    C_ABI_CHECK_TRUE(&ctx, ok, "avid_validate_packet accepted valid C11 call");
    C_ABI_CHECK_EQ_ZU(&ctx, out.s_x, 5, "decoded s_x == 5 across C11/C++ ABI");
    C_ABI_CHECK_EQ_ZU(&ctx, out.s_z, 11, "decoded s_z == 11 across C11/C++ ABI");
    C_ABI_CHECK_EQ_ZU(&ctx, out.epoch_id, 3, "decoded epoch_id == 3 across C11/C++ ABI");
    C_ABI_CHECK_EQ_ZU(&ctx, out.seq_id, 0, "decoded seq_id == 0 across C11/C++ ABI");
    C_ABI_CHECK_EQ_ZU(&ctx, v.valid_count, 1, "validator valid_count == 1 across C11/C++ ABI");

    /* Note unpacked struct (12B) vs wire word (8B) difference */
    c_abi_note(&ctx, "Wire word is uint64_t (8 bytes); unpacked avid_syndrome_packet_t is 12 bytes (1-byte padding at byte 3); C11 and C++17 layouts match bit-for-bit.");

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "C11/C++17 layout equality and clean extern \"C\" call for packet_validator.h");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "sizeof(pkt)=%zu, sizeof(validator)=%zu, extern \"C\" call succeeded (checks_failed=%d)",
             sizeof(avid_syndrome_packet_t), sizeof(avid_packet_validator_t), ctx.checks_failed);

    return c_abi_finalize(&ctx, (ctx.checks_failed == 0) ? "PASS" : "FAIL", "");
}
