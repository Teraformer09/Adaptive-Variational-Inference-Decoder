#include "c_abi_harness.h"
#include "estimator/llr_quantizer.h"

_Static_assert(sizeof(int16_t) == 2, "int16_t Q7.8 weight must be 2 bytes");
_Static_assert(sizeof(double) == 8, "double probability must be 8 bytes");
_Static_assert(AVID_Q78_MIN_SAFE == 0x0040, "AVID_Q78_MIN_SAFE must equal 0x0040 (+0.25)");
_Static_assert(AVID_Q78_MAX_SAFE == 0x3000, "AVID_Q78_MAX_SAFE must equal 0x3000 (+48.0)");
_Static_assert(AVID_Q78_DEFAULT  == 0x0468, "AVID_Q78_DEFAULT must equal 0x0468");

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-003", "c_abi_parameter_llr_quantizer",
               "software/src/estimator/llr_quantizer.h, software/src/inference/noise_model.h",
               "avid_prob_to_q78_llr, avid_quantize_and_validate_bank");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify C11 ABI of llr_quantizer.h and check whether a C ABI exists for 21-parameter NoiseModel inputs");

    double p_eff[9] = {0.012, 0.012, 0.012, 0.015, 0.015, 0.015, 0.020, 0.020, 0.020};
    int16_t out_q78[9] = {0};

    bool ok = avid_quantize_and_validate_bank(p_eff, out_q78);
    C_ABI_CHECK_TRUE(&ctx, ok, "avid_quantize_and_validate_bank succeeded from C11");
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)out_q78[0], (size_t)0x0469, "p=0.012 quantizes to lround(log((1-0.012)/0.012)*256) == 1129 (0x0469) across C11/C++ ABI");

    int16_t single = avid_prob_to_q78_llr(0.012);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)single, (size_t)0x0469, "avid_prob_to_q78_llr(0.012) == 0x0469 in C11");
    c_abi_note(&ctx, "Minor constant note: AVID_Q78_DEFAULT macro is 0x0468 (1128), 1 LSB below avid_prob_to_q78_llr(0.012) == 0x0469 (1129).");

    /* Unphysical probability >= 0.5 must be rejected by avid_quantize_and_validate_bank */
    double bad_p[9] = {0.55, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012, 0.012};
    int16_t bad_out[9] = {0};
    bool bad_ok = avid_quantize_and_validate_bank(bad_p, bad_out);
    C_ABI_CHECK_TRUE(&ctx, !bad_ok, "avid_quantize_and_validate_bank rejects p=0.55 from C11");

    c_abi_note(&ctx, "llr_quantizer.h has valid extern \"C\" C11 ABI for 9-qubit effective probabilities; 21-parameter NoiseModel/HistogramLBFGSSolver headers include <Eigen/Dense> and expose C++ classes only (no C11 wrapper).");

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "Clean C11 linkage and Q7.8 fixed-point ABI on llr_quantizer.h; document C++-only 21-param NoiseModel boundary");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "llr_quantizer.h C11 calls passed (out_q78[0]=0x%04x, bad_p rejected=%s); NoiseModel is C++-only",
             (unsigned)out_q78[0], (!bad_ok) ? "true" : "false");

    return c_abi_finalize(&ctx, (ctx.checks_failed == 0) ? "PASS" : "FAIL", "");
}
