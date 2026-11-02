/* ===========================================================================
 * llr_quantizer.cpp — Signed Q7.8 Fixed-Point LLR Quantizer (SF8)
 * =========================================================================*/
#include "llr_quantizer.h"
#include <cmath>
#include <algorithm>

int16_t avid_prob_to_q78_llr(double p) {
    double pc = std::clamp(p, 1e-5, 0.49);
    double llr = std::log((1.0 - pc) / pc);
    long raw = std::lround(llr * 256.0);
    if (raw > 32767) raw = 32767;
    if (raw < -32768) raw = -32768;
    return (int16_t)raw;
}

bool avid_quantize_and_validate_bank(const double p_eff[9],
                                     int16_t out_q78[9]) {
    bool all_safe = true;
    for (int i = 0; i < 9; i++) {
        if (!std::isfinite(p_eff[i]) || p_eff[i] <= 0.0 || p_eff[i] >= 0.5) {
            all_safe = false;
            out_q78[i] = AVID_Q78_DEFAULT;
            continue;
        }
        int16_t q = avid_prob_to_q78_llr(p_eff[i]);
        if (q < AVID_Q78_MIN_SAFE || q > AVID_Q78_MAX_SAFE) {
            all_safe = false;
            q = std::clamp(q, AVID_Q78_MIN_SAFE, AVID_Q78_MAX_SAFE);
        }
        out_q78[i] = q;
    }
    return all_safe;
}
