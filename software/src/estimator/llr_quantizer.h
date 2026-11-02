/* ===========================================================================
 * llr_quantizer.h — Signed Q7.8 Fixed-Point LLR Quantizer (SF8)
 * Enforces `architecture/contracts/llr_numeric_contract.yaml`.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AVID_Q78_MIN_SAFE   ((int16_t)0x0040)  /* +0.25 LLR minimum safe positive prior */
#define AVID_Q78_MAX_SAFE   ((int16_t)0x3000)  /* +48.0 LLR maximum safe positive prior */
#define AVID_Q78_DEFAULT    ((int16_t)0x0468)  /* ~4.406 LLR (p = 0.012) */

int16_t avid_prob_to_q78_llr(double p);

bool avid_quantize_and_validate_bank(const double p_eff[9],
                                     int16_t out_q78[9]);

#ifdef __cplusplus
}
#endif
