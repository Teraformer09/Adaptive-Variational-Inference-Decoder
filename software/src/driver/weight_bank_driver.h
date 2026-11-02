/* ===========================================================================
 * weight_bank_driver.h — Shadow/Active Atomic Weight Bank Driver (SF5)
 * Wraps `weight_reg_driver.h` with explicit shadow verification & epoch tracking.
 * =========================================================================*/
#pragma once
#include "weight_reg_driver.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

int avid_weight_bank_write_and_commit(wrd_handle_t *h,
                                      const int16_t weights_q78[WRD_NUM_QUBITS],
                                      uint16_t *out_swap_count);

#ifdef __cplusplus
}
#endif
