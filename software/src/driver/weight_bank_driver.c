/* ===========================================================================
 * weight_bank_driver.c — Shadow/Active Atomic Weight Bank Driver (SF5)
 * =========================================================================*/
#include "weight_bank_driver.h"

int avid_weight_bank_write_and_commit(wrd_handle_t *h,
                                      const int16_t weights_q78[WRD_NUM_QUBITS],
                                      uint16_t *out_swap_count) {
    if (!h) return -1;
    wrd_write_all_weights(h, weights_q78);
    int rc = wrd_commit_and_wait(h, 1000);
    if (out_swap_count) {
        *out_swap_count = wrd_get_swap_count(h);
    }
    return rc;
}
