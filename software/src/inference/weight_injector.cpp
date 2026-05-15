/* ===========================================================================
 * weight_injector.cpp — AVID Phase 3.4
 * =========================================================================*/
#include "weight_injector.h"
#include <cmath>
#include <cstdio>

void WeightInjector::inject(const NoiseModel &model) {
    model.to_llr(llr_x_buf_, llr_z_buf_);
    wrd_write_all(drv_, llr_x_buf_, llr_z_buf_);
    wrd_commit(drv_);
}

void WeightInjector::inject_default(float p0) {
    float llr = (float)(std::log((1.0f - p0) / p0));
    for (int i = 0; i < NM_N_QUBIT; i++) {
        llr_x_buf_[i] = llr;
        llr_z_buf_[i] = llr;
    }
    wrd_write_all(drv_, llr_x_buf_, llr_z_buf_);
    wrd_commit(drv_);
    printf("[injector] Default weights injected: p0=%.3f LLR=%.4f\n", p0, llr);
}

uint32_t WeightInjector::last_epoch() const {
    return wrd_read_epoch(drv_);
}
