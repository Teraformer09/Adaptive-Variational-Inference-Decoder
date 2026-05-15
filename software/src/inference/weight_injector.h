/* ===========================================================================
 * weight_injector.h — AVID Phase 3.4
 * Translates optimised NoiseModel parameters → Q7.8 LLRs → FPGA injection.
 *
 * Flow: NoiseModel::to_llr() → Q7.8 clamp → wrd_write_all() → wrd_commit()
 * =========================================================================*/
#pragma once
#include "noise_model.h"
#include "../driver/weight_reg_driver.h"

class WeightInjector {
public:
    WeightInjector(wrd_handle_t *driver) : drv_(driver) {}

    /**
     * inject() — Write current model parameters to the FPGA shadow bank
     *            and commit the atomic swap.
     * Thread-safety: call from the Slow-Path thread only.
     */
    void inject(const NoiseModel &model);

    /**
     * inject_default() — Reset FPGA weights to log((1-p0)/p0) for all qubits.
     */
    void inject_default(float p0 = 0.01f);

    /** Last injected epoch (from wrd_read_epoch()) */
    uint32_t last_epoch() const;

private:
    wrd_handle_t *drv_;
    float llr_x_buf_[NM_N_QUBIT];  /* Pre-allocated float LLR buffers */
    float llr_z_buf_[NM_N_QUBIT];
};
