#include "c_abi_harness.h"
#include "c_abi_cxx_layout_oracle.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "driver/weight_reg_driver.h"
#include "driver/weight_bank_driver.h"
#include "hal/avid_hal.h"
#include <dlfcn.h>

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-004", "c_abi_runtime_config_hal",
               "software/src/control/causal_scheduler.h, fallback_fsm.h, hal/avid_hal.h, driver/weight_reg_driver.h",
               "avid_scheduler_init, avid_fallback_init, avid_hal_open, wrd_open");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify C11 compatibility of runtime control, driver, and HAL headers (including avid_hal.h C11 preprocessing and extern \"C\" linkage)");

    c_abi_cxx_retain_prod_symbols();

    /* 1. Driver C11 calls (weight_reg_driver.c & weight_bank_driver.c are pure C with extern "C" headers) */
    wrd_handle_t wrd;
    int rc_open = wrd_open(&wrd, WRD_BASE_ADDR);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)rc_open, 0, "wrd_open succeeded from C11");
    int16_t weights[9] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    uint16_t swaps = 0;
    int rc_commit = avid_weight_bank_write_and_commit(&wrd, weights, &swaps);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)rc_commit, 0, "avid_weight_bank_write_and_commit succeeded from C11");
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)wrd_read_active_weight(&wrd, 0), 1024, "wrd_read_active_weight(0) == 1024");
    wrd_close(&wrd);

    /* 2. Direct C11 calls & extern "C" linkage check on causal_scheduler.h & fallback_fsm.h */
    avid_causal_scheduler_t sched;
    avid_scheduler_init(&sched, 16, 10000, 4);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)sched.commit_delay_rounds, 4, "avid_scheduler_init C11 call sets commit_delay_rounds == 4");

    avid_fast_burst_detector_t det;
    avid_fast_detector_init(&det, 0.35, 4);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)det.subwindow_rounds, 16, "avid_fast_detector_init C11 call sets subwindow_rounds == 16");

    avid_fallback_fsm_t fsm;
    avid_fallback_init(&fsm, 1024);
    C_ABI_CHECK_EQ_ZU(&ctx, (size_t)fsm.state, (size_t)AVID_FSM_NORMAL_ADAPTIVE, "avid_fallback_init C11 call sets NORMAL_ADAPTIVE");

    void *c_sched_sym = dlsym(NULL, "avid_scheduler_init");
    void *c_det_sym   = dlsym(NULL, "avid_fast_detector_init");
    void *c_fsm_sym   = dlsym(NULL, "avid_fallback_init");

    if (!c_sched_sym) {
        c_abi_note(&ctx, "ABI DEFECT: causal_scheduler.h lacks extern \"C\"; dlsym(\"avid_scheduler_init\") == NULL!");
    }
    if (!c_det_sym) {
        c_abi_note(&ctx, "ABI DEFECT: causal_scheduler.h lacks extern \"C\"; dlsym(\"avid_fast_detector_init\") == NULL!");
    }
    if (!c_fsm_sym) {
        c_abi_note(&ctx, "ABI DEFECT: fallback_fsm.h lacks extern \"C\"; dlsym(\"avid_fallback_init\") == NULL!");
    }

    C_ABI_CHECK_TRUE(&ctx, c_sched_sym != NULL, "causal_scheduler.h must export C symbol avid_scheduler_init");
    C_ABI_CHECK_TRUE(&ctx, c_det_sym != NULL, "causal_scheduler.h must export C symbol avid_fast_detector_init");
    C_ABI_CHECK_TRUE(&ctx, c_fsm_sym != NULL, "fallback_fsm.h must export C symbol avid_fallback_init");

    /* 3. Empirically compile-check hal/avid_hal.h with C11 compiler (-std=c11 -fsyntax-only) */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "%s -std=c11 -x c -fsyntax-only -I\"%s/software/src\" \"%s/software/src/hal/avid_hal.h\" >/dev/null 2>&1",
             ctx.c_compiler, ctx.repo_root, ctx.repo_root);
    int hal_rc = system(cmd);
    bool hal_c11_ok = (hal_rc == 0);
    if (!hal_c11_ok) {
        c_abi_note(&ctx, "ABI DEFECT: software/src/hal/avid_hal.h fails `cc -std=c11 -fsyntax-only` because lines 9-10 use `#include <cstdint>` and `#include <cstddef>` outside `#ifdef __cplusplus` despite having `extern \"C\"` at line 14!");
    }
    C_ABI_CHECK_TRUE(&ctx, hal_c11_ok, "hal/avid_hal.h must compile cleanly in C11 mode (-std=c11)");

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "All C-facing control, driver, and HAL headers compile under C11 and export unmangled C symbols");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "wrd/weight_bank C11 OK; avid_scheduler_init=%s, avid_fallback_init=%s, avid_hal.h -std=c11=%s",
             c_sched_sym ? "FOUND" : "NULL",
             c_fsm_sym ? "FOUND" : "NULL",
             hal_c11_ok ? "PASS" : "FAIL (<cstdint> in C11)");

    return c_abi_finalize(&ctx, (ctx.checks_failed == 0) ? "PASS" : "FAIL",
                          (ctx.checks_failed > 0) ? "C_ABI_HEADER_AND_LINKAGE_MISMATCH" : "");
}
