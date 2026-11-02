/* ===========================================================================
 * fallback_fsm.h — 4-State Fault Containment & Fallback Machine (A09 / Q18–Q19)
 * Enforces `architecture/contracts/fault_policy.yaml`.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    AVID_FSM_NORMAL_ADAPTIVE   = 0,
    AVID_FSM_HOLD_LAST_GOOD    = 1,
    AVID_FSM_SAFE_STATIC_PRIOR = 2,
    AVID_FSM_DEGRADED_BYPASS   = 3
} avid_fallback_state_t;

typedef struct {
    avid_fallback_state_t state;
    uint32_t              consecutive_holds;
    uint32_t              max_holds_before_static;
    int16_t               last_good_q78[9];
    int16_t               static_prior_q78[9];
    uint64_t              fallback_triggers;
} avid_fallback_fsm_t;

void avid_fallback_init(avid_fallback_fsm_t *fsm, int16_t default_q78);

avid_fallback_state_t avid_fallback_step(avid_fallback_fsm_t *fsm,
                                         bool estimator_ok,
                                         bool deadline_ok,
                                         bool bounds_ok,
                                         bool bus_ok,
                                         const int16_t candidate_q78[9],
                                         int16_t out_selected_q78[9],
                                         bool *out_should_commit);
