/* ===========================================================================
 * fallback_fsm.cpp — 4-State Fault Containment & Fallback Machine (A09 / Q18)
 * =========================================================================*/
#include "fallback_fsm.h"
#include <string.h>

static inline bool is_valid_fsm_state_ptr(const avid_fallback_state_t *st_ptr) {
    static_assert(sizeof(unsigned int) == sizeof(avid_fallback_state_t),
                  "avid_fallback_state_t width must match unsigned int");
    unsigned int raw = 0;
    memcpy(&raw, st_ptr, sizeof(raw));
    return (raw == (unsigned int)AVID_FSM_NORMAL_ADAPTIVE ||
            raw == (unsigned int)AVID_FSM_HOLD_LAST_GOOD ||
            raw == (unsigned int)AVID_FSM_SAFE_STATIC_PRIOR ||
            raw == (unsigned int)AVID_FSM_DEGRADED_BYPASS);
}

void avid_fallback_init(avid_fallback_fsm_t *fsm, int16_t default_q78) {
    if (!fsm) {
        return;
    }
    memset(fsm, 0, sizeof(*fsm));
    fsm->state = AVID_FSM_NORMAL_ADAPTIVE;
    fsm->max_holds_before_static = 3;
    for (int i = 0; i < 9; i++) {
        fsm->last_good_q78[i]    = default_q78;
        fsm->static_prior_q78[i] = default_q78;
    }
}

avid_fallback_state_t avid_fallback_step(avid_fallback_fsm_t *fsm,
                                         bool estimator_ok,
                                         bool deadline_ok,
                                         bool bounds_ok,
                                         bool bus_ok,
                                         const int16_t candidate_q78[9],
                                         int16_t out_selected_q78[9],
                                         bool *out_should_commit) {
    if (!fsm) {
        if (out_should_commit) {
            *out_should_commit = false;
        }
        if (out_selected_q78) {
            memset(out_selected_q78, 0, sizeof(int16_t) * 9);
        }
        return AVID_FSM_DEGRADED_BYPASS;
    }

    if (!out_selected_q78 || !out_should_commit) {
        fsm->fallback_triggers++;
        fsm->state = AVID_FSM_SAFE_STATIC_PRIOR;
        if (out_should_commit) {
            *out_should_commit = false;
        }
        if (out_selected_q78) {
            memcpy(out_selected_q78, fsm->static_prior_q78, sizeof(int16_t) * 9);
        }
        return fsm->state;
    }

    /* Sanitize out-of-domain incoming enum state without forming an invalid enum rvalue */
    const bool incoming_state_corrupt = !is_valid_fsm_state_ptr(&fsm->state);
    if (incoming_state_corrupt) {
        fsm->fallback_triggers++;
        fsm->state = AVID_FSM_SAFE_STATIC_PRIOR;
        memcpy(out_selected_q78, fsm->static_prior_q78, sizeof(int16_t) * 9);
        *out_should_commit = bus_ok;
        if (!bus_ok) {
            fsm->state = AVID_FSM_DEGRADED_BYPASS;
        }
        return fsm->state;
    }

    if (!bus_ok) {
        fsm->state = AVID_FSM_DEGRADED_BYPASS;
        fsm->fallback_triggers++;
        memcpy(out_selected_q78, fsm->static_prior_q78, sizeof(int16_t) * 9);
        *out_should_commit = false;
        return fsm->state;
    }

    if (!candidate_q78) {
        fsm->fallback_triggers++;
        fsm->consecutive_holds++;
        fsm->state = AVID_FSM_SAFE_STATIC_PRIOR;
        memcpy(out_selected_q78, fsm->static_prior_q78, sizeof(int16_t) * 9);
        *out_should_commit = false;
        return fsm->state;
    }

    if (estimator_ok && deadline_ok && bounds_ok) {
        fsm->state = AVID_FSM_NORMAL_ADAPTIVE;
        fsm->consecutive_holds = 0;
        memcpy(fsm->last_good_q78, candidate_q78, sizeof(int16_t) * 9);
        memcpy(out_selected_q78, candidate_q78, sizeof(int16_t) * 9);
        *out_should_commit = true;
        return fsm->state;
    }

    fsm->fallback_triggers++;
    fsm->consecutive_holds++;

    if (!bounds_ok || fsm->consecutive_holds >= fsm->max_holds_before_static) {
        fsm->state = AVID_FSM_SAFE_STATIC_PRIOR;
        memcpy(out_selected_q78, fsm->static_prior_q78, sizeof(int16_t) * 9);
        *out_should_commit = true;
        return fsm->state;
    }

    fsm->state = AVID_FSM_HOLD_LAST_GOOD;
    memcpy(out_selected_q78, fsm->last_good_q78, sizeof(int16_t) * 9);
    *out_should_commit = false;
    return fsm->state;
}
