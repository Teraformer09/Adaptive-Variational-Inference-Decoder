/* ===========================================================================
 * causal_scheduler.cpp — Dual-Timescale Causal Epoch Scheduler (A04 / Q01)
 * =========================================================================*/
#include "causal_scheduler.h"
#include <string.h>

void avid_scheduler_init(avid_causal_scheduler_t *sched,
                         uint64_t fast_window,
                         uint64_t slow_window,
                         uint64_t commit_delay) {
    memset(sched, 0, sizeof(*sched));
    sched->fast_window_rounds  = fast_window;
    sched->slow_window_rounds  = slow_window;
    sched->commit_delay_rounds = commit_delay;
}

void avid_scheduler_schedule_commit(avid_causal_scheduler_t *sched,
                                    uint64_t window_end_round,
                                    const int16_t new_weights_q78[9]) {
    memcpy(sched->pending_weights_q78, new_weights_q78, sizeof(int16_t) * 9);
    sched->pending_commit_round = window_end_round + sched->commit_delay_rounds;
    sched->has_pending_commit   = true;
}

bool avid_scheduler_poll_commit_ready(avid_causal_scheduler_t *sched,
                                      uint64_t round_now,
                                      int16_t out_weights_q78[9]) {
    sched->current_round = round_now;
    if (sched->has_pending_commit && round_now >= sched->pending_commit_round) {
        memcpy(out_weights_q78, sched->pending_weights_q78, sizeof(int16_t) * 9);
        sched->has_pending_commit = false;
        sched->active_epoch_id++;
        return true;
    }
    return false;
}
