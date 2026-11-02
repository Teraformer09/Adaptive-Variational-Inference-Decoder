/* ===========================================================================
 * causal_scheduler.h — Dual-Timescale Causal Epoch Scheduler (A04 / Q01)
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint64_t current_round;
    uint64_t fast_window_rounds;
    uint64_t slow_window_rounds;
    uint64_t commit_delay_rounds;
    uint64_t pending_commit_round;
    bool     has_pending_commit;
    int16_t  pending_weights_q78[9];
    uint16_t active_epoch_id;
} avid_causal_scheduler_t;

void avid_scheduler_init(avid_causal_scheduler_t *sched,
                         uint64_t fast_window,
                         uint64_t slow_window,
                         uint64_t commit_delay);

void avid_scheduler_schedule_commit(avid_causal_scheduler_t *sched,
                                    uint64_t window_end_round,
                                    const int16_t new_weights_q78[9]);

bool avid_scheduler_poll_commit_ready(avid_causal_scheduler_t *sched,
                                      uint64_t round_now,
                                      int16_t out_weights_q78[9]);
