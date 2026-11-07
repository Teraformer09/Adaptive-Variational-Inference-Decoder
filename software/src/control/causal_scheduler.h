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

/* ===========================================================================
 * Fast 16-Round Sliding-Subwindow Burst Detector (system_timing.yaml Contract)
 * - 16-round sliding subwindow
 * - Syndrome-density threshold = 0.35
 * - 4-round debounce
 * - Worst-case trigger latency = 20 rounds (20 us @ 1 MHz)
 * =========================================================================*/
typedef struct {
    uint32_t subwindow_rounds;
    double   density_threshold;
    uint32_t debounce_rounds;
    uint32_t worst_case_trigger_latency_rounds;
    uint8_t  ring_nonzero[16];
    uint32_t ring_sum;
    uint32_t debounce_counter;
    uint32_t trigger_count;
    double   peak_density;
    bool     burst_active;
} avid_fast_burst_detector_t;

void avid_fast_detector_init(avid_fast_burst_detector_t *det,
                             double density_threshold,
                             uint32_t debounce_rounds);

bool avid_fast_detector_step(avid_fast_burst_detector_t *det,
                             uint64_t round_idx,
                             uint8_t sx,
                             uint8_t sz);
