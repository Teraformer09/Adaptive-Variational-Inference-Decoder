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

void avid_fast_detector_init(avid_fast_burst_detector_t *det,
                             double density_threshold,
                             uint32_t debounce_rounds) {
    memset(det, 0, sizeof(*det));
    det->subwindow_rounds  = 16u;
    det->density_threshold = (density_threshold > 0.0) ? density_threshold : 0.35;
    det->debounce_rounds   = (debounce_rounds > 0u) ? debounce_rounds : 4u;
    det->worst_case_trigger_latency_rounds = det->subwindow_rounds + det->debounce_rounds;
}

bool avid_fast_detector_step(avid_fast_burst_detector_t *det,
                             uint64_t round_idx,
                             uint8_t sx,
                             uint8_t sz) {
    uint32_t slot = (uint32_t)(round_idx & 15u);
    uint8_t old_v = det->ring_nonzero[slot];
    uint8_t new_v = (((sx & 0xFu) != 0u) || ((sz & 0xFu) != 0u)) ? 1u : 0u;
    det->ring_nonzero[slot] = new_v;
    det->ring_sum = det->ring_sum - (uint32_t)old_v + (uint32_t)new_v;

    uint32_t valid_len = (round_idx < 15u) ? (uint32_t)(round_idx + 1u) : 16u;
    double density = (double)det->ring_sum / (double)valid_len;
    if (density > det->peak_density) {
        det->peak_density = density;
    }

    if (round_idx >= 15u && density >= det->density_threshold) {
        det->debounce_counter++;
        if (det->debounce_counter == det->debounce_rounds) {
            det->trigger_count++;
            det->burst_active = true;
            return true;
        }
    } else {
        det->debounce_counter = 0u;
        det->burst_active     = false;
    }
    return false;
}
