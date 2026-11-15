/* ===========================================================================
 * causal_scheduler.cpp — Dual-Timescale Causal Epoch Scheduler (A04 / Q01)
 * =========================================================================*/
#include "causal_scheduler.h"
#include <cmath>
#include <string.h>

void avid_scheduler_init(avid_causal_scheduler_t *sched,
                         uint64_t fast_window,
                         uint64_t slow_window,
                         uint64_t commit_delay) {
    if (!sched) {
        return;
    }
    memset(sched, 0, sizeof(*sched));
    sched->fast_window_rounds  = fast_window;
    sched->slow_window_rounds  = slow_window;
    sched->commit_delay_rounds = commit_delay;
}

void avid_scheduler_schedule_commit(avid_causal_scheduler_t *sched,
                                    uint64_t window_end_round,
                                    const int16_t new_weights_q78[9]) {
    if (!sched || !new_weights_q78) {
        return;
    }
    if (sched->has_scheduled_window && window_end_round < sched->last_scheduled_window_end) {
        sched->stale_schedule_drops++;
        return;
    }
    memcpy(sched->pending_weights_q78, new_weights_q78, sizeof(int16_t) * 9);
    sched->pending_commit_round      = window_end_round + sched->commit_delay_rounds;
    sched->has_pending_commit        = true;
    sched->has_scheduled_window      = true;
    sched->last_scheduled_window_end = window_end_round;
}

bool avid_scheduler_poll_commit_ready(avid_causal_scheduler_t *sched,
                                      uint64_t round_now,
                                      int16_t out_weights_q78[9]) {
    if (!sched || !out_weights_q78) {
        return false;
    }
    sched->current_round = round_now;
    if (sched->has_pending_commit && round_now >= sched->pending_commit_round) {
        memcpy(out_weights_q78, sched->pending_weights_q78, sizeof(int16_t) * 9);
        sched->has_pending_commit = false;
        sched->active_epoch_id++;
        return true;
    }
    return false;
}

bool avid_fast_detector_init_checked(avid_fast_burst_detector_t *det,
                                     double density_threshold,
                                     uint32_t debounce_rounds) {
    if (!det) {
        return false;
    }
    memset(det, 0, sizeof(*det));
    det->subwindow_rounds = 16u;

    const bool thresh_ok = std::isfinite(density_threshold) &&
                           (density_threshold > 0.0) &&
                           (density_threshold <= 1.0);
    const bool debounce_ok = (debounce_rounds > 0u) && (debounce_rounds <= 1024u);

    det->density_threshold = thresh_ok ? density_threshold : 0.35;
    det->debounce_rounds   = debounce_ok ? debounce_rounds : 4u;
    det->worst_case_trigger_latency_rounds = det->subwindow_rounds + det->debounce_rounds;
    det->config_valid      = (thresh_ok && debounce_ok);
    return det->config_valid;
}

void avid_fast_detector_init(avid_fast_burst_detector_t *det,
                             double density_threshold,
                             uint32_t debounce_rounds) {
    (void)avid_fast_detector_init_checked(det, density_threshold, debounce_rounds);
}

bool avid_fast_detector_step(avid_fast_burst_detector_t *det,
                             uint64_t round_idx,
                             uint8_t sx,
                             uint8_t sz) {
    if (!det) {
        return false;
    }
    if (det->has_seen_round && round_idx <= det->last_round_idx) {
        det->temporal_regress_count++;
        return false;
    }
    det->has_seen_round = true;
    det->last_round_idx = round_idx;

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
