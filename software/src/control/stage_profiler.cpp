/* ===========================================================================
 * stage_profiler.cpp — Per-Stage Monotonic Latency Profiler (A04 / Q01)
 * =========================================================================*/
#include "stage_profiler.h"
#include <time.h>

static uint64_t monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void avid_profiler_init(avid_stage_profiler_t *p, const char *name, double deadline_us) {
    p->stage_name      = name;
    p->deadline_us     = deadline_us;
    p->start_ns        = 0;
    p->last_elapsed_us = 0.0;
    p->max_elapsed_us  = 0.0;
    p->call_count      = 0;
    p->deadline_misses = 0;
}

void avid_profiler_begin(avid_stage_profiler_t *p) {
    p->start_ns = monotonic_ns();
}

bool avid_profiler_end(avid_stage_profiler_t *p) {
    uint64_t end_ns = monotonic_ns();
    double elapsed_us = (double)(end_ns - p->start_ns) * 1e-3;
    p->last_elapsed_us = elapsed_us;
    if (elapsed_us > p->max_elapsed_us) {
        p->max_elapsed_us = elapsed_us;
    }
    p->call_count++;
    if (elapsed_us > p->deadline_us) {
        p->deadline_misses++;
        return false;
    }
    return true;
}
