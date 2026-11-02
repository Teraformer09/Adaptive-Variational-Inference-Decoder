/* ===========================================================================
 * stage_profiler.h — Per-Stage Monotonic Latency Profiler (A04 / Q01)
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    const char *stage_name;
    double      deadline_us;
    uint64_t    start_ns;
    double      last_elapsed_us;
    double      max_elapsed_us;
    uint64_t    call_count;
    uint64_t    deadline_misses;
} avid_stage_profiler_t;

void avid_profiler_init(avid_stage_profiler_t *p, const char *name, double deadline_us);
void avid_profiler_begin(avid_stage_profiler_t *p);
bool avid_profiler_end(avid_stage_profiler_t *p);
