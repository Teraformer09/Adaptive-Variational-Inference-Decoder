/* ===========================================================================
 * cycle_accurate_bp_engine.h — Bit-Exact, Cycle-Accurate Q7.8 Min-Sum BP Engine
 * Policy Items: A01 / Q02, A06 / Q06, SF6, SF8
 * Matches `check_node_minsum.sv` and `bp_engine.sv` bit-for-bit.
 * =========================================================================*/
#pragma once
#include <stdint.h>
#include <stdbool.h>

#define BP_N_VARS    9
#define BP_N_CHECKS  4
#define BP_MAX_DEG   4

typedef struct {
    uint8_t  syndrome_mask;
    uint16_t correction_mask;
    int16_t  posteriors_q78[BP_N_VARS];
    int      iterations_K;
    int      cycles_fused;     /* L(K) = 2 + 5*K (17 cycles @ K=3) */
    int      cycles_legacy;    /* L_legacy(K) = 2 + 6*K (20 cycles @ K=3) */
    double   latency_ns_fused; /* 68.0 ns @ 250 MHz, K=3 */
    bool     syndrome_satisfied;
} avid_bp_result_t;

class CycleAccurateBPEngine {
public:
    CycleAccurateBPEngine(int default_K = 3, double clk_mhz = 250.0);

    static int cycle_count_fused(int K)  { return 2 + 5 * K; }
    static int cycle_count_legacy(int K) { return 2 + 6 * K; }

    avid_bp_result_t decode(uint8_t syndrome_mask,
                            const int16_t active_weights_q78[BP_N_VARS],
                            int K = -1) const;

private:
    int    default_K_;
    double clk_period_ns_;
};
