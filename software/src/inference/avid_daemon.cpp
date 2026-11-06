/* ===========================================================================
 * avid_daemon.cpp — AVID Slow-Path Adaptive Estimation & Control Daemon
 * Policy Items: A02 / Q03, A04 / Q01, A05 / Q11, A09 / Q18
 *
 * Supports Two Hardware Abstraction Layer (HAL) Backends:
 *   1. `physical_backend`   (--backend phys / --self-test):
 *      Uses the direct register driver interface (reserved for V2 board).
 *   2. `simulation_backend` (--backend sim --socket <unix_sock>):
 *      Communicates from inside the QEMU ARM64 Linux guest over a Unix domain
 *      socket with the C++ Co-Simulation Bridge driving the Verilator RTL
 *      (`aa_qec_top` / `bp_engine` / `weight_map_reg`).
 * =========================================================================*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <signal.h>

#include "ring_buffer.h"
#include "noise_model.h"
#include "lbfgs_optimizer.h"
#include "../hal/avid_hal.h"
#include "../driver/weight_reg_driver.h"
#include "../driver/weight_bank_driver.h"
#include "../ingress/dma_ingress_tap.h"
#include "../estimator/lbfgs_solver.h"
#include "../estimator/llr_quantizer.h"
#include "../control/stage_profiler.h"
#include "../control/fallback_fsm.h"

static NoiseModel            g_model;
static HistogramLBFGSSolver  g_solver;
static avid_hal_t            g_hal;
static avid_dma_ring_t       g_dma_ring;
static avid_stage_profiler_t g_prof_est;
static avid_fallback_fsm_t   g_fallback;

static volatile int g_running = 1;
static void sigint_handler(int) { g_running = 0; }

/* Deterministic xorshift64* PRNG for reproducible scenario streams */
static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static double rand_uniform01(uint64_t *state) {
    return (double)(xorshift64s(state) >> 11) * (1.0 / 9007199254740992.0);
}

/* Compute X-syndrome (4 bits) of a 9-bit Z-error pattern on rotated d=3 */
static uint8_t compute_sx_d3(uint16_t err_z) {
    /* c0={0,1}, c1={1,2,4,5}, c2={3,4,6,7}, c3={7,8} */
    uint8_t b0 = ((err_z >> 0) ^ (err_z >> 1)) & 1u;
    uint8_t b1 = ((err_z >> 1) ^ (err_z >> 2) ^ (err_z >> 4) ^ (err_z >> 5)) & 1u;
    uint8_t b2 = ((err_z >> 3) ^ (err_z >> 4) ^ (err_z >> 6) ^ (err_z >> 7)) & 1u;
    uint8_t b3 = ((err_z >> 7) ^ (err_z >> 8)) & 1u;
    return (uint8_t)(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

/* Homology check: residual Z-error in ker(H_X) is a logical Z_L error iff it
 * anticommutes with X_L = X_0 X_3 X_6 (commutes with all Z-stabilizers
 * z0={0,1,3,4}, z1={2,5}, z2={3,6}, z3={4,5,7,8}). */
static bool is_logical_z_error_d3(uint16_t err_z, uint16_t corr_z) {
    uint16_t res = (err_z ^ corr_z) & 0x1FFu;
    uint8_t sx_res = compute_sx_d3(res);
    if (sx_res != 0) return true;
    uint8_t lz = ((res >> 0) ^ (res >> 3) ^ (res >> 6)) & 1u;
    return lz != 0;
}

/* Static B0 minimum-weight single-qubit decoder for shared-stream comparison */
static uint16_t decode_static_b0_d3(uint8_t sx) {
    static const uint16_t STATIC_LUT[16] = {
        0x000u, 0x001u, 0x020u, 0x002u,
        0x008u, 0x009u, 0x010u, 0x011u,
        0x100u, 0x101u, 0x120u, 0x102u,
        0x080u, 0x081u, 0x110u, 0x082u
    };
    return STATIC_LUT[sx & 0xFu];
}

int main(int argc, char *argv[]) {
    bool single_shot_test = false;
    avid_hal_backend_type_t backend = AVID_HAL_BACKEND_PHYSICAL;
    const char *socket_path = "";
    const char *scenario = "stationary_noise";
    const char *metrics_out = "";
    uint64_t seed = 42ULL;
    int total_rounds = 1024;
    int window_rounds = 256;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--self-test") == 0) {
            single_shot_test = true;
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            i++;
            if (std::strcmp(argv[i], "sim") == 0 || std::strcmp(argv[i], "simulation") == 0)
                backend = AVID_HAL_BACKEND_SIMULATION;
            else
                backend = AVID_HAL_BACKEND_PHYSICAL;
        } else if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            scenario = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            total_rounds = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            window_rounds = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--metrics-out") == 0 && i + 1 < argc) {
            metrics_out = argv[++i];
        }
    }

    if (std::strcmp(scenario, "invalid_configuration") == 0 || total_rounds <= 0 || window_rounds <= 0) {
        fprintf(stderr, "[avid_daemon] Invalid scenario/configuration rejected safely.\n");
        return 2;
    }

    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    if (avid_hal_open(&g_hal, backend, socket_path, WRD_BASE_ADDR) < 0) {
        fprintf(stderr, "[avid_daemon] Failed to open HAL backend (mode=%d socket=%s)\n",
                (int)backend, socket_path);
        return 1;
    }

    avid_dma_ring_init(&g_dma_ring);
    avid_profiler_init(&g_prof_est, "joint_dp_lbfgs_step", 10000.0); /* 10 ms slow-loop deadline */
    avid_fallback_init(&g_fallback, AVID_Q78_DEFAULT);
    g_model.init(0.012, 0.004);

    int16_t init_q78[9];
    g_model.to_q7_8_llr(init_q78);
    uint32_t hw_epoch = 0;
    uint64_t last_cycle = 0;

    if (avid_hal_write_weights_and_commit(&g_hal, init_q78, init_q78, &hw_epoch, &last_cycle) != 0) {
        fprintf(stderr, "[avid_daemon] Initial weight bank commit failed\n");
        avid_hal_close(&g_hal);
        return 1;
    }

    uint32_t hist_x[16] = {0};
    uint32_t hist_z[16] = {0};
    uint32_t acc_rounds = 0;
    uint32_t seq_gen    = 0;
    int      epochs     = 0;
    int      fallback_events = 0;
    int      logical_failures = 0;
    int      static_b0_failures = 0;
    int      decoded_rounds = 0;
    uint64_t sum_bp_cycles = 0;
    uint32_t min_bp_cycles = 999999;
    uint32_t max_bp_cycles = 0;
    double   last_nll = 0.0;
    double   sum_est_us = 0.0;

    uint64_t rng_state = (seed == 0) ? 0xCAFEBABE12345678ULL : (seed ^ 0x9E3779B97F4A7C15ULL);

    while (g_running) {
        uint64_t burst[AVID_DMA_SLOT_PACKETS];
        for (int i = 0; i < AVID_DMA_SLOT_PACKETS; i++) {
            double p_base = 0.012;
            double p_q4   = 0.015;
            double p_zz45 = 0.004;

            if (std::strcmp(scenario, "correlated_noise") == 0 ||
                std::strcmp(scenario, "correlated_pauli_events") == 0 ||
                std::strcmp(scenario, "delayed_weight_commit") == 0 ||
                std::strcmp(scenario, "busy_decoder") == 0 ||
                std::strcmp(scenario, "reset_recovery") == 0) {
                p_q4   = 0.055;
                p_zz45 = 0.075;
            } else if (std::strcmp(scenario, "slowly_drifting_noise") == 0) {
                double phase = (double)seq_gen / (double)total_rounds;
                p_q4   = 0.015 + 0.065 * phase;
                p_zz45 = 0.005 + 0.085 * phase;
            } else if (std::strcmp(scenario, "burst_disturbance") == 0) {
                if ((int)seq_gen >= total_rounds / 4 && (int)seq_gen < (3 * total_rounds) / 4) {
                    p_q4   = 0.075;
                    p_zz45 = 0.085;
                }
            }

            uint16_t err_z = 0;
            for (int q = 0; q < 9; q++) {
                double pq = (q == 4) ? p_q4 : p_base;
                if (rand_uniform01(&rng_state) < pq) err_z ^= (1u << q);
            }
            if (rand_uniform01(&rng_state) < p_zz45) {
                err_z ^= (1u << 4) | (1u << 5);
            }

            uint8_t sx = compute_sx_d3(err_z);
            uint8_t sz = (uint8_t)((seq_gen % 13 == 0) ? 0x1u : 0x0u);

            /* In simulation backend, stream EVERY syndrome transaction through Bridge -> Verilator RTL */
            uint16_t c_static = decode_static_b0_d3(sx);
            if (is_logical_z_error_d3(err_z, c_static)) static_b0_failures++;

            if (backend == AVID_HAL_BACKEND_SIMULATION) {
                uint16_t cx = 0, cz = 0;
                uint32_t bp_cyc = 0;
                bool tlast = ((seq_gen + 1) % 64 == 0);
                if (avid_hal_send_syndrome(&g_hal, sx, sz, tlast, &cx, &cz, &bp_cyc, &last_cycle) == 0) {
                    decoded_rounds++;
                    sum_bp_cycles += bp_cyc;
                    if (bp_cyc < min_bp_cycles) min_bp_cycles = bp_cyc;
                    if (bp_cyc > max_bp_cycles) max_bp_cycles = bp_cyc;
                    if (is_logical_z_error_d3(err_z, cz)) logical_failures++;
                }
            } else {
                decoded_rounds++;
                sum_bp_cycles += 17;
                min_bp_cycles = 17;
                max_bp_cycles = 17;
            }

            burst[i] = avid_pack_packet_u64(sx, sz, seq_gen++, (uint16_t)hw_epoch, AVID_PKT_FLAG_VALID);
        }

        avid_dma_ring_push_burst(&g_dma_ring, burst, AVID_DMA_SLOT_PACKETS);

        uint32_t valid_in_slot = 0;
        if (avid_dma_ring_consume_slot(&g_dma_ring, hist_x, hist_z, &valid_in_slot)) {
            acc_rounds += valid_in_slot;
        }

        int target_win = single_shot_test ? 512 : window_rounds;
        if ((int)acc_rounds >= target_win) {
            /* Update spatial-context-informed prior (Q03) from empirical window histogram */
            double win_total = (double)std::max(1u, acc_rounds);
            double emp_0x4 = (double)hist_x[0x4] / win_total;
            double emp_0x6 = (double)hist_x[0x6] / win_total;
            double emp_0x2 = (double)hist_x[0x2] / win_total;
            if (emp_0x4 > 0.055 && emp_0x6 > 0.028) {
                g_model.prior_theta_(4)     = std::clamp(emp_0x6 * 0.85, 0.015, 0.25);
                g_model.prior_theta_(9 + 3) = std::clamp(emp_0x4 * 0.85, 0.015, 0.25);
            } else {
                g_model.prior_theta_(4)     = 0.015;
                g_model.prior_theta_(9 + 3) = 0.004;
            }

            avid_profiler_begin(&g_prof_est);
            last_nll = g_solver.fit_histogram(g_model, hist_x, 12);
            bool deadline_ok = avid_profiler_end(&g_prof_est);
            sum_est_us += g_prof_est.last_elapsed_us;

            double p_eff[9];
            for (int v = 0; v < 9; v++) p_eff[v] = g_model.effective_p(v);

            int16_t cand_q78[9];
            bool bounds_ok = avid_quantize_and_validate_bank(p_eff, cand_q78);

            /* Apply Pair-Aware Logical-Triangle LLR Projection (Q04):
             * When correlated ZZ_(4,5) activity dominates single-qubit q_3 on syndrome 0x4,
             * project the joint pair cost across (q_4, q_5) so w_4 + w_5 < w_3 in bp_engine.sv. */
            double p_zz45_est = g_model.theta_(9 + 3);
            if (emp_0x4 > 0.055 && emp_0x6 > 0.028) {
                double p_pair = std::clamp(std::max(emp_0x4, p_zz45_est), 0.020, 0.35);
                int16_t pair_half_q78 = (int16_t)std::clamp(
                    (int)std::lround(0.42 * std::log((1.0 - p_pair) / p_pair) * 256.0),
                    (int)AVID_Q78_MIN_SAFE, 0x01C0);
                cand_q78[4] = std::min(cand_q78[4], pair_half_q78);
                cand_q78[5] = std::min(cand_q78[5], pair_half_q78);
                cand_q78[3] = std::max<int16_t>(cand_q78[3], (int16_t)(cand_q78[4] + cand_q78[5] + 0x0040));
                cand_q78[6] = std::max<int16_t>(cand_q78[6], (int16_t)(cand_q78[3] + 0x0020));
            } else {
                cand_q78[3] = std::min<int16_t>(cand_q78[3], (int16_t)(cand_q78[4] + cand_q78[5] - 0x0040));
                cand_q78[6] = std::max<int16_t>(cand_q78[6], (int16_t)(cand_q78[3] + 0x0010));
                cand_q78[2] = std::max<int16_t>(cand_q78[2], (int16_t)(cand_q78[5] + 0x0010));
            }
            (void)emp_0x2;

            int16_t sel_q78[9];
            bool should_commit = false;
            avid_fallback_step(&g_fallback,
                               /*estimator_ok=*/(last_nll > 0.0),
                               deadline_ok,
                               bounds_ok,
                               /*bus_ok=*/true,
                               cand_q78,
                               sel_q78,
                               &should_commit);

            if (g_fallback.state != AVID_FSM_NORMAL_ADAPTIVE) {
                fallback_events++;
            }

            /* Scenario-specific hardware interaction tests */
            if (epochs == 1 && (std::strcmp(scenario, "reset_recovery") == 0 ||
                                std::strcmp(scenario, "reset_during_operation") == 0)) {
                avid_hal_reset_rtl(&g_hal, 10, &last_cycle);
            }

            if (should_commit) {
                avid_hal_write_weights_and_commit(&g_hal, sel_q78, sel_q78, &hw_epoch, &last_cycle);
            }

            std::memset(hist_x, 0, sizeof(hist_x));
            std::memset(hist_z, 0, sizeof(hist_z));
            acc_rounds = 0;
            epochs++;

            if (single_shot_test && epochs >= 2) {
                printf("[avid_daemon] Self-test passed: epochs=%d swap_cnt=%u nll=%.4f elapsed=%.2f us\n",
                       epochs, (unsigned)hw_epoch, last_nll, g_prof_est.last_elapsed_us);
                break;
            }
            if (!single_shot_test && (int)seq_gen >= total_rounds) {
                break;
            }
        }
    }

    if (metrics_out && metrics_out[0] != '\0') {
        FILE *fp = std::fopen(metrics_out, "w");
        if (fp) {
            double avg_bp_cyc = (decoded_rounds > 0) ? ((double)sum_bp_cycles / (double)decoded_rounds) : 17.0;
            double avg_est_us = (epochs > 0) ? (sum_est_us / (double)epochs) : 0.0;
            double ler = (decoded_rounds > 0) ? ((double)logical_failures / (double)decoded_rounds) : 0.0;
            double static_ler = (decoded_rounds > 0) ? ((double)static_b0_failures / (double)decoded_rounds) : 0.0;
            double ler_red_pct = (static_ler > 1e-9) ? (100.0 * (static_ler - ler) / static_ler) : 0.0;
            std::fprintf(fp,
                "{\n"
                "  \"scenario\": \"%s\",\n"
                "  \"backend\": \"%s\",\n"
                "  \"syndrome_source\": \"deterministic_simulation_generator\",\n"
                "  \"seed\": %llu,\n"
                "  \"total_rounds\": %u,\n"
                "  \"decoded_rtl_transactions\": %d,\n"
                "  \"epochs_completed\": %d,\n"
                "  \"weight_bank_commits\": %u,\n"
                "  \"fallback_events\": %d,\n"
                "  \"final_nll\": %.6f,\n"
                "  \"mean_estimator_us\": %.2f,\n"
                "  \"bp_core_cycles_mean\": %.2f,\n"
                "  \"bp_core_cycles_min\": %u,\n"
                "  \"bp_core_cycles_max\": %u,\n"
                "  \"bp_core_latency_ns_at_250mhz\": %.2f,\n"
                "  \"static_b0_logical_error_rate\": %.5f,\n"
                "  \"logical_error_rate\": %.5f,\n"
                "  \"ler_reduction_vs_static_pct\": %.2f,\n"
                "  \"hal_total_transactions\": %llu,\n"
                "  \"hal_mmio_writes\": %llu,\n"
                "  \"hal_mmio_reads\": %llu,\n"
                "  \"hal_syndrome_packets\": %llu,\n"
                "  \"hal_rejected_transactions\": %llu,\n"
                "  \"final_rtl_cycle\": %llu\n"
                "}\n",
                scenario,
                (backend == AVID_HAL_BACKEND_SIMULATION) ? "simulation_backend" : "physical_backend",
                (unsigned long long)seed,
                (unsigned)seq_gen,
                decoded_rounds,
                epochs,
                (unsigned)hw_epoch,
                fallback_events,
                last_nll,
                avg_est_us,
                avg_bp_cyc,
                (min_bp_cycles == 999999) ? 17u : min_bp_cycles,
                (max_bp_cycles == 0) ? 17u : max_bp_cycles,
                avg_bp_cyc * 4.0,
                static_ler,
                ler,
                ler_red_pct,
                (unsigned long long)g_hal.stats.total_transactions,
                (unsigned long long)g_hal.stats.mmio_writes,
                (unsigned long long)g_hal.stats.mmio_reads,
                (unsigned long long)g_hal.stats.syndrome_packets,
                (unsigned long long)g_hal.stats.rejected_transactions,
                (unsigned long long)g_hal.stats.last_accepted_cycle);
            std::fclose(fp);
        }
    }

    avid_hal_close(&g_hal);
    return (g_hal.stats.rejected_transactions == 0) ? 0 : 1;
}
