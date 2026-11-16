/* ===========================================================================
 * avid_hal_guest_daemon.cpp — Layer 4 QEMU ARM Cortex-A9 / ARM64 Guest Client
 * Scope: Layer 4 ONLY (HAL <-> Unix-Socket RPC Bridge <-> Verilator RTL).
 *
 * Compiles natively for x86_64 and cross-compiles cleanly for:
 *   - 32-bit ARMv7-A Cortex-A9 (arm-linux-gnueabihf, ILP32)
 *   - 64-bit ARMv8-A Cortex-A53 (aarch64-linux-gnu, LP64)
 *
 * Emits deterministic per-transaction guest JSONL traces (--trace-out) and
 * guest execution metrics (--metrics-out) for live QEMU <-> Bridge <-> RTL
 * lockstep verification (L4-LOCK-01..09, Gate G5).
 * =========================================================================*/
#include "../hal/avid_hal.h"
#include "../driver/weight_reg_driver.h"
#include "../driver/weight_bank_driver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char *guest_isa_name() {
#if defined(__aarch64__)
    return "aarch64";
#elif defined(__arm__)
    return "arm_cortex_a9";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "unknown";
#endif
}

static bool is_little_endian_runtime() {
    uint32_t probe = 0x01020304u;
    unsigned char b0 = *reinterpret_cast<const unsigned char *>(&probe);
    return b0 == 0x04u;
}

static uint64_t xorshift64s(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

static uint8_t compute_sx_d3(uint16_t err_z) {
    uint8_t b0 = ((err_z >> 0) ^ (err_z >> 1)) & 1u;
    uint8_t b1 = ((err_z >> 1) ^ (err_z >> 2) ^ (err_z >> 4) ^ (err_z >> 5)) & 1u;
    uint8_t b2 = ((err_z >> 3) ^ (err_z >> 4) ^ (err_z >> 6) ^ (err_z >> 7)) & 1u;
    uint8_t b3 = ((err_z >> 7) ^ (err_z >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

static uint8_t compute_sz_d3(uint16_t err_x) {
    uint8_t b0 = ((err_x >> 0) ^ (err_x >> 1) ^ (err_x >> 3) ^ (err_x >> 4)) & 1u;
    uint8_t b1 = ((err_x >> 2) ^ (err_x >> 5)) & 1u;
    uint8_t b2 = ((err_x >> 3) ^ (err_x >> 6)) & 1u;
    uint8_t b3 = ((err_x >> 4) ^ (err_x >> 5) ^ (err_x >> 7) ^ (err_x >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

static int run_self_test(const char *metrics_out) {
    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR) != 0) {
        std::fprintf(stderr, "[avid_daemon self-test] avid_hal_open(PHYSICAL) failed\n");
        return 1;
    }

    uint64_t cyc = 0;
    uint32_t rdata = 0;
    if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0 || cyc != 10u) return 1;
    if (avid_hal_mmio_read(&hal, WRD_OFF_SHADOW_X(0), &rdata, &cyc) != 0) return 1;
    if ((rdata & 0xFFFFu) != WRD_DEFAULT_Q78) return 1;

    /* Reject invalid/read-only MMIO accesses cleanly */
    if (avid_hal_mmio_write(&hal, WRD_OFF_EPOCH, 99u, 0xFu, &cyc) == 0) return 1;
    if (avid_hal_mmio_write(&hal, WRD_OFF_MAX_ITER, 0u, 0x1u, &cyc) == 0) return 1;
    if (avid_hal_mmio_read(&hal, 0x990u, &rdata, &cyc) == 0) return 1;

    const int16_t wx[9] = {0x0190, 0x0240, 0x0190, 0x0190, 0x0140, 0x0260, 0x0260, 0x0240, 0x0190};
    const int16_t wz[9] = {0x01A0, 0x0250, 0x01A0, 0x01A0, 0x0130, 0x0270, 0x0270, 0x0250, 0x01A0};
    uint32_t epoch_after = 0;
    if (avid_hal_write_weights_and_commit(&hal, wx, wz, &epoch_after, &cyc) != 0) return 1;
    if (epoch_after != 1u) return 1;

    uint16_t corr_x = 0, corr_z = 0;
    uint32_t bp_cycles = 0;
    /* q4 Z-error (sx = 0x6) and q4 X-error (sz = 0x9) simultaneously */
    if (avid_hal_send_syndrome(&hal, 0x6u, 0x9u, true, &corr_x, &corr_z, &bp_cycles, &cyc) != 0) {
        return 1;
    }
    if (corr_x != (1u << 4) || corr_z != (1u << 4) || bp_cycles != 17u) {
        std::fprintf(stderr,
                     "[avid_daemon self-test] decode mismatch corr_x=0x%x corr_z=0x%x bp=%u\n",
                     corr_x, corr_z, bp_cycles);
        return 1;
    }

    avid_hal_close(&hal);

    if (metrics_out && metrics_out[0] != '\0') {
        FILE *fp = std::fopen(metrics_out, "w");
        if (fp) {
            std::fprintf(fp,
                         "{\n"
                         "  \"mode\": \"self_test\",\n"
                         "  \"guest_isa\": \"%s\",\n"
                         "  \"pointer_bits\": %zu,\n"
                         "  \"little_endian\": %s,\n"
                         "  \"final_cycle\": %llu,\n"
                         "  \"status\": \"PASS\"\n"
                         "}\n",
                         guest_isa_name(),
                         sizeof(void *) * 8u,
                         is_little_endian_runtime() ? "true" : "false",
                         static_cast<unsigned long long>(cyc));
            std::fclose(fp);
        }
    }

    std::printf("[avid_daemon] self-test PASS (isa=%s ptr_bits=%zu le=%d final_cycle=%llu)\n",
                guest_isa_name(),
                sizeof(void *) * 8u,
                is_little_endian_runtime() ? 1 : 0,
                static_cast<unsigned long long>(cyc));
    return 0;
}

int main(int argc, char **argv) {
    bool self_test = false;
    bool reset_mid_run = false;
    avid_hal_backend_type_t backend = AVID_HAL_BACKEND_PHYSICAL;
    const char *socket_path = "/tmp/avid_cosim.sock";
    const char *trace_out = "";
    const char *metrics_out = "";
    uint64_t seed = 42ULL;
    uint32_t total_rounds = 64u;
    uint32_t window_rounds = 16u;
    uint32_t k_iters = 3u;
    uint32_t osd_mode = 1u;
    int timeout_ms = 3000;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--self-test") == 0) {
            self_test = true;
        } else if (std::strcmp(argv[i], "--reset-mid-run") == 0) {
            reset_mid_run = true;
        } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char *b = argv[++i];
            if (std::strcmp(b, "sim") == 0 || std::strcmp(b, "simulation") == 0) {
                backend = AVID_HAL_BACKEND_SIMULATION;
            } else if (std::strcmp(b, "physical") == 0 || std::strcmp(b, "phys") == 0) {
                backend = AVID_HAL_BACKEND_PHYSICAL;
            } else {
                std::fprintf(stderr, "[avid_daemon] Invalid --backend: %s\n", b);
                return 2;
            }
        } else if (std::strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (std::strcmp(argv[i], "--trace-out") == 0 && i + 1 < argc) {
            trace_out = argv[++i];
        } else if (std::strcmp(argv[i], "--metrics-out") == 0 && i + 1 < argc) {
            metrics_out = argv[++i];
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            total_rounds = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            window_rounds = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--k-iters") == 0 && i + 1 < argc) {
            k_iters = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--osd-mode") == 0 && i + 1 < argc) {
            osd_mode = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            timeout_ms = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "[avid_daemon] Unknown or incomplete CLI argument: %s\n", argv[i]);
            return 2;
        }
    }

    if (self_test) {
        return run_self_test(metrics_out);
    }

    if (total_rounds == 0u || window_rounds == 0u ||
        k_iters < AVID_HAL_MIN_BP_ITER || k_iters > AVID_HAL_MAX_BP_ITER ||
        osd_mode > 2u) {
        std::fprintf(stderr, "[avid_daemon] Invalid runtime configuration bounds\n");
        return 2;
    }

    avid_hal_t hal;
    if (avid_hal_open(&hal, backend, socket_path, WRD_BASE_ADDR) != 0) {
        std::fprintf(stderr, "[avid_daemon] Failed to open HAL backend (socket=%s)\n", socket_path);
        return 1;
    }
    avid_hal_set_timeout_ms(&hal, timeout_ms);

    FILE *tfp = (trace_out[0] != '\0') ? std::fopen(trace_out, "w") : nullptr;
    auto emit_guest_event = [&](const char *kind, uint64_t tx_before, uint64_t tx_after,
                                uint32_t round_idx, uint32_t epoch, uint8_t sx, uint8_t sz,
                                bool tlast, uint16_t corr_x, uint16_t corr_z, uint32_t bp_cyc,
                                uint64_t accepted_c, uint64_t completed_c, bool irq) {
        if (!tfp) return;
        std::fprintf(tfp,
                     "{\"guest_isa\":\"%s\",\"ptr_bits\":%zu,\"event\":\"%s\","
                     "\"tx_id_first\":%llu,\"tx_id_last\":%llu,\"round\":%u,\"hw_epoch\":%u,"
                     "\"sx\":%u,\"sz\":%u,\"tlast\":%s,\"corr_x\":\"0x%04x\",\"corr_z\":\"0x%04x\","
                     "\"bp_cycles\":%u,\"accepted_cycle\":%llu,\"completed_cycle\":%llu,"
                     "\"epoch_irq\":%u}\n",
                     guest_isa_name(), sizeof(void *) * 8u, kind,
                     static_cast<unsigned long long>(tx_before),
                     static_cast<unsigned long long>(tx_after),
                     round_idx, epoch,
                     static_cast<unsigned>(sx), static_cast<unsigned>(sz),
                     tlast ? "true" : "false",
                     static_cast<unsigned>(corr_x), static_cast<unsigned>(corr_z),
                     bp_cyc,
                     static_cast<unsigned long long>(accepted_c),
                     static_cast<unsigned long long>(completed_c),
                     irq ? 1u : 0u);
        std::fflush(tfp);
    };

    uint64_t cyc = 0;
    uint64_t tx0 = hal.next_tx_id;
    if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0) {
        if (tfp) std::fclose(tfp);
        avid_hal_close(&hal);
        return 1;
    }
    emit_guest_event("RESET", tx0, hal.next_tx_id - 1ULL, 0u, 0u, 0u, 0u, false,
                     0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);

    tx0 = hal.next_tx_id;
    if (avid_hal_set_bp_iterations(&hal, k_iters, &cyc) != 0 ||
        avid_hal_set_osd_mode(&hal, osd_mode, &cyc) != 0) {
        if (tfp) std::fclose(tfp);
        avid_hal_close(&hal);
        return 1;
    }
    emit_guest_event("CONFIG", tx0, hal.next_tx_id - 1ULL, 0u, 0u, 0u, 0u, false,
                     0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);

    int16_t wx[9] = {0x0190, 0x0240, 0x0190, 0x0190, 0x0140, 0x0260, 0x0260, 0x0240, 0x0190};
    int16_t wz[9] = {0x01A0, 0x0250, 0x01A0, 0x01A0, 0x0130, 0x0270, 0x0270, 0x0250, 0x01A0};
    uint32_t hw_epoch = 0u;
    tx0 = hal.next_tx_id;
    if (avid_hal_write_weights_and_commit(&hal, wx, wz, &hw_epoch, &cyc) != 0 || hw_epoch != 1u) {
        if (tfp) std::fclose(tfp);
        avid_hal_close(&hal);
        return 1;
    }
    emit_guest_event("COMMIT", tx0, hal.next_tx_id - 1ULL, 0u, hw_epoch, 0u, 0u, false,
                     0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);

    uint64_t rng = (seed == 0ULL) ? 0x9E3779B97F4A7C15ULL : seed;
    uint32_t syndrome_satisfied = 0u;
    uint32_t irq_observed = 0u;
    uint32_t epoch_commits = 1u;

    for (uint32_t r = 0; r < total_rounds; r++) {
        if (reset_mid_run && r == total_rounds / 2u) {
            tx0 = hal.next_tx_id;
            if (avid_hal_reset_rtl(&hal, 10u, &cyc) != 0) return 1;
            uint32_t ep_after_rst = 99u;
            if (avid_hal_mmio_read(&hal, WRD_OFF_EPOCH, &ep_after_rst, &cyc) != 0 ||
                ep_after_rst != 0u) {
                return 1;
            }
            hw_epoch = 0u;
            emit_guest_event("MID_RUN_RESET", tx0, hal.next_tx_id - 1ULL, r, hw_epoch, 0u, 0u, false,
                             0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);

            if (avid_hal_set_bp_iterations(&hal, k_iters, &cyc) != 0 ||
                avid_hal_set_osd_mode(&hal, osd_mode, &cyc) != 0) {
                return 1;
            }
            tx0 = hal.next_tx_id;
            if (avid_hal_write_weights_and_commit(&hal, wx, wz, &hw_epoch, &cyc) != 0 ||
                hw_epoch != 1u) {
                return 1;
            }
            epoch_commits++;
            emit_guest_event("COMMIT_POST_RESET", tx0, hal.next_tx_id - 1ULL, r, hw_epoch, 0u, 0u,
                             false, 0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);
        }

        uint64_t draw = xorshift64s(&rng);
        uint8_t sx = static_cast<uint8_t>(draw & 0xFu);
        uint8_t sz = static_cast<uint8_t>((draw >> 4) & 0xFu);
        bool tlast = (((r + 1u) % window_rounds) == 0u);

        uint16_t corr_x = 0u, corr_z = 0u;
        uint32_t bp_cyc = 0u;
        tx0 = hal.next_tx_id;
        if (avid_hal_send_syndrome(&hal, sx, sz, tlast, &corr_x, &corr_z, &bp_cyc, &cyc) != 0) {
            if (tfp) std::fclose(tfp);
            avid_hal_close(&hal);
            return 1;
        }
        if (compute_sx_d3(corr_z) == sx && compute_sz_d3(corr_x) == sz) {
            syndrome_satisfied++;
        }
        if (hal.stats.last_epoch_irq) {
            irq_observed++;
        }
        emit_guest_event("SYNDROME", tx0, hal.next_tx_id - 1ULL, r, hw_epoch, sx, sz, tlast,
                         corr_x, corr_z, bp_cyc, hal.stats.last_accepted_cycle,
                         hal.stats.last_completed_cycle, hal.stats.last_epoch_irq);

        if (tlast && (r + 1u) < total_rounds) {
            /* Deterministic slow-loop weight adaptation step across X and Z banks */
            int hot_q = static_cast<int>((r / window_rounds) % 9u);
            for (int q = 0; q < 9; q++) {
                wx[q] = static_cast<int16_t>((q == hot_q) ? 0x0118 : (0x0190 + 8 * q));
                wz[q] = static_cast<int16_t>((q == (8 - hot_q)) ? 0x0120 : (0x01A0 + 8 * q));
            }
            tx0 = hal.next_tx_id;
            uint32_t prev_epoch = hw_epoch;
            if (avid_hal_write_weights_and_commit(&hal, wx, wz, &hw_epoch, &cyc) != 0 ||
                hw_epoch != prev_epoch + 1u) {
                if (tfp) std::fclose(tfp);
                avid_hal_close(&hal);
                return 1;
            }
            /* Read back active_x[hot_q] and active_z[8-hot_q] to verify bank agreement */
            uint32_t act_x = 0u, act_z = 0u;
            if (avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(hot_q), &act_x, &cyc) != 0 ||
                avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(8 - hot_q), &act_z, &cyc) != 0) {
                return 1;
            }
            if ((act_x & 0xFFFFu) != static_cast<uint16_t>(wx[hot_q]) ||
                (act_z & 0xFFFFu) != static_cast<uint16_t>(wz[8 - hot_q])) {
                std::fprintf(stderr, "[avid_daemon] Active weight bank readback mismatch!\n");
                return 1;
            }
            epoch_commits++;
            emit_guest_event("COMMIT", tx0, hal.next_tx_id - 1ULL, r, hw_epoch, 0u, 0u, false,
                             0u, 0u, 0u, hal.stats.last_accepted_cycle, hal.stats.last_completed_cycle, false);
        }
    }

    if (tfp) std::fclose(tfp);

    if (metrics_out[0] != '\0') {
        FILE *mfp = std::fopen(metrics_out, "w");
        if (mfp) {
            std::fprintf(mfp,
                         "{\n"
                         "  \"guest_isa\": \"%s\",\n"
                         "  \"pointer_bits\": %zu,\n"
                         "  \"little_endian\": %s,\n"
                         "  \"backend\": \"%s\",\n"
                         "  \"seed\": %llu,\n"
                         "  \"rounds\": %u,\n"
                         "  \"window_rounds\": %u,\n"
                         "  \"k_iters\": %u,\n"
                         "  \"osd_mode\": %u,\n"
                         "  \"syndrome_satisfied\": %u,\n"
                         "  \"irq_observed\": %u,\n"
                         "  \"epoch_commits\": %u,\n"
                         "  \"final_hw_epoch\": %u,\n"
                         "  \"mmio_writes\": %llu,\n"
                         "  \"mmio_reads\": %llu,\n"
                         "  \"syndrome_packets\": %llu,\n"
                         "  \"rejected_transactions\": %llu,\n"
                         "  \"id_mismatch_errors\": %llu,\n"
                         "  \"final_accepted_cycle\": %llu,\n"
                         "  \"final_completed_cycle\": %llu\n"
                         "}\n",
                         guest_isa_name(),
                         sizeof(void *) * 8u,
                         is_little_endian_runtime() ? "true" : "false",
                         (backend == AVID_HAL_BACKEND_SIMULATION) ? "simulation" : "physical",
                         static_cast<unsigned long long>(seed),
                         total_rounds, window_rounds, k_iters, osd_mode,
                         syndrome_satisfied, irq_observed, epoch_commits, hw_epoch,
                         static_cast<unsigned long long>(hal.stats.mmio_writes),
                         static_cast<unsigned long long>(hal.stats.mmio_reads),
                         static_cast<unsigned long long>(hal.stats.syndrome_packets),
                         static_cast<unsigned long long>(hal.stats.rejected_transactions),
                         static_cast<unsigned long long>(hal.stats.id_mismatch_errors),
                         static_cast<unsigned long long>(hal.stats.last_accepted_cycle),
                         static_cast<unsigned long long>(hal.stats.last_completed_cycle));
            std::fclose(mfp);
        }
    }

    avid_hal_close(&hal);
    return 0;
}
