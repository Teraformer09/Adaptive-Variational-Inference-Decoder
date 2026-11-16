#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "control/r2d2_lstm_controller.h"
#include "control/stage_profiler.h"
#include "custom_engine/cycle_accurate_bp_engine.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include "inference/noise_model.h"
#include "ingress/dma_ingress_tap.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

namespace {

static uint64_t next_u64(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rand01(uint64_t *st) {
    return static_cast<double>(next_u64(st) >> 11) * (1.0 / 9007199254740992.0);
}

static uint8_t sample_sx_from_theta(const NoiseModel::ParamVec &theta, uint64_t *rng) {
    uint8_t sx = 0;
    for (int e = 0; e < NM_N_PARAMS; ++e) {
        if (rand01(rng) < theta(e)) {
            sx ^= NM_EVENT_SIGNATURES[e];
        }
    }
    return sx & 0xFu;
}

static double extract_json_number(const std::string &body, const std::string &key, double def_val = -1.0) {
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def_val;
    pos += pat.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) pos++;
    return std::strtod(body.c_str() + pos, nullptr);
}

static double percentile_sorted(const std::vector<double> &sorted_vals, double pct) {
    if (sorted_vals.empty()) return 0.0;
    double idx = (pct / 100.0) * static_cast<double>(sorted_vals.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (hi >= sorted_vals.size()) hi = sorted_vals.size() - 1;
    double frac = idx - static_cast<double>(lo);
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-026");
    const std::string tid = opts.test_id;

    if (tid == "L3-FLOW-026") {
        TestContext ctx(opts, tid, "observation_window_timing_boundaries",
                        "software/src/inference/avid_daemon.cpp",
                        "main (acc_rounds == window_rounds)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Configure window_rounds in {256, 1000, 10000} and verify exact slow-loop execution cadence without premature adaptation";
        ctx.record().independent_oracle = "total_rounds == epochs_completed * window_rounds and fast_burst_subepoch_adaptations == 0 on stationary stream";

        const int windows[3] = {256, 1000, 10000};
        for (int w : windows) {
            const int rounds = 2 * w;
            std::string mpath = "/tmp/l3_flow026_" + std::to_string(w) + "_" + std::to_string(::getpid()) + ".json";
            std::string cmd = opts.daemon_bin +
                " --scenario stationary_noise --rounds " + std::to_string(rounds) +
                " --window " + std::to_string(w) + " --seed 2026 --backend physical"
                " --checkpoint " + opts.repo_root + "/checkpoints/r2d2_v1_trained.bin"
                " --metrics-out " + mpath;
            SubprocessResult res = run_command_capture(cmd);
            ASSERT_EQ(res.exit_code, 0);

            std::ifstream ifs(mpath);
            std::stringstream ss;
            ss << ifs.rdbuf();
            std::string body = ss.str();
            ::unlink(mpath.c_str());

            int epochs = static_cast<int>(std::lround(extract_json_number(body, "epochs_completed")));
            int sub_adapt = static_cast<int>(std::lround(extract_json_number(body, "fast_burst_subepoch_adaptations")));
            int total_r = static_cast<int>(std::lround(extract_json_number(body, "total_rounds")));
            ctx.log("window=" + std::to_string(w) + " total_rounds=" + std::to_string(total_r) +
                    " epochs=" + std::to_string(epochs) + " sub_adapt=" + std::to_string(sub_adapt));
            ASSERT_EQ(total_r, rounds);
            ASSERT_EQ(epochs, 2);
            ASSERT_EQ(sub_adapt, 0);
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-027") {
        TestContext ctx(opts, tid, "evidence_availability_prefix_causality",
                        "software/src/control/causal_scheduler.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "avid_fast_detector_step + HistogramLBFGSSolver::fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Inject burst starting at round=1000 vs pure stationary stream; verify rounds 0..999 show zero advance knowledge";
        ctx.record().independent_oracle = "Slow MAP estimates at r=256,512,768 and fast detector states across r=0..999 are bit-identical";

        auto run_prefix = [&](bool inject_burst_at_1000, double out_theta4[3], uint32_t *out_trig_before_1000) {
            uint64_t rng = 0x123456789ABCDEF0ULL;
            NoiseModel::ParamVec th_stat, th_burst;
            th_stat.setConstant(0.001);
            for (int q = 0; q < 9; ++q) th_stat(q) = 0.004;
            th_burst = th_stat;
            th_burst(4) = 0.28;
            th_burst(9 + 3) = 0.28;

            avid_fast_burst_detector_t det{};
            avid_fast_detector_init(&det, 0.35, 4u);
            NoiseModel model;
            model.set_basis_z(false);
            model.init(0.012, 0.004);
            HistogramLBFGSSolver solver;
            uint32_t hist[16] = {0};
            int ep_idx = 0;

            for (int r = 0; r < 1200; ++r) {
                const auto &th = (inject_burst_at_1000 && r >= 1000) ? th_burst : th_stat;
                uint8_t sx = sample_sx_from_theta(th, &rng);
                avid_fast_detector_step(&det, static_cast<uint64_t>(r), sx, 0u);
                if (r == 999) {
                    *out_trig_before_1000 = det.trigger_count;
                }
                hist[sx]++;
                if ((r + 1) == 256 || (r + 1) == 512 || (r + 1) == 768) {
                    solver.fit_histogram(model, hist, 20, nullptr);
                    out_theta4[ep_idx++] = model.theta_(4);
                    std::memset(hist, 0, sizeof(hist));
                }
            }
        };

        double th_clean[3] = {0}, th_future_burst[3] = {0};
        uint32_t trig_clean = 99, trig_future_burst = 99;
        run_prefix(false, th_clean, &trig_clean);
        run_prefix(true, th_future_burst, &trig_future_burst);

        ASSERT_EQ(trig_clean, trig_future_burst);
        ASSERT_EQ(trig_future_burst, 0u);
        for (int k = 0; k < 3; ++k) {
            ASSERT_EQ(th_clean[k], th_future_burst[k]);
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-028") {
        TestContext ctx(opts, tid, "commit_delay_enforcement_tc_ge_te_plus_dc",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_schedule_commit -> avid_scheduler_poll_commit_ready", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Set commit_delay=100 after window_end=256 and poll rounds 256..360";
        ctx.record().independent_oracle = "poll_commit_ready is false for 256..355, true strictly once at 356, and false thereafter";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 100u);
        const int16_t cand[9] = {710, 720, 730, 740, 510, 750, 760, 770, 780};
        avid_scheduler_schedule_commit(&sched, 256u, cand);

        int16_t out[9] = {0};
        for (uint64_t r = 256; r <= 355; ++r) {
            ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, r, out));
        }
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 356u, out));
        ASSERT_ARRAY_EQ(out, cand, 9);
        ASSERT_EQ(sched.active_epoch_id, 1u);
        ASSERT_FALSE(avid_scheduler_poll_commit_ready(&sched, 357u, out));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-029") {
        TestContext ctx(opts, tid, "burst_onset_detection_latency_bound",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Start burst at round=100 on avid_fast_burst_detector_t (threshold=0.35, debounce=4) and record trigger round";
        ctx.record().independent_oracle = "16-subwindow reaches >=6/16 at r=105; 4-round debounce triggers at r_trig=108 in [103, 119] (<=20 rounds)";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        int r_trig = -1;
        for (int r = 0; r < 160; ++r) {
            uint8_t sx = (r >= 100) ? 0x6u : 0x0u;
            bool fired = avid_fast_detector_step(&det, static_cast<uint64_t>(r), sx, 0u);
            if (fired && r_trig < 0) {
                r_trig = r;
            }
        }
        ctx.log("first_burst_evidence_round=100 trigger_round=" + std::to_string(r_trig));
        ASSERT_IN_RANGE(r_trig, 103, 119);
        ASSERT_EQ(r_trig, 108);
        ASSERT_EQ(det.trigger_count, 1u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-030") {
        TestContext ctx(opts, tid, "burst_end_window_drain_recovery",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_fast_detector_step", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "End 60-round burst at round=160 (return to sx=0) and record round when burst_active clears";
        ctx.record().independent_oracle = "burst_active clears within <=16 rounds after round=160 once 16-round sliding density drops below 0.35";

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        int r_clear = -1;
        for (int r = 0; r <= 185; ++r) {
            uint8_t sx = (r >= 100 && r < 160) ? 0x6u : 0x0u;
            avid_fast_detector_step(&det, static_cast<uint64_t>(r), sx, 0u);
            if (r == 159) {
                ASSERT_TRUE(det.burst_active);
            }
            if (r >= 160 && !det.burst_active && r_clear < 0) {
                r_clear = r;
            }
        }
        ctx.log("burst_end=160 burst_clear_round=" + std::to_string(r_clear));
        ASSERT_IN_RANGE(r_clear, 161, 176);
        ASSERT_FALSE(det.burst_active);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-031") {
        TestContext ctx(opts, tid, "slow_loop_latency_distribution_vs_10ms_budget",
                        "software/src/control/stage_profiler.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "avid_profiler_begin/end + fit_histogram + step_causal", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Benchmark 100 iterations of dual-basis fit_histogram + R2D2 step_causal + Q7.8 quantization under 10,000 us budget";
        ctx.record().independent_oracle = "p50, p95, p99, max < 10,000 us and profiler.deadline_misses == 0";

        AvidR2D2LSTMController r2d2;
        std::string err;
        ASSERT_TRUE(r2d2.load_checkpoint(opts.repo_root + "/checkpoints/r2d2_v1_trained.bin", &err));

        avid_stage_profiler_t prof{};
#ifdef AVID_SANITIZER_BUILD
        const double slow_budget_us = 250000.0;
#else
        const double slow_budget_us = 10000.0;
#endif
        avid_profiler_init(&prof, "slow_loop_full_epoch", slow_budget_us);

        NoiseModel model_z, model_x;
        model_z.set_basis_z(false);
        model_x.set_basis_z(true);
        HistogramLBFGSSolver solver;

        const uint32_t hx[16] = {210, 4, 3, 5, 4, 2, 14, 1, 3, 1, 2, 1, 3, 1, 1, 1};
        const uint32_t hz[16] = {235, 3, 2, 2, 2, 1, 3, 1, 2, 1, 1, 1, 1, 0, 1, 0};
        std::vector<double> samples_us;
        samples_us.reserve(100);

        for (int iter = 0; iter < 100; ++iter) {
            model_z.init(0.012, 0.004);
            model_x.init(0.005, 0.002);
            avid_profiler_begin(&prof);
            LBFGSFitDiagnostics dz{}, dx{};
            double nll_z = solver.fit_histogram(model_z, hx, 20, &dz);
            solver.fit_histogram(model_x, hz, 12, &dx);
            double gains[9] = {1.0};
            uint32_t rec_k = 3;
            double pair_scale = 1.0;
            r2d2.step_causal(hx, hz, 256, model_z.theta_.data(), nll_z, gains, &rec_k, 0.15, &pair_scale);
            double peff[9];
            int16_t q78[9];
            for (int q = 0; q < 9; ++q) peff[q] = model_z.effective_p(q);
            bool bounds_ok = avid_quantize_and_validate_bank(peff, q78);
            bool ddl_ok = avid_profiler_end(&prof);
            ASSERT_TRUE(bounds_ok);
            ASSERT_TRUE(ddl_ok);
            samples_us.push_back(prof.last_elapsed_us);
        }

        std::sort(samples_us.begin(), samples_us.end());
        double p50 = percentile_sorted(samples_us, 50.0);
        double p95 = percentile_sorted(samples_us, 95.0);
        double p99 = percentile_sorted(samples_us, 99.0);
        double pmax = samples_us.back();
        ctx.log("slow_loop_us: p50=" + std::to_string(p50) + " p95=" + std::to_string(p95) +
                " p99=" + std::to_string(p99) + " max=" + std::to_string(pmax));
        ASSERT_EQ(prof.deadline_misses, 0u);
        ASSERT_IN_RANGE(p99, 0.01, slow_budget_us);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-032") {
        TestContext ctx(opts, tid, "fast_loop_isolation_from_slow_lbfgs",
                        "software/src/custom_engine/cycle_accurate_bp_engine.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "CycleAccurateBPEngine::decode || HistogramLBFGSSolver::fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run HistogramLBFGSSolver::fit_histogram on Thread A concurrently with CycleAccurateBPEngine::decode on Thread B";
        ctx.record().independent_oracle = "Zero shared lock contention; every decode reports cycles_fused == 17 and latency_ns_fused == 68.0";

        std::atomic<bool> slow_done{false};
        std::atomic<int> bad_cycles{0};
        std::atomic<uint64_t> fast_decodes{0};

        std::thread slow_thr([&]() {
            NoiseModel m;
            m.set_basis_z(false);
            HistogramLBFGSSolver solver;
            const uint32_t hx[16] = {195, 5, 4, 8, 5, 3, 18, 2, 4, 2, 3, 2, 3, 1, 1, 0};
            for (int i = 0; i < 120; ++i) {
                m.init(0.012, 0.004);
                solver.fit_histogram(m, hx, 20, nullptr);
            }
            slow_done.store(true, std::memory_order_release);
        });

        std::thread fast_thr([&]() {
            CycleAccurateBPEngine bp(3, 250.0);
            const int16_t w[9] = {1128, 1128, 1128, 1128, 720, 1128, 1128, 1128, 1128};
            uint64_t count = 0;
            while (!slow_done.load(std::memory_order_acquire) || count < 10000) {
                avid_bp_result_t r = bp.decode(static_cast<uint8_t>(count & 0xFu), w, 3);
                if (r.cycles_fused != 17 || std::abs(r.latency_ns_fused - 68.0) > 1e-9) {
                    bad_cycles.fetch_add(1, std::memory_order_relaxed);
                }
                count++;
                if (count >= 50000 && slow_done.load(std::memory_order_acquire)) break;
            }
            fast_decodes.store(count, std::memory_order_release);
        });

        slow_thr.join();
        fast_thr.join();
        ctx.log("fast_decodes=" + std::to_string(fast_decodes.load()) + " bad_cycles=" + std::to_string(bad_cycles.load()));
        ASSERT_EQ(bad_cycles.load(), 0);
        ASSERT_TRUE(fast_decodes.load() >= 10000u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-033") {
        TestContext ctx(opts, tid, "stale_estimate_and_epoch_protection",
                        "software/src/control/causal_scheduler.cpp, software/src/ingress/dma_ingress_tap.cpp",
                        "avid_scheduler_schedule_commit + avid_dma_ring_set_active_epoch", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Attempt to schedule commit with stale window_end < last_scheduled_window_end and push stale-epoch DMA packets";
        ctx.record().independent_oracle = "Scheduler increments stale_schedule_drops without overwriting fresh weights; DMA ring increments stale_epoch_drops";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 10u);
        const int16_t fresh_w[9] = {900, 900, 900, 900, 520, 900, 900, 900, 900};
        const int16_t stale_w[9] = {1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200};
        avid_scheduler_schedule_commit(&sched, 512u, fresh_w);
        avid_scheduler_schedule_commit(&sched, 256u, stale_w);
        ASSERT_EQ(sched.stale_schedule_drops, 1u);

        int16_t out[9] = {0};
        ASSERT_TRUE(avid_scheduler_poll_commit_ready(&sched, 522u, out));
        ASSERT_ARRAY_EQ(out, fresh_w, 9);

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        avid_dma_ring_set_active_epoch(&ring, 4u);
        uint64_t pkts[2] = {
            avid_pack_packet_u64(0x6u, 0x0u, 0u, 2u, AVID_PKT_FLAG_VALID), /* stale epoch 2 < 4 */
            avid_pack_packet_u64(0x3u, 0x0u, 1u, 4u, AVID_PKT_FLAG_VALID)  /* active epoch 4 */
        };
        ASSERT_TRUE(avid_dma_ring_push_burst(&ring, pkts, 2u));
        uint32_t hx[16] = {0}, hz[16] = {0}, valid_n = 0;
        ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &valid_n));
        ASSERT_EQ(ring.stale_epoch_drops, 1u);
        ASSERT_EQ(valid_n, 1u);
        ASSERT_EQ(hx[6], 0u);
        ASSERT_EQ(hx[3], 1u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-034") {
        TestContext ctx(opts, tid, "short_burst_60us_vs_10ms_dilution_and_fast_detection",
                        "software/src/control/causal_scheduler.cpp, software/src/inference/avid_daemon.cpp",
                        "avid_fast_detector_step vs 10000-round slow MAP", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Inject 60-round burst at r=9940..9999 inside a 10,000-round window; compare fast 16-round detector vs slow MAP dilution";
        ctx.record().independent_oracle = "Fast detector triggers in <=20 rounds (r<=9960), while 10,000-round slow MAP shift is <0.015 due to 0.6% dilution";

        uint64_t rng_stat = 0x55AA55AA12345678ULL;
        uint64_t rng_mix  = 0x55AA55AA12345678ULL;
        NoiseModel::ParamVec th_stat, th_burst;
        th_stat.setConstant(0.004);
        for (int q = 0; q < 9; ++q) th_stat(q) = 0.012;
        th_burst = th_stat;
        th_burst(4) = 0.35;
        th_burst(9 + 3) = 0.35;

        uint32_t hist_stat[16] = {0}, hist_mix[16] = {0};
        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        int r_trig = -1;

        for (int r = 0; r < 10000; ++r) {
            uint8_t sx_s = sample_sx_from_theta(th_stat, &rng_stat);
            hist_stat[sx_s]++;

            uint8_t sx_m = sample_sx_from_theta((r >= 9940) ? th_burst : th_stat, &rng_mix);
            hist_mix[sx_m]++;
            if (avid_fast_detector_step(&det, static_cast<uint64_t>(r), sx_m, 0u) && r >= 9940 && r_trig < 0) {
                r_trig = r;
            }
        }

        NoiseModel m_stat, m_mix;
        m_stat.set_basis_z(false);
        m_mix.set_basis_z(false);
        m_stat.init(0.012, 0.004);
        m_mix.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        solver.fit_histogram(m_stat, hist_stat, 20, nullptr);
        solver.fit_histogram(m_mix, hist_mix, 20, nullptr);

        double slow_shift = std::abs(m_mix.theta_(4) - m_stat.theta_(4));
        ctx.log("burst_start=9940 fast_trigger_round=" + std::to_string(r_trig) +
                " slow_10ms_theta4_shift=" + std::to_string(slow_shift));
        ASSERT_IN_RANGE(r_trig, 9943, 9960);
        ASSERT_IN_RANGE(slow_shift, 0.0, 0.015);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-035") {
        TestContext ctx(opts, tid, "system_timing_yaml_claim_cross_validation",
                        "architecture/contracts/system_timing.yaml, software/src/inference/avid_daemon.cpp",
                        "avid_fast_detector_init + CycleAccurateBPEngine + avid_daemon", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Cross-check architecture/contracts/system_timing.yaml parameters against live C++ structures and daemon telemetry";
        ctx.record().independent_oracle = "subwindow=16, debounce=4, worst_case=20 rounds, v1_window=256, v2_window=10000, BP K=3 cycles=17 (68 ns), AXIS=22 (88 ns)";

        std::ifstream yfs(opts.repo_root + "/architecture/contracts/system_timing.yaml");
        std::stringstream yss;
        yss << yfs.rdbuf();
        std::string ybody = yss.str();
        ASSERT_TRUE(ybody.find("sliding_subwindow_rounds: 16") != std::string::npos);
        ASSERT_TRUE(ybody.find("debounce_rounds: 4") != std::string::npos);
        ASSERT_TRUE(ybody.find("worst_case_trigger_latency_rounds: 20") != std::string::npos);
        ASSERT_TRUE(ybody.find("v1_emulation_window_rounds: 256") != std::string::npos);
        ASSERT_TRUE(ybody.find("v2_production_window_rounds: 10000") != std::string::npos);
        ASSERT_TRUE(ybody.find("k3_core_cycles_mode_0_1: 17") != std::string::npos);
        ASSERT_TRUE(ybody.find("k3_axis_total_cycles_mode_0_1: 22") != std::string::npos);

        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        ASSERT_EQ(det.subwindow_rounds, 16u);
        ASSERT_EQ(det.debounce_rounds, 4u);
        ASSERT_EQ(det.worst_case_trigger_latency_rounds, 20u);
        ASSERT_EQ(CycleAccurateBPEngine::cycle_count_fused(3), 17);

        std::string mpath = "/tmp/l3_flow035_" + std::to_string(::getpid()) + ".json";
        std::string cmd = opts.daemon_bin +
            " --scenario stationary_noise --rounds 512 --window 256 --seed 2026 --backend physical --controller-mode static"
            " --metrics-out " + mpath;
        SubprocessResult res = run_command_capture(cmd);
        ASSERT_EQ(res.exit_code, 0);

        std::ifstream ifs(mpath);
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string body = ss.str();
        ::unlink(mpath.c_str());

        ASSERT_NEAR(extract_json_number(body, "bp_core_cycles_mean"), 17.0, 1e-6);
        ASSERT_NEAR(extract_json_number(body, "bp_core_latency_ns_at_250mhz"), 68.0, 1e-6);
        ASSERT_NEAR(extract_json_number(body, "axis_total_cycles_mean"), 22.0, 1e-6);
        ASSERT_NEAR(extract_json_number(body, "axis_total_latency_ns_at_250mhz"), 88.0, 1e-6);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
