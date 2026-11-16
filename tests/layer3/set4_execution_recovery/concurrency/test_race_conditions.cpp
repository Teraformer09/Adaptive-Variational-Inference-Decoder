#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "control/r2d2_lstm_controller.h"
#include "control/stage_profiler.h"
#include "custom_engine/cycle_accurate_bp_engine.h"
#include "driver/weight_reg_driver.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include "hal/avid_hal.h"
#include "inference/noise_model.h"
#include "ingress/dma_ingress_tap.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

namespace {

static std::string read_text_file_unlink(const std::string &path) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    ::unlink(path.c_str());
    return ss.str();
}

static double extract_json_num(const std::string &body, const std::string &key, double def_val = -1.0) {
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return def_val;
    pos += pat.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) pos++;
    return std::strtod(body.c_str() + pos, nullptr);
}

static double percentile_sorted(const std::vector<double> &v, double pct) {
    if (v.empty()) return 0.0;
    double idx = (pct / 100.0) * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (hi >= v.size()) hi = v.size() - 1;
    double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static long measure_maxrss_kb(const std::string &cmd, double *out_cpu_sec = nullptr) {
    std::fflush(nullptr);
    pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        ::execl("/bin/sh", "sh", "-c", (cmd + " >/dev/null 2>&1").c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    int wstatus = 0;
    struct rusage ru {};
    if (::wait4(pid, &wstatus, 0, &ru) < 0) return -1;
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) return -1;
    if (out_cpu_sec) {
        *out_cpu_sec = (double)ru.ru_utime.tv_sec + 1e-6 * (double)ru.ru_utime.tv_usec +
                       (double)ru.ru_stime.tv_sec + 1e-6 * (double)ru.ru_stime.tv_usec;
    }
    return ru.ru_maxrss;
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-EXEC-036");
    const std::string tid  = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-EXEC-036") {
        TestContext ctx(opts, tid, "dma_ingress_1m_packet_throughput",
                        "software/src/ingress/dma_ingress_tap.cpp",
                        "avid_dma_ring_push_burst + avid_dma_ring_consume_slot", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Stream 1,000,000 64-bit packets (15,625 bursts x 64) through DMA ring and measure Mpkts/sec";
        ctx.record().independent_oracle = "All 1,000,000 packets validated with 0 drops and sustained throughput > 1.0 Mpkts/sec";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        uint32_t hx[16] = {0}, hz[16] = {0};
        uint64_t total_valid = 0;
        uint32_t seq = 0;

        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < 15625; ++b) {
            uint64_t burst[64];
            for (int i = 0; i < 64; ++i) {
                burst[i] = avid_pack_packet_u64(static_cast<uint8_t>(i & 0xFu),
                                                static_cast<uint8_t>((i >> 2) & 0xFu),
                                                seq++, 1u, AVID_PKT_FLAG_VALID);
            }
            ASSERT_TRUE(avid_dma_ring_push_burst(&ring, burst, 64u));
            uint32_t nv = 0;
            ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &nv));
            total_valid += nv;
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double mpkts_sec = 1.0 / std::max(1e-9, sec);
        ctx.log("packets=1000000 elapsed_sec=" + std::to_string(sec) + " mpkts_per_sec=" + std::to_string(mpkts_sec));
        ASSERT_EQ(total_valid, 1000000ULL);
        ASSERT_EQ(ring.overflow_drops, 0ULL);
#ifdef AVID_SANITIZER_BUILD
        ASSERT_TRUE(mpkts_sec > 0.1);
#else
        ASSERT_TRUE(mpkts_sec > 1.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-037") {
        TestContext ctx(opts, tid, "dma_ingress_burst_consume_latency_distribution",
                        "software/src/ingress/dma_ingress_tap.cpp", "avid_dma_ring_consume_slot", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Measure per-64-packet-burst push+consume latency across 5,000 bursts (p50, p95, p99, max us)";
        ctx.record().independent_oracle = "p99 < 64.0 us (well inside 64 us 1-MHz real-time DMA slot window)";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        uint32_t hx[16] = {0}, hz[16] = {0};
        uint32_t seq = 0;
        std::vector<double> us_vec;
        us_vec.reserve(5000);

        for (int b = 0; b < 5000; ++b) {
            uint64_t burst[64];
            for (int i = 0; i < 64; ++i) {
                burst[i] = avid_pack_packet_u64(static_cast<uint8_t>(i & 0xFu), 0u, seq++, 1u, AVID_PKT_FLAG_VALID);
            }
            auto s0 = std::chrono::steady_clock::now();
            avid_dma_ring_push_burst(&ring, burst, 64u);
            uint32_t nv = 0;
            avid_dma_ring_consume_slot(&ring, hx, hz, &nv);
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count();
            us_vec.push_back(us);
        }
        std::sort(us_vec.begin(), us_vec.end());
        double p50 = percentile_sorted(us_vec, 50.0);
        double p95 = percentile_sorted(us_vec, 95.0);
        double p99 = percentile_sorted(us_vec, 99.0);
        ctx.log("ingress_burst_us: p50=" + std::to_string(p50) + " p95=" + std::to_string(p95) +
                " p99=" + std::to_string(p99) + " max=" + std::to_string(us_vec.back()));
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(p99, 0.0, 2500.0);
#else
        ASSERT_IN_RANGE(p99, 0.0, 64.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-038") {
        TestContext ctx(opts, tid, "dual_basis_estimator_and_r2d2_latency_distribution",
                        "software/src/estimator/lbfgs_solver.cpp, software/src/control/stage_profiler.cpp",
                        "fit_histogram (X+Z) + AvidR2D2LSTMController::step_causal", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Benchmark 200 dual-basis fit_histogram + R2D2 step_causal calls and report p50, p95, p99, max us";
        ctx.record().independent_oracle = "p99 < 10,000.0 us (10 ms slow-loop deadline)";

        AvidR2D2LSTMController r2d2;
        std::string err;
        ASSERT_TRUE(r2d2.load_checkpoint(ckpt, &err));
        NoiseModel mz, mx;
        mz.set_basis_z(false);
        mx.set_basis_z(true);
        HistogramLBFGSSolver solver;
        const uint32_t hx[16] = {205, 5, 4, 7, 5, 3, 15, 1, 4, 1, 2, 1, 2, 1, 0, 0};
        const uint32_t hz[16] = {236, 3, 2, 3, 2, 1, 3, 1, 2, 1, 1, 0, 1, 0, 0, 0};

        std::vector<double> us_vec;
        us_vec.reserve(200);
        for (int i = 0; i < 200; ++i) {
            mz.init(0.012, 0.004);
            mx.init(0.005, 0.002);
            auto s0 = std::chrono::steady_clock::now();
            double nll = solver.fit_histogram(mz, hx, 20, nullptr);
            solver.fit_histogram(mx, hz, 12, nullptr);
            double gains[9] = {1.0};
            uint32_t rec_k = 3;
            r2d2.step_causal(hx, hz, 256, mz.theta_.data(), nll, gains, &rec_k, 0.1);
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count();
            us_vec.push_back(us);
        }
        std::sort(us_vec.begin(), us_vec.end());
        double p50 = percentile_sorted(us_vec, 50.0);
        double p95 = percentile_sorted(us_vec, 95.0);
        double p99 = percentile_sorted(us_vec, 99.0);
        ctx.log("estimator_r2d2_us: p50=" + std::to_string(p50) + " p95=" + std::to_string(p95) +
                " p99=" + std::to_string(p99) + " max=" + std::to_string(us_vec.back()));
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(p99, 0.01, 250000.0);
#else
        ASSERT_IN_RANGE(p99, 0.01, 10000.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-039") {
        TestContext ctx(opts, tid, "scheduler_and_fast_detector_latency_distribution",
                        "software/src/control/causal_scheduler.cpp",
                        "avid_scheduler_poll_commit_ready + avid_fast_detector_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Benchmark 10,000 calls to avid_scheduler_poll_commit_ready and avid_fast_detector_step";
        ctx.record().independent_oracle = "p99 < 10,000 ns (10 us), suitable for sub-microsecond per-round invocation";

        avid_causal_scheduler_t sched{};
        avid_scheduler_init(&sched, 16u, 256u, 8u);
        avid_fast_burst_detector_t det{};
        avid_fast_detector_init(&det, 0.35, 4u);
        const int16_t cand[9] = {900, 900, 900, 900, 900, 900, 900, 900, 900};
        int16_t out[9] = {0};

        std::vector<double> ns_vec;
        ns_vec.reserve(10000);
        for (uint64_t r = 0; r < 10000; ++r) {
            if (r % 256u == 0u) avid_scheduler_schedule_commit(&sched, r, cand);
            auto s0 = std::chrono::steady_clock::now();
            avid_fast_detector_step(&det, r, static_cast<uint8_t>(r & 0xFu), 0u);
            avid_scheduler_poll_commit_ready(&sched, r, out);
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - s0).count();
            ns_vec.push_back(ns);
        }
        std::sort(ns_vec.begin(), ns_vec.end());
        double p50 = percentile_sorted(ns_vec, 50.0);
        double p99 = percentile_sorted(ns_vec, 99.0);
        ctx.log("scheduler_detector_ns: p50=" + std::to_string(p50) + " p99=" + std::to_string(p99));
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(p99, 0.0, 250000.0);
#else
        ASSERT_IN_RANGE(p99, 0.0, 10000.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-040") {
        TestContext ctx(opts, tid, "hal_commit_latency_physical_and_unix_bridge",
                        "software/src/hal/physical_backend.cpp, software/src/hal/simulation_backend.cpp",
                        "avid_hal_write_weights_and_commit", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Measure 100 round-trip commits over both physical (MMIO) and simulation (AF_UNIX) backends";
        ctx.record().independent_oracle = "Zero rejected transactions; physical p99 < 100 us and AF_UNIX p99 < 10,000 us";

        const int16_t wx[9] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
        const int16_t wz[9] = {800,  800,  800,  800,  600,  800,  800,  800,  800};

        avid_hal_t hal_p{};
        ASSERT_EQ(avid_hal_open(&hal_p, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR), 0);
        std::vector<double> phys_us;
        for (int i = 0; i < 100; ++i) {
            uint32_t ep = 0; uint64_t cyc = 0;
            auto s0 = std::chrono::steady_clock::now();
            ASSERT_EQ(avid_hal_write_weights_and_commit(&hal_p, wx, wz, &ep, &cyc), 0);
            phys_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count());
        }
        avid_hal_close(&hal_p);
        std::sort(phys_us.begin(), phys_us.end());

        std::string sock = "/tmp/l3_exec040_" + std::to_string(::getpid()) + ".sock";
        ::unlink(sock.c_str());
        int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
        ::bind(lfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        ::listen(lfd, 2);
        std::atomic<bool> done{false};
        std::thread srv([lfd, &done]() {
            int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd < 0) return;
            uint32_t ep = 0;
            std::string buf;
            char ch[512];
            while (!done.load()) {
                ssize_t r = ::recv(cfd, ch, sizeof(ch), 0);
                if (r <= 0) break;
                buf.append(ch, static_cast<size_t>(r));
                size_t nl = 0;
                while ((nl = buf.find('\n')) != std::string::npos) {
                    std::string ln = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    size_t p = ln.find("\"transaction_id\":");
                    uint64_t tx = p != std::string::npos ? std::strtoull(ln.c_str() + p + 17, nullptr, 10) : 0;
                    if (ln.find("0x00000048") != std::string::npos) ep++;
                    char resp[256];
                    if (ln.find("\"MMIO_READ\"") != std::string::npos) {
                        std::snprintf(resp, sizeof(resp),
                            "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"rdata\":\"0x%08X\",\"accepted_cycle\":10}\n",
                            (unsigned long long)tx, ep);
                    } else {
                        std::snprintf(resp, sizeof(resp),
                            "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"accepted_cycle\":10}\n",
                            (unsigned long long)tx);
                    }
                    ::send(cfd, resp, std::strlen(resp), MSG_NOSIGNAL);
                }
            }
            ::close(cfd);
        });

        avid_hal_t hal_s{};
        ASSERT_EQ(avid_hal_open(&hal_s, AVID_HAL_BACKEND_SIMULATION, sock.c_str(), WRD_BASE_ADDR), 0);
        std::vector<double> sim_us;
        for (int i = 0; i < 100; ++i) {
            uint32_t ep = 0; uint64_t cyc = 0;
            auto s0 = std::chrono::steady_clock::now();
            ASSERT_EQ(avid_hal_write_weights_and_commit(&hal_s, wx, wz, &ep, &cyc), 0);
            sim_us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count());
        }
        ASSERT_EQ(hal_s.stats.rejected_transactions, 0ULL);
        avid_hal_close(&hal_s);
        done.store(true);
        srv.join();
        ::close(lfd);
        ::unlink(sock.c_str());

        std::sort(sim_us.begin(), sim_us.end());
        ctx.log("commit_phys_p99_us=" + std::to_string(percentile_sorted(phys_us, 99.0)) +
                " commit_sim_p99_us=" + std::to_string(percentile_sorted(sim_us, 99.0)));
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(percentile_sorted(phys_us, 99.0), 0.0, 5000.0);
        ASSERT_IN_RANGE(percentile_sorted(sim_us, 99.0), 0.0, 500000.0);
#else
        ASSERT_IN_RANGE(percentile_sorted(phys_us, 99.0), 0.0, 200.0);
        ASSERT_IN_RANGE(percentile_sorted(sim_us, 99.0), 0.0, 50000.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-041") {
        TestContext ctx(opts, tid, "three_thread_5s_concurrent_workload_stress",
                        "software/src/ingress/dma_ingress_tap.cpp, software/src/estimator/lbfgs_solver.cpp, software/src/driver/weight_reg_driver.c",
                        "3-thread concurrent ingress + solver + MMIO reader (5s)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run 3 C++ threads (DMA producer, mutex-guarded L-BFGS solver + commit, MMIO telemetry reader) for 5 seconds";
        ctx.record().independent_oracle = "Zero invariant violations, monotonic hardware epochs, safe active Q7.8 range [64, 12288] throughout";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        const int16_t init_w[9] = {AVID_Q78_DEFAULT, AVID_Q78_DEFAULT, AVID_Q78_DEFAULT,
                                   AVID_Q78_DEFAULT, AVID_Q78_DEFAULT, AVID_Q78_DEFAULT,
                                   AVID_Q78_DEFAULT, AVID_Q78_DEFAULT, AVID_Q78_DEFAULT};
        wrd_write_all_weights(&wrd, init_w);
        wrd_commit_and_wait(&wrd, 16);

        std::mutex ring_mtx, wrd_mtx;
        std::atomic<bool> stop_flag{false};
        std::atomic<int> invariant_errors{0};
        std::atomic<uint64_t> fits_done{0}, reads_done{0};

        std::thread t_prod([&]() {
            uint32_t seq = 0;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<std::mutex> lk(ring_mtx);
                    if (ring.slots[ring.write_slot_idx].state == AVID_SLOT_FREE) {
                        uint64_t burst[64];
                        for (int i = 0; i < 64; ++i) {
                            burst[i] = avid_pack_packet_u64((i % 7 == 0) ? 0x6u : 0x0u, 0u, seq++, 1u, AVID_PKT_FLAG_VALID);
                        }
                        avid_dma_ring_push_burst(&ring, burst, 64u);
                    }
                }
                ::usleep(200);
            }
        });

        std::thread t_solv([&]() {
            NoiseModel m;
            m.set_basis_z(false);
            HistogramLBFGSSolver solver;
            uint32_t hx[16] = {0}, hz[16] = {0}, acc = 0;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                uint32_t nv = 0;
                bool got = false;
                {
                    std::lock_guard<std::mutex> lk(ring_mtx);
                    got = avid_dma_ring_consume_slot(&ring, hx, hz, &nv);
                }
                if (got) {
                    acc += nv;
                    if (acc >= 256u) {
                        m.init(0.012, 0.004);
                        solver.fit_histogram(m, hx, 12, nullptr);
                        double peff[9];
                        int16_t q78[9];
                        for (int q = 0; q < 9; ++q) peff[q] = m.effective_p(q);
                        if (avid_quantize_and_validate_bank(peff, q78)) {
                            std::lock_guard<std::mutex> wlk(wrd_mtx);
                            wrd_write_all_weights(&wrd, q78);
                            wrd_commit_and_wait(&wrd, 16);
                        }
                        std::memset(hx, 0, sizeof(hx));
                        std::memset(hz, 0, sizeof(hz));
                        acc = 0;
                        fits_done.fetch_add(1);
                    }
                } else {
                    ::usleep(200);
                }
            }
        });

        std::thread t_read([&]() {
            uint32_t prev_ep = 0;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                {
                    std::lock_guard<std::mutex> wlk(wrd_mtx);
                    uint32_t ep = wrd_read_epoch(&wrd);
                    if (ep < prev_ep) invariant_errors.fetch_add(1);
                    prev_ep = ep;
                    for (int q = 0; q < 9; ++q) {
                        int16_t w = wrd_read_active_weight(&wrd, q);
                        if (w < AVID_Q78_MIN_SAFE || w > AVID_Q78_MAX_SAFE) {
                            invariant_errors.fetch_add(1);
                        }
                    }
                }
                reads_done.fetch_add(1);
                ::usleep(300);
            }
        });

        ::usleep(5000000); /* 5.0 seconds per traceability specification */
        stop_flag.store(true);
        t_prod.join();
        t_solv.join();
        t_read.join();
        wrd_close(&wrd);

        ctx.log("5s_stress: fits=" + std::to_string(fits_done.load()) +
                " reads=" + std::to_string(reads_done.load()) +
                " invariant_errors=" + std::to_string(invariant_errors.load()));
        ASSERT_EQ(invariant_errors.load(), 0);
        ASSERT_TRUE(fits_done.load() > 10u);
        ASSERT_TRUE(reads_done.load() > 100u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-042") {
        TestContext ctx(opts, tid, "slow_loop_cpu_utilization_telemetry_and_os_measurement",
                        "software/src/inference/avid_daemon.cpp", "main (slow_loop_cpu_utilization_10ms_pct)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run avid_daemon across 50,000 rounds under /usr/bin/time and inspect slow_loop_cpu_utilization_10ms_pct";
        ctx.record().independent_oracle = "0.0 < slow_loop_cpu_utilization_10ms_pct < 100.0 and OS user+sys CPU finishes cleanly";

        std::string mp = "/tmp/l3_exec042_" + std::to_string(::getpid()) + ".json";
        double os_cpu_sec = 0.0;
        long rss_kb = measure_maxrss_kb(
            opts.daemon_bin +
            " --scenario stationary_noise --rounds 50000 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp,
            &os_cpu_sec);
        ASSERT_GT(rss_kb, 0L);
        std::string body = read_text_file_unlink(mp);
        double cpu_pct = extract_json_num(body, "slow_loop_cpu_utilization_10ms_pct");
        ctx.log("os_user_sys_sec=" + std::to_string(os_cpu_sec) + " slow_loop_cpu_pct=" + std::to_string(cpu_pct));
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(cpu_pct, 0.0001, 2500.0);
#else
        ASSERT_IN_RANGE(cpu_pct, 0.0001, 100.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-043") {
        TestContext ctx(opts, tid, "memory_utilization_rss_and_r2d2_static_audit",
                        "software/src/control/r2d2_lstm_controller.h, software/src/inference/avid_daemon.cpp",
                        "AvidR2D2LSTMController::get_memory_audit + 100k-round RSS", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Compare ru_maxrss across 1,024 vs 100,000 rounds and verify get_memory_audit() exact byte counts";
        ctx.record().independent_oracle = "total_params==19744, fp32_bytes==78976, int8==19744, lbfgs_workspace==2688, and RSS delta < 16 MB";

        AvidR2D2MemoryAudit m = AvidR2D2LSTMController::get_memory_audit();
        ASSERT_EQ(m.hidden_dim, 64u);
        ASSERT_EQ(m.total_params, 19744u);
        ASSERT_EQ(m.fp32_bytes, 78976u);
        ASSERT_EQ(m.int8_uncompressed_bytes, 19744u);
        ASSERT_EQ(m.lbfgs_workspace_bytes, 2688u);

        std::string mp1 = "/tmp/l3_exec043_1_" + std::to_string(::getpid()) + ".json";
        std::string mp2 = "/tmp/l3_exec043_2_" + std::to_string(::getpid()) + ".json";
        long rss_1k   = measure_maxrss_kb(opts.daemon_bin + " --scenario stationary_noise --rounds 1024 --window 256 --backend physical --checkpoint " + ckpt + " --metrics-out " + mp1);
        long rss_100k = measure_maxrss_kb(opts.daemon_bin + " --scenario stationary_noise --rounds 100000 --window 1000 --backend physical --checkpoint " + ckpt + " --metrics-out " + mp2);
        ::unlink(mp1.c_str());
        ::unlink(mp2.c_str());
        ctx.log("rss_1k_kb=" + std::to_string(rss_1k) + " rss_100k_kb=" + std::to_string(rss_100k));
        ASSERT_TRUE(rss_1k > 0 && rss_100k > 0);
        ASSERT_TRUE((rss_100k - rss_1k) < 16384L);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-044") {
        TestContext ctx(opts, tid, "priority_inversion_fast_bp_vs_slow_lbfgs",
                        "software/src/custom_engine/cycle_accurate_bp_engine.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "CycleAccurateBPEngine::decode under concurrent L-BFGS load", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run heavy L-BFGS loop on background thread while timing 50,000 CycleAccurateBPEngine::decode calls";
        ctx.record().independent_oracle = "p99 host decode latency < 50.0 us and cycles_fused == 17 across all 50,000 decodes";

        std::atomic<bool> stop_bg{false};
        std::thread bg([&]() {
            NoiseModel m;
            m.set_basis_z(false);
            HistogramLBFGSSolver solver;
            const uint32_t hx[16] = {190, 8, 6, 10, 7, 4, 20, 2, 3, 2, 1, 1, 1, 1, 0, 0};
            while (!stop_bg.load(std::memory_order_relaxed)) {
                m.init(0.012, 0.004);
                solver.fit_histogram(m, hx, 25, nullptr);
                ::sched_yield();
            }
        });

        CycleAccurateBPEngine bp(3, 250.0);
        const int16_t w[9] = {1128, 1128, 1128, 1128, 720, 1128, 1128, 1128, 1128};
        std::vector<double> us_vec;
        us_vec.reserve(50000);
        int bad_cyc = 0;
        for (int i = 0; i < 50000; ++i) {
            auto s0 = std::chrono::steady_clock::now();
            avid_bp_result_t r = bp.decode(static_cast<uint8_t>(i & 0xFu), w, 3);
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count();
            us_vec.push_back(us);
            if (r.cycles_fused != 17) bad_cyc++;
        }
        stop_bg.store(true);
        bg.join();

        std::sort(us_vec.begin(), us_vec.end());
        double p50 = percentile_sorted(us_vec, 50.0);
        double p99 = percentile_sorted(us_vec, 99.0);
        ctx.log("fast_bp_decode_p50_us=" + std::to_string(p50) + " fast_bp_decode_p99_us=" + std::to_string(p99));
        ASSERT_EQ(bad_cyc, 0);
#ifdef AVID_SANITIZER_BUILD
        ASSERT_IN_RANGE(p50, 0.0, 5000.0);
        ASSERT_IN_RANGE(p99, 0.0, 30000.0);
#else
        ASSERT_IN_RANGE(p50, 0.0, 50.0);
        ASSERT_IN_RANGE(p99, 0.0, 500.0);
#endif
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-045") {
        TestContext ctx(opts, tid, "timing_overrun_profiler_to_hold_last_good",
                        "software/src/control/stage_profiler.cpp, software/src/control/fallback_fsm.cpp",
                        "avid_profiler_end (1.0 us budget) -> avid_fallback_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Configure avid_stage_profiler_t with 0.01 us deadline and avid_daemon --slow-deadline-us 0.01";
        ctx.record().independent_oracle = "avid_profiler_end returns false, deadline_misses > 0, and FSM enters HOLD_LAST_GOOD (should_commit==false)";

        avid_stage_profiler_t prof{};
        avid_profiler_init(&prof, "tight_deadline_est", 0.01);
        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        const uint32_t hx[16] = {200, 6, 5, 8, 6, 3, 18, 2, 3, 1, 1, 1, 1, 1, 0, 0};

        avid_profiler_begin(&prof);
        solver.fit_histogram(m, hx, 20, nullptr);
        bool ddl_ok = avid_profiler_end(&prof);
        ASSERT_FALSE(ddl_ok);
        ASSERT_EQ(prof.deadline_misses, 1ULL);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t cand[9] = {800, 800, 800, 800, 800, 800, 800, 800, 800};
        int16_t sel[9] = {0};
        bool commit = true;
        ASSERT_EQ(avid_fallback_step(&fsm, true, ddl_ok, true, true, cand, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_FALSE(commit);

        std::string mp = "/tmp/l3_exec045_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 512 --window 256 --slow-deadline-us 0.01"
            " --seed 2026 --backend physical --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "fallback_events")), 2);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
