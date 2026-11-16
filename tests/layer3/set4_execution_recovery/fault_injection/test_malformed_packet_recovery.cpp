#include "test_assertions.h"
#include "test_common.h"
#include "control/fallback_fsm.h"
#include "driver/weight_reg_driver.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include "hal/avid_hal.h"
#include "inference/noise_model.h"
#include "ingress/dma_ingress_tap.h"
#include "ingress/packet_validator.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

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

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-EXEC-023");
    const std::string tid  = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-EXEC-023") {
        TestContext ctx(opts, tid, "packet_duplication_rejection",
                        "software/src/ingress/packet_validator.cpp, software/src/ingress/dma_ingress_tap.cpp",
                        "avid_validate_packet + avid_dma_ring_consume_slot", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Inject duplicate sequence number back-to-back (seq=6, 7, 7, 8) into DMA ring";
        ctx.record().independent_oracle = "Second seq=7 is rejected by sequence continuity check, avoiding double histogram count";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        uint64_t pkts[4] = {
            avid_pack_packet_u64(0x1u, 0x0u, 6u, 1u, AVID_PKT_FLAG_VALID),
            avid_pack_packet_u64(0x6u, 0x0u, 7u, 1u, AVID_PKT_FLAG_VALID),
            avid_pack_packet_u64(0x6u, 0x0u, 7u, 1u, AVID_PKT_FLAG_VALID), /* duplicate seq=7 */
            avid_pack_packet_u64(0x2u, 0x0u, 8u, 1u, AVID_PKT_FLAG_VALID)
        };
        ASSERT_TRUE(avid_dma_ring_push_burst(&ring, pkts, 4u));
        uint32_t hx[16] = {0}, hz[16] = {0}, n_val = 0;
        ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &n_val));
        ASSERT_EQ(n_val, 3u);
        ASSERT_EQ(hx[6], 1u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-024") {
        TestContext ctx(opts, tid, "empty_observation_window_safe_hold",
                        "software/src/estimator/lbfgs_solver.cpp, software/src/control/fallback_fsm.cpp",
                        "fit_histogram (hist=0) -> avid_fallback_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Force all-zero histogram hist={0}*16 in solver and pass estimator_ok=false into FSM";
        ctx.record().independent_oracle = "fit_histogram returns 0.0 with converged==false; FSM enters HOLD_LAST_GOOD with should_commit==false";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        const uint32_t zero_hist[16] = {0};
        LBFGSFitDiagnostics diag{};
        double nll = solver.fit_histogram(m, zero_hist, 20, &diag);
        ASSERT_EQ(nll, 0.0);
        ASSERT_FALSE(diag.converged);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t cand[9] = {900, 900, 900, 900, 900, 900, 900, 900, 900};
        int16_t sel[9] = {0};
        bool commit = true;
        ASSERT_EQ(avid_fallback_step(&fsm, diag.converged && nll > 0.0, true, true, true, cand, sel, &commit),
                  AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_FALSE(commit);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-025") {
        TestContext ctx(opts, tid, "optimizer_non_convergence_commit_guard",
                        "software/src/estimator/lbfgs_solver.cpp, software/src/inference/avid_daemon.cpp",
                        "fit_histogram(max_iters=0) + --lbfgs-max-iters 0", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Run solver and avid_daemon with --lbfgs-max-iters 0 so diag.converged==false";
        ctx.record().independent_oracle = "diag.converged==false triggers fallback_events >= 1 and blocks un-converged adaptive commits";

        NoiseModel m;
        m.set_basis_z(false);
        m.init(0.012, 0.004);
        HistogramLBFGSSolver solver;
        const uint32_t hx[16] = {180, 10, 8, 12, 10, 4, 25, 1, 2, 1, 1, 1, 1, 0, 0, 0};
        LBFGSFitDiagnostics diag{};
        solver.fit_histogram(m, hx, 0, &diag);
        ASSERT_FALSE(diag.converged);
        ASSERT_EQ(diag.iterations_executed, 0);

        std::string mp = "/tmp/l3_exec025_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 512 --window 256 --lbfgs-max-iters 0"
            " --seed 2026 --backend physical --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "fallback_events")), 2);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "adaptive_epoch_commits")), 0);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-026") {
        TestContext ctx(opts, tid, "optimizer_nan_injection_containment",
                        "software/src/estimator/llr_quantizer.cpp, software/src/control/fallback_fsm.cpp",
                        "avid_quantize_and_validate_bank + avid_fallback_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Inject NaN into NoiseModel::theta_ and p_eff bank passed to quantizer and FSM";
        ctx.record().independent_oracle = "validate_parameter_bounds==false, quantizer returns false & sets 0x0468, FSM enters SAFE_STATIC_PRIOR";

        NoiseModel::ParamVec th;
        th.setConstant(0.012);
        th(4) = std::numeric_limits<double>::quiet_NaN();
        ASSERT_FALSE(NoiseModel::validate_parameter_bounds(th));

        double peff[9] = {0.012, 0.012, 0.012, 0.012, std::numeric_limits<double>::quiet_NaN(), 0.012, 0.012, 0.012, 0.012};
        int16_t q78[9] = {0};
        bool bounds_ok = avid_quantize_and_validate_bank(peff, q78);
        ASSERT_FALSE(bounds_ok);
        ASSERT_EQ(q78[4], AVID_Q78_DEFAULT);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool commit = false;
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, bounds_ok, true, q78, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(sel[4], AVID_Q78_DEFAULT);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-027") {
        TestContext ctx(opts, tid, "out_of_bounds_probability_estimate_rejection",
                        "software/src/estimator/llr_quantizer.cpp, software/src/control/fallback_fsm.cpp",
                        "avid_quantize_and_validate_bank (p_eff=0.65)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Set p_eff[0..8] = 0.65 (>= 0.5) and pass through quantizer and safety FSM";
        ctx.record().independent_oracle = "Quantizer returns false, replaces candidates with 0x0468, and triggers SAFE_STATIC_PRIOR";

        double peff[9];
        for (int q = 0; q < 9; ++q) peff[q] = 0.65;
        int16_t q78[9] = {0};
        bool bounds_ok = avid_quantize_and_validate_bank(peff, q78);
        ASSERT_FALSE(bounds_ok);
        for (int q = 0; q < 9; ++q) ASSERT_EQ(q78[q], AVID_Q78_DEFAULT);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        int16_t sel[9] = {0};
        bool commit = false;
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, bounds_ok, true, q78, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-028") {
        TestContext ctx(opts, tid, "quantization_overflow_and_safe_floor_saturation",
                        "software/src/estimator/llr_quantizer.cpp",
                        "avid_prob_to_q78_llr + avid_quantize_and_validate_bank", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Evaluate extreme p=1e-12 and near-half p=0.489 (raw Q7.8=11 < 0x0040) through quantizer";
        ctx.record().independent_oracle = "p=1e-12 clamps deterministically at 1e-5 (2947 <= 0x3000); p=0.489 flags all_safe=false and clamps to 0x0040";

        int16_t q_tiny = avid_prob_to_q78_llr(1e-12);
        ASSERT_IN_RANGE(q_tiny, AVID_Q78_MIN_SAFE, AVID_Q78_MAX_SAFE);

        double peff[9] = {1e-12, 0.012, 0.012, 0.012, 0.489, 0.012, 0.012, 0.012, 0.012};
        int16_t q78[9] = {0};
        bool safe = avid_quantize_and_validate_bank(peff, q78);
        ASSERT_FALSE(safe);
        ASSERT_EQ(q78[4], AVID_Q78_MIN_SAFE);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-029") {
        TestContext ctx(opts, tid, "unix_socket_commit_timeout_detection",
                        "software/src/hal/simulation_backend.cpp",
                        "avid_hal_set_timeout_ms + avid_hal_write_weights_and_commit", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Connect simulation_backend (timeout_ms=60) to AF_UNIX mock server that stalls past SO_RCVTIMEO";
        ctx.record().independent_oracle = "avid_hal_write_weights_and_commit returns -1, timeout_transactions>=1, rejected_transactions>=1";

        std::string sock = "/tmp/l3_exec029_" + std::to_string(::getpid()) + ".sock";
        ::unlink(sock.c_str());
        int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
        ::bind(lfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        ::listen(lfd, 2);

        std::thread stall_srv([lfd]() {
            int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd >= 0) {
                char buf[256];
                (void)::recv(cfd, buf, sizeof(buf), 0);
                ::usleep(250000); /* 250 ms > 60 ms timeout */
                ::close(cfd);
            }
        });

        avid_hal_t hal{};
        ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, sock.c_str(), WRD_BASE_ADDR), 0);
        avid_hal_set_timeout_ms(&hal, 60u);
        const int16_t w[9] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
        uint32_t ep = 0;
        uint64_t cyc = 0;
        int rc = avid_hal_write_weights_and_commit(&hal, w, w, &ep, &cyc);
        ASSERT_EQ(rc, -1);
        ASSERT_TRUE(hal.stats.timeout_transactions >= 1u);
        ASSERT_TRUE(hal.stats.rejected_transactions >= 1u);

        avid_hal_close(&hal);
        stall_srv.join();
        ::close(lfd);
        ::unlink(sock.c_str());
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-030") {
        TestContext ctx(opts, tid, "bridge_transaction_id_and_protocol_mismatch_rejection",
                        "software/src/hal/simulation_backend.cpp", "send_bridge_request", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Mock AF_UNIX bridge responds with wrong transaction_id (tx_id+99) and protocol_version=2";
        ctx.record().independent_oracle = "Both responses rejected (-1); id_mismatch_errors==1 and rejected_transactions==2";

        std::string sock = "/tmp/l3_exec030_" + std::to_string(::getpid()) + ".sock";
        ::unlink(sock.c_str());
        int lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
        ::bind(lfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        ::listen(lfd, 2);

        std::thread bad_srv([lfd]() {
            int cfd = ::accept(lfd, nullptr, nullptr);
            if (cfd >= 0) {
                char buf[512];
                /* Req 1 -> wrong transaction_id */
                if (::recv(cfd, buf, sizeof(buf), 0) > 0) {
                    const char *r1 = "{\"protocol_version\":1,\"transaction_id\":999999,\"status\":\"OK\",\"rdata\":\"0x00000001\",\"accepted_cycle\":10}\n";
                    ::send(cfd, r1, std::strlen(r1), MSG_NOSIGNAL);
                }
                /* Req 2 -> wrong protocol_version */
                if (::recv(cfd, buf, sizeof(buf), 0) > 0) {
                    const char *r2 = "{\"protocol_version\":2,\"transaction_id\":1001,\"status\":\"OK\",\"rdata\":\"0x00000001\",\"accepted_cycle\":20}\n";
                    ::send(cfd, r2, std::strlen(r2), MSG_NOSIGNAL);
                }
                ::close(cfd);
            }
        });

        avid_hal_t hal{};
        ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, sock.c_str(), WRD_BASE_ADDR), 0);
        uint32_t rdata = 0;
        uint64_t cyc = 0;
        ASSERT_EQ(avid_hal_mmio_read(&hal, 0x050u, &rdata, &cyc), -1);
        ASSERT_EQ(avid_hal_mmio_read(&hal, 0x050u, &rdata, &cyc), -1);
        ASSERT_EQ(hal.stats.id_mismatch_errors, 1u);
        ASSERT_EQ(hal.stats.rejected_transactions, 2u);

        avid_hal_close(&hal);
        bad_srv.join();
        ::close(lfd);
        ::unlink(sock.c_str());
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-031") {
        TestContext ctx(opts, tid, "decoder_busy_status_commit_timeout",
                        "software/src/driver/weight_reg_driver.c", "wrd_commit_and_wait", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Lock emulated status_reg bit 0 (commit_pending=1) and invoke wrd_commit_and_wait(..., 100)";
        ctx.record().independent_oracle = "wrd_commit_and_wait returns -1 cleanly without overwriting active weight bank";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        const int16_t w_ok[9] = {950, 950, 950, 950, 950, 950, 950, 950, 950};
        wrd_write_all_weights(&wrd, w_ok);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);

        const int16_t w_new[9] = {450, 450, 450, 450, 450, 450, 450, 450, 450};
        wrd_write_all_weights(&wrd, w_new);
        wrd.sim_regs[WRD_OFF_STATUS / 4u] = 0x1u;
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 100), -1);
        ASSERT_EQ(wrd_read_active_weight(&wrd, 0), 950);
        wrd_close(&wrd);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-033") {
        TestContext ctx(opts, tid, "memory_pressure_rlimit_as_clean_execution",
                        "software/src/inference/ring_buffer.h, software/src/inference/lbfgs_optimizer.h",
                        "avid_daemon under 128 MB RLIMIT_AS", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Build release avid_daemon (no sanitizer shadow map) and execute under setrlimit(RLIMIT_AS, 128 MB)";
        ctx.record().independent_oracle = "Succeeds cleanly (exit 0) within static pre-allocated footprint (<128 MB virtual address space)";

        std::string bdir = "/tmp/l3_exec033_build_" + std::to_string(::getpid());
        std::string mp   = "/tmp/l3_exec033_" + std::to_string(::getpid()) + ".json";
        run_command_capture("rm -rf " + bdir);
        ASSERT_EQ(run_command_capture(
            "cmake -S " + opts.repo_root + "/software -B " + bdir +
            " -DCMAKE_BUILD_TYPE=Release && cmake --build " + bdir + " --target avid_daemon -j4").exit_code, 0);

        std::string bin_rel = bdir + "/avid_daemon";
        pid_t pid = ::fork();
        if (pid == 0) {
            struct rlimit rl{};
            rl.rlim_cur = 128ULL * 1024ULL * 1024ULL;
            rl.rlim_max = 128ULL * 1024ULL * 1024ULL;
            ::setrlimit(RLIMIT_AS, &rl);
            ::execl(bin_rel.c_str(), bin_rel.c_str(),
                    "--scenario", "correlated_noise",
                    "--rounds", "1024",
                    "--window", "256",
                    "--backend", "physical",
                    "--checkpoint", ckpt.c_str(),
                    "--metrics-out", mp.c_str(),
                    static_cast<char *>(nullptr));
            _exit(127);
        }
        int st = 0;
        ::waitpid(pid, &st, 0);
        run_command_capture("rm -rf " + bdir);
        ASSERT_TRUE(WIFEXITED(st));
        ASSERT_EQ(WEXITSTATUS(st), 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-034") {
        TestContext ctx(opts, tid, "repeated_100x_safety_trigger_stability",
                        "software/src/control/fallback_fsm.cpp", "avid_fallback_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Trigger avid_fallback_step with estimator failures for 100 consecutive calls";
        ctx.record().independent_oracle = "FSM transitions HOLD(1..2) -> SAFE_STATIC_PRIOR(>=3) and remains stable with 0x0468 static weights";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t bad_cand[9] = {200, 200, 200, 200, 200, 200, 200, 200, 200};
        int16_t sel[9] = {0};
        bool commit = false;
        for (int i = 0; i < 100; ++i) {
            avid_fallback_state_t s = avid_fallback_step(&fsm, false, true, true, true, bad_cand, sel, &commit);
            if (i >= 2) {
                ASSERT_EQ(s, AVID_FSM_SAFE_STATIC_PRIOR);
                for (int q = 0; q < 9; ++q) ASSERT_EQ(sel[q], AVID_Q78_DEFAULT);
            }
        }
        ASSERT_EQ(fsm.state, AVID_FSM_SAFE_STATIC_PRIOR);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-035") {
        TestContext ctx(opts, tid, "sticky_recovery_block_while_fault_persists",
                        "software/src/control/fallback_fsm.cpp", "avid_fallback_step", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Keep bounds_ok=false or bus_ok=false across 25 steps while other flags are healthy";
        ctx.record().independent_oracle = "FSM never returns to NORMAL_ADAPTIVE while bounds_ok==false or bus_ok==false";

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t cand[9] = {900, 900, 900, 900, 900, 900, 900, 900, 900};
        int16_t sel[9] = {0};
        bool commit = false;
        for (int i = 0; i < 25; ++i) {
            ASSERT_EQ(avid_fallback_step(&fsm, true, true, false, true, cand, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        }
        for (int i = 0; i < 25; ++i) {
            ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, false, cand, sel, &commit), AVID_FSM_DEGRADED_BYPASS);
        }
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
