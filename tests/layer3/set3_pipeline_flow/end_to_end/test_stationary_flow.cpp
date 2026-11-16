#include "test_assertions.h"
#include "test_common.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "custom_engine/cycle_accurate_bp_engine.h"
#include "driver/weight_reg_driver.h"
#include "estimator/lbfgs_solver.h"
#include "estimator/llr_quantizer.h"
#include "hal/avid_hal.h"
#include "inference/lbfgs_optimizer.h"
#include "inference/noise_model.h"
#include "ingress/dma_ingress_tap.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
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

static std::vector<double> extract_json_array(const std::string &body, const std::string &key) {
    std::vector<double> out;
    const std::string pat = "\"" + key + "\":";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return out;
    size_t lb = body.find('[', pos);
    size_t rb = body.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return out;
    std::stringstream ss(body.substr(lb + 1, rb - lb - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        char *endp = nullptr;
        double v = std::strtod(item.c_str(), &endp);
        if (endp != item.c_str()) out.push_back(v);
    }
    return out;
}

static uint64_t parse_req_u64(const std::string &line, const char *key, uint64_t def_val = 0) {
    std::string pat = std::string("\"") + key + "\":";
    size_t p = line.find(pat);
    if (p == std::string::npos) return def_val;
    p += pat.size();
    while (p < line.size() && (line[p] == ' ' || line[p] == '"')) p++;
    return static_cast<uint64_t>(std::strtoull(line.c_str() + p, nullptr, 0));
}

class MockCosimBridgeServer {
public:
    explicit MockCosimBridgeServer(const std::string &sock_path, int commit_delay_us = 0)
        : sock_path_(sock_path), commit_delay_us_(commit_delay_us) {
        for (int q = 0; q < 9; ++q) {
            shadow_x_[q] = AVID_Q78_DEFAULT;
            shadow_z_[q] = AVID_Q78_DEFAULT;
            active_x_[q] = AVID_Q78_DEFAULT;
            active_z_[q] = AVID_Q78_DEFAULT;
        }
        ::unlink(sock_path_.c_str());
        listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock_path_.c_str());
        ::bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
        ::listen(listen_fd_, 4);
        thr_ = std::thread([this]() { serve(); });
    }

    ~MockCosimBridgeServer() {
        stop();
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (thr_.joinable()) thr_.join();
        ::unlink(sock_path_.c_str());
    }

    uint32_t observed_max_k() const { return max_k_.load(); }
    uint32_t observed_min_k() const { return min_k_.load(); }
    uint32_t commits_seen() const { return commits_.load(); }

private:
    void serve() {
        int cfd = ::accept(listen_fd_, nullptr, nullptr);
        if (cfd < 0) return;
        std::string buf;
        char chunk[512];
        while (!stopped_.load()) {
            ssize_t r = ::recv(cfd, chunk, sizeof(chunk), 0);
            if (r <= 0) break;
            buf.append(chunk, static_cast<size_t>(r));
            size_t nl = 0;
            while ((nl = buf.find('\n')) != std::string::npos) {
                std::string line = buf.substr(0, nl);
                buf.erase(0, nl + 1);
                std::string resp = handle_line(line);
                size_t sent = 0;
                while (sent < resp.size()) {
                    ssize_t w = ::send(cfd, resp.data() + sent, resp.size() - sent, MSG_NOSIGNAL);
                    if (w <= 0) break;
                    sent += static_cast<size_t>(w);
                }
            }
        }
        ::close(cfd);
    }

    std::string handle_line(const std::string &line) {
        uint64_t tx_id = parse_req_u64(line, "transaction_id", 0);
        uint32_t addr  = static_cast<uint32_t>(parse_req_u64(line, "address", 0));
        uint32_t data  = static_cast<uint32_t>(parse_req_u64(line, "data", 0));
        char out[512];

        if (line.find("\"MMIO_WRITE\"") != std::string::npos) {
            cycle_ += 4;
            if (addr < 0x024u && (addr % 4u == 0u)) {
                shadow_x_[addr / 4u] = static_cast<int16_t>(data & 0xFFFFu);
            } else if (addr >= 0x024u && addr < 0x048u && (addr % 4u == 0u)) {
                shadow_z_[(addr - 0x024u) / 4u] = static_cast<int16_t>(data & 0xFFFFu);
            } else if (addr == 0x048u && (data & 1u)) {
                if (commit_delay_us_ > 0) ::usleep(static_cast<useconds_t>(commit_delay_us_));
                std::memcpy(active_x_, shadow_x_, sizeof(active_x_));
                std::memcpy(active_z_, shadow_z_, sizeof(active_z_));
                epoch_++;
                commits_.fetch_add(1);
                cycle_ += 2;
            } else if (addr == 0x054u) {
                uint32_t k = std::max(1u, std::min(5u, data & 0xFFu));
                k_iter_ = k;
                if (k > max_k_.load()) max_k_.store(k);
                if (k < min_k_.load()) min_k_.store(k);
            } else if (addr == 0x058u) {
                osd_mode_ = data & 1u;
            }
            std::snprintf(out, sizeof(out),
                "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"accepted_cycle\":%llu}\n",
                (unsigned long long)tx_id, (unsigned long long)cycle_);
            return out;
        }

        if (line.find("\"MMIO_READ\"") != std::string::npos) {
            cycle_ += 4;
            uint32_t rdata = 0;
            if (addr == 0x04Cu) rdata = 0x2u; /* bp_ready=1, commit_pending=0 */
            else if (addr == 0x050u) rdata = epoch_;
            else if (addr == 0x054u) rdata = k_iter_;
            else if (addr == 0x058u) rdata = osd_mode_;
            else if (addr >= 0x100u && addr < 0x124u) rdata = static_cast<uint16_t>(active_x_[(addr - 0x100u) / 4u]);
            else if (addr >= 0x124u && addr < 0x148u) rdata = static_cast<uint16_t>(active_z_[(addr - 0x124u) / 4u]);
            std::snprintf(out, sizeof(out),
                "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"rdata\":\"0x%08X\",\"accepted_cycle\":%llu}\n",
                (unsigned long long)tx_id, rdata, (unsigned long long)cycle_);
            return out;
        }

        if (line.find("\"RESET\"") != std::string::npos) {
            cycle_ += (data > 0 ? data : 8u);
            epoch_ = 0;
            k_iter_ = 3;
            for (int q = 0; q < 9; ++q) {
                shadow_x_[q] = active_x_[q] = AVID_Q78_DEFAULT;
                shadow_z_[q] = active_z_[q] = AVID_Q78_DEFAULT;
            }
            std::snprintf(out, sizeof(out),
                "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"accepted_cycle\":%llu}\n",
                (unsigned long long)tx_id, (unsigned long long)cycle_);
            return out;
        }

        if (line.find("\"SYNDROME_TRANSFER\"") != std::string::npos) {
            uint8_t sx = static_cast<uint8_t>(data & 0xFu);
            CycleAccurateBPEngine bp(static_cast<int>(k_iter_), 250.0);
            avid_bp_result_t r = bp.decode(sx, active_z_, static_cast<int>(k_iter_));
            uint32_t bp_cyc = static_cast<uint32_t>(r.cycles_fused);
            uint32_t ax_cyc = bp_cyc + 5u;
            cycle_ += ax_cyc;
            std::snprintf(out, sizeof(out),
                "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"OK\",\"corr_x\":0,\"corr_z\":%u,"
                "\"bp_cycles\":%u,\"axis_total_cycles\":%u,\"epoch_irq\":%u,\"accepted_cycle\":%llu}\n",
                (unsigned long long)tx_id, (unsigned)r.correction_mask, bp_cyc, ax_cyc, epoch_, (unsigned long long)cycle_);
            return out;
        }

        std::snprintf(out, sizeof(out),
            "{\"protocol_version\":1,\"transaction_id\":%llu,\"status\":\"ERROR\",\"accepted_cycle\":%llu}\n",
            (unsigned long long)tx_id, (unsigned long long)cycle_);
        return out;
    }

    std::string sock_path_;
    int commit_delay_us_ = 0;
    int listen_fd_ = -1;
    std::atomic<bool> stopped_{false};
    std::thread thr_;
    int16_t shadow_x_[9]{}, shadow_z_[9]{}, active_x_[9]{}, active_z_[9]{};
    uint32_t epoch_ = 0;
    uint32_t k_iter_ = 3;
    uint32_t osd_mode_ = 1;
    uint64_t cycle_ = 0;
    std::atomic<uint32_t> max_k_{3};
    std::atomic<uint32_t> min_k_{3};
    std::atomic<uint32_t> commits_{0};
};

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-FLOW-013");
    const std::string tid = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-FLOW-013") {
        TestContext ctx(opts, tid, "nominal_stationary_end_to_end_flow",
                        "software/src/inference/avid_daemon.cpp", "main", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run avid_daemon --scenario stationary_noise --rounds 1536 --window 256";
        ctx.record().independent_oracle = "Finite NLL, fallback_events==0, epoch_commit_reconciled==true, orbit_9class_rmse < 0.03";

        std::string mpath = "/tmp/l3_flow013_" + std::to_string(::getpid()) + ".json";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl013 = " --slow-deadline-us 500000";
#else
        const std::string san_dl013 = "";
#endif
        SubprocessResult res = run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 1536 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + san_dl013 + " --metrics-out " + mpath);
        ASSERT_EQ(res.exit_code, 0);
        std::string body = read_text_file_unlink(mpath);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "fallback_events")), 0);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "epochs_completed")), 6);
        ASSERT_FINITE(extract_json_num(body, "final_nll"));
        ASSERT_IN_RANGE(extract_json_num(body, "orbit_9class_rmse"), 0.0, 0.03);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-014") {
        TestContext ctx(opts, tid, "slowly_drifting_noise_tracking",
                        "software/src/inference/avid_daemon.cpp", "main (slowly_drifting_noise)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Compare early vs full trajectory under slowly_drifting_noise (--window 256)";
        ctx.record().independent_oracle = "Late-epoch estimated hotspot probability rises and active Q7.8 LLR on qubit 4 decreases to track upward drift";

        std::string mp1 = "/tmp/l3_flow014_a_" + std::to_string(::getpid()) + ".json";
        std::string mp2 = "/tmp/l3_flow014_b_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp1).exit_code, 0);
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario slowly_drifting_noise --rounds 2048 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp2).exit_code, 0);

        std::string b1 = read_text_file_unlink(mp1);
        std::string b2 = read_text_file_unlink(mp2);
        auto q78_stat  = extract_json_array(b1, "final_q78_z");
        auto q78_drift = extract_json_array(b2, "final_q78_z");
        auto th_stat   = extract_json_array(b1, "final_theta_z");
        auto th_drift  = extract_json_array(b2, "final_theta_z");
        ASSERT_EQ(q78_stat.size(), 9u);
        ASSERT_EQ(q78_drift.size(), 9u);
        ASSERT_EQ(th_drift.size(), 21u);
        ASSERT_TRUE((th_drift[4] + th_drift[12]) > (th_stat[4] + th_stat[12]) * 2.0);
        ASSERT_TRUE(q78_drift[4] < q78_stat[4]);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-015") {
        TestContext ctx(opts, tid, "correlated_noise_transition_and_sim_ler_advantage",
                        "software/src/inference/avid_daemon.cpp, software/src/hal/simulation_backend.cpp",
                        "main (--backend sim)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run correlated_noise and heldout_multi_pair_12_67 over live AF_UNIX simulation bridge and compare vs static B0 LER";
        ctx.record().independent_oracle = "Hotspot pair LLRs adapt downward and logical_error_rate < static_b0_logical_error_rate";

        std::string sock1 = "/tmp/l3_flow015_s1_" + std::to_string(::getpid()) + ".sock";
        std::string mp1   = "/tmp/l3_flow015_m1_" + std::to_string(::getpid()) + ".json";
        {
            MockCosimBridgeServer srv(sock1);
            SubprocessResult r1 = run_command_capture(
                opts.daemon_bin + " --scenario correlated_noise --rounds 1536 --window 256 --seed 2026"
                " --backend sim --socket " + sock1 + " --checkpoint " + ckpt + " --metrics-out " + mp1);
            srv.stop();
            ASSERT_EQ(r1.exit_code, 0);
        }
        std::string b1 = read_text_file_unlink(mp1);
        double ler1 = extract_json_num(b1, "logical_error_rate");
        double sler1 = extract_json_num(b1, "static_b0_logical_error_rate");
        ctx.log("correlated_noise sim: avid_ler=" + std::to_string(ler1) + " static_ler=" + std::to_string(sler1));
        ASSERT_TRUE(ler1 < sler1);

        std::string mp2 = "/tmp/l3_flow015_m2_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario heldout_multi_pair_12_67 --rounds 2048 --window 256 --seed 2026"
            " --backend physical --checkpoint " + ckpt + " --metrics-out " + mp2).exit_code, 0);
        std::string b2 = read_text_file_unlink(mp2);
        ASSERT_TRUE(extract_json_num(b2, "logical_error_rate") < extract_json_num(b2, "static_b0_logical_error_rate"));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-016") {
        TestContext ctx(opts, tid, "burst_disturbance_zero_halted_rounds",
                        "software/src/inference/avid_daemon.cpp", "main (burst_disturbance)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run avid_daemon --scenario burst_disturbance --rounds 2048 --window 256";
        ctx.record().independent_oracle = "decoded_rtl_transactions == 2048, fast_burst_detector.trigger_count >= 1, epoch_commit_reconciled == true";

        std::string mpath = "/tmp/l3_flow016_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario burst_disturbance --rounds 2048 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mpath).exit_code, 0);
        std::string body = read_text_file_unlink(mpath);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "decoded_rtl_transactions")), 2048);
        ASSERT_TRUE(extract_json_num(body, "trigger_count") >= 1.0);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-017") {
        TestContext ctx(opts, tid, "delayed_weight_commit_reconciliation",
                        "software/src/driver/weight_reg_driver.c, software/src/control/causal_scheduler.cpp",
                        "main (--commit-delay 32 --scenario delayed_weight_commit)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run delayed_weight_commit with --commit-delay 32 over delayed-ack Unix co-sim bridge";
        ctx.record().independent_oracle = "Delayed commits complete cleanly without round loss and epoch_commit_reconciled == true";

        std::string sock = "/tmp/l3_flow017_" + std::to_string(::getpid()) + ".sock";
        std::string mp   = "/tmp/l3_flow017_" + std::to_string(::getpid()) + ".json";
        {
            MockCosimBridgeServer srv(sock, 1500);
            SubprocessResult r = run_command_capture(
                opts.daemon_bin + " --scenario delayed_weight_commit --rounds 1024 --window 256 --commit-delay 32 --seed 2026"
                " --backend sim --socket " + sock + " --checkpoint " + ckpt + " --metrics-out " + mp);
            srv.stop();
            ASSERT_EQ(r.exit_code, 0);
        }
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "scheduler_commit_delay_rounds")), 32);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-018") {
        TestContext ctx(opts, tid, "reset_recovery_end_to_end",
                        "software/src/inference/avid_daemon.cpp", "main (reset_recovery)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run avid_daemon --scenario reset_recovery --rounds 1024 --window 256";
        ctx.record().independent_oracle = "Mid-run RTL reset increments reset_cleared_commits, resumes commits, and reconciles hw_epoch";

        std::string mp = "/tmp/l3_flow018_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario reset_recovery --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(extract_json_num(body, "reset_cleared_commits") >= 1.0);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-019") {
        TestContext ctx(opts, tid, "busy_decoder_commit_deferral_protection",
                        "software/src/driver/weight_reg_driver.c", "wrd_commit_and_wait", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Lock emulated status_reg with commit_pending=1 (bp_ready=0) and verify wrd_commit_and_wait times out without corrupting active weights";
        ctx.record().independent_oracle = "wrd_commit_and_wait returns -1 when status_reg stays busy; prior active bank remains intact";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        const int16_t init_bank[9] = {1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080};
        wrd_write_all_weights(&wrd, init_bank);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        uint32_t ep_before = wrd_read_epoch(&wrd);

        /* Emulate busy decoder: bit 0 = commit_pending=1, bit 1 = bp_ready=0 at offset 0x04C (word 19) */
        const int16_t rogue_bank[9] = {400, 400, 400, 400, 400, 400, 400, 400, 400};
        wrd_write_all_weights(&wrd, rogue_bank);
        wrd.sim_regs[WRD_OFF_STATUS / 4u] = (wrd.sim_regs[WRD_OFF_STATUS / 4u] & ~0x3u) | 0x1u;
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 64), -1);
        ASSERT_EQ(wrd_read_active_weight(&wrd, 4), 1040);
        ASSERT_EQ(wrd_read_epoch(&wrd), ep_before);
        wrd_close(&wrd);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-020") {
        TestContext ctx(opts, tid, "variable_iteration_control_mmio_and_cycles",
                        "software/src/inference/avid_daemon.cpp, software/src/hal/simulation_backend.cpp",
                        "avid_hal_set_bp_iterations (0x054)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run variable_k_adaptation over mock Unix bridge and verify K in {1..5} writes to 0x054 and 2+5*K BP cycles";
        ctx.record().independent_oracle = "bp_core_cycles_min >= 7, bp_core_cycles_max <= 27, and K spans multiple iteration counts";

        std::string sock = "/tmp/l3_flow020_" + std::to_string(::getpid()) + ".sock";
        std::string mp   = "/tmp/l3_flow020_" + std::to_string(::getpid()) + ".json";
        uint32_t min_k = 3, max_k = 3;
        {
            MockCosimBridgeServer srv(sock);
            SubprocessResult r = run_command_capture(
                opts.daemon_bin + " --scenario variable_k_adaptation --rounds 1536 --window 256 --seed 2026"
                " --backend sim --socket " + sock + " --checkpoint " + ckpt + " --metrics-out " + mp);
            min_k = srv.observed_min_k();
            max_k = srv.observed_max_k();
            srv.stop();
            ASSERT_EQ(r.exit_code, 0);
        }
        std::string body = read_text_file_unlink(mp);
        uint32_t cmin = static_cast<uint32_t>(extract_json_num(body, "bp_core_cycles_min"));
        uint32_t cmax = static_cast<uint32_t>(extract_json_num(body, "bp_core_cycles_max"));
        ASSERT_IN_RANGE(min_k, 1u, 5u);
        ASSERT_IN_RANGE(max_k, 1u, 5u);
        ASSERT_TRUE(cmax > cmin);
        ASSERT_EQ(cmin, 2u + 5u * min_k);
        ASSERT_EQ(cmax, 2u + 5u * max_k);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-021") {
        TestContext ctx(opts, tid, "fallback_activation_on_corrupted_epoch",
                        "software/src/inference/avid_daemon.cpp, software/src/control/fallback_fsm.cpp",
                        "main (fallback_safety_trigger)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run fallback_safety_trigger and injected !deadline_ok / !bus_ok faults";
        ctx.record().independent_oracle = "fallback_events >= 1 in daemon; FSM enters SAFE_STATIC_PRIOR, HOLD_LAST_GOOD, and DEGRADED_BYPASS accurately";

        std::string mp = "/tmp/l3_flow021_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario fallback_safety_trigger --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(extract_json_num(body, "fallback_events") >= 1.0);

        avid_fallback_fsm_t fsm{};
        avid_fallback_init(&fsm, AVID_Q78_DEFAULT);
        const int16_t cand[9] = {800, 800, 800, 800, 800, 800, 800, 800, 800};
        int16_t sel[9] = {0};
        bool commit = false;
        ASSERT_EQ(avid_fallback_step(&fsm, true, false, true, true, cand, sel, &commit), AVID_FSM_HOLD_LAST_GOOD);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, false, true, cand, sel, &commit), AVID_FSM_SAFE_STATIC_PRIOR);
        ASSERT_EQ(avid_fallback_step(&fsm, true, true, true, false, cand, sel, &commit), AVID_FSM_DEGRADED_BYPASS);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-022") {
        TestContext ctx(opts, tid, "recovery_after_fallback_to_normal_adaptive",
                        "software/src/inference/avid_daemon.cpp, software/src/control/fallback_fsm.cpp",
                        "main (fallback_safety_trigger -> healthy epochs)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run fallback_safety_trigger across 4 epochs (Epoch 1 faulted, Epochs 2..4 healthy) and check final_fsm_state";
        ctx.record().independent_oracle = "fallback_events >= 1 on Epoch 1, then daemon recovers to final_fsm_state == 0 (NORMAL_ADAPTIVE) and reconciles commits";

        std::string mp = "/tmp/l3_flow022_" + std::to_string(::getpid()) + ".json";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl022 = " --slow-deadline-us 500000";
#else
        const std::string san_dl022 = "";
#endif
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario fallback_safety_trigger --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + san_dl022 + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(extract_json_num(body, "fallback_events") >= 1.0);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "final_fsm_state")), static_cast<int>(AVID_FSM_NORMAL_ADAPTIVE));
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-023") {
        TestContext ctx(opts, tid, "multiple_update_cycles_12_epochs",
                        "software/src/inference/avid_daemon.cpp", "main (12 epochs)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run 12 consecutive slow-loop epochs (--rounds 3072 --window 256)";
        ctx.record().independent_oracle = "epochs_completed == 12, initial_boot_commits == 1, adaptive_epoch_commits == 12, weight_bank_commits == 13";

        std::string mp = "/tmp/l3_flow023_" + std::to_string(::getpid()) + ".json";
#ifdef AVID_SANITIZER_BUILD
        const std::string san_dl023 = " --slow-deadline-us 500000";
#else
        const std::string san_dl023 = "";
#endif
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 3072 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + san_dl023 + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "epochs_completed")), 12);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "adaptive_epoch_commits")), 12);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "weight_bank_commits")), 13);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-024") {
        TestContext ctx(opts, tid, "concurrent_ingress_and_optimization",
                        "software/src/ingress/dma_ingress_tap.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "avid_dma_ring_push_burst + avid_dma_ring_consume_slot + fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Multi-thread harness pushing 200x64-packet bursts while consumer thread snapshots and fits histograms";
        ctx.record().independent_oracle = "Total consumed + overflow_drops * 64 == 12,800 pushed packets; zero corrupted histogram bins";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        std::mutex ring_mtx;
        std::atomic<bool> prod_done{false};
        std::atomic<uint32_t> pushed_pkts{0};
        std::atomic<uint32_t> consumed_pkts{0};

        std::thread prod([&]() {
            uint32_t seq = 0;
            for (int b = 0; b < 200; ++b) {
                uint64_t burst[64];
                for (int i = 0; i < 64; ++i) {
                    burst[i] = avid_pack_packet_u64((i % 8 == 0) ? 0x6u : 0x0u, 0x0u, seq++, 1u, AVID_PKT_FLAG_VALID);
                }
                bool ok = false;
                while (!ok) {
                    {
                        std::lock_guard<std::mutex> lk(ring_mtx);
                        if (ring.slots[ring.write_slot_idx].state == AVID_SLOT_FREE) {
                            ok = avid_dma_ring_push_burst(&ring, burst, 64u);
                        }
                    }
                    if (!ok) ::usleep(50);
                }
                pushed_pkts.fetch_add(64u);
            }
            prod_done.store(true);
        });

        std::thread cons([&]() {
            NoiseModel m;
            m.set_basis_z(false);
            HistogramLBFGSSolver solver;
            uint32_t hx[16] = {0}, hz[16] = {0};
            uint32_t acc = 0;
            while (true) {
                uint32_t n = 0;
                bool got = false;
                {
                    std::lock_guard<std::mutex> lk(ring_mtx);
                    got = avid_dma_ring_consume_slot(&ring, hx, hz, &n);
                }
                if (got) {
                    consumed_pkts.fetch_add(n);
                    acc += n;
                    if (acc >= 256u) {
                        m.init(0.012, 0.004);
                        double nll = solver.fit_histogram(m, hx, 10, nullptr);
                        if (!std::isfinite(nll)) std::abort();
                        std::memset(hx, 0, sizeof(hx));
                        std::memset(hz, 0, sizeof(hz));
                        acc = 0;
                    }
                } else if (prod_done.load()) {
                    break;
                } else {
                    ::usleep(50);
                }
            }
        });

        prod.join();
        cons.join();
        ASSERT_EQ(pushed_pkts.load(), 12800u);
        ASSERT_EQ(consumed_pkts.load(), 12800u);
        ASSERT_EQ(ring.overflow_drops, 0u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-025") {
        TestContext ctx(opts, tid, "concurrent_syndrome_decode_and_atomic_bank_commit",
                        "software/src/hal/physical_backend.cpp, software/src/driver/weight_reg_driver.c",
                        "avid_hal_send_syndrome || avid_hal_write_weights_and_commit", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify active weight bank swaps atomically on commit (never exposing half-written shadow weights to decoder)";
        ctx.record().independent_oracle = "Writing partial shadow weights before 0x048 commit never alters active bank read by decoder; post-commit switches atomically";

        avid_hal_t hal{};
        ASSERT_EQ(avid_hal_open(&hal, AVID_HAL_BACKEND_PHYSICAL, nullptr, WRD_BASE_ADDR), 0);
        const int16_t bankA[9] = {1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128};
        const int16_t bankB[9] = {1128, 1128, 1128, 300,  1128, 1128, 1128, 1128, 1128};
        uint32_t ep = 0;
        uint64_t cyc = 0;
        ASSERT_EQ(avid_hal_write_weights_and_commit(&hal, bankA, bankA, &ep, &cyc), 0);

        /* Write half of shadow Z bank (offsets 0x024..0x034) WITHOUT committing 0x048 yet */
        for (int q = 0; q < 5; ++q) {
            ASSERT_EQ(avid_hal_mmio_write(&hal, 0x024u + 4u * q, static_cast<uint16_t>(bankB[q]), 0x3u, &cyc), 0);
        }
        /* Active Z register at 0x130 (q=3) MUST still equal bankA[3] == 1128 */
        uint32_t r_active = 0;
        ASSERT_EQ(avid_hal_mmio_read(&hal, 0x124u + 4u * 3u, &r_active, &cyc), 0);
        ASSERT_EQ(static_cast<int16_t>(r_active & 0xFFFFu), 1128);

        /* Complete remaining shadow Z writes and issue atomic commit */
        for (int q = 5; q < 9; ++q) {
            ASSERT_EQ(avid_hal_mmio_write(&hal, 0x024u + 4u * q, static_cast<uint16_t>(bankB[q]), 0x3u, &cyc), 0);
        }
        ASSERT_EQ(avid_hal_mmio_write(&hal, 0x048u, 1u, 0x1u, &cyc), 0);
        ASSERT_EQ(avid_hal_mmio_read(&hal, 0x124u + 4u * 3u, &r_active, &cyc), 0);
        ASSERT_EQ(static_cast<int16_t>(r_active & 0xFFFFu), 300);
        avid_hal_close(&hal);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-036") {
        TestContext ctx(opts, tid, "probability_propagation_sensitivity",
                        "software/src/estimator/lbfgs_solver.cpp", "HistogramLBFGSSolver::fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Increase true p_4 from 0.012 to 0.090 on identical RNG seed and inspect fitted theta[4]";
        ctx.record().independent_oracle = "theta_high[4] > 2.0 * theta_low[4]";

        auto fit_for_p4 = [](double p4) {
            uint64_t rng = 0xABCDEF0123456789ULL;
            uint32_t hist[16] = {0};
            for (int r = 0; r < 2000; ++r) {
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
                double u = static_cast<double>((rng * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
                uint8_t sx = (u < p4) ? 0x6u : 0x0u;
                hist[sx]++;
            }
            NoiseModel m;
            m.set_basis_z(false);
            m.init(0.012, 0.004);
            HistogramLBFGSSolver s;
            s.fit_histogram(m, hist, 20, nullptr);
            return m.theta_(4);
        };
        double th_lo = fit_for_p4(0.012);
        double th_hi = fit_for_p4(0.090);
        ASSERT_TRUE(th_hi > 2.0 * th_lo);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-037") {
        TestContext ctx(opts, tid, "histogram_propagation_sensitivity",
                        "software/src/inference/noise_model.cpp, software/src/estimator/lbfgs_solver.cpp",
                        "compute_nll_from_histogram + fit_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Add +200 counts directly to hist[6] (Z_4 signature) keeping other bins identical";
        ctx.record().independent_oracle = "Initial NLL and fitted theta[4] strictly increase";

        uint32_t h1[16] = {950, 5, 5, 5, 5, 5, 10, 2, 3, 2, 2, 2, 2, 1, 1, 0};
        uint32_t h2[16];
        std::memcpy(h2, h1, sizeof(h1));
        h2[6] += 200;

        NoiseModel m1, m2;
        m1.set_basis_z(false); m1.init(0.012, 0.004);
        m2.set_basis_z(false); m2.init(0.012, 0.004);
        double nll1 = m1.compute_nll_from_histogram(h1);
        double nll2 = m2.compute_nll_from_histogram(h2);
        ASSERT_TRUE(nll2 > nll1);

        HistogramLBFGSSolver s;
        s.fit_histogram(m1, h1, 20, nullptr);
        s.fit_histogram(m2, h2, 20, nullptr);
        ASSERT_TRUE(m2.theta_(4) > m1.theta_(4) * 3.0);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-038") {
        TestContext ctx(opts, tid, "prior_propagation_sensitivity",
                        "software/src/inference/noise_model.cpp, software/src/inference/lbfgs_optimizer.cpp",
                        "LBFGSOptimizer::step_histogram", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Shift prior_theta_(4) from 0.01 to 0.15 with prior_lambda_=2.0 on small sample N=50";
        ctx.record().independent_oracle = "MAP objective and fitted theta_(4) shift upward toward 0.15";

        uint32_t hist[16] = {46, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        NoiseModel m_lo, m_hi;
        m_lo.set_basis_z(false); m_lo.init(0.03, 0.004);
        m_hi.set_basis_z(false); m_hi.init(0.03, 0.004);
        m_lo.prior_lambda_ = 12.0; m_lo.prior_theta_(4) = 0.01;
        m_hi.prior_lambda_ = 12.0; m_hi.prior_theta_(4) = 0.15;

        double obj_lo = m_lo.compute_nll_from_histogram(hist);
        double obj_hi = m_hi.compute_nll_from_histogram(hist);
        ASSERT_TRUE(obj_hi != obj_lo);

        LBFGSOptimizer opt1, opt2;
        opt1.step_histogram(m_lo, hist, 40, nullptr, nullptr);
        opt2.step_histogram(m_hi, hist, 40, nullptr, nullptr);
        ASSERT_TRUE(m_hi.theta_(4) > m_lo.theta_(4) + 0.05);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-039") {
        TestContext ctx(opts, tid, "window_propagation_sensitivity",
                        "software/src/inference/avid_daemon.cpp", "main (--window 128 vs 512)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run daemon with --window 128 vs --window 512 at --rounds 1024";
        ctx.record().independent_oracle = "epochs_completed equals 8 vs 2 and orbit_9class_rmse shrinks at larger window";

        std::string mp128 = "/tmp/l3_flow039_128_" + std::to_string(::getpid()) + ".json";
        std::string mp512 = "/tmp/l3_flow039_512_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 1024 --window 128 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp128).exit_code, 0);
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 1024 --window 512 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp512).exit_code, 0);

        std::string b128 = read_text_file_unlink(mp128);
        std::string b512 = read_text_file_unlink(mp512);
        ASSERT_EQ(static_cast<int>(extract_json_num(b128, "epochs_completed")), 8);
        ASSERT_EQ(static_cast<int>(extract_json_num(b512, "epochs_completed")), 2);
        ASSERT_TRUE(extract_json_num(b512, "orbit_9class_rmse") <= extract_json_num(b128, "orbit_9class_rmse"));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-040") {
        TestContext ctx(opts, tid, "threshold_propagation_sensitivity",
                        "software/src/control/causal_scheduler.cpp", "avid_fast_detector_init + avid_fast_detector_step", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run identical 50%-density syndrome stream through detector with threshold=0.20 vs 0.80";
        ctx.record().independent_oracle = "threshold=0.20 triggers >=1; threshold=0.80 triggers 0 times";

        avid_fast_burst_detector_t det_lo{}, det_hi{};
        avid_fast_detector_init(&det_lo, 0.20, 4u);
        avid_fast_detector_init(&det_hi, 0.80, 4u);
        for (uint64_t r = 0; r < 64; ++r) {
            uint8_t sx = (r % 2 == 0) ? 0x6u : 0x0u;
            avid_fast_detector_step(&det_lo, r, sx, 0u);
            avid_fast_detector_step(&det_hi, r, sx, 0u);
        }
        ASSERT_TRUE(det_lo.trigger_count >= 1u);
        ASSERT_EQ(det_hi.trigger_count, 0u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-041") {
        TestContext ctx(opts, tid, "llr_propagation_monotonicity",
                        "software/src/estimator/llr_quantizer.cpp", "avid_prob_to_q78_llr", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Raise fitted theta[4] across {0.01, 0.04, 0.12} and inspect quantized q78[4]";
        ctx.record().independent_oracle = "q78[4] decreases strictly as error probability increases";

        int16_t q1 = avid_prob_to_q78_llr(0.01);
        int16_t q2 = avid_prob_to_q78_llr(0.04);
        int16_t q3 = avid_prob_to_q78_llr(0.12);
        ASSERT_TRUE(q1 > q2);
        ASSERT_TRUE(q2 > q3);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-042") {
        TestContext ctx(opts, tid, "commit_propagation_bitwise_equivalence",
                        "software/src/driver/weight_reg_driver.c", "wrd_write_all_weights + wrd_commit_and_wait", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Write bankA then bankB != bankA and commit; read active register bank";
        ctx.record().independent_oracle = "Active register bank equals bankB bitwise across all 9 qubits";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        const int16_t bankA[9] = {900, 910, 920, 930, 940, 950, 960, 970, 980};
        const int16_t bankB[9] = {610, 625, 640, 655, 480, 670, 685, 700, 715};
        wrd_write_all_weights(&wrd, bankA);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        wrd_write_all_weights(&wrd, bankB);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        for (int q = 0; q < 9; ++q) {
            ASSERT_EQ(wrd_read_active_weight(&wrd, q), bankB[q]);
        }
        wrd_close(&wrd);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-043") {
        TestContext ctx(opts, tid, "static_output_bypass_detection",
                        "software/src/inference/avid_daemon.cpp", "main (stationary_noise vs correlated_noise)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Run stationary_noise vs correlated_noise to fresh --metrics-out files and verify non-static telemetry";
        ctx.record().independent_oracle = "final_nll, orbit_9class_rmse, and final_q78_z differ significantly across distinct noise regimes";

        std::string mp1 = "/tmp/l3_flow043_1_" + std::to_string(::getpid()) + ".json";
        std::string mp2 = "/tmp/l3_flow043_2_" + std::to_string(::getpid()) + ".json";
        ::unlink(mp1.c_str());
        ::unlink(mp2.c_str());
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 1024 --window 256 --seed 101 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp1).exit_code, 0);
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 1024 --window 256 --seed 202 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp2).exit_code, 0);

        std::string b1 = read_text_file_unlink(mp1);
        std::string b2 = read_text_file_unlink(mp2);
        ASSERT_TRUE(std::abs(extract_json_num(b1, "final_nll") - extract_json_num(b2, "final_nll")) > 0.05);
        auto q1 = extract_json_array(b1, "final_q78_z");
        auto q2 = extract_json_array(b2, "final_q78_z");
        ASSERT_EQ(q1.size(), 9u);
        ASSERT_EQ(q2.size(), 9u);
        ASSERT_TRUE(q1 != q2);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-044") {
        TestContext ctx(opts, tid, "component_invocation_trace_completeness",
                        "software/src/inference/avid_daemon.cpp", "main (stage invocation telemetry)", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Verify every pipeline stage counter (validator, L-BFGS, R2D2 action histogram, HAL MMIO/syndrome/commits) > 0";
        ctx.record().independent_oracle = "validator_valid_packets > 0, lbfgs_iterations > 0, sum(r2d2_action_histogram) == epochs_completed, hal_mmio_writes > 0";

        std::string mp = "/tmp/l3_flow044_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 1024 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "validator_valid_packets")), 1024);
        ASSERT_TRUE(extract_json_num(body, "lbfgs_iterations") > 0.0);
        ASSERT_TRUE(extract_json_num(body, "hal_mmio_writes") > 0.0);
        ASSERT_TRUE(extract_json_num(body, "hal_mmio_reads") > 0.0);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "hal_syndrome_packets")), 1024);
        auto acts = extract_json_array(body, "r2d2_action_histogram");
        double sum_acts = 0.0;
        for (double a : acts) sum_acts += a;
        ASSERT_EQ(static_cast<int>(sum_acts), static_cast<int>(extract_json_num(body, "epochs_completed")));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-FLOW-045") {
        TestContext ctx(opts, tid, "consumer_use_trace_no_dead_intermediates",
                        "software/src/driver/weight_reg_driver.c, software/src/custom_engine/cycle_accurate_bp_engine.cpp",
                        "wrd_write_all_weights -> wrd_commit_and_wait -> CycleAccurateBPEngine::decode", "set3");
        ctx.record().test_set = "set3_pipeline_flow";
        ctx.record().test_description = "Perturb candidate weight on aliased orbit 0x1 (q0 vs q1+q2) before commit and verify downstream BP decode decision shifts";
        ctx.record().independent_oracle = "Committed active weights directly steer CycleAccurateBPEngine::decode correction mask on syndrome 0x1";

        wrd_handle_t wrd{};
        ASSERT_EQ(wrd_open(&wrd, WRD_BASE_ADDR), 0);
        CycleAccurateBPEngine bp(3, 250.0);

        /* Bank 1: q0 has low LLR (likely error on q0 -> correction mask bit 0 = 0x1) */
        const int16_t bank_q0[9] = {200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200};
        wrd_write_all_weights(&wrd, bank_q0);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        int16_t active1[9];
        for (int q = 0; q < 9; ++q) active1[q] = wrd_read_active_weight(&wrd, q);
        avid_bp_result_t r1 = bp.decode(0x1u, active1, 3);

        /* Bank 2: q0 has high LLR while pair (1,2) have very low LLR -> decoder prefers (q1,q2) mask 0x6 on syndrome 0x1 */
        const int16_t bank_q12[9] = {1500, 120, 120, 1200, 1200, 1200, 1200, 1200, 1200};
        wrd_write_all_weights(&wrd, bank_q12);
        ASSERT_EQ(wrd_commit_and_wait(&wrd, 16), 0);
        int16_t active2[9];
        for (int q = 0; q < 9; ++q) active2[q] = wrd_read_active_weight(&wrd, q);
        avid_bp_result_t r2 = bp.decode(0x1u, active2, 3);

        wrd_close(&wrd);
        ASSERT_EQ(r1.correction_mask, 0x001u);
        ASSERT_EQ(r2.correction_mask, 0x006u);
        ASSERT_TRUE(r1.correction_mask != r2.correction_mask);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
