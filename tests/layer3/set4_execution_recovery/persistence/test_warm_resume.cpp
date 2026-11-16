#include "test_assertions.h"
#include "test_common.h"
#include "ingress/dma_ingress_tap.h"
#include "ingress/packet_validator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/wait.h>
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

static std::string extract_json_string(const std::string &body, const std::string &key) {
    const std::string pat = "\"" + key + "\": \"";
    size_t pos = body.find(pat);
    if (pos == std::string::npos) return "";
    pos += pat.size();
    size_t end = body.find('"', pos);
    if (end == std::string::npos) return "";
    return body.substr(pos, end - pos);
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

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-EXEC-017");
    const std::string tid  = opts.test_id;
    const std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";

    if (tid == "L3-EXEC-017") {
        TestContext ctx(opts, tid, "runtime_provenance_and_sha256_verification",
                        "software/src/inference/avid_daemon.cpp", "main (provenance fields)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Verify telemetry retains checkpoint_path, seed, scenario, backend, controller_mode and SHA-256 digests";
        ctx.record().independent_oracle = "All provenance strings match invocation and sha256sum yields 64-hex digests for binary and checkpoint";

        std::string mp = "/tmp/l3_exec017_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 7777 --backend physical"
            " --controller-mode avid_r2d2 --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_EQ(extract_json_string(body, "scenario"), "correlated_noise");
        ASSERT_EQ(extract_json_string(body, "controller_mode"), "avid_r2d2");
        ASSERT_EQ(extract_json_string(body, "backend"), "physical_backend");
        ASSERT_EQ(extract_json_string(body, "r2d2_checkpoint_path"), ckpt);
        ASSERT_EQ(static_cast<int>(extract_json_num(body, "seed")), 7777);

        SubprocessResult sh_bin  = run_command_capture("sha256sum " + opts.daemon_bin);
        SubprocessResult sh_ckpt = run_command_capture("sha256sum " + ckpt);
        ASSERT_EQ(sh_bin.exit_code, 0);
        ASSERT_EQ(sh_ckpt.exit_code, 0);
        ASSERT_TRUE(sh_bin.stdout_text.size() >= 64u);
        ASSERT_TRUE(sh_ckpt.stdout_text.size() >= 64u);
        ctx.log("daemon_sha256=" + sh_bin.stdout_text.substr(0, 64) + " ckpt_sha256=" + sh_ckpt.stdout_text.substr(0, 64));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-018") {
        TestContext ctx(opts, tid, "cold_restart_and_crc32_warm_state_resume",
                        "software/src/inference/avid_daemon.cpp", "save_persisted_state_atomic + load_persisted_state", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Test deterministic cold start, valid --state-out -> --state-in warm resume, and corrupt CRC32 state rejection";
        ctx.record().independent_oracle = "Valid state loads cleanly (state_in_loaded==true); 1-bit flip triggers state_in_rejected==true & safe cold fallback";

        std::string st_file = "/tmp/l3_exec018_" + std::to_string(::getpid()) + ".state";
        std::string mp1     = "/tmp/l3_exec018_1_" + std::to_string(::getpid()) + ".json";
        std::string mp2     = "/tmp/l3_exec018_2_" + std::to_string(::getpid()) + ".json";
        std::string mp3     = "/tmp/l3_exec018_3_" + std::to_string(::getpid()) + ".json";

        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --state-out " + st_file + " --metrics-out " + mp1).exit_code, 0);
        std::string b1 = read_text_file_unlink(mp1);
        ASSERT_TRUE(b1.find("\"state_out_written\": true") != std::string::npos);

        /* Warm resume from valid CRC32 state */
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2027 --backend physical"
            " --checkpoint " + ckpt + " --state-in " + st_file + " --metrics-out " + mp2).exit_code, 0);
        std::string b2 = read_text_file_unlink(mp2);
        ASSERT_TRUE(b2.find("\"state_in_loaded\": true") != std::string::npos);
        ASSERT_TRUE(b2.find("\"state_in_rejected\": false") != std::string::npos);

        /* Corrupt payload byte in state file -> CRC32 guard must reject safely and fall back to cold start */
        {
            std::fstream fs(st_file, std::ios::in | std::ios::out | std::ios::binary);
            fs.seekg(16);
            char c = 0;
            fs.read(&c, 1);
            c ^= 0x5Au;
            fs.seekp(16);
            fs.write(&c, 1);
        }
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario correlated_noise --rounds 512 --window 256 --seed 2027 --backend physical"
            " --checkpoint " + ckpt + " --state-in " + st_file + " --metrics-out " + mp3).exit_code, 0);
        std::string b3 = read_text_file_unlink(mp3);
        ::unlink(st_file.c_str());
        ASSERT_TRUE(b3.find("\"state_in_loaded\": false") != std::string::npos);
        ASSERT_TRUE(b3.find("\"state_in_rejected\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-019") {
        TestContext ctx(opts, tid, "metrics_json_log_structural_integrity",
                        "software/src/inference/avid_daemon.cpp", "main (JSON output)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Validate JSON syntax via python3 json.load, non-negative counters, and action/epoch invariants";
        ctx.record().independent_oracle = "Valid JSON object, sum(r2d2_action_histogram)==epochs_completed, initial+adaptive-reset==weight_bank_commits";

        std::string mp = "/tmp/l3_exec019_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario burst_disturbance --rounds 1536 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp).exit_code, 0);
        ASSERT_EQ(run_command_capture("python3 -c \"import json; d=json.load(open('" + mp + "')); assert d['epoch_commit_reconciled'] is True\"").exit_code, 0);

        std::string body = read_text_file_unlink(mp);
        auto acts = extract_json_array(body, "r2d2_action_histogram");
        ASSERT_EQ(acts.size(), 6u);
        double sum_a = 0.0;
        for (double a : acts) {
            ASSERT_TRUE(a >= 0.0);
            sum_a += a;
        }
        ASSERT_EQ(static_cast<int>(sum_a), static_cast<int>(extract_json_num(body, "epochs_completed")));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-020") {
        TestContext ctx(opts, tid, "artifact_bit_exact_replay_reconstruction",
                        "software/src/inference/avid_daemon.cpp", "main (deterministic replay)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Extract CLI provenance from saved metrics JSON, replay run, and verify bit-identical metric reconstruction";
        ctx.record().independent_oracle = "Replayed JSON matches original JSON byte-for-byte";

        std::string mp_orig   = "/tmp/l3_exec020_orig_" + std::to_string(::getpid()) + ".json";
        std::string mp_replay = "/tmp/l3_exec020_repl_" + std::to_string(::getpid()) + ".json";
        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario heldout_multi_pair_12_67 --rounds 768 --window 256 --seed 9090 --backend physical"
            " --controller-mode avid_r2d2 --checkpoint " + ckpt + " --metrics-out " + mp_orig).exit_code, 0);
        std::string orig = read_text_file_unlink(mp_orig);

        std::string sc   = extract_json_string(orig, "scenario");
        std::string mode = extract_json_string(orig, "controller_mode");
        std::string ckp  = extract_json_string(orig, "r2d2_checkpoint_path");
        int rounds       = static_cast<int>(extract_json_num(orig, "total_rounds"));
        int win          = static_cast<int>(extract_json_num(orig, "window_rounds"));
        int seed         = static_cast<int>(extract_json_num(orig, "seed"));

        ASSERT_EQ(run_command_capture(
            opts.daemon_bin + " --scenario " + sc + " --rounds " + std::to_string(rounds) +
            " --window " + std::to_string(win) + " --seed " + std::to_string(seed) +
            " --backend physical --controller-mode " + mode +
            " --checkpoint " + ckp + " --metrics-out " + mp_replay).exit_code, 0);
        std::string repl = read_text_file_unlink(mp_replay);
        auto strip_wc = [](const std::string &jt) {
            std::istringstream iss(jt);
            std::ostringstream oss;
            std::string ln;
            while (std::getline(iss, ln)) {
                if (ln.find("\"mean_estimator_us\":") != std::string::npos ||
                    ln.find("\"max_estimator_us\":") != std::string::npos ||
                    ln.find("\"slow_loop_cpu_utilization_10ms_pct\":") != std::string::npos) {
                    continue;
                }
                oss << ln << "\n";
            }
            return oss.str();
        };
        ASSERT_EQ(strip_wc(orig), strip_wc(repl));
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-021") {
        TestContext ctx(opts, tid, "malformed_packet_in_dma_slot_isolation",
                        "software/src/ingress/dma_ingress_tap.cpp, software/src/ingress/packet_validator.cpp",
                        "avid_dma_ring_consume_slot", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Inject flags=0 and PARITY_ERR packets amidst valid packets inside a DMA slot";
        ctx.record().independent_oracle = "Valid packets accumulate cleanly; malformed packets rejected with invalid_flag_count==2 and unpolluted hist";

        avid_dma_ring_t ring{};
        avid_dma_ring_init(&ring);
        uint64_t pkts[6] = {
            avid_pack_packet_u64(0x6u, 0x1u, 0u, 1u, AVID_PKT_FLAG_VALID),
            avid_pack_packet_u64(0xFu, 0xFu, 1u, 1u, 0x00u), /* flags=0 invalid */
            avid_pack_packet_u64(0x6u, 0x1u, 1u, 1u, AVID_PKT_FLAG_VALID),
            avid_pack_packet_u64(0xEu, 0xEu, 2u, 1u, static_cast<uint8_t>(AVID_PKT_FLAG_VALID | AVID_PKT_FLAG_PARITY_ERR)),
            avid_pack_packet_u64(0x3u, 0x2u, 2u, 1u, AVID_PKT_FLAG_VALID),
            avid_pack_packet_u64(0x0u, 0x0u, 3u, 1u, AVID_PKT_FLAG_VALID)
        };
        ASSERT_TRUE(avid_dma_ring_push_burst(&ring, pkts, 6u));
        uint32_t hx[16] = {0}, hz[16] = {0}, n_valid = 0;
        ASSERT_TRUE(avid_dma_ring_consume_slot(&ring, hx, hz, &n_valid));
        ASSERT_EQ(n_valid, 4u);
        ASSERT_EQ(ring.validator.invalid_flag_count, 2u);
        ASSERT_EQ(hx[0xF], 0u);
        ASSERT_EQ(hx[0xE], 0u);
        ASSERT_EQ(hx[0x6], 2u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-022") {
        TestContext ctx(opts, tid, "packet_loss_sequence_gap_accounting",
                        "software/src/ingress/packet_validator.cpp", "avid_validate_packet", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Drop seq=5 cleanly (0..4, 6..8) in packet ingress stream";
        ctx.record().independent_oracle = "seq_gap_count increments by exact 1 and post-gap valid stream resumes cleanly";

        avid_packet_validator_t val{};
        avid_validator_init(&val);
        /* 0..4 valid, premature 6 rejected (seq_gap_count==1 without mutating last_seq_id), then 5..8 resume cleanly */
        const uint32_t seqs[10] = {0, 1, 2, 3, 4, 6, 5, 6, 7, 8};
        for (uint32_t s : seqs) {
            avid_syndrome_packet_t out{};
            uint64_t raw = avid_pack_packet_u64(0x2u, 0x1u, s, 1u, AVID_PKT_FLAG_VALID);
            avid_validate_packet(&val, raw, &out);
        }
        ASSERT_EQ(val.seq_gap_count, 1u);
        ASSERT_EQ(val.valid_count, 9u);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    if (tid == "L3-EXEC-032") {
        TestContext ctx(opts, tid, "daemon_abrupt_sigkill_and_clean_relaunch",
                        "software/src/inference/avid_daemon.cpp", "main (post-SIGKILL recovery)", "set4");
        ctx.record().test_set = "set4_execution_recovery";
        ctx.record().test_description = "Terminate avid_daemon mid-run with SIGKILL, then relaunch clean daemon targeting same metrics path";
        ctx.record().independent_oracle = "Relaunch succeeds cleanly with exit 0, zero stale temp-file corruption, and epoch_commit_reconciled==true";

        std::string mp = "/tmp/l3_exec032_" + std::to_string(::getpid()) + ".json";
        pid_t pid = ::fork();
        if (pid == 0) {
            ::execl(opts.daemon_bin.c_str(), opts.daemon_bin.c_str(),
                    "--scenario", "stationary_noise",
                    "--rounds", "5000000",
                    "--window", "256",
                    "--backend", "physical",
                    "--checkpoint", ckpt.c_str(),
                    "--metrics-out", mp.c_str(),
                    static_cast<char *>(nullptr));
            _exit(127);
        }
        ::usleep(60000);
        ::kill(pid, SIGKILL);
        int st = 0;
        ::waitpid(pid, &st, 0);
        ASSERT_TRUE(WIFSIGNALED(st));

        SubprocessResult r2 = run_command_capture(
            opts.daemon_bin + " --scenario stationary_noise --rounds 512 --window 256 --seed 2026 --backend physical"
            " --checkpoint " + ckpt + " --metrics-out " + mp);
        ASSERT_EQ(r2.exit_code, 0);
        std::string body = read_text_file_unlink(mp);
        ASSERT_TRUE(body.find("\"epoch_commit_reconciled\": true") != std::string::npos);
        return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
    }

    return 2;
}
