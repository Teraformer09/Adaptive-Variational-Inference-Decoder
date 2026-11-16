#include "test_assertions.h"
#include "test_common.h"
#include "fixture_loader.h"
#include "ingress/packet_validator.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace avid::layer3_test;

namespace {

bool file_exists(const std::string &path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool extract_json_u64(const std::string &json, const std::string &key, uint64_t *out) {
    std::string pat = "\"" + key + "\":";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char *endp = nullptr;
    unsigned long long val = std::strtoull(json.c_str() + pos, &endp, 10);
    if (endp == json.c_str() + pos) return false;
    *out = static_cast<uint64_t>(val);
    return true;
}

bool extract_json_double(const std::string &json, const std::string &key, double *out) {
    std::string pat = "\"" + key + "\":";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char *endp = nullptr;
    double val = std::strtod(json.c_str() + pos, &endp);
    if (endp == json.c_str() + pos) return false;
    *out = val;
    return true;
}

bool extract_json_bool(const std::string &json, const std::string &key, bool *out) {
    std::string pat = "\"" + key + "\":";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    pos += pat.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (json.compare(pos, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

bool extract_json_u64_array16(const std::string &json, const std::string &key, std::array<uint64_t, 16> *out) {
    std::string pat = "\"" + key + "\":";
    size_t pos = json.find(pat);
    if (pos == std::string::npos) return false;
    size_t lb = json.find('[', pos + pat.size());
    size_t rb = json.find(']', lb == std::string::npos ? 0 : lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) return false;
    std::string body = json.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(body);
    std::string tok;
    size_t idx = 0;
    while (std::getline(ss, tok, ',')) {
        if (idx >= 16) return false;
        char *endp = nullptr;
        unsigned long long v = std::strtoull(tok.c_str(), &endp, 10);
        if (endp == tok.c_str()) return false;
        (*out)[idx++] = static_cast<uint64_t>(v);
    }
    return idx == 16;
}

void write_raw_file(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void append_le32(std::vector<uint8_t> &v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x & 0xFFu));
    v.push_back(static_cast<uint8_t>((x >> 8) & 0xFFu));
    v.push_back(static_cast<uint8_t>((x >> 16) & 0xFFu));
    v.push_back(static_cast<uint8_t>((x >> 24) & 0xFFu));
}

} // namespace

int main(int argc, char **argv) {
    CliOptions opts = parse_cli_options(argc, argv, "L3-INT-005");
    TestContext ctx(opts, opts.test_id, "stage6_producer_to_daemon_consumer_compatibility",
                    "software/src/ingress/packet_validator.cpp, software/src/inference/avid_daemon.cpp",
                    "stage6_stratified_sampler.py -> AvidInputAdapter -> stim_to_bin (0xAAEC0001 + dropout flags) -> avid_daemon --stim-bin",
                    "set1");
    ctx.record().test_set = "set1_integration";
    ctx.record().test_description =
        "Upstream Stage-6 producer compatibility — verify avid_daemon --stim-bin parses 16B 0xAAEC0001 Stage-6 headers, rejects malformed/legacy-u16 headers, preserves valid-round H_X/H_Z histograms across readout dropouts without false sequence gaps, and matches live Python Stage-6 output";
    ctx.record().input_contract =
        "16-byte 0xAAEC0001 Stage-6 container (is_u64=1, d=3, 4x4 checks) + 64-bit packet payload with flags=0x00 on readout dropouts";
    ctx.record().independent_oracle =
        "Exact agreement between Stage-6 Python NPZ valid-round syndromes and C++ avid_daemon validator counters + 16-state histograms";
    ctx.record().fixture = "tests/layer3/fixtures/packets/valid/stage6_stratified_dropout_256.bin";

    std::string daemon = opts.daemon_bin;
    ASSERT_TRUE(!daemon.empty());
    std::string ckpt = opts.repo_root + "/checkpoints/r2d2_v1_trained.bin";
    std::string pid_s = std::to_string(::getpid());

    /* 1. Negative 0xAAEC0001 container checks (truncated, legacy u16, invalid code geometry) */
    {
        std::vector<uint8_t> trunc_hdr;
        append_le32(trunc_hdr, 0xAAEC0001u);
        append_le32(trunc_hdr, 4u);
        append_le32(trunc_hdr, 1u);
        trunc_hdr.insert(trunc_hdr.end(), {4u, 4u, 3u, 0u}); /* 16B header claiming 4 rounds, 0B payload */
        std::string p_trunc = "/tmp/avid_s6_trunc_" + pid_s + ".bin";
        write_raw_file(p_trunc, trunc_hdr);
        auto r_trunc = run_command_capture(
            "\"" + daemon + "\" --backend physical --controller-mode heuristic --stim-bin \"" + p_trunc + "\" --rounds 4");
        ::unlink(p_trunc.c_str());
        ASSERT_EQ(r_trunc.exit_code, 2);

        /* Legacy u16 stream: 16B header + 4*2B = 24B (divisible by 8, so ONLY header inspection catches is_u64=0!) */
        std::vector<uint8_t> legacy_u16;
        append_le32(legacy_u16, 0xAAEC0001u);
        append_le32(legacy_u16, 4u);
        append_le32(legacy_u16, 0u); /* is_u64 = 0 */
        legacy_u16.insert(legacy_u16.end(), {4u, 4u, 3u, 0u});
        legacy_u16.resize(24u, 0x01u);
        std::string p_u16 = "/tmp/avid_s6_u16_" + pid_s + ".bin";
        write_raw_file(p_u16, legacy_u16);
        auto r_u16 = run_command_capture(
            "\"" + daemon + "\" --backend physical --controller-mode heuristic --stim-bin \"" + p_u16 + "\" --rounds 4");
        ::unlink(p_u16.c_str());
        ASSERT_EQ(r_u16.exit_code, 2);

        /* Wrong code geometry (d=5, 12x12 checks) */
        std::vector<uint8_t> bad_dim;
        append_le32(bad_dim, 0xAAEC0001u);
        append_le32(bad_dim, 2u);
        append_le32(bad_dim, 1u);
        bad_dim.insert(bad_dim.end(), {12u, 12u, 5u, 0u});
        bad_dim.resize(32u, 0x01u);
        std::string p_dim = "/tmp/avid_s6_dim_" + pid_s + ".bin";
        write_raw_file(p_dim, bad_dim);
        auto r_dim = run_command_capture(
            "\"" + daemon + "\" --backend physical --controller-mode heuristic --stim-bin \"" + p_dim + "\" --rounds 2");
        ::unlink(p_dim.c_str());
        ASSERT_EQ(r_dim.exit_code, 2);
    }

    /* 2. Deterministic Stage-6 256-round fixture with 7 readout dropouts + multi-pass wrap check */
    std::string fix_bin = ctx.fixture_path("packets/valid/stage6_stratified_dropout_256.bin");
    std::string fix_exp_path = ctx.fixture_path("packets/expected/stage6_stratified_dropout_256_expected.json");
    std::string fix_exp_json;
    ASSERT_TRUE(load_text_fixture(fix_exp_path, fix_exp_json));

    uint64_t exp_valid = 0, exp_drop = 0;
    std::array<uint64_t, 16> exp_hx{}, exp_hz{};
    ASSERT_TRUE(extract_json_u64(fix_exp_json, "valid_rounds", &exp_valid));
    ASSERT_TRUE(extract_json_u64(fix_exp_json, "dropout_rounds", &exp_drop));
    ASSERT_TRUE(extract_json_u64_array16(fix_exp_json, "expected_hist_x", &exp_hx));
    ASSERT_TRUE(extract_json_u64_array16(fix_exp_json, "expected_hist_z", &exp_hz));

    std::string m_fix = "/tmp/avid_s6_fix_metrics_" + pid_s + ".json";
    auto r_fix = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + fix_bin + "\" --rounds 256 --window 64 --metrics-out \"" + m_fix + "\"");
    ASSERT_EQ(r_fix.exit_code, 0);

    std::string j_fix;
    ASSERT_TRUE(load_text_fixture(m_fix, j_fix));
    ::unlink(m_fix.c_str());

    bool hdr_det = false;
    uint64_t v_valid = 0, v_inv = 0, v_gaps = 999, rb_cnt = 0, epochs_c = 0;
    double final_nll = 0.0;
    std::array<uint64_t, 16> got_hx{}, got_hz{};
    ASSERT_TRUE(extract_json_bool(j_fix, "stim_stream_header_detected", &hdr_det));
    ASSERT_TRUE(extract_json_u64(j_fix, "validator_valid_packets", &v_valid));
    ASSERT_TRUE(extract_json_u64(j_fix, "validator_invalid_flags", &v_inv));
    ASSERT_TRUE(extract_json_u64(j_fix, "validator_seq_gaps", &v_gaps));
    ASSERT_TRUE(extract_json_u64(j_fix, "ring_buffer_count", &rb_cnt));
    ASSERT_TRUE(extract_json_u64(j_fix, "epochs_completed", &epochs_c));
    ASSERT_TRUE(extract_json_double(j_fix, "final_nll", &final_nll));
    ASSERT_TRUE(extract_json_u64_array16(j_fix, "total_hist_x", &got_hx));
    ASSERT_TRUE(extract_json_u64_array16(j_fix, "total_hist_z", &got_hz));

    ASSERT_TRUE(hdr_det);
    ASSERT_EQ(v_valid, exp_valid);
    ASSERT_EQ(v_inv, exp_drop);
    ASSERT_EQ(v_gaps, 0ULL);
    ASSERT_EQ(rb_cnt, exp_valid);
    ASSERT_EQ(epochs_c, 4ULL);
    ASSERT_FINITE(final_nll);
    ASSERT_TRUE(final_nll > 0.0);
    ASSERT_ARRAY_EQ(got_hx.data(), exp_hx.data(), 16);
    ASSERT_ARRAY_EQ(got_hz.data(), exp_hz.data(), 16);

    /* Multi-pass EOF wrap (512 rounds over 256-round headered file) must seek to byte 16, not 0 */
    std::string m_wrap = "/tmp/avid_s6_wrap_metrics_" + pid_s + ".json";
    auto r_wrap = run_command_capture(
        "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
        "\" --stim-bin \"" + fix_bin + "\" --rounds 512 --window 64 --metrics-out \"" + m_wrap + "\"");
    ASSERT_EQ(r_wrap.exit_code, 0);
    std::string j_wrap;
    ASSERT_TRUE(load_text_fixture(m_wrap, j_wrap));
    ::unlink(m_wrap.c_str());
    uint64_t wrap_valid = 0, wrap_inv = 0, wrap_gaps = 999;
    ASSERT_TRUE(extract_json_u64(j_wrap, "validator_valid_packets", &wrap_valid));
    ASSERT_TRUE(extract_json_u64(j_wrap, "validator_invalid_flags", &wrap_inv));
    ASSERT_TRUE(extract_json_u64(j_wrap, "validator_seq_gaps", &wrap_gaps));
    ASSERT_EQ(wrap_valid, 2ULL * exp_valid);
    ASSERT_EQ(wrap_inv, 2ULL * exp_drop);
    ASSERT_EQ(wrap_gaps, 0ULL);

    /* 3. Live Cross-Branch Stage-6 Producer -> Layer-3 Consumer Verification (when Layer 1 checkout is available) */
    std::string layer1_root;
    const char *env_l1 = std::getenv("AVID_LAYER1_ROOT");
    if (env_l1 && env_l1[0] != '\0' &&
        file_exists(std::string(env_l1) + "/simulation/stimulus/seven_stage/stage6_stratified_sampler.py")) {
        layer1_root = env_l1;
    } else if (file_exists(opts.repo_root + "/../layer1/simulation/stimulus/seven_stage/stage6_stratified_sampler.py")) {
        layer1_root = opts.repo_root + "/../layer1";
    }

    bool live_stage6_verified = false;
    uint64_t live_valid = 0, live_drop = 0;
    if (!layer1_root.empty()) {
        std::string py_script = "/tmp/avid_s6_gen_" + pid_s + ".py";
        std::string live_bin = "/tmp/avid_s6_live_" + pid_s + ".bin";
        std::string live_oracle = "/tmp/avid_s6_live_oracle_" + pid_s + ".json";
        std::string live_metrics = "/tmp/avid_s6_live_metrics_" + pid_s + ".json";

        {
            std::ofstream pf(py_script);
            pf << "import json, sys, numpy as np\n"
               << "sys.path.insert(0, " << "\"" << layer1_root << "\")\n"
               << "from simulation.stimulus.seven_stage.stage6_stratified_sampler import generate_single_stratified_trajectory\n"
               << "from simulation.stimulus.noise_framework.avid_interface.avid_input_adapter import AvidInputAdapter\n"
               << "from simulation.stimulus.stim_to_bin import bits_to_nibble\n"
               << "traj, prov, _ = generate_single_stratified_trajectory(\n"
               << "    trajectory_id='live_l3_int_005', split_id='train', master_seed=10042,\n"
               << "    duration_seconds=25.6, assignment_policy='curriculum_burst_enriched',\n"
               << "    allowed_profile_ids=['PD_06', 'PD_13', 'PD_15'], rounds_per_composite_sample=1)\n"
               << "adapter = AvidInputAdapter()\n"
               << "adapter.write_packet_stream(traj, '" << live_bin << "', u64_format=True)\n"
               << "chk = adapter.validate_stream('" << live_bin << "', expected_trajectory=traj)\n"
               << "vx = np.all(np.asarray(traj['validity_mask_x']) != 0, axis=1)\n"
               << "vz = np.all(np.asarray(traj['validity_mask_z']) != 0, axis=1)\n"
               << "vm = vx & vz\n"
               << "hx = [0]*16\n"
               << "hz = [0]*16\n"
               << "for r in range(int(traj['total_rounds'])):\n"
               << "    if vm[r]:\n"
               << "        hx[bits_to_nibble(traj['syndromes_x'][r])] += 1\n"
               << "        hz[bits_to_nibble(traj['syndromes_z'][r])] += 1\n"
               << "with open('" << live_oracle << "', 'w') as f:\n"
               << "    json.dump({'n_rounds': int(chk['n_rounds']), 'valid_rounds': int(chk['valid_packets']),\n"
               << "               'dropout_rounds': int(chk['dropout_packets']), 'expected_hist_x': hx,\n"
               << "               'expected_hist_z': hz}, f)\n";
        }

        auto r_py = run_command_capture("python3 \"" + py_script + "\"");
        ::unlink(py_script.c_str());
        if (r_py.exit_code != 0) {
            ctx.log("Live Stage-6 Python helper failed: " + r_py.stderr_text);
        }
        ASSERT_EQ(r_py.exit_code, 0);

        std::string j_ora;
        ASSERT_TRUE(load_text_fixture(live_oracle, j_ora));
        ::unlink(live_oracle.c_str());

        uint64_t live_n = 0;
        std::array<uint64_t, 16> live_exp_hx{}, live_exp_hz{};
        ASSERT_TRUE(extract_json_u64(j_ora, "n_rounds", &live_n));
        ASSERT_TRUE(extract_json_u64(j_ora, "valid_rounds", &live_valid));
        ASSERT_TRUE(extract_json_u64(j_ora, "dropout_rounds", &live_drop));
        ASSERT_TRUE(extract_json_u64_array16(j_ora, "expected_hist_x", &live_exp_hx));
        ASSERT_TRUE(extract_json_u64_array16(j_ora, "expected_hist_z", &live_exp_hz));
        ASSERT_EQ(live_n, 256ULL);
        ASSERT_TRUE(live_drop > 0ULL);

        auto r_live = run_command_capture(
            "\"" + daemon + "\" --backend physical --controller-mode avid_r2d2 --checkpoint \"" + ckpt +
            "\" --stim-bin \"" + live_bin + "\" --rounds 256 --window 64 --metrics-out \"" + live_metrics + "\"");
        ::unlink(live_bin.c_str());
        ASSERT_EQ(r_live.exit_code, 0);

        std::string j_live;
        ASSERT_TRUE(load_text_fixture(live_metrics, j_live));
        ::unlink(live_metrics.c_str());

        bool live_hdr = false;
        uint64_t got_lv = 0, got_ld = 0, got_lg = 999;
        double live_nll = 0.0;
        std::array<uint64_t, 16> got_lhx{}, got_lhz{};
        ASSERT_TRUE(extract_json_bool(j_live, "stim_stream_header_detected", &live_hdr));
        ASSERT_TRUE(extract_json_u64(j_live, "validator_valid_packets", &got_lv));
        ASSERT_TRUE(extract_json_u64(j_live, "validator_invalid_flags", &got_ld));
        ASSERT_TRUE(extract_json_u64(j_live, "validator_seq_gaps", &got_lg));
        ASSERT_TRUE(extract_json_double(j_live, "final_nll", &live_nll));
        ASSERT_TRUE(extract_json_u64_array16(j_live, "total_hist_x", &got_lhx));
        ASSERT_TRUE(extract_json_u64_array16(j_live, "total_hist_z", &got_lhz));

        ASSERT_TRUE(live_hdr);
        ASSERT_EQ(got_lv, live_valid);
        ASSERT_EQ(got_ld, live_drop);
        ASSERT_EQ(got_lg, 0ULL);
        ASSERT_FINITE(live_nll);
        ASSERT_TRUE(live_nll > 0.0);
        ASSERT_ARRAY_EQ(got_lhx.data(), live_exp_hx.data(), 16);
        ASSERT_ARRAY_EQ(got_lhz.data(), live_exp_hz.data(), 16);
        live_stage6_verified = true;
    }

    ctx.record().expected["stage6_compatibility"] =
        "0xAAEC0001 header parsed, malformed/u16 headers rejected (rc=2), readout dropouts excluded without false seq gaps, 32/32 histogram bins exact";
    ctx.record().actual["fixture_256r"] =
        "valid=" + std::to_string(v_valid) + ", dropouts=" + std::to_string(v_inv) +
        ", seq_gaps=" + std::to_string(v_gaps) + ", hist_bins_32_match=true, wrap_512_seq_gaps=0";
    ctx.record().actual["live_stage6_python_producer"] =
        live_stage6_verified
            ? ("EXECUTED_AND_MATCHED (valid=" + std::to_string(live_valid) +
               ", dropouts=" + std::to_string(live_drop) + ", seq_gaps=0, hist_bins_32_match=true)")
            : "SKIPPED_STANDALONE_CLONE (verified via deterministic Stage-6 fixture)";

    return ctx.finalize(ctx.has_assertion_failures() ? TestStatus::FAIL : TestStatus::PASS);
}
