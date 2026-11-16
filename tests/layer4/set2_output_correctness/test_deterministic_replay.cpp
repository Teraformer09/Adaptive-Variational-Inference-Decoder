/* ===========================================================================
 * test_deterministic_replay.cpp — Layer 4 Set 2.4 Deterministic Replay Tests
 * Canonical IDs: L4-DET-01 .. L4-DET-06 (Gates G2, G8)
 * Executes 5 independent Bridge+Verilator runs and verifies bit-exact traces.
 * =========================================================================*/
#include "../common/l4_test_harness.hpp"
#include "../../../software/src/hal/avid_hal.h"

struct ReplaySnapshot {
    std::string trace_content;
    std::string metrics_content;
    std::vector<uint16_t> corr_x_seq;
    std::vector<uint16_t> corr_z_seq;
    std::vector<uint64_t> comp_cyc_seq;
    uint32_t final_epoch = 0;
    uint32_t active_checksum = 0;
    uint64_t final_cycle = 0;
};

static bool execute_single_replay(const std::string &bridge_bin, int run_idx, ReplaySnapshot &out) {
    avid_l4::BridgeProcessGuard br;
    if (!br.start(bridge_bin, "set2_det_" + std::to_string(run_idx), 4, true)) return false;

    avid_hal_t hal;
    if (avid_hal_open(&hal, AVID_HAL_BACKEND_SIMULATION, br.sock_path.c_str(), WRD_BASE_ADDR) != 0) return false;

    uint64_t cyc = 0;
    if (avid_hal_reset_rtl(&hal, 12u, &cyc) != 0) return false;
    if (avid_hal_set_bp_iterations(&hal, 4u, &cyc) != 0) return false;
    if (avid_hal_set_osd_mode(&hal, 1u, &cyc) != 0) return false;

    for (int ep = 0; ep < 3; ep++) {
        if (ep == 1) {
            /* Inject deterministic mid-session RESET and reconfigure */
            if (avid_hal_reset_rtl(&hal, 8u, &cyc) != 0) return false;
            if (avid_hal_set_bp_iterations(&hal, 3u, &cyc) != 0) return false;
            if (avid_hal_set_osd_mode(&hal, 2u, &cyc) != 0) return false;
        }
        int16_t wx[9], wz[9];
        for (int q = 0; q < 9; q++) {
            wx[q] = static_cast<int16_t>(0x0130 + ep * 17 + q * 5);
            wz[q] = static_cast<int16_t>(0x01F0 - ep * 11 + q * 7);
        }
        uint32_t hw_ep = 0;
        if (avid_hal_write_weights_and_commit(&hal, wx, wz, &hw_ep, &cyc) != 0) return false;
        out.final_epoch = hw_ep;

        for (int r = 0; r < 16; r++) {
            uint8_t sx = static_cast<uint8_t>((ep * 5 + r * 3 + 1) & 0xF);
            uint8_t sz = static_cast<uint8_t>((ep * 7 + r * 9 + 2) & 0xF);
            uint16_t cx = 0, cz = 0;
            uint32_t bp = 0;
            if (avid_hal_send_syndrome(&hal, sx, sz, (r == 15), &cx, &cz, &bp, &cyc) != 0) return false;
            out.corr_x_seq.push_back(cx);
            out.corr_z_seq.push_back(cz);
            out.comp_cyc_seq.push_back(hal.stats.last_completed_cycle);
        }
    }

    uint32_t csum = 0;
    for (int q = 0; q < 9; q++) {
        uint32_t ax = 0, az = 0;
        avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_X(q), &ax, &cyc);
        avid_hal_mmio_read(&hal, WRD_OFF_ACTIVE_Z(q), &az, &cyc);
        csum = csum * 131u + ax * 17u + az;
    }
    out.active_checksum = csum;
    out.final_cycle = hal.stats.last_completed_cycle;

    avid_hal_close(&hal);
    br.graceful_shutdown(999999ULL);

    std::ifstream tf(br.tx_log_path);
    out.trace_content.assign((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    std::ifstream mf(br.metrics_path);
    out.metrics_content.assign((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
    unlink(br.tx_log_path.c_str());
    unlink(br.metrics_path.c_str());
    return !out.trace_content.empty() && !out.metrics_content.empty();
}

int main(int argc, char **argv) {
    avid_l4::L4Suite suite("set2_deterministic_replay", "Set2", argc, argv);
    const std::string sim_mode = "verilator_c++_rtl";

    ReplaySnapshot runs[5];
    bool all_ran = true;
    for (int i = 0; i < 5; i++) {
        if (!execute_single_replay(suite.bridge_bin_path(), i, runs[i])) {
            all_ran = false;
            break;
        }
    }
    if (!all_ran) return 2;

    suite.run_case("L4-DET-01", "G2,G8", "5 independent Verilator bridge runs produce identical corr_x and corr_z sequences", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        for (int i = 1; i < 5; i++) {
            if (runs[i].corr_x_seq != runs[0].corr_x_seq || runs[i].corr_z_seq != runs[0].corr_z_seq) return false;
        }
        ev = "48x5 corrections bit-exact across all 5 runs";
        return runs[0].corr_x_seq.size() == 48u;
    });

    suite.run_case("L4-DET-02", "G2,G8", "5 independent runs produce identical accepted_cycle & completed_cycle timelines", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        for (int i = 1; i < 5; i++) {
            if (runs[i].comp_cyc_seq != runs[0].comp_cyc_seq || runs[i].final_cycle != runs[0].final_cycle) return false;
        }
        ev = "final_cycle=" + std::to_string(cyc) + " identical across 5 runs";
        return true;
    });

    suite.run_case("L4-DET-03", "G2,G8", "5 independent runs converge to identical post-reset epoch_count and active register bank", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        for (int i = 1; i < 5; i++) {
            if (runs[i].final_epoch != runs[0].final_epoch ||
                runs[i].active_checksum != runs[0].active_checksum) return false;
        }
        ev = "final_epoch=" + std::to_string(runs[0].final_epoch) +
             " active_checksum=" + std::to_string(runs[0].active_checksum);
        return runs[0].final_epoch == 2u;
    });

    suite.run_case("L4-DET-04", "G2,G8", "m3_transactions.jsonl files are byte-for-byte identical across all 5 runs", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        for (int i = 1; i < 5; i++) {
            if (runs[i].trace_content != runs[0].trace_content) return false;
        }
        ev = "JSONL bytes=" + std::to_string(runs[0].trace_content.size()) + " byte-identical across 5 runs";
        return true;
    });

    suite.run_case("L4-DET-05", "G2,G8", "Bridge --metrics-out JSON files are byte-for-byte identical across all 5 runs", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        for (int i = 1; i < 5; i++) {
            if (runs[i].metrics_content != runs[0].metrics_content) return false;
        }
        ev = "metrics JSON bytes=" + std::to_string(runs[0].metrics_content.size()) + " byte-identical";
        return true;
    });

    suite.run_case("L4-DET-06", "G2,G8", "Verify zero wall-clock jitter leakage into simulated RTL time (sim_time_ns == 4 * sim_cycle)", sim_mode,
                   [&](uint64_t &cyc, std::string &ev) {
        cyc = runs[0].final_cycle;
        uint64_t f_cyc = 0, f_ns = 0;
        avid_l4::extract_json_u64_simple(runs[0].metrics_content, "final_sim_cycle", &f_cyc);
        avid_l4::extract_json_u64_simple(runs[0].metrics_content, "final_sim_time_ns", &f_ns);
        ev = "final_sim_cycle=" + std::to_string(f_cyc) + " final_sim_time_ns=" + std::to_string(f_ns);
        return f_cyc > 0ULL && f_ns == f_cyc * 4ULL;
    });

    return suite.finalize();
}
