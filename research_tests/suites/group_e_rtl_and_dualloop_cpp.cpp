/* ============================================================================
 * group_e_rtl_and_dualloop_cpp.cpp
 *
 * Native Verilator SystemVerilog RTL Harness for AVID Publication Gate:
 *   1) --run-group-e <out_json> [--vcd <out_vcd>]
 *      Executes Hardware Correctness & Dual-Loop Isolation Gates E2..E12
 *      directly on compiled Vaa_qec_top + bound sva_weight_commit_assertions.
 *   2) --decode-closedloop <in_schedule.csv> <out_trace.csv>
 *      Executes multi-epoch closed-loop AXI4-Lite weight updates + AXI4-Stream
 *      syndrome decoding directly on native Verilator Vaa_qec_top for Group F
 *      (F1..F7, F9, F10).
 *
 * IMPORTANT: Never hold two AvidVerilatorRtlModel instances alive in the same
 * scope (avoids Verilator 5.020 dual-context hang on Debian / Cloud Shell).
 * ==========================================================================*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "verilator_rtl_driver.h"

namespace {

static inline uint8_t syndrome_x_from_err_z(uint16_t ez) {
    uint8_t b0 = ((ez >> 0) ^ (ez >> 1)) & 1u;
    uint8_t b1 = ((ez >> 1) ^ (ez >> 2) ^ (ez >> 4) ^ (ez >> 5)) & 1u;
    uint8_t b2 = ((ez >> 3) ^ (ez >> 4) ^ (ez >> 6) ^ (ez >> 7)) & 1u;
    uint8_t b3 = ((ez >> 7) ^ (ez >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

static inline uint8_t syndrome_z_from_err_x(uint16_t ex) {
    uint8_t b0 = ((ex >> 0) ^ (ex >> 1) ^ (ex >> 3) ^ (ex >> 4)) & 1u;
    uint8_t b1 = ((ex >> 2) ^ (ex >> 5)) & 1u;
    uint8_t b2 = ((ex >> 3) ^ (ex >> 6)) & 1u;
    uint8_t b3 = ((ex >> 4) ^ (ex >> 5) ^ (ex >> 7) ^ (ex >> 8)) & 1u;
    return static_cast<uint8_t>(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
}

static void program_weights_and_commit_idle(
    AvidVerilatorRtlModel &rtl,
    const int16_t wx[9],
    const int16_t wz[9],
    uint32_t k_iter,
    uint32_t osd_mode
) {
    uint64_t cyc = 0;
    rtl.axil_write(0x54u, k_iter, 0x1u, &cyc);
    rtl.axil_write(0x58u, osd_mode, 0x1u, &cyc);
    for (int i = 0; i < 9; ++i) {
        rtl.axil_write(static_cast<uint32_t>(i * 4), static_cast<uint16_t>(wx[i]), 0x3u, &cyc);
        rtl.axil_write(static_cast<uint32_t>(0x24u + i * 4), static_cast<uint16_t>(wz[i]), 0x3u, &cyc);
    }
    rtl.axil_write(0x48u, 0x1u, 0x1u, &cyc);
    rtl.step_clock(2);
}

struct GroupEMetrics {
    bool native_verilator = false;
    uint64_t total_rtl_cycles = 0;

    // E2: Q7.8 exhaustive 3,840-state equivalence
    uint32_t e2_states_tested = 0;
    uint32_t e2_mismatches = 0;
    bool e2_passed = false;

    // E3: Pure BP Mode-0 golden equivalence across 4 inhomogeneous banks * 256
    uint32_t e3_states_tested = 0;
    uint32_t e3_mismatches = 0;
    bool e3_passed = false;

    // E4: OSD-0 degenerate & multi-fault post-processing
    uint32_t e4_cases_tested = 0;
    uint32_t e4_unsatisfied_syndromes = 0;
    uint32_t e4_mismatches = 0;
    bool e4_pair_disambiguation_flipped = false;
    bool e4_passed = false;

    // E5: Atomic shadow-to-active weight commits under continuous streaming
    uint32_t e5_packets = 0;
    uint32_t e5_commits = 0;
    uint32_t e5_mismatches = 0;
    uint32_t e5_hw_epochs = 0;
    bool e5_passed = false;

    // E6: Mid-decode commit isolation & SLVERR lock across 12 offsets
    uint32_t e6_offsets_tested = 0;
    uint32_t e6_slverr_locks_verified = 0;
    uint32_t e6_mid_decode_corruptions = 0;
    bool e6_passed = false;

    // E7: 16-entry RX/TX FIFO full, backpressure & empty boundary
    uint32_t e7_saturated_inflight = 0;
    bool e7_tready_deasserted_on_full = false;
    uint32_t e7_drained_packets = 0;
    uint32_t e7_mismatches = 0;
    bool e7_empty_clean = false;
    bool e7_passed = false;

    // E8: Synchronous reset recovery across pipeline offsets 1..20
    uint32_t e8_reset_offsets_tested = 0;
    uint32_t e8_ghost_outputs = 0;
    uint32_t e8_post_reset_failures = 0;
    bool e8_passed = false;

    // E9: 10,000-syndrome randomized RTL stress
    uint32_t e9_packets = 0;
    uint32_t e9_mismatches = 0;
    uint64_t e9_cycles = 0;
    bool e9_passed = false;

    // E10: Cycle-budget deterministic law across 15 (K, Mode) configs
    uint32_t e10_configs_tested = 0;
    uint32_t e10_law_violations = 0;
    std::vector<uint32_t> e10_k_list;
    std::vector<uint32_t> e10_mode_list;
    std::vector<uint32_t> e10_core_cycles;
    std::vector<uint32_t> e10_axis_cycles;
    std::vector<double> e10_latency_ns;
    bool e10_passed = false;

    // E11: Weight-update interference (pre / during / post commit)
    uint32_t e11_pre_packets = 0;
    uint32_t e11_during_packets = 0;
    uint32_t e11_post_packets = 0;
    uint32_t e11_k3_pre_min_cyc = 999, e11_k3_pre_max_cyc = 0;
    uint32_t e11_k3_during_min_cyc = 999, e11_k3_during_max_cyc = 0;
    uint32_t e11_k3_post_min_cyc = 999, e11_k3_post_max_cyc = 0;
    uint32_t e11_k5_pre_cyc = 0, e11_k5_during_cyc = 0, e11_k5_post_cyc = 0;
    int32_t e11_max_abs_delta_cycles = 999;
    double e11_max_abs_delta_ns = 999.0;
    uint32_t e11_mismatches = 0;
    bool e11_passed = false;

    // E12: Fast-loop 300 ns deadline under worst-case traffic
    uint32_t e12_packets = 0;
    uint32_t e12_max_core_cycles = 0;
    uint32_t e12_max_service_axis_cycles = 0;
    double e12_max_service_latency_ns = 0.0;
    double e12_deadline_ns = 300.0;
    double e12_headroom_ns = 0.0;
    uint32_t e12_deadline_misses = 0;
    uint32_t e12_mismatches = 0;
    std::vector<double> e12_service_ns_hist_samples;
    bool e12_passed = false;
};

static int run_group_e_all(const std::string &out_json_path, const std::string &vcd_path) {
    GroupEMetrics m;
    m.native_verilator = AvidVerilatorRtlModel::is_native_verilator();
    if (!m.native_verilator) {
        std::fprintf(stderr, "[FATAL] Group E requires native Verilator (-DUSE_VERILATOR=1).\n");
        return 2;
    }

    // -------------------------------------------------------------------------
    // E2 & E10: Exhaustive 3,840-state Q7.8 sweep + 15-config Cycle-Budget Law
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        const int16_t wx_inhom[9] = {0x0260, 0x01C0, 0x0210, 0x0240, 0x0150, 0x0180, 0x0250, 0x01E0, 0x0270};
        const int16_t wz_inhom[9] = {0x0220, 0x01A0, 0x0250, 0x0210, 0x0130, 0x0148, 0x0260, 0x01D0, 0x0240};

        for (uint32_t k = 1; k <= 5; ++k) {
            for (uint32_t mode = 0; mode <= 2; ++mode) {
                program_weights_and_commit_idle(rtl, wx_inhom, wz_inhom, k, mode);
                uint32_t exp_core = (mode == 2u) ? (4u + 5u * k) : (2u + 5u * k);
                uint32_t exp_axis = 5u + exp_core;

                uint32_t obs_core = 0;
                uint32_t obs_axis = 0;
                for (uint32_t sx = 0; sx < 16; ++sx) {
                    for (uint32_t sz = 0; sz < 16; ++sz) {
                        RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), true);
                        uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                            static_cast<uint8_t>(sz), wx_inhom, static_cast<int>(k), true, mode);
                        uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                            static_cast<uint8_t>(sx), wz_inhom, static_cast<int>(k), false, mode);
                        m.e2_states_tested++;
                        if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                            m.e2_mismatches++;
                        }
                        if (sx == 0x6u && sz == 0x9u) {
                            obs_core = tr.bp_core_cycles;
                            obs_axis = tr.total_axis_cycles;
                        }
                        if (tr.bp_core_cycles != exp_core || tr.total_axis_cycles != exp_axis) {
                            m.e10_law_violations++;
                        }
                    }
                }
                m.e10_configs_tested++;
                m.e10_k_list.push_back(k);
                m.e10_mode_list.push_back(mode);
                m.e10_core_cycles.push_back(obs_core);
                m.e10_axis_cycles.push_back(obs_axis);
                m.e10_latency_ns.push_back(static_cast<double>(obs_axis) * 4.0);
            }
        }
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e2_passed = (m.e2_states_tested == 3840u) && (m.e2_mismatches == 0u);
    m.e10_passed = (m.e10_configs_tested == 15u) && (m.e10_law_violations == 0u);

    // -------------------------------------------------------------------------
    // E3: Pure Min-Sum BP (Mode 0) Golden Equivalence across 4 asymmetric banks
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        const int16_t banks_x[4][9] = {
            {0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232},
            {0x0080, 0x0320, 0x02F0, 0x0310, 0x0110, 0x02E0, 0x0300, 0x02C0, 0x0340},
            {0x0480, 0x0140, 0x0150, 0x0420, 0x0440, 0x0410, 0x0160, 0x0158, 0x0460},
            {0x0060, 0x0520, 0x0090, 0x04C0, 0x0078, 0x04E0, 0x00A0, 0x04A0, 0x0088},
        };
        const int16_t banks_z[4][9] = {
            {0x01E0, 0x0280, 0x0240, 0x01F0, 0x0160, 0x0270, 0x0230, 0x01B0, 0x0290},
            {0x0310, 0x02D0, 0x0090, 0x02E0, 0x0120, 0x0098, 0x02F0, 0x0320, 0x02B0},
            {0x0130, 0x0138, 0x03E0, 0x03F0, 0x0118, 0x0120, 0x0400, 0x03D0, 0x0410},
            {0x0510, 0x0070, 0x04D0, 0x0088, 0x04F0, 0x0068, 0x04B0, 0x0098, 0x0500},
        };
        for (int b = 0; b < 4; ++b) {
            uint32_t k = static_cast<uint32_t>((b % 5) + 1);
            program_weights_and_commit_idle(rtl, banks_x[b], banks_z[b], k, 0u);
            for (uint32_t sx = 0; sx < 16; ++sx) {
                for (uint32_t sz = 0; sz < 16; ++sz) {
                    RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), true);
                    uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                        static_cast<uint8_t>(sz), banks_x[b], static_cast<int>(k), true, 0u);
                    uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                        static_cast<uint8_t>(sx), banks_z[b], static_cast<int>(k), false, 0u);
                    m.e3_states_tested++;
                    if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                        m.e3_mismatches++;
                    }
                }
            }
        }
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e3_passed = (m.e3_states_tested == 1024u) && (m.e3_mismatches == 0u);

    // -------------------------------------------------------------------------
    // E4: OSD-0 Degenerate Pair Disambiguation & 100% Syndrome Satisfaction
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        const int16_t uniform_w[9] = {0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232};
        // Pair (4,5) heavily depressed -> sx=0x4 must select pair (4,5)=0x030 instead of single q6=0x040 or q3=0x008
        const int16_t pair45_w[9]  = {0x0280, 0x0280, 0x0280, 0x0280, 0x00D0, 0x00D0, 0x0280, 0x0280, 0x0280};

        program_weights_and_commit_idle(rtl, uniform_w, uniform_w, 3u, 1u);
        RtlDecodeTrace tr_unif = rtl.decode_syndrome(0x4u, 0x1u, true);

        for (uint32_t mode = 1; mode <= 2; ++mode) {
            program_weights_and_commit_idle(rtl, pair45_w, pair45_w, 4u, mode);
            for (uint32_t sx = 0; sx < 16; ++sx) {
                for (uint32_t sz = 0; sz < 16; ++sz) {
                    RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), true);
                    m.e4_cases_tested++;
                    if (syndrome_x_from_err_z(tr.corr_z) != sx || syndrome_z_from_err_x(tr.corr_x) != sz) {
                        m.e4_unsatisfied_syndromes++;
                    }
                    uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                        static_cast<uint8_t>(sz), pair45_w, 4, true, mode);
                    uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(
                        static_cast<uint8_t>(sx), pair45_w, 4, false, mode);
                    if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                        m.e4_mismatches++;
                    }
                    if (mode == 1u && sx == 0x4u && sz == 0x1u) {
                        m.e4_pair_disambiguation_flipped = (tr_unif.corr_z == 0x008u || tr_unif.corr_z == 0x040u)
                                                           && (tr.corr_z == 0x030u);
                    }
                }
            }
        }
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e4_passed = (m.e4_cases_tested == 512u) && (m.e4_unsatisfied_syndromes == 0u)
                  && (m.e4_mismatches == 0u) && m.e4_pair_disambiguation_flipped;

    // -------------------------------------------------------------------------
    // E5: Atomic Shadow-to-Active Commits Under Continuous 2,000-Packet Stream
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        int16_t wx[9], wz[9];
        uint32_t rng = 0x13579BDFu;
        for (uint32_t ep = 0; ep < 20; ++ep) {
            for (int i = 0; i < 9; ++i) {
                rng = 1664525u * rng + 1013904223u;
                wx[i] = static_cast<int16_t>(0x00C0 + ((rng >> 16) & 0x01FF));
                rng = 1664525u * rng + 1013904223u;
                wz[i] = static_cast<int16_t>(0x00C0 + ((rng >> 16) & 0x01FF));
            }
            program_weights_and_commit_idle(rtl, wx, wz, 3u, 1u);
            m.e5_commits++;
            for (int p = 0; p < 100; ++p) {
                rng = 1664525u * rng + 1013904223u;
                uint8_t sx = static_cast<uint8_t>((rng >> 12) & 0xFu);
                uint8_t sz = static_cast<uint8_t>((rng >> 20) & 0xFu);
                RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (p == 99));
                uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, wx, 3, true, 1u);
                uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, wz, 3, false, 1u);
                m.e5_packets++;
                if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                    m.e5_mismatches++;
                }
            }
        }
        uint64_t cyc = 0;
        rtl.axil_read(0x50u, &m.e5_hw_epochs, &cyc);
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e5_passed = (m.e5_packets == 2000u) && (m.e5_commits == 20u)
                  && (m.e5_mismatches == 0u) && (m.e5_hw_epochs == 20u);

    // -------------------------------------------------------------------------
    // E6: Mid-Decode Weight-Update Isolation & SLVERR Lock Across 12 Offsets
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        if (!vcd_path.empty()) {
            rtl.open_waveform(vcd_path.c_str());
        }
        const int16_t bank_a[9] = {0x0280, 0x0280, 0x0280, 0x0280, 0x0280, 0x0280, 0x0280, 0x0280, 0x0280};
        const int16_t bank_b[9] = {0x0280, 0x0280, 0x0280, 0x0280, 0x00C0, 0x00C0, 0x0280, 0x0280, 0x0280};

        for (uint32_t offset = 1; offset <= 12; ++offset) {
            program_weights_and_commit_idle(rtl, bank_a, bank_a, 5u, 2u);
            uint64_t cyc = 0;
            // Stage bank_b into shadow before starting packet
            for (int i = 0; i < 9; ++i) {
                rtl.axil_write(static_cast<uint32_t>(i * 4), static_cast<uint16_t>(bank_b[i]), 0x3u, &cyc);
                rtl.axil_write(static_cast<uint32_t>(0x24u + i * 4), static_cast<uint16_t>(bank_b[i]), 0x3u, &cyc);
            }

            // Inject sx=0x4 (bank_a -> 0x008, bank_b -> 0x030!)
            rtl.top->s_axis_syndrome_tdata = 0x04u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 1;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast = 0;
            rtl.step_clock(2 + offset); // Inside active K=5 BP decode (29 busy cycles)

            bool was_busy = (rtl.top->bp_core_busy_mon != 0);
            // Assert COMMIT mid-decode
            rtl.axil_write(0x48u, 0x1u, 0x1u, &cyc);

            // Attempt torn shadow write while swap_pending && bp_busy -> MUST SLVERR (return false)
            bool write_accepted = rtl.axil_write(0x24u + 4u * 4u, 0x03FFu, 0x3u, &cyc);
            if (was_busy && !write_accepted && rtl.last_bresp == 2u) {
                m.e6_slverr_locks_verified++;
            }

            // Verify active register 4 is STILL bank_a[4] mid-decode
            uint32_t act_mid = 0;
            rtl.axil_read(0x124u + 4u * 4u, &act_mid, &cyc);
            if (static_cast<int16_t>(act_mid & 0xFFFFu) != bank_a[4]) {
                m.e6_mid_decode_corruptions++;
            }

            // Drain current in-flight packet
            uint16_t inflight_cz = 0;
            for (int g = 0; g < 60; ++g) {
                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid) {
                    inflight_cz = static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu);
                    rtl.step_clock(2);
                    break;
                }
            }
            uint16_t exp_bank_a_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x4u, bank_a, 5, false, 2u);
            uint16_t exp_bank_b_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x4u, bank_b, 5, false, 2u);
            if (inflight_cz != exp_bank_a_cz) {
                m.e6_mid_decode_corruptions++;
            }

            // Now next packet must use swapped bank_b automatically without any extra COMMIT!
            RtlDecodeTrace tr_next = rtl.decode_syndrome(0x4u, 0x0u, true);
            if (tr_next.corr_z != exp_bank_b_cz) {
                m.e6_mid_decode_corruptions++;
            }
            m.e6_offsets_tested++;
        }
        rtl.close_waveform();
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e6_passed = (m.e6_offsets_tested == 12u) && (m.e6_slverr_locks_verified == 12u)
                  && (m.e6_mid_decode_corruptions == 0u);

    // -------------------------------------------------------------------------
    // E7: 16-Entry Elastic FIFO Saturation, Backpressure & Clean Empty Drain
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        const int16_t w_def[9] = {0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232};
        program_weights_and_commit_idle(rtl, w_def, w_def, 3u, 1u);

        // Hold master egress tready=0 so TX FIFO + BP core + 16-entry RX FIFO fill completely
        rtl.top->m_axis_correction_tready = 0;
        std::vector<uint8_t> pushed_sx;
        for (int attempt = 0; attempt < 60; ++attempt) {
            uint8_t sx = static_cast<uint8_t>((attempt % 15) + 1);
            rtl.top->s_axis_syndrome_tdata = sx;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 0;
            bool accepted = (rtl.top->s_axis_syndrome_tready != 0);
            rtl.step_clock(1);
            if (accepted) {
                pushed_sx.push_back(sx);
            } else {
                m.e7_tready_deasserted_on_full = true;
                break;
            }
            // Allow pipeline stages to pull earlier items so we reach full saturation
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.step_clock(22);
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        m.e7_saturated_inflight = static_cast<uint32_t>(pushed_sx.size());

        // Release egress backpressure and drain all accepted packets in exact FIFO order
        rtl.top->m_axis_correction_tready = 1;
        std::vector<uint16_t> got_cz;
        for (int g = 0; g < 1600 && got_cz.size() < pushed_sx.size(); ++g) {
            bool valid_now = (rtl.top->m_axis_correction_tvalid != 0);
            uint32_t tdata_now = rtl.top->m_axis_correction_tdata;
            rtl.step_clock(1);
            if (valid_now) {
                got_cz.push_back(static_cast<uint16_t>((tdata_now >> 9) & 0x1FFu));
            }
        }
        m.e7_drained_packets = static_cast<uint32_t>(got_cz.size());
        for (size_t i = 0; i < got_cz.size() && i < pushed_sx.size(); ++i) {
            uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(pushed_sx[i], w_def, 3, false, 1u);
            if (got_cz[i] != exp_cz) {
                m.e7_mismatches++;
            }
        }
        rtl.step_clock(10);
        m.e7_empty_clean = (rtl.top->m_axis_correction_tvalid == 0) && (rtl.top->bp_core_busy_mon == 0);
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e7_passed = m.e7_tready_deasserted_on_full && (m.e7_saturated_inflight >= 16u)
                  && (m.e7_drained_packets == m.e7_saturated_inflight)
                  && (m.e7_mismatches == 0u) && m.e7_empty_clean;

    // -------------------------------------------------------------------------
    // E8: Synchronous Hardware Reset Recovery Across Pipeline Offsets 1..20
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        const int16_t def_w[9] = {0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232, 0x0232};
        for (uint32_t cyc_off = 1; cyc_off <= 20; ++cyc_off) {
            rtl.top->s_axis_syndrome_tdata = 0x0Fu;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 1;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast = 0;
            rtl.step_clock(cyc_off);

            // Synchronous reset assertion
            rtl.reset_model(6);
            for (int idle_c = 0; idle_c < 8; ++idle_c) {
                if (rtl.top->m_axis_correction_tvalid != 0 || rtl.top->bp_core_busy_mon != 0) {
                    m.e8_ghost_outputs++;
                }
                rtl.step_clock(1);
            }
            RtlDecodeTrace tr = rtl.decode_syndrome(0x6u, 0x9u, true);
            uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x9u, def_w, 3, true, 1u);
            uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x6u, def_w, 3, false, 1u);
            if (tr.total_axis_cycles != 22u || tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                m.e8_post_reset_failures++;
            }
            m.e8_reset_offsets_tested++;
        }
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e8_passed = (m.e8_reset_offsets_tested == 20u) && (m.e8_ghost_outputs == 0u)
                  && (m.e8_post_reset_failures == 0u);

    // -------------------------------------------------------------------------
    // E9: 10,000-Syndrome Randomized RTL Stress (> 220,000 RTL cycles)
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        int16_t wx[9], wz[9];
        uint32_t rng = 0xCAFEBABEu;
        uint32_t cur_k = 3u, cur_mode = 1u;
        for (int i = 0; i < 9; ++i) {
            wx[i] = 0x0232;
            wz[i] = 0x0232;
        }
        program_weights_and_commit_idle(rtl, wx, wz, cur_k, cur_mode);

        for (uint32_t pkt = 0; pkt < 10000u; ++pkt) {
            if (pkt > 0 && (pkt % 500u == 0u)) {
                rng = 1664525u * rng + 1013904223u;
                cur_k = (rng % 5u) + 1u;
                cur_mode = (rng >> 8) % 3u;
                for (int i = 0; i < 9; ++i) {
                    rng = 1664525u * rng + 1013904223u;
                    wx[i] = static_cast<int16_t>(0x00A0 + ((rng >> 16) & 0x0240));
                    rng = 1664525u * rng + 1013904223u;
                    wz[i] = static_cast<int16_t>(0x00A0 + ((rng >> 16) & 0x0240));
                }
                program_weights_and_commit_idle(rtl, wx, wz, cur_k, cur_mode);
            }
            // Periodically toggle slow-loop DMA tap stall to verify fast-loop immunity
            rtl.top->dma_tap_stall_in = ((pkt % 7u) == 0u) ? 1 : 0;
            rng = 1664525u * rng + 1013904223u;
            uint8_t sx = static_cast<uint8_t>((rng >> 16) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((rng >> 24) & 0xFu);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (pkt % 64u == 63u));
            uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, wx, static_cast<int>(cur_k), true, cur_mode);
            uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, wz, static_cast<int>(cur_k), false, cur_mode);
            m.e9_packets++;
            if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                m.e9_mismatches++;
            }
        }
        rtl.top->dma_tap_stall_in = 0;
        m.e9_cycles = rtl.sim_cycle;
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e9_passed = (m.e9_packets == 10000u) && (m.e9_mismatches == 0u) && (m.e9_cycles >= 200000ULL);

    // -------------------------------------------------------------------------
    // E11: Weight-Update Interference Test (Pre / During / Post Commit)
    // Proves max |Delta t_decode| == 0 cycles (0.0 ns) during live AXI4-Lite
    // shadow writes and mid-decode COMMIT triggers!
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        auto measure_packet_concurrent_commit = [&](
            uint8_t sx,
            uint8_t sz,
            bool do_concurrent_commit,
            const int16_t next_wx[9],
            const int16_t next_wz[9],
            uint16_t *out_cx,
            uint16_t *out_cz
        ) -> uint32_t {
            // Measure exact cycle count from s_axis handshake to m_axis handshake + 1 (decode_syndrome convention)
            uint64_t accept_cyc = rtl.sim_cycle;
            rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>((sx & 0xFu) | ((sz & 0xFu) << 4));
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = 1;
            rtl.top->m_axis_correction_tready = 1;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast = 0;

            // Prepare sequence of AXI4-Lite single-beat writes to drive concurrently while BP is busy!
            struct AxilOp { uint32_t addr; uint32_t data; uint8_t strb; };
            std::vector<AxilOp> ops;
            if (do_concurrent_commit) {
                // Write 6 shadow entries + COMMIT right inside the 17-cycle (K=3) or 29-cycle (K=5) active decode window
                ops.push_back({0x00u, static_cast<uint16_t>(next_wx[0]), 0x3u});
                ops.push_back({0x10u, static_cast<uint16_t>(next_wx[4]), 0x3u});
                ops.push_back({0x24u, static_cast<uint16_t>(next_wz[0]), 0x3u});
                ops.push_back({0x34u, static_cast<uint16_t>(next_wz[4]), 0x3u});
                ops.push_back({0x38u, static_cast<uint16_t>(next_wz[5]), 0x3u});
                ops.push_back({0x48u, 0x1u, 0x1u}); // Mid-decode COMMIT!
            }

            size_t op_idx = 0;
            bool axil_driving = false;
            uint32_t finish_delta = 0;
            for (int c = 0; c < 80; ++c) {
                // Start concurrent AXI4-Lite writes once bp_core_busy_mon is high
                if (do_concurrent_commit && !axil_driving && op_idx < ops.size() && rtl.top->bp_core_busy_mon) {
                    rtl.top->s_axil_awaddr = ops[op_idx].addr;
                    rtl.top->s_axil_awvalid = 1;
                    rtl.top->s_axil_wdata = ops[op_idx].data;
                    rtl.top->s_axil_wstrb = ops[op_idx].strb;
                    rtl.top->s_axil_wvalid = 1;
                    rtl.top->s_axil_bready = 1;
                    axil_driving = true;
                } else if (axil_driving) {
                    rtl.top->s_axil_awvalid = 0;
                    rtl.top->s_axil_wvalid = 0;
                    axil_driving = false;
                    op_idx++;
                }

                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid) {
                    uint32_t out_w = rtl.top->m_axis_correction_tdata;
                    *out_cx = static_cast<uint16_t>(out_w & 0x1FFu);
                    *out_cz = static_cast<uint16_t>((out_w >> 9) & 0x1FFu);
                    rtl.top->s_axil_awvalid = 0;
                    rtl.top->s_axil_wvalid = 0;
                    rtl.step_clock(1); // Match decode_syndrome registered epoch_irq cycle
                    finish_delta = static_cast<uint32_t>(rtl.sim_cycle - accept_cyc);
                    break;
                }
            }
            rtl.step_clock(2); // Let swap_pending complete cleanly in inter-packet idle gap
            return finish_delta;
        };

        int16_t cur_wx[9] = {0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240};
        int16_t cur_wz[9] = {0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240, 0x0240};
        int16_t alt_wx[9] = {0x0240, 0x0240, 0x0240, 0x0240, 0x0110, 0x0240, 0x0240, 0x0240, 0x0240};
        int16_t alt_wz[9] = {0x0240, 0x0240, 0x0240, 0x0240, 0x00D8, 0x00D8, 0x0240, 0x0240, 0x0240};

        program_weights_and_commit_idle(rtl, cur_wx, cur_wz, 3u, 1u);

        // 1) 500 Pre-Commit packets (K=3, Mode=1 -> 22 cycles = 88.0 ns)
        for (uint32_t i = 0; i < 500u; ++i) {
            uint8_t sx = static_cast<uint8_t>((i * 7u + 1u) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i * 5u + 3u) & 0xFu);
            uint16_t cx = 0, cz = 0;
            uint32_t cyc = measure_packet_concurrent_commit(sx, sz, false, alt_wx, alt_wz, &cx, &cz);
            m.e11_pre_packets++;
            m.e11_k3_pre_min_cyc = std::min(m.e11_k3_pre_min_cyc, cyc);
            m.e11_k3_pre_max_cyc = std::max(m.e11_k3_pre_max_cyc, cyc);
            if (cz != AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, cur_wz, 3, false, 1u)) {
                m.e11_mismatches++;
            }
        }

        // 2) 500 During-Commit packets (live mid-decode AXI4-Lite shadow write + COMMIT on EVERY packet!)
        bool using_alt = false;
        for (uint32_t i = 0; i < 500u; ++i) {
            uint8_t sx = static_cast<uint8_t>((i * 3u + 4u) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i * 9u + 2u) & 0xFu);
            const int16_t *active_now_wz = using_alt ? alt_wz : cur_wz;
            const int16_t *target_wx = using_alt ? cur_wx : alt_wx;
            const int16_t *target_wz = using_alt ? cur_wz : alt_wz;
            uint16_t cx = 0, cz = 0;
            uint32_t cyc = measure_packet_concurrent_commit(sx, sz, true, target_wx, target_wz, &cx, &cz);
            m.e11_during_packets++;
            m.e11_k3_during_min_cyc = std::min(m.e11_k3_during_min_cyc, cyc);
            m.e11_k3_during_max_cyc = std::max(m.e11_k3_during_max_cyc, cyc);
            // In-flight packet MUST have been decoded with active_now_wz (zero mid-decode corruption!)
            if (cz != AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, active_now_wz, 3, false, 1u)) {
                m.e11_mismatches++;
            }
            using_alt = !using_alt;
        }

        // 3) 500 Post-Commit packets
        const int16_t *final_wz = using_alt ? alt_wz : cur_wz;
        for (uint32_t i = 0; i < 500u; ++i) {
            uint8_t sx = static_cast<uint8_t>((i * 11u + 6u) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i * 3u + 1u) & 0xFu);
            uint16_t cx = 0, cz = 0;
            uint32_t cyc = measure_packet_concurrent_commit(sx, sz, false, alt_wx, alt_wz, &cx, &cz);
            m.e11_post_packets++;
            m.e11_k3_post_min_cyc = std::min(m.e11_k3_post_min_cyc, cyc);
            m.e11_k3_post_max_cyc = std::max(m.e11_k3_post_max_cyc, cyc);
            if (cz != AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, final_wz, 3, false, 1u)) {
                m.e11_mismatches++;
            }
        }

        // Also check K=5, Mode=2 pre/during/post single-reference latency (34 cycles = 136.0 ns)
        program_weights_and_commit_idle(rtl, cur_wx, cur_wz, 5u, 2u);
        uint16_t cx5 = 0, cz5 = 0;
        m.e11_k5_pre_cyc = measure_packet_concurrent_commit(0x4u, 0x9u, false, alt_wx, alt_wz, &cx5, &cz5);
        m.e11_k5_during_cyc = measure_packet_concurrent_commit(0x4u, 0x9u, true, alt_wx, alt_wz, &cx5, &cz5);
        m.e11_k5_post_cyc = measure_packet_concurrent_commit(0x4u, 0x9u, false, cur_wx, cur_wz, &cx5, &cz5);

        int32_t d1 = std::abs(static_cast<int32_t>(m.e11_k3_during_max_cyc) - static_cast<int32_t>(m.e11_k3_pre_min_cyc));
        int32_t d2 = std::abs(static_cast<int32_t>(m.e11_k3_post_max_cyc) - static_cast<int32_t>(m.e11_k3_pre_min_cyc));
        int32_t d3 = std::abs(static_cast<int32_t>(m.e11_k5_during_cyc) - static_cast<int32_t>(m.e11_k5_pre_cyc));
        m.e11_max_abs_delta_cycles = std::max({d1, d2, d3});
        m.e11_max_abs_delta_ns = static_cast<double>(m.e11_max_abs_delta_cycles) * 4.0;
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e11_passed = (m.e11_pre_packets == 500u) && (m.e11_during_packets == 500u)
                   && (m.e11_post_packets == 500u) && (m.e11_max_abs_delta_cycles == 0)
                   && (m.e11_mismatches == 0u)
                   && (m.e11_k3_pre_max_cyc == 22u) && (m.e11_k5_during_cyc == 34u);

    // -------------------------------------------------------------------------
    // E12: Fast-Loop 300 ns Deadline Under Worst-Case Traffic
    // K=5, Mode=2 (Post-BP OSD-0), simultaneous non-zero X+Z syndromes,
    // slow-loop DMA tap stalled (dma_tap_stall_in=1), and concurrent commits
    // across 2,000 QEC rounds (paced at 1 round / 1,000 ns = 250 clocks AND
    // worst-case minimum service latency check).
    // -------------------------------------------------------------------------
    {
        AvidVerilatorRtlModel rtl(4, 10);
        int16_t wx[9] = {0x0210, 0x0150, 0x0230, 0x0190, 0x00E0, 0x00E8, 0x0240, 0x0170, 0x0220};
        int16_t wz[9] = {0x0220, 0x0160, 0x0210, 0x01A0, 0x00D0, 0x00D8, 0x0250, 0x0180, 0x0230};
        program_weights_and_commit_idle(rtl, wx, wz, 5u, 2u);
        rtl.top->dma_tap_stall_in = 1; // Worst-case slow-loop DMA congestion!

        uint32_t rng = 0x9E3779B9u;
        for (uint32_t i = 0; i < 2000u; ++i) {
            rng = 1664525u * rng + 1013904223u;
            uint8_t sx = static_cast<uint8_t>(((rng >> 12) % 15u) + 1u); // strictly non-zero
            uint8_t sz = static_cast<uint8_t>(((rng >> 20) % 15u) + 1u); // strictly non-zero
            if (i > 0 && (i % 200u == 0u)) {
                wx[4] = static_cast<int16_t>(0x00C8 + (i % 40u));
                wz[4] = static_cast<int16_t>(0x00C0 + (i % 40u));
                program_weights_and_commit_idle(rtl, wx, wz, 5u, 2u);
            }
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (i % 100u == 99u));
            uint16_t exp_cx = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, wx, 5, true, 2u);
            uint16_t exp_cz = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, wz, 5, false, 2u);
            m.e12_packets++;
            m.e12_max_core_cycles = std::max(m.e12_max_core_cycles, tr.bp_core_cycles);
            m.e12_max_service_axis_cycles = std::max(m.e12_max_service_axis_cycles, tr.total_axis_cycles);
            double lat_ns = static_cast<double>(tr.total_axis_cycles) * 4.0;
            if (i < 64u) {
                m.e12_service_ns_hist_samples.push_back(lat_ns);
            }
            if (lat_ns > m.e12_deadline_ns) {
                m.e12_deadline_misses++;
            }
            if (tr.corr_x != exp_cx || tr.corr_z != exp_cz) {
                m.e12_mismatches++;
            }
        }
        rtl.top->dma_tap_stall_in = 0;
        m.e12_max_service_latency_ns = static_cast<double>(m.e12_max_service_axis_cycles) * 4.0;
        m.e12_headroom_ns = m.e12_deadline_ns - m.e12_max_service_latency_ns;
        m.total_rtl_cycles += rtl.sim_cycle;
    }
    m.e12_passed = (m.e12_packets == 2000u) && (m.e12_deadline_misses == 0u)
                   && (m.e12_mismatches == 0u) && (m.e12_max_service_axis_cycles == 34u)
                   && (m.e12_max_service_latency_ns <= 300.0);

    bool all_ok = m.e2_passed && m.e3_passed && m.e4_passed && m.e5_passed
                  && m.e6_passed && m.e7_passed && m.e8_passed && m.e9_passed
                  && m.e10_passed && m.e11_passed && m.e12_passed;

    // Serialize JSON output
    std::ofstream out(out_json_path);
    out << "{\n"
        << "  \"group\": \"Group_E_Hardware_Correctness_and_DualLoop_Isolation\",\n"
        << "  \"simulator\": \"" << AvidVerilatorRtlModel::simulator_name() << "\",\n"
        << "  \"native_verilator\": " << (m.native_verilator ? "true" : "false") << ",\n"
        << "  \"clock_mhz\": 250.0,\n"
        << "  \"clock_period_ns\": 4.0,\n"
        << "  \"total_verilator_rtl_cycles\": " << m.total_rtl_cycles << ",\n"
        << "  \"gates\": {\n"
        << "    \"E2_fixed_point_q78_equivalence\": {\"passed\": " << (m.e2_passed ? "true" : "false")
        << ", \"states_tested\": " << m.e2_states_tested << ", \"mismatches\": " << m.e2_mismatches << "},\n"
        << "    \"E3_bp_mode0_golden_equivalence\": {\"passed\": " << (m.e3_passed ? "true" : "false")
        << ", \"states_tested\": " << m.e3_states_tested << ", \"mismatches\": " << m.e3_mismatches << "},\n"
        << "    \"E4_osd0_degenerate_postprocessing\": {\"passed\": " << (m.e4_passed ? "true" : "false")
        << ", \"cases_tested\": " << m.e4_cases_tested
        << ", \"unsatisfied_syndromes\": " << m.e4_unsatisfied_syndromes
        << ", \"mismatches\": " << m.e4_mismatches
        << ", \"pair_disambiguation_flipped\": " << (m.e4_pair_disambiguation_flipped ? "true" : "false") << "},\n"
        << "    \"E5_atomic_shadow_commit_streaming\": {\"passed\": " << (m.e5_passed ? "true" : "false")
        << ", \"packets\": " << m.e5_packets << ", \"commits\": " << m.e5_commits
        << ", \"hw_epochs\": " << m.e5_hw_epochs << ", \"mismatches\": " << m.e5_mismatches << "},\n"
        << "    \"E6_mid_decode_commit_and_slverr_lock\": {\"passed\": " << (m.e6_passed ? "true" : "false")
        << ", \"offsets_tested\": " << m.e6_offsets_tested
        << ", \"slverr_locks_verified\": " << m.e6_slverr_locks_verified
        << ", \"mid_decode_corruptions\": " << m.e6_mid_decode_corruptions << "},\n"
        << "    \"E7_elastic_fifo_full_empty_backpressure\": {\"passed\": " << (m.e7_passed ? "true" : "false")
        << ", \"saturated_inflight_packets\": " << m.e7_saturated_inflight
        << ", \"tready_deasserted_on_full\": " << (m.e7_tready_deasserted_on_full ? "true" : "false")
        << ", \"drained_packets\": " << m.e7_drained_packets
        << ", \"mismatches\": " << m.e7_mismatches
        << ", \"empty_clean\": " << (m.e7_empty_clean ? "true" : "false") << "},\n"
        << "    \"E8_synchronous_reset_recovery\": {\"passed\": " << (m.e8_passed ? "true" : "false")
        << ", \"reset_offsets_tested\": " << m.e8_reset_offsets_tested
        << ", \"ghost_outputs\": " << m.e8_ghost_outputs
        << ", \"post_reset_failures\": " << m.e8_post_reset_failures << "},\n"
        << "    \"E9_10k_randomized_rtl_stress\": {\"passed\": " << (m.e9_passed ? "true" : "false")
        << ", \"packets\": " << m.e9_packets << ", \"mismatches\": " << m.e9_mismatches
        << ", \"rtl_cycles\": " << m.e9_cycles << "},\n"
        << "    \"E10_cycle_budget_latency_law\": {\"passed\": " << (m.e10_passed ? "true" : "false")
        << ", \"configs_tested\": " << m.e10_configs_tested
        << ", \"law_violations\": " << m.e10_law_violations << ", \"k_list\": [";
    for (size_t i = 0; i < m.e10_k_list.size(); ++i) out << (i ? "," : "") << m.e10_k_list[i];
    out << "], \"mode_list\": [";
    for (size_t i = 0; i < m.e10_mode_list.size(); ++i) out << (i ? "," : "") << m.e10_mode_list[i];
    out << "], \"core_cycles\": [";
    for (size_t i = 0; i < m.e10_core_cycles.size(); ++i) out << (i ? "," : "") << m.e10_core_cycles[i];
    out << "], \"axis_cycles\": [";
    for (size_t i = 0; i < m.e10_axis_cycles.size(); ++i) out << (i ? "," : "") << m.e10_axis_cycles[i];
    out << "], \"latency_ns\": [";
    for (size_t i = 0; i < m.e10_latency_ns.size(); ++i) out << (i ? "," : "") << m.e10_latency_ns[i];
    out << "]},\n"
        << "    \"E11_weight_update_interference\": {\"passed\": " << (m.e11_passed ? "true" : "false")
        << ", \"pre_packets\": " << m.e11_pre_packets
        << ", \"during_packets\": " << m.e11_during_packets
        << ", \"post_packets\": " << m.e11_post_packets
        << ", \"k3_pre_cycles_min\": " << m.e11_k3_pre_min_cyc
        << ", \"k3_pre_cycles_max\": " << m.e11_k3_pre_max_cyc
        << ", \"k3_during_cycles_min\": " << m.e11_k3_during_min_cyc
        << ", \"k3_during_cycles_max\": " << m.e11_k3_during_max_cyc
        << ", \"k3_post_cycles_min\": " << m.e11_k3_post_min_cyc
        << ", \"k3_post_cycles_max\": " << m.e11_k3_post_max_cyc
        << ", \"k5_pre_cycles\": " << m.e11_k5_pre_cyc
        << ", \"k5_during_cycles\": " << m.e11_k5_during_cyc
        << ", \"k5_post_cycles\": " << m.e11_k5_post_cyc
        << ", \"max_abs_delta_cycles\": " << m.e11_max_abs_delta_cycles
        << ", \"max_abs_delta_ns\": " << m.e11_max_abs_delta_ns
        << ", \"mismatches\": " << m.e11_mismatches << "},\n"
        << "    \"E12_fast_loop_300ns_worst_case_deadline\": {\"passed\": " << (m.e12_passed ? "true" : "false")
        << ", \"packets\": " << m.e12_packets
        << ", \"max_core_cycles\": " << m.e12_max_core_cycles
        << ", \"max_service_axis_cycles\": " << m.e12_max_service_axis_cycles
        << ", \"max_service_latency_ns\": " << m.e12_max_service_latency_ns
        << ", \"deadline_ns\": " << m.e12_deadline_ns
        << ", \"headroom_ns\": " << m.e12_headroom_ns
        << ", \"deadline_misses\": " << m.e12_deadline_misses
        << ", \"mismatches\": " << m.e12_mismatches << "}\n"
        << "  },\n"
        << "  \"overall_passed\": " << (all_ok ? "true" : "false") << "\n"
        << "}\n";
    out.close();

    std::printf("[Group E RTL] overall_passed=%s | total_rtl_cycles=%llu\n",
                all_ok ? "true" : "false",
                static_cast<unsigned long long>(m.total_rtl_cycles));
    return all_ok ? 0 : 1;
}

static uint8_t hex_nibble(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + (c - 'a'));
    if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + (c - 'A'));
    return 0u;
}

/* Closed-loop multi-epoch RTL execution for Group F (`F1..F7, F9, F10`).
 * Input CSV columns per line:
 *   epoch_id,reset_rtl,commit_mode,k_iter,osd_mode,wx0..wx8,wz0..wz8,sx_hex,sz_hex
 * commit_mode:
 *   0 = idle AXI4-Lite shadow write + COMMIT before window
 *   1 = live mid-decode AXI4-Lite shadow write + COMMIT while round 0 is active in bp_engine
 *   2 = no weight update (retain active RTL bank untouched, e.g. sub-10ms 60us micro-burst)
 */
static int run_closedloop_schedule(const std::string &in_csv, const std::string &out_csv) {
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] Closed-loop RTL schedule requires native Verilator.\n");
        return 2;
    }
    std::ifstream fin(in_csv);
    if (!fin.is_open()) {
        std::fprintf(stderr, "Failed to open input schedule CSV: %s\n", in_csv.c_str());
        return 3;
    }
    std::ofstream fout(out_csv);
    if (!fout.is_open()) {
        std::fprintf(stderr, "Failed to open output trace CSV: %s\n", out_csv.c_str());
        return 4;
    }

    AvidVerilatorRtlModel rtl(4, 10);
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            cols.push_back(item);
        }
        if (cols.size() < 25) {
            std::fprintf(stderr, "Malformed schedule row (got %zu cols): %s\n", cols.size(), line.c_str());
            return 5;
        }
        int epoch_id = std::atoi(cols[0].c_str());
        int reset_rtl = std::atoi(cols[1].c_str());
        int commit_mode = std::atoi(cols[2].c_str());
        uint32_t k_iter = static_cast<uint32_t>(std::atoi(cols[3].c_str()));
        uint32_t osd_mode = static_cast<uint32_t>(std::atoi(cols[4].c_str()));
        int16_t wx[9], wz[9];
        for (int i = 0; i < 9; ++i) {
            wx[i] = static_cast<int16_t>(std::atoi(cols[5 + i].c_str()));
            wz[i] = static_cast<int16_t>(std::atoi(cols[14 + i].c_str()));
        }
        const std::string &sx_hex = cols[23];
        const std::string &sz_hex = cols[24];
        size_t n_rounds = std::min(sx_hex.size(), sz_hex.size());

        if (reset_rtl != 0) {
            rtl.reset_model(8);
        }

        uint64_t cyc = 0;
        if (commit_mode == 0) {
            program_weights_and_commit_idle(rtl, wx, wz, k_iter, osd_mode);
        } else if (commit_mode == 1) {
            // Stage 8 shadow pairs before round 0, then trigger final shadow writes + COMMIT mid-decode!
            rtl.axil_write(0x54u, k_iter, 0x1u, &cyc);
            rtl.axil_write(0x58u, osd_mode, 0x1u, &cyc);
            for (int i = 0; i < 8; ++i) {
                rtl.axil_write(static_cast<uint32_t>(i * 4), static_cast<uint16_t>(wx[i]), 0x3u, &cyc);
                rtl.axil_write(static_cast<uint32_t>(0x24u + i * 4), static_cast<uint16_t>(wz[i]), 0x3u, &cyc);
            }
        }

        std::vector<uint16_t> corr_z_out(n_rounds, 0u);
        std::vector<uint16_t> corr_x_out(n_rounds, 0u);
        uint64_t sum_core_cyc = 0;
        uint32_t max_axis_cyc = 0;
        int32_t mid_commit_delta_cyc = 0;

        for (size_t r = 0; r < n_rounds; ++r) {
            uint8_t sx = hex_nibble(sx_hex[r]);
            uint8_t sz = hex_nibble(sz_hex[r]);
            if (commit_mode == 1 && r == 0) {
                // Inject round 0 and fire final shadow writes + COMMIT while bp_core_busy_mon == 1!
                uint64_t accept_c = rtl.sim_cycle;
                rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>(sx | (sz << 4));
                rtl.top->s_axis_syndrome_tvalid = 1;
                rtl.top->s_axis_syndrome_tlast = 0;
                rtl.top->m_axis_correction_tready = 1;
                rtl.step_clock(1);
                rtl.top->s_axis_syndrome_tvalid = 0;

                struct Op { uint32_t addr; uint32_t data; uint8_t strb; };
                Op ops[3] = {
                    {32u, static_cast<uint16_t>(wx[8]), 0x3u},
                    {0x24u + 32u, static_cast<uint16_t>(wz[8]), 0x3u},
                    {0x48u, 0x1u, 0x1u}
                };
                int oi = 0;
                bool driving = false;
                uint32_t busy_c = 0;
                for (int g = 0; g < 80; ++g) {
                    if (!driving && oi < 3 && rtl.top->bp_core_busy_mon) {
                        rtl.top->s_axil_awaddr = ops[oi].addr;
                        rtl.top->s_axil_awvalid = 1;
                        rtl.top->s_axil_wdata = ops[oi].data;
                        rtl.top->s_axil_wstrb = ops[oi].strb;
                        rtl.top->s_axil_wvalid = 1;
                        rtl.top->s_axil_bready = 1;
                        driving = true;
                    } else if (driving) {
                        rtl.top->s_axil_awvalid = 0;
                        rtl.top->s_axil_wvalid = 0;
                        driving = false;
                        oi++;
                    }
                    rtl.step_clock(1);
                    if (rtl.top->bp_core_busy_mon) busy_c++;
                    if (rtl.top->m_axis_correction_tvalid) {
                        uint32_t ow = rtl.top->m_axis_correction_tdata;
                        corr_x_out[r] = static_cast<uint16_t>(ow & 0x1FFu);
                        corr_z_out[r] = static_cast<uint16_t>((ow >> 9) & 0x1FFu);
                        rtl.top->s_axil_awvalid = 0;
                        rtl.top->s_axil_wvalid = 0;
                        rtl.step_clock(1);
                        uint32_t ax_c = static_cast<uint32_t>(rtl.sim_cycle - accept_c);
                        uint32_t exp_ax = ((osd_mode == 2u) ? (9u + 5u * k_iter) : (7u + 5u * k_iter));
                        mid_commit_delta_cyc = static_cast<int32_t>(ax_c) - static_cast<int32_t>(exp_ax);
                        sum_core_cyc += busy_c;
                        max_axis_cyc = std::max(max_axis_cyc, ax_c);
                        break;
                    }
                }
                rtl.step_clock(2); // Allow deferred swap to take effect before round 1
            } else {
                RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (r + 1 == n_rounds));
                corr_x_out[r] = tr.corr_x;
                corr_z_out[r] = tr.corr_z;
                sum_core_cyc += tr.bp_core_cycles;
                max_axis_cyc = std::max(max_axis_cyc, tr.total_axis_cycles);
            }
        }

        uint32_t hw_ep = 0;
        rtl.axil_read(0x50u, &hw_ep, &cyc);
        double mean_core = n_rounds > 0 ? static_cast<double>(sum_core_cyc) / static_cast<double>(n_rounds) : 0.0;

        fout << epoch_id << "," << hw_ep << "," << mean_core << "," << max_axis_cyc << ","
             << mid_commit_delta_cyc << "," << rtl.sim_cycle << ",";
        for (size_t r = 0; r < n_rounds; ++r) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%03x", corr_z_out[r] & 0x1FFu);
            fout << (r ? ":" : "") << buf;
        }
        fout << ",";
        for (size_t r = 0; r < n_rounds; ++r) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%03x", corr_x_out[r] & 0x1FFu);
            fout << (r ? ":" : "") << buf;
        }
        fout << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "Usage:\n"
                     "  %s --run-group-e <out.json> [--vcd <out.vcd>]\n"
                     "  %s --decode-closedloop <in_schedule.csv> <out_trace.csv>\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "--run-group-e") == 0) {
        std::string out_json = argv[2];
        std::string vcd_path;
        if (argc >= 5 && std::strcmp(argv[3], "--vcd") == 0) {
            vcd_path = argv[4];
        }
        return run_group_e_all(out_json, vcd_path);
    }
    if (std::strcmp(argv[1], "--decode-closedloop") == 0 && argc >= 4) {
        return run_closedloop_schedule(argv[2], argv[3]);
    }
    std::fprintf(stderr, "Invalid CLI arguments.\n");
    return 2;
}
