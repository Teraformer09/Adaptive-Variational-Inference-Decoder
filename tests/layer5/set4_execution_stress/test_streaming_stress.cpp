/* ===========================================================================
 * test_streaming_stress.cpp — Layer 5 Set 4.2 High-Volume Streaming & Race Stress
 * Canonical IDs: L5-FLT-01 .. L5-FLT-10 (Gates G2, G3, G4, G5, G6, G7, G8)
 * Executes 10,000-syndrome continuous streaming (>220,000 RTL cycles), multi-
 * epoch live commits, K=1/K=5 extremes, and exhaustive cycle-by-cycle COMMIT
 * and aresetn race sweeps on native Verilator Vaa_qec_top (--assert).
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set4_streaming_stress", "Set4", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] test_streaming_stress requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-FLT-01: 10,000-syndrome continuous streaming stress (>220,000 RTL cycles, 0 mismatches) */
    suite.run_case("L5-FLT-01", "G5,G6,G8", "10,000-syndrome continuous streaming stress (>220,000 RTL cycles) completes with 0 mismatches & 0 deadlocks", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 10);
        uint32_t s = 0x13579BDFu;
        int mismatches = 0;
        int irq_pulses = 0;
        for (int i = 0; i < 10000; i++) {
            s = s * 1664525u + 1013904223u;
            uint8_t sx = static_cast<uint8_t>((s >> 16) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((s >> 20) & 0xFu);
            bool tlast = ((i + 1) % 250 == 0);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, tlast);
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, rtl.active_x, 3, true, 1u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, rtl.active_z, 3, false, 1u);
            if (tr.corr_x != ex || tr.corr_z != ez || tr.bp_core_cycles != 17u || tr.total_axis_cycles != 22u) {
                mismatches++;
            }
            if (tr.epoch_irq_pulsed) irq_pulses++;
        }
        cyc = rtl.sim_cycle;
        det = "syndromes=10000 rtl_cycles=" + std::to_string(cyc) +
              " mismatches=" + std::to_string(mismatches) + " epoch_irqs=" + std::to_string(irq_pulses);
        return (mismatches == 0) && (cyc >= 220000ULL) && (irq_pulses == 40);
    });

    /* L5-FLT-02: 2,000-syndrome stream interleaved with 20 live adaptive weight-bank commits */
    suite.run_case("L5-FLT-02", "G4,G5,G6", "2,000-syndrome stream with 20 live adaptive commits (including mid-decode deferred swaps) has 0 mismatches", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int mismatches = 0;
        uint64_t c = 0;
        for (int ep = 0; ep < 20; ep++) {
            int hot_q = ep % 9;
            int16_t wx[9], wz[9];
            for (int q = 0; q < 9; q++) {
                wx[q] = (q == hot_q) ? static_cast<int16_t>(0x0090 + ep * 4) : static_cast<int16_t>(0x0280);
                wz[q] = (q == (8 - hot_q)) ? static_cast<int16_t>(0x0088 + ep * 4) : static_cast<int16_t>(0x0290);
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), static_cast<uint16_t>(wx[q]), 0x3u, &c);
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), static_cast<uint16_t>(wz[q]), 0x3u, &c);
            }
            /* Issue COMMIT mid-decode on round 0 of each epoch so swap defers to round 1 */
            int16_t prev_wx[9], prev_wz[9];
            for (int q = 0; q < 9; q++) {
                prev_wx[q] = rtl.active_x[q];
                prev_wz[q] = rtl.active_z[q];
            }
            uint64_t acc = 0;
            rtl.start_async_syndrome(0x2u, 0x8u, false, &acc);
            rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
            RtlDecodeTrace tr0{};
            rtl.finish_async_syndrome(&tr0);
            uint16_t ex0 = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x8u, prev_wx, 3, true, 1u);
            uint16_t ez0 = AvidVerilatorRtlModel::eval_rtl_bp_reference(0x2u, prev_wz, 3, false, 1u);
            if (tr0.corr_x != ex0 || tr0.corr_z != ez0) mismatches++;

            for (int r = 1; r < 100; r++) {
                uint8_t sx = static_cast<uint8_t>((ep * 7 + r) & 0xFu);
                uint8_t sz = static_cast<uint8_t>((ep * 11 + r * 3) & 0xFu);
                RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (r == 99));
                uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, wx, 3, true, 1u);
                uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, wz, 3, false, 1u);
                if (tr.corr_x != ex || tr.corr_z != ez) mismatches++;
            }
        }
        uint32_t ep_hw = 0;
        rtl.axil_read(avid_l5::L5_OFF_EPOCH_COUNT, &ep_hw, &c);
        cyc = rtl.sim_cycle;
        det = "rounds=2000 epochs=" + std::to_string(ep_hw) + " mismatches=" + std::to_string(mismatches);
        return (mismatches == 0) && (ep_hw == 20u);
    });

    /* L5-FLT-03: 1,000-syndrome worst-case latency stream (K=5, OSD_MODE=2: 29 core / 34 AXI cycles) */
    suite.run_case("L5-FLT-03", "G2,G5,G6", "1,000-syndrome worst-case K=5 OSD_MODE=2 stream holds exact 29-cycle core / 34-cycle AXI latency", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, 5u, 0x1u, &c);
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 2u, 0x1u, &c);
        int bad = 0;
        for (int i = 0; i < 1000; i++) {
            uint8_t sx = static_cast<uint8_t>((i * 5 + 1) & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i * 9 + 3) & 0xFu);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (i == 999));
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, rtl.active_x, 5, true, 2u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, rtl.active_z, 5, false, 2u);
            if (tr.bp_core_cycles != 29u || tr.total_axis_cycles != 34u || tr.corr_x != ex || tr.corr_z != ez) bad++;
        }
        cyc = rtl.sim_cycle;
        det = "rounds=1000 core=29 axis=34 (136.0ns<=300ns) bad=" + std::to_string(bad);
        return (bad == 0);
    });

    /* L5-FLT-04: 1,000-syndrome minimum-latency stream (K=1, OSD_MODE=0: 7 core / 12 AXI cycles) */
    suite.run_case("L5-FLT-04", "G2,G5,G6", "1,000-syndrome minimum-iteration K=1 OSD_MODE=0 stream holds exact 7-cycle core / 12-cycle AXI latency", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, 1u, 0x1u, &c);
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 0u, 0x1u, &c);
        int bad = 0;
        for (int i = 0; i < 1000; i++) {
            uint8_t sx = static_cast<uint8_t>(i & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i >> 4) & 0xFu);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (i == 999));
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, rtl.active_x, 1, true, 0u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, rtl.active_z, 1, false, 0u);
            if (tr.bp_core_cycles != 7u || tr.total_axis_cycles != 12u || tr.corr_x != ex || tr.corr_z != ez) bad++;
        }
        cyc = rtl.sim_cycle;
        det = "rounds=1000 core=7 axis=12 (48.0ns) bad=" + std::to_string(bad);
        return (bad == 0);
    });

    /* L5-FLT-05: 1,000-syndrome bursty stream with periodic egress backpressure & DMA tap stall */
    suite.run_case("L5-FLT-05", "G3,G5,G6", "1,000-syndrome pipelined stream under periodic egress stalls & DMA tap drops preserves 100% accuracy", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const size_t N = 1000;
        std::vector<uint8_t> sxs(N), szs(N);
        for (size_t i = 0; i < N; i++) {
            sxs[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFu);
            szs[i] = static_cast<uint8_t>((i * 13 + 5) & 0xFu);
        }
        size_t sent = 0;
        uint32_t expected_drops = 0;
        std::vector<std::pair<uint16_t, uint16_t>> outs;
        outs.reserve(N);

        for (int step = 0; step < 35000 && outs.size() < N; step++) {
            bool dma_stall = ((step % 5) < 2);
            rtl.top->dma_tap_stall_in         = dma_stall ? 1 : 0;
            rtl.top->m_axis_correction_tready = ((step % 7) != 0) ? 1 : 0;

            if (sent < N && rtl.top->s_axis_syndrome_tready) {
                rtl.top->s_axis_syndrome_tdata  = static_cast<uint16_t>(sxs[sent] | (szs[sent] << 4));
                rtl.top->s_axis_syndrome_tvalid = 1;
                rtl.top->s_axis_syndrome_tlast  = ((sent + 1) % 100 == 0) ? 1 : 0;
            } else {
                rtl.top->s_axis_syndrome_tvalid = 0;
                rtl.top->s_axis_syndrome_tlast  = 0;
            }

            bool in_hs  = (rtl.top->s_axis_syndrome_tvalid && rtl.top->s_axis_syndrome_tready);
            bool out_hs = (rtl.top->m_axis_correction_tvalid && rtl.top->m_axis_correction_tready);
            uint32_t out_w = rtl.top->m_axis_correction_tdata;

            rtl.step_clock(1);

            if (in_hs) {
                sent++;
                if (dma_stall) expected_drops++;
            }
            if (out_hs) {
                outs.push_back({static_cast<uint16_t>(out_w & 0x1FFu),
                                static_cast<uint16_t>((out_w >> 9) & 0x1FFu)});
            }
        }
        rtl.top->s_axis_syndrome_tvalid = 0;
        bool ok = (outs.size() == N) && (rtl.top->dma_tap_drop_count_mon == expected_drops);
        for (size_t i = 0; i < N && ok; i++) {
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(szs[i], rtl.active_x, 3, true, 1u);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sxs[i], rtl.active_z, 3, false, 1u);
            if (outs[i].first != ex || outs[i].second != ez) ok = false;
        }
        cyc = rtl.sim_cycle;
        det = "decoded=" + std::to_string(outs.size()) + "/1000 drops=" +
              std::to_string(rtl.top->dma_tap_drop_count_mon) + "/" + std::to_string(expected_drops);
        return ok;
    });

    /* L5-FLT-06: 512-syndrome signed Q7.8 negative + positive mixed weight stress */
    suite.run_case("L5-FLT-06", "G5,G6", "512-syndrome stream with mixed negative/positive Q7.8 priors matches golden reference bit-for-bit", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        int16_t wx[9] = {-512, 320, -64, 480, 128, -256, 600, 96, -128};
        int16_t wz[9] = {400, -300, 256, -128, 512, 64, -200, 350, -32};
        for (int q = 0; q < 9; q++) {
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), static_cast<uint16_t>(wx[q]), 0x3u, &c);
            rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), static_cast<uint16_t>(wz[q]), 0x3u, &c);
        }
        rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
        int bad = 0;
        for (int i = 0; i < 512; i++) {
            uint8_t sx = static_cast<uint8_t>(i & 0xFu);
            uint8_t sz = static_cast<uint8_t>((i >> 4) & 0xFu);
            uint32_t mode = (i < 256) ? 0u : 1u;
            if (i == 0 || i == 256) rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, mode, 0x1u, &c);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, false);
            uint16_t ex = AvidVerilatorRtlModel::eval_rtl_bp_reference(sz, wx, 3, true, mode);
            uint16_t ez = AvidVerilatorRtlModel::eval_rtl_bp_reference(sx, wz, 3, false, mode);
            if (tr.corr_x != ex || tr.corr_z != ez) bad++;
        }
        cyc = rtl.sim_cycle;
        det = "signed_mixed_vectors=512 mismatches=" + std::to_string(bad);
        return (bad == 0);
    });

    /* L5-FLT-07: 64-epoch variable-window stream ({1, 4, 8, 16} rounds/epoch) */
    suite.run_case("L5-FLT-07", "G2,G3,G5", "64 variable-length epochs ({1, 4, 8, 16} rounds) assert epoch_irq strictly 64 times", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int lens[4] = {1, 4, 8, 16};
        int irqs = 0;
        int total_pkts = 0;
        for (int ep = 0; ep < 64; ep++) {
            int L = lens[ep & 3];
            for (int r = 0; r < L; r++) {
                bool is_last = (r == L - 1);
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>((ep + r) & 0xFu),
                                                        static_cast<uint8_t>((ep * 3 + r) & 0xFu),
                                                        is_last);
                total_pkts++;
                if (tr.epoch_irq_pulsed) irqs++;
                if (tr.epoch_irq_pulsed != is_last) return false;
            }
        }
        cyc = rtl.sim_cycle;
        det = "epochs=64 total_pkts=" + std::to_string(total_pkts) +
              " irqs=" + std::to_string(irqs) + " rx_epoch_mon=" + std::to_string(rtl.top->rx_epoch_count_mon);
        return (irqs == 64) && (rtl.top->rx_epoch_count_mon == 64u);
    });

    /* L5-FLT-08: Exhaustive cycle-by-cycle COMMIT race sweep (dt = 0..24 cycles from syndrome accept) */
    suite.run_case("L5-FLT-08", "G4,G5,G7", "Cycle-by-cycle COMMIT race sweep across dt=0..24 cycles proves 0 SVA violations & 0 torn decodes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint64_t total_c = 0;
        int clean_phases = 0;
        for (uint32_t dt = 0; dt <= 24; dt++) {
            AvidVerilatorRtlModel rtl(4, 6);
            uint64_t c = 0;
            /* Old bank prefers q2 on sx=0x2; new shadow bank prefers q5 on sx=0x2 */
            int16_t old_z[9], new_z[9];
            for (int q = 0; q < 9; q++) {
                old_z[q] = (q == 2) ? 0x0080 : 0x0300;
                new_z[q] = (q == 5) ? 0x0080 : 0x0300;
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), static_cast<uint16_t>(old_z[q]), 0x3u, &c);
            }
            rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
            for (int q = 0; q < 9; q++) {
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), static_cast<uint16_t>(new_z[q]), 0x3u, &c);
            }

            /* Accept syndrome sx=0x2 on cycle 0, then wait dt cycles before driving COMMIT */
            rtl.top->s_axis_syndrome_tdata  = 0x02u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 1;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast  = 0;

            uint16_t first_cz = 0;
            bool got_first = false;
            for (uint32_t s = 0; s < dt; s++) {
                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid && !got_first) {
                    first_cz = static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu);
                    got_first = true;
                }
            }
            /* Drive COMMIT handshake */
            rtl.top->s_axil_awaddr  = avid_l5::L5_OFF_COMMIT;
            rtl.top->s_axil_awvalid = 1;
            rtl.top->s_axil_wdata   = 1u;
            rtl.top->s_axil_wstrb   = 0x1u;
            rtl.top->s_axil_wvalid  = 1;
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid && !got_first) {
                first_cz = static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu);
                got_first = true;
            }
            rtl.top->s_axil_awvalid = 0;
            rtl.top->s_axil_wvalid  = 0;

            for (int g = 0; g < 35; g++) {
                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid && !got_first) {
                    first_cz = static_cast<uint16_t>((rtl.top->m_axis_correction_tdata >> 9) & 0x1FFu);
                    got_first = true;
                }
            }
            /* First decode must cleanly use either old_z (0x004) or new_z (0x020 if commit won before ST_INIT) — NEVER a torn invalid correction */
            bool first_valid = got_first && (first_cz == 0x004u || first_cz == 0x020u) &&
                               (avid_l5::gf2_hx_mul_ez(first_cz) == 0x2u);
            /* Next decode MUST strictly use new_z (0x020) */
            RtlDecodeTrace tr2 = rtl.decode_syndrome(0x2u, 0x0u, true);
            if (first_valid && tr2.corr_z == 0x020u) clean_phases++;
            total_c += rtl.sim_cycle;
        }
        cyc = total_c;
        det = "clean_commit_phases=" + std::to_string(clean_phases) + "/25";
        return (clean_phases == 25);
    });

    /* L5-FLT-09: Exhaustive cycle-by-cycle aresetn race sweep (dt = 0..24 cycles from syndrome accept) */
    suite.run_case("L5-FLT-09", "G5", "Cycle-by-cycle aresetn race sweep across dt=0..24 cycles recovers cleanly on every clock phase", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint64_t total_c = 0;
        int clean_phases = 0;
        for (uint32_t dt = 0; dt <= 24; dt++) {
            AvidVerilatorRtlModel rtl(4, 6);
            rtl.top->s_axis_syndrome_tdata  = 0x96u;
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast  = 1;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast  = 0;
            if (dt > 0) rtl.step_clock(dt);

            rtl.reset_model(5);
            RtlDecodeTrace rec = rtl.decode_syndrome(0x6u, 0x9u, true);
            if (rec.bp_core_cycles == 17u && rec.total_axis_cycles == 22u &&
                avid_l5::gf2_hz_mul_ex(rec.corr_x) == 0x9u &&
                avid_l5::gf2_hx_mul_ez(rec.corr_z) == 0x6u) {
                clean_phases++;
            }
            total_c += rtl.sim_cycle;
        }
        cyc = total_c;
        det = "clean_reset_phases=" + std::to_string(clean_phases) + "/25";
        return (clean_phases == 25);
    });

    /* L5-FLT-10: 10x model construct/configure/stream/destruct lifecycle determinism */
    suite.run_case("L5-FLT-10", "G5,G8", "10 consecutive Vaa_qec_top construct/stream/destruct lifecycles produce identical cycle & output hashes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint64_t ref_cyc = 0;
        uint32_t ref_hash = 0;
        bool all_same = true;
        for (int inst = 0; inst < 10; inst++) {
            AvidVerilatorRtlModel rtl(4, 8);
            uint64_t c = 0;
            rtl.axil_write(0x000u, 0x0190u, 0x3u, &c);
            rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
            uint32_t h = 2166136261u;
            for (int r = 0; r < 64; r++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(r & 0xFu),
                                                        static_cast<uint8_t>((r * 5) & 0xFu),
                                                        (r == 63));
                h = (h ^ tr.corr_x) * 16777619u;
                h = (h ^ tr.corr_z) * 16777619u;
            }
            if (inst == 0) {
                ref_cyc  = rtl.sim_cycle;
                ref_hash = h;
            } else if (rtl.sim_cycle != ref_cyc || h != ref_hash) {
                all_same = false;
            }
            cyc += rtl.sim_cycle;
        }
        det = "instances=10 per_inst_cyc=" + std::to_string(ref_cyc) +
              " fnv1a=0x" + std::to_string(ref_hash) + " identical=" + std::to_string(all_same);
        return all_same;
    });

    return suite.finalize();
}
