/* ===========================================================================
 * test_correction_output_contract.cpp — Layer 5 Set 2.2 Correction Egress Tests
 * Canonical IDs: L5-O01 .. L5-O16 (Gates G2, G6, G8)
 * Verifies m_axis_correction bit-packing, zero-syndrome fixed point, single-qubit
 * recovery, selective 1-cycle epoch_irq, and deterministic replay on Vaa_qec_top.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"
#include <random>

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set2_correction_output_contract", "Set2", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) return 2;
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-O01: m_axis_correction_tdata bit-packing: [8:0]=corr_x, [17:9]=corr_z, [31:18]=0 */
    suite.run_case("L5-O01", "G2", "m_axis_correction_tdata packs corr_x in [8:0], corr_z in [17:9], and zeros [31:18]", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axis_syndrome_tdata = 0x96u; /* sx=0x6 (q4 Z), sz=0x9 (q4 X) */
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        uint32_t raw_word = 0xFFFFFFFFu;
        for (int g = 0; g < 35; g++) {
            rtl.step_clock(1);
            if (rtl.top->m_axis_correction_tvalid) {
                raw_word = rtl.top->m_axis_correction_tdata;
                rtl.step_clock(1);
                break;
            }
        }
        uint32_t cx = raw_word & 0x1FFu;
        uint32_t cz = (raw_word >> 9) & 0x1FFu;
        uint32_t upper = raw_word >> 18;
        cyc = rtl.sim_cycle;
        det = "raw_tdata=0x" + std::to_string(raw_word) + " cx=" + std::to_string(cx) +
              " cz=" + std::to_string(cz) + " upper14=" + std::to_string(upper);
        return cx == 0x10u && cz == 0x10u && upper == 0u;
    });

    /* L5-O02: Zero-syndrome (0, 0) -> (0, 0) fixed-point invariant across all K=1..5 and OSD_MODE=0..2 */
    suite.run_case("L5-O02", "G2,G6", "Zero-syndrome (sx=0, sz=0) maps strictly to (corr_x=0, corr_z=0) across all 15 (K, OSD_MODE) states", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint32_t m = 0; m <= 2; m++) {
            rtl.axil_write(0x058u, m, 0x1u, &cyc);
            for (uint32_t k = 1; k <= 5; k++) {
                rtl.axil_write(0x054u, k, 0x1u, &cyc);
                RtlDecodeTrace tr = rtl.decode_syndrome(0u, 0u, false);
                if (tr.corr_x != 0u || tr.corr_z != 0u) return false;
            }
        }
        cyc = rtl.sim_cycle;
        det = "15/15 (K, OSD_MODE) zero-syndrome decodes produced exact (0, 0)";
        return true;
    });

    /* L5-O03: Pure X-syndrome (sx != 0, sz == 0) produces corr_x == 0 across all 15 sx */
    suite.run_case("L5-O03", "G2,G6", "All 15 non-zero X-syndromes with sz=0 produce corr_x==0 and valid non-zero corr_z", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint8_t sx = 1; sx <= 15; sx++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, 0u, false);
            if (tr.corr_x != 0u || tr.corr_z == 0u || avid_l5::gf2_hx_mul_ez(tr.corr_z) != sx) return false;
        }
        cyc = rtl.sim_cycle;
        det = "15/15 X-syndromes satisfied with zero cross-basis leakage into corr_x";
        return true;
    });

    /* L5-O04: Pure Z-syndrome (sx == 0, sz != 0) produces corr_z == 0 across all 15 sz */
    suite.run_case("L5-O04", "G2,G6", "All 15 non-zero Z-syndromes with sx=0 produce corr_z==0 and valid non-zero corr_x", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (uint8_t sz = 1; sz <= 15; sz++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(0u, sz, false);
            if (tr.corr_z != 0u || tr.corr_x == 0u || avid_l5::gf2_hz_mul_ex(tr.corr_x) != sz) return false;
        }
        cyc = rtl.sim_cycle;
        det = "15/15 Z-syndromes satisfied with zero cross-basis leakage into corr_z";
        return true;
    });

    /* L5-O05: Exact single-qubit Z-error recovery across all 9 data qubits q=0..8 */
    suite.run_case("L5-O05", "G2,G6", "All 9 single-qubit Z-errors e_Z=(1<<q) recovered uniquely under qubit-calibrated priors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int q = 0; q < 9; q++) {
            for (int i = 0; i < 9; i++) {
                uint16_t w = (i == q) ? 0x00A0u : 0x0320u;
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), w, 0x3u, &cyc);
            }
            rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
            uint8_t sx = avid_l5::gf2_hx_mul_ez(static_cast<uint16_t>(1u << q));
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, 0u, false);
            if (tr.corr_z != (1u << q) || tr.corr_x != 0u) return false;
        }
        cyc = rtl.sim_cycle;
        det = "9/9 single-qubit Z errors recovered as exact (1 << q)";
        return true;
    });

    /* L5-O06: Exact single-qubit X-error recovery across all 9 data qubits q=0..8 */
    suite.run_case("L5-O06", "G2,G6", "All 9 single-qubit X-errors e_X=(1<<q) recovered uniquely under qubit-calibrated priors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int q = 0; q < 9; q++) {
            for (int i = 0; i < 9; i++) {
                uint16_t w = (i == q) ? 0x00A0u : 0x0320u;
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), w, 0x3u, &cyc);
            }
            rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
            uint8_t sz = avid_l5::gf2_hz_mul_ex(static_cast<uint16_t>(1u << q));
            RtlDecodeTrace tr = rtl.decode_syndrome(0u, sz, false);
            if (tr.corr_x != (1u << q) || tr.corr_z != 0u) return false;
        }
        cyc = rtl.sim_cycle;
        det = "9/9 single-qubit X errors recovered as exact (1 << q)";
        return true;
    });

    /* L5-O07: Simultaneous Pauli-Y (joint X + Z) single-qubit error recovery */
    suite.run_case("L5-O07", "G2,G6", "Simultaneous Pauli-Y errors (e_X = e_Z = 1<<q) decoded in parallel across all 9 qubits", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        for (int q = 0; q < 9; q++) {
            for (int i = 0; i < 9; i++) {
                uint16_t w = (i == q) ? 0x00A0u : 0x0320u;
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(i), w, 0x3u, &cyc);
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(i), w, 0x3u, &cyc);
            }
            rtl.axil_write(0x048u, 1u, 0x1u, &cyc);
            uint8_t sx = avid_l5::gf2_hx_mul_ez(static_cast<uint16_t>(1u << q));
            uint8_t sz = avid_l5::gf2_hz_mul_ex(static_cast<uint16_t>(1u << q));
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, false);
            if (tr.corr_x != (1u << q) || tr.corr_z != (1u << q)) return false;
        }
        cyc = rtl.sim_cycle;
        det = "9/9 Pauli-Y errors recovered simultaneously on corr_x and corr_z";
        return true;
    });

    /* L5-O08: m_axis_correction_tlast propagation across mixed 16-beat tlast pattern */
    suite.run_case("L5-O08", "G2", "m_axis_correction_tlast matches s_axis_syndrome_tlast 1-to-1 across 16 mixed packets", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const bool tlasts[16] = {0,0,0,1, 0,1,0,0, 1,1,0,0, 0,0,0,1};
        for (int i = 0; i < 16; i++) {
            rtl.top->s_axis_syndrome_tdata = static_cast<uint16_t>((i & 0xFu));
            rtl.top->s_axis_syndrome_tvalid = 1;
            rtl.top->s_axis_syndrome_tlast = tlasts[i] ? 1 : 0;
            rtl.step_clock(1);
            rtl.top->s_axis_syndrome_tvalid = 0;
            rtl.top->s_axis_syndrome_tlast = 0;
            bool seen = false, got_last = false;
            for (int g = 0; g < 35; g++) {
                rtl.step_clock(1);
                if (rtl.top->m_axis_correction_tvalid) {
                    seen = true;
                    got_last = (rtl.top->m_axis_correction_tlast != 0);
                    rtl.step_clock(1);
                    break;
                }
            }
            if (!seen || got_last != tlasts[i]) return false;
        }
        cyc = rtl.sim_cycle;
        det = "16/16 mixed tlast flags propagated 1-to-1 to m_axis_correction_tlast";
        return true;
    });

    /* L5-O09: Single-cycle epoch_irq pulse width on s_axis_tlast=1 */
    suite.run_case("L5-O09", "G2", "epoch_irq asserts for strictly 1 clock cycle (4.0 ns) on correction handshake with tlast=1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->s_axis_syndrome_tdata = 0x06u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        int irq_high_cycles = 0;
        for (int c = 0; c < 35; c++) {
            rtl.step_clock(1);
            if (rtl.top->epoch_irq) irq_high_cycles++;
        }
        cyc = rtl.sim_cycle;
        det = "epoch_irq high duration=" + std::to_string(irq_high_cycles) + " cycle";
        return irq_high_cycles == 1;
    });

    /* L5-O10: Zero epoch_irq assertions across 100 consecutive non-terminal (tlast=0) syndromes */
    suite.run_case("L5-O10", "G2", "epoch_irq remains 0 across 100 consecutive syndromes with s_axis_syndrome_tlast=0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int irq_count = 0;
        for (int i = 0; i < 100; i++) {
            RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(i & 0xFu), static_cast<uint8_t>((i >> 4) & 0xFu), false);
            if (tr.epoch_irq_pulsed) irq_count++;
        }
        cyc = rtl.sim_cycle;
        det = "100x tlast=0 syndromes -> irq_count=" + std::to_string(irq_count);
        return irq_count == 0;
    });

    /* L5-O11: epoch_irq gated by m_axis_correction_tready (never fires while stalled!) */
    suite.run_case("L5-O11", "G2,G5", "epoch_irq remains 0 while tvalid=1 & tlast=1 is stalled by m_axis_correction_tready=0, firing once upon release", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.top->m_axis_correction_tready = 0;
        rtl.top->s_axis_syndrome_tdata = 0x06u;
        rtl.top->s_axis_syndrome_tvalid = 1;
        rtl.top->s_axis_syndrome_tlast = 1;
        rtl.step_clock(1);
        rtl.top->s_axis_syndrome_tvalid = 0;
        rtl.top->s_axis_syndrome_tlast = 0;
        int premature_irq = 0;
        for (int c = 0; c < 35; c++) {
            rtl.step_clock(1);
            if (rtl.top->epoch_irq) premature_irq++;
        }
        bool stalled_tlast = (rtl.top->m_axis_correction_tvalid == 1) && (rtl.top->m_axis_correction_tlast == 1);
        rtl.top->m_axis_correction_tready = 1;
        rtl.step_clock(1); /* Handshake occurs on this edge -> epoch_irq=1 after edge! */
        bool pulsed_after = (rtl.top->epoch_irq == 1);
        rtl.step_clock(1);
        bool cleared_next = (rtl.top->epoch_irq == 0);
        cyc = rtl.sim_cycle;
        det = "premature_irq=" + std::to_string(premature_irq) + " pulsed_on_release=" + std::to_string(pulsed_after);
        return premature_irq == 0 && stalled_tlast && pulsed_after && cleared_next;
    });

    /* L5-O12: Idle egress bus cleanliness (tdata=0, tlast=0 when TX FIFO empty) */
    suite.run_case("L5-O12", "G2", "m_axis_correction_tdata and tlast drive 0 cleanly whenever m_axis_correction_tvalid==0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        rtl.decode_syndrome(0xFu, 0xFu, true);
        rtl.step_clock(4);
        bool clean = (rtl.top->m_axis_correction_tvalid == 0) &&
                     (rtl.top->m_axis_correction_tdata == 0u) &&
                     (rtl.top->m_axis_correction_tlast == 0);
        cyc = rtl.sim_cycle;
        det = "idle tvalid=0, tdata=0x0, tlast=0";
        return clean;
    });

    /* L5-O13: Deterministic bit-for-bit & cycle-for-cycle replay across 5 independent instances */
    suite.run_case("L5-O13", "G2,G8", "5 independent Vaa_qec_top instances produce bit-for-bit and cycle-for-cycle identical traces", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        std::vector<uint64_t> digests;
        uint64_t total_c = 0;
        for (int run = 0; run < 5; run++) {
            AvidVerilatorRtlModel rtl(4, 10);
            uint64_t c = 0;
            for (int q = 0; q < 9; q++) {
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), 0x0120u + 13u * q, 0x3u, &c);
                rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), 0x01E0u + 7u * q, 0x3u, &c);
            }
            rtl.axil_write(0x048u, 1u, 0x1u, &c);
            uint64_t h = 1469598103934665603ULL;
            for (int s = 0; s < 32; s++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(s & 0xF), static_cast<uint8_t>((s * 3) & 0xF), (s % 8) == 7);
                h = (h ^ tr.corr_x) * 1099511628211ULL;
                h = (h ^ tr.corr_z) * 1099511628211ULL;
                h = (h ^ tr.end_cycle) * 1099511628211ULL;
            }
            digests.push_back(h);
            total_c += rtl.sim_cycle;
        }
        cyc = total_c;
        for (size_t i = 1; i < digests.size(); i++) {
            if (digests[i] != digests[0]) return false;
        }
        det = "5/5 independent runs matched FNV-1a trace digest=" + std::to_string(digests[0]);
        return true;
    });

    /* L5-O14: Deterministic bit-for-bit replay across 5 consecutive resets in one instance */
    suite.run_case("L5-O14", "G2,G8", "5 consecutive reset_model() cycles inside one Vaa_qec_top instance produce identical outputs & deltas", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        std::vector<uint64_t> hashes;
        for (int r = 0; r < 5; r++) {
            rtl.reset_model(8);
            uint64_t base_c = rtl.sim_cycle, dummy = 0;
            rtl.axil_write(0x000u, 0x00C0u, 0x3u, &dummy);
            rtl.axil_write(0x048u, 1u, 0x1u, &dummy);
            uint64_t h = 0;
            for (uint8_t s = 0; s < 16; s++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(s, static_cast<uint8_t>(15u - s), s == 15);
                h = (h * 1315423911ULL) ^ ((uint64_t)tr.corr_x << 32) ^ ((uint64_t)tr.corr_z << 16) ^ (tr.end_cycle - base_c);
            }
            hashes.push_back(h);
        }
        cyc = rtl.sim_cycle;
        for (size_t i = 1; i < hashes.size(); i++) {
            if (hashes[i] != hashes[0]) return false;
        }
        det = "5/5 post-reset trajectories identical (hash=" + std::to_string(hashes[0]) + ")";
        return true;
    });

    /* L5-O15: Deterministic seeded 256-round PRNG stream replay (seed=42 vs seed=42 & seed=108) */
    suite.run_case("L5-O15", "G2,G8", "256-round seeded syndrome replay is bit-exact for identical seeds (42==42) and distinct for 42!=108", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        auto run_seeded = [](uint32_t seed, uint64_t &out_cyc) -> uint64_t {
            AvidVerilatorRtlModel rtl(4, 8);
            std::mt19937 rng(seed);
            uint64_t h = 2166136261ULL;
            for (int r = 0; r < 256; r++) {
                uint8_t sx = static_cast<uint8_t>(rng() & 0xFu);
                uint8_t sz = static_cast<uint8_t>(rng() & 0xFu);
                RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, (r & 63) == 63);
                h = (h ^ tr.corr_x) * 16777619ULL;
                h = (h ^ tr.corr_z) * 16777619ULL;
            }
            out_cyc += rtl.sim_cycle;
            return h;
        };
        uint64_t h42_a = run_seeded(42u, cyc);
        uint64_t h42_b = run_seeded(42u, cyc);
        uint64_t h108  = run_seeded(108u, cyc);
        det = "h(42)=" + std::to_string(h42_a) + " == h(42) and != h(108)=" + std::to_string(h108);
        return h42_a == h42_b && h42_a != h108;
    });

    /* L5-O16: Reference equivalence between Vaa_qec_top and eval_rtl_bp_reference across all OSD modes */
    suite.run_case("L5-O16", "G2,G6", "Vaa_qec_top matches C++ golden reference across OSD_MODE 0, 1, 2 for all 16 orthogonal pairs", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int mism = 0;
        for (uint32_t m = 0; m <= 2; m++) {
            rtl.axil_write(0x058u, m, 0x1u, &cyc);
            for (uint8_t s = 0; s < 16; s++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(s, s, false);
                uint16_t rx = AvidVerilatorRtlModel::eval_rtl_bp_reference(s, rtl.active_x, 3, true, m);
                uint16_t rz = AvidVerilatorRtlModel::eval_rtl_bp_reference(s, rtl.active_z, 3, false, m);
                if (tr.corr_x != rx || tr.corr_z != rz) mism++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "48/48 Mode 0..2 decodes matched C++ golden model (mismatches=" + std::to_string(mism) + ")";
        return mism == 0;
    });

    return suite.finalize();
}
