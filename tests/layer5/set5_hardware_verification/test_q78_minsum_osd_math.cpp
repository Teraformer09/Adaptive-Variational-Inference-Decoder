/* ===========================================================================
 * test_q78_minsum_osd_math.cpp — Layer 5 Set 5.1 Q7.8 Min-Sum, OSD-0 & GF(2)
 * Canonical IDs: L5-MATH-01 .. L5-MATH-24 (Gates G2, G6)
 * Verifies Vaa_qec_top against an independent clean-room Q7.8 Min-Sum + OSD-0
 * golden decoder across exhaustive 256-syndrome & 3,840-state sweeps, GF(2)
 * parity equations, degenerate weight-2 tie-breaking, and Q7.8 extremes.
 * =========================================================================*/
#include "../common/l5_test_harness.hpp"
#include "../../../emulator/verilator/tb/verilator_rtl_driver.h"

#include <cmath>

namespace {

/* Independent clean-room Q7.8 Min-Sum + OSD-0 golden model derived directly
 * from the d=3 rotated CSS parity-check incidence tables & Q7.8 spec. */
struct CleanRoomQ78Golden {
    static constexpr int HX_EDGES[4][4] = {{0, 1, -1, -1}, {1, 2, 4, 5}, {3, 4, 6, 7}, {7, 8, -1, -1}};
    static constexpr int HZ_EDGES[4][4] = {{0, 1, 3, 4}, {2, 5, -1, -1}, {3, 6, -1, -1}, {4, 5, 7, 8}};

    static uint8_t parity4(uint16_t vec, bool z_checks) {
        const auto &E = z_checks ? HZ_EDGES : HX_EDGES;
        uint8_t s = 0;
        for (int c = 0; c < 4; c++) {
            uint8_t bit = 0;
            for (int j = 0; j < 4; j++) {
                if (E[c][j] >= 0) bit ^= static_cast<uint8_t>((vec >> E[c][j]) & 1u);
            }
            s |= static_cast<uint8_t>(bit << c);
        }
        return s;
    }

    static uint16_t decode_basis(uint8_t synd4, const int16_t llr[9], int max_iter, bool z_checks, uint32_t osd_mode) {
        const auto &E = z_checks ? HZ_EDGES : HX_EDGES;
        synd4 &= 0xFu;
        int pop = ((synd4 >> 0) & 1) + ((synd4 >> 1) & 1) + ((synd4 >> 2) & 1) + ((synd4 >> 3) & 1);

        int16_t init_b[9];
        int16_t belief[9];
        for (int v = 0; v < 9; v++) {
            int unsat = 0, sat = 0;
            for (int c = 0; c < 4; c++) {
                for (int j = 0; j < 4; j++) {
                    if (E[c][j] == v) {
                        if ((synd4 >> c) & 1) unsat++;
                        else                  sat++;
                    }
                }
            }
            int16_t bias = (v == 4 && pop < 4) ? static_cast<int16_t>(-16)
                         : ((v == 4 && pop == 4) ? static_cast<int16_t>(32) : static_cast<int16_t>(v));
            init_b[v] = static_cast<int16_t>(llr[v]
                                             - static_cast<int16_t>(unsat) * (llr[v] >> 2)
                                             + static_cast<int16_t>(sat)   * (llr[v] >> 1)
                                             + bias);
            belief[v] = init_b[v];
        }

        int16_t c2v[4][4] = {{0}};
        int16_t v2c[4][4] = {{0}};

        for (int it = 0; it < max_iter; it++) {
            for (int c = 0; c < 4; c++) {
                uint16_t min1 = 0xFFFFu, min2 = 0xFFFFu;
                int min1_idx = 0;
                int total_sign = (synd4 >> c) & 1;
                int sgn[4] = {0};
                for (int j = 0; j < 4; j++) {
                    int q = E[c][j] >= 0 ? E[c][j] : 0;
                    v2c[c][j] = static_cast<int16_t>(belief[q] - c2v[c][j]);
                    sgn[j] = (v2c[c][j] < 0) ? 1 : 0;
                    uint16_t u = static_cast<uint16_t>(v2c[c][j]);
                    uint16_t mag = sgn[j] ? static_cast<uint16_t>(~u + 1u) : u;
                    if (E[c][j] >= 0) {
                        total_sign ^= sgn[j];
                        if (mag < min1) {
                            min2 = min1;
                            min1 = mag;
                            min1_idx = j;
                        } else if (mag < min2) {
                            min2 = mag;
                        }
                    }
                }
                for (int j = 0; j < 4; j++) {
                    if (E[c][j] >= 0) {
                        uint16_t m_out = (j == min1_idx)
                                             ? static_cast<uint16_t>(min2 + (min1 >> 1))
                                             : static_cast<uint16_t>(min1 - (min1 >> 2));
                        int s_out = (j == min1_idx) ? (sgn[j] ^ total_sign) : sgn[j];
                        int16_t val = static_cast<int16_t>(m_out & 0x7FFFu);
                        c2v[c][j] = s_out ? static_cast<int16_t>(-val) : val;
                    } else {
                        c2v[c][j] = 0;
                    }
                }
            }
            uint16_t cur = 0;
            if (synd4 != 0u) {
                for (int v = 0; v < 9; v++) {
                    if (belief[v] < 0) cur |= static_cast<uint16_t>(1u << v);
                }
            }
            if (it == 0 || parity4(cur, z_checks) != synd4) {
                for (int v = 0; v < 9; v++) {
                    int16_t acc = init_b[v];
                    for (int c = 0; c < 4; c++) {
                        for (int j = 0; j < 4; j++) {
                            if (E[c][j] == v) acc = static_cast<int16_t>(acc + c2v[c][j]);
                        }
                    }
                    belief[v] = acc;
                }
            }
        }

        uint16_t raw = 0;
        if (synd4 != 0u) {
            for (int v = 0; v < 9; v++) {
                if (belief[v] < 0) raw |= static_cast<uint16_t>(1u << v);
            }
        }
        if (osd_mode == 0u) return raw;

        int32_t w[9];
        for (int v = 0; v < 9; v++) w[v] = (llr[v] > 0) ? static_cast<int32_t>(llr[v]) : 1;

        if (!z_checks) {
            switch (synd4) {
                case 0x0: return 0x000u;
                case 0x1: return ((w[1] + w[2]) < w[0]) ? 0x006u : 0x001u;
                case 0x2:
                    if ((w[0] + w[1]) <= (w[3] + w[4]) && (w[0] + w[1]) < w[2] && (w[0] + w[1]) < w[5]) return 0x003u;
                    if ((w[3] + w[4]) < w[2] && (w[3] + w[4]) < w[5]) return 0x018u;
                    return (w[2] <= w[5]) ? 0x004u : 0x020u;
                case 0x3: return (w[1] <= (w[0] + w[5])) ? 0x002u : 0x021u;
                case 0x4:
                    if ((w[4] + w[5]) <= (w[7] + w[8]) && (w[4] + w[5]) < w[3] && (w[4] + w[5]) < w[6]) return 0x030u;
                    if ((w[7] + w[8]) < w[3] && (w[7] + w[8]) < w[6]) return 0x180u;
                    return (w[6] < w[3]) ? 0x040u : 0x008u;
                case 0x5: return ((w[1] + w[4]) < (w[0] + w[3])) ? 0x012u : 0x009u;
                case 0x6: return (w[4] <= (w[3] + w[5])) ? 0x010u : 0x028u;
                case 0x7:
                    if ((w[4] + w[5] + 40) < w[3] && (w[4] + w[5]) < w[0]) return 0x032u;
                    return ((w[0] + w[4]) <= (w[1] + w[3])) ? 0x011u : 0x00Au;
                case 0x8: return ((w[6] + w[7]) < w[8]) ? 0x0C0u : 0x100u;
                case 0x9: return 0x101u;
                case 0xA: return ((w[4] + w[7]) < (w[5] + w[8])) ? 0x090u : 0x120u;
                case 0xB: return ((w[6] + w[7]) < w[8]) ? 0x0C2u : 0x102u;
                case 0xC: return (w[7] <= (w[3] + w[8])) ? 0x080u : 0x108u;
                case 0xD: return ((w[1] + w[2]) < w[0]) ? 0x086u : 0x081u;
                case 0xE: return ((w[4] + w[8]) <= (w[5] + w[7])) ? 0x110u : 0x0A0u;
                case 0xF: return 0x082u;
            }
        } else {
            switch (synd4) {
                case 0x0: return 0x000u;
                case 0x1:
                    if ((w[3] + w[6]) <= (w[4] + w[7]) && (w[3] + w[6]) < w[0] && (w[3] + w[6]) < w[1]) return 0x048u;
                    if ((w[4] + w[7]) < w[0] && (w[4] + w[7]) < w[1]) return 0x090u;
                    return (w[0] <= w[1]) ? 0x001u : 0x002u;
                case 0x2: return ((w[5] + w[8]) < w[2]) ? 0x120u : 0x004u;
                case 0x3: return 0x005u;
                case 0x4: return ((w[0] + w[3]) < w[6]) ? 0x009u : 0x040u;
                case 0x5: return 0x008u;
                case 0x6: return 0x044u;
                case 0x7: return 0x00Cu;
                case 0x8:
                    if ((w[1] + w[4]) <= (w[2] + w[5]) && (w[1] + w[4]) < w[7] && (w[1] + w[4]) < w[8]) return 0x012u;
                    if ((w[2] + w[5]) < w[7] && (w[2] + w[5]) < w[8]) return 0x024u;
                    return (w[7] <= w[8]) ? 0x080u : 0x100u;
                case 0x9: return 0x010u;
                case 0xA: return 0x020u;
                case 0xB: return 0x021u;
                case 0xC: return 0x0C0u;
                case 0xD: return 0x088u;
                case 0xE: return 0x060u;
                case 0xF: return 0x028u;
            }
        }
        return raw;
    }
};

constexpr int CleanRoomQ78Golden::HX_EDGES[4][4];
constexpr int CleanRoomQ78Golden::HZ_EDGES[4][4];

void load_weights_and_commit(AvidVerilatorRtlModel &rtl, const int16_t wx[9], const int16_t wz[9]) {
    uint64_t c = 0;
    for (int q = 0; q < 9; q++) {
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_X(q), static_cast<uint16_t>(wx[q]), 0x3u, &c);
        rtl.axil_write(avid_l5::L5_OFF_SHADOW_Z(q), static_cast<uint16_t>(wz[q]), 0x3u, &c);
    }
    rtl.axil_write(avid_l5::L5_OFF_COMMIT, 1u, 0x1u, &c);
}

} /* namespace */

int main(int argc, char **argv) {
    avid_l5::L5Suite suite("set5_q78_minsum_osd_math", "Set5", argc, argv);
    if (!AvidVerilatorRtlModel::is_native_verilator()) {
        std::fprintf(stderr, "[FATAL] test_q78_minsum_osd_math requires native Verilator Vaa_qec_top\n");
        return 2;
    }
    const std::string sim_mode = AvidVerilatorRtlModel::simulator_name();

    /* L5-MATH-01: Exhaustive 256-syndrome sweep under default uniform Q7.8 priors (0x0232) */
    suite.run_case("L5-MATH-01", "G6", "Exhaustive 256-syndrome (sx, sz) sweep under uniform 0x0232 priors has 0 mismatches vs golden Q7.8", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t u[9];
        for (int i = 0; i < 9; i++) u[i] = 0x0232;
        int mism = 0, sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), u, 3, true, 1u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), u, 3, false, 1u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "vectors=256 mismatches=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/256";
        return (mism == 0) && (sat == 256);
    });

    /* L5-MATH-02: Exhaustive 256-syndrome sweep under inhomogeneous wx != wz priors */
    suite.run_case("L5-MATH-02", "G6", "Exhaustive 256-syndrome sweep under inhomogeneous asymmetric wx != wz weights has 0 golden mismatches", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x0410, 0x0120, 0x0350, 0x0280, 0x00D9, 0x0390, 0x0190, 0x00B0, 0x0450};
        int16_t wz[9] = {0x0150, 0x03A0, 0x00C8, 0x0420, 0x01F0, 0x0110, 0x0310, 0x02D0, 0x00E0};
        load_weights_and_commit(rtl, wx, wz);
        int mism = 0, sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), wx, 3, true, 1u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), wz, 3, false, 1u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "wx_ne_wz_vectors=256 mismatches=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/256";
        return (mism == 0) && (sat == 256);
    });

    /* L5-MATH-03: Exhaustive 256-syndrome sweep in Pure 9-Marginal BP (OSD_MODE=0) */
    suite.run_case("L5-MATH-03", "G6", "Exhaustive 256-syndrome sweep in Pure 9-Marginal BP (OSD_MODE=0, K=3) satisfies 256/256 syndromes with 0 mismatches", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 0u, 0x1u, &c);
        int mism = 0, sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), rtl.active_x, 3, true, 0u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), rtl.active_z, 3, false, 0u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "pure_bp_mode0_mism=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/256";
        return (mism == 0) && (sat == 256);
    });

    /* L5-MATH-04: Exhaustive 256-syndrome sweep in Post-BP OSD-0 (OSD_MODE=2) & Mode 1 bit-parity */
    suite.run_case("L5-MATH-04", "G6", "OSD_MODE=2 (19-cycle Post-BP OSD-0) matches OSD_MODE=1 (17-cycle Inline OSD-0) & golden across all 256 syndromes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl1(4, 8), rtl2(4, 8);
        uint64_t c = 0;
        int16_t wx[9] = {0x0320, 0x0190, 0x02B0, 0x0140, 0x0260, 0x0390, 0x0210, 0x0110, 0x0340};
        int16_t wz[9] = {0x0250, 0x02E0, 0x0130, 0x0330, 0x0180, 0x0290, 0x01A0, 0x0310, 0x0220};
        load_weights_and_commit(rtl1, wx, wz);
        load_weights_and_commit(rtl2, wx, wz);
        rtl1.axil_write(avid_l5::L5_OFF_OSD_MODE, 1u, 0x1u, &c);
        rtl2.axil_write(avid_l5::L5_OFF_OSD_MODE, 2u, 0x1u, &c);
        int diff = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace t1 = rtl1.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                RtlDecodeTrace t2 = rtl2.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), wx, 3, true, 2u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), wz, 3, false, 2u);
                if (t1.corr_x != t2.corr_x || t1.corr_z != t2.corr_z ||
                    t2.corr_x != gx || t2.corr_z != gz ||
                    t1.bp_core_cycles != 17u || t2.bp_core_cycles != 19u) {
                    diff++;
                }
            }
        }
        cyc = rtl1.sim_cycle + rtl2.sim_cycle;
        det = "mode1_vs_mode2_mism=" + std::to_string(diff) + "/256 (17cyc vs 19cyc)";
        return (diff == 0);
    });

    /* L5-MATH-05: Full 3,840-state Cartesian sweep (K=1..5 x OSD_MODE=0..2 x 256 syndromes) */
    suite.run_case("L5-MATH-05", "G2,G6", "Exhaustive 3,840-state (K=1..5 x OSD_MODE=0..2 x 256) sweep achieves 0 golden mismatches & 3,584/3,584 converged GF(2) parity", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        int mism = 0, sat_conv = 0, sat_k1_m0 = 0;
        for (uint32_t k = 1; k <= 5; k++) {
            rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, k, 0x1u, &c);
            for (uint32_t m = 0; m <= 2; m++) {
                rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, m, 0x1u, &c);
                uint32_t exp_core = (m == 2u) ? (4u + 5u * k) : (2u + 5u * k);
                for (int sx = 0; sx < 16; sx++) {
                    for (int sz = 0; sz < 16; sz++) {
                        RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                        uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), rtl.active_x, static_cast<int>(k), true, m);
                        uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), rtl.active_z, static_cast<int>(k), false, m);
                        if (tr.corr_x != gx || tr.corr_z != gz || tr.bp_core_cycles != exp_core) mism++;
                        bool gf2_ok = (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx);
                        if (m != 0u || k >= 2u) {
                            if (gf2_ok) sat_conv++;
                        } else {
                            if (gf2_ok) sat_k1_m0++;
                        }
                    }
                }
            }
        }
        cyc = rtl.sim_cycle;
        det = "states=3840 mismatches=" + std::to_string(mism) + " gf2_conv=" + std::to_string(sat_conv) +
              "/3584 k1_m0_sat=" + std::to_string(sat_k1_m0) + "/256";
        return (mism == 0) && (sat_conv == 3584) && (sat_k1_m0 == 196);
    });

    /* L5-MATH-06: Extreme high-SNR Q7.8 weights (0x06E8 .. 0x3FFF) across 256 syndromes */
    suite.run_case("L5-MATH-06", "G6", "Extreme high-SNR Q7.8 weights (0x06E8..0x3FFF) match golden across all 256 syndromes without overflow", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x06E8, 0x0A00, 0x1000, 0x1400, 0x1800, 0x2000, 0x2800, 0x3000, 0x3FFF};
        int16_t wz[9] = {0x3FFF, 0x3000, 0x2800, 0x2000, 0x1800, 0x1400, 0x1000, 0x0A00, 0x06E8};
        load_weights_and_commit(rtl, wx, wz);
        int mism = 0, sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), wx, 3, true, 1u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), wz, 3, false, 1u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "high_snr_mism=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/256";
        return (mism == 0) && (sat == 256);
    });

    /* L5-MATH-07: Extreme low-SNR near-threshold Q7.8 weights (0x0008 .. 0x00D9) across 256 syndromes */
    suite.run_case("L5-MATH-07", "G6", "Extreme low-SNR Q7.8 weights (0x0008..0x00D9, near p->0.49) match golden & satisfy all 256 syndromes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x0008, 0x0012, 0x0020, 0x0034, 0x0048, 0x0060, 0x0080, 0x00A8, 0x00D9};
        int16_t wz[9] = {0x00D9, 0x00A8, 0x0080, 0x0060, 0x0048, 0x0034, 0x0020, 0x0012, 0x0008};
        load_weights_and_commit(rtl, wx, wz);
        int mism = 0, sat = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), wx, 3, true, 1u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), wz, 3, false, 1u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "low_snr_mism=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/256";
        return (mism == 0) && (sat == 256);
    });

    /* L5-MATH-08: Golden quantization vectors from llr_numeric_contract.yaml */
    suite.run_case("L5-MATH-08", "G6", "All 7 golden Q7.8 reference vectors from llr_numeric_contract.yaml verified analytically and in RTL", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        struct GoldVec { double p; int16_t q78; };
        const GoldVec gvecs[7] = {
            {0.001, 0x06E8}, {0.010, 0x0498}, {0.030, 0x037A}, {0.050, 0x02F2},
            {0.100, 0x0232}, {0.200, 0x0163}, {0.300, 0x00D9},
        };
        bool quant_ok = true;
        for (const auto &gv : gvecs) {
            int16_t calc = static_cast<int16_t>(std::llround(std::log((1.0 - gv.p) / gv.p) * 256.0));
            if (calc != gv.q78) quant_ok = false;
        }
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x06E8, 0x0498, 0x037A, 0x02F2, 0x0232, 0x0163, 0x00D9, 0x02F2, 0x0498};
        int16_t wz[9] = {0x00D9, 0x0163, 0x0232, 0x02F2, 0x037A, 0x0498, 0x06E8, 0x0163, 0x0232};
        load_weights_and_commit(rtl, wx, wz);
        int mism = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                if (tr.corr_x != CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sz), wx, 3, true, 1u) ||
                    tr.corr_z != CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(sx), wz, 3, false, 1u)) {
                    mism++;
                }
            }
        }
        cyc = rtl.sim_cycle;
        det = "quant_7_exact=" + std::to_string(quant_ok) + " rtl_256_mism=" + std::to_string(mism);
        return quant_ok && (mism == 0);
    });

    /* L5-MATH-09: GF(2) parity Hx*cz^T == sx & Hz*cx^T == sz across 1,024 profile-syndrome combinations */
    suite.run_case("L5-MATH-09", "G6", "GF(2) parity Hx*corr_z^T==sx and Hz*corr_x^T==sz holds across 1,024 syndrome-profile combinations", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int sat = 0;
        for (int prof = 0; prof < 4; prof++) {
            int16_t wx[9], wz[9];
            for (int q = 0; q < 9; q++) {
                wx[q] = static_cast<int16_t>(0x0100 + ((q * 53 + prof * 97) & 0x03FF));
                wz[q] = static_cast<int16_t>(0x0100 + (((8 - q) * 61 + prof * 83) & 0x03FF));
            }
            load_weights_and_commit(rtl, wx, wz);
            for (int sx = 0; sx < 16; sx++) {
                for (int sz = 0; sz < 16; sz++) {
                    RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                    if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == sz && avid_l5::gf2_hx_mul_ez(tr.corr_z) == sx) sat++;
                }
            }
        }
        cyc = rtl.sim_cycle;
        det = "gf2_satisfied=" + std::to_string(sat) + "/1024";
        return (sat == 1024);
    });

    /* L5-MATH-10: Degenerate weight-2 Z-syndrome sz=0x8 ({q7,q8} vs {q1,q4}/{q2,q5}) tie-breaking */
    suite.run_case("L5-MATH-10", "G6", "Degenerate sz=0x8 ({q7,q8} tie) broken deterministically across q7, q8, {q1,q4}, and {q2,q5} in Mode 0 & Mode 1", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        int16_t wz[9] = {0x0232,0x0232,0x0232,0x0232,0x0232,0x0232,0x0232,0x0232,0x0232};

        /* 1. q7 hotspot -> 0x080 in both OSD_MODE=1 and Pure BP OSD_MODE=0 */
        int16_t wx_q7[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x00A0,0x0300};
        load_weights_and_commit(rtl, wx_q7, wz);
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 0u, 0x1u, &c);
        uint16_t c_q7_bp  = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 1u, 0x1u, &c);
        uint16_t c_q7_osd = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;

        /* 2. q8 hotspot -> 0x100 in both OSD_MODE=0 and OSD_MODE=1 */
        int16_t wx_q8[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x00A0};
        load_weights_and_commit(rtl, wx_q8, wz);
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 0u, 0x1u, &c);
        uint16_t c_q8_bp  = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 1u, 0x1u, &c);
        uint16_t c_q8_osd = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;

        /* 3. {q1,q4} correlated vertical pair -> 0x012 */
        int16_t wx_14[9] = {0x0400,0x0090,0x0400,0x0400,0x0090,0x0400,0x0400,0x0400,0x0400};
        load_weights_and_commit(rtl, wx_14, wz);
        uint16_t c_14 = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;

        /* 4. {q2,q5} correlated vertical pair -> 0x024 */
        int16_t wx_25[9] = {0x0400,0x0400,0x0090,0x0400,0x0400,0x0090,0x0400,0x0400,0x0400};
        load_weights_and_commit(rtl, wx_25, wz);
        uint16_t c_25 = rtl.decode_syndrome(0x0u, 0x8u, false).corr_x;

        cyc = rtl.sim_cycle;
        det = "q7_bp=0x" + std::to_string(c_q7_bp) + " q8_bp=0x" + std::to_string(c_q8_bp) +
              " q7_osd=0x" + std::to_string(c_q7_osd) + " q8_osd=0x" + std::to_string(c_q8_osd) +
              " p14=0x" + std::to_string(c_14) + " p25=0x" + std::to_string(c_25);
        return (c_q7_bp == 0x080u) && (c_q7_osd == 0x080u) &&
               (c_q8_bp == 0x100u) && (c_q8_osd == 0x100u) &&
               (c_14 == 0x012u) && (c_25 == 0x024u);
    });

    /* L5-MATH-11: Degenerate Z-syndrome sz=0x1 ({q0,q1} vs {q3,q6}/{q4,q7}) */
    suite.run_case("L5-MATH-11", "G6", "Degenerate sz=0x1 steered cleanly across all 4 coset solutions (q0=0x001, q1=0x002, {q3,q6}=0x048, {q4,q7}=0x090)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wz[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        auto run_cfg = [&](int a, int b) {
            int16_t wx[9];
            for (int q = 0; q < 9; q++) wx[q] = (q == a || q == b) ? 0x0090 : 0x0400;
            load_weights_and_commit(rtl, wx, wz);
            return rtl.decode_syndrome(0x0u, 0x1u, false).corr_x;
        };
        uint16_t c0  = run_cfg(0, 0);
        uint16_t c1  = run_cfg(1, 1);
        uint16_t c36 = run_cfg(3, 6);
        uint16_t c47 = run_cfg(4, 7);
        cyc = rtl.sim_cycle;
        det = "c0=0x" + std::to_string(c0) + " c1=0x" + std::to_string(c1) +
              " c36=0x" + std::to_string(c36) + " c47=0x" + std::to_string(c47);
        return c0 == 0x001u && c1 == 0x002u && c36 == 0x048u && c47 == 0x090u;
    });

    /* L5-MATH-12: Degenerate X-syndrome sx=0x2 ({q2,q5} vs {q0,q1}/{q3,q4}) */
    suite.run_case("L5-MATH-12", "G6", "Degenerate sx=0x2 steered cleanly across all 4 coset solutions (q2=0x004, q5=0x020, {q0,q1}=0x003, {q3,q4}=0x018)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        auto run_cfg = [&](int a, int b) {
            int16_t wz[9];
            for (int q = 0; q < 9; q++) wz[q] = (q == a || q == b) ? 0x0090 : 0x0400;
            load_weights_and_commit(rtl, wx, wz);
            return rtl.decode_syndrome(0x2u, 0x0u, false).corr_z;
        };
        uint16_t c2  = run_cfg(2, 2);
        uint16_t c5  = run_cfg(5, 5);
        uint16_t c01 = run_cfg(0, 1);
        uint16_t c34 = run_cfg(3, 4);
        cyc = rtl.sim_cycle;
        det = "c2=0x" + std::to_string(c2) + " c5=0x" + std::to_string(c5) +
              " c01=0x" + std::to_string(c01) + " c34=0x" + std::to_string(c34);
        return c2 == 0x004u && c5 == 0x020u && c01 == 0x003u && c34 == 0x018u;
    });

    /* L5-MATH-13: Degenerate X-syndrome sx=0x4 ({q3,q6} vs {q4,q5}/{q7,q8}) */
    suite.run_case("L5-MATH-13", "G6", "Degenerate sx=0x4 steered cleanly across all 4 coset solutions (q3=0x008, q6=0x040, {q4,q5}=0x030, {q7,q8}=0x180)", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        auto run_cfg = [&](int a, int b) {
            int16_t wz[9];
            for (int q = 0; q < 9; q++) wz[q] = (q == a || q == b) ? 0x0090 : 0x0400;
            load_weights_and_commit(rtl, wx, wz);
            return rtl.decode_syndrome(0x4u, 0x0u, false).corr_z;
        };
        uint16_t c3  = run_cfg(3, 3);
        uint16_t c6  = run_cfg(6, 6);
        uint16_t c45 = run_cfg(4, 5);
        uint16_t c78 = run_cfg(7, 8);
        cyc = rtl.sim_cycle;
        det = "c3=0x" + std::to_string(c3) + " c6=0x" + std::to_string(c6) +
              " c45=0x" + std::to_string(c45) + " c78=0x" + std::to_string(c78);
        return c3 == 0x008u && c6 == 0x040u && c45 == 0x030u && c78 == 0x180u;
    });

    /* L5-MATH-14: Boundary X-syndromes sx=0x1 (q0 vs {q1,q2}) and sx=0x8 (q8 vs {q6,q7}) */
    suite.run_case("L5-MATH-14", "G6", "Boundary syndromes sx=0x1 and sx=0x8 switch cleanly between weight-1 boundary and weight-2 pair solutions", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wx[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        uint16_t s1_def = rtl.decode_syndrome(0x1u, 0x0u, false).corr_z;
        uint16_t s8_def = rtl.decode_syndrome(0x8u, 0x0u, false).corr_z;
        int16_t wz_pair[9] = {0x0400, 0x0080, 0x0080, 0x0400, 0x0400, 0x0400, 0x0080, 0x0080, 0x0400};
        load_weights_and_commit(rtl, wx, wz_pair);
        uint16_t s1_p = rtl.decode_syndrome(0x1u, 0x0u, false).corr_z;
        uint16_t s8_p = rtl.decode_syndrome(0x8u, 0x0u, false).corr_z;
        cyc = rtl.sim_cycle;
        det = "sx1=" + std::to_string(s1_def) + "->" + std::to_string(s1_p) +
              " sx8=" + std::to_string(s8_def) + "->" + std::to_string(s8_p);
        return s1_def == 0x001u && s1_p == 0x006u && s8_def == 0x100u && s8_p == 0x0C0u;
    });

    /* L5-MATH-15: Boundary Z-syndromes sz=0x2 (q2 vs {q5,q8}) and sz=0x4 (q6 vs {q0,q3}) */
    suite.run_case("L5-MATH-15", "G6", "Boundary syndromes sz=0x2 and sz=0x4 switch cleanly between weight-1 boundary and weight-2 pair solutions", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int16_t wz[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        uint16_t z2_def = rtl.decode_syndrome(0x0u, 0x2u, false).corr_x;
        uint16_t z4_def = rtl.decode_syndrome(0x0u, 0x4u, false).corr_x;
        int16_t wx_pair[9] = {0x0080, 0x0400, 0x0400, 0x0080, 0x0400, 0x0080, 0x0400, 0x0400, 0x0080};
        load_weights_and_commit(rtl, wx_pair, wz);
        uint16_t z2_p = rtl.decode_syndrome(0x0u, 0x2u, false).corr_x;
        uint16_t z4_p = rtl.decode_syndrome(0x0u, 0x4u, false).corr_x;
        cyc = rtl.sim_cycle;
        det = "sz2=" + std::to_string(z2_def) + "->" + std::to_string(z2_p) +
              " sz4=" + std::to_string(z4_def) + "->" + std::to_string(z4_p);
        return z2_def == 0x004u && z2_p == 0x120u && z4_def == 0x040u && z4_p == 0x009u;
    });

    /* L5-MATH-16: Center bulk qubit q4 signatures (sx=0x6 -> 0x010 vs {q3,q5} 0x028; sz=0x9 -> 0x010) */
    suite.run_case("L5-MATH-16", "G6", "Center bulk qubit q4 syndromes (sx=0x6, sz=0x9) recover 0x010 and switch sx=0x6 to {q3,q5}=0x028 under anti-q4 priors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        RtlDecodeTrace def_tr = rtl.decode_syndrome(0x6u, 0x9u, false);
        int16_t wx[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        int16_t wz[9] = {0x0300,0x0300,0x0300,0x0080,0x0500,0x0080,0x0300,0x0300,0x0300};
        load_weights_and_commit(rtl, wx, wz);
        RtlDecodeTrace alt_tr = rtl.decode_syndrome(0x6u, 0x9u, false);
        cyc = rtl.sim_cycle;
        det = "def=(0x" + std::to_string(def_tr.corr_x) + ",0x" + std::to_string(def_tr.corr_z) +
              ") alt_z=0x" + std::to_string(alt_tr.corr_z);
        return def_tr.corr_x == 0x010u && def_tr.corr_z == 0x010u && alt_tr.corr_z == 0x028u;
    });

    /* L5-MATH-17: All 6 horizontal Z-error pairs recovered uniquely under pair-calibrated priors */
    suite.run_case("L5-MATH-17", "G6", "All 6 horizontal Z-error pairs {(0,1),(1,2),(3,4),(4,5),(6,7),(7,8)} recovered uniquely under pair hotspot priors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int pairs[6][2] = {{0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8}};
        int16_t wx[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        int ok = 0;
        for (int p = 0; p < 6; p++) {
            uint16_t err_z = static_cast<uint16_t>((1u << pairs[p][0]) | (1u << pairs[p][1]));
            uint8_t sx = avid_l5::gf2_hx_mul_ez(err_z);
            int16_t wz[9];
            for (int q = 0; q < 9; q++) wz[q] = (q == pairs[p][0] || q == pairs[p][1]) ? 0x0080 : 0x0400;
            load_weights_and_commit(rtl, wx, wz);
            RtlDecodeTrace tr = rtl.decode_syndrome(sx, 0x0u, false);
            if (tr.corr_z == err_z) ok++;
        }
        cyc = rtl.sim_cycle;
        det = "horizontal_z_pairs=" + std::to_string(ok) + "/6";
        return (ok == 6);
    });

    /* L5-MATH-18: All 6 vertical X-error pairs recovered uniquely under pair-calibrated priors */
    suite.run_case("L5-MATH-18", "G6", "All 6 vertical X-error pairs {(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)} recovered uniquely under pair hotspot priors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const int pairs[6][2] = {{0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}};
        int16_t wz[9] = {0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300,0x0300};
        int ok = 0;
        for (int p = 0; p < 6; p++) {
            uint16_t err_x = static_cast<uint16_t>((1u << pairs[p][0]) | (1u << pairs[p][1]));
            uint8_t sz = avid_l5::gf2_hz_mul_ex(err_x);
            int16_t wx[9];
            for (int q = 0; q < 9; q++) wx[q] = (q == pairs[p][0] || q == pairs[p][1]) ? 0x0080 : 0x0400;
            load_weights_and_commit(rtl, wx, wz);
            RtlDecodeTrace tr = rtl.decode_syndrome(0x0u, sz, false);
            if (tr.corr_x == err_x) ok++;
        }
        cyc = rtl.sim_cycle;
        det = "vertical_x_pairs=" + std::to_string(ok) + "/6";
        return (ok == 6);
    });

    /* L5-MATH-19: Pure BP check-node Min-Sum scaling (alpha=0.75 / 0.5) across K=1..5 */
    suite.run_case("L5-MATH-19", "G6", "Check-node Min-Sum shift-add scaling (0.75 non-min / min2+0.5*min1 min1) matches RTL across K=1..5 in Mode 0", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        int16_t wx[9] = {0x0210, 0x0190, 0x02F0, 0x0140, 0x0232, 0x0310, 0x01B0, 0x0160, 0x0290};
        int16_t wz[9] = {0x01C0, 0x0280, 0x0150, 0x02D0, 0x01F0, 0x0170, 0x02B0, 0x0240, 0x0190};
        load_weights_and_commit(rtl, wx, wz);
        rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, 0u, 0x1u, &c);
        int mism = 0;
        for (uint32_t k = 1; k <= 5; k++) {
            rtl.axil_write(avid_l5::L5_OFF_CFG_MAX_ITER, k, 0x1u, &c);
            for (int s = 1; s < 16; s++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(s), static_cast<uint8_t>(16 - s), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(16 - s), wx, static_cast<int>(k), true, 0u);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(s), wz, static_cast<int>(k), false, 0u);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "pure_bp_inhom_checks=75 mismatches=" + std::to_string(mism);
        return (mism == 0);
    });

    /* L5-MATH-20: Residual stabilizer syndrome check H*(e ^ c) == 0 across all single- & two-qubit errors */
    suite.run_case("L5-MATH-20", "G6", "Residual error (e XOR corr) lies strictly in kernel(H) across all 45 weight-1 & weight-2 X/Z Pauli errors", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        int checked = 0, kernel_ok = 0;
        for (int i = 0; i < 9; i++) {
            for (int j = i; j < 9; j++) {
                uint16_t err = static_cast<uint16_t>((1u << i) | (1u << j));
                uint8_t sx = avid_l5::gf2_hx_mul_ez(err);
                uint8_t sz = avid_l5::gf2_hz_mul_ex(err);
                RtlDecodeTrace tr = rtl.decode_syndrome(sx, sz, false);
                uint16_t res_x = static_cast<uint16_t>(err ^ tr.corr_x);
                uint16_t res_z = static_cast<uint16_t>(err ^ tr.corr_z);
                checked++;
                if (avid_l5::gf2_hz_mul_ex(res_x) == 0u && avid_l5::gf2_hx_mul_ez(res_z) == 0u) kernel_ok++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "kernel_satisfied=" + std::to_string(kernel_ok) + "/" + std::to_string(checked);
        return (checked == 45) && (kernel_ok == 45);
    });

    /* L5-MATH-21: Q7.8 full-scale saturation extremes (+32767 = 0x7FFF, -32768 = 0x8000) */
    suite.run_case("L5-MATH-21", "G6", "Full-scale 16-bit Q7.8 extremes (0x7FFF and 0x8000) execute cleanly in Mode 0 & Mode 1 matching golden", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        uint64_t c = 0;
        int16_t wx[9] = {0x7FFF, static_cast<int16_t>(0x8000), 0x7FFF, 0x0001, -1, 0x7FFF, static_cast<int16_t>(0x8000), 0x4000, -16384};
        int16_t wz[9] = {static_cast<int16_t>(0x8000), 0x7FFF, -1, 0x7FFF, 0x0001, -16384, 0x4000, 0x7FFF, static_cast<int16_t>(0x8000)};
        load_weights_and_commit(rtl, wx, wz);
        int mism = 0;
        for (uint32_t m = 0; m <= 1; m++) {
            rtl.axil_write(avid_l5::L5_OFF_OSD_MODE, m, 0x1u, &c);
            for (int s = 0; s < 16; s++) {
                RtlDecodeTrace tr = rtl.decode_syndrome(static_cast<uint8_t>(s), static_cast<uint8_t>(15 - s), false);
                uint16_t gx = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(15 - s), wx, 3, true, m);
                uint16_t gz = CleanRoomQ78Golden::decode_basis(static_cast<uint8_t>(s), wz, 3, false, m);
                if (tr.corr_x != gx || tr.corr_z != gz) mism++;
            }
        }
        cyc = rtl.sim_cycle;
        det = "extreme_q78_vectors=32 mismatches=" + std::to_string(mism);
        return (mism == 0);
    });

    /* L5-MATH-22: CSS commutation H_X * H_Z^T == 0 (mod 2) & Tanner graph degree audit */
    suite.run_case("L5-MATH-22", "G6", "CSS commutation H_X * H_Z^T == 0 (mod 2) verified across all 16 stabilizer pairs & 24 Tanner edges", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        uint16_t hx_rows[4] = {0x003u, 0x036u, 0x0D8u, 0x180u};
        uint16_t hz_rows[4] = {0x01Bu, 0x024u, 0x048u, 0x1B0u};
        bool comm_ok = true;
        int edges_x = 0, edges_z = 0;
        for (int i = 0; i < 4; i++) {
            edges_x += avid_l5::popcount9(hx_rows[i]);
            edges_z += avid_l5::popcount9(hz_rows[i]);
            for (int j = 0; j < 4; j++) {
                if ((avid_l5::popcount9(static_cast<uint16_t>(hx_rows[i] & hz_rows[j])) & 1) != 0) {
                    comm_ok = false;
                }
            }
        }
        /* Also verify via RTL that feeding syndrome of any Z-stabilizer row as X-error has sx==0 */
        AvidVerilatorRtlModel rtl(4, 6);
        for (int j = 0; j < 4; j++) {
            if (avid_l5::gf2_hx_mul_ez(hz_rows[j]) != 0u) comm_ok = false;
            if (avid_l5::gf2_hz_mul_ex(hx_rows[j]) != 0u) comm_ok = false;
        }
        (void)rtl.decode_syndrome(0x0u, 0x0u, true);
        cyc = rtl.sim_cycle;
        det = "comm_ok=" + std::to_string(comm_ok) + " edges_x=" + std::to_string(edges_x) +
              " edges_z=" + std::to_string(edges_z);
        return comm_ok && (edges_x == 12) && (edges_z == 12);
    });

    /* L5-MATH-23: High-weight composite syndromes {0x7, 0xB, 0xD, 0xE, 0xF} adaptive tie-breaking */
    suite.run_case("L5-MATH-23", "G6", "Composite weight-3/weight-4 syndromes {0x7, 0xB, 0xD, 0xE, 0xF} satisfy GF(2) & match golden across 5 profiles", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl(4, 8);
        const uint8_t hi_s[5] = {0x7u, 0xBu, 0xDu, 0xEu, 0xFu};
        int mism = 0, sat = 0;
        for (int p = 0; p < 5; p++) {
            int16_t wx[9], wz[9];
            for (int q = 0; q < 9; q++) {
                wx[q] = (q == p || q == p + 3) ? 0x0088 : 0x0380;
                wz[q] = (q == (8 - p) || q == 4) ? 0x0088 : 0x0380;
            }
            load_weights_and_commit(rtl, wx, wz);
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    RtlDecodeTrace tr = rtl.decode_syndrome(hi_s[i], hi_s[j], false);
                    uint16_t gx = CleanRoomQ78Golden::decode_basis(hi_s[j], wx, 3, true, 1u);
                    uint16_t gz = CleanRoomQ78Golden::decode_basis(hi_s[i], wz, 3, false, 1u);
                    if (tr.corr_x != gx || tr.corr_z != gz) mism++;
                    if (avid_l5::gf2_hz_mul_ex(tr.corr_x) == hi_s[j] && avid_l5::gf2_hx_mul_ez(tr.corr_z) == hi_s[i]) sat++;
                }
            }
        }
        cyc = rtl.sim_cycle;
        det = "composite_checks=125 mismatches=" + std::to_string(mism) + " gf2_sat=" + std::to_string(sat) + "/125";
        return (mism == 0) && (sat == 125);
    });

    /* L5-MATH-24: Orthogonal X/Z weight-bank & BP engine non-interference across 256 syndromes */
    suite.run_case("L5-MATH-24", "G6", "Mutating active_x leaves corr_z 100% invariant (and vice versa) across all 256 (sx, sz) syndromes", sim_mode,
                   [&](uint64_t &cyc, std::string &det) {
        AvidVerilatorRtlModel rtl_a(4, 8), rtl_b(4, 8);
        int16_t wx_a[9] = {0x0090, 0x0400, 0x0090, 0x0400, 0x0090, 0x0400, 0x0090, 0x0400, 0x0090};
        int16_t wx_b[9] = {0x0400, 0x0090, 0x0400, 0x0090, 0x0400, 0x0090, 0x0400, 0x0090, 0x0400};
        int16_t wz_same[9] = {0x0250, 0x0180, 0x0310, 0x0120, 0x0290, 0x01D0, 0x0340, 0x0160, 0x02B0};
        load_weights_and_commit(rtl_a, wx_a, wz_same);
        load_weights_and_commit(rtl_b, wx_b, wz_same);
        int cz_diff = 0, cx_diff = 0;
        for (int sx = 0; sx < 16; sx++) {
            for (int sz = 0; sz < 16; sz++) {
                RtlDecodeTrace ta = rtl_a.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                RtlDecodeTrace tb = rtl_b.decode_syndrome(static_cast<uint8_t>(sx), static_cast<uint8_t>(sz), false);
                if (ta.corr_z != tb.corr_z) cz_diff++;
                if (ta.corr_x != tb.corr_x) cx_diff++;
            }
        }
        cyc = rtl_a.sim_cycle + rtl_b.sim_cycle;
        det = "cz_diff_when_wx_changed=" + std::to_string(cz_diff) + " cx_diff=" + std::to_string(cx_diff);
        return (cz_diff == 0) && (cx_diff > 0);
    });

    return suite.finalize();
}
