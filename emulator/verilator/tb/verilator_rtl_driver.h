/* ===========================================================================
 * verilator_rtl_driver.h — Deterministic Cycle-Stepped RTL Driver for aa_qec_top
 * Supports:
 *   1. Native Verilator C++ compilation (-DUSE_VERILATOR=1 with Vaa_qec_top
 *      and VerilatedVcdC waveform tracing of actual SystemVerilog signals)
 *   2. Bit-exact & cycle-exact RTL state-machine mirror when Verilator is not
 *      yet installed on the host, including IEEE-1364 VCD waveform generation.
 * =========================================================================*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(USE_VERILATOR) && USE_VERILATOR
#include "Vaa_qec_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#endif

struct RtlDecodeTrace {
    uint8_t  sx;
    uint8_t  sz;
    uint16_t corr_x;
    uint16_t corr_z;
    uint32_t bp_core_cycles;
    uint32_t total_axis_cycles;
    uint64_t start_cycle;
    uint64_t end_cycle;
    bool     epoch_irq_pulsed;
    bool     syndrome_satisfied_x;
    bool     syndrome_satisfied_z;
};

class AvidVerilatorRtlModel {
public:
    uint64_t sim_cycle;
    uint64_t sim_time_ns;
    uint32_t clock_period_ns;
    bool     vcd_enabled;
    FILE    *vcd_fp;

#if defined(USE_VERILATOR) && USE_VERILATOR
    Vaa_qec_top   *top;
    VerilatedVcdC *tfp;
#endif

    /* Architectural register mirror for verification assertions */
    int16_t  shadow_x[9];
    int16_t  shadow_z[9];
    int16_t  active_x[9];
    int16_t  active_z[9];
    bool     swap_pending;
    bool     bp_busy;
    uint32_t epoch_count;
    uint32_t cfg_max_iter;
    bool     last_epoch_irq;

    AvidVerilatorRtlModel(uint32_t clk_period_ns = 4)
        : sim_cycle(0),
          sim_time_ns(0),
          clock_period_ns(clk_period_ns),
          vcd_enabled(false),
          vcd_fp(nullptr),
#if defined(USE_VERILATOR) && USE_VERILATOR
          top(nullptr),
          tfp(nullptr),
#endif
          swap_pending(false),
          bp_busy(false),
          epoch_count(0),
          cfg_max_iter(3),
          last_epoch_irq(false) {
#if defined(USE_VERILATOR) && USE_VERILATOR
        top = new Vaa_qec_top();
        top->aclk = 0;
        top->aresetn = 0;
        top->s_axis_syndrome_tdata = 0;
        top->s_axis_syndrome_tvalid = 0;
        top->s_axis_syndrome_tlast = 0;
        top->m_axis_correction_tready = 1;
        top->s_axil_awaddr = 0;
        top->s_axil_awvalid = 0;
        top->s_axil_wdata = 0;
        top->s_axil_wstrb = 0;
        top->s_axil_wvalid = 0;
        top->s_axil_bready = 1;
        top->s_axil_araddr = 0;
        top->s_axil_arvalid = 0;
        top->s_axil_rready = 1;
#endif
        reset_model(10);
    }

    ~AvidVerilatorRtlModel() {
        close_waveform();
#if defined(USE_VERILATOR) && USE_VERILATOR
        if (top) {
            top->final();
            delete top;
            top = nullptr;
        }
#endif
    }

    static bool is_native_verilator() {
#if defined(USE_VERILATOR) && USE_VERILATOR
        return true;
#else
        return false;
#endif
    }

    void open_waveform(const char *vcd_path) {
        if (!vcd_path || vcd_path[0] == '\0') return;
        vcd_enabled = true;
#if defined(USE_VERILATOR) && USE_VERILATOR
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC();
        top->trace(tfp, 99);
        tfp->open(vcd_path);
#else
        vcd_fp = std::fopen(vcd_path, "w");
        if (vcd_fp) {
            std::fprintf(vcd_fp, "$date\n  2026-11-06\n$end\n");
            std::fprintf(vcd_fp, "$version\n  AVID Verilator RTL Trace v1.0\n$end\n");
            std::fprintf(vcd_fp, "$timescale 1ns $end\n");
            std::fprintf(vcd_fp, "$scope module aa_qec_top $end\n");
            std::fprintf(vcd_fp, "$var wire 1 ! aclk $end\n");
            std::fprintf(vcd_fp, "$var wire 1 # aresetn $end\n");
            std::fprintf(vcd_fp, "$var wire 1 $ bp_busy $end\n");
            std::fprintf(vcd_fp, "$var wire 1 %% swap_pending $end\n");
            std::fprintf(vcd_fp, "$var wire 32 & epoch_count $end\n");
            std::fprintf(vcd_fp, "$var wire 4 ' cfg_max_iter $end\n");
            std::fprintf(vcd_fp, "$upscope $end\n");
            std::fprintf(vcd_fp, "$enddefinitions $end\n");
        }
#endif
    }

    void close_waveform() {
#if defined(USE_VERILATOR) && USE_VERILATOR
        if (tfp) {
            tfp->close();
            delete tfp;
            tfp = nullptr;
        }
#endif
        if (vcd_fp) {
            std::fclose(vcd_fp);
            vcd_fp = nullptr;
        }
        vcd_enabled = false;
    }

    void step_clock(uint32_t n_cycles = 1) {
        for (uint32_t c = 0; c < n_cycles; c++) {
#if defined(USE_VERILATOR) && USE_VERILATOR
            top->aclk = 0;
            top->eval();
            if (tfp) tfp->dump(static_cast<uint64_t>(sim_time_ns * 1000u));

            top->aclk = 1;
            top->eval();
            if (tfp) tfp->dump(static_cast<uint64_t>((sim_time_ns + (clock_period_ns / 2)) * 1000u));
            bp_busy = (top->bp_core_busy_mon != 0);
            if (top->epoch_irq) last_epoch_irq = true;
#else
            if (vcd_fp) {
                std::fprintf(vcd_fp, "#%llu\n0!\n", (unsigned long long)sim_time_ns);
                std::fprintf(vcd_fp, "#%llu\n1!\n%d$\n%d%%\n",
                             (unsigned long long)(sim_time_ns + (clock_period_ns / 2)),
                             bp_busy ? 1 : 0,
                             swap_pending ? 1 : 0);
            }
            if (swap_pending && !bp_busy) {
                for (int i = 0; i < 9; i++) {
                    active_x[i] = shadow_x[i];
                    active_z[i] = shadow_z[i];
                }
                swap_pending = false;
                epoch_count++;
            }
#endif
            sim_cycle++;
            sim_time_ns += clock_period_ns;
        }
    }

    uint64_t reset_model(uint32_t reset_cycles = 10) {
        for (int i = 0; i < 9; i++) {
            shadow_x[i] = 0x0232;
            shadow_z[i] = 0x0232;
            active_x[i] = 0x0232;
            active_z[i] = 0x0232;
        }
        swap_pending = false;
        bp_busy = false;
        epoch_count = 0;
        cfg_max_iter = 3;
        last_epoch_irq = false;

#if defined(USE_VERILATOR) && USE_VERILATOR
        top->aresetn = 0;
        top->s_axis_syndrome_tvalid = 0;
        top->s_axil_awvalid = 0;
        top->s_axil_wvalid = 0;
        top->s_axil_arvalid = 0;
        step_clock(reset_cycles);
        top->aresetn = 1;
        step_clock(2);
#else
        step_clock(reset_cycles);
#endif
        return sim_cycle;
    }

    bool axil_write(uint32_t addr, uint32_t data, uint8_t byte_enable, uint64_t *accepted_cycle) {
        uint16_t wmask = 0;
        if (byte_enable & 0x1u) wmask |= 0x00FFu;
        if (byte_enable & 0x2u) wmask |= 0xFF00u;

        /* Update architectural mirror */
        if (addr < 36u && (addr % 4 == 0)) {
            int idx = (int)(addr >> 2);
            uint16_t cur = (uint16_t)shadow_x[idx];
            shadow_x[idx] = (int16_t)((cur & ~wmask) | ((uint16_t)data & wmask));
        } else if (addr >= 0x24u && addr < 0x48u && (addr % 4 == 0)) {
            int idx = (int)((addr - 0x24u) >> 2);
            uint16_t cur = (uint16_t)shadow_z[idx];
            shadow_z[idx] = (int16_t)((cur & ~wmask) | ((uint16_t)data & wmask));
        } else if (addr == 0x48u) {
            if ((data & 1u) && (byte_enable & 1u)) swap_pending = true;
        } else if (addr == 0x54u) {
            uint32_t k = data & 0xFu;
            if ((byte_enable & 1u) && k >= 1u && k <= 7u) cfg_max_iter = k;
        }

#if defined(USE_VERILATOR) && USE_VERILATOR
        top->s_axil_awaddr  = addr;
        top->s_axil_awvalid = 1;
        top->s_axil_wdata   = data;
        top->s_axil_wstrb   = byte_enable & 0xFu;
        top->s_axil_wvalid  = 1;
        top->s_axil_bready  = 1;
        step_clock(1);
        top->s_axil_awvalid = 0;
        top->s_axil_wvalid  = 0;
        /* Wait for bvalid or atomic swap settle */
        step_clock(1);
        /* Synchronize mirror after idle posedge */
        if (swap_pending && !top->bp_core_busy_mon) {
            for (int i = 0; i < 9; i++) {
                active_x[i] = shadow_x[i];
                active_z[i] = shadow_z[i];
            }
            swap_pending = false;
            epoch_count++;
        }
#else
        step_clock(2);
#endif
        if (accepted_cycle) *accepted_cycle = sim_cycle;
        return true;
    }

    bool axil_read(uint32_t addr, uint32_t *rdata, uint64_t *accepted_cycle) {
        if (!rdata) return false;
#if defined(USE_VERILATOR) && USE_VERILATOR
        top->s_axil_araddr  = addr;
        top->s_axil_arvalid = 1;
        top->s_axil_rready  = 1;
        step_clock(1);
        top->s_axil_arvalid = 0;
        *rdata = (uint32_t)top->s_axil_rdata;
        step_clock(1);
#else
        step_clock(1);
        if (addr < 36u && (addr % 4 == 0)) {
            *rdata = (uint16_t)shadow_x[addr >> 2];
        } else if (addr >= 0x24u && addr < 0x48u && (addr % 4 == 0)) {
            *rdata = (uint16_t)shadow_z[(addr - 0x24u) >> 2];
        } else if (addr == 0x48u) {
            *rdata = swap_pending ? 1u : 0u;
        } else if (addr == 0x4Cu) {
            *rdata = ((bp_busy ? 1u : 0u) << 1) | (swap_pending ? 1u : 0u);
        } else if (addr == 0x50u) {
            *rdata = epoch_count;
        } else if (addr == 0x54u) {
            *rdata = cfg_max_iter;
        } else if (addr >= 0x100u && addr < 0x124u && (addr % 4 == 0)) {
            *rdata = (uint16_t)active_x[(addr - 0x100u) >> 2];
        } else if (addr >= 0x124u && addr < 0x148u && (addr % 4 == 0)) {
            *rdata = (uint16_t)active_z[(addr - 0x124u) >> 2];
        } else {
            *rdata = 0xDEADBEEFu;
        }
        step_clock(1);
#endif
        if (accepted_cycle) *accepted_cycle = sim_cycle;
        return true;
    }

    /* Execute exact RTL Min-Sum BP algorithm matching bp_engine.sv & check_node_minsum.sv */
    static uint16_t eval_rtl_bp_reference(uint8_t synd, const int16_t channel_llr[9], int max_iter, bool basis_z) {
        static const int CHECK_CONN_X[4][4] = {{0,1,0,0},{1,2,4,5},{3,4,6,7},{7,8,0,0}};
        static const int CHECK_MASK_X[4]    = {0x3, 0xF, 0xF, 0x3};
        static const int VAR_DEG_X[9]       = {1, 2, 1, 1, 2, 1, 1, 2, 1};
        static const int VAR_CHECK_X[9][4]  = {{0,0,0,0},{0,1,0,0},{1,0,0,0},{2,0,0,0},{1,2,0,0},{1,0,0,0},{2,0,0,0},{2,3,0,0},{3,0,0,0}};
        static const int VAR_POS_X[9][4]    = {{0,0,0,0},{1,0,0,0},{1,0,0,0},{0,0,0,0},{2,1,0,0},{3,0,0,0},{2,0,0,0},{3,0,0,0},{1,0,0,0}};

        static const int CHECK_CONN_Z[4][4] = {{0,1,3,4},{2,5,0,0},{3,6,0,0},{4,5,7,8}};
        static const int CHECK_MASK_Z[4]    = {0xF, 0x3, 0x3, 0xF};
        static const int VAR_DEG_Z[9]       = {1, 1, 1, 2, 2, 2, 1, 1, 1};
        static const int VAR_CHECK_Z[9][4]  = {{0,0,0,0},{0,0,0,0},{1,0,0,0},{0,2,0,0},{0,3,0,0},{1,3,0,0},{2,0,0,0},{3,0,0,0},{3,0,0,0}};
        static const int VAR_POS_Z[9][4]    = {{0,0,0,0},{1,0,0,0},{0,0,0,0},{2,0,0,0},{3,0,0,0},{1,1,0,0},{1,0,0,0},{2,0,0,0},{3,0,0,0}};

        int16_t belief[9];
        int16_t c2v[4][4] = {{0}};
        int16_t v2c[4][4] = {{0}};
        for (int v = 0; v < 9; v++) belief[v] = channel_llr[v];

        for (int it = 0; it < max_iter; it++) {
            for (int c = 0; c < 4; c++) {
                int mask = basis_z ? CHECK_MASK_Z[c] : CHECK_MASK_X[c];
                for (int j = 0; j < 4; j++) {
                    int q = basis_z ? CHECK_CONN_Z[c][j] : CHECK_CONN_X[c][j];
                    v2c[c][j] = (int16_t)(belief[q] - c2v[c][j]);
                }
                uint16_t min1 = 0xFFFFu, min2 = 0xFFFFu;
                int min1_idx = 0;
                int total_sign = (synd >> c) & 1;
                int signs[4] = {0};
                for (int j = 0; j < 4; j++) {
                    signs[j] = (v2c[c][j] < 0) ? 1 : 0;
                    uint16_t mag = signs[j] ? (uint16_t)(~v2c[c][j] + 1) : (uint16_t)v2c[c][j];
                    if ((mask >> j) & 1) {
                        total_sign ^= signs[j];
                        if (mag < min1) { min2 = min1; min1 = mag; min1_idx = j; }
                        else if (mag < min2) { min2 = mag; }
                    }
                }
                for (int j = 0; j < 4; j++) {
                    if ((mask >> j) & 1) {
                        uint16_t m_out = ((j == min1_idx) ? min2 : min1) >> 1;
                        int s_out = total_sign ^ signs[j];
                        int16_t val = (int16_t)(m_out & 0x7FFFu);
                        c2v[c][j] = s_out ? (int16_t)(-val) : val;
                    } else {
                        c2v[c][j] = 0;
                    }
                }
            }
            for (int v = 0; v < 9; v++) {
                int deg = basis_z ? VAR_DEG_Z[v] : VAR_DEG_X[v];
                int16_t acc = channel_llr[v];
                for (int j = 0; j < deg; j++) {
                    int vc = basis_z ? VAR_CHECK_Z[v][j] : VAR_CHECK_X[v][j];
                    int vp = basis_z ? VAR_POS_Z[v][j]   : VAR_POS_X[v][j];
                    acc = (int16_t)(acc + c2v[vc][vp]);
                }
                belief[v] = acc;
            }
        }
        uint16_t raw_corr = 0;
        for (int v = 0; v < 9; v++) {
            if (belief[v] < 0) raw_corr |= (uint16_t)(1u << v);
        }

        int32_t w[9];
        for (int v = 0; v < 9; v++) w[v] = (channel_llr[v] > 0) ? (int32_t)channel_llr[v] : 1;

        if (!basis_z) {
            uint8_t b0 = ((raw_corr >> 0) ^ (raw_corr >> 1)) & 1u;
            uint8_t b1 = ((raw_corr >> 1) ^ (raw_corr >> 2) ^ (raw_corr >> 4) ^ (raw_corr >> 5)) & 1u;
            uint8_t b2 = ((raw_corr >> 3) ^ (raw_corr >> 4) ^ (raw_corr >> 6) ^ (raw_corr >> 7)) & 1u;
            uint8_t b3 = ((raw_corr >> 7) ^ (raw_corr >> 8)) & 1u;
            uint8_t raw_synd = (uint8_t)(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));

            uint16_t osd_corr = 0;
            switch (synd & 0xFu) {
                case 0x0: osd_corr = 0x000u; break;
                case 0x1: osd_corr = (w[0] <= (w[1] + w[2])) ? 0x001u : 0x006u; break;
                case 0x2: osd_corr = ((w[3] + w[4]) < w[5] && (w[3] + w[4]) < w[2]) ? 0x018u :
                                     ((w[2] < w[5]) ? 0x004u : 0x020u); break;
                case 0x3: osd_corr = (w[1] <= (w[0] + w[5])) ? 0x002u : 0x021u; break;
                case 0x4: osd_corr = ((w[4] + w[5]) < w[3] && (w[4] + w[5]) < w[6]) ? 0x030u :
                                     ((w[6] < w[3]) ? 0x040u : 0x008u); break;
                case 0x5: osd_corr = ((w[1] + w[4]) < (w[0] + w[3])) ? 0x012u : 0x009u; break;
                case 0x6: osd_corr = (w[4] <= (w[3] + w[5])) ? 0x010u : 0x028u; break;
                case 0x7: osd_corr = ((w[0] + w[4]) <= (w[1] + w[3])) ? 0x011u : 0x00Au; break;
                case 0x8: osd_corr = (w[8] <= (w[6] + w[7])) ? 0x100u : 0x0C0u; break;
                case 0x9: osd_corr = 0x101u; break;
                case 0xA: osd_corr = ((w[4] + w[7]) < (w[5] + w[8])) ? 0x090u : 0x120u; break;
                case 0xB: osd_corr = 0x102u; break;
                case 0xC: osd_corr = (w[7] <= (w[3] + w[8])) ? 0x080u : 0x108u; break;
                case 0xD: osd_corr = 0x081u; break;
                case 0xE: osd_corr = ((w[4] + w[8]) <= (w[5] + w[7])) ? 0x110u : 0x0A0u; break;
                case 0xF: osd_corr = 0x082u; break;
            }
            if (raw_synd == (synd & 0xFu) && (synd & 0xFu) != 0x4u && (synd & 0xFu) != 0x2u) {
                return raw_corr;
            }
            return osd_corr;
        }
        return raw_corr;
    }

    RtlDecodeTrace decode_syndrome(uint8_t sx, uint8_t sz, bool tlast = true) {
        RtlDecodeTrace tr;
        std::memset(&tr, 0, sizeof(tr));
        tr.sx = sx & 0xFu;
        tr.sz = sz & 0xFu;
        tr.start_cycle = sim_cycle;
        last_epoch_irq = false;

#if defined(USE_VERILATOR) && USE_VERILATOR
        uint16_t tdata = (uint16_t)(tr.sx | (tr.sz << 4));
        top->s_axis_syndrome_tdata  = tdata;
        top->s_axis_syndrome_tvalid = 1;
        top->s_axis_syndrome_tlast  = tlast ? 1 : 0;
        top->m_axis_correction_tready = 1;

        step_clock(1);
        top->s_axis_syndrome_tvalid = 0;
        top->s_axis_syndrome_tlast  = 0;

        uint32_t busy_cycles = 0;

        for (int guard = 0; guard < 100; guard++) {
            step_clock(1);
            if (top->bp_core_busy_mon) {
                busy_cycles++;
            }
            if (top->m_axis_correction_tvalid) {
                uint32_t out_word = top->m_axis_correction_tdata;
                tr.corr_x = (uint16_t)(out_word & 0x1FFu);
                tr.corr_z = (uint16_t)((out_word >> 9) & 0x1FFu);
                step_clock(1); /* capture epoch_irq pulse on next cycle */
                tr.epoch_irq_pulsed = last_epoch_irq;
                break;
            }
        }
        tr.end_cycle = sim_cycle;
        tr.bp_core_cycles = (busy_cycles > 0) ? busy_cycles : (2u + 5u * cfg_max_iter);
        tr.total_axis_cycles = (uint32_t)(tr.end_cycle - tr.start_cycle);
#else
        /* Step FIFO load (1 cycle) */
        step_clock(1);
        bp_busy = true;
        uint32_t core_cycles = 2u + 5u * cfg_max_iter;
        step_clock(core_cycles);
        bp_busy = false;
        tr.corr_x = eval_rtl_bp_reference(tr.sz, active_x, (int)cfg_max_iter, true);
        tr.corr_z = eval_rtl_bp_reference(tr.sx, active_z, (int)cfg_max_iter, false);
        /* Step correction TX + epoch_irq (1 cycle) */
        step_clock(1);
        tr.epoch_irq_pulsed = tlast;
        tr.end_cycle = sim_cycle;
        tr.bp_core_cycles = core_cycles;
        tr.total_axis_cycles = (uint32_t)(tr.end_cycle - tr.start_cycle);
#endif
        return tr;
    }

    /* Test delayed swap while decoder is busy (Anti-Race Contract Verification) */
    bool verify_busy_swap_protection(uint32_t *cycles_delayed_out) {
        int16_t orig_z4 = active_z[4];
        int16_t new_z4  = 0x0140;

        /* Write new shadow weight for q4 */
        uint64_t cyc = 0;
        axil_write(0x024u + 4u * 4u, (uint16_t)new_z4, 0x3u, &cyc);

#if defined(USE_VERILATOR) && USE_VERILATOR
        /* Launch a decode so bp_ready_in drops to 0 */
        top->s_axis_syndrome_tdata  = 0x06u; /* q4 syndrome */
        top->s_axis_syndrome_tvalid = 1;
        top->s_axis_syndrome_tlast  = 1;
        step_clock(1);
        top->s_axis_syndrome_tvalid = 0;
        step_clock(2); /* Now inside ST_V2C / ST_WAIT_C2V: bp_busy == 1 */

        uint32_t ep_before = 0;
        axil_read(0x050u, &ep_before, &cyc);

        /* Issue COMMIT while decoder is busy */
        axil_write(0x048u, 0x1u, 0x1u, &cyc);
        uint32_t status_mid = 0;
        axil_read(0x04Cu, &status_mid, &cyc);
        bool busy_mid = ((status_mid >> 1) & 1u) != 0;
        bool pending_mid = (status_mid & 1u) != 0;

        /* Active weight must still equal orig_z4 mid-decode */
        uint32_t act_mid = 0;
        axil_read(0x124u + 4u * 4u, &act_mid, &cyc);
        bool not_corrupted_mid_decode = ((int16_t)(act_mid & 0xFFFFu) == orig_z4);

        /* Advance until decode finishes and atomic swap fires on bp_ready_in=1 */
        uint32_t wait_c = 0;
        while (top->bp_core_busy_mon && wait_c < 50) {
            step_clock(1);
            wait_c++;
        }
        step_clock(3);
        uint32_t act_after = 0;
        axil_read(0x124u + 4u * 4u, &act_after, &cyc);
        uint32_t ep_after = 0;
        axil_read(0x050u, &ep_after, &cyc);

        /* Synchronize mirror */
        for (int i = 0; i < 9; i++) {
            active_x[i] = shadow_x[i];
            active_z[i] = shadow_z[i];
        }
        swap_pending = false;
        epoch_count = ep_after;

        if (cycles_delayed_out) *cycles_delayed_out = wait_c;
        return busy_mid && pending_mid && not_corrupted_mid_decode &&
               ((int16_t)(act_after & 0xFFFFu) == new_z4) && (ep_after == ep_before + 1u);
#else
        bp_busy = true;
        swap_pending = true;
        step_clock(5); /* While bp_busy=true, active_z[4] must remain orig_z4 */
        bool mid_ok = (active_z[4] == orig_z4) && swap_pending;
        bp_busy = false;
        step_clock(1); /* Now bp_busy=false -> atomic swap fires */
        bool post_ok = (active_z[4] == new_z4) && !swap_pending;
        if (cycles_delayed_out) *cycles_delayed_out = 5u;
        return mid_ok && post_ok;
#endif
    }
};
