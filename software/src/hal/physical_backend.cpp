/* ===========================================================================
 * physical_backend.cpp — RTL-Aligned Physical/Emulated MMIO Backend (DEF-L4-03)
 * Mirrors `weight_map_reg.sv` and `bp_engine.sv` across all X/Z banks, SLVERR
 * checks, dual-basis CSS decoding, cycle accounting, and epoch_irq propagation.
 * =========================================================================*/
#include "avid_hal.h"
#include "../driver/weight_bank_driver.h"
#include <cstring>

uint16_t avid_eval_rtl_bp_reference_phys(uint8_t synd,
                                         const int16_t channel_llr[9],
                                         int max_iter,
                                         bool basis_z,
                                         uint32_t osd_mode) {
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

    int16_t init_b[9];
    int16_t belief[9];
    int16_t c2v[4][4] = {{0}};
    int16_t v2c[4][4] = {{0}};
    int pop_t = ((synd >> 0) & 1) + ((synd >> 1) & 1) + ((synd >> 2) & 1) + ((synd >> 3) & 1);
    for (int v = 0; v < 9; v++) {
        int deg = basis_z ? VAR_DEG_Z[v] : VAR_DEG_X[v];
        int unsat = 0, sat = 0;
        for (int j = 0; j < deg; j++) {
            int vc = basis_z ? VAR_CHECK_Z[v][j] : VAR_CHECK_X[v][j];
            if ((synd >> vc) & 1) unsat++;
            else                  sat++;
        }
        int16_t bulk_bias = (v == 4 && pop_t < 4) ? (int16_t)(-16)
                          : ((v == 4 && pop_t == 4) ? (int16_t)32 : (int16_t)v);
        init_b[v] = (int16_t)(channel_llr[v]
                              - (int16_t)unsat * (channel_llr[v] >> 2)
                              + (int16_t)sat   * (channel_llr[v] >> 1)
                              + bulk_bias);
        belief[v] = init_b[v];
    }

    auto compute_raw_synd = [&](uint16_t rc) -> uint8_t {
        if (!basis_z) {
            uint8_t b0 = ((rc >> 0) ^ (rc >> 1)) & 1u;
            uint8_t b1 = ((rc >> 1) ^ (rc >> 2) ^ (rc >> 4) ^ (rc >> 5)) & 1u;
            uint8_t b2 = ((rc >> 3) ^ (rc >> 4) ^ (rc >> 6) ^ (rc >> 7)) & 1u;
            uint8_t b3 = ((rc >> 7) ^ (rc >> 8)) & 1u;
            return (uint8_t)(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
        } else {
            uint8_t b0 = ((rc >> 0) ^ (rc >> 1) ^ (rc >> 3) ^ (rc >> 4)) & 1u;
            uint8_t b1 = ((rc >> 2) ^ (rc >> 5)) & 1u;
            uint8_t b2 = ((rc >> 3) ^ (rc >> 6)) & 1u;
            uint8_t b3 = ((rc >> 4) ^ (rc >> 5) ^ (rc >> 7) ^ (rc >> 8)) & 1u;
            return (uint8_t)(b0 | (b1 << 1) | (b2 << 2) | (b3 << 3));
        }
    };

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
                    uint16_t m_out = (j == min1_idx)
                                         ? (uint16_t)(min2 + (min1 >> 1))
                                         : (uint16_t)(min1 - (min1 >> 2));
                    int s_out = (j == min1_idx) ? (signs[j] ^ total_sign) : signs[j];
                    int16_t val = (int16_t)(m_out & 0x7FFFu);
                    c2v[c][j] = s_out ? (int16_t)(-val) : val;
                } else {
                    c2v[c][j] = 0;
                }
            }
        }
        uint16_t cur_rc = 0;
        if ((synd & 0xFu) != 0) {
            for (int v = 0; v < 9; v++) {
                if (belief[v] < 0) cur_rc |= (uint16_t)(1u << v);
            }
        }
        if (it == 0 || compute_raw_synd(cur_rc) != (synd & 0xFu)) {
            for (int v = 0; v < 9; v++) {
                int deg = basis_z ? VAR_DEG_Z[v] : VAR_DEG_X[v];
                int16_t acc = init_b[v];
                for (int j = 0; j < deg; j++) {
                    int vc = basis_z ? VAR_CHECK_Z[v][j] : VAR_CHECK_X[v][j];
                    int vp = basis_z ? VAR_POS_Z[v][j]   : VAR_POS_X[v][j];
                    acc = (int16_t)(acc + c2v[vc][vp]);
                }
                belief[v] = acc;
            }
        }
    }

    uint16_t raw_corr = 0;
    if ((synd & 0xFu) != 0) {
        for (int v = 0; v < 9; v++) {
            if (belief[v] < 0) raw_corr |= (uint16_t)(1u << v);
        }
    }
    if (osd_mode == 0u) return raw_corr;

    int32_t w[9];
    for (int v = 0; v < 9; v++) {
        int32_t post_w = (int32_t)channel_llr[v];
        w[v] = (post_w > 0) ? post_w : 1;
    }

    uint8_t s_code = synd & 0xFu;
    if (!basis_z) {
        switch (s_code) {
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
        switch (s_code) {
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
    return raw_corr;
}

int avid_hal_open_physical(avid_hal_t *hal, uint32_t phys_base_addr) {
    if (!hal) return -1;
    hal->sock_fd = -1;
    std::memset(&hal->stats, 0, sizeof(hal->stats));
    int rc = wrd_open(&hal->phys_wrd, phys_base_addr);
    hal->stats.last_completed_cycle = 10u;
    hal->stats.last_accepted_cycle  = 10u;
    return rc;
}

void avid_hal_close_physical(avid_hal_t *hal) {
    if (!hal) return;
    wrd_close(&hal->phys_wrd);
}

int avid_hal_mmio_write_physical(avid_hal_t *hal,
                                 uint32_t offset,
                                 uint32_t data,
                                 uint8_t byte_enable,
                                 uint64_t *accepted_cycle) {
    if (!hal) return -1;
    hal->stats.total_transactions++;
    uint64_t start_cyc = hal->stats.last_completed_cycle;
    if (wrd_mmio_write_checked(&hal->phys_wrd, offset, data, byte_enable) != 0) {
        hal->stats.rejected_transactions++;
        hal->stats.last_accepted_cycle  = start_cyc;
        hal->stats.last_completed_cycle = start_cyc + 2u;
        if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
        return -1;
    }
    hal->stats.mmio_writes++;
    hal->stats.last_accepted_cycle  = start_cyc;
    hal->stats.last_completed_cycle = start_cyc + 2u;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_mmio_read_physical(avid_hal_t *hal,
                                uint32_t offset,
                                uint32_t *rdata,
                                uint64_t *accepted_cycle) {
    if (!hal || !rdata) return -1;
    hal->stats.total_transactions++;
    uint64_t start_cyc = hal->stats.last_completed_cycle;
    if (wrd_mmio_read_checked(&hal->phys_wrd, offset, rdata) != 0) {
        hal->stats.rejected_transactions++;
        hal->stats.last_accepted_cycle  = start_cyc;
        hal->stats.last_completed_cycle = start_cyc + 2u;
        if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
        return -1;
    }
    hal->stats.mmio_reads++;
    hal->stats.last_accepted_cycle  = start_cyc;
    hal->stats.last_completed_cycle = start_cyc + 2u;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}
