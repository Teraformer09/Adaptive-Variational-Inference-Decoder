/* ===========================================================================
 * physical_backend.cpp — Physical MMIO Register Backend (Reserved for V2)
 * =========================================================================*/
#include "avid_hal.h"
#include "../driver/weight_bank_driver.h"
#include "../custom_engine/cycle_accurate_bp_engine.h"
#include <cstring>

int avid_hal_open_physical(avid_hal_t *hal, uint32_t phys_base_addr) {
    if (!hal) return -1;
    hal->sock_fd = -1;
    std::memset(&hal->stats, 0, sizeof(hal->stats));
    return wrd_open(&hal->phys_wrd, phys_base_addr);
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
    uint32_t mask = 0;
    for (int b = 0; b < 4; b++) {
        if ((byte_enable >> b) & 1u) mask |= (0xFFu << (8 * b));
    }
    uint32_t idx = (offset >> 2) & 0x1Fu;
    if (hal->phys_wrd.is_emulated) {
        hal->phys_wrd.sim_regs[idx] = (hal->phys_wrd.sim_regs[idx] & ~mask) | (data & mask);
        if (offset == 0x048u && (data & 1u) && (byte_enable & 1u)) {
            wrd_commit(&hal->phys_wrd);
        }
    } else if (hal->phys_wrd.mapped_base) {
        volatile uint32_t *reg = (volatile uint32_t *)((uint8_t *)hal->phys_wrd.mapped_base + offset);
        *reg = (*reg & ~mask) | (data & mask);
    }
    hal->stats.total_transactions++;
    hal->stats.mmio_writes++;
    hal->stats.last_accepted_cycle += 4;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}

int avid_hal_mmio_read_physical(avid_hal_t *hal,
                                uint32_t offset,
                                uint32_t *rdata,
                                uint64_t *accepted_cycle) {
    if (!hal || !rdata) return -1;
    uint32_t idx = (offset >> 2) & 0x1Fu;
    if (offset >= 0x100u && offset < 0x124u) {
        int q = (int)((offset - 0x100u) >> 2);
        *rdata = (uint16_t)wrd_read_active_weight(&hal->phys_wrd, q);
    } else if (offset >= 0x124u && offset < 0x148u) {
        int q = (int)((offset - 0x124u) >> 2);
        *rdata = (uint16_t)wrd_read_active_weight(&hal->phys_wrd, q);
    } else if (offset == 0x050u) {
        *rdata = wrd_read_epoch(&hal->phys_wrd);
    } else if (hal->phys_wrd.is_emulated) {
        *rdata = hal->phys_wrd.sim_regs[idx];
    } else if (hal->phys_wrd.mapped_base) {
        volatile uint32_t *reg = (volatile uint32_t *)((const uint8_t *)hal->phys_wrd.mapped_base + offset);
        *rdata = *reg;
    } else {
        *rdata = 0xDEADBEEFu;
    }
    hal->stats.total_transactions++;
    hal->stats.mmio_reads++;
    hal->stats.last_accepted_cycle += 4;
    if (accepted_cycle) *accepted_cycle = hal->stats.last_accepted_cycle;
    return 0;
}
