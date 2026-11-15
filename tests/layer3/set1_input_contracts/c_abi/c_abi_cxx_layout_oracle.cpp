#include "c_abi_cxx_layout_oracle.h"
#include "ingress/packet_validator.h"
#include "ingress/dma_ingress_tap.h"
#include "inference/ring_buffer.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "driver/weight_reg_driver.h"
#include "driver/dma_manager.h"
#include <cstddef>

extern "C" {

void c_abi_cxx_get_packet_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_syndrome_packet_t);
    out->align = alignof(avid_syndrome_packet_t);
    out->off0  = offsetof(avid_syndrome_packet_t, s_x);
    out->off1  = offsetof(avid_syndrome_packet_t, s_z);
    out->off2  = offsetof(avid_syndrome_packet_t, flags);
    out->off3  = offsetof(avid_syndrome_packet_t, epoch_id);
    out->off4  = offsetof(avid_syndrome_packet_t, seq_id);
}

void c_abi_cxx_get_validator_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_packet_validator_t);
    out->align = alignof(avid_packet_validator_t);
    out->off0  = offsetof(avid_packet_validator_t, last_seq_id);
    out->off1  = offsetof(avid_packet_validator_t, has_seen_first);
    out->off2  = offsetof(avid_packet_validator_t, valid_count);
    out->off3  = offsetof(avid_packet_validator_t, seq_gap_count);
    out->off4  = offsetof(avid_packet_validator_t, invalid_flag_count);
}

void c_abi_cxx_get_dma_slot_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_dma_slot_t);
    out->align = alignof(avid_dma_slot_t);
    out->off0  = offsetof(avid_dma_slot_t, packets);
    out->off1  = offsetof(avid_dma_slot_t, count);
    out->off2  = offsetof(avid_dma_slot_t, state);
    out->off3  = 0;
    out->off4  = 0;
}

void c_abi_cxx_get_dma_ring_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_dma_ring_t);
    out->align = alignof(avid_dma_ring_t);
    out->off0  = offsetof(avid_dma_ring_t, slots);
    out->off1  = offsetof(avid_dma_ring_t, write_slot_idx);
    out->off2  = offsetof(avid_dma_ring_t, read_slot_idx);
    out->off3  = offsetof(avid_dma_ring_t, overflow_drops);
    out->off4  = offsetof(avid_dma_ring_t, validator);
}

void c_abi_cxx_get_syndrome_round_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(syndrome_round_t);
    out->align = alignof(syndrome_round_t);
    out->off0  = offsetof(syndrome_round_t, x);
    out->off1  = offsetof(syndrome_round_t, z);
    out->off2  = 0;
    out->off3  = 0;
    out->off4  = 0;
}

void c_abi_cxx_get_ring_buffer_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(ring_buffer_t);
    out->align = alignof(ring_buffer_t);
    out->off0  = offsetof(ring_buffer_t, buf);
    out->off1  = offsetof(ring_buffer_t, head);
    out->off2  = offsetof(ring_buffer_t, count);
    out->off3  = 0;
    out->off4  = 0;
}

void c_abi_cxx_get_scheduler_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_causal_scheduler_t);
    out->align = alignof(avid_causal_scheduler_t);
    out->off0  = offsetof(avid_causal_scheduler_t, current_round);
    out->off1  = offsetof(avid_causal_scheduler_t, commit_delay_rounds);
    out->off2  = offsetof(avid_causal_scheduler_t, has_pending_commit);
    out->off3  = offsetof(avid_causal_scheduler_t, pending_weights_q78);
    out->off4  = offsetof(avid_causal_scheduler_t, active_epoch_id);
}

void c_abi_cxx_get_burst_detector_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_fast_burst_detector_t);
    out->align = alignof(avid_fast_burst_detector_t);
    out->off0  = offsetof(avid_fast_burst_detector_t, subwindow_rounds);
    out->off1  = offsetof(avid_fast_burst_detector_t, density_threshold);
    out->off2  = offsetof(avid_fast_burst_detector_t, debounce_rounds);
    out->off3  = offsetof(avid_fast_burst_detector_t, peak_density);
    out->off4  = offsetof(avid_fast_burst_detector_t, burst_active);
}

void c_abi_cxx_get_fallback_fsm_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(avid_fallback_fsm_t);
    out->align = alignof(avid_fallback_fsm_t);
    out->off0  = offsetof(avid_fallback_fsm_t, state);
    out->off1  = offsetof(avid_fallback_fsm_t, consecutive_holds);
    out->off2  = offsetof(avid_fallback_fsm_t, last_good_q78);
    out->off3  = offsetof(avid_fallback_fsm_t, static_prior_q78);
    out->off4  = offsetof(avid_fallback_fsm_t, fallback_triggers);
}

void c_abi_cxx_get_wrd_handle_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(wrd_handle_t);
    out->align = alignof(wrd_handle_t);
    out->off0  = offsetof(wrd_handle_t, fd);
    out->off1  = offsetof(wrd_handle_t, mapped_base);
    out->off2  = offsetof(wrd_handle_t, base_addr);
    out->off3  = offsetof(wrd_handle_t, sim_regs);
    out->off4  = offsetof(wrd_handle_t, is_emulated);
}

void c_abi_cxx_get_dma_handle_layout(c_abi_struct_metrics_t *out) {
    out->size  = sizeof(dma_handle_t);
    out->align = alignof(dma_handle_t);
    out->off0  = offsetof(dma_handle_t, fd_mem);
    out->off1  = offsetof(dma_handle_t, dma_base);
    out->off2  = offsetof(dma_handle_t, syndrome_buf);
    out->off3  = offsetof(dma_handle_t, dma_base_addr);
    out->off4  = offsetof(dma_handle_t, max_syndrome_bytes);
}

static volatile const void *g_retained_syms[5];

void c_abi_cxx_retain_prod_symbols(void) {
    g_retained_syms[0] = reinterpret_cast<const void *>(&avid_validator_init);
    g_retained_syms[1] = reinterpret_cast<const void *>(&avid_dma_ring_init);
    g_retained_syms[2] = reinterpret_cast<const void *>(&avid_scheduler_init);
    g_retained_syms[3] = reinterpret_cast<const void *>(&avid_fast_detector_init);
    g_retained_syms[4] = reinterpret_cast<const void *>(&avid_fallback_init);
}

} /* extern "C" */
