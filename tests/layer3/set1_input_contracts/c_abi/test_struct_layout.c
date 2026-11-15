#include "c_abi_harness.h"
#include "c_abi_cxx_layout_oracle.h"
#include "ingress/packet_validator.h"
#include "ingress/dma_ingress_tap.h"
#include "inference/ring_buffer.h"
#include "control/causal_scheduler.h"
#include "control/fallback_fsm.h"
#include "driver/weight_reg_driver.h"
#include "driver/dma_manager.h"

static void compare_metrics(c_abi_ctx_t *ctx, const char *name,
                            const c_abi_struct_metrics_t *c11_m,
                            const c_abi_struct_metrics_t *cxx_m) {
    char lbl[128];
    snprintf(lbl, sizeof(lbl), "%s sizeof C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->size, cxx_m->size, lbl);
    snprintf(lbl, sizeof(lbl), "%s alignof C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->align, cxx_m->align, lbl);
    snprintf(lbl, sizeof(lbl), "%s off0 C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->off0, cxx_m->off0, lbl);
    snprintf(lbl, sizeof(lbl), "%s off1 C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->off1, cxx_m->off1, lbl);
    snprintf(lbl, sizeof(lbl), "%s off2 C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->off2, cxx_m->off2, lbl);
    snprintf(lbl, sizeof(lbl), "%s off3 C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->off3, cxx_m->off3, lbl);
    snprintf(lbl, sizeof(lbl), "%s off4 C11==C++17", name);
    C_ABI_CHECK_EQ_ZU(ctx, c11_m->off4, cxx_m->off4, lbl);
}

int main(int argc, char **argv) {
    c_abi_ctx_t ctx;
    c_abi_init(&ctx, argc, argv, "L3-ABI-005", "c_abi_struct_layout_parity",
               "software/src/{ingress,inference,control,driver}/*.h",
               "sizeof, _Alignof, offsetof across 11 public Layer 3 structs");
    snprintf(ctx.description, sizeof(ctx.description),
             "Verify exact sizeof, _Alignof, and member offsetof parity between C11 (-std=c11) and C++17 (-std=c++17) across all 11 shared C structs");

    c_abi_struct_metrics_t cxx;

    c_abi_struct_metrics_t c_pkt = {
        sizeof(avid_syndrome_packet_t), _Alignof(avid_syndrome_packet_t),
        offsetof(avid_syndrome_packet_t, s_x), offsetof(avid_syndrome_packet_t, s_z),
        offsetof(avid_syndrome_packet_t, flags), offsetof(avid_syndrome_packet_t, epoch_id),
        offsetof(avid_syndrome_packet_t, seq_id)
    };
    c_abi_cxx_get_packet_layout(&cxx);
    compare_metrics(&ctx, "avid_syndrome_packet_t", &c_pkt, &cxx);

    c_abi_struct_metrics_t c_val = {
        sizeof(avid_packet_validator_t), _Alignof(avid_packet_validator_t),
        offsetof(avid_packet_validator_t, last_seq_id), offsetof(avid_packet_validator_t, has_seen_first),
        offsetof(avid_packet_validator_t, valid_count), offsetof(avid_packet_validator_t, seq_gap_count),
        offsetof(avid_packet_validator_t, invalid_flag_count)
    };
    c_abi_cxx_get_validator_layout(&cxx);
    compare_metrics(&ctx, "avid_packet_validator_t", &c_val, &cxx);

    c_abi_struct_metrics_t c_slot = {
        sizeof(avid_dma_slot_t), _Alignof(avid_dma_slot_t),
        offsetof(avid_dma_slot_t, packets), offsetof(avid_dma_slot_t, count),
        offsetof(avid_dma_slot_t, state), 0, 0
    };
    c_abi_cxx_get_dma_slot_layout(&cxx);
    compare_metrics(&ctx, "avid_dma_slot_t", &c_slot, &cxx);

    c_abi_struct_metrics_t c_ring = {
        sizeof(avid_dma_ring_t), _Alignof(avid_dma_ring_t),
        offsetof(avid_dma_ring_t, slots), offsetof(avid_dma_ring_t, write_slot_idx),
        offsetof(avid_dma_ring_t, read_slot_idx), offsetof(avid_dma_ring_t, overflow_drops),
        offsetof(avid_dma_ring_t, validator)
    };
    c_abi_cxx_get_dma_ring_layout(&cxx);
    compare_metrics(&ctx, "avid_dma_ring_t", &c_ring, &cxx);

    c_abi_struct_metrics_t c_sr = {
        sizeof(syndrome_round_t), _Alignof(syndrome_round_t),
        offsetof(syndrome_round_t, x), offsetof(syndrome_round_t, z), 0, 0, 0
    };
    c_abi_cxx_get_syndrome_round_layout(&cxx);
    compare_metrics(&ctx, "syndrome_round_t", &c_sr, &cxx);

    c_abi_struct_metrics_t c_rb = {
        sizeof(ring_buffer_t), _Alignof(ring_buffer_t),
        offsetof(ring_buffer_t, buf), offsetof(ring_buffer_t, head),
        offsetof(ring_buffer_t, count), 0, 0
    };
    c_abi_cxx_get_ring_buffer_layout(&cxx);
    compare_metrics(&ctx, "ring_buffer_t", &c_rb, &cxx);

    c_abi_struct_metrics_t c_sched = {
        sizeof(avid_causal_scheduler_t), _Alignof(avid_causal_scheduler_t),
        offsetof(avid_causal_scheduler_t, current_round), offsetof(avid_causal_scheduler_t, commit_delay_rounds),
        offsetof(avid_causal_scheduler_t, has_pending_commit), offsetof(avid_causal_scheduler_t, pending_weights_q78),
        offsetof(avid_causal_scheduler_t, active_epoch_id)
    };
    c_abi_cxx_get_scheduler_layout(&cxx);
    compare_metrics(&ctx, "avid_causal_scheduler_t", &c_sched, &cxx);

    c_abi_struct_metrics_t c_det = {
        sizeof(avid_fast_burst_detector_t), _Alignof(avid_fast_burst_detector_t),
        offsetof(avid_fast_burst_detector_t, subwindow_rounds), offsetof(avid_fast_burst_detector_t, density_threshold),
        offsetof(avid_fast_burst_detector_t, debounce_rounds), offsetof(avid_fast_burst_detector_t, peak_density),
        offsetof(avid_fast_burst_detector_t, burst_active)
    };
    c_abi_cxx_get_burst_detector_layout(&cxx);
    compare_metrics(&ctx, "avid_fast_burst_detector_t", &c_det, &cxx);

    c_abi_struct_metrics_t c_fsm = {
        sizeof(avid_fallback_fsm_t), _Alignof(avid_fallback_fsm_t),
        offsetof(avid_fallback_fsm_t, state), offsetof(avid_fallback_fsm_t, consecutive_holds),
        offsetof(avid_fallback_fsm_t, last_good_q78), offsetof(avid_fallback_fsm_t, static_prior_q78),
        offsetof(avid_fallback_fsm_t, fallback_triggers)
    };
    c_abi_cxx_get_fallback_fsm_layout(&cxx);
    compare_metrics(&ctx, "avid_fallback_fsm_t", &c_fsm, &cxx);

    c_abi_struct_metrics_t c_wrd = {
        sizeof(wrd_handle_t), _Alignof(wrd_handle_t),
        offsetof(wrd_handle_t, fd), offsetof(wrd_handle_t, mapped_base),
        offsetof(wrd_handle_t, base_addr), offsetof(wrd_handle_t, sim_regs),
        offsetof(wrd_handle_t, is_emulated)
    };
    c_abi_cxx_get_wrd_handle_layout(&cxx);
    compare_metrics(&ctx, "wrd_handle_t", &c_wrd, &cxx);

    c_abi_struct_metrics_t c_dma = {
        sizeof(dma_handle_t), _Alignof(dma_handle_t),
        offsetof(dma_handle_t, fd_mem), offsetof(dma_handle_t, dma_base),
        offsetof(dma_handle_t, syndrome_buf), offsetof(dma_handle_t, dma_base_addr),
        offsetof(dma_handle_t, max_syndrome_bytes)
    };
    c_abi_cxx_get_dma_handle_layout(&cxx);
    compare_metrics(&ctx, "dma_handle_t", &c_dma, &cxx);

    snprintf(ctx.expected_summary, sizeof(ctx.expected_summary),
             "77/77 sizeof, alignof, and offsetof checks match identically between C11 and C++17");
    snprintf(ctx.actual_summary, sizeof(ctx.actual_summary),
             "77/77 checks run, %d failed (pkt=%zuB, dma_ring=%zuB, ring_buf=%zuB, sched=%zuB, fsm=%zuB, wrd=%zuB)",
             ctx.checks_failed, sizeof(avid_syndrome_packet_t), sizeof(avid_dma_ring_t),
             sizeof(ring_buffer_t), sizeof(avid_causal_scheduler_t), sizeof(avid_fallback_fsm_t), sizeof(wrd_handle_t));

    return c_abi_finalize(&ctx, (ctx.checks_failed == 0) ? "PASS" : "FAIL", "");
}
