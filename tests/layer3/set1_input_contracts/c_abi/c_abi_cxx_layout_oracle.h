#ifndef AVID_LAYER3_C_ABI_CXX_LAYOUT_ORACLE_H
#define AVID_LAYER3_C_ABI_CXX_LAYOUT_ORACLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t size;
    size_t align;
    size_t off0;
    size_t off1;
    size_t off2;
    size_t off3;
    size_t off4;
} c_abi_struct_metrics_t;

void c_abi_cxx_get_packet_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_validator_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_dma_slot_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_dma_ring_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_syndrome_round_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_ring_buffer_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_scheduler_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_burst_detector_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_fallback_fsm_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_wrd_handle_layout(c_abi_struct_metrics_t *out);
void c_abi_cxx_get_dma_handle_layout(c_abi_struct_metrics_t *out);

/* Retains C++-compiled production symbols in binary so dlsym() can inspect mangled vs unmangled names */
void c_abi_cxx_retain_prod_symbols(void);

#ifdef __cplusplus
}
#endif

#endif /* AVID_LAYER3_C_ABI_CXX_LAYOUT_ORACLE_H */
