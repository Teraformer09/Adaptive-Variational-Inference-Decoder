add_subdirectory(c_abi)

# Helper function to compile a C++ Set 1 test executable
function(add_layer3_set1_exe tgt_name src_rel)
    add_executable(${tgt_name} ${src_rel})
    target_link_libraries(${tgt_name} PRIVATE avid_layer3_test_common avid_layer3_prod)
    target_compile_options(${tgt_name} PRIVATE -Wall -Wextra -g)
endfunction()

# 1. Packet contract executables (9 files)
add_layer3_set1_exe(set1_test_packet_valid            packet/test_packet_valid.cpp)
add_layer3_set1_exe(set1_test_packet_invalid          packet/test_packet_invalid.cpp)
add_layer3_set1_exe(set1_test_packet_endianness       packet/test_packet_endianness.cpp)
add_layer3_set1_exe(set1_test_packet_flags            packet/test_packet_flags.cpp)
add_layer3_set1_exe(set1_test_packet_field_decode     packet/test_packet_field_decode.cpp)
add_layer3_set1_exe(set1_test_packet_epoch            packet/test_packet_epoch.cpp)
add_layer3_set1_exe(set1_test_packet_sequence         packet/test_packet_sequence.cpp)
add_layer3_set1_exe(set1_test_packet_no_side_effect   packet/test_packet_no_side_effect.cpp)
add_layer3_set1_exe(set1_test_packet_boundary         packet/test_packet_boundary.cpp)

# 2. DMA & histogram executables (7 files)
add_layer3_set1_exe(set1_test_dma_ingress                 dma_histogram/test_dma_ingress.cpp)
add_layer3_set1_exe(set1_test_histogram_accumulation      dma_histogram/test_histogram_accumulation.cpp)
add_layer3_set1_exe(set1_test_histogram_channel_isolation dma_histogram/test_histogram_channel_isolation.cpp)
add_layer3_set1_exe(set1_test_histogram_snapshot          dma_histogram/test_histogram_snapshot.cpp)
add_layer3_set1_exe(set1_test_histogram_reset             dma_histogram/test_histogram_reset.cpp)
add_layer3_set1_exe(set1_test_histogram_overflow          dma_histogram/test_histogram_overflow.cpp)
add_layer3_set1_exe(set1_test_histogram_invalid_packet    dma_histogram/test_histogram_invalid_packet.cpp)

# 3. Estimator input executables (7 files)
add_layer3_set1_exe(set1_test_parameter_dimensions        estimator_inputs/test_parameter_dimensions.cpp)
add_layer3_set1_exe(set1_test_parameter_ordering          estimator_inputs/test_parameter_ordering.cpp)
add_layer3_set1_exe(set1_test_parameter_bounds            estimator_inputs/test_parameter_bounds.cpp)
add_layer3_set1_exe(set1_test_parameter_nan_inf           estimator_inputs/test_parameter_nan_inf.cpp)
add_layer3_set1_exe(set1_test_histogram_input_validation  estimator_inputs/test_histogram_input_validation.cpp)
add_layer3_set1_exe(set1_test_prior_validation            estimator_inputs/test_prior_validation.cpp)
add_layer3_set1_exe(set1_test_empty_observation           estimator_inputs/test_empty_observation.cpp)

# 4. Runtime configuration executables (6 files)
add_layer3_set1_exe(set1_test_config_valid          runtime_config/test_config_valid.cpp)
add_layer3_set1_exe(set1_test_config_missing        runtime_config/test_config_missing.cpp)
add_layer3_set1_exe(set1_test_config_types          runtime_config/test_config_types.cpp)
add_layer3_set1_exe(set1_test_config_bounds         runtime_config/test_config_bounds.cpp)
add_layer3_set1_exe(set1_test_config_unknown_fields runtime_config/test_config_unknown_fields.cpp)
add_layer3_set1_exe(set1_test_config_consumption    runtime_config/test_config_consumption.cpp)

# 5. Production integration executables (4 files)
add_layer3_set1_exe(set1_test_production_ingress_path    integration/test_production_ingress_path.cpp)
add_layer3_set1_exe(set1_test_daemon_histogram_path      integration/test_daemon_histogram_path.cpp)
add_layer3_set1_exe(set1_test_ring_buffer_connectivity   integration/test_ring_buffer_connectivity.cpp)
add_layer3_set1_exe(set1_test_invalid_input_propagation  integration/test_invalid_input_propagation.cpp)

# Register all 40 canonical Set 1 test IDs (L3-IP-001 .. L3-IP-040)
set(SET1_CANONICAL_TESTS
    "L3-IP-001|set1_test_packet_valid"
    "L3-IP-002|set1_test_packet_invalid"
    "L3-IP-003|set1_test_packet_endianness"
    "L3-IP-004|set1_test_packet_flags"
    "L3-IP-005|set1_test_packet_field_decode"
    "L3-IP-006|set1_test_packet_epoch"
    "L3-IP-007|set1_test_packet_sequence"
    "L3-IP-008|set1_test_packet_no_side_effect"
    "L3-IP-009|set1_test_packet_boundary"
    "L3-IP-010|set1_test_dma_ingress"
    "L3-IP-011|set1_test_histogram_accumulation"
    "L3-IP-012|set1_test_histogram_input_validation"
    "L3-IP-013|set1_test_empty_observation"
    "L3-IP-014|set1_test_histogram_accumulation"
    "L3-IP-015|set1_test_histogram_input_validation"
    "L3-IP-016|set1_test_histogram_input_validation"
    "L3-IP-017|set1_test_histogram_overflow"
    "L3-IP-018|set1_test_histogram_reset"
    "L3-IP-019|set1_test_histogram_snapshot"
    "L3-IP-020|set1_test_histogram_channel_isolation"
    "L3-IP-021|set1_test_parameter_dimensions"
    "L3-IP-022|set1_test_parameter_ordering"
    "L3-IP-023|set1_test_parameter_bounds"
    "L3-IP-024|set1_test_parameter_nan_inf"
    "L3-IP-025|set1_test_prior_validation"
    "L3-IP-026|set1_test_prior_validation"
    "L3-IP-027|set1_test_parameter_bounds"
    "L3-IP-028|set1_test_parameter_nan_inf"
    "L3-IP-029|set1_test_empty_observation"
    "L3-IP-030|set1_test_packet_epoch"
    "L3-IP-031|set1_test_config_consumption"
    "L3-IP-032|set1_test_config_valid"
    "L3-IP-033|set1_test_config_bounds"
    "L3-IP-034|set1_test_config_bounds"
    "L3-IP-035|set1_test_config_types"
    "L3-IP-036|set1_test_config_unknown_fields"
    "L3-IP-037|set1_test_config_consumption"
    "L3-IP-038|set1_test_config_types"
    "L3-IP-039|set1_test_config_valid"
    "L3-IP-040|set1_test_config_bounds"
)

foreach(pair IN LISTS SET1_CANONICAL_TESTS)
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 tid)
    list(GET parts 1 exe_tgt)
    add_test(
        NAME ${tid}
        COMMAND ${exe_tgt}
            --test-id ${tid}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_REPO_ROOT}/validation/layer3
            --daemon-bin $<TARGET_FILE:avid_daemon_under_test>
    )
endforeach()

# Register supplementary Set 1 DMA, config-missing, and production-integration tests
set(SET1_SUPPLEMENTARY_TESTS
    "L3-DMA-INVALID-PKT|set1_test_histogram_invalid_packet"
    "L3-CFG-MISSING|set1_test_config_missing"
    "L3-INT-001|set1_test_production_ingress_path"
    "L3-INT-002|set1_test_daemon_histogram_path"
    "L3-INT-003|set1_test_ring_buffer_connectivity"
    "L3-INT-004|set1_test_invalid_input_propagation"
)

foreach(pair IN LISTS SET1_SUPPLEMENTARY_TESTS)
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 tid)
    list(GET parts 1 exe_tgt)
    add_test(
        NAME ${tid}
        COMMAND ${exe_tgt}
            --test-id ${tid}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_REPO_ROOT}/validation/layer3
            --daemon-bin $<TARGET_FILE:avid_daemon_under_test>
    )
endforeach()
