function(add_layer3_set5_exe tgt_name src_rel)
    add_executable(${tgt_name} ${src_rel})
    target_link_libraries(${tgt_name} PRIVATE avid_layer3_test_common avid_layer3_prod)
    target_compile_options(${tgt_name} PRIVATE -Wall -Wextra -g)
endfunction()

add_layer3_set5_exe(set5_test_dp_vs_bruteforce           likelihood/test_dp_vs_bruteforce.cpp)
add_layer3_set5_exe(set5_test_finite_difference_gradient gradient/test_finite_difference_gradient.cpp)
add_layer3_set5_exe(set5_test_lbfgs_standalone           optimizer/test_lbfgs_standalone.cpp)
add_layer3_set5_exe(set5_test_jacobian_rank              identifiability/test_jacobian_rank.cpp)
add_layer3_set5_exe(set5_test_llr_formula                quantization/test_llr_formula.cpp)
add_layer3_set5_exe(set5_test_window_length              scheduler/test_window_length.cpp)
add_layer3_set5_exe(set5_test_fsm_transitions            safety_fsm/test_fsm_transitions.cpp)

set(SET5_CANONICAL_TESTS
    "L3-MATH-001|set5_test_dp_vs_bruteforce"
    "L3-MATH-002|set5_test_dp_vs_bruteforce"
    "L3-MATH-003|set5_test_dp_vs_bruteforce"
    "L3-MATH-004|set5_test_dp_vs_bruteforce"
    "L3-MATH-005|set5_test_dp_vs_bruteforce"
    "L3-MATH-006|set5_test_dp_vs_bruteforce"
    "L3-MATH-007|set5_test_dp_vs_bruteforce"
    "L3-MATH-008|set5_test_dp_vs_bruteforce"
    "L3-MATH-009|set5_test_finite_difference_gradient"
    "L3-MATH-010|set5_test_finite_difference_gradient"
    "L3-MATH-011|set5_test_finite_difference_gradient"
    "L3-MATH-012|set5_test_finite_difference_gradient"
    "L3-MATH-013|set5_test_finite_difference_gradient"
    "L3-MATH-014|set5_test_finite_difference_gradient"
    "L3-MATH-015|set5_test_lbfgs_standalone"
    "L3-MATH-016|set5_test_lbfgs_standalone"
    "L3-MATH-017|set5_test_lbfgs_standalone"
    "L3-MATH-018|set5_test_lbfgs_standalone"
    "L3-MATH-019|set5_test_lbfgs_standalone"
    "L3-MATH-020|set5_test_lbfgs_standalone"
    "L3-MATH-021|set5_test_lbfgs_standalone"
    "L3-MATH-022|set5_test_lbfgs_standalone"
    "L3-MATH-023|set5_test_lbfgs_standalone"
    "L3-MATH-024|set5_test_lbfgs_standalone"
    "L3-MATH-025|set5_test_jacobian_rank"
    "L3-MATH-026|set5_test_jacobian_rank"
    "L3-MATH-027|set5_test_jacobian_rank"
    "L3-MATH-028|set5_test_jacobian_rank"
    "L3-MATH-029|set5_test_jacobian_rank"
    "L3-MATH-030|set5_test_jacobian_rank"
    "L3-MATH-031|set5_test_jacobian_rank"
    "L3-MATH-032|set5_test_jacobian_rank"
    "L3-MATH-033|set5_test_llr_formula"
    "L3-MATH-034|set5_test_llr_formula"
    "L3-MATH-035|set5_test_llr_formula"
    "L3-MATH-036|set5_test_llr_formula"
    "L3-MATH-037|set5_test_llr_formula"
    "L3-MATH-038|set5_test_llr_formula"
    "L3-MATH-039|set5_test_llr_formula"
    "L3-MATH-040|set5_test_llr_formula"
    "L3-MATH-041|set5_test_llr_formula"
    "L3-MATH-042|set5_test_llr_formula"
    "L3-MATH-043|set5_test_window_length"
    "L3-MATH-044|set5_test_window_length"
    "L3-MATH-045|set5_test_window_length"
    "L3-MATH-046|set5_test_window_length"
    "L3-MATH-047|set5_test_window_length"
    "L3-MATH-048|set5_test_window_length"
    "L3-MATH-049|set5_test_window_length"
    "L3-MATH-050|set5_test_window_length"
    "L3-MATH-051|set5_test_window_length"
    "L3-MATH-052|set5_test_window_length"
    "L3-MATH-053|set5_test_fsm_transitions"
    "L3-MATH-054|set5_test_fsm_transitions"
    "L3-MATH-055|set5_test_fsm_transitions"
    "L3-MATH-056|set5_test_fsm_transitions"
    "L3-MATH-057|set5_test_fsm_transitions"
    "L3-MATH-058|set5_test_fsm_transitions"
    "L3-MATH-059|set5_test_fsm_transitions"
    "L3-MATH-060|set5_test_fsm_transitions"
    "L3-MATH-061|set5_test_fsm_transitions"
    "L3-MATH-062|set5_test_fsm_transitions"
)

foreach(pair IN LISTS SET5_CANONICAL_TESTS)
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 tid)
    list(GET parts 1 exe_tgt)
    add_test(
        NAME ${tid}
        COMMAND ${exe_tgt}
            --test-id ${tid}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_VALIDATION_DIR}
            --daemon-bin $<TARGET_FILE:avid_daemon_under_test>
            --ctest-strict
    )
endforeach()
