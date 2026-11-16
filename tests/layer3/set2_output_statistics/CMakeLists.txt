# Set 2 — Output Contract & Statistical Estimator Validation (51 Canonical IDs)
find_program(AVID_PYTHON3_EXECUTABLE NAMES python3 REQUIRED)

function(avid_add_set2_target target_name src_path)
    add_executable(${target_name} ${src_path})
    target_link_libraries(${target_name} PRIVATE avid_layer3_test_common)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -g)
endfunction()

avid_add_set2_target(set2_parameter_output_shape estimator_output/test_parameter_output_shape.cpp)
avid_add_set2_target(set2_q78_output             llr_quantization/test_q78_output.cpp)
avid_add_set2_target(set2_scheduler_decision     scheduler_output/test_scheduler_decision.cpp)
avid_add_set2_target(set2_fsm_action_output      fsm_output/test_fsm_action_output.cpp)

add_executable(set2_stat_mc_driver statistical/stat_mc_driver.cpp)
target_link_libraries(set2_stat_mc_driver PRIVATE avid_layer3_prod)
target_compile_options(set2_stat_mc_driver PRIVATE -Wall -Wextra -O2)

function(avid_register_set2_cpp_case test_id target_name)
    add_test(
        NAME ${test_id}
        COMMAND $<TARGET_FILE:${target_name}>
            --test-id ${test_id}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_VALIDATION_DIR}
            --daemon-bin $<TARGET_FILE:avid_daemon_under_test>
            --ctest-strict
    )
endfunction()

function(avid_register_set2_py_case test_id script_rel)
    add_test(
        NAME ${test_id}
        COMMAND ${AVID_PYTHON3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script_rel}
            --test-id ${test_id}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_VALIDATION_DIR}
            --mc-driver $<TARGET_FILE:set2_stat_mc_driver>
            --ctest-strict
    )
endfunction()

foreach(idx RANGE 1 10)
    string(PREPEND idx_pad "00${idx}")
    string(SUBSTRING "${idx_pad}" 1 3 num3)
    if(idx EQUAL 10)
        set(num3 "010")
    else()
        set(num3 "00${idx}")
    endif()
    avid_register_set2_cpp_case("L3-OP-${num3}" set2_parameter_output_shape)
endforeach()

foreach(idx RANGE 11 20)
    avid_register_set2_cpp_case("L3-OP-0${idx}" set2_q78_output)
endforeach()

foreach(idx RANGE 21 28)
    avid_register_set2_cpp_case("L3-OP-0${idx}" set2_scheduler_decision)
endforeach()

foreach(idx RANGE 29 36)
    avid_register_set2_cpp_case("L3-OP-0${idx}" set2_fsm_action_output)
endforeach()

avid_register_set2_py_case(L3-STAT-001 statistical/test_stationary_recovery.py)
avid_register_set2_py_case(L3-STAT-002 statistical/test_stationary_recovery.py)
avid_register_set2_py_case(L3-STAT-003 statistical/test_stationary_recovery.py)
avid_register_set2_py_case(L3-STAT-004 statistical/test_correlated_noise.py)
avid_register_set2_py_case(L3-STAT-005 statistical/test_model_mismatch.py)
avid_register_set2_py_case(L3-STAT-006 statistical/test_sample_size_sensitivity.py)
avid_register_set2_py_case(L3-STAT-007 statistical/test_seed_variance.py)
avid_register_set2_py_case(L3-STAT-008 statistical/test_prior_sensitivity.py)
avid_register_set2_py_case(L3-STAT-009 statistical/test_initialization_sensitivity.py)
avid_register_set2_py_case(L3-STAT-010 statistical/test_model_mismatch.py)
avid_register_set2_py_case(L3-STAT-011 statistical/test_identifiable_parameter_recovery.py)
avid_register_set2_py_case(L3-STAT-012 statistical/test_drift_recovery.py)
avid_register_set2_py_case(L3-STAT-013 statistical/test_burst_contamination.py)
avid_register_set2_py_case(L3-STAT-014 statistical/test_false_trigger_rate.py)
avid_register_set2_py_case(L3-STAT-015 statistical/test_confidence_calibration.py)
