# Set 3 — Pipeline and System-Flow Validation (45 Canonical IDs: L3-FLOW-001..045)

function(avid_add_set3_target target_name src_path)
    add_executable(${target_name} ${src_path})
    target_link_libraries(${target_name} PRIVATE avid_layer3_test_common avid_layer3_prod)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -g)
endfunction()

avid_add_set3_target(set3_packet_to_histogram         ingress_to_histogram/test_packet_to_histogram.cpp)
avid_add_set3_target(set3_histogram_snapshot_delivery histogram_to_estimator/test_histogram_snapshot_delivery.cpp)
avid_add_set3_target(set3_solver_call_path            optimizer_execution/test_solver_call_path.cpp)
avid_add_set3_target(set3_parameter_to_llr            estimator_to_llr/test_parameter_to_llr.cpp)
avid_add_set3_target(set3_commit_scheduling           scheduler_integration/test_commit_scheduling.cpp)
avid_add_set3_target(set3_fsm_to_hal_action           fsm_to_hal/test_fsm_to_hal_action.cpp)
avid_add_set3_target(set3_fast_slow_loop_separation   scheduler_integration/test_fast_slow_loop_separation.cpp)
avid_add_set3_target(set3_stationary_flow             end_to_end/test_stationary_flow.cpp)

function(avid_register_set3_case test_id target_name)
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

avid_register_set3_case(L3-FLOW-001 set3_packet_to_histogram)
avid_register_set3_case(L3-FLOW-002 set3_histogram_snapshot_delivery)
avid_register_set3_case(L3-FLOW-003 set3_histogram_snapshot_delivery)
avid_register_set3_case(L3-FLOW-004 set3_solver_call_path)
avid_register_set3_case(L3-FLOW-005 set3_solver_call_path)
avid_register_set3_case(L3-FLOW-006 set3_parameter_to_llr)
avid_register_set3_case(L3-FLOW-007 set3_commit_scheduling)
avid_register_set3_case(L3-FLOW-008 set3_commit_scheduling)

foreach(idx RANGE 9 12)
    if(idx LESS 10)
        avid_register_set3_case("L3-FLOW-00${idx}" set3_fsm_to_hal_action)
    else()
        avid_register_set3_case("L3-FLOW-0${idx}" set3_fsm_to_hal_action)
    endif()
endforeach()

foreach(idx RANGE 13 25)
    avid_register_set3_case("L3-FLOW-0${idx}" set3_stationary_flow)
endforeach()

foreach(idx RANGE 26 35)
    avid_register_set3_case("L3-FLOW-0${idx}" set3_fast_slow_loop_separation)
endforeach()

foreach(idx RANGE 36 45)
    avid_register_set3_case("L3-FLOW-0${idx}" set3_stationary_flow)
endforeach()
