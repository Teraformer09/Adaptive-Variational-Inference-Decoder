# Set 4 — Execution, Persistence and Fault-Recovery Validation (45 Canonical IDs: L3-EXEC-001..045)

function(avid_add_set4_target target_name src_path)
    add_executable(${target_name} ${src_path})
    target_link_libraries(${target_name} PRIVATE avid_layer3_test_common avid_layer3_prod)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -g)
endfunction()

avid_add_set4_target(set4_daemon_startup            startup_shutdown/test_daemon_startup.cpp)
avid_add_set4_target(set4_long_running_daemon       runtime_stress/test_long_running_daemon.cpp)
avid_add_set4_target(set4_warm_resume               persistence/test_warm_resume.cpp)
avid_add_set4_target(set4_malformed_packet_recovery fault_injection/test_malformed_packet_recovery.cpp)
avid_add_set4_target(set4_race_conditions           concurrency/test_race_conditions.cpp)

function(avid_register_set4_case test_id target_name)
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

foreach(idx RANGE 1 9)
    avid_register_set4_case("L3-EXEC-00${idx}" set4_daemon_startup)
endforeach()

foreach(idx RANGE 10 16)
    avid_register_set4_case("L3-EXEC-0${idx}" set4_long_running_daemon)
endforeach()

foreach(idx RANGE 17 22)
    avid_register_set4_case("L3-EXEC-0${idx}" set4_warm_resume)
endforeach()

foreach(idx RANGE 23 31)
    avid_register_set4_case("L3-EXEC-0${idx}" set4_malformed_packet_recovery)
endforeach()

avid_register_set4_case(L3-EXEC-032 set4_warm_resume)

foreach(idx RANGE 33 35)
    avid_register_set4_case("L3-EXEC-0${idx}" set4_malformed_packet_recovery)
endforeach()

foreach(idx RANGE 36 45)
    avid_register_set4_case("L3-EXEC-0${idx}" set4_race_conditions)
endforeach()
