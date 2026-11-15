add_library(avid_layer3_c_abi_oracle STATIC
    c_abi_cxx_layout_oracle.cpp
)
target_link_libraries(avid_layer3_c_abi_oracle PUBLIC avid_layer3_prod)

set(C_ABI_SPECS
    "L3-ABI-001|c_abi_test_packet_abi|test_packet_abi.c"
    "L3-ABI-002|c_abi_test_histogram_abi|test_histogram_abi.c"
    "L3-ABI-003|c_abi_test_parameter_abi|test_parameter_abi.c"
    "L3-ABI-004|c_abi_test_runtime_config_abi|test_runtime_config_abi.c"
    "L3-ABI-005|c_abi_test_struct_layout|test_struct_layout.c"
    "L3-ABI-006|c_abi_test_null_pointer|test_null_pointer.c"
    "L3-ABI-007|c_abi_test_buffer_ownership|test_buffer_ownership.c"
)

foreach(spec IN LISTS C_ABI_SPECS)
    string(REPLACE "|" ";" parts "${spec}")
    list(GET parts 0 abi_id)
    list(GET parts 1 tgt_name)
    list(GET parts 2 src_file)

    add_executable(${tgt_name} ${src_file})
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE C)
    target_compile_options(${tgt_name} PRIVATE -Wall -Wextra -g)
    target_include_directories(${tgt_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_options(${tgt_name} PRIVATE "-rdynamic")
    target_link_libraries(${tgt_name} PRIVATE avid_layer3_c_abi_oracle avid_layer3_prod dl m)

    add_test(
        NAME ${abi_id}
        COMMAND ${tgt_name}
            --test-id ${abi_id}
            --repo-root ${AVID_REPO_ROOT}
            --validation-dir ${AVID_REPO_ROOT}/validation/layer3
            --c-compiler ${CMAKE_C_COMPILER}
    )
endforeach()
