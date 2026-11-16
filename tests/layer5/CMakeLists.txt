cmake_minimum_required(VERSION 3.16)
project(AVID_Layer5_RTL_Validation_Suite CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(VLT_OBJ_DIR "${REPO_ROOT}/firmware/build_layer5/obj_standalone")

find_program(VERILATOR_BIN verilator
    HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/verilator/bin"
)
if(NOT VERILATOR_BIN)
    message(FATAL_ERROR "Native 'verilator' executable is required to build Layer 5 RTL validation tests.")
endif()

execute_process(
    COMMAND "${VERILATOR_BIN}" --getenv VERILATOR_ROOT
    OUTPUT_VARIABLE VERILATOR_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE VLT_RC
)
if(NOT VLT_RC EQUAL 0 OR NOT EXISTS "${VERILATOR_ROOT}/include/verilated.h")
    message(FATAL_ERROR "Failed to locate VERILATOR_ROOT include directory (got '${VERILATOR_ROOT}').")
endif()

if(NOT EXISTS "${VLT_OBJ_DIR}/Vaa_qec_top__ALL.a")
    message(FATAL_ERROR
        "Missing prebuilt Verilator archive '${VLT_OBJ_DIR}/Vaa_qec_top__ALL.a'. "
        "Compile Vaa_qec_top first via scripts/run_layer5.sh."
    )
endif()

find_package(Threads REQUIRED)

set(VLT_LINK_OBJECTS
    "${VLT_OBJ_DIR}/verilated.o"
    "${VLT_OBJ_DIR}/verilated_vcd_c.o"
    "${VLT_OBJ_DIR}/verilated_threads.o"
    "${VLT_OBJ_DIR}/Vaa_qec_top__ALL.a"
)

function(add_l5_rtl_test TARGET_NAME SRC_REL_PATH)
    add_executable(${TARGET_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_REL_PATH}")
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_VERILATOR=1)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${REPO_ROOT}/emulator/verilator/tb"
    )
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${VLT_OBJ_DIR}"
        "${VERILATOR_ROOT}/include"
        "${VERILATOR_ROOT}/include/vltstd"
    )
    target_compile_options(${TARGET_NAME} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${VLT_LINK_OBJECTS}
        Threads::Threads
        atomic
    )
endfunction()

# Set 1 — Input, AXI4-Lite & AXI4-Stream Ingress Contracts (44 IDs)
add_l5_rtl_test(test_axis_ingress_contract  "set1_input_contracts/test_axis_ingress_contract.cpp")
add_l5_rtl_test(test_axil_register_contract "set1_input_contracts/test_axil_register_contract.cpp")

# Set 2 — Output, Deterministic Cycle Law & Correction Stream Contracts (38 IDs)
add_l5_rtl_test(test_cycle_latency_law          "set2_output_contracts/test_cycle_latency_law.cpp")
add_l5_rtl_test(test_correction_output_contract "set2_output_contracts/test_correction_output_contract.cpp")

# Set 3 — Pipeline, Ingress Tap & Atomic Weight-Bank System Flow (36 IDs)
add_l5_rtl_test(test_ingress_tap_pipeline "set3_pipeline_flow/test_ingress_tap_pipeline.cpp")
add_l5_rtl_test(test_weight_bank_flow     "set3_pipeline_flow/test_weight_bank_flow.cpp")

# Set 4 — Execution, Backpressure, Reset, Stress & Artifact Persistence (32 C++ IDs + 8 Python IDs)
add_l5_rtl_test(test_reset_and_backpressure "set4_execution_stress/test_reset_and_backpressure.cpp")
add_l5_rtl_test(test_streaming_stress       "set4_execution_stress/test_streaming_stress.cpp")

# Set 5 — Hardware-Specific Q7.8 Min-Sum, OSD-0, SVA & Constraints (24 C++ IDs + 18 Python IDs)
add_l5_rtl_test(test_q78_minsum_osd_math "set5_hardware_verification/test_q78_minsum_osd_math.cpp")
