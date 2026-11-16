cmake_minimum_required(VERSION 3.16)
project(AVID_Layer4_Validation_Suite CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(VLT_OBJ_DIR "${REPO_ROOT}/emulator/verilator/models/obj_standalone")

find_program(VERILATOR_BIN verilator
    HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/verilator/bin"
)
if(NOT VERILATOR_BIN)
    message(FATAL_ERROR "Native 'verilator' executable is required to build Layer 4 validation tests.")
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
        "Run 'AVID_REQUIRE_NATIVE_VERILATOR=1 bash emulator/verilator/build.sh' first."
    )
endif()

find_package(Threads REQUIRED)

# Shared Layer 4 HAL & driver static library
add_library(avid_l4_hal_lib STATIC
    "${REPO_ROOT}/software/src/driver/weight_reg_driver.c"
    "${REPO_ROOT}/software/src/driver/weight_bank_driver.c"
    "${REPO_ROOT}/software/src/hal/physical_backend.cpp"
    "${REPO_ROOT}/software/src/hal/simulation_backend.cpp"
    "${REPO_ROOT}/software/src/custom_engine/cycle_accurate_bp_engine.cpp"
)
target_include_directories(avid_l4_hal_lib PUBLIC
    "${REPO_ROOT}/software/src"
)
target_compile_options(avid_l4_hal_lib PRIVATE -O2 -Wall -Wextra)

# C++ cycle-exact reference fallback bridge (built WITHOUT -DUSE_VERILATOR for L4-CFID-07 & L4-LIM-02)
add_executable(avid_cosim_bridge_fallback
    "${REPO_ROOT}/emulator/bridge/src/cosim_bridge_main.cpp"
)
target_include_directories(avid_cosim_bridge_fallback PRIVATE
    "${REPO_ROOT}/emulator/bridge/include"
    "${REPO_ROOT}/emulator/verilator/tb"
    "${REPO_ROOT}/software/src"
)
target_compile_options(avid_cosim_bridge_fallback PRIVATE -O2 -Wall)
target_link_libraries(avid_cosim_bridge_fallback PRIVATE Threads::Threads)

set(VLT_LINK_OBJECTS
    "${VLT_OBJ_DIR}/verilated.o"
    "${VLT_OBJ_DIR}/verilated_vcd_c.o"
    "${VLT_OBJ_DIR}/verilated_threads.o"
    "${VLT_OBJ_DIR}/Vaa_qec_top__ALL.a"
)

function(add_l4_cpp_test TARGET_NAME SRC_REL_PATH)
    add_executable(${TARGET_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_REL_PATH}")
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_VERILATOR=1)
    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${REPO_ROOT}/software/src"
        "${REPO_ROOT}/emulator/bridge/include"
        "${REPO_ROOT}/emulator/verilator/tb"
    )
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${VLT_OBJ_DIR}"
        "${VERILATOR_ROOT}/include"
        "${VERILATOR_ROOT}/include/vltstd"
    )
    target_compile_options(${TARGET_NAME} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(${TARGET_NAME} PRIVATE
        avid_l4_hal_lib
        ${VLT_LINK_OBJECTS}
        Threads::Threads
        atomic
    )
endfunction()

# Set 1 — Input Contracts (5 C++ binaries)
add_l4_cpp_test(test_rpc_schema            "set1_input_contracts/test_rpc_schema.cpp")
add_l4_cpp_test(test_mmio_contract         "set1_input_contracts/test_mmio_contract.cpp")
add_l4_cpp_test(test_syndrome_transfer     "set1_input_contracts/test_syndrome_transfer.cpp")
add_l4_cpp_test(test_transaction_ordering  "set1_input_contracts/test_transaction_ordering.cpp")
add_l4_cpp_test(test_socket_recovery       "set1_input_contracts/test_socket_recovery.cpp")

# Set 2 — Output & Transaction Correctness (4 C++ binaries)
add_l4_cpp_test(test_acknowledgements         "set2_output_correctness/test_acknowledgements.cpp")
add_l4_cpp_test(test_cycle_accounting         "set2_output_correctness/test_cycle_accounting.cpp")
add_l4_cpp_test(test_rtl_response_consistency "set2_output_correctness/test_rtl_response_consistency.cpp")
add_l4_cpp_test(test_deterministic_replay     "set2_output_correctness/test_deterministic_replay.cpp")

# Set 4 — Execution, Runtime & Recovery (4 C++ binaries; 1 Python script run separately)
add_l4_cpp_test(test_bridge_startup    "set4_execution_recovery/test_bridge_startup.cpp")
add_l4_cpp_test(test_runtime_stress    "set4_execution_recovery/test_runtime_stress.cpp")
add_l4_cpp_test(test_fault_injection   "set4_execution_recovery/test_fault_injection.cpp")
add_l4_cpp_test(test_restart_recovery  "set4_execution_recovery/test_restart_recovery.cpp")

# Set 5 — Hardware-Interface & Co-Simulation-Specific Tests (6 C++ binaries)
add_l4_cpp_test(test_axil_protocol          "set5_hardware_cosim/test_axil_protocol.cpp")
add_l4_cpp_test(test_axis_syndrome_protocol "set5_hardware_cosim/test_axis_syndrome_protocol.cpp")
add_l4_cpp_test(test_cycle_fidelity         "set5_hardware_cosim/test_cycle_fidelity.cpp")
add_l4_cpp_test(test_decoder_equivalence    "set5_hardware_cosim/test_decoder_equivalence.cpp")
add_l4_cpp_test(test_weight_bank_semantics  "set5_hardware_cosim/test_weight_bank_semantics.cpp")
add_l4_cpp_test(test_cosim_fidelity_limits  "set5_hardware_cosim/test_cosim_fidelity_limits.cpp")
