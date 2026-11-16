cmake_minimum_required(VERSION 3.16)
project(AVID_CoSim_Bridge CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

option(AVID_USE_VERILATOR "Compile native Verilator Vaa_qec_top into bridge and standalone_rtl_tb" OFF)

set(AVID_RTL_SV_FILES
    "${REPO_ROOT}/firmware/rtl/decoder/check_node_minsum.sv"
    "${REPO_ROOT}/firmware/rtl/ingress/axis_ingress_tap.sv"
    "${REPO_ROOT}/firmware/rtl/ingress/axis_syndrome_rx.sv"
    "${REPO_ROOT}/firmware/rtl/decoder/axis_correction_tx.sv"
    "${REPO_ROOT}/firmware/rtl/config/weight_map_reg.sv"
    "${REPO_ROOT}/firmware/rtl/decoder/bp_engine.sv"
    "${REPO_ROOT}/firmware/rtl/top/aa_qec_top.sv"
)

if(AVID_USE_VERILATOR)
    find_program(VERILATOR_BIN verilator HINTS "$ENV{HOME}/.local/bin" "$ENV{HOME}/.local/verilator/bin")
    if(NOT VERILATOR_BIN)
        message(FATAL_ERROR "AVID_USE_VERILATOR=ON requested, but 'verilator' executable was not found")
    endif()
    set(VERILATOR_FLAGS --cc --exe --build --build-jobs 4 --trace --assert -Wno-fatal --top-module aa_qec_top)
    set(VERILATOR_CFLAGS "-O2 -std=c++17 -DUSE_VERILATOR=1 -I${REPO_ROOT}/emulator/verilator/tb -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${REPO_ROOT}/software/src")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/avid_cosim_bridge"
        COMMAND ${VERILATOR_BIN} ${VERILATOR_FLAGS}
            -Mdir "${CMAKE_CURRENT_BINARY_DIR}/obj_bridge"
            -CFLAGS "${VERILATOR_CFLAGS}"
            ${AVID_RTL_SV_FILES}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cosim_bridge_main.cpp"
            -o avid_cosim_bridge
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/obj_bridge/avid_cosim_bridge"
            "${CMAKE_CURRENT_BINARY_DIR}/avid_cosim_bridge"
        DEPENDS ${AVID_RTL_SV_FILES}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/cosim_bridge_main.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/bridge_protocol.h"
            "${REPO_ROOT}/emulator/verilator/tb/verilator_rtl_driver.h"
        VERBATIM
    )
    add_custom_target(avid_cosim_bridge ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/avid_cosim_bridge")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/standalone_rtl_tb"
        COMMAND ${VERILATOR_BIN} ${VERILATOR_FLAGS}
            -Mdir "${CMAKE_CURRENT_BINARY_DIR}/obj_standalone"
            -CFLAGS "${VERILATOR_CFLAGS}"
            ${AVID_RTL_SV_FILES}
            "${REPO_ROOT}/emulator/verilator/tb/standalone_rtl_tb.cpp"
            -o standalone_rtl_tb
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/obj_standalone/standalone_rtl_tb"
            "${CMAKE_CURRENT_BINARY_DIR}/standalone_rtl_tb"
        DEPENDS ${AVID_RTL_SV_FILES}
            "${REPO_ROOT}/emulator/verilator/tb/standalone_rtl_tb.cpp"
            "${REPO_ROOT}/emulator/verilator/tb/verilator_rtl_driver.h"
        VERBATIM
    )
    add_custom_target(standalone_rtl_tb ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/standalone_rtl_tb")
else()
    add_executable(avid_cosim_bridge
        src/cosim_bridge_main.cpp
    )
    target_include_directories(avid_cosim_bridge PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${REPO_ROOT}/emulator/verilator/tb
    )
    target_compile_options(avid_cosim_bridge PRIVATE -O2 -Wall)

    add_executable(standalone_rtl_tb
        ${REPO_ROOT}/emulator/verilator/tb/standalone_rtl_tb.cpp
    )
    target_include_directories(standalone_rtl_tb PRIVATE
        ${REPO_ROOT}/emulator/verilator/tb
    )
    target_compile_options(standalone_rtl_tb PRIVATE -O2 -Wall)
endif()

add_executable(test_bridge_transactions
    tests/test_bridge_transactions.cpp
    ${REPO_ROOT}/software/src/hal/physical_backend.cpp
    ${REPO_ROOT}/software/src/hal/simulation_backend.cpp
    ${REPO_ROOT}/software/src/driver/weight_reg_driver.c
    ${REPO_ROOT}/software/src/driver/weight_bank_driver.c
    ${REPO_ROOT}/software/src/custom_engine/cycle_accurate_bp_engine.cpp
)
target_include_directories(test_bridge_transactions PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${REPO_ROOT}/software/src
)
target_compile_options(test_bridge_transactions PRIVATE -O2 -Wall)
