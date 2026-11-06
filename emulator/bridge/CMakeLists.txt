cmake_minimum_required(VERSION 3.16)
project(AVID_CoSim_Bridge CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

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
