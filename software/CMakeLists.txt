cmake_minimum_required(VERSION 3.16)
project(AVID_Layer4_Guest_HAL CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

set(AVID_L4_HAL_SOURCES
    src/driver/weight_reg_driver.c
    src/driver/weight_bank_driver.c
    src/hal/physical_backend.cpp
    src/hal/simulation_backend.cpp
    src/custom_engine/cycle_accurate_bp_engine.cpp
)

add_executable(avid_daemon
    ${AVID_L4_HAL_SOURCES}
    src/guest/avid_hal_guest_daemon.cpp
)

target_include_directories(avid_daemon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(avid_daemon PRIVATE -O2 -Wall -Wextra)

install(TARGETS avid_daemon DESTINATION bin)
