cmake_minimum_required(VERSION 3.16)
project(AA_QEC CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

# =============================================================================
# Cross-compilation for Zynq ARM (uncomment on x86 host)
# =============================================================================
# set(CMAKE_SYSTEM_NAME      Linux)
# set(CMAKE_SYSTEM_PROCESSOR arm)
# set(CMAKE_C_COMPILER       arm-linux-gnueabihf-gcc)
# set(CMAKE_CXX_COMPILER     arm-linux-gnueabihf-g++)
# set(CMAKE_FIND_ROOT_PATH   /usr/arm-linux-gnueabihf)
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# =============================================================================
# Eigen3 (header-only — no link needed if using INTERFACE target)
# =============================================================================
find_package(Eigen3 REQUIRED NO_MODULE)

# =============================================================================
# avid_daemon — the adaptive inference process
# =============================================================================
add_executable(avid_daemon
    src/driver/weight_reg_driver.c
    src/driver/dma_manager.c
    src/inference/ring_buffer.h       # Header-only component
    src/inference/noise_model.cpp
    src/inference/lbfgs_optimizer.cpp
    src/inference/weight_injector.cpp
    src/inference/avid_daemon.cpp
)

target_include_directories(avid_daemon PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(avid_daemon PRIVATE Eigen3::Eigen)

# ARM NEON vectorisation for Eigen matrix operations
target_compile_options(avid_daemon PRIVATE
    -O3
    -ffast-math
    $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},arm>:-march=armv7-a -mfpu=neon -mfloat-abi=hard>
    $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>:-march=armv8-a>
)

# Suppress Eigen warnings on ARM
target_compile_definitions(avid_daemon PRIVATE
    EIGEN_DONT_PARALLELIZE          # Single-threaded: no OpenMP contention
    EIGEN_NO_DEBUG                  # Disable bounds checks in release
)

# =============================================================================
# Install target (on Zynq Linux: sudo make install → /usr/local/bin/)
# =============================================================================
install(TARGETS avid_daemon DESTINATION bin)

# =============================================================================
# Build instructions
# =============================================================================
message(STATUS "")
message(STATUS "AVID Software Build")
message(STATUS "  Compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "  Eigen3   : ${EIGEN3_INCLUDE_DIR}")
message(STATUS "")
message(STATUS "Build on Zynq:")
message(STATUS "  mkdir build && cd build && cmake .. && make -j4")
message(STATUS "")
message(STATUS "Cross-compile from x86:")
message(STATUS "  Uncomment the cross-compilation block above, then:")
message(STATUS "  mkdir build_arm && cd build_arm && cmake .. && make -j8")
