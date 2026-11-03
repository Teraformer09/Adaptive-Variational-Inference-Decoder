cmake_minimum_required(VERSION 3.16)
project(AA_QEC CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD   11)

find_package(Eigen3 QUIET NO_MODULE)

set(AVID_CORE_SOURCES
    src/driver/weight_reg_driver.c
    src/driver/weight_bank_driver.c
    src/driver/dma_manager.c
    src/ingress/packet_validator.cpp
    src/ingress/dma_ingress_tap.cpp
    src/inference/noise_model.cpp
    src/inference/lbfgs_optimizer.cpp
    src/estimator/joint_dp_estimator.cpp
    src/estimator/lbfgs_solver.cpp
    src/estimator/llr_quantizer.cpp
    src/control/causal_scheduler.cpp
    src/control/stage_profiler.cpp
    src/control/fallback_fsm.cpp
    src/custom_engine/cycle_accurate_bp_engine.cpp
)

add_executable(avid_daemon
    ${AVID_CORE_SOURCES}
    src/inference/avid_daemon.cpp
)

add_executable(test_baseline_suite
    ${AVID_CORE_SOURCES}
    tests/test_baseline_suite.cpp
)

if(Eigen3_FOUND)
    target_include_directories(avid_daemon PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(avid_daemon PRIVATE Eigen3::Eigen)
    target_include_directories(test_baseline_suite PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_baseline_suite PRIVATE Eigen3::Eigen)
else()
    message(STATUS "System Eigen3 not found — using built-in zero-allocation Eigen/Dense header")
    target_include_directories(avid_daemon PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
    target_include_directories(test_baseline_suite PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
endif()

target_compile_options(avid_daemon PRIVATE -O3 -Wall)
target_compile_definitions(avid_daemon PRIVATE EIGEN_DONT_PARALLELIZE EIGEN_NO_DEBUG)

target_compile_options(test_baseline_suite PRIVATE -O2 -Wall)
target_compile_definitions(test_baseline_suite PRIVATE EIGEN_DONT_PARALLELIZE)

install(TARGETS avid_daemon DESTINATION bin)
