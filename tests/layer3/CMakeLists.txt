cmake_minimum_required(VERSION 3.20)

project(AVID_Layer3_Validation LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AVID_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(AVID_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

set(AVID_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
get_filename_component(AVID_REPO_ROOT "${AVID_REPO_ROOT}" REALPATH)
set(AVID_VALIDATION_DIR "${AVID_REPO_ROOT}/validation/layer3" CACHE PATH "Output validation directory for test logs and raw_results")

set(AVID_SANITIZER_FLAGS "")
if(AVID_ENABLE_ASAN)
    list(APPEND AVID_SANITIZER_FLAGS "-fsanitize=address" "-fno-omit-frame-pointer")
endif()
if(AVID_ENABLE_UBSAN)
    list(APPEND AVID_SANITIZER_FLAGS "-fsanitize=undefined" "-fno-omit-frame-pointer")
endif()

if(AVID_SANITIZER_FLAGS)
    add_compile_options(${AVID_SANITIZER_FLAGS})
    add_link_options(${AVID_SANITIZER_FLAGS})
endif()

find_package(Threads REQUIRED)

# Compile actual unmodified production sources directly from software/src and deployment/cpp
set(AVID_PROD_SOURCES
    ${AVID_REPO_ROOT}/software/src/driver/weight_reg_driver.c
    ${AVID_REPO_ROOT}/software/src/driver/weight_bank_driver.c
    ${AVID_REPO_ROOT}/software/src/driver/dma_manager.c
    ${AVID_REPO_ROOT}/software/src/hal/physical_backend.cpp
    ${AVID_REPO_ROOT}/software/src/hal/simulation_backend.cpp
    ${AVID_REPO_ROOT}/software/src/ingress/packet_validator.cpp
    ${AVID_REPO_ROOT}/software/src/ingress/dma_ingress_tap.cpp
    ${AVID_REPO_ROOT}/software/src/inference/noise_model.cpp
    ${AVID_REPO_ROOT}/software/src/inference/lbfgs_optimizer.cpp
    ${AVID_REPO_ROOT}/software/src/estimator/joint_dp_estimator.cpp
    ${AVID_REPO_ROOT}/software/src/estimator/lbfgs_solver.cpp
    ${AVID_REPO_ROOT}/software/src/estimator/llr_quantizer.cpp
    ${AVID_REPO_ROOT}/software/src/control/causal_scheduler.cpp
    ${AVID_REPO_ROOT}/software/src/control/stage_profiler.cpp
    ${AVID_REPO_ROOT}/software/src/control/fallback_fsm.cpp
    ${AVID_REPO_ROOT}/software/src/custom_engine/cycle_accurate_bp_engine.cpp
    ${AVID_REPO_ROOT}/deployment/cpp/model_loader.cpp
    ${AVID_REPO_ROOT}/deployment/cpp/r2d2_runtime.cpp
)

add_library(avid_layer3_prod STATIC ${AVID_PROD_SOURCES})
target_compile_definitions(avid_layer3_prod PUBLIC EIGEN_DONT_PARALLELIZE)
target_compile_options(avid_layer3_prod PRIVATE -Wall -g)

# Include both software/src and software/include across Layer 3 production translation units
target_include_directories(avid_layer3_prod PUBLIC
    ${AVID_REPO_ROOT}/software/src
    ${AVID_REPO_ROOT}/software/include
    ${AVID_REPO_ROOT}
)
target_link_libraries(avid_layer3_prod PUBLIC Threads::Threads m dl)

# Also build the unmodified production daemon executable for Set 1 integration & config consumption tests
add_executable(avid_daemon_under_test
    ${AVID_REPO_ROOT}/software/src/inference/avid_daemon.cpp
)
target_link_libraries(avid_daemon_under_test PRIVATE avid_layer3_prod)
target_compile_options(avid_daemon_under_test PRIVATE -Wall -g)

enable_testing()

add_subdirectory(common)
add_subdirectory(set1_input_contracts)
add_subdirectory(set2_output_statistics)
add_subdirectory(set3_pipeline_flow)
add_subdirectory(set4_execution_recovery)
add_subdirectory(set5_control_mathematics)
