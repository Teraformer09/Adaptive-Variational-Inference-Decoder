add_library(avid_layer3_test_common STATIC
    test_common.cpp
    test_result.cpp
    test_environment.cpp
    test_utils/binary_packet_builder.cpp
    test_utils/histogram_builder.cpp
    test_utils/fixture_loader.cpp
)

target_include_directories(avid_layer3_test_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils
)

target_compile_definitions(avid_layer3_test_common PUBLIC
    AVID_DEFAULT_REPO_ROOT="${AVID_REPO_ROOT}"
    AVID_BUILD_ASAN=$<BOOL:${AVID_ENABLE_ASAN}>
    AVID_BUILD_UBSAN=$<BOOL:${AVID_ENABLE_UBSAN}>
)

target_link_libraries(avid_layer3_test_common PUBLIC avid_layer3_prod)
target_compile_options(avid_layer3_test_common PRIVATE -Wall -Wextra -g)
