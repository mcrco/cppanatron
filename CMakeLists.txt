cmake_minimum_required(VERSION 3.20)
project(cppanatron VERSION 0.1.0 LANGUAGES CXX)

option(CPPANATRON_BUILD_TESTS "Build cppanatron tests" ON)

add_library(cppanatron
    src/map.cpp
    src/board.cpp
)
add_library(cppanatron::cppanatron ALIAS cppanatron)

target_compile_features(cppanatron PUBLIC cxx_std_20)
target_include_directories(
    cppanatron
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(cppanatron PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cppanatron PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(CPPANATRON_BUILD_TESTS)
    enable_testing()
    add_executable(cppanatron_tests tests/test_board.cpp)
    target_link_libraries(cppanatron_tests PRIVATE cppanatron::cppanatron)
    add_test(NAME cppanatron_tests COMMAND cppanatron_tests)
endif()
