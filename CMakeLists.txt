cmake_minimum_required(VERSION 3.20)
project(cppanatron VERSION 0.4.0 LANGUAGES CXX)

option(CPPANATRON_BUILD_TESTS "Build cppanatron tests" ON)
option(CPPANATRON_BUILD_PYTHON "Build the cppanatron Python extension" OFF)

add_library(cppanatron
    src/map.cpp
    src/board.cpp
    src/game.cpp
    src/setup.cpp
    src/action_space.cpp
    src/graph_dims.cpp
    src/graph_action_space.cpp
    src/graph_encoder.cpp
    src/vec_env.cpp
    src/value_player.cpp
    src/observation.cpp
    src/batch.cpp
    src/mcts.cpp
    src/hidden_cards.cpp
    src/tracker.cpp
    src/flat_codec.cpp
    src/search_pool.cpp
)
add_library(cppanatron::cppanatron ALIAS cppanatron)

find_package(Threads REQUIRED)
target_link_libraries(cppanatron PUBLIC Threads::Threads)

target_compile_features(cppanatron PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(cppanatron PUBLIC Threads::Threads)
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

    add_executable(cppanatron_tracker_tests tests/test_tracker.cpp)
    target_link_libraries(cppanatron_tracker_tests PRIVATE cppanatron::cppanatron)
    add_test(NAME cppanatron_tracker_tests COMMAND cppanatron_tracker_tests)

    add_executable(cppanatron_search_pool_tests tests/test_search_pool.cpp)
    target_link_libraries(cppanatron_search_pool_tests PRIVATE cppanatron::cppanatron)
    add_test(NAME cppanatron_search_pool_tests COMMAND cppanatron_search_pool_tests)

    add_executable(cppanatron_action_dump tools/dump_action_space.cpp)
    target_link_libraries(cppanatron_action_dump PRIVATE cppanatron::cppanatron)

    add_executable(cppanatron_setup_dump tools/dump_setup.cpp)
    target_link_libraries(cppanatron_setup_dump PRIVATE cppanatron::cppanatron)

    add_executable(cppanatron_search_bench tools/bench_search_pool.cpp)
    target_link_libraries(cppanatron_search_bench PRIVATE cppanatron::cppanatron)
endif()

if(CPPANATRON_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_core
        NB_STATIC
        src/python/module.cpp
        src/python/types.cpp
        src/python/maps.cpp
        src/python/graph.cpp
        src/python/vec_env.cpp
        src/python/build_info.cpp
        src/python/game.cpp
        src/python/observation.cpp
        src/python/search.cpp
        src/python/tracker.cpp
        src/python/search_pool.cpp
    )
    target_link_libraries(_core PRIVATE cppanatron::cppanatron)

    # build_info(): compiler and flags at configure time, and the engine
    # repo's git commit and dirty flag at every build, so a rebuild after a
    # commit or an edit reports the current state.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
    string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${build_type_upper}}" CPPANATRON_CXX_FLAGS)
    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    configure_file(cmake/build_config.h.in "${generated_dir}/cppanatron_build_config.h" @ONLY)
    add_custom_target(cppanatron_git_info ALL
        COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DOUTPUT=${generated_dir}/cppanatron_git.h"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_info.cmake"
        BYPRODUCTS "${generated_dir}/cppanatron_git.h"
        COMMENT "Recording the engine's git commit")
    add_dependencies(_core cppanatron_git_info)
    target_include_directories(_core PRIVATE "${generated_dir}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(_core PRIVATE -Wall -Wextra -Werror)
    endif()

    # Import the freshly built extension as a top-level module so an installed
    # cppanatron package can never shadow it during stub generation.
    nanobind_add_stub(_core_stub
        MODULE _core
        OUTPUT _core.pyi
        MARKER_FILE py.typed
        PYTHON_PATH $<TARGET_FILE_DIR:_core>
        DEPENDS _core
    )

    install(TARGETS _core LIBRARY DESTINATION cppanatron)
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi
            ${CMAKE_CURRENT_BINARY_DIR}/py.typed
        DESTINATION cppanatron)
endif()
