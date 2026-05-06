cmake_minimum_required(VERSION 3.22)

project(tokenspeed_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TOKENSPEED_SCHEDULER_BUILD_PYTHON "Build Python bindings" OFF)
option(TOKENSPEED_SCHEDULER_BUILD_TESTS "Build tests" OFF)

include(FetchContent)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.1
)
FetchContent_MakeAvailable(spdlog)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(tokenspeed_scheduler_core STATIC
    csrc/core/token_container.cpp

    csrc/resource/types.cpp
    csrc/resource/allocator/kv_allocator.cpp
    csrc/resource/allocator/owned_pages.cpp
    csrc/resource/allocator/page_allocator.cpp
    csrc/resource/allocator/req_pool_allocator.cpp
    csrc/resource/page_container.cpp
    csrc/resource/radix_tree/radix_tree.cpp
    csrc/resource/radix_tree/tree_node.cpp
    csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp
    csrc/resource/kv_prefix_cache/cache_coordinator.cpp
    csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp
    csrc/resource/hybrid_prefix_cache/mamba_eviction_manager.cpp
    csrc/resource/allocator/mamba_chunk_allocator.cpp
    csrc/resource/allocator/local_mamba_allocator.cpp

    csrc/fsm/forward_events.cpp
    csrc/fsm/cache_events.cpp
    csrc/fsm/pd_events.cpp

    csrc/scheduler/operations/cache.cpp
    csrc/scheduler/operations/forward.cpp

    csrc/scheduler/kv_cache_events.cpp
    csrc/scheduler/request.cpp
    csrc/scheduler/outside_event_handler.cpp
    csrc/scheduler/scheduler.cpp
)

target_include_directories(tokenspeed_scheduler_core
    PUBLIC
        csrc
)

target_link_libraries(tokenspeed_scheduler_core
    PUBLIC
        OpenSSL::Crypto
        spdlog::spdlog
)

if(TOKENSPEED_SCHEDULER_BUILD_PYTHON)
    nanobind_add_module(tokenspeed_scheduler_ext
        bindings/python_module.cpp
    )

    target_link_libraries(tokenspeed_scheduler_ext
        PRIVATE
            tokenspeed_scheduler_core
    )

    target_include_directories(tokenspeed_scheduler_ext
        PRIVATE
            csrc
    )

    install(TARGETS tokenspeed_scheduler_ext
        LIBRARY DESTINATION tokenspeed_scheduler
    )
endif()

if(TOKENSPEED_SCHEDULER_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(tokenspeed_scheduler_tests
        tests/cpp/test_scheduler_plan.cpp
        tests/cpp/test_outside_event_handler.cpp
        tests/cpp/test_write_back.cpp
        tests/cpp/test_load_back.cpp
        tests/cpp/test_retract.cpp
        tests/cpp/test_prefix_cache_host_match.cpp
        tests/cpp/test_abort.cpp
        tests/cpp/test_basic_lifecycle.cpp
        tests/cpp/test_chunked_prefill.cpp
        tests/cpp/test_batch_scheduling.cpp
        tests/cpp/test_prefetch.cpp
        tests/cpp/test_owned_pages.cpp
        tests/cpp/test_retract_abort_pages.cpp
        tests/cpp/test_mamba_slot.cpp
        tests/cpp/test_mamba_eviction.cpp
        tests/cpp/test_mamba_cache.cpp
        tests/cpp/test_mamba_integration.cpp
        tests/cpp/test_kv_cache_events.cpp
    )

    target_link_libraries(tokenspeed_scheduler_tests
        PRIVATE
            tokenspeed_scheduler_core
            GTest::gtest
            GTest::gtest_main
    )

    target_include_directories(tokenspeed_scheduler_tests
        PRIVATE
            csrc
            tests/cpp
    )

    include(GoogleTest)
    gtest_discover_tests(tokenspeed_scheduler_tests)
endif()
