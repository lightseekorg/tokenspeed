cmake_minimum_required(VERSION 3.22)

project(tokenspeed_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TOKENSPEED_SCHEDULER_BUILD_PYTHON "Build Python bindings" OFF)
option(TOKENSPEED_SCHEDULER_BUILD_TESTS "Build tests" OFF)

include(FetchContent)

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import tokenspeed_spdlog; print(tokenspeed_spdlog.cmake_prefix_path())"
    OUTPUT_VARIABLE tokenspeed_spdlog_ROOT
    ERROR_VARIABLE tokenspeed_spdlog_ERROR
    RESULT_VARIABLE tokenspeed_spdlog_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT tokenspeed_spdlog_RESULT EQUAL 0)
    message(FATAL_ERROR "tokenspeed-spdlog is required. Install it with `pip install tokenspeed-spdlog`. ${tokenspeed_spdlog_ERROR}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${tokenspeed_spdlog_ROOT}")
find_package(spdlog CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(tokenspeed_scheduler_core STATIC
    csrc/core/token_container.cpp

    csrc/resource/allocator/req_pool_allocator.cpp

    csrc/fsm/forward_events.cpp
    csrc/fsm/pd_events.cpp

    csrc/scheduler/operations/forward.cpp

    csrc/scheduler/kv_cache_events.cpp
    csrc/scheduler/request.cpp
    csrc/scheduler/types.cpp
    csrc/scheduler/outside_event_handler.cpp
    csrc/scheduler/scheduler.cpp

    csrc/cache/core/cache_config.cpp
    csrc/cache/coordinator/cache_admission.cpp
    csrc/cache/coordinator/cache_coordinator.cpp
    csrc/cache/core/cache_block_ref.cpp
    csrc/scheduler/operations/cache.cpp
    csrc/cache/tier/transfer_manager.cpp
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

# OpenSSL 3.0 deprecates the one-shot SHA256_* API in favor of EVP_*. We keep
# the SHA256_* calls in prefix_hasher.h; suppress the deprecation warnings.
target_compile_definitions(tokenspeed_scheduler_core
    PUBLIC
        OPENSSL_SUPPRESS_DEPRECATED
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
        tests/cpp/test_prefix_hasher.cpp
        tests/cpp/test_block_pool.cpp
        tests/cpp/test_cache_block_ref.cpp
        tests/cpp/test_full_attn_group.cpp
        tests/cpp/test_swa_group.cpp
        tests/cpp/test_match_host_pages.cpp
        tests/cpp/test_cache_coordinator.cpp
        tests/cpp/test_joint_match_invariants.cpp
        tests/cpp/test_forward_cache_ops.cpp
        tests/cpp/test_forward_batch.cpp
        tests/cpp/test_kv_cache_lifecycle.cpp
        tests/cpp/test_cache_scenarios.cpp
        tests/cpp/test_scheduler_plan.cpp
        tests/cpp/test_outside_event_handler.cpp
        tests/cpp/test_cache_operations.cpp
        tests/cpp/test_req_pool_allocator.cpp
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
