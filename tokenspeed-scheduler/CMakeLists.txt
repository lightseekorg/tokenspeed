cmake_minimum_required(VERSION 3.22)

project(tokenspeed_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TOKENSPEED_SCHEDULER_BUILD_PYTHON "Build Python bindings" OFF)
option(TOKENSPEED_SCHEDULER_BUILD_TESTS "Build tests" OFF)
option(TOKENSPEED_FLAT_KVCACHE "Use the new flat KV-cache (KvCacheCoordinator) path in the FSM instead of the radix LocalKVAllocator path" OFF)

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

    csrc/resource/types.cpp
    csrc/resource/allocator/kv_allocator.cpp
    csrc/resource/allocator/owned_pages.cpp
    csrc/resource/allocator/page_allocator.cpp
    csrc/resource/allocator/paged_cache_group.cpp
    csrc/resource/allocator/req_pool_allocator.cpp
    csrc/resource/page_container.cpp
    csrc/resource/radix_tree/radix_tree.cpp
    csrc/resource/radix_tree/tree_node.cpp
    csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp
    csrc/resource/kv_prefix_cache/cache_coordinator.cpp
    csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp
    csrc/resource/hybrid_prefix_cache/mamba_eviction_manager.cpp
    csrc/resource/allocator/mamba_chunk_allocator.cpp
    csrc/resource/allocator/mamba_host_allocator.cpp
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

    csrc/cache/kv_cache_coordinator.cpp
    csrc/cache/block_ref.cpp
    csrc/cache/forward_cache_ops.cpp
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
# the SHA256_* calls in page_hasher.h; suppress the deprecation warnings.
target_compile_definitions(tokenspeed_scheduler_core
    PUBLIC
        OPENSSL_SUPPRESS_DEPRECATED
)

if(TOKENSPEED_FLAT_KVCACHE)
    target_compile_definitions(tokenspeed_scheduler_core PUBLIC TOKENSPEED_FLAT_KVCACHE=1)
endif()

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
        tests/cpp/test_page_hasher.cpp
        tests/cpp/test_block_pool.cpp
        tests/cpp/test_block_ref.cpp
        tests/cpp/test_full_attn_manager.cpp
        tests/cpp/test_swa_manager.cpp
        tests/cpp/test_match_host_pages.cpp
        tests/cpp/test_kv_cache_coordinator.cpp
        tests/cpp/test_forward_cache_ops.cpp
        tests/cpp/test_flat_forward_operation.cpp
        tests/cpp/test_flat_kvcache_lifecycle.cpp
        tests/cpp/test_flat_kvcache_scenarios.cpp
        tests/cpp/test_scheduler_plan.cpp
        tests/cpp/test_outside_event_handler.cpp
        tests/cpp/test_write_back.cpp
        tests/cpp/test_load_back.cpp
        tests/cpp/test_cache_operation_kinds.cpp
        tests/cpp/test_retract.cpp
        tests/cpp/test_prefix_cache_host_match.cpp
        tests/cpp/test_abort.cpp
        tests/cpp/test_basic_lifecycle.cpp
        tests/cpp/test_chunked_prefill.cpp
        tests/cpp/test_batch_scheduling.cpp
        tests/cpp/test_prefetch.cpp
        tests/cpp/test_owned_pages.cpp
        tests/cpp/test_paged_cache_prefix_match.cpp
        tests/cpp/test_paged_cache_attach_loop.cpp
        tests/cpp/test_paged_cache_eviction.cpp
        tests/cpp/test_paged_cache_family_split.cpp
        tests/cpp/test_paged_cache_l2_offload.cpp
        tests/cpp/test_paged_cache_prefix_hit_commit.cpp
        tests/cpp/test_paged_cache_replay.cpp
        tests/cpp/test_retract_abort_pages.cpp
        tests/cpp/test_req_pool_allocator.cpp
        tests/cpp/test_mamba_slot.cpp
        tests/cpp/test_mamba_eviction.cpp
        tests/cpp/test_mamba_cache.cpp
        tests/cpp/test_mamba_integration.cpp
        tests/cpp/test_kv_cache_events.cpp
        tests/cpp/test_eviction_lru.cpp
        tests/cpp/test_host_node_ref_lifetime.cpp
        tests/cpp/test_hybrid_takefirst_overflow.cpp
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
