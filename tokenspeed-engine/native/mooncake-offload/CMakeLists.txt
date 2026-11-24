cmake_minimum_required(VERSION 3.22)
project(tokenspeed_mooncake_offload LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED MOONCAKE_SOURCE_DIR)
    message(FATAL_ERROR "MOONCAKE_SOURCE_DIR must name the deployed Mooncake SDK source")
endif()
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
add_library(tokenspeed_mooncake_offload SHARED client.cpp)
target_include_directories(tokenspeed_mooncake_offload PRIVATE
    "${NLOHMANN_JSON_INCLUDE_DIR}"
    "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include"
    "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/thirdparty"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/standalone")
target_link_libraries(tokenspeed_mooncake_offload PRIVATE
    OpenSSL::SSL OpenSSL::Crypto Threads::Threads ${CMAKE_DL_LIBS})
install(TARGETS tokenspeed_mooncake_offload LIBRARY DESTINATION lib)

include(CTest)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_executable(offload_test_provider test_provider.cpp)
    target_include_directories(offload_test_provider PRIVATE
        "${NLOHMANN_JSON_INCLUDE_DIR}"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/thirdparty"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/standalone")
    target_link_libraries(offload_test_provider PRIVATE OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
    add_test(NAME offload_provider_restart COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_restart.py"
        "$<TARGET_FILE:tokenspeed_mooncake_offload>" "$<TARGET_FILE:offload_test_provider>")
endif()

# Build against the same SDK that the Store client uses. Keeping discovery in
# its own library avoids linking the protocol-only bridge to SDK class layouts.
if(DEFINED MOONCAKE_STORE_LIBRARY)
    add_library(tokenspeed_mooncake_discovery SHARED discovery.cpp)
    target_compile_definitions(tokenspeed_mooncake_discovery PRIVATE STORE_USE_ETCD)
    target_include_directories(tokenspeed_mooncake_discovery PRIVATE
        "${NLOHMANN_JSON_INCLUDE_DIR}"
        "${MOONCAKE_SOURCE_DIR}/mooncake-store/include"
        "${MOONCAKE_SOURCE_DIR}/mooncake-store/include/cachelib_memory_allocator"
        "${MOONCAKE_SOURCE_DIR}/mooncake-store/include/cachelib_memory_allocator/include"
        "${MOONCAKE_SOURCE_DIR}/mooncake-store/include/cachelib_memory_allocator/fake_include"
        "${MOONCAKE_SOURCE_DIR}/mooncake-common/include"
        "${MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/include"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/thirdparty"
        "${MOONCAKE_SOURCE_DIR}/extern/yalantinglibs/include/ylt/standalone")
    target_link_libraries(tokenspeed_mooncake_discovery PRIVATE "${MOONCAKE_STORE_LIBRARY}" Threads::Threads)
    install(TARGETS tokenspeed_mooncake_discovery LIBRARY DESTINATION lib)
endif()
