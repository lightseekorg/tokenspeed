cmake_minimum_required(VERSION 3.22)
project(tokenspeed_mooncake_engine LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
foreach(required MOONCAKE_SOURCE_DIR MOONCAKE_LIBRARY_DIR ASCEND_HOME)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} must name the selected SDK installation")
  endif()
endforeach()
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
add_library(tokenspeed_transfer_engine SHARED transfer_engine_tokenspeed.cpp)
target_compile_definitions(tokenspeed_transfer_engine PRIVATE USE_ASCEND_DIRECT)
target_include_directories(tokenspeed_transfer_engine PRIVATE
  "${MOONCAKE_SOURCE_DIR}/mooncake-transfer-engine/include"
  "${MOONCAKE_SOURCE_DIR}/mooncake-common/include"
  "${ASCEND_HOME}/include" ${JSONCPP_INCLUDE_DIRS})
target_link_directories(tokenspeed_transfer_engine PRIVATE
  "${MOONCAKE_LIBRARY_DIR}" "${ASCEND_HOME}/lib64")
target_link_libraries(tokenspeed_transfer_engine PRIVATE transfer_engine ascendcl Threads::Threads)
set_target_properties(tokenspeed_transfer_engine PROPERTIES
  INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
