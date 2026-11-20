cmake_minimum_required(VERSION 3.18)
project(ascend_memory_saver LANGUAGES CXX)

option(SAVER_FAKE_ACL "Build only host-side state-machine tests" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

if(SAVER_FAKE_ACL)
  set(saver_target saver_state_tests)
  add_executable(${saver_target} src/ascend_saver.cpp tests/fake_acl.cpp tests/state_tests.cpp)
  target_compile_definitions(${saver_target} PRIVATE SAVER_FAKE_ACL=1)
  target_include_directories(${saver_target} PRIVATE tests)
  enable_testing()
  add_test(NAME saver_state_tests COMMAND ${saver_target})
  # Each callback fault is terminal for its process.
  add_test(NAME saver_void_free_wrong_context COMMAND ${saver_target} wrong-context)
  add_test(NAME saver_void_free_wrong_device COMMAND ${saver_target} wrong-device)
  add_test(NAME saver_failed_malloc_rollback COMMAND ${saver_target} malloc-rollback)
else()
  set(saver_target ascend_memory_saver)
  add_library(${saver_target} SHARED src/ascend_saver.cpp)
  set_target_properties(${saver_target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
  if(NOT ASCEND_HOME_PATH)
    message(FATAL_ERROR "Set ASCEND_HOME_PATH to the target CANN installation")
  endif()
  find_path(ACL_INCLUDE_DIR acl/acl_rt.h PATHS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    NO_DEFAULT_PATH REQUIRED)
  find_library(ACL_RT_LIBRARY acl_rt PATHS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    NO_DEFAULT_PATH REQUIRED)
  target_include_directories(${saver_target} PRIVATE "${ACL_INCLUDE_DIR}")
  target_link_libraries(${saver_target} PRIVATE "${ACL_RT_LIBRARY}")
  target_link_options(${saver_target} PRIVATE "LINKER:--no-undefined")
  include(GNUInstallDirs)
  install(TARGETS ${saver_target} LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

target_include_directories(${saver_target} PRIVATE include)
target_link_libraries(${saver_target} PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(${saver_target} PRIVATE -Wall -Wextra -Wpedantic)
endif()
