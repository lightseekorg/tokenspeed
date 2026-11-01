# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# The scheduler core and its Python module as static libraries for a binary
# that embeds the interpreter: the module is registered as a built-in, so the
# objects the binary creates and the ones Python reads come from one build.

cmake_minimum_required(VERSION 3.22)
project(ts_scheduler_embed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED TOKENSPEED_SCHEDULER_DIR)
    message(FATAL_ERROR "TOKENSPEED_SCHEDULER_DIR must point at tokenspeed-scheduler/")
endif()

# The scheduler's own build finds these too, in its directory scope; the
# module below needs them here.
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

set(TOKENSPEED_SCHEDULER_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(TOKENSPEED_SCHEDULER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${TOKENSPEED_SCHEDULER_DIR}" scheduler)

nanobind_build_library(nanobind-static)

add_library(ts_scheduler_embed STATIC
    "${TOKENSPEED_SCHEDULER_DIR}/bindings/python_module.cpp"
    cxx/embed.cc
)
nanobind_compile_options(ts_scheduler_embed)
target_include_directories(ts_scheduler_embed PRIVATE "${TOKENSPEED_SCHEDULER_DIR}/csrc")
target_link_libraries(ts_scheduler_embed PUBLIC tokenspeed_scheduler_core nanobind-static Python::Module)
