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

# xgrammar's C++ core as a static library, from the source release of the
# version the Python package pins (python/pyproject.toml). The wheels ship
# neither headers nor the library. Set FETCHCONTENT_SOURCE_DIR_XGRAMMAR to an
# unpacked source tree to build offline.
cmake_minimum_required(VERSION 3.18)
project(ts_grammar LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  xgrammar
  URL https://files.pythonhosted.org/packages/03/0f/29f521535729be740779c2c98b630ef6607cf28be9b669895dd2b821ff63/xgrammar-0.2.8.tar.gz
  URL_HASH SHA256=8305646fd2b319d6c3003f51a1665e54cbbe9cba1fb636bc41d0b46434ccbb80
)
# xgrammar reads its options from config.cmake in the top binary directory
# first: the core only, no Python bindings.
file(WRITE ${CMAKE_BINARY_DIR}/config.cmake
     "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n"
     "set(XGRAMMAR_BUILD_CXX_TESTS OFF)\n"
     "set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n"
     "set(XGRAMMAR_ENABLE_COVERAGE OFF)\n"
     "set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF)\n")
# xgrammar builds with -flto=auto and -Werror. LTO objects in a static
# archive need the LTO plugin at the final link, which Cargo's linker
# invocation does not bring; a newer compiler's warnings must not fail it.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto -Wno-error")
FetchContent_MakeAvailable(xgrammar)

# Where build.rs finds the headers.
file(WRITE ${CMAKE_BINARY_DIR}/xgrammar_source_dir.txt "${xgrammar_SOURCE_DIR}")
