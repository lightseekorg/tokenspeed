// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
namespace tokenspeed_marlin_detail {
template <class... Args>
[[noreturn]] inline void check_fail(const char* condition, Args&&... args) {
  std::ostringstream message;
  message << "Marlin MXFP8 check failed: " << condition << ": ";
  (message << ... << std::forward<Args>(args));
  throw std::runtime_error(message.str());
}
}
#define STD_TORCH_CHECK(condition, ...) \
  do { if (!(condition)) { \
    ::tokenspeed_marlin_detail::check_fail(#condition __VA_OPT__(,) __VA_ARGS__); \
  } } while (false)
