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

#include <algorithm>
#include <charconv>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include "full_cache_benchmark.h"

// JSON output shared by the CPU benchmarks: one object per invocation, every
// timing as its raw samples plus their summary, in milliseconds per call.
namespace tokenspeed::benchmark {

inline int ParseNonnegative(std::string_view value) {
    int result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    Require(error == std::errc{} && end == value.data() + value.size() && result >= 0, "expected nonnegative integer");
    return result;
}

inline void PrintDistribution(std::vector<double> values) {
    std::cout << "{\"samples\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
    }
    std::ranges::sort(values);
    const auto n = values.size();
    std::cout << "],\"median\":" << (values[(n - 1) / 2] + values[n / 2]) / 2
              << ",\"p95\":" << values[(n * 95 + 99) / 100 - 1] << ",\"min\":" << values.front()
              << ",\"max\":" << values.back() << '}';
}

template <typename Sample>
void PrintTiming(const char* name, const std::vector<Sample>& samples, Duration Sample::*member) {
    std::vector<double> wall, cpu;
    for (const auto& sample : samples) {
        wall.push_back((sample.*member).wall_ms);
        cpu.push_back((sample.*member).cpu_ms);
    }
    std::cout << ",\"" << name << "\":{\"wall_ms\":";
    PrintDistribution(std::move(wall));
    std::cout << ",\"thread_cpu_ms\":";
    PrintDistribution(std::move(cpu));
    std::cout << '}';
}

}  // namespace tokenspeed::benchmark
