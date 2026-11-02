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

// CPU-only sorter benchmark; excludes source-vector preparation. This does not
// measure complete admission, request latency or GPU serving performance.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include "cache/coordinator/admission_order.h"

int main() {
    using namespace tokenspeed;
    std::mt19937_64 random(20261101);
    std::uint64_t checksum = 0;
    std::cout << "n,distribution,sample,comparison_ns,radix_ns\n";
    for (const int size : {64, 128, 640, 2560, 100000}) {
        for (const std::string_view distribution : {"dense", "sparse", "equal", "mixed"}) {
            std::vector<AdmissionVictimCandidate> source;
            for (int i = 0; i < size; ++i) {
                const std::int64_t position =
                    distribution == "equal"
                        ? 0
                        : random() % (distribution == "sparse" ? 2147483648ULL : static_cast<std::uint64_t>(size));
                source.push_back({.group_id = 0,
                                  .location = {.lcm_block_id = i / 4, .slot_index = i % 4},
                                  .last_access_epoch = 1,
                                  .eviction_tier = distribution == "mixed" && i % 2
                                                       ? AdmissionEvictionTier::kEstablishedBoundary
                                                       : AdmissionEvictionTier::kClosedPrefix,
                                  .position_rank = -position});
            }
            auto expected = source;
            std::ranges::sort(expected, {}, AdmissionEvictionKey);
            auto actual = source;
            SortAdmissionGroupEpoch(actual);
            for (int i = 0; i < size; ++i) {
                if (AdmissionEvictionKey(expected[i]) != AdmissionEvictionKey(actual[i])) return 1;
            }
            const int repetitions = std::max(10, 100000 / size);
            for (int sample = 0; sample < 9; ++sample) {
                double elapsed[2]{};
                for (int arm_index = 0; arm_index < 2; ++arm_index) {
                    const int arm = (sample + arm_index) % 2;
                    for (int repeat = 0; repeat < repetitions; ++repeat) {
                        auto candidates = source;
                        const auto start = std::chrono::steady_clock::now();
                        if (arm == 0)
                            std::ranges::sort(candidates, {}, AdmissionEvictionKey);
                        else
                            SortAdmissionGroupEpoch(candidates);
                        elapsed[arm] +=
                            std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
                        checksum += static_cast<std::uint64_t>(candidates[repeat % size].location.lcm_block_id);
                    }
                }
                std::cout << size << ',' << distribution << ',' << sample << ',' << elapsed[0] / repetitions << ','
                          << elapsed[1] / repetitions << '\n';
            }
        }
    }
    std::cerr << "checksum=" << checksum << '\n';
}
