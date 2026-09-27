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

// Timing of BlockPool balanced allocation and release. Not a test: it prints
// JSON and asserts only that the workload it timed was the intended one.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cache/allocator/group_allocator.h"
#include "cache/core/acquire_plan.h"
#include "cache/core/block_pool.h"
#include "cache/core/block_table.h"

namespace tokenspeed {
namespace {

void Require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "block_pool benchmark: " << message << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Reproduce with the same optimized build and an otherwise idle CPU:
//   cmake -S tokenspeed-scheduler -B /tmp/tokenspeed-scheduler-bench \
//     -DCMAKE_BUILD_TYPE=Release -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
//     -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
//   cmake --build /tmp/tokenspeed-scheduler-bench \
//     --target tokenspeed_scheduler_block_pool_benchmark -j4
//   /tmp/tokenspeed-scheduler-bench/tokenspeed_scheduler_block_pool_benchmark
// One warmup and seven samples per case; each sample grows 16 requests by one
// block for 64 rounds, then frees all requests. The fragmented case starts with
// one hole per seeded parent. Setup, seeding, assertions and final seed cleanup
// are outside the timed regions. JSON reports raw batch ns and median ns/block;
// timings are informational, with no performance threshold in CI.
void AllocateAndRelease() {
    using Clock = std::chrono::steady_clock;
    constexpr std::int32_t packing = 16;
    constexpr std::int32_t buckets = 4;
    constexpr int requests = 16;
    constexpr int rounds = 64;
    constexpr int samples = 7;
    constexpr int operations = requests * rounds;
    for (std::int32_t parents : {256, 4096}) {
        for (std::int32_t fragmented : {0, parents * 3 / 4}) {
            std::array<std::int64_t, samples> acquire_ns{};
            std::array<std::int64_t, samples> release_ns{};
            for (int sample = -1; sample < samples; ++sample) {
                BlockPool pool(parents, {packing});
                pool.RegisterGroup(0, packing, buckets);
                GroupAllocator allocator(packing, 0, buckets);
                auto seeds = pool.AcquireBlocks(0, fragmented * packing, std::vector<std::int32_t>(buckets, 0));
                Require(seeds.size() == static_cast<std::size_t>(fragmented * packing), "seeding failed");
                for (auto& seed : seeds) {
                    const auto location = seed->Location();
                    if (location.slot_index == (location.lcm_block_id - 1) % packing) {
                        seed.reset();
                    }
                }
                std::array<BlockTable, requests> tables;
                const auto start = Clock::now();
                for (int round = 0; round < rounds; ++round) {
                    for (auto& table : tables) {
                        Require(allocator.Acquire(pool, table, AcquirePlan{.num_blocks = 1}),
                                "benchmark workload exceeds configured capacity");
                    }
                }
                const auto acquired = Clock::now();
                for (const auto& table : tables) {
                    Require(table.NumBlocks() == rounds, "request table did not grow by one block per round");
                }
                const auto release_start = Clock::now();
                for (auto& table : tables) {
                    allocator.Free(table);
                }
                const auto released = Clock::now();
                Require(pool.NumOccupiedSlots() == fragmented * (packing - 1), "seed holes were not preserved");
                seeds.clear();
                Require(pool.NumEmptyLcmBlocks() == parents, "parents were not all released");
                Require(pool.NumOccupiedSlots() == 0, "slots were not all released");
                if (sample >= 0) {
                    acquire_ns[sample] = std::chrono::duration_cast<std::chrono::nanoseconds>(acquired - start).count();
                    release_ns[sample] =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(released - release_start).count();
                }
            }
            const auto report = [&](const char* operation, const auto& timings) {
                auto ordered = timings;
                std::ranges::sort(ordered);
                std::cout << "{\"benchmark\":\"block_pool\",\"operation\":\"" << operation
                          << "\",\"parents\":" << parents << ",\"packing\":" << packing << ",\"buckets\":" << buckets
                          << ",\"fragmented_parents\":" << fragmented << ",\"requests\":" << requests
                          << ",\"rounds\":" << rounds << ",\"warmups\":1,\"samples\":" << samples
                          << ",\"blocks_per_sample\":" << operations
                          << ",\"median_ns_per_block\":" << static_cast<double>(ordered[samples / 2]) / operations
                          << ",\"batch_ns\":[";
                for (int i = 0; i < samples; ++i) {
                    std::cout << (i ? "," : "") << timings[i];
                }
                std::cout << "]}" << std::endl;
            };
            report("acquire", acquire_ns);
            report("release", release_ns);
        }
    }
}

}  // namespace
}  // namespace tokenspeed

int main() {
    tokenspeed::AllocateAndRelease();
    return 0;
}
