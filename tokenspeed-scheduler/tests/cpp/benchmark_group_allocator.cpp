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

// CPU timing, not an end-to-end serving benchmark. Build/run from the repo root:
//   cmake -S tokenspeed-scheduler -B /tmp/tokenspeed-scheduler-bench \
//     -DCMAKE_BUILD_TYPE=Release -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
//     -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=OFF
//   cmake --build /tmp/tokenspeed-scheduler-bench \
//     --target tokenspeed_scheduler_group_allocator_benchmark -j4
//   /tmp/tokenspeed-scheduler-bench/tokenspeed_scheduler_group_allocator_benchmark
// Run the same source/build settings on each checkout to compare actual Acquire.
// Eight cases: (32 requests x 32768 slots, 64 x 65536), append 1/2, shards 1/4.
// Packing is 4; each table retains two live slots behind a long null prefix.
// Sparse setup establishes the reclaimed frontier before timing. Each timed
// batch only appends and reclaims; setup, address snapshots, validation, summary
// and output are outside timing. The first append can still move the long table;
// it is reported separately and included in eight warmups. Four windows of 32
// steps follow. All cases finish before JSONL output. Address changes estimate
// old-slot reference bytes moved, not payload bytes or measured bandwidth.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <time.h>
#include <vector>

#include "cache/allocator/group_allocator.h"

namespace tokenspeed {
namespace {

constexpr int kPacking = 4, kRetained = 2, kWarmups = 8, kWindows = 4, kSteps = 32;
using Clock = std::chrono::steady_clock;

void Require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "group_allocator benchmark: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::int64_t ThreadCpuNs() {
    timespec time{};
    Require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0, "thread CPU clock unavailable");
    return static_cast<std::int64_t>(time.tv_sec) * 1000000000 + time.tv_nsec;
}

struct Sample {
    std::int64_t wall_ns = 0, thread_cpu_ns = 0;
    std::uint64_t data_address_changes = 0, old_slot_ref_bytes_moved_estimate = 0;
};

struct Result {
    int batch, initial_slots, append, shards;
    Sample first_append;
    std::array<Sample, kWindows * kSteps> steady{};
};

Sample Advance(GroupAllocator& allocator, BlockPool& pool, std::vector<BlockTable>& tables, int append) {
    std::array<std::uintptr_t, 64> addresses{};
    for (std::size_t i = 0; i < tables.size(); ++i) {
        addresses[i] = reinterpret_cast<std::uintptr_t>(tables[i].Blocks().data());
    }
    const int old_slots = tables.front().NumBlocks();
    bool success = true;
    const auto cpu_start = ThreadCpuNs();
    const auto start = Clock::now();
    for (auto& table : tables) {
        success &= allocator.Acquire(pool, table, AcquirePlan{.num_blocks = append},
                                     /*first_placed_slot=*/table.NumBlocks(), /*placed=*/{});
        allocator.ReclaimExpired(pool, table, table.NumBlocks() - kRetained);
    }
    const auto end = Clock::now();
    const auto cpu_end = ThreadCpuNs();
    Require(success, "append exceeds pool capacity");
    Sample sample{std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), cpu_end - cpu_start};
    for (std::size_t i = 0; i < tables.size(); ++i) {
        if (addresses[i] != reinterpret_cast<std::uintptr_t>(tables[i].Blocks().data())) {
            ++sample.data_address_changes;
            sample.old_slot_ref_bytes_moved_estimate += static_cast<std::uint64_t>(old_slots) * sizeof(CacheBlockRef);
        }
    }
    return sample;
}

void Verify(const GroupAllocator& allocator, const BlockPool& pool, const std::vector<BlockTable>& tables, int slots) {
    Require(pool.NumOccupiedSlots() == static_cast<int>(tables.size()) * kRetained, "incorrect live occupancy");
    for (const auto& table : tables) {
        Require(table.NumBlocks() == slots && table.AvailableTokens() == 0, "incorrect logical size or tail capacity");
        Require(table.ReclaimedPrefixBlocks() == slots - kRetained, "incorrect reclaimed frontier");
        const auto ids = allocator.BlockTablePageIds(table);
        Require(ids.size() == static_cast<std::size_t>(slots), "incorrect page table length");
        for (int i = 0; i < slots; ++i) {
            const auto& ref = table.Blocks()[i];
            if (i < slots - kRetained) {
                Require(!ref && ids[i] == 0, "expired slot is not null/zero");
            } else {
                Require(static_cast<bool>(ref), "live tail is null");
                const auto location = ref->Location();
                const auto expected = 1 + (location.lcm_block_id - 1) * kPacking + location.slot_index;
                Require(ref.IsOwnedBy(pool) && ids[i] == expected && ids[i] > 0, "incorrect live page mapping");
                Require(allocator.BucketOf(location) == (ids[i] - 1) % allocator.ShardCount(),
                        "incorrect shard mapping");
            }
        }
    }
}

Result Measure(int batch, int initial_slots, int append, int shards) {
    const int parents = batch * 4 + 16;
    BlockPool pool(parents, {kPacking});
    pool.RegisterGroup(0, kPacking, shards);
    GroupAllocator allocator(kPacking, 0, shards);
    std::vector<BlockTable> tables(batch);
    for (auto& table : tables) {
        Require(allocator.Acquire(pool, table,
                                  AcquirePlan{.num_blocks = kRetained,
                                              .suffix_start = initial_slots - kRetained,
                                              .table_blocks_after = initial_slots},
                                  /*first_placed_slot=*/table.NumBlocks(), /*placed=*/{}),
                "sparse setup failed");
    }
    Verify(allocator, pool, tables, initial_slots);
    Result result{batch, initial_slots, append, shards, Advance(allocator, pool, tables, append)};
    Verify(allocator, pool, tables, initial_slots + append);
    for (int i = 1; i < kWarmups; ++i) {
        Advance(allocator, pool, tables, append);
    }
    Verify(allocator, pool, tables, initial_slots + kWarmups * append);
    for (int window = 0; window < kWindows; ++window) {
        for (int step = 0; step < kSteps; ++step) {
            result.steady[window * kSteps + step] = Advance(allocator, pool, tables, append);
        }
        Verify(allocator, pool, tables, initial_slots + (kWarmups + (window + 1) * kSteps) * append);
    }
    for (auto& table : tables) {
        allocator.Free(table);
        Require(table.NumBlocks() == 0 && table.ReclaimedPrefixBlocks() == 0 && table.AvailableTokens() == 0,
                "Free did not reset table");
    }
    Require(pool.NumOccupiedSlots() == 0 && pool.NumEmptyLcmBlocks() == parents, "Free leaked pool slots");
    return result;
}

void PrintSample(const Sample& sample) {
    std::cout << "{\"wall_ns\":" << sample.wall_ns << ",\"thread_cpu_ns\":" << sample.thread_cpu_ns
              << ",\"data_address_changes\":" << sample.data_address_changes
              << ",\"old_slot_ref_bytes_moved_estimate\":" << sample.old_slot_ref_bytes_moved_estimate << '}';
}

void PrintSummary(const Result& result, std::int64_t Sample::*member) {
    std::array<std::int64_t, kWindows * kSteps> ordered{};
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        ordered[i] = result.steady[i].*member;
    }
    const double mean = std::accumulate(ordered.begin(), ordered.end(), 0.0) / ordered.size();
    std::ranges::sort(ordered);
    const double median = (ordered[ordered.size() / 2 - 1] + ordered[ordered.size() / 2]) / 2.0;
    // Nearest-rank p95 of the 128 timed batch samples.
    std::cout << "{\"mean\":" << mean << ",\"median\":" << median
              << ",\"p95\":" << ordered[(ordered.size() * 95 + 99) / 100 - 1] << '}';
}

void Report(const Result& result) {
    std::uint64_t changes = 0, bytes = 0;
    for (const auto& sample : result.steady) {
        changes += sample.data_address_changes;
        bytes += sample.old_slot_ref_bytes_moved_estimate;
    }
    std::cout << std::setprecision(12) << "{\"benchmark\":\"group_allocator\",\"batch\":" << result.batch
              << ",\"initial_logical_slots_per_request\":" << result.initial_slots
              << ",\"append_per_step\":" << result.append << ",\"shards\":" << result.shards
              << ",\"packing\":" << kPacking << ",\"retained_live_tail\":" << kRetained
              << ",\"pool_parents\":" << result.batch * 4 + 16
              << ",\"sizeof_cache_block_ref\":" << sizeof(CacheBlockRef) << ",\"warmup_steps\":" << kWarmups
              << ",\"windows\":" << kWindows << ",\"steps_per_window\":" << kSteps << ",\"first_append\":";
    PrintSample(result.first_append);
    std::cout << ",\"steady_wall_ns\":";
    PrintSummary(result, &Sample::wall_ns);
    std::cout << ",\"steady_thread_cpu_ns\":";
    PrintSummary(result, &Sample::thread_cpu_ns);
    std::cout << ",\"steady_data_address_changes\":" << changes
              << ",\"steady_old_slot_ref_bytes_moved_estimate\":" << bytes << ",\"final_logical_slots_per_request\":"
              << result.initial_slots + (kWarmups + kWindows * kSteps) * result.append
              << ",\"final_occupied_slots\":0,\"final_empty_parents\":" << result.batch * 4 + 16
              << ",\"steady_samples\":[";
    for (std::size_t i = 0; i < result.steady.size(); ++i) {
        std::cout << (i ? "," : "");
        PrintSample(result.steady[i]);
    }
    std::cout << "]}\n";
}

}  // namespace
}  // namespace tokenspeed

int main() {
    std::vector<tokenspeed::Result> results;
    for (int batch : {32, 64}) {
        for (int append : {1, 2}) {
            for (int shards : {1, 4}) {
                results.push_back(tokenspeed::Measure(batch, batch * 1024, append, shards));
            }
        }
    }
    for (const auto& result : results) {
        tokenspeed::Report(result);
    }
    return 0;
}
