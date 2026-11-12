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

// CPU microbenchmark of ForwardBatch construction. Prefix lengths are
// logical slots in the sliding group. The workload also has a dense History
// group and a two-slot State group with occasional prefill checkpoints.
// Run without arguments for the sweep, or pass BATCH RELEASED_PREFIX_SLOTS
// [REPEATS] to select one point. Use identical source on both revisions.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <time.h>
#include <utility>
#include <variant>
#include <vector>

#include "cache/allocator/group_allocator.h"
#include "scheduler/operations/forward.h"

namespace tokenspeed {
namespace {

constexpr int kWarmups = 4;
constexpr int kRepeats = 32;
constexpr int kFullColumns = 256;
using Clock = std::chrono::steady_clock;

void Require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::int64_t ThreadCpuNs() {
    timespec time{};
    Require(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0, "thread CPU clock unavailable");
    return static_cast<std::int64_t>(time.tv_sec) * 1000000000 + time.tv_nsec;
}

struct TableSize {
    std::size_t rows = 0;
    std::size_t absolute_entries = 0;
};

TableSize Verify(const ForwardBatch& batch, const std::vector<ForwardOperation>& ops,
                 const std::vector<std::size_t>& order) {
    Require(batch.request_ids.size() == order.size(), "wrong request count");
    const auto tables = batch.BlockTables();
    Require(tables.size() == 3, "wrong cache group count");
    TableSize size;
    for (const auto& [group, rows] : tables) {
        Require(rows.size() == order.size(), "wrong group row count");
        std::size_t width = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            const auto& op =
                std::visit([](const auto& inner) -> const ForwardOperationBase& { return inner; }, ops[order[i]]);
            Require(batch.request_ids[i] == op.request_id, "wrong request order");
            const auto& row = rows[i];
            std::size_t pages = 0;
            if (const auto it = op.block_tables.find(group); it != op.block_tables.end()) {
                const auto& source = it->second;
                const auto expected = source.allocator.BlockTablePageIds(source.table);
                pages = expected.size() + (source.checkpoint_page_id != 0);
                Require(row.size() >= pages, "truncated block table row");
                Require(std::equal(expected.begin(), expected.end(), row.begin()), "wrong allocator page ids");
                Require(std::all_of(row.begin(), row.begin() + source.table.ReclaimedPrefixBlocks(),
                                    [](std::int32_t page) { return page == 0; }),
                        "released prefix must contain page zero");
                if (source.checkpoint_page_id != 0) {
                    Require(row[expected.size()] == source.checkpoint_page_id, "wrong checkpoint page id");
                }
            }
            Require(std::all_of(row.begin() + pages, row.end(), [](std::int32_t page) { return page == -1; }),
                    "wrong missing row or tail padding");
            width = std::max(width, pages);
            size.absolute_entries += row.size();
        }
        Require(width > 0 &&
                    std::all_of(rows.begin(), rows.end(), [width](const auto& row) { return row.size() == width; }),
                "wrong padded block table width");
        size.rows += rows.size();
    }
    return size;
}

void PrintSummary(std::vector<std::int64_t> samples) {
    std::ranges::sort(samples);
    const std::size_t n = samples.size();
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / n;
    const double median = (samples[(n - 1) / 2] + samples[n / 2]) / 2.0;
    std::cout << "{\"mean\":" << mean << ",\"median\":" << median << ",\"p95\":" << samples[(n * 95 + 99) / 100 - 1]
              << '}';
}

void Run(int batch_size, int released_prefix_slots, int repeats) {
    constexpr int kHistoryPacking = 4, kStatePacking = 2;
    constexpr std::uint32_t kFull = 0, kWindow = 1, kState = 2;
    BlockPool pool(batch_size * (kFullColumns / kHistoryPacking + 8),
                   {kHistoryPacking, kHistoryPacking, kStatePacking});
    GroupAllocator full(kHistoryPacking, kFull, 1);
    GroupAllocator window(kHistoryPacking, kWindow, 1);
    GroupAllocator state(kStatePacking, kState, 1);
    std::vector<BlockTable> full_tables(batch_size), window_tables(batch_size), state_tables(batch_size);
    std::vector<CacheBlockRef> checkpoints(batch_size);
    std::vector<ForwardOperation> ops;
    ops.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        const bool prefill = i % 8 == 0;
        if (prefill) {
            PrefillOperation extend;
            static constexpr std::int32_t kInput[] = {1, 2, 3, 4};
            static constexpr std::int32_t kShifted[] = {2, 3, 4, 5};
            extend.input_ids = kInput;
            extend.shifted_input_ids = kShifted;
            extend.extend_state_checkpoint_len = 2;
            ops.emplace_back(std::move(extend));
        } else {
            DecodeOperation decode;
            decode.decode_input_id = 1;
            ops.emplace_back(std::move(decode));
        }
        auto& op = std::visit([](auto& inner) -> ForwardOperationBase& { return inner; }, ops.back());
        op.request_id = "request-" + std::to_string(i);
        op.request_pool_index = i;
        op.input_length = 4;

        Require(full.Acquire(pool, full_tables[i], AcquirePlan{.num_blocks = kFullColumns - i % 4}, 0, {}),
                "full table setup failed");
        op.block_tables.emplace("full", ForwardBlockTable{full, full_tables[i], 0});
        if (i % 16 != 15) {
            const int released = released_prefix_slots - (i / 4) % 4;
            const int retained = 2 + i % 4;
            const AcquirePlan plan{
                .num_blocks = retained, .suffix_start = released, .table_blocks_after = released + retained};
            Require(window.Acquire(pool, window_tables[i], plan, 0, {}), "sliding table setup failed");
            Require(window_tables[i].ReclaimedPrefixBlocks() == released, "wrong released prefix length");
            op.block_tables.emplace("window", ForwardBlockTable{window, window_tables[i], 0});
        }
        if (i % 16 != 14) {
            state.ShapeStateTable(state_tables[i], {});
            if (i % 2 != 0) {
                Require(state.AcquireStateInput(pool, state_tables[i]), "state input setup failed");
            }
            Require(state.AcquireStateLive(pool, state_tables[i]), "live state setup failed");
            std::int32_t checkpoint_page_id = 0;
            if (prefill) {
                checkpoints[i] = state.AcquireStateCheckpoint(pool, state_tables[i]);
                Require(static_cast<bool>(checkpoints[i]), "checkpoint setup failed");
                checkpoint_page_id = state.ResolveCacheBlockId(*checkpoints[i]);
                Require(checkpoint_page_id > 0, "checkpoint needs a physical page");
            }
            op.block_tables.emplace("state", ForwardBlockTable{state, state_tables[i], checkpoint_page_id});
        }
    }

    std::vector<std::size_t> order;
    order.reserve(batch_size);
    for (bool prefill : {true, false}) {
        for (std::size_t i = 0; i < ops.size(); ++i) {
            if ((i % 8 == 0) == prefill) {
                order.push_back(i);
            }
        }
    }

    // Check every table entry before the warmups and again after the last
    // sample. Intermediate batches escape through a compiler barrier so the
    // constructor stays observable without a competing full-table scan.
    const TableSize size = Verify(ForwardBatch{ops}, ops, order);
    for (int i = 0; i < kWarmups; ++i) {
        const ForwardBatch batch{ops};
        asm volatile("" : : "g"(&batch) : "memory");
    }

    std::vector<std::int64_t> wall, cpu;
    wall.reserve(repeats);
    cpu.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        auto input = ops;
        const auto cpu_start = ThreadCpuNs();
        const auto start = Clock::now();
        const ForwardBatch batch{std::move(input)};
        const auto end = Clock::now();
        const auto cpu_end = ThreadCpuNs();
        asm volatile("" : : "g"(&batch) : "memory");
        wall.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        cpu.push_back(cpu_end - cpu_start);
        if (i == repeats - 1) {
            Verify(batch, ops, order);
        }
    }

    std::cout << "{\"benchmark\":\"forward_batch_constructor\",\"batch\":" << batch_size
              << ",\"max_released_prefix_slots\":" << released_prefix_slots << ",\"table_rows\":" << size.rows
              << ",\"absolute_entries\":" << size.absolute_entries << ",\"warmups\":" << kWarmups
              << ",\"repeats\":" << repeats << ",\"wall_ns\":";
    PrintSummary(std::move(wall));
    std::cout << ",\"thread_cpu_ns\":";
    PrintSummary(std::move(cpu));
    std::cout << "}\n";
}

int ParsePositive(std::string_view value) {
    int result = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    Require(error == std::errc{} && end == value.data() + value.size() && result > 0, "expected a positive integer");
    return result;
}

}  // namespace
}  // namespace tokenspeed

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            for (int batch : {64, 128, 256}) {
                for (int prefix : {65536, 131072}) {
                    tokenspeed::Run(batch, prefix, tokenspeed::kRepeats);
                }
            }
        } else {
            tokenspeed::Require(argc == 3 || argc == 4, "usage: benchmark [BATCH RELEASED_PREFIX_SLOTS [REPEATS]]");
            const int batch = tokenspeed::ParsePositive(argv[1]);
            const int prefix = tokenspeed::ParsePositive(argv[2]);
            tokenspeed::Require(batch >= 64 && batch <= 256 && prefix >= 4 && prefix <= 131072,
                                "require 64 <= BATCH <= 256 and 4 <= RELEASED_PREFIX_SLOTS <= 131072");
            tokenspeed::Run(batch, prefix, argc == 4 ? tokenspeed::ParsePositive(argv[3]) : tokenspeed::kRepeats);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "forward_batch benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
