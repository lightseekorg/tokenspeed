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

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {
namespace {

constexpr std::int32_t kDemand = 4;
constexpr std::int32_t kPacking = 4;
constexpr std::array<std::int32_t, 3> kPoolSizes{1000, 10000, 100000};
constexpr std::int32_t kRepeats = 5;

std::uint64_t ConsumeLocations(const std::vector<CacheBlockRef>& blocks) {
    std::uint64_t checksum = 0;
    for (const CacheBlockRef& block : blocks) {
        const CacheBlockLocation location = block->Location();
        checksum += static_cast<std::uint64_t>(location.lcm_block_id) * 17U +
                    static_cast<std::uint64_t>(location.slot_index + 1);
    }
    return checksum;
}

std::uint64_t ConsumeLocations(const std::vector<PrefixCacheIndex::EvictionCandidate>& candidates) {
    std::uint64_t checksum = 0;
    for (const PrefixCacheIndex::EvictionCandidate& candidate : candidates) {
        checksum += static_cast<std::uint64_t>(candidate.location.lcm_block_id) * 17U +
                    static_cast<std::uint64_t>(candidate.location.slot_index + 1);
    }
    return checksum;
}

template <typename Operation>
void Measure(const char* workload, std::int32_t pool_size, std::int32_t demand, std::int32_t iterations,
             Operation&& operation) {
    std::array<double, kRepeats> timings{};
    std::uint64_t checksum = 0;
    for (std::int32_t repeat = 0; repeat < kRepeats; ++repeat) {
        std::uint64_t repeat_checksum = 0;
        const auto begin = std::chrono::steady_clock::now();
        for (std::int32_t iteration = 0; iteration < iterations; ++iteration) {
            repeat_checksum += operation();
        }
        const auto end = std::chrono::steady_clock::now();
        timings[static_cast<std::size_t>(repeat)] =
            std::chrono::duration<double, std::nano>(end - begin).count() / iterations;
        checksum += repeat_checksum;
    }
    std::ranges::sort(timings);
    std::cout << workload << ',' << pool_size << ',' << demand << ',' << iterations << ',' << timings[kRepeats / 2]
              << ',' << checksum << '\n';
}

std::int32_t IterationsFor(std::int32_t pool_size) {
    return pool_size <= 1000 ? 10000 : (pool_size <= 10000 ? 3000 : 1000);
}

void MeasureAcquireBlocksPackingOne(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    Measure("acquire_blocks_packing_1", pool_size, kDemand, iterations, [&] {
        std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/1, /*num=*/kDemand);
        const std::uint64_t checksum = ConsumeLocations(blocks);
        blocks.clear();
        return checksum;
    });
}

std::vector<CacheBlockRef> MakeOneHolePerParent(BlockPool& pool, std::int32_t pool_size) {
    std::vector<CacheBlockRef> retained =
        pool.AcquireBlocks(/*group_id=*/0, /*packing=*/kPacking, /*num=*/pool_size * kPacking);
    if (static_cast<std::int32_t>(retained.size()) != pool_size * kPacking) {
        std::abort();
    }
    for (std::int32_t i = kPacking - 1; i < pool_size * kPacking; i += kPacking) {
        retained[static_cast<std::size_t>(i)].reset();
    }
    return retained;
}

void MeasureFragmentedAllocations(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    std::vector<CacheBlockRef> retained = MakeOneHolePerParent(pool, pool_size);
    Measure("acquire_blocks_packing_4_one_hole_per_parent", pool_size, kDemand, iterations, [&] {
        std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/kPacking, /*num=*/kDemand);
        const std::uint64_t checksum = ConsumeLocations(blocks);
        blocks.clear();
        return checksum;
    });
}

void MeasureFullAllocations(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size + kDemand);
    std::vector<CacheBlockRef> retained =
        pool.AcquireBlocks(/*group_id=*/0, /*packing=*/kPacking, /*num=*/pool_size * kPacking);
    Measure("acquire_blocks_packing_4_full_parents", pool_size, kDemand, iterations, [&] {
        std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/kPacking, /*num=*/kDemand);
        const std::uint64_t checksum = ConsumeLocations(blocks);
        blocks.clear();
        return checksum;
    });
}

void MeasureLargeBatchControlMaintenance(std::int32_t pool_size, std::int32_t iterations) {
    const std::int32_t demand = std::min<std::int32_t>(512, pool_size);
    BlockPool pool(pool_size);
    std::vector<CacheBlockRef> retained = MakeOneHolePerParent(pool, pool_size);
    Measure("acquire_blocks_packing_4_large_batch_one_hole_per_parent", pool_size, demand, iterations, [&] {
        std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/kPacking, /*num=*/demand);
        const std::uint64_t checksum = ConsumeLocations(blocks);
        blocks.clear();
        return checksum;
    });
}

void MeasureHostStyleAcquire(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    std::vector<CacheBlockRef> retained = MakeOneHolePerParent(pool, pool_size);
    const std::array<std::uint32_t, kDemand> group_ids{0, 0, 0, 0};
    const std::array<std::int32_t, 1> packings{kPacking};
    Measure("acquire_available_in_order_one_hole_per_parent", pool_size, kDemand, iterations, [&] {
        std::vector<CacheBlockRef> blocks = pool.AcquireAvailableBlocksInOrder(group_ids, packings);
        const std::uint64_t checksum = ConsumeLocations(blocks);
        blocks.clear();
        return checksum;
    });
}

void MeasureEvictableCandidates(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    PrefixCacheIndex index(/*group_id=*/0);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/1, /*num=*/pool_size);
    if (static_cast<std::int32_t>(blocks.size()) != pool_size) {
        std::abort();
    }
    for (std::int32_t i = 0; i < pool_size; ++i) {
        index.Register(pool, blocks[static_cast<std::size_t>(i)],
                       CacheKey{.group_id = 0, .content_hash = std::to_string(i), .page_offset = 0},
                       /*access_epoch=*/static_cast<std::uint64_t>(i));
    }
    blocks.clear();
    Measure("evictable_candidates", pool_size, pool_size, iterations,
            [&] { return ConsumeLocations(index.EvictableCandidates(pool)); });
}

CacheCoordinator MakeAdmissionCoordinator(BlockPool& pool) {
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    return MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
}

void FillAdmissionCache(CacheCoordinator& coordinator, BlockPool& pool, std::int32_t pool_size) {
    for (std::int32_t i = 0; i < pool_size; ++i) {
        CacheBlockRef block = pool.AcquireBlock(/*group_id=*/0, /*packing=*/1);
        if (!block) {
            std::abort();
        }
        coordinator.GroupPrefixIndex(0).Register(
            pool, block,
            CacheKey{.group_id = 0, .content_hash = "admission-cache-" + std::to_string(i), .page_offset = 0},
            /*access_epoch=*/static_cast<std::uint64_t>(i + 1));
    }
}

void MeasureAdmission(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    CacheCoordinator coordinator = MakeAdmissionCoordinator(pool);
    FillAdmissionCache(coordinator, pool, pool_size);
    BlockTable table;
    std::array<GroupDemand, 1> no_demand{GroupDemand{.table = &table}};
    Measure("admit_no_demand_cached_pool", pool_size, 0, iterations, [&] {
        const std::optional<CacheCoordinator::AdmissionResult> result =
            coordinator.Admit(coordinator.ProbePrefix({}), no_demand);
        if (!result) {
            std::abort();
        }
        return result->access_epoch;
    });

    std::array<GroupDemand, 1> demand{GroupDemand{.table = &table, .num_tokens = 4}};
    std::uint64_t next_key = static_cast<std::uint64_t>(pool_size);
    Measure("admit_small_demand_evict_and_restore", pool_size, 1, iterations, [&] {
        const std::optional<CacheCoordinator::AdmissionResult> result =
            coordinator.Admit(coordinator.ProbePrefix({}), demand);
        if (!result) {
            std::abort();
        }
        std::uint64_t checksum = result->access_epoch;
        for (const std::vector<std::int32_t>& page_ids : result->new_page_ids) {
            for (std::int32_t page_id : page_ids) {
                checksum += static_cast<std::uint64_t>(page_id);
            }
        }
        coordinator.Free(std::span{&table, std::size_t{1}});
        CacheBlockRef block = pool.AcquireBlock(/*group_id=*/0, /*packing=*/1);
        if (!block) {
            std::abort();
        }
        ++next_key;
        coordinator.GroupPrefixIndex(0).Register(
            pool, block,
            CacheKey{.group_id = 0, .content_hash = "admission-cache-" + std::to_string(next_key), .page_offset = 0},
            /*access_epoch=*/next_key);
        return checksum;
    });
}

// Distinct request epochs with pinned cache entries force admission to inspect
// the entire index before rejecting. Setup and pins stay outside the timing.
void MeasurePinnedAdmission(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    CacheCoordinator coordinator = MakeAdmissionCoordinator(pool);
    std::vector<CacheBlockRef> pins =
        pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/pool_size);
    for (std::int32_t i = 0; i < pool_size; ++i) {
        coordinator.GroupPrefixIndex(0).Register(
            pool, pins[static_cast<std::size_t>(i)],
            CacheKey{.group_id = 0, .content_hash = "pinned-" + std::to_string(i), .page_offset = 0},
            /*access_epoch=*/static_cast<std::uint64_t>(i + 1), /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
            /*newly_cached=*/nullptr);
    }
    BlockTable table;
    const std::array demand{GroupDemand{.table = &table, .num_tokens = 4}};
    Measure("admit_small_demand_all_pinned", pool_size, 1, iterations, [&] {
        const auto result = coordinator.Admit(coordinator.ProbePrefix({}), demand, std::nullopt);
        if (result) {
            std::abort();
        }
        return std::uint64_t{1};
    });
}

void MeasureAvailableLcmBlocks(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(pool_size);
    CacheCoordinator coordinator = MakeAdmissionCoordinator(pool);
    FillAdmissionCache(coordinator, pool, pool_size);
    Measure("num_available_lcm_blocks_cached_pool", pool_size, pool_size, iterations,
            [&] { return static_cast<std::uint64_t>(coordinator.NumAvailableLcmBlocks()); });
}

// Every Host parent is cached, so the batch can only be served by evicting.
// Each iteration drops one cache entry and restores it, keeping the tier full.
void MeasureHostBlockAcquisition(std::int32_t pool_size, std::int32_t iterations) {
    BlockPool pool(/*num_lcm_blocks=*/1);
    BlockPool host_pool(pool_size);
    const std::array specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool, &host_pool,
                                                   /*stream_device_cache_to_host=*/true);
    for (std::int32_t i = 0; i < pool_size; ++i) {
        CacheBlockRef block = host_pool.AcquireBlock(/*group_id=*/0, /*packing=*/1);
        if (!block) {
            std::abort();
        }
        coordinator.GroupPrefixIndex(0).Register(
            host_pool, block,
            CacheKey{.group_id = 0, .content_hash = "host-cache-" + std::to_string(i), .page_offset = 0},
            /*access_epoch=*/static_cast<std::uint64_t>(i + 1));
    }

    const std::array<std::uint32_t, 1> group_ids{0};
    std::uint64_t next_key = static_cast<std::uint64_t>(pool_size);
    Measure("acquire_host_blocks_full_cache", pool_size, 1, iterations, [&] {
        CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(group_ids);
        if (batch.stats.allocated != 1) {
            std::abort();
        }
        ++next_key;
        coordinator.CacheHostBlock(
            batch.blocks.front(),
            CacheKey{.group_id = 0, .content_hash = "host-cache-" + std::to_string(next_key), .page_offset = 0});
        return static_cast<std::uint64_t>(batch.stats.same_group_scans + batch.stats.cross_group_scans);
    });
}

}  // namespace
}  // namespace tokenspeed

int main() {
    std::cout << "workload,pool_size,demand,iterations,ns_per_op_median,checksum\n";
    for (const std::int32_t pool_size : tokenspeed::kPoolSizes) {
        const std::int32_t iterations = tokenspeed::IterationsFor(pool_size);
        tokenspeed::MeasureAcquireBlocksPackingOne(pool_size, iterations);
        tokenspeed::MeasureFragmentedAllocations(pool_size, iterations);
        tokenspeed::MeasureFullAllocations(pool_size, iterations);
        tokenspeed::MeasureLargeBatchControlMaintenance(pool_size, iterations);
        tokenspeed::MeasureHostStyleAcquire(pool_size, iterations);
        tokenspeed::MeasureEvictableCandidates(pool_size, iterations);
        tokenspeed::MeasureAdmission(pool_size, iterations);
        tokenspeed::MeasurePinnedAdmission(pool_size, std::min(iterations, 100));
        tokenspeed::MeasureAvailableLcmBlocks(pool_size, iterations);
        tokenspeed::MeasureHostBlockAcquisition(pool_size, iterations);
    }
    return 0;
}
