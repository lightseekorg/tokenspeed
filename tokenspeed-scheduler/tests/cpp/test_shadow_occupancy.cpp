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

#include <cstdint>
#include <memory>
#include <memory_resource>
#include <list>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "cache/cache_group.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/parent_occupancy.h"
#include "cache/coordinator/shadow_occupancy.h"

namespace tokenspeed::test {
namespace {

TEST(ParentOccupancy, SparseParentsKeepZeroAndRestoredCountsAcrossGrowthAndUnwind) {
    for (const std::int32_t stride : {4096, 104729}) {
        PlanningScratch storage;
        for (int round = 0; round < 3; ++round) {
            try {
                auto lease = storage.Acquire();
                ParentOccupancy counts{&lease};
                EXPECT_EQ(counts.Find(1), nullptr);
                for (int i = 1; i <= 16384; ++i) {
                    const auto parent = static_cast<std::int32_t>(static_cast<std::int64_t>(i) * stride);
                    counts.GetOrInsert(parent, i % 4 + 1) -= 1;
                    EXPECT_EQ(counts.GetOrInsert(parent, 99), i % 4);
                }
                EXPECT_EQ(counts.size(), 16384u);
                for (int i = 16384; i > 0; --i) {
                    const auto parent = static_cast<std::int32_t>(static_cast<std::int64_t>(i) * stride);
                    auto* occupied = counts.Find(parent);
                    ASSERT_NE(occupied, nullptr);
                    EXPECT_EQ(*occupied, i % 4);
                    ++*occupied;
                    EXPECT_EQ(counts.GetOrInsert(parent, 99), i % 4 + 1);
                }
                EXPECT_EQ(counts.Find(1), nullptr);
                throw 1;
            } catch (int) {
            }
        }
    }
}

TEST(CacheEvictionCandidates, PrescanReplaysPolicyOrderAcrossPinnedEpochsAndSnapshotLimit) {
    for (const std::size_t threshold : {16u, 9000u, 12000u}) {
        BlockPool pool(10032, {1, 1});
        std::vector<CacheGroup> groups;
        std::vector<CacheBlockRef> pinned;
        for (std::uint32_t group = 0; group < 2; ++group) {
            const CacheGroupSpec spec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4};
            groups.emplace_back(spec, std::make_unique<GroupAllocator>(1, group, 1), MakePrefixMatcher(spec));
            for (int i = 0; i < (group == 0 ? 10000 : 32); ++i) {
                auto block = pool.AcquireBlock(group);
                const CacheKey key{.group_id = group, .content_hash = std::to_string(i)};
                groups.back().Index().Register(pool, block, key, static_cast<std::uint64_t>(i / 16 + 1), i,
                                               CacheBoundaryKind::kChunk, nullptr, nullptr);
                if (i >= 16 && i < 32) pinned.push_back(block);
            }
        }
        CacheEvictionCandidates replay{groups, pool, nullptr, nullptr, CheckpointAging{}, nullptr};
        std::size_t seen = 0;
        const bool fits = replay.PreScanUntil([&](std::uint32_t, auto entries) {
            seen += entries.size();
            return seen >= threshold;
        });
        EXPECT_EQ(fits, threshold <= 10000);
        CacheEvictionCandidates direct{groups, pool, nullptr, nullptr, CheckpointAging{}, nullptr};
        using Key = std::tuple<std::uint32_t, std::int32_t, std::int32_t, std::uint64_t, std::int64_t>;
        const auto drain = [](CacheEvictionCandidates& candidates) {
            std::vector<Key> result;
            while (const auto candidate = candidates.Next(std::numeric_limits<std::uint64_t>::max())) {
                result.emplace_back(candidate->group_id, candidate->location.lcm_block_id,
                                    candidate->location.slot_index, candidate->rank_epoch, candidate->position_rank);
            }
            return result;
        };
        const auto expected = drain(direct);
        EXPECT_EQ(expected.size(), 10000u);
        EXPECT_EQ(drain(replay), expected);
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
    }
}

TEST(FixedSlotResource, AlignedNodesKeepTheirAddressAcrossGrowthAndReuse) {
    struct alignas(256) Value {
        int value;
    };
    FixedSlotResource storage{16};
    std::pmr::list<Value> values{&storage};
    values.push_back(Value{73});
    const auto* anchor = &values.front();
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 4096; ++i) values.push_back(Value{i});
        EXPECT_EQ(&values.front(), anchor);
        EXPECT_EQ(anchor->value, 73);
        auto it = std::next(values.begin());
        for (int i = 0; i < 4096; ++i, ++it) {
            EXPECT_EQ(it->value, i);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&*it) % alignof(Value), 0);
        }
        values.erase(std::next(values.begin()), values.end());
    }
}

TEST(ShadowOccupancy, ConcurrentPlansAndReusedScratchHaveIndependentCapacity) {
    const CacheGroupSpec history{.cache_blocks_per_lcm_block = 2, .block_granularity = 4};
    std::vector<CacheGroup> groups;
    groups.emplace_back(history, std::make_unique<GroupAllocator>(2, 0, 1), MakePrefixMatcher(history));
    BlockPool pool{2, {2}};
    pool.RegisterGroup(0, 2, 1);
    const auto held = pool.AcquireBlocks(0, 4);
    const std::vector<std::vector<std::int64_t>> needed{{2}};
    const std::vector<std::int64_t> zero{0};
    for (int round = 0; round < 4; ++round) {
        ShadowOccupancy first{pool, groups};
        ShadowOccupancy second{pool, groups};
        ShadowOccupancy third{pool, groups};
        EXPECT_FALSE(first.Fits(needed, zero, zero));
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        EXPECT_FALSE(third.Fits(needed, zero, zero));
        first.Remove(0, held[0]->Location());
        first.Remove(0, held[1]->Location());
        EXPECT_TRUE(first.Fits(needed, zero, zero));
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        second.Remove(0, held[2]->Location());
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        second.Remove(0, held[3]->Location());
        EXPECT_TRUE(second.Fits(needed, zero, zero));
        first.Restore(0, held[0]->Location());
        EXPECT_FALSE(first.Fits(needed, zero, zero));
        EXPECT_TRUE(second.Fits(needed, zero, zero));
        EXPECT_FALSE(third.Fits(needed, zero, zero));
        EXPECT_EQ(pool.NumOccupiedSlots(), 4);
    }
}

TEST(PlanningScratch, AlignedOverflowAndNestedLeasesKeepTheirOwnValues) {
    struct alignas(512) Value {
        int value;
    };
    PlanningScratch scratch;
    for (int round = 0; round < 3; ++round) {
        auto first = scratch.Acquire();
        auto second = scratch.Acquire();
        auto third = scratch.Acquire();
        std::pmr::vector<Value> large{&first};
        std::pmr::vector<Value> small{&second};
        std::pmr::vector<Value> nested{&third};
        for (int i = 0; i < 2048; ++i) large.push_back(Value{i});
        small.push_back(Value{42});
        nested.push_back(Value{99});
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(large.data()) % alignof(Value), 0);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small.data()) % alignof(Value), 0);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(nested.data()) % alignof(Value), 0);
        for (int i = 0; i < 2048; ++i) EXPECT_EQ(large[i].value, i);
        EXPECT_EQ(small[0].value, 42);
        EXPECT_EQ(nested[0].value, 99);
    }
}

TEST(ShadowOccupancy, AReleasedHostParentCannotAlsoLeaveHistorySlotsBehind) {
    const CacheGroupSpec history{.cache_blocks_per_lcm_block = 6, .block_granularity = 4};
    const CacheGroupSpec state{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0};
    std::vector<CacheGroup> groups;
    groups.emplace_back(history, std::make_unique<GroupAllocator>(6, 0, 1), MakePrefixMatcher(history));
    groups.emplace_back(state, std::make_unique<GroupAllocator>(1, 1, 1), MakePrefixMatcher(state));
    BlockPool host{1, {8, 1}};
    host.RegisterGroup(0, 8, 1);
    host.RegisterGroup(1, 1, 1);
    const CacheBlockRef held = host.AcquireBlock(0);
    ASSERT_TRUE(held);
    ShadowOccupancy shadow{host, groups};
    shadow.Remove(0, held->Location());

    const std::vector<std::int64_t> dense{0, 0};
    const std::vector<std::int64_t> pairs{0, 0};
    const std::vector<std::vector<std::int64_t>> mixed{{2}, {1}};
    const std::vector<std::vector<std::int64_t>> history_only{{8}, {0}};
    const std::vector<std::vector<std::int64_t>> state_only{{0}, {1}};
    EXPECT_FALSE(shadow.Fits(mixed, dense, pairs));
    EXPECT_TRUE(shadow.Fits(history_only, dense, pairs));
    EXPECT_TRUE(shadow.Fits(state_only, dense, pairs));
    EXPECT_EQ(host.NumEmptyLcmBlocks(), 0);  // The plan has not mutated storage.
}

TEST(HostPacking, ABlockIdUsesTheOwningPoolsPacking) {
    GroupAllocator allocator{4, 0, 2};
    BlockPool device{2, {4}};
    BlockPool host{2, {6}};
    device.RegisterGroup(0, 4, 2);
    host.RegisterGroup(0, 6, 2);
    const std::vector<std::int32_t> buckets{0, 1, 0, 1, 0, 1, 0, 1};
    const auto device_blocks = device.AcquireBlocksInBuckets(0, std::span{buckets}.first(6));
    const auto host_blocks = host.AcquireBlocksInBuckets(0, buckets);
    ASSERT_EQ(device_blocks.size(), 6);
    ASSERT_EQ(host_blocks.size(), 8);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*device_blocks.back()), 6);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*host_blocks.back()), 8);
    EXPECT_EQ(host_blocks.back()->Location().lcm_block_id, 2);
    EXPECT_EQ(host_blocks.back()->Location().slot_index, 1);
    EXPECT_EQ((allocator.ResolveCacheBlockId(*host_blocks.back()) - 1) % 2, 1);

    // Both pools use parent 2, but their first child IDs differ. Holes and
    // a return to the Device pool must preserve every logical table slot.
    BlockTable table = BlockTable::FromBlocks(
        /*group_id=*/0, {device_blocks[4], device_blocks[5], {}, host_blocks[6], host_blocks[7], device_blocks[5]}, 0);
    EXPECT_EQ(allocator.BlockTablePageIds(table), (std::vector<std::int32_t>{5, 6, 0, 7, 8, 6}));
    allocator.ReclaimExpired(device, table, 2);
    EXPECT_EQ(allocator.BlockTablePageIds(table), (std::vector<std::int32_t>{0, 0, 0, 7, 8, 6}));
}

TEST(BlockTableAccounting, ReleasedPrefixesAndSharedParentsCountOnce) {
    BlockPool device{4, {4}};
    const std::vector specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = 4,
                                           .cache_blocks_per_lcm_block = 4,
                                           .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, device, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const auto blocks = device.AcquireBlocks(0, 8);
    ASSERT_EQ(blocks.size(), 8u);
    std::vector<BlockTable> first{BlockTable::FromBlocks(/*group_id=*/0, {blocks.begin(), blocks.begin() + 7}, 0)};
    std::vector<BlockTable> second{
        BlockTable::FromBlocks(/*group_id=*/0, {blocks[0], {}, {}, {}, {}, {}, {}, blocks[7]}, 0)};
    const std::vector<std::span<const BlockTable>> requests{first, second};
    coordinator.Allocator(0).ReclaimExpired(device, first[0], 6);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 2);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 7);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, first[0], 7);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 8);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 0);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 2) << "the retained copies still occupy the pool";
}

TEST(HostPacking, SnapshotAdmissionCountsTheSnapshotPoolsChildren) {
    for (const std::int32_t host_packing : {2, 6}) {
        BlockPool device{2, {4}};
        BlockPool snapshots{1, {host_packing}};
        const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 4, .block_granularity = 2}};
        auto coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device, /*enable_l3_storage=*/false,
                                           /*host_pool=*/nullptr, &snapshots, /*stream_device_cache_to_host=*/false,
                                           /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
        const std::int32_t blocks = host_packing == 2 ? 3 : 5;
        const std::int32_t tokens = blocks * 2;
        std::vector<BlockTable> tables{BlockTable::FromBlocks(/*group_id=*/0, device.AcquireBlocks(0, blocks), 0)};
        const std::vector<std::vector<ImageSlot>> host_slots(1);
        const bool fits = blocks <= host_packing;
        EXPECT_EQ(coordinator.SnapshotPoolHolds(tables, tokens, host_slots), fits);
        EXPECT_EQ(snapshots.NumEmptyLcmBlocks(), 1);
        auto image = coordinator.TakeImage(tables, tokens, host_slots);
        ASSERT_EQ(image.has_value(), fits);
        if (fits) {
            EXPECT_EQ(image->store_pairs.size(), blocks);
            EXPECT_EQ(snapshots.NumOccupiedSlots(), blocks);
        } else {
            EXPECT_EQ(snapshots.NumEmptyLcmBlocks(), 1);
            EXPECT_EQ(snapshots.NumOccupiedSlots(), 0);
        }
    }
}

}  // namespace
}  // namespace tokenspeed::test
