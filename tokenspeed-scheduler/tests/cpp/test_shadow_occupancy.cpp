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
#include <vector>

#include <gtest/gtest.h>

#include "cache/cache_group.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/shadow_occupancy.h"

namespace tokenspeed::test {
namespace {

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
    const std::vector<std::vector<std::int64_t>> mixed{{2}, {1}};
    const std::vector<std::vector<std::int64_t>> history_only{{8}, {0}};
    const std::vector<std::vector<std::int64_t>> state_only{{0}, {1}};
    EXPECT_FALSE(shadow.Fits(mixed, dense));
    EXPECT_TRUE(shadow.Fits(history_only, dense));
    EXPECT_TRUE(shadow.Fits(state_only, dense));
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
}

TEST(HostPacking, SnapshotAdmissionCountsTheSnapshotPoolsChildren) {
    for (const std::int32_t host_packing : {2, 6}) {
        BlockPool device{2, {4}};
        BlockPool snapshots{1, {host_packing}};
        const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 4, .block_granularity = 2}};
        auto coordinator =
            MakeCoordinator(specs, /*prefix_granularity=*/2, device, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                            &snapshots, /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        const std::int32_t blocks = host_packing == 2 ? 3 : 5;
        const std::int32_t tokens = blocks * 2;
        std::vector<BlockTable> tables{BlockTable::FromBlocks(device.AcquireBlocks(0, blocks), 0)};
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
