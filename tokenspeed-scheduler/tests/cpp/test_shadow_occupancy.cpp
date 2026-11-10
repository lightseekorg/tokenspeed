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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cache/cache_group.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/shadow_occupancy.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"

namespace tokenspeed::test {
namespace {

std::vector<std::string> ContentHashesFor(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<std::span<const std::int32_t>> spans;
    spans.reserve(pages.size());
    for (const auto& page : pages) {
        spans.emplace_back(page.data(), page.size());
    }
    return ComputePrefixHashes(spans, "");
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
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(requests), 2);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 7);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, first[0], 7);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 8);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(requests), 0);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 2) << "the retained copies still occupy the pool";
}

// CacheCoordinator::NumActiveLcmBlocks() (the ActiveParents O(1) gauge) must
// always agree with CacheCoordinator::CountActiveLcmBlocks() (the reference
// walk) over every request table set still live through the coordinator.
// Each test below checks both against a hand-derived expectation.

TEST(BlockTableAccounting, AdmissionCountsDistinctParentsAndFreeDropsThem) {
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));  // 3 blocks, 3 distinct parents
    const std::vector<std::span<const BlockTable>> live{tables};
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 3);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 3);

    coordinator.Free(tables);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 0);
}

TEST(BlockTableAccounting, SharedPrefixParentsCountOnceAcrossRequests) {
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const std::vector<std::string> ch = ContentHashesFor({{0, 0, 0, 0}, {1, 1, 1, 1}});

    std::vector<BlockTable> a(1);
    ASSERT_TRUE(AdmitForTest(coordinator, a, /*num_tokens=*/8));  // 2 fresh blocks
    CacheFullBlocksForTest(coordinator, a, ch);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2);

    CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 8);
    std::vector<BlockTable> b(1);
    ASSERT_TRUE(AdmitForTest(coordinator, b, std::move(prefix), GroupDemand{}));  // prefix hit, no new blocks
    ASSERT_EQ(b[0].NumBlocks(), 2);

    // a and b share the same two cache blocks (hence the same two parents):
    // the gauge and the reference walk over both live tables must still
    // read 2, not 4.
    const std::vector<std::span<const BlockTable>> live_both{a, b};
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live_both), 2);

    coordinator.Free(a);
    const std::vector<std::span<const BlockTable>> live_b{b};
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "b still owns the shared parents";
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live_b), 2);

    coordinator.Free(b);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
}

TEST(BlockTableAccounting, PackedParentSharedByTwoRequestsCountsOnce) {
    // One LCM parent packs 2 cache blocks; two different requests each take
    // one, filling the parent between them.
    BlockPool pool(1, {2});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 2, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> a(1);
    ASSERT_TRUE(AdmitForTest(coordinator, a, /*num_tokens=*/4));  // 1 block
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 1);

    std::vector<BlockTable> b(1);
    ASSERT_TRUE(AdmitForTest(coordinator, b, /*num_tokens=*/4));  // the parent's other slot
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 1) << "the one parent is shared by a and b";
    const std::vector<std::span<const BlockTable>> live{a, b};
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 1);

    coordinator.Free(a);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 1) << "b still occupies the parent";

    coordinator.Free(b);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
}

TEST(BlockTableAccounting, CoordinatorReclaimExpiredDropsSlidOutParents) {
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = 4,
                                           .cache_blocks_per_lcm_block = 1,
                                           .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/20));  // 5 blocks
    const std::vector<std::span<const BlockTable>> live{tables};
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 5);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 5);

    // skipped = num_computed_tokens - sliding_window + 1 = 20 - 4 + 1 = 17;
    // 17 / block_granularity(4) = 4 pages expire, leaving 1 live block.
    coordinator.ReclaimExpired(tables, /*num_computed_tokens=*/20);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 1);
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 1);

    coordinator.Free(tables);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
}

TEST(BlockTableAccounting, CopyMoveAndScopeExitTrackActiveParents) {
    // The pool must outlive every table built against it.
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));  // 2 blocks
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2);

    {
        std::vector<BlockTable> copy = tables;  // NOLINT(performance-unnecessary-copy-initialization)
        EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "a copy shares its source's parents, counted once";
    }
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "dropping the copy leaves the original counted";

    std::vector<BlockTable> moved = std::move(tables);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "a move transfers the count, it does not change it";

    {
        std::vector<BlockTable> scoped = std::move(moved);
        EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2);
    }
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0)
        << "a filled table going out of scope without Free still decrements the gauge";
}

TEST(BlockTableAccounting, FromBlocksTableCountsNowhere) {
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const auto blocks = pool.AcquireBlocks(0, 2);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
    {
        BlockTable table = BlockTable::FromBlocks(/*group_id=*/0, blocks, 0);
        EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0) << "a table built outside the allocator counts nowhere";
    }
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
}

TEST(BlockTableAccounting, PublicationDedupeReplacesLosingTableBlockInPlace) {
    // A and B are each admitted with their own fresh blocks for the same two
    // pages (AdmitForTest's num_tokens overload probes an empty hash list, so
    // neither gets a prefix hit). A publishes first and becomes canonical;
    // when B publishes the same hashes, PrefixCacheIndex::Register replaces
    // B's blocks with A's canonical ones in place
    // (BlockTable::PublishWindow / RegisterFullBlocks), and BlockTable::refresh
    // re-derives B's slots' page ids and ParentRefs from the swapped blocks.
    BlockPool pool(8, {1});
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const std::vector<std::string> ch = ContentHashesFor({{0, 0, 0, 0}, {1, 1, 1, 1}});

    std::vector<BlockTable> a(1);
    ASSERT_TRUE(AdmitForTest(coordinator, a, /*num_tokens=*/8));  // 2 fresh blocks
    std::vector<BlockTable> b(1);
    ASSERT_TRUE(AdmitForTest(coordinator, b, /*num_tokens=*/8));  // 2 more fresh blocks, distinct from a's

    const std::vector<std::span<const BlockTable>> live{a, b};
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 4) << "a and b each hold 2 distinct fresh parents";
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 4);
    const std::vector<std::int32_t> a_ids = coordinator.Allocator(0).BlockTablePageIds(a[0]);
    const std::vector<std::int32_t> b_ids_before = coordinator.Allocator(0).BlockTablePageIds(b[0]);
    EXPECT_NE(a_ids, b_ids_before) << "a and b start on different physical blocks";
    const std::int32_t empty_before = pool.NumEmptyLcmBlocks();

    CacheFullBlocksForTest(coordinator, a, ch);  // a's blocks become the canonical entries
    CacheFullBlocksForTest(coordinator, b, ch);  // b dedupes against a's canonical blocks

    const std::vector<std::int32_t> b_ids_after = coordinator.Allocator(0).BlockTablePageIds(b[0]);
    ASSERT_EQ(b_ids_after, a_ids) << "the dedupe must have swapped b's slots onto a's blocks";

    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "b's slots now share a's 2 parents instead of owning 2 more";
    EXPECT_EQ(coordinator.CountActiveLcmBlocks(live), 2);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), empty_before + 2) << "b's replaced fresh blocks returned to the pool";

    coordinator.Free(a);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 2) << "b still holds the shared canonical parents";
    coordinator.Free(b);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(), 0);
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
