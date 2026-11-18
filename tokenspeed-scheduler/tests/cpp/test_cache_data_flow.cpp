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
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cache/allocator/group_allocator.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/group_geometry.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_hasher.h"

namespace tokenspeed::test {
namespace {

TEST(CacheDataFlow, OneConstAllocatorPlacesIndependentTablesInTheirSuppliedPools) {
    const GroupAllocator allocator{/*cache_blocks_per_lcm_block=*/2, /*group_id=*/0,
                                   /*shard_count=*/1, /*kept_head_blocks=*/0};
    const GroupGeometry geometry{4};
    BlockPool first_pool{2, {2}};
    BlockPool second_pool{2, {2}};
    BlockTable first;
    BlockTable second;
    const AcquirePlan first_plan = geometry.PlanAcquire(first, /*num_tokens=*/7, /*reserve_tokens=*/3);
    const AcquirePlan second_plan = geometry.PlanAcquire(second, /*num_tokens=*/2, /*reserve_tokens=*/0);
    ASSERT_TRUE(allocator.Acquire(first_pool, first, first_plan, /*first_placed_slot=*/0, {}));
    ASSERT_TRUE(allocator.Acquire(second_pool, second, second_plan, /*first_placed_slot=*/0, {}));
    EXPECT_EQ(first.NumBlocks(), 3);
    EXPECT_EQ(first.AvailableTokens(), 5);
    EXPECT_EQ(second.NumBlocks(), 1);
    EXPECT_EQ(second.AvailableTokens(), 2);
    allocator.ConsumeReservedTokens(first, 2);
    allocator.ReclaimExpired(first_pool, first, /*expired_block_end=*/2);
    EXPECT_EQ(first.AvailableTokens(), 3);
    EXPECT_EQ(first_pool.NumOccupiedSlots(), 1);
    EXPECT_EQ(second.AvailableTokens(), 2);
    EXPECT_EQ(second_pool.NumOccupiedSlots(), 1);
    allocator.Free(first);
    allocator.Free(second);
    EXPECT_EQ(first_pool.NumOccupiedSlots(), 0);
    EXPECT_EQ(second_pool.NumOccupiedSlots(), 0);
}

TEST(CacheDataFlow, HostSuffixReturnsIndependentTierResultsAndLeavesItsInputUnchanged) {
    BlockPool device_pool{4, {1}};
    BlockPool host_pool{4, {1}};
    const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const std::vector<std::int32_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const auto all_tokens = std::span{tokens};
    const std::vector<std::span<const std::int32_t>> pages{all_tokens.subspan(0, 4), all_tokens.subspan(4, 4),
                                                           all_tokens.subspan(8, 4)};
    const auto hashes = ComputePrefixHashes(pages, "");
    {
        auto block = device_pool.AcquireBlock(0);
        coordinator.GroupPrefixIndex(0).Register(
            device_pool, block, CacheKey{.group_id = 0, .content_hash = ContentHash{hashes[0]}},
            /*access_epoch=*/1, /*logical_block_index=*/0, CacheBoundaryKind::kEndpoint, nullptr, nullptr);
    }
    {
        auto block = host_pool.AcquireBlock(0);
        coordinator.GroupPrefixIndex(0).Register(
            host_pool, block, CacheKey{.group_id = 0, .content_hash = ContentHash{hashes[1]}},
            /*access_epoch=*/1, /*logical_block_index=*/1, CacheBoundaryKind::kEndpoint, nullptr, nullptr);
    }
    const auto& prober = std::as_const(coordinator).Prober();
    const auto& store = std::as_const(coordinator).Store();
    const PromptPrefix prompt{.prefix_hashes = hashes, .tokens = tokens, .max_tokens = 12};
    const auto keys = prober.PreparePrefixKeys(hashes);
    const PrefixProbe::Tier device = prober.ProbeTier(prompt, *keys, store.MatchOrder(), /*claim_partial=*/true);
    ASSERT_EQ(device.num_common_tokens, 4);
    const auto visible = prober.ProbeHostSuffix(prompt, *keys, store.MatchOrder(), /*above_tokens=*/4,
                                                /*extra_hits=*/nullptr, device);
    const std::unordered_set<CacheKey, CacheKeyHash> assumed{(*keys)[0][2]};
    const auto extended =
        prober.ProbeHostSuffix(prompt, *keys, store.MatchOrder(), /*above_tokens=*/4, &assumed, device);
    EXPECT_EQ(device.num_common_tokens, 4);
    EXPECT_EQ(device.full_page_tokens, 4);
    ASSERT_EQ(device.per_group.size(), 1u);
    EXPECT_EQ(device.per_group[0].hits, (std::vector<std::uint8_t>{1}));
    EXPECT_EQ(visible.device.num_common_tokens, 4);
    EXPECT_EQ(visible.host.num_common_tokens, 8);
    EXPECT_EQ(visible.host.per_group[0].hits, (std::vector<std::uint8_t>{1}));
    EXPECT_EQ(extended.device.num_common_tokens, 4);
    EXPECT_EQ(extended.host.num_common_tokens, 12);
    EXPECT_EQ(extended.host.per_group[0].hits, (std::vector<std::uint8_t>{1, 1}));
    EXPECT_EQ(prober.ProbePromptPrefix(prompt).host.num_common_tokens, 8);
    EXPECT_EQ(device_pool.NumOccupiedSlots(), 1);
    EXPECT_EQ(host_pool.NumOccupiedSlots(), 1);
}

TEST(CacheDataFlow, StoreMetadataKeepsMatchOrderAndGeometryAcrossAllocation) {
    BlockPool pool{8, {1, 1, 1, 1}};
    const std::vector specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = 4,
                                           .cache_blocks_per_lcm_block = 1,
                                           .block_granularity = 2},
                            CacheGroupSpec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4},
                            CacheGroupSpec{.attn_kind = AttnKind::kState,
                                           .cache_blocks_per_lcm_block = 1,
                                           .block_granularity = 0,
                                           .checkpoint_granularity = 2},
                            CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = 8,
                                           .replayable = true,
                                           .cache_blocks_per_lcm_block = 1,
                                           .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const CacheStore& store = std::as_const(coordinator).Store();
    for (const bool allocate : {false, true}) {
        auto block = allocate ? pool.AcquireBlock(0) : CacheBlockRef{};
        EXPECT_EQ(store.MatchOrder(), (std::vector<std::size_t>{1, 0, 2}));
        EXPECT_EQ(store.RoutingHistoryGroups(), (std::vector<std::size_t>{1}));
        EXPECT_EQ(store.Geometry()[0].BlockGranularity(), 2);
        EXPECT_EQ(store.Geometry()[1].BlockGranularity(), 4);
        EXPECT_FALSE(store.Geometry()[2].HasTokenSpan());
        EXPECT_EQ(store.CheckpointGranularity(), 2);
        EXPECT_EQ(store.ReplayWindowTokens(), 8);
        EXPECT_EQ(store.NumStateGroups(), 1);
        EXPECT_EQ(store.DevicePool().NumOccupiedSlots(), allocate ? 1 : 0);
    }
}

}  // namespace
}  // namespace tokenspeed::test
