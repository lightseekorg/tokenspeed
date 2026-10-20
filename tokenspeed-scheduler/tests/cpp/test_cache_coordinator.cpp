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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/cache_group.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/cache_types.h"
#include "cache/allocator/group_allocator.h"
#include "cache/prefix/prefix_matcher.h"
#include "prefix_test_group.h"
#include "cache_test_access.h"
#include "cache/prefix/prefix_hasher.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {
namespace {

std::vector<std::int32_t> BlockIds(const std::vector<CacheBlockRef>& refs) {
    std::vector<std::int32_t> ids;
    ids.reserve(refs.size());
    for (const CacheBlockRef& ref : refs) {
        ids.push_back(ref ? ref->Location().lcm_block_id : 0);
    }
    return ids;
}

using token_span = std::span<const std::int32_t>;

std::vector<std::string> ContentHashes(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const auto& p : pages) {
        spans.emplace_back(p.data(), p.size());
    }
    return ComputePrefixHashes(spans, "");
}

std::uint64_t NextTestAccessEpoch() {
    static std::uint64_t next_access_epoch = 0;
    return ++next_access_epoch;
}

CacheKey Key(const std::string& content_hash, std::uint32_t group_id, std::int32_t page_offset = 0) {
    return CacheKey{
        .group_id = group_id,
        .content_hash = content_hash,
        .page_offset = page_offset,
    };
}

// Cache then free, so the block is prefix-hittable via MatchPrefix.
std::int32_t CacheForGroup(CacheCoordinator& coordinator, BlockPool& pool, const std::string& content_hash,
                           std::uint32_t group_id) {
    const CacheKey key = Key(content_hash, group_id);
    const std::int32_t group_index = static_cast<std::int32_t>(group_id);
    CacheBlockRef got = pool.AcquireBlock(group_id);
    const std::int32_t id = got->Location().lcm_block_id;
    coordinator.GroupPrefixIndex(group_index)
        .Register(pool, got, key, NextTestAccessEpoch(), /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                  /*newly_cached=*/nullptr);
    got.reset();
    return id;
}

CacheBlockLocation CacheBoundaryForGroup(CacheCoordinator& coordinator, BlockPool& pool,
                                         const std::string& content_hash, std::uint32_t group_id,
                                         std::uint64_t access_epoch, std::int32_t logical_block_index,
                                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk) {
    const std::int32_t group_index = static_cast<std::int32_t>(group_id);
    CacheBlockRef block_ref = pool.AcquireBlock(group_id);
    _assert(static_cast<bool>(block_ref), "test cache block allocation failed");
    const CacheBlockLocation location = block_ref->Location();
    coordinator.GroupPrefixIndex(group_index)
        .Register(pool, block_ref, Key(content_hash, group_id), access_epoch, logical_block_index, boundary_kind,
                  /*newly_cached=*/nullptr);
    block_ref.reset();
    return location;
}

// Asserts no null hole inside the last min(len, pages_needed) blocks.
void ExpectSwaWindowIntact(const PrefixMatch& m, std::int32_t window, std::int32_t block_granularity) {
    std::int32_t len = static_cast<std::int32_t>(m.blocks.size());
    std::int32_t pages_needed = (window - 1 + block_granularity - 1) / block_granularity;
    std::int32_t need = std::min(len, pages_needed);
    for (std::int32_t i = len - need; i < len; ++i) {
        EXPECT_TRUE(m.blocks[static_cast<std::size_t>(i)])
            << "null hole inside the last window at slot " << i << " of " << len;
    }
}

TEST(FullAttnMatcherTest, ProbeRequiresExplicitL3HitSet) {
    BlockPool pool(8, {1});
    PrefixCacheIndex index(/*group_id=*/0);
    const CacheKey key{.group_id = 0, .content_hash = "l3"};
    const std::array keys{key};
    std::unordered_set<CacheKey, CacheKeyHash> l3_hits{key};

    EXPECT_TRUE(FullAttnMatcher{}
                    .Probe(index, pool, keys, /*begin_blocks=*/0, /*max_blocks=*/1, /*extra_hits=*/nullptr)
                    .hits.empty());
    const GroupPrefixProbe probe =
        FullAttnMatcher{}.Probe(index, pool, keys, /*begin_blocks=*/0, /*max_blocks=*/1, &l3_hits);
    ASSERT_EQ(probe.hits.size(), 1u);
    EXPECT_EQ(probe.hits.front(), 1);
}

TEST(CacheGroupTest, HoldsSpecGroupIdManager) {
    BlockPool pool(8, {1});
    auto mgr = std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1, /*group_id=*/7, /*shard_count=*/1);
    CacheGroup g(
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        std::move(mgr), std::make_unique<FullAttnMatcher>());
    EXPECT_EQ(g.Id(), 7u);
    EXPECT_EQ(g.Spec().attn_kind, AttnKind::kFull);
}

TEST(MakeCoordinatorTest, BuildsOneGroupPerSpec) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coord.NumGroups(), 2);
}

TEST(MakeCoordinatorTest, UsesOneCacheBlockPerLcmBlockByDefault) {
    BlockPool pool(1, {1});
    const std::array specs{CacheGroupSpec{.block_granularity = 4}};

    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coordinator.Allocator(0).CacheBlocksPerLcmBlock(), 1);
}

TEST(MakeCoordinatorTest, Qwen35UsesUniformLogicalPWithDifferentPacking) {
    BlockPool pool(32, {8, 1, 1, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 8, .block_granularity = 128},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    CacheCoordinator coord =
        MakeCoordinator(specs, /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coord.PrefixGranularity(), 128);
    EXPECT_EQ(coord.GroupBlockGranularity(0), 128);
    EXPECT_TRUE(coord.GroupIsState(1));
    EXPECT_EQ(coord.BlocksPerBoundary(0), 1);
    EXPECT_EQ(coord.BlocksPerBoundary(1), 1);
    EXPECT_EQ(coord.Allocator(0).CacheBlocksPerLcmBlock(), 8);
    EXPECT_EQ(coord.Allocator(1).CacheBlocksPerLcmBlock(), 1);

    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/128));
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    for (std::size_t i = 1; i < tables.size(); ++i) {
        ASSERT_EQ(tables[i].NumBlocks(), 2) << "a State table is two slots";
        EXPECT_TRUE(tables[i].Blocks()[0]) << "no checkpoint: a fresh zeroed block is the zero state";
        EXPECT_TRUE(tables[i].Blocks()[1]) << "the live slot";
    }

    // The page completes at a natural checkpoint: every group publishes.
    const std::vector<std::string> hashes = ContentHashes({std::vector<std::int32_t>(128, 7)});
    CacheCompletedBlocksForTest(coord, tables, hashes, NextTestAccessEpoch(), /*first_new_prefix_page=*/0,
                                /*num_computed_tokens=*/128, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/true);
    coord.Free(tables);
    CoordinatorMatch match = MatchPrefixForTest(coord, hashes).device;
    EXPECT_EQ(match.num_common_tokens, 128);
    ASSERT_EQ(match.per_group.size(), specs.size());
    for (const PrefixMatch& group_match : match.per_group) {
        EXPECT_EQ(group_match.NumHitBlocks(), 1);
    }
}

TEST(MakeCoordinatorTest, ManagersMayUseSmallerPagesThanTheCoordinatorDomain) {
    BlockPool pool(32, {2, 3});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 8},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 3,
         .block_granularity = 2},
    };

    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/8, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coordinator.PrefixGranularity(), 8);
    EXPECT_EQ(coordinator.GroupBlockGranularity(0), 8);
    EXPECT_EQ(coordinator.GroupBlockGranularity(1), 2);
}

// A parent held only by an unpinned cache entry is available to eviction but
// not empty: the per-step gauge counts it as cached, the leak check as free.
TEST(CacheCoordinatorCapacityTest, EmptyParentsExcludeCacheOnlyParents) {
    BlockPool pool(2, {2});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    PrefixCacheIndex& index = coordinator.GroupPrefixIndex(0);
    const CacheKey key = Key("cached", /*group_id=*/0);

    CacheBlockRef request_ref = pool.AcquireBlock(/*group_id=*/0);
    index.Register(pool, request_ref, key, /*access_epoch=*/1, /*logical_block_index=*/0, CacheBoundaryKind::kChunk,
                   /*newly_cached=*/nullptr);
    EXPECT_EQ(coordinator.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(coordinator.NumAvailableLcmBlocks(), 1);

    request_ref.reset();
    EXPECT_EQ(coordinator.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(coordinator.NumAvailableLcmBlocks(), 2);

    EXPECT_TRUE(index.Evict(pool, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}).has_value());
    EXPECT_EQ(coordinator.NumEmptyLcmBlocks(), 2);
    EXPECT_EQ(coordinator.NumAvailableLcmBlocks(), 2);
}

TEST(CacheCoordinatorCapacityTest, GroupAvailablePagesTracksLocalSlotsAndEmptyParents) {
    BlockPool pool(3, {4, 2});
    const std::array specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coordinator.GroupAvailablePages(0), 12);
    EXPECT_EQ(coordinator.GroupAvailablePages(1), 6);
    CacheBlockRef group_zero = pool.AcquireBlock(/*group_id=*/0);
    EXPECT_EQ(coordinator.GroupAvailablePages(0), 11);
    EXPECT_EQ(coordinator.GroupAvailablePages(1), 4);
    CacheBlockRef group_one = pool.AcquireBlock(/*group_id=*/1);
    EXPECT_EQ(coordinator.GroupAvailablePages(0), 7);
    EXPECT_EQ(coordinator.GroupAvailablePages(1), 3);
}

TEST(CacheCoordinatorTest, ExpandsOneLogicalHashIntoPerGroupCacheBlocks) {
    BlockPool pool(32, {2, 3});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 8},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 5,
         .cache_blocks_per_lcm_block = 3,
         .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/8, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    ASSERT_EQ(tables[0].NumBlocks(), 1);
    ASSERT_EQ(tables[1].NumBlocks(), 4);

    const std::vector<std::string> hashes = ContentHashes({std::vector<std::int32_t>(8, 7)});
    CacheCompletedBlocksForTest(coordinator, tables, hashes, NextTestAccessEpoch(), /*first_new_prefix_page=*/0,
                                /*num_computed_tokens=*/8, CacheBoundaryKind::kChunk,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[0], 1, 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[0], 1, 1)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[0], 1, 2)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[0], 1, 3)));
    coordinator.Free(tables);

    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 8);
    ASSERT_EQ(probe.device.per_group[0].hits.size(), 1u);
    ASSERT_EQ(probe.device.per_group[1].hits.size(), 4u);
    EXPECT_EQ(probe.device.per_group[1].hits, (std::vector<std::uint8_t>{0, 0, 1, 1}));
}

TEST(MakeCoordinatorTest, RejectsNonPositiveLogicalPOrPacking) {
    BlockPool pool(8, {1});
    const std::vector<CacheGroupSpec> valid = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
    };
    EXPECT_THROW(
        MakeCoordinator(valid, /*prefix_granularity=*/0, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);
    EXPECT_THROW(
        MakeCoordinator(valid, /*prefix_granularity=*/-1, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);

    const std::vector<CacheGroupSpec> zero_packing = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 0, .block_granularity = 128},
    };
    EXPECT_THROW(
        MakeCoordinator(zero_packing, /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);
}

TEST(CacheCoordinatorTest, RejectsManagerGeometryThatDiffersFromDomainOrSpec) {
    BlockPool pool(8, {1});
    std::vector<CacheGroup> wrong_p;
    wrong_p.emplace_back(
        CacheGroupSpec{.attn_kind = AttnKind::kFull,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 48},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1, /*group_id=*/0, /*shard_count=*/1),
        std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(
        CacheCoordinator(std::move(wrong_p), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                         /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);

    std::vector<CacheGroup> wrong_k;
    wrong_k.emplace_back(
        CacheGroupSpec{.attn_kind = AttnKind::kFull,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 8,
                       .block_granularity = 128},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1, /*group_id=*/0, /*shard_count=*/1),
        std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(
        CacheCoordinator(std::move(wrong_k), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                         /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);

    // A spec sharded across two owners while its allocator balances nothing
    // would silently disable placement balancing; both must agree.
    BlockPool sharded_pool(8, {2});
    std::vector<CacheGroup> wrong_shards;
    wrong_shards.emplace_back(
        CacheGroupSpec{.attn_kind = AttnKind::kFull,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 2,
                       .block_granularity = 128,
                       .shard_count = 2},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/2, /*group_id=*/0, /*shard_count=*/1),
        std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(CacheCoordinator(std::move(wrong_shards), /*prefix_granularity=*/128, sharded_pool,
                                  /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                  /*stream_device_cache_to_host=*/false),
                 std::runtime_error);
}

TEST(CacheCoordinatorTest, RejectsManagerGroupIdThatDiffersFromItsIndex) {
    BlockPool pool(8, {1});
    std::vector<CacheGroup> groups;
    groups.emplace_back(CacheGroupSpec{.attn_kind = AttnKind::kFull,
                                       .sliding_window = 0,
                                       .cache_blocks_per_lcm_block = 1,
                                       .block_granularity = 128},
                        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1,
                                                         /*group_id=*/1, /*shard_count=*/1),
                        std::make_unique<FullAttnMatcher>());

    EXPECT_THROW(
        CacheCoordinator(std::move(groups), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                         /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false),
        std::runtime_error);
}

TEST(GroupAllocatorTest, ResolvesAffineKernelPageIdsAndRejectsInvalidLocations) {
    FullAttnManager full(/*block_granularity=*/128, /*cache_blocks_per_lcm_block=*/8);
    FullAttnManager state(/*block_granularity=*/128, /*cache_blocks_per_lcm_block=*/1);

    EXPECT_EQ(full.ResolveCacheBlockId({.lcm_block_id = 1, .slot_index = 0}), 1);
    EXPECT_EQ(full.ResolveCacheBlockId({.lcm_block_id = 1, .slot_index = 7}), 8);
    EXPECT_EQ(full.ResolveCacheBlockId({.lcm_block_id = 2, .slot_index = 0}), 9);
    EXPECT_EQ(state.ResolveCacheBlockId({.lcm_block_id = 2, .slot_index = 0}), 2);

    EXPECT_THROW(full.ResolveCacheBlockId({.lcm_block_id = 0, .slot_index = 0}), std::runtime_error);
    EXPECT_THROW(full.ResolveCacheBlockId({.lcm_block_id = 1, .slot_index = -1}), std::runtime_error);
    EXPECT_THROW(full.ResolveCacheBlockId({.lcm_block_id = 1, .slot_index = 8}), std::runtime_error);
    EXPECT_THROW(full.ResolveCacheBlockId({.lcm_block_id = std::numeric_limits<std::int32_t>::max(), .slot_index = 7}),
                 std::runtime_error);
}

TEST(CacheKeyTest, NamespaceGroupAndOffsetArePartOfIdentity) {
    const CacheKey base{
        .namespace_id = kDefaultCacheNamespaceId,
        .group_id = 0,
        .content_hash = "hash",
    };
    CacheKey other_group = base;
    other_group.group_id = 1;
    CacheKey other_namespace = base;
    other_namespace.namespace_id = 7;
    CacheKey other_offset = base;
    other_offset.page_offset = 1;

    EXPECT_NE(base, other_group);
    EXPECT_NE(base, other_namespace);
    EXPECT_NE(base, other_offset);
}

TEST(CacheKeyTest, HashPreservesNamespaceGroupAndContentIdentity) {
    const CacheKey base{
        .namespace_id = kDefaultCacheNamespaceId,
        .group_id = 0,
        .content_hash = "hash",
    };
    CacheKey other_namespace = base;
    other_namespace.namespace_id = 1;
    CacheKey other_group = base;
    other_group.group_id = 1;
    CacheKey other_content = base;
    other_content.content_hash = "other";
    CacheKey other_offset = base;
    other_offset.page_offset = 1;

    std::unordered_set<CacheKey, CacheKeyHash> keys{
        base, base, other_namespace, other_group, other_content, other_offset,
    };
    EXPECT_EQ(keys.size(), 5u);
}

TEST(MakeCoordinatorTest, UsesOneLogicalGranularityAcrossGroups) {
    BlockPool pool(16, {8, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 8, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    EXPECT_EQ(coord.PrefixGranularity(), 4);
    EXPECT_EQ(coord.GroupBlockGranularity(0), coord.GroupBlockGranularity(1));
}

TEST(CoordinatorMatchTest, BothGroupsAllMiss) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(CoordinatorMatchTest, CommonIsMinCoverageFullDeeperThanSwa) {
    // full caches 4 contiguous pages; swa (window 10 -> pages_needed 3)
    // caches only the last 3. Common = min(4, 3) = 3.
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    // swa front 3-run (a TAIL run would null-pad back to index 0 -> coverage 4).
    CacheForGroup(coord, pool, ch[0], 1);
    CacheForGroup(coord, pool, ch[1], 1);
    CacheForGroup(coord, pool, ch[2], 1);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 12);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 3u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 3u);
    // Full had 4 real hits, truncated to 3 -> num_hit recomputed to 3.
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 3);
}

TEST(CoordinatorMatchTest, TrimmedFullHitsDoNotRefreshAccessEpoch) {
    BlockPool pool(11, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);
    CacheForGroup(coord, pool, ch[2], 0);
    const std::int32_t trimmed = CacheForGroup(coord, pool, ch[3], 0);
    CacheForGroup(coord, pool, ch[0], 1);
    CacheForGroup(coord, pool, ch[1], 1);
    CacheForGroup(coord, pool, ch[2], 1);
    const CacheBlockLocation trimmed_location{.lcm_block_id = trimmed, .slot_index = 0};
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> before =
        coord.GroupPrefixIndex(0).MetadataFor(pool, trimmed_location);
    ASSERT_TRUE(before);
    CoordinatorMatch match = MatchPrefixForTest(coord, ch).device;
    ASSERT_EQ(match.num_common_tokens, 12);

    const std::optional<PrefixCacheIndex::CachedBlockMetadata> after =
        coord.GroupPrefixIndex(0).MetadataFor(pool, trimmed_location);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->last_access_epoch, before->last_access_epoch);
}

TEST(CoordinatorMatchTest, ProbeDoesNotRefreshAccessEpoch) {
    BlockPool pool(3, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    const std::int32_t oldest = CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);
    const CacheBlockLocation oldest_location{.lcm_block_id = oldest, .slot_index = 0};
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> before =
        coord.GroupPrefixIndex(0).MetadataFor(pool, oldest_location);
    ASSERT_TRUE(before);

    CacheCoordinator::PrefixProbe probe = coord.ProbePrefix(std::span<const std::string>{ch}.first(1));
    EXPECT_EQ(probe.device.num_common_tokens, 4);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> after =
        coord.GroupPrefixIndex(0).MetadataFor(pool, oldest_location);
    ASSERT_TRUE(after);
    EXPECT_EQ(after->last_access_epoch, before->last_access_epoch);
}

TEST(CacheCoordinatorTest, ClearDeviceCacheLeavesHostCacheUntouched) {
    BlockPool device_pool(4, {1});
    BlockPool host_pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});

    CacheForGroup(coordinator, device_pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, host_pool, hashes[0], /*group_id=*/0);

    ASSERT_TRUE(coordinator.ClearDeviceCache());
    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 4);
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), 4);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 3);
}

TEST(CacheCoordinatorTest, ClearCacheRemovesDeviceAndHostEntries) {
    BlockPool device_pool(4, {1});
    BlockPool host_pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});

    CacheForGroup(coordinator, device_pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, host_pool, hashes[0], /*group_id=*/0);

    ASSERT_TRUE(coordinator.CacheIsClearable(/*include_host=*/true));
    ASSERT_TRUE(coordinator.ClearCache());
    EXPECT_TRUE(coordinator.CacheIsClearable(/*include_host=*/true));
    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 0);
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), 4);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 4);
}

TEST(CacheCoordinatorTest, ClearDeviceCacheRejectsPinnedEntryWithoutPartialMutation) {
    BlockPool pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[1], /*group_id=*/0);
    CacheBlockRef pin = coordinator.AcquireDeviceCachedBlock(Key(hashes[0], /*group_id=*/0));

    EXPECT_FALSE(coordinator.ClearDeviceCache());
    EXPECT_FALSE(coordinator.CacheIsClearable(/*include_host=*/false));
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 8);

    pin.reset();
    EXPECT_TRUE(coordinator.CacheIsClearable(/*include_host=*/false));
    EXPECT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 0);
}

std::vector<GroupDemand> FreshDemands(std::vector<BlockTable>& tables, std::span<const std::int32_t> tokens) {
    EXPECT_EQ(tables.size(), tokens.size());
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (std::size_t i = 0; i < tables.size(); ++i) {
        demands.push_back(GroupDemand{.table = &tables[i], .extent = DenseGrowth{tokens[i]}});
    }
    return demands;
}

TEST(CacheCoordinatorAdmissionTest, ReportsOnlyFreshlyAllocatedKernelPageIds) {
    BlockPool pool(4, {4, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> first_demands = {
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{5}},
        GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kRotate}},
    };
    const std::optional<CacheCoordinator::AdmissionResult> first =
        coordinator.Admit(coordinator.ProbePrefix({}), first_demands, RequestProgress{}, std::nullopt);

    ASSERT_TRUE(first);
    ASSERT_EQ(first->new_page_ids.size(), 2u);
    EXPECT_EQ(first->new_page_ids[0], coordinator.Allocator(0).BlockTablePageIds(tables[0]));
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_EQ(first->new_page_ids[1], coordinator.Allocator(1).BlockTablePageIds(tables[1]))
        << "both State slots are fresh: the zeroed zero state and the live slot";

    // One more token fits the full group's block, and a decode step releases
    // the State input without acquiring: nothing fresh to zero.
    std::vector<GroupDemand> tail_demands = {
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{1}},
        GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kRelease}},
    };
    const std::optional<CacheCoordinator::AdmissionResult> tail =
        coordinator.Admit(coordinator.ProbePrefix({}), tail_demands, RequestProgress{}, first->access_epoch);

    ASSERT_TRUE(tail);
    ASSERT_EQ(tail->new_page_ids.size(), 2u);
    EXPECT_TRUE(tail->new_page_ids[0].empty());
    EXPECT_TRUE(tail->new_page_ids[1].empty());
}

TEST(CacheCoordinatorAdmissionTest, AcquireMatchedFullPrefixUsesOneAccessEpoch) {
    BlockPool pool(8, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});

    std::vector<BlockTable> cached_tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, cached_tables, /*num_tokens=*/16));
    CacheFullBlocksForTest(coordinator, cached_tables, hashes);
    std::vector<CacheBlockLocation> locations;
    for (const CacheBlockRef& block_ref : cached_tables[0].Blocks()) {
        locations.push_back(block_ref->Location());
    }
    coordinator.Free(cached_tables);

    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> no_new_tokens{0};
    std::vector<GroupDemand> hit_demands = FreshDemands(hit_tables, no_new_tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands, RequestProgress{}, std::nullopt));

    std::vector<std::uint64_t> access_epochs;
    for (CacheBlockLocation location : locations) {
        const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
            coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
        ASSERT_TRUE(metadata);
        access_epochs.push_back(metadata->last_access_epoch);
    }
    ASSERT_EQ(access_epochs.size(), 4u);
    EXPECT_TRUE(
        std::ranges::all_of(access_epochs, [&](std::uint64_t epoch) { return epoch == access_epochs.front(); }));
}

TEST(CacheCoordinatorAdmissionTest, LaterChunksReuseRequestAccessEpoch) {
    BlockPool pool(8, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    std::vector<BlockTable> tables(coordinator.NumGroups());

    const std::optional<CacheCoordinator::AdmissionResult> first = AdmitForTest(coordinator, tables, /*num_tokens=*/8);
    ASSERT_TRUE(first);

    std::vector<GroupDemand> second_demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{8}}};
    const RequestProgress second_progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = std::span<const std::string>(hashes).first(2),
                .first_new_prefix_page = 0,
                .boundary_kind = CacheBoundaryKind::kChunk,
            },
        .num_computed_tokens = 8,
    };
    const std::optional<CacheCoordinator::AdmissionResult> second =
        coordinator.Admit(coordinator.ProbePrefix({}), second_demands, second_progress, first->access_epoch);
    ASSERT_TRUE(second);
    EXPECT_EQ(second->access_epoch, first->access_epoch);

    std::vector<GroupDemand> final_publication{GroupDemand{.table = &tables[0]}};
    const RequestProgress final_progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 2,
                .boundary_kind = CacheBoundaryKind::kEndpoint,
            },
        .num_computed_tokens = 16,
    };
    const std::optional<CacheCoordinator::AdmissionResult> final =
        coordinator.Admit(coordinator.ProbePrefix({}), final_publication, final_progress, first->access_epoch);
    ASSERT_TRUE(final);

    for (const CacheBlockRef& block_ref : tables[0].Blocks()) {
        const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
            coordinator.GroupPrefixIndex(0).MetadataFor(pool, block_ref->Location());
        ASSERT_TRUE(metadata);
        EXPECT_EQ(metadata->last_access_epoch, first->access_epoch);
    }
}

TEST(CacheCoordinatorAdmissionTest, ProbeAndRejectedAdmissionDoNotAdvanceAccessEpoch) {
    BlockPool pool(1, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}});
    std::vector<BlockTable> owner_tables(coordinator.NumGroups());

    const std::optional<CacheCoordinator::AdmissionResult> owner =
        AdmitForTest(coordinator, owner_tables, /*num_tokens=*/4);
    ASSERT_TRUE(owner);
    coordinator.CacheFullBlocks(owner_tables, hashes, owner->access_epoch, /*first_slot=*/0, CacheBoundaryKind::kChunk);
    const CacheBlockLocation location = owner_tables[0].Blocks().front()->Location();

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
        coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, owner->access_epoch);

    std::vector<BlockTable> rejected_tables(coordinator.NumGroups());
    std::vector<GroupDemand> rejected_demands{
        GroupDemand{.table = &rejected_tables[0], .extent = DenseGrowth{4}},
    };
    EXPECT_FALSE(coordinator.Admit(std::move(probe), rejected_demands, RequestProgress{}, std::nullopt));
    metadata = coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, owner->access_epoch);

    coordinator.Free(owner_tables);
    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    std::vector<GroupDemand> hit_demands{
        GroupDemand{.table = &hit_tables[0]},
    };
    const std::optional<CacheCoordinator::AdmissionResult> hit =
        coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(hit);
    EXPECT_EQ(hit->access_epoch, owner->access_epoch + 1);
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionPrefersOlderRequestBeforePosition) {
    BlockPool pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> old_hashes = ContentHashes({{1, 1, 1, 1}});
    const std::vector<std::string> new_hashes = ContentHashes({{2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});

    std::vector<BlockTable> old_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> old_request =
        AdmitForTest(coordinator, old_tables, /*num_tokens=*/4);
    ASSERT_TRUE(old_request);
    coordinator.CacheFullBlocks(old_tables, old_hashes, old_request->access_epoch, /*first_slot=*/0,
                                CacheBoundaryKind::kChunk);
    coordinator.Free(old_tables);

    std::vector<BlockTable> new_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> new_request =
        AdmitForTest(coordinator, new_tables, /*num_tokens=*/12);
    ASSERT_TRUE(new_request);
    coordinator.CacheFullBlocks(new_tables, new_hashes, new_request->access_epoch, /*first_slot=*/0,
                                CacheBoundaryKind::kChunk);
    coordinator.Free(new_tables);

    std::vector<BlockTable> contender(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, contender, /*num_tokens=*/4));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(old_hashes[0], 0)));
    for (const std::string& hash : new_hashes) {
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hash, 0)));
    }
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionIsSuffixFirstWithinOneAccessEpoch) {
    BlockPool pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});

    std::vector<BlockTable> cached_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> cached_request =
        AdmitForTest(coordinator, cached_tables, /*num_tokens=*/16);
    ASSERT_TRUE(cached_request);
    coordinator.CacheFullBlocks(cached_tables, hashes, cached_request->access_epoch, /*first_slot=*/0,
                                CacheBoundaryKind::kChunk);
    coordinator.Free(cached_tables);

    std::vector<BlockTable> contender(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, contender, /*num_tokens=*/4));
    for (std::size_t i = 0; i + 1 < hashes.size(); ++i) {
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[i], 0)));
    }
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes.back(), 0)));
}

TEST(CacheCoordinatorAdmissionTest, RejectedCachedHitDoesNotRefreshAccessEpoch) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t cached_hit = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef pinned = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(pinned);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    const CacheBlockLocation cached_location{.lcm_block_id = cached_hit, .slot_index = 0};
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> before =
        coordinator.GroupPrefixIndex(0).MetadataFor(pool, cached_location);
    ASSERT_TRUE(before);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    for (int retry = 0; retry < 2; ++retry) {
        CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(std::span<const std::string>{hashes}.first(1));
        EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands, RequestProgress{}, std::nullopt));
        EXPECT_TRUE(tables[0].Blocks().empty());
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, cached_location));
        const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
            coordinator.GroupPrefixIndex(0).MetadataFor(pool, cached_location);
        ASSERT_TRUE(metadata);
        EXPECT_EQ(metadata->last_access_epoch, before->last_access_epoch);
        EXPECT_FALSE(metadata->was_acquired);
    }
}

TEST(CacheCoordinatorAdmissionTest, UsesPoolAllocationOrderForEmptyParents) {
    BlockPool pool(3, {1});
    std::vector<CacheBlockRef> held = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(held.size(), 3u);
    const CacheBlockLocation first_released = held[2]->Location();
    const CacheBlockLocation second_released = held[0]->Location();
    held[2].reset();
    held[0].reset();

    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), first_released);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), second_released);
}

TEST(CacheCoordinatorAdmissionTest, PacksSlotsFromOneNewParentTogether) {
    BlockPool pool(2, {2, 1});
    std::vector<CacheBlockRef> held = pool.AcquireBlocks(/*group_id=*/1, /*num=*/2);
    ASSERT_EQ(held.size(), 2u);
    const CacheBlockLocation first_released = held[1]->Location();
    held[1].reset();
    held[0].reset();

    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, first_released.lcm_block_id);
    EXPECT_EQ(tables[0].Blocks()[1]->Location().lcm_block_id, first_released.lcm_block_id);
    EXPECT_NE(tables[0].Blocks()[0]->Location().slot_index, tables[0].Blocks()[1]->Location().slot_index);
}

TEST(CacheCoordinatorAdmissionTest, FreeCachedHitCostsNoAdditionalPlacement) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, hit_parent);
    EXPECT_NE(tables[0].Blocks()[1]->Location().lcm_block_id, hit_parent);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(CacheCoordinatorAdmissionTest, ReservedCapacityLivesInBlockTableUntilConsumed) {
    BlockPool pool(3, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{1};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    demands[0].reserve_tokens = 8;

    const std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(result);

    EXPECT_TRUE(result->load_pairs.empty());
    EXPECT_EQ(tables[0].NumBlocks(), 3);
    EXPECT_EQ(coordinator.GroupBlocksNeededFor(0, tables[0], /*num_tokens=*/8), 0);
    const std::int32_t free_before_consume = pool.NumEmptyLcmBlocks();
    coordinator.ConsumeReservedTokens(tables, /*num_tokens=*/8);
    EXPECT_EQ(tables[0].AvailableTokens(), 3);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before_consume);
}

TEST(CacheCoordinatorAdmissionTest, HeterogeneousGroupsSharePartialAndEmptyParents) {
    BlockPool pool(2, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{4, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt));

    ASSERT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), (CacheBlockLocation{.lcm_block_id = hit_parent, .slot_index = 1}));
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, hit_parent);
}

TEST(CacheCoordinatorAdmissionTest, PacksDemandBeforeConsumingAnotherGroupsOnlyParent) {
    BlockPool pool(2, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(coordinator, pool, hashes[0], 0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    const std::int32_t packed_parent = tables[0].Blocks()[0]->Location().lcm_block_id;
    EXPECT_EQ(tables[0].Blocks()[1]->Location().lcm_block_id, packed_parent);
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, packed_parent);
}

TEST(CacheCoordinatorAdmissionTest, RetriesWithCompactPackingWhenGreedyFreeSlotsWouldReject) {
    BlockPool pool(3, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t first_partial_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef fills_first_parent = pool.AcquireBlock(/*group_id=*/0);
    const std::int32_t second_partial_parent = CacheForGroup(coordinator, pool, hashes[1], 0);
    fills_first_parent.reset();
    ASSERT_NE(first_partial_parent, second_partial_parent);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 8};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, tables[0].Blocks()[1]->Location().lcm_block_id);
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, tables[1].Blocks()[1]->Location().lcm_block_id);
}

TEST(CacheCoordinatorAdmissionTest, KeepsCachedChildrenWhenUnpackedPlacementFits) {
    BlockPool pool(3, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef fills_first_parent = pool.AcquireBlock(/*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    fills_first_parent.reset();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
}

TEST(CacheCoordinatorAdmissionTest, UsesEmptyParentBeforeEvictingCachedChild) {
    BlockPool pool(2, {2});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 1);
    EXPECT_EQ(tables[0].NumBlocks(), 2);
}

TEST(CacheCoordinatorAdmissionTest, DoesNotShareFreeSlotsFromBoundForeignParent) {
    BlockPool pool(1, {4, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    CacheBlockRef pinned = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(pinned);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    EXPECT_TRUE(tables[0].Blocks().empty());
    EXPECT_TRUE(tables[1].Blocks().empty());
}

TEST(CacheCoordinatorAdmissionTest, EvictsOldestCachedChildNeededForCapacity) {
    BlockPool pool(1, {2});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionUsesAccessEpochInsteadOfPermanentHitClass) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> first_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> first =
        AdmitForTest(coordinator, first_tables, /*num_tokens=*/4);
    ASSERT_TRUE(first);
    coordinator.CacheFullBlocks(first_tables, std::span{hashes}.first(1), first->access_epoch, /*first_slot=*/0,
                                CacheBoundaryKind::kChunk);
    coordinator.Free(first_tables);

    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> no_new_tokens{0};
    std::vector<GroupDemand> hit_demands = FreshDemands(hit_tables, no_new_tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(std::span{hashes}.first(1)), hit_demands, RequestProgress{},
                                  std::nullopt));
    coordinator.Free(hit_tables);

    std::vector<BlockTable> second_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> second =
        AdmitForTest(coordinator, second_tables, /*num_tokens=*/4);
    ASSERT_TRUE(second);
    coordinator.CacheFullBlocks(second_tables, std::span{hashes}.subspan(1), second->access_epoch, /*first_slot=*/0,
                                CacheBoundaryKind::kChunk);
    coordinator.Free(second_tables);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    std::vector<BlockTable> fresh_tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, fresh_tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, OrdinarySwaKeepsLongerUnhitFrontier) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    constexpr std::uint64_t kSameAccessEpoch = 7;

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/7);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/3);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, MixedGroupTieEvictsNonClosedBeforeFullHistory) {
    BlockPool pool(3, {1, 1, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    constexpr std::uint64_t kSameAccessEpoch = 7;

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/1);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/1, kSameAccessEpoch,
                          /*logical_block_index=*/3, CacheBoundaryKind::kEndpoint);
    CacheBoundaryForGroup(coordinator, pool, hashes[2], /*group_id=*/2, kSameAccessEpoch,
                          /*logical_block_index=*/9);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .extent = DenseGrowth{4}},
        {.table = &tables[1], .extent = StateStep{StateStep::Kind::kKeep}},
        {.table = &tables[2]},
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[1], 1)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(2).Contains(pool, Key(hashes[2], 2)));
}

TEST(CacheCoordinatorAdmissionTest, RetriesWithFreshCursorsAfterPinnedAdmissionFails) {
    BlockPool pool(3, {1, 1, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    constexpr std::uint64_t kSameAccessEpoch = 7;

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/1, CacheBoundaryKind::kChunk);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/1, kSameAccessEpoch,
                          /*logical_block_index=*/3, CacheBoundaryKind::kEndpoint);
    CacheBoundaryForGroup(coordinator, pool, hashes[2], /*group_id=*/2, /*access_epoch=*/1,
                          /*logical_block_index=*/9, CacheBoundaryKind::kChunk);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .extent = DenseGrowth{4}},
        {.table = &tables[1], .extent = StateStep{StateStep::Kind::kKeep}},
        {.table = &tables[2]},
    };
    std::vector<CacheBlockRef> pins;
    for (std::uint32_t group = 0; group < 3; ++group) {
        pins.push_back(coordinator.GroupPrefixIndex(group).Find(pool, Key(hashes[group], group, 0)));
        ASSERT_TRUE(pins.back());
    }
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));
    for (const BlockTable& table : tables) {
        EXPECT_EQ(table.NumBlocks(), 0);
    }
    // A new plan must revisit exhausted streams after references are released.
    // The oldest epoch stays pinned; the unpinned tie still favors state.
    pins[0].reset();
    pins[1].reset();
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0, 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[1], 1, 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(2).Contains(pool, Key(hashes[2], 2, 0)));
}

TEST(CacheCoordinatorAdmissionTest, SkipsProtectedOldestEpochAndEvictsLaterEpochs) {
    BlockPool pool(3, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});

    // The first candidate epoch is entirely protected by the incoming prefix.
    // The two later epochs must still be considered to make space for the
    // remaining two pages of the request.
    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/0, CacheBoundaryKind::kChunk);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, /*access_epoch=*/2,
                          /*logical_block_index=*/1, CacheBoundaryKind::kChunk);
    CacheBoundaryForGroup(coordinator, pool, hashes[2], /*group_id=*/0, /*access_epoch=*/3,
                          /*logical_block_index=*/2, CacheBoundaryKind::kChunk);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .extent = DenseGrowth{8}},
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(std::span{hashes}.first(1)), demands, RequestProgress{},
                                  std::nullopt));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], /*group_id=*/0, /*page_offset=*/0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], /*group_id=*/0, /*page_offset=*/0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], /*group_id=*/0, /*page_offset=*/0)));
    EXPECT_EQ(tables[0].NumBlocks(), 3);
}

TEST(CacheCoordinatorAdmissionTest, ProspectiveUncachedReclaimDoesNotEvictCachedBlock) {
    BlockPool pool(3, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::string cached_hash = ContentHashes({{1, 1, 1, 1}}).front();
    CacheBoundaryForGroup(coordinator, pool, cached_hash, /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/0, CacheBoundaryKind::kChunk);

    // Two prefill steps fill the pool: the cached checkpoint, the first
    // chunk's output (now slot 0) and the second chunk's live slot.
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = StateStep{StateStep::Kind::kRotate}}};
    const RequestProgress progress{.num_computed_tokens = 4};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    // The third step's rotation releases the request's own unpublished input
    // block, which funds its fresh live slot: the cached checkpoint stays.
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{.num_computed_tokens = 8},
                                  std::nullopt));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(cached_hash, 0)));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(CacheCoordinatorAdmissionTest, RebindsOnlyAfterEvictingWholeForeignParent) {
    BlockPool pool(1, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    ASSERT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{1});
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
}

TEST(CacheCoordinatorAdmissionTest, UsesLocalSlotAfterEvictingAnotherWholeParent) {
    BlockPool pool(2, {2});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    const std::int32_t old_full_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    EXPECT_EQ(CacheForGroup(coordinator, pool, hashes[1], 0), old_full_parent);
    const std::int32_t newest_partial_parent = CacheForGroup(coordinator, pool, hashes[2], 0);
    ASSERT_NE(old_full_parent, newest_partial_parent);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, newest_partial_parent);
    EXPECT_EQ(tables[0].Blocks()[1]->Location().lcm_block_id, old_full_parent);
    EXPECT_EQ(tables[0].Blocks()[2]->Location().lcm_block_id, old_full_parent);
}

TEST(CacheCoordinatorAdmissionTest, RestoresOlderSiblingWhenMiddleCandidateFreesRequiredParent) {
    BlockPool pool(2, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    GroupAllocator& manager = coordinator.Allocator(0);
    std::vector<CacheBlockRef> cached = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(cached.size(), 3u);
    ASSERT_EQ(cached[0]->Location().lcm_block_id, cached[1]->Location().lcm_block_id);
    ASSERT_NE(cached[0]->Location().lcm_block_id, cached[2]->Location().lcm_block_id);
    const std::int32_t middle_candidate_parent = cached[2]->Location().lcm_block_id;

    coordinator.GroupPrefixIndex(0).Register(pool, cached[0], Key(hashes[0], 0), /*access_epoch=*/1,
                                             /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                             /*newly_cached=*/nullptr);
    coordinator.GroupPrefixIndex(0).Register(pool, cached[1], Key(hashes[1], 0), /*access_epoch=*/3,
                                             /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                             /*newly_cached=*/nullptr);
    coordinator.GroupPrefixIndex(0).Register(pool, cached[2], Key(hashes[2], 0), /*access_epoch=*/2,
                                             /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                             /*newly_cached=*/nullptr);
    cached.clear();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_EQ(tables[1].Blocks()[0]->Location().lcm_block_id, middle_candidate_parent);
}

TEST(CacheCoordinatorAdmissionTest, ProspectiveHitParentCannotBecomeVictim) {
    BlockPool pool(2, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t protected_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    CacheForGroup(coordinator, pool, hashes[0], 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(std::span<const std::string>{hashes}.first(1));

    EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands, RequestProgress{}, std::nullopt));
    EXPECT_EQ(pool.BoundGroup(protected_parent), std::optional<std::uint32_t>{0});
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
    EXPECT_TRUE(tables[0].Blocks().empty());
    EXPECT_TRUE(tables[1].Blocks().empty());
}

TEST(CacheCoordinatorAdmissionTest, ReclaimsTableOwnerBeforeEvictingProspectiveVictim) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, tables, hashes);
    const CacheBlockLocation reclaimed = tables[0].Blocks()[0]->Location();

    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = DenseGrowth{4}}};
    const RequestProgress progress{.num_computed_tokens = 8};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt));

    ASSERT_EQ(tables[0].NumBlocks(), 3);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    ASSERT_TRUE(tables[0].Blocks()[2]);
    EXPECT_EQ(tables[0].Blocks()[2]->Location(), reclaimed);
}

TEST(CacheCoordinatorAdmissionTest, EvictsProspectiveVictimCachedDuringCommit) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    const CacheBlockLocation reclaimed = tables[0].Blocks()[0]->Location();

    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = DenseGrowth{4}}};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 0,
                .boundary_kind = CacheBoundaryKind::kChunk,
            },
        .num_computed_tokens = 8,
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    ASSERT_TRUE(tables[0].Blocks()[2]);
    EXPECT_EQ(tables[0].Blocks()[2]->Location(), reclaimed);
}

TEST(CacheCoordinatorAdmissionTest, RejectsProspectiveVictimWithAnExtraOwner) {
    BlockPool pool(2, {1});
    const std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, tables, hashes);

    const std::array<CacheKey, 1> first_key{Key(hashes[0], /*group_id=*/0)};
    GroupAllocator& manager = coordinator.Allocator(0);
    PrefixMatch extra_owner = coordinator.GroupPrefixIndex(0).AcquireMatched(
        pool, first_key, /*begin_blocks=*/0,
        coordinator.GroupMatcher(0).Probe(coordinator.GroupPrefixIndex(0), pool, first_key, /*begin_blocks=*/0,
                                          /*max_blocks=*/1, /*extra_hits=*/nullptr),
        NextTestAccessEpoch());
    ASSERT_EQ(extra_owner.NumHitBlocks(), 1);

    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = DenseGrowth{4}}};
    const RequestProgress progress{.num_computed_tokens = 8};
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_TRUE(tables[0].Blocks()[0]);
}

TEST(CoordinatorMatchTest, SwaMissForcesZeroCommon) {
    // full caches 2 pages, swa caches nothing -> common = min(2, 0) = 0.
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    EXPECT_EQ(m.per_group[0].blocks.size(), 0u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 0u);
}

TEST(CoordinatorAllocTest, ColdStartAllocatesAlignedPages) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    EXPECT_EQ(prefix.device.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.extent = DenseGrowth{8}}));
    // 8 tokens / page 4 = 2 pages in EACH group; tables aligned.
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, ClaimsCommonPrefixThenAllocatesRemainder) {
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    // swa window 4 -> pages_needed 1, so a single cached front page is a hit.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[0], 1);

    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    // 8 tokens total, 1 page (4 tokens) common -> 4 uncached tokens -> +1 page each.
    ASSERT_TRUE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.extent = DenseGrowth{4}}));
    EXPECT_EQ(tables[0].NumBlocks(), 2);  // 1 claimed + 1 allocated
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, CrossGroupShortfallAllocatesNothing) {
    BlockPool pool(5, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();
    // 12 tokens -> 3 pages per group = 6 needed, only 5 free -> fail, nothing taken.
    EXPECT_FALSE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.extent = DenseGrowth{12}}));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);  // untouched, not rolled back
}

TEST(CoordinatorStepTest, AcquireKeepsGroupsAligned) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 page each
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 more each
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorStepTest, AcquireShortfallAllocatesNothing) {
    BlockPool pool(3, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();
    // 2 pages per group (8 tokens) = 4 blocks, only 3 free -> fail, nothing taken.
    EXPECT_FALSE(AdmitForTest(coord, tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(CoordinatorStepTest, CacheFullBlocksThenMatchHits) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 page each
    CacheFullBlocksForTest(coord, tables, ch);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
}

TEST(CoordinatorStepTest, FreeReturnsAllGroups) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 8));  // 2 pages each = 4 blocks
    std::int32_t free_mid = pool.NumEmptyLcmBlocks();
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_mid + 4);
}

TEST(CoordinatorStepTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});

    // Request A: cold, allocate 2 pages each, cache both, free.
    {
        CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
        EXPECT_EQ(prefix.device.num_common_tokens, 0);
        std::vector<BlockTable> a(2);
        ASSERT_TRUE(AdmitForTest(coord, a, std::move(prefix), GroupDemand{.extent = DenseGrowth{8}}));
        CacheFullBlocksForTest(coord, a, ch);
        coord.Free(a);
    }
    // Request B: shares the prefix -> common 2 pages in both groups.
    {
        CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
        EXPECT_EQ(prefix.device.num_common_tokens, 8);
        std::vector<BlockTable> b(2);
        ASSERT_TRUE(AdmitForTest(coord, b, std::move(prefix), GroupDemand{}));
        EXPECT_EQ(b[0].NumBlocks(), 2);
        EXPECT_EQ(b[1].NumBlocks(), 2);
        coord.Free(b);
    }
}

TEST(CoordinatorStepTest, CacheFullBlocksAtSlotOffsetExtendsPrefix) {
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 24));                   // 6 pages each
    CacheFullBlocksForTest(coord, tables, std::span(ch).first(4));  // prefill path: slots 0..3
    CacheFullBlocksForTest(coord, tables, std::span(ch).subspan(4), /*first_slot=*/4);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 24);
    ASSERT_EQ(m.per_group.size(), 2u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 6u);
    for (std::size_t s = 0; s < 6; ++s) {
        EXPECT_EQ(m.per_group[0].blocks[s], tables[0].Blocks()[s]) << "slot " << s;
    }
    // swa window 4 -> pages_needed 1: tail run maps to the offset-registered slot-5 block.
    ASSERT_EQ(m.per_group[1].blocks.size(), 6u);
    EXPECT_EQ(m.per_group[1].blocks[5], tables[1].Blocks()[5]);
}

TEST(CoordinatorStepTest, CacheFullBlocksAtOffsetSkipsSwaHoles) {
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 24));  // 6 pages each
    // num_computed=20 -> swa skipped = 20-4+1 = 17 -> 17/4 = 4 pages punched:
    // swa slots 0..3 are null holes.
    coord.ReclaimExpired(tables, /*num_computed_tokens=*/20);
    ASSERT_FALSE(tables[1].Blocks()[3]);
    ASSERT_TRUE(tables[1].Blocks()[4]);

    CacheFullBlocksForTest(coord, tables, std::span(ch).subspan(2), /*first_slot=*/2);
    for (std::size_t s = 2; s < 6; ++s) {
        EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[s], 0))) << "full slot " << s;
    }
    EXPECT_FALSE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[2], 1)));
    EXPECT_FALSE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[3], 1)));
    EXPECT_TRUE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[4], 1)));
    EXPECT_TRUE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[5], 1)));
}

TEST(CoordinatorStepTest, CacheFullBlocksRejectsOutOfRangeFirstSlot) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{7, 7, 7, 7}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 8));  // 2 pages each
    EXPECT_THROW(CacheFullBlocksForTest(coord, tables, ch, /*first_slot=*/2), std::runtime_error);
    EXPECT_THROW(CacheFullBlocksForTest(coord, tables, ch, /*first_slot=*/-1), std::runtime_error);
}

TEST(CoordinatorMatchTest, SwaRunCutByFullBoundDropsToNoValidMatch) {
    // full covers 4; swa's tail run {2,3,4} bounded to 4 leaves run {2,3} <
    // pages_needed 3 with holes at 0,1 -> no valid swa match, common = 0.
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);
    CacheForGroup(coord, pool, ch[2], 0);
    CacheForGroup(coord, pool, ch[3], 0);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[4], 1);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_granularity=*/4);
}

TEST(CoordinatorMatchTest, FullShorterThanSwaBoundsSwaWithRunIntact) {
    // full covers 4; swa caches 1..4. Bounded to 4 the run {1,2,3} still reaches
    // pages_needed 3, so common stays 4 -- hole only OUTSIDE the last window.
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);
    CacheForGroup(coord, pool, ch[2], 0);
    CacheForGroup(coord, pool, ch[3], 0);
    CacheForGroup(coord, pool, ch[1], 1);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[4], 1);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_TRUE(m.per_group[1].blocks[1]);
    EXPECT_TRUE(m.per_group[1].blocks[2]);
    EXPECT_TRUE(m.per_group[1].blocks[3]);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_granularity=*/4);
}

TEST(CoordinatorMatchTest, SwaShorterThanFullTruncatesFull) {
    // swa's best valid match is 4 blocks [null, b1, b2, b3]; full truncates 5 -> 4.
    BlockPool pool(64, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[1], 1);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_granularity=*/4);
}

TEST(CoordinatorMatchTest, TwoSwaGroupsSharedBoundaryMatches) {
    // pages_needed 3. full: 5; both SWA groups cache {1,2,3}, so each accepts
    // the SAME boundary 4 in the single sweep -- no cascade, full truncates 5 -> 4.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    for (std::uint32_t g : {1u, 2u}) {
        CacheForGroup(coord, pool, ch[1], g);
        CacheForGroup(coord, pool, ch[2], g);
        CacheForGroup(coord, pool, ch[3], g);
    }

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 4);
    for (std::size_t i = 1; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 4u) << "group " << i;
        EXPECT_FALSE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_EQ(m.per_group[i].NumHitBlocks(), 3) << "group " << i;
        ExpectSwaWindowIntact(m.per_group[i], /*window=*/10, /*block_granularity=*/4);
    }
}

TEST(CoordinatorMatchTest, TwoSwaGroupsCascadingShrinkConverges) {
    // swaA first accepts boundary 5 (run {2,3,4}), then swaB shrinks the bound to 4 (run {1,2,3})
    // UNDER swaA's match; re-matching swaA at 4 fails its window ({1} missing) and cascades both
    // groups down to boundary 1 -- the greatest boundary ALL groups support (block 0 shared).
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[0], 1);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[4], 1);
    CacheForGroup(coord, pool, ch[0], 2);
    CacheForGroup(coord, pool, ch[1], 2);
    CacheForGroup(coord, pool, ch[2], 2);
    CacheForGroup(coord, pool, ch[3], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 1u) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_EQ(m.per_group[i].NumHitBlocks(), 1) << "group " << i;
    }
}

TEST(CoordinatorMatchTest, SwaGroupOrderDoesNotChangeConvergedCommon) {
    // The cascade fixture above with the two window groups swapped: convergence must land on the
    // same greatest common boundary regardless of sweep order among non-closed groups.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // = swaB above
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // = swaA above
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 1);
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);
    CacheForGroup(coord, pool, ch[2], 0);
    CacheForGroup(coord, pool, ch[3], 0);
    CacheForGroup(coord, pool, ch[0], 2);
    CacheForGroup(coord, pool, ch[2], 2);
    CacheForGroup(coord, pool, ch[3], 2);
    CacheForGroup(coord, pool, ch[4], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
}

TEST(CoordinatorMatchTest, MultiWindowThreeGroupsSharedBoundary) {
    // Mixed window sizes on one pool (the W=128 / W=4 / full shape, scaled to P=2: W=6 needs a
    // 3-page tail, W=2 needs 1): all three groups support boundary 5 and the pool stays unified.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[4], 1);
    CacheForGroup(coord, pool, ch[4], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 10);
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 5);
    EXPECT_EQ(m.per_group[1].blocks.size(), 5u);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 3);  // holes at 0,1
    EXPECT_EQ(m.per_group[2].blocks.size(), 5u);
    EXPECT_EQ(m.per_group[2].NumHitBlocks(), 1);  // holes at 0..3
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/6, /*block_granularity=*/2);
    ExpectSwaWindowIntact(m.per_group[2], /*window=*/2, /*block_granularity=*/2);
}

TEST(CoordinatorMatchTest, MultiWindowCascadeToZero) {
    // W=2's only cached page (3) forces boundary 4, where W=6 cannot cover {1,2,3} -> its
    // re-match collapses to 0 and drags the small window with it: no boundary works, common = 0.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[2], 1);
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[4], 1);
    CacheForGroup(coord, pool, ch[3], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    for (const PrefixMatch& g : m.per_group) {
        EXPECT_TRUE(g.blocks.empty());
    }
}

TEST(CoordinatorMatchTest, DeepCascadeRequiresSecondConvergeSweep) {
    // Forces the converge loop through a SECOND productive sweep: swaA re-matches to boundary 3,
    // then swaB's re-match lands at 2 UNDER swaA's already re-matched boundary, so swaA must
    // re-match again. Hand-computed greatest common boundary: 2 pages (A@3 valid but B@3 lacks
    // block 2; at 2 both windows clamp to begin over {0,1}).
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // swaA, needed = 3
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // swaB, needed = 3
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0},
                                                 {1, 1, 1, 1},
                                                 {2, 2, 2, 2},
                                                 {3, 3, 3, 3},
                                                 {4, 4, 4, 4},
                                                 {5, 5, 5, 5},
                                                 {6, 6, 6, 6},
                                                 {7, 7, 7, 7}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    for (int j : {0, 1, 2, 4, 5, 6}) CacheForGroup(coord, pool, ch[static_cast<std::size_t>(j)], 1);  // swaA
    for (int j : {0, 1, 3, 4, 5}) CacheForGroup(coord, pool, ch[static_cast<std::size_t>(j)], 2);     // swaB

    // Sweep: A {4,5,6} -> 7; B {3,4,5} -> 6. Converge pass 1: A@6 -> run {0,1,2} -> 3;
    // B@3 -> bottoming run {0,1} -> 2. Pass 2: A@2 -> bottoming {0,1} -> 2. Stable.
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 2u) << "group " << i;
        EXPECT_EQ(m.per_group[i].NumHitBlocks(), 2) << "group " << i;
    }
}

TEST(CoordinatorMatchTest, MultiWindowSlideCreditSumsPerWindow) {
    // Retention is per-window: at 10 computed tokens W=6 has slid past 2 pages, W=2 past 4,
    // full past none -- the gate credit is their sum over the shared pool.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/10));

    std::int32_t credit = 0;
    for (std::int32_t i = 0; i < coord.NumGroups(); ++i) {
        credit += coord.GroupBlocksReclaimableAt(i, tables[static_cast<std::size_t>(i)],
                                                 /*num_computed_tokens=*/10,
                                                 /*count_uncached=*/true);
    }
    EXPECT_EQ(credit, 6);  // 0 (full) + 2 (W=6) + 4 (W=2)

    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    coord.ReclaimExpired(tables, /*num_computed_tokens=*/10);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before + 6) << "reclaim must deliver exactly the credited pages";
    coord.Free(tables);
}

TEST(CoordinatorMatchTest, AllFullGroupsMinTruncationUnchanged) {
    BlockPool pool(32, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[0], 1);
    CacheForGroup(coord, pool, ch[1], 1);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 2);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 2);
}

TEST(CoordinatorMatchTest, SingleFullGroupUnchanged) {
    BlockPool pool(16, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 1u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 2);
}

TEST(CoordinatorMatchTest, SwaOnlyConfigKeepsTailRunWithLeadingHoles) {
    // No full bound: tail run {2,3,4} covers the window; leading holes null-pad to page 0.
    BlockPool pool(32, {1});
    std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 10,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(coord, pool, ch[2], 0);
    CacheForGroup(coord, pool, ch[3], 0);
    CacheForGroup(coord, pool, ch[4], 0);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 20);
    ASSERT_EQ(m.per_group.size(), 1u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 5u);
    EXPECT_FALSE(m.per_group[0].blocks[0]);
    EXPECT_FALSE(m.per_group[0].blocks[1]);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 3);
    ExpectSwaWindowIntact(m.per_group[0], /*window=*/10, /*block_granularity=*/4);
}

TEST(CoordinatorAllocTest, RejectedAdmissionLeavesCachedPrefixUnclaimed) {
    BlockPool pool(5, {1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[0], 1);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();

    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    // Uncached 8 tokens -> 2 pages/group = 4 needed; 5 parents - 2 cached parents = 3 free -> fail.
    EXPECT_FALSE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.extent = DenseGrowth{8}}));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(CacheCoordinatorReclaimExpired, OnlySlidingWindowGroupEvicts) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    // 6 tokens -> 3 pages per group.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    ASSERT_EQ(tables[1].NumBlocks(), 3);

    const std::vector<std::int32_t> full_snapshot = BlockTableLcmBlockIds(tables[0]);

    // num_computed_tokens=5 -> swa skipped=5-4+1=2 -> skipped_blocks=2/2=1 -> page 0 evicted.
    coordinator.ReclaimExpired(tables, /*num_computed_tokens=*/5);

    ASSERT_EQ(tables[0].NumBlocks(), 3);
    const auto full_after = tables[0].Blocks();
    for (std::int32_t i = 0; i < tables[0].NumBlocks(); ++i) {
        ASSERT_TRUE(full_after[i]) << "full group got a null hole at " << i;
        EXPECT_EQ(full_after[i]->Location().lcm_block_id, full_snapshot[i]) << "full group block " << i << " changed";
    }

    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_TRUE(tables[1].Blocks()[1]);
    EXPECT_TRUE(tables[1].Blocks()[2]);
}

TEST(CoordinatorMatchTest, ThreeGroupsCommonIsMinCoverageAcrossAll) {
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    // Shortest window group first in index order: group 2's deeper match trims to
    // group 1's bound inside the sweep (the reverse order would be a cascade).
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[0], 1);
    CacheForGroup(coord, pool, ch[1], 1);
    CacheForGroup(coord, pool, ch[0], 2);
    CacheForGroup(coord, pool, ch[1], 2);
    CacheForGroup(coord, pool, ch[2], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "common = min(4, 2, 3) pages, in tokens (P=4)";
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[2].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].NumHitBlocks(), 2);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 2);
    EXPECT_EQ(m.per_group[2].NumHitBlocks(), 2);
}

TEST(CoordinatorMatchTest, ThreeGroupsOneAllMissForcesZeroCommon) {
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // Groups 0 and 2 fully cache both pages; group 1 caches nothing. The all-miss
    // group zeroes the bound before group 2 matches (the reverse would be a cascade).
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0) << "one group all-miss -> common 0";
}

TEST(CacheCoordinatorStoreCandidates, CollectsKeysWithoutPinningDeviceBlocks) {
    BlockPool pool(/*num_lcm_blocks=*/16, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    BlockPool host_pool(4, {1, 1});
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    CacheFullBlocksForTest(coordinator, tables, hashes, /*first_slot=*/0);
    std::vector<CacheCoordinator::StoreCandidate> pending = coordinator.TakePendingStores();

    ASSERT_EQ(pending.size(), 2u);  // sliding-window group only; full-attention deferred
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
    for (const BlockTable& table : tables) {
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            EXPECT_EQ(block_ref.use_count(), 2) << "only the request table and Device cache own the block";
        }
    }
    // Typed keys keep the group distinct without changing the content hash.
    std::unordered_set<CacheKey, CacheKeyHash> keys;
    for (const auto& c : pending) keys.insert(c.key);
    EXPECT_EQ(keys.size(), 2u);
    for (std::size_t i = 0; i < pending.size(); ++i) {
        EXPECT_EQ(pending[i].key, Key(hashes[i], /*group_id=*/1)) << "candidate " << i;
    }

    // Re-registering the same hashes yields nothing new (IsCached skip).
    CacheFullBlocksForTest(coordinator, tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());

    coordinator.Free(tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 12);  // four Manager-owned cache entries remain resident
}

TEST(CacheCoordinatorStoreCandidates, DisabledByDefaultCollectsNothing) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    CacheFullBlocksForTest(coordinator, tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());
    coordinator.Free(tables);
}

// Caller-side slide credit (the scheduler's batch gates run this exact per-group loop).
std::int32_t SlideCredit(const CacheCoordinator& coord, std::span<const BlockTable> tables,
                         std::int32_t num_computed_tokens) {
    std::int32_t total_freed = 0;
    for (std::int32_t i = 0; i < coord.NumGroups(); ++i) {
        total_freed += coord.GroupBlocksReclaimableAt(i, tables[static_cast<std::size_t>(i)], num_computed_tokens,
                                                      /*count_uncached=*/!coord.StreamsDeviceCacheToHost());
    }
    return total_freed;
}

// The exact slide-credit rule: collection-on credits a slide-out block only when it is cached;
// a pending store key does not pin the Device block before an operation is emitted.
TEST(CacheCoordinatorStoreCandidates, SlideCreditExcludesUncachedOnlyWhenCollecting) {
    BlockPool pool(16, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator off = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                           /*snapshot_pool=*/nullptr,
                                           /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(off.NumGroups());
    ASSERT_TRUE(AdmitForTest(off, tables, /*num_tokens=*/8));  // pages 0..3; N=8 slides out pages 0,1
    EXPECT_EQ(SlideCredit(off, tables, 8), 2) << "collection-off counts uncached ref-1 blocks";
    off.Free(tables);

    BlockPool host_pool(4, {1});
    CacheCoordinator on = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                          /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<BlockTable> tables2(on.NumGroups());
    ASSERT_TRUE(AdmitForTest(on, tables2, 8));
    EXPECT_EQ(SlideCredit(on, tables2, 8), 0) << "collection-on excludes uncached slide-out blocks";

    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    CacheFullBlocksForTest(on, tables2, hashes, 0);
    EXPECT_EQ(SlideCredit(on, tables2, 8), 2) << "pending store keys do not change ownership";
    on.TakePendingStores();
    on.Free(tables2);
}

TEST(CacheCoordinatorHostReplacement, ReusesOneColdChildBeforeRebindingAParent) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(1, {2, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    const CacheKey first = Key("first", 0);
    const CacheKey second = Key("second", 0);
    CacheBlockRef first_ref = host_pool.AcquireBlock(0);
    CacheBlockRef second_ref = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(first_ref, first);
    coordinator.CacheHostBlock(second_ref, second);
    first_ref.reset();
    second_ref.reset();

    CacheBlockRef replacement = coordinator.AcquireHostBlock(0, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(host_pool.BoundGroup(replacement->Location().lcm_block_id), 0u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 1);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(first));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(second));
}

TEST(CacheCoordinatorHostReplacement, RebindsACompleteEvictableParentAcrossGroups) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(1, {2, 1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    CacheBlockRef first = host_pool.AcquireBlock(0);
    CacheBlockRef second = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(first, Key("first", 0));
    coordinator.CacheHostBlock(second, Key("second", 0));
    first.reset();
    second.reset();

    CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(host_pool.BoundGroup(replacement->Location().lcm_block_id), 1u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
}

// Publish a host page for (hash, group) directly (the scheduler's store path minus the
// D2H write): allocate -> hash -> free leaves it cached-and-evictable, like a committed store.
std::int32_t HostPut(CacheCoordinator& coordinator, BlockPool& host_pool, const std::string& content_hash,
                     std::uint32_t gid) {
    const CacheKey key = Key(content_hash, gid);
    GroupAllocator& manager = coordinator.Allocator(static_cast<std::int32_t>(gid));
    CacheBlockRef block_ref = host_pool.AcquireBlock(gid);
    const std::int32_t id = block_ref->Location().lcm_block_id;
    coordinator.GroupPrefixIndex(static_cast<std::int32_t>(gid))
        .Register(host_pool, block_ref, key, NextTestAccessEpoch(), /*logical_block_index=*/-1,
                  CacheBoundaryKind::kChunk,
                  /*newly_cached=*/nullptr);
    block_ref.reset();
    return id;
}

TEST(CacheCoordinatorHostReplacement, BatchReusesSameGroupVictimsWithOneScan) {
    BlockPool device_pool(4, {2});
    BlockPool host_pool(2, {2});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 2,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    HostPut(coordinator, host_pool, "old-2", 0);
    HostPut(coordinator, host_pool, "old-3", 0);

    const std::array<std::uint32_t, 3> groups{0, 0, 0};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.requested, 3u);
    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_EQ(batch.stats.unallocated, 0u);
    EXPECT_EQ(batch.stats.same_group_scans, 1u);
    EXPECT_EQ(batch.stats.cross_group_scans, 0u);
}

TEST(CacheCoordinatorHostReplacement, BatchPreservesEmptyRefForPinnedShortfall) {
    BlockPool device_pool(2, {1});
    BlockPool host_pool(2, {1});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(pinned);
    coordinator.CacheHostBlock(pinned, Key("pinned", 0));

    const std::array<std::uint32_t, 2> groups{0, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    EXPECT_TRUE(batch.blocks[0]);
    EXPECT_FALSE(batch.blocks[1]);
    EXPECT_EQ(batch.stats.requested, 2u);
    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
    EXPECT_EQ(batch.stats.same_group_scans, 1u);
    EXPECT_EQ(batch.stats.cross_group_scans, 1u);
}

TEST(CacheCoordinatorHostReplacement, BatchGivesScarceFreeCapacityToFirstCandidateGroup) {
    BlockPool device_pool(2, {1, 1});
    BlockPool host_pool(1, {1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    const std::array<std::uint32_t, 2> groups{1, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    ASSERT_TRUE(batch.blocks[0]);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[0]->Location().lcm_block_id), 1u);
    EXPECT_FALSE(batch.blocks[1]);
    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
}

TEST(CacheCoordinatorHostReplacement, BatchGivesInterleavedGroupsFreeCapacityInCandidateOrder) {
    BlockPool device_pool(3, {1, 1});
    BlockPool host_pool(2, {1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    const std::array<std::uint32_t, 3> groups{0, 1, 0};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    ASSERT_TRUE(batch.blocks[0]);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[0]->Location().lcm_block_id), 0u);
    ASSERT_TRUE(batch.blocks[1]);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[1]->Location().lcm_block_id), 1u);
    EXPECT_FALSE(batch.blocks[2]);
    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
}

TEST(CacheCoordinatorHostReplacement, BatchRebindsParentsOnceAcrossMixedGroups) {
    BlockPool device_pool(4, {2, 2});
    BlockPool host_pool(2, {2, 2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    HostPut(coordinator, host_pool, "old-2", 0);
    HostPut(coordinator, host_pool, "old-3", 0);

    const std::array<std::uint32_t, 3> groups{1, 1, 0};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    ASSERT_TRUE(batch.blocks[0]);
    ASSERT_TRUE(batch.blocks[1]);
    ASSERT_TRUE(batch.blocks[2]);
    EXPECT_EQ(batch.blocks[0]->Location().lcm_block_id, batch.blocks[1]->Location().lcm_block_id);
    EXPECT_NE(batch.blocks[0]->Location().slot_index, batch.blocks[1]->Location().slot_index);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[0]->Location().lcm_block_id), 1u);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[2]->Location().lcm_block_id), 0u);
    EXPECT_NE(batch.blocks[0]->Location().lcm_block_id, batch.blocks[2]->Location().lcm_block_id);
    EXPECT_EQ(batch.stats.same_group_scans, 2u);
    EXPECT_EQ(batch.stats.cross_group_scans, 1u);
    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_EQ(batch.stats.unallocated, 0u);
}

TEST(CacheCoordinatorHostReplacement, BatchCrossGroupRebindingDoesNotRescanPlacementPoolPerVictim) {
    BlockPool device_pool(4, {2, 2});
    BlockPool host_pool(2, {2, 2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    HostPut(coordinator, host_pool, "old-2", 0);
    HostPut(coordinator, host_pool, "old-3", 0);

    const std::array<std::uint32_t, 3> groups{1, 1, 1};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.cross_group_scans, 1u);
}

TEST(CacheCoordinatorHostReplacement, BatchCrossGroupRebindingChoosesLowestCacheValueParent) {
    BlockPool device_pool(3, {1, 1, 1});
    BlockPool host_pool(2, {1, 1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    const std::int32_t older_parent = HostPut(coordinator, host_pool, "older", 0);
    const std::int32_t newer_parent = HostPut(coordinator, host_pool, "newer", 1);

    const std::array<std::uint32_t, 1> groups{2};
    const std::array<std::int32_t, 1> buckets{0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_TRUE(batch.blocks[0]);
    EXPECT_EQ(batch.blocks[0]->Location().lcm_block_id, older_parent);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("older", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("newer", 1)));
    EXPECT_EQ(host_pool.BoundGroup(newer_parent), 1u);
}

TEST(CacheCoordinatorHostReplacement, BatchDoesNotRebindPinnedParent) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(1, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/true);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    CacheBlockRef pinned = coordinator.GroupPrefixIndex(0).Find(host_pool, Key("old-0", 0));
    ASSERT_TRUE(pinned);

    const std::array<std::uint32_t, 1> groups{1};
    const std::array<std::int32_t, 1> buckets{0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    EXPECT_FALSE(batch.blocks[0]);
    EXPECT_EQ(host_pool.BoundGroup(1), 0u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-1", 0)));
    EXPECT_EQ(batch.stats.cross_group_scans, 1u);
    EXPECT_EQ(batch.stats.allocated, 0u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
}

// Cache slots [0, blocks) in the DEVICE pool for every group, so the merged MatchPrefix's
// device boundary lands exactly there (SWA's bottom-clamped run accepts any such floor).
void SeedDeviceFloor(BlockPool& pool, CacheCoordinator& coord, std::span<const std::string> ch, std::int32_t blocks) {
    for (std::int32_t g = 0; g < coord.NumGroups(); ++g) {
        for (std::int32_t j = 0; j < blocks; ++j) {
            (void)HostPut(coord, pool, ch[static_cast<std::size_t>(j)], static_cast<std::uint32_t>(g));
        }
    }
}

// Fixture constants: full P=2 + SWA W=4 -> pages_needed = (4-1+2-1)/2 = 2.
std::vector<CacheGroupSpec> HostExtSpecs() {
    return {
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2}};
}

TEST(CacheCoordinatorHostExtension, PreservesAllNullWindowExtensionSlots) {
    BlockPool pool(3, {1, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 1,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(3, {1, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{0, 0}, {1, 1}, {2, 2}});
    for (const std::string& hash : hashes) {
        (void)HostPut(coordinator, host_pool, hash, /*gid=*/0);
    }

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    const std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt);

    ASSERT_TRUE(result);
    EXPECT_EQ(result->host_prefix_tokens, 6);
    EXPECT_EQ(result->load_pairs.size(), 3u);
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_TRUE(std::ranges::all_of(tables[0].Blocks(),
                                    [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
    EXPECT_TRUE(std::ranges::none_of(tables[1].Blocks(),
                                     [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
}

TEST(CacheCoordinatorHostExtension, BothGroupsFullyPresent) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    std::vector<std::int32_t> fp, sp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 2; j <= 3; ++j) sp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);  // boundary 4 blocks * P=2 (floor 1 + extension 3)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2]}));
    // SWA tail at boundary 4 needs blocks [2, 4); extension slot for block 1 is a hole.
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{0, sp[0], sp[1]}));
    EXPECT_EQ(m.per_group[0].NumHitBlocks() + m.per_group[1].NumHitBlocks(), 5);
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 5);
}

TEST(CacheCoordinatorHostExtension, SwaTailMissShrinksBoundary) {
    // swa misses block 3 -> boundary shrinks to 3; tail at 3 = blocks [1, 3), which hits.
    // ext = 2, swa start = max(1, 3-2) = 1 = dev -> no holes; full's block-3 page stays unpinned.
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    std::vector<std::int32_t> fp, sp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 1; j <= 2; ++j) sp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 6);  // boundary 3 blocks * P=2 (floor 1 + extension 2)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1]}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{sp[0], sp[1]}));
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 4);
}

TEST(CacheCoordinatorHostExtension, FullGapCapsExtension) {
    // full misses block 2 -> boundary 2; swa tail at 2: start = max(1, 2-2) = 1 -> needs block 1 only.
    // ext = 1, both groups = {block-1 page}, 2 pins -- swa's deeper blocks 2..3 stay unused.
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    const std::int32_t fp1 = HostPut(coord, host_pool, ch[1], 0);
    (void)HostPut(coord, host_pool, ch[3], 0);  // gap at block 2
    std::vector<std::int32_t> sp;
    for (int j = 1; j <= 3; ++j) sp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 4);  // boundary 2 blocks * P=2 (floor 1 + extension 1)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp1}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{sp[0]}));
}

TEST(CacheCoordinatorHostExtension, EmptyStoreZeroExtension) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(5, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 2) << "no extension: boundary stays at the device floor";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(CacheCoordinatorHostExtension, DeviceBoundaryRespected) {
    // Host holds only blocks 0..1 (below dev=2): zero extension, and those entries stay unpinned.
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(5, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    for (int j = 0; j <= 1; ++j) {
        (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0);
        (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1);
    }

    SeedDeviceFloor(pool, coord, ch, 2);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 4) << "below-floor host pages extend nothing";
    // The below-dev entries were never probed: all four stay evictable.
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 0);
    EXPECT_TRUE(host_pool.AcquireBlock(/*group_id=*/0));
}

TEST(CacheCoordinatorHostExtension, MatchPinsPagesUntilResultDies) {
    BlockPool pool(16, {1, 1});
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6, {1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    for (int j = 1; j <= 3; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j = 2; j <= 3; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 5);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
    CacheBlockRef last_free = host_pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(last_free);
    EXPECT_FALSE(host_pool.AcquireBlock(/*group_id=*/0)) << "pinned cache entries must not be selected as victims";
    last_free.reset();
    m = {};
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 0);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
}

TEST(CacheCoordinatorHostExtension, DeepCascadeConverges) {
    // Host twin of the deep cascade (dev = 0 so the block math mirrors the device test):
    // convergence must land on extension 2 and take no refs along the way.
    BlockPool pool(64, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    BlockPool host_pool(32, {1, 1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0},
                                                 {1, 1, 1, 1},
                                                 {2, 2, 2, 2},
                                                 {3, 3, 3, 3},
                                                 {4, 4, 4, 4},
                                                 {5, 5, 5, 5},
                                                 {6, 6, 6, 6},
                                                 {7, 7, 7, 7}});

    for (int j = 0; j <= 7; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j : {0, 1, 2, 4, 5, 6}) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1);
    for (int j : {0, 1, 3, 4, 5}) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);  // boundary 2 blocks * P=4 (floor 0)
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 2u) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[1]) << "group " << i;
    }
    // Abandoned intermediate matches release; only the final 2 pages/group stay pinned.
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 6);
}

TEST(CacheCoordinatorHostExtension, MultiWindowGroupsExtendTogether) {
    // Host-tier twin of the multi-window device case (P=2, full + W=6 + W=2, dev=1): every
    // group supports host boundary 5, holes pad below each window's tail run.
    BlockPool pool(16, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(16, {1, 1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}});

    std::vector<std::int32_t> fp, ap;
    for (int j = 1; j <= 4; ++j) fp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 2; j <= 4; ++j) ap.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1));
    const std::int32_t bp4 = HostPut(coord, host_pool, ch[4], 2);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 10);  // boundary 5 blocks * P=2 (floor 1 + extension 4)
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2], fp[3]}));
    const std::int32_t hole = 0;
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{hole, ap[0], ap[1], ap[2]}));
    EXPECT_EQ(BlockIds(m.per_group[2].blocks), (std::vector<std::int32_t>{hole, hole, hole, bp4}));
    EXPECT_EQ(m.per_group[0].NumHitBlocks() + m.per_group[1].NumHitBlocks() + m.per_group[2].NumHitBlocks(),
              8);  // 4 full + 3 W=6 + 1 W=2
}

TEST(CacheCoordinatorHostExtension, MultiWindowCascadeConvergesToZeroExtension) {
    // W=2's only host page (block 3) caps the boundary at 4, where W=6 cannot cover {1,2,3};
    // its re-match collapses to the device boundary and the extension converges to zero with
    // nothing left pinned.
    BlockPool pool(16, {1, 1, 1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.attn_kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(16, {1, 1, 1});
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}});

    for (int j = 1; j <= 4; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j = 2; j <= 4; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1);
    (void)HostPut(coord, host_pool, ch[3], 2);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 2) << "extension converges to zero: boundary = device floor";
    for (const PrefixMatch& g : m.per_group) {
        EXPECT_TRUE(g.blocks.empty());
    }
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 0);
}

// ---------------------------------------------------------------------------
// Mamba-analog semantics: vLLM reduces a mamba/linear-attention group to the
// cache machinery via (a) hit = ONE aligned state snapshot found right-to-left,
// padded with nulls ([null]*i + [state]); (b) retention = only the last token's
// state lives (skipped = n-1, exactly our W=2 slide rule); (c) L2 = sliding
// window of one block. These tests pin that our SwaManager with a one-page
// window already produces those exact shapes -- the machinery a MambaManager
// (AttnKind::kState) would reuse unchanged.
// ---------------------------------------------------------------------------

TEST(CompletedBoundaryTest, HistoricalHashesWithoutBoundaryAreNotPublished) {
    BlockPool pool(8, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<GroupDemand> demands{{.table = &tables[0]}};

    // Retention progress alone names no completed page: nothing is published,
    // however many pages the table has filled.
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{.num_computed_tokens = 4},
                                  std::nullopt));
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
    coordinator.Free(tables);
}

TEST(CompletedBoundaryTest, RejectsEmptyCompletedPageRange) {
    BlockPool pool(8, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}});
    const std::vector<GroupDemand> demands{{.table = &tables[0]}};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 1,
                .boundary_kind = CacheBoundaryKind::kChunk,
            },
        .num_computed_tokens = 4,
    };

    EXPECT_THROW(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt), std::runtime_error);
    EXPECT_THROW(coordinator.CacheCompletedBlocks(tables, progress, /*access_epoch=*/1), std::runtime_error);
    coordinator.Free(tables);
}

TEST(CompletedBoundaryTest, RejectsCompletedPagesWithoutComputedProgress) {
    BlockPool pool(8, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}});
    const std::vector<GroupDemand> demands{{.table = &tables[0]}};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 0,
                .boundary_kind = CacheBoundaryKind::kChunk,
            },
    };

    EXPECT_THROW(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt), std::runtime_error);
    EXPECT_THROW(coordinator.CacheCompletedBlocks(tables, progress, /*access_epoch=*/1), std::runtime_error);
    coordinator.Free(tables);
}

TEST(CompletedBoundaryTest, PublicationWithoutCompletedPagesIsRejected) {
    BlockPool pool(8, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_THROW(coordinator.CacheCompletedBlocks(tables, RequestProgress{.num_computed_tokens = 4},
                                                  /*access_epoch=*/1),
                 std::runtime_error);
    coordinator.Free(tables);
}

TEST(SwaRegistrationTest, SwaBoundaryRequiresTrailingWindow) {
    BlockPool pool(32, {1});
    std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 9,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/16));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    std::vector<GroupDemand> demands{{.table = &tables[0]}};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = ch,
                // Only page 3 is new; the two-page resume tail crosses the prior chunk.
                .first_new_prefix_page = 3,
                .boundary_kind = CacheBoundaryKind::kChunk,
            },
        .num_computed_tokens = 16,
    };
    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[0], 0)));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[1], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[2], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[3], 0)));
    coord.Free(tables);
}

TEST(SwaRegistrationTest, UnalignedEndpointPublishesTrailingFullPages) {
    BlockPool pool(32, {1});
    std::vector<CacheGroupSpec> specs = {{.attn_kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 9,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/14));
    std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    std::vector<GroupDemand> demands{{.table = &tables[0]}};
    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = hashes,
                .first_new_prefix_page = 0,
                .boundary_kind = CacheBoundaryKind::kEndpoint,
            },
        .num_computed_tokens = 14,
    };

    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands, progress, std::nullopt));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    coord.Free(tables);
}

// ---- Bounded replay --------------------------------------------------------
//
// A sliding group declared replayable (CacheGroupSpec::replayable) leaves
// prefix caching: never matched, published or streamed, regenerated by the
// model from re-fed prompt tokens, while retention still slides its private
// pages out. Scheduler-level scenarios live in test_cache_scenarios.cpp
// (BoundedReplaySuite).

// full (closed, g=4) + replayable swa (W=8, g=4): P=4 so every prefix page is
// one block in both groups.
std::vector<CacheGroupSpec> ReplaySpecs() {
    return {
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 8,
                       .replayable = true,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 4}};
}

TEST(BoundedReplayCoordinator, MakeCoordinatorRejectsReplayOnFullOrStateGroups) {
    BlockPool pool(8, {1});
    const std::vector<CacheGroupSpec> full_replay = {CacheGroupSpec{.attn_kind = AttnKind::kFull,
                                                                    .sliding_window = 0,
                                                                    .replayable = true,
                                                                    .cache_blocks_per_lcm_block = 1,
                                                                    .block_granularity = 4}};
    EXPECT_THROW(
        MakeCoordinator(full_replay, 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr, false),
        std::runtime_error);
    const std::vector<CacheGroupSpec> state_replay = {CacheGroupSpec{.attn_kind = AttnKind::kState,
                                                                     .sliding_window = 0,
                                                                     .replayable = true,
                                                                     .cache_blocks_per_lcm_block = 1,
                                                                     .block_granularity = 4}};
    EXPECT_THROW(
        MakeCoordinator(state_replay, 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr, false),
        std::runtime_error);
}

TEST(BoundedReplayCoordinator, ReplayableGroupNeverConstrainsTheCommonBoundary) {
    BlockPool pool(16, {1, 1});
    CacheCoordinator coord =
        MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr, false);
    EXPECT_EQ(coord.ReplayWindowTokens(), 8);
    EXPECT_FALSE(coord.GroupIsReplayable(0));
    EXPECT_TRUE(coord.GroupIsReplayable(1));

    // full caches 3 pages, swa caches nothing: without replay the common
    // boundary would collapse to 0 (SwaMissForcesZeroCommon); with it the
    // closed group alone decides.
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coord, pool, hash, 0);
    }
    const CacheCoordinator::PrefixProbe probe = coord.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 12);
    EXPECT_EQ(probe.device.kv_tokens, 12);
    EXPECT_EQ(probe.device.per_group[0].hits, (std::vector<std::uint8_t>{1, 1, 1}));
    EXPECT_TRUE(probe.device.per_group[1].hits.empty());

    const CoordinatorMatch match = MatchPrefixForTest(coord, hashes).device;
    EXPECT_EQ(match.num_common_tokens, 12);
    EXPECT_EQ(match.per_group[0].NumHitBlocks(), 3);
    EXPECT_TRUE(match.per_group[1].blocks.empty());
}

TEST(BoundedReplayCoordinator, AdmitMaterializesReplayableSuffixFromWindowBegin) {
    BlockPool pool(32, {1, 1});
    CacheCoordinator coord =
        MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr, false);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coord, pool, hash, 0);
    }

    // P = 16 (4 pages) hits on full; the request extends by 6 new tokens. The
    // caller states the same dense demand for every group; Admit itself
    // materializes the replayable group from s = 16 - 8 = 8: pages
    // [2, ceil(22/4) = 6).
    std::vector<BlockTable> tables(2);
    std::vector<GroupDemand> demands = {
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{6}},
        GroupDemand{.table = &tables[1], .extent = DenseGrowth{6}},
    };
    const std::optional<CacheCoordinator::AdmissionResult> admitted =
        coord.Admit(coord.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt);
    ASSERT_TRUE(admitted.has_value());
    EXPECT_EQ(admitted->device_prefix_tokens, 16);
    EXPECT_EQ(admitted->junction_tokens, 0);

    // Closed group: 4 claimed hit pages + ceil(6/4) = 2 fresh.
    ASSERT_EQ(tables[0].NumBlocks(), 6);
    for (std::int32_t slot = 0; slot < 6; ++slot) {
        EXPECT_TRUE(tables[0].Blocks()[static_cast<std::size_t>(slot)]) << "full slot " << slot;
    }
    EXPECT_EQ(admitted->new_page_ids[0].size(), 2u);

    // Replayable group: holes below s/g, fresh private pages from there.
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t slot = 2; slot < 6; ++slot) {
        EXPECT_TRUE(tables[1].Blocks()[static_cast<std::size_t>(slot)]) << "swa slot " << slot;
    }
    EXPECT_EQ(tables[1].ReclaimedPrefixBlocks(), 2);
    EXPECT_EQ(tables[1].AvailableTokens(), 24 - 22);
    EXPECT_EQ(admitted->new_page_ids[1].size(), 4u);
    EXPECT_EQ(coord.GroupPrefixIndex(1).NumEntries(pool), 0);
}

TEST(BoundedReplayCoordinator, ReplayableGroupNeverPublishesOrStreams) {
    BlockPool pool(32, {1, 1});
    BlockPool host_pool(16, {1, 1});
    CacheCoordinator coord = MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, &host_pool,
                                             /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/true);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/16));

    // Chunk and endpoint publication both skip the replayable group.
    for (const CacheBoundaryKind kind : {CacheBoundaryKind::kChunk, CacheBoundaryKind::kEndpoint}) {
        CacheCompletedBlocksForTest(coord, tables, hashes, NextTestAccessEpoch(), /*first_new_prefix_page=*/0,
                                    /*num_computed_tokens=*/16, kind, /*stream_completed_to_host=*/true,
                                    /*publish_state=*/false);
    }
    EXPECT_EQ(coord.GroupPrefixIndex(0).NumEntries(pool), 4);
    EXPECT_EQ(coord.GroupPrefixIndex(1).NumEntries(pool), 0);
    EXPECT_EQ(coord.GroupPrefixIndex(1).NumEntries(host_pool), 0);
    for (const CacheCoordinator::StoreCandidate& store : coord.TakePendingStores()) {
        EXPECT_EQ(store.key.group_id, 0u) << "a replayable group must never be queued for D2H";
    }

    // The exact-range registration used by host extensions skips it too.
    CacheFullBlocksForTest(coord, tables, hashes);
    coord.QueueCachedBlocksForStore(hashes);
    EXPECT_EQ(coord.GroupPrefixIndex(1).NumEntries(pool), 0);
    for (const CacheCoordinator::StoreCandidate& store : coord.TakePendingStores()) {
        EXPECT_EQ(store.key.group_id, 0u);
    }

    // Residency of the boundary counts the closed group only: kComplete, not
    // permanently kPartial because of the replayable group.
    EXPECT_EQ(coord.DeviceBoundaryResidency(Key(hashes[3], 0), /*tail_tokens=*/0),
              CacheCoordinator::BoundaryResidency::kComplete);
}

TEST(BoundedReplayCoordinator, RetentionStillReclaimsReplayablePages) {
    BlockPool pool(32, {1, 1});
    CacheCoordinator coord =
        MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr, false);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/24));
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    for (const CacheBlockRef& block : tables[1].Blocks()) {
        EXPECT_TRUE(block);
    }

    // Window 8 at 24 computed tokens: skipped = 17 -> 4 fully slid-out pages.
    EXPECT_EQ(coord.GroupBlocksReclaimableAt(1, tables[1], /*num_computed_tokens=*/24, /*count_uncached=*/true), 4);
    coord.ReclaimExpired(tables, /*num_computed_tokens=*/24);
    for (std::int32_t slot = 0; slot < 4; ++slot) {
        EXPECT_FALSE(tables[1].Blocks()[static_cast<std::size_t>(slot)]) << "swa slot " << slot;
    }
    for (std::int32_t slot = 4; slot < 6; ++slot) {
        EXPECT_TRUE(tables[1].Blocks()[static_cast<std::size_t>(slot)]) << "swa slot " << slot;
    }
    // The closed group never expires.
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
}

// ---------------------------------------------------------------------------
// Retraction images: Snapshot records a request's data slots -- published
// ones as pinned Host L2 entries, the rest in the snapshot pool, each in its
// Device block's bucket -- and Restore rebuilds the identical shape.
// ---------------------------------------------------------------------------

// The shape a table presents to its consumers: block count, which slots are
// holes, tail capacity and reclaimed prefix. Two tables with equal shapes are
// interchangeable to the kernels once the restore copies the bytes.
struct TableShape {
    std::int32_t num_blocks{0};
    std::vector<bool> filled;
    std::int32_t available_tokens{0};
    std::int32_t reclaimed_prefix_blocks{0};

    bool operator==(const TableShape&) const = default;
};

TableShape ShapeOf(const BlockTable& table) {
    TableShape shape{
        .num_blocks = table.NumBlocks(),
        .available_tokens = table.AvailableTokens(),
        .reclaimed_prefix_blocks = table.ReclaimedPrefixBlocks(),
    };
    for (const CacheBlockRef& block : table.Blocks()) {
        shape.filled.push_back(static_cast<bool>(block));
    }
    return shape;
}

// Admits a request with one GroupDemand per group, like the scheduler does.
std::optional<CacheCoordinator::AdmissionResult> AdmitDemands(CacheCoordinator& coordinator,
                                                              std::vector<BlockTable>& tables,
                                                              std::vector<GroupDemand> demands) {
    for (std::size_t i = 0; i < tables.size(); ++i) {
        demands[i].table = &tables[i];
    }
    return coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt);
}

std::optional<CacheCoordinator::AdmissionResult> RestoreDemands(CacheCoordinator& coordinator,
                                                                const RetractionImage& snapshot,
                                                                std::vector<BlockTable>& tables,
                                                                std::vector<GroupDemand> demands,
                                                                std::uint64_t access_epoch) {
    for (std::size_t i = 0; i < tables.size(); ++i) {
        demands[i].table = &tables[i];
    }
    return coordinator.Restore(snapshot, demands, access_epoch);
}

std::vector<std::vector<ImageSlot>> NoHostSlots(std::int32_t num_groups) {
    return std::vector<std::vector<ImageSlot>>(static_cast<std::size_t>(num_groups));
}

TEST(RetractionImageTest, PublishedSlotsRideHostCacheAndRestoreAsKeyedLoads) {
    BlockPool pool(8, {1});
    BlockPool host_pool(8, {1});
    BlockPool snapshot_pool(8, {1});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                                   &snapshot_pool, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));  // 3 blocks: two full pages + a tail
    const std::vector<std::string> hashes{"h0", "h1"};
    CacheFullBlocksForTest(coordinator, tables, hashes);

    // The two published pages name their keys; the unaligned tail does not.
    auto published = coordinator.PublishedDataSlots(tables, /*num_tokens=*/5);
    ASSERT_EQ(published.size(), 1u);
    ASSERT_EQ(published[0].size(), 2u);
    EXPECT_EQ(published[0][0].slot_index, 0);
    EXPECT_EQ(published[0][0].key, Key("h0", 0));
    EXPECT_EQ(published[0][1].slot_index, 1);
    EXPECT_EQ(published[0][1].key, Key("h1", 0));
    EXPECT_EQ(published[0][0].block, tables[0].Blocks()[0]);

    // The transfer manager swaps the Host L2 block in for each published slot
    // (here: one already cached on Host, one freshly allocated for the store).
    // (as StartRetractionStores does: the published slot with its Host block
    // swapped in, keeping how the Device entry was published)
    EXPECT_EQ(published[0][0].logical_block_index, 0);
    EXPECT_EQ(published[0][1].logical_block_index, 1);
    std::vector<std::vector<ImageSlot>> host_slots(1);
    CacheBlockRef warm = coordinator.AcquireHostBlock(0, /*bucket=*/0);
    coordinator.CacheHostBlock(warm, Key("h0", 0));
    host_slots[0].push_back(published[0][0]);
    host_slots[0][0].block = warm;
    CacheBlockRef fresh = coordinator.AcquireHostBlock(0, /*bucket=*/0);
    host_slots[0].push_back(published[0][1]);
    host_slots[0][1].block = fresh;

    auto image = coordinator.TakeImage(tables, /*num_tokens=*/5, host_slots);
    ASSERT_TRUE(image);
    ASSERT_EQ(image->image.tables[0].slots.size(), 3u);
    EXPECT_TRUE(image->image.tables[0].slots[0].InHostCache());
    EXPECT_TRUE(image->image.tables[0].slots[1].InHostCache());
    EXPECT_FALSE(image->image.tables[0].slots[2].InHostCache());
    EXPECT_EQ(image->image.tables[0].slots[2].slot_index, 2);
    ASSERT_EQ(image->store_pairs.size(), 1u) << "only the tail page is copied to the snapshot pool";
    EXPECT_EQ(image->store_pairs[0].source, tables[0].Blocks()[2]);
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 7);
    EXPECT_EQ(image->image.tables[0].available_tokens, 1);

    // Publish the store's destination as its ACK would, then free the victim
    // (the store pairs and the published-slot listing hold Device refs only
    // until the transfer manager has resolved them).
    coordinator.CacheHostBlock(fresh, Key("h1", 0));
    warm.reset();
    fresh.reset();
    image->store_pairs.clear();
    published.clear();
    coordinator.Free(tables);
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 2) << "the image pins its Host L2 entries";
    // The victim's published Device blocks are gone from the Device index too,
    // so the restore must copy the published slots back from Host L2.
    ASSERT_TRUE(coordinator.ClearDeviceCache());

    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore);
    EXPECT_EQ(restored[0].NumBlocks(), 3);
    EXPECT_EQ(restored[0].AvailableTokens(), 1);
    EXPECT_TRUE(restore->new_page_ids[0].empty()) << "the copies fill every rebuilt page; the reserve fits the tail";
    ASSERT_EQ(restore->load_pairs.size(), 2u);
    EXPECT_EQ(restore->load_pairs[0].key, Key("h0", 0));
    EXPECT_EQ(restore->load_pairs[1].key, Key("h1", 0));
    EXPECT_TRUE(restore->load_pairs[0].source.IsOwnedBy(host_pool));
    EXPECT_EQ(restore->load_pairs[0].destination, restored[0].Blocks()[0]);
    // The rows carry how the entries were published before the retraction,
    // and the ACK's republication keeps that metadata.
    EXPECT_EQ(restore->load_pairs[0].logical_block_index, 0);
    EXPECT_EQ(restore->load_pairs[1].logical_block_index, 1);
    EXPECT_EQ(restore->load_pairs[1].boundary_kind, CacheBoundaryKind::kChunk);
    coordinator.CacheDeviceBlock(restore->load_pairs[1].destination, restore->load_pairs[1].key,
                                 restore->load_pairs[1].logical_block_index, restore->load_pairs[1].boundary_kind);
    const auto republished = coordinator.GroupPrefixIndex(0).MetadataFor(pool, restored[0].Blocks()[1]->Location());
    ASSERT_TRUE(republished);
    EXPECT_EQ(republished->logical_block_index, 1);
    EXPECT_EQ(republished->boundary_kind, CacheBoundaryKind::kChunk);
    ASSERT_EQ(restore->snapshot_pairs.size(), 1u);
    EXPECT_TRUE(restore->snapshot_pairs[0].key.content_hash.empty());
    EXPECT_TRUE(restore->snapshot_pairs[0].source.IsOwnedBy(snapshot_pool));
    EXPECT_EQ(restore->snapshot_pairs[0].destination, restored[0].Blocks()[2]);

    coordinator.Free(restored);
    restore.reset();  // the copy pairs pin both ends until the runtime's ACK
    host_slots.clear();
    image.reset();
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0) << "dropping the image unpins the Host entries";
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 2) << "they stay cached for ordinary reuse";
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 8);
}

TEST(RetractionImageTest, SnapshotRefusesWithoutMutationWhenThePoolCannotHoldTheImage) {
    BlockPool pool(8, {1, 1});
    BlockPool snapshot_pool(3, {1, 1});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                                                   /*host_pool=*/nullptr, &snapshot_pool,
                                                   /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));  // 2 + 2 blocks > 3 snapshot parents
    CacheBlockRef pinned = snapshot_pool.AcquireBlock(0);
    ASSERT_TRUE(pinned);

    EXPECT_FALSE(coordinator.TakeImage(tables, /*num_tokens=*/4, NoHostSlots(2)));
    EXPECT_EQ(snapshot_pool.NumEmptyLcmBlocks(), 2) << "the first group's blocks were not taken for a failing image";
    EXPECT_EQ(snapshot_pool.NumOccupiedSlots(), 1);

    pinned.reset();
    EXPECT_FALSE(coordinator.TakeImage(tables, /*num_tokens=*/4, NoHostSlots(2))) << "4 blocks never fit 3 parents";
    EXPECT_TRUE(coordinator.TakeImage(tables, /*num_tokens=*/2, NoHostSlots(2))) << "1 + 1 blocks do";
    coordinator.Free(tables);
}

TEST(RetractionImageTest, SnapshotIsRefusedWithoutASnapshotPool) {
    BlockPool pool(8, {1});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                                                   /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                                   /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    EXPECT_FALSE(coordinator.HasSnapshotPool());
    EXPECT_FALSE(coordinator.TakeImage(tables, /*num_tokens=*/4, NoHostSlots(1)));
    coordinator.Free(tables);
}

TEST(RetractionImageTest, RestoreEvictsCacheOnlyBlocksAndFailsCleanlyWhenPinned) {
    BlockPool pool(4, {1});
    BlockPool snapshot_pool(8, {1});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                                                   /*host_pool=*/nullptr, &snapshot_pool,
                                                   /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));  // 3 of 4 Device parents
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/6, NoHostSlots(1));
    ASSERT_TRUE(image);
    image->store_pairs.clear();  // resolved into the wire op; the Device sources are released
    coordinator.Free(tables);

    // Fill the Device pool with cache-only entries of another prompt: the
    // restore needs 3 + 1 reserve = 4 blocks, so it must evict them.
    for (std::int32_t i = 0; i < 4; ++i) {
        CacheForGroup(coordinator, pool, "other-" + std::to_string(i), /*group_id=*/0);
    }
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    CacheBlockRef pin = coordinator.GroupPrefixIndex(0).Find(pool, Key("other-0", 0));
    ASSERT_TRUE(pin);

    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    EXPECT_FALSE(RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1))
        << "one pinned entry leaves only three evictable parents for four blocks";
    EXPECT_EQ(restored[0].NumBlocks(), 0) << "a failed restore leaves nothing allocated";
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0) << "and evicts nothing";
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 4);

    pin.reset();
    const auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore);
    EXPECT_EQ(restored[0].NumBlocks(), 4);
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0) << "the restore evicted the cache-only blocks";
    ASSERT_EQ(restore->new_page_ids.size(), 1u);
    EXPECT_EQ(restore->new_page_ids[0].size(), 1u) << "only the re-reserved page is zeroed; the copies fill the rest";
    coordinator.Free(restored);
}

TEST(RetractionImageTest, RestoreClaimsStillCachedDeviceBlocksInsteadOfCopying) {
    BlockPool pool(8, {1});
    BlockPool host_pool(8, {1});
    BlockPool snapshot_pool(8, {1});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                                   &snapshot_pool, /*stream_device_cache_to_host=*/false);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));
    const std::vector<std::string> hashes{"h0", "h1"};
    CacheFullBlocksForTest(coordinator, tables, hashes);
    const std::vector<CacheBlockRef> original(tables[0].Blocks().begin(), tables[0].Blocks().end());

    std::vector<std::vector<ImageSlot>> host_slots = coordinator.PublishedDataSlots(tables, 5);
    for (ImageSlot& slot : host_slots[0]) {
        CacheBlockRef host = coordinator.AcquireHostBlock(0, /*bucket=*/0);
        coordinator.CacheHostBlock(host, slot.key);
        slot.block = std::move(host);
    }
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/5, host_slots);
    ASSERT_TRUE(image);
    image->store_pairs.clear();
    host_slots.clear();
    coordinator.Free(tables);
    // The victim's published blocks survive as cache-only Device entries (the
    // test's `original` copies keep the old tail block alive as well).
    ASSERT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 5);

    // Fill the pool so that copying both published pages would not fit: the
    // restore needs only the tail page because it claims the two cached blocks
    // as a prefix hit would, and the reserve fits the imaged tail capacity.
    std::vector<CacheBlockRef> others = pool.AcquireBlocks(0, 4);
    ASSERT_EQ(others.size(), 4u);
    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore);
    EXPECT_EQ(restored[0].Blocks()[0], original[0]) << "slot 0 is the still-cached canonical block";
    EXPECT_EQ(restored[0].Blocks()[1], original[1]);
    EXPECT_NE(restored[0].Blocks()[2], original[2]) << "the tail page is a fresh block filled by the copy";
    EXPECT_TRUE(restore->load_pairs.empty()) << "nothing to copy from Host L2";
    ASSERT_EQ(restore->snapshot_pairs.size(), 1u);
    EXPECT_TRUE(restore->new_page_ids[0].empty()) << "claimed and copied pages alike need no zeroing";
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumPinnedEntries(pool), 2) << "the claims pin the entries";
    coordinator.Free(restored);
}

TEST(RetractionImageTest, ShardedImageKeepsEveryCopyInItsBucket) {
    // packing 4, two owners: block v is owned by (v - 1) % 2 == slot % 2.
    BlockPool pool(4, {4});
    BlockPool host_pool(4, {4});
    BlockPool snapshot_pool(2, {4});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                                   &snapshot_pool, /*stream_device_cache_to_host=*/false);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/10));  // 5 blocks, balanced over both buckets
    std::vector<std::int32_t> device_buckets;
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        device_buckets.push_back(allocator.BucketOf(block->Location()));
    }
    EXPECT_EQ(std::ranges::count(device_buckets, 0), 3);
    EXPECT_EQ(std::ranges::count(device_buckets, 1), 2);

    // Published slots 0 and 1 take Host blocks in their own buckets.
    const std::vector<std::string> hashes{"h0", "h1"};
    CacheFullBlocksForTest(coordinator, tables, hashes);
    std::vector<std::vector<ImageSlot>> host_slots(1);
    std::vector<std::vector<ImageSlot>> published = coordinator.PublishedDataSlots(tables, 10);
    ASSERT_EQ(published[0].size(), 2u);
    for (ImageSlot& slot : published[0]) {
        CacheBlockRef host = coordinator.AcquireHostBlock(0, allocator.BucketOf(slot.block->Location()));
        ASSERT_TRUE(host);
        EXPECT_EQ(allocator.BucketOf(host->Location()), device_buckets[static_cast<std::size_t>(slot.slot_index)]);
        coordinator.CacheHostBlock(host, slot.key);
        slot.block = std::move(host);
        host_slots[0].push_back(std::move(slot));
    }
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/10, host_slots);
    ASSERT_TRUE(image);
    for (const BlockTransfer& pair : image->store_pairs) {
        EXPECT_EQ(allocator.BucketOf(pair.source->Location()), allocator.BucketOf(pair.destination->Location()));
    }
    for (const ImageSlot& slot : image->image.tables[0].slots) {
        EXPECT_EQ(allocator.BucketOf(slot.block->Location()),
                  device_buckets[static_cast<std::size_t>(slot.slot_index)]);
    }
    image->store_pairs.clear();
    coordinator.Free(tables);

    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    const auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore);
    for (const BlockTransfer& pair : restore->load_pairs) {
        EXPECT_EQ(allocator.BucketOf(pair.source->Location()), allocator.BucketOf(pair.destination->Location()));
    }
    for (const BlockTransfer& pair : restore->snapshot_pairs) {
        EXPECT_EQ(allocator.BucketOf(pair.source->Location()), allocator.BucketOf(pair.destination->Location()));
    }
    coordinator.Free(restored);
}

TEST(RetractionImageTest, BucketConstrainedRestoreFailsCleanlyWhileOtherBucketsHaveRoom) {
    // packing 2, two owners, two parents: the image holds two bucket-1 blocks,
    // so it needs both parents' slot 1, and a resident request pins one.
    BlockPool pool(2, {2});
    BlockPool snapshot_pool(2, {2});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                                                   /*host_pool=*/nullptr, &snapshot_pool,
                                                   /*stream_device_cache_to_host=*/false);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));  // 2 blocks, one per bucket
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/4, NoHostSlots(1));
    ASSERT_TRUE(image);
    image->store_pairs.clear();
    coordinator.Free(tables);

    // A resident holds slot 1 of parent 1 (bucket 1), leaving bucket 1 one
    // hole in parent 2 only: the image's bucket-0 block fits anywhere, but
    // nothing lets a bucket-1 block take a bucket-0 hole.
    std::vector<CacheBlockRef> resident = pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1});
    ASSERT_EQ(resident.size(), 1u);
    std::vector<CacheBlockRef> resident_too = pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1});
    ASSERT_EQ(resident_too.size(), 1u);
    ASSERT_EQ(pool.NumFreeSlots(0), 2) << "two bucket-0 holes remain";

    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 0}};
    EXPECT_FALSE(RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1))
        << "the bucket-1 slot has no home although two holes are free";
    EXPECT_EQ(restored[0].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeSlots(0), 2) << "nothing was allocated";

    resident_too.clear();
    const auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore);
    for (const ImageSlot& slot : image->image.tables[0].slots) {
        EXPECT_EQ(allocator.BucketOf(restored[0].Blocks()[static_cast<std::size_t>(slot.slot_index)]->Location()),
                  allocator.BucketOf(slot.block->Location()));
    }
    coordinator.Free(restored);
}

TEST(RetractionImageTest, HostAllocationFollowsTheSourceBucket) {
    BlockPool pool(2, {2});
    BlockPool host_pool(2, {2});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    // Fill Host with evictable entries of bucket 0 and bucket 1 alike.
    for (std::int32_t i = 0; i < 4; ++i) {
        CacheBlockRef block = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{i % 2}).front();
        coordinator.CacheHostBlock(block, Key("old-" + std::to_string(i), 0));
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    const std::array<std::uint32_t, 2> groups{0, 0};
    const std::array<std::int32_t, 2> buckets{1, 1};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);
    ASSERT_TRUE(batch.blocks[0]);
    ASSERT_TRUE(batch.blocks[1]);
    EXPECT_EQ(allocator.BucketOf(batch.blocks[0]->Location()), 1);
    EXPECT_EQ(allocator.BucketOf(batch.blocks[1]->Location()), 1);
    EXPECT_EQ(batch.stats.same_group_scans, 1u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 2) << "only bucket-1 entries were evicted";
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-2", 0)));
}

// ---------------------------------------------------------------------------
// Tail copies under page-cyclic sharding: a hitter's copy of a claimed tail
// block and an owner's copy-in each take a fresh block in the source block's
// bucket, so a hole in another bucket cannot hold them. The pool has three
// parents. Group 0 shards over `shard_count` buckets (slot s of a parent is
// bucket s % shard_count); group 1 is a State group whose every block takes a
// parent of its own.
// ---------------------------------------------------------------------------
class ShardedTailCopyTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;

    ShardedTailCopyTest(std::int32_t packing, std::int32_t block_granularity, std::int32_t shard_count)
        : pool_(/*num_lcm_blocks=*/3, {packing, 1}),
          coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  CacheGroupSpec{.attn_kind = AttnKind::kFull,
                                 .cache_blocks_per_lcm_block = packing,
                                 .block_granularity = block_granularity,
                                 .shard_count = shard_count},
                  CacheGroupSpec{
                      .attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
              },
              kP, pool_, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/false)} {}

    // A group-0 block held outside any request, in `bucket`.
    std::vector<CacheBlockRef> Resident(std::int32_t bucket) {
        std::vector<CacheBlockRef> block = pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{bucket});
        EXPECT_EQ(block.size(), 1u);
        return block;
    }

    // Group-0 blocks held outside any request until `bucket` has no free
    // slot, counting the slots of empty parents.
    std::vector<std::vector<CacheBlockRef>> FillBucket(std::int32_t bucket) {
        std::vector<std::vector<CacheBlockRef>> residents;
        while (pool_.FreeSlotsByBucket(0)[static_cast<std::size_t>(bucket)] > 0 || pool_.NumEmptyLcmBlocks() > 0) {
            residents.push_back(Resident(bucket));
            if (residents.back().empty()) {
                break;  // Resident's EXPECT has failed; do not spin
            }
        }
        return residents;
    }

    // The prompt's first three tokens published as a tail (no full page).
    RequestProgress TailAtThree(bool owner_continues, bool publish_state) const {
        return RequestProgress{
            .completed_pages =
                CompletedPages{
                    .prefix_hashes = {},
                    .first_new_prefix_page = 0,
                    .tail = TailPublication{.tail_tokens = 3, .hash = tail_hash_},
                    .owner_continues = owner_continues,
                    .publish_state = publish_state,
                    .boundary_kind = CacheBoundaryKind::kEndpoint,
                },
            .num_computed_tokens = 3,
        };
    }

    std::int32_t BucketOf(const CacheBlockRef& block) const {
        return coordinator_.Allocator(0).BucketOf(block->Location());
    }

    std::optional<CacheBlockLocation> CachedTailBlock() const {
        return CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(tail_hash_, 0));
    }

    BlockPool pool_;
    CacheCoordinator coordinator_;
    const std::vector<std::int32_t> prompt_{1, 2, 3, 99};
    const std::string tail_hash_ = HashPrefixPage(token_span(prompt_.data(), 3), "");
};

class ShardedTailCopyPackingFourTest : public ShardedTailCopyTest {
protected:
    // Slots 0 and 2 are bucket 0, slots 1 and 3 bucket 1.
    ShardedTailCopyPackingFourTest()
        : ShardedTailCopyTest(/*packing=*/4, /*block_granularity=*/kP, /*shard_count=*/2) {}

    // An owner computes three tokens and publishes them as a tail, its block
    // in bucket 0 and its state checkpoint beside it. Sets `source` to the
    // tail block.
    void CacheTailInBucketZero(CacheBlockLocation& source) {
        std::vector<BlockTable> owner(2);
        ASSERT_TRUE(AdmitForTest(coordinator_, owner, 3));
        source = owner[0].Blocks()[0]->Location();
        EXPECT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
        coordinator_.CacheCompletedBlocks(owner, TailAtThree(/*owner_continues=*/false, /*publish_state=*/true),
                                          CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        coordinator_.Free(owner);
    }

    // A hitter's probe of the prompt: it matches the cached tail.
    CacheCoordinator::PrefixProbe ProbeTail() {
        return coordinator_.ProbePromptPrefix(
            CacheCoordinator::PromptPrefix{.prefix_hashes = {}, .tokens = prompt_, .max_tokens = 3});
    }

    // An unrelated full page cached from the block a dense growth takes next;
    // sets `location` to its block.
    void CacheUnrelatedPage(CacheBlockLocation& location) {
        std::vector<BlockTable> filler(2);
        ASSERT_TRUE(AdmitForTest(coordinator_, filler, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                 StateStep{StateStep::Kind::kKeep}));
        location = filler[0].Blocks()[0]->Location();
        CacheFullBlocksForTest(coordinator_, filler, unrelated_page_);
        coordinator_.Free(filler);
    }

    std::optional<CacheBlockLocation> CachedUnrelatedPage() const {
        return CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(unrelated_page_[0], 0));
    }

    const std::vector<std::string> unrelated_page_ = ContentHashes({{5, 6, 7, 8}});
};

TEST_F(ShardedTailCopyPackingFourTest, HitterCopyWaitsForAHoleInTheSourceBucket) {
    CacheBlockLocation source;
    ASSERT_NO_FATAL_FAILURE(CacheTailInBucketZero(source));
    // Group 0's two holes are both in bucket 1; the one empty parent left is
    // what the hitter's State live slot needs.
    std::vector<CacheBlockRef> resident = Resident(0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 2);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 1);

    ASSERT_EQ(ProbeTail().device.num_common_tokens, 3);
    std::vector<BlockTable> hitter(2);
    EXPECT_FALSE(AdmitForTest(coordinator_, hitter, ProbeTail(), GroupDemand{.extent = DenseGrowth{1}}))
        << "the copy of the bucket-0 tail block would take the State group's empty parent";
    EXPECT_EQ(hitter[0].NumBlocks(), 0);
    EXPECT_EQ(pool_.NumEmptyLcmBlocks(), 1) << "a refused admission takes nothing";

    // A bucket-0 hole lets the copy land beside its source.
    resident.clear();
    const auto admitted = AdmitForTest(coordinator_, hitter, ProbeTail(), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(admitted);
    ASSERT_EQ(admitted->pages_to_copy[0].size(), 1u);
    EXPECT_EQ(admitted->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveCacheBlockId(source));
    EXPECT_EQ(BucketOf(hitter[0].Blocks()[0]), 0);
    EXPECT_EQ(pool_.NumEmptyLcmBlocks(), 0);
    coordinator_.Free(hitter);
}

TEST_F(ShardedTailCopyPackingFourTest, HitterCopyIsFundedByAnEvictionInTheSourceBucket) {
    CacheBlockLocation source;
    ASSERT_NO_FATAL_FAILURE(CacheTailInBucketZero(source));
    // Bucket 0's only other slot holds an unrelated cached page; bucket 1
    // has a hole, and the one empty parent is the State live slot's.
    std::vector<CacheBlockRef> resident = Resident(1);
    CacheBlockLocation page;
    ASSERT_NO_FATAL_FAILURE(CacheUnrelatedPage(page));
    ASSERT_EQ(coordinator_.Allocator(0).BucketOf(page), 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 1);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> hitter(2);
    const auto admitted = AdmitForTest(coordinator_, hitter, ProbeTail(), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(admitted) << "evicting the bucket-0 page makes room for the copy";
    ASSERT_EQ(admitted->pages_to_copy[0].size(), 1u);
    EXPECT_EQ(admitted->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveCacheBlockId(source));
    EXPECT_EQ(hitter[0].Blocks()[0]->Location(), page) << "the copy lands in the evicted page's slot";
    EXPECT_FALSE(CachedUnrelatedPage().has_value());
    EXPECT_EQ(pool_.NumEmptyLcmBlocks(), 0);
    coordinator_.Free(hitter);
}

TEST_F(ShardedTailCopyPackingFourTest, HitterCopyIsNotFundedByAnEvictionInAnotherBucket) {
    CacheBlockLocation source;
    ASSERT_NO_FATAL_FAILURE(CacheTailInBucketZero(source));
    // Bucket 0 is full; the only cached entry to evict besides the hit is a
    // page in bucket 1, and the one empty parent is the State live slot's.
    std::vector<CacheBlockRef> resident = Resident(0);
    CacheBlockLocation page;
    ASSERT_NO_FATAL_FAILURE(CacheUnrelatedPage(page));
    ASSERT_EQ(coordinator_.Allocator(0).BucketOf(page), 1);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 1);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> hitter(2);
    EXPECT_FALSE(AdmitForTest(coordinator_, hitter, ProbeTail(), GroupDemand{.extent = DenseGrowth{1}}))
        << "a bucket-1 hole cannot hold the copy of the bucket-0 tail block";
    EXPECT_EQ(CachedUnrelatedPage(), page) << "a refused admission evicts nothing";
    EXPECT_EQ(pool_.NumEmptyLcmBlocks(), 1);
}

TEST_F(ShardedTailCopyPackingFourTest, OwnerCopyInWithoutAHoleInItsBucketPublishesNoTail) {
    // A lone request at its first decode: its prompt-end tail block is in
    // bucket 0, which has no hole, and no parent is empty; the rotation's
    // released state input funds only the State group's live slot.
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, 3));
    const CacheBlockLocation own = owner[0].Blocks()[0]->Location();
    ASSERT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
    std::vector<CacheBlockRef> resident = Resident(0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 2);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    const auto decode = AdmitForTest(coordinator_, owner, GroupDemand{.extent = DenseGrowth{1}},
                                     TailAtThree(/*owner_continues=*/true, /*publish_state=*/true));
    ASSERT_TRUE(decode) << "the owner keeps writing its private block instead of waiting";
    EXPECT_TRUE(decode->pages_to_copy[0].empty());
    EXPECT_EQ(owner[0].Blocks()[0]->Location(), own) << "no copy-in";
    EXPECT_EQ(coordinator_.NumTailEntries(), 0u);
    EXPECT_FALSE(CachedTailBlock().has_value()) << "nothing is published at the tail";
    coordinator_.Free(owner);
}

TEST_F(ShardedTailCopyPackingFourTest, OwnerCopyInIsNotFundedByEvictingTheTailItDedupesOnto) {
    // The owner holds its bucket-0 tail block; its State group is not shaped
    // yet, so its first rotation below needs two blocks.
    std::vector<CacheBlockRef> resident = Resident(0);
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{3}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    const CacheBlockLocation own = owner[0].Blocks()[0]->Location();
    ASSERT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
    ASSERT_EQ(owner[1].NumBlocks(), 0);
    // Another request publishes the same tail first, from a bucket-1 block,
    // with its state checkpoint (a parent of its own) beside it.
    std::vector<BlockTable> other(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, 3));
    const CacheBlockLocation canonical = other[0].Blocks()[0]->Location();
    ASSERT_EQ(BucketOf(other[0].Blocks()[0]), 1);
    coordinator_.CacheCompletedBlocks(other, TailAtThree(/*owner_continues=*/false, /*publish_state=*/true),
                                      CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(other);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 0);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 1);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 1);

    // The copy-in of the canonical block fits the bucket-1 hole, and the
    // State group's two blocks need one parent more. The only cache entry to
    // evict for it is the tail's state checkpoint, but evicting it evicts the
    // whole tail, the canonical block too: the owner would then copy its own
    // bucket-0 block, into the parent the State group needs.
    const auto decode = AdmitForTest(
        coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{1}},
        TailAtThree(/*owner_continues=*/true, /*publish_state=*/false), StateStep{StateStep::Kind::kRotate});
    ASSERT_TRUE(decode) << "the owner keeps writing its private block instead of waiting";
    EXPECT_TRUE(decode->pages_to_copy[0].empty()) << "no copy-in";
    EXPECT_EQ(owner[0].Blocks()[0]->Location(), own);
    EXPECT_NE(CachedTailBlock(), std::optional<CacheBlockLocation>{canonical})
        << "the other request's tail made room for the State group";
    EXPECT_EQ(owner[1].NumBlocks(), 2);
    EXPECT_EQ(pool_.NumEmptyLcmBlocks(), 0);
    coordinator_.Free(owner);
}

class ShardedTailCopyPackingSixTest : public ShardedTailCopyTest {
protected:
    // Slots 0, 2 and 4 are bucket 0; slots 1, 3 and 5 bucket 1.
    ShardedTailCopyPackingSixTest() : ShardedTailCopyTest(/*packing=*/6, /*block_granularity=*/kP, /*shard_count=*/2) {}

    // Another request publishes the owner's tail first from a bucket-1 block
    // while the owner's own tail block is in bucket 0, so the owner's
    // publication dedupes onto that block and the copy-in copies it. Sets
    // `canonical` to that block; `residents_` keeps bucket 1 full and, with
    // own_bucket_full, bucket 0 too.
    void PublishCanonicalInBucketOne(std::vector<BlockTable>& owner, bool own_bucket_full,
                                     CacheBlockLocation& canonical) {
        std::vector<CacheBlockRef> kept = Resident(0);
        std::vector<CacheBlockRef> owner_room = Resident(0);  // freed below for the owner's block
        std::vector<CacheBlockRef> spare = Resident(0);
        std::vector<BlockTable> other(2);
        ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{3}}, RequestProgress{},
                                 StateStep{StateStep::Kind::kKeep}));
        canonical = other[0].Blocks()[0]->Location();
        EXPECT_EQ(BucketOf(other[0].Blocks()[0]), 1);
        owner_room.clear();
        ASSERT_TRUE(AdmitForTest(coordinator_, owner, 3));
        EXPECT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
        coordinator_.CacheCompletedBlocks(other, TailAtThree(/*owner_continues=*/false, /*publish_state=*/false),
                                          CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        coordinator_.Free(other);
        if (!own_bucket_full) {
            spare.clear();
        }
        residents_ = {std::move(kept), std::move(spare), Resident(1), Resident(1)};
    }

    std::vector<std::vector<CacheBlockRef>> residents_;
};

TEST_F(ShardedTailCopyPackingSixTest, OwnerCopyInIsCountedInTheBucketOfTheBlockItDedupesOnto) {
    std::vector<BlockTable> owner(2);
    CacheBlockLocation canonical;
    ASSERT_NO_FATAL_FAILURE(PublishCanonicalInBucketOne(owner, /*own_bucket_full=*/false, canonical));
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 1) << "the owner's own bucket has a hole";
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[1], 0) << "the canonical block's bucket has none";
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    const auto decode = AdmitForTest(coordinator_, owner, GroupDemand{.extent = DenseGrowth{1}},
                                     TailAtThree(/*owner_continues=*/true, /*publish_state=*/true));
    ASSERT_TRUE(decode);
    EXPECT_TRUE(decode->pages_to_copy[0].empty()) << "no bucket-1 slot for the copy: the owner publishes no tail";
    EXPECT_EQ(CachedTailBlock(), canonical) << "the other request's tail stays cached";
    coordinator_.Free(owner);
}

TEST_F(ShardedTailCopyPackingSixTest, OwnerCopyInKeepsTheBlockItDedupesOnto) {
    std::vector<BlockTable> owner(2);
    CacheBlockLocation canonical;
    ASSERT_NO_FATAL_FAILURE(PublishCanonicalInBucketOne(owner, /*own_bucket_full=*/true, canonical));
    ASSERT_EQ(pool_.NumFreeSlots(0), 0);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    // Evicting the canonical block would open a bucket-1 hole, but the
    // publication would then keep the owner's own bucket-0 block and copy
    // that one instead.
    const auto decode = AdmitForTest(coordinator_, owner, GroupDemand{.extent = DenseGrowth{1}},
                                     TailAtThree(/*owner_continues=*/true, /*publish_state=*/true));
    ASSERT_TRUE(decode);
    EXPECT_TRUE(decode->pages_to_copy[0].empty());
    EXPECT_EQ(CachedTailBlock(), canonical) << "the block the copy-in would read stays cached";
    coordinator_.Free(owner);
}

TEST_F(ShardedTailCopyPackingSixTest, OwnerCopyInMayEvictTheTailItDedupesOntoInItsOwnBucket) {
    // Another request publishes the owner's tail first from a bucket-0
    // block, the bucket of the owner's own tail block too.
    std::vector<BlockTable> other(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{3}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    ASSERT_EQ(BucketOf(other[0].Blocks()[0]), 0);
    std::vector<CacheBlockRef> resident = Resident(1);
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, 3));
    const CacheBlockLocation own = owner[0].Blocks()[0]->Location();
    ASSERT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
    coordinator_.CacheCompletedBlocks(other, TailAtThree(/*owner_continues=*/false, /*publish_state=*/false),
                                      CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(other);
    const auto bucket_zero = FillBucket(0);
    const auto bucket_one = FillBucket(1);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    // Whichever block the copy-in reads, its copy lands in bucket 0, so the
    // canonical block may be evicted for the room; the publication then
    // keeps the owner's own block and copies that one.
    const auto decode = AdmitForTest(coordinator_, owner, GroupDemand{.extent = DenseGrowth{1}},
                                     TailAtThree(/*owner_continues=*/true, /*publish_state=*/true));
    ASSERT_TRUE(decode);
    ASSERT_EQ(decode->pages_to_copy[0].size(), 1u);
    EXPECT_EQ(decode->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveCacheBlockId(own));
    EXPECT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
    EXPECT_EQ(CachedTailBlock(), own) << "the owner's block is published at the tail";
    coordinator_.Free(owner);
}

// A tail past a full page, each page two blocks: the tail is table slots 2
// and 3, published under the tail's hash at page offsets 0 and 1.
class ShardedTailCopyPastAPageTest : public ShardedTailCopyTest {
protected:
    // Slots 0 and 2 are bucket 0, slots 1 and 3 bucket 1.
    ShardedTailCopyPastAPageTest()
        : ShardedTailCopyTest(/*packing=*/4, /*block_granularity=*/kP / 2, /*shard_count=*/2) {}

    // Seven computed tokens: the full page, which this point does not
    // publish, and a three-token tail.
    RequestProgress TailAtSeven(bool owner_continues) const {
        return RequestProgress{
            .completed_pages =
                CompletedPages{
                    .prefix_hashes = std::span<const std::string>(hashes_).first(1),
                    .first_new_prefix_page = 1,
                    .tail = TailPublication{.tail_tokens = 3, .hash = hashes_[1]},
                    .owner_continues = owner_continues,
                    .boundary_kind = CacheBoundaryKind::kEndpoint,
                },
            .num_computed_tokens = 7,
        };
    }

    // The page's hash, then the tail's.
    const std::vector<std::string> hashes_ = ContentHashes({{1, 2, 3, 4}, {5, 6, 7}});
};

TEST_F(ShardedTailCopyPastAPageTest, OwnerCopyInReadsTheCanonicalBlockAtEachPageOffset) {
    // The owner's tail blocks, slots 2 and 3, are in buckets 0 and 1.
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{7}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    ASSERT_EQ(BucketOf(owner[0].Blocks()[2]), 0);
    ASSERT_EQ(BucketOf(owner[0].Blocks()[3]), 1);
    // Another request publishes the same tail first from two bucket-0
    // blocks: its table grows by the page, then, with bucket 1 held
    // elsewhere, by the tail.
    std::vector<BlockTable> other(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    const auto held = FillBucket(1);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{3}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    ASSERT_EQ(BucketOf(other[0].Blocks()[2]), 0);
    ASSERT_EQ(BucketOf(other[0].Blocks()[3]), 0);
    const CacheBlockLocation canonical_first = other[0].Blocks()[2]->Location();
    const CacheBlockLocation canonical_second = other[0].Blocks()[3]->Location();
    coordinator_.CacheCompletedBlocks(other, TailAtSeven(/*owner_continues=*/false),
                                      CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(other);
    // Bucket 0 has room for two copies, bucket 1 for none.
    const auto bucket_one = FillBucket(1);
    ASSERT_EQ(pool_.FreeSlotsByBucket(0)[0], 2);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    // The copy-in fits only as copies of the canonical blocks: a copy of the
    // owner's own slot-3 block would need room in bucket 1.
    const auto decode =
        AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{1}},
                     TailAtSeven(/*owner_continues=*/true), StateStep{StateStep::Kind::kKeep});
    ASSERT_TRUE(decode);
    const GroupAllocator& allocator = coordinator_.Allocator(0);
    const std::vector<std::pair<std::int32_t, std::int32_t>> copies{
        {allocator.ResolveCacheBlockId(canonical_first),
         allocator.ResolveCacheBlockId(owner[0].Blocks()[2]->Location())},
        {allocator.ResolveCacheBlockId(canonical_second),
         allocator.ResolveCacheBlockId(owner[0].Blocks()[3]->Location())},
    };
    EXPECT_EQ(decode->pages_to_copy[0], copies) << "each slot copies the canonical block at its page offset";
    EXPECT_EQ(BucketOf(owner[0].Blocks()[3]), 0) << "slot 3's copy lands in its source's bucket";
    EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(hashes_[1], 0, /*page_offset=*/1)),
              canonical_second);
    coordinator_.Free(owner);
}

TEST_F(ShardedTailCopyPastAPageTest, OwnerCopyInKeepsTheCanonicalBlockOfAnotherBucketThanItsTailSlot) {
    // Another request's tail blocks, slots 2 and 3, are in buckets 0 and 1.
    std::vector<BlockTable> other(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{7}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    const CacheBlockLocation canonical_first = other[0].Blocks()[2]->Location();
    const CacheBlockLocation canonical_second = other[0].Blocks()[3]->Location();
    ASSERT_EQ(BucketOf(other[0].Blocks()[2]), 0);
    ASSERT_EQ(BucketOf(other[0].Blocks()[3]), 1);
    // The owner's page, slots 0 and 1, is in buckets 0 and 1 too, but with
    // bucket 1 held elsewhere both its tail blocks are in bucket 0.
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    const auto held = FillBucket(1);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{3}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    ASSERT_EQ(BucketOf(owner[0].Blocks()[0]), 0);
    ASSERT_EQ(BucketOf(owner[0].Blocks()[1]), 1);
    ASSERT_EQ(BucketOf(owner[0].Blocks()[2]), 0);
    ASSERT_EQ(BucketOf(owner[0].Blocks()[3]), 0);
    coordinator_.CacheCompletedBlocks(other, TailAtSeven(/*owner_continues=*/false),
                                      CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(other);
    const auto bucket_zero = FillBucket(0);
    const auto bucket_one = FillBucket(1);
    ASSERT_EQ(pool_.NumFreeSlots(0), 0);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    // Evicting the canonical blocks would open a hole in each bucket, but the
    // publication would then keep the owner's own blocks and copy them, both
    // into bucket 0.
    const auto decode =
        AdmitForTest(coordinator_, owner, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{1}},
                     TailAtSeven(/*owner_continues=*/true), StateStep{StateStep::Kind::kKeep});
    ASSERT_TRUE(decode);
    EXPECT_TRUE(decode->pages_to_copy[0].empty()) << "the owner publishes no tail";
    EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(hashes_[1], 0)), canonical_first);
    EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(hashes_[1], 0, /*page_offset=*/1)),
              canonical_second);
    coordinator_.Free(owner);
}

// The same pool with group 0 unsharded: every block is in bucket 0.
class UnshardedTailCopyTest : public ShardedTailCopyTest {
protected:
    UnshardedTailCopyTest() : ShardedTailCopyTest(/*packing=*/4, /*block_granularity=*/kP, /*shard_count=*/1) {}
};

TEST_F(UnshardedTailCopyTest, OwnerCopyInMayEvictTheTailItDedupesOnto) {
    // Another request publishes the owner's tail first; nothing else is free.
    std::vector<BlockTable> other(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, other, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{3}},
                             RequestProgress{}, StateStep{StateStep::Kind::kKeep}));
    std::vector<BlockTable> owner(2);
    ASSERT_TRUE(AdmitForTest(coordinator_, owner, 3));
    const CacheBlockLocation own = owner[0].Blocks()[0]->Location();
    coordinator_.CacheCompletedBlocks(other, TailAtThree(/*owner_continues=*/false, /*publish_state=*/false),
                                      CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(other);
    const auto residents = FillBucket(0);
    ASSERT_EQ(pool_.NumEmptyLcmBlocks(), 0);

    // The canonical block is evicted for the copy's room; the publication
    // then keeps the owner's own block and copies that one.
    const auto decode = AdmitForTest(coordinator_, owner, GroupDemand{.extent = DenseGrowth{1}},
                                     TailAtThree(/*owner_continues=*/true, /*publish_state=*/true));
    ASSERT_TRUE(decode);
    ASSERT_EQ(decode->pages_to_copy[0].size(), 1u);
    EXPECT_EQ(decode->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveCacheBlockId(own));
    EXPECT_EQ(CachedTailBlock(), own) << "the owner's block is published at the tail";
    coordinator_.Free(owner);
}

// ---------------------------------------------------------------------------
// Tail publication: the partial page at a final frontier is published under
// hash@T beside the full pages, and the entries of one tail live and die as
// a unit across groups.
// ---------------------------------------------------------------------------
class TailPublicationTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;
    // Group 0: full attention with 2-token blocks (two blocks per page);
    // group 1: full attention with 4-token blocks; group 2: a sliding window
    // of 3 tokens over 2-token blocks (lookback of one block).
    TailPublicationTest()
        : pool_(/*num_lcm_blocks=*/64, {1, 1, 1}),
          coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
                  CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
                  CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                 .sliding_window = 3,
                                 .cache_blocks_per_lcm_block = 1,
                                 .block_granularity = 2},
              },
              kP, pool_, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/false)} {}

    // Admits a request over `tokens` and publishes everything below
    // num_computed_tokens as a final frontier. Returns the tail published.
    TailPublication PublishFinal(std::vector<BlockTable>& tables, const std::vector<std::int32_t>& tokens,
                                 std::int32_t num_computed_tokens) {
        EXPECT_TRUE(AdmitForTest(coordinator_, tables, static_cast<std::int32_t>(tokens.size())));
        const std::int32_t full_pages = num_computed_tokens / kP;
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < full_pages; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        hashes_ = ContentHashes(pages);
        const std::int32_t tail_tokens = num_computed_tokens - full_pages * kP;
        TailPublication tail{
            .tail_tokens = tail_tokens,
            .hash = HashPrefixPage(token_span(tokens.data() + full_pages * kP, static_cast<std::size_t>(tail_tokens)),
                                   hashes_.empty() ? std::string{} : hashes_.back()),
        };
        const RequestProgress progress{
            .completed_pages =
                CompletedPages{
                    .prefix_hashes = hashes_, .first_new_prefix_page = 0, .tail = tail, .owner_continues = false},
            .num_computed_tokens = num_computed_tokens,
        };
        coordinator_.CacheCompletedBlocks(tables, progress, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        return tail;
    }

    bool Cached(std::uint32_t group, const std::string& hash, std::int32_t offset) const {
        return coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(group))
            .Contains(pool_, Key(hash, group, offset));
    }

    BlockPool pool_;
    CacheCoordinator coordinator_;
    std::vector<std::string> hashes_;
};

TEST_F(TailPublicationTest, PublishesThePartialPageUnderHashAtT) {
    std::vector<BlockTable> tables(3);
    // 11 computed tokens: two full pages, then rows [8, 11) of page 2.
    const TailPublication tail = PublishFinal(tables, MakeTokens(12), /*num_computed_tokens=*/11);
    ASSERT_EQ(tail.tail_tokens, 3);
    ASSERT_EQ(hashes_.size(), 2u);
    EXPECT_EQ(coordinator_.NumTailEntries(), 1u);
    // Group 0 (g = 2): blocks [8,10) and the partial [10,11) -> offsets 0, 1.
    EXPECT_TRUE(Cached(0, tail.hash, 0));
    EXPECT_TRUE(Cached(0, tail.hash, 1));
    EXPECT_FALSE(Cached(0, tail.hash, 2));
    // Group 1 (g = 4): the single partial block.
    EXPECT_TRUE(Cached(1, tail.hash, 0));
    EXPECT_FALSE(Cached(1, tail.hash, 1));
    // Group 2 (sliding, W = 3): the window behind 11 is rows [9, 11), which
    // the two tail blocks [8, 10) and [10, 11) hold -- one block more than a
    // page-aligned boundary's lookback.
    EXPECT_TRUE(Cached(2, tail.hash, 1));
    EXPECT_TRUE(Cached(2, tail.hash, 0));
    EXPECT_TRUE(Cached(0, hashes_[1], 1)) << "the full pages are published as before";
    EXPECT_EQ(coordinator_.TailBlocksFor(0, 3), 2);
    EXPECT_EQ(coordinator_.TailBlocksFor(1, 3), 1);
    EXPECT_EQ(coordinator_.DeviceBoundaryResidency(Key(tail.hash, 0), tail.tail_tokens),
              CacheCoordinator::BoundaryResidency::kComplete);
    coordinator_.Free(tables);
}

TEST_F(TailPublicationTest, AnAlignedFrontierPublishesNoTail) {
    std::vector<BlockTable> tables(3);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 8));
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheCompletedBlocksForTest(coordinator_, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    EXPECT_EQ(coordinator_.NumTailEntries(), 0u);
    coordinator_.Free(tables);
}

TEST_F(TailPublicationTest, EvictingOneTailBlockTakesItsCompanionsAndPrunesTheTable) {
    std::vector<BlockTable> tables(3);
    const TailPublication tail = PublishFinal(tables, MakeTokens(12), /*num_computed_tokens=*/11);
    coordinator_.Free(tables);  // cache-only: every tail block is evictable
    ASSERT_EQ(coordinator_.NumTailEntries(), 1u);

    // Evicting group 1's partial block through the planner's path takes group
    // 0's two blocks and the sliding group's block with it: a tail is usable
    // only whole.
    CacheBlockLocation location{};
    {
        const CacheBlockRef victim = coordinator_.GroupPrefixIndex(1).Find(pool_, Key(tail.hash, 1, 0));
        ASSERT_TRUE(static_cast<bool>(victim));
        location = victim->Location();
    }
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, 1, location));
    EXPECT_FALSE(Cached(0, tail.hash, 0));
    EXPECT_FALSE(Cached(0, tail.hash, 1));
    EXPECT_FALSE(Cached(2, tail.hash, 1));
    EXPECT_EQ(coordinator_.NumTailEntries(), 0u) << "no tier holds the hash any more";
    EXPECT_TRUE(Cached(0, hashes_[1], 0)) << "the full pages stay";
    EXPECT_EQ(coordinator_.DeviceBoundaryResidency(Key(tail.hash, 0), tail.tail_tokens),
              CacheCoordinator::BoundaryResidency::kNone);
}

// A sliding group's acceptance is not monotone in T: it can hold the lookback
// behind a tail while the lookback behind the page boundary below it is gone.
// The probe judges it per candidate, so the tail stays resumable.
TEST_F(TailPublicationTest, ASlidingGroupMissingThePageLookbackStillResumesAtTheTail) {
    std::vector<BlockTable> tables(3);
    const std::vector<std::int32_t> tokens = MakeTokens(12);
    const TailPublication tail = PublishFinal(tables, tokens, /*num_computed_tokens=*/11);
    coordinator_.Free(tables);
    // The sliding group (W=3 over 2-token blocks) needs the rows [T-2, T)
    // behind a boundary: behind the tail at 11 the blocks [8, 10) and
    // [10, 11), behind the page boundary 8 the block [6, 8). Evict the
    // latter alone.
    CacheBlockLocation location{};
    {
        const CacheBlockRef victim = coordinator_.GroupPrefixIndex(2).Find(pool_, Key(hashes_[1], 2, 1));
        ASSERT_TRUE(static_cast<bool>(victim));
        location = victim->Location();
    }
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, 2, location));
    EXPECT_TRUE(Cached(2, tail.hash, 1));

    const CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(
        CacheCoordinator::PromptPrefix{.prefix_hashes = hashes_, .tokens = tokens, .max_tokens = 11});
    EXPECT_EQ(probe.device.num_common_tokens, 11);
    ASSERT_TRUE(probe.device.tail.has_value());
    EXPECT_EQ(probe.device.tail->hash, tail.hash);
    EXPECT_EQ(probe.device.tail->per_group[2].hits, (std::vector<std::uint8_t>{1, 1}));

    // Below the tail the page boundary 8 is not resumable for the sliding
    // group, and a sliding group publishes only its newest boundary's
    // lookback, so nothing below 8 is resumable either.
    const CacheCoordinator::PrefixProbe below = coordinator_.ProbePromptPrefix(
        CacheCoordinator::PromptPrefix{.prefix_hashes = hashes_, .tokens = tokens, .max_tokens = 10});
    EXPECT_EQ(below.device.num_common_tokens, 0);
}

// Oracle: whatever the page size P, the block size g and the window W, a
// prompt resuming at an unaligned T must hold, in the sliding group, every
// block that intersects the window [T - W + 1, T) the first query after T
// attends -- the owner published them and the hitter claimed (or copied)
// them; blocks below the window stay holes.
TEST(SlidingWindowOracleTest, AHitterHoldsEveryBlockOfTheWindowBehindAnUnalignedBoundary) {
    std::mt19937 rng{11};
    for (std::int32_t trial = 0; trial < 160; ++trial) {
        const std::int32_t g = std::array{1, 2, 4}[rng() % 3];
        const std::int32_t P = g * static_cast<std::int32_t>(1 + rng() % 4);
        const std::int32_t W = 2 + static_cast<std::int32_t>(rng() % (3 * P));
        BlockPool pool(/*num_lcm_blocks=*/512, {1, 1});
        CacheCoordinator coord = MakeCoordinator(
            std::vector<CacheGroupSpec>{
                CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = P},
                CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                               .sliding_window = W,
                               .cache_blocks_per_lcm_block = 1,
                               .block_granularity = g},
            },
            P, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
            /*stream_device_cache_to_host=*/false);
        // The owner computes L tokens and publishes its frontier T < L as a
        // final point, T unaligned.
        const std::int32_t T = P + 1 + static_cast<std::int32_t>(rng() % (3 * P));
        if (T % P == 0) {
            continue;
        }
        const std::int32_t L = T + 1 + static_cast<std::int32_t>(rng() % P);
        const std::vector<std::int32_t> tokens = MakeTokens(L, 1000 * (trial + 1));
        std::vector<BlockTable> owner(2);
        ASSERT_TRUE(AdmitForTest(coord, owner, L));
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < T / P; ++page) {
            pages.emplace_back(tokens.begin() + page * P, tokens.begin() + (page + 1) * P);
        }
        const std::vector<std::string> hashes = ContentHashes(pages);
        const std::int32_t tail_tokens = T % P;
        const TailPublication tail{
            .tail_tokens = tail_tokens,
            .hash = HashPrefixPage(token_span(tokens.data() + (T / P) * P, static_cast<std::size_t>(tail_tokens)),
                                   hashes.empty() ? std::string{} : hashes.back()),
        };
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = hashes, .first_new_prefix_page = 0, .tail = tail},
            .num_computed_tokens = T,
        };
        coord.CacheCompletedBlocks(owner, progress, CacheCoordinatorTestAccess::NextAccessEpoch(coord));
        coord.Free(owner);

        // A prompt sharing exactly T tokens resumes there.
        std::vector<std::int32_t> prompt(tokens.begin(), tokens.begin() + T);
        prompt.push_back(7);
        const CacheCoordinator::PrefixProbe probe = coord.ProbePromptPrefix(
            CacheCoordinator::PromptPrefix{.prefix_hashes = hashes, .tokens = prompt, .max_tokens = T});
        ASSERT_EQ(probe.device.num_common_tokens, T) << "P=" << P << " g=" << g << " W=" << W << " T=" << T;
        std::vector<BlockTable> hitter(2);
        ASSERT_TRUE(
            AdmitForTest(coord, hitter, CacheCoordinator::PrefixProbe{probe}, GroupDemand{.extent = DenseGrowth{1}}))
            << "P=" << P << " g=" << g << " W=" << W << " T=" << T;
        const std::span<const CacheBlockRef> row = hitter[1].Blocks();
        const std::int32_t first_window_row = std::max(0, T - W + 1);
        for (std::int32_t block = 0; block * g < T; ++block) {
            const std::int32_t block_end = std::min((block + 1) * g, T);
            const bool in_window = block_end > first_window_row;
            ASSERT_LT(static_cast<std::size_t>(block), row.size());
            EXPECT_EQ(static_cast<bool>(row[static_cast<std::size_t>(block)]), in_window)
                << "P=" << P << " g=" << g << " W=" << W << " T=" << T << " block " << block;
        }
        coord.Free(hitter);
    }
}

TEST_F(TailPublicationTest, CapDropsTheLeastRecentlyUsedTailUnderOnePriorWithItsBlocks) {
    // Nine requests sharing page 0 and ending at nine different offsets of
    // page 1 publish nine tails under the same prior; the first tail is hit
    // by a probe before the ninth arrives.
    std::vector<std::vector<BlockTable>> tables;
    std::vector<TailPublication> tails;
    std::vector<std::vector<std::int32_t>> streams;
    for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior + 1; ++extra) {
        std::vector<std::int32_t> tokens = MakeTokens(kP);
        for (std::int32_t i = 0; i < 3; ++i) {
            tokens.push_back(1000 * extra + i);
        }
        if (extra == TailTable::kMaxTailsPerPrior + 1) {
            const auto hit = CacheCoordinatorTestAccess::MatchPromptPrefix(
                coordinator_, CacheCoordinator::PromptPrefix{
                                  .prefix_hashes = hashes_, .tokens = streams.front(), .max_tokens = kP + 3});
            ASSERT_EQ(hit.device.num_common_tokens, kP + 3) << "the first tail is a hit";
        }
        tables.emplace_back(3);
        tails.push_back(PublishFinal(tables.back(), tokens, /*num_computed_tokens=*/kP + 3));
        streams.push_back(std::move(tokens));
        coordinator_.Free(tables.back());
    }
    EXPECT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    EXPECT_EQ(coordinator_.TailCapEvictions(), 1);
    EXPECT_TRUE(Cached(0, tails.front().hash, 0)) << "the hit kept the first tail";
    EXPECT_FALSE(Cached(0, tails[1].hash, 0)) << "the least recently used tail's blocks left the index";
    EXPECT_FALSE(Cached(1, tails[1].hash, 0));
    EXPECT_FALSE(Cached(2, tails[1].hash, 1));
    for (std::size_t i = 2; i < tails.size(); ++i) {
        EXPECT_TRUE(Cached(0, tails[i].hash, 0)) << "tail " << i;
    }
}

}  // namespace
}  // namespace tokenspeed::test
