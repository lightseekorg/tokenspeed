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
#include <set>
#include <span>
#include <string>
#include <unordered_set>
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
    CacheBlockRef got = pool.AcquireBlock(group_id, coordinator.Allocator(group_index).CacheBlocksPerLcmBlock());
    const std::int32_t id = got->Location().lcm_block_id;
    coordinator.GroupPrefixIndex(group_index).Register(pool, got, key, NextTestAccessEpoch());
    got.reset();
    return id;
}

CacheBlockLocation CacheBoundaryForGroup(CacheCoordinator& coordinator, BlockPool& pool,
                                         const std::string& content_hash, std::uint32_t group_id,
                                         std::uint64_t access_epoch, std::int32_t logical_block_index,
                                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk) {
    const std::int32_t group_index = static_cast<std::int32_t>(group_id);
    CacheBlockRef block_ref = pool.AcquireBlock(group_id, coordinator.Allocator(group_index).CacheBlocksPerLcmBlock());
    _assert(static_cast<bool>(block_ref), "test cache block allocation failed");
    const CacheBlockLocation location = block_ref->Location();
    coordinator.GroupPrefixIndex(group_index)
        .Register(pool, block_ref, Key(content_hash, group_id), access_epoch, logical_block_index, boundary_kind);
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

TEST(CacheGroupTest, HoldsSpecGroupIdManager) {
    BlockPool pool(8);
    auto mgr = std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1, /*group_id=*/7);
    CacheGroup g(
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        std::move(mgr), std::make_unique<FullAttnMatcher>());
    EXPECT_EQ(g.Id(), 7u);
    EXPECT_EQ(g.Spec().kind, AttnKind::kFull);
}

TEST(MakeCoordinatorTest, BuildsOneGroupPerSpec) {
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    EXPECT_EQ(coord.NumGroups(), 2);
}

TEST(MakeCoordinatorTest, UsesOneCacheBlockPerLcmBlockByDefault) {
    BlockPool pool(1);
    const std::array specs{CacheGroupSpec{.block_granularity = 4}};

    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);

    EXPECT_EQ(coordinator.Allocator(0).CacheBlocksPerLcmBlock(), 1);
}

TEST(MakeCoordinatorTest, Qwen35UsesUniformLogicalPWithDifferentPacking) {
    BlockPool pool(32);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 8, .block_granularity = 128},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
    };
    CacheCoordinator coord = MakeCoordinator(specs, /*prefix_granularity=*/128, pool);

    EXPECT_EQ(coord.PrefixGranularity(), 128);
    EXPECT_EQ(coord.GroupBlockGranularity(0), 128);
    EXPECT_EQ(coord.GroupBlockGranularity(1), 128);
    EXPECT_EQ(coord.Allocator(0).CacheBlocksPerLcmBlock(), 8);
    EXPECT_EQ(coord.Allocator(1).CacheBlocksPerLcmBlock(), 1);

    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/128));
    for (const BlockTable& table : tables) {
        EXPECT_EQ(table.NumBlocks(), 1);
    }

    const std::vector<std::string> hashes = ContentHashes({std::vector<std::int32_t>(128, 7)});
    CacheFullBlocksForTest(coord, tables, hashes);
    coord.Free(tables);
    CoordinatorMatch match = MatchPrefixForTest(coord, hashes).device;
    EXPECT_EQ(match.num_common_tokens, 128);
    ASSERT_EQ(match.per_group.size(), specs.size());
    for (const PrefixMatch& group_match : match.per_group) {
        EXPECT_EQ(group_match.blocks.size(), 1u);
    }
}

TEST(MakeCoordinatorTest, ManagersMayUseSmallerPagesThanTheCoordinatorDomain) {
    BlockPool pool(32);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 8},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 3,
         .block_granularity = 2},
    };

    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/8, pool);

    EXPECT_EQ(coordinator.PrefixGranularity(), 8);
    EXPECT_EQ(coordinator.GroupBlockGranularity(0), 8);
    EXPECT_EQ(coordinator.GroupBlockGranularity(1), 2);
}

TEST(CacheCoordinatorTest, ExpandsOneLogicalHashIntoPerGroupCacheBlocks) {
    BlockPool pool(32);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 8},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 5,
         .cache_blocks_per_lcm_block = 3,
         .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/8, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    ASSERT_EQ(tables[0].NumBlocks(), 1);
    ASSERT_EQ(tables[1].NumBlocks(), 4);

    const std::vector<std::string> hashes = ContentHashes({std::vector<std::int32_t>(8, 7)});
    coordinator.CacheCompletedBlocks(tables, hashes, NextTestAccessEpoch(), /*first_new_prefix_page=*/0,
                                     /*num_computed_tokens=*/8, CacheBoundaryKind::kChunk);
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
    BlockPool pool(8);
    const std::vector<CacheGroupSpec> valid = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
    };
    EXPECT_THROW(MakeCoordinator(valid, /*prefix_granularity=*/0, pool), std::runtime_error);
    EXPECT_THROW(MakeCoordinator(valid, /*prefix_granularity=*/-1, pool), std::runtime_error);

    const std::vector<CacheGroupSpec> zero_packing = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 0, .block_granularity = 128},
    };
    EXPECT_THROW(MakeCoordinator(zero_packing, /*prefix_granularity=*/128, pool), std::runtime_error);
}

TEST(CacheCoordinatorTest, RejectsManagerGeometryThatDiffersFromDomainOrSpec) {
    BlockPool pool(8);
    std::vector<CacheGroup> wrong_p;
    wrong_p.emplace_back(
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 48},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1), std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(CacheCoordinator(std::move(wrong_p), /*prefix_granularity=*/128, pool), std::runtime_error);

    std::vector<CacheGroup> wrong_k;
    wrong_k.emplace_back(
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 8, .block_granularity = 128},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1), std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(CacheCoordinator(std::move(wrong_k), /*prefix_granularity=*/128, pool), std::runtime_error);
}

TEST(CacheCoordinatorTest, RejectsManagerGroupIdThatDiffersFromItsIndex) {
    BlockPool pool(8);
    std::vector<CacheGroup> groups;
    groups.emplace_back(
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 128},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1,
                                         /*group_id=*/1),
        std::make_unique<FullAttnMatcher>());

    EXPECT_THROW(CacheCoordinator(std::move(groups), /*prefix_granularity=*/128, pool), std::runtime_error);
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
    BlockPool pool(16);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 8, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    EXPECT_EQ(coord.PrefixGranularity(), 4);
    EXPECT_EQ(coord.GroupBlockGranularity(0), coord.GroupBlockGranularity(1));
}

TEST(CoordinatorMatchTest, BothGroupsAllMiss) {
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(11);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
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
    BlockPool device_pool(4);
    BlockPool host_pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, &host_pool);
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
    BlockPool device_pool(4);
    BlockPool host_pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, &host_pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});

    CacheForGroup(coordinator, device_pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, host_pool, hashes[0], /*group_id=*/0);

    ASSERT_TRUE(coordinator.ClearCache());
    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 0);
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), 4);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 4);
}

TEST(CacheCoordinatorTest, ClearDeviceCacheRejectsPinnedEntryWithoutPartialMutation) {
    BlockPool pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[1], /*group_id=*/0);
    CacheBlockRef pin = coordinator.AcquireDeviceCachedBlock(Key(hashes[0], /*group_id=*/0));

    EXPECT_FALSE(coordinator.ClearDeviceCache());
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 8);

    pin.reset();
    EXPECT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 0);
}

std::vector<GroupDemand> FreshDemands(std::vector<BlockTable>& tables, std::span<const std::int32_t> tokens) {
    EXPECT_EQ(tables.size(), tokens.size());
    std::vector<GroupDemand> demands;
    demands.reserve(tables.size());
    for (std::size_t i = 0; i < tables.size(); ++i) {
        demands.push_back(GroupDemand{.table = &tables[i], .num_tokens = tokens[i]});
    }
    return demands;
}

TEST(CacheCoordinatorAdmissionTest, ReportsOnlyFreshlyAllocatedKernelPageIds) {
    BlockPool pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> first_tokens{5, 1};
    std::vector<GroupDemand> first_demands = FreshDemands(tables, first_tokens);

    const std::optional<CacheCoordinator::AdmissionResult> first =
        coordinator.Admit(coordinator.ProbePrefix({}), first_demands);

    ASSERT_TRUE(first);
    ASSERT_EQ(first->new_page_ids.size(), 2u);
    EXPECT_EQ(first->new_page_ids[0], coordinator.Allocator(0).BlockTablePageIds(tables[0]));
    EXPECT_EQ(first->new_page_ids[1], coordinator.Allocator(1).BlockTablePageIds(tables[1]));

    const std::array<std::int32_t, 2> tail_tokens{1, 1};
    std::vector<GroupDemand> tail_demands = FreshDemands(tables, tail_tokens);
    const std::optional<CacheCoordinator::AdmissionResult> tail =
        coordinator.Admit(coordinator.ProbePrefix({}), tail_demands, first->access_epoch);

    ASSERT_TRUE(tail);
    ASSERT_EQ(tail->new_page_ids.size(), 2u);
    EXPECT_TRUE(tail->new_page_ids[0].empty());
    EXPECT_TRUE(tail->new_page_ids[1].empty());
}

TEST(CacheCoordinatorAdmissionTest, AcquireMatchedFullPrefixUsesOneAccessEpoch) {
    BlockPool pool(8);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands));

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
    BlockPool pool(8);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    std::vector<BlockTable> tables(coordinator.NumGroups());

    const std::optional<CacheCoordinator::AdmissionResult> first = AdmitForTest(coordinator, tables, /*num_tokens=*/8);
    ASSERT_TRUE(first);

    std::vector<GroupDemand> second_demands{
        GroupDemand{
            .table = &tables[0],
            .num_tokens = 8,
            .prefix_hashes = std::span<const std::string>(hashes).first(2),
            .new_prefix_hash_begin = 0,
            .completed_boundary_kind = CacheBoundaryKind::kChunk,
            .num_computed_tokens = 8,
        },
    };
    const std::optional<CacheCoordinator::AdmissionResult> second =
        coordinator.Admit(coordinator.ProbePrefix({}), second_demands, first->access_epoch);
    ASSERT_TRUE(second);
    EXPECT_EQ(second->access_epoch, first->access_epoch);

    std::vector<GroupDemand> final_publication{
        GroupDemand{
            .table = &tables[0],
            .prefix_hashes = hashes,
            .new_prefix_hash_begin = 2,
            .completed_boundary_kind = CacheBoundaryKind::kEndpoint,
            .num_computed_tokens = 16,
        },
    };
    const std::optional<CacheCoordinator::AdmissionResult> final =
        coordinator.Admit(coordinator.ProbePrefix({}), final_publication, first->access_epoch);
    ASSERT_TRUE(final);

    for (const CacheBlockRef& block_ref : tables[0].Blocks()) {
        const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
            coordinator.GroupPrefixIndex(0).MetadataFor(pool, block_ref->Location());
        ASSERT_TRUE(metadata);
        EXPECT_EQ(metadata->last_access_epoch, first->access_epoch);
    }
}

TEST(CacheCoordinatorAdmissionTest, ProbeAndRejectedAdmissionDoNotAdvanceAccessEpoch) {
    BlockPool pool(1);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}});
    std::vector<BlockTable> owner_tables(coordinator.NumGroups());

    const std::optional<CacheCoordinator::AdmissionResult> owner =
        AdmitForTest(coordinator, owner_tables, /*num_tokens=*/4);
    ASSERT_TRUE(owner);
    coordinator.CacheFullBlocks(owner_tables, hashes, owner->access_epoch);
    const CacheBlockLocation location = owner_tables[0].Blocks().front()->Location();

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
        coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, owner->access_epoch);

    std::vector<BlockTable> rejected_tables(coordinator.NumGroups());
    std::vector<GroupDemand> rejected_demands{
        GroupDemand{.table = &rejected_tables[0], .num_tokens = 4},
    };
    EXPECT_FALSE(coordinator.Admit(std::move(probe), rejected_demands));
    metadata = coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, owner->access_epoch);

    coordinator.Free(owner_tables);
    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    std::vector<GroupDemand> hit_demands{
        GroupDemand{.table = &hit_tables[0]},
    };
    const std::optional<CacheCoordinator::AdmissionResult> hit =
        coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands);
    ASSERT_TRUE(hit);
    EXPECT_EQ(hit->access_epoch, owner->access_epoch + 1);
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionPrefersOlderRequestBeforePosition) {
    BlockPool pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> old_hashes = ContentHashes({{1, 1, 1, 1}});
    const std::vector<std::string> new_hashes = ContentHashes({{2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});

    std::vector<BlockTable> old_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> old_request =
        AdmitForTest(coordinator, old_tables, /*num_tokens=*/4);
    ASSERT_TRUE(old_request);
    coordinator.CacheFullBlocks(old_tables, old_hashes, old_request->access_epoch);
    coordinator.Free(old_tables);

    std::vector<BlockTable> new_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> new_request =
        AdmitForTest(coordinator, new_tables, /*num_tokens=*/12);
    ASSERT_TRUE(new_request);
    coordinator.CacheFullBlocks(new_tables, new_hashes, new_request->access_epoch);
    coordinator.Free(new_tables);

    std::vector<BlockTable> contender(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, contender, /*num_tokens=*/4));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(old_hashes[0], 0)));
    for (const std::string& hash : new_hashes) {
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hash, 0)));
    }
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionIsSuffixFirstWithinOneAccessEpoch) {
    BlockPool pool(4);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});

    std::vector<BlockTable> cached_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> cached_request =
        AdmitForTest(coordinator, cached_tables, /*num_tokens=*/16);
    ASSERT_TRUE(cached_request);
    coordinator.CacheFullBlocks(cached_tables, hashes, cached_request->access_epoch);
    coordinator.Free(cached_tables);

    std::vector<BlockTable> contender(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, contender, /*num_tokens=*/4));
    for (std::size_t i = 0; i + 1 < hashes.size(); ++i) {
        EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[i], 0)));
    }
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes.back(), 0)));
}

TEST(CacheCoordinatorAdmissionTest, RejectedCachedHitDoesNotRefreshAccessEpoch) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t cached_hit = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef pinned = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
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
        EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands));
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
    BlockPool pool(3);
    std::vector<CacheBlockRef> held = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/3);
    ASSERT_EQ(held.size(), 3u);
    const CacheBlockLocation first_released = held[2]->Location();
    const CacheBlockLocation second_released = held[0]->Location();
    held[2].reset();
    held[0].reset();

    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), first_released);
    EXPECT_EQ(tables[0].Blocks()[1]->Location(), second_released);
}

TEST(CacheCoordinatorAdmissionTest, PacksSlotsFromOneNewParentTogether) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> held = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);
    ASSERT_EQ(held.size(), 2u);
    const CacheBlockLocation first_released = held[1]->Location();
    held[1].reset();
    held[0].reset();

    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, first_released.lcm_block_id);
    EXPECT_EQ(tables[0].Blocks()[1]->Location().lcm_block_id, first_released.lcm_block_id);
    EXPECT_NE(tables[0].Blocks()[0]->Location().slot_index, tables[0].Blocks()[1]->Location().slot_index);
}

TEST(CacheCoordinatorAdmissionTest, FreeCachedHitCostsNoAdditionalPlacement) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, hit_parent);
    EXPECT_NE(tables[0].Blocks()[1]->Location().lcm_block_id, hit_parent);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(CacheCoordinatorAdmissionTest, ReservedCapacityLivesInBlockTableUntilConsumed) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{1};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    demands[0].reserve_tokens = 8;

    const std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator.Admit(coordinator.ProbePrefix({}), demands);
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
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{4, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands));

    ASSERT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[0].Blocks()[0]->Location(), (CacheBlockLocation{.lcm_block_id = hit_parent, .slot_index = 1}));
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, hit_parent);
}

TEST(CacheCoordinatorAdmissionTest, PacksDemandBeforeConsumingAnotherGroupsOnlyParent) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(coordinator, pool, hashes[0], 0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    const std::int32_t packed_parent = tables[0].Blocks()[0]->Location().lcm_block_id;
    EXPECT_EQ(tables[0].Blocks()[1]->Location().lcm_block_id, packed_parent);
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, packed_parent);
}

TEST(CacheCoordinatorAdmissionTest, RetriesWithCompactPackingWhenGreedyFreeSlotsWouldReject) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t first_partial_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef fills_first_parent = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/2);
    const std::int32_t second_partial_parent = CacheForGroup(coordinator, pool, hashes[1], 0);
    fills_first_parent.reset();
    ASSERT_NE(first_partial_parent, second_partial_parent);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 8};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[0].Blocks()[0]->Location().lcm_block_id, tables[0].Blocks()[1]->Location().lcm_block_id);
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_NE(tables[1].Blocks()[0]->Location().lcm_block_id, tables[1].Blocks()[1]->Location().lcm_block_id);
}

TEST(CacheCoordinatorAdmissionTest, KeepsCachedChildrenWhenUnpackedPlacementFits) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef fills_first_parent = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/2);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    fills_first_parent.reset();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
}

TEST(CacheCoordinatorAdmissionTest, UsesEmptyParentBeforeEvictingCachedChild) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 1);
    EXPECT_EQ(tables[0].NumBlocks(), 2);
}

TEST(CacheCoordinatorAdmissionTest, DoesNotShareFreeSlotsFromBoundForeignParent) {
    BlockPool pool(1);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    CacheBlockRef pinned = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/4);
    ASSERT_TRUE(pinned);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    EXPECT_TRUE(tables[0].Blocks().empty());
    EXPECT_TRUE(tables[1].Blocks().empty());
}

TEST(CacheCoordinatorAdmissionTest, EvictsOldestCachedChildNeededForCapacity) {
    BlockPool pool(1);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, FullEvictionUsesAccessEpochInsteadOfPermanentHitClass) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> first_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> first =
        AdmitForTest(coordinator, first_tables, /*num_tokens=*/4);
    ASSERT_TRUE(first);
    coordinator.CacheFullBlocks(first_tables, std::span{hashes}.first(1), first->access_epoch);
    coordinator.Free(first_tables);

    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> no_new_tokens{0};
    std::vector<GroupDemand> hit_demands = FreshDemands(hit_tables, no_new_tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(std::span{hashes}.first(1)), hit_demands));
    coordinator.Free(hit_tables);

    std::vector<BlockTable> second_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> second =
        AdmitForTest(coordinator, second_tables, /*num_tokens=*/4);
    ASSERT_TRUE(second);
    coordinator.CacheFullBlocks(second_tables, std::span{hashes}.subspan(1), second->access_epoch);
    coordinator.Free(second_tables);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    std::vector<BlockTable> fresh_tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, fresh_tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, StateEvictionUsesAccessEpochBeforePosition) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    CacheBlockRef late_checkpoint = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    coordinator.GroupPrefixIndex(0).Register(pool, late_checkpoint, Key(hashes[0], 0), NextTestAccessEpoch(),
                                             /*logical_block_index=*/10);
    late_checkpoint.reset();

    CacheBlockRef early_checkpoint = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    coordinator.GroupPrefixIndex(0).Register(pool, early_checkpoint, Key(hashes[1], 0), NextTestAccessEpoch(),
                                             /*logical_block_index=*/0);
    early_checkpoint.reset();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, OrdinaryStateKeepsLongerUnhitFrontier) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
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

TEST(CacheCoordinatorAdmissionTest, OrdinarySwaKeepsLongerUnhitFrontier) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
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

TEST(CacheCoordinatorAdmissionTest, EndpointUsesNormalValueWithinSameEpoch) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    constexpr std::uint64_t kSameAccessEpoch = 7;

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/2, CacheBoundaryKind::kEndpoint);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/6);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, PromotedBoundaryUsesNormalValueWithinSameEpoch) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    constexpr std::uint64_t kSameAccessEpoch = 7;

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/2, CacheBoundaryKind::kPromoted);
    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, kSameAccessEpoch,
                          /*logical_block_index=*/6);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, ActualHitPromotesChunkBoundaryToNormalValue) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/2);
    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> hit =
        AdmitForTest(coordinator, hit_tables, coordinator.ProbePrefix(std::span{hashes}.first(1)), GroupDemand{});
    ASSERT_TRUE(hit);
    coordinator.Free(hit_tables);

    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, hit->access_epoch,
                          /*logical_block_index=*/6);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, EndpointIsNotBelowEveryPreviouslyHitBlock) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    GroupAllocator& manager = coordinator.Allocator(0);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    CacheBoundaryForGroup(coordinator, pool, hashes[0], /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/2);
    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    const std::optional<CacheCoordinator::AdmissionResult> old_hit =
        AdmitForTest(coordinator, hit_tables, coordinator.ProbePrefix(std::span{hashes}.first(1)), GroupDemand{});
    ASSERT_TRUE(old_hit);
    coordinator.Free(hit_tables);

    CacheBoundaryForGroup(coordinator, pool, hashes[1], /*group_id=*/0, old_hit->access_epoch + 1,
                          /*logical_block_index=*/6, CacheBoundaryKind::kEndpoint);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
}

TEST(CacheCoordinatorAdmissionTest, MixedGroupTieEvictsNonClosedBeforeFullHistory) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
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
        {.table = &tables[0], .num_tokens = 4},
        {.table = &tables[1]},
        {.table = &tables[2]},
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[1], 1)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(2).Contains(pool, Key(hashes[2], 2)));
}

TEST(CacheCoordinatorAdmissionTest, ProspectiveUncachedReclaimDoesNotEvictCachedBlock) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::string cached_hash = ContentHashes({{1, 1, 1, 1}}).front();
    CacheBoundaryForGroup(coordinator, pool, cached_hash, /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .num_tokens = 4, .num_computed_tokens = 8},
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(cached_hash, 0)));
}

TEST(CacheCoordinatorAdmissionTest, QwenScaleChunkLifecyclePublishesOneStateSnapshotPerChunk) {
    constexpr std::int32_t kBlockTokens = 128;
    constexpr std::int32_t kPromptPages = 256;
    constexpr std::int32_t kChunkPages = 64;
    // Keep this publication-contract test out of eviction-policy territory.
    BlockPool pool(/*num_lcm_blocks=*/512);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.kind = AttnKind::kMambaState,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.kind = AttnKind::kMambaState,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = kBlockTokens},
        {.kind = AttnKind::kFull,
         .sliding_window = 0,
         .cache_blocks_per_lcm_block = 32,
         .block_granularity = kBlockTokens},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, kBlockTokens, pool);
    std::vector<std::string> hashes;
    hashes.reserve(kPromptPages);
    for (std::int32_t page = 0; page < kPromptPages; ++page) {
        hashes.push_back(std::to_string(page));
    }

    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::optional<std::uint64_t> access_epoch;
    for (std::int32_t chunk = 0; chunk < kPromptPages / kChunkPages; ++chunk) {
        const std::int32_t first_page = chunk * kChunkPages;
        std::vector<GroupDemand> demands;
        demands.reserve(specs.size());
        for (std::size_t group = 0; group < specs.size(); ++group) {
            demands.push_back(GroupDemand{
                .table = &tables[group],
                .num_tokens = kChunkPages * kBlockTokens,
                .prefix_hashes = std::span<const std::string>{hashes}.first(first_page),
                .new_prefix_hash_begin = std::max(first_page - kChunkPages, 0),
                .completed_boundary_kind = first_page == 0 ? std::nullopt : std::optional{CacheBoundaryKind::kChunk},
                .num_computed_tokens = first_page * kBlockTokens,
                .reserve_tokens = chunk == kPromptPages / kChunkPages - 1 ? 1 : 0,
            });
        }
        const std::optional<CacheCoordinator::AdmissionResult> admission =
            coordinator.Admit(coordinator.ProbePrefix({}), demands, access_epoch);
        ASSERT_TRUE(admission) << "chunk " << chunk;
        access_epoch = admission->access_epoch;
    }

    std::vector<GroupDemand> decode_demands;
    decode_demands.reserve(specs.size());
    for (std::size_t group = 0; group < specs.size(); ++group) {
        decode_demands.push_back(GroupDemand{
            .table = &tables[group],
            .num_tokens = 1,
            .prefix_hashes = hashes,
            .new_prefix_hash_begin = kPromptPages - kChunkPages,
            .completed_boundary_kind = CacheBoundaryKind::kEndpoint,
            .num_computed_tokens = kPromptPages * kBlockTokens,
        });
    }
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), decode_demands, access_epoch));
    coordinator.Free(tables);

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 4);
    EXPECT_EQ(coordinator.GroupPrefixIndex(1).NumEntries(pool), 4);
    EXPECT_EQ(coordinator.GroupPrefixIndex(2).NumEntries(pool), 4);
    EXPECT_EQ(coordinator.GroupPrefixIndex(3).NumEntries(pool), kPromptPages);
    EXPECT_EQ(MatchPrefixForTest(coordinator, hashes).device.num_common_tokens, kPromptPages * kBlockTokens);
}

TEST(CacheCoordinatorAdmissionTest, RebindsOnlyAfterEvictingWholeForeignParent) {
    BlockPool pool(1);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    ASSERT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{1});
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
}

TEST(CacheCoordinatorAdmissionTest, UsesLocalSlotAfterEvictingAnotherWholeParent) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
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
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    GroupAllocator& manager = coordinator.Allocator(0);
    std::vector<CacheBlockRef> cached = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/2, /*num=*/3);
    ASSERT_EQ(cached.size(), 3u);
    ASSERT_EQ(cached[0]->Location().lcm_block_id, cached[1]->Location().lcm_block_id);
    ASSERT_NE(cached[0]->Location().lcm_block_id, cached[2]->Location().lcm_block_id);
    const std::int32_t middle_candidate_parent = cached[2]->Location().lcm_block_id;

    coordinator.GroupPrefixIndex(0).Register(pool, cached[0], Key(hashes[0], 0), /*access_epoch=*/1);
    coordinator.GroupPrefixIndex(0).Register(pool, cached[1], Key(hashes[1], 0), /*access_epoch=*/3);
    coordinator.GroupPrefixIndex(0).Register(pool, cached[2], Key(hashes[2], 0), /*access_epoch=*/2);
    cached.clear();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    EXPECT_EQ(tables[1].Blocks()[0]->Location().lcm_block_id, middle_candidate_parent);
}

TEST(CacheCoordinatorAdmissionTest, ProspectiveHitParentCannotBecomeVictim) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t protected_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    CacheForGroup(coordinator, pool, hashes[0], 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(std::span<const std::string>{hashes}.first(1));

    EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands));
    EXPECT_EQ(pool.BoundGroup(protected_parent), std::optional<std::uint32_t>{0});
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
    EXPECT_TRUE(tables[0].Blocks().empty());
    EXPECT_TRUE(tables[1].Blocks().empty());
}

TEST(CacheCoordinatorAdmissionTest, ReclaimsTableOwnerBeforeEvictingProspectiveVictim) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, tables, hashes);
    const CacheBlockLocation reclaimed = tables[0].Blocks()[0]->Location();

    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .num_tokens = 4, .num_computed_tokens = 8},
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    ASSERT_EQ(tables[0].NumBlocks(), 3);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    ASSERT_TRUE(tables[0].Blocks()[2]);
    EXPECT_EQ(tables[0].Blocks()[2]->Location(), reclaimed);
}

TEST(CacheCoordinatorAdmissionTest, EvictsProspectiveVictimCachedDuringCommit) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    const CacheBlockLocation reclaimed = tables[0].Blocks()[0]->Location();

    std::vector<GroupDemand> demands = {
        {
            .table = &tables[0],
            .num_tokens = 4,
            .prefix_hashes = hashes,
            .new_prefix_hash_begin = 0,
            .completed_boundary_kind = CacheBoundaryKind::kChunk,
            .num_computed_tokens = 8,
        },
    };
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));

    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    ASSERT_TRUE(tables[0].Blocks()[2]);
    EXPECT_EQ(tables[0].Blocks()[2]->Location(), reclaimed);
}

TEST(CacheCoordinatorAdmissionTest, RejectsProspectiveVictimWithAnExtraOwner) {
    BlockPool pool(2);
    const std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                                .sliding_window = 5,
                                                .cache_blocks_per_lcm_block = 1,
                                                .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, tables, hashes);

    const std::array<CacheKey, 1> first_key{Key(hashes[0], /*group_id=*/0)};
    GroupAllocator& manager = coordinator.Allocator(0);
    PrefixMatch extra_owner = coordinator.GroupPrefixIndex(0).AcquireMatched(
        pool, first_key, /*begin_blocks=*/0,
        coordinator.GroupMatcher(0).Probe(coordinator.GroupPrefixIndex(0), pool, first_key, /*begin_blocks=*/0,
                                          /*max_blocks=*/1),
        NextTestAccessEpoch());
    ASSERT_EQ(extra_owner.NumHitBlocks(), 1);

    std::vector<GroupDemand> demands = {
        {.table = &tables[0], .num_tokens = 4, .num_computed_tokens = 8},
    };
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_TRUE(tables[0].Blocks()[0]);
}

TEST(CoordinatorMatchTest, SwaMissForcesZeroCommon) {
    // full caches 2 pages, swa caches nothing -> common = min(2, 0) = 0.
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[1], 0);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    EXPECT_EQ(m.per_group[0].blocks.size(), 0u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 0u);
}

TEST(CoordinatorAllocTest, ColdStartAllocatesAlignedPages) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    EXPECT_EQ(prefix.device.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.num_tokens = 8}));
    // 8 tokens / page 4 = 2 pages in EACH group; tables aligned.
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, ClaimsCommonPrefixThenAllocatesRemainder) {
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    // swa window 4 -> pages_needed 1, so a single cached front page is a hit.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[0], 1);

    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    // 8 tokens total, 1 page (4 tokens) common -> 4 uncached tokens -> +1 page each.
    ASSERT_TRUE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.num_tokens = 4}));
    EXPECT_EQ(tables[0].NumBlocks(), 2);  // 1 claimed + 1 allocated
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, CrossGroupShortfallAllocatesNothing) {
    BlockPool pool(5);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();
    // 12 tokens -> 3 pages per group = 6 needed, only 5 free -> fail, nothing taken.
    EXPECT_FALSE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.num_tokens = 12}));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);  // untouched, not rolled back
}

TEST(CoordinatorStepTest, AcquireKeepsGroupsAligned) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 page each
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 more each
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorStepTest, AcquireShortfallAllocatesNothing) {
    BlockPool pool(3);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();
    // 2 pages per group (8 tokens) = 4 blocks, only 3 free -> fail, nothing taken.
    EXPECT_FALSE(AdmitForTest(coord, tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(CoordinatorStepTest, CacheFullBlocksThenMatchHits) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 4));  // 1 page each
    CacheFullBlocksForTest(coord, tables, ch);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
}

TEST(CoordinatorStepTest, FreeReturnsAllGroups) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 8));  // 2 pages each = 4 blocks
    std::int32_t free_mid = pool.NumEmptyLcmBlocks();
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_mid + 4);
}

TEST(CoordinatorStepTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});

    // Request A: cold, allocate 2 pages each, cache both, free.
    {
        CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
        EXPECT_EQ(prefix.device.num_common_tokens, 0);
        std::vector<BlockTable> a(2);
        ASSERT_TRUE(AdmitForTest(coord, a, std::move(prefix), GroupDemand{.num_tokens = 8}));
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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{7, 7, 7, 7}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coord, tables, 8));  // 2 pages each
    EXPECT_THROW(CacheFullBlocksForTest(coord, tables, ch, /*first_slot=*/2), std::runtime_error);
    EXPECT_THROW(CacheFullBlocksForTest(coord, tables, ch, /*first_slot=*/-1), std::runtime_error);
}

TEST(CoordinatorMatchTest, SwaRunCutByFullBoundDropsToNoValidMatch) {
    // full covers 4; swa's tail run {2,3,4} bounded to 4 leaves run {2,3} <
    // pages_needed 3 with holes at 0,1 -> no valid swa match, common = 0.
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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

TEST(CoordinatorPromotionTest, ProbePreservesClosedCoverageBeforeWindowConvergence) {
    BlockPool pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, pool, hash, /*group_id=*/0);
    }
    CacheForGroup(coordinator, pool, hashes[3], /*group_id=*/1);

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);

    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.prefix_closed_tokens, 32);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = FreshDemands(tables, std::array<std::int32_t, 2>{0, 0});
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->promotion_boundary_tokens, 32);
}

TEST(CoordinatorPromotionTest, ClosedCoverageIsMinimumAcrossClosedGroups) {
    BlockPool pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, pool, hash, /*group_id=*/0);
    }
    for (const std::string& hash : std::span{hashes}.first(6)) {
        CacheForGroup(coordinator, pool, hash, /*group_id=*/1);
    }
    CacheForGroup(coordinator, pool, hashes[3], /*group_id=*/2);

    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);

    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.prefix_closed_tokens, 24);
}

TEST(CoordinatorPromotionTest, LongerWindowHitDoesNotCreatePromotion) {
    BlockPool pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : std::span{hashes}.first(4)) {
        CacheForGroup(coordinator, pool, hash, /*group_id=*/0);
    }
    for (std::size_t i : {1u, 2u, 3u, 5u, 6u, 7u}) {
        CacheForGroup(coordinator, pool, hashes[i], /*group_id=*/1);
    }

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.prefix_closed_tokens, 16);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = FreshDemands(tables, std::array<std::int32_t, 2>{0, 0});
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->promotion_boundary_tokens, 0);
}

TEST(CoordinatorPromotionTest, PureWindowGroupsHaveNoClosedCoverage) {
    BlockPool pool(32);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 5,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    CacheForGroup(coordinator, pool, hashes[2], /*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[3], /*group_id=*/0);

    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);

    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.prefix_closed_tokens, 0);
}

TEST(CoordinatorPromotionTest, HostTierPreservesDevicePromotion) {
    BlockPool device_pool(64);
    BlockPool host_pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, &host_pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, device_pool, hash, /*group_id=*/0);
    }
    CacheForGroup(coordinator, device_pool, hashes[3], /*group_id=*/1);

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    ASSERT_EQ(probe.device.num_common_tokens, 16);
    ASSERT_EQ(probe.device.prefix_closed_tokens, 32);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = FreshDemands(tables, std::array<std::int32_t, 2>{0, 0});
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->promotion_boundary_tokens, 32);
}

TEST(CoordinatorPromotionTest, HostClosedCoverageCreatesPromotion) {
    BlockPool device_pool(64);
    BlockPool host_pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, &host_pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, host_pool, hash, /*group_id=*/0);
    }
    CacheForGroup(coordinator, host_pool, hashes[3], /*group_id=*/1);

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    ASSERT_EQ(probe.host.num_common_tokens, 16);
    ASSERT_EQ(probe.host.prefix_closed_tokens, 32);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = FreshDemands(tables, std::array<std::int32_t, 2>{0, 0});
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->promotion_boundary_tokens, 32);
}

TEST(CoordinatorPromotionTest, HostHitCoveringClosedBoundaryDoesNotCreatePromotion) {
    BlockPool device_pool(64);
    BlockPool host_pool(64);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, &host_pool,
                                                   /*stream_device_cache_to_host=*/false);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0},
                                                           {1, 1, 1, 1},
                                                           {2, 2, 2, 2},
                                                           {3, 3, 3, 3},
                                                           {4, 4, 4, 4},
                                                           {5, 5, 5, 5},
                                                           {6, 6, 6, 6},
                                                           {7, 7, 7, 7}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, device_pool, hash, /*group_id=*/0);
        CacheForGroup(coordinator, host_pool, hash, /*group_id=*/0);
    }
    CacheForGroup(coordinator, device_pool, hashes[3], /*group_id=*/1);
    CacheForGroup(coordinator, host_pool, hashes[7], /*group_id=*/1);

    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    ASSERT_EQ(probe.device.num_common_tokens, 16);
    ASSERT_EQ(probe.device.prefix_closed_tokens, 32);
    ASSERT_EQ(probe.host.num_common_tokens, 32);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands = FreshDemands(tables, std::array<std::int32_t, 2>{0, 0});
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->promotion_boundary_tokens, 0);
}

TEST(CoordinatorMatchTest, TwoSwaGroupsSharedBoundaryMatches) {
    // pages_needed 3. full: 5; both SWA groups cache {1,2,3}, so each accepts
    // the SAME boundary 4 in the single sweep -- no cascade, full truncates 5 -> 4.
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // = swaB above
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // = swaA above
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // swaA, needed = 3
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},  // swaB, needed = 3
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool);
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
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 10,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(5);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 4,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(coord, pool, ch[0], 0);
    CacheForGroup(coord, pool, ch[0], 1);
    std::int32_t free_before = pool.NumEmptyLcmBlocks();

    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    ASSERT_EQ(prefix.device.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    // Uncached 8 tokens -> 2 pages/group = 4 needed; 5 parents - 2 cached parents = 3 free -> fail.
    EXPECT_FALSE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.num_tokens = 8}));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(CacheCoordinatorReclaimExpired, OnlySlidingWindowGroupEvicts) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

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
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 40,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // Groups 0 and 2 fully cache both pages; group 1 caches nothing. The all-miss
    // group zeroes the bound before group 2 matches (the reverse would be a cascade).
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 2);

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0) << "one group all-miss -> common 0";
}

TEST(CacheCoordinatorStoreCandidates, CollectsKeysWithoutPinningDeviceBlocks) {
    BlockPool pool(/*num_lcm_blocks=*/16);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    BlockPool host_pool(4);
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool, &host_pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    CacheFullBlocksForTest(coordinator, tables, hashes, /*first_slot=*/0);
    std::vector<CacheCoordinator::StoreCandidate> pending = coordinator.TakePendingStores();

    ASSERT_EQ(pending.size(), 4u);  // 2 pages x 2 groups, group-wrapped keys
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
    for (const BlockTable& table : tables) {
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            EXPECT_EQ(block_ref.use_count(), 2) << "only the request table and Device cache own the block";
        }
    }
    // Typed keys keep the group distinct without changing the content hash.
    std::unordered_set<CacheKey, CacheKeyHash> keys;
    for (const auto& c : pending) keys.insert(c.key);
    EXPECT_EQ(keys.size(), 4u);
    // Collection is group-major: 2 pages for group 0, then 2 for group 1.
    for (std::size_t i = 0; i < pending.size(); ++i) {
        EXPECT_EQ(pending[i].key, Key(hashes[i % 2], static_cast<std::uint32_t>(i / 2))) << "candidate " << i;
    }

    // Re-registering the same hashes yields nothing new (IsCached skip).
    CacheFullBlocksForTest(coordinator, tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());

    coordinator.Free(tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 12);  // four Manager-owned cache entries remain resident
}

TEST(CacheCoordinatorStoreCandidates, DisabledByDefaultCollectsNothing) {
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator off = MakeCoordinator(specs, 2, pool);
    std::vector<BlockTable> tables(off.NumGroups());
    ASSERT_TRUE(AdmitForTest(off, tables, /*num_tokens=*/8));  // pages 0..3; N=8 slides out pages 0,1
    EXPECT_EQ(SlideCredit(off, tables, 8), 2) << "collection-off counts uncached ref-1 blocks";
    off.Free(tables);

    BlockPool host_pool(4);
    CacheCoordinator on = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool device_pool(2);
    BlockPool host_pool(1);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, &host_pool);
    const CacheKey first = Key("first", 0);
    const CacheKey second = Key("second", 0);
    CacheBlockRef first_ref = host_pool.AcquireBlock(0, 2);
    CacheBlockRef second_ref = host_pool.AcquireBlock(0, 2);
    coordinator.CacheHostBlock(first_ref, first);
    coordinator.CacheHostBlock(second_ref, second);
    first_ref.reset();
    second_ref.reset();

    CacheBlockRef replacement = coordinator.AcquireHostBlock(0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(host_pool.BoundGroup(replacement->Location().lcm_block_id), 0u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 1);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(first));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(second));
}

TEST(CacheCoordinatorHostReplacement, RebindsACompleteEvictableParentAcrossGroups) {
    BlockPool device_pool(2);
    BlockPool host_pool(1);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, &host_pool);
    CacheBlockRef first = host_pool.AcquireBlock(0, 2);
    CacheBlockRef second = host_pool.AcquireBlock(0, 2);
    coordinator.CacheHostBlock(first, Key("first", 0));
    coordinator.CacheHostBlock(second, Key("second", 0));
    first.reset();
    second.reset();

    CacheBlockRef replacement = coordinator.AcquireHostBlock(1);

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
    CacheBlockRef block_ref = host_pool.AcquireBlock(gid, manager.CacheBlocksPerLcmBlock());
    const std::int32_t id = block_ref->Location().lcm_block_id;
    coordinator.GroupPrefixIndex(static_cast<std::int32_t>(gid))
        .Register(host_pool, block_ref, key, NextTestAccessEpoch());
    block_ref.reset();
    return id;
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
    return {CacheGroupSpec{
                .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
            CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                           .sliding_window = 4,
                           .cache_blocks_per_lcm_block = 1,
                           .block_granularity = 2}};
}

TEST(CacheCoordinatorHostExtension, PreservesAllNullWindowExtensionSlots) {
    BlockPool pool(3);
    const std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 1,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(3);
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool, &host_pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0}, {1, 1}, {2, 2}});
    for (const std::string& hash : hashes) {
        (void)HostPut(coordinator, host_pool, hash, /*gid=*/0);
    }

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    const std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator.Admit(coordinator.ProbePrefix(hashes), demands);

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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(5);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(5);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    EXPECT_TRUE(host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1));
}

TEST(CacheCoordinatorHostExtension, MatchPinsPagesUntilResultDies) {
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    for (int j = 1; j <= 3; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j = 2; j <= 3; ++j) (void)HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 1);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 5);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
    CacheBlockRef last_free = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(last_free);
    EXPECT_FALSE(host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1))
        << "pinned cache entries must not be selected as victims";
    last_free.reset();
    m = {};
    EXPECT_EQ(coord.NumPinnedHostCachedBlocks(), 0);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 1);
}

TEST(CacheCoordinatorHostExtension, DeepCascadeConverges) {
    // Host twin of the deep cascade (dev = 0 so the block math mirrors the device test):
    // convergence must land on extension 2 and take no refs along the way.
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 10,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    BlockPool host_pool(32);
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(16);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 6,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 2,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 2},
    };
    BlockPool host_pool(16);
    CacheCoordinator coord = MakeCoordinator(specs, 2, pool, &host_pool);
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
// (AttnKind::kMambaState) would reuse unchanged.
// ---------------------------------------------------------------------------

TEST(MambaAnalogTest, HitIsSingleSnapshotPlusLeadingHoles) {
    // needed = ceil((5-1)/4) = 1: the match is the RIGHTMOST cached block with
    // null holes below -- byte-identical to MambaManager.find_longest_cache_hit.
    BlockPool pool(16);
    SwaManager mgr(/*block_granularity=*/4, /*sliding_window=*/5);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (std::int32_t slot : {0, 1, 3}) {
        CacheBlockRef block_ref = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
        mgr.RegisterCachedBlock(pool, block_ref, Key(ch[static_cast<std::size_t>(slot)], 0), NextTestAccessEpoch());
        block_ref.reset();
    }

    std::vector<CacheKey> keys;
    for (const std::string& h : ch) keys.push_back(Key(h, 0));
    PrefixMatch m =
        mgr.AcquireMatchedBlocks(pool, keys, /*begin_blocks=*/0,
                                 mgr.Probe(pool, keys, /*begin_blocks=*/0, /*max_blocks=*/4), NextTestAccessEpoch());
    ASSERT_EQ(m.blocks.size(), 4u);
    EXPECT_FALSE(m.blocks[0]);
    EXPECT_FALSE(m.blocks[1]);
    EXPECT_FALSE(m.blocks[2]);
    EXPECT_TRUE(m.blocks[3]);
    EXPECT_EQ(m.NumHitBlocks(), 1);
}

TEST(MambaAnalogTest, RetentionKeepsOnlyTheLastStateBlock) {
    // vLLM: get_num_skipped_tokens = n-1 (only the last token's state matters).
    // Our slide rule skips n-W+1 tokens, so W=2 IS that policy: at 16 computed
    // tokens pages 0..2 free and only the tail state page survives.
    BlockPool pool(16);
    SwaManager mgr(/*block_granularity=*/4, /*sliding_window=*/2);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, /*num_tokens=*/16));
    ASSERT_EQ(table.NumBlocks(), 4);

    EXPECT_EQ(mgr.BlocksReclaimableAt(table, /*num_computed_tokens=*/16, /*count_uncached=*/true), 3);
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    mgr.ReclaimExpired(pool, table, 16);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before + 3);
    ASSERT_EQ(table.NumBlocks(), 4) << "holes preserve slot alignment";
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_FALSE(table.Blocks()[2]);
    EXPECT_TRUE(table.Blocks()[3]) << "the live state block";
    mgr.Free(table);
}

TEST(MambaAnalogTest, HybridFullSwaMambaComposesUnderOnePool) {
    // Task-4 shape: full + real window (W=8) + mamba-analog (W=5, needed=1),
    // three groups sharing ONE BlockPool; common = min over all three.
    BlockPool pool(64);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 8,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 5,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(coord, pool, h, 0);
    CacheForGroup(coord, pool, ch[2], 1);  // W=8 needs a 2-page tail run
    CacheForGroup(coord, pool, ch[3], 1);
    CacheForGroup(coord, pool, ch[3], 2);  // mamba-analog: one snapshot at the boundary

    CacheCoordinator::PrefixProbe prefix = coord.ProbePrefix(ch);
    EXPECT_EQ(prefix.device.num_common_tokens, 16);
    ASSERT_EQ(prefix.device.per_group.size(), 3u);
    EXPECT_EQ(std::ranges::count(prefix.device.per_group[0].hits, std::uint8_t{1}), 4);
    EXPECT_EQ(std::ranges::count(prefix.device.per_group[1].hits, std::uint8_t{1}), 2);
    EXPECT_EQ(std::ranges::count(prefix.device.per_group[2].hits, std::uint8_t{1}), 1);

    // Claim + acquire keeps the pool unified and balanced across all three.
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, std::move(prefix), GroupDemand{.num_tokens = 4}));
    coord.Free(tables);
}

TEST(MambaAnalogTest, HostTierStoresAndMatchesTheSnapshotOnly) {
    // vLLM offloading treats a mamba group as sliding_window_size_in_blocks=1:
    // the host index needs only the boundary snapshot; earlier keys stay holes.
    BlockPool pool(16);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kSlidingWindow,
         .sliding_window = 5,
         .cache_blocks_per_lcm_block = 1,
         .block_granularity = 4},
    };
    BlockPool host_pool(8);
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});

    std::vector<std::int32_t> fp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(coord, host_pool, ch[static_cast<std::size_t>(j)], 0));
    const std::int32_t snapshot = HostPut(coord, host_pool, ch[3], 1);  // ONLY the boundary snapshot

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = MatchPrefixForTest(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 16);  // boundary 4 blocks * P=4 (floor 1 + extension 3)
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2]}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{0, 0, snapshot}));
    EXPECT_EQ(m.per_group[0].NumHitBlocks() + m.per_group[1].NumHitBlocks(), 4);
}

// kMambaState is the named form of the analog pinned above: MakeCoordinator maps it to the
// W=2 machinery, so a mixed full+state model converges with single-snapshot state semantics.
TEST(MambaStateKindTest, FactoryMapsStateKindToAlignSemantics) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (int j = 0; j <= 2; ++j) CacheForGroup(coord, pool, ch[static_cast<std::size_t>(j)], 0);
    CacheForGroup(coord, pool, ch[2], 1);  // ONLY the boundary snapshot for the state group

    CoordinatorMatch m = MatchPrefixForTest(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 12);  // full covers 3 pages; state resumes off snapshot @2
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[1].NumHitBlocks(), 1);
    ASSERT_EQ(m.per_group[1].blocks.size(), 3u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_FALSE(m.per_group[1].blocks[1]);
    EXPECT_TRUE(m.per_group[1].blocks[2]);  // [null, null, snapshot]
}

TEST(MambaStateKindTest, StateGroupRetentionKeepsOnlyLastPage) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/16));  // 4 pages
    coord.ReclaimExpired(tables, /*num_computed_tokens=*/16);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    EXPECT_FALSE(tables[0].Blocks()[2]);
    EXPECT_TRUE(tables[0].Blocks()[3]);  // skipped = n-1: only the live state page
    coord.Free(tables);
}

TEST(CompletedBoundaryTest, HistoricalHashesWithoutBoundaryAreNotPublished) {
    BlockPool pool(8);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}});
    const std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = hashes,
        .new_prefix_hash_begin = 1,
        .completed_boundary_kind = std::nullopt,
        .num_computed_tokens = 4,
    }};

    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
    coordinator.Free(tables);
}

TEST(CompletedBoundaryTest, RejectsNewHashesWithoutBoundaryKind) {
    BlockPool pool(8);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/4, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}});
    const std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = hashes,
        .new_prefix_hash_begin = 0,
        .completed_boundary_kind = std::nullopt,
        .num_computed_tokens = 4,
    }};

    EXPECT_THROW(coordinator.Admit(coordinator.ProbePrefix({}), demands), std::runtime_error);
    coordinator.Free(tables);
}

TEST(MambaStateRegistrationTest, MambaPublishesOnlyChunkBoundary) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/12));  // 3 pages
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    std::vector<GroupDemand> demands;
    for (BlockTable& table : tables) {
        demands.push_back(GroupDemand{
            .table = &table,
            .prefix_hashes = ch,
            .new_prefix_hash_begin = 0,
            .completed_boundary_kind = CacheBoundaryKind::kChunk,
            .num_computed_tokens = 12,
        });
    }
    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[0], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[2], 0)));
    EXPECT_FALSE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[0], 1)));
    EXPECT_FALSE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[1], 1)));
    EXPECT_TRUE(coord.GroupPrefixIndex(1).Contains(pool, Key(ch[2], 1)));
    coord.Free(tables);
}

TEST(MambaStateRegistrationTest, MambaPublishesAlignedEndpoint) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/12));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = ch,
        .new_prefix_hash_begin = 0,
        .completed_boundary_kind = CacheBoundaryKind::kEndpoint,
        .num_computed_tokens = 12,
    }};
    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[0], 0)));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[1], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[2], 0)));
    const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
        coord.GroupPrefixIndex(0).MetadataFor(pool, tables[0].Blocks()[2]->Location());
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, CacheBoundaryKind::kEndpoint);
    coord.Free(tables);
}

TEST(MambaStateRegistrationTest, MambaPublishesNoUnalignedBoundary) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/12));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = ch,
        .new_prefix_hash_begin = 0,
        .completed_boundary_kind = CacheBoundaryKind::kEndpoint,
        .num_computed_tokens = 10,
    }};
    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands));
    EXPECT_EQ(coord.GroupPrefixIndex(0).NumEntries(pool), 0);
    coord.Free(tables);
}

TEST(DecodeDestinationTest, AdmitMaterializesOnlyRequestedStateSuffix) {
    BlockPool pool(/*num_lcm_blocks=*/5);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands{
        {
            .table = &tables[0],
            .num_tokens = 8,
            .reserve_tokens = 1,
        },
        {
            .table = &tables[1],
            .num_tokens = 8,
            .reserve_tokens = 1,
            .materialized_suffix_start = 3,
        },
    };

    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands));
    ASSERT_EQ(tables[0].NumBlocks(), 5);
    EXPECT_TRUE(std::ranges::all_of(tables[0].Blocks(),
                                    [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
    ASSERT_EQ(tables[1].NumBlocks(), 5);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    EXPECT_FALSE(tables[1].Blocks()[2]);
    EXPECT_TRUE(tables[1].Blocks()[3]);
    EXPECT_TRUE(tables[1].Blocks()[4]);
    EXPECT_EQ(tables[0].AvailableTokens(), 2);
    EXPECT_EQ(tables[1].AvailableTokens(), 2);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
    coordinator.Free(tables);
}

TEST(DecodeDestinationTest, HistoryGroupsDeterminePrefixAndStateGetsAlignedHoles) {
    BlockPool pool(/*num_lcm_blocks=*/8);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool);
    const std::vector<std::string> hashes = ContentHashes({{1, 1}, {2, 2}, {3, 3}});
    for (const std::string& hash : hashes) {
        CacheForGroup(coordinator, pool, hash, /*group_id=*/0);
    }

    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 0);

    CacheCoordinator::PrefixProbe probe = coordinator.ProbeDecodeDevicePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 6);
    ASSERT_EQ(probe.device.per_group.size(), 2u);
    ASSERT_EQ(probe.device.per_group[0].hits.size(), 3u);
    EXPECT_TRUE(std::ranges::all_of(probe.device.per_group[0].hits, [](std::int32_t page_id) { return page_id != 0; }));
    ASSERT_EQ(probe.device.per_group[1].hits.size(), 3u);
    EXPECT_TRUE(std::ranges::all_of(probe.device.per_group[1].hits, [](std::int32_t page_id) { return page_id == 0; }));

    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands{
        {
            .table = &tables[0],
            .num_tokens = 2,
            .reserve_tokens = 1,
        },
        {
            .table = &tables[1],
            .num_tokens = 8,
            .reserve_tokens = 1,
            .materialized_suffix_start = 3,
        },
    };
    const std::optional<CacheCoordinator::AdmissionResult> admission = coordinator.Admit(std::move(probe), demands);
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->device_prefix_tokens, 6);
    ASSERT_EQ(tables[0].NumBlocks(), 5);
    EXPECT_TRUE(std::ranges::all_of(tables[0].Blocks(),
                                    [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
    ASSERT_EQ(tables[1].NumBlocks(), 5);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    EXPECT_FALSE(tables[1].Blocks()[2]);
    EXPECT_TRUE(tables[1].Blocks()[3]);
    EXPECT_TRUE(tables[1].Blocks()[4]);
    coordinator.Free(tables);
}

TEST(DecodeDestinationTest, SparseAdmissionFailureLeavesAllGroupsUnchanged) {
    BlockPool pool(/*num_lcm_blocks=*/4);
    std::vector<CacheGroupSpec> specs = {
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.kind = AttnKind::kMambaState, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> demands{
        {
            .table = &tables[0],
            .num_tokens = 8,
            .reserve_tokens = 1,
        },
        {
            .table = &tables[1],
            .num_tokens = 8,
            .reserve_tokens = 1,
            .materialized_suffix_start = 3,
        },
    };

    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 4);
}

TEST(SwaRegistrationTest, SwaBoundaryRequiresTrailingWindow) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 9,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/16));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = ch,
        // Only page 3 is new; the two-page resume tail crosses the prior chunk.
        .new_prefix_hash_begin = 3,
        .completed_boundary_kind = CacheBoundaryKind::kChunk,
        .num_computed_tokens = 16,
    }};
    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[0], 0)));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[1], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[2], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(ch[3], 0)));
    coord.Free(tables);
}

TEST(SwaRegistrationTest, UnalignedEndpointPublishesTrailingFullPages) {
    BlockPool pool(32);
    std::vector<CacheGroupSpec> specs = {{.kind = AttnKind::kSlidingWindow,
                                          .sliding_window = 9,
                                          .cache_blocks_per_lcm_block = 1,
                                          .block_granularity = 4}};
    CacheCoordinator coord = MakeCoordinator(specs, 4, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(AdmitForTest(coord, tables, /*num_tokens=*/14));
    std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    std::vector<GroupDemand> demands{{
        .table = &tables[0],
        .prefix_hashes = hashes,
        .new_prefix_hash_begin = 0,
        .completed_boundary_kind = CacheBoundaryKind::kEndpoint,
        .num_computed_tokens = 14,
    }};

    ASSERT_TRUE(coord.Admit(coord.ProbePrefix({}), demands));
    EXPECT_FALSE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_TRUE(coord.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    coord.Free(tables);
}

TEST(GroupAllocatorBoundaryTest, BoundaryPromotionIsMonotonic) {
    BlockPool pool(1);
    FullAttnManager manager(/*block_granularity=*/4);
    CacheBlockRef block_ref = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheKey key = Key(std::string(64, 'a'), 0);

    manager.RegisterCachedBlock(pool, block_ref, key, /*access_epoch=*/1, /*logical_block_index=*/0,
                                CacheBoundaryKind::kChunk);
    manager.RegisterCachedBlock(pool, block_ref, key, /*access_epoch=*/2, /*logical_block_index=*/0,
                                CacheBoundaryKind::kPromoted);
    manager.RegisterCachedBlock(pool, block_ref, key, /*access_epoch=*/3, /*logical_block_index=*/0,
                                CacheBoundaryKind::kChunk);

    const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
        manager.CachedBlockMetadataFor(pool, block_ref->Location());
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->boundary_kind, CacheBoundaryKind::kPromoted);
}

}  // namespace
}  // namespace tokenspeed::test
