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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
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
#include "cache/tier/transfer_manager.h"
#include "unit_test_helper.h"

namespace tokenspeed::test {
namespace {

TEST(CacheCoordinatorAdmissionBuckets, HostHitDestinationsPreserveSourceOwnership) {
    for (const std::int32_t shards : {1, 2, 4}) {
        SCOPED_TRACE(shards);
        BlockPool device_pool(4, {8});
        BlockPool host_pool(4, {8});
        const std::array specs{CacheGroupSpec{.attn_kind = AttnKind::kFull,
                                              .cache_blocks_per_lcm_block = 8,
                                              .block_granularity = 4,
                                              .shard_count = shards}};
        auto coordinator = MakeCoordinator(specs, 4, device_pool, false, &host_pool, nullptr, false, true);
        const std::vector<std::string> hashes{"first", "second", "third", "fourth"};
        std::vector<std::int32_t> source_buckets;
        for (std::size_t i = 0; i < hashes.size(); ++i) {
            const std::int32_t bucket = static_cast<std::int32_t>((i + 1) % shards);
            auto source = coordinator.AcquireHostBlock(0, bucket);
            ASSERT_TRUE(source);
            source_buckets.push_back(coordinator.Allocator(0).BucketOf(source->Location()));
            coordinator.CacheHostBlock(source, CacheKey{.group_id = 0, .content_hash = hashes[i]}, std::nullopt);
        }
        std::vector<BlockTable> tables(1);
        const std::array demands{GroupDemand{.table = &tables[0], .extent = DenseGrowth{4}}};
        auto admitted =
            coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt, false);
        ASSERT_TRUE(admitted);
        EXPECT_EQ(admitted->host_prefix_tokens, 16);
        ASSERT_EQ(admitted->load_pairs.size(), hashes.size());
        for (std::size_t i = 0; i < admitted->load_pairs.size(); ++i) {
            const auto& transfer = admitted->load_pairs[i];
            EXPECT_EQ(coordinator.Allocator(0).BucketOf(transfer.source->Location()), source_buckets[i]);
            EXPECT_EQ(coordinator.Allocator(0).BucketOf(transfer.destination->Location()), source_buckets[i]);
        }
        coordinator.Free(tables);
    }
}

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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    EXPECT_EQ(coord.NumGroups(), 2);
}

TEST(MakeCoordinatorTest, UsesOneCacheBlockPerLcmBlockByDefault) {
    BlockPool pool(1, {1});
    const std::array specs{CacheGroupSpec{.block_granularity = 4}};

    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    CacheCoordinator coord = MakeCoordinator(specs, /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                                             /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    EXPECT_THROW(MakeCoordinator(valid, /*prefix_granularity=*/0, pool, /*enable_l3_storage=*/false,
                                 /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                 /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
                 std::runtime_error);
    EXPECT_THROW(MakeCoordinator(valid, /*prefix_granularity=*/-1, pool, /*enable_l3_storage=*/false,
                                 /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                 /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
                 std::runtime_error);

    const std::vector<CacheGroupSpec> zero_packing = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 0, .block_granularity = 128},
    };
    EXPECT_THROW(MakeCoordinator(zero_packing, /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                                 /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                 /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
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
    EXPECT_THROW(CacheCoordinator(std::move(wrong_p), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                                  /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                  /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
                 std::runtime_error);

    std::vector<CacheGroup> wrong_k;
    wrong_k.emplace_back(
        CacheGroupSpec{.attn_kind = AttnKind::kFull,
                       .sliding_window = 0,
                       .cache_blocks_per_lcm_block = 8,
                       .block_granularity = 128},
        std::make_unique<GroupAllocator>(/*cache_blocks_per_lcm_block=*/1, /*group_id=*/0, /*shard_count=*/1),
        std::make_unique<FullAttnMatcher>());
    EXPECT_THROW(CacheCoordinator(std::move(wrong_k), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                                  /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                  /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
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
                                  /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
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

    EXPECT_THROW(CacheCoordinator(std::move(groups), /*prefix_granularity=*/128, pool, /*enable_l3_storage=*/false,
                                  /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                  /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false),
                 std::runtime_error);
}

TEST(GroupAllocatorTest, ResolvesAffineKernelPageIdsAndRejectsInvalidLocations) {
    FullAttnManager full(/*block_granularity=*/128, /*cache_blocks_per_lcm_block=*/8);
    FullAttnManager state(/*block_granularity=*/128, /*cache_blocks_per_lcm_block=*/1);

    EXPECT_EQ(full.ResolveDeviceCacheBlockId({.lcm_block_id = 1, .slot_index = 0}), 1);
    EXPECT_EQ(full.ResolveDeviceCacheBlockId({.lcm_block_id = 1, .slot_index = 7}), 8);
    EXPECT_EQ(full.ResolveDeviceCacheBlockId({.lcm_block_id = 2, .slot_index = 0}), 9);
    EXPECT_EQ(state.ResolveDeviceCacheBlockId({.lcm_block_id = 2, .slot_index = 0}), 2);

    EXPECT_THROW(full.ResolveDeviceCacheBlockId({.lcm_block_id = 0, .slot_index = 0}), std::runtime_error);
    EXPECT_THROW(full.ResolveDeviceCacheBlockId({.lcm_block_id = 1, .slot_index = -1}), std::runtime_error);
    EXPECT_THROW(full.ResolveDeviceCacheBlockId({.lcm_block_id = 1, .slot_index = 8}), std::runtime_error);
    EXPECT_THROW(
        full.ResolveDeviceCacheBlockId({.lcm_block_id = std::numeric_limits<std::int32_t>::max(), .slot_index = 7}),
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
    CacheCoordinator coord = MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                                             /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});

    CacheForGroup(coordinator, device_pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, host_pool, hashes[0], /*group_id=*/0);

    ASSERT_TRUE(coordinator.ClearDeviceCache());
    const CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 4);
    EXPECT_EQ(device_pool.NumEmptyLcmBlocks(), 4);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 3);
    const CacheCoordinator::DeviceEvictionCounts& flushed = coordinator.DeviceEvictions().flush;
    EXPECT_EQ(flushed.blocks, 1u);
    EXPECT_EQ(flushed.host_resident, 1u) << "the Host tier still holds the page";
}

TEST(CacheCoordinatorTest, ClearCacheRemovesDeviceAndHostEntries) {
    BlockPool device_pool(4, {1});
    BlockPool host_pool(4, {1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], /*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[1], /*group_id=*/0);
    CacheBlockRef pin = coordinator.AcquireDeviceCachedBlock(Key(hashes[0], /*group_id=*/0));

    EXPECT_FALSE(coordinator.ClearDeviceCache());
    EXPECT_FALSE(coordinator.CacheIsClearable(/*include_host=*/false));
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 8);
    EXPECT_EQ(coordinator.DeviceEvictions().flush.blocks, 0u) << "a refused clear evicts nothing";

    pin.reset();
    EXPECT_TRUE(coordinator.CacheIsClearable(/*include_host=*/false));
    EXPECT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_EQ(coordinator.ProbePrefix(hashes).device.num_common_tokens, 0);
    EXPECT_EQ(coordinator.DeviceEvictions().flush.blocks, 2u);
    EXPECT_EQ(coordinator.DeviceEvictions().flush.host_resident, 0u) << "no Host pool";
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    std::vector<GroupDemand> first_demands = {
        GroupDemand{.table = &tables[0], .extent = DenseGrowth{5}},
        GroupDemand{.table = &tables[1], .extent = StateStep{StateStep::Kind::kRotate}},
    };
    const std::optional<CacheCoordinator::AdmissionResult> first =
        coordinator.Admit(coordinator.ProbePrefix({}), first_demands, RequestProgress{}, std::nullopt,
                          /*evict_blocks_past_segments=*/false);

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
        coordinator.Admit(coordinator.ProbePrefix({}), tail_demands, RequestProgress{}, first->access_epoch,
                          /*evict_blocks_past_segments=*/false);

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
        coordinator.Admit(coordinator.ProbePrefix({}), second_demands, second_progress, first->access_epoch,
                          /*evict_blocks_past_segments=*/false);
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
        coordinator.Admit(coordinator.ProbePrefix({}), final_publication, final_progress, first->access_epoch,
                          /*evict_blocks_past_segments=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    EXPECT_FALSE(coordinator.Admit(std::move(probe), rejected_demands, RequestProgress{}, std::nullopt,
                                   /*evict_blocks_past_segments=*/false));
    metadata = coordinator.GroupPrefixIndex(0).MetadataFor(pool, location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, owner->access_epoch);

    coordinator.Free(owner_tables);
    std::vector<BlockTable> hit_tables(coordinator.NumGroups());
    std::vector<GroupDemand> hit_demands{
        GroupDemand{.table = &hit_tables[0]},
    };
    const std::optional<CacheCoordinator::AdmissionResult> hit =
        coordinator.Admit(coordinator.ProbePrefix(hashes), hit_demands, RequestProgress{}, std::nullopt,
                          /*evict_blocks_past_segments=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
        EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands, RequestProgress{}, std::nullopt,
                                       /*evict_blocks_past_segments=*/false));
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 1> tokens{1};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    demands[0].reserve_tokens = 8;

    const std::optional<CacheCoordinator::AdmissionResult> result = coordinator.Admit(
        coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt, /*evict_blocks_past_segments=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    const std::int32_t hit_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{4, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(coordinator, pool, hashes[0], 0);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheBlockRef fills_first_parent = pool.AcquireBlock(/*group_id=*/0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    fills_first_parent.reset();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{8, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
}

TEST(CacheCoordinatorAdmissionTest, UsesEmptyParentBeforeEvictingCachedChild) {
    BlockPool pool(2, {2});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    CacheBlockRef pinned = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(pinned);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                   /*evict_blocks_past_segments=*/false));

    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    EXPECT_TRUE(tables[0].Blocks().empty());
    EXPECT_TRUE(tables[1].Blocks().empty());
}

TEST(CacheCoordinatorAdmissionTest, EvictsOldestCachedChildNeededForCapacity) {
    BlockPool pool(1, {2});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                  std::nullopt, /*evict_blocks_past_segments=*/false));
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, Key(hashes[1], 1)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(2).Contains(pool, Key(hashes[2], 2)));
}

TEST(CacheCoordinatorAdmissionTest, MergesGroupEpochsWithPinnedEntriesInEitherGroupOrder) {
    for (std::uint32_t state_group : {0u, 1u}) {
        SCOPED_TRACE(state_group);
        const std::uint32_t full_group = 1 - state_group;
        BlockPool pool(3, {1, 1});
        std::vector<CacheGroupSpec> specs(2);
        specs[state_group] = CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1};
        specs[full_group] =
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4};
        CacheCoordinator coordinator = MakeCoordinator(specs, 4, pool, false, nullptr, nullptr, false, false);
        const auto hashes = ContentHashes({{1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
        CacheBoundaryForGroup(coordinator, pool, hashes[0], full_group, 1, 0, CacheBoundaryKind::kEndpoint);
        CacheBoundaryForGroup(coordinator, pool, hashes[1], state_group, 2, 0, CacheBoundaryKind::kEndpoint);
        CacheBoundaryForGroup(coordinator, pool, hashes[2], full_group, 3, 1, CacheBoundaryKind::kEndpoint);
        CacheBlockRef pinned = coordinator.GroupPrefixIndex(full_group).Find(pool, Key(hashes[0], full_group));
        ASSERT_TRUE(pinned);
        std::vector<BlockTable> tables(2);
        std::vector<GroupDemand> demands(2);
        demands[state_group] = GroupDemand{.table = &tables[state_group], .extent = StateStep{StateStep::Kind::kKeep}};
        demands[full_group] = GroupDemand{.table = &tables[full_group], .extent = DenseGrowth{4}};
        ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt, false));
        EXPECT_TRUE(coordinator.GroupPrefixIndex(full_group).Contains(pool, Key(hashes[0], full_group)));
        EXPECT_FALSE(coordinator.GroupPrefixIndex(state_group).Contains(pool, Key(hashes[1], state_group)));
        EXPECT_TRUE(coordinator.GroupPrefixIndex(full_group).Contains(pool, Key(hashes[2], full_group)));
    }
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                        /*write_back_evictions=*/false);
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
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                   /*evict_blocks_past_segments=*/false));
    for (const BlockTable& table : tables) {
        EXPECT_EQ(table.NumBlocks(), 0);
    }
    // A new plan must revisit exhausted streams after references are released.
    // The oldest epoch stays pinned; the unpinned tie still favors state.
    pins[0].reset();
    pins[1].reset();
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                        /*write_back_evictions=*/false);
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
                                  std::nullopt, /*evict_blocks_past_segments=*/false));

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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::string cached_hash = ContentHashes({{1, 1, 1, 1}}).front();
    CacheBoundaryForGroup(coordinator, pool, cached_hash, /*group_id=*/0, /*access_epoch=*/1,
                          /*logical_block_index=*/0, CacheBoundaryKind::kChunk);

    // Two prefill steps fill the pool: the cached checkpoint, the first
    // chunk's output (now slot 0) and the second chunk's live slot.
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = StateStep{StateStep::Kind::kRotate}}};
    const RequestProgress progress{.num_computed_tokens = 4};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    // The third step's rotation releases the request's own unpublished input
    // block, which funds its fresh live slot: the cached checkpoint stays.
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{.num_computed_tokens = 8},
                                  std::nullopt, /*evict_blocks_past_segments=*/false));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(cached_hash, 0)));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(CacheCoordinatorAdmissionTest, RebindsOnlyAfterEvictingWholeForeignParent) {
    BlockPool pool(1, {2, 1});
    const std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    ASSERT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 4, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    const std::int32_t protected_parent = CacheForGroup(coordinator, pool, hashes[0], 0);
    CacheForGroup(coordinator, pool, hashes[1], 0);
    CacheForGroup(coordinator, pool, hashes[0], 1);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 4};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(std::span<const std::string>{hashes}.first(1));

    EXPECT_FALSE(coordinator.Admit(std::move(prefix), demands, RequestProgress{}, std::nullopt,
                                   /*evict_blocks_past_segments=*/false));
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, tables, hashes);
    const CacheBlockLocation reclaimed = tables[0].Blocks()[0]->Location();

    std::vector<GroupDemand> demands = {{.table = &tables[0], .extent = DenseGrowth{4}}};
    const RequestProgress progress{.num_computed_tokens = 8};
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                  /*evict_blocks_past_segments=*/false));

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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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
    EXPECT_FALSE(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                   /*evict_blocks_past_segments=*/false));
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                                   /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                                   /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                           /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(off.NumGroups());
    ASSERT_TRUE(AdmitForTest(off, tables, /*num_tokens=*/8));  // pages 0..3; N=8 slides out pages 0,1
    EXPECT_EQ(SlideCredit(off, tables, 8), 2) << "collection-off counts uncached ref-1 blocks";
    off.Free(tables);

    BlockPool host_pool(4, {1});
    CacheCoordinator on = MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool,
                                          /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                          /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const CacheKey first = Key("first", 0);
    const CacheKey second = Key("second", 0);
    CacheBlockRef first_ref = host_pool.AcquireBlock(0);
    CacheBlockRef second_ref = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(first_ref, first, std::nullopt);
    coordinator.CacheHostBlock(second_ref, second, std::nullopt);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    CacheBlockRef first = host_pool.AcquireBlock(0);
    CacheBlockRef second = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(first, Key("first", 0), std::nullopt);
    coordinator.CacheHostBlock(second, Key("second", 0), std::nullopt);
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

void HostHit(CacheCoordinator& coordinator, BlockPool& host_pool, const CacheKey& key, std::uint64_t access_epoch) {
    const auto group_index = static_cast<std::int32_t>(key.group_id);
    PrefixCacheIndex& index = coordinator.GroupPrefixIndex(group_index);
    const std::array keys{key};
    const GroupPrefixProbe probe =
        coordinator.GroupMatcher(group_index)
            .Probe(index, host_pool, keys, /*begin_blocks=*/0, /*max_blocks=*/1, /*extra_hits=*/nullptr);
    PrefixMatch match = index.AcquireMatched(host_pool, keys, /*begin_blocks=*/0, probe, access_epoch);
    ASSERT_EQ(match.NumHitBlocks(), 1);
}

TEST(CacheCoordinatorHostReplacement, SameGroupUsesLastAccessAfterPrefixHits) {
    for (const bool refresh_old : {false, true}) {
        SCOPED_TRACE(refresh_old);
        BlockPool device_pool(2, {1});
        BlockPool host_pool(2, {1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2}};
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                            /*snapshot_pool=*/nullptr,
                            /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
        HostPut(coordinator, host_pool, "old", 0);
        HostHit(coordinator, host_pool, Key("old", 0), NextTestAccessEpoch());
        HostPut(coordinator, host_pool, "new", 0);
        if (refresh_old) {
            HostHit(coordinator, host_pool, Key("old", 0), NextTestAccessEpoch());
        }

        CacheBlockRef replacement = coordinator.AcquireHostBlock(0, /*bucket=*/0);

        ASSERT_TRUE(replacement);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("old", 0)), refresh_old);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("new", 0)), !refresh_old);
    }
}

TEST(CacheCoordinatorHostReplacement, CrossGroupUsesMostRecentChildAccess) {
    for (const bool refresh_old : {false, true}) {
        SCOPED_TRACE(refresh_old);
        BlockPool device_pool(3, {2, 2, 1});
        BlockPool host_pool(2, {2, 2, 1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        };
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                            /*snapshot_pool=*/nullptr,
                            /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
        const std::int32_t old_parent = HostPut(coordinator, host_pool, "old-0", 0);
        HostPut(coordinator, host_pool, "old-1", 0);
        HostHit(coordinator, host_pool, Key("old-0", 0), NextTestAccessEpoch());
        const std::int32_t new_parent = HostPut(coordinator, host_pool, "new-0", 1);
        HostPut(coordinator, host_pool, "new-1", 1);
        if (refresh_old) {
            HostHit(coordinator, host_pool, Key("old-0", 0), NextTestAccessEpoch());
        }

        CacheBlockRef replacement = coordinator.AcquireHostBlock(2, /*bucket=*/0);

        ASSERT_TRUE(replacement);
        EXPECT_EQ(replacement->Location().lcm_block_id, refresh_old ? new_parent : old_parent);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("old-0", 0)), refresh_old);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("old-1", 0)), refresh_old);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("new-0", 1)), !refresh_old);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("new-1", 1)), !refresh_old);
    }
}

TEST(CacheCoordinatorHostReplacement, EqualEpochDoesNotProtectPreviouslyAcquiredEntry) {
    BlockPool device_pool(2, {2});
    BlockPool host_pool(1, {2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::uint64_t access_epoch = NextTestAccessEpoch();
    const CacheBlockLocation first = CacheBoundaryForGroup(coordinator, host_pool, "first", 0, access_epoch, -1);
    const CacheBlockLocation second = CacheBoundaryForGroup(coordinator, host_pool, "second", 0, access_epoch, -1);
    const bool first_is_earlier =
        std::pair{first.lcm_block_id, first.slot_index} < std::pair{second.lcm_block_id, second.slot_index};
    const CacheKey acquired = Key(first_is_earlier ? "first" : "second", 0);
    const CacheKey other = Key(first_is_earlier ? "second" : "first", 0);
    HostHit(coordinator, host_pool, acquired, access_epoch);

    CacheBlockRef replacement = coordinator.AcquireHostBlock(0, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(acquired));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(other));
}

TEST(CacheCoordinatorHostReplacement, GlobalLruAllowsStateCheckpointCapacityToGrow) {
    BlockPool device_pool(4, {2, 1});
    BlockPool host_pool(4, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    for (int i = 0; i < 6; ++i) {
        CacheBlockRef ref = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(ref, Key("old-" + std::to_string(i), 0), std::nullopt);
    }
    {
        CacheBlockRef ref = host_pool.AcquireBlock(1);
        coordinator.CacheHostBlock(ref, Key("new-0", 1), std::nullopt);
    }

    for (int i = 1; i <= 3; ++i) {
        CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);
        ASSERT_TRUE(replacement);
        coordinator.CacheHostBlock(replacement, Key("new-" + std::to_string(i), 1), std::nullopt);
    }

    for (int i = 0; i < 6; ++i) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-" + std::to_string(i), 0)));
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("new-" + std::to_string(i), 1)));
    }
}

TEST(CacheCoordinatorHostReplacement, GlobalLruComparesRefreshedParentsWithSameGroupChildren) {
    for (const bool refresh_other_group : {false, true}) {
        SCOPED_TRACE(refresh_other_group);
        BlockPool device_pool(2, {2, 1});
        BlockPool host_pool(2, {2, 1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        };
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                            /*snapshot_pool=*/nullptr,
                            /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
        const std::int32_t other_parent = HostPut(coordinator, host_pool, "other-0", 0);
        HostPut(coordinator, host_pool, "other-1", 0);
        const std::int32_t own_parent = HostPut(coordinator, host_pool, "own", 1);
        if (refresh_other_group) {
            HostHit(coordinator, host_pool, Key("other-0", 0), NextTestAccessEpoch());
        }

        CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);

        ASSERT_TRUE(replacement);
        EXPECT_EQ(replacement->Location().lcm_block_id, refresh_other_group ? own_parent : other_parent);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("own", 1)), !refresh_other_group);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("other-0", 0)), refresh_other_group);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key("other-1", 0)), refresh_other_group);
    }
}

TEST(CacheCoordinatorHostReplacement, GlobalLruProtectsAParentWithARecentSibling) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(2, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "cold-sibling", 0);
    const std::int32_t own_parent = HostPut(coordinator, host_pool, "own", 1);
    HostPut(coordinator, host_pool, "hot-sibling", 0);

    CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location().lcm_block_id, own_parent);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("own", 1)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("cold-sibling", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("hot-sibling", 0)));
}

TEST(CacheCoordinatorHostReplacement, GlobalLruPrefersOneChildWhenAccessEpochsTie) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(2, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::uint64_t epoch = NextTestAccessEpoch();
    CacheBoundaryForGroup(coordinator, host_pool, "other-0", 0, epoch, -1);
    CacheBoundaryForGroup(coordinator, host_pool, "other-1", 0, epoch, -1);
    const CacheBlockLocation own = CacheBoundaryForGroup(coordinator, host_pool, "own", 1, epoch, -1);

    CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), own);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("own", 1)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("other-0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("other-1", 0)));
}

TEST(CacheCoordinatorHostReplacement, GlobalLruSkipsPinnedParentsWithoutFallingBackToSameGroup) {
    BlockPool device_pool(3, {2, 1, 1});
    BlockPool host_pool(3, {2, 1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "pinned", 0);
    HostPut(coordinator, host_pool, "cold-sibling", 0);
    const std::int32_t middle_parent = HostPut(coordinator, host_pool, "middle", 2);
    HostPut(coordinator, host_pool, "newest", 1);
    CacheBlockRef pin = coordinator.FindHostCachedBlock(Key("pinned", 0));
    ASSERT_TRUE(pin);

    CacheBlockRef replacement = coordinator.AcquireHostBlock(1, /*bucket=*/0);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location().lcm_block_id, middle_parent);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("middle", 2)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("newest", 1)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("pinned", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("cold-sibling", 0)));
}

TEST(CacheCoordinatorHostReplacement, GlobalLruBatchSkipsReboundEntriesAndProtectsNewAllocations) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(2, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::int32_t older_parent = HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    const std::int32_t newer_parent = HostPut(coordinator, host_pool, "newer", 1);
    const std::array<std::uint32_t, 3> groups{1, 0, 1};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};

    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_TRUE(batch.blocks[0]);
    EXPECT_EQ(batch.blocks[0]->Location().lcm_block_id, older_parent);
    ASSERT_TRUE(batch.blocks[1]);
    EXPECT_EQ(batch.blocks[1]->Location().lcm_block_id, newer_parent);
    EXPECT_EQ(host_pool.BoundGroup(older_parent), 1u);
    EXPECT_EQ(host_pool.BoundGroup(newer_parent), 0u);
    EXPECT_FALSE(batch.blocks[2]);
    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
}

TEST(CacheCoordinatorHostReplacement, GlobalLruMixesChildAndParentVictimsAcrossBuckets) {
    BlockPool device_pool(3, {4, 4});
    BlockPool host_pool(3, {4, 4});
    const std::array specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 2, .shard_count = 2},
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                   /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                                   /*write_back_evictions=*/false);
    std::vector<CacheBlockRef> first_group = host_pool.AcquireBlocks(0, 8);
    std::vector<CacheBlockRef> second_group = host_pool.AcquireBlocks(1, 4);
    ASSERT_EQ(first_group.size(), 8u);
    ASSERT_EQ(second_group.size(), 4u);
    const CacheBlockLocation cold_child = first_group.front()->Location();
    const std::int32_t cold_parent = first_group.back()->Location().lcm_block_id;
    const std::int32_t middle_parent = second_group.front()->Location().lcm_block_id;
    for (std::size_t i = 0; i < first_group.size(); ++i) {
        const bool shares_cold_child = first_group[i]->Location().lcm_block_id == cold_child.lcm_block_id;
        const std::uint64_t epoch = shares_cold_child ? (i == 0 ? 1 : 100) : 10;
        coordinator.GroupPrefixIndex(0).Register(host_pool, first_group[i], Key("first-" + std::to_string(i), 0), epoch,
                                                 /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                                 /*newly_cached=*/nullptr);
    }
    for (std::size_t i = 0; i < second_group.size(); ++i) {
        coordinator.GroupPrefixIndex(1).Register(host_pool, second_group[i], Key("second-" + std::to_string(i), 1),
                                                 /*access_epoch=*/20, /*logical_block_index=*/-1,
                                                 CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
    }
    first_group.clear();
    second_group.clear();

    // Child eviction first, then whole-parent eviction opens holes in both
    // buckets. Later requests reuse those holes and skip rebound candidates.
    const std::array<std::uint32_t, 5> groups{0, 1, 1, 0, 0};
    const std::array<std::int32_t, 5> buckets{0, 1, 0, 1, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.blocks[0]->Location(), cold_child);
    EXPECT_EQ(batch.blocks[1]->Location().lcm_block_id, cold_parent);
    EXPECT_EQ(batch.blocks[2]->Location().lcm_block_id, cold_parent);
    EXPECT_EQ(batch.blocks[3]->Location().lcm_block_id, middle_parent);
    EXPECT_EQ(batch.blocks[4]->Location().lcm_block_id, middle_parent);
    for (std::size_t i = 0; i < groups.size(); ++i) {
        EXPECT_EQ(host_pool.BoundGroup(batch.blocks[i]->Location().lcm_block_id), groups[i]);
        EXPECT_EQ(coordinator.Allocator(static_cast<std::int32_t>(groups[i])).BucketOf(batch.blocks[i]->Location()),
                  buckets[i]);
    }
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("first-0", 0)));
    for (int i = 1; i < 4; ++i) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("first-" + std::to_string(i), 0)));
    }
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 3);
    EXPECT_EQ(batch.stats.group_scans, 2u);
    EXPECT_EQ(batch.stats.parent_candidates, 3u);
}

TEST(CacheCoordinatorHostReplacement, GlobalLruRefreshesParentScratchBetweenBatches) {
    for (const std::uint32_t first_group : {0u, 1u}) {
        SCOPED_TRACE(first_group);
        BlockPool device_pool(2, {1, 1});
        BlockPool host_pool(2, {1, 1});
        const std::array specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        };
        CacheCoordinator coordinator = MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                                       /*write_back_evictions=*/false);
        const std::int32_t oldest_parent = HostPut(coordinator, host_pool, "oldest", 0);
        HostPut(coordinator, host_pool, "other", 1);
        {
            CacheBlockRef replacement = coordinator.AcquireHostBlock(first_group, /*bucket=*/0);
            ASSERT_TRUE(replacement);
            ASSERT_EQ(replacement->Location().lcm_block_id, oldest_parent);
            coordinator.GroupPrefixIndex(static_cast<std::int32_t>(first_group))
                .Register(host_pool, replacement, Key("replacement", first_group), NextTestAccessEpoch(),
                          /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                          /*newly_cached=*/nullptr);
        }
        HostHit(coordinator, host_pool, Key("other", 1), NextTestAccessEpoch());

        CacheBlockRef replacement = coordinator.AcquireHostBlock(1u - first_group, /*bucket=*/0);

        ASSERT_TRUE(replacement);
        EXPECT_EQ(replacement->Location().lcm_block_id, oldest_parent);
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("replacement", first_group)));
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("other", 1)));
    }
}

TEST(CacheCoordinatorHostReplacement, GlobalLruComparesOnlyChildrenInTheRequestedBucket) {
    BlockPool device_pool(2, {2, 2});
    BlockPool host_pool(2, {2, 2});
    const std::array specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const auto put = [&](std::uint32_t group, std::int32_t bucket, const std::string& hash) {
        const std::array buckets{bucket};
        std::vector<CacheBlockRef> refs = host_pool.AcquireBlocksInBuckets(group, buckets);
        ASSERT_EQ(refs.size(), 1u);
        coordinator.CacheHostBlock(refs.front(), Key(hash, group), std::nullopt);
    };
    put(0, 0, "own-cold");
    put(1, 0, "other-0");
    put(1, 1, "other-1");
    put(0, 1, "own-hot");

    CacheBlockRef replacement = coordinator.AcquireHostBlock(0, /*bucket=*/1);

    ASSERT_TRUE(replacement);
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(replacement->Location()), 1);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("own-cold", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("own-hot", 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("other-0", 1)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("other-1", 1)));
}

TEST(CacheCoordinatorHostReplacement, SingleGroupBatchUsesChildrenWithoutParentCandidates) {
    BlockPool device_pool(4, {2});
    BlockPool host_pool(2, {2});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 2,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
    EXPECT_EQ(batch.stats.group_scans, 1u);
    EXPECT_EQ(batch.stats.parent_candidates, 0u);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator.HostEvictions().lru.device_resident, 0u);
}

TEST(CacheCoordinatorHostReplacement, BoundedHistoryScanPreservesGlobalLruAndRepeatedAllocation) {
    BlockPool device_pool(257, {1, 2});
    BlockPool host_pool(257, {1, 2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::int32_t donor_parent = HostPut(coordinator, host_pool, "donor-0", 1);
    HostPut(coordinator, host_pool, "donor-1", 1);
    for (std::int32_t i = 0; i < 256; ++i) {
        HostPut(coordinator, host_pool, "old-" + std::to_string(i), 0);
    }
    const std::array<std::uint32_t, 1> groups{0};
    const std::array<std::int32_t, 1> buckets{0};
    auto first = coordinator.AcquireHostBlocks(groups, buckets);
    ASSERT_TRUE(first.blocks[0]);
    EXPECT_EQ(first.blocks[0]->Location().lcm_block_id, donor_parent);
    EXPECT_EQ(first.stats.block_candidates, 3u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("donor-0", 1)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("donor-1", 1)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));

    auto second = coordinator.AcquireHostBlocks(groups, buckets);
    ASSERT_TRUE(second.blocks[0]);
    EXPECT_EQ(second.stats.block_candidates, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-1", 0)));

    auto third = coordinator.AcquireHostBlocks(groups, buckets);
    ASSERT_TRUE(third.blocks[0]);
    EXPECT_EQ(third.stats.block_candidates, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-1", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-2", 0)));
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 254);
}

TEST(CacheCoordinatorHostReplacement, BoundedHistoryScanPreservesBucketsEpochTiesAndPins) {
    BlockPool device_pool(2, {4});
    BlockPool host_pool(2, {4});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 4,
        .block_granularity = 2,
        .shard_count = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::array<std::uint64_t, 8> epochs{40, 10, 10, 10, 5, 20, 10, 30};
    std::vector<CacheBlockLocation> locations;
    CacheBlockRef pinned;
    for (std::int32_t i = 0; i < 8; ++i) {
        auto refs = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{i % 2});
        ASSERT_EQ(refs.size(), 1u);
        locations.push_back(refs.front()->Location());
        coordinator.GroupPrefixIndex(0).Register(host_pool, refs.front(), Key("old-" + std::to_string(i), 0), epochs[i],
                                                 /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                                                 /*newly_cached=*/nullptr);
        if (i == 4) pinned = refs.front();
    }
    const std::array<std::uint32_t, 3> groups{0, 0, 0};
    const std::array<std::int32_t, 3> buckets{1, 0, 1};
    auto batch = coordinator.AcquireHostBlocks(groups, buckets);
    const std::array<std::int32_t, 3> victim_indices{1, 2, 3};
    for (std::size_t i = 0; i < groups.size(); ++i) {
        ASSERT_TRUE(batch.blocks[i]);
        EXPECT_EQ(batch.blocks[i]->Location(), locations[victim_indices[i]]);
        EXPECT_EQ(coordinator.Allocator(0).BucketOf(batch.blocks[i]->Location()), buckets[i]);
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-" + std::to_string(victim_indices[i]), 0)));
    }
    for (const std::int32_t i : {0, 4, 5, 6, 7}) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-" + std::to_string(i), 0)));
    }
    EXPECT_EQ(batch.stats.block_candidates, 4u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 5);
}

TEST(CacheCoordinatorHostReplacement, BatchPreservesEmptyRefForPinnedShortfall) {
    BlockPool device_pool(2, {1});
    BlockPool host_pool(2, {1});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    CacheBlockRef pinned = host_pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(pinned);
    coordinator.CacheHostBlock(pinned, Key("pinned", 0), std::nullopt);

    const std::array<std::uint32_t, 2> groups{0, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_EQ(batch.blocks.size(), groups.size());
    EXPECT_TRUE(batch.blocks[0]);
    EXPECT_FALSE(batch.blocks[1]);
    EXPECT_EQ(batch.stats.requested, 2u);
    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(batch.stats.unallocated, 1u);
    EXPECT_EQ(batch.stats.group_scans, 1u);
    EXPECT_EQ(batch.stats.parent_candidates, 0u);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
}

TEST(CacheCoordinatorHostReplacement, BatchGivesScarceFreeCapacityToFirstCandidateGroup) {
    BlockPool device_pool(2, {1, 1});
    BlockPool host_pool(1, {1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    HostPut(coordinator, host_pool, "old-2", 0);
    HostPut(coordinator, host_pool, "old-3", 0);
    HostPut(coordinator, device_pool, "old-1", 0);

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
    EXPECT_EQ(batch.stats.group_scans, 2u);
    EXPECT_EQ(batch.stats.parent_candidates, 2u);
    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_EQ(batch.stats.unallocated, 0u);
    // Group 0 gave its older parent, old-0 and old-1, whole to group 1's
    // requests and then one entry to its own.
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("old-1", 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator.HostEvictions().lru.device_resident, 1u) << "the Device index still holds \"old-1\"";
}

TEST(CacheCoordinatorHostReplacement, CountsTheVictimsTheDeviceStillHolds) {
    BlockPool device_pool(2, {1});
    BlockPool host_pool(2, {1});
    const std::array specs{CacheGroupSpec{
        .attn_kind = AttnKind::kFull,
        .cache_blocks_per_lcm_block = 1,
        .block_granularity = 2,
    }};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "host-only", 0);
    HostPut(coordinator, host_pool, "resident", 0);
    HostPut(coordinator, device_pool, "resident", 0);

    const std::array<std::uint32_t, 2> groups{0, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 2u);
    EXPECT_EQ(coordinator.HostEvictions().lru.device_resident, 1u) << "the Device index still holds \"resident\"";
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(device_pool, Key("resident", 0)))
        << "the residency check leaves the Device entry alone";
}

TEST(CacheCoordinatorHostReplacement, FlushCountsHostEvictionsAndUndrainedStoreCandidates) {
    BlockPool device_pool(2, {1, 1});
    BlockPool host_pool(5, {1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "a", 0);
    HostPut(coordinator, host_pool, "b", 0);
    HostPut(coordinator, host_pool, "c", 1);
    HostPut(coordinator, host_pool, "e", 0);
    HostPut(coordinator, host_pool, "e", 1);
    HostPut(coordinator, device_pool, "d", 0);
    HostPut(coordinator, device_pool, "d", 1);
    coordinator.QueueCachedBlocksForStore(std::array<std::string, 1>{"d"});

    ASSERT_TRUE(coordinator.ClearDeviceCache());
    EXPECT_EQ(coordinator.StoreCandidatesDroppedByFlush(), 2u) << "one queued candidate per group, never drained";
    EXPECT_EQ(coordinator.HostEvictions().flush.blocks, 0u) << "the Device flush leaves Host alone";

    ASSERT_TRUE(coordinator.ClearCache());
    EXPECT_EQ(coordinator.StoreCandidatesDroppedByFlush(), 2u);
    EXPECT_EQ(coordinator.HostEvictions().flush.blocks, 5u) << "three Host entries in group 0, two in group 1";
    EXPECT_EQ(coordinator.HostEvictions().flush.device_resident, 0u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
}

TEST(CacheCoordinatorHostReplacement, BatchCrossGroupRebindingDoesNotRescanPlacementPoolPerVictim) {
    BlockPool device_pool(4, {2, 2});
    BlockPool host_pool(2, {2, 2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "old-0", 0);
    HostPut(coordinator, host_pool, "old-1", 0);
    HostPut(coordinator, host_pool, "old-2", 0);
    HostPut(coordinator, host_pool, "old-3", 0);

    const std::array<std::uint32_t, 3> groups{1, 1, 1};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.parent_candidates, 2u);
}

TEST(CacheCoordinatorHostReplacement, BatchCrossGroupRebindingChoosesLowestCacheValueParent) {
    BlockPool device_pool(3, {1, 1, 1});
    BlockPool host_pool(2, {1, 1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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

// The K3 shape on both tiers: P = 4, one Full group (four blocks per parent)
// and three State groups (one per parent).
const std::vector<CacheGroupSpec> kK3HostSpecs{
    {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
    {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
};

TEST(CacheCoordinatorHostReplacement, AHostStateEvictionTakesTheOtherStateGroupsAtItsHash) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(4, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    // A page's three checkpoints and then its KV fill the Host, so group 1's
    // checkpoint is the least recently used entry; the Device still holds
    // group 2's checkpoint.
    for (std::uint32_t group : {1u, 2u, 3u, 0u}) {
        HostPut(coordinator, host_pool, "h", group);
    }
    HostPut(coordinator, device_pool, "h", 2);

    const std::array<std::uint32_t, 1> groups{1};
    const std::array<std::int32_t, 1> buckets{0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    ASSERT_TRUE(batch.blocks[0]);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("h", group))) << "group " << group;
    }
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("h", 0))) << "a checkpoint takes no KV along";
    EXPECT_TRUE(coordinator.GroupPrefixIndex(2).Contains(device_pool, Key("h", 2))) << "the Device tier is untouched";
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 2) << "the siblings' parents are empty";
    // Every entry that left counts under the reason of the eviction that took
    // it.
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator.HostEvictions().lru.device_resident, 1u) << "the Device still holds group 2's";
}

// Two checkpoints fill the Host and a batch stores one block per State group:
// the older checkpoint's eviction frees room for all three, so the newer one
// stays whole.
TEST(CacheCoordinatorHostReplacement, StateCascadeDoesNotOverEvict) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(6, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    for (const char* hash : {"older", "newer"}) {
        for (std::uint32_t group = 1; group <= 3; ++group) {
            HostPut(coordinator, host_pool, hash, group);
        }
    }

    const std::array<std::uint32_t, 3> groups{1, 2, 3};
    const std::array<std::int32_t, 3> buckets{0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("older", group))) << "group " << group;
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("newer", group))) << "group " << group;
    }
    // Groups 2 and 3 take the room group 1's victim freed: only the older
    // checkpoint's three entries leave.
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 0);
}

// Two blocks of one State group: the first victim's cascade frees the room
// the second needs.
TEST(CacheCoordinatorHostReplacement, AStateCascadeFundsTheSameGroupsNextBlock) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(6, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    for (const char* hash : {"older", "newer"}) {
        for (std::uint32_t group = 1; group <= 3; ++group) {
            HostPut(coordinator, host_pool, hash, group);
        }
    }

    const std::array<std::uint32_t, 2> groups{1, 1};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_EQ(batch.stats.allocated, 2u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("older", group))) << "group " << group;
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("newer", group))) << "group " << group;
    }
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 1);
}

// A whole-parent eviction for one group takes a checkpoint whose sibling
// frees a parent of its own: a later group of the batch places there instead
// of taking the next least recently used parent, group 0's.
TEST(CacheCoordinatorHostReplacement, AParentEvictionsStateCascadeFundsALaterGroup) {
    BlockPool device_pool(4, {2, 2, 1, 1});
    BlockPool host_pool(3, {2, 2, 1, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::int32_t h_parent = HostPut(coordinator, host_pool, "h", 2);
    HostPut(coordinator, host_pool, "x0", 0);
    HostPut(coordinator, host_pool, "x1", 0);
    HostPut(coordinator, host_pool, "h", 3);

    // Group 0's first block takes group 2's older parent whole rather than
    // its own child; group 1 holds no entry.
    const std::array<std::uint32_t, 4> groups{0, 0, 1, 1};
    const std::array<std::int32_t, 4> buckets{0, 0, 0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.allocated, 4u);
    EXPECT_EQ(batch.stats.unallocated, 0u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("x0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("x1", 0)));
    EXPECT_EQ(batch.blocks[0]->Location().lcm_block_id, h_parent) << "group 2's parent went whole to group 0";
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 2u) << "h in group 2 and, by the cascade, in group 3";
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 2);
}

// A full K3 Host whose checkpoints are older than its KV, and a batch of one
// KV block and then one checkpoint block: the KV block's least recently used
// victim is a checkpoint's whole parent, whose cascade frees the room the
// checkpoint block takes, so no KV entry is evicted.
TEST(CacheCoordinatorHostReplacement, AKvBlocksCheckpointVictimFundsTheStateBlock) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(4, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::int32_t checkpoint_parent = HostPut(coordinator, host_pool, "h", 1);
    for (std::uint32_t group = 2; group <= 3; ++group) {
        HostPut(coordinator, host_pool, "h", group);
    }
    for (const char* hash : {"k0", "k1", "k2", "k3"}) {
        HostPut(coordinator, host_pool, hash, 0);
    }
    ASSERT_EQ(coordinator.NumFreeHostLcmBlocks(), 0);

    const std::array<std::uint32_t, 2> groups{0, 1};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.allocated, 2u);
    for (const char* hash : {"k0", "k1", "k2", "k3"}) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
    }
    EXPECT_EQ(batch.blocks[0]->Location().lcm_block_id, checkpoint_parent)
        << "group 1's checkpoint parent went whole to the KV block";
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u) << "the checkpoint and its two siblings";
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 1);
}

// Every KV entry pinned and the checkpoint block first: its own victim's
// cascade frees two parents, and the KV block, which would otherwise take a
// sibling's parent whole, places in one without a victim.
TEST(CacheCoordinatorHostReplacement, AStateCascadeFundsAKvBlockWithoutAVictim) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(4, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    std::vector<CacheBlockRef> pinned;
    for (const char* hash : {"k0", "k1", "k2", "k3"}) {
        HostPut(coordinator, host_pool, hash, 0);
        pinned.push_back(coordinator.GroupPrefixIndex(0).Find(host_pool, Key(hash, 0)));
        ASSERT_TRUE(pinned.back()) << hash;
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        HostPut(coordinator, host_pool, "h", group);
    }

    const std::array<std::uint32_t, 2> groups{1, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u) << "one checkpoint and its two siblings; the KV block none";
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 1);
}

// A checkpoint the cascade would take but something still pins stays: the
// first block takes the oldest checkpoint parent whole, whose cascade frees
// the room the second block takes instead of the older KV parent, and leaves
// the pinned sibling indexed.
TEST(CacheCoordinatorHostReplacement, AStateCascadeLeavesAPinnedSibling) {
    BlockPool device_pool(8, {4, 1, 1, 1});
    BlockPool host_pool(4, {4, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        kK3HostSpecs, 4, device_pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "h", 1);
    HostPut(coordinator, host_pool, "h", 2);
    HostPut(coordinator, host_pool, "k", 0);
    HostPut(coordinator, host_pool, "h", 3);
    CacheBlockRef pinned = coordinator.GroupPrefixIndex(1).Find(host_pool, Key("h", 1));
    ASSERT_TRUE(pinned);

    const std::array<std::uint32_t, 2> groups{1, 2};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("h", 1))) << "a pinned sibling stays";
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("h", 2)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("h", 3)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("k", 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 2u) << "h in group 2 and, by the cascade, in group 3";
    EXPECT_EQ(coordinator.NumFreeHostLcmBlocks(), 0);
}

// A cascade that takes some of a candidate parent's entries leaves it a
// valid candidate: a later block evicts the rest and places there.
TEST(CacheCoordinatorHostReplacement, AParentACascadePartlyEvictedStillGoesWhole) {
    BlockPool device_pool(4, {2, 1, 2});
    BlockPool host_pool(2, {2, 1, 2});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 2, .block_granularity = 0},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    HostPut(coordinator, host_pool, "a", 1);
    const std::int32_t shared_parent = HostPut(coordinator, host_pool, "a", 2);
    ASSERT_EQ(HostPut(coordinator, host_pool, "b", 2), shared_parent);

    // Group 1's own checkpoint ties with its parent and goes alone, taking
    // group 2's "a" from the shared parent; the Full block then takes what is
    // left of that parent.
    const std::array<std::uint32_t, 2> groups{1, 0};
    const std::array<std::int32_t, 2> buckets{0, 0};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);

    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& ref) { return static_cast<bool>(ref); }));
    EXPECT_EQ(batch.blocks[1]->Location().lcm_block_id, shared_parent);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u) << "a in groups 1 and 2, then b";

    // The same Host again, with group 1's block alone, shows the parent the
    // Full block took was partly evicted: group 1's checkpoint takes group 2's
    // "a" from it and leaves "b".
    batch.blocks.clear();
    HostPut(coordinator, host_pool, "a", 1);
    const std::int32_t refilled_parent = HostPut(coordinator, host_pool, "a", 2);
    ASSERT_EQ(HostPut(coordinator, host_pool, "b", 2), refilled_parent);
    const std::array<std::uint32_t, 1> checkpoint_group{1};
    const std::array<std::int32_t, 1> checkpoint_bucket{0};
    batch = coordinator.AcquireHostBlocks(checkpoint_group, checkpoint_bucket);
    ASSERT_TRUE(batch.blocks[0]);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("a", 2))) << "the cascade took it";
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("b", 2)));
}

TEST(CacheCoordinatorHostReplacement, BatchDoesNotRebindPinnedParent) {
    BlockPool device_pool(2, {2, 1});
    BlockPool host_pool(1, {2, 1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, device_pool, /*enable_l3_storage=*/false, &host_pool,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
    EXPECT_EQ(batch.stats.parent_candidates, 1u);
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
    CacheCoordinator coordinator = MakeCoordinator(
        specs, /*prefix_granularity=*/2, pool, /*enable_l3_storage=*/false, &host_pool,
        /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes = ContentHashes({{0, 0}, {1, 1}, {2, 2}});
    for (const std::string& hash : hashes) {
        (void)HostPut(coordinator, host_pool, hash, /*gid=*/0);
    }

    std::vector<BlockTable> tables(coordinator.NumGroups());
    const std::array<std::int32_t, 2> tokens{0, 0};
    std::vector<GroupDemand> demands = FreshDemands(tables, tokens);
    const std::optional<CacheCoordinator::AdmissionResult> result =
        coordinator.Admit(coordinator.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt,
                          /*evict_blocks_past_segments=*/false);

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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
                                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                             /*write_back_evictions=*/false);
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
// The Host-suffix rule (a hit is a Device KV prefix plus a Host suffix) on the
// K3 shape: P = 4, one Full group (one block per page, four per parent) and
// three State groups (one per parent). A hit takes the Device's KV prefix, then
// a Host suffix, and each State group's checkpoint from whichever tier holds
// it, the Device first. Layouts are registered straight into each tier's index;
// the prompt is 40 tokens, of which nine pages are hashed and 37 may be
// matched. The parameter turns L3 on, which must not change any local match.
// ---------------------------------------------------------------------------

class HostSlidingSuffixTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;

    void SetUp() override {
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < 4; ++page) {
            pages.emplace_back(tokens_.begin() + page * kP, tokens_.begin() + (page + 1) * kP);
        }
        hashes_ = ContentHashes(pages);
        std::vector<BlockTable> tables(3);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, 16));
        CacheCompletedBlocksForTest(coordinator_, tables, hashes_,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/16,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
        coordinator_.Free(tables);
        for (std::uint32_t group = 0; group < 3; ++group) {
            for (std::size_t page = group == 0 ? 0 : 3; page < 4; ++page) {
                CacheBlockRef copy = host_.AcquireBlock(group);
                coordinator_.CacheHostBlock(copy, Key(hashes_[page], group), std::nullopt);
            }
            if (group != 0) {
                const auto location =
                    CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, group, Key(hashes_.back(), group));
                ASSERT_TRUE(location);
                ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, group, *location));
            }
        }
    }

    void HoldHostCapacity() {
        held_ = host_.AcquireBlocks(0, host_.NumFreeSlots(0));
        for (CacheBlockRef& block : host_.AcquireBlocks(1, host_.NumEmptyLcmBlocks())) {
            held_.push_back(std::move(block));
        }
    }

    CacheCoordinator::PromptPrefix Prompt() const {
        return {.prefix_hashes = hashes_, .tokens = tokens_, .max_tokens = 16};
    }

    BlockPool device_{32, {4, 1, 4}};
    BlockPool host_{32, {4, 1, 4}};
    CacheCoordinator coordinator_{MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            {.attn_kind = AttnKind::kSlidingWindow,
             .sliding_window = 3,
             .cache_blocks_per_lcm_block = 4,
             .block_granularity = kP}},
        kP, device_, /*enable_l3_storage=*/false, &host_, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true)};
    const std::vector<std::int32_t> tokens_{MakeTokens(16)};
    std::vector<std::string> hashes_;
    std::vector<CacheBlockRef> held_;
};

// A cached Device state can outlive its Device window while pinned by a
// request. The Host window remains useful to a later joint prefix match.
class DeviceStateHostWindowPressureTest : public HostSlidingSuffixTest {
protected:
    void SetUp() override {
        const auto publish = [&](const std::vector<std::int32_t>& tokens) {
            std::vector<std::vector<std::int32_t>> pages;
            for (int page = 0; page < 4; ++page)
                pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
            const auto hashes = ContentHashes(pages);
            std::vector<BlockTable> tables(3);
            EXPECT_TRUE(AdmitForTest(coordinator_, tables, 16));
            CacheCompletedBlocksForTest(coordinator_, tables, hashes,
                                        CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_), 0, 16,
                                        CacheBoundaryKind::kEndpoint, false, true);
            coordinator_.Free(tables);
            return hashes;
        };
        const auto copy = [&](const CacheKey& key) {
            auto block = host_.AcquireBlock(key.group_id);
            coordinator_.CacheHostBlock(block, key, std::nullopt);
        };
        const auto evict = [&](const CacheKey& key) {
            const auto location = CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, key.group_id, key);
            ASSERT_TRUE(location);
            ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, key.group_id, *location));
        };
        const auto older = publish(MakeTokens(16, 100));
        older_checkpoint_ = older.back();
        copy(Key(older_checkpoint_, 1));
        evict(Key(older_checkpoint_, 1));
        hashes_ = publish(tokens_);
        for (const auto& hash : hashes_) copy(Key(hash, 0));
        copy(Key(hashes_.back(), 2));
        device_checkpoint_pin_ = coordinator_.GroupPrefixIndex(1).Find(device_, Key(hashes_.back(), 1));
        ASSERT_TRUE(device_checkpoint_pin_);
        evict(Key(hashes_.back(), 2));
    }
    std::string older_checkpoint_;
    CacheBlockRef device_checkpoint_pin_{};
};
TEST_F(DeviceStateHostWindowPressureTest, HostWindowNeededByDeviceStateIsNotDead) {
    auto before = coordinator_.ProbePromptPrefix(Prompt());
    const auto before_hit = std::max(before.device.num_common_tokens, before.host.num_common_tokens);
    ASSERT_EQ(before_hit, 16);
    ASSERT_TRUE(coordinator_.GroupPrefixIndex(1).Contains(device_, Key(hashes_.back(), 1)));
    ASSERT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 1)));
    HoldHostCapacity();
    const std::array<std::uint32_t, 1> groups{1};
    const std::array<std::int32_t, 1> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const auto allocated = coordinator_.AcquireHostSegments(groups, buckets, segments);
    ASSERT_EQ(allocated.stats.allocated, 1u);
    auto after = coordinator_.ProbePromptPrefix(Prompt());
    const auto hit = std::max(after.device.num_common_tokens, after.host.num_common_tokens);
    EXPECT_EQ(hit, 16);
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(1).Contains(device_, Key(hashes_.back(), 1)));
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 2)));
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_checkpoint_, 1)));
}

TEST(JointStateHostPressureTest, StateAcrossTiersIsNotDeadHostState) {
    BlockPool device(32, {4, 1, 1, 1});
    BlockPool host(32, {4, 1, 1, 1});
    auto coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 4},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        4, device, /*enable_l3_storage=*/false, &host, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const auto key = [](const std::string& hash, std::uint32_t group) {
        return CacheKey{.group_id = group, .content_hash = hash};
    };
    const auto publish = [&](std::int32_t start) {
        const auto tokens = MakeTokens(16, start);
        std::vector<std::span<const std::int32_t>> pages;
        for (int i = 0; i < 4; ++i) pages.emplace_back(tokens.data() + i * 4, 4);
        const auto hashes = ComputePrefixHashes(pages, "");
        std::vector<BlockTable> tables(4);
        EXPECT_TRUE(AdmitForTest(coordinator, tables, 16));
        CacheCompletedBlocksForTest(coordinator, tables, hashes,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator), 0, 16,
                                    CacheBoundaryKind::kEndpoint, false, true);
        coordinator.Free(tables);
        return hashes;
    };
    const auto copy = [&](const CacheKey& cache_key) {
        auto block = host.AcquireBlock(cache_key.group_id);
        coordinator.CacheHostBlock(block, cache_key, std::nullopt);
    };
    const auto evict = [&](const CacheKey& cache_key) {
        const auto location = CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, cache_key.group_id, cache_key);
        ASSERT_TRUE(location);
        ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, cache_key.group_id, *location));
    };
    const auto older = publish(100);
    for (const auto group : {1u, 2u, 3u}) copy(key(older.back(), group));
    evict(key(older.back(), 1));
    const auto hashes = publish(1);
    copy(key(hashes.back(), 2));
    copy(key(hashes.back(), 3));
    // A request pins one State group while its other groups are demoted to
    // the Host. The same logical checkpoint remains complete across both tiers.
    auto pin = coordinator.GroupPrefixIndex(1).Find(device, key(hashes.back(), 1));
    ASSERT_TRUE(pin);
    evict(key(hashes.back(), 2));
    const auto tokens = MakeTokens(17);
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes, .tokens = tokens, .max_tokens = 16};
    const auto before = coordinator.ProbePromptPrefix(prompt);
    ASSERT_EQ(std::max(before.device.num_common_tokens, before.host.num_common_tokens), 16);
    auto held = host.AcquireBlocks(0, host.NumFreeSlots(0));
    for (auto& block : host.AcquireBlocks(1, host.NumEmptyLcmBlocks())) held.push_back(std::move(block));
    const std::array<std::uint32_t, 1> groups{1};
    const std::array<std::int32_t, 1> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const auto allocated = coordinator.AcquireHostSegments(groups, buckets, segments);
    ASSERT_EQ(allocated.stats.allocated, 1u);
    const auto after = coordinator.ProbePromptPrefix(prompt);
    const auto hit = std::max(after.device.num_common_tokens, after.host.num_common_tokens);
    EXPECT_EQ(hit, 16);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key(hashes.back(), 2)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(key(hashes.back(), 3)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key(older.back(), 1)));
}

TEST_F(HostSlidingSuffixTest, FullDeviceKvResumesWithAHostWindowAndCheckpoint) {
    auto probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device_only_hit_tokens, 0);
    EXPECT_EQ(probe.device.full_page_tokens, 12);
    EXPECT_EQ(probe.host.num_common_tokens, 16);
    std::vector<BlockTable> tables(3);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->device_prefix_tokens, 12);
    EXPECT_EQ(result->host_prefix_tokens, 16);
    ASSERT_EQ(result->load_pairs.size(), 3u) << "one Full page, one checkpoint, one window block";
    std::array<std::int32_t, 3> loads{};
    for (const auto& pair : result->load_pairs) {
        ++loads[pair.group_id];
    }
    EXPECT_EQ(loads, (std::array<std::int32_t, 3>{1, 1, 1}));
    coordinator_.Free(tables);
}

class HostSuffixHitTest : public ::testing::TestWithParam<bool> {
protected:
    static constexpr std::int32_t kP = 4;
    static constexpr std::int32_t kMaxTokens = 37;

    HostSuffixHitTest()
        : coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
              kP, device_, /*enable_l3_storage=*/GetParam(), &host_, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false)} {
        std::vector<std::span<const std::int32_t>> pages;
        for (std::size_t start = 0; start + kP <= 36; start += kP) {
            pages.emplace_back(tokens_.data() + start, kP);
        }
        hashes_ = ComputePrefixHashes(pages, "");
    }

    CacheCoordinator::PromptPrefix Prompt() const {
        return {.prefix_hashes = hashes_, .tokens = tokens_, .max_tokens = kMaxTokens};
    }
    // The prompt's hash at an unaligned boundary.
    ContentHash TailHash(std::int32_t tokens) const {
        const std::int32_t page_tokens = tokens - tokens % kP;
        return HashPrefixPage(std::span<const std::int32_t>(tokens_).subspan(static_cast<std::size_t>(page_tokens),
                                                                             static_cast<std::size_t>(tokens % kP)),
                              hashes_[static_cast<std::size_t>(page_tokens / kP) - 1]);
    }
    CacheBlockLocation Put(BlockPool& pool, const std::string& hash, std::uint32_t group) {
        return CacheBoundaryForGroup(coordinator_, pool, hash, group,
                                     CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                     /*logical_block_index=*/-1);
    }

    BlockPool device_{64, {4, 1, 1, 1}};
    BlockPool host_{64, {4, 1, 1, 1}};
    CacheCoordinator coordinator_;
    const std::vector<std::int32_t> tokens_ = MakeTokens(40);
    std::vector<std::string> hashes_;
};

INSTANTIATE_TEST_SUITE_P(L3OffAndOn, HostSuffixHitTest, ::testing::Bool());

// A checkpoint on the Device's last full page, aligned: the Device keeps the KV
// of pages h0-h3, the Host the three checkpoints at 16. The hit is 16 with no
// KV loaded; a Host copy of the same pages changes nothing.
TEST_P(HostSuffixHitTest, ADeviceKvPrefixResumesAtAHostCheckpointOnItsLastPage) {
    std::vector<CacheBlockLocation> device_pages;
    for (std::size_t page = 0; page < 4; ++page) {
        device_pages.push_back(Put(device_, hashes_[page], 0));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(host_, hashes_[3], group);
    }
    for (const bool host_duplicates : {false, true}) {
        if (host_duplicates) {
            for (std::size_t page = 0; page < 4; ++page) {
                Put(host_, hashes_[page], 0);
            }
        }
        CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
        EXPECT_EQ(probe.device_only_hit_tokens, 0) << "no checkpoint on the Device";
        EXPECT_EQ(probe.device.num_common_tokens, 16);
        EXPECT_EQ(probe.device.full_page_tokens, 16);
        EXPECT_EQ(probe.host.num_common_tokens, 16);
        EXPECT_EQ(probe.host.full_page_tokens, 16);
        std::vector<BlockTable> tables(4);
        const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(result);
        EXPECT_EQ(result->device_prefix_tokens, 16);
        EXPECT_EQ(result->host_prefix_tokens, 16);
        EXPECT_EQ(result->junction_tokens, 0);
        ASSERT_EQ(result->load_pairs.size(), 3u) << "host_duplicates=" << host_duplicates;
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(result->load_pairs[i].group_id, i + 1) << "only the checkpoints are loaded";
        }
        ASSERT_EQ(tables[0].NumBlocks(), 5);
        for (std::size_t page = 0; page < 4; ++page) {
            EXPECT_EQ(tables[0].Blocks()[page]->Location(), device_pages[page]) << "the Device's own KV page";
        }
    }
}

// A Host tail -- the KV tail and its checkpoints -- continues the Device's full
// pages: one tail block and three checkpoints load, and the tail lands on a
// fresh block with two tokens of room that the growth fills.
TEST_P(HostSuffixHitTest, AHostTailWithItsCheckpointsResumesAfterTheDevicePages) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    const ContentHash tail = TailHash(18);
    for (std::uint32_t group = 0; group <= 3; ++group) {
        Put(host_, tail, group);
    }
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 2, .hash = tail});
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_FALSE(probe.device.tail);
    EXPECT_EQ(probe.host.num_common_tokens, 18);
    EXPECT_EQ(probe.host.full_page_tokens, 16);
    ASSERT_TRUE(probe.host.tail);
    EXPECT_EQ(probe.host.tail->tail_tokens, 2);
    EXPECT_EQ(probe.host.kv_tokens, 18) << "the Host's KV tail";
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{2}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->load_pairs.size(), 4u);
    EXPECT_EQ(std::ranges::count(result->load_pairs, 0u, &BlockTransfer::group_id), 1);
    EXPECT_TRUE(result->pages_to_copy[0].empty());
    EXPECT_EQ(tables[0].NumBlocks(), 5) << "four claimed pages and the tail's destination";
    EXPECT_EQ(tables[0].AvailableTokens(), 0) << "the two tokens of growth fill the tail's block";
}

// The Device holds the KV tail at 18, the Host only its checkpoints: the
// hitter copies the Device's tail block and loads the three checkpoints.
TEST_P(HostSuffixHitTest, ADeviceKvTailIsCopiedWhenOnlyItsCheckpointIsOnHost) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    const ContentHash tail = TailHash(18);
    const CacheBlockLocation device_tail = Put(device_, tail, 0);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(host_, tail, group);
    }
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 2, .hash = tail});
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 18);
    EXPECT_EQ(probe.device.full_page_tokens, 16);
    ASSERT_TRUE(probe.device.tail);
    EXPECT_EQ(probe.device.tail->group_keys[0].size(), 1u);
    EXPECT_TRUE(probe.device.tail->group_keys[1].empty());
    EXPECT_EQ(probe.host.num_common_tokens, 18);
    ASSERT_TRUE(probe.host.tail);
    EXPECT_TRUE(probe.host.tail->group_keys[0].empty()) << "the KV tail comes whole from the Device";
    EXPECT_EQ(probe.host.kv_tokens, 0) << "the Host adds no KV";
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{2}});
    ASSERT_TRUE(result);
    ASSERT_EQ(result->pages_to_copy[0].size(), 1u);
    EXPECT_EQ(result->pages_to_copy[0][0].first, coordinator_.Allocator(0).ResolveDeviceCacheBlockId(device_tail));
    EXPECT_EQ(result->load_pairs.size(), 3u);
    EXPECT_EQ(std::ranges::count(result->load_pairs, 0u, &BlockTransfer::group_id), 0);
}

// Two tails in one page, as a prompt end and the finish after a short output
// publish them: the Device holds the tail at 18 with all its checkpoints, the
// Host (write-through) that one and the checkpoint boundary at 19. The Host
// suffix is probed above the Device-only hit 18, inside the page it ends in, so
// 19 resumes: the Host's tail block and three checkpoints load.
TEST_P(HostSuffixHitTest, AHostTailAboveADeviceTailInTheSamePageResumes) {
    std::vector<CacheBlockLocation> device_pages;
    for (std::size_t page = 0; page < 4; ++page) {
        device_pages.push_back(Put(device_, hashes_[page], 0));
    }
    const ContentHash short_tail = TailHash(18);
    const ContentHash long_tail = TailHash(19);
    for (std::uint32_t group = 0; group <= 3; ++group) {
        Put(device_, short_tail, group);
        Put(host_, short_tail, group);
        Put(host_, long_tail, group);
    }
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 2, .hash = short_tail});
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 3, .hash = long_tail});
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device_only_hit_tokens, 18);
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.full_page_tokens, 16);
    EXPECT_FALSE(probe.device.tail) << "the Device's tail at 18 gives way";
    EXPECT_EQ(probe.host.num_common_tokens, 19);
    EXPECT_EQ(probe.host.full_page_tokens, 16);
    ASSERT_TRUE(probe.host.tail);
    EXPECT_EQ(probe.host.tail->tail_tokens, 3);
    EXPECT_EQ(probe.host.kv_tokens, 19);
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->host_prefix_tokens, 19);
    EXPECT_EQ(result->junction_tokens, 0);
    EXPECT_EQ(result->load_pairs.size(), 4u);
    EXPECT_EQ(std::ranges::count(result->load_pairs, 0u, &BlockTransfer::group_id), 1);
    EXPECT_TRUE(result->pages_to_copy[0].empty());
    ASSERT_EQ(tables[0].NumBlocks(), 5) << "four claimed pages and the tail's destination";
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(tables[0].Blocks()[page]->Location(), device_pages[page]);
    }
}

// The same two tails for the storage query: the Device holds the checkpoint
// boundary at 18, the Host that one and only the KV block of the checkpoint
// boundary at 19. The query and the reservation reach 19 above the Device-only
// hit and ask for the three checkpoints alone.
TEST_P(HostSuffixHitTest, AStorageQueryAboveADeviceTailReachesALongerTailInTheSamePage) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    const ContentHash short_tail = TailHash(18);
    const ContentHash long_tail = TailHash(19);
    for (std::uint32_t group = 0; group <= 3; ++group) {
        Put(device_, short_tail, group);
        Put(host_, short_tail, group);
    }
    Put(host_, long_tail, 0);
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 2, .hash = short_tail});
    coordinator_.CacheHostTail(hashes_[3], TailPublication{.tail_tokens = 3, .hash = long_tail});
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes_, .tokens = tokens_, .max_tokens = 19};
    EXPECT_EQ(coordinator_.ProbePromptPrefix(prompt).host.num_common_tokens, 18);
    const std::vector<CacheKey> checkpoints{Key(long_tail, 1), Key(long_tail, 2), Key(long_tail, 3)};
    const CacheCoordinator::StorageQuery query = coordinator_.MissingStorageKeys(prompt, /*max_keys=*/64);
    EXPECT_EQ(query.max_tokens, 19);
    EXPECT_EQ(query.keys, checkpoints);
    const CacheCoordinator::StorageReservation reservation =
        coordinator_.ReserveStoragePrefix(prompt, query.keys, /*max_blocks=*/64);
    EXPECT_EQ(reservation.keys, checkpoints);
    EXPECT_EQ(reservation.blocks.size(), 3u);
    EXPECT_EQ(reservation.tail_prior, hashes_[3]);
    ASSERT_TRUE(reservation.tail);
    EXPECT_EQ(reservation.tail->tail_tokens, 3);
    EXPECT_EQ(reservation.tail->hash, long_tail);
}

// A shorter checkpoint under longer Device KV: the Device's KV reaches 32, the
// only checkpoint is the Host's at 16. The hit stops at 16, claims the four
// Device pages below it, and the first chunk is cut at the KV boundary 32.
TEST_P(HostSuffixHitTest, AShorterHostCheckpointResumesUnderLongerDeviceKv) {
    std::vector<CacheBlockLocation> device_pages;
    for (std::size_t page = 0; page < 8; ++page) {
        device_pages.push_back(Put(device_, hashes_[page], 0));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(host_, hashes_[3], group);
    }
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.full_page_tokens, 16);
    EXPECT_EQ(probe.device.kv_tokens, 32);
    EXPECT_EQ(probe.host.num_common_tokens, 16);
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->junction_tokens, 32);
    EXPECT_EQ(result->load_pairs.size(), 3u);
    ASSERT_EQ(tables[0].NumBlocks(), 5);
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(tables[0].Blocks()[page]->Location(), device_pages[page]);
    }
}

// KV on Host alone, no checkpoint on either tier: nothing resumes, and the
// first chunk is cut at the Host's KV reach 32.
TEST_P(HostSuffixHitTest, HostOnlyKvSetsTheJunctionAtItsReach) {
    for (std::size_t page = 0; page < 8; ++page) {
        Put(host_, hashes_[page], 0);
    }
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.device.kv_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 0) << "the Device alone resumes furthest";
    EXPECT_EQ(probe.host.kv_tokens, 32);
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->host_prefix_tokens, 0);
    EXPECT_EQ(result->junction_tokens, 32);
    EXPECT_TRUE(result->load_pairs.empty());
}

// The Device's KV h0-h3 continued by the Host's h4-h7, no checkpoint: the
// junction is the two tiers' KV reach 32, not the Device's 16.
TEST_P(HostSuffixHitTest, HostKvPastTheDeviceKvSetsTheJunctionAtTheHostReach) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    for (std::size_t page = 4; page < 8; ++page) {
        Put(host_, hashes_[page], 0);
    }
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.device.kv_tokens, 16);
    EXPECT_EQ(probe.host.num_common_tokens, 0) << "the Device alone resumes furthest";
    EXPECT_EQ(probe.host.kv_tokens, 32);
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->junction_tokens, 32);
    EXPECT_TRUE(result->load_pairs.empty());
}

// The Device's checkpoint is at 8 but its KV reaches 16; the Host continues
// the KV with h4, h5 and holds the checkpoint at 24. The Host suffix starts
// at the Device's KV coverage, not at its checkpoint: h2, h3 are not loaded.
TEST_P(HostSuffixHitTest, AHostSuffixContinuesTheDevicePrefixAboveAShorterDeviceCheckpoint) {
    std::vector<CacheBlockLocation> device_pages;
    for (std::size_t page = 0; page < 4; ++page) {
        device_pages.push_back(Put(device_, hashes_[page], 0));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(device_, hashes_[1], group);
        Put(host_, hashes_[5], group);
    }
    Put(host_, hashes_[4], 0);
    Put(host_, hashes_[5], 0);
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device_only_hit_tokens, 8);
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.full_page_tokens, 16) << "the Device's KV coverage, not its checkpoint";
    EXPECT_EQ(probe.host.num_common_tokens, 24);
    EXPECT_EQ(probe.host.full_page_tokens, 24);
    EXPECT_EQ(probe.host.per_group[0].hits, (std::vector<std::uint8_t>{1, 1}));
    EXPECT_EQ(probe.host.kv_tokens, 24);
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->device_prefix_tokens, 16);
    EXPECT_EQ(result->host_prefix_tokens, 24);
    EXPECT_EQ(result->load_pairs.size(), 5u) << "h4, h5 and three checkpoints";
    EXPECT_EQ(std::ranges::count(result->load_pairs, 0u, &BlockTransfer::group_id), 2);
    ASSERT_EQ(tables[0].NumBlocks(), 7);
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(tables[0].Blocks()[page]->Location(), device_pages[page]) << "h0-h3 are claimed, not loaded";
    }
}

// A gap stops a tier's run, so the hit is the longest contiguous Device
// prefix plus Host suffix. (a) The Host lacks h5: its checkpoint at 24 is
// unreachable and the Device's own hit at 8 stands, with nothing loaded; (b)
// a Host checkpoint at 20 is reachable through h4. (c) After a flush the
// Device lacks h2: its h3 beyond the gap is never stitched in, so the Host's
// checkpoint at 16 needs the Host's own h2 and h3.
TEST_P(HostSuffixHitTest, AGapDegradesToTheLongestContiguousRun) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(device_, hashes_[1], group);
        Put(host_, hashes_[5], group);
    }
    Put(host_, hashes_[4], 0);
    {
        CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
        EXPECT_EQ(probe.device.num_common_tokens, 8);
        EXPECT_EQ(probe.host.num_common_tokens, 8) << "the Device alone resumes furthest";
        std::vector<BlockTable> tables(4);
        const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(result);
        EXPECT_TRUE(result->load_pairs.empty());
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(host_, hashes_[4], group);
    }
    {
        CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
        EXPECT_EQ(probe.device.full_page_tokens, 16);
        EXPECT_EQ(probe.host.num_common_tokens, 20);
        std::vector<BlockTable> tables(4);
        const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(result);
        EXPECT_EQ(result->load_pairs.size(), 4u) << "h4 and three checkpoints";
    }

    ASSERT_TRUE(coordinator_.ClearCache());
    for (const std::size_t page : {0, 1, 3}) {
        Put(device_, hashes_[page], 0);
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        Put(device_, hashes_[1], group);
        Put(host_, hashes_[3], group);
    }
    Put(host_, hashes_[3], 0);
    EXPECT_EQ(coordinator_.ProbePromptPrefix(Prompt()).host.num_common_tokens, 8) << "the Host lacks h2";
    Put(host_, hashes_[2], 0);
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.full_page_tokens, 8);
    EXPECT_EQ(probe.host.num_common_tokens, 16);
    EXPECT_EQ(probe.host.per_group[0].hits, (std::vector<std::uint8_t>{1, 1})) << "h2 and h3 from the Host";
}

// A checkpoint is matched only when every State group holds it on some tier:
// two of three on Host, or one on each tier and none for the third, is no
// boundary at all.
TEST_P(HostSuffixHitTest, APartialHostCheckpointIsNeverMatched) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    Put(host_, hashes_[3], 1);
    Put(host_, hashes_[3], 2);
    for (const bool device_has_first : {false, true}) {
        if (device_has_first) {
            Put(device_, hashes_[3], 1);
        }
        CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
        EXPECT_EQ(probe.device.num_common_tokens, 0) << "device_has_first=" << device_has_first;
        EXPECT_EQ(probe.host.num_common_tokens, 0) << "device_has_first=" << device_has_first;
        std::vector<BlockTable> tables(4);
        const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(result);
        EXPECT_TRUE(result->load_pairs.empty());
    }
}

// Each State group takes its checkpoint from the Device when it holds it and
// from the Host otherwise: the first group claims its Device block as slot
// 0, the other two load theirs.
TEST_P(HostSuffixHitTest, EachStateGroupMayComeFromEitherTier) {
    for (std::size_t page = 0; page < 4; ++page) {
        Put(device_, hashes_[page], 0);
    }
    const CacheBlockLocation device_state = Put(device_, hashes_[3], 1);
    Put(host_, hashes_[3], 1);
    Put(host_, hashes_[3], 2);
    Put(host_, hashes_[3], 3);
    CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(probe.device.num_common_tokens, 16);
    EXPECT_EQ(probe.device.per_group[1].hits, (std::vector<std::uint8_t>{0, 0, 0, 1})) << "from page 0";
    EXPECT_TRUE(probe.host.per_group[1].hits.empty()) << "the Device's checkpoint comes first";
    EXPECT_EQ(probe.host.per_group[2].hits, (std::vector<std::uint8_t>{0, 0, 0, 1})) << "from page 0";
    std::vector<BlockTable> tables(4);
    const auto result = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    ASSERT_EQ(result->load_pairs.size(), 2u);
    EXPECT_EQ(result->load_pairs[0].group_id, 2u);
    EXPECT_EQ(result->load_pairs[1].group_id, 3u);
    EXPECT_EQ(tables[1].Blocks()[0]->Location(), device_state);
    EXPECT_EQ(tables[2].Blocks()[0], result->load_pairs[0].destination);
    EXPECT_EQ(tables[3].Blocks()[0], result->load_pairs[1].destination);
}

// Full + SWA (W = 8, two blocks of lookback) + State. The Device holds the
// Full pages h0-h3 and the window blocks h2, h3 behind 16; the checkpoint at
// 12 is on Host. (a) The window behind 12 needs block 1, which only the Host
// holds, so the split drops to 4 and the Host supplies h1, h2 in both groups
// -- a hit the probe from the Device's full pages (0) misses for want of a
// Host h0. (b) Once the Device holds block 1 the split stays at 12 and only
// the checkpoint loads.
TEST(HostSuffixHitSlidingWindowTest, ASlidingWindowCapLowersTheSplit) {
    BlockPool device(64, {1, 1, 1});
    BlockPool host(64, {1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
            {.attn_kind = AttnKind::kSlidingWindow,
             .sliding_window = 8,
             .cache_blocks_per_lcm_block = 1,
             .block_granularity = 4},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        /*prefix_granularity=*/4, device, /*enable_l3_storage=*/false, &host, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const std::vector<std::string> hashes =
        ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}});
    for (std::size_t page = 0; page < 4; ++page) {
        CacheForGroup(coordinator, device, hashes[page], 0);
    }
    CacheForGroup(coordinator, device, hashes[2], 1);
    CacheForGroup(coordinator, device, hashes[3], 1);
    for (std::uint32_t group = 0; group <= 1; ++group) {
        CacheForGroup(coordinator, host, hashes[1], group);
        CacheForGroup(coordinator, host, hashes[2], group);
    }
    CacheForGroup(coordinator, host, hashes[2], 2);
    {
        CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
        EXPECT_EQ(probe.device_only_hit_tokens, 0);
        EXPECT_EQ(probe.device.kv_tokens, 16);
        EXPECT_EQ(probe.device.full_page_tokens, 4) << "the window behind 12 lowers the split";
        EXPECT_EQ(probe.device.per_group[0].hits, (std::vector<std::uint8_t>{1}));
        EXPECT_EQ(probe.device.per_group[1].hits, (std::vector<std::uint8_t>{0}));
        EXPECT_EQ(probe.host.num_common_tokens, 12);
        EXPECT_EQ(probe.host.full_page_tokens, 12);
        EXPECT_EQ(probe.host.per_group[0].hits, (std::vector<std::uint8_t>{1, 1}));
        EXPECT_EQ(probe.host.per_group[1].hits, (std::vector<std::uint8_t>{1, 1}));
        std::vector<BlockTable> tables(3);
        const auto result = AdmitForTest(coordinator, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(result);
        EXPECT_EQ(result->load_pairs.size(), 5u) << "h1, h2 in both KV groups and the checkpoint";
    }
    CacheForGroup(coordinator, device, hashes[1], 1);
    CacheCoordinator::PrefixProbe probe = coordinator.ProbePrefix(hashes);
    EXPECT_EQ(probe.device.full_page_tokens, 12);
    EXPECT_EQ(probe.device.per_group[1].hits, (std::vector<std::uint8_t>{0, 1, 1}));
    EXPECT_EQ(probe.host.num_common_tokens, 12);
    std::vector<BlockTable> tables(3);
    const auto result = AdmitForTest(coordinator, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(result);
    ASSERT_EQ(result->load_pairs.size(), 1u);
    EXPECT_EQ(result->load_pairs[0].group_id, 2u);
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
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    const std::vector<GroupDemand> demands{{.table = &tables[0]}};

    // Retention progress alone names no completed page: nothing is published,
    // however many pages the table has filled.
    ASSERT_TRUE(coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{.num_computed_tokens = 4},
                                  std::nullopt, /*evict_blocks_past_segments=*/false));
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
    coordinator.Free(tables);
}

TEST(CompletedBoundaryTest, RejectsEmptyCompletedPageRange) {
    BlockPool pool(8, {1});
    std::vector<CacheGroupSpec> specs = {
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
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

    EXPECT_THROW(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                   /*evict_blocks_past_segments=*/false),
                 std::runtime_error);
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
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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

    EXPECT_THROW(coordinator.Admit(coordinator.ProbePrefix({}), demands, progress, std::nullopt,
                                   /*evict_blocks_past_segments=*/false),
                 std::runtime_error);
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
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    ASSERT_TRUE(
        coord.Admit(coord.ProbePrefix({}), demands, progress, std::nullopt, /*evict_blocks_past_segments=*/false));
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
                                             /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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

    ASSERT_TRUE(
        coord.Admit(coord.ProbePrefix({}), demands, progress, std::nullopt, /*evict_blocks_past_segments=*/false));
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
    EXPECT_THROW(MakeCoordinator(full_replay, 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr,
                                 false, /*write_back_evictions=*/false),
                 std::runtime_error);
    const std::vector<CacheGroupSpec> state_replay = {CacheGroupSpec{.attn_kind = AttnKind::kState,
                                                                     .sliding_window = 0,
                                                                     .replayable = true,
                                                                     .cache_blocks_per_lcm_block = 1,
                                                                     .block_granularity = 4}};
    EXPECT_THROW(MakeCoordinator(state_replay, 4, pool, /*enable_l3_storage=*/false, nullptr, /*snapshot_pool=*/nullptr,
                                 false, /*write_back_evictions=*/false),
                 std::runtime_error);
}

TEST(BoundedReplayCoordinator, ReplayableGroupNeverConstrainsTheCommonBoundary) {
    BlockPool pool(16, {1, 1});
    CacheCoordinator coord = MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                             /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false);
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
    CacheCoordinator coord = MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                             /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false);
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
    const std::optional<CacheCoordinator::AdmissionResult> admitted = coord.Admit(
        coord.ProbePrefix(hashes), demands, RequestProgress{}, std::nullopt, /*evict_blocks_past_segments=*/false);
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
                                             /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
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
    CacheCoordinator coord = MakeCoordinator(ReplaySpecs(), 4, pool, /*enable_l3_storage=*/false, nullptr,
                                             /*snapshot_pool=*/nullptr, false, /*write_back_evictions=*/false);
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
    return coordinator.Admit(coordinator.ProbePrefix({}), demands, RequestProgress{}, std::nullopt,
                             /*evict_blocks_past_segments=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    coordinator.CacheHostBlock(warm, Key("h0", 0), std::nullopt);
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
    coordinator.CacheHostBlock(fresh, Key("h1", 0), std::nullopt);
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
                                 restore->load_pairs[1].logical_block_index, restore->load_pairs[1].boundary_kind,
                                 /*access_epoch=*/1);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));
    const std::vector<std::string> hashes{"h0", "h1"};
    CacheFullBlocksForTest(coordinator, tables, hashes);
    const std::vector<CacheBlockRef> original(tables[0].Blocks().begin(), tables[0].Blocks().end());

    std::vector<std::vector<ImageSlot>> host_slots = coordinator.PublishedDataSlots(tables, 5);
    for (ImageSlot& slot : host_slots[0]) {
        CacheBlockRef host = coordinator.AcquireHostBlock(0, /*bucket=*/0);
        coordinator.CacheHostBlock(host, slot.key, std::nullopt);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
        coordinator.CacheHostBlock(host, slot.key, std::nullopt);
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
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
                                                   /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                                   /*write_back_evictions=*/false);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    // Fill Host with evictable entries of bucket 0 and bucket 1 alike.
    for (std::int32_t i = 0; i < 4; ++i) {
        CacheBlockRef block = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{i % 2}).front();
        coordinator.CacheHostBlock(block, Key("old-" + std::to_string(i), 0), std::nullopt);
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    const std::array<std::uint32_t, 2> groups{0, 0};
    const std::array<std::int32_t, 2> buckets{1, 1};
    CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostBlocks(groups, buckets);
    ASSERT_TRUE(batch.blocks[0]);
    ASSERT_TRUE(batch.blocks[1]);
    EXPECT_EQ(allocator.BucketOf(batch.blocks[0]->Location()), 1);
    EXPECT_EQ(allocator.BucketOf(batch.blocks[1]->Location()), 1);
    EXPECT_EQ(batch.stats.group_scans, 1u);
    EXPECT_EQ(coordinator.NumHostCachedBlocks(), 2) << "only bucket-1 entries were evicted";
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-0", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("old-2", 0)));
}

// ---------------------------------------------------------------------------
// Restores whose claims pin the only cache they could evict -- the request's
// own pages above a page it claims, or its own checkpoint beside one -- each
// under write-through and under write-back (segment eviction).
// ---------------------------------------------------------------------------

// The image keeps page 3, already on the Host, in Host L2 and pages 0-2 in
// the snapshot pool. The restore claims page 3 from the Device and needs fresh
// blocks for pages 0-2 and the reserve, with one empty parent; the only cache
// on the Device is the request's own pages 0-2, the ancestors of page 3.
// Write-through evicts them block by block. Under write-back the segment of
// each holds the claimed page 3, so the claims leave no plan; claiming
// nothing, the restore copies page 3 back from Host L2 too and evicts pages
// 0-3 as one segment, which the plan's drain writes back: pages 0-2 are
// copied, page 3 rides the Host entry it has.
TEST(RestoreWriteBackTest, ARestoreEvictsItsOwnPagesAboveAPageItClaims) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    for (const bool write_back : {false, true}) {
        SCOPED_TRACE(write_back);
        BlockPool pool(5, {1});
        BlockPool host_pool(8, {1});
        BlockPool snapshot_pool(8, {1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/write_back);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
        const std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        CacheCompletedBlocksForTest(coordinator, tables, hashes,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        std::vector<std::vector<ImageSlot>> host_slots = coordinator.PublishedDataSlots(tables, /*num_tokens=*/8);
        ASSERT_EQ(host_slots[0].size(), 4u);
        host_slots[0].erase(host_slots[0].begin(), host_slots[0].begin() + 3);
        CacheBlockRef host_block = coordinator.AcquireHostBlock(0, /*bucket=*/0);
        coordinator.CacheHostBlock(host_block, host_slots[0][0].key, std::nullopt);
        host_slots[0][0].block = std::move(host_block);
        auto image = coordinator.TakeImage(tables, /*num_tokens=*/8, host_slots);
        ASSERT_TRUE(image);
        image->store_pairs.clear();
        host_slots.clear();
        coordinator.Free(tables);
        ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);
        ASSERT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 4);

        std::vector<BlockTable> restored(1);
        std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
        const auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
        ASSERT_TRUE(restore) << "the request's own pages are the room it needs";
        EXPECT_EQ(restored[0].NumBlocks(), 5);
        EXPECT_EQ(restore->snapshot_pairs.size(), 3u);
        if (write_back) {
            EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, 4u) << "pages 0-3, one segment";
            ASSERT_EQ(restore->load_pairs.size(), 1u);
            EXPECT_EQ(restore->load_pairs[0].key, Key(hashes[3], 0)) << "page 3, republished at the ACK";
            EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
            TierTransferManager transfers(coordinator);
            const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
            ASSERT_TRUE(op.has_value());
            ASSERT_EQ(op->transfers.size(), 3u);
            for (std::size_t row = 0; row < 3; ++row) {
                EXPECT_EQ(op->transfers[row].content_hash, hashes[2 - row]) << "deepest page first";
            }
            const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kEviction);
            EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{3}));
            EXPECT_EQ(stats.host_cached, 1u) << "page 3";
            EXPECT_EQ(stats.no_placement, 0u);
            EXPECT_EQ(coordinator.DeviceEvictions().request_reclaim.blocks, 0u);
            transfers.CompleteWriteBack(op->op_id);
            coordinator.CollectDeadEntries();
            for (const std::string& hash : hashes) {
                EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
            }
        } else {
            EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, 3u) << "pages 0-2";
            EXPECT_TRUE(restore->load_pairs.empty()) << "page 3 is claimed";
            EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumPinnedEntries(pool), 1);
        }
        coordinator.Free(restored);
    }
}

// A request retracted at 4, the end of page 1, with a State group: the
// retraction put pages 0 and 1 and State slot 1 (the checkpoint at 4 it
// published) in Host L2, and slot 0 (the zero state it started from) in the
// snapshot pool. The restore claims pages 0 and 1 from the Device; slot 1 is
// the one the request keeps writing, so it is a fresh copy like slot 0, and
// with the reserve that is three blocks against two empty parents. The only
// other cache on the Device is the checkpoint at 4, beside the claimed page 1.
// Write-through evicts the checkpoint alone. Under write-back its segment
// holds page 1, so the claims leave no plan; claiming nothing, the restore
// copies pages 0 and 1 back from Host L2 too and evicts them with the
// checkpoint as one segment. The image's Host entries hold all three, so the
// plan's drain copies nothing: the segment rides them.
TEST(RestoreWriteBackTest, ARestoreEvictsItsOwnCheckpointBesideAPageItClaims) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const bool write_back : {false, true}) {
        SCOPED_TRACE(write_back);
        BlockPool pool(5, {1, 1});
        BlockPool host_pool(8, {1, 1});
        BlockPool snapshot_pool(8, {1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/write_back);
        std::vector<BlockTable> tables(2);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
        const std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
        CacheCompletedBlocksForTest(coordinator, tables, hashes,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
        std::vector<std::vector<ImageSlot>> host_slots = coordinator.PublishedDataSlots(tables, /*num_tokens=*/4);
        ASSERT_EQ(host_slots[0].size(), 2u);
        ASSERT_EQ(host_slots[1].size(), 1u);
        ASSERT_EQ(host_slots[1][0].slot_index, 1);
        for (std::vector<ImageSlot>& slots : host_slots) {
            for (ImageSlot& slot : slots) {
                CacheBlockRef host_block = coordinator.AcquireHostBlock(slot.key.group_id, /*bucket=*/0);
                coordinator.CacheHostBlock(host_block, slot.key, std::nullopt);
                slot.block = std::move(host_block);
            }
        }
        auto image = coordinator.TakeImage(tables, /*num_tokens=*/4, host_slots);
        ASSERT_TRUE(image);
        ASSERT_EQ(image->store_pairs.size(), 1u) << "State slot 0";
        image->store_pairs.clear();
        host_slots.clear();
        coordinator.Free(tables);
        ASSERT_EQ(pool.NumEmptyLcmBlocks(), 2);

        std::vector<BlockTable> restored(2);
        std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1},
                                         GroupDemand{.extent = StateStep{StateStep::Kind::kKeep}}};
        const auto restore = RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
        ASSERT_TRUE(restore) << "the request's own checkpoint is the room it needs";
        EXPECT_EQ(restored[0].NumBlocks(), 3);
        EXPECT_EQ(restore->snapshot_pairs.size(), 1u);
        if (write_back) {
            EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, 3u) << "pages 0-1 and the checkpoint";
            ASSERT_EQ(restore->load_pairs.size(), 3u);
            EXPECT_EQ(restore->load_pairs[0].key, Key(hashes[0], 0));
            EXPECT_EQ(restore->load_pairs[1].key, Key(hashes[1], 0));
            EXPECT_TRUE(restore->load_pairs[2].key.content_hash.empty()) << "State slot 1 stays private";
            EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
            EXPECT_EQ(coordinator.GroupPrefixIndex(1).NumEntries(pool), 0);
            TierTransferManager transfers(coordinator);
            EXPECT_FALSE(transfers.StartEvictionStores().has_value());
            const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kEviction);
            EXPECT_EQ(stats.host_cached, 3u);
            EXPECT_EQ(stats.no_placement, 0u);
            EXPECT_EQ(coordinator.DeviceEvictions().request_reclaim.blocks, 0u);
        } else {
            EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, 1u) << "the checkpoint";
            ASSERT_EQ(restore->load_pairs.size(), 1u) << "State slot 1";
            EXPECT_TRUE(restore->load_pairs[0].key.content_hash.empty());
            EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumPinnedEntries(pool), 2);
        }
        coordinator.Free(restored);
    }
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
              /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false)} {}

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
    EXPECT_EQ(admitted->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveDeviceCacheBlockId(source));
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
    EXPECT_EQ(admitted->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveDeviceCacheBlockId(source));
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
    EXPECT_EQ(decode->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveDeviceCacheBlockId(own));
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
        {allocator.ResolveDeviceCacheBlockId(canonical_first),
         allocator.ResolveDeviceCacheBlockId(owner[0].Blocks()[2]->Location())},
        {allocator.ResolveDeviceCacheBlockId(canonical_second),
         allocator.ResolveDeviceCacheBlockId(owner[0].Blocks()[3]->Location())},
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
    EXPECT_EQ(decode->pages_to_copy[0].front().first, coordinator_.Allocator(0).ResolveDeviceCacheBlockId(own));
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
    TailPublicationTest() : TailPublicationTest(/*with_host=*/false, /*write_back_evictions=*/false) {}
    // with_host gives the coordinator a Host tier of the same shape, which
    // only the test itself feeds; write_back_evictions (which needs it) the
    // write-back policy's eviction, with its collection of dead entries.
    TailPublicationTest(bool with_host, bool write_back_evictions)
        : pool_(/*num_lcm_blocks=*/64, {1, 1, 1}),
          host_pool_(/*num_lcm_blocks=*/64, {1, 1, 1}),
          coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
                  CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4},
                  CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                 .sliding_window = 3,
                                 .cache_blocks_per_lcm_block = 1,
                                 .block_granularity = 2},
              },
              kP, pool_, /*enable_l3_storage=*/false, with_host ? &host_pool_ : nullptr, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/false, write_back_evictions)} {}

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

    // Nine requests sharing page 0 end at nine different offsets of page 1:
    // nine tails under one prior, so the ninth drops the first. before_ninth
    // sees the first eight tails just before the ninth is published, to pin
    // or store what the test needs then.
    std::vector<TailPublication> PublishNineTails(
        const std::function<void(const std::vector<TailPublication>&)>& before_ninth) {
        std::vector<TailPublication> tails;
        for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior + 1; ++extra) {
            std::vector<std::int32_t> tokens = MakeTokens(kP);
            for (std::int32_t i = 0; i < 3; ++i) {
                tokens.push_back(1000 * extra + i);
            }
            if (extra == TailTable::kMaxTailsPerPrior + 1) {
                before_ninth(tails);
            }
            std::vector<BlockTable> tables(3);
            tails.push_back(PublishFinal(tables, tokens, /*num_computed_tokens=*/kP + 3));
            coordinator_.Free(tables);
        }
        return tails;
    }

    bool Cached(std::uint32_t group, const std::string& hash, std::int32_t offset) const {
        return coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(group))
            .Contains(pool_, Key(hash, group, offset));
    }

    BlockPool pool_;
    BlockPool host_pool_;
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
    // 0's two blocks and the sliding group's two blocks with it: a tail is
    // usable only whole.
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
    EXPECT_FALSE(Cached(2, tail.hash, 0));
    // The victim and its four companions count under the same reason; there
    // is no Host tier to hold any of them.
    const CacheCoordinator::DeviceEvictionCounts& evicted = coordinator_.DeviceEvictions().admission;
    EXPECT_EQ(evicted.blocks, 5u);
    EXPECT_EQ(evicted.host_resident, 0u);
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
            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
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
    EXPECT_FALSE(Cached(0, tails[1].hash, 1));
    EXPECT_FALSE(Cached(2, tails[1].hash, 0));
    // The cap evicts the dropped tail's blocks itself: the three rows [4, 7)
    // in two blocks of group 0, one of group 1, and two of the sliding group
    // (its window behind 7 starts at row 5).
    EXPECT_EQ(coordinator_.DeviceEvictions().tail_cap.blocks, 5u);
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 0u) << "the pool never ran short";
    for (std::size_t i = 2; i < tails.size(); ++i) {
        EXPECT_TRUE(Cached(0, tails[i].hash, 0)) << "tail " << i;
    }
}

TEST_F(TailPublicationTest, CapCountsOnlyTheBlocksItEvictsAndLeavesAPinnedCompanionIndexed) {
    // The ninth tail drops the first, whose first block in group 0 is still
    // referenced, as a request table would hold it.
    CacheBlockRef pinned;
    const std::vector<TailPublication> tails = PublishNineTails([&](const std::vector<TailPublication>& published) {
        pinned = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(published.front().hash, 0, 0));
    });
    ASSERT_TRUE(static_cast<bool>(pinned));
    EXPECT_EQ(coordinator_.TailCapEvictions(), 1);
    EXPECT_TRUE(Cached(0, tails.front().hash, 0)) << "the pinned companion stays indexed";
    EXPECT_FALSE(Cached(0, tails.front().hash, 1));
    EXPECT_FALSE(Cached(1, tails.front().hash, 0));
    EXPECT_FALSE(Cached(2, tails.front().hash, 0));
    EXPECT_FALSE(Cached(2, tails.front().hash, 1));
    // Only the blocks the cap evicted count: group 0's second block, but not
    // its pinned first one.
    EXPECT_EQ(coordinator_.DeviceEvictions().tail_cap.blocks, 4u);
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 0u) << "the pool never ran short";

    // Released, it is an ordinary cache entry, left to the eviction order:
    // without write-back nothing collects dead entries.
    pinned.reset();
    coordinator_.CollectDeadEntries();
    EXPECT_TRUE(Cached(0, tails.front().hash, 0));
}

// The tail cap with a Host tier: a dropped tail can never be matched again,
// so its Host entries leave with its Device blocks.
class TailCapHostTest : public TailPublicationTest {
protected:
    TailCapHostTest() : TailPublicationTest(/*with_host=*/true, /*write_back_evictions=*/false) {}

    // Copies every Device entry of a published tail to the Host, as a
    // write-through store's ACK publishes it. Returns how many it copied.
    std::int32_t StoreOnHost(const TailPublication& tail) {
        std::int32_t stored = 0;
        coordinator_.ForEachBoundaryKey(
            CacheKey{.content_hash = tail.hash}, tail.tail_tokens, [&](std::size_t i, const CacheKey& key) {
                if (!coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(i)).Contains(pool_, key)) {
                    return;
                }
                CacheBlockRef block = host_pool_.AcquireBlock(key.group_id);
                coordinator_.CacheHostBlock(block, key, std::nullopt);
                ++stored;
            });
        return stored;
    }
};

TEST_F(TailCapHostTest, CapDropsTheTailsHostEntriesToo) {
    // The two oldest tails are on the Host before the ninth arrives.
    const std::vector<TailPublication> tails = PublishNineTails([&](const std::vector<TailPublication>& published) {
        EXPECT_EQ(StoreOnHost(published[0]), 5) << "two, one and two blocks in groups 0, 1 and 2";
        EXPECT_EQ(StoreOnHost(published[1]), 5);
    });
    EXPECT_EQ(coordinator_.TailCapEvictions(), 1);
    EXPECT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5) << "only the second tail's five Host entries are left";
    coordinator_.ForEachBoundaryKey(
        CacheKey{.content_hash = tails[0].hash}, tails[0].tail_tokens, [&](std::size_t, const CacheKey& key) {
            EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(key)) << "group " << key.group_id;
            EXPECT_FALSE(Cached(key.group_id, tails[0].hash, key.page_offset)) << "group " << key.group_id;
        });
    coordinator_.ForEachBoundaryKey(
        CacheKey{.content_hash = tails[1].hash}, tails[1].tail_tokens, [&](std::size_t, const CacheKey& key) {
            EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(key)) << "group " << key.group_id;
        });
    // Each entry leaves after the Device's, so none is Device-resident.
    EXPECT_EQ(coordinator_.HostEvictions().tail_cap.blocks, 5u);
    EXPECT_EQ(coordinator_.HostEvictions().tail_cap.device_resident, 0u);
    EXPECT_EQ(coordinator_.DeviceEvictions().tail_cap.blocks, 5u) << "the Device side is unchanged";
}

TEST_F(TailCapHostTest, CapLeavesAPinnedHostEntryOfTheTail) {
    // The two oldest tails are on the Host before the ninth arrives, and a
    // load-back reads the first one's group 0 block.
    CacheBlockRef held;
    const std::vector<TailPublication> tails = PublishNineTails([&](const std::vector<TailPublication>& published) {
        EXPECT_EQ(StoreOnHost(published[0]), 5);
        EXPECT_EQ(StoreOnHost(published[1]), 5);
        held = coordinator_.AcquireHostCachedBlock(Key(published[0].hash, 0));
    });
    ASSERT_TRUE(static_cast<bool>(held));
    EXPECT_EQ(coordinator_.TailCapEvictions(), 1);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tails[0].hash, 0))) << "the load-back's source stays";
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(tails[0].hash, 0, /*page_offset=*/1)));
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(tails[0].hash, 1)));
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 6);
    EXPECT_EQ(coordinator_.HostEvictions().tail_cap.blocks, 4u) << "only the unpinned ones count: 1, 1 and 2";
    held.reset();
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tails[0].hash, 0))) << "an ordinary entry again";
}

// The tail cap under the write-back policy: what it leaves of a dropped tail
// because a block was in use is a dead entry, which no probe enumerates.
class TailCapWriteBackTest : public TailPublicationTest {
protected:
    TailCapWriteBackTest() : TailPublicationTest(/*with_host=*/true, /*write_back_evictions=*/true) {}
};

// CapCountsOnlyTheBlocksItEvictsAndLeavesAPinnedCompanionIndexed when writing
// back: the pinned companion waits, and the first collection once it is
// released takes it and the tail's node; nothing else is dead.
TEST_F(TailCapWriteBackTest, ACappedTailLeftoverIsCollectedOnceUnpinned) {
    CacheBlockRef pinned;
    const std::vector<TailPublication> tails = PublishNineTails([&](const std::vector<TailPublication>& published) {
        pinned = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(published.front().hash, 0, 0));
    });
    ASSERT_TRUE(static_cast<bool>(pinned));
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    ASSERT_TRUE(Cached(0, tails.front().hash, 0));
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_);
    const CacheCoordinator::DeviceEvictionCounts& collected = coordinator_.DeviceEvictions().dead_entry_collection;

    coordinator_.CollectDeadEntries();
    EXPECT_TRUE(Cached(0, tails.front().hash, 0)) << "still in use";
    EXPECT_EQ(collected.blocks, 0u);

    pinned.reset();
    coordinator_.CollectDeadEntries();
    EXPECT_FALSE(Cached(0, tails.front().hash, 0));
    EXPECT_EQ(tree.Find(tails.front().hash), nullptr);
    EXPECT_EQ(collected.blocks, 1u) << "the pinned companion alone";
    EXPECT_EQ(collected.host_resident, 0u);
    for (std::size_t i = 1; i < tails.size(); ++i) {
        EXPECT_TRUE(Cached(0, tails[i].hash, 0)) << "tail " << i;
    }
    EXPECT_TRUE(Cached(0, hashes_[0], 0)) << "the prior page";
    EXPECT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
}

// The same leftover, never collected: an admission takes it as the oldest
// victim. No probe enumerates a tail without its hint, so it is not written
// back (a dead entry, not written back); the next victim, the second tail, is.
TEST_F(TailCapWriteBackTest, ACappedTailIsNotWrittenBack) {
    CacheBlockRef pinned;
    const std::vector<TailPublication> tails = PublishNineTails([&](const std::vector<TailPublication>& published) {
        pinned = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(published.front().hash, 0, 0));
    });
    ASSERT_TRUE(Cached(0, tails.front().hash, 0));
    pinned.reset();
    // One block per group, none free.
    const std::vector<CacheBlockRef> held = pool_.AcquireBlocks(0, pool_.NumEmptyLcmBlocks());
    std::vector<BlockTable> tables(3);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 2));
    coordinator_.Free(tables);

    EXPECT_FALSE(Cached(0, tails.front().hash, 0));
    EXPECT_EQ(coordinator_.DeadEntriesNotWrittenBack(), 1u);
    std::vector<CacheKey> recorded;
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        for (const CacheCoordinator::EvictedBlock& member : segment.members) {
            recorded.push_back(member.key);
        }
        // The plan's drain finds no Host room for it.
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }
    EXPECT_EQ(recorded.size(), 5u) << "the second tail's two, one and two blocks";
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_);
    for (const CacheKey& key : recorded) {
        EXPECT_EQ(key.content_hash, tails[1].hash);
        if (const BoundaryTree::Node* node = tree.Find(key.content_hash)) {
            EXPECT_EQ(node->pending_host_stores, 0) << "no store holds it once settled";
            EXPECT_EQ(node->pending_host_checkpoint_stores, 0);
        }
    }
}

// ACappedTailIsNotWrittenBack under the empty prior, with the first tail's
// node started by a Host entry (an L3 GET's) before the tail's publication:
// that publication has no parent to link the node to, but gives it its
// boundary, so the admission still finds the capped tail dead.
TEST_F(TailCapWriteBackTest, ACappedTailWhoseNodeAHostEntryStartedIsNotWrittenBack) {
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_);
    std::vector<TailPublication> tails;
    CacheBlockRef pinned;
    for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior + 1; ++extra) {
        const std::vector<std::int32_t> tokens{1000 * extra, 1000 * extra + 1, 1000 * extra + 2};
        if (extra == 1) {
            CacheBlockRef block = host_pool_.AcquireBlock(1);
            coordinator_.CacheHostBlock(block, Key(HashPrefixPage(token_span(tokens), std::string{}), 1), std::nullopt);
        }
        if (extra == TailTable::kMaxTailsPerPrior + 1) {
            pinned = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(tails.front().hash, 0, 0));
        }
        std::vector<BlockTable> tables(3);
        tails.push_back(PublishFinal(tables, tokens, /*num_computed_tokens=*/3));
        coordinator_.Free(tables);
        if (extra == 1) {
            const BoundaryTree::Node* node = tree.Find(tails.front().hash);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->parent, nullptr);
            EXPECT_EQ(node->boundary_tokens, 3);
            EXPECT_EQ(node->tail_tokens, 3);
        }
    }
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    ASSERT_TRUE(Cached(0, tails.front().hash, 0));
    pinned.reset();
    // One block per group, none free.
    const std::vector<CacheBlockRef> held = pool_.AcquireBlocks(0, pool_.NumEmptyLcmBlocks());
    std::vector<BlockTable> tables(3);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 2));
    coordinator_.Free(tables);

    EXPECT_FALSE(Cached(0, tails.front().hash, 0));
    EXPECT_EQ(coordinator_.DeadEntriesNotWrittenBack(), 1u);
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        for (const CacheCoordinator::EvictedBlock& member : segment.members) {
            EXPECT_NE(member.key.content_hash, tails.front().hash) << "the capped tail is not recorded";
        }
        // The plan's drain finds no Host room for it.
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }
}

// A write-back of a tail is in flight when the cap drops the tail; the Host
// held one of its blocks already, and a load is reading that entry. The cap
// queues the tail, but the collection of that plan finds it still on its way
// to the Host, not dead. The settlement cancels the tail and cannot evict the
// entry in use -- without State groups no checkpoint boundary flips and the
// tail's node stays, so nothing but the settlement itself queues it again --
// and the first collection once the load let go takes it.
TEST_F(TailCapWriteBackTest, ACancelledTailInUseIsCollectedOnceReleased) {
    TierTransferManager transfers{coordinator_};
    std::vector<std::int32_t> tokens = MakeTokens(kP);
    for (std::int32_t i = 0; i < 3; ++i) {
        tokens.push_back(1000 + i);
    }
    std::vector<BlockTable> tables(3);
    const TailPublication tail = PublishFinal(tables, tokens, /*num_computed_tokens=*/kP + 3);
    coordinator_.Free(tables);
    {
        CacheBlockRef block = host_pool_.AcquireBlock(0);
        coordinator_.CacheHostBlock(block, Key(tail.hash, 0, 0), std::nullopt);
    }
    // The tail's five blocks are the oldest (with the prior page's window
    // block, a set of its own).
    std::vector<CacheBlockRef> held = pool_.AcquireBlocks(0, pool_.NumEmptyLcmBlocks());
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 2));
    coordinator_.Free(tables);
    held.clear();
    const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(std::ranges::count(op->transfers, tail.hash, &CacheTransfer::content_hash), 4)
        << "all but the one the Host held";
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    CacheBlockRef load = coordinator_.AcquireHostCachedBlock(Key(tail.hash, 0, 0));
    for (std::int32_t extra = 2; extra <= TailTable::kMaxTailsPerPrior + 1; ++extra) {
        std::vector<std::int32_t> other = MakeTokens(kP);
        for (std::int32_t i = 0; i < 3; ++i) {
            other.push_back(1000 * extra + i);
        }
        PublishFinal(tables, other, /*num_computed_tokens=*/kP + 3);
        coordinator_.Free(tables);
    }
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    // A plan's collection while the store is out: the tail it queued is not
    // dead yet.
    coordinator_.CollectDeadEntries();
    ASSERT_NE(CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_).Find(tail.hash), nullptr);

    const std::vector<HostPublication> published = transfers.CompleteWriteBack(op->op_id);
    EXPECT_EQ(published.size(), 1u) << "the prior page's window block";
    EXPECT_TRUE(std::ranges::none_of(
        published, [&](const HostPublication& entry) { return entry.key.content_hash == tail.hash; }));
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, 0, 0))) << "in use";
    coordinator_.CollectDeadEntries();
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, 0, 0)));
    load.reset();
    coordinator_.CollectDeadEntries();
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, 0, 0)));
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 1u);
    EXPECT_EQ(CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_).Find(tail.hash), nullptr);
}

// ---------------------------------------------------------------------------
// A state checkpoint resumes only with every State group's block at its hash,
// so the blocks leave the Device together; the KV of its page stays. K3
// shape: P = 4, one Full group (one block per page, four per parent) and three
// State groups (one per parent), in a pool of exactly seven parents.
// ---------------------------------------------------------------------------
class StateCheckpointCascadeTest : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;

    StateCheckpointCascadeTest()
        : pool_(/*num_lcm_blocks=*/7, {4, 1, 1, 1}),
          coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
              kP, pool_, /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
              /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false)},
          hashes_{ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}})} {}

    // A request over three pages publishes them with its checkpoint at their
    // end (hashes_[2]) and finishes: one Full parent holds the three pages,
    // three State parents the checkpoints, and the zero states' three parents
    // are empty again.
    void PublishCheckpointAndFinish() {
        std::vector<BlockTable> tables(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, 3 * kP));
        CacheCompletedBlocksForTest(coordinator_, tables, hashes_,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
        coordinator_.Free(tables);
        ASSERT_EQ(coordinator_.NumEmptyLcmBlocks(), 3);
    }

    bool Cached(std::uint32_t group, std::size_t page) const {
        return coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(group))
            .Contains(pool_, Key(hashes_[page], group));
    }

    CacheBlockLocation CheckpointLocation(std::uint32_t group) const {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, group, Key(hashes_[2], group));
        _assert(location.has_value(), "the checkpoint is published");
        return *location;
    }

    BlockPool pool_;
    CacheCoordinator coordinator_;
    std::vector<std::string> hashes_;
};

TEST_F(StateCheckpointCascadeTest, EvictingOneStateCheckpointEvictsTheOtherStateGroupsAtItsHash) {
    PublishCheckpointAndFinish();
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, 1, CheckpointLocation(1)));

    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(Cached(group, 2)) << "group " << group;
    }
    for (std::size_t page = 0; page < 3; ++page) {
        EXPECT_TRUE(Cached(0, page)) << "a checkpoint takes no KV along: page " << page;
    }
    EXPECT_EQ(coordinator_.DeviceBoundaryResidency(Key(hashes_[2], 0), /*tail_tokens=*/0),
              CacheCoordinator::BoundaryResidency::kPartial);
    EXPECT_EQ(coordinator_.NumEmptyLcmBlocks(), 6) << "three State parents freed";
    // The checkpoint and the two it takes along count under the same reason.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 3u);
}

TEST_F(StateCheckpointCascadeTest, APinnedStateSiblingStaysWhenItsCheckpointIsEvicted) {
    PublishCheckpointAndFinish();
    // Group 3's checkpoint is held, as a request table resuming on it would.
    CacheBlockRef pinned = coordinator_.GroupPrefixIndex(3).Find(pool_, Key(hashes_[2], 3));
    ASSERT_TRUE(static_cast<bool>(pinned));

    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, 2, CheckpointLocation(2)));
    EXPECT_FALSE(Cached(1, 2));
    EXPECT_FALSE(Cached(2, 2));
    EXPECT_TRUE(Cached(3, 2)) << "the pinned sibling stays indexed";
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 2u) << "group 2's checkpoint and group 1's";

    pinned.reset();
    EXPECT_TRUE(Cached(3, 2)) << "released, it is an ordinary cache entry again";
    EXPECT_EQ(coordinator_.NumEmptyLcmBlocks(), 5);
}

// A new request's six State blocks (a zero state and a live slot per group)
// need three parents beyond the three empty ones: the planner takes the three
// checkpoints, oldest-first group 1's, whose eviction takes the other two;
// their own planned evictions then find them gone, and the admission fits.
TEST_F(StateCheckpointCascadeTest, PlannedCheckpointVictimsLeaveTogether) {
    PublishCheckpointAndFinish();
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));

    for (std::size_t group = 1; group <= 3; ++group) {
        EXPECT_EQ(tables[group].NumBlocks(), 2) << "group " << group;
        EXPECT_FALSE(Cached(static_cast<std::uint32_t>(group), 2)) << "group " << group;
    }
    for (std::size_t page = 0; page < 3; ++page) {
        EXPECT_TRUE(Cached(0, page)) << "the Full group's hole funds its block: page " << page;
    }
    // Each checkpoint counts once: a planned eviction that finds its victim
    // already gone counts nothing.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 3u);
    coordinator_.Free(tables);
}

// A prefill in chunks: each chunk's rotation releases the request's slot 0.
// The third chunk's slot 0 is the checkpoint the second one published at 4;
// the pool's seven parents are all in use, so it funds the new live slots and
// is evicted -- counted as the request's own reclaim, not as an admission's
// victim. (The zero states the second chunk releases were never cached.)
TEST_F(StateCheckpointCascadeTest, RequestReclaimCountsApartFromAdmission) {
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));
    for (std::int32_t pages = 1; pages <= 2; ++pages) {
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = std::span<const std::string>(hashes_).first(
                                                  static_cast<std::size_t>(pages)),
                                              .first_new_prefix_page = pages - 1,
                                              .publish_state = true,
                                              .boundary_kind = CacheBoundaryKind::kChunk},
            .num_computed_tokens = pages * kP,
        };
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, progress))
            << "chunk " << pages + 1;
    }

    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(Cached(group, 0)) << "group " << group;
        EXPECT_TRUE(Cached(group, 1)) << "the checkpoint at 8 is the new slot 0: group " << group;
    }
    // The three State groups' checkpoints at 4, and nothing else.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.request_reclaim.blocks, 3u);
    EXPECT_EQ(evictions.admission.blocks, 0u);
    coordinator_.Free(tables);
}

// ---------------------------------------------------------------------------
// The boundary tree through the coordinator: a publication links its nodes
// before any group registers under them, and a node lives exactly while some
// tier holds its hash or a child hangs under it. K3 shape: P = 4, one Full
// group (one block per page, four per parent) and three State groups (one per
// parent), on the Device and the Host tier alike; the Host is fed only by the
// test. BoundaryTreeCoordinatorTest evicts block by block, as under
// write-through; SegmentEvictionTest (below) by segments, as under write-back.
// ---------------------------------------------------------------------------
class BoundaryTreeCoordinatorFixture : public ::testing::Test {
protected:
    static constexpr std::int32_t kP = 4;
    static constexpr std::size_t kDevice = static_cast<std::size_t>(CacheTier::kDevice);
    using Node = BoundaryTree::Node;
    using CheckpointBoundaryCounts = std::array<std::int32_t, 2>;  // checkpoint_boundaries_in_subtree by CacheTier

    BoundaryTreeCoordinatorFixture(bool write_back_evictions, std::int32_t full_shard_count,
                                   std::int32_t num_snapshot_pages)
        : pool_(/*num_lcm_blocks=*/32, {4, 1, 1, 1}),
          host_pool_(/*num_lcm_blocks=*/32, {4, 1, 1, 1}),
          snapshot_pool_(num_snapshot_pages, {4, 1, 1, 1}),
          coordinator_{MakeCoordinator(
              std::vector<CacheGroupSpec>{
                  {.attn_kind = AttnKind::kFull,
                   .cache_blocks_per_lcm_block = 4,
                   .block_granularity = kP,
                   .shard_count = full_shard_count},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
                  {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
              kP, pool_, /*enable_l3_storage=*/false, &host_pool_,
              /*snapshot_pool=*/num_snapshot_pages > 0 ? &snapshot_pool_ : nullptr,
              /*stream_device_cache_to_host=*/false, write_back_evictions)} {}

    const BoundaryTree& Tree() const { return CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator_); }

    const Node& NodeOf(const std::string& hash) const {
        const Node* node = Tree().Find(hash);
        _assert(node != nullptr, "the hash has a tree node");
        return *node;
    }

    // Admits a request over `tokens` and publishes all of them as a final
    // frontier: the full pages (hashes_), the tail at an unaligned end, and
    // every State group's checkpoint at the end. Returns the tail.
    std::optional<TailPublication> PublishFinal(std::vector<BlockTable>& tables,
                                                const std::vector<std::int32_t>& tokens) {
        const auto num_tokens = static_cast<std::int32_t>(tokens.size());
        EXPECT_TRUE(AdmitForTest(coordinator_, tables, num_tokens));
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < num_tokens / kP; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        hashes_ = ContentHashes(pages);
        std::optional<TailPublication> tail;
        if (num_tokens % kP != 0) {
            tail = TailPublication{
                .tail_tokens = num_tokens % kP,
                .hash = HashPrefixPage(token_span(tokens).subspan(pages.size() * kP),
                                       hashes_.empty() ? std::string{} : hashes_.back()),
            };
        }
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = hashes_,
                                              .first_new_prefix_page = 0,
                                              .tail = tail,
                                              .publish_state = true,
                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
            .num_computed_tokens = num_tokens,
        };
        coordinator_.CacheCompletedBlocks(tables, progress, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        return tail;
    }

    // Registers a Host copy of a Device entry, as a write-through store's ACK
    // does.
    void StoreOnHost(const CacheKey& key) {
        CacheBlockRef block = host_pool_.AcquireBlock(key.group_id);
        coordinator_.CacheHostBlock(block, key, std::nullopt);
    }

    // A planned Device eviction of the entry under `key`, with its cascade.
    void EvictOnDevice(const CacheKey& key) {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, key.group_id, key);
        ASSERT_TRUE(location.has_value());
        ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator_, key.group_id, *location));
    }

    // Two finished sessions over the same first two pages, A before B, each
    // ending in a tail of 2 with its checkpoint: s0 s1 a2 a3 tA and s0 s1 b2
    // b3 tB. B's publication touches s0 and s1, so A's own entries are the
    // oldest.
    void PublishTwoSessionsSharingTwoPages() {
        const std::vector<std::int32_t> a = MakeTokens(18);
        std::vector<std::int32_t> b = MakeTokens(18, 100);
        std::copy_n(a.begin(), 2 * kP, b.begin());
        std::vector<BlockTable> a_tables(4);
        a_tail_ = *PublishFinal(a_tables, a);
        a_hashes_ = hashes_;
        coordinator_.Free(a_tables);
        std::vector<BlockTable> b_tables(4);
        b_tail_ = *PublishFinal(b_tables, b);
        b_hashes_ = hashes_;
        coordinator_.Free(b_tables);
    }

    // Holds every free slot and every empty parent of `pool` (the Device's
    // or the Host's) but `spare_parents`, so a demand needing more is funded
    // by evictions.
    void HoldFreeCapacity(BlockPool& pool, std::int32_t spare_parents) {
        for (CacheBlockRef& block : pool.AcquireBlocks(0, pool.NumFreeSlots(0))) {
            held_.push_back(std::move(block));
        }
        for (CacheBlockRef& block : pool.AcquireBlocks(1, pool.NumEmptyLcmBlocks() - spare_parents)) {
            held_.push_back(std::move(block));
        }
        ASSERT_EQ(pool.NumFreeSlots(0), 0);
        ASSERT_EQ(pool.NumEmptyLcmBlocks(), spare_parents);
    }

    BlockPool pool_;
    BlockPool host_pool_;
    // The coordinator has a snapshot pool only when it has pages.
    BlockPool snapshot_pool_;
    CacheCoordinator coordinator_;
    std::vector<std::string> hashes_;
    std::vector<std::string> a_hashes_;
    std::vector<std::string> b_hashes_;
    TailPublication a_tail_;
    TailPublication b_tail_;
    // Declared after the coordinator, so released before the pools.
    std::vector<CacheBlockRef> held_;
};

class BoundaryTreeCoordinatorTest : public BoundaryTreeCoordinatorFixture {
protected:
    BoundaryTreeCoordinatorTest()
        : BoundaryTreeCoordinatorFixture(/*write_back_evictions=*/false, /*full_shard_count=*/1,
                                         /*num_snapshot_pages=*/0) {}
};

TEST_F(BoundaryTreeCoordinatorTest, TreeNodesFollowPublication) {
    std::vector<BlockTable> tables(4);
    // 22 tokens: five pages and a tail of 2, the checkpoint under the tail.
    const std::optional<TailPublication> tail = PublishFinal(tables, MakeTokens(22));
    coordinator_.Free(tables);
    ASSERT_TRUE(tail.has_value());
    ASSERT_EQ(hashes_.size(), 5u);

    EXPECT_EQ(Tree().Size(), 6u);
    EXPECT_EQ(NodeOf(hashes_[0]).parent, nullptr);
    for (std::size_t page = 0; page < hashes_.size(); ++page) {
        const Node& node = NodeOf(hashes_[page]);
        if (page > 0) {
            EXPECT_EQ(node.parent, &NodeOf(hashes_[page - 1])) << "page " << page;
        }
        EXPECT_EQ(node.boundary_tokens, static_cast<std::int32_t>(page + 1) * kP);
        EXPECT_EQ(node.tail_tokens, 0);
        // The checkpoint at 22 is the only checkpoint boundary; every page
        // below counts it.
        EXPECT_FALSE(node.has_checkpoint[kDevice]) << "page " << page;
        EXPECT_EQ(node.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0})) << "page " << page;
    }
    const Node& tail_node = NodeOf(tail->hash);
    EXPECT_EQ(tail_node.parent, &NodeOf(hashes_[4]));
    EXPECT_EQ(tail_node.boundary_tokens, 22);
    EXPECT_EQ(tail_node.tail_tokens, 2);
    EXPECT_TRUE(tail_node.has_checkpoint[kDevice]);
    EXPECT_EQ(tail_node.checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_EQ(Tree().FirstHostLeaf(), nullptr) << "nothing on the Host";
}

// The in-forward checkpoint at 8 sits on a page the same publication
// hashes. Its nodes are linked before the State groups register, so it
// counts in its ancestors as it is published -- by a finish
// (CacheCompletedBlocks) and by the next chunk's admission alike.
TEST_F(BoundaryTreeCoordinatorTest, ANewCheckpointCountsInItsAncestorsWhenItIsPublished) {
    for (const bool by_admission : {false, true}) {
        SCOPED_TRACE(by_admission);
        const std::vector<std::int32_t> tokens = MakeTokens(12, by_admission ? 100 : 1);
        // The chunk [0, 12) writes the state at 8 into a block of its own.
        std::vector<BlockTable> tables(4);
        std::optional<CacheCoordinator::AdmissionResult> chunk =
            AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{12}},
                         RequestProgress{}, StateStep{.kind = StateStep::Kind::kRotate, .checkpoint = true});
        ASSERT_TRUE(chunk.has_value());
        std::vector<CacheBlockRef> checkpoints = std::move(chunk->state_checkpoint_blocks);
        const std::vector<std::string> hashes = ContentHashes({{tokens.begin(), tokens.begin() + 4},
                                                               {tokens.begin() + 4, tokens.begin() + 8},
                                                               {tokens.begin() + 8, tokens.end()}});
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = hashes,
                                              .first_new_prefix_page = 0,
                                              .state_checkpoint_pages = 2,
                                              .state_checkpoint_blocks = checkpoints},
            .num_computed_tokens = 12,
        };
        if (by_admission) {
            ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                     GroupDemand{.extent = DenseGrowth{4}}, progress));
        } else {
            coordinator_.CacheCompletedBlocks(tables, progress,
                                              CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        }
        EXPECT_TRUE(NodeOf(hashes[1]).has_checkpoint[kDevice]);
        EXPECT_EQ(NodeOf(hashes[1]).parent, &NodeOf(hashes[0]));
        EXPECT_EQ(NodeOf(hashes[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
        EXPECT_EQ(NodeOf(hashes[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
        EXPECT_EQ(NodeOf(hashes[2]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}))
            << "the page after the checkpoint";
        coordinator_.Free(tables);
    }
}

TEST_F(BoundaryTreeCoordinatorTest, ANodeLivesWhileEitherTierHoldsItsHash) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(12));  // three pages, the checkpoint under the last
    coordinator_.Free(tables);
    // The last page and its checkpoint are on the Host too.
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(hashes_[2], group));
    }
    const Node* last = &NodeOf(hashes_[2]);
    EXPECT_TRUE(last->on_host);
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(Tree().FirstHostLeaf(), last);

    // The Device evicts the page and, with it, the checkpoint: the Host keeps
    // the node, linked as it was.
    EvictOnDevice(Key(hashes_[2], 0));
    ASSERT_EQ(Tree().Find(hashes_[2]), last);
    EXPECT_EQ(last->parent, &NodeOf(hashes_[1]));
    EXPECT_EQ(last->has_checkpoint, (std::array<bool, 2>{false, true}));
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));

    // The Host evicts the checkpoint (its cascade takes every State group's)
    // and then the page: the node leaves, the pages below stay.
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(hashes_[2], 2)));
    EXPECT_EQ(last->has_checkpoint, (std::array<bool, 2>{false, false}));
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_TRUE(last->on_host);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(hashes_[2], 0)));
    EXPECT_EQ(Tree().Find(hashes_[2]), nullptr);
    EXPECT_EQ(Tree().Size(), 2u);
    EXPECT_EQ(NodeOf(hashes_[1]).first_child, nullptr);
    EXPECT_EQ(Tree().FirstHostLeaf(), nullptr);
}

TEST_F(BoundaryTreeCoordinatorTest, AGapNodeKeepsItsChildrenLinked) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(12));
    coordinator_.Free(tables);

    // The middle page leaves the Device; the last page still hangs under it.
    EvictOnDevice(Key(hashes_[1], 0));
    const Node* gap = Tree().Find(hashes_[1]);
    ASSERT_NE(gap, nullptr);
    EXPECT_EQ(coordinator_.DeviceBoundaryResidency(Key(hashes_[1], 0), /*tail_tokens=*/0),
              CacheCoordinator::BoundaryResidency::kNone);
    EXPECT_EQ(gap->first_child, &NodeOf(hashes_[2]));
    EXPECT_EQ(NodeOf(hashes_[2]).parent, gap);
    EXPECT_EQ(gap->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}))
        << "it still counts the checkpoint below it";
    // Nothing reaches the last page past the gap, but without write-back
    // nothing collects it either: it ages out through the eviction order.
    coordinator_.CollectDeadEntries();
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[2], 0)));
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 0u);

    // Once the page below leaves too, nothing hangs under the gap and it goes
    // with it.
    EvictOnDevice(Key(hashes_[2], 0));
    EXPECT_EQ(Tree().Find(hashes_[2]), nullptr);
    EXPECT_EQ(Tree().Find(hashes_[1]), nullptr);
    EXPECT_EQ(Tree().Size(), 1u);
    EXPECT_EQ(NodeOf(hashes_[0]).first_child, nullptr);
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
}

TEST_F(BoundaryTreeCoordinatorTest, ClearCacheEmptiesTheBoundaryTree) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(12));
    coordinator_.Free(tables);
    StoreOnHost(Key(hashes_[0], 0));
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(hashes_[2], group));
    }
    ASSERT_EQ(Tree().Size(), 3u);

    ASSERT_TRUE(coordinator_.ClearCache());
    EXPECT_EQ(Tree().Size(), 0u);
    EXPECT_EQ(Tree().FirstHostLeaf(), nullptr);

    // The next publication starts the tree afresh.
    std::vector<BlockTable> next(4);
    PublishFinal(next, MakeTokens(8));
    coordinator_.Free(next);
    EXPECT_EQ(Tree().Size(), 2u);
    EXPECT_EQ(NodeOf(hashes_[1]).parent, &NodeOf(hashes_[0]));
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
}

TEST_F(BoundaryTreeCoordinatorTest, ClearDeviceCacheKeepsTheNodesHostEntriesNeed) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(16));  // four pages, the checkpoint under the last
    coordinator_.Free(tables);
    // The Host holds the first page and the last with its checkpoint.
    StoreOnHost(Key(hashes_[0], 0));
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(hashes_[3], group));
    }

    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5) << "Host L2 is untouched";
    // The two pages between hold nothing any more but link the last one.
    EXPECT_EQ(Tree().Size(), 4u);
    for (std::size_t page = 1; page < 4; ++page) {
        EXPECT_EQ(NodeOf(hashes_[page]).parent, &NodeOf(hashes_[page - 1])) << "page " << page;
    }
    for (const std::string& hash : hashes_) {
        EXPECT_EQ(NodeOf(hash).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    }
    EXPECT_FALSE(NodeOf(hashes_[1]).on_host);
    // The first page is a Host leaf too: no child of it is on the Host.
    EXPECT_EQ(Tree().FirstHostLeaf(), &NodeOf(hashes_[0]));
    EXPECT_EQ(NodeOf(hashes_[0]).next_host_leaf, &NodeOf(hashes_[3]));
}

// The tail hint goes exactly when the last entry under the tail hash leaves
// either tier, as before the boundary tree (TailPublicationTest's
// EvictingOneTailBlockTakesItsCompanionsAndPrunesTheTable), and the tail's
// node goes with it.
TEST_F(BoundaryTreeCoordinatorTest, TailHintPruningIsUnchangedWithoutPendingStores) {
    std::vector<BlockTable> first(4);
    const TailPublication device_only = *PublishFinal(first, MakeTokens(22));
    coordinator_.Free(first);
    std::vector<BlockTable> second(4);
    const TailPublication host_too = *PublishFinal(second, MakeTokens(22, 100));
    coordinator_.Free(second);
    StoreOnHost(Key(host_too.hash, 0));
    ASSERT_EQ(coordinator_.NumTailEntries(), 2u);

    // One block of a Device-only tail takes the rest; the hint and the node
    // leave with the last of them.
    EvictOnDevice(Key(device_only.hash, 1));
    EXPECT_EQ(coordinator_.DeviceBoundaryResidency(Key(device_only.hash, 0), device_only.tail_tokens),
              CacheCoordinator::BoundaryResidency::kNone);
    EXPECT_EQ(coordinator_.NumTailEntries(), 1u);
    EXPECT_EQ(Tree().Find(device_only.hash), nullptr);

    // The Host still holds the other tail's KV block: hint and node stay
    // until it leaves too.
    EvictOnDevice(Key(host_too.hash, 0));
    EXPECT_EQ(coordinator_.NumTailEntries(), 1u);
    ASSERT_NE(Tree().Find(host_too.hash), nullptr);
    EXPECT_TRUE(NodeOf(host_too.hash).on_host);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(host_too.hash, 0)));
    EXPECT_EQ(coordinator_.NumTailEntries(), 0u);
    EXPECT_EQ(Tree().Find(host_too.hash), nullptr);
    EXPECT_EQ(Tree().Size(), 10u) << "both sessions' five pages stay on the Device";
}

// A Host entry no publication linked -- an L3 GET, or a store whose Device
// entry left before its ACK -- starts a node without a parent.
TEST_F(BoundaryTreeCoordinatorTest, AHostEntryWithoutAPublicationStartsANodeOfItsOwn) {
    StoreOnHost(Key("orphan", 0));
    StoreOnHost(Key("orphan", 1));
    const Node& node = NodeOf("orphan");
    EXPECT_EQ(node.parent, nullptr);
    EXPECT_EQ(node.boundary_tokens, 0);
    EXPECT_TRUE(node.on_host);
    EXPECT_EQ(node.has_checkpoint, (std::array<bool, 2>{false, true}));
    EXPECT_EQ(Tree().FirstHostLeaf(), &node);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key("orphan", 0)));
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key("orphan", 1)));
    EXPECT_EQ(Tree().Size(), 0u);
}

// A restore's republished Device entry whose node left while no tier held the
// hash starts a node without a parent, and its eviction takes the node along.
TEST_F(BoundaryTreeCoordinatorTest, ARepublishedDeviceEntryWithoutANodeStartsOneOfItsOwn) {
    {
        CacheBlockRef page = pool_.AcquireBlock(0);
        coordinator_.CacheDeviceBlock(page, Key("restored", 0), /*logical_block_index=*/0, CacheBoundaryKind::kEndpoint,
                                      /*access_epoch=*/1);
        CacheBlockRef checkpoint = pool_.AcquireBlock(1);
        coordinator_.CacheDeviceBlock(checkpoint, Key("restored", 1), /*logical_block_index=*/-1,
                                      CacheBoundaryKind::kEndpoint, /*access_epoch=*/1);
    }
    const Node* node = Tree().Find("restored");
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->parent, nullptr);
    EXPECT_EQ(node->boundary_tokens, 0);
    EXPECT_FALSE(node->on_host);
    EXPECT_EQ(node->has_checkpoint, (std::array<bool, 2>{true, false}));
    EXPECT_EQ(node->checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    // The page leaves the Device and takes the checkpoint along.
    EvictOnDevice(Key("restored", 0));
    EXPECT_EQ(Tree().Size(), 0u);
}

// A request holds a page whose node an entry started without a parent -- a
// Host page (an L3 GET's, or a write-through ACK's after its Device entry
// left) its admission loaded and published on the Device, as a Host
// extension is -- and publishes the pages after it. The search for their
// parent goes on past that node, so the publication links it under the page
// before it too, which then counts the Host page and the Device checkpoint
// boundary at 12.
TEST_F(BoundaryTreeCoordinatorTest, APublicationLinksTheNodeOfAPageItHolds) {
    const std::vector<std::int32_t> tokens = MakeTokens(12);
    const std::vector<std::string> hashes = ContentHashes({{tokens.begin(), tokens.begin() + 4},
                                                           {tokens.begin() + 4, tokens.begin() + 8},
                                                           {tokens.begin() + 8, tokens.end()}});
    StoreOnHost(Key(hashes[1], 0));
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 12));
    const std::uint64_t epoch = CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_);
    for (std::int32_t page = 0; page < 2; ++page) {
        coordinator_.CacheFullBlocks(tables, std::span{hashes}.subspan(page, 1), epoch, /*first_slot=*/page,
                                     CacheBoundaryKind::kChunk);
    }
    ASSERT_EQ(NodeOf(hashes[1]).parent, nullptr);
    const RequestProgress progress{
        .completed_pages = CompletedPages{.prefix_hashes = hashes,
                                          .first_new_prefix_page = 2,
                                          .publish_state = true,
                                          .boundary_kind = CacheBoundaryKind::kEndpoint},
        .num_computed_tokens = 12,
    };
    coordinator_.CacheCompletedBlocks(tables, progress, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
    coordinator_.Free(tables);

    EXPECT_EQ(NodeOf(hashes[0]).parent, nullptr);
    EXPECT_EQ(NodeOf(hashes[1]).parent, &NodeOf(hashes[0]));
    EXPECT_EQ(NodeOf(hashes[1]).boundary_tokens, 2 * kP);
    EXPECT_EQ(NodeOf(hashes[2]).parent, &NodeOf(hashes[1]));
    EXPECT_EQ(NodeOf(hashes[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_EQ(NodeOf(hashes[0]).children_on_host, 1);
    EXPECT_EQ(Tree().FirstHostLeaf(), &NodeOf(hashes[1]));
}

// The Host's checkpoint at 8 (an L3 GET's, or a write-through ACK's after its
// Device entry left) starts a node without a parent. The publication of the
// pages through it links that node under the page before it, which then counts
// the Host checkpoint boundary and the Device checkpoint boundary at 12 under
// it, as it would had the publication come first.
TEST_F(BoundaryTreeCoordinatorTest, APublicationLinksTheNodeAHostEntryStarted) {
    const std::vector<std::int32_t> tokens = MakeTokens(12);
    const std::vector<std::string> hashes = ContentHashes({{tokens.begin(), tokens.begin() + 4},
                                                           {tokens.begin() + 4, tokens.begin() + 8},
                                                           {tokens.begin() + 8, tokens.end()}});
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(hashes[1], group));
    }
    ASSERT_EQ(NodeOf(hashes[1]).parent, nullptr);
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, tokens);
    coordinator_.Free(tables);
    ASSERT_EQ(hashes_, hashes);

    EXPECT_EQ(NodeOf(hashes[1]).parent, &NodeOf(hashes[0]));
    EXPECT_EQ(NodeOf(hashes[1]).boundary_tokens, 2 * kP);
    EXPECT_EQ(NodeOf(hashes[2]).parent, &NodeOf(hashes[1]));
    EXPECT_EQ(NodeOf(hashes[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    EXPECT_EQ(NodeOf(hashes[0]).children_on_host, 1);
    EXPECT_EQ(Tree().FirstHostLeaf(), &NodeOf(hashes[1]));

    // Both checkpoint boundaries leave; the pages stay on the Device.
    EvictOnDevice(Key(hashes[2], 1));
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(hashes[1], 1)));
    EXPECT_EQ(NodeOf(hashes[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    EXPECT_EQ(NodeOf(hashes[0]).children_on_host, 0);
    EXPECT_EQ(Tree().Size(), 3u);
}

// A Host hit at 16 on the Device's pages 0 and 1: its admission publishes
// the Host's pages 2 and 3 on the Device one by one, with no publication to
// link them. They still hang under the pages before them, since some tier
// held each of them from its first publication on. The hit's checkpoints
// stay private, so the Host alone holds those at 16.
TEST_F(BoundaryTreeCoordinatorTest, AHostExtensionKeepsItsPagesLinkedToTheirParents) {
    std::vector<std::int32_t> tokens = MakeTokens(16);
    std::vector<BlockTable> published(4);
    PublishFinal(published, tokens);
    coordinator_.Free(published);
    const std::vector<std::string> hashes = hashes_;
    for (const std::string& hash : hashes) {
        StoreOnHost(Key(hash, 0));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(hashes[3], group));
    }
    // The Device keeps pages 0 and 1: page 3 takes its checkpoints along.
    EvictOnDevice(Key(hashes[3], 0));
    EvictOnDevice(Key(hashes[2], 0));

    tokens.push_back(77);
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes, .tokens = tokens, .max_tokens = 4 * kP});
    ASSERT_EQ(probe.device.full_page_tokens, 2 * kP);
    ASSERT_EQ(probe.host.num_common_tokens, 4 * kP);
    std::vector<BlockTable> tables(4);
    const std::optional<CacheCoordinator::AdmissionResult> admitted =
        AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(admitted.has_value());
    for (std::int32_t page = 2; page < 4; ++page) {
        coordinator_.CacheFullBlocks(tables, std::span<const std::string>(hashes).subspan(page, 1),
                                     admitted->access_epoch, page, CacheBoundaryKind::kChunk);
    }

    EXPECT_EQ(NodeOf(hashes[0]).parent, nullptr);
    for (std::size_t page = 1; page < hashes.size(); ++page) {
        EXPECT_EQ(NodeOf(hashes[page]).parent, &NodeOf(hashes[page - 1])) << "page " << page;
        EXPECT_EQ(NodeOf(hashes[page]).boundary_tokens, static_cast<std::int32_t>(page + 1) * kP);
    }
    EXPECT_EQ(NodeOf(hashes[3]).has_checkpoint, (std::array<bool, 2>{false, true}));
    coordinator_.Free(tables);
}

// The same without a prefix-closed group, under either write policy: no
// group holds the first page, so only the link keeps its node, and it leaves
// with the last checkpoint boundary under it instead of staying behind for
// good.
TEST(BoundaryTreeStateOnlyTest, APageAboveALinkedNodeLeavesWithTheLastCheckpointUnderIt) {
    constexpr std::int32_t kP = 4;
    const std::vector<CacheGroupSpec> specs(
        3, CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1});
    for (const bool write_back_evictions : {false, true}) {
        SCOPED_TRACE(write_back_evictions);
        BlockPool pool(/*num_lcm_blocks=*/8, {1, 1, 1});
        BlockPool host_pool(/*num_lcm_blocks=*/4, {1, 1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
                            /*stream_device_cache_to_host=*/false, write_back_evictions);
        const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
        const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        for (std::uint32_t group = 0; group < 3; ++group) {
            CacheBlockRef block = host_pool.AcquireBlock(group);
            coordinator.CacheHostBlock(block, Key(hashes[1], group), std::nullopt);
        }
        std::vector<BlockTable> tables(3);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, 3 * kP));
        CacheCompletedBlocksForTest(coordinator, tables, hashes,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
        coordinator.Free(tables);
        ASSERT_EQ(tree.Size(), 3u);
        EXPECT_EQ(tree.Find(hashes[1])->parent, tree.Find(hashes[0]));
        EXPECT_EQ(tree.Find(hashes[0])->checkpoint_boundaries_in_subtree, (std::array<std::int32_t, 2>{1, 1}));

        // The Device's checkpoint at 12 and then the Host's at 8 leave, each
        // taking the other groups' at its hash along.
        const std::optional<CacheBlockLocation> device_checkpoint =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hashes[2], 0));
        ASSERT_TRUE(device_checkpoint &&
                    CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *device_checkpoint));
        ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator, Key(hashes[1], 0)));
        EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
        EXPECT_EQ(tree.Size(), 0u);
    }
}

// Without a prefix-closed group a publication may register nothing at all (a
// chunk end of State groups alone): the nodes it linked do not stay, whether
// a finish (CacheCompletedBlocks) or the next chunk's admission publishes it.
// The next publication that registers a checkpoint links the whole chain
// again.
TEST(BoundaryTreeStateOnlyTest, APublicationThatRegistersNothingLeavesNoNode) {
    constexpr std::int32_t kP = 4;
    for (const bool by_admission : {false, true}) {
        SCOPED_TRACE(by_admission);
        BlockPool pool(/*num_lcm_blocks=*/8, {1});
        CacheCoordinator coordinator = MakeCoordinator(
            std::vector<CacheGroupSpec>{{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1}}, kP, pool,
            /*enable_l3_storage=*/false, /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr,
            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, 8));
        const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
        const RequestProgress chunk_end{
            .completed_pages = CompletedPages{.prefix_hashes = hashes,
                                              .first_new_prefix_page = 0,
                                              .publish_state = false,
                                              .boundary_kind = CacheBoundaryKind::kChunk},
            .num_computed_tokens = 8,
        };
        if (by_admission) {
            ASSERT_TRUE(AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{4}}, chunk_end));
        } else {
            coordinator.CacheCompletedBlocks(tables, chunk_end,
                                             CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
        }
        EXPECT_EQ(tree.Size(), 0u);

        // The prompt end at 8 publishes the checkpoint over pages published
        // before.
        const RequestProgress prompt_end{
            .completed_pages = CompletedPages{.prefix_hashes = hashes,
                                              .first_new_prefix_page = 2,
                                              .publish_state = true,
                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
            .num_computed_tokens = 8,
        };
        if (by_admission) {
            ASSERT_TRUE(AdmitForTest(coordinator, tables, GroupDemand{.extent = DenseGrowth{4}}, prompt_end));
        } else {
            coordinator.CacheCompletedBlocks(tables, prompt_end,
                                             CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
        }
        coordinator.Free(tables);
        ASSERT_EQ(tree.Size(), 2u);
        EXPECT_EQ(tree.Find(hashes[1])->parent, tree.Find(hashes[0]));
        EXPECT_TRUE(tree.Find(hashes[1])->has_checkpoint[static_cast<std::size_t>(CacheTier::kDevice)]);
        EXPECT_EQ(tree.Find(hashes[0])->checkpoint_boundaries_in_subtree, (std::array<std::int32_t, 2>{1, 0}));
        EXPECT_EQ(tree.Find(hashes[1])->checkpoint_boundaries_in_subtree, (std::array<std::int32_t, 2>{1, 0}));
    }
}

// The scenario of SegmentEvictionTest.ALeafSegmentLeavesWholeWithItsCheckpoint
// without write-back: the planner takes one block at a time, as it did before
// segments. A's group-1 checkpoint alone funds the block; its eviction takes
// the rest of A's tail along, and a3 and a2 stay.
TEST_F(BoundaryTreeCoordinatorTest, WriteThroughEvictsBlocksAsBefore) {
    PublishTwoSessionsSharingTwoPages();
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));

    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail_.hash, group)))
            << "group " << group;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(b_tail_.hash, group)))
            << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[page], 0))) << "page " << page;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes_[page], 0))) << "page " << page;
    }
    // A's tail: its KV block and three checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 4u);
    coordinator_.Free(tables);
}

// ---------------------------------------------------------------------------
// Evicting Device cache by segments (write_back_evictions): a victim leaves
// with its checkpoint boundary, the pages only that checkpoint boundary depends
// on, and everything the Device holds under them, so no Device page outlives
// its parent. The K3 shape of BoundaryTreeCoordinatorFixture, its content
// published only through CacheCompletedBlocks so every entry is linked; each
// test holds the pool's free capacity, so its admission must evict.
// ---------------------------------------------------------------------------
class SegmentEvictionTest : public BoundaryTreeCoordinatorFixture {
protected:
    SegmentEvictionTest()
        : BoundaryTreeCoordinatorFixture(/*write_back_evictions=*/true, /*full_shard_count=*/1,
                                         /*num_snapshot_pages=*/0) {}
};

// A's checkpoint at its tail is the oldest candidate. Its segment climbs to a2,
// below the pages B shares, and leaves whole, deepest first: the tail's
// checkpoints and KV block, a3, a2. Nothing of it lies under the candidate.
TEST_F(SegmentEvictionTest, ALeafSegmentLeavesWholeWithItsCheckpoint) {
    PublishTwoSessionsSharingTwoPages();
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    // The plan's drain finds no Host room: every segment is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }

    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail_.hash, group)))
            << "group " << group;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(b_tail_.hash, group)))
            << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[page], 0)), page < 2)
            << "page " << page;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes_[page], 0))) << "page " << page;
    }
    EXPECT_EQ(Tree().Size(), 5u) << "s0, s1, b2, b3 and tB";
    EXPECT_EQ(NodeOf(b_hashes_[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    // A's tail (its KV block and three checkpoints), a3 and a2.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 6u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    coordinator_.Free(tables);
}

// Once A's checkpoint boundary left, B's is the only one under s1 and s0: B's
// segment climbs to the root and takes the shared pages along, and the tree is
// empty.
TEST_F(SegmentEvictionTest, ASharedPrefixLeavesWithItsLastSegment) {
    PublishTwoSessionsSharingTwoPages();
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> first(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, first, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> second(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, second, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    // The plan's drain finds no Host room: every segment is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }

    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes_[page], 0))) << "page " << page;
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(b_tail_.hash, group)))
            << "group " << group;
    }
    EXPECT_EQ(Tree().Size(), 0u);
    // a2 and a3, then s0, s1, b2 and b3; each tail's KV block and three
    // checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 14u);
    coordinator_.Free(first);
    coordinator_.Free(second);
}

// A checkpoint at 8 under a deeper one at 16, which the second publication left
// the older: the page at 8 still serves the deeper checkpoint boundary, so only
// the three checkpoints at 8 leave.
TEST_F(SegmentEvictionTest, ANestedCheckpointLeavesWithoutItsPages) {
    std::vector<BlockTable> shorter(4);
    PublishFinal(shorter, MakeTokens(8));
    coordinator_.Free(shorter);
    std::vector<BlockTable> longer(4);
    PublishFinal(longer, MakeTokens(16));
    coordinator_.Free(longer);
    ASSERT_EQ(NodeOf(hashes_[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 0}));
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    // The plan's drain finds no Host room: every segment is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }

    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes_[1], group))) << "group " << group;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes_[3], group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    EXPECT_FALSE(NodeOf(hashes_[1]).has_checkpoint[kDevice]);
    EXPECT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 3u) << "the three checkpoints at 8";
    coordinator_.Free(tables);
}

// A request reading A's group-3 checkpoint pins one member of A's checkpoint
// boundary: no part of A's segment may leave, so the planner passes over all of
// it to B's, and A's tail keeps its KV block, which one block at a time the
// group-1 checkpoint's eviction would have taken along.
TEST_F(SegmentEvictionTest, APinnedCheckpointKeepsItsWholeSegment) {
    std::vector<BlockTable> a_tables(4);
    const TailPublication a_tail = *PublishFinal(a_tables, MakeTokens(18));
    const std::vector<std::string> a_hashes = hashes_;
    coordinator_.Free(a_tables);
    std::vector<BlockTable> b_tables(4);
    const TailPublication b_tail = *PublishFinal(b_tables, MakeTokens(18, 100));
    coordinator_.Free(b_tables);
    const CacheBlockRef pinned = coordinator_.GroupPrefixIndex(3).Find(pool_, Key(a_tail.hash, 3));
    ASSERT_TRUE(static_cast<bool>(pinned));
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));

    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail.hash, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(b_tail.hash, group)))
            << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes[page], 0))) << "page " << page;
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    // B's four pages, and its tail's KV block and three checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 8u);
    coordinator_.Free(tables);
}

// The new request resumes at 8 on A's first two Device pages and a Host
// checkpoint there, so those pages are hits it keeps: A's segment climbs no
// higher than a2. Had it taken a1 and a0, it could not have left at all.
TEST_F(SegmentEvictionTest, TheSegmentClimbStopsAtProtectedHitPages) {
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    std::vector<BlockTable> a_tables(4);
    const TailPublication a_tail = *PublishFinal(a_tables, tokens);
    coordinator_.Free(a_tables);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(hashes_[1], group));
    }
    // Six State blocks (each group's loaded checkpoint and live slot) and a
    // Full block: three spare parents, and the segment's three State parents
    // and three Full slots.
    HoldFreeCapacity(pool_, /*spare_parents=*/3);
    const std::vector<std::string> prompt_hashes{hashes_[0], hashes_[1]};
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = prompt_hashes, .tokens = tokens, .max_tokens = 2 * kP});
    ASSERT_EQ(probe.device.num_common_tokens, 2 * kP);
    const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
    std::vector<BlockTable> tables(4);
    const std::optional<CacheCoordinator::AdmissionResult> result =
        AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{kP}});
    ASSERT_TRUE(result);
    EXPECT_EQ(result->device_prefix_tokens, 2 * kP);
    EXPECT_EQ(result->load_pairs.size(), 3u);

    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0)), page < 2) << "page " << page;
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail.hash, group)))
            << "group " << group;
    }
    // a2, a3, and the tail's KV block and three checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks - before.admission.blocks, 6u);
    coordinator_.Free(tables);
}

// Two requests publish s0 and s1 at their chunk ends, then their own pages,
// tails and checkpoints at their finishes, so s1 and s0 are the oldest entries.
// Both checkpoint boundaries depend on them: they are passed over, and A's
// segment leaves without them.
TEST_F(SegmentEvictionTest, APageTwoCheckpointBoundariesShareWaitsForTheLastOfThem) {
    const std::vector<std::int32_t> a = MakeTokens(18);
    std::vector<std::int32_t> b = MakeTokens(18, 100);
    std::copy_n(a.begin(), 2 * kP, b.begin());
    std::array<std::vector<BlockTable>, 2> tables{std::vector<BlockTable>(4), std::vector<BlockTable>(4)};
    std::array<std::vector<std::string>, 2> hashes;
    std::array<TailPublication, 2> tails;
    for (std::size_t r = 0; r < 2; ++r) {
        const std::vector<std::int32_t>& tokens = r == 0 ? a : b;
        ASSERT_TRUE(AdmitForTest(coordinator_, tables[r], 18));
        hashes[r] = ContentHashes({{tokens.begin(), tokens.begin() + 4},
                                   {tokens.begin() + 4, tokens.begin() + 8},
                                   {tokens.begin() + 8, tokens.begin() + 12},
                                   {tokens.begin() + 12, tokens.begin() + 16}});
        tails[r] = TailPublication{.tail_tokens = 2,
                                   .hash = HashPrefixPage(token_span(tokens).subspan(4 * kP), hashes[r].back())};
        CacheCompletedBlocksForTest(coordinator_, tables[r], std::span<const std::string>(hashes[r]).first(2),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * kP,
                                    CacheBoundaryKind::kChunk, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
    }
    for (std::size_t r = 0; r < 2; ++r) {
        coordinator_.CacheCompletedBlocks(
            tables[r],
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = hashes[r],
                                                              .first_new_prefix_page = 2,
                                                              .tail = tails[r],
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = 18},
            CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        coordinator_.Free(tables[r]);
    }
    ASSERT_EQ(NodeOf(hashes[0][1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 0}));
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
    std::vector<BlockTable> next(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, next, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));

    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes[0][page], 0)), page < 2)
            << "page " << page;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes[1][page], 0))) << "page " << page;
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(tails[0].hash, group)))
            << "group " << group;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(tails[1].hash, group)))
            << "group " << group;
    }
    // A's third and fourth pages, and its tail's KV block and three
    // checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks - before.admission.blocks, 6u);
    coordinator_.Free(next);
}

// A demand only the whole cache but one Full slot funds: 31 Full blocks from
// a pool of 32 parents whose free capacity is held. A's segment and B's leave
// s1 and s0, which both checkpoint boundaries depend on, for later; once the
// candidates run out, the shared pages are tried again without the checkpoint
// boundaries this plan already takes. s1 then leaves with everything under it
// (taking both segments in), and s0, its single slot not needed, stays. Block
// by block (write-through) the same admission fits; without the second try it
// would fail every time, the cache unchanged.
TEST_F(SegmentEvictionTest, SharedPagesLeaveOnceEveryCheckpointBoundaryUnderThemLeaves) {
    PublishTwoSessionsSharingTwoPages();
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                             GroupDemand{.extent = DenseGrowth{31 * kP}}, RequestProgress{},
                             StateStep{.kind = StateStep::Kind::kKeep}));
    // The plan's drain finds no Host room: every segment is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }

    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[page], 0)), page == 0)
            << "page " << page;
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes_[page], 0)), page == 0)
            << "page " << page;
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail_.hash, group)))
            << "group " << group;
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(b_tail_.hash, group)))
            << "group " << group;
    }
    EXPECT_EQ(Tree().Size(), 1u) << "s0";
    // a2, a3, b2, b3 and s1; each tail's KV block and three checkpoints. The
    // segments keep their reason inside s1's set.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 13u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    coordinator_.Free(tables);
}

// The chunked prefill of StateCheckpointCascadeTest's
// RequestReclaimCountsApartFromAdmission, evicting by segments: the third
// chunk's released checkpoint at 4 leaves, the other groups' at its hash with
// it, as the request's own reclaim, and the page under it stays.
TEST_F(SegmentEvictionTest, ReleasedCheckpointsCountAsTheRequestsOwnReclaim) {
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));
    for (std::int32_t pages = 1; pages <= 2; ++pages) {
        if (pages == 2) {
            // The Full block takes the one spare parent; the three new live
            // slots need the released checkpoints' parents.
            HoldFreeCapacity(pool_, /*spare_parents=*/1);
        }
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = std::span<const std::string>(hashes).first(
                                                  static_cast<std::size_t>(pages)),
                                              .first_new_prefix_page = pages - 1,
                                              .publish_state = true,
                                              .boundary_kind = CacheBoundaryKind::kChunk},
            .num_computed_tokens = pages * kP,
        };
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, progress))
            << "chunk " << pages + 1;
    }

    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes[0], group))) << "group " << group;
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes[1], group))) << "group " << group;
    }
    // The three checkpoints at 4.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.request_reclaim.blocks, 3u);
    EXPECT_EQ(evictions.admission.blocks, 0u);
    coordinator_.Free(tables);
}

// B resumes at A's tail at 18 (its KV block copied, its checkpoints claimed
// as slot 0) and computes to 24; the next chunk's rotation releases slot 0.
// That chunk needs three State parents for the new live slots and one Full
// slot, and nothing else is free: the released checkpoints' set holds the
// tail's KV block too, as their eviction takes it along, so the plan counts
// its slot and fits. Every block of the tail leaves as B's own reclaim.
TEST_F(SegmentEvictionTest, AReleasedTailCheckpointLeavesWithItsTailsKvBlock) {
    std::vector<BlockTable> a_tables(4);
    const std::vector<std::int32_t> a = MakeTokens(18);
    const TailPublication a_tail = *PublishFinal(a_tables, a);
    coordinator_.Free(a_tables);
    std::vector<std::int32_t> b = MakeTokens(24, 100);
    std::copy_n(a.begin(), 18, b.begin());
    const std::vector<std::string> b_hashes = ContentHashes({{b.begin(), b.begin() + 4},
                                                             {b.begin() + 4, b.begin() + 8},
                                                             {b.begin() + 8, b.begin() + 12},
                                                             {b.begin() + 12, b.begin() + 16},
                                                             {b.begin() + 16, b.begin() + 20}});
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = b_hashes, .tokens = b, .max_tokens = 23});
    ASSERT_EQ(probe.device.num_common_tokens, 18);
    ASSERT_TRUE(probe.device.tail.has_value());
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{6}}));
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{.num_computed_tokens = 24}));

    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail.hash, group)))
            << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    // The tail's KV block and three checkpoints.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.request_reclaim.blocks, 4u);
    EXPECT_EQ(evictions.admission.blocks, 0u);
    coordinator_.Free(tables);
}

// B resumes at A's tail at 18, as in AReleasedTailCheckpointLeavesWithItsTailsKvBlock,
// and its finish marks that tail superseded while B's table still holds its
// checkpoints (a fused finish publishes before the release). The mark only
// re-dates the tail's entries, to epoch 0. Once B releases them, the next
// admission that needs room takes them first and drops them as request
// reclaim, never written back.
TEST_F(SegmentEvictionTest, AMarkUnderAHeldCheckpointTakesEffectOnceItIsReleased) {
    std::vector<BlockTable> a_tables(4);
    const std::vector<std::int32_t> a = MakeTokens(18);
    const TailPublication a_tail = *PublishFinal(a_tables, a);
    coordinator_.Free(a_tables);
    std::vector<std::int32_t> b = MakeTokens(24, 100);
    std::copy_n(a.begin(), 18, b.begin());
    const std::vector<std::string> b_hashes = ContentHashes({{b.begin(), b.begin() + 4},
                                                             {b.begin() + 4, b.begin() + 8},
                                                             {b.begin() + 8, b.begin() + 12},
                                                             {b.begin() + 12, b.begin() + 16},
                                                             {b.begin() + 16, b.begin() + 20}});
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = b_hashes, .tokens = b, .max_tokens = 23});
    std::vector<BlockTable> tables(4);
    const std::optional<CacheCoordinator::AdmissionResult> admitted =
        AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{6}});
    ASSERT_TRUE(admitted.has_value());
    ASSERT_EQ(admitted->resumed_tail_hash, a_tail.hash);
    const auto metadata_of = [&](std::uint32_t group) {
        const CacheBlockRef block = coordinator_.GroupPrefixIndex(group).Find(pool_, Key(a_tail.hash, group));
        return coordinator_.GroupPrefixIndex(group).MetadataFor(pool_, block->Location());
    };
    std::vector<PrefixCacheIndex::CachedBlockMetadata> before;
    for (std::uint32_t group = 0; group < 4; ++group) {
        before.push_back(*metadata_of(group));
        EXPECT_EQ(before.back().last_access_epoch, admitted->access_epoch) << "group " << group;
    }

    coordinator_.MarkSupersededStateInput(a_tail.hash, admitted->access_epoch);
    for (std::uint32_t group = 0; group < 4; ++group) {
        const PrefixCacheIndex::CachedBlockMetadata after = *metadata_of(group);
        EXPECT_EQ(after.last_access_epoch, 0u) << "group " << group;
        EXPECT_EQ(after.logical_block_index, before[group].logical_block_index) << "group " << group;
        EXPECT_EQ(after.boundary_kind, before[group].boundary_kind) << "group " << group;
        EXPECT_EQ(after.was_acquired, before[group].was_acquired) << "group " << group;
    }

    coordinator_.Free(tables);
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> next(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, next, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    // The room needs A's four pages too. With the tail gone, no checkpoint
    // boundary lies under them on either tier, so they are dead and dropped as
    // well: nothing is written back.
    EXPECT_TRUE(coordinator_.TakeEvictedSegments().empty());
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(a_tail.hash, group)))
            << "group " << group;
    }
    // The tail's KV block and three checkpoints leave as the request's
    // reclaim, A's four pages as the admission's victims.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.request_reclaim.blocks, 4u);
    EXPECT_EQ(evictions.admission.blocks, 4u);
    coordinator_.Free(next);
}

// C resumes at A's tail after B did, so the tail's entries carry C's epoch:
// B's finish leaves them unmarked, C's input until C finishes, and C's
// finish marks them.
TEST_F(SegmentEvictionTest, AnEarlierFinishLeavesALaterHittersInputUnmarked) {
    std::vector<BlockTable> a_tables(4);
    const std::vector<std::int32_t> a = MakeTokens(18);
    const TailPublication a_tail = *PublishFinal(a_tables, a);
    coordinator_.Free(a_tables);
    std::vector<std::uint64_t> epochs;
    for (const std::int32_t start : {100, 200}) {
        std::vector<std::int32_t> next = MakeTokens(22, start);
        std::copy_n(a.begin(), 18, next.begin());
        const std::vector<std::string> next_hashes = ContentHashes({{next.begin(), next.begin() + 4},
                                                                    {next.begin() + 4, next.begin() + 8},
                                                                    {next.begin() + 8, next.begin() + 12},
                                                                    {next.begin() + 12, next.begin() + 16},
                                                                    {next.begin() + 16, next.begin() + 20}});
        CacheCoordinator::PrefixProbe probe =
            coordinator_.ProbePromptPrefix({.prefix_hashes = next_hashes, .tokens = next, .max_tokens = 21});
        std::vector<BlockTable> tables(4);
        const std::optional<CacheCoordinator::AdmissionResult> admitted =
            AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
        ASSERT_TRUE(admitted.has_value());
        ASSERT_EQ(admitted->resumed_tail_hash, a_tail.hash);
        epochs.push_back(admitted->access_epoch);
        coordinator_.Free(tables);
    }
    ASSERT_LT(epochs[0], epochs[1]);
    const auto epochs_now = [&] {
        std::vector<std::uint64_t> now;
        for (std::uint32_t group = 0; group < 4; ++group) {
            const CacheBlockRef block = coordinator_.GroupPrefixIndex(group).Find(pool_, Key(a_tail.hash, group));
            now.push_back(
                coordinator_.GroupPrefixIndex(group).MetadataFor(pool_, block->Location())->last_access_epoch);
        }
        return now;
    };

    coordinator_.MarkSupersededStateInput(a_tail.hash, epochs[0]);
    EXPECT_EQ(epochs_now(), std::vector<std::uint64_t>(4, epochs[1])) << "B's finish";
    coordinator_.MarkSupersededStateInput(a_tail.hash, epochs[1]);
    EXPECT_EQ(epochs_now(), std::vector<std::uint64_t>(4, 0)) << "C's finish";
}

// A's two pages, published without a checkpoint, are older than B's two and
// share a Full parent with them. A fresh request needs six State parents and
// a Full block, five parents are spare: A's set (a1, then a0 taking it in)
// frees no parent; B's segment frees its three State parents and, with A's,
// the shared Full parent. Restoring newest first, B's set is still needed and
// A's is not: it returns whole, where block by block a1 would stay evicted.
TEST_F(SegmentEvictionTest, TheRestorePassReturnsWholeSets) {
    const std::vector<std::int32_t> a = MakeTokens(2 * kP);
    const std::vector<std::string> a_hashes = ContentHashes({{a.begin(), a.begin() + kP}, {a.begin() + kP, a.end()}});
    std::vector<BlockTable> a_tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, a_tables, 2 * kP));
    CacheCompletedBlocksForTest(coordinator_, a_tables, a_hashes,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator_.Free(a_tables);
    std::vector<BlockTable> b_tables(4);
    PublishFinal(b_tables, MakeTokens(2 * kP, 100));
    coordinator_.Free(b_tables);
    HoldFreeCapacity(pool_, /*spare_parents=*/5);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));

    for (const std::string& hash : a_hashes) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hash, 0)));
    }
    for (const std::string& hash : hashes_) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hash, 0)));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes_[1], group))) << "group " << group;
    }
    // B's two pages and three checkpoints.
    EXPECT_EQ(coordinator_.DeviceEvictions().admission.blocks, 5u);
    coordinator_.Free(tables);
}

// A set that cannot leave passes over only the blocks that would expand to it
// again. X = x0 x1, its checkpoint at x1, is the oldest; A resumed at X and
// keeps running: its table holds x0, x1 and its chunk's page a2, and its next
// chunk released X's checkpoint from slot 0. B = x0 x1 b2 b3, newer, keeps its
// checkpoint on the Host alone. X's checkpoint is the first candidate: its
// segment cannot climb past x1 and holds a2, in use, so it stays -- but b2 and
// b3 beside a2 form sets of their own, and b3 alone funds the Full block a
// fresh request needs.
TEST_F(SegmentEvictionTest, AFailedSetLeavesAnIdleBranchBesideItsPinsEvictable) {
    const std::vector<std::int32_t> x = MakeTokens(2 * kP);
    std::vector<BlockTable> x_tables(4);
    PublishFinal(x_tables, x);
    const std::vector<std::string> x_hashes = hashes_;
    coordinator_.Free(x_tables);

    std::vector<std::int32_t> a = x;
    const std::vector<std::int32_t> a_more = MakeTokens(kP, 200);
    a.insert(a.end(), a_more.begin(), a_more.end());
    const std::vector<std::string> a_hashes = ContentHashes(
        {{a.begin(), a.begin() + kP}, {a.begin() + kP, a.begin() + 2 * kP}, {a.begin() + 2 * kP, a.end()}});
    const std::vector<std::string> a_prompt{a_hashes[0], a_hashes[1]};
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = a_prompt, .tokens = a, .max_tokens = 2 * kP});
    ASSERT_EQ(probe.device.num_common_tokens, 2 * kP);
    std::vector<BlockTable> a_tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, a_tables, std::move(probe), GroupDemand{.extent = DenseGrowth{kP}}));
    ASSERT_TRUE(
        AdmitForTest(coordinator_, a_tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                     RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = a_hashes,
                                                                       .first_new_prefix_page = 2,
                                                                       .publish_state = false,
                                                                       .boundary_kind = CacheBoundaryKind::kChunk},
                                     .num_computed_tokens = 3 * kP}));

    std::vector<std::int32_t> b = x;
    const std::vector<std::int32_t> b_more = MakeTokens(2 * kP, 300);
    b.insert(b.end(), b_more.begin(), b_more.end());
    std::vector<BlockTable> b_tables(4);
    PublishFinal(b_tables, b);
    const std::vector<std::string> b_hashes = hashes_;
    coordinator_.Free(b_tables);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(b_hashes[3], group));
    }
    EvictOnDevice(Key(b_hashes[3], 1));  // the State cascade takes groups 2 and 3 along
    ASSERT_EQ(NodeOf(x_hashes[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 1}));
    ASSERT_EQ(NodeOf(b_hashes[3]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    ASSERT_EQ(NodeOf(a_hashes[2]).parent, &NodeOf(x_hashes[1]));
    for (std::uint32_t group = 1; group <= 3; ++group) {
        const CacheBlockRef x_checkpoint = coordinator_.GroupPrefixIndex(group).Find(pool_, Key(x_hashes[1], group));
        ASSERT_EQ(x_checkpoint.use_count(), 2) << "only the index holds X's checkpoint, group " << group;
        ASSERT_LT(
            coordinator_.GroupPrefixIndex(group).MetadataFor(pool_, x_checkpoint->Location())->last_access_epoch,
            coordinator_.GroupPrefixIndex(0)
                .MetadataFor(pool_, *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(b_hashes[3], 0)))
                ->last_access_epoch)
            << "X's checkpoint is older than B's pages, group " << group;
    }

    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));

    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes[3], 0)));
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes[2], 0)));
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(x_hashes[1], group))) << "group " << group;
    }
    // b3; the setup's eviction of B's three Device checkpoints before it.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 4u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    EXPECT_EQ(evictions.request_reclaim.blocks, 0u);
    coordinator_.Free(tables);
    coordinator_.Free(a_tables);
}

// The same with nothing running: the hits an admission keeps hold a set back
// alike. X = x0 x1, its checkpoint at x1, is the oldest; C = x0 x1 c2 c3 and
// then B = x0 x1 b2 b3 keep their checkpoints on the Host alone. R, C's next
// turn, resumes at 16 on C's Device pages and its Host checkpoint, so x0 x1
// c2 c3 are hits it keeps and X's checkpoint is not. X's segment holds c2
// and c3 and stays, but B's branch beside them may leave: R's six State
// blocks take the six spare parents, and b3 alone funds its Full block.
TEST_F(SegmentEvictionTest, KeptHitsLeaveAnIdleBranchBesideThemEvictable) {
    const std::vector<std::int32_t> x = MakeTokens(2 * kP);
    std::vector<BlockTable> x_tables(4);
    PublishFinal(x_tables, x);
    const std::vector<std::string> x_hashes = hashes_;
    coordinator_.Free(x_tables);
    std::array<std::vector<std::int32_t>, 2> branch;
    std::array<std::vector<std::string>, 2> branch_hashes;
    for (std::size_t s = 0; s < 2; ++s) {  // C, then B
        branch[s] = x;
        const std::vector<std::int32_t> more = MakeTokens(2 * kP, 300 + 100 * static_cast<std::int32_t>(s));
        branch[s].insert(branch[s].end(), more.begin(), more.end());
        std::vector<BlockTable> tables(4);
        PublishFinal(tables, branch[s]);
        branch_hashes[s] = hashes_;
        coordinator_.Free(tables);
        for (std::uint32_t group = 1; group <= 3; ++group) {
            StoreOnHost(Key(branch_hashes[s][3], group));
        }
        EvictOnDevice(Key(branch_hashes[s][3], 1));
    }
    const std::vector<std::string>& c_hashes = branch_hashes[0];
    const std::vector<std::string>& b_hashes = branch_hashes[1];
    ASSERT_EQ(NodeOf(x_hashes[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 2}));

    std::vector<std::int32_t> r = branch[0];
    const std::vector<std::int32_t> r_more = MakeTokens(kP, 600);
    r.insert(r.end(), r_more.begin(), r_more.end());
    CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = c_hashes, .tokens = r, .max_tokens = 4 * kP});
    ASSERT_EQ(probe.device.num_common_tokens, 4 * kP);
    HoldFreeCapacity(pool_, /*spare_parents=*/6);
    std::vector<BlockTable> tables(4);
    const std::optional<CacheCoordinator::AdmissionResult> result =
        AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{kP}});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->device_prefix_tokens, 4 * kP);
    EXPECT_EQ(result->load_pairs.size(), 3u) << "C's checkpoint from the Host";

    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes[3], 0)));
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(b_hashes[2], 0)));
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(c_hashes[3], 0)));
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(x_hashes[1], group))) << "group " << group;
    }
    // b3; the setup's evictions of C's and B's three Device checkpoints each
    // before it.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 7u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    coordinator_.Free(tables);
}

// A session the Device holds with no checkpoint under it, its checkpoint on
// the Host alone: what a request leaves behind that resumed on the whole
// session from the Host (its admission publishes the pages it loads back on
// the Device, the checkpoint goes to its own slot) and was then aborted. No
// segment climbs, so each page the demand needs is a candidate of its own,
// deepest first, whose set takes the one under it over whole; the first two
// pages stay. The four that leave form one segment, deepest first, each held
// on the Host already.
TEST_F(SegmentEvictionTest, ASessionWithoutADeviceCheckpointLeavesFromItsSuffix) {
    std::vector<BlockTable> published(4);
    PublishFinal(published, MakeTokens(6 * kP));
    coordinator_.Free(published);
    for (const std::string& hash : hashes_) {
        StoreOnHost(Key(hash, 0));
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(hashes_[5], group));
    }
    EvictOnDevice(Key(hashes_[5], 1));  // the State cascade takes groups 2 and 3 along
    for (const std::string& hash : hashes_) {
        ASSERT_TRUE(NodeOf(hash).on_device);
        ASSERT_EQ(NodeOf(hash).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    }

    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                             GroupDemand{.extent = DenseGrowth{4 * kP}}, RequestProgress{},
                             StateStep{.kind = StateStep::Kind::kKeep}));
    std::vector<CacheCoordinator::EvictedSegment> segments = coordinator_.TakeEvictedSegments();
    ASSERT_EQ(segments.size(), 1u);
    std::vector<std::string> order;
    for (const CacheCoordinator::EvictedBlock& member : segments[0].members) {
        order.push_back(member.key.content_hash);
        EXPECT_TRUE(static_cast<bool>(member.host_block)) << "held on the Host since it was stored there";
    }
    EXPECT_EQ(order, (std::vector<std::string>{hashes_[5], hashes_[4], hashes_[3], hashes_[2]}));
    coordinator_.SettleEvictedSegment(std::move(segments[0]), /*placed_on_host=*/true);

    for (std::size_t page = 0; page < 6; ++page) {
        EXPECT_EQ(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0)), page < 2) << "page " << page;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_[page], 0))) << "page " << page;
    }
    // The last four pages, each a victim chosen for itself.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator_.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks - before.admission.blocks, 4u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    coordinator_.Free(tables);
}

// Without a State group nothing is a checkpoint boundary, and a candidate's set
// is its page and everything under it. A Full group of four blocks per parent
// in a pool of one parent; the finish published p2 and p3 after a chunk end
// published p0 and p1, so p1 is the oldest candidate past p0's suffix order.
TEST(SegmentEvictionWithoutStateTest, AMiddlePageTakesItsDeviceSuffix) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/1, {4});
    BlockPool host_pool(/*num_lcm_blocks=*/1, {4});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> hashes =
        ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 4 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, std::span<const std::string>(hashes).first(2),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * kP, CacheBoundaryKind::kChunk,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/2, /*num_computed_tokens=*/4 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    ASSERT_EQ(pool.NumFreeSlots(0), 0);

    std::vector<BlockTable> next(1);
    ASSERT_TRUE(AdmitForTest(coordinator, next, kP));
    // The plan's drain finds no Host room: the segment is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator.TakeEvictedSegments()) {
        coordinator.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[page], 0)), page == 0) << "page " << page;
    }
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 1u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 2u);
    // No page is left on the Device without its parent.
    EXPECT_EQ(CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator).Size(), 1u);
    coordinator.Free(next);
}

// p3 is the first candidate and its set is p3 alone; p2's set is p2 and p3,
// so it takes p3's set in whole, and p3 still counts as a victim chosen for
// itself, not as p2's stranded suffix.
TEST(SegmentEvictionWithoutStateTest, ASetAbsorbsTheSetsBelowIt) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/1, {4});
    BlockPool host_pool(/*num_lcm_blocks=*/1, {4});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> hashes =
        ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 4 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);

    std::vector<BlockTable> next(1);
    ASSERT_TRUE(AdmitForTest(coordinator, next, 2 * kP));
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[page], 0)), page < 2) << "page " << page;
    }
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 2u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    coordinator.Free(next);
}

// A sliding window's block only shortens which boundaries its prefix resumes
// at, so it leaves alone even when evicting by segments: the finish published
// the two window blocks behind 16 (window 8), the oldest candidates; the fresh
// request's two blocks take their parents, and no Full page goes.
TEST(SegmentEvictionWithoutStateTest, SlidingWindowBlocksStillLeaveAlone) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/8, {1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP},
            {.attn_kind = AttnKind::kSlidingWindow,
             .sliding_window = 2 * kP,
             .cache_blocks_per_lcm_block = 1,
             .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> hashes =
        ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 4 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    ASSERT_EQ(coordinator.GroupPrefixIndex(1).NumEntries(pool), 2);
    const std::vector<CacheBlockRef> held = pool.AcquireBlocks(0, pool.NumEmptyLcmBlocks());

    std::vector<BlockTable> next(2);
    ASSERT_TRUE(AdmitForTest(coordinator, next, kP));
    EXPECT_EQ(coordinator.GroupPrefixIndex(1).NumEntries(pool), 0);
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 4);
    EXPECT_EQ(coordinator.DeviceEvictions().admission.blocks, 2u) << "the two window blocks";
    coordinator.Free(next);
}

// Without a prefix-closed group the Device may hold nothing at a page above a
// checkpoint it holds: a State group keeps no entry for the pages before its
// checkpoint. A segment's climb stops below such a page, which the walk down
// would not enter: the checkpoint at 12 leaves alone. One State group in a
// pool of four blocks; the checkpoint holds one, the test two more, and a
// fresh request needs two.
TEST(SegmentEvictionWithoutAClosedGroupTest, TheClimbStopsBelowAPageTheDeviceDoesNotHold) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/4, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/4, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1}}, kP, pool,
        /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 3 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    coordinator.Free(tables);
    ASSERT_FALSE(tree.Find(hashes[1])->on_device);
    ASSERT_EQ(tree.Find(hashes[0])->checkpoint_boundaries_in_subtree, (std::array<std::int32_t, 2>{1, 0}));
    const std::vector<CacheBlockRef> held = pool.AcquireBlocks(0, pool.NumEmptyLcmBlocks() - 1);

    std::vector<BlockTable> next(1);
    ASSERT_TRUE(AdmitForTest(coordinator, next, kP));
    // The plan's drain finds no Host room: the segment is dropped.
    std::vector<CacheCoordinator::EvictedSegment> segments = coordinator.TakeEvictedSegments();
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].members.size(), 1u);
    coordinator.SettleEvictedSegment(std::move(segments[0]), /*placed_on_host=*/false);
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[2], 0)));
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 1u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    EXPECT_EQ(tree.Size(), 0u);
    coordinator.Free(next);
}

// The same over the pages a sliding window left behind: a window of two pages
// keeps blocks at the last two of four pages, over the checkpoint at 16. The
// fresh request needs three parents, all held but those the cache fills: the
// checkpoint's segment climbs to the window's first page and no further, and
// takes in the two window blocks, chosen alone before it.
TEST(SegmentEvictionWithoutAClosedGroupTest, TheClimbStopsBelowAPageTheWindowLeftBehind) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/8, {1, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(std::vector<CacheGroupSpec>{{.attn_kind = AttnKind::kSlidingWindow,
                                                     .sliding_window = 2 * kP,
                                                     .cache_blocks_per_lcm_block = 1,
                                                     .block_granularity = kP},
                                                    {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1}},
                        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
    const std::vector<std::string> hashes =
        ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 4 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/true);
    coordinator.Free(tables);
    ASSERT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 2);
    ASSERT_FALSE(tree.Find(hashes[1])->on_device);
    ASSERT_TRUE(tree.Find(hashes[2])->on_device);
    const std::vector<CacheBlockRef> held = pool.AcquireBlocks(0, pool.NumEmptyLcmBlocks());

    std::vector<BlockTable> next(2);
    ASSERT_TRUE(AdmitForTest(coordinator, next, kP));
    // The plan's drain finds no Host room: the segment is dropped.
    std::vector<CacheCoordinator::EvictedSegment> segments = coordinator.TakeEvictedSegments();
    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].members.size(), 3u);
    coordinator.SettleEvictedSegment(std::move(segments[0]), /*placed_on_host=*/false);
    EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
    EXPECT_EQ(coordinator.GroupPrefixIndex(1).NumEntries(pool), 0);
    // The two window blocks and the checkpoint.
    const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
    EXPECT_EQ(evictions.admission.blocks, 3u);
    EXPECT_EQ(evictions.stranded_suffix.blocks, 0u);
    EXPECT_EQ(tree.Size(), 0u);
    coordinator.Free(next);
}

// Two coordinators fed the same publications and admissions evict the same
// segments in the same order: every decision follows the tree's insertion order
// and the planner's keys, never an address. K3 shape in a pool of 16 parents,
// the Full group's pages dealt over the parameter's KV-parallel shards;
// sessions of 18 tokens fork from the first at varying pages.
class SegmentEvictionMirrorTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, SegmentEvictionMirrorTest, ::testing::Values(1, 2, 4));

TEST_P(SegmentEvictionMirrorTest, EvictionSetsMatchOnTwoMirroredCoordinators) {
    constexpr std::int32_t kP = 4;
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull,
         .cache_blocks_per_lcm_block = 4,
         .block_granularity = kP,
         .shard_count = GetParam()},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    BlockPool first_pool(/*num_lcm_blocks=*/16, {4, 1, 1, 1});
    BlockPool second_pool(/*num_lcm_blocks=*/16, {4, 1, 1, 1});
    BlockPool first_host(/*num_lcm_blocks=*/1, {4, 1, 1, 1});
    BlockPool second_host(/*num_lcm_blocks=*/1, {4, 1, 1, 1});
    std::array<CacheCoordinator, 2> coordinators{
        MakeCoordinator(specs, kP, first_pool, /*enable_l3_storage=*/false, &first_host, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true),
        MakeCoordinator(specs, kP, second_pool, /*enable_l3_storage=*/false, &second_host, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true)};
    const std::array<BlockPool*, 2> pools{&first_pool, &second_pool};
    const std::vector<std::int32_t> trunk = MakeTokens(18);
    for (std::int32_t session = 0; session < 12; ++session) {
        std::vector<std::int32_t> tokens = MakeTokens(18, 100 * (session + 1));
        std::copy_n(trunk.begin(), session % 4 * kP, tokens.begin());
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < 4; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        const std::vector<std::string> hashes = ContentHashes(pages);
        const TailPublication tail{.tail_tokens = 2,
                                   .hash = HashPrefixPage(token_span(tokens).subspan(4 * kP), hashes.back())};
        std::array<std::optional<CacheCoordinator::AdmissionResult>, 2> admitted;
        for (std::size_t c = 0; c < 2; ++c) {
            std::vector<BlockTable> tables(4);
            admitted[c] = AdmitForTest(coordinators[c], tables, 18);
            if (admitted[c]) {
                coordinators[c].CacheCompletedBlocks(
                    tables,
                    RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = hashes,
                                                                      .first_new_prefix_page = 0,
                                                                      .tail = tail,
                                                                      .publish_state = true,
                                                                      .boundary_kind = CacheBoundaryKind::kEndpoint},
                                    .num_computed_tokens = 18},
                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinators[c]));
            }
            coordinators[c].Free(tables);
        }
        ASSERT_EQ(admitted[0].has_value(), admitted[1].has_value()) << "session " << session;
        if (admitted[0]) {
            EXPECT_EQ(admitted[0]->new_page_ids, admitted[1]->new_page_ids) << "session " << session;
        }
        const CacheCoordinator::DeviceEvictionStats& lhs = coordinators[0].DeviceEvictions();
        const CacheCoordinator::DeviceEvictionStats& rhs = coordinators[1].DeviceEvictions();
        EXPECT_EQ(lhs.admission.blocks, rhs.admission.blocks) << "session " << session;
        EXPECT_EQ(lhs.stranded_suffix.blocks, rhs.stranded_suffix.blocks) << "session " << session;
        EXPECT_EQ(lhs.request_reclaim.blocks, rhs.request_reclaim.blocks) << "session " << session;
        for (std::size_t group = 0; group < specs.size(); ++group) {
            EXPECT_EQ(coordinators[0].GroupPrefixIndex(static_cast<std::int32_t>(group)).NumEntries(*pools[0]),
                      coordinators[1].GroupPrefixIndex(static_cast<std::int32_t>(group)).NumEntries(*pools[1]))
                << "session " << session;
        }
        EXPECT_EQ(pools[0]->NumEmptyLcmBlocks(), pools[1]->NumEmptyLcmBlocks()) << "session " << session;
    }
    EXPECT_GT(coordinators[0].DeviceEvictions().admission.blocks, 0u) << "the pool forced segment evictions";
}

// ---------------------------------------------------------------------------
// Host placement planned per segment (AcquireHostSegments): one joint fit
// over every segment, a segment placed whole or not at all, and victims taken
// whole -- a Host checkpoint boundary with the Host pages only it depends on,
// or a page no Host checkpoint boundary depends on -- dead entries first, then
// duplicates and checkpoint boundaries a Device checkpoint boundary supersedes,
// then Host-only cache. The K3 shape of BoundaryTreeCoordinatorFixture; Host
// content is published on the Device first and then copied to the Host, so it
// hangs under the tree nodes the publication linked. Each test holds the Host
// pool's free capacity, so a demand must be funded by victims.
// ---------------------------------------------------------------------------
class HostSegmentPlanTest : public BoundaryTreeCoordinatorFixture {
protected:
    using HostSegmentRows = CacheCoordinator::HostSegmentRows;

    HostSegmentPlanTest() : HostSegmentPlanTest(/*full_shard_count=*/1) {}
    explicit HostSegmentPlanTest(std::int32_t full_shard_count)
        : BoundaryTreeCoordinatorFixture(/*write_back_evictions=*/true, full_shard_count, /*num_snapshot_pages=*/0) {}

    // The session over `tokens` as a write-back leaves it: published on the
    // Device (PublishFinal), every entry it published copied to the Host,
    // shallowest boundary first -- one the Host holds already stays as it is
    // -- and then evicted from the Device. Returns the tail.
    std::optional<TailPublication> WriteBack(const std::vector<std::int32_t>& tokens) {
        std::vector<BlockTable> tables(4);
        const std::optional<TailPublication> tail = PublishFinal(tables, tokens);
        coordinator_.Free(tables);
        std::vector<std::pair<std::string, std::int32_t>> boundaries;
        for (const std::string& hash : hashes_) {
            boundaries.emplace_back(hash, 0);
        }
        if (tail) {
            boundaries.emplace_back(tail->hash, tail->tail_tokens);
        }
        for (const auto& [hash, tail_tokens] : boundaries) {
            coordinator_.ForEachBoundaryKey(Key(hash, 0), tail_tokens, [&](std::size_t, const CacheKey& key) {
                if (coordinator_.GroupPrefixIndex(key.group_id).Contains(pool_, key) &&
                    !coordinator_.ContainsHostCachedBlock(key)) {
                    StoreOnHost(key);
                }
            });
        }
        for (const auto& [hash, tail_tokens] : boundaries) {
            coordinator_.ForEachBoundaryKey(Key(hash, 0), tail_tokens, [&](std::size_t, const CacheKey& key) {
                // An earlier eviction's cascade may have taken it.
                if (coordinator_.GroupPrefixIndex(key.group_id).Contains(pool_, key)) {
                    EvictOnDevice(key);
                }
            });
        }
        return tail;
    }

    // A session of four pages starting at token `start`, published on the
    // Device with its checkpoints at the last page; the checkpoints of
    // `state_groups` are copied to the Host and, unless the Device keeps its
    // own, evicted from the Device, the KV staying (a Host checkpoint on the
    // Device's last page). Returns the last page's hash.
    std::string CheckpointOnHost(std::int32_t start, const std::vector<std::uint32_t>& state_groups,
                                 bool device_keeps_checkpoint) {
        std::vector<BlockTable> tables(4);
        PublishFinal(tables, MakeTokens(4 * kP, start));
        coordinator_.Free(tables);
        for (const std::uint32_t group : state_groups) {
            StoreOnHost(Key(hashes_.back(), group));
        }
        if (!device_keeps_checkpoint) {
            // The other State groups' go along; the page stays.
            EvictOnDevice(Key(hashes_.back(), 1));
        }
        return hashes_.back();
    }

    const std::vector<std::uint32_t> state_group_ids_{1, 2, 3};
    const std::vector<std::int32_t> zero_buckets_{0, 0, 0};
};

class HostKvDuplicateReclaimTest : public HostSegmentPlanTest {
protected:
    void PublishKvCopiesAndHostOnlyCheckpoints() {
        const auto before = coordinator_.NumHostCachedBlocks();
        CheckpointOnHost(1, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
        for (const std::string& hash : hashes_) {
            StoreOnHost(Key(hash, 0));
        }
        ASSERT_EQ(coordinator_.NumHostCachedBlocks(), before + 7);
    }

    CacheCoordinator::HostAllocationBatch Allocate(std::span<const std::uint32_t> groups) {
        const std::vector<std::int32_t> buckets(groups.size(), 0);
        const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = groups.size()}};
        return coordinator_.AcquireHostSegments(groups, buckets, segments);
    }

    void ExpectHostCheckpoints() {
        for (const std::uint32_t group : state_group_ids_) {
            EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), group))) << group;
            EXPECT_FALSE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes_.back(), group))) << group;
        }
    }
};

TEST_F(HostKvDuplicateReclaimTest, ReclaimsKvWithoutTheOnlyHostCheckpointsAndStillLoadsThem) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_.front(), 0)));
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.device_resident, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
    ExpectHostCheckpoints();

    auto probe = coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = MakeTokens(16), .max_tokens = 16});
    EXPECT_EQ(probe.device.full_page_tokens, 16);
    EXPECT_EQ(probe.host.num_common_tokens, 16);
    std::vector<BlockTable> tables(4);
    const auto admission = AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{4}});
    ASSERT_TRUE(admission);
    ASSERT_EQ(admission->load_pairs.size(), 3u);
    for (const auto& pair : admission->load_pairs) {
        EXPECT_GT(pair.group_id, 0u) << "the Device KV is reused; only Host checkpoints load";
    }
    coordinator_.Free(tables);
}

TEST_F(HostKvDuplicateReclaimTest, PinnedCheckpointDoesNotPreventReclaimingItsKvCopy) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, PinnedKvCopyIsSkipped) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheBlockRef pinned = coordinator_.AcquireHostCachedBlock(Key(hashes_.front(), 0));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.front(), 0)));
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_[1], 0)));
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, AWholeKvParentCanFundAnotherGroupWithoutDroppingState) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{1});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 4u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
    for (const std::string& hash : hashes_) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hash, 0)));
    }
    ExpectHostCheckpoints();
    const auto probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = MakeTokens(16), .max_tokens = 16});
    EXPECT_EQ(probe.host.num_common_tokens, 16);
}

TEST_F(HostKvDuplicateReclaimTest, UnneededCopiesAreRestoredWhenTheSegmentCannotFit) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(state_group_ids_);
    EXPECT_EQ(batch.stats.allocated, 0u) << "one KV parent cannot fund three State parents";
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 7);
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, FreeCapacityKeepsAllDuplicates) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 7);
}

TEST_F(HostKvDuplicateReclaimTest, EachPlanRechecksAnAncestorStoreThatStartsAndSettles) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(kP, 1));
    const std::string ancestor = hashes_.front();
    coordinator_.Free(tables);
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheKey first_kv = Key(hashes_.front(), 0);
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    // Populate a scan's answers before an ancestor begins its write-back.
    auto first = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(first.stats.allocated, 1u);
    coordinator_.CacheHostBlock(first.blocks[0], first_kv, std::nullopt);
    first.blocks.clear();

    CacheCoordinator::EvictedSegment pending;
    ASSERT_TRUE(CacheCoordinatorTestAccess::RecordDeviceEviction(coordinator_, Key(ancestor, 1), pending));
    ASSERT_FALSE(pending.members.empty());
    const auto protected_plan = Allocate(std::array<std::uint32_t, 1>{0});
    EXPECT_EQ(protected_plan.stats.allocated, 0u);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(first_kv));

    coordinator_.SettleEvictedSegment(std::move(pending), /*placed_on_host=*/false);
    const auto settled = Allocate(std::array<std::uint32_t, 1>{0});
    EXPECT_EQ(settled.stats.allocated, 1u);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(first_kv));
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, OlderHostStateLeavesBeforeNewerUnheldDeviceKvCopies) {
    const std::string old = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    PublishKvCopiesAndHostOnlyCheckpoints();
    const std::vector<std::string> newer = hashes_;
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{1});
    ASSERT_EQ(batch.stats.allocated, 1u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(old, group)));
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(newer.back(), group)));
    }
    for (const std::string& hash : newer) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hash, 0)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u);
}

TEST_F(HostKvDuplicateReclaimTest, HeldDeviceKvCopiesLeaveBeforeOlderHostState) {
    const std::string old = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    PublishKvCopiesAndHostOnlyCheckpoints();
    std::vector<CacheBlockRef> request_history;
    for (const std::string& hash : hashes_) {
        request_history.push_back(coordinator_.GroupPrefixIndex(0).Find(pool_, Key(hash, 0)));
    }
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{1});
    ASSERT_EQ(batch.stats.allocated, 1u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(old, group)));
    }
    ExpectHostCheckpoints();
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 4u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
}

TEST_F(HostKvDuplicateReclaimTest, AnOlderUnheldDeviceKvCopyLeavesBeforeNewerHostState) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const std::vector<std::string> older = hashes_;
    const std::string newer = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{1});
    ASSERT_EQ(batch.stats.allocated, 1u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(older.back(), group)));
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(newer, group)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 4u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
}

TEST_F(HostKvDuplicateReclaimTest, ADeviceHeldCopyKeepsPriorityAmongUnheldCopiesAndOlderState) {
    const std::string older = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheKey held_key = Key(hashes_[2], 0);
    const CacheBlockRef request_history = coordinator_.GroupPrefixIndex(0).Find(pool_, held_key);
    ASSERT_TRUE(request_history);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    for (const std::string& hash : hashes_) {
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(hash, 0)), hash != held_key.content_hash);
    }
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(older, group)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, OldStateCapacityCanBeReusedByKvInTheSharedPool) {
    const std::string old = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    PublishKvCopiesAndHostOnlyCheckpoints();
    const std::vector<std::string> newer = hashes_;
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(old, group)));
    }
    for (const std::string& hash : newer) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hash, 0)));
    }
    EXPECT_EQ(coordinator_.ProbePrefix(newer).host.num_common_tokens, 4 * kP);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u);
}

TEST_F(HostKvDuplicateReclaimTest, ADeviceKvSuffixBeyondAGapDoesNotReplaceHostKv) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    EvictOnDevice(Key(hashes_.front(), 0));
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    EXPECT_EQ(batch.stats.allocated, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 7);
    const auto probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = MakeTokens(16), .max_tokens = 16});
    EXPECT_EQ(probe.host.num_common_tokens, 16) << "Host KV above the Device gap stays contiguous";
}

TEST_F(HostKvDuplicateReclaimTest, RepublishingTheDeviceGapMakesItsHostCopiesReclaimable) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    EvictOnDevice(Key(hashes_.front(), 0));
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(4 * kP, 1));
    coordinator_.Free(tables);
    EvictOnDevice(Key(hashes_.back(), 1));
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_.front(), 0)));
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, ARepublishedHostCopyCanBeReclaimedAgain) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    const CacheKey key = Key(hashes_.front(), 0);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, key));
    StoreOnHost(key);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(key));
    ExpectHostCheckpoints();
}

TEST_F(HostKvDuplicateReclaimTest, CacheClearAndRepublicationKeepReclaimWorking) {
    PublishKvCopiesAndHostOnlyCheckpoints();
    ASSERT_TRUE(coordinator_.ClearCache());
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    PublishKvCopiesAndHostOnlyCheckpoints();
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto batch = Allocate(std::array<std::uint32_t, 1>{0});
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
    ExpectHostCheckpoints();
}

TEST_F(HostSlidingSuffixTest, IndependentKvReclaimKeepsThePagesNeededByTheHostWindow) {
    const CacheBlockRef checkpoint = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 1));
    HoldHostCapacity();
    const auto before = coordinator_.ProbePromptPrefix(Prompt());
    ASSERT_EQ(before.host.num_common_tokens, 16);
    ASSERT_EQ(before.device.full_page_tokens, 12);
    const std::array<std::uint32_t, 1> groups{0};
    const std::array<std::int32_t, 1> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const auto batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_.front(), 0)));
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 0)))
        << "the Host-only window requires Full KV below the Device prefix";
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 1)));
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 2)));
    const auto after = coordinator_.ProbePromptPrefix(Prompt());
    EXPECT_EQ(after.host.num_common_tokens, before.host.num_common_tokens);
    EXPECT_EQ(after.device.full_page_tokens, before.device.full_page_tokens);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 1u);
}

// Free capacity places every segment: no victim is even looked for.
TEST_F(HostSegmentPlanTest, FreeCapacityPlacesEverySegmentWithoutEvicting) {
    WriteBack(MakeTokens(18));
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 8);
    const std::vector<std::uint32_t> groups{1, 2, 3, 0, 0, 0, 0};
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3},
                              HostSegmentRows{.first_row = 3, .past_end_row = 7}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 7u);
    EXPECT_TRUE(std::ranges::all_of(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
}

TEST_F(HostSegmentPlanTest, KvTraversalDoesNotRefreshAnOlderHostCheckpoint) {
    WriteBack(MakeTokens(4 * kP, 1));
    const std::string older_checkpoint = hashes_.back();
    WriteBack(MakeTokens(4 * kP, 100));
    const std::string newer_checkpoint = hashes_.back();
    WriteBack(MakeTokens(5 * kP, 1));
    const std::vector<std::string> longer_prefix = hashes_;
    // Resuming beyond the older checkpoint uses its KV pages and only the
    // longer prefix's State. The older State's last access stays unchanged.
    auto hit = CacheCoordinatorTestAccess::MatchPrefix(coordinator_, longer_prefix);
    ASSERT_EQ(hit.host.num_common_tokens, 5 * kP);
    ASSERT_EQ(hit.device.num_common_tokens, 0);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = state_group_ids_.size()}};
    const auto batch = coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);
    ASSERT_EQ(batch.stats.allocated, state_group_ids_.size());
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_checkpoint, group)));
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(newer_checkpoint, group)));
    }
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(older_checkpoint, 0)));
    EXPECT_EQ(coordinator_.ProbePrefix(longer_prefix).host.num_common_tokens, 5 * kP);
}

TEST_F(HostSegmentPlanTest, KvPinDoesNotPinAnUnusedHostCheckpoint) {
    WriteBack(MakeTokens(4 * kP, 1));
    const std::vector<std::string> prefix = hashes_;
    const CacheBlockRef kv_load = coordinator_.AcquireHostCachedBlock(Key(prefix.back(), 0));
    ASSERT_TRUE(kv_load);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = state_group_ids_.size()}};
    const auto batch = coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);
    ASSERT_EQ(batch.stats.allocated, state_group_ids_.size());
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(prefix.back(), group)));
    }
    for (const std::string& hash : prefix) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hash, 0)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, state_group_ids_.size());
    EXPECT_EQ(coordinator_.ProbePrefix(prefix).host.num_common_tokens, 0);
}

TEST_F(HostSegmentPlanTest, AStatePinStillProtectsTheWholeHostCheckpoint) {
    WriteBack(MakeTokens(4 * kP, 1));
    const CacheBlockRef kv_load = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 0));
    const CacheBlockRef state_load = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 2));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = state_group_ids_.size()}};
    const auto batch = coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);
    EXPECT_EQ(batch.stats.allocated, 0u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), group)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
}

TEST_F(HostSegmentPlanTest, UnneededStateReclaimBehindAKvPinIsRolledBack) {
    WriteBack(MakeTokens(4 * kP, 1));
    const CacheBlockRef kv_load = coordinator_.AcquireHostCachedBlock(Key(hashes_.back(), 0));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array<std::uint32_t, 4> groups{1, 1, 2, 3};
    const std::array<std::int32_t, 4> buckets{};
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = groups.size()}};
    const auto batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
    EXPECT_EQ(batch.stats.allocated, 0u);
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), group)));
    }
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_.back(), 0)));
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 0u);
}

// Every Host entry is pinned by a load, so no victim can fund the segment: it
// is dropped and nothing changes -- no entry leaves, the one empty parent
// stays the next one the pool hands out.
TEST_F(HostSegmentPlanTest, NoEvictionWhenNoSegmentCanFit) {
    const TailPublication tail = *WriteBack(MakeTokens(18));
    std::vector<CacheBlockRef> loads;
    for (const std::string& hash : hashes_) {
        loads.push_back(coordinator_.AcquireHostCachedBlock(Key(hash, 0)));
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        loads.push_back(coordinator_.AcquireHostCachedBlock(Key(tail.hash, group)));
    }
    ASSERT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 8);
    // The session took parents 1-5; parent 32 stays empty.
    HoldFreeCapacity(host_pool_, /*spare_parents=*/1);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_TRUE(
        std::ranges::none_of(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    EXPECT_EQ(batch.stats.requested, 3u);
    EXPECT_EQ(batch.stats.allocated, 0u);
    EXPECT_EQ(batch.stats.unallocated, 3u);
    const CacheCoordinator::HostEvictionStats& host_evictions = coordinator_.HostEvictions();
    EXPECT_EQ(host_evictions.dead_entry.blocks + host_evictions.duplicate.blocks + host_evictions.superseded.blocks +
                  host_evictions.lru.blocks,
              0u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
    EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(host_pool_.AcquireBlock(1)->Location().lcm_block_id, 32);
}

// A KV-only demand of one Full parent, the oldest Host checkpoint boundary A's:
// its three checkpoints leave with A's KV -- its tail block and its four pages,
// which empty a Full parent of their own -- never one checkpoint alone; B
// stays.
TEST_F(HostSegmentPlanTest, KvOnlyDemandTakesWholeSegments) {
    const TailPublication a_tail = *WriteBack(MakeTokens(18));
    const std::vector<std::string> a_hashes = hashes_;
    const TailPublication b_tail = *WriteBack(MakeTokens(18, 100));
    const std::vector<std::string> b_hashes = hashes_;
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::vector<std::uint32_t> groups(4, 0);
    const std::vector<std::int32_t> buckets(4, 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 4}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 4u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(a_tail.hash, group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_tail.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(a_hashes[page], 0))) << "page " << page;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_hashes[page], 0))) << "page " << page;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 8u) << "A's four pages, its tail block and checkpoints";
    // A's tail block left a hole in the Full parent it shared with B's pages;
    // the rest open the first parent A's eviction emptied, its group-1
    // checkpoint's.
    std::vector<CacheBlockLocation> placed;
    for (const CacheBlockRef& block : batch.blocks) {
        placed.push_back(block->Location());
    }
    EXPECT_EQ(placed, (std::vector<CacheBlockLocation>{{2, 0}, {3, 0}, {3, 1}, {3, 2}}));
}

// The victim classes in order, one call per victim, each freeing exactly the
// parents its call needs: the dead entry -- a checkpoint boundary missing a
// State group's checkpoint, the newest Host entry -- then the duplicates of
// what the Device resumes, the newest Device copy first though its Host copy is
// the newer, and the Host-only checkpoint boundary last though its Host copy is
// the oldest.
TEST_F(HostSegmentPlanTest, DeadEntriesGoFirstThenNewestDeviceDuplicatesThenHostOnly) {
    const std::string host_only = CheckpointOnHost(300, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    const std::string older_duplicate = CheckpointOnHost(200, {1, 2, 3}, /*device_keeps_checkpoint=*/true);
    const std::string newer_duplicate = CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/true);
    const std::string partial = CheckpointOnHost(400, {1, 2}, /*device_keeps_checkpoint=*/false);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const auto on_host = [&](const std::string& hash) {
        return coordinator_.ContainsHostCachedBlock(Key(hash, 1)) || coordinator_.ContainsHostCachedBlock(Key(hash, 2));
    };

    std::vector<CacheCoordinator::HostAllocationBatch> batches;
    const std::array two_rows{HostSegmentRows{.first_row = 0, .past_end_row = 2}};
    batches.push_back(coordinator_.AcquireHostSegments(std::span(state_group_ids_).first(2),
                                                       std::span(zero_buckets_).first(2), two_rows));
    EXPECT_EQ(batches.back().stats.allocated, 2u);
    EXPECT_FALSE(on_host(partial));
    EXPECT_TRUE(on_host(newer_duplicate) && on_host(older_duplicate) && on_host(host_only));
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks, 2u) << "the partial checkpoint boundary's two";

    const std::array three_rows{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    for (const auto& [expected_gone, duplicate] :
         {std::pair{&newer_duplicate, true}, std::pair{&older_duplicate, true}, std::pair{&host_only, false}}) {
        SCOPED_TRACE(*expected_gone);
        const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
        batches.push_back(coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, three_rows));
        const CacheCoordinator::HostEvictionStats& after = coordinator_.HostEvictions();
        EXPECT_EQ(batches.back().stats.allocated, 3u);
        EXPECT_FALSE(on_host(*expected_gone));
        EXPECT_EQ(after.duplicate.blocks - before.duplicate.blocks, duplicate ? 3u : 0u);
        EXPECT_EQ(after.duplicate.device_resident - before.duplicate.device_resident, duplicate ? 3u : 0u);
        EXPECT_EQ(after.lru.blocks - before.lru.blocks, duplicate ? 0u : 3u);
    }
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
}

// A turn ends at 16 and the next at 18; the Device keeps both checkpoint
// boundaries, the Host has copies of both. The second publication touched the
// page at 16 too, so both Device copies date from it: at one epoch the
// shallower duplicate goes first, keeping the longer frontier -- the
// checkpoints at 16, not the checkpoint boundary at 18.
TEST_F(HostSegmentPlanTest, AtOneDeviceEpochTheShallowerDuplicateGoesFirst) {
    std::vector<BlockTable> earlier_tables(4);
    PublishFinal(earlier_tables, MakeTokens(16));
    coordinator_.Free(earlier_tables);
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    coordinator_.Free(tables);
    for (const auto& [hash, tail_tokens] : {std::pair{hashes_[3], 0}, std::pair{tail.hash, tail.tail_tokens}}) {
        coordinator_.ForEachBoundaryKey(Key(hash, 0), tail_tokens,
                                        [&](std::size_t, const CacheKey& key) { StoreOnHost(key); });
    }
    const auto device_epoch = [&](const CacheKey& key) {
        return coordinator_.GroupPrefixIndex(key.group_id)
            .MetadataFor(pool_, *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, key.group_id, key))
            ->last_access_epoch;
    };
    ASSERT_EQ(device_epoch(Key(hashes_[3], 0)), device_epoch(Key(tail.hash, 1)));
    ASSERT_LT(device_epoch(Key(hashes_[3], 1)), device_epoch(Key(hashes_[3], 0)));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_[3], group))) << "group " << group;
    }
    // The three checkpoints at 16.
    EXPECT_EQ(coordinator_.HostEvictions().duplicate.blocks, 3u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate.device_resident, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
    }
}

TEST_F(HostSegmentPlanTest, ADeeperDeviceCheckpointDoesNotAgeARecentHostCheckpoint) {
    const TailPublication cold_tail = *WriteBack(MakeTokens(18, 100));
    const std::vector<std::int32_t> continuation = MakeTokens(26);
    WriteBack({continuation.begin(), continuation.begin() + 16});
    const std::vector<std::string> shorter_hashes = hashes_;
    const std::string shorter_checkpoint = hashes_.back();
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, continuation);
    coordinator_.Free(tables);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = state_group_ids_.size()}};
    const auto batch = coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);
    ASSERT_EQ(batch.stats.allocated, state_group_ids_.size());
    for (const std::uint32_t group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(shorter_checkpoint, group)));
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(cold_tail.hash, group)));
    }
    // The request can branch immediately after the earlier checkpoint. The
    // deeper State follows other tokens and cannot restore this boundary.
    std::vector<std::int32_t> branch{continuation.begin(), continuation.begin() + 16};
    branch.push_back(999);
    const auto probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = shorter_hashes, .tokens = branch, .max_tokens = 16});
    EXPECT_EQ(probe.host.num_common_tokens, 16);
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
}

// An older checkpoint still leaves by LRU when a newer Device checkpoint
// follows it. The newer unrelated Host session and the Device resume stay.
TEST_F(HostSegmentPlanTest, AnOlderHostCheckpointLeavesByLruDespiteADeeperDeviceCheckpoint) {
    const std::vector<std::int32_t> a = MakeTokens(26);
    WriteBack({a.begin(), a.begin() + 16});
    const std::string earlier = hashes_.back();
    const TailPublication b_tail = *WriteBack(MakeTokens(18, 100));
    const std::vector<std::string> b_hashes = hashes_;
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, a);
    coordinator_.Free(tables);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(earlier, group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_tail.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_[page], 0))) << "page " << page;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_hashes[page], 0))) << "page " << page;
    }
    // A's four pages and three checkpoints; the pages are on the Device too,
    // its checkpoints are not.
    const CacheCoordinator::HostEvictionStats& host_evictions = coordinator_.HostEvictions();
    EXPECT_EQ(host_evictions.superseded.blocks, 0u);
    EXPECT_EQ(host_evictions.lru.blocks, 7u);
    EXPECT_EQ(host_evictions.lru.device_resident, 4u);
    const CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = a, .max_tokens = 26});
    EXPECT_EQ(probe.device.num_common_tokens, 26);
}

// A tail and the next full page are siblings in the boundary tree. The
// Device's deeper checkpoint may follow different tokens inside that page,
// so it cannot supersede the newer Host-only tail checkpoint.
TEST_F(HostSegmentPlanTest, ATailOnAnotherDeviceBranchIsNotSuperseded) {
    const std::vector<std::int32_t> older = MakeTokens(18, 100);
    const TailPublication older_tail = *WriteBack(older);
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    const TailPublication tail = *WriteBack(tokens);
    const std::vector<std::string> prefix_hashes = hashes_;
    std::vector<std::int32_t> branch = MakeTokens(26);
    branch[16] = 999;
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, branch);
    coordinator_.Free(tables);
    ASSERT_EQ(coordinator_.ProbePromptPrefix({.prefix_hashes = prefix_hashes, .tokens = tokens, .max_tokens = 18})
                  .host.num_common_tokens,
              18);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_tail.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
    EXPECT_EQ(coordinator_.ProbePromptPrefix({.prefix_hashes = prefix_hashes, .tokens = tokens, .max_tokens = 18})
                  .host.num_common_tokens,
              18);
}

// An aligned checkpoint published by another request is not evidence that
// the newer Host tail continued along the Device branch. The old unrelated
// session leaves by LRU when it alone funds the demand.
TEST_F(HostSegmentPlanTest, AnAlignedCheckpointDoesNotHideATailOnAnotherDeviceBranch) {
    const std::vector<std::int32_t> older = MakeTokens(18, 100);
    const TailPublication older_tail = *WriteBack(older);
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    WriteBack({tokens.begin(), tokens.begin() + 16});
    const TailPublication tail = *WriteBack(tokens);
    const std::vector<std::string> prefix_hashes = hashes_;
    std::vector<std::int32_t> branch = MakeTokens(26);
    branch[16] = 999;
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, branch);
    coordinator_.Free(tables);
    // The live request keeps these Device pages, so their Host copies fund
    // the demand before either independent Host-only tail can leave by LRU.
    std::vector<CacheBlockRef> request_history;
    for (const std::string& hash : prefix_hashes) {
        request_history.push_back(coordinator_.GroupPrefixIndex(0).Find(pool_, Key(hash, 0)));
    }
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::vector<std::uint32_t> groups{0, 0, 0, 0, 1, 2, 3};
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = groups.size()}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, groups.size());
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_tail.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks, 0u)
        << "the older session alone funds the demand; unneeded duplicates are restored";
    EXPECT_EQ(coordinator_.ProbePromptPrefix({.prefix_hashes = prefix_hashes, .tokens = tokens, .max_tokens = 18})
                  .host.num_common_tokens,
              18);
}

// A request actually resumes from the Host tail and publishes its deeper
// checkpoint under the admission's epoch. That proves the continuation even
// though the tail and the new full pages are siblings in the boundary tree.
TEST_F(HostSegmentPlanTest, AResumedHostTailIsSupersededByItsDeviceContinuation) {
    const TailPublication older = *WriteBack(MakeTokens(18, 100));
    const std::vector<std::int32_t> tokens = MakeTokens(26);
    const TailPublication tail = *WriteBack({tokens.begin(), tokens.begin() + 18});
    const std::vector<std::string> input_hashes = hashes_;
    {
        std::vector<BlockTable> tables(4);
        const auto admitted = AdmitForTest(
            coordinator_, tables,
            coordinator_.ProbePromptPrefix({.prefix_hashes = input_hashes, .tokens = tokens, .max_tokens = 18}),
            GroupDemand{.extent = DenseGrowth{8}});
        ASSERT_TRUE(admitted.has_value());
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < 6; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        hashes_ = ContentHashes(pages);
        const TailPublication end{
            .tail_tokens = 2,
            .hash = HashPrefixPage(token_span(tokens).subspan(24), hashes_.back()),
        };
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = hashes_,
                                              .tail = end,
                                              .publish_state = true,
                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
            .num_computed_tokens = 26,
        };
        coordinator_.CacheCompletedBlocks(tables, progress, admitted->access_epoch);
        coordinator_.Free(tables);
    }
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(older.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 8u);
    EXPECT_EQ(coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = tokens, .max_tokens = 26})
                  .device.num_common_tokens,
              26);
}

// The next prompt can share the aligned checkpoint and diverge before the
// deeper Device frontier. Its KV remains usable while Host State ages by LRU.
TEST_F(HostSegmentPlanTest, AnAlignedHostCheckpointSurvivesADeviceContinuationAndAnOlderVictim) {
    const std::string older = CheckpointOnHost(100, {1, 2, 3}, false);
    const std::string useful = CheckpointOnHost(1, {1, 2, 3}, false);
    const std::vector<std::int32_t> tokens = MakeTokens(26);
    const std::vector<std::string> input_hashes = hashes_;
    {
        std::vector<BlockTable> tables(4);
        const auto admitted = AdmitForTest(
            coordinator_, tables,
            coordinator_.ProbePromptPrefix({.prefix_hashes = input_hashes, .tokens = tokens, .max_tokens = 16}),
            GroupDemand{.extent = DenseGrowth{10}});
        ASSERT_TRUE(admitted.has_value());
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < 6; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        hashes_ = ContentHashes(pages);
        const TailPublication end{.tail_tokens = 2,
                                  .hash = HashPrefixPage(token_span(tokens).subspan(24), hashes_.back())};
        const RequestProgress progress{.completed_pages = CompletedPages{.prefix_hashes = hashes_,
                                                                         .tail = end,
                                                                         .publish_state = true,
                                                                         .boundary_kind = CacheBoundaryKind::kEndpoint},
                                       .num_computed_tokens = 26};
        coordinator_.CacheCompletedBlocks(tables, progress, admitted->access_epoch);
        coordinator_.Free(tables);
    }
    if (coordinator_.GroupPrefixIndex(1).Contains(pool_, Key(useful, 1))) {
        EvictOnDevice(Key(useful, 1));
    }
    std::vector<std::int32_t> future(tokens.begin(), tokens.begin() + 16);
    future.push_back(9999);
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = input_hashes, .tokens = future, .max_tokens = 16};
    const auto before = coordinator_.ProbePromptPrefix(prompt);
    ASSERT_EQ(std::max(before.device.num_common_tokens, before.host.num_common_tokens), 16);
    ASSERT_FALSE(coordinator_.GroupPrefixIndex(1).Contains(pool_, Key(useful, 1)));
    ASSERT_TRUE(coordinator_.ContainsHostCachedBlock(Key(useful, 1)));
    HoldFreeCapacity(host_pool_, 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const auto allocation = coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);
    ASSERT_EQ(allocation.stats.allocated, 3u);
    const auto after = coordinator_.ProbePromptPrefix(prompt);
    const auto hit_after = std::max(after.device.num_common_tokens, after.host.num_common_tokens);
    const auto kv_after = std::max(after.device.kv_tokens, after.host.kv_tokens);
    EXPECT_EQ(kv_after, 16);
    EXPECT_EQ(hit_after, 16);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older, 1)));
    for (const auto group : state_group_ids_) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(useful, group)));
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u);
}

// The Device's KV with the Host's checkpoint is no duplicate (the Device
// alone cannot resume there) and nothing supersedes it: it is Host-only
// cache, and the older Host-only session goes first. The Host checkpoint
// still resumes on the Device's KV.
TEST_F(HostSegmentPlanTest, DeviceKvWithHostCheckpointIsNotADuplicate) {
    const std::vector<std::int32_t> older = MakeTokens(16, 100);
    WriteBack(older);
    const std::vector<std::string> older_hashes = hashes_;
    const std::string checkpoint = CheckpointOnHost(1, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    const std::vector<std::string> checkpoint_pages = hashes_;
    ASSERT_EQ(coordinator_.DeviceBoundaryResidency(Key(checkpoint, 0), /*tail_tokens=*/0),
              CacheCoordinator::BoundaryResidency::kPartial);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(checkpoint, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_hashes[3], group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 7u) << "the older session's pages and checkpoints";
    const CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(
        {.prefix_hashes = checkpoint_pages, .tokens = MakeTokens(16, 1), .max_tokens = 16});
    EXPECT_EQ(probe.device.full_page_tokens, 16);
    EXPECT_EQ(probe.host.num_common_tokens, 16);
}

// The Host holds a checkpoint at 16, the older, and a deeper one at 22 on the
// same chain: the page at 16 still serves the deeper checkpoint boundary, so
// only the three checkpoints at 16 leave, and the Host still resumes at 22.
TEST_F(HostSegmentPlanTest, ANestedCheckpointLeavesWithoutItsPages) {
    const std::vector<std::int32_t> tokens = MakeTokens(22);
    WriteBack({tokens.begin(), tokens.begin() + 16});
    const TailPublication tail = *WriteBack(tokens);
    ASSERT_EQ(NodeOf(hashes_[3]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 2}));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(hashes_[3], group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u) << "the three checkpoints at 16";
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 9) << "five pages, the tail block and its checkpoints";
    const CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = tokens, .max_tokens = 22});
    EXPECT_EQ(probe.host.num_common_tokens, 22);
}

// Two Host-only sessions share their first two pages. The older one's segment
// climbs to its own pages only: the shared pages stay for the newer checkpoint
// boundary.
TEST_F(HostSegmentPlanTest, ALeafHostSegmentStopsBelowPagesAnotherCheckpointShares) {
    const std::vector<std::int32_t> a = MakeTokens(18);
    std::vector<std::int32_t> b = MakeTokens(18, 100);
    std::copy_n(a.begin(), 2 * kP, b.begin());
    const TailPublication a_tail = *WriteBack(a);
    const std::vector<std::string> a_hashes = hashes_;
    const TailPublication b_tail = *WriteBack(b);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(a_tail.hash, group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_tail.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(a_hashes[page], 0)), page < 2) << "page " << page;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_[page], 0))) << "page " << page;
    }
    // A's last two pages, and its tail block and three checkpoints.
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks - before.lru.blocks, 6u);
}

// The same sessions, and a demand both checkpoint boundaries must fund in one
// plan: once the older checkpoint boundary is taken, the newer one is the last
// Host checkpoint boundary under the shared pages, so its segment climbs to the
// root and the Host is left empty.
TEST_F(HostSegmentPlanTest, SharedHostPagesLeaveWithTheLastCheckpointUnderThem) {
    const std::vector<std::int32_t> a = MakeTokens(18);
    std::vector<std::int32_t> b = MakeTokens(18, 100);
    std::copy_n(a.begin(), 2 * kP, b.begin());
    WriteBack(a);
    WriteBack(b);
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 14);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::vector<std::uint32_t> groups{1, 2, 3, 1, 2, 3};
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 6}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 6u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(Tree().Size(), 0u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 14u) << "every Host entry";
}

// A dead entry (a checkpoint boundary missing a State group's checkpoint) goes
// first but frees two parents of the three needed; the Host-only checkpoint
// boundary after it frees three. The restore pass returns the dead entry: the
// checkpoint boundary alone suffices.
TEST_F(HostSegmentPlanTest, TheRestorePassReturnsUnneededVictims) {
    const std::string partial = CheckpointOnHost(100, {1, 2}, /*device_keeps_checkpoint=*/false);
    const std::string whole = CheckpointOnHost(200, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(whole, group))) << "group " << group;
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(partial, group)), group <= 2) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks, 0u);
}

// A load pins the older session's tail KV, not its State. Reclaim that
// checkpoint alone, retaining both sessions' history and the newer State.
TEST_F(HostSegmentPlanTest, PinnedHistoryKeepsPagesWhileUnusedStateLeaves) {
    const TailPublication a_tail = *WriteBack(MakeTokens(18));
    const std::vector<std::string> a_hashes = hashes_;
    const TailPublication b_tail = *WriteBack(MakeTokens(18, 100));
    const CacheBlockRef load = coordinator_.AcquireHostCachedBlock(Key(a_tail.hash, 0));
    ASSERT_TRUE(static_cast<bool>(load));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(a_tail.hash, group)), group == 0) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_tail.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(a_hashes[page], 0))) << "page " << page;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_[page], 0))) << "page " << page;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks - before.lru.blocks, 3u);
    const auto newer_probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = hashes_, .tokens = MakeTokens(18, 100), .max_tokens = 18});
    EXPECT_EQ(newer_probe.host.num_common_tokens, 18);
}

// A load pins the older session's second page: its segment climbs no higher
// than the page under it and leaves without the two pages it cannot take.
TEST_F(HostSegmentPlanTest, TheClimbStopsBelowAPageInUse) {
    const TailPublication a_tail = *WriteBack(MakeTokens(18));
    const std::vector<std::string> a_hashes = hashes_;
    const TailPublication b_tail = *WriteBack(MakeTokens(18, 100));
    const CacheBlockRef load = coordinator_.AcquireHostCachedBlock(Key(a_hashes[1], 0));
    ASSERT_TRUE(static_cast<bool>(load));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(a_tail.hash, group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(b_tail.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(Key(a_hashes[page], 0)), page < 2) << "page " << page;
    }
    // A's last two pages, and its tail block and three checkpoints.
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks - before.lru.blocks, 6u);
}

// Three segments of three, three and twelve State blocks against three
// Host-only checkpoint boundaries of three checkpoints each: the last segment
// cannot be funded and goes whole, and the restore pass returns the checkpoint
// boundary taken only for it.
TEST_F(HostSegmentPlanTest, ADroppedSegmentEvictsNothing) {
    const std::array checkpoint_hashes{CheckpointOnHost(100, {1, 2, 3}, /*device_keeps_checkpoint=*/false),
                                       CheckpointOnHost(200, {1, 2, 3}, /*device_keeps_checkpoint=*/false),
                                       CheckpointOnHost(300, {1, 2, 3}, /*device_keeps_checkpoint=*/false)};
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    std::vector<std::uint32_t> groups;
    for (std::int32_t copy = 0; copy < 6; ++copy) {
        groups.insert(groups.end(), {1, 2, 3});
    }
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3},
                              HostSegmentRows{.first_row = 3, .past_end_row = 6},
                              HostSegmentRows{.first_row = 6, .past_end_row = 18}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 6u);
    EXPECT_EQ(batch.stats.unallocated, 12u);
    for (std::size_t row = 0; row < groups.size(); ++row) {
        EXPECT_EQ(static_cast<bool>(batch.blocks[row]), row < 6) << "row " << row;
    }
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(checkpoint_hashes[0], group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(checkpoint_hashes[1], group))) << "group " << group;
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(checkpoint_hashes[2], group))) << "group " << group;
    }
    // The first two checkpoint boundaries' three checkpoints each.
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks - before.lru.blocks, 6u);
}

// The oldest checkpoint boundary, X, has only its checkpoints on the Host, over
// the Device's KV; three newer Host checkpoint boundaries Y hang under it, each
// two pages and its checkpoints. Segments of three and twelve State blocks: the
// second cannot be funded and is dropped after every checkpoint boundary was
// taken. X's checkpoints are no Host page, so each Y continues the Device's
// prefix without them: the restore pass returns every Y, and X alone funds the
// first segment.
TEST_F(HostSegmentPlanTest, ADroppedSegmentCostsNoSessionUnderCheckpointsOverTheDevicesKv) {
    const std::string x = CheckpointOnHost(1, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    std::vector<std::vector<std::string>> branches;
    for (const std::int32_t start : {500, 600, 700}) {
        std::vector<std::int32_t> tokens = MakeTokens(4 * kP, 1);
        const std::vector<std::int32_t> branch = MakeTokens(2 * kP, start);
        tokens.insert(tokens.end(), branch.begin(), branch.end());
        std::vector<BlockTable> tables(4);
        PublishFinal(tables, tokens);
        coordinator_.Free(tables);
        StoreOnHost(Key(hashes_[4], 0));
        for (std::uint32_t group = 0; group < 4; ++group) {
            StoreOnHost(Key(hashes_[5], group));
        }
        // The other State groups' go along.
        EvictOnDevice(Key(hashes_[5], 1));
        EvictOnDevice(Key(hashes_[5], 0));
        EvictOnDevice(Key(hashes_[4], 0));
        branches.push_back({hashes_[4], hashes_[5]});
    }
    ASSERT_EQ(NodeOf(x).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 4}));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    std::vector<std::uint32_t> groups;
    for (std::int32_t copy = 0; copy < 5; ++copy) {
        groups.insert(groups.end(), {1, 2, 3});
    }
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3},
                              HostSegmentRows{.first_row = 3, .past_end_row = 15}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_EQ(batch.stats.unallocated, 12u);
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(x, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u) << "X's three checkpoints";
    for (const std::vector<std::string>& branch : branches) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(branch[0], 0)));
        for (std::uint32_t group = 0; group < 4; ++group) {
            EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(branch[1], group))) << "group " << group;
        }
    }
}

// X's checkpoints are on the Host over the Device's KV, and the page under X is
// on the Host without a checkpoint boundary: a dead entry, taken first, then X,
// a leaf checkpoint boundary. X's checkpoints alone fund the segment and its
// page stays on the Device, so the restore pass returns the page under it.
TEST_F(HostSegmentPlanTest, APageUnderCheckpointsOverTheDevicesKvReturnsWithoutThem) {
    const std::string x = CheckpointOnHost(1, {1, 2, 3}, /*device_keeps_checkpoint=*/false);
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(5 * kP, 1));
    coordinator_.Free(tables);
    const std::string page = hashes_[4];
    StoreOnHost(Key(page, 0));
    // The other State groups' go along.
    EvictOnDevice(Key(page, 1));
    EvictOnDevice(Key(page, 0));
    ASSERT_EQ(NodeOf(page).parent, &NodeOf(x));
    ASSERT_EQ(NodeOf(x).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(page, 0)));
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(x, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u) << "X's three checkpoints";
}

// Session A's tail was written to the Host, and its Device copy then lost all
// but one checkpoint, still pinned when the tail's KV left. That Device
// checkpoint boundary is A's own, not a deeper one: A's Host checkpoint
// boundary is Host-only cache, and the older Host-only session goes first.
TEST_F(HostSegmentPlanTest, AHostTailIsNotSupersededByItsOwnDeviceCheckpoint) {
    const TailPublication older = *WriteBack(MakeTokens(18, 100));
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    coordinator_.Free(tables);
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(tail.hash, group));
    }
    {
        const CacheBlockRef pinned = coordinator_.AcquireDeviceCachedBlock(Key(tail.hash, 2));
        EvictOnDevice(Key(tail.hash, 0));
    }
    ASSERT_TRUE(NodeOf(tail.hash).has_checkpoint[kDevice]);
    ASSERT_EQ(coordinator_.DeviceBoundaryResidency(Key(tail.hash, 0), tail.tail_tokens),
              CacheCoordinator::BoundaryResidency::kPartial);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
}

// Session A's checkpoint boundary at 18 is a Host tail; the Device then holds
// A's pages and a checkpoint boundary at 16, on the tail's page but shallower
// than the tail. Nothing deeper than 18 is on the Device, so A's checkpoint
// boundary is Host-only cache, and the older Host-only session goes first.
TEST_F(HostSegmentPlanTest, AShallowerDeviceCheckpointDoesNotSupersedeATail) {
    const TailPublication older = *WriteBack(MakeTokens(18, 100));
    const std::vector<std::string> older_hashes = hashes_;
    const TailPublication tail = *WriteBack(MakeTokens(18));
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(16));
    coordinator_.Free(tables);
    ASSERT_TRUE(NodeOf(hashes_[3]).has_checkpoint[kDevice]);
    ASSERT_EQ(NodeOf(tail.hash).parent, &NodeOf(hashes_[3]));
    ASSERT_FALSE(NodeOf(tail.hash).has_checkpoint[kDevice]);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator_.AcquireHostSegments(state_group_ids_, zero_buckets_, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, group))) << "group " << group;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older.hash, group))) << "group " << group;
    }
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_[page], 0))) << "page " << page;
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(older_hashes[page], 0))) << "page " << page;
    }
    EXPECT_EQ(coordinator_.HostEvictions().superseded.blocks, 0u);
    // The older session's four pages, and its tail block and three checkpoints.
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 8u);
}

// ---------------------------------------------------------------------------
// Host placement per segment under KV parallelism: the fixture's Full group
// dealt over two shards, a block's bucket its slot % 2. A row is fitted and
// placed in its Device block's bucket only, so a victim that frees room in
// another bucket funds nothing.
// ---------------------------------------------------------------------------
class ShardedHostSegmentPlanTest : public HostSegmentPlanTest {
protected:
    ShardedHostSegmentPlanTest() : HostSegmentPlanTest(/*full_shard_count=*/2) {}
};

// The Host holds one page P of a session in bucket 0, its parent's bucket-1
// slots held and its other bucket-0 slot free; every other Host parent is
// held. Two segments of Full rows: one row in bucket 0, then a row in bucket 0
// and one in bucket 1. Bucket 0 can place both bucket-0 rows (the free slot,
// and P's once P, the only victim, is evicted), but bucket 1 cannot place the
// other row: the second segment is dropped whole, the restore pass returns P,
// and the first segment takes the free slot. Nothing leaves the Host. Segments
// are dropped from the back, so with the two-bucket segment first both go.
TEST_F(ShardedHostSegmentPlanTest, ABucketTheHostCannotFillDropsItsSegmentWhole) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(4 * kP));
    coordinator_.Free(tables);
    CacheBlockRef page = host_pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
    coordinator_.CacheHostBlock(page, Key(hashes_[0], 0), std::nullopt);
    page.reset();
    for (CacheBlockRef& block : host_pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 2>{1, 1})) {
        held_.push_back(std::move(block));
    }
    for (CacheBlockRef& block : host_pool_.AcquireBlocks(1, host_pool_.NumEmptyLcmBlocks())) {
        held_.push_back(std::move(block));
    }
    ASSERT_EQ(host_pool_.FreeSlotsByBucket(0)[0], 1);
    ASSERT_EQ(host_pool_.FreeSlotsByBucket(0)[1], 0);
    ASSERT_EQ(host_pool_.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0, 0, 0};
    for (const bool two_bucket_segment_first : {false, true}) {
        SCOPED_TRACE(two_bucket_segment_first);
        const std::vector<std::int32_t> buckets =
            two_bucket_segment_first ? std::vector<std::int32_t>{0, 1, 0} : std::vector<std::int32_t>{0, 0, 1};
        const std::size_t first_segment_rows = two_bucket_segment_first ? 2 : 1;
        const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = first_segment_rows},
                                  HostSegmentRows{.first_row = first_segment_rows, .past_end_row = 3}};
        const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

        EXPECT_EQ(batch.stats.allocated, two_bucket_segment_first ? 0u : 1u);
        EXPECT_EQ(static_cast<bool>(batch.blocks[0]), !two_bucket_segment_first);
        EXPECT_FALSE(static_cast<bool>(batch.blocks[1]));
        EXPECT_FALSE(static_cast<bool>(batch.blocks[2]));
        if (batch.blocks[0]) {
            EXPECT_EQ(coordinator_.Allocator(0).BucketOf(batch.blocks[0]->Location()), 0);
        }
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(hashes_[0], 0)));
        for (const auto reason :
             {&CacheCoordinator::HostEvictionStats::lru, &CacheCoordinator::HostEvictionStats::duplicate,
              &CacheCoordinator::HostEvictionStats::superseded, &CacheCoordinator::HostEvictionStats::dead_entry,
              &CacheCoordinator::HostEvictionStats::tail_cap, &CacheCoordinator::HostEvictionStats::flush,
              &CacheCoordinator::HostEvictionStats::duplicate_kv}) {
            EXPECT_EQ((coordinator_.HostEvictions().*reason).blocks, 0u);
        }
    }
}

// The oldest victim X, a page no Host checkpoint boundary is under (dead), sits
// in bucket 0 beside held blocks and frees a bucket-0 slot only; the newer
// Host-only checkpoint boundary Y frees its three State parents. One Full row
// in bucket 1 takes X, which does not fit it, then Y, which does; the restore
// pass returns X, and the row lands in a parent Y emptied, in bucket 1. The
// bucket-wise TheRestorePassReturnsUnneededVictims.
TEST_F(ShardedHostSegmentPlanTest, AVictimInTheWrongBucketIsReturned) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(4 * kP));
    coordinator_.Free(tables);
    const std::string x = hashes_[0];
    CacheBlockRef page = host_pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
    coordinator_.CacheHostBlock(page, Key(x, 0), std::nullopt);
    page.reset();
    const std::string y = CheckpointOnHost(200, state_group_ids_, /*device_keeps_checkpoint=*/false);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::vector<std::uint32_t> groups{0};
    const std::vector<std::int32_t> buckets{1};
    const std::array segments{HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 1u);
    ASSERT_TRUE(static_cast<bool>(batch.blocks[0]));
    EXPECT_EQ(coordinator_.Allocator(0).BucketOf(batch.blocks[0]->Location()), 1);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(x, 0)));
    for (std::uint32_t group = 1; group <= 3; ++group) {
        EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(y, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 3u);
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks, 0u);
}

// A Host suffix loads into Device blocks of its Host blocks' buckets, and an
// admission evicts for those buckets. A KV-only coordinator (one Full group,
// four blocks per parent, two shards: MLA or DSA under KV parallelism); with
// State groups any Device segment frees whole State parents, which serve every
// bucket. Session s resumes at its two Device pages and its third page from the
// Host, in bucket 1, and grows one block. The Device's first parent holds s's
// pages, a bucket-0 hole and a held block, its second a's page in bucket 0 and
// b's in bucket 1 beside held blocks, a older than b. a goes first and frees no
// bucket-1 slot, then b, and the restore pass returns a: only b is evicted and
// recorded, and the load's two ends share bucket 1.
TEST(ShardedSegmentEvictionTest, ADeviceBucketShortageEvictsOnlyWhatFundsTheBucket) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/2, {4});
    BlockPool host_pool(/*num_lcm_blocks=*/1, {4});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP, .shard_count = 2}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::int32_t> s = MakeTokens(3 * kP + 1);
    std::vector<std::vector<std::int32_t>> pages;
    for (std::int32_t page = 0; page < 3; ++page) {
        pages.emplace_back(s.begin() + page * kP, s.begin() + (page + 1) * kP);
    }
    const std::vector<std::string> s_hashes = ContentHashes(pages);
    const std::vector<std::string> a_and_b{ContentHashes({MakeTokens(kP, 100)})[0],
                                           ContentHashes({MakeTokens(kP, 200)})[0]};
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 3 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, s_hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    // s's third page leaves the Device for the Host's bucket 1.
    CacheBlockRef host_page = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1}).front();
    coordinator.CacheHostBlock(host_page, Key(s_hashes[2], 0), std::nullopt);
    host_page.reset();
    const std::optional<CacheBlockLocation> third_page =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(s_hashes[2], 0));
    ASSERT_TRUE(third_page && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *third_page));
    // The first parent's two free slots are held while a and b take the
    // second parent's; the slot beside each in its bucket is held after it.
    std::vector<CacheBlockRef> held = pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 2>{0, 1});
    for (std::int32_t session = 0; session < 2; ++session) {
        ASSERT_TRUE(AdmitForTest(coordinator, tables, kP));
        CacheCompletedBlocksForTest(coordinator, tables, std::span<const std::string>(a_and_b).subspan(session, 1),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        held.push_back(pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{session}).front());
    }
    held[0].reset();
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[0], 1);
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[1], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    for (std::int32_t session = 0; session < 2; ++session) {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(a_and_b[session], 0));
        ASSERT_TRUE(location.has_value());
        ASSERT_EQ(coordinator.Allocator(0).BucketOf(*location), session) << "a in bucket 0, b in bucket 1";
    }
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = s_hashes, .tokens = s, .max_tokens = 3 * kP};
    const std::uint64_t evicted_before = coordinator.DeviceEvictions().admission.blocks;

    const std::optional<CacheCoordinator::AdmissionResult> admitted =
        AdmitForTest(coordinator, tables, coordinator.ProbePromptPrefix(prompt), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(admitted);
    ASSERT_EQ(admitted->load_pairs.size(), 1u);
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(admitted->load_pairs[0].source->Location()), 1);
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(admitted->load_pairs[0].destination->Location()), 1);
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[1], 0)));
    EXPECT_EQ(coordinator.DeviceEvictions().admission.blocks - evicted_before, 1u);
    std::vector<CacheCoordinator::EvictedSegment> segments = coordinator.TakeEvictedSegments();
    ASSERT_EQ(segments.size(), 1u);
    ASSERT_EQ(segments[0].members.size(), 1u);
    EXPECT_EQ(segments[0].members[0].key, Key(a_and_b[1], 0));
    coordinator.SettleEvictedSegment(std::move(segments[0]), /*placed_on_host=*/false);
    coordinator.Free(tables);
}

// As above, but a live request holds b: no victim funds bucket 1, so the
// admission waits -- nothing is evicted or recorded, and no Host entry stays
// pinned -- until the request lets go, when the same admission evicts b.
TEST(ShardedSegmentEvictionTest, ADeviceBucketThatCannotBeFundedMakesTheAdmissionWait) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/2, {4});
    BlockPool host_pool(/*num_lcm_blocks=*/1, {4});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = kP, .shard_count = 2}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::int32_t> s = MakeTokens(3 * kP + 1);
    std::vector<std::vector<std::int32_t>> pages;
    for (std::int32_t page = 0; page < 3; ++page) {
        pages.emplace_back(s.begin() + page * kP, s.begin() + (page + 1) * kP);
    }
    const std::vector<std::string> s_hashes = ContentHashes(pages);
    const std::vector<std::string> a_and_b{ContentHashes({MakeTokens(kP, 100)})[0],
                                           ContentHashes({MakeTokens(kP, 200)})[0]};
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 3 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, s_hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    // s's third page leaves the Device for the Host's bucket 1.
    CacheBlockRef host_page = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1}).front();
    coordinator.CacheHostBlock(host_page, Key(s_hashes[2], 0), std::nullopt);
    host_page.reset();
    const std::optional<CacheBlockLocation> third_page =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(s_hashes[2], 0));
    ASSERT_TRUE(third_page && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *third_page));
    // The first parent's two free slots are held while a and b take the
    // second parent's; the slot beside each in its bucket is held after it.
    std::vector<CacheBlockRef> held = pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 2>{0, 1});
    for (std::int32_t session = 0; session < 2; ++session) {
        ASSERT_TRUE(AdmitForTest(coordinator, tables, kP));
        CacheCompletedBlocksForTest(coordinator, tables, std::span<const std::string>(a_and_b).subspan(session, 1),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        held.push_back(pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{session}).front());
    }
    held[0].reset();
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[0], 1);
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[1], 0);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);
    for (std::int32_t session = 0; session < 2; ++session) {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(a_and_b[session], 0));
        ASSERT_TRUE(location.has_value());
        ASSERT_EQ(coordinator.Allocator(0).BucketOf(*location), session) << "a in bucket 0, b in bucket 1";
    }
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = s_hashes, .tokens = s, .max_tokens = 3 * kP};
    const std::uint64_t evicted_before = coordinator.DeviceEvictions().admission.blocks;
    CacheBlockRef reader = coordinator.GroupPrefixIndex(0).Find(pool, Key(a_and_b[1], 0));
    ASSERT_TRUE(static_cast<bool>(reader));
    EXPECT_FALSE(AdmitForTest(coordinator, tables, coordinator.ProbePromptPrefix(prompt),
                              GroupDemand{.extent = DenseGrowth{1}}));
    EXPECT_EQ(coordinator.DeviceEvictions().admission.blocks, evicted_before);
    EXPECT_TRUE(coordinator.TakeEvictedSegments().empty());
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[0], 0)));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[1], 0)));
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);

    reader.reset();
    const std::optional<CacheCoordinator::AdmissionResult> admitted =
        AdmitForTest(coordinator, tables, coordinator.ProbePromptPrefix(prompt), GroupDemand{.extent = DenseGrowth{1}});
    ASSERT_TRUE(admitted);
    ASSERT_EQ(admitted->load_pairs.size(), 1u);
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(admitted->load_pairs[0].source->Location()),
              coordinator.Allocator(0).BucketOf(admitted->load_pairs[0].destination->Location()));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[0], 0)));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(a_and_b[1], 0)));
    for (CacheCoordinator::EvictedSegment& segment : coordinator.TakeEvictedSegments()) {
        coordinator.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }
    coordinator.Free(tables);
}

// The batch-greedy starvation of #2593 (placing a batch block by block, an
// earlier block can take the room a later one needs), without State groups:
// a pinned entry P and C share group 0's parent, V fills group 1's.
// Group 0's segment first takes nothing group 1's needs: one joint fit
// evicts V for group 1 and C for group 0, and both segments are placed.
TEST(HostSegmentPlanWithoutStateTest, LaterSegmentNotStarvedByAnEarlierOne) {
    BlockPool pool(/*num_lcm_blocks=*/4, {2, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/2, {2, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}},
        /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false, &host_pool,
        /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
        /*write_back_evictions=*/true);
    // V first, the oldest; P stays pinned by the reference kept here.
    CacheBlockRef v = host_pool.AcquireBlock(1);
    coordinator.CacheHostBlock(v, Key("v", 1), std::nullopt);
    v.reset();
    CacheBlockRef pinned = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(pinned, Key("p", 0), std::nullopt);
    CacheBlockRef c = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(c, Key("c", 0), std::nullopt);
    c.reset();
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0, 1};
    const std::vector<std::int32_t> buckets{0, 0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1},
                              CacheCoordinator::HostSegmentRows{.first_row = 1, .past_end_row = 2}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 2u);
    ASSERT_TRUE(batch.blocks[0] && batch.blocks[1]);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[0]->Location().lcm_block_id), 0u);
    EXPECT_EQ(host_pool.BoundGroup(batch.blocks[1]->Location().lcm_block_id), 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("v", 1)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("c", 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("p", 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 2u);
}

// The mixed-packing variant of that starvation with a State group, on
// a hand-built pool: group 0's parent holds A and the newer B, the State
// group's parent the oldest checkpoint S. With State groups a page no
// checkpoint resumes is a dead entry, so A and B are taken before S; the
// restore pass then returns B, which the joint fit does not need -- {A, S}
// leave, not {S, A, B}.
TEST(HostSegmentPlanSmallPoolTest, MixedPackingEvictsNoNewerEntry) {
    BlockPool pool(/*num_lcm_blocks=*/4, {2, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/2, {2, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        /*prefix_granularity=*/4, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    for (const auto& [hash, group] : {std::pair{"s", 1u}, std::pair{"a", 0u}, std::pair{"b", 0u}}) {
        CacheBlockRef block = host_pool.AcquireBlock(group);
        coordinator.CacheHostBlock(block, Key(hash, group), std::nullopt);
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0, 1};
    const std::vector<std::int32_t> buckets{0, 0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 2}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("a", 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key("s", 1)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("b", 0)));
    EXPECT_EQ(coordinator.HostEvictions().dead_entry.blocks, 1u) << "A";
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 1u) << "S";
}

// Without State groups a tail serves only while the probe enumerates it: the
// per-prior cap dropped this one while a load pinned its Host block, so the
// block stayed without its hint. It is a dead entry, taken before the older
// Host-only page.
TEST(HostSegmentPlanWithoutStateTest, ACappedTailIsADeadEntry) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/16, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/3, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> older = ContentHashes({{9, 9, 9, 9}});
    {
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, kP));
        CacheCompletedBlocksForTest(coordinator, tables, older,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        CacheBlockRef block = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(block, Key(older[0], 0), std::nullopt);
    }
    const std::optional<CacheBlockLocation> older_on_device =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(older[0], 0));
    ASSERT_TRUE(older_on_device && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *older_on_device));
    // Nine tails of three tokens under one page; the first is copied to the
    // Host and pinned there before the ninth makes the cap drop it.
    const std::vector<std::string> page = ContentHashes({{1, 2, 3, 4}});
    std::vector<TailPublication> tails;
    CacheBlockRef load;
    for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior + 1; ++extra) {
        if (extra == TailTable::kMaxTailsPerPrior + 1) {
            CacheBlockRef block = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(block, Key(tails[0].hash, 0), std::nullopt);
            load = coordinator.AcquireHostCachedBlock(Key(tails[0].hash, 0));
        }
        const std::vector<std::int32_t> tail_tokens{1000 * extra, 1000 * extra + 1, 1000 * extra + 2};
        tails.push_back(TailPublication{.tail_tokens = 3, .hash = HashPrefixPage(token_span(tail_tokens), page[0])});
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, kP + 3));
        coordinator.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = page,
                                                              .first_new_prefix_page = 0,
                                                              .tail = tails.back(),
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = kP + 3},
            CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
        coordinator.Free(tables);
    }
    ASSERT_EQ(coordinator.TailCapEvictions(), 1);
    load.reset();
    ASSERT_TRUE(coordinator.ContainsHostCachedBlock(Key(tails[0].hash, 0)));
    const CacheBlockRef held = host_pool.AcquireBlock(0);
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0};
    const std::vector<std::int32_t> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(tails[0].hash, 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(older[0], 0)));
    EXPECT_EQ(coordinator.HostEvictions().dead_entry.blocks, 1u);
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
}

// The Device holds p and c, the Host p (a duplicate) and g, the suffix that
// continues the Device's p and c. p is a Host leaf, the first victim; g hangs
// under c, a page off the Host, so it does not leave with p.
TEST(HostSegmentPlanWithoutStateTest, AHostSuffixUnderADevicePageStaysWhenTheLeafAboveItGoes) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/3, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 3 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/3 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    for (const std::size_t page : {0, 2}) {
        CacheBlockRef block = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(block, Key(hashes[page], 0), std::nullopt);
    }
    const std::optional<CacheBlockLocation> g_on_device =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hashes[2], 0));
    ASSERT_TRUE(g_on_device && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *g_on_device));
    const CacheBlockRef held = host_pool.AcquireBlock(0);
    const std::vector<std::uint32_t> groups{0};
    const std::vector<std::int32_t> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hashes[0], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hashes[2], 0)));
    EXPECT_EQ(coordinator.HostEvictions().duplicate.blocks, 1u);
    EXPECT_EQ(coordinator.HostEvictions().duplicate.device_resident, 1u);
}

// Group 0 (two blocks per parent) holds c1 beside a pinned entry and c0 alone;
// group 1 needs a whole parent. c1, the Host leaf, goes first but frees only a
// hole; c0, a Host leaf by then, empties its parent. The restore pass cannot
// return c1 though the fit no longer needs it: the Host would keep c1 without
// c0.
TEST(HostSegmentPlanWithoutStateTest, TheRestorePassKeepsGoneAPageWhoseParentLeft) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {2, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/3, {2, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = kP},
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, 2 * kP));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * kP,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    CacheBlockRef pinned = host_pool.AcquireBlock(0);
    coordinator.CacheHostBlock(pinned, Key("z", 0), std::nullopt);
    for (const std::size_t page : {1, 0}) {
        CacheBlockRef block = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(block, Key(hashes[page], 0), std::nullopt);
    }
    for (const std::string& hash : hashes) {
        for (std::uint32_t group = 0; group < 2; ++group) {
            const std::optional<CacheBlockLocation> location =
                CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, group, Key(hash, group));
            ASSERT_TRUE(location && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, group, *location));
        }
    }
    const CacheBlockRef held = host_pool.AcquireBlock(1);
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{1};
    const std::vector<std::int32_t> buckets{0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hashes[0], 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hashes[1], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key("z", 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 2u);
}

// p0 alone, the oldest, and the newer chain c0 c1 fill the Host. Segments of
// one and three blocks take p0, c1 and then c0, a Host leaf by then, before the
// second segment is dropped. The restore pass returns c0 first, and with c0
// back c1 returns under it: only p0, which the first segment needs, leaves.
TEST(HostSegmentPlanWithoutStateTest, TheRestorePassReturnsAPageOnceThePageAboveItIsBack) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/3, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> p = ContentHashes({{9, 9, 9, 9}});
    const std::vector<std::string> c = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    for (const std::vector<std::string>* chain : {&p, &c}) {
        std::vector<BlockTable> tables(1);
        const auto tokens = static_cast<std::int32_t>(chain->size()) * kP;
        ASSERT_TRUE(AdmitForTest(coordinator, tables, tokens));
        CacheCompletedBlocksForTest(coordinator, tables, *chain,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/tokens,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        for (const std::string& hash : *chain) {
            CacheBlockRef block = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
        }
        for (const std::string& hash : *chain) {
            const std::optional<CacheBlockLocation> location =
                CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hash, 0));
            ASSERT_TRUE(location && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
        }
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0, 0, 0, 0};
    const std::vector<std::int32_t> buckets{0, 0, 0, 0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1},
                              CacheCoordinator::HostSegmentRows{.first_row = 1, .past_end_row = 4}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(batch.stats.unallocated, 3u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(p[0], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(c[0], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(c[1], 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 1u);
}

// Without State groups the candidates are the Host leaves: a chain c0-c3 and
// a newer one d0-d1. Three parents are needed: c3 goes, which leaves c2 a
// leaf, which goes next, then c1. c0 and d stay.
TEST(HostSegmentPlanWithoutStateTest, HostLeavesGoDeepestPageFirst) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/8, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    std::vector<std::vector<std::string>> chains;
    for (const auto& [num_pages, start] : {std::pair{4, 1}, std::pair{2, 100}}) {
        const std::vector<std::int32_t> tokens = MakeTokens(num_pages * kP, start);
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < num_pages; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        chains.push_back(ContentHashes(pages));
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, num_pages * kP));
        CacheCompletedBlocksForTest(coordinator, tables, chains.back(),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/num_pages * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        for (const std::string& hash : chains.back()) {
            CacheBlockRef block = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
        }
        for (const std::string& hash : chains.back()) {
            const std::optional<CacheBlockLocation> location =
                CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hash, 0));
            ASSERT_TRUE(location && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
        }
    }
    const std::vector<CacheBlockRef> held = host_pool.AcquireBlocks(0, host_pool.NumEmptyLcmBlocks());
    const std::vector<std::uint32_t> groups{0, 0, 0};
    const std::vector<std::int32_t> buckets{0, 0, 0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    ASSERT_EQ(batch.stats.allocated, 3u);
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(Key(chains[0][page], 0)), page == 0) << "page " << page;
    }
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(chains[1][0], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(chains[1][1], 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    // Each freed parent opens in the order it emptied: c3's, c2's, c1's.
    std::vector<std::int32_t> parents;
    for (const CacheBlockRef& block : batch.blocks) {
        parents.push_back(block->Location().lcm_block_id);
    }
    EXPECT_EQ(parents, (std::vector<std::int32_t>{4, 3, 2}));
}

// Two chains share c0: a = c0 a1 a2, the older, and b = c0 b1. Three parents
// are needed: a2, then a1, which a2 left a Host leaf, then b1. c0 becomes a
// Host leaf only once every branch under it is gone, so it stays though it is
// older than b1.
TEST(HostSegmentPlanWithoutStateTest, ASharedPageWaitsForEveryBranch) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/4, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    const std::vector<std::string> a = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    const std::vector<std::string> b = ContentHashes({{1, 2, 3, 4}, {50, 60, 70, 80}});
    for (const std::vector<std::string>* chain : {&a, &b}) {
        std::vector<BlockTable> tables(1);
        const auto tokens = static_cast<std::int32_t>(chain->size()) * kP;
        ASSERT_TRUE(AdmitForTest(coordinator, tables, tokens));
        CacheCompletedBlocksForTest(coordinator, tables, *chain,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/tokens,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        for (const std::string& hash : *chain) {
            if (!coordinator.ContainsHostCachedBlock(Key(hash, 0))) {
                CacheBlockRef block = host_pool.AcquireBlock(0);
                coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
            }
        }
    }
    for (const std::string& hash : {b[1], a[2], a[1], a[0]}) {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hash, 0));
        ASSERT_TRUE(location && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups{0, 0, 0};
    const std::vector<std::int32_t> buckets{0, 0, 0};
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 3}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, 3u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(a[0], 0)));
    for (const std::string& hash : {a[1], a[2], b[1]}) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hash, 0)));
    }
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
}

// A wide fork: 64 one-page branches under c0, on the Host alone, Host-dated
// oldest first in the order they hang under it, and a segment that needs all
// their rows. Every branch goes; c0, which becomes a Host leaf once the last
// branch goes, stays.
TEST(HostSegmentPlanWithoutStateTest, AWideForkPageBecomesAHostLeafOnceItsLastBranchGoes) {
    constexpr std::int32_t kP = 4;
    constexpr std::size_t kBranches = 64;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/kBranches + 2, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kP}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    std::string c0;
    std::vector<std::string> branches;
    for (std::size_t branch = 0; branch < kBranches; ++branch) {
        const auto first = static_cast<std::int32_t>(1000 + kP * branch);
        const std::vector<std::string> chain = ContentHashes({{1, 2, 3, 4}, {first, first + 1, first + 2, first + 3}});
        c0 = chain[0];
        branches.push_back(chain[1]);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, 2 * kP));
        CacheCompletedBlocksForTest(coordinator, tables, chain,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        coordinator.Free(tables);
        for (const std::string& hash : chain) {
            if (!coordinator.ContainsHostCachedBlock(Key(hash, 0))) {
                CacheBlockRef block = host_pool.AcquireBlock(0);
                coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
            }
        }
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(chain[1], 0));
        ASSERT_TRUE(location && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
    }
    const std::optional<CacheBlockLocation> c0_on_device =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(c0, 0));
    ASSERT_TRUE(c0_on_device && CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *c0_on_device));
    const std::vector<CacheBlockRef> held = host_pool.AcquireBlocks(0, host_pool.NumEmptyLcmBlocks());
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const std::vector<std::uint32_t> groups(kBranches, 0);
    const std::vector<std::int32_t> buckets(kBranches, 0);
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = kBranches}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator.AcquireHostSegments(groups, buckets, segments);

    EXPECT_EQ(batch.stats.allocated, kBranches);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(c0, 0)));
    for (const std::string& hash : branches) {
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hash, 0)));
    }
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, kBranches);
}

// Two coordinators fed the same publications, Host copies and Device
// evictions plan the same victims and the same rows: every decision follows the
// tree's insertion order and the plan's ranks, never an address. K3 shape, the
// Full group's pages dealt over the parameter's KV-parallel shards; sessions of
// 18 tokens fork from the first at varying pages, every other one keeps its
// Device copy.
class HostSegmentPlanMirrorTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, HostSegmentPlanMirrorTest, ::testing::Values(1, 2, 4));

TEST_P(HostSegmentPlanMirrorTest, PlansMatchOnTwoMirroredCoordinators) {
    constexpr std::int32_t kP = 4;
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull,
         .cache_blocks_per_lcm_block = 4,
         .block_granularity = kP,
         .shard_count = GetParam()},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    BlockPool first_pool(/*num_lcm_blocks=*/64, {4, 1, 1, 1});
    BlockPool second_pool(/*num_lcm_blocks=*/64, {4, 1, 1, 1});
    BlockPool first_host(/*num_lcm_blocks=*/40, {4, 1, 1, 1});
    BlockPool second_host(/*num_lcm_blocks=*/40, {4, 1, 1, 1});
    std::array<CacheCoordinator, 2> coordinators{
        MakeCoordinator(specs, kP, first_pool, /*enable_l3_storage=*/false, &first_host, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true),
        MakeCoordinator(specs, kP, second_pool, /*enable_l3_storage=*/false, &second_host, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true)};
    const std::array<BlockPool*, 2> pools{&first_pool, &second_pool};
    const std::array<BlockPool*, 2> hosts{&first_host, &second_host};
    std::vector<CacheKey> keys;
    const std::vector<std::int32_t> trunk = MakeTokens(18);
    for (std::int32_t session = 0; session < 8; ++session) {
        std::vector<std::int32_t> tokens = MakeTokens(18, 100 * (session + 1));
        std::copy_n(trunk.begin(), session % 4 * kP, tokens.begin());
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < 4; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        const std::vector<std::string> hashes = ContentHashes(pages);
        const TailPublication tail{.tail_tokens = 2,
                                   .hash = HashPrefixPage(token_span(tokens).subspan(4 * kP), hashes.back())};
        std::vector<std::pair<std::string, std::int32_t>> boundaries;
        for (const std::string& hash : hashes) {
            boundaries.emplace_back(hash, 0);
        }
        boundaries.emplace_back(tail.hash, tail.tail_tokens);
        for (std::size_t c = 0; c < 2; ++c) {
            CacheCoordinator& coordinator = coordinators[c];
            std::vector<BlockTable> tables(4);
            ASSERT_TRUE(AdmitForTest(coordinator, tables, 18));
            coordinator.CacheCompletedBlocks(
                tables,
                RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = hashes,
                                                                  .first_new_prefix_page = 0,
                                                                  .tail = tail,
                                                                  .publish_state = true,
                                                                  .boundary_kind = CacheBoundaryKind::kEndpoint},
                                .num_computed_tokens = 18},
                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
            coordinator.Free(tables);
            for (const auto& [hash, tail_tokens] : boundaries) {
                coordinator.ForEachBoundaryKey(Key(hash, 0), tail_tokens, [&](std::size_t, const CacheKey& key) {
                    if (coordinator.GroupPrefixIndex(key.group_id).Contains(*pools[c], key) &&
                        !coordinator.ContainsHostCachedBlock(key)) {
                        CacheBlockRef block = hosts[c]->AcquireBlock(key.group_id);
                        coordinator.CacheHostBlock(block, key, std::nullopt);
                        if (c == 0) {
                            keys.push_back(key);
                        }
                    }
                });
            }
            if (session % 2 == 1) {
                continue;  // the Device keeps its copy
            }
            for (const auto& [hash, tail_tokens] : boundaries) {
                coordinator.ForEachBoundaryKey(Key(hash, 0), tail_tokens, [&](std::size_t, const CacheKey& key) {
                    if (const std::optional<CacheBlockLocation> location =
                            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, key.group_id, key)) {
                        CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, key.group_id, *location);
                    }
                });
            }
        }
    }
    std::array<std::vector<CacheBlockRef>, 2> held;
    for (std::size_t c = 0; c < 2; ++c) {
        held[c] = hosts[c]->AcquireBlocks(0, hosts[c]->NumFreeSlots(0));
        for (CacheBlockRef& block : hosts[c]->AcquireBlocks(1, hosts[c]->NumEmptyLcmBlocks())) {
            held[c].push_back(std::move(block));
        }
    }
    const std::vector<std::uint32_t> groups{1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3};
    // The Full rows spread over the buckets (row % shards), as a drain's rows
    // follow their Device blocks; the State groups are unsharded.
    std::vector<std::int32_t> buckets;
    for (std::size_t row = 0; row < groups.size(); ++row) {
        buckets.push_back(groups[row] == 0 ? static_cast<std::int32_t>(row) % GetParam() : 0);
    }
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 3},
                              CacheCoordinator::HostSegmentRows{.first_row = 3, .past_end_row = 8},
                              CacheCoordinator::HostSegmentRows{.first_row = 8, .past_end_row = 14}};
    std::array<std::vector<CacheCoordinator::HostAllocationBatch>, 2> batches;
    for (std::int32_t round = 0; round < 3; ++round) {
        for (std::size_t c = 0; c < 2; ++c) {
            batches[c].push_back(coordinators[c].AcquireHostSegments(groups, buckets, segments));
        }
        const CacheCoordinator::HostAllocationBatch& lhs = batches[0].back();
        const CacheCoordinator::HostAllocationBatch& rhs = batches[1].back();
        for (std::size_t row = 0; row < groups.size(); ++row) {
            ASSERT_EQ(static_cast<bool>(lhs.blocks[row]), static_cast<bool>(rhs.blocks[row])) << "row " << row;
            if (lhs.blocks[row]) {
                EXPECT_EQ(lhs.blocks[row]->Location(), rhs.blocks[row]->Location()) << "row " << row;
                EXPECT_EQ(coordinators[0]
                              .Allocator(static_cast<std::int32_t>(groups[row]))
                              .BucketOf(lhs.blocks[row]->Location()),
                          buckets[row])
                    << "row " << row;
            }
        }
        EXPECT_EQ(lhs.stats.allocated, rhs.stats.allocated) << "round " << round;
        // Equal running totals after every round: equal victim counts by
        // reason in each round; the Contains loop below pins which entries.
        for (const auto reason :
             {&CacheCoordinator::HostEvictionStats::lru, &CacheCoordinator::HostEvictionStats::duplicate,
              &CacheCoordinator::HostEvictionStats::superseded, &CacheCoordinator::HostEvictionStats::dead_entry,
              &CacheCoordinator::HostEvictionStats::tail_cap, &CacheCoordinator::HostEvictionStats::flush,
              &CacheCoordinator::HostEvictionStats::duplicate_kv}) {
            EXPECT_EQ((coordinators[0].HostEvictions().*reason).blocks,
                      (coordinators[1].HostEvictions().*reason).blocks)
                << "round " << round;
            EXPECT_EQ((coordinators[0].HostEvictions().*reason).device_resident,
                      (coordinators[1].HostEvictions().*reason).device_resident)
                << "round " << round;
        }
        for (const CacheKey& key : keys) {
            EXPECT_EQ(coordinators[0].ContainsHostCachedBlock(key), coordinators[1].ContainsHostCachedBlock(key))
                << "round " << round;
        }
    }
    EXPECT_LT(coordinators[0].NumHostCachedBlocks(), static_cast<std::int32_t>(keys.size())) << "the plans evicted";
}

// ---------------------------------------------------------------------------
// Dead prefix entries under the write-back policy: what no probe can resume
// from any more -- the pages no checkpoint boundary is left under on either
// tier, a tail the cap dropped, what lies at and under a gap (a page whose KV
// neither tier holds) -- is queued as evictions leave it and collected on both
// tiers, judged again when collected; an entry in use makes its root wait for
// the next collection. The K3 shape of BoundaryTreeCoordinatorFixture; each
// test calls CollectDeadEntries itself, as the scheduler does once per plan.
// ---------------------------------------------------------------------------
class DeadEntryTest : public BoundaryTreeCoordinatorFixture {
protected:
    DeadEntryTest()
        : BoundaryTreeCoordinatorFixture(/*write_back_evictions=*/true, /*full_shard_count=*/1,
                                         /*num_snapshot_pages=*/0) {}

    // PublishTwoSessionsSharingTwoPages, then the end of A as a write-back
    // leaves it: its last page and its tail with the checkpoints copied to
    // the Host and evicted from the Device, which keeps s0, s1 and a2, the
    // Device prefix the Host suffix continues.
    void LeaveTheEndOfAOnTheHost() {
        PublishTwoSessionsSharingTwoPages();
        StoreOnHost(Key(a_hashes_[3], 0));
        for (std::uint32_t group = 0; group < 4; ++group) {
            StoreOnHost(Key(a_tail_.hash, group));
        }
        // The tail's KV block takes its checkpoints along.
        EvictOnDevice(Key(a_tail_.hash, 0));
        EvictOnDevice(Key(a_hashes_[3], 0));
        ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 5);
    }

    // A request resumes on the Device at the checkpoint another request
    // published at 4 (on p0) and computes to 8. Its next chunk publishes p1
    // without a checkpoint, acquires the blocks of one at 12 its forward
    // writes, and rotates the checkpoint at 4 out of slot 0; only that
    // checkpoint's parents fund the chunk, so it leaves as the request's own
    // reclaim. No checkpoint boundary is left under p0 then, while the
    // request's table still holds p0 and p1. hashes_ become the request's first
    // three pages; returns the chunk's checkpoint blocks.
    std::vector<CacheBlockRef> DropTheCheckpointALiveRequestResumedAt(std::vector<BlockTable>& tables) {
        std::vector<BlockTable> publisher(4);
        PublishFinal(publisher, MakeTokens(kP));
        coordinator_.Free(publisher);
        const std::vector<std::int32_t> tokens = MakeTokens(4 * kP);
        hashes_ = ContentHashes({{tokens.begin(), tokens.begin() + kP},
                                 {tokens.begin() + kP, tokens.begin() + 2 * kP},
                                 {tokens.begin() + 2 * kP, tokens.begin() + 3 * kP}});
        CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(
            {.prefix_hashes = std::span<const std::string>(hashes_).first(1), .tokens = tokens, .max_tokens = kP});
        EXPECT_EQ(probe.device.num_common_tokens, kP);
        EXPECT_TRUE(AdmitForTest(coordinator_, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{kP}}));
        // The chunk's Full block takes a spare parent, its three live slots
        // and three checkpoint blocks the other three and the three the
        // released checkpoint frees.
        HoldFreeCapacity(pool_, /*spare_parents=*/4);
        const RequestProgress progress{
            .completed_pages = CompletedPages{.prefix_hashes = std::span<const std::string>(hashes_).first(2),
                                              .first_new_prefix_page = 1,
                                              .boundary_kind = CacheBoundaryKind::kChunk},
            .num_computed_tokens = 2 * kP,
        };
        const CacheCoordinator::DeviceEvictionStats before = coordinator_.DeviceEvictions();
        std::optional<CacheCoordinator::AdmissionResult> chunk =
            AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                         progress, StateStep{.kind = StateStep::Kind::kRotate, .checkpoint = true});
        EXPECT_TRUE(chunk.has_value());
        EXPECT_EQ(coordinator_.DeviceEvictions().request_reclaim.blocks - before.request_reclaim.blocks, 3u)
            << "the three checkpoints at 4";
        return chunk ? std::move(chunk->state_checkpoint_blocks) : std::vector<CacheBlockRef>{};
    }
};

// A session's only checkpoint boundary is its tail at 18 with its checkpoints;
// the Host holds a copy of everything it published. Eight later requests over
// its four pages end in tails of their own without a checkpoint, so the
// per-prior cap drops the session's tail from both tiers, and nothing under its
// pages resumes any more: a collection takes the pages from both tiers and the
// eight tails, KV that no checkpoint follows, from the Device. Each node's
// Device entry leaves first, so the Device evictions count the Host copies.
TEST_F(DeadEntryTest, DroppingTheLastCheckpointCollectsItsDeadPagesOnBothTiers) {
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    coordinator_.Free(tables);
    const std::vector<std::string> pages = hashes_;
    for (const std::string& page : pages) {
        StoreOnHost(Key(page, 0));
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(tail.hash, group));
    }
    for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior; ++extra) {
        std::vector<std::int32_t> tokens = MakeTokens(4 * kP);
        tokens.push_back(1000 * extra);
        std::vector<BlockTable> other(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, other, 4 * kP + 1));
        coordinator_.CacheCompletedBlocks(
            other,
            RequestProgress{
                .completed_pages =
                    CompletedPages{.prefix_hashes = pages,
                                   .first_new_prefix_page = 0,
                                   .tail = TailPublication{.tail_tokens = 1,
                                                           .hash = HashPrefixPage(token_span(tokens).subspan(4 * kP),
                                                                                  pages.back())},
                                   .boundary_kind = CacheBoundaryKind::kEndpoint},
                .num_computed_tokens = 4 * kP + 1},
            CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
        coordinator_.Free(other);
    }
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 4) << "the tail's Host entries left with it";
    ASSERT_EQ(Tree().Size(), 12u) << "four pages and eight tails";

    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(Tree().Size(), 0u);
    EXPECT_EQ(coordinator_.NumTailEntries(), 0u);
    for (const std::string& page : pages) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(page, 0)));
    }
    // The four pages and the eight tails' KV blocks on the Device, the pages
    // on the Host.
    const CacheCoordinator::DeviceEvictionCounts& collected = coordinator_.DeviceEvictions().dead_entry_collection;
    EXPECT_EQ(collected.blocks, 12u);
    EXPECT_EQ(collected.host_resident, 4u);
    const CacheCoordinator::HostEvictionCounts& host = coordinator_.HostEvictions().dead_entry;
    EXPECT_EQ(host.blocks - before.dead_entry.blocks, 4u);
    EXPECT_EQ(host.device_resident - before.dead_entry.device_resident, 0u);
}

// Session A's turn k-1 ends in a checkpoint boundary at 18, turn k in one at
// 26; a write-back left the Device A's first three pages and the Host the rest
// of both turns, and B shares A's first two pages with its checkpoint boundary
// on the Device. One Host plan needs the room of both of A's checkpoint
// boundaries: their nodes go, and with the second the Host pages above both of
// them up to the Device's prefix. That leaves a2, on the Device, with no
// checkpoint boundary under it; the collection takes it, and the pages B's
// checkpoint boundary depends on stay.
TEST_F(DeadEntryTest, TakingBothCheckpointBoundariesOfASessionCollectsThePagesAboveThem) {
    const std::vector<std::int32_t> a = MakeTokens(26);
    std::vector<std::int32_t> b = MakeTokens(18, 100);
    std::copy_n(a.begin(), 2 * kP, b.begin());
    std::vector<BlockTable> b_tables(4);
    PublishFinal(b_tables, b);
    coordinator_.Free(b_tables);
    std::vector<BlockTable> earlier_tables(4);
    const TailPublication earlier = *PublishFinal(earlier_tables, {a.begin(), a.begin() + 18});
    coordinator_.Free(earlier_tables);
    std::vector<BlockTable> later_tables(4);
    const TailPublication later = *PublishFinal(later_tables, a);
    coordinator_.Free(later_tables);
    for (std::size_t page = 3; page < 6; ++page) {
        StoreOnHost(Key(hashes_[page], 0));
    }
    for (const TailPublication& tail : {earlier, later}) {
        for (std::uint32_t group = 0; group < 4; ++group) {
            StoreOnHost(Key(tail.hash, group));
        }
    }
    // Each tail's KV block takes its checkpoints along.
    EvictOnDevice(Key(later.hash, 0));
    EvictOnDevice(Key(earlier.hash, 0));
    for (std::size_t page = 6; page > 3; --page) {
        EvictOnDevice(Key(hashes_[page - 1], 0));
    }
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const std::vector<std::uint32_t> groups{1, 2, 3, 1, 2, 3};
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    const std::array segments{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 6}};
    const CacheCoordinator::HostAllocationBatch batch = coordinator_.AcquireHostSegments(groups, buckets, segments);
    ASSERT_EQ(batch.stats.allocated, 6u);
    EXPECT_EQ(Tree().Find(earlier.hash), nullptr);
    EXPECT_EQ(Tree().Find(later.hash), nullptr);
    for (std::size_t page = 3; page < 6; ++page) {
        EXPECT_EQ(Tree().Find(hashes_[page]), nullptr) << "page " << page;
    }
    ASSERT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[2], 0)));
    EXPECT_EQ(NodeOf(hashes_[2]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));

    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[2], 0)));
    EXPECT_EQ(Tree().Find(hashes_[2]), nullptr);
    for (std::size_t page = 0; page < 2; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    EXPECT_EQ(Tree().Size(), 5u) << "s0, s1, b2, b3 and B's tail";
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 1u) << "a2";
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.host_resident, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 0u)
        << "the Host plan took A's Host pages itself";
}

// A's tail leaves the Device, and with it the last checkpoint boundary under a2
// and a3. Before the next collection, session C publishes s0 s1 a2 c3 with a
// checkpoint boundary of its own: a2 is alive again, a3 is not. The collection
// takes a3 and keeps what C's checkpoint boundary depends on.
TEST_F(DeadEntryTest, ACheckpointPublishedAboveADeadBranchLeavesTheBranchToCollect) {
    PublishTwoSessionsSharingTwoPages();
    EvictOnDevice(Key(a_tail_.hash, 0));
    ASSERT_EQ(NodeOf(a_hashes_[2]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    const std::vector<std::int32_t> a = MakeTokens(18);
    std::vector<std::int32_t> c = MakeTokens(18, 200);
    std::copy_n(a.begin(), 3 * kP, c.begin());
    std::vector<BlockTable> c_tables(4);
    const TailPublication c_tail = *PublishFinal(c_tables, c);
    coordinator_.Free(c_tables);
    ASSERT_EQ(NodeOf(a_hashes_[2]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));
    ASSERT_EQ(NodeOf(a_hashes_[3]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));

    coordinator_.CollectDeadEntries();
    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[3], 0)));
    EXPECT_EQ(Tree().Find(a_hashes_[3]), nullptr);
    for (std::size_t page = 0; page < 4; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(c_tail.hash, group))) << "group " << group;
    }
    EXPECT_EQ(Tree().Size(), 8u) << "s0, s1, a2, c3, C's tail, b2, b3 and B's tail";
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 1u) << "a3";
}

// The request's own reclaim drops the only checkpoint boundary under p0 while
// the request is live: its table holds every page of the dead subtree, so the
// collection takes nothing and the root waits. Once the request goes without
// publishing a checkpoint (an abort), the next collection takes both pages.
TEST_F(DeadEntryTest, ARequestReclaimDropCollectsNothingUnderALiveRequest) {
    std::vector<BlockTable> tables(4);
    std::vector<CacheBlockRef> checkpoints = DropTheCheckpointALiveRequestResumedAt(tables);
    const CacheCoordinator::DeviceEvictionCounts& collected = coordinator_.DeviceEvictions().dead_entry_collection;

    coordinator_.CollectDeadEntries();
    for (std::size_t page = 0; page < 2; ++page) {
        EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    EXPECT_EQ(collected.blocks, 0u);

    checkpoints.clear();
    coordinator_.Free(tables);
    coordinator_.CollectDeadEntries();
    for (std::size_t page = 0; page < 2; ++page) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
    }
    EXPECT_EQ(collected.blocks, 2u);
    EXPECT_EQ(Tree().Size(), 0u);
}

// The same drop, then the request's next chunk publishes p2 with the
// checkpoint at 12 its forward wrote, and its next point releases its hold on
// that checkpoint (as AdvanceRequestProgress does). The waiting root is no
// longer dead when collected, so it is dropped from the queue: the new
// checkpoint and its pages stay, also once the request finishes.
TEST_F(DeadEntryTest, ARevivedRootIsNotCollected) {
    std::vector<BlockTable> tables(4);
    std::vector<CacheBlockRef> checkpoints = DropTheCheckpointALiveRequestResumedAt(tables);
    coordinator_.CollectDeadEntries();
    held_.clear();
    const RequestProgress progress{
        .completed_pages = CompletedPages{.prefix_hashes = hashes_,
                                          .first_new_prefix_page = 2,
                                          .state_checkpoint_pages = 3,
                                          .state_checkpoint_blocks = checkpoints,
                                          .boundary_kind = CacheBoundaryKind::kChunk},
        .num_computed_tokens = 3 * kP,
    };
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             progress));
    checkpoints.clear();
    ASSERT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{1, 0}));

    for (const bool finished : {false, true}) {
        SCOPED_TRACE(finished);
        if (finished) {
            coordinator_.Free(tables);
        }
        coordinator_.CollectDeadEntries();
        for (std::size_t page = 0; page < 3; ++page) {
            EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[page], 0))) << "page " << page;
        }
        for (std::uint32_t group = 1; group <= 3; ++group) {
            EXPECT_TRUE(coordinator_.GroupPrefixIndex(group).Contains(pool_, Key(hashes_[2], group)))
                << "group " << group;
        }
        EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 0u);
    }
}

// A's last page and tail are a Host suffix of the Device's s0 s1 a2. An
// admission's segment takes a2, the oldest Device page and the only one no
// other checkpoint boundary's segment shares, and the Host has no room to write
// it back: a2 becomes a gap, and nothing reaches the Host suffix any more. The
// collection takes it from the Host; B keeps its pages.
TEST_F(DeadEntryTest, AGapStrandsAndCollectsTheHostSuffixBelowIt) {
    LeaveTheEndOfAOnTheHost();
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{kP}},
                             RequestProgress{}, StateStep{.kind = StateStep::Kind::kKeep}));
    // The plan's drain finds no Host room for a2's segment: it is dropped.
    for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
        coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
    }
    ASSERT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[2], 0)));
    ASSERT_NE(Tree().Find(a_hashes_[2]), nullptr) << "a gap: the Host suffix hangs under it";
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5);

    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(Tree().Find(a_hashes_[2]), nullptr);
    EXPECT_EQ(Tree().Size(), 5u) << "s0, s1, b2, b3 and B's tail";
    // The tail's KV block and a3; the group-1 checkpoint takes the others
    // along.
    const CacheCoordinator::HostEvictionCounts& host = coordinator_.HostEvictions().dead_entry;
    EXPECT_EQ(host.blocks - before.dead_entry.blocks, 5u);
    EXPECT_EQ(host.device_resident - before.dead_entry.device_resident, 0u);
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 0u);
    coordinator_.Free(tables);
}

// A session over p0..p3 and a tail of 2, its checkpoint at 8 written back
// alone: the Host holds it over the Device's KV of p1, and the suffix past p1
// with the checkpoint boundary at 18, so a probe resumes at 18. Once the Device
// evicts p1's KV, no tier holds that page's KV, and nothing at or under it
// resumes -- its own checkpoints included, though the Host still holds them on
// the page and they and the checkpoint boundary under it keep p0 counted. The
// page is queued as a gap: the collection takes its subtree from the Host and
// then p0, which no checkpoint boundary is left under, from the Device.
TEST_F(DeadEntryTest, APageWhoseKvLeftBothTiersCutsOffItsOwnCheckpoints) {
    const std::vector<std::int32_t> a = MakeTokens(18);
    std::vector<BlockTable> nested_tables(4);
    PublishFinal(nested_tables, {a.begin(), a.begin() + 2 * kP});
    coordinator_.Free(nested_tables);
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, a);
    coordinator_.Free(tables);
    for (std::uint32_t group = 1; group < 4; ++group) {
        StoreOnHost(Key(hashes_[1], group));
    }
    for (std::size_t page = 2; page < 4; ++page) {
        StoreOnHost(Key(hashes_[page], 0));
    }
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(tail.hash, group));
    }
    // The tail's KV block takes its checkpoints along, the checkpoint at 8
    // the other State groups' at its hash; p1's KV stays.
    EvictOnDevice(Key(tail.hash, 0));
    for (std::size_t page = 4; page > 2; --page) {
        EvictOnDevice(Key(hashes_[page - 1], 0));
    }
    EvictOnDevice(Key(hashes_[1], 1));
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 9);
    const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = hashes_, .tokens = a, .max_tokens = 18};
    ASSERT_EQ(coordinator_.ProbePromptPrefix(prompt).host.num_common_tokens, 18);

    EvictOnDevice(Key(hashes_[1], 0));
    const CacheCoordinator::PrefixProbe probe = coordinator_.ProbePromptPrefix(prompt);
    EXPECT_EQ(probe.device.num_common_tokens, 0);
    EXPECT_EQ(probe.host.num_common_tokens, 0);
    ASSERT_EQ(NodeOf(hashes_[0]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 2}))
        << "the checkpoint at 8 and the checkpoint boundary at 18";

    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(hashes_[0], 0)));
    EXPECT_EQ(Tree().Size(), 0u);
    // The tail's KV block, p3 and p2; each State group's checkpoints at 18
    // and at 8, the group-1 ones taking the others along.
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 9u);
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 1u) << "p0";
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.host_resident, 0u);
}

// The same gap with a load reading A's last page off the Host: the collection
// takes the tail and passes over the page; the root waits, and the first
// collection after the load's ACK released the page takes it.
TEST_F(DeadEntryTest, PinnedDeadEntriesWaitForTheNextPlan) {
    LeaveTheEndOfAOnTheHost();
    CacheBlockRef load = coordinator_.AcquireHostCachedBlock(Key(a_hashes_[3], 0));
    ASSERT_TRUE(static_cast<bool>(load));
    EvictOnDevice(Key(a_hashes_[2], 0));

    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    for (std::int32_t plan = 0; plan < 2; ++plan) {
        coordinator_.CollectDeadEntries();
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(a_hashes_[3], 0))) << "plan " << plan;
        EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 1) << "plan " << plan;
        EXPECT_NE(Tree().Find(a_hashes_[2]), nullptr) << "plan " << plan;
    }
    load.reset();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(Tree().Find(a_hashes_[2]), nullptr);
    EXPECT_EQ(Tree().Size(), 5u);
    // The tail's KV block and checkpoints, then a3.
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 5u);
}

// A Device-only clear (ClearL1Cache) under write-back leaves the Device prefix
// of A's Host suffix gone, but Host L2 untouched: nothing it queued is
// collected.
TEST_F(DeadEntryTest, ClearL1CacheCollectsNothingOnHost) {
    LeaveTheEndOfAOnTheHost();
    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(a_hashes_[3], 0)));
    for (std::uint32_t group = 0; group < 4; ++group) {
        EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(a_tail_.hash, group))) << "group " << group;
    }
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 0u);
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 0u);
}

// After that clear B's end is a Host suffix too, under the gaps s0 and s1,
// and a load reads A's last page. Once the Host evicts A's checkpoints, no
// checkpoint boundary is left under a2; A's tail, which keeps its KV block, is
// queued. The climb from it stops at the page in use, and taking the tail
// leaves that page the deepest one of its branch, queued in turn: from it the
// collection climbs through a2 and the gaps above, so B's suffix under them
// goes as well, though B's checkpoint boundary is still there, and only the
// page in use waits.
TEST_F(DeadEntryTest, TheCollectionClimbsThroughTheGapsAboveADeadRoot) {
    LeaveTheEndOfAOnTheHost();
    StoreOnHost(Key(b_hashes_[3], 0));
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(b_tail_.hash, group));
    }
    EvictOnDevice(Key(b_tail_.hash, 0));
    EvictOnDevice(Key(b_hashes_[3], 0));
    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    CacheBlockRef load = coordinator_.AcquireHostCachedBlock(Key(a_hashes_[3], 0));
    ASSERT_TRUE(static_cast<bool>(load));
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(a_tail_.hash, 1)));
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 7);
    ASSERT_EQ(NodeOf(a_hashes_[2]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 0}));
    ASSERT_EQ(NodeOf(a_hashes_[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{0, 1}))
        << "B's checkpoint boundary";

    // The test access counted A's checkpoints under lru.
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 1);
    EXPECT_TRUE(coordinator_.ContainsHostCachedBlock(Key(a_hashes_[3], 0)));
    // A's tail block, B's last page, tail block and checkpoints.
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 6u);

    load.reset();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(Tree().Size(), 0u);
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 7u) << "and a3";
}

// Only a prefix-closed group needs every page below a boundary. With State
// groups alone, a checkpoint at 8 resumes though nothing holds p0 any more
// (its own checkpoint left): p0 is no gap, and the collection keeps what
// hangs under it.
TEST(DeadEntryWithoutAClosedGroupTest, AStateCheckpointResumesPastAPageHoldingNothing) {
    constexpr std::int32_t kP = 4;
    BlockPool pool(/*num_lcm_blocks=*/8, {1});
    BlockPool host_pool(/*num_lcm_blocks=*/8, {1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1}}, kP, pool,
        /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
        /*write_back_evictions=*/true);
    const std::vector<std::int32_t> tokens = MakeTokens(3 * kP);
    const std::vector<std::string> hashes =
        ContentHashes({{tokens.begin(), tokens.begin() + kP}, {tokens.begin() + kP, tokens.begin() + 2 * kP}});
    // Checkpoints at 4 and at 8, from two requests.
    for (std::size_t pages = 1; pages <= 2; ++pages) {
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, static_cast<std::int32_t>(pages) * kP));
        CacheCompletedBlocksForTest(coordinator, tables, std::span<const std::string>(hashes).first(pages),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, static_cast<std::int32_t>(pages) * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);
        coordinator.Free(tables);
    }
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(
        coordinator, 0, *CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hashes[0], 0))));
    ASSERT_NE(CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator).Find(hashes[0]), nullptr) << "p1 hangs under p0";

    coordinator.CollectDeadEntries();
    EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(hashes[1], 0)));
    EXPECT_EQ(coordinator.DeviceEvictions().dead_entry_collection.blocks, 0u);
    const CacheCoordinator::PrefixProbe probe =
        coordinator.ProbePromptPrefix({.prefix_hashes = hashes, .tokens = tokens, .max_tokens = 2 * kP});
    EXPECT_EQ(probe.device.num_common_tokens, 2 * kP);
}

}  // namespace
// ---------------------------------------------------------------------------
// The write-back transaction on the K3 fixture: an admission under write-back
// records what its cache victims leave behind, the plan's drain
// (TierTransferManager::StartEvictionStores) places it on the Host and issues
// the stream-ordered copies, and each segment publishes, whole, at the last
// ACK it waits for (SettleEvictedSegment).
// ---------------------------------------------------------------------------
class WriteBackTransactionTest : public BoundaryTreeCoordinatorFixture {
protected:
    WriteBackTransactionTest() : WriteBackTransactionTest(/*full_shard_count=*/1, /*num_snapshot_pages=*/0) {}
    WriteBackTransactionTest(std::int32_t full_shard_count, std::int32_t num_snapshot_pages)
        : BoundaryTreeCoordinatorFixture(/*write_back_evictions=*/true, full_shard_count, num_snapshot_pages) {}

    // An admission of one Full block with no free Device slot: it evicts the
    // oldest segment. Then the plan's drain, whose op it returns (nullopt
    // when it copies nothing).
    std::optional<WriteBackOperation> EvictOldestAndDrain() {
        const std::size_t held_before = held_.size();
        HoldFreeCapacity(pool_, /*spare_parents=*/0);
        std::vector<BlockTable> tables(4);
        EXPECT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                 StateStep{.kind = StateStep::Kind::kKeep}));
        coordinator_.Free(tables);
        // The Device room it held goes back; Host room a test held stays held.
        held_.resize(held_before);
        return transfers_.StartEvictionStores();
    }

    // Eight requests over `pages` publish tails one token past them without
    // a checkpoint: the per-prior cap drops the oldest tail under the last
    // page.
    void PublishEightTailsWithoutCheckpoints(const std::vector<std::string>& pages) {
        const auto page_tokens = static_cast<std::int32_t>(pages.size()) * kP;
        for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior; ++extra) {
            std::vector<std::int32_t> tokens = MakeTokens(page_tokens);
            tokens.push_back(1000 * extra);
            std::vector<BlockTable> tables(4);
            ASSERT_TRUE(AdmitForTest(coordinator_, tables, page_tokens + 1));
            coordinator_.CacheCompletedBlocks(
                tables,
                RequestProgress{
                    .completed_pages =
                        CompletedPages{
                            .prefix_hashes = pages,
                            .first_new_prefix_page = 0,
                            .tail = TailPublication{.tail_tokens = 1,
                                                    .hash = HashPrefixPage(token_span(tokens).subspan(
                                                                               static_cast<std::size_t>(page_tokens)),
                                                                           pages.back())},
                            .boundary_kind = CacheBoundaryKind::kEndpoint},
                    .num_computed_tokens = page_tokens + 1},
                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_));
            coordinator_.Free(tables);
        }
    }

    // Every key a published boundary holds (ForEachBoundaryKey).
    std::vector<CacheKey> KeysOf(const std::vector<std::string>& pages, const std::optional<TailPublication>& tail) {
        std::vector<CacheKey> keys;
        for (const std::string& page : pages) {
            coordinator_.ForEachBoundaryKey(Key(page, 0), 0, [&](std::size_t, const CacheKey& key) {
                // A page holds a checkpoint only at a natural point.
                if (!coordinator_.GroupIsState(key.group_id)) {
                    keys.push_back(key);
                }
            });
        }
        if (tail) {
            coordinator_.ForEachBoundaryKey(Key(tail->hash, 0), tail->tail_tokens,
                                            [&](std::size_t, const CacheKey& key) { keys.push_back(key); });
        }
        return keys;
    }

    std::int32_t HostEntries(const std::vector<CacheKey>& keys) const {
        return static_cast<std::int32_t>(std::ranges::count_if(
            keys, [&](const CacheKey& key) { return coordinator_.ContainsHostCachedBlock(key); }));
    }

    // The slot-state blob slot of a retraction image; it outlives transfers_,
    // whose snapshot store holds it until its ACK.
    SnapshotSlotAllocator blob_slots_{1};
    TierTransferManager transfers_{coordinator_};
};

// A block is recorded only once its eviction succeeded, and only once. A
// request reading A's group-3 checkpoint keeps A's whole segment out of the
// first admission, which takes B's (its tail's KV block and checkpoints
// leave through the group-1 checkpoint's cascade, recorded there and not
// again); A's segment, with the pages B's no longer needs, goes with the
// second admission once the request let go.
TEST_F(WriteBackTransactionTest, AVictimIsRecordedOnlyOnceEvicted) {
    PublishTwoSessionsSharingTwoPages();
    CacheBlockRef reader = coordinator_.GroupPrefixIndex(3).Find(pool_, Key(a_tail_.hash, 3));
    ASSERT_TRUE(static_cast<bool>(reader));
    const auto record = [&] {
        HoldFreeCapacity(pool_, /*spare_parents=*/0);
        std::vector<BlockTable> tables(4);
        EXPECT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                 StateStep{.kind = StateStep::Kind::kKeep}));
        coordinator_.Free(tables);
        held_.clear();
        std::vector<CacheKey> keys;
        for (CacheCoordinator::EvictedSegment& segment : coordinator_.TakeEvictedSegments()) {
            for (const CacheCoordinator::EvictedBlock& member : segment.members) {
                keys.push_back(member.key);
            }
            // The plan's drain finds no Host room for it: no store holds a
            // node or shows on the Host for the next admission.
            coordinator_.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
        }
        for (const CacheKey& key : keys) {
            if (const Node* node = Tree().Find(key.content_hash)) {
                EXPECT_EQ(node->pending_host_stores, 0);
                EXPECT_EQ(node->pending_host_checkpoint_stores, 0);
            }
        }
        return keys;
    };
    const auto sorted = [](std::vector<CacheKey> keys) {
        std::ranges::sort(
            keys, {}, [](const CacheKey& key) { return std::tuple{key.content_hash, key.group_id, key.page_offset}; });
        return keys;
    };
    EXPECT_EQ(sorted(record()), sorted(KeysOf({b_hashes_[2], b_hashes_[3]}, b_tail_)));
    reader.reset();
    EXPECT_EQ(sorted(record()), sorted(KeysOf(a_hashes_, a_tail_)));
}

// Two segments of two plans: each publishes at its own op's ACK, whole,
// whichever ACK comes first.
TEST_F(WriteBackTransactionTest, AckOutOfOrderPublishesASegmentWhole) {
    for (const bool first_acked_first : {true, false}) {
        SCOPED_TRACE(first_acked_first);
        std::vector<BlockTable> tables(4);
        const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18, 100 * (first_acked_first ? 1 : 3)));
        const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
        coordinator_.Free(tables);
        const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
        ASSERT_TRUE(x_op.has_value());
        const TailPublication y_tail = *PublishFinal(tables, MakeTokens(18, 100 * (first_acked_first ? 2 : 4)));
        const std::vector<CacheKey> y = KeysOf(hashes_, y_tail);
        coordinator_.Free(tables);
        const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
        ASSERT_TRUE(y_op.has_value());
        ASSERT_EQ(x_op->transfers.size(), 8u);
        ASSERT_EQ(y_op->transfers.size(), 8u);

        transfers_.CompleteWriteBack(first_acked_first ? x_op->op_id : y_op->op_id);
        EXPECT_EQ(HostEntries(x), first_acked_first ? 8 : 0);
        EXPECT_EQ(HostEntries(y), first_acked_first ? 0 : 8);
        transfers_.CompleteWriteBack(first_acked_first ? y_op->op_id : x_op->op_id);
        EXPECT_EQ(HostEntries(x), 8);
        EXPECT_EQ(HostEntries(y), 8);
    }
}

// Y continues X's four pages, which X's write-back still carries: Y's segment
// borrows them from X's op and copies only its tail and checkpoints in its
// own. It publishes once both ops landed -- in either order -- and a repeated
// ACK changes nothing.
TEST_F(WriteBackTransactionTest, ASegmentPublishesOnlyAtItsLastAck) {
    for (const bool own_op_first : {true, false}) {
        SCOPED_TRACE(own_op_first);
        const std::int32_t start = own_op_first ? 1 : 500;
        std::vector<BlockTable> tables(4);
        const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18, start));
        const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
        coordinator_.Free(tables);
        const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
        ASSERT_TRUE(x_op.has_value());
        std::vector<std::int32_t> y_tokens = MakeTokens(18, start);
        y_tokens[16] = 9000;
        const TailPublication y_tail = *PublishFinal(tables, y_tokens);
        coordinator_.Free(tables);
        const std::vector<CacheKey> y_own = KeysOf({}, y_tail);
        const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
        ASSERT_TRUE(y_op.has_value());
        EXPECT_EQ(y_op->transfers.size(), 4u) << "the tail's KV block and its three checkpoints";
        const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
        EXPECT_EQ(stats.in_flight, own_op_first ? 4u : 8u) << "the four pages, borrowed";

        if (own_op_first) {
            EXPECT_TRUE(transfers_.CompleteWriteBack(y_op->op_id).empty()) << "Y still waits for X's op";
            EXPECT_EQ(HostEntries(y_own), 0);
            EXPECT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u + 8u) << "X's segment, then Y's";
        } else {
            EXPECT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u);
            EXPECT_EQ(HostEntries(y_own), 0);
            EXPECT_EQ(transfers_.CompleteWriteBack(y_op->op_id).size(), 8u);
        }
        EXPECT_EQ(HostEntries(x), 8);
        EXPECT_EQ(HostEntries(y_own), 4);
        const std::int32_t host_entries = coordinator_.NumHostCachedBlocks();
        EXPECT_TRUE(transfers_.CompleteWriteBack(x_op->op_id).empty());
        EXPECT_TRUE(transfers_.CompleteWriteBack(y_op->op_id).empty());
        EXPECT_EQ(coordinator_.NumHostCachedBlocks(), host_entries);
    }
}

// The Host holds every block of X already: the eviction copies nothing and
// issues no op, and the segment settles at once, each entry with the newer of
// its Host and Device epochs -- the Device's for the pages, which a hit
// touched after their Host copies were made, the Host's for the tail and its
// checkpoints, copied after that hit.
TEST_F(WriteBackTransactionTest, ASegmentAlreadyOnTheHostSettlesWithoutAnOp) {
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, tokens);
    coordinator_.Free(tables);
    const std::vector<CacheKey> keys = KeysOf(hashes_, tail);
    for (std::size_t page = 0; page < 4; ++page) {
        StoreOnHost(keys[page]);
    }
    CacheCoordinatorTestAccess::MatchPromptPrefix(
        coordinator_, CacheCoordinator::PromptPrefix{.prefix_hashes = hashes_, .tokens = tokens, .max_tokens = 18});
    for (std::size_t member = 4; member < keys.size(); ++member) {
        StoreOnHost(keys[member]);
    }
    const auto epoch_on = [&](const BlockPool& pool, const CacheKey& key) {
        const PrefixCacheIndex& index = coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id));
        return index.MetadataFor(pool, index.Find(pool, key)->Location())->last_access_epoch;
    };
    // The epochs each entry is published with: the Device's for a page, the
    // Host's own for the rest.
    std::vector<std::uint64_t> expected;
    for (std::size_t member = 0; member < keys.size(); ++member) {
        const std::uint64_t device = epoch_on(pool_, keys[member]);
        const std::uint64_t host = epoch_on(host_pool_, keys[member]);
        ASSERT_EQ(device > host, member < 4) << "member " << member;
        expected.push_back(std::max(device, host));
    }

    EXPECT_FALSE(EvictOldestAndDrain().has_value());
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.ops, 0u);
    EXPECT_EQ(stats.host_cached, 8u);
    EXPECT_EQ(stats.cancelled, 0u);
    EXPECT_FALSE(transfers_.HasAnyInFlight());
    for (std::size_t member = 0; member < keys.size(); ++member) {
        EXPECT_EQ(epoch_on(host_pool_, keys[member]), expected[member]) << "member " << member;
    }
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
}

// Y continues X's four pages with a page and a tail of its own, and the Host
// already holds Y's page at 20: Y's segment holds that entry, borrows X's
// pages from X's op still in flight and copies only its tail and
// checkpoints. Nothing of Y publishes at its own ACK; X's lands it all.
TEST_F(WriteBackTransactionTest, MembersAlreadyOnTheHostBorrowedAndCopiedPublishTogether) {
    std::vector<BlockTable> tables(4);
    const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    std::vector<std::int32_t> y_tokens = MakeTokens(16);
    for (std::int32_t i = 0; i < 6; ++i) {
        y_tokens.push_back(700 + i);
    }
    const TailPublication y_tail = *PublishFinal(tables, y_tokens);
    coordinator_.Free(tables);
    StoreOnHost(Key(hashes_[4], 0));
    const std::vector<CacheKey> y_own = KeysOf({}, y_tail);
    const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
    ASSERT_TRUE(y_op.has_value());
    EXPECT_EQ(y_op->transfers.size(), 4u);
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.host_cached, 1u);
    EXPECT_EQ(stats.in_flight, 4u);
    EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{5 + 1, 1 + 1, 1 + 1, 1 + 1}));
    EXPECT_EQ(stats.ops, 2u);

    EXPECT_TRUE(transfers_.CompleteWriteBack(y_op->op_id).empty());
    EXPECT_EQ(HostEntries(y_own), 0);
    EXPECT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u + 9u);
    EXPECT_EQ(HostEntries(x), 8);
    EXPECT_EQ(HostEntries(y_own), 4);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8 + 1 + 4);
    EXPECT_EQ(stats.cancelled, 0u);
}

// X's write-back is in flight when the same prompt publishes X's pages on
// the Device again and a publication store is queued for them: the store
// copies them itself rather than counting them in flight. An eviction
// write-back's tickets publish with their segment, which may wait for other
// ops or be cancelled, not at their own op's ACK. Both land on one Host
// entry per key.
TEST_F(WriteBackTransactionTest, APublicationStoreCopiesWhatAnEvictionWriteBackCarries) {
    std::vector<BlockTable> tables(4);
    const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    ASSERT_EQ(x_op->transfers.size(), 8u);
    PublishFinal(tables, MakeTokens(18));
    coordinator_.QueueCachedBlocksForStore(hashes_);
    const std::optional<WriteBackOperation> store =
        transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication);
    ASSERT_TRUE(store.has_value());
    EXPECT_EQ(store->transfers.size(), 4u) << "X's four pages";
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kPublication).in_flight, 0u);
    EXPECT_EQ(transfers_.CompleteWriteBack(store->op_id).size(), 4u);
    EXPECT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u);
    coordinator_.Free(tables);
    EXPECT_EQ(HostEntries(x), 8);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
}

// X's write-back is in flight when eight tails without a checkpoint arrive
// under its last page and the per-prior cap drops X's tail: at the ACK the tail
// and its checkpoints are cancelled, and X's pages, which no checkpoint
// boundary depends on any more, with them. Nothing is published; the Host
// blocks the copy filled are free again; the eight tails stay.
TEST_F(WriteBackTransactionTest, ACappedTailIsNotPublishedAtItsAck) {
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 8u);
    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    ASSERT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    ASSERT_NE(Tree().Find(tail.hash), nullptr) << "held by the store";

    EXPECT_TRUE(transfers_.CompleteWriteBack(op->op_id).empty());
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 8u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), 32);
    EXPECT_EQ(Tree().Find(tail.hash), nullptr);
    EXPECT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
}

// As above, the cap drops X's tail hint while X's write-back is out, but X
// is published again before the ACK: the hint is back -- the republished
// tail is the newest under the page, so the cap drops another -- and the
// settlement judges the hint as it then stands. Every member is published once,
// nothing is cancelled, and the store leaves no Host block or tree node hold
// behind.
TEST_F(WriteBackTransactionTest, ATailPublishedAgainBeforeTheAckIsPublished) {
    std::vector<BlockTable> tables(4);
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    const TailPublication tail = *PublishFinal(tables, tokens);
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 8u);
    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    PublishFinal(tables, tokens);
    coordinator_.Free(tables);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 2);
    ASSERT_EQ(coordinator_.NumTailEntries(), static_cast<std::size_t>(TailTable::kMaxTailsPerPrior));
    ASSERT_EQ(NodeOf(tail.hash).pending_host_stores, 4) << "the tail block and its three checkpoints";

    EXPECT_EQ(transfers_.CompleteWriteBack(op->op_id).size(), 8u);
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.ops, 1u);
    EXPECT_EQ(stats.cancelled, 0u);
    EXPECT_EQ(HostEntries(KeysOf(pages, tail)), 8);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
    EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), 32 - 5) << "two KV parents and three checkpoint parents";
    for (const std::string& hash : {pages[0], pages[1], pages[2], pages[3], tail.hash}) {
        EXPECT_EQ(NodeOf(hash).pending_host_stores, 0);
        EXPECT_EQ(NodeOf(hash).pending_host_checkpoint_stores, 0);
    }
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(tail.hash, 0)))
        << "the Device keeps the republished tail";
}

// A request reading X's first page keeps it on the Device, so X's segment
// stops below it: p0 is a Device prefix the Host suffix continues. While the
// write-back is out, p0 serves the checkpoint boundary on its way; the cap
// cancels that checkpoint boundary, and its settlement queues what it leaves
// dead, so the next collection takes p0 -- with the pages and tails the capping
// requests left on the Device, none of which a checkpoint boundary depends on.
TEST_F(WriteBackTransactionTest, AStoreThatSettlesDeadIsCollected) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(18));
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    CacheBlockRef reader = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(pages[0], 0));
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 7u) << "p1 to p3, the tail block and its checkpoints";
    reader.reset();
    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);

    coordinator_.CollectDeadEntries();
    EXPECT_TRUE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(pages[0], 0)))
        << "the checkpoint boundary on its way still depends on it";
    EXPECT_TRUE(transfers_.CompleteWriteBack(op->op_id).empty());
    coordinator_.CollectDeadEntries();
    for (const std::string& page : pages) {
        EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(page, 0)));
    }
    EXPECT_EQ(coordinator_.DeviceEvictions().dead_entry_collection.blocks, 4u + 8u) << "the pages and the eight tails";
    EXPECT_EQ(Tree().Size(), 0u);
}

// A's segment -- a2, a3, the tail block and its checkpoints -- is still out
// when the next admission takes B with s0 and s1, which no checkpoint boundary
// on the Device shares any more, and the full Host drops B's segment: s1 is a
// gap above A's checkpoint boundary on its way. The collection takes nothing
// that has not landed and waits; A's segment publishes under the gap at its
// ACK, and the next collection takes it.
TEST_F(WriteBackTransactionTest, ASegmentLandingUnderAGapIsCollectedOnceItLands) {
    PublishTwoSessionsSharingTwoPages();
    const std::optional<WriteBackOperation> a_op = EvictOldestAndDrain();
    ASSERT_TRUE(a_op.has_value());
    ASSERT_EQ(a_op->transfers.size(), 6u);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    EXPECT_FALSE(EvictOldestAndDrain().has_value());
    ASSERT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).no_placement, 8u) << "B's, with s0 and s1";
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(transfers_.CompleteWriteBack(a_op->op_id).size(), 6u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 6);
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0) << "nothing resumes under the gap";
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 6u)
        << "a2, a3, the tail block and its checkpoints";
    EXPECT_EQ(Tree().Size(), 0u);
}

// X's tail block was on the Host already, so X's segment holds that entry
// instead of copying it. The cap drops X's tail meanwhile; the held entry
// cannot leave then, and leaves at the settlement, as a dead entry.
TEST_F(WriteBackTransactionTest, ACancelledMemberAlreadyOnTheHostLeavesIt) {
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    StoreOnHost(Key(tail.hash, 0));
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 7u);
    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_TRUE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, 0))) << "held by the store";
    EXPECT_EQ(coordinator_.HostEvictions().tail_cap.blocks, 0u);

    EXPECT_TRUE(transfers_.CompleteWriteBack(op->op_id).empty());
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(tail.hash, 0)));
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks, 1u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 8u);
}

// One admission evicts X and Y; the Host has room for Y's segment alone, and
// the one entry it could otherwise evict is X's tail block, which X's own
// segment holds. X, the older, is dropped whole: its held entry stays as it
// was, and nothing else of X reaches the Host.
TEST_F(WriteBackTransactionTest, AMemberAlreadyOnTheHostSurvivesTheDrainsOwnPlanning) {
    std::vector<BlockTable> tables(4);
    const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
    coordinator_.Free(tables);
    const TailPublication y_tail = *PublishFinal(tables, MakeTokens(18, 100));
    const std::vector<CacheKey> y = KeysOf(hashes_, y_tail);
    coordinator_.Free(tables);
    StoreOnHost(Key(x_tail.hash, 0));
    // Y's two Full parents and three State parents.
    HoldFreeCapacity(host_pool_, /*spare_parents=*/5);
    HoldFreeCapacity(pool_, /*spare_parents=*/0);
    // A first chunk's three zero states and three live slots.
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> op = transfers_.StartEvictionStores();
    ASSERT_TRUE(op.has_value());
    EXPECT_EQ(op->transfers.size(), 8u) << "Y's";
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.no_placement, 7u) << "X's but the held one";
    EXPECT_EQ(stats.host_cached, 1u);
    EXPECT_EQ(HostEntries(x), 1);
    transfers_.CompleteWriteBack(op->op_id);
    EXPECT_EQ(HostEntries(x), 1);
    EXPECT_EQ(HostEntries(y), 8);
}

// The Device holds s0 s1 a2 of session A, the Host the rest of A, a3 and its
// tail with the checkpoints; the Host is full. An admission takes a2, the
// page that suffix continues, and records it for the Host. The Host plan
// passes A's suffix over -- with its parent on its way, it would arrive with
// nothing under it -- so a2's segment cannot be placed and is dropped. a2 is
// then a gap, and the collection takes the suffix it strands.
TEST_F(WriteBackTransactionTest, AnInFlightSegmentAboveKeepsItsHostSuffix) {
    PublishTwoSessionsSharingTwoPages();
    StoreOnHost(Key(a_hashes_[3], 0));
    for (std::uint32_t group = 0; group < 4; ++group) {
        StoreOnHost(Key(a_tail_.hash, group));
    }
    EvictOnDevice(Key(a_tail_.hash, 0));
    EvictOnDevice(Key(a_hashes_[3], 0));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    EXPECT_FALSE(EvictOldestAndDrain().has_value());
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.no_placement, 1u) << "a2";
    EXPECT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(a_hashes_[2], 0)));
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5) << "the Host plan took none of A's suffix";
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
}

// X's checkpoints at 16 are on the Host alone, over the Device's KV of X's
// four pages; Y is on the Host alone, its entries newer; the Host is full. A
// request reading x2 keeps X's first three pages, so the admission takes x3 and
// records its KV for the Host, which X's Host checkpoint boundary keeps alive.
// The Host plan passes X's checkpoints over, older as they are -- the page on
// its way would arrive with no checkpoint boundary under it -- and takes Y:
// x3's KV lands under X's checkpoints, and X resumes at 16 on the Device's
// pages and the Host's.
TEST_F(WriteBackTransactionTest, AStoreOnItsWayKeepsTheHostCheckpointItArrivesUnder) {
    std::vector<BlockTable> tables(4);
    const std::vector<std::int32_t> x = MakeTokens(4 * kP);
    PublishFinal(tables, x);
    coordinator_.Free(tables);
    const std::vector<std::string> x_pages = hashes_;
    for (std::uint32_t group = 1; group <= 3; ++group) {
        StoreOnHost(Key(x_pages[3], group));
    }
    EvictOnDevice(Key(x_pages[3], 1));
    const TailPublication y_tail = *PublishFinal(tables, MakeTokens(18, 100));
    coordinator_.Free(tables);
    const std::vector<CacheKey> y = KeysOf(hashes_, y_tail);
    for (const CacheKey& key : y) {
        StoreOnHost(key);
    }
    EvictOnDevice(Key(y_tail.hash, 0));
    for (std::size_t page = hashes_.size(); page > 0; --page) {
        EvictOnDevice(Key(hashes_[page - 1], 0));
    }
    CacheBlockRef reader = coordinator_.GroupPrefixIndex(0).Find(pool_, Key(x_pages[2], 0));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 1u);
    EXPECT_EQ(op->transfers.front().content_hash, x_pages[3]);
    EXPECT_EQ(HostEntries(y), 0);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 3) << "X's checkpoints";
    EXPECT_EQ(transfers_.CompleteWriteBack(op->op_id).size(), 1u);
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 0u);
    reader.reset();
    std::vector<std::int32_t> next = x;
    next.push_back(77);
    const CacheCoordinator::PrefixProbe probe =
        coordinator_.ProbePromptPrefix({.prefix_hashes = x_pages, .tokens = next, .max_tokens = 4 * kP});
    EXPECT_EQ(probe.host.num_common_tokens, 4 * kP);
}

// One request publishes a checkpoint at 8 at a chunk end and its final one at
// 16, under one access epoch. The plan's first admission takes the nested
// checkpoint at 8 alone; the second the segment of the checkpoint boundary at
// 16, which now climbs through 8 to the root. Placed apart, the checkpoint at 8
// -- first in the drain's order at an equal epoch -- could reach the Host
// without the KV of its page; merged, the two arrive together or not at all:
// (a) with room for the checkpoints at 8 only, neither is copied; (b) with room
// for both, one op carries them and one ACK publishes them.
TEST_F(WriteBackTransactionTest, TwoAdmitsInOnePlanMergeDependentSegments) {
    for (const bool room_for_both : {false, true}) {
        SCOPED_TRACE(room_for_both);
        const std::vector<std::int32_t> tokens = MakeTokens(16, room_for_both ? 100 : 1);
        const std::vector<std::string> pages = ContentHashes({{tokens.begin(), tokens.begin() + kP},
                                                              {tokens.begin() + kP, tokens.begin() + 2 * kP},
                                                              {tokens.begin() + 2 * kP, tokens.begin() + 3 * kP},
                                                              {tokens.begin() + 3 * kP, tokens.end()}});
        std::vector<BlockTable> tables(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, 2 * kP));
        const std::optional<CacheCoordinator::AdmissionResult> second = AdmitForTest(
            coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{2 * kP}},
            RequestProgress{
                .completed_pages = CompletedPages{.prefix_hashes = std::span<const std::string>(pages).first(2),
                                                  .first_new_prefix_page = 0,
                                                  .publish_state = true,
                                                  .boundary_kind = CacheBoundaryKind::kChunk},
                .num_computed_tokens = 2 * kP});
        ASSERT_TRUE(second.has_value());
        coordinator_.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = pages,
                                                              .first_new_prefix_page = 2,
                                                              .publish_state = true,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = 4 * kP},
            second->access_epoch);
        coordinator_.Free(tables);
        ASSERT_EQ(NodeOf(pages[1]).checkpoint_boundaries_in_subtree, (CheckpointBoundaryCounts{2, 0}));
        for (std::int32_t admission = 0; admission < 2; ++admission) {
            HoldFreeCapacity(pool_, /*spare_parents=*/0);
            ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                     GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                     StateStep{.kind = StateStep::Kind::kKeep}));
            coordinator_.Free(tables);
            held_.clear();
            ASSERT_EQ(NodeOf(pages[1]).has_checkpoint[kDevice], false);
        }
        ASSERT_FALSE(coordinator_.GroupPrefixIndex(0).Contains(pool_, Key(pages[1], 0)));
        const std::vector<CacheKey> keys = [&] {
            std::vector<CacheKey> all = KeysOf(pages, std::nullopt);
            for (const std::size_t page : {1, 3}) {
                for (std::uint32_t group = 1; group <= 3; ++group) {
                    all.push_back(Key(pages[page], group));
                }
            }
            return all;
        }();
        if (!room_for_both) {
            HoldFreeCapacity(host_pool_, /*spare_parents=*/3);
        }
        const std::optional<WriteBackOperation> op = transfers_.StartEvictionStores();
        held_.clear();
        const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
        if (!room_for_both) {
            EXPECT_FALSE(op.has_value());
            EXPECT_EQ(stats.no_placement, 10u);
            EXPECT_EQ(HostEntries(keys), 0);
            continue;
        }
        ASSERT_TRUE(op.has_value());
        EXPECT_EQ(op->transfers.size(), 10u);
        EXPECT_EQ(HostEntries(keys), 0);
        EXPECT_EQ(transfers_.CompleteWriteBack(op->op_id).size(), 10u);
        EXPECT_EQ(HostEntries(keys), 10);
    }
}

// The nested checkpoint at 8 is evicted, published again and evicted again in
// one build. Its two segments hang under no segment's KV page, yet record the
// same keys: they are merged, and the drain copies each key once, the second
// segment's members sharing the first one's copies.
TEST_F(WriteBackTransactionTest, SegmentsRecordingOneKeyAreMergedAndCopyItOnce) {
    const auto publish = [&](std::int32_t num_tokens) {
        std::vector<BlockTable> tables(4);
        PublishFinal(tables, MakeTokens(num_tokens));
        coordinator_.Free(tables);
    };
    const auto evict_oldest = [&] {
        HoldFreeCapacity(pool_, /*spare_parents=*/0);
        std::vector<BlockTable> tables(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, coordinator_.ProbePrefix({}),
                                 GroupDemand{.extent = DenseGrowth{kP}}, RequestProgress{},
                                 StateStep{.kind = StateStep::Kind::kKeep}));
        coordinator_.Free(tables);
        held_.clear();
    };
    // Each time the checkpoint at 8 is the oldest entry, under the one at 16.
    for (std::int32_t round = 0; round < 2; ++round) {
        publish(8);
        publish(16);
        evict_oldest();
    }
    const std::string page = hashes_[1];
    std::vector<CacheKey> keys;
    for (std::uint32_t group = 1; group <= 3; ++group) {
        keys.push_back(Key(page, group));
    }
    ASSERT_EQ(coordinator_.DeviceEvictions().admission.blocks, 6u) << "the three checkpoints at 8, twice";
    const std::optional<WriteBackOperation> op = transfers_.StartEvictionStores();
    ASSERT_TRUE(op.has_value());
    EXPECT_EQ(op->transfers.size(), 3u);
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).in_flight, 3u) << "the second three";
    transfers_.CompleteWriteBack(op->op_id);
    EXPECT_EQ(HostEntries(keys), 3);
    EXPECT_EQ(NodeOf(page).pending_host_stores, 0);
}

// A key published, evicted, published again from another block and evicted
// again in one build is copied once for both records: a Host with room for
// one block takes it, and the second record counts as in flight. On a Host
// whose one block another entry pins, the segment is dropped and both records
// count as no_placement.
TEST(EvictionWriteBackTest, AKeyEvictedTwiceInOneBuildIsCopiedOnce) {
    for (const std::int32_t free_host_blocks : {0, 1, 2}) {
        SCOPED_TRACE(free_host_blocks);
        const std::vector<CacheGroupSpec> specs{
            CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        };
        BlockPool pool(/*num_lcm_blocks=*/2, {1});
        BlockPool host_pool(std::max(free_host_blocks, 1), {1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, /*prefix_granularity=*/2, pool, /*enable_l3_storage=*/false, &host_pool,
                            /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                            /*write_back_evictions=*/true);
        TierTransferManager transfers(coordinator);
        CacheBlockRef pinned;
        if (free_host_blocks == 0) {
            pinned = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(pinned, Key(ContentHashes({{91, 92}})[0], 0), std::nullopt);
        }
        const std::vector<std::string> page = ContentHashes({{1, 2}});
        for (std::int32_t round = 0; round < 2; ++round) {
            std::vector<BlockTable> publisher(1);
            ASSERT_TRUE(AdmitForTest(coordinator, publisher, /*num_tokens=*/2));
            CacheCompletedBlocksForTest(coordinator, publisher, page,
                                        CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                        /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2,
                                        CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                        /*publish_state=*/false);
            coordinator.Free(publisher);
            // Both Device blocks: the page leaves.
            std::vector<BlockTable> evictor(1);
            ASSERT_TRUE(AdmitForTest(coordinator, evictor, /*num_tokens=*/4));
            coordinator.Free(evictor);
        }
        ASSERT_EQ(coordinator.DeviceEvictions().admission.blocks, 2u);
        const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
        const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kEviction);
        if (free_host_blocks == 0) {
            EXPECT_FALSE(op.has_value());
            EXPECT_EQ(stats.no_placement, 2u);
            EXPECT_EQ(stats.in_flight, 0u);
            EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(page[0], 0)));
            continue;
        }
        ASSERT_TRUE(op.has_value());
        EXPECT_EQ(op->transfers.size(), 1u);
        EXPECT_EQ(stats.in_flight, 1u);
        transfers.CompleteWriteBack(op->op_id);
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(page[0], 0)));
    }
}

// The Host dropped X's last page after X was written back; X is published on
// the Device again and evicted again. Everything else of it is still on the
// Host and is held, not copied; the last page is copied a second time, a
// recopy.
TEST_F(WriteBackTransactionTest, AKeyWrittenBackBeforeCountsAsRecopied) {
    std::vector<BlockTable> tables(4);
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    PublishFinal(tables, tokens);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> first = EvictOldestAndDrain();
    ASSERT_TRUE(first.has_value());
    transfers_.CompleteWriteBack(first->op_id);
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(hashes_[3], 0)));
    PublishFinal(tables, tokens);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> second = EvictOldestAndDrain();
    ASSERT_TRUE(second.has_value());
    ASSERT_EQ(second->transfers.size(), 1u);
    EXPECT_EQ(second->transfers.front().content_hash, hashes_[3]);
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.recopied, 1u);
    EXPECT_EQ(stats.host_cached, 7u);
    EXPECT_EQ(stats.blocks[0], 5u + 1u);
    transfers_.CompleteWriteBack(second->op_id);
}

// The same through the Host's own reclaim. X's earlier turn is written back;
// Y and X's longer turn are published on the Device without acquiring X's
// input checkpoint. With no continuation proof, Host LRU reclaims the older
// X checkpoint and its pages. X's longer turn copies those pages again.
TEST_F(WriteBackTransactionTest, PagesOfAReclaimedHostCheckpointAreRecopied) {
    const std::vector<std::int32_t> tokens = MakeTokens(26);
    std::vector<BlockTable> tables(4);
    const TailPublication earlier = *PublishFinal(tables, {tokens.begin(), tokens.begin() + 18});
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    ASSERT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u);
    PublishFinal(tables, MakeTokens(18, 100));
    coordinator_.Free(tables);
    const TailPublication later = *PublishFinal(tables, tokens);
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);

    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
    ASSERT_TRUE(y_op.has_value());
    EXPECT_EQ(y_op->transfers.size(), 8u);
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(coordinator_.HostEvictions().duplicate_kv.blocks - before.duplicate_kv.blocks, 0u);
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks - before.lru.blocks, 8u)
        << "the older checkpoint, its tail block and four pages leave together";
    EXPECT_EQ(HostEntries(KeysOf({pages.begin(), pages.begin() + 4}, earlier)), 0);
    transfers_.CompleteWriteBack(y_op->op_id);
    held_.clear();

    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    EXPECT_EQ(op->transfers.size(), 10u) << "six pages, the tail block and its checkpoints";
    EXPECT_EQ(stats.recopied, 4u);
    EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{4 + 1 + 4 + 1 + 6 + 1, 3, 3, 3}));
    EXPECT_EQ(transfers_.CompleteWriteBack(op->op_id).size(), 10u);
    EXPECT_EQ(HostEntries(KeysOf(pages, later)), 10);
}

// A request publishes a checkpoint at 8 at a chunk end and its final one at
// 16. The first eviction writes back the nested checkpoints at 8 alone, the
// second the segment of the checkpoint boundary at 16, which then climbs
// through 8: the KV of page 8 is copied for the first time, no recopy, though
// the Host already holds the checkpoints at its hash.
TEST_F(WriteBackTransactionTest, AFirstCopyBesideWrittenBackCheckpointsIsNoRecopy) {
    const std::vector<std::int32_t> tokens = MakeTokens(16);
    const std::vector<std::string> pages = ContentHashes({{tokens.begin(), tokens.begin() + kP},
                                                          {tokens.begin() + kP, tokens.begin() + 2 * kP},
                                                          {tokens.begin() + 2 * kP, tokens.begin() + 3 * kP},
                                                          {tokens.begin() + 3 * kP, tokens.end()}});
    std::vector<BlockTable> tables(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, tables, 2 * kP));
    const std::optional<CacheCoordinator::AdmissionResult> second = AdmitForTest(
        coordinator_, tables, coordinator_.ProbePrefix({}), GroupDemand{.extent = DenseGrowth{2 * kP}},
        RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = std::span<const std::string>(pages).first(2),
                                                          .first_new_prefix_page = 0,
                                                          .publish_state = true,
                                                          .boundary_kind = CacheBoundaryKind::kChunk},
                        .num_computed_tokens = 2 * kP});
    ASSERT_TRUE(second.has_value());
    coordinator_.CacheCompletedBlocks(
        tables,
        RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = pages,
                                                          .first_new_prefix_page = 2,
                                                          .publish_state = true,
                                                          .boundary_kind = CacheBoundaryKind::kEndpoint},
                        .num_computed_tokens = 4 * kP},
        second->access_epoch);
    coordinator_.Free(tables);

    const std::optional<WriteBackOperation> first = EvictOldestAndDrain();
    ASSERT_TRUE(first.has_value());
    ASSERT_EQ(first->transfers.size(), 3u) << "the checkpoints at 8";
    transfers_.CompleteWriteBack(first->op_id);
    const std::optional<WriteBackOperation> last = EvictOldestAndDrain();
    ASSERT_TRUE(last.has_value());
    ASSERT_EQ(last->transfers.size(), 7u) << "the four pages and the checkpoints at 16";
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.recopied, 0u);
    EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{4, 2, 2, 2}));
    transfers_.CompleteWriteBack(last->op_id);
}

// A Host entry an eviction wrote back is the Device entry it was: the epoch
// of its last Device access, its logical block index and boundary kind, and
// never a prefix hit's.
TEST_F(WriteBackTransactionTest, HostEntriesInheritDeviceMetadata) {
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, MakeTokens(18));
    coordinator_.Free(tables);
    const std::vector<CacheKey> keys = KeysOf(hashes_, tail);
    std::vector<PrefixCacheIndex::CachedBlockMetadata> device;
    for (const CacheKey& key : keys) {
        const PrefixCacheIndex& index = coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id));
        device.push_back(*index.MetadataFor(pool_, index.Find(pool_, key)->Location()));
    }
    const std::uint64_t next_epoch = CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_);
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    transfers_.CompleteWriteBack(op->op_id);
    for (std::size_t member = 0; member < keys.size(); ++member) {
        SCOPED_TRACE(member);
        const PrefixCacheIndex& index = coordinator_.GroupPrefixIndex(static_cast<std::int32_t>(keys[member].group_id));
        const PrefixCacheIndex::CachedBlockMetadata host =
            *index.MetadataFor(host_pool_, index.Find(host_pool_, keys[member])->Location());
        EXPECT_EQ(host.last_access_epoch, device[member].last_access_epoch);
        EXPECT_LT(host.last_access_epoch, next_epoch);
        EXPECT_EQ(host.logical_block_index, device[member].logical_block_index);
        EXPECT_NE(host.logical_block_index, -1);
        EXPECT_EQ(host.boundary_kind, device[member].boundary_kind);
        EXPECT_FALSE(host.was_acquired);
    }
}

// The write-back transaction under KV parallelism: the fixture's Full group
// dealt over two shards, a block's bucket its slot % 2.
class ShardedWriteBackTransactionTest : public WriteBackTransactionTest {
protected:
    ShardedWriteBackTransactionTest() : WriteBackTransactionTest(/*full_shard_count=*/2, /*num_snapshot_pages=*/0) {}
};

// MembersAlreadyOnTheHostBorrowedAndCopiedPublishTogether with the members in
// other buckets than their Device copies. X's pages alternate buckets 0 and 1,
// and X's write-back carries them into Host blocks of the same buckets. The
// Device's bucket-0 slots are held before Y is admitted, so every Full block of
// Y lands in bucket 1, and the Host holds Y's page at 20 in bucket 0. Y's
// segment borrows X's pages and holds that entry, whatever their buckets, and
// copies only its tail and checkpoints, each row within one bucket; it
// publishes at X's ACK.
TEST_F(ShardedWriteBackTransactionTest, MembersInAnotherBucketThanTheirDeviceCopyAreNotCopied) {
    std::vector<BlockTable> tables(4);
    const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18));
    const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    // The bucket of each of X's pages on the Host.
    std::vector<std::int32_t> x_host_buckets;
    for (std::size_t page = 0; page < 4; ++page) {
        const auto row = std::ranges::find_if(x_op->transfers, [&](const CacheTransfer& transfer) {
            return transfer.group_id == 0 && transfer.content_hash == x[page].content_hash;
        });
        ASSERT_NE(row, x_op->transfers.end());
        x_host_buckets.push_back((row->destination_page - 1) % 2);
    }
    ASSERT_EQ(x_host_buckets, (std::vector<std::int32_t>{0, 1, 0, 1}));
    // Six bucket-0 slots held in three parents leave six bucket-1 slots, as
    // many as Y's Full blocks.
    const std::size_t held_before = held_.size();
    for (CacheBlockRef& block : pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 6>{})) {
        held_.push_back(std::move(block));
    }
    std::vector<std::int32_t> y_tokens = MakeTokens(16);
    for (std::int32_t i = 0; i < 6; ++i) {
        y_tokens.push_back(700 + i);
    }
    const TailPublication y_tail = *PublishFinal(tables, y_tokens);
    coordinator_.Free(tables);
    held_.resize(held_before);
    for (std::size_t page = 0; page < 5; ++page) {
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator_, 0, Key(hashes_[page], 0));
        ASSERT_TRUE(location.has_value());
        ASSERT_EQ(coordinator_.Allocator(0).BucketOf(*location), 1) << "page " << page;
    }
    CacheBlockRef page = host_pool_.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
    coordinator_.CacheHostBlock(page, Key(hashes_[4], 0), std::nullopt);
    page.reset();
    const std::vector<CacheKey> y_own = KeysOf({}, y_tail);
    const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
    ASSERT_TRUE(y_op.has_value());
    ASSERT_EQ(y_op->transfers.size(), 4u) << "the tail's KV block and its three checkpoints";
    for (const CacheTransfer& transfer : y_op->transfers) {
        EXPECT_EQ(transfer.content_hash, y_tail.hash);
        const std::int32_t shards = coordinator_.Allocator(static_cast<std::int32_t>(transfer.group_id)).ShardCount();
        EXPECT_EQ((transfer.source_page - 1) % shards, (transfer.destination_page - 1) % shards);
    }
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.host_cached, 1u);
    EXPECT_EQ(stats.in_flight, 4u);
    EXPECT_EQ(stats.ops, 2u);

    EXPECT_TRUE(transfers_.CompleteWriteBack(y_op->op_id).empty());
    EXPECT_EQ(HostEntries(y_own), 0);
    EXPECT_EQ(transfers_.CompleteWriteBack(x_op->op_id).size(), 8u + 9u);
    EXPECT_EQ(HostEntries(x), 8);
    EXPECT_EQ(HostEntries(y_own), 4);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8 + 1 + 4);
    EXPECT_EQ(stats.cancelled, 0u);
}

// ---------------------------------------------------------------------------
// A restore under the write-back policy evicts cache for the tables it
// rebuilds through the admission planner, and records what it evicts as an
// admission does -- each victim set as one segment, every set it takes -- so
// the plan's drain (StartEvictionStores, called here where the scheduler calls
// it, right after the build) writes it back and the ACK publishes it on the
// Host. One Full group, a block per page of two tokens, and a snapshot pool
// for the retracted request's image.
// ---------------------------------------------------------------------------

// With the demand of an admission -- six blocks on a Device whose six parents
// hold two other prompts' pages, X's four and Y's two -- a restore records the
// same two segments, one per prompt with every page of it, and its drain
// copies the same rows into the same Host blocks; at the ACK every page is on
// the Host. Each path runs twice from the same start: once to read its
// records, once to drain them.
TEST(RestoreWriteBackTest, ARestoreWritesBackWhatItEvicts) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    const std::vector<std::string> x = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
    const std::vector<std::string> y = ContentHashes({{9, 10}, {11, 12}});
    // The admission's, then the restore's: the keys each recorded, segment by
    // segment, and the rows each drain copied.
    std::vector<std::vector<std::vector<CacheKey>>> recorded;
    std::vector<std::vector<CacheTransfer>> copied;
    for (const bool restore : {false, true}) {
        for (const bool drain : {false, true}) {
            SCOPED_TRACE(testing::Message() << "restore " << restore << ", drain " << drain);
            BlockPool pool(6, {1});
            BlockPool host_pool(8, {1});
            BlockPool snapshot_pool(8, {1});
            CacheCoordinator coordinator =
                MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                                /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
            TierTransferManager transfers(coordinator);
            // A retracted request's five blocks, imaged into the snapshot pool.
            std::vector<BlockTable> tables(1);
            ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/10));
            auto image = coordinator.TakeImage(tables, /*num_tokens=*/10, NoHostSlots(1));
            ASSERT_TRUE(image);
            image->store_pairs.clear();
            coordinator.Free(tables);
            // X's four pages, then Y's two, fill the Device.
            ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
            CacheCompletedBlocksForTest(coordinator, tables, x,
                                        CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                        /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8,
                                        CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                        /*publish_state=*/false);
            coordinator.Free(tables);
            ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
            CacheCompletedBlocksForTest(coordinator, tables, y,
                                        CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                        /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4,
                                        CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                        /*publish_state=*/false);
            coordinator.Free(tables);
            ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

            // The image's five blocks and the reserve's one, or six fresh ones.
            if (restore) {
                std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
                ASSERT_TRUE(RestoreDemands(coordinator, image->image, tables, demands, /*access_epoch=*/1));
            } else {
                ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));
            }
            EXPECT_EQ(coordinator.GroupPrefixIndex(0).NumEntries(pool), 0);
            if (!drain) {
                std::vector<std::vector<CacheKey>>& segment_keys = recorded.emplace_back();
                for (CacheCoordinator::EvictedSegment& segment : coordinator.TakeEvictedSegments()) {
                    std::vector<CacheKey>& member_keys = segment_keys.emplace_back();
                    for (const CacheCoordinator::EvictedBlock& member : segment.members) {
                        member_keys.push_back(member.key);
                    }
                    coordinator.SettleEvictedSegment(std::move(segment), /*placed_on_host=*/false);
                }
                coordinator.Free(tables);
                continue;
            }
            const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
            ASSERT_TRUE(op.has_value());
            copied.push_back(op->transfers);
            transfers.CompleteWriteBack(op->op_id);
            for (const std::vector<std::string>& prompt : {x, y}) {
                for (const std::string& hash : prompt) {
                    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
                }
            }
            EXPECT_FALSE(transfers.HasAnyInFlight());
            coordinator.Free(tables);
        }
    }
    ASSERT_EQ(recorded.size(), 2u);
    // Deepest page first, X (the older) before Y.
    EXPECT_EQ(recorded[0], (std::vector<std::vector<CacheKey>>{
                               {Key(x[3], 0), Key(x[2], 0), Key(x[1], 0), Key(x[0], 0)},
                               {Key(y[1], 0), Key(y[0], 0)},
                           }));
    EXPECT_EQ(recorded[1], recorded[0]);
    ASSERT_EQ(copied.size(), 2u);
    EXPECT_EQ(copied[0].size(), 6u);
    EXPECT_EQ(copied[1], copied[0]);
}

// The Host holds pages 2 and 3, written back earlier, and the Device the
// pages 0 and 1 they continue. A restore that evicts pages 0 and 1 writes
// them back, so after its ACK the Host holds the whole chain and the
// collection of dead entries takes none of it. Dropped, they would leave
// pages 2 and 3 under a gap on both tiers, for the collection to take too.
TEST(RestoreWriteBackTest, ARestoreKeepsTheHostSuffixItsVictimsContinue) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    BlockPool pool(4, {1});
    BlockPool host_pool(8, {1});
    BlockPool snapshot_pool(8, {1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    TierTransferManager transfers(coordinator);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/6, NoHostSlots(1));
    ASSERT_TRUE(image);
    image->store_pairs.clear();
    coordinator.Free(tables);
    const std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheCompletedBlocksForTest(coordinator, tables, hashes, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    coordinator.Free(tables);
    for (const std::string& hash : {hashes[2], hashes[3]}) {
        CacheBlockRef block = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
        const std::optional<CacheBlockLocation> location =
            CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(hash, 0));
        ASSERT_TRUE(location.has_value());
        ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *location));
    }
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 2);

    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    ASSERT_TRUE(RestoreDemands(coordinator, image->image, tables, demands, /*access_epoch=*/1));
    EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, 2u);
    const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
    ASSERT_TRUE(op.has_value()) << "the restore recorded pages 0 and 1";
    EXPECT_EQ(op->transfers.size(), 2u);
    transfers.CompleteWriteBack(op->op_id);
    coordinator.CollectDeadEntries();
    for (const std::string& hash : hashes) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
    }
    coordinator.Free(tables);
}

// The oldest entry a restore short of one block finds is a tail a turn's
// finish marked superseded (MarkSupersededStateInput). It leaves as the
// reclaim of the request that superseded it, as under an admission and under
// either write policy: recorded nowhere, written back never, and not counted
// as the restore's, so every block counted as evicted for the cache still
// lands in one outcome of the drain.
TEST(RestoreWriteBackTest, ARestoreDropsASupersededInputRatherThanWritingItBack) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    for (const bool write_back : {true, false}) {
        SCOPED_TRACE(write_back);
        BlockPool pool(4, {1});
        BlockPool host_pool(8, {1});
        BlockPool snapshot_pool(8, {1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/!write_back, /*write_back_evictions=*/write_back);
        TierTransferManager transfers(coordinator);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2));
        auto image = coordinator.TakeImage(tables, /*num_tokens=*/2, NoHostSlots(1));
        ASSERT_TRUE(image);
        image->store_pairs.clear();
        coordinator.Free(tables);
        // Two pages and a one-token tail, the tail marked superseded once published.
        const std::vector<std::int32_t> tokens = MakeTokens(5);
        const std::vector<std::string> pages = ContentHashes({{tokens[0], tokens[1]}, {tokens[2], tokens[3]}});
        const TailPublication tail{.tail_tokens = 1,
                                   .hash = HashPrefixPage(token_span(tokens).subspan(4), pages.back())};
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));
        const std::uint64_t epoch = CacheCoordinatorTestAccess::NextAccessEpoch(coordinator);
        coordinator.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = pages,
                                                              .first_new_prefix_page = 0,
                                                              .tail = tail,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = 5},
            epoch);
        coordinator.Free(tables);
        coordinator.MarkSupersededStateInput(tail.hash, epoch);
        ASSERT_EQ(pool.NumEmptyLcmBlocks(), 1);

        std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
        ASSERT_TRUE(RestoreDemands(coordinator, image->image, tables, demands, /*access_epoch=*/1));
        EXPECT_FALSE(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(tail.hash, 0)).has_value());
        const CacheCoordinator::DeviceEvictionStats& evictions = coordinator.DeviceEvictions();
        EXPECT_EQ(evictions.request_reclaim.blocks, 1u);
        EXPECT_EQ(evictions.restore.blocks, 0u);
        EXPECT_FALSE(transfers.StartEvictionStores().has_value()) << "nothing was recorded";
        EXPECT_EQ(coordinator.NumHostCachedBlocks(), 0);
        const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kEviction);
        std::uint64_t outcomes =
            stats.host_cached + stats.in_flight + stats.no_placement + coordinator.DeadEntriesNotWrittenBack();
        for (const std::uint64_t blocks : stats.blocks) {
            outcomes += blocks;
        }
        EXPECT_EQ(outcomes, evictions.admission.blocks + evictions.stranded_suffix.blocks + evictions.restore.blocks);
        coordinator.Free(tables);
    }
}

// A restore whose only victim is a superseded tail adds a record that stays
// empty, and erases it before it returns. An admission later in the same
// plan then records the one segment the drain takes, dated by its newest
// member; and a cache clear before any drain finds no record left behind.
TEST(RestoreWriteBackTest, AnEmptyRestoreRecordDoesNotShiftTheAdmissionsEpochs) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    for (const bool admit_after_restore : {true, false}) {
        SCOPED_TRACE(admit_after_restore);
        BlockPool pool(4, {1});
        BlockPool host_pool(8, {1});
        BlockPool snapshot_pool(8, {1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2));
        auto image = coordinator.TakeImage(tables, /*num_tokens=*/2, NoHostSlots(1));
        ASSERT_TRUE(image);
        image->store_pairs.clear();
        coordinator.Free(tables);
        const std::vector<std::int32_t> tokens = MakeTokens(5);
        const std::vector<std::string> pages = ContentHashes({{tokens[0], tokens[1]}, {tokens[2], tokens[3]}});
        const TailPublication tail{.tail_tokens = 1,
                                   .hash = HashPrefixPage(token_span(tokens).subspan(4), pages.back())};
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));
        const std::uint64_t epoch = CacheCoordinatorTestAccess::NextAccessEpoch(coordinator);
        coordinator.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = pages,
                                                              .first_new_prefix_page = 0,
                                                              .tail = tail,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = 5},
            epoch);
        coordinator.Free(tables);
        coordinator.MarkSupersededStateInput(tail.hash, epoch);

        std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
        ASSERT_TRUE(RestoreDemands(coordinator, image->image, tables, demands, /*access_epoch=*/1));
        ASSERT_EQ(coordinator.DeviceEvictions().request_reclaim.blocks, 1u);
        if (!admit_after_restore) {
            EXPECT_TRUE(coordinator.ClearDeviceCache());
            coordinator.Free(tables);
            continue;
        }
        // One block on a full Device: page 1 leaves, a segment of its own.
        std::vector<BlockTable> admitted(1);
        ASSERT_TRUE(AdmitForTest(coordinator, admitted, /*num_tokens=*/2));
        std::vector<CacheCoordinator::EvictedSegment> segments = coordinator.TakeEvictedSegments();
        ASSERT_EQ(segments.size(), 1u);
        ASSERT_EQ(segments[0].members.size(), 1u);
        EXPECT_EQ(segments[0].members[0].key, Key(pages[1], 0));
        EXPECT_EQ(segments[0].newest_access_epoch, epoch);
        coordinator.SettleEvictedSegment(std::move(segments[0]), /*placed_on_host=*/false);
        coordinator.Free(admitted);
        coordinator.Free(tables);
    }
}

// A restore evicts X's four pages, and its write-back stays unacknowledged
// across three more plans, each draining and collecting dead entries as the
// scheduler does: the first admits a request that evicts Y's two pages and
// drains them in an op of its own, the other two drain nothing. Until the
// restore's ACK, the pending stores alone hold X's tree nodes -- neither tier
// holds X, the collection leaves it -- and nothing of X is published, while
// Y's segment publishes at its own ACK. Once both ACKs are in, no Host entry
// is pinned, no store is out, and the cache clears.
TEST(RestoreWriteBackTest, ARestoreRecordWaitsForItsAckAcrossPlans) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    BlockPool pool(6, {1});
    BlockPool host_pool(8, {1});
    BlockPool snapshot_pool(8, {1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    TierTransferManager transfers(coordinator);
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/6, NoHostSlots(1));
    ASSERT_TRUE(image);
    image->store_pairs.clear();
    coordinator.Free(tables);
    // X's four pages, then Y's two, fill the Device.
    const std::vector<std::string> x = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    CacheCompletedBlocksForTest(coordinator, tables, x, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    coordinator.Free(tables);
    const std::vector<std::string> y = ContentHashes({{9, 10}, {11, 12}});
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    CacheCompletedBlocksForTest(coordinator, tables, y, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    coordinator.Free(tables);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 0);

    // The image's three blocks and the reserve's one: X, the older, leaves.
    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    ASSERT_TRUE(RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1));
    const std::optional<WriteBackOperation> restore_op = transfers.StartEvictionStores();
    ASSERT_TRUE(restore_op.has_value());
    ASSERT_EQ(restore_op->transfers.size(), 4u);
    coordinator.CollectDeadEntries();

    // Two blocks on the full Device: Y leaves, in an op of its own.
    std::vector<BlockTable> admitted(1);
    ASSERT_TRUE(AdmitForTest(coordinator, admitted, /*num_tokens=*/4));
    const std::optional<WriteBackOperation> admission_op = transfers.StartEvictionStores();
    ASSERT_TRUE(admission_op.has_value());
    EXPECT_EQ(admission_op->transfers.size(), 2u);
    coordinator.CollectDeadEntries();
    for (std::int32_t plan = 0; plan < 2; ++plan) {
        EXPECT_FALSE(transfers.StartEvictionStores().has_value());
        coordinator.CollectDeadEntries();
    }
    for (const std::string& hash : x) {
        const BoundaryTree::Node* node = tree.Find(hash);
        ASSERT_NE(node, nullptr) << hash;
        EXPECT_EQ(node->pending_host_stores, 1) << hash;
        EXPECT_FALSE(node->on_device) << hash;
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
    }

    EXPECT_EQ(transfers.CompleteWriteBack(admission_op->op_id).size(), 2u) << "Y's pages";
    EXPECT_EQ(transfers.CompleteWriteBack(restore_op->op_id).size(), 4u) << "X's pages";
    for (const std::string& hash : x) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
        EXPECT_EQ(tree.Find(hash)->pending_host_stores, 0) << hash;
    }
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers.HasAnyInFlight());
    coordinator.Free(restored);
    coordinator.Free(admitted);
    EXPECT_TRUE(coordinator.ClearCache());
}

// ARestoreWritesBackWhatItEvicts under KV parallelism, a parent holding one
// block of each bucket. X's pages fill the first parent, page j in bucket j,
// and the Host holds X's deepest page, written back earlier, in bucket 0. The
// image's two published pages ride Host entries in bucket 1, page 1's own
// bucket and not page 0's; page 0 is still cached on the Device, page 1 is
// not. With one bucket-1 hole left on the Device, the restore is short of
// room only in the bucket of the image's tail, yet X's pages leave deepest
// first, down to the one in that bucket, as one segment over every bucket
// they span: every member is written back, each copy within one bucket, X's
// deepest page as its Host entry. The restore claims page 0, copying nothing
// for it although its Host entry lies in another bucket, loads page 1 within
// bucket 1, and copies the tail onto the page X left in the tail's bucket,
// behind that page's write-back: both copies stay in that bucket.
class ShardedRestoreWriteBackTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, ShardedRestoreWriteBackTest, ::testing::Values(2, 8));

TEST_P(ShardedRestoreWriteBackTest, ARestoreWritesBackWhatItEvicts) {
    const std::int32_t shards = GetParam();
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull,
                       .cache_blocks_per_lcm_block = shards,
                       .block_granularity = 2,
                       .shard_count = shards},
    };
    BlockPool pool(3, {shards});
    BlockPool host_pool(4, {shards});
    BlockPool snapshot_pool(2, {shards});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    TierTransferManager transfers(coordinator);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    std::vector<std::vector<std::int32_t>> x_pages;
    for (std::int32_t page = 0; page < shards; ++page) {
        x_pages.push_back({2 * page + 1, 2 * page + 2});
    }
    const std::vector<std::string> x = ContentHashes(x_pages);
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2 * shards));
    CacheCompletedBlocksForTest(coordinator, tables, x, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * shards,
                                CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                /*publish_state=*/false);
    coordinator.Free(tables);
    CacheBlockRef written_back = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
    coordinator.CacheHostBlock(written_back, Key(x.back(), 0), std::nullopt);
    written_back.reset();

    const std::vector<std::string> image_pages = ContentHashes({{101, 102}, {103, 104}});
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/5));
    CacheCompletedBlocksForTest(coordinator, tables, image_pages,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/5, CacheBoundaryKind::kChunk,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    std::vector<std::int32_t> image_buckets;
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        image_buckets.push_back(allocator.BucketOf(block->Location()));
    }
    ASSERT_EQ(image_buckets, (std::vector<std::int32_t>{0, 1, 2 % shards}));
    const std::int32_t tail_bucket = image_buckets[2];
    const CacheBlockLocation page0 = tables[0].Blocks()[0]->Location();
    std::vector<std::vector<ImageSlot>> host_slots = coordinator.PublishedDataSlots(tables, /*num_tokens=*/5);
    ASSERT_EQ(host_slots[0].size(), 2u);
    for (ImageSlot& slot : host_slots[0]) {
        CacheBlockRef host = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1}).front();
        coordinator.CacheHostBlock(host, slot.key, std::nullopt);
        slot.block = std::move(host);
    }
    auto image = coordinator.TakeImage(tables, /*num_tokens=*/5, host_slots);
    ASSERT_TRUE(image);
    image->store_pairs.clear();
    host_slots.clear();
    coordinator.Free(tables);
    const std::optional<CacheBlockLocation> page1 =
        CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, Key(image_pages[1], 0));
    ASSERT_TRUE(page1.has_value());
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictCachedBlock(coordinator, 0, *page1));
    // Every Device hole and empty parent held but one bucket-1 hole.
    std::vector<CacheBlockRef> held = pool.AcquireBlocks(0, pool.NumFreeSlots(0) + shards * pool.NumEmptyLcmBlocks());
    held.erase(std::ranges::find_if(
        held, [&](const CacheBlockRef& block) { return allocator.BucketOf(block->Location()) == 1; }));

    std::vector<BlockTable> restored(1);
    std::vector<GroupDemand> demands{GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1}};
    const std::optional<CacheCoordinator::AdmissionResult> restore =
        RestoreDemands(coordinator, image->image, restored, demands, /*access_epoch=*/1);
    ASSERT_TRUE(restore.has_value());
    EXPECT_EQ(coordinator.DeviceEvictions().restore.blocks, static_cast<std::uint64_t>(shards - tail_bucket));
    EXPECT_EQ(restored[0].Blocks()[0]->Location(), page0) << "page 0 is claimed";
    EXPECT_NE(allocator.BucketOf(image->image.tables[0].slots[0].block->Location()), allocator.BucketOf(page0));
    ASSERT_EQ(restore->load_pairs.size(), 1u) << "page 1 alone is loaded";
    EXPECT_EQ(restore->load_pairs[0].key, Key(image_pages[1], 0));
    EXPECT_EQ(allocator.BucketOf(restore->load_pairs[0].source->Location()),
              allocator.BucketOf(restore->load_pairs[0].destination->Location()));
    ASSERT_EQ(restore->snapshot_pairs.size(), 1u);
    const BlockTransfer& tail_copy = restore->snapshot_pairs[0];
    EXPECT_EQ(allocator.BucketOf(tail_copy.source->Location()), allocator.BucketOf(tail_copy.destination->Location()));

    const std::optional<WriteBackOperation> op = transfers.StartEvictionStores();
    ASSERT_TRUE(op.has_value());
    EXPECT_EQ(op->transfers.size(), static_cast<std::size_t>(shards - tail_bucket - 1)) << "all but the deepest";
    EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kEviction).host_cached, 1u);
    for (const CacheTransfer& transfer : op->transfers) {
        // A page's bucket is (page id - 1) % shard_count (BucketOf).
        EXPECT_EQ((transfer.source_page - 1) % shards, (transfer.destination_page - 1) % shards);
    }
    const auto tail_page_write_back =
        std::ranges::find(op->transfers, allocator.ResolveDeviceCacheBlockId(tail_copy.destination->Location()),
                          &CacheTransfer::source_page);
    ASSERT_NE(tail_page_write_back, op->transfers.end()) << "the tail lands on a page X's write-back reads";
    EXPECT_EQ((tail_page_write_back->destination_page - 1) % shards, allocator.BucketOf(tail_copy.source->Location()));
    transfers.CompleteWriteBack(op->op_id);
    for (std::int32_t page = 0; page < shards; ++page) {
        const CacheKey key = Key(x[static_cast<std::size_t>(page)], 0);
        EXPECT_EQ(coordinator.ContainsHostCachedBlock(key), page >= tail_bucket) << "page " << page;
        EXPECT_EQ(CacheCoordinatorTestAccess::DeviceLocationOf(coordinator, 0, key).has_value(), page < tail_bucket)
            << "page " << page;
    }
    EXPECT_FALSE(transfers.HasAnyInFlight());
    coordinator.Free(restored);
}

// ---------------------------------------------------------------------------
// A retraction image's Host L2 leg is planned before anything is acquired or
// evicted for it (TierTransferManager::SplitRetractionImage): the published
// slots the Host serves already, the rest as rows by boundary, how many
// boundaries the Host takes (CacheCoordinator::NumHostBoundariesThatFit) and
// whether the snapshot pool holds what is left. The victim probe asks the
// split; StartRetractionStores carries out the same plan, so an image the
// split holds is always taken and one it refuses leaves the Host as it was.
// One Full group, a block per page of two tokens, unless a test says else.
// ---------------------------------------------------------------------------

// Four Host blocks hold other prompts' pages. The image has six published
// pages and a snapshot pool of one block: the Host could take four of them
// by evicting every entry, which leaves two for the pool. The split refuses
// the image on the Host's account, and every entry stays.
TEST(RetractionImageSplitTest, ARefusedImageLeavesTheHostAsItWas) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    BlockPool pool(8, {1});
    BlockPool host_pool(4, {1});
    BlockPool snapshot_pool(1, {1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}, {97, 98}});
    for (const std::string& hash : cached) {
        CacheBlockRef block = host_pool.AcquireBlock(0);
        coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
    }
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));
    const std::vector<std::string> image_pages = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}});
    CacheCompletedBlocksForTest(coordinator, tables, image_pages,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/12, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);

    const TierTransferManager::RetractionImageSplit split =
        transfers.SplitRetractionImage(tables, /*num_tokens=*/12, coordinator.PublishedDataSlots(tables, 12));
    EXPECT_EQ(split.shortfall, ImageShortfall::kHostL2);
    EXPECT_EQ(split.boundaries.size(), 6u);
    EXPECT_EQ(split.num_host_boundaries, 4u) << "the Host takes four pages by evicting all four entries";
    for (const std::string& hash : cached) {
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
    }
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    coordinator.Free(tables);
}

// A Full group of four blocks per parent and a State group. The Host's two
// parents both hold Full entries: X four unpinned ones, the oldest, and Y one
// pinned and three unpinned. The image needs a Full block and a State block
// at its one page. Freeing every unpinned entry would make room for both --
// the Full block in Y, X emptied for the State block -- but the allocator
// takes the oldest Full entry, in X, for the Full block, which keeps X, and
// Y never empties: the State block has no parent. The Host takes no
// boundary, and keeps every entry: with a snapshot pool for the State
// group's working slot alone, the split refuses the image on the Host's
// account; with one that holds both rows as well, the retraction takes the
// image and evicts nothing for the boundary the Host does not take.
TEST(RetractionImageSplitTest, AnImageFitsOnlyWhereTheAllocatorWouldPlaceIt) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 4, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const std::int32_t snapshot_parents : {1, 3}) {
        SCOPED_TRACE(snapshot_parents);
        BlockPool pool(8, {4, 1});
        BlockPool host_pool(2, {4, 1});
        BlockPool snapshot_pool(snapshot_parents, {4, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> cached =
            ContentHashes({{91, 92}, {93, 94}, {95, 96}, {97, 98}, {81, 82}, {83, 84}, {85, 86}, {87, 88}});
        CacheBlockRef pinned;
        for (const std::string& hash : cached) {
            CacheBlockRef block = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(block, Key(hash, 0), std::nullopt);
            if (hash == cached[4]) {
                pinned = std::move(block);
            }
        }
        ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
        std::vector<BlockTable> tables(2);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2));
        const std::vector<std::string> image_page = ContentHashes({{1, 2}});
        CacheCompletedBlocksForTest(coordinator, tables, image_page,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/2, coordinator.PublishedDataSlots(tables, 2));
        ASSERT_EQ(split.boundaries.size(), 1u);
        EXPECT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 1}))
            << "one boundary: the Full row, then the State row";
        EXPECT_EQ(split.num_host_boundaries, 0u);
        if (snapshot_parents == 1) {
            EXPECT_EQ(split.shortfall, ImageShortfall::kHostL2);
        } else {
            ASSERT_FALSE(split.shortfall.has_value());
            std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
                "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
                /*num_tokens=*/2, std::move(split));
            ASSERT_TRUE(stores.has_value());
            EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 2u);
        }
        for (const std::string& hash : cached) {
            EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(hash, 0))) << hash;
        }
        EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
        coordinator.Free(tables);
    }
}

// A Full group and two State groups, one block per parent each; the Host's
// four parents hold, oldest first, the State checkpoints of one hash in the
// first State group, a Full entry F, a Full entry G, and the second State
// group's checkpoint at that hash. The image's one page needs a Full block
// and both State blocks. The Full block's victim is the oldest parent, the
// first checkpoint, which takes its sibling along: two parents free. The
// first State block fits beside it, and the second takes F. G stays: the
// plan counts the sibling the cascade frees rather than taking another
// victim for its room.
TEST(RetractionImageSplitTest, ACheckpointVictimFreesItsStateSiblingInThePlan) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    BlockPool pool(8, {1, 1, 1});
    BlockPool host_pool(4, {1, 1, 1});
    BlockPool snapshot_pool(2, {1, 1, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    SnapshotSlotAllocator blob_slots(1);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}});
    const std::vector<CacheKey> host_keys{Key(cached[0], 1), Key(cached[1], 0), Key(cached[2], 0), Key(cached[0], 2)};
    for (const CacheKey& key : host_keys) {
        CacheBlockRef block = host_pool.AcquireBlock(key.group_id);
        coordinator.CacheHostBlock(block, key, std::nullopt);
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    std::vector<BlockTable> tables(3);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2));
    CacheCompletedBlocksForTest(coordinator, tables, ContentHashes({{1, 2}}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/true);

    TierTransferManager::RetractionImageSplit split =
        transfers.SplitRetractionImage(tables, /*num_tokens=*/2, coordinator.PublishedDataSlots(tables, 2));
    ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 1, 2}));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
        /*num_tokens=*/2, std::move(split));
    ASSERT_TRUE(stores.has_value());
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[0]));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[1]));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(host_keys[2])) << "G is no victim";
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[3]));
    coordinator.Free(tables);
}

// A Full group of two blocks per parent and a State group. The Host's
// parents hold, oldest first, a Full entry c, a State checkpoint w, a second
// Full entry beside c and a State checkpoint q. The image's two pages need a
// Full block each and a State block at the second. The first Full block
// takes c's place, which keeps c's parent from going whole, as in
// AcquireHostBlocks; the second Full block takes w's parent, whose other
// slot could hold the first Full block as well, and still the State block
// takes q, not c's parent: the second Full entry stays.
TEST(RetractionImageSplitTest, ARowsChildVictimKeepsItsParentFromGoingWhole) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    BlockPool pool(8, {2, 1});
    BlockPool host_pool(3, {2, 1});
    BlockPool snapshot_pool(1, {2, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    SnapshotSlotAllocator blob_slots(1);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}, {97, 98}});
    const std::vector<CacheKey> host_keys{Key(cached[0], 0), Key(cached[1], 1), Key(cached[2], 0), Key(cached[3], 1)};
    {
        // c and the second Full entry share a parent; w and q have one each.
        std::vector<CacheBlockRef> blocks{host_pool.AcquireBlock(0), host_pool.AcquireBlock(1),
                                          host_pool.AcquireBlock(0), host_pool.AcquireBlock(1)};
        for (std::size_t i = 0; i < host_keys.size(); ++i) {
            coordinator.CacheHostBlock(blocks[i], host_keys[i], std::nullopt);
        }
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    CacheCompletedBlocksForTest(coordinator, tables, ContentHashes({{1, 2}, {3, 4}}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/true);

    TierTransferManager::RetractionImageSplit split =
        transfers.SplitRetractionImage(tables, /*num_tokens=*/4, coordinator.PublishedDataSlots(tables, 4));
    ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 0, 1}));
    ASSERT_EQ(split.num_host_boundaries, 2u);
    ASSERT_FALSE(split.shortfall.has_value());
    std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
        /*num_tokens=*/4, std::move(split));
    ASSERT_TRUE(stores.has_value());
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[0]));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[1]));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(host_keys[2])) << "the second Full entry is no victim";
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[3]));
    coordinator.Free(tables);
}

// Two Full groups A and B, two blocks per parent. The Host's parent X holds
// a pinned A entry and an unpinned one, Y a B entry, the oldest, and a free
// slot. One boundary needs an A block and a B block. The B block fits Y's
// free slot, so Y cannot go whole, though it is the oldest parent: the A
// block takes X's unpinned entry, as AcquireHostBlocks places the B block
// before it picks a victim. Taking Y would leave the B block without room.
TEST(RetractionImageSplitTest, ARowInFreeRoomKeepsItsParentFromGoingWhole) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    BlockPool pool(8, {2, 2});
    BlockPool host_pool(2, {2, 2});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}});
    CacheBlockRef pinned = host_pool.AcquireBlock(0);
    {
        CacheBlockRef x_entry = host_pool.AcquireBlock(0);
        CacheBlockRef y_entry = host_pool.AcquireBlock(1);
        coordinator.CacheHostBlock(y_entry, Key(cached[0], 1), std::nullopt);
        coordinator.CacheHostBlock(x_entry, Key(cached[1], 0), std::nullopt);
    }
    coordinator.CacheHostBlock(pinned, Key(cached[2], 0), std::nullopt);
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    const std::vector<std::uint32_t> group_ids{0, 1};
    const std::vector<std::int32_t> buckets{0, 0};
    const std::vector<CacheCoordinator::HostSegmentRows> boundaries{{.first_row = 0, .past_end_row = 2}};
    EXPECT_EQ(coordinator.NumHostBoundariesThatFit(group_ids, buckets, boundaries), 1u);
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator.AcquireHostBoundaries(group_ids, buckets, boundaries);
    EXPECT_EQ(batch.stats.allocated, 2u);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(cached[0], 1))) << "Y's entry is no victim";
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(cached[1], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(cached[2], 0)));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 1u);
}

// A Full group and a State group, one block per parent each. The Host holds
// a State checkpoint, the oldest, and a Full entry. A Full block's victim is
// the older of the two: the State parent, whole, rather than the Full
// group's own younger entry.
TEST(RetractionImageSplitTest, AnOlderParentOfAnotherGroupBeatsAYoungerEntry) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    BlockPool pool(8, {1, 1});
    BlockPool host_pool(2, {1, 1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}});
    const std::vector<CacheKey> host_keys{Key(cached[0], 1), Key(cached[1], 0)};
    for (const CacheKey& key : host_keys) {
        CacheBlockRef block = host_pool.AcquireBlock(key.group_id);
        coordinator.CacheHostBlock(block, key, std::nullopt);
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    const std::vector<std::uint32_t> group_ids{0};
    const std::vector<std::int32_t> buckets{0};
    const std::vector<CacheCoordinator::HostSegmentRows> boundaries{{.first_row = 0, .past_end_row = 1}};
    EXPECT_EQ(coordinator.NumHostBoundariesThatFit(group_ids, buckets, boundaries), 1u);
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator.AcquireHostBoundaries(group_ids, buckets, boundaries);
    EXPECT_EQ(batch.stats.allocated, 1u);
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(host_keys[0]));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(host_keys[1])) << "the younger Full entry is no victim";
}

// KV parallelism over two shards, two Full blocks per parent; the Host's one
// parent holds an entry in bucket 0, the older, and one in bucket 1. A block
// in bucket 1 takes the bucket-1 entry: the oldest of its own bucket, and one
// entry rather than the whole parent its newest occupant dates. The older
// bucket-0 entry stays.
TEST(RetractionImageSplitTest, ARowTakesTheOldestEntryOfItsBucketNotItsWholeParent) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    BlockPool pool(4, {2});
    BlockPool host_pool(1, {2});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}});
    for (std::int32_t bucket = 0; bucket < 2; ++bucket) {
        CacheBlockRef block = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{bucket}).front();
        coordinator.CacheHostBlock(block, Key(cached[static_cast<std::size_t>(bucket)], 0), std::nullopt);
    }
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);

    const std::vector<std::uint32_t> group_ids{0};
    const std::vector<std::int32_t> buckets{1};
    const std::vector<CacheCoordinator::HostSegmentRows> boundaries{{.first_row = 0, .past_end_row = 1}};
    EXPECT_EQ(coordinator.NumHostBoundariesThatFit(group_ids, buckets, boundaries), 1u);
    const CacheCoordinator::HostAllocationBatch batch =
        coordinator.AcquireHostBoundaries(group_ids, buckets, boundaries);
    ASSERT_EQ(batch.stats.allocated, 1u);
    EXPECT_EQ(coordinator.Allocator(0).BucketOf(batch.blocks.front()->Location()), 1);
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(cached[0], 0))) << "the bucket-0 entry is no victim";
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(cached[1], 0)));
}

// KV parallelism over two shards, two Full blocks per parent: a page's block
// lies in bucket page % 2, and its Host copy must too. Of the Host's two
// parents, one holds a pinned entry in bucket 0 and the other an unpinned
// one, each with bucket 1 free: the first page takes the unpinned entry's
// place (its parent empties, a parent for both buckets), the second page the
// other parent's bucket-1 block, and the third page finds no Host block in
// bucket 0 -- the Host takes two boundaries, though it has a free block in
// bucket 1 for the fourth. Without a snapshot pool block for the last two
// pages, the split refuses the image and nothing is evicted; with one, the
// retraction places exactly the first two pages, each beside its Device
// block's bucket, and the last two ride the pool as the split predicted.
TEST(RetractionImageSplitTest, OneFullHostBucketDecidesTheSplitBeforeAnyEviction) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    for (const std::int32_t snapshot_parents : {0, 1}) {
        SCOPED_TRACE(snapshot_parents);
        BlockPool pool(4, {2});
        BlockPool host_pool(2, {2});
        BlockPool snapshot_pool(snapshot_parents, {2});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}});
        CacheBlockRef pinned = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
        coordinator.CacheHostBlock(pinned, Key(cached[0], 0), std::nullopt);
        {
            CacheBlockRef unpinned = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{0}).front();
            coordinator.CacheHostBlock(unpinned, Key(cached[1], 0), std::nullopt);
        }
        ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
        std::vector<BlockTable> tables(1);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
        const std::vector<std::string> image_pages = ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}});
        CacheCompletedBlocksForTest(coordinator, tables, image_pages,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/8, coordinator.PublishedDataSlots(tables, 8));
        EXPECT_EQ(split.buckets, (std::vector<std::int32_t>{0, 1, 0, 1}));
        EXPECT_EQ(split.num_host_boundaries, 2u);
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(cached[1], 0))) << "the split evicts nothing";
        if (snapshot_parents == 0) {
            EXPECT_EQ(split.shortfall, ImageShortfall::kHostL2);
            coordinator.Free(tables);
            continue;
        }
        ASSERT_FALSE(split.shortfall.has_value());
        std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
            /*num_tokens=*/8, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        const std::vector<CacheTransfer>& copies = stores->host_store->transfers;
        ASSERT_EQ(copies.size(), 2u);
        for (std::size_t page = 0; page < copies.size(); ++page) {
            EXPECT_EQ(copies[page].content_hash, image_pages[page]);
            EXPECT_EQ((copies[page].source_page - 1) % 2, (copies[page].destination_page - 1) % 2)
                << "page " << page << " keeps its owner";
        }
        const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kRetraction);
        EXPECT_EQ(stats.snapshot_fallback, 2u);
        EXPECT_EQ(stats.image_refused, 0u);
        EXPECT_EQ(stores->snapshot_store.transfers.size(), 2u) << "the last two pages ride the pool";
        EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(cached[1], 0))) << "the first page took its place";
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(cached[0], 0)));
        coordinator.Free(tables);
    }
}

// KV parallelism over two shards, two Full blocks per parent. Another request
// publishes the image's one page first, from a block in bucket 0; the
// victim's own block for that page lies in bucket 1, and its publication
// re-points the slot to the indexed block. The probe, made before that
// publication on PublishedDataSlotsAfter, reads the slot on the indexed block
// as well: the row goes to bucket 0, where the Host's one parent has a free
// block, not to bucket 1, which a pinned entry fills. The probe's split and
// the one made after the publication agree.
TEST(RetractionImageSplitTest, TheProbeCountsARowInTheBucketOfTheBlockItsPublicationDedupesOnto) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    BlockPool pool(4, {2});
    BlockPool host_pool(1, {2});
    BlockPool snapshot_pool(1, {2});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    TierTransferManager transfers(coordinator);
    const GroupAllocator& allocator = coordinator.Allocator(0);
    // Both compute the page. The publisher's block opens a parent at its
    // bucket-0 slot; the victim's takes that parent's bucket-1 hole, which
    // goes before a new parent.
    std::vector<BlockTable> publisher(1);
    ASSERT_TRUE(AdmitForTest(coordinator, publisher, /*num_tokens=*/2));
    std::vector<BlockTable> victim(1);
    ASSERT_TRUE(AdmitForTest(coordinator, victim, /*num_tokens=*/2));
    ASSERT_EQ(allocator.BucketOf(publisher[0].Blocks()[0]->Location()), 0);
    ASSERT_EQ(allocator.BucketOf(victim[0].Blocks()[0]->Location()), 1);
    const std::vector<std::string> page = ContentHashes({{1, 2}});
    CacheCompletedBlocksForTest(coordinator, publisher, page, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    CacheBlockRef pinned = host_pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 1>{1}).front();
    coordinator.CacheHostBlock(pinned, Key(ContentHashes({{91, 92}})[0], 0), std::nullopt);

    const RequestProgress progress{
        .completed_pages =
            CompletedPages{
                .prefix_hashes = page,
                .first_new_prefix_page = 0,
                .boundary_kind = CacheBoundaryKind::kEndpoint,
            },
        .num_computed_tokens = 2,
    };
    const std::vector<std::vector<ImageSlot>> probed =
        coordinator.PublishedDataSlotsAfter(victim, /*num_tokens=*/2, progress);
    ASSERT_EQ(probed[0].size(), 1u);
    EXPECT_EQ(probed[0][0].block, publisher[0].Blocks()[0]) << "the block the publication leaves the slot on";
    const TierTransferManager::RetractionImageSplit probe =
        transfers.SplitRetractionImage(victim, /*num_tokens=*/2, probed);
    EXPECT_EQ(probe.buckets, (std::vector<std::int32_t>{0}));
    EXPECT_EQ(probe.num_host_boundaries, 1u);

    coordinator.CacheCompletedBlocks(victim, progress, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
    ASSERT_EQ(victim[0].Blocks()[0], publisher[0].Blocks()[0]);
    const TierTransferManager::RetractionImageSplit retraction =
        transfers.SplitRetractionImage(victim, /*num_tokens=*/2, coordinator.PublishedDataSlots(victim, 2));
    EXPECT_EQ(retraction.buckets, probe.buckets);
    EXPECT_EQ(retraction.num_host_boundaries, probe.num_host_boundaries);
    EXPECT_EQ(retraction.shortfall, probe.shortfall);
    coordinator.Free(victim);
    coordinator.Free(publisher);
}

// The boundaries go shallowest first whatever order the groups are declared
// in; K3 declares a State group before its Full group. A Full group and a
// State group, a block per page; the image's three pages publish the State
// checkpoint at the last one. The Host's one empty parent takes the first
// page's Full block, and the snapshot pool of four blocks the rest: the later
// pages' Full blocks and both State slots. Numbered in declaration order, the
// State checkpoint would put the last page first, whose two rows the one
// parent cannot take, and the pool could not hold the rest.
TEST(RetractionImageSplitTest, BoundariesGoShallowestFirstWhateverTheGroupOrder) {
    const CacheGroupSpec full{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2};
    const CacheGroupSpec state{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0};
    for (const bool state_first : {false, true}) {
        SCOPED_TRACE(state_first);
        const std::vector<CacheGroupSpec> specs =
            state_first ? std::vector<CacheGroupSpec>{state, full} : std::vector<CacheGroupSpec>{full, state};
        const std::uint32_t full_group = state_first ? 1 : 0;
        BlockPool pool(8, {1, 1});
        BlockPool host_pool(1, {1, 1});
        BlockPool snapshot_pool(4, {1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        std::vector<BlockTable> tables(2);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
        const std::vector<std::string> image_pages = ContentHashes({{1, 2}, {3, 4}, {5, 6}});
        CacheCompletedBlocksForTest(coordinator, tables, image_pages,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/6,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/true);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/6, coordinator.PublishedDataSlots(tables, 6));
        ASSERT_EQ(split.boundaries.size(), 3u);
        for (std::size_t page = 0; page < split.boundaries.size(); ++page) {
            EXPECT_EQ(split.slots[split.boundaries[page].first_row].key.content_hash, image_pages[page])
                << "boundary " << page;
        }
        EXPECT_EQ(split.num_host_boundaries, 1u);
        ASSERT_FALSE(split.shortfall.has_value());
        std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
            /*num_tokens=*/6, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        ASSERT_EQ(stores->host_store->transfers.size(), 1u);
        EXPECT_EQ(stores->host_store->transfers[0].content_hash, image_pages[0]);
        EXPECT_EQ(stores->host_store->transfers[0].group_id, full_group);
        EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 3u);
        coordinator.Free(tables);
    }
}

// Random images on random Hosts, placed block by block and, as under the
// write-back policy, by segments first. Per geometry -- a Full group alone or
// with State groups, its blocks dealt over one, two or eight KV-parallel shards
// (eight Full blocks and one State block per parent is K3's KVP8 shape) --
// each trial fills a Host of two to six parents with other prompts' entries,
// some pinned, and the Host's copies of none, some or all of the image's own
// pages (all of them: no row is left to copy), then splits an image of one to
// eight pages beside a snapshot pool of one to four parents. The split
// mutates nothing, and the plan of its first j boundaries counts min(j, k) of
// them, k the whole plan's count. An image the split holds is taken: the Host
// copies the rows of its first k boundaries, in order, and nothing else, the
// pool takes every later row, no image is refused, and no pinned entry or
// copy the image reuses moves. An image the split refuses is never taken, and
// leaves every Host entry where it was.
TEST(RetractionImageSplitTest, ProbeAndCommitAgreeOnEveryImage) {
    struct Geometry {
        std::int32_t shards;
        std::int32_t packing;
        std::int32_t num_state_groups;
    };
    std::mt19937 rng(20261026);
    std::size_t num_taken = 0;
    std::size_t num_taken_after_evicting = 0;
    std::size_t num_taken_with_pool_rows = 0;
    std::size_t num_refused = 0;
    std::size_t num_without_rows = 0;
    std::size_t num_short_of_the_host = 0;
    for (const bool place_by_segments : {false, true}) {
        for (const Geometry geometry :
             {Geometry{1, 1, 0}, Geometry{1, 2, 2}, Geometry{2, 2, 1}, Geometry{2, 4, 2}, Geometry{8, 8, 2}}) {
            std::vector<CacheGroupSpec> specs{CacheGroupSpec{.attn_kind = AttnKind::kFull,
                                                             .cache_blocks_per_lcm_block = geometry.packing,
                                                             .block_granularity = 2,
                                                             .shard_count = geometry.shards}};
            std::vector<std::int32_t> packings{geometry.packing};
            for (std::int32_t i = 0; i < geometry.num_state_groups; ++i) {
                specs.push_back(CacheGroupSpec{
                    .attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0});
                packings.push_back(1);
            }
            const auto num_groups = static_cast<std::uint32_t>(specs.size());
            for (std::int32_t trial = 0; trial < 120; ++trial) {
                SCOPED_TRACE(testing::Message() << "by segments " << place_by_segments << ", shards " << geometry.shards
                                                << ", packing " << geometry.packing << ", State groups "
                                                << geometry.num_state_groups << ", trial " << trial);
                BlockPool pool(32, packings);
                BlockPool host_pool(2 + static_cast<std::int32_t>(rng() % 5), packings);
                BlockPool snapshot_pool(1 + static_cast<std::int32_t>(rng() % 4), packings);
                CacheCoordinator coordinator =
                    MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                                    /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/place_by_segments);
                // Outlives the transfer manager, whose snapshot store holds a slot.
                SnapshotSlotAllocator blob_slots(1);
                TierTransferManager transfers(coordinator);
                std::vector<BlockTable> tables(num_groups);
                const auto num_pages = 1 + static_cast<std::int32_t>(rng() % 8);
                ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2 * num_pages));
                std::vector<std::vector<std::int32_t>> page_tokens;
                for (std::int32_t page = 0; page < num_pages; ++page) {
                    page_tokens.push_back({1000 * trial + 2 * page + 1, 1000 * trial + 2 * page + 2});
                }
                CacheCompletedBlocksForTest(coordinator, tables, ContentHashes(page_tokens),
                                            CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                            /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2 * num_pages,
                                            CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                            /*publish_state=*/geometry.num_state_groups > 0 && rng() % 2 == 0);
                const std::vector<std::vector<ImageSlot>> published =
                    coordinator.PublishedDataSlots(tables, 2 * num_pages);

                // The Host: the image's own pages (none, some or all of them, each
                // in its Device block's bucket) and other prompts' entries, in a
                // random order, while blocks are left. One in four is pinned.
                std::vector<CacheKey> host_keys;
                std::vector<CacheBlockRef> pinned;
                const std::uint32_t reuse = rng() % 3;
                std::vector<std::pair<CacheKey, std::int32_t>> placements;
                for (const std::vector<ImageSlot>& slots : published) {
                    for (const ImageSlot& slot : slots) {
                        if (reuse == 2 || (reuse == 1 && rng() % 2 == 0)) {
                            placements.emplace_back(slot.key,
                                                    coordinator.Allocator(static_cast<std::int32_t>(slot.key.group_id))
                                                        .BucketOf(slot.block->Location()));
                        }
                    }
                }
                // Other prompts' hashes repeat across groups, so a State entry may
                // have siblings at its hash.
                for (std::int32_t other = 0; other < 6 * geometry.packing; ++other) {
                    const std::uint32_t group_id = rng() % num_groups;
                    placements.emplace_back(Key("other" + std::to_string(rng() % (2 * geometry.packing)), group_id),
                                            static_cast<std::int32_t>(rng() % (group_id == 0 ? geometry.shards : 1)));
                }
                std::shuffle(placements.begin(), placements.end(), rng);
                for (const auto& [key, bucket] : placements) {
                    if (coordinator.ContainsHostCachedBlock(key)) {
                        continue;
                    }
                    std::vector<CacheBlockRef> blocks =
                        host_pool.AcquireBlocksInBuckets(key.group_id, std::array<std::int32_t, 1>{bucket});
                    if (blocks.empty()) {
                        continue;
                    }
                    coordinator.CacheHostBlock(blocks.front(), key, std::nullopt);
                    host_keys.push_back(key);
                    if (rng() % 4 == 0) {
                        pinned.push_back(std::move(blocks.front()));
                    }
                }
                const auto host_locations = [&] {
                    std::vector<std::optional<CacheBlockLocation>> locations;
                    for (const CacheKey& key : host_keys) {
                        const CacheBlockRef block = coordinator.FindHostCachedBlock(key);
                        locations.push_back(block ? std::optional{block->Location()} : std::nullopt);
                    }
                    return locations;
                };
                const std::vector<std::optional<CacheBlockLocation>> before = host_locations();

                TierTransferManager::RetractionImageSplit split =
                    transfers.SplitRetractionImage(tables, 2 * num_pages, published);
                EXPECT_EQ(host_locations(), before) << "the split mutates nothing";
                const std::size_t k = split.num_host_boundaries;
                ASSERT_LE(k, split.boundaries.size());
                for (std::size_t j = 0; j <= split.boundaries.size(); ++j) {
                    const std::size_t num_rows = j == 0 ? 0 : split.boundaries[j - 1].past_end_row;
                    EXPECT_EQ(coordinator.NumHostBoundariesThatFit(std::span(split.group_ids).first(num_rows),
                                                                   std::span(split.buckets).first(num_rows),
                                                                   std::span(split.boundaries).first(j)),
                              std::min(j, k))
                        << j << " boundaries";
                }
                num_without_rows += split.slots.empty() ? 1 : 0;
                num_short_of_the_host += k < split.boundaries.size() ? 1 : 0;
                if (split.shortfall) {
                    ++num_refused;
                    EXPECT_EQ(host_locations(), before);
                    coordinator.Free(tables);
                    continue;
                }
                ++num_taken;
                const std::int32_t num_cached_before = coordinator.NumHostCachedBlocks();
                const std::size_t num_host_rows = k == 0 ? 0 : split.boundaries[k - 1].past_end_row;
                std::vector<CacheTransfer> expected_copies;
                for (std::size_t row = 0; row < num_host_rows; ++row) {
                    expected_copies.push_back(CacheTransfer{.group_id = split.group_ids[row],
                                                            .content_hash = split.slots[row].key.content_hash,
                                                            .page_offset = split.slots[row].key.page_offset});
                }
                const std::size_t num_pool_rows = split.slots.size() - num_host_rows;
                std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
                    "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()),
                    tables, 2 * num_pages, std::move(split));
                ASSERT_TRUE(stores.has_value());
                std::vector<CacheTransfer> copies;
                if (stores->host_store) {
                    for (const CacheTransfer& copy : stores->host_store->transfers) {
                        copies.push_back(CacheTransfer{.group_id = copy.group_id,
                                                       .content_hash = copy.content_hash,
                                                       .page_offset = copy.page_offset});
                    }
                }
                EXPECT_EQ(copies, expected_copies);
                const TierTransferManager::WriteBackStats& stats =
                    transfers.WriteBackStatsFor(HostStoreCause::kRetraction);
                EXPECT_EQ(stats.snapshot_fallback, num_pool_rows);
                EXPECT_EQ(stats.image_refused, 0u);
                num_taken_after_evicting += coordinator.NumHostCachedBlocks() < num_cached_before ? 1 : 0;
                num_taken_with_pool_rows += num_pool_rows > 0 ? 1 : 0;
                const std::vector<std::optional<CacheBlockLocation>> after = host_locations();
                for (std::size_t i = 0; i < host_keys.size(); ++i) {
                    const bool is_pinned = std::ranges::any_of(
                        pinned, [&](const CacheBlockRef& block) { return block->Location() == before[i]; });
                    const bool reused = host_keys[i].content_hash.rfind("other", 0) != 0;
                    if (is_pinned || reused) {
                        EXPECT_EQ(after[i], before[i]) << host_keys[i].content_hash;
                    }
                }
                coordinator.Free(tables);
            }
        }
    }
    // Every case occurred.
    EXPECT_GT(num_taken, 0u);
    EXPECT_GT(num_taken_after_evicting, 0u);
    EXPECT_GT(num_taken_with_pool_rows, 0u);
    EXPECT_GT(num_refused, 0u);
    EXPECT_GT(num_without_rows, 0u);
    EXPECT_GT(num_short_of_the_host, 0u);
}

// ---------------------------------------------------------------------------
// A retraction image's Host L2 leg under the write-back policy
// (write_back_evictions): of the leading boundaries the Host can take, it takes
// room by whole segments first, as an eviction write-back does, then block by
// block for the boundaries the segment step drops
// (CacheCoordinator::NumHostBoundariesThatFit). The write-back transaction's
// K3 fixture with a snapshot pool. Unless a test says else, an image is a
// request's one page of Full KV published without a checkpoint: its leg is
// one Full block, and the rest of it rides the pool.
// ---------------------------------------------------------------------------
class ImageWriteBackTransactionTest : public WriteBackTransactionTest {
protected:
    ImageWriteBackTransactionTest() : WriteBackTransactionTest(/*full_shard_count=*/1, /*num_snapshot_pages=*/32) {}
};

// The Host holds two finished sessions over the same two pages and nothing
// else -- s0 s1 a2 a3 tA and s0 s1 b2 b3 tB with their checkpoints, s1 its
// least recently used entry -- and no room. By segments the leg takes a leaf
// of the Host's tree, A's own pages and tail, whole, and B still resumes from
// the Host. Block by block it would take s1, and the next collection both
// sessions under that gap.
TEST_F(ImageWriteBackTransactionTest, AnImagesL2LegTakesAHostLeafNotAMiddlePage) {
    PublishTwoSessionsSharingTwoPages();
    const std::vector<CacheKey> a = KeysOf(a_hashes_, a_tail_);
    const std::vector<CacheKey> b = KeysOf(b_hashes_, b_tail_);
    StoreOnHost(a[1]);
    StoreOnHost(a[0]);
    for (std::size_t member = 2; member < a.size(); ++member) {
        StoreOnHost(a[member]);
    }
    for (std::size_t member = 2; member < b.size(); ++member) {
        StoreOnHost(b[member]);
    }
    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    std::vector<BlockTable> image(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, image, kP));
    CacheCompletedBlocksForTest(coordinator_, image, ContentHashes({MakeTokens(kP, 900)}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(image, kP, coordinator_.PublishedDataSlots(image, kP));
    ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0}));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, kP,
        std::move(split));
    ASSERT_TRUE(stores.has_value());
    coordinator_.Free(image);

    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 6u) << "a2, a3, A's tail block and its checkpoints";
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(HostEntries(b), 8) << "B resumes from the Host";
    EXPECT_EQ(HostEntries(a), 2) << "s0 and s1";
}

// A's write-back is in flight -- a2, a3, A's tail block and checkpoints,
// under s1 -- and the Host holds B with s0 and s1, all of one age, s1 and s0
// in its first blocks, and no room. s0 and s1 are no victim by segments: A
// would land under a gap. The leg takes B's own pages and tail; A publishes
// at its ACK and resumes from the Host. Block by block the leg would take s1,
// the first of the oldest entries.
TEST_F(ImageWriteBackTransactionTest, AnImagesL2LegDoesNotCutAboveAnInFlightSegment) {
    const std::vector<std::int32_t> a_tokens = MakeTokens(18);
    std::vector<std::int32_t> b_tokens = MakeTokens(18, 100);
    std::copy_n(a_tokens.begin(), 2 * kP, b_tokens.begin());
    std::vector<BlockTable> tables(4);
    const TailPublication a_tail = *PublishFinal(tables, a_tokens);
    const std::vector<CacheKey> a = KeysOf(hashes_, a_tail);
    coordinator_.Free(tables);
    // Host copies of s1 and s0, made before B's publication touches them: B's
    // write-back publishes them at B's age, as its own blocks.
    StoreOnHost(a[1]);
    StoreOnHost(a[0]);
    const TailPublication b_tail = *PublishFinal(tables, b_tokens);
    const std::vector<CacheKey> b = KeysOf(hashes_, b_tail);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> a_op = EvictOldestAndDrain();
    ASSERT_TRUE(a_op.has_value());
    ASSERT_EQ(a_op->transfers.size(), 6u) << "a2, a3, A's tail block and its checkpoints";
    const std::optional<WriteBackOperation> b_op = EvictOldestAndDrain();
    ASSERT_TRUE(b_op.has_value());
    ASSERT_EQ(b_op->transfers.size(), 6u) << "B's own; s0 and s1 are on the Host";
    ASSERT_EQ(transfers_.CompleteWriteBack(b_op->op_id).size(), 8u);
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    std::vector<BlockTable> image(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, image, kP));
    CacheCompletedBlocksForTest(coordinator_, image, ContentHashes({MakeTokens(kP, 900)}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(image, kP, coordinator_.PublishedDataSlots(image, kP));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, kP,
        std::move(split));
    ASSERT_TRUE(stores.has_value());
    coordinator_.Free(image);
    EXPECT_EQ(HostEntries(b), 2) << "s0 and s1";
    EXPECT_EQ(coordinator_.HostEvictions().lru.blocks, 6u) << "b2, b3, B's tail block and its checkpoints";

    EXPECT_EQ(transfers_.CompleteWriteBack(a_op->op_id).size(), 6u);
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(HostEntries(a), 8) << "A resumes from the Host";
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 0u);
}

// As above, A's write-back is in flight under s1, but B's own entries have
// left the Host and a reader holds s0: the Host holds s0 and s1 alone, and no
// room, and no victim by segments is left. The leg falls back to single
// blocks and takes s1 -- the image comes before the cache. A still publishes
// at its ACK (its checkpoint boundary on its way counts on the Host, and its
// settlement does not look above its node), and the next collection takes it
// under the gap. Nothing stays pinned or in flight once the image's copies
// land and it is let go.
TEST_F(ImageWriteBackTransactionTest, AnImagesL2LegFallsBackToSingleBlocksAboveAnInFlightSegment) {
    PublishTwoSessionsSharingTwoPages();
    const std::optional<WriteBackOperation> a_op = EvictOldestAndDrain();
    ASSERT_TRUE(a_op.has_value());
    ASSERT_EQ(a_op->transfers.size(), 6u);
    const std::optional<WriteBackOperation> b_op = EvictOldestAndDrain();
    ASSERT_TRUE(b_op.has_value());
    ASSERT_EQ(transfers_.CompleteWriteBack(b_op->op_id).size(), 8u) << "B with s0 and s1";
    ASSERT_TRUE(CacheCoordinatorTestAccess::EvictHostCachedBlock(coordinator_, Key(b_tail_.hash, 1)));
    coordinator_.CollectDeadEntries();
    ASSERT_EQ(coordinator_.NumHostCachedBlocks(), 2) << "s0 and s1";
    CacheBlockRef reader = coordinator_.AcquireHostCachedBlock(Key(b_hashes_[0], 0));
    ASSERT_TRUE(static_cast<bool>(reader));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    std::vector<BlockTable> image(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, image, kP));
    CacheCompletedBlocksForTest(coordinator_, image, ContentHashes({MakeTokens(kP, 900)}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(image, kP, coordinator_.PublishedDataSlots(image, kP));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, kP,
        std::move(split));
    ASSERT_TRUE(stores.has_value());
    coordinator_.Free(image);
    EXPECT_FALSE(coordinator_.ContainsHostCachedBlock(Key(b_hashes_[1], 0))) << "s1";

    EXPECT_EQ(transfers_.CompleteWriteBack(a_op->op_id).size(), 6u) << "A publishes";
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 0u);
    const CacheCoordinator::HostEvictionStats before = coordinator_.HostEvictions();
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.HostEvictions().dead_entry.blocks - before.dead_entry.blocks, 6u)
        << "a2, a3, A's tail block and its checkpoints, under the gap";

    reader.reset();
    ASSERT_TRUE(stores->host_store.has_value());
    transfers_.CompleteWriteBack(stores->host_store->op_id);
    transfers_.CompleteSnapshotStore(stores->snapshot_store.op_id);
    stores.reset();
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers_.HasAnyInFlight());
}

// An image's Host entries stay pinned until its restore. A first image's
// page lands on the Host first, its oldest Full entry; a session Y follows,
// on the Host alone, and the Host has no room. A second image's leg takes Y
// by segments. Then a session Z, on the Host alone, whose tail KV a reader
// holds. A third image's leg reclaims Z's unpinned State while retaining its
// KV. Neither step touches the earlier images' entries.
TEST_F(ImageWriteBackTransactionTest, AnImagesL2LegLeavesAnotherImagesEntries) {
    std::vector<BlockTable> first(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, first, kP));
    const std::vector<std::string> first_page = ContentHashes({MakeTokens(kP, 900)});
    CacheCompletedBlocksForTest(coordinator_, first, first_page,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(first, kP, coordinator_.PublishedDataSlots(first, kP));
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> first_image = transfers_.StartRetractionStores(
        "first", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), first, kP,
        std::move(split));
    ASSERT_TRUE(first_image.has_value());
    ASSERT_TRUE(first_image->host_store.has_value());
    transfers_.CompleteWriteBack(first_image->host_store->op_id);
    transfers_.CompleteSnapshotStore(first_image->snapshot_store.op_id);
    coordinator_.Free(first);
    const CacheKey first_key = Key(first_page[0], 0);
    const CacheBlockLocation first_location = coordinator_.FindHostCachedBlock(first_key)->Location();

    std::vector<BlockTable> tables(4);
    const TailPublication y_tail = *PublishFinal(tables, MakeTokens(18, 300));
    const std::vector<CacheKey> y = KeysOf(hashes_, y_tail);
    coordinator_.Free(tables);
    for (const CacheKey& key : y) {
        StoreOnHost(key);
    }
    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    std::vector<BlockTable> second(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, second, kP));
    const std::vector<std::string> second_page = ContentHashes({MakeTokens(kP, 950)});
    CacheCompletedBlocksForTest(coordinator_, second, second_page,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    split = transfers_.SplitRetractionImage(second, kP, coordinator_.PublishedDataSlots(second, kP));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> second_image = transfers_.StartRetractionStores(
        "second", /*request_pool_index=*/2, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), second, kP,
        std::move(split));
    ASSERT_TRUE(second_image.has_value());
    ASSERT_TRUE(second_image->host_store.has_value());
    transfers_.CompleteWriteBack(second_image->host_store->op_id);
    transfers_.CompleteSnapshotStore(second_image->snapshot_store.op_id);
    coordinator_.Free(second);
    EXPECT_EQ(HostEntries(y), 0) << "Y, whole";
    EXPECT_EQ(coordinator_.FindHostCachedBlock(first_key)->Location(), first_location);
    const CacheKey second_key = Key(second_page[0], 0);
    const CacheBlockLocation second_location = coordinator_.FindHostCachedBlock(second_key)->Location();

    held_.clear();
    const TailPublication z_tail = *PublishFinal(tables, MakeTokens(18, 400));
    const std::vector<CacheKey> z = KeysOf(hashes_, z_tail);
    coordinator_.Free(tables);
    for (const CacheKey& key : z) {
        StoreOnHost(key);
    }
    ASSERT_TRUE(coordinator_.ClearDeviceCache());
    const CacheBlockRef reader = coordinator_.AcquireHostCachedBlock(Key(z_tail.hash, 0));
    ASSERT_TRUE(static_cast<bool>(reader));
    HoldFreeCapacity(host_pool_, /*spare_parents=*/0);

    std::vector<BlockTable> third(4);
    ASSERT_TRUE(AdmitForTest(coordinator_, third, kP));
    CacheCompletedBlocksForTest(coordinator_, third, ContentHashes({MakeTokens(kP, 990)}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/kP, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    split = transfers_.SplitRetractionImage(third, kP, coordinator_.PublishedDataSlots(third, kP));
    ASSERT_EQ(split.num_host_boundaries, 1u);
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> third_image = transfers_.StartRetractionStores(
        "third", /*request_pool_index=*/3, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), third, kP,
        std::move(split));
    ASSERT_TRUE(third_image.has_value());
    coordinator_.Free(third);
    for (const CacheKey& key : z) {
        EXPECT_EQ(coordinator_.ContainsHostCachedBlock(key), key.group_id == 0);
    }
    EXPECT_EQ(HostEntries(z), 5) << "Z's four pages and pinned tail KV remain";
    EXPECT_EQ(coordinator_.FindHostCachedBlock(first_key)->Location(), first_location);
    EXPECT_EQ(coordinator_.FindHostCachedBlock(second_key)->Location(), second_location);
}

// A retracted session's pages are cache on the Device once the victim lets
// go, and an admission of the same plan evicts them while the image's own
// write-back carries every one of them to the Host. The eviction's drain
// borrows that store's blocks -- they publish at its ACK -- and copies
// nothing: each block crosses once, and the segment publishes at the image
// store's ACK, on the same Host entries.
TEST_F(ImageWriteBackTransactionTest, AnEvictionDrainBorrowsAnImagesTicket) {
    std::vector<BlockTable> image(4);
    const TailPublication tail = *PublishFinal(image, MakeTokens(18));
    const std::vector<CacheKey> keys = KeysOf(hashes_, tail);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(image, 18, coordinator_.PublishedDataSlots(image, 18));
    ASSERT_EQ(split.slots.size(), 8u) << "four pages, the tail block and its checkpoints";
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, 18,
        std::move(split));
    ASSERT_TRUE(stores.has_value());
    ASSERT_TRUE(stores->host_store.has_value());
    ASSERT_EQ(stores->host_store->transfers.size(), 8u);
    coordinator_.Free(image);

    EXPECT_FALSE(EvictOldestAndDrain().has_value()) << "nothing to copy";
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.in_flight, 8u);
    EXPECT_EQ(stats.ops, 0u);
    EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{0, 0, 0, 0}));
    EXPECT_EQ(transfers_.CompleteWriteBack(stores->host_store->op_id).size(), 8u + 8u)
        << "the image's entries, then the segment's, the same";
    EXPECT_EQ(HostEntries(keys), 8);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
    EXPECT_EQ(stats.cancelled, 0u);
}

// X's eviction write-back is in flight when a request publishes X's prompt on
// the Device again and is retracted: its image's slots are keys that
// write-back carries, but those tickets publish only through X's segment,
// which may wait for other ops or be cancelled. The image borrows none of
// them: it takes Host blocks of its own and copies every slot. The per-prior
// cap then drops the prompt's tail, and X's segment is cancelled at its ACK;
// the image's copies land and publish all the same.
TEST_F(ImageWriteBackTransactionTest, AnImageDoesNotBorrowAnEvictionWriteBacksTicket) {
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, tokens);
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    ASSERT_EQ(x_op->transfers.size(), 8u);

    std::vector<BlockTable> image(4);
    const TailPublication tail = *PublishFinal(image, tokens);
    const std::vector<CacheKey> keys = KeysOf(pages, tail);
    TierTransferManager::RetractionImageSplit split =
        transfers_.SplitRetractionImage(image, 18, coordinator_.PublishedDataSlots(image, 18));
    EXPECT_TRUE(split.pending_store_ops.empty()) << "nothing borrowed";
    ASSERT_EQ(split.slots.size(), 8u);
    ASSERT_EQ(split.num_host_boundaries, split.boundaries.size());
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, 18,
        std::move(split));
    ASSERT_TRUE(stores.has_value());
    ASSERT_TRUE(stores->host_store.has_value());
    EXPECT_EQ(stores->host_store->transfers.size(), 8u);
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kRetraction).in_flight, 0u);
    coordinator_.Free(image);

    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 1);
    EXPECT_TRUE(transfers_.CompleteWriteBack(x_op->op_id).empty());
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 8u);
    EXPECT_EQ(transfers_.CompleteWriteBack(stores->host_store->op_id).size(), 8u);
    EXPECT_EQ(HostEntries(keys), 8);
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 8) << "the image's";
}

// The segment step plans every boundary of an image and places the leading
// ones the Host has room for -- none, some or all of three, dropping them
// from the deepest. One Full group, a block per page, under the write-back
// policy; the Host holds the parameter's number of an older prompt's pages,
// alone, and no other room. The leg takes their places, a boundary each, and
// the pool takes the rest.
class ImageHostPrefixTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(HostPrefixes, ImageHostPrefixTest, ::testing::Values(0, 2, 3));

TEST_P(ImageHostPrefixTest, TheSegmentStepPlacesTheBoundariesTheHostCanTake) {
    const std::int32_t num_host_pages = GetParam();
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    BlockPool pool(8, {1});
    BlockPool host_pool(3, {1});
    BlockPool snapshot_pool(3, {1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    // Outlives the transfer manager, whose snapshot store holds a slot.
    SnapshotSlotAllocator blob_slots(1);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> older = ContentHashes({{91, 92}, {93, 94}, {95, 96}});
    std::vector<BlockTable> tables(1);
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    CacheCompletedBlocksForTest(coordinator, tables, older, CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/6, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    coordinator.Free(tables);
    std::vector<CacheBlockRef> held;
    for (std::int32_t page = 0; page < 3; ++page) {
        CacheBlockRef block = host_pool.AcquireBlock(0);
        if (page < num_host_pages) {
            coordinator.CacheHostBlock(block, Key(older[static_cast<std::size_t>(page)], 0), std::nullopt);
        } else {
            held.push_back(std::move(block));
        }
    }
    ASSERT_TRUE(coordinator.ClearDeviceCache());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    const std::vector<std::string> image_pages = ContentHashes({{1, 2}, {3, 4}, {5, 6}});
    CacheCompletedBlocksForTest(coordinator, tables, image_pages,
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/6, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers.SplitRetractionImage(tables, /*num_tokens=*/6, coordinator.PublishedDataSlots(tables, 6));
    ASSERT_EQ(split.boundaries.size(), 3u);
    EXPECT_EQ(split.num_host_boundaries, static_cast<std::size_t>(num_host_pages));
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
        /*num_tokens=*/6, std::move(split));
    ASSERT_TRUE(stores.has_value());
    std::vector<std::string> copied;
    if (stores->host_store) {
        for (const CacheTransfer& copy : stores->host_store->transfers) {
            copied.push_back(copy.content_hash);
        }
    }
    EXPECT_EQ(copied, std::vector<std::string>(image_pages.begin(), image_pages.begin() + num_host_pages));
    EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback,
              static_cast<std::uint64_t>(3 - num_host_pages));
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, static_cast<std::uint64_t>(num_host_pages))
        << "the older prompt's pages";
    coordinator.Free(tables);
}

// Both steps in one image, the fallback planning against the Host the
// segment step leaves. One Full group, a block per page, under the write-back
// policy. The Host's five blocks hold an older prompt Z, z0 z1 z2, its last
// page read, and then a prompt Y, y0 y1, on the Host alone: z0 and z1 are the
// oldest entries, but no victim by segments, Z's leaf being in use. The
// image's three pages: by segments Y's two pages, a leaf at a time, make room
// for the first two; the fallback places the third in z0's place, its room
// counted beside the room Y left, and z1 stays.
TEST(ImageSegmentsThenBlocksTest, TheFallbackPlansOnTheHostTheSegmentsLeave) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    BlockPool pool(8, {1});
    BlockPool host_pool(5, {1});
    BlockPool snapshot_pool(1, {1});
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    // Outlives the transfer manager, whose snapshot store holds a slot.
    SnapshotSlotAllocator blob_slots(1);
    TierTransferManager transfers(coordinator);
    const std::vector<std::string> z = ContentHashes({{91, 92}, {93, 94}, {95, 96}});
    const std::vector<std::string> y = ContentHashes({{81, 82}, {83, 84}});
    std::vector<BlockTable> tables(1);
    for (const std::vector<std::string>& prompt : {z, y}) {
        const auto num_tokens = 2 * static_cast<std::int32_t>(prompt.size());
        ASSERT_TRUE(AdmitForTest(coordinator, tables, num_tokens));
        CacheCompletedBlocksForTest(coordinator, tables, prompt,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, num_tokens, CacheBoundaryKind::kEndpoint,
                                    /*stream_completed_to_host=*/false, /*publish_state=*/false);
        coordinator.Free(tables);
        for (const std::string& page : prompt) {
            CacheBlockRef block = host_pool.AcquireBlock(0);
            coordinator.CacheHostBlock(block, Key(page, 0), std::nullopt);
        }
    }
    ASSERT_TRUE(coordinator.ClearDeviceCache());
    ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
    const CacheBlockRef reader = coordinator.AcquireHostCachedBlock(Key(z.back(), 0));

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    CacheCompletedBlocksForTest(coordinator, tables, ContentHashes({{1, 2}, {3, 4}, {5, 6}}),
                                CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                /*first_new_prefix_page=*/0, /*num_computed_tokens=*/6, CacheBoundaryKind::kEndpoint,
                                /*stream_completed_to_host=*/false, /*publish_state=*/false);
    TierTransferManager::RetractionImageSplit split =
        transfers.SplitRetractionImage(tables, /*num_tokens=*/6, coordinator.PublishedDataSlots(tables, 6));
    ASSERT_EQ(split.num_host_boundaries, 3u);
    ASSERT_FALSE(split.shortfall.has_value());
    const std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
        "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
        /*num_tokens=*/6, std::move(split));
    ASSERT_TRUE(stores.has_value());
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(y[0], 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(y[1], 0)));
    EXPECT_FALSE(coordinator.ContainsHostCachedBlock(Key(z[0], 0)));
    EXPECT_TRUE(coordinator.ContainsHostCachedBlock(Key(z[1], 0))) << "the fallback counts the room Y left";
    EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 3u);
    coordinator.Free(tables);
}

// Where the two steps place fewer boundaries than the per-block plan alone,
// the per-block plan stands: the image comes before the cache, and the
// write-back policy places it as write-through does. A Full group of two
// blocks per parent and two State groups. The Host's five parents hold other
// prompts' entries, oldest first: Full entries z and x in one parent; x's
// checkpoint in the second State group, pinned; a Full entry y beside a free
// slot; a checkpoint w in the first State group; y's checkpoint in that
// group. The image's four pages need a Full block each, its last page a
// block of each State group too, and the snapshot pool holds the State
// groups' working slots alone. Block by block the Host takes all four
// boundaries, evicting z, x and y for Full blocks and w and y's checkpoint
// for the State blocks. By segments w alone, a dead entry, makes room for the
// first three pages; the fallback then finds no parent for the last page's
// second State block -- z's parent keeps the Full block that takes z's place,
// y's the free slot the rows need -- and the image would be refused.
TEST(ImageSegmentsThenBlocksTest, TheBlockByBlockPlanStandsWhereTheStepsPlaceFewer) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const bool write_back_evictions : {false, true}) {
        SCOPED_TRACE(write_back_evictions);
        BlockPool pool(8, {2, 1, 1});
        BlockPool host_pool(5, {2, 1, 1});
        BlockPool snapshot_pool(2, {2, 1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, write_back_evictions);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}, {97, 98}});
        const CacheKey z = Key(cached[0], 0);
        const CacheKey x = Key(cached[1], 0);
        const CacheKey x_checkpoint = Key(cached[1], 2);
        const CacheKey y = Key(cached[2], 0);
        const CacheKey w = Key(cached[3], 1);
        const CacheKey y_checkpoint = Key(cached[2], 1);
        CacheBlockRef pinned;
        for (const CacheKey& key : {z, x, x_checkpoint, y, w, y_checkpoint}) {
            CacheBlockRef block = host_pool.AcquireBlock(key.group_id);
            coordinator.CacheHostBlock(block, key, std::nullopt);
            if (key.group_id == 2) {
                pinned = std::move(block);
            }
        }
        ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
        ASSERT_EQ(host_pool.NumFreeSlots(0), 1) << "beside y";
        std::vector<BlockTable> tables(3);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
        CacheCompletedBlocksForTest(coordinator, tables, ContentHashes({{1, 2}, {3, 4}, {5, 6}, {7, 8}}),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/8,
                                    CacheBoundaryKind::kEndpoint,
                                    /*stream_completed_to_host=*/false, /*publish_state=*/true);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/8, coordinator.PublishedDataSlots(tables, 8));
        ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 0, 0, 0, 1, 2}));
        ASSERT_EQ(split.num_host_boundaries, 4u);
        ASSERT_FALSE(split.shortfall.has_value());
        const std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
            /*num_tokens=*/8, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        EXPECT_EQ(stores->host_store->transfers.size(), 6u);
        EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 0u);
        for (const CacheKey& key : {z, x, y, w, y_checkpoint}) {
            EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key)) << key.content_hash << " " << key.group_id;
        }
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(x_checkpoint));
        EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 5u);
        coordinator.Free(tables);
    }
}

// The segment step can place more boundaries than the per-block plan alone.
// A Full group of two blocks per parent and two State groups, under each Host
// write policy. The Host's three parents hold other prompts' entries,
// oldest first: a Full entry x, pinned; a checkpoint v in the first State
// group; w's checkpoint in that group; w's Full entry, beside x. The image's
// one page needs a Full block and a block of each State group, and the
// snapshot pool holds the State groups' working slots alone. Block by block
// the Full row takes v's parent whole, older than w's Full entry, the first
// State row w's checkpoint, and the second State row finds no parent left:
// the image would be refused. By segments v and w, dead checkpoint boundaries
// (neither has the second State group's checkpoint), go whole, w's Full entry
// with its checkpoint, and the three rows fit: the Full row in w's slot, each
// State row in a parent.
TEST(ImageSegmentsThenBlocksTest, TheSegmentStepPlacesABoundaryTheBlockByBlockPlanCannot) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const bool write_back_evictions : {false, true}) {
        SCOPED_TRACE(write_back_evictions);
        BlockPool pool(8, {2, 1, 1});
        BlockPool host_pool(3, {2, 1, 1});
        BlockPool snapshot_pool(2, {2, 1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, write_back_evictions);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}});
        const CacheKey x = Key(cached[0], 0);
        const CacheKey v = Key(cached[1], 1);
        const CacheKey w_checkpoint = Key(cached[2], 1);
        const CacheKey w = Key(cached[2], 0);
        CacheBlockRef pinned;
        for (const CacheKey& key : {x, v, w_checkpoint, w}) {
            CacheBlockRef block = host_pool.AcquireBlock(key.group_id);
            coordinator.CacheHostBlock(block, key, std::nullopt);
            if (key == x) {
                pinned = std::move(block);
            }
        }
        ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
        ASSERT_EQ(host_pool.NumFreeSlots(0), 0) << "w beside x";
        std::vector<BlockTable> tables(3);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/2));
        CacheCompletedBlocksForTest(
            coordinator, tables, ContentHashes({{1, 2}}), CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
            /*first_new_prefix_page=*/0, /*num_computed_tokens=*/2, CacheBoundaryKind::kEndpoint,
            /*stream_completed_to_host=*/false, /*publish_state=*/true);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/2, coordinator.PublishedDataSlots(tables, 2));
        if (!write_back_evictions) {
            EXPECT_EQ(split.num_host_boundaries, 0u);
            EXPECT_EQ(split.shortfall, ImageShortfall::kHostL2);
            coordinator.Free(tables);
            continue;
        }
        ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 1, 2}));
        ASSERT_EQ(split.num_host_boundaries, 1u);
        ASSERT_FALSE(split.shortfall.has_value());
        const std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
            /*num_tokens=*/2, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        EXPECT_EQ(stores->host_store->transfers.size(), 3u);
        EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 0u);
        for (const CacheKey& key : {v, w_checkpoint, w}) {
            EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key)) << key.content_hash << " " << key.group_id;
        }
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(x));
        EXPECT_EQ(coordinator.HostEvictions().dead_entry.blocks, 3u);
        EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
        coordinator.Free(tables);
    }
}

// The segment step places more boundaries than the per-block plan also where
// that plan places some. The previous test's groups and Host, and a fourth
// parent holding a pinned Full entry y beside a free slot; the image's first
// page needs a Full block, its second a Full block and a block of each State
// group. Block by block the first page's row takes the free slot beside y,
// and the second page's rows fall short as the previous test's do: the Full
// row takes v's parent whole, the first State row w's checkpoint, and the
// second State row finds no parent left. By segments v and w go whole as
// before, and both pages fit: the Full rows beside y and in w's slot, each
// State row in a parent.
TEST(ImageSegmentsThenBlocksTest, TheSegmentStepPlacesMoreBoundariesThanTheBlockByBlockPlanDoes) {
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
    };
    for (const bool write_back_evictions : {false, true}) {
        SCOPED_TRACE(write_back_evictions);
        BlockPool pool(8, {2, 1, 1});
        BlockPool host_pool(4, {2, 1, 1});
        BlockPool snapshot_pool(2, {2, 1, 1});
        CacheCoordinator coordinator =
            MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, &host_pool, &snapshot_pool,
                            /*stream_device_cache_to_host=*/false, write_back_evictions);
        // Outlives the transfer manager, whose snapshot store holds a slot.
        SnapshotSlotAllocator blob_slots(1);
        TierTransferManager transfers(coordinator);
        const std::vector<std::string> cached = ContentHashes({{91, 92}, {93, 94}, {95, 96}, {97, 98}});
        const CacheKey x = Key(cached[0], 0);
        const CacheKey v = Key(cached[1], 1);
        const CacheKey w_checkpoint = Key(cached[2], 1);
        const CacheKey w = Key(cached[2], 0);
        const CacheKey y = Key(cached[3], 0);
        std::vector<CacheBlockRef> pinned;
        for (const CacheKey& key : {x, v, w_checkpoint, w, y}) {
            CacheBlockRef block = host_pool.AcquireBlock(key.group_id);
            coordinator.CacheHostBlock(block, key, std::nullopt);
            if (key == x || key == y) {
                pinned.push_back(std::move(block));
            }
        }
        ASSERT_EQ(host_pool.NumEmptyLcmBlocks(), 0);
        ASSERT_EQ(host_pool.NumFreeSlots(0), 1) << "beside y";
        std::vector<BlockTable> tables(3);
        ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
        CacheCompletedBlocksForTest(coordinator, tables, ContentHashes({{1, 2}, {3, 4}}),
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/4,
                                    CacheBoundaryKind::kEndpoint,
                                    /*stream_completed_to_host=*/false, /*publish_state=*/true);

        TierTransferManager::RetractionImageSplit split =
            transfers.SplitRetractionImage(tables, /*num_tokens=*/4, coordinator.PublishedDataSlots(tables, 4));
        if (!write_back_evictions) {
            EXPECT_EQ(split.num_host_boundaries, 1u);
            EXPECT_EQ(split.shortfall, ImageShortfall::kHostL2);
            coordinator.Free(tables);
            continue;
        }
        ASSERT_EQ(split.group_ids, (std::vector<std::uint32_t>{0, 0, 1, 2}));
        ASSERT_EQ(split.num_host_boundaries, 2u);
        ASSERT_FALSE(split.shortfall.has_value());
        const std::optional<TierTransferManager::RetractionStores> stores = transfers.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), tables,
            /*num_tokens=*/4, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        EXPECT_EQ(stores->host_store->transfers.size(), 4u);
        EXPECT_EQ(transfers.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 0u);
        for (const CacheKey& key : {v, w_checkpoint, w}) {
            EXPECT_FALSE(coordinator.ContainsHostCachedBlock(key)) << key.content_hash << " " << key.group_id;
        }
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(x));
        EXPECT_TRUE(coordinator.ContainsHostCachedBlock(y));
        EXPECT_EQ(coordinator.HostEvictions().dead_entry.blocks, 3u);
        EXPECT_EQ(coordinator.HostEvictions().lru.blocks, 0u);
        coordinator.Free(tables);
    }
}

// Under KV parallelism the leg's Host blocks lie in their Device blocks'
// buckets, placed by segments or by the fallback, as the split predicts. The
// write-back transaction's K3 fixture, its Full group dealt over the
// parameter's shards, with a snapshot pool. Two finished sessions, X and Y,
// three pages and a tail each, stay on the Device and are copied to the Host,
// which has no other room; a reader holds Y's tail checkpoint. An image's
// pages, one per bucket, need a Host block each. By segments its leg takes
// X's Host copy whole, a duplicate of the boundary the Device resumes. Then
// no victim by segments is left, and a second such image's leg falls back to
// Y's oldest KV entry of each bucket.
class ShardedImageWriteBackTest : public WriteBackTransactionTest, public ::testing::WithParamInterface<std::int32_t> {
protected:
    ShardedImageWriteBackTest()
        : WriteBackTransactionTest(/*full_shard_count=*/GetParam(), /*num_snapshot_pages=*/32) {}
};

INSTANTIATE_TEST_SUITE_P(KvShards, ShardedImageWriteBackTest, ::testing::Values(2, 4));

TEST_P(ShardedImageWriteBackTest, AnImagesHostBlocksKeepTheirDeviceBlocksBuckets) {
    const std::int32_t shards = GetParam();
    std::vector<BlockTable> tables(4);
    const TailPublication x_tail = *PublishFinal(tables, MakeTokens(14));
    const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
    coordinator_.Free(tables);
    const TailPublication y_tail = *PublishFinal(tables, MakeTokens(14, 100));
    const std::vector<CacheKey> y = KeysOf(hashes_, y_tail);
    coordinator_.Free(tables);
    for (const std::vector<CacheKey>& session : {x, y}) {
        for (const CacheKey& key : session) {
            StoreOnHost(key);
        }
    }
    const CacheBlockRef reader = coordinator_.AcquireHostCachedBlock(Key(y_tail.hash, 1));
    ASSERT_TRUE(static_cast<bool>(reader));

    for (const bool by_fallback : {false, true}) {
        SCOPED_TRACE(by_fallback);
        HoldFreeCapacity(host_pool_, /*spare_parents=*/0);
        std::vector<BlockTable> image(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, image, shards * kP));
        std::vector<std::vector<std::int32_t>> image_tokens;
        for (std::int32_t page = 0; page < shards; ++page) {
            image_tokens.push_back(MakeTokens(kP, (by_fallback ? 950 : 900) + kP * page));
        }
        const std::vector<std::string> image_pages = ContentHashes(image_tokens);
        CacheCompletedBlocksForTest(coordinator_, image, image_pages,
                                    CacheCoordinatorTestAccess::NextAccessEpoch(coordinator_),
                                    /*first_new_prefix_page=*/0, /*num_computed_tokens=*/shards * kP,
                                    CacheBoundaryKind::kEndpoint, /*stream_completed_to_host=*/false,
                                    /*publish_state=*/false);
        TierTransferManager::RetractionImageSplit split =
            transfers_.SplitRetractionImage(image, shards * kP, coordinator_.PublishedDataSlots(image, shards * kP));
        std::vector<std::int32_t> image_buckets;
        for (const CacheBlockRef& block : image[0].Blocks()) {
            image_buckets.push_back(coordinator_.Allocator(0).BucketOf(block->Location()));
        }
        ASSERT_EQ(split.buckets, image_buckets);
        ASSERT_EQ(std::set<std::int32_t>(image_buckets.begin(), image_buckets.end()).size(),
                  static_cast<std::size_t>(shards))
            << "a page per bucket";
        ASSERT_EQ(split.num_host_boundaries, static_cast<std::size_t>(shards));
        ASSERT_FALSE(split.shortfall.has_value());
        const std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image,
            shards * kP, std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        const std::vector<CacheTransfer>& copies = stores->host_store->transfers;
        ASSERT_EQ(copies.size(), static_cast<std::size_t>(shards));
        for (std::size_t page = 0; page < copies.size(); ++page) {
            EXPECT_EQ(copies[page].content_hash, image_pages[page]);
            // A page's bucket is (page id - 1) % shard_count (BucketOf).
            EXPECT_EQ((copies[page].source_page - 1) % shards, (copies[page].destination_page - 1) % shards)
                << "page " << page;
        }
        transfers_.CompleteSnapshotStore(stores->snapshot_store.op_id);
        coordinator_.Free(image);
        EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kRetraction).snapshot_fallback, 0u);
        EXPECT_EQ(coordinator_.HostEvictions().duplicate.blocks, 7u) << "X's Host copy, whole";
        EXPECT_EQ(coordinator_.HostEvictions().lru.blocks + coordinator_.HostEvictions().duplicate_kv.blocks,
                  by_fallback ? static_cast<std::uint64_t>(shards) : 0u);
        EXPECT_EQ(HostEntries(x), 0);
        // Y's KV entries come first, one per slot of one parent: the first
        // `shards` are the oldest of each bucket.
        for (std::size_t i = 0; i < y.size(); ++i) {
            EXPECT_EQ(coordinator_.ContainsHostCachedBlock(y[i]), !by_fallback || i >= static_cast<std::size_t>(shards))
                << "Y's entry " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// The eviction write-back across plans. A plan builds -- its admissions and
// restores evict and record segments -- then drains the records into one op
// (StartEvictionStores) and collects the dead entries (CollectDeadEntries),
// as Scheduler::NextExecutionPlan does; the runtime acknowledges each op any
// number of plans later. Until its last ACK, a segment's stores hold its tree
// nodes (pending_host_stores); a later drain borrows the block of a key a
// store still carries instead of copying it again; and the settlement at the
// last ACK publishes the members, or cancels those no probe could resume
// from any more.
// ---------------------------------------------------------------------------

// ASegmentPublishesOnlyAtItsLastAck's two ops: Y's segment borrows X's four
// pages from X's op and copies its tail and checkpoints in an op of its own,
// so it waits for both, in either ACK order. Between the two ACKs a request
// over Y's prompt is retracted. Y's own keys are then landed but unpublished,
// or on Y's op still out; an image borrows only from a store whose ACK
// publishes, so it copies them once more, and X's pages too while X's op is
// out. The segment publishes at its last ACK; the image's ACK merges each
// copy into the entry the segment published (the index redirects the
// publication), and the image follows it. One Host entry per key; once the
// image is let go nothing is pinned or in flight, and the extra copies' Host
// blocks are free again.
TEST_F(ImageWriteBackTransactionTest, AnImageBetweenASegmentsTwoAcksCopiesItsKeysAgain) {
    for (const bool own_op_first : {true, false}) {
        SCOPED_TRACE(own_op_first);
        std::vector<BlockTable> tables(4);
        const TailPublication x_tail = *PublishFinal(tables, MakeTokens(18));
        const std::vector<CacheKey> x = KeysOf(hashes_, x_tail);
        coordinator_.Free(tables);
        const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
        ASSERT_TRUE(x_op.has_value());
        std::vector<std::int32_t> y_tokens = MakeTokens(18);
        y_tokens[16] = 9000;
        const TailPublication y_tail = *PublishFinal(tables, y_tokens);
        coordinator_.Free(tables);
        const std::vector<CacheKey> y_own = KeysOf({}, y_tail);
        const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
        ASSERT_TRUE(y_op.has_value());
        ASSERT_EQ(y_op->transfers.size(), 4u);
        EXPECT_EQ(transfers_.CompleteWriteBack(own_op_first ? y_op->op_id : x_op->op_id).size(),
                  own_op_first ? 0u : 8u);
        EXPECT_EQ(HostEntries(y_own), 0);
        const std::int32_t empty_parents = host_pool_.NumEmptyLcmBlocks();
        const std::int32_t free_kv_slots = host_pool_.NumFreeSlots(0);

        std::vector<BlockTable> image(4);
        PublishFinal(image, y_tokens);
        TierTransferManager::RetractionImageSplit split =
            transfers_.SplitRetractionImage(image, 18, coordinator_.PublishedDataSlots(image, 18));
        EXPECT_TRUE(split.pending_store_ops.empty()) << "nothing borrowed";
        ASSERT_EQ(split.slots.size(), own_op_first ? 8u : 4u);
        ASSERT_FALSE(split.shortfall.has_value());
        std::optional<TierTransferManager::RetractionStores> stores = transfers_.StartRetractionStores(
            "image", /*request_pool_index=*/1, std::make_shared<SnapshotSlotIndex>(blob_slots_.Allocate()), image, 18,
            std::move(split));
        ASSERT_TRUE(stores.has_value());
        ASSERT_TRUE(stores->host_store.has_value());
        coordinator_.Free(image);

        EXPECT_EQ(transfers_.CompleteWriteBack(own_op_first ? x_op->op_id : y_op->op_id).size(),
                  own_op_first ? 8u + 8u : 8u);
        EXPECT_EQ(HostEntries(x), 8);
        EXPECT_EQ(HostEntries(y_own), 4);
        EXPECT_EQ(stores->host_store->transfers.size(), own_op_first ? 8u : 4u);
        stores->image.FollowPublished(transfers_.CompleteWriteBack(stores->host_store->op_id));
        for (const ImageTable& table : stores->image.tables) {
            for (const ImageSlot& slot : table.slots) {
                if (slot.InHostCache()) {
                    EXPECT_EQ(slot.block, coordinator_.FindHostCachedBlock(slot.key));
                }
            }
        }
        transfers_.CompleteSnapshotStore(stores->snapshot_store.op_id);
        stores.reset();
        EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
        EXPECT_FALSE(transfers_.HasAnyInFlight());
        EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), empty_parents);
        EXPECT_EQ(host_pool_.NumFreeSlots(0), free_kv_slots);
        EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8 + 4);
        EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 0u);
        ASSERT_TRUE(coordinator_.ClearCache());
    }
}

// AnEvictionDrainBorrowsAnImagesTicket with an earlier eviction write-back's
// ticket. The same prompt is published on the Device again while its
// write-back is still out -- neither tier indexes it, so a later request
// computed it again -- and a later plan evicts it again. That plan's drain
// borrows every member from the op still out and issues no store of its own:
// the second segment waits for the first one's op, and the two hold the tree
// nodes together (pending_host_stores). The one ACK publishes both, in drain
// order, onto the same Host entries: each key crossed once, and nothing stays
// pinned or in flight.
TEST_F(WriteBackTransactionTest, ADrainBorrowingAnEarlierWriteBackIssuesNoStore) {
    const std::vector<std::int32_t> tokens = MakeTokens(18);
    std::vector<BlockTable> tables(4);
    const TailPublication tail = *PublishFinal(tables, tokens);
    const std::vector<std::string> pages = hashes_;
    const std::vector<CacheKey> keys = KeysOf(pages, tail);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> op = EvictOldestAndDrain();
    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->transfers.size(), 8u);
    PublishFinal(tables, tokens);
    coordinator_.Free(tables);

    EXPECT_FALSE(EvictOldestAndDrain().has_value()) << "every member borrowed";
    const TierTransferManager::WriteBackStats& stats = transfers_.WriteBackStatsFor(HostStoreCause::kEviction);
    EXPECT_EQ(stats.ops, 1u);
    EXPECT_EQ(stats.in_flight, 8u);
    EXPECT_EQ(stats.blocks, (std::vector<std::uint64_t>{5, 1, 1, 1}));
    for (const std::string& page : pages) {
        EXPECT_EQ(NodeOf(page).pending_host_stores, 2) << "a hold per segment";
    }
    EXPECT_EQ(NodeOf(tail.hash).pending_host_stores, 2 * 4) << "the tail block and its checkpoints, twice";
    EXPECT_EQ(HostEntries(keys), 0);

    EXPECT_EQ(transfers_.CompleteWriteBack(op->op_id).size(), 8u + 8u) << "the first segment, then the second";
    EXPECT_TRUE(transfers_.CompleteWriteBack(op->op_id).empty());
    EXPECT_EQ(HostEntries(keys), 8);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 8);
    EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), 32 - 5) << "two KV parents and three checkpoint parents";
    EXPECT_EQ(stats.cancelled, 0u);
    for (const std::string& hash : {pages[0], pages[1], pages[2], pages[3], tail.hash}) {
        EXPECT_EQ(NodeOf(hash).pending_host_stores, 0);
    }
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers_.HasAnyInFlight());
}

// ASegmentPublishesOnlyAtItsLastAck's two ops, Y's own landing first. Eight
// tails without a checkpoint then arrive under the last page, and the
// per-prior cap drops X's tail and Y's while both segments are out. Each
// segment is judged at its last ACK, as the Host then stands: the settlement
// cancels a member no probe could resume from. X's settles first: its tail
// and checkpoints are cancelled, and its pages publish, Y's checkpoints still
// on their way under them. Y's then cancels all eight of its members: its
// own copies, landed as they were, and the pages it borrowed, which X's
// settlement just published. Once the next collection has run, nothing is
// left of either session or of the eight tails, and nothing stays pinned or
// in flight.
TEST_F(WriteBackTransactionTest, TheCapCancelsASegmentBetweenItsTwoAcks) {
    std::vector<BlockTable> tables(4);
    PublishFinal(tables, MakeTokens(18));
    const std::vector<std::string> pages = hashes_;
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> x_op = EvictOldestAndDrain();
    ASSERT_TRUE(x_op.has_value());
    std::vector<std::int32_t> y_tokens = MakeTokens(18);
    y_tokens[16] = 9000;
    PublishFinal(tables, y_tokens);
    coordinator_.Free(tables);
    const std::optional<WriteBackOperation> y_op = EvictOldestAndDrain();
    ASSERT_TRUE(y_op.has_value());
    ASSERT_EQ(y_op->transfers.size(), 4u);
    EXPECT_TRUE(transfers_.CompleteWriteBack(y_op->op_id).empty()) << "Y waits for X's op";

    PublishEightTailsWithoutCheckpoints(pages);
    ASSERT_EQ(coordinator_.TailCapEvictions(), 2);
    transfers_.CompleteWriteBack(x_op->op_id);
    EXPECT_EQ(transfers_.WriteBackStatsFor(HostStoreCause::kEviction).cancelled, 4u + 8u);
    coordinator_.CollectDeadEntries();
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 0);
    EXPECT_EQ(host_pool_.NumEmptyLcmBlocks(), 32);
    EXPECT_EQ(Tree().Size(), 0u);
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers_.HasAnyInFlight());
}

// Restore-only plans: a plan's one piece of work is a restore, which takes no
// forward, so nothing in the plan waits for its write-back, and the ACK may
// come any number of plans later. Five retracted requests' images, in the
// snapshot pool, are restored in five plans; each restore finds three Device
// parents spare and evicts the oldest of five finished sessions for the rest,
// recording it as one segment; every plan drains and collects as the scheduler
// does, and each op's ACK arrives one to five plans after its drain, out of
// order, while later plans run. Until its ACK a session's stores alone hold
// its tree nodes (pending_host_stores): neither tier holds it, and the
// collection leaves it. Each ACK publishes its segment once, whole; a repeated
// one publishes nothing. Once every ACK is in, the Host holds the five
// sessions, no entry is pinned, nothing is in flight, and the cache clears.
TEST_F(ImageWriteBackTransactionTest, RestoreOnlyPlansPublishEachSegmentOnceAtItsLateAck) {
    std::vector<RetractionImage> images;
    for (std::int32_t request = 0; request < 5; ++request) {
        std::vector<BlockTable> tables(4);
        ASSERT_TRUE(AdmitForTest(coordinator_, tables, kP));
        std::optional<CacheCoordinator::ImageTaken> image = coordinator_.TakeImage(tables, kP, NoHostSlots(4));
        ASSERT_TRUE(image.has_value());
        image->store_pairs.clear();
        coordinator_.Free(tables);
        images.push_back(std::move(image->image));
    }
    std::vector<std::vector<CacheKey>> sessions;
    for (std::int32_t session = 0; session < 5; ++session) {
        std::vector<BlockTable> tables(4);
        const TailPublication tail = *PublishFinal(tables, MakeTokens(18, 100 * (session + 1)));
        sessions.push_back(KeysOf(hashes_, tail));
        coordinator_.Free(tables);
    }
    // Before which plan each restore's write-back is acknowledged: one to five
    // plans after the restore's own.
    const std::array<std::int32_t, 5> ack_plan{0 + 3, 1 + 1, 2 + 5, 3 + 2, 4 + 4};
    std::vector<WriteBackOperation> ops;
    for (std::int32_t plan = 0; plan <= 8; ++plan) {
        SCOPED_TRACE(plan);
        std::vector<BlockTable> restored(4);
        for (std::size_t session = 0; session < ops.size(); ++session) {
            if (ack_plan[session] == plan) {
                EXPECT_EQ(transfers_.CompleteWriteBack(ops[session].op_id).size(), 8u) << "session " << session;
                EXPECT_TRUE(transfers_.CompleteWriteBack(ops[session].op_id).empty()) << "session " << session;
            }
        }
        if (plan < 5) {
            HoldFreeCapacity(pool_, /*spare_parents=*/3);
            ASSERT_TRUE(RestoreDemands(coordinator_, images[static_cast<std::size_t>(plan)], restored,
                                       {GroupDemand{.extent = DenseGrowth{0}, .reserve_tokens = 1},
                                        GroupDemand{.extent = StateStep{StateStep::Kind::kKeep}},
                                        GroupDemand{.extent = StateStep{StateStep::Kind::kKeep}},
                                        GroupDemand{.extent = StateStep{StateStep::Kind::kKeep}}},
                                       /*access_epoch=*/1));
        }
        std::optional<WriteBackOperation> op = transfers_.StartEvictionStores();
        ASSERT_EQ(op.has_value(), plan < 5);
        if (op) {
            EXPECT_EQ(op->transfers.size(), 8u);
            ops.push_back(std::move(*op));
        }
        coordinator_.CollectDeadEntries();
        for (std::size_t session = 0; session < ops.size(); ++session) {
            const bool landed = ack_plan[session] <= plan;
            for (const CacheKey& key : sessions[session]) {
                const Node* node = Tree().Find(key.content_hash);
                ASSERT_NE(node, nullptr) << "session " << session;
                EXPECT_EQ(node->pending_host_stores > 0, !landed) << "session " << session;
                EXPECT_FALSE(node->on_device) << "session " << session;
                EXPECT_EQ(coordinator_.ContainsHostCachedBlock(key), landed) << "session " << session;
            }
        }
        coordinator_.Free(restored);
    }
    EXPECT_EQ(coordinator_.DeviceEvictions().restore.blocks, 5u * 8u);
    EXPECT_EQ(coordinator_.NumHostCachedBlocks(), 5 * 8);
    EXPECT_EQ(coordinator_.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers_.HasAnyInFlight());
    EXPECT_TRUE(coordinator_.ClearCache());
}

// Random K3 sessions -- forked from earlier ones at random pages, resumed on
// what either tier holds, with a checkpoint at a chunk end now and then --
// published and evicted under pressure on a Device of 64 Full blocks, written
// back to a Host of 48 (16 and 12 parents at four Full blocks per parent) by
// each round's drain, the ACKs landing zero to three rounds late, out of order
// and now and then twice, the dead entries collected every round, and now and
// then eight tails without a checkpoint capping a session's own. A next turn
// that resumed at a tail checkpoint on the Device marks it superseded once it
// publishes, as the P role's finish does, so marked entries leave in segments
// beside recorded ones. After every round a hash has a tree node exactly while
// either tier holds an entry of it, a store is bringing one, or a node hangs
// under it, and every page or tail the Device holds has the page it follows
// there; every eviction an admission recorded or dropped lands in exactly one
// outcome of the drain; and once every ACK is in, no node holds a store and no
// Host entry is pinned. The parameter deals the Full group's pages over that
// many KV-parallel shards, so every copy row keeps its two ends in one bucket;
// a parent holds at least one Full block per shard, and the pools keep their
// Full capacity.
class WriteBackRandomTest : public ::testing::TestWithParam<std::int32_t> {};

INSTANTIATE_TEST_SUITE_P(KvShards, WriteBackRandomTest, ::testing::Values(1, 2, 4, 8));

TEST_P(WriteBackRandomTest, TreeNodesLiveExactlyAsLongAsAnEntryOrAStore) {
    constexpr std::int32_t kP = 4;
    const std::int32_t packing = std::max(4, GetParam());
    BlockPool pool(/*num_lcm_blocks=*/64 / packing, {packing, 1, 1, 1});
    BlockPool host_pool(/*num_lcm_blocks=*/48 / packing, {packing, 1, 1, 1});
    CacheCoordinator coordinator = MakeCoordinator(
        std::vector<CacheGroupSpec>{
            {.attn_kind = AttnKind::kFull,
             .cache_blocks_per_lcm_block = packing,
             .block_granularity = kP,
             .shard_count = GetParam()},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            {.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}},
        kP, pool, /*enable_l3_storage=*/false, &host_pool, /*snapshot_pool=*/nullptr,
        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/true);
    TierTransferManager transfers{coordinator};
    std::mt19937 rng(20261029);
    std::vector<std::vector<std::int32_t>> sessions;
    // Every boundary ever published, with its tail tokens (0 for a page).
    std::vector<std::pair<std::string, std::int32_t>> boundaries;
    struct Ack {
        std::int32_t due;
        std::uint32_t op_id;
    };
    std::vector<Ack> acks;
    const auto land = [&](std::int32_t round, bool all) {
        std::vector<Ack> due;
        std::erase_if(acks, [&](const Ack& ack) {
            if (!all && ack.due > round) {
                return false;
            }
            due.push_back(ack);
            return true;
        });
        std::ranges::shuffle(due, rng);
        for (const Ack& ack : due) {
            transfers.CompleteWriteBack(ack.op_id);
            if (std::bernoulli_distribution(0.05)(rng)) {
                EXPECT_TRUE(transfers.CompleteWriteBack(ack.op_id).empty()) << "a repeated ACK";
            }
        }
    };
    const BoundaryTree& tree = CacheCoordinatorTestAccess::BoundaryTreeOf(coordinator);
    const auto check_nodes = [&](std::int32_t round) {
        for (const auto& [hash, tail_tokens] : boundaries) {
            bool held = false;
            coordinator.ForEachBoundaryKey(Key(hash, 0), tail_tokens, [&](std::size_t, const CacheKey& key) {
                held = held ||
                       coordinator.GroupPrefixIndex(static_cast<std::int32_t>(key.group_id)).Contains(pool, key) ||
                       coordinator.ContainsHostCachedBlock(key);
            });
            const BoundaryTree::Node* node = tree.Find(hash);
            EXPECT_EQ(node != nullptr,
                      held || (node != nullptr && (node->pending_host_stores > 0 || node->first_child != nullptr)))
                << "round " << round;
            if (node == nullptr) {
                EXPECT_FALSE(held) << "round " << round << ": an entry without a node";
            }
            if (node != nullptr && node->parent != nullptr &&
                coordinator.GroupPrefixIndex(0).Contains(pool, Key(hash, 0))) {
                EXPECT_TRUE(coordinator.GroupPrefixIndex(0).Contains(pool, Key(*node->parent->hash, 0)))
                    << "round " << round << ": a Device page or tail outlives the page it follows";
            }
        }
    };
    std::int32_t forks = 0;
    std::int32_t hits = 0;
    std::int32_t capped = 0;
    std::int32_t marked = 0;
    // Marked entries the admissions dropped (no request here reclaims its own
    // slots).
    std::uint64_t reclaimed_by_admissions = 0;
    for (std::int32_t round = 0; round < 300; ++round) {
        std::vector<std::int32_t> tokens =
            MakeTokens(std::uniform_int_distribution<std::int32_t>(2 * kP, 7 * kP - 1)(rng), 1000 * (round + 1));
        const double draw = std::uniform_real_distribution<double>(0, 1)(rng);
        if (!sessions.empty() && draw < 0.7) {
            // One of the last eight sessions, which the two tiers may still
            // hold.
            const std::vector<std::int32_t>& base =
                sessions[sessions.size() - 1 -
                         std::uniform_int_distribution<std::size_t>(
                             0, std::min<std::size_t>(sessions.size(), 8) - 1)(rng)];
            if (draw < 0.4 && base.size() < static_cast<std::size_t>(9 * kP)) {
                // The session's next turn: all of it, and a few tokens more.
                tokens.resize(static_cast<std::size_t>(std::uniform_int_distribution<std::int32_t>(1, 2 * kP)(rng)));
                tokens.insert(tokens.begin(), base.begin(), base.end());
            } else {
                // A fork at some point of it.
                const auto shared = std::min<std::size_t>(
                    base.size(), static_cast<std::size_t>(std::uniform_int_distribution<std::int32_t>(
                                     kP, static_cast<std::int32_t>(tokens.size()) - 1)(rng)));
                std::copy_n(base.begin(), shared, tokens.begin());
            }
            ++forks;
        }
        const auto num_tokens = static_cast<std::int32_t>(tokens.size());
        const std::int32_t full_pages = num_tokens / kP;
        std::vector<std::vector<std::int32_t>> pages;
        for (std::int32_t page = 0; page < full_pages; ++page) {
            pages.emplace_back(tokens.begin() + page * kP, tokens.begin() + (page + 1) * kP);
        }
        const std::vector<std::string> hashes = ContentHashes(pages);
        std::optional<TailPublication> tail;
        if (num_tokens % kP != 0) {
            tail = TailPublication{
                .tail_tokens = num_tokens % kP,
                .hash = HashPrefixPage(token_span(tokens).subspan(static_cast<std::size_t>(full_pages * kP)),
                                       hashes.back())};
        }
        CacheCoordinator::PrefixProbe probe =
            coordinator.ProbePromptPrefix({.prefix_hashes = std::span<const std::string>(hashes).first(
                                               static_cast<std::size_t>((num_tokens - 1) / kP)),
                                           .tokens = tokens,
                                           .max_tokens = num_tokens - 1});
        const std::int32_t hit = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
        // The Host suffix's full pages, which the Scheduler publishes on the
        // Device at admission (the Host keeps its copies).
        const std::int32_t device_full_pages = probe.device.full_page_tokens / kP;
        const std::int32_t extension_pages =
            std::max(probe.host.full_page_tokens - probe.device.full_page_tokens, 0) / kP;
        hits += hit > 0 ? 1 : 0;
        std::vector<BlockTable> tables(4);
        const std::uint64_t reclaimed_before = coordinator.DeviceEvictions().request_reclaim.blocks;
        const std::optional<CacheCoordinator::AdmissionResult> admitted =
            AdmitForTest(coordinator, tables, std::move(probe), GroupDemand{.extent = DenseGrowth{num_tokens - hit}});
        reclaimed_by_admissions += coordinator.DeviceEvictions().request_reclaim.blocks - reclaimed_before;
        if (admitted) {
            for (std::int32_t page = 0; page < extension_pages; ++page) {
                coordinator.CacheFullBlocks(
                    tables,
                    std::span<const std::string>(hashes).subspan(static_cast<std::size_t>(device_full_pages + page), 1),
                    admitted->access_epoch, device_full_pages + page, CacheBoundaryKind::kChunk);
            }
            coordinator.CacheCompletedBlocks(
                tables,
                RequestProgress{.completed_pages = CompletedPages{.prefix_hashes = hashes,
                                                                  .first_new_prefix_page = hit / kP,
                                                                  .tail = tail,
                                                                  .publish_state = true,
                                                                  .boundary_kind = CacheBoundaryKind::kEndpoint},
                                .num_computed_tokens = num_tokens},
                admitted->access_epoch);
            // As the P role's finish does under write-back: the end state
            // just published supersedes the tail checkpoint the session
            // resumed from on the Device.
            if (admitted->resumed_tail_hash) {
                coordinator.MarkSupersededStateInput(*admitted->resumed_tail_hash, admitted->access_epoch);
                ++marked;
            }
            for (const std::string& hash : hashes) {
                boundaries.emplace_back(hash, 0);
            }
            if (tail) {
                boundaries.emplace_back(tail->hash, tail->tail_tokens);
            }
            sessions.push_back(tokens);
        }
        coordinator.Free(tables);
        // Now and then eight tails without a checkpoint after the last full
        // page of this session: the per-prior cap drops the session's own.
        if (admitted && full_pages > 0 && std::bernoulli_distribution(0.1)(rng)) {
            ++capped;
            for (std::int32_t extra = 1; extra <= TailTable::kMaxTailsPerPrior; ++extra) {
                std::vector<std::int32_t> other(tokens.begin(), tokens.begin() + full_pages * kP);
                other.push_back(100000 * (round + 1) + extra);
                std::vector<BlockTable> other_tables(4);
                if (AdmitForTest(coordinator, other_tables, full_pages * kP + 1)) {
                    const TailPublication other_tail{
                        .tail_tokens = 1,
                        .hash = HashPrefixPage(token_span(other).subspan(static_cast<std::size_t>(full_pages * kP)),
                                               hashes.back())};
                    coordinator.CacheCompletedBlocks(
                        other_tables,
                        RequestProgress{
                            .completed_pages = CompletedPages{.prefix_hashes = hashes,
                                                              .first_new_prefix_page = 0,
                                                              .tail = other_tail,
                                                              .boundary_kind = CacheBoundaryKind::kEndpoint},
                            .num_computed_tokens = full_pages * kP + 1},
                        CacheCoordinatorTestAccess::NextAccessEpoch(coordinator));
                    boundaries.emplace_back(other_tail.hash, 1);
                }
                coordinator.Free(other_tables);
            }
        }
        // The plan's drain and collection; the ACKs of earlier plans land.
        if (const std::optional<WriteBackOperation> op = transfers.StartEvictionStores()) {
            for (const CacheTransfer& transfer : op->transfers) {
                ASSERT_GT(transfer.source_page, 0);
                // A page's bucket is (page id - 1) % shard_count (BucketOf).
                const std::int32_t shards =
                    coordinator.Allocator(static_cast<std::int32_t>(transfer.group_id)).ShardCount();
                EXPECT_EQ((transfer.source_page - 1) % shards, (transfer.destination_page - 1) % shards)
                    << "round " << round;
            }
            acks.push_back(
                Ack{.due = round + std::uniform_int_distribution<std::int32_t>(0, 3)(rng), .op_id = op->op_id});
        }
        coordinator.CollectDeadEntries();
        land(round, /*all=*/false);
        check_nodes(round);
    }
    land(/*round=*/0, /*all=*/true);
    coordinator.CollectDeadEntries();
    check_nodes(300);
    for (const auto& [hash, tail_tokens] : boundaries) {
        if (const BoundaryTree::Node* node = tree.Find(hash)) {
            EXPECT_EQ(node->pending_host_stores, 0);
            EXPECT_EQ(node->pending_host_checkpoint_stores, 0);
        }
    }
    EXPECT_EQ(coordinator.NumPinnedHostCachedBlocks(), 0);
    EXPECT_FALSE(transfers.HasAnyInFlight());

    // Every recorded or dropped eviction landed in one outcome. (A marked
    // entry evicted is counted as request reclaim, recorded nowhere.)
    const TierTransferManager::WriteBackStats& stats = transfers.WriteBackStatsFor(HostStoreCause::kEviction);
    const CacheCoordinator::DeviceEvictionStats& device_evictions = coordinator.DeviceEvictions();
    std::uint64_t outcomes =
        stats.host_cached + stats.in_flight + stats.no_placement + coordinator.DeadEntriesNotWrittenBack();
    std::uint64_t copied = 0;
    for (const std::uint64_t blocks : stats.blocks) {
        outcomes += blocks;
        copied += blocks;
    }
    EXPECT_EQ(outcomes, device_evictions.admission.blocks + device_evictions.stranded_suffix.blocks);
    // The run forked and continued sessions, resumed them, capped tails in
    // flight, and exercised cache pressure. Cancellation and re-copy counts
    // depend on replacement choices; the transaction cases check them exactly.
    EXPECT_GT(forks, 0);
    EXPECT_GT(hits, 0);
    EXPECT_GT(capped, 0);
    EXPECT_GT(marked, 0);
    EXPECT_GT(reclaimed_by_admissions, 0u) << "no marked checkpoint was evicted";
    EXPECT_GT(stats.ops, 0u);
    EXPECT_GT(stats.host_cached, 0u);
    EXPECT_GT(stats.in_flight, 0u);
    EXPECT_GT(stats.no_placement, 0u);
    EXPECT_LE(stats.cancelled, copied);
    EXPECT_LE(stats.recopied, copied);
    EXPECT_GT(coordinator.DeadEntriesNotWrittenBack(), 0u);
    // The collection counts under dead_entry too, so the Host plan's other
    // classes show it chose victims.
    const CacheCoordinator::HostEvictionStats& host_evictions = coordinator.HostEvictions();
    EXPECT_GT(host_evictions.duplicate.blocks + host_evictions.superseded.blocks + host_evictions.lru.blocks +
                  host_evictions.duplicate_kv.blocks,
              0u)
        << "the Host plan chose victims";
    EXPECT_GT(device_evictions.dead_entry_collection.blocks, 0u);
}

TEST(StoragePrefixReservationTest, DistributesRestoredPagesAcrossCyclicOwners) {
    BlockPool pool(4, {2});
    BlockPool host_pool(4, {2});
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 2, .block_granularity = 2, .shard_count = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/true, &host_pool, /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/true, /*write_back_evictions=*/false);
    const auto tokens = MakeTokens(7);
    const auto hashes = ComputePrefixHashes(
        std::array{token_span(tokens.data(), 2), token_span(tokens.data() + 2, 2), token_span(tokens.data() + 4, 2)},
        "");
    const CacheCoordinator::PromptPrefix prompt{
        .prefix_hashes = hashes,
        .tokens = tokens,
        .max_tokens = 7,
        .tail =
            TailPublication{.tail_tokens = 1, .hash = HashPrefixPage(token_span(tokens.data() + 6, 1), hashes.back())}};
    const auto query = coordinator.MissingStorageKeys(prompt, 16);
    ASSERT_EQ(query.keys.size(), 4u);
    const auto reservation = coordinator.ReserveStoragePrefix(prompt, query.keys, 4);
    ASSERT_EQ(reservation.blocks.size(), 4u);
    ASSERT_TRUE(reservation.tail);
    EXPECT_EQ(reservation.keys.back().content_hash, prompt.tail->hash);
    for (std::size_t i = 0; i < reservation.blocks.size(); ++i) {
        EXPECT_EQ(coordinator.Allocator(0).BucketOf(reservation.blocks[i]->Location()),
                  static_cast<std::int32_t>(i % 2));
    }
}

TEST(StoragePrefixReservationTest, QueryBudgetIncludesPromptEndStateBeforeHistoricalCheckpoints) {
    constexpr std::int32_t kGrain = 128;
    BlockPool pool(128, std::vector<std::int32_t>(8, 1));
    BlockPool host_pool(128, std::vector<std::int32_t>(8, 1));
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = kGrain}};
    for (int i = 0; i < 7; ++i) {
        specs.push_back(
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0});
    }
    auto coordinator = MakeCoordinator(specs, kGrain, pool, /*enable_l3_storage=*/true, &host_pool,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/true,
                                       /*write_back_evictions=*/false);
    const auto tokens = MakeTokens(4097);
    std::vector<token_span> pages;
    for (std::size_t begin = 0; begin + kGrain <= tokens.size(); begin += kGrain) {
        pages.emplace_back(tokens.data() + begin, kGrain);
    }
    const auto hashes = ComputePrefixHashes(pages, "");
    const TailPublication tail{.tail_tokens = 1,
                               .hash = HashPrefixPage(token_span(tokens.data() + 4096, 1), hashes.back())};
    const CacheCoordinator::PromptPrefix prompt{
        .prefix_hashes = hashes, .tokens = tokens, .max_tokens = 4097, .tail = tail};
    const auto query = coordinator.MissingStorageKeys(prompt, 256);
    EXPECT_EQ(query.max_tokens, 4097);
    EXPECT_LE(query.keys.size(), 256u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 128);
    EXPECT_EQ(host_pool.NumEmptyLcmBlocks(), 128);

    // A writer publishes State at its natural prompt end, not at every
    // historical Full boundary. The complete 4097 hit needs only 40 objects.
    std::vector<CacheKey> hits;
    for (const auto& key : query.keys) {
        if (key.group_id == 0 || key.content_hash == tail.hash) hits.push_back(key);
    }
    const auto reservation = coordinator.ReserveStoragePrefix(prompt, hits, 128);
    ASSERT_TRUE(reservation.tail);
    EXPECT_EQ(reservation.tail->hash, tail.hash);
    EXPECT_EQ(reservation.keys.size(), 40u);
    EXPECT_EQ(reservation.blocks.size(), 40u);
}

}  // namespace tokenspeed::test
