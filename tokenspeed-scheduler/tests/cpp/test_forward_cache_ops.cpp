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
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache_test_access.h"
#include "integration_test_helper.h"
#include "scheduler/operations/cache.h"
#include "cache/prefix/prefix_hasher.h"
#include "scheduler/types.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasLegacyGranularityField =
    requires(T value) { value.block_size; } || requires(T value) { value.cache_block_tokens; };

static_assert(!HasLegacyGranularityField<CacheGroupSpec>);

CacheCoordinator MakeTwoGroup(BlockPool& pool) {
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    return MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                           /*snapshot_pool=*/nullptr,
                           /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
}

TEST(ForwardCacheOpsFree, ReturnsAllPagesToPool) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    ASSERT_LT(pool.NumEmptyLcmBlocks(), free_before);

    FreeRequest(coordinator, tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(ForwardCacheOpsPrefill, FirstChunkAcquiresPagesForTokens) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(ForwardCacheOpsPrefill, FirstChunkClaimsHitThenAcquiresOnlyRemainder) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    // W=16: the SWA bounded match needs ceil((16-1)/2) = 8 > 4 contiguous pages,
    // so all 4 prefix pages stay real hits and nothing slides out of window.
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                       .sliding_window = 16,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    // r1: 8 tokens -> 4 pages/group; freed blocks keep their hashes (prefix-hittable).
    std::vector<std::string> hashes8(4);
    for (std::size_t i = 0; i < hashes8.size(); ++i) {
        hashes8[i] = std::string(64, static_cast<char>('a' + i));
    }
    std::vector<BlockTable> r1(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, r1, /*num_tokens=*/8));
    CacheFullBlocksForTest(coordinator, r1, hashes8);
    const std::vector<std::int32_t> r1_full_ids = BlockTableLcmBlockIds(r1[0]);
    const std::vector<std::int32_t> r1_swa_ids = BlockTableLcmBlockIds(r1[1]);
    FreeRequest(coordinator, r1);

    // r2: same 8-token prefix, 12-token prefill target -> 4 NEW tokens.
    {
        const CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(hashes8);
        ASSERT_EQ(prefix.device.num_common_tokens, 8);
        ASSERT_EQ(std::ranges::count(prefix.device.per_group[1].hits, std::uint8_t{1}), 4)
            << "W=16 must keep every SWA prefix page real";
    }

    // Claimed pages carry no available capacity: the next allocation starts a fresh block.
    {
        std::vector<BlockTable> probe(coordinator.NumGroups());
        ASSERT_TRUE(AdmitForTest(coordinator, probe, coordinator.ProbePrefix(hashes8), GroupDemand{}));
        EXPECT_EQ(probe[0].AvailableTokens(), 0);
        EXPECT_EQ(probe[1].AvailableTokens(), 0);
        EXPECT_EQ(coordinator.GroupBlocksNeededFor(0, probe[0], /*num_tokens=*/4), 2);
        EXPECT_EQ(coordinator.GroupBlocksNeededFor(1, probe[1], /*num_tokens=*/4), 2);
        FreeRequest(coordinator, probe);
    }

    const std::int32_t free_before = pool.NumEmptyLcmBlocks();
    std::vector<BlockTable> r2(coordinator.NumGroups());
    CacheCoordinator::PrefixProbe prefix = coordinator.ProbePrefix(hashes8);
    ASSERT_TRUE(AdmitForTest(coordinator, r2, std::move(prefix), GroupDemand{.extent = DenseGrowth{4}}));

    // Per-group table: 4 claimed prefix pages + ceil(4 new / 2) = 2 fresh = 6.
    ASSERT_EQ(r2[0].NumBlocks(), 6);
    ASSERT_EQ(r2[1].NumBlocks(), 6);

    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(r2[0].Blocks()[i]->Location().lcm_block_id, r1_full_ids[i]) << "full slot " << i;
        EXPECT_EQ(r2[1].Blocks()[i]->Location().lcm_block_id, r1_swa_ids[i]) << "swa slot " << i;
    }

    // Match already pinned the 4 prefix pages/group before free_before; claim
    // only transfers those refs. This operation allocates 2 new pages/group.
    EXPECT_EQ(free_before - pool.NumEmptyLcmBlocks(), 4);
}

TEST(ForwardCacheOpsPrefill, ChunkAcquiresAndCachesFullBlocks) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    // Second chunk: 4 more tokens -> +2 pages/group.
    // num_computed = 4 -> skipped = 4-4+1 = 1 -> 1/2 = 0 pages slid out yet.
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{4},
                             },
                             RequestProgress{
                                 .completed_pages =
                                     CompletedPages{
                                         .prefix_hashes = hashes2,
                                         .first_new_prefix_page = 0,
                                         .boundary_kind = CacheBoundaryKind::kChunk,
                                     },
                                 .num_computed_tokens = 4,
                             }));
    EXPECT_EQ(tables[0].NumBlocks(), 4);
    EXPECT_EQ(tables[1].NumBlocks(), 4);
    for (const CacheBlockRef& block : tables[1].Blocks()) {
        EXPECT_TRUE(block) << "num_computed=4, W=4: no page is fully out of window yet";
    }
}

// Register-before-punch: CacheFullBlocks skips holes, so punched pages' hashes
// must be registered before the slide.
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowAndKeepsPunchedPageHashes) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // Chunk 0: 8 tokens -> 4 pages/group (chunk >> window is fine: the slide
    // happens on the next admission.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    const std::int32_t free_before_chunk = pool.NumEmptyLcmBlocks();

    // Chunk 1: num_computed = 8 -> skipped = 8-4+1 = 5 -> 5/2 = 2 pages fully
    // out of window: SWA slots 0,1 punched, then 2 fresh pages acquired.
    std::vector<std::string> hashes{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'),
                                    std::string(64, 'd')};
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{4},
                             },
                             RequestProgress{
                                 .completed_pages =
                                     CompletedPages{
                                         .prefix_hashes = hashes,
                                         .first_new_prefix_page = 0,
                                         .boundary_kind = CacheBoundaryKind::kChunk,
                                     },
                                 .num_computed_tokens = 8,
                             }));

    EXPECT_EQ(tables[0].NumBlocks(), 6);
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t i = 2; i < 6; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }

    // Only the two-page resume tail is cached. The older two SWA pages become
    // free before four fresh pages are acquired, for a net cost of two.
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before_chunk - 2);

    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, CacheKey{.group_id = 1, .content_hash = hashes[0]}));
    EXPECT_FALSE(coordinator.GroupPrefixIndex(1).Contains(pool, CacheKey{.group_id = 1, .content_hash = hashes[1]}));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(1).Contains(pool, CacheKey{.group_id = 1, .content_hash = hashes[2]}));
    EXPECT_TRUE(coordinator.GroupPrefixIndex(1).Contains(pool, CacheKey{.group_id = 1, .content_hash = hashes[3]}));
}

// The first decode step (query at position P) only reads keys back to P - W + 1.
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowBeforeAcquire) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 12-token prefill -> 6 pages/group, tails full.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/12));
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    // num_computed = 12 -> skipped = 12-4+1 = 9 -> 9/2 = 4 pages punched
    // (slots 0..3); reserve 1 token -> 1 fresh page/group.
    std::vector<std::string> hashes(6, "");
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .completed_pages =
                                     CompletedPages{
                                         .prefix_hashes = hashes,
                                         .first_new_prefix_page = 0,
                                         .boundary_kind = CacheBoundaryKind::kChunk,
                                     },
                                 .num_computed_tokens = 12,
                             }));

    ASSERT_EQ(tables[1].NumBlocks(), 7);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(tables[1].Blocks()[i]) << "slot " << i;
    }
    for (std::int32_t i = 4; i < 7; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    // Pool: the uncached SWA prefix frees four parents, then one page per group
    // consumes two parents.
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before + 2);
}

TEST(ForwardCacheOpsDecode, StepAcquiresAndSlidesSwaWindow) {
    BlockPool pool(/*num_lcm_blocks=*/64, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // swa window=4, prefix_granularity=2
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));  // 3 pages/group

    for (std::int32_t computed = 7; computed <= 13; ++computed) {
        ASSERT_TRUE(AdmitForTest(coordinator, tables,
                                 GroupDemand{
                                     .extent = DenseGrowth{1},
                                 },
                                 RequestProgress{
                                     .num_computed_tokens = computed,
                                 }));
    }
    // 13 tokens -> ceil(13/2) = 7 pages.
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    std::int32_t full_nulls = 0;
    for (const CacheBlockRef& block : tables[0].Blocks()) {
        if (!block) ++full_nulls;
    }
    EXPECT_EQ(full_nulls, 0);
    std::int32_t swa_active = 0;
    for (const CacheBlockRef& block : tables[1].Blocks()) {
        if (block) ++swa_active;
    }
    EXPECT_LE(swa_active, 3);
}

TEST(ForwardCacheOpsDecode, DecodeStepRegistersFilledPages) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 8 tokens -> 4 full pages; pages 0-1 registered at prefill time.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));
    std::vector<std::string> hashes(4);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    CacheFullBlocksForTest(coordinator, tables, std::span<const std::string>(hashes).first(2));
    ASSERT_EQ(MatchPrefixForTest(coordinator, hashes).device.num_common_tokens, 4);

    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .completed_pages =
                                     CompletedPages{
                                         .prefix_hashes = hashes,
                                         .first_new_prefix_page = 2,
                                         .boundary_kind = CacheBoundaryKind::kChunk,
                                     },
                                 .num_computed_tokens = 8,
                             }));

    // Registration maps slots to this request's physical pages, not copies.
    const CoordinatorMatch hit = MatchPrefixForTest(coordinator, hashes).device;
    EXPECT_EQ(hit.num_common_tokens, 8);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(hit.per_group[0].blocks[i]->Location().lcm_block_id, tables[0].Blocks()[i]->Location().lcm_block_id)
            << "slot " << i;
    }
}

TEST(ForwardCacheOpsDecode, AdmissionWithEmptyHashesOnlySlidesAndAllocates) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));  // 4 pages/group
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .extent = DenseGrowth{1},
                             },
                             RequestProgress{
                                 .num_computed_tokens = 8,
                             }));

    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
    EXPECT_EQ(tables[0].NumBlocks(), 5);
    EXPECT_EQ(tables[1].NumBlocks(), 5);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t g = 0; g < coordinator.NumGroups(); ++g) {
        EXPECT_EQ(coordinator.GroupPrefixIndex(g).NumEntries(pool), 0);
        EXPECT_EQ(tables[static_cast<std::size_t>(g)].AvailableTokens(), 1);
    }
}

TEST(MakeSpecsFromConfigTest, TranslatesCacheGroups) {
    SchedulerConfig config;
    config.prefix_granularity = 16;
    CacheGroupConfig full_grp;
    full_grp.group_id = "full";
    full_grp.block_granularity = 16;
    full_grp.attn_kind = AttnKind::kFull;
    CacheGroupConfig swa_grp;
    swa_grp.group_id = "swa";
    swa_grp.block_granularity = 16;
    swa_grp.attn_kind = AttnKind::kSlidingWindow;
    swa_grp.sliding_window_tokens = 128;
    config.cache_groups = {full_grp, swa_grp};

    std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].attn_kind, AttnKind::kFull);
    EXPECT_EQ(specs[0].sliding_window, 0);
    EXPECT_EQ(specs[0].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[1].attn_kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].sliding_window, 128);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, StateGroupCarriesNoSpan) {
    SchedulerConfig config;
    config.prefix_granularity = 4;
    CacheGroupConfig full_grp;
    full_grp.group_id = "full_attention";
    full_grp.block_granularity = 4;
    full_grp.attn_kind = AttnKind::kFull;
    CacheGroupConfig state_grp;
    state_grp.group_id = "linear_attention";
    state_grp.block_granularity = 0;
    state_grp.attn_kind = AttnKind::kState;
    config.cache_groups = {full_grp, state_grp};

    std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].attn_kind, AttnKind::kFull);
    EXPECT_EQ(specs[1].attn_kind, AttnKind::kState);
    EXPECT_EQ(specs[1].sliding_window, 0);
    EXPECT_EQ(specs[1].block_granularity, 0) << "a State group has no token span";
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, Qwen35Fp8UsesOneLogicalPAndPerGroupPacking) {
    SchedulerConfig config;
    config.prefix_granularity = 128;
    CacheGroupConfig full;
    full.group_id = "full";
    full.block_granularity = 128;
    full.cache_blocks_per_lcm_block = 16;
    CacheGroupConfig state0;
    state0.group_id = "state0";
    state0.block_granularity = 0;
    state0.attn_kind = AttnKind::kState;
    CacheGroupConfig state1 = state0;
    state1.group_id = "state1";
    CacheGroupConfig state2 = state0;
    state2.group_id = "state2";
    config.cache_groups = {full, state0, state1, state2};

    std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);

    ASSERT_EQ(specs.size(), 4u);
    EXPECT_EQ(specs[0].cache_blocks_per_lcm_block, 16);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[2].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[3].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, PreservesPerGroupCachePageTokens) {
    SchedulerConfig config;
    config.prefix_granularity = 256;
    CacheGroupConfig history;
    history.group_id = "history";
    history.block_granularity = 256;
    CacheGroupConfig fine;
    fine.group_id = "compressor";
    fine.block_granularity = 4;
    config.cache_groups = {history, fine};

    const std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);

    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].block_granularity, 256);
    EXPECT_EQ(specs[1].block_granularity, 4);
}

// A configuration SchedulerConfig::Validate() accepts, so that each rejection
// test below can perturb exactly one field.
SchedulerConfig MakeValidConfig() {
    SchedulerConfig config;
    config.prefix_granularity = 128;
    config.device_allocator.total_pages = 32;
    config.max_scheduled_tokens = 1024;
    config.max_batch_size = 8;
    CacheGroupConfig group;
    group.group_id = "full";
    group.block_granularity = 128;
    group.total_pages = config.device_allocator.total_pages;
    config.cache_groups = {group};
    SetTestSnapshotPool(config);
    return config;
}

// Every message naming a group must identify which one: a model mixes several.
void ExpectRejectedNamingGroup(const SchedulerConfig& config, const std::string& group_id) {
    try {
        WithHostPackingFromDevice(config).Validate();
        FAIL() << "invalid config was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find(group_id), std::string::npos) << error.what();
    }
}

TEST(SchedulerConfigValidateTest, AcceptsAValidConfig) {
    EXPECT_NO_THROW(WithHostPackingFromDevice(MakeValidConfig()).Validate());
}

TEST(SchedulerConfigValidateTest, HostPackingIsExplicitAndMayDifferFromDevicePacking) {
    auto config = MakeValidConfig();
    EXPECT_THROW(config.Validate(), std::invalid_argument);
    config.host_group_packing = {3};
    EXPECT_NO_THROW(config.Validate());
    config.host_group_packing = {0};
    EXPECT_THROW(config.Validate(), std::invalid_argument);
    config.host_group_packing = {3, 1};
    EXPECT_THROW(config.Validate(), std::invalid_argument);
    config.cache_groups.front().shard_count = 2;
    config.cache_groups.front().cache_blocks_per_lcm_block = 2;
    config.host_group_packing = {3};
    EXPECT_THROW(config.Validate(), std::invalid_argument);
    config.host_group_packing = {6};
    EXPECT_NO_THROW(config.Validate());
}

TEST(SchedulerConfigValidateTest, RejectsNonPositiveGlobalP) {
    SchedulerConfig config = MakeValidConfig();
    for (const std::int32_t prefix_granularity : {0, -1}) {
        config.prefix_granularity = prefix_granularity;
        EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument);
    }
}

TEST(SchedulerConfigValidateTest, RejectsNonPositivePerGroupPacking) {
    SchedulerConfig config = MakeValidConfig();
    for (const std::int32_t packing : {0, -1}) {
        config.cache_groups[0].cache_blocks_per_lcm_block = packing;
        ExpectRejectedNamingGroup(config, config.cache_groups[0].group_id);
    }
}

TEST(SchedulerConfigValidateTest, RejectsNonPositiveBlockGranularity) {
    SchedulerConfig config = MakeValidConfig();
    for (const std::int32_t block_granularity : {0, -1}) {
        config.cache_groups[0].block_granularity = block_granularity;
        ExpectRejectedNamingGroup(config, config.cache_groups[0].group_id);
    }
}

TEST(SchedulerConfigValidateTest, RejectsBlockGranularityThatDoesNotDivideP) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].block_granularity = 48;
    ExpectRejectedNamingGroup(config, config.cache_groups[0].group_id);
}

TEST(SchedulerConfigValidateTest, RejectsMissingSlidingWindowWithGroupId) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].group_id = "missing_window";
    config.cache_groups[0].attn_kind = AttnKind::kSlidingWindow;
    ExpectRejectedNamingGroup(config, "missing_window");
}

TEST(SchedulerConfigValidateTest, RejectsNonPositiveSlidingWindowWithGroupId) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].group_id = "nonpositive_window";
    config.cache_groups[0].attn_kind = AttnKind::kSlidingWindow;
    for (const std::int32_t window : {0, -1}) {
        config.cache_groups[0].sliding_window_tokens = window;
        ExpectRejectedNamingGroup(config, "nonpositive_window");
    }
}

TEST(SchedulerConfigValidateTest, RejectsWindowOnNonSlidingGroupWithGroupId) {
    // Only a sliding-window group slides rows out, so only it carries a
    // window: a State group holds checkpoints and a full group every row.
    SchedulerConfig config = MakeValidConfig();
    CacheGroupConfig& group = config.cache_groups[0];
    group.group_id = "windowed";
    group.sliding_window_tokens = 256;
    group.attn_kind = AttnKind::kState;
    group.block_granularity = 0;
    ExpectRejectedNamingGroup(config, "windowed");
    group.attn_kind = AttnKind::kFull;
    group.block_granularity = 128;
    ExpectRejectedNamingGroup(config, "windowed");

    group.attn_kind = AttnKind::kSlidingWindow;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
}

TEST(SchedulerConfigValidateTest, StateGroupHasNoTokenSpanAndNoChunkAlignmentRule) {
    // A State group's two slots hold whole states: it declares no span, and
    // the chunk budget owes it nothing (a chunk may be any length).
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].group_id = "state";
    config.cache_groups[0].attn_kind = AttnKind::kState;
    ExpectRejectedNamingGroup(config, "state");
    config.cache_groups[0].block_granularity = 0;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.max_scheduled_tokens = 1;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    // A full group still declares a positive span that divides P.
    config.cache_groups[0].attn_kind = AttnKind::kFull;
    ExpectRejectedNamingGroup(config, "state");
}

TEST(SchedulerConfigValidateTest, RejectsPdTransferPolicyMismatch) {
    SchedulerConfig config = MakeValidConfig();
    // A P or D role IS the cache-transfer PD protocol, so it demands an
    // explicit transfer_policy; the fused role (MakeValidConfig's default)
    // never does.
    config.role = Role::kD;
    CacheGroupConfig& history = config.cache_groups[0];

    // A full group ships its full suffix; leaving the policy to the default
    // or claiming a snapshot layout are both rejected.
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::LatestSnapshot;
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::FullSuffix;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());

    // A State group is the mirror image.
    history.attn_kind = AttnKind::kState;
    history.block_granularity = 0;
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::LatestSnapshot;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
}

TEST(SchedulerConfigValidateTest, ReplayableRequiresASlidingWindowGroup) {
    SchedulerConfig config = MakeValidConfig();
    CacheGroupConfig& group = config.cache_groups[0];
    group.group_id = "swa";
    group.attn_kind = AttnKind::kSlidingWindow;
    group.sliding_window_tokens = 256;
    group.replayable = true;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());

    // Only a sliding-window group can be regenerated: a full group is
    // prefix-closed and shared, and a State group cannot slide at all.
    SchedulerConfig full = MakeValidConfig();
    full.cache_groups[0].replayable = true;
    ExpectRejectedNamingGroup(full, full.cache_groups[0].group_id);
    group.attn_kind = AttnKind::kState;
    group.sliding_window_tokens.reset();
    group.block_granularity = 0;
    ExpectRejectedNamingGroup(config, "swa");
}

TEST(SchedulerConfigValidateTest, ReplayNeedsBudgetForAWindowOnEveryRole) {
    SchedulerConfig config = MakeValidConfig();
    CacheGroupConfig swa;
    swa.group_id = "swa";
    swa.block_granularity = 64;
    swa.total_pages = config.device_allocator.total_pages;
    swa.attn_kind = AttnKind::kSlidingWindow;
    swa.sliding_window_tokens = 128;
    swa.replayable = true;
    config.cache_groups.push_back(swa);
    // P = 128 = W: the hit chunk needs W plus max(W, P) = 256 tokens of budget.
    config.max_scheduled_tokens = 255;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument)
        << "a hit chunk needs replay plus a window or page";
    config.max_scheduled_tokens = 256;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    // A window smaller than the prefix page still needs replay plus a page:
    // W = 2 leaves 6 of an 8-token budget, which no page-aligned chunk fits.
    SchedulerConfig small = config;
    small.cache_groups[1].sliding_window_tokens = 2;
    small.max_scheduled_tokens = 129;
    EXPECT_THROW(WithHostPackingFromDevice(small).Validate(), std::invalid_argument);
    small.max_scheduled_tokens = 130;
    EXPECT_NO_THROW(WithHostPackingFromDevice(small).Validate());

    // A State group beside a replayable one adds no rule of its own: chunks
    // are cut by budget and the replay window alone.
    SchedulerConfig with_state = config;
    CacheGroupConfig state;
    state.group_id = "state";
    state.block_granularity = 0;
    state.total_pages = config.device_allocator.total_pages;
    state.attn_kind = AttnKind::kState;
    with_state.cache_groups.push_back(state);
    EXPECT_NO_THROW(WithHostPackingFromDevice(with_state).Validate());

    // PD roles transfer a replayable group as any sliding window's retained
    // tail; the prefill role replays locally, the decode role never does.
    for (const Role role : {Role::kP, Role::kD}) {
        SchedulerConfig pd = config;
        pd.role = role;
        for (CacheGroupConfig& group : pd.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(pd);
        EXPECT_NO_THROW(WithHostPackingFromDevice(pd).Validate());
    }
}

TEST(SchedulerConfigValidateTest, StateCheckpointInForwardRequiresPrefixCaching) {
    // The checkpoint only pays off through a prefix hit on it; without prefix
    // caching it would hold a State block for nothing.
    SchedulerConfig config = MakeValidConfig();
    config.state_checkpoint_in_forward = true;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.disable_prefix_cache = true;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument);
    config.state_checkpoint_in_forward = false;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());

    // The D role computes no prompt, so it takes the flag without prefix
    // caching (PD launchers share one argument list between the roles); the
    // P role still refuses it.
    for (const Role role : {Role::kP, Role::kD}) {
        SchedulerConfig pd = MakeValidConfig();
        pd.role = role;
        for (CacheGroupConfig& group : pd.cache_groups) {
            group.transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(pd);
        pd.disable_prefix_cache = true;
        EXPECT_NO_THROW(WithHostPackingFromDevice(pd).Validate());
        pd.state_checkpoint_in_forward = true;
        if (role == Role::kD) {
            EXPECT_NO_THROW(WithHostPackingFromDevice(pd).Validate());
        } else {
            EXPECT_THROW(WithHostPackingFromDevice(pd).Validate(), std::invalid_argument);
        }
    }
}

TEST(SchedulerConfigValidateTest, SnapshotPoolAndRetractedSlotsAreStatedTogether) {
    // The pool is explicit on every role: 0 pages says nothing, the null page
    // alone says "never retract" (and takes no blob slots), anything above it
    // needs blob slots -- and the P role, which never retracts, takes neither.
    SchedulerConfig config = MakeValidConfig();
    config.snapshot_allocator.total_pages = 0;
    config.max_retracted_requests = 0;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument);
    config.snapshot_allocator.total_pages = 1;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.max_retracted_requests = 4;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument) << "slots without a pool";
    config.snapshot_allocator.total_pages = 16;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.max_retracted_requests = 0;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument) << "a pool without slots";
    config.max_retracted_requests = -1;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument);

    config.max_retracted_requests = 4;
    config.role = Role::kP;
    config.cache_groups[0].transfer_policy = CacheTransferPolicy::FullSuffix;
    EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument) << "the P role never retracts";
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
}

TEST(SchedulerConfigValidateTest, ForcedRetractionRequiresASnapshotPool) {
    // The knob retracts through the capacity path, which images the victim:
    // with the null page alone there is nowhere to put the image.
    SchedulerConfig config = MakeValidConfig();
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    for (const std::int32_t interval : {3, -3}) {
        config.debug_force_retraction_interval = interval;
        EXPECT_THROW(WithHostPackingFromDevice(config).Validate(), std::invalid_argument)
            << "interval " << interval << " without a pool";
    }
    config.snapshot_allocator.total_pages = 16;
    config.max_retracted_requests = 4;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.debug_force_retraction_interval = 0;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
}

TEST(SchedulerConfigValidateTest, RetractionDiagnosticsNameTheServerArgs) {
    // The runtime surfaces these messages verbatim at startup, so each one
    // names the server arg the operator has to change.
    SchedulerConfig config = MakeValidConfig();
    config.snapshot_allocator.total_pages = 16;
    config.max_retracted_requests = 0;
    try {
        WithHostPackingFromDevice(config).Validate();
        FAIL() << "a pool without slots must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("--retraction-snapshot-max-requests"), std::string::npos)
            << error.what();
        EXPECT_NE(std::string(error.what()).find("--retraction-snapshot-host-gb"), std::string::npos) << error.what();
    }
    config.snapshot_allocator.total_pages = 1;
    config.debug_force_retraction_interval = 2;
    try {
        WithHostPackingFromDevice(config).Validate();
        FAIL() << "the knob without a pool must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("--debug-force-retraction-interval"), std::string::npos)
            << error.what();
    }
}

TEST(SchedulerConfigValidateTest, L3StorageAcceptsCyclicOwnerGroups) {
    SchedulerConfig config = MakeValidConfig();
    config.host_allocator.total_pages = 32;
    config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    config.enable_l3_storage = true;
    config.storage_config = StorageConfig{
        8, 32, 4, 32, 16, 2, 100, 100, StoragePressure{8, 1048576, 1000}, StoragePressure{4, 524288, 500}};
    config.cache_groups[0].cache_blocks_per_lcm_block = 2;
    config.cache_groups[0].shard_count = 2;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    config.cache_groups[0].shard_count = 1;
    EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
}

TEST(SchedulerConfigValidateTest, AHostCacheRequiresAnExplicitWritePolicy) {
    // What feeds Host L2 is a cache policy, so a Host tier states it on every
    // role; without a Host tier (the null page alone, or disabled) there is
    // nothing to feed and the policy may stay unspecified.
    for (const Role role : {Role::kFused, Role::kP, Role::kD}) {
        SCOPED_TRACE(static_cast<int>(role));
        SchedulerConfig config = MakeValidConfig();
        config.role = role;
        if (role != Role::kFused) {
            config.cache_groups[0].transfer_policy = CacheTransferPolicy::FullSuffix;
        }
        SetTestSnapshotPool(config);
        config.host_allocator.total_pages = 1;
        EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
        config.host_allocator.total_pages = 2;
        ExpectRejectedNamingGroup(config, "explicit host_write_policy");
        config.disable_l2_cache = true;
        EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
        config.disable_l2_cache = false;
        config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        EXPECT_NO_THROW(WithHostPackingFromDevice(config).Validate());
    }
}

// Write-back runs on every role but D (which parallel layouts run it is the
// runtime's to decide): it needs a Host tier, takes a sharded group and the
// fused role's snapshot pool, and supports cached and replayable windows.
// It refuses the D role, L3 and more groups than the 64 bits that mark a
// hash's written-back groups.
TEST(SchedulerConfigValidateTest, WriteBackOnEveryRoleButTheDecodeRole) {
    const auto prefill = [] {
        SchedulerConfig config = MakeValidConfig();
        config.role = Role::kP;
        config.cache_groups[0].transfer_policy = CacheTransferPolicy::FullSuffix;
        SetTestSnapshotPool(config);
        config.host_allocator.total_pages = 2;
        config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        return config;
    };
    EXPECT_NO_THROW(WithHostPackingFromDevice(prefill()).Validate());
    SchedulerConfig no_host = prefill();
    no_host.host_allocator.total_pages = 1;
    ExpectRejectedNamingGroup(no_host, "requires a Host L2 cache");
    SchedulerConfig disabled_host = prefill();
    disabled_host.disable_l2_cache = true;
    ExpectRejectedNamingGroup(disabled_host, "requires a Host L2 cache");
    SchedulerConfig fused = prefill();
    fused.role = Role::kFused;
    SetTestSnapshotPool(fused);
    ASSERT_TRUE(fused.HasSnapshotPool());
    EXPECT_NO_THROW(WithHostPackingFromDevice(fused).Validate());
    EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(fused)});
    SchedulerConfig decode = prefill();
    decode.role = Role::kD;
    SetTestSnapshotPool(decode);
    ExpectRejectedNamingGroup(decode, "not available on the D role");
    SchedulerConfig l3 = prefill();
    l3.enable_l3_storage = true;
    l3.storage_config = StorageConfig{4, 16, 1, 16, 32, 2, 100, 100, {8, 1048576, 1000}, {4, 524288, 500}};
    ExpectRejectedNamingGroup(l3, "excludes L3 storage");
    SchedulerConfig sharded = prefill();
    sharded.cache_groups[0].cache_blocks_per_lcm_block = 2;
    sharded.cache_groups[0].shard_count = 2;
    EXPECT_NO_THROW(WithHostPackingFromDevice(sharded).Validate());
    EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(sharded)});

    // Both cached and replayable sliding windows share the write-back path.
    for (const bool replayable : {false, true}) {
        SCOPED_TRACE(replayable);
        SchedulerConfig sliding = prefill();
        CacheGroupConfig swa = sliding.cache_groups[0];
        swa.group_id = "swa";
        swa.attn_kind = AttnKind::kSlidingWindow;
        swa.sliding_window_tokens = 128;
        swa.replayable = replayable;
        sliding.cache_groups.push_back(swa);
        sliding.max_scheduled_tokens = 256;
        sliding.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
        EXPECT_NO_THROW(WithHostPackingFromDevice(sliding).Validate()) << "write-through takes it";
        sliding.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(sliding)});
    }

    // 64 groups construct a write-back scheduler; a 65th is refused before
    // any pool or the cache coordinator is built.
    SchedulerConfig wide = prefill();
    for (std::int32_t i = 1; i < 64; ++i) {
        CacheGroupConfig group = wide.cache_groups[0];
        group.group_id = "full" + std::to_string(i);
        wide.cache_groups.push_back(group);
    }
    EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(wide)});
    CacheGroupConfig last = wide.cache_groups[0];
    last.group_id = "full64";
    wide.cache_groups.push_back(last);
    ExpectRejectedNamingGroup(wide, "at most 64 cache groups");
    EXPECT_THROW(Scheduler{WithHostPackingFromDevice(wide)}, std::invalid_argument);
    wide.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteThrough;
    EXPECT_NO_THROW(WithHostPackingFromDevice(wide).Validate()) << "write-through keeps no bit per group";
}

TEST(SchedulerConfigValidateTest, CacheGroupConfigRejectsNonPositivePacking) {
    CacheGroupConfig group;
    group.group_id = "full";
    group.block_granularity = 1;
    group.total_pages = 2;

    group.cache_blocks_per_lcm_block = 0;
    EXPECT_THROW(group.Validate(), std::invalid_argument);

    group.cache_blocks_per_lcm_block = -1;
    EXPECT_THROW(group.Validate(), std::invalid_argument);
}

TEST(ForwardCacheOpsBuildBlockTables, TwoGroupsRowsAndIds) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    // 6 tokens, prefix_granularity 2 -> 3 pages per group.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    ASSERT_TRUE(built.count("full"));
    ASSERT_TRUE(built.count("swa"));
    EXPECT_EQ(built.at("full").size(), 3u);
    EXPECT_EQ(built.at("swa").size(), 3u);
    for (std::int32_t id : built.at("full")) {
        EXPECT_GT(id, 0);
    }
    // Rows match the source span verbatim: no compaction, null hole = 0 in its slot.
    const std::vector<std::int32_t> expected_full = coordinator.Allocator(0).BlockTablePageIds(tables[0]);
    const std::vector<std::int32_t> expected_swa = coordinator.Allocator(1).BlockTablePageIds(tables[1]);
    EXPECT_EQ(built.at("full"), expected_full);
    EXPECT_EQ(built.at("swa"), expected_swa);
}

TEST(ForwardCacheOpsBuildBlockTables, SwaRowGetsNullHoleAfterAdvance) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    // Window = 4 tokens = 2 pages, so 8 tokens leave earlier pages out of window.
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));  // 4 pages/group
    coordinator.ReclaimExpired(tables, /*num_computed_tokens=*/8);

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);
    for (std::int32_t id : built.at("full")) {
        EXPECT_GT(id, 0);
    }
    const auto& swa = built.at("swa");
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end());
    const std::vector<std::int32_t> expected_swa = coordinator.Allocator(1).BlockTablePageIds(tables[1]);
    EXPECT_EQ(swa, expected_swa);
}

TEST(ForwardCacheOpsBuildBlockTables, FreshTablesProduceEmptyRows) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built.at("full").empty());
    EXPECT_TRUE(built.at("swa").empty());
}

TEST(ForwardCacheOpsBuildBlockTables, SingleGroupRowMatchesSource) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1});
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, 2, pool, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                        /*snapshot_pool=*/nullptr,
                        /*stream_device_cache_to_host=*/false, /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));  // 2 pages

    std::vector<std::string> group_ids{"only"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(built.size(), 1u);
    const std::vector<std::int32_t> expected = coordinator.Allocator(0).BlockTablePageIds(tables[0]);
    EXPECT_EQ(built.at("only"), expected);
    // Sanity: keyed by the supplied group_id, not a bare index.
    EXPECT_EQ(built.count("0"), 0u);
}

TEST(ForwardCacheOpsBuildBlockTables, KeyMatchesSuppliedGroupIdStrings) {
    BlockPool pool(/*num_lcm_blocks=*/32, {1, 1});
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    std::vector<std::string> group_ids{"alpha", "beta"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built.count("alpha"));
    EXPECT_TRUE(built.count("beta"));
    const std::vector<std::int32_t> expected_alpha = coordinator.Allocator(0).BlockTablePageIds(tables[0]);
    EXPECT_EQ(built.at("alpha"), expected_alpha);
}

TEST(ForwardCacheOpsBuildBlockTables, ChildSlotsWithinOneParentHaveDistinctKernelPageIds) {
    BlockPool pool(/*num_lcm_blocks=*/4, {2});
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    const std::vector<std::int32_t> parent_ids = BlockTableLcmBlockIds(tables[0]);
    ASSERT_EQ(parent_ids.size(), 2u);
    EXPECT_EQ(parent_ids[0], parent_ids[1]);

    const auto built = BuildBlockTables(coordinator, tables, std::vector<std::string>{"full"});
    ASSERT_EQ(built.at("full").size(), 2u);
    EXPECT_EQ(built.at("full"), (std::vector<std::int32_t>{1, 2}));
    EXPECT_NE(built.at("full"), parent_ids);
}

TEST(ForwardCacheOpsBuildBlockTables, ResolvesEachGroupsPackingRecipe) {
    BlockPool pool(/*num_lcm_blocks=*/16, {2, 1});
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.attn_kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator =
        MakeCoordinator(specs, /*prefix_granularity=*/2, pool, /*enable_l3_storage=*/false,
                        /*host_pool=*/nullptr, /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                        /*write_back_evictions=*/false);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    const std::vector<std::string> group_ids{"packed", "single"};
    const auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_EQ(built.at("packed"),
              (std::vector<std::int32_t>{
                  coordinator.Allocator(0).ResolveDeviceCacheBlockId(tables[0].Blocks()[0]->Location()),
                  coordinator.Allocator(0).ResolveDeviceCacheBlockId(tables[0].Blocks()[1]->Location()),
              }));
    EXPECT_EQ(built.at("single"),
              (std::vector<std::int32_t>{
                  coordinator.Allocator(1).ResolveDeviceCacheBlockId(tables[1].Blocks()[0]->Location()),
                  coordinator.Allocator(1).ResolveDeviceCacheBlockId(tables[1].Blocks()[1]->Location()),
              }));
}

}  // namespace
}  // namespace tokenspeed::test
