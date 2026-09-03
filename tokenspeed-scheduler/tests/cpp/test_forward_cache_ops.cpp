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
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache_test_access.h"
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
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 4,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    return MakeCoordinator(specs, 2, pool);
}

TEST(ForwardCacheOpsFree, ReturnsAllPagesToPool) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));
    ASSERT_LT(pool.NumEmptyLcmBlocks(), free_before);

    FreeRequest(coordinator, tables);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
}

TEST(AlignPrefillChunkTest, StopsAtPromotionBoundary) {
    EXPECT_EQ(AlignPrefillChunk(/*first_pos=*/16, /*unscheduled=*/24, /*token_budget=*/24,
                                /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/32),
              16);
}

TEST(AlignPrefillChunkTest, KeepsFuturePromotionWhenBudgetFallsShort) {
    EXPECT_EQ(AlignPrefillChunk(/*first_pos=*/16, /*unscheduled=*/24, /*token_budget=*/8,
                                /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(AlignPrefillChunkTest, LaterChunkStopsAtPromotionBoundary) {
    EXPECT_EQ(AlignPrefillChunk(/*first_pos=*/24, /*unscheduled=*/16, /*token_budget=*/16,
                                /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(AlignPrefillChunkTest, EndpointBeforePromotionWins) {
    EXPECT_EQ(AlignPrefillChunk(/*first_pos=*/24, /*unscheduled=*/4, /*token_budget=*/16,
                                /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/32),
              4);
}

TEST(AlignPrefillChunkTest, ReachedPromotionUsesOrdinaryPageAlignment) {
    EXPECT_EQ(AlignPrefillChunk(/*first_pos=*/32, /*unscheduled=*/16, /*token_budget=*/10,
                                /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/32),
              8);
}

TEST(FinalAlignedTailTokensTest, FindsSubPageTailAfterAlignedBody) {
    const std::optional<std::int32_t> tail =
        FinalAlignedTailTokens(/*first_pos=*/16, /*unscheduled=*/11, /*token_budget=*/16,
                               /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/0);
    EXPECT_EQ(tail, 3);
}

TEST(FinalAlignedTailTokensTest, LeavesAStandaloneSubPageWhole) {
    EXPECT_EQ(FinalAlignedTailTokens(/*first_pos=*/24, /*unscheduled=*/3, /*token_budget=*/16,
                                     /*prefix_granularity=*/4, /*promotion_boundary_tokens=*/0),
              std::nullopt);
}

TEST(ForwardCacheOpsPrefill, FirstChunkAcquiresPagesForTokens) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(ForwardCacheOpsPrefill, FirstChunkClaimsHitThenAcquiresOnlyRemainder) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    // W=16: the SWA bounded match needs ceil((16-1)/2) = 8 > 4 contiguous pages,
    // so all 4 prefix pages stay real hits and nothing slides out of window.
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
        CacheGroupSpec{.kind = AttnKind::kSlidingWindow,
                       .sliding_window = 16,
                       .cache_blocks_per_lcm_block = 1,
                       .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);

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
    ASSERT_TRUE(AdmitForTest(coordinator, r2, std::move(prefix), GroupDemand{.num_tokens = 4}));

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
    BlockPool pool(/*num_lcm_blocks=*/32);
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    // Second chunk: 4 more tokens -> +2 pages/group.
    // num_computed = 4 -> skipped = 4-4+1 = 1 -> 1/2 = 0 pages slid out yet.
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 4,
                                 .prefix_hashes = hashes2,
                                 .completed_boundary_kind = CacheBoundaryKind::kChunk,
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
    BlockPool pool(/*num_lcm_blocks=*/32);
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
                                 .num_tokens = 4,
                                 .prefix_hashes = hashes,
                                 .completed_boundary_kind = CacheBoundaryKind::kChunk,
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
    BlockPool pool(/*num_lcm_blocks=*/32);
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
                                 .num_tokens = 1,
                                 .prefix_hashes = hashes,
                                 .completed_boundary_kind = CacheBoundaryKind::kChunk,
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
    BlockPool pool(/*num_lcm_blocks=*/64);
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // swa window=4, prefix_granularity=2
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/6));  // 3 pages/group

    for (std::int32_t computed = 7; computed <= 13; ++computed) {
        ASSERT_TRUE(AdmitForTest(coordinator, tables,
                                 GroupDemand{
                                     .num_tokens = 1,
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
    BlockPool pool(/*num_lcm_blocks=*/32);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
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
                                 .num_tokens = 1,
                                 .prefix_hashes = hashes,
                                 .new_prefix_hash_begin = 2,
                                 .completed_boundary_kind = CacheBoundaryKind::kChunk,
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
    BlockPool pool(/*num_lcm_blocks=*/32);
    CacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/8));  // 4 pages/group
    const std::int32_t free_before = pool.NumEmptyLcmBlocks();

    ASSERT_TRUE(AdmitForTest(coordinator, tables,
                             GroupDemand{
                                 .num_tokens = 1,
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
    full_grp.rows_per_page = 16;
    full_grp.entry_stride_tokens = 1;
    full_grp.retention = CacheGroupConfig::Retention::FullHistory;
    CacheGroupConfig swa_grp;
    swa_grp.group_id = "swa";
    swa_grp.rows_per_page = 16;
    swa_grp.entry_stride_tokens = 1;
    swa_grp.retention = CacheGroupConfig::Retention::SlidingWindow;
    swa_grp.sliding_window_tokens = 128;
    config.cache_groups = {full_grp, swa_grp};

    std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[0].sliding_window, 0);
    EXPECT_EQ(specs[0].cache_blocks_per_lcm_block, 1);
    EXPECT_EQ(specs[1].kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].sliding_window, 128);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, StateFamilyMapsToMambaStateKind) {
    SchedulerConfig config;
    config.prefix_granularity = 4;
    CacheGroupConfig full_grp;
    full_grp.group_id = "full_attention";
    full_grp.rows_per_page = 4;
    full_grp.entry_stride_tokens = 1;
    full_grp.retention = CacheGroupConfig::Retention::FullHistory;
    CacheGroupConfig state_grp;
    state_grp.group_id = "linear_attention";
    state_grp.rows_per_page = 4;
    state_grp.entry_stride_tokens = 1;
    state_grp.family = CacheGroupFamily::State;
    config.cache_groups = {full_grp, state_grp};

    std::vector<CacheGroupSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[1].kind, AttnKind::kMambaState);
    EXPECT_EQ(specs[1].sliding_window, 0);
    EXPECT_EQ(specs[1].cache_blocks_per_lcm_block, 1);
}

TEST(MakeSpecsFromConfigTest, Qwen35Fp8UsesOneLogicalPAndPerGroupPacking) {
    SchedulerConfig config;
    config.prefix_granularity = 128;
    CacheGroupConfig full;
    full.group_id = "full";
    full.rows_per_page = 128;
    full.entry_stride_tokens = 1;
    full.cache_blocks_per_lcm_block = 16;
    CacheGroupConfig state0;
    state0.group_id = "state0";
    state0.rows_per_page = 128;
    state0.entry_stride_tokens = 1;
    state0.family = CacheGroupFamily::State;
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
    history.rows_per_page = 64;
    history.entry_stride_tokens = 4;
    CacheGroupConfig state;
    state.group_id = "compressor_state";
    state.family = CacheGroupFamily::State;
    state.rows_per_page = 4;
    state.entry_stride_tokens = 1;
    config.cache_groups = {history, state};

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
    group.rows_per_page = 128;
    group.entry_stride_tokens = 1;
    group.total_pages = config.device_allocator.total_pages;
    config.cache_groups = {group};
    return config;
}

// Every message naming a group must identify which one: a model mixes several.
void ExpectRejectedNamingGroup(const SchedulerConfig& config, const std::string& group_id) {
    try {
        config.Validate();
        FAIL() << "invalid config was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find(group_id), std::string::npos) << error.what();
    }
}

TEST(SchedulerConfigValidateTest, AcceptsAValidConfig) {
    EXPECT_NO_THROW(MakeValidConfig().Validate());
}

TEST(SchedulerConfigValidateTest, RejectsNonPositiveGlobalP) {
    SchedulerConfig config = MakeValidConfig();
    for (const std::int32_t prefix_granularity : {0, -1}) {
        config.prefix_granularity = prefix_granularity;
        EXPECT_THROW(config.Validate(), std::invalid_argument);
    }
}

TEST(SchedulerConfigValidateTest, RejectsNonPositivePerGroupPacking) {
    SchedulerConfig config = MakeValidConfig();
    for (const std::int32_t packing : {0, -1}) {
        config.cache_groups[0].cache_blocks_per_lcm_block = packing;
        ExpectRejectedNamingGroup(config, config.cache_groups[0].group_id);
    }
}

TEST(SchedulerConfigValidateTest, RejectsBlockGranularityThatDoesNotDivideP) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].rows_per_page = 48;
    ExpectRejectedNamingGroup(config, config.cache_groups[0].group_id);
}

TEST(SchedulerConfigValidateTest, RejectsMissingSlidingWindowWithGroupId) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].group_id = "missing_window";
    config.cache_groups[0].retention = CacheGroupConfig::Retention::SlidingWindow;
    ExpectRejectedNamingGroup(config, "missing_window");
}

TEST(SchedulerConfigValidateTest, RejectsNonPositiveSlidingWindowWithGroupId) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].group_id = "nonpositive_window";
    config.cache_groups[0].retention = CacheGroupConfig::Retention::SlidingWindow;
    for (const std::int32_t window : {0, -1}) {
        config.cache_groups[0].sliding_window_tokens = window;
        ExpectRejectedNamingGroup(config, "nonpositive_window");
    }
}

TEST(SchedulerConfigValidateTest, RejectsSnapshotStateGroupBelowOneCacheBlock) {
    SchedulerConfig config = MakeValidConfig();
    config.cache_groups[0].family = CacheGroupFamily::State;
    config.max_scheduled_tokens = config.prefix_granularity - 1;
    EXPECT_THROW(config.Validate(), std::invalid_argument);

    config.max_scheduled_tokens = config.prefix_granularity;
    EXPECT_NO_THROW(config.Validate());
}

TEST(SchedulerConfigValidateTest, RejectsPdTransferPolicyMismatch) {
    SchedulerConfig config = MakeValidConfig();
    // A P or D role IS the cache-transfer PD protocol, so it demands an
    // explicit transfer_policy; the fused role (MakeValidConfig's default)
    // never does.
    config.role = Role::kD;
    CacheGroupConfig& history = config.cache_groups[0];

    // A History group ships its full suffix; leaving the policy to the default
    // or claiming a snapshot layout are both rejected.
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::LatestSnapshot;
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::FullSuffix;
    EXPECT_NO_THROW(config.Validate());

    // A snapshot-state group is the mirror image.
    history.family = CacheGroupFamily::State;
    ExpectRejectedNamingGroup(config, history.group_id);
    history.transfer_policy = CacheTransferPolicy::LatestSnapshot;
    EXPECT_NO_THROW(config.Validate());
}

TEST(SchedulerConfigValidateTest, CacheGroupConfigRejectsNonPositivePacking) {
    CacheGroupConfig group;
    group.group_id = "full";
    group.rows_per_page = 1;
    group.entry_stride_tokens = 1;
    group.total_pages = 2;

    group.cache_blocks_per_lcm_block = 0;
    EXPECT_THROW(group.Validate(), std::invalid_argument);

    group.cache_blocks_per_lcm_block = -1;
    EXPECT_THROW(group.Validate(), std::invalid_argument);
}

TEST(ForwardCacheOpsBuildBlockTables, TwoGroupsRowsAndIds) {
    BlockPool pool(/*num_lcm_blocks=*/32);
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
    BlockPool pool(/*num_lcm_blocks=*/32);
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
    BlockPool pool(/*num_lcm_blocks=*/32);
    CacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    std::vector<std::string> group_ids{"full", "swa"};
    auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(built.size(), 2u);
    EXPECT_TRUE(built.at("full").empty());
    EXPECT_TRUE(built.at("swa").empty());
}

TEST(ForwardCacheOpsBuildBlockTables, SingleGroupRowMatchesSource) {
    BlockPool pool(/*num_lcm_blocks=*/32);
    std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{
            .kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, 2, pool);
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
    BlockPool pool(/*num_lcm_blocks=*/32);
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
    BlockPool pool(/*num_lcm_blocks=*/4);
    const std::vector<CacheGroupSpec> specs{
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool);
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
    BlockPool pool(/*num_lcm_blocks=*/16);
    const std::vector<CacheGroupSpec> specs{
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 2, .block_granularity = 2},
        {.kind = AttnKind::kFull, .sliding_window = 0, .cache_blocks_per_lcm_block = 1, .block_granularity = 2},
    };
    CacheCoordinator coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(AdmitForTest(coordinator, tables, /*num_tokens=*/4));

    const std::vector<std::string> group_ids{"packed", "single"};
    const auto built = BuildBlockTables(coordinator, tables, group_ids);

    ASSERT_EQ(tables[0].NumBlocks(), 2);
    ASSERT_EQ(tables[1].NumBlocks(), 2);
    EXPECT_EQ(built.at("packed"), (std::vector<std::int32_t>{
                                      coordinator.Allocator(0).ResolveCacheBlockId(tables[0].Blocks()[0]->Location()),
                                      coordinator.Allocator(0).ResolveCacheBlockId(tables[0].Blocks()[1]->Location()),
                                  }));
    EXPECT_EQ(built.at("single"), (std::vector<std::int32_t>{
                                      coordinator.Allocator(1).ResolveCacheBlockId(tables[1].Blocks()[0]->Location()),
                                      coordinator.Allocator(1).ResolveCacheBlockId(tables[1].Blocks()[1]->Location()),
                                  }));
}

}  // namespace
}  // namespace tokenspeed::test
