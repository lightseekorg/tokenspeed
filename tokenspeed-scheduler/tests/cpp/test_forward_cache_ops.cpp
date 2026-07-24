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
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/forward_cache_ops.h"
#include "cache/kv_cache_coordinator.h"
#include "resource/allocator/paged_cache_group.h"
#include "scheduler/page_hasher.h"
#include "scheduler/types.h"

namespace tokenspeed::test {
namespace {

KvCacheCoordinator MakeTwoGroup(BlockPool& pool) {
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/4},
    };
    return MakeCoordinator(specs, pool);
}

TEST(ForwardCacheOpsFree, ReturnsAllPagesToPool) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    const std::int32_t free_before = pool.NumFreeBlocks();

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));
    ASSERT_LT(pool.NumFreeBlocks(), free_before);

    FreeRequest(coordinator, tables);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(ForwardCacheOpsPrefill, FirstChunkAcquiresPagesForTokens) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // Zero hit (default-constructed match: miss path) -> Acquire 4 tokens -> 2 pages/group.
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, CoordinatorMatch{}, /*num_new_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(ForwardCacheOpsPrefill, FirstChunkClaimsHitThenAcquiresOnlyRemainder) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    // W=16: the SWA bounded match needs ceil((16-1)/2) = 8 > 4 contiguous pages,
    // so all 4 prefix pages stay real hits and nothing slides out of window.
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/16},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);

    // r1: 8 tokens -> 4 pages/group; freed blocks keep their hashes (prefix-hittable).
    std::vector<std::string> hashes8(4);
    for (std::size_t i = 0; i < hashes8.size(); ++i) {
        hashes8[i] = std::string(64, static_cast<char>('a' + i));
    }
    std::vector<BlockTable> r1(coordinator.NumGroups());
    ASSERT_TRUE(PrefillFirstChunk(coordinator, r1, CoordinatorMatch{}, /*num_new_tokens=*/8));
    coordinator.CacheFullBlocks(r1, hashes8);
    const std::vector<std::int32_t> r1_full_ids = BlockTablePageIds(r1[0]);
    const std::vector<std::int32_t> r1_swa_ids = BlockTablePageIds(r1[1]);
    FreeRequest(coordinator, r1);

    // r2: same 8-token prefix, 12-token prefill target -> 4 NEW tokens.
    CoordinatorMatch hit = coordinator.AcquirePrefix(coordinator.ProbePrefix(hashes8)).device;
    ASSERT_EQ(hit.num_common_tokens, 8);
    ASSERT_EQ(hit.per_group[1].num_hit_blocks, 4) << "W=16 must keep every SWA prefix page real";

    // Claimed pages carry no tail credit (spec §4.3): tail_avail_ stays 0, so
    // Four new tokens need ceil(4/2) = 2 pages in each group.
    {
        std::vector<BlockTable> probe(coordinator.NumGroups());
        CoordinatorMatch probe_hit = hit;
        coordinator.ClaimCommonPrefix(probe, std::move(probe_hit));
        EXPECT_EQ(probe[0].TailAvailableTokens(), 0);
        EXPECT_EQ(probe[1].TailAvailableTokens(), 0);
        EXPECT_EQ(coordinator.BlocksNeededByPool(probe, /*num_tokens=*/4), (PoolDemand{4}));
        FreeRequest(coordinator, probe);
    }

    const std::int32_t free_before = pool.NumFreeBlocks();
    std::vector<BlockTable> r2(coordinator.NumGroups());
    ASSERT_TRUE(PrefillFirstChunk(coordinator, r2, std::move(hit), /*num_new_tokens=*/4));

    // Per-group table: 4 claimed prefix pages + ceil(4 new / 2) = 2 fresh = 6.
    ASSERT_EQ(r2[0].NumBlocks(), 6);
    ASSERT_EQ(r2[1].NumBlocks(), 6);

    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(r2[0].Blocks()[i]->BlockId(), r1_full_ids[i]) << "full slot " << i;
        EXPECT_EQ(r2[1].Blocks()[i]->BlockId(), r1_swa_ids[i]) << "swa slot " << i;
    }

    // Match already pinned the 4 prefix pages/group before free_before; claim
    // only transfers those refs. This operation allocates 2 new pages/group.
    EXPECT_EQ(free_before - pool.NumFreeBlocks(), 4);
}

TEST(ForwardCacheOpsPrefill, ChunkAcquiresAndCachesFullBlocks) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, CoordinatorMatch{}, /*num_new_tokens=*/4));

    // Second chunk: 4 more tokens -> +2 pages/group.
    // num_computed = 4 -> skipped = 4-4+1 = 1 -> 1/2 = 0 pages slid out yet.
    std::vector<std::string> hashes2{std::string(64, 'a'), std::string(64, 'b')};
    ASSERT_TRUE(PrefillChunk(coordinator, tables, hashes2, /*num_tokens=*/4, /*num_computed_tokens=*/4));
    EXPECT_EQ(tables[0].NumBlocks(), 4);
    EXPECT_EQ(tables[1].NumBlocks(), 4);
    for (const BlockRef& block : tables[1].Blocks()) {
        EXPECT_TRUE(block) << "num_computed=4, W=4: no page is fully out of window yet";
    }
}

// Register-before-punch: CacheFullBlocks skips holes, so punched pages' hashes
// must be registered before the slide.
TEST(ForwardCacheOpsPrefill, ChunkSlidesSwaWindowAndKeepsPunchedPageHashes) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // Chunk 0: 8 tokens -> 4 pages/group (chunk >> window is fine: the slide
    // happens on the NEXT op; SWA footprint contract in forward_cache_ops.h).
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, CoordinatorMatch{}, /*num_new_tokens=*/8));
    const std::int32_t free_before_chunk = pool.NumFreeBlocks();

    // Chunk 1: num_computed = 8 -> skipped = 8-4+1 = 5 -> 5/2 = 2 pages fully
    // out of window: SWA slots 0,1 punched, then 2 fresh pages acquired.
    std::vector<std::string> hashes{std::string(64, 'a'), std::string(64, 'b'), std::string(64, 'c'),
                                    std::string(64, 'd')};
    ASSERT_TRUE(PrefillChunk(coordinator, tables, hashes, /*num_tokens=*/4, /*num_computed_tokens=*/8));

    EXPECT_EQ(tables[0].NumBlocks(), 6);
    for (const BlockRef& block : tables[0].Blocks()) {
        EXPECT_TRUE(block);
    }
    ASSERT_EQ(tables[1].NumBlocks(), 6);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[1]);
    for (std::int32_t i = 2; i < 6; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }

    // Pool: the slide freed 2 SWA pages, the acquire took 2/group = 4 -> net -2.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before_chunk + 2 - 4);

    for (const std::string& h : {hashes[0], hashes[1]}) {
        EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(h, /*group_id=*/1)))
            << "slid-out page must keep its registered hash";
    }
}

// The first decode step (query at position P) only reads keys back to P - W + 1.
TEST(ForwardCacheOpsPrefill, FinalizeSlidesSwaWindowBeforeReserveAcquire) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // page=2, W=4
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 12-token prefill -> 6 pages/group, tails full.
    ASSERT_TRUE(PrefillFirstChunk(coordinator, tables, CoordinatorMatch{}, /*num_new_tokens=*/12));
    const std::int32_t free_before = pool.NumFreeBlocks();

    // num_computed = 12 -> skipped = 12-4+1 = 9 -> 9/2 = 4 pages punched
    // (slots 0..3); reserve 1 token -> 1 fresh page/group.
    std::vector<std::string> hashes(6, "");
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    ASSERT_TRUE(FinalizePrefillAndReserveDecode(coordinator, tables, hashes, /*reserve_tokens=*/1,
                                                /*num_computed_tokens=*/12));

    ASSERT_EQ(tables[1].NumBlocks(), 7);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_FALSE(tables[1].Blocks()[i]) << "slot " << i;
    }
    for (std::int32_t i = 4; i < 7; ++i) {
        EXPECT_TRUE(tables[1].Blocks()[i]) << "slot " << i;
    }
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    // Pool: slide freed 4, reserve acquire took 1/group = 2 -> net +2.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 4 - 2);
}

TEST(ForwardCacheOpsDecode, StepAcquiresAndSlidesSwaWindow) {
    BlockPool pool(/*total_num_blocks=*/64, /*enable_caching=*/true);
    KvCacheCoordinator coordinator = MakeTwoGroup(pool);  // swa window=4, block_size=2
    std::vector<BlockTable> tables(coordinator.NumGroups());

    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));  // 3 pages/group

    for (std::int32_t computed = 7; computed <= 13; ++computed) {
        ASSERT_TRUE(DecodeStep(coordinator, tables, /*content_hashes=*/{}, /*first_page_slot=*/0,
                               /*num_tokens=*/1, /*num_computed_tokens=*/computed));
    }
    // 13 tokens -> ceil(13/2) = 7 pages.
    EXPECT_EQ(tables[0].NumBlocks(), 7);
    std::int32_t full_nulls = 0;
    for (const BlockRef& block : tables[0].Blocks()) {
        if (!block) ++full_nulls;
    }
    EXPECT_EQ(full_nulls, 0);
    std::int32_t swa_active = 0;
    for (const BlockRef& block : tables[1].Blocks()) {
        if (block) ++swa_active;
    }
    EXPECT_LE(swa_active, 3);
}

TEST(ForwardCacheOpsDecode, DecodeStepRegistersFilledPages) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());

    // 8 tokens -> 4 full pages; pages 0-1 registered at prefill time.
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/8));
    std::vector<std::string> hashes(4);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        hashes[i] = std::string(64, static_cast<char>('a' + i));
    }
    coordinator.CacheFullBlocks(tables, std::span<const std::string>(hashes).first(2));
    ASSERT_EQ(coordinator.AcquirePrefix(coordinator.ProbePrefix(hashes)).device.num_common_tokens, 4);

    const std::vector<std::string> fresh(hashes.begin() + 2, hashes.end());
    ASSERT_TRUE(DecodeStep(coordinator, tables, fresh, /*first_page_slot=*/2,
                           /*num_tokens=*/1, /*num_computed_tokens=*/8));

    // Registration maps slots to this request's physical pages, not copies.
    const CoordinatorMatch hit = coordinator.AcquirePrefix(coordinator.ProbePrefix(hashes)).device;
    EXPECT_EQ(hit.num_common_tokens, 8);
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(hit.per_group[0].blocks[i]->BlockId(), tables[0].Blocks()[i]->BlockId()) << "slot " << i;
    }
}

TEST(ForwardCacheOpsDecode, DecodeStepWithEmptyHashesUnchanged) {
    BlockPool pool_new(/*total_num_blocks=*/32, /*enable_caching=*/true);
    BlockPool pool_old(/*total_num_blocks=*/32, /*enable_caching=*/true);
    KvCacheCoordinator coordinator_new = MakeTwoGroup(pool_new);  // page=2, W=4
    KvCacheCoordinator coordinator_old = MakeTwoGroup(pool_old);
    std::vector<BlockTable> tables_new(coordinator_new.NumGroups());
    std::vector<BlockTable> tables_old(coordinator_old.NumGroups());
    ASSERT_TRUE(coordinator_new.Acquire(tables_new, /*num_tokens=*/8));  // 4 pages/group
    ASSERT_TRUE(coordinator_old.Acquire(tables_old, /*num_tokens=*/8));

    ASSERT_TRUE(DecodeStep(coordinator_new, tables_new, /*content_hashes=*/{}, /*first_page_slot=*/0,
                           /*num_tokens=*/1, /*num_computed_tokens=*/8));
    for (std::int32_t g = 0; g < coordinator_old.NumGroups(); ++g) {
        coordinator_old.GroupManager(g).ReclaimExpired(pool_old, tables_old[static_cast<std::size_t>(g)],
                                                       /*num_computed_tokens=*/8);
    }
    ASSERT_TRUE(coordinator_old.Acquire(tables_old, /*num_tokens=*/1));

    EXPECT_EQ(pool_new.NumFreeBlocks(), pool_old.NumFreeBlocks());
    ASSERT_EQ(tables_new.size(), tables_old.size());
    for (std::size_t g = 0; g < tables_new.size(); ++g) {
        EXPECT_EQ(BlockTablePageIds(tables_new[g]), BlockTablePageIds(tables_old[g])) << "group " << g;
        EXPECT_EQ(tables_new[g].TailAvailableTokens(), tables_old[g].TailAvailableTokens()) << "group " << g;
    }
}

TEST(ForwardCacheOpsSpecs, TranslatesPagedCacheGroups) {
    SchedulerConfig config;
    config.block_size = 16;
    PagedCacheGroupConfig full_grp;
    full_grp.group_id = "full";
    full_grp.rows_per_page = 8;
    full_grp.entry_stride_tokens = 2;
    full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupConfig swa_grp;
    swa_grp.group_id = "swa";
    swa_grp.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
    swa_grp.sliding_window_tokens = 128;
    config.paged_cache_groups = {full_grp, swa_grp};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[0].block_size, 16);
    EXPECT_EQ(specs[0].sliding_window, 0);
    EXPECT_EQ(specs[0].group_id, "full");
    EXPECT_EQ(specs[0].rows_per_page, 8);
    EXPECT_EQ(specs[0].entry_stride_tokens, 2);
    EXPECT_EQ(specs[1].kind, AttnKind::kSlidingWindow);
    EXPECT_EQ(specs[1].block_size, 16);
    EXPECT_EQ(specs[1].sliding_window, 128);
    EXPECT_EQ(specs[1].group_id, "swa");
    EXPECT_EQ(specs[1].rows_per_page, 16);
    EXPECT_EQ(specs[1].entry_stride_tokens, 1);
}

TEST(ForwardCacheOpsSpecs, StateFamilyMapsToMambaStateKind) {
    SchedulerConfig config;
    config.block_size = 4;
    PagedCacheGroupConfig full_grp;
    full_grp.group_id = "full_attention";
    full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
    PagedCacheGroupConfig state_grp;
    state_grp.group_id = "linear_attention";
    state_grp.family = PagedCacheGroupFamily::State;
    config.paged_cache_groups = {full_grp, state_grp};

    std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config);
    ASSERT_EQ(specs.size(), 2u);
    EXPECT_EQ(specs[0].kind, AttnKind::kFull);
    EXPECT_EQ(specs[1].kind, AttnKind::kMambaState);
    EXPECT_EQ(specs[1].block_size, 4);
    EXPECT_EQ(specs[1].sliding_window, 0);
}

TEST(ForwardCacheOpsSpecs, CanonicalFlatConfigMapsPrefixRolesByCacheMode) {
    struct Case {
        const char* name;
        bool disable_prefix_cache;
        KvPrefixRole expected_prefix_role;
        bool coordinator_is_valid;
    };
    const Case cases[] = {
        {"prefix enabled", false, KvPrefixRole::kContinuationState, false},
        {"prefix disabled", true, KvPrefixRole::kNone, true},
    };

    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        SchedulerConfig config;
        config.block_size = 8;
        config.disable_prefix_cache = test_case.disable_prefix_cache;
        config.flat_block_pools = {
            FlatBlockPoolConfig{.pool_id = "v4.indexer.state", .total_blocks = 9, .bytes_per_block = 64},
        };
        PagedCacheGroupConfig state;
        state.group_id = "v4.c4a.indexer_compressor_state";
        state.rows_per_page = 4;
        state.entry_stride_tokens = 1;
        state.total_pages = 9;
        state.block_size = 4;
        state.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        state.sliding_window_tokens = 8;
        state.family = PagedCacheGroupFamily::State;
        state.pool_id = "v4.indexer.state";
        state.prefix_role = PrefixRole::ContinuationState;
        state.table_layout = TableLayout::BoundedWindow;
        state.owner_mask = 0b001;
        config.paged_cache_groups = {state};

        BlockPoolSet pools(MakeFlatBlockPoolConfigs(config));
        const std::vector<KvCacheSpec> specs = MakeSpecsFromConfig(config, pools);
        ASSERT_EQ(specs.size(), 1u);
        EXPECT_EQ(specs[0].prefix_role, test_case.expected_prefix_role);

        if (!test_case.disable_prefix_cache) {
            EXPECT_EQ(specs[0].kind, AttnKind::kSlidingWindow);
            EXPECT_EQ(specs[0].block_size, 4);
            EXPECT_EQ(specs[0].sliding_window, 8);
            EXPECT_EQ(specs[0].pool_index, pools.IndexOf("v4.indexer.state"));
            EXPECT_EQ(specs[0].table_layout, KvTableLayout::kBoundedWindow);
            EXPECT_EQ(specs[0].owner_mask, 0b001u);
        }
        if (test_case.coordinator_is_valid) {
            EXPECT_NO_THROW(MakeCoordinator(specs, pools));
        } else {
            EXPECT_THROW(MakeCoordinator(specs, pools), std::invalid_argument);
        }
    }
}

}  // namespace
}  // namespace tokenspeed::test
