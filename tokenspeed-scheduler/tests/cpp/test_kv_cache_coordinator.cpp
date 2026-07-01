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

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_group.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/cache_types.h"
#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

// One content-hash page (no group_id), via the no-group hasher.
std::vector<std::string> ContentHashes(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const auto& p : pages) {
        spans.emplace_back(p.data(), p.size());
    }
    return ComputePagedHashes(spans, "");
}

// Cache a content page under group_id's wrapped key, through the pool, then free
// it so MatchPrefix can hit it. Returns the physical block.
CacheBlock* CacheForGroup(BlockPool& pool, const std::string& content_hash, std::uint32_t group_id) {
    std::string key = MakeKeyWithGroupId(content_hash, group_id);
    std::vector<CacheBlock*> got = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(got.front(), key);
    pool.FreeBlocks(got);
    return got.front();
}

TEST(CacheGroupTest, HoldsSpecGroupIdManager) {
    BlockPool pool(8);
    auto mgr = std::make_unique<FullAttnManager>(pool, 4);
    CacheGroup g(KvCacheSpec{AttnKind::kFull, 4, 0}, /*group_id=*/0, std::move(mgr));
    EXPECT_EQ(g.GroupId(), 0u);
    EXPECT_EQ(g.Spec().page_size, 4);
    EXPECT_EQ(g.Spec().kind, AttnKind::kFull);
}

TEST(MakeCoordinatorTest, BuildsOneGroupPerSpec) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    EXPECT_EQ(coord.NumGroups(), 2);
}

TEST(MakeCoordinatorTest, RejectsMismatchedPageSize) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 8, 10},  // different page_size
    };
    EXPECT_THROW(MakeCoordinator(specs, pool), std::runtime_error);
}

TEST(CoordinatorMatchTest, BothGroupsAllMiss) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(CoordinatorMatchTest, CommonIsMinCoverageFullDeeperThanSwa) {
    // full caches 4 contiguous pages; swa (window 10 -> contiguous_needed 3)
    // caches only the last 3. Common = min(4, 3) = 3.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    // full (group 0): cache all 4 pages.
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    // swa (group 1): cache pages 0,1,2 -> a 3-run at the front; index 3 misses,
    // so the run sits at the front and coverage is exactly 3 (not 4: a tail run
    // would null-pad back to index 0 and yield coverage 4).
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 3);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 3u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 3u);
    // Full had 4 real hits, truncated to 3 -> num_hit recomputed to 3.
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
}

TEST(CoordinatorMatchTest, SwaMissForcesZeroCommon) {
    // full caches 2 pages, swa caches nothing -> common = min(2, 0) = 0.
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    EXPECT_EQ(m.per_group[0].blocks.size(), 0u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 0u);
}

TEST(CoordinatorAllocTest, ColdStartAllocatesAlignedPages) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CoordinatorMatch hit = coord.MatchPrefix(ch);
    EXPECT_EQ(hit.num_common_blocks, 0);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, hit);              // no hits -> no-op
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/8));
    // 8 tokens / page 4 = 2 pages in EACH group; tables aligned.
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, ClaimsCommonPrefixThenAllocatesRemainder) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    // swa window 4 -> contiguous_needed 1, so a single cached front page is a hit.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // Cache page 0 in BOTH groups so the common prefix is 1 page.
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);

    CoordinatorMatch hit = coord.MatchPrefix(ch);
    ASSERT_EQ(hit.num_common_blocks, 1);

    std::vector<BlockTable> tables(2);
    // 8 tokens total, 1 page (4 tokens) common -> 4 uncached tokens -> +1 page each.
    coord.ClaimCommonPrefix(tables, hit);              // claim the 1 cached page each
    ASSERT_TRUE(coord.Acquire(tables, 8 - hit.num_common_blocks * 4));
    EXPECT_EQ(tables[0].NumBlocks(), 2);  // 1 claimed + 1 allocated
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, CrossGroupShortfallAllocatesNothing) {
    // Tight pool: the combined demand exceeds free blocks -> check-then-act
    // allocates NOTHING (no partial state, no rollback needed).
    BlockPool pool(5);  // 4 usable after null reservation
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CoordinatorMatch hit = coord.MatchPrefix(ch);  // all miss, common 0
    ASSERT_EQ(hit.num_common_blocks, 0);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    coord.ClaimCommonPrefix(tables, hit);          // no hits -> no-op
    // 12 tokens -> 3 pages per group = 6 needed, only 4 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 12));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);   // untouched, not rolled back
}

TEST(CoordinatorStepTest, AcquireKeepsGroupsAligned) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 page each
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 more each
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorStepTest, AcquireShortfallAllocatesNothing) {
    BlockPool pool(3);  // 2 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    // 2 pages per group (8 tokens) = 4 blocks, only 2 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CoordinatorStepTest, CacheFullBlocksThenMatchHits) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));   // 1 page each
    coord.CacheFullBlocks(tables, ch, /*num_full_blocks=*/1);

    // A fresh request now hits the common prefix (1 page) in both groups.
    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 1);
}

TEST(CoordinatorStepTest, FreeReturnsAllGroups) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 8));   // 2 pages each = 4 blocks
    std::int32_t free_mid = pool.NumFreeBlocks();
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_mid + 4);
}

TEST(CoordinatorStepTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});

    // Request A: cold, allocate 2 pages each, cache both, free.
    {
        CoordinatorMatch m = coord.MatchPrefix(ch);
        EXPECT_EQ(m.num_common_blocks, 0);
        std::vector<BlockTable> a(2);
        coord.ClaimCommonPrefix(a, m);
        ASSERT_TRUE(coord.Acquire(a, 8));
        coord.CacheFullBlocks(a, ch, 2);
        coord.Free(a);
    }
    // Request B: shares the prefix -> common 2 pages in both groups.
    {
        CoordinatorMatch m = coord.MatchPrefix(ch);
        EXPECT_EQ(m.num_common_blocks, 2);
        std::vector<BlockTable> b(2);
        coord.ClaimCommonPrefix(b, m);
        ASSERT_TRUE(coord.Acquire(b, 8 - m.num_common_blocks * 4));
        EXPECT_EQ(b[0].NumBlocks(), 2);
        EXPECT_EQ(b[1].NumBlocks(), 2);
        coord.Free(b);
    }
}

TEST(CoordinatorMatchTest, SwaHoleTruncationRecomputesNumHit) {
    // full caches 5 pages; swa (window 10 -> contiguous_needed 3) caches a 3-run
    // at indices 2,3,4 -> swa coverage is [null,null,b,b,b] = 5 (leading holes
    // null-pad back to 0). full coverage 5 too. To force a truncation that cuts
    // SWA real hits, make full SHALLOWER: full caches only 4 pages -> common 4.
    // swa truncated to 4: [null,null,b,b] -> num_hit recomputed 4->2.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    // full (group 0): cache pages 0..3 only (coverage 4).
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    // swa (group 1): cache pages 2,3,4 -> a 3-run at the tail -> coverage 5,
    // real hits at indices 2,3,4, holes at 0,1.
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 4);   // min(4, 5)
    ASSERT_EQ(m.per_group.size(), 2u);
    // full truncated 5->4? no, full coverage is 4 already; swa truncated 5->4.
    EXPECT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_TRUE(m.per_group[1].blocks[0]->IsNull());
    EXPECT_TRUE(m.per_group[1].blocks[1]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[2]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[3]->IsNull());
    // swa had 3 real hits over coverage 5; truncated to 4 drops the rightmost
    // real hit (index 4) -> 2 real remain.
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
}

TEST(CoordinatorAllocTest, AcquireShortfallLeavesClaimedPrefixForCallerToFree) {
    // With check-then-act, ClaimCommonPrefix and Acquire are separate steps.
    // A capacity-short Acquire allocates nothing, but the already-claimed prefix
    // stays in the tables -- the caller (FSM retract path) Frees them. This test
    // verifies that contract: claim succeeds, Acquire fails cleanly (no new
    // blocks), and an explicit Free fully restores the pool.
    // swa window 4 -> contiguous_needed 1, so a single cached page is a hit.
    BlockPool pool(6);  // 5 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    // Cache page 0 in both groups -> common prefix 1.
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);
    std::int32_t free_before = pool.NumFreeBlocks();  // after caching, before claim

    CoordinatorMatch hit = coord.MatchPrefix(ch);
    ASSERT_EQ(hit.num_common_blocks, 1);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, hit);   // claim 1 cached page each (2 blocks)
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    // Remainder: 12 tokens, 1 page common (4 tokens) -> uncached 8 -> 2 pages per
    // group = 4 needed. 5 usable - 2 claimed = 3 free < 4 -> Acquire fails, takes
    // nothing; the claimed prefix remains in the tables.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 1);    // claimed prefix still there
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    // Caller frees the request -> pool fully restored (claimed refs balanced,
    // cached pages return to free list still reusable).
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

// AdvanceWindow only touches sliding-window groups; full-attention groups are
// left completely untouched (spec.kind gate). page_size=2 for both groups,
// swa window=4.
TEST(KvCacheCoordinatorAdvanceWindow, OnlySlidingWindowGroupEvicts) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    // Allocate 6 tokens -> 3 pages per group (page_size=2).
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    ASSERT_EQ(tables[1].NumBlocks(), 3);

    auto full_before = tables[0].Blocks();
    std::vector<CacheBlock*> full_snapshot(full_before.begin(), full_before.end());

    // num_computed_tokens=5 -> swa skipped=5-4+1=2 -> skipped_blocks=2/2=1:
    // swa group evicts page 0 (one page slides fully out of window).
    coordinator.AdvanceWindow(tables, /*num_computed_tokens=*/5);

    // Full group untouched: same length, same blocks, no nulls.
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    auto full_after = tables[0].Blocks();
    for (std::int32_t i = 0; i < tables[0].NumBlocks(); ++i) {
        EXPECT_EQ(full_after[i], full_snapshot[i]) << "full group block " << i << " changed";
        EXPECT_NE(full_after[i], pool.NullBlock()) << "full group got a null hole at " << i;
    }

    // Swa group: page 0 became a null hole, length unchanged.
    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_EQ(tables[1].Blocks()[0], pool.NullBlock());
    EXPECT_NE(tables[1].Blocks()[1], pool.NullBlock());
    EXPECT_NE(tables[1].Blocks()[2], pool.NullBlock());
}

}  // namespace
}  // namespace tokenspeed::test
