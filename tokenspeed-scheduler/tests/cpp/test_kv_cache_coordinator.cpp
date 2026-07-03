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

// Asserts the sliding-window validity invariant on a group's match at its
// claimed length: the last min(len, contiguous_needed) blocks are real cached
// pages, i.e. no null hole inside the last window of the claimed prefix.
void ExpectSwaWindowIntact(const PrefixMatch& m, std::int32_t window, std::int32_t page_size) {
    std::int32_t len = static_cast<std::int32_t>(m.blocks.size());
    std::int32_t contiguous_needed = (window - 1 + page_size - 1) / page_size;
    std::int32_t need = std::min(len, contiguous_needed);
    for (std::int32_t i = len - need; i < len; ++i) {
        EXPECT_FALSE(m.blocks[static_cast<std::size_t>(i)]->IsNull())
            << "null hole inside the last window at slot " << i << " of " << len;
    }
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

TEST(CoordinatorMatchTest, SwaRunCutByFullBoundDropsToNoValidMatch) {
    // full caches pages 0..3 (coverage 4); swa (window 10 -> contiguous_needed 3)
    // caches a 3-run at the TAIL (indices 2,3,4, unbounded coverage 5). Bounded
    // to full's 4, swa's remaining run {2,3} is only 2 long and pages 0,1 are
    // holes, so NO valid swa match of any length <= 4 exists (claiming 4 would
    // mark token 16 computed while its window spans the page-1 hole -> attention
    // would read garbage KV). The coordinator must lower common to 0 for
    // everyone, not keep [null,null,b,b] and call it a 4-block hit.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, FullShorterThanSwaBoundsSwaWithRunIntact) {
    // full caches pages 0..3 (coverage 4); swa caches 1..4 (unbounded coverage 5
    // with the run ending at 4). Bounded to 4, swa's run {1,2,3} still reaches
    // contiguous_needed 3 against the bounded end, so common stays 4 and the
    // swa match is [null, b1, b2, b3] -- hole only OUTSIDE the last window.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 4);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_TRUE(m.per_group[1].blocks[0]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[1]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[2]->IsNull());
    EXPECT_FALSE(m.per_group[1].blocks[3]->IsNull());
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, SwaShorterThanFullTruncatesFull) {
    // full caches all 5 pages; swa caches 1..3 -> its best valid match is 4
    // blocks [null, b1, b2, b3] (run {1,2,3} ends at index 3). common drops to
    // 4 and the FULL match truncates 5 -> 4 (always safe for full attention).
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 4);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_TRUE(m.per_group[1].blocks[0]->IsNull());
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, TwoSwaGroupsIterateToFixpoint) {
    // Fixpoint iteration: lowering the bound for one SWA group can invalidate
    // another's already-computed match, whose re-match lowers the bound AGAIN.
    // window 10, page 4 -> contiguous_needed 3; 6 pages.
    //   full (group 0): caches all 6           -> initial common 6.
    //   swaA (group 1): caches {0, 2, 3, 4}    -> bound 6: run {2,3,4}, keep 5 -> common 5.
    //   swaB (group 2): caches {0, 1, 2, 3}    -> bound 5: run {1,2,3}, keep 4 -> common 4.
    //   swaA RE-match at 4: run {2,3} too short, falls back to the left-end run
    //   {0}, keep 1 -> common 1 (re-match shortened again -> iteration needed).
    //   swaB RE-match at 1: page 0 cached, keep 1 -> fixpoint common = 1.
    // At length 1 the window clamps to the sequence start, so a single real
    // page 0 is a valid match for both SWA groups.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes(
        {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);
    CacheForGroup(pool, ch[2], 2);
    CacheForGroup(pool, ch[3], 2);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 1);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 1u) << "group " << i;
        EXPECT_FALSE(m.per_group[i].blocks[0]->IsNull()) << "group " << i;
        EXPECT_EQ(m.per_group[i].num_hit_blocks, 1) << "group " << i;
    }
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*page_size=*/4);
    ExpectSwaWindowIntact(m.per_group[2], /*window=*/10, /*page_size=*/4);
}

TEST(CoordinatorMatchTest, AllFullGroupsMinTruncationUnchanged) {
    // No SWA groups: behavior is the original min-then-truncate (any prefix of
    // a full match is valid, no bounded re-match needed).
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SingleFullGroupUnchanged) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2);
    ASSERT_EQ(m.per_group.size(), 1u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SwaOnlyConfigKeepsTailRunWithLeadingHoles) {
    // No full groups: the initial bound is the whole prompt, so a single SWA
    // group behaves exactly like its unbounded MatchPrefix -- tail run {2,3,4}
    // covers the window, leading holes null-pad back to page 0.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[4], 0);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 5);
    ASSERT_EQ(m.per_group.size(), 1u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 5u);
    EXPECT_TRUE(m.per_group[0].blocks[0]->IsNull());
    EXPECT_TRUE(m.per_group[0].blocks[1]->IsNull());
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[0], /*window=*/10, /*page_size=*/4);
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

// Three groups (full + two sliding-window): the common prefix is the MINIMUM
// coverage across all three, not just across two. full caches 4 pages, swa_a
// caches a 3-run at the front, swa_b only a 2-run at the front. Common should be
// min(4, 3, 2) = 2, and every group's per-group hit is truncated to 2.
TEST(CoordinatorMatchTest, ThreeGroupsCommonIsMinCoverageAcrossAll) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    // group 0 (full): all 4 pages.
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    // group 1 (swa_a): front 3-run.
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    // group 2 (swa_b): front 2-run -> the binding minimum.
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 2) << "common = min(4, 3, 2)";
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[2].blocks.size(), 2u);
    // Each group's num_hit recomputed to the truncated common length.
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[2].num_hit_blocks, 2);
}

// Three groups where one sliding-window group has ZERO hits: that forces the
// common prefix to 0 for everyone, even though the other two fully hit.
TEST(CoordinatorMatchTest, ThreeGroupsOneAllMissForcesZeroCommon) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // groups 0 and 1 fully cache both pages; group 2 caches nothing.
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    for (const std::string& h : ch) CacheForGroup(pool, h, 1);

    CoordinatorMatch m = coord.MatchPrefix(ch);
    EXPECT_EQ(m.num_common_blocks, 0) << "one group all-miss -> common 0";
}

}  // namespace
}  // namespace tokenspeed::test
