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

#include <span>
#include <string>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "scheduler/page_hasher.h"
#include "cache/swa_manager.h"

namespace tokenspeed::test {
namespace {

std::vector<std::int32_t> BlockIds(const std::vector<BlockRef>& refs) {
    std::vector<std::int32_t> ids;
    ids.reserve(refs.size());
    for (const BlockRef& ref : refs) {
        ids.push_back(ref ? ref->BlockId() : 0);
    }
    return ids;
}

using token_span = std::span<const std::int32_t>;

std::string RealKey(const std::vector<std::int32_t>& tokens, uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> keys = ComputePagedHashesWithGroup(pages, "", group_id);
    return keys.front();
}

// Cache then free, so the page is prefix-hittable via MatchPrefix.
std::int32_t CacheOnePage(BlockPool& pool, const std::string& key) {
    BlockRef got = pool.AcquireBlock();
    const std::int32_t id = got->BlockId();
    pool.CacheFullBlock(got, key);
    got.reset();
    return id;
}

TEST(SwaManagerTest, ConstructsWithWindow) {
    BlockPool pool(8);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/10);
    BlockTable table;
    EXPECT_EQ(table.NumBlocks(), 0);
}

TEST(SwaManagerTest, MatchAllMissReturnsEmpty) {
    BlockPool pool(8);
    SwaManager mgr(4, 10);
    std::vector<std::string> hashes = {RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)};
    PrefixMatch m = mgr.Match(pool, hashes, 0, static_cast<std::int32_t>(hashes.size()));
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_TRUE(m.blocks.empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // no hit, so nothing pinned
}

TEST(SwaManagerTest, MatchStopsAfterContiguousNeededFromRight) {
    // block_size 4, window 10 -> pages_needed = ceil(9/4) = 3.
    BlockPool pool(16);
    SwaManager mgr(4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    const std::int32_t b1 = CacheOnePage(pool, h1);
    const std::int32_t b2 = CacheOnePage(pool, h2);
    const std::int32_t b3 = CacheOnePage(pool, h3);

    std::vector<std::string> keys{h0, h1, h2, h3, h4};
    PrefixMatch m = mgr.Match(pool, keys, 0, 5);
    // Right->left: h4 miss; h3,h2,h1 hit -> run reaches 3, stop. run_end = 3.
    // keep [0..3] -> [NULL, b1, b2, b3]; num_hit_blocks = 3.
    ASSERT_EQ(m.blocks.size(), 4u);
    EXPECT_FALSE(m.blocks[0]);
    EXPECT_EQ(m.blocks[1]->BlockId(), b1);
    EXPECT_EQ(m.blocks[2]->BlockId(), b2);
    EXPECT_EQ(m.blocks[3]->BlockId(), b3);
    EXPECT_EQ(m.num_hit_blocks, 3);
}

TEST(SwaManagerTest, BoundedMatchEnforcesRunAgainstBoundedEnd) {
    // Tail 3-run {2,3,4}. Bounded to 4 the run {2,3} < pages_needed 3 with
    // holes at 0,1 -> the bounded overload re-scans and returns empty.
    BlockPool pool(16);
    SwaManager mgr(4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    CacheOnePage(pool, h2);
    CacheOnePage(pool, h3);
    CacheOnePage(pool, h4);
    std::vector<std::string> hashes{h0, h1, h2, h3, h4};

    PrefixMatch unbounded = mgr.Match(pool, hashes, 0, /*max_blocks=*/5);
    EXPECT_EQ(unbounded.blocks.size(), 5u);
    EXPECT_EQ(unbounded.num_hit_blocks, 3);

    PrefixMatch bounded = mgr.Match(pool, hashes, 0, /*max_blocks=*/4);
    EXPECT_TRUE(bounded.blocks.empty());
    EXPECT_EQ(bounded.num_hit_blocks, 0);
}

TEST(SwaManagerTest, MatchTrimsTailAfterWindow) {
    // pages_needed = ceil((4-1)/4) = 1 -> any single hit (from the right) suffices.
    BlockPool pool(16);
    SwaManager mgr(4, 4);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    const std::int32_t b0 = CacheOnePage(pool, h0);
    const std::int32_t b2 = CacheOnePage(pool, h2);  // h1 left uncached

    // Right->left: h2 hits, run 1 >= pages_needed -> keep [0..2].
    std::vector<std::string> keys{h0, h1, h2};
    PrefixMatch m = mgr.Match(pool, keys, 0, 3);
    ASSERT_EQ(m.blocks.size(), 3u);
    EXPECT_FALSE(m.blocks[0]);
    EXPECT_FALSE(m.blocks[1]);
    EXPECT_EQ(m.blocks[2]->BlockId(), b2);
    EXPECT_EQ(m.num_hit_blocks, 1);
    (void)b0;
}

TEST(SwaManagerTest, MatchAcceptsRunShorterThanContiguousNeeded) {
    // window 10 -> pages_needed 3, but prompt is only 2 pages, both cached.
    BlockPool pool(16);
    SwaManager mgr(4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    const std::int32_t b0 = CacheOnePage(pool, h0);
    const std::int32_t b1 = CacheOnePage(pool, h1);

    // Run reaches the left end at 2 < 3; run > 0 -> accept, keep [b0, b1].
    std::vector<std::string> keys{h0, h1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), b0);
    EXPECT_EQ(m.blocks[1]->BlockId(), b1);
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(SwaManagerTest, MatchRequiresContiguityNotAnyHit) {
    // h2 miss splits runs {h3,h4} and {h0,h1}; neither reaches 3, so the
    // surviving run is the LEFT one: keep [0..1] = [b0, b1].
    BlockPool pool(16);
    SwaManager mgr(4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    const std::int32_t b0 = CacheOnePage(pool, h0);
    const std::int32_t b1 = CacheOnePage(pool, h1);
    CacheOnePage(pool, h3);
    CacheOnePage(pool, h4);  // h2 left uncached

    std::vector<std::string> keys{h0, h1, h2, h3, h4};
    PrefixMatch m = mgr.Match(pool, keys, 0, 5);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), b0);
    EXPECT_EQ(m.blocks[1]->BlockId(), b1);
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(SwaManagerTest, SpeculativeHitsDoNotRefreshEvictionOrder) {
    BlockPool pool(7);
    SwaManager mgr(4, 10);  // pages_needed = 3
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    const std::int32_t b0 = CacheOnePage(pool, h0);
    const std::int32_t b1 = CacheOnePage(pool, h1);
    const std::int32_t b3 = CacheOnePage(pool, h3);
    CacheOnePage(pool, h4);

    std::vector<std::string> keys{h0, h1, h2, h3, h4};
    PrefixMatch match = mgr.Match(pool, keys, 0, 5);
    ASSERT_EQ(BlockIds(match.blocks), (std::vector<std::int32_t>{b0, b1}));

    std::vector<BlockRef> acquired = pool.AcquireBlocks(3);
    ASSERT_EQ(acquired.size(), 3u);
    EXPECT_EQ(acquired.back()->BlockId(), b3);
}

// Pins the device-tier W=1 semantic: no lookback means every boundary is resumable,
// so the match covers the full bounded range with holes and claims no real page.
TEST(SwaManagerTest, MatchWindowOneCoversAllAsHoles) {
    BlockPool pool(8);
    SwaManager mgr(4, /*sliding_window=*/1);  // pages_needed = 0
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    CacheOnePage(pool, h0);  // a real cached page must NOT shrink or anchor the match

    std::vector<std::string> keys{h0, "k1", "k2"};
    PrefixMatch m = mgr.Match(pool, keys, 0, 3);
    EXPECT_EQ(BlockIds(m.blocks), (std::vector<std::int32_t>{0, 0, 0}));
    EXPECT_EQ(m.num_hit_blocks, 0);
}

TEST(SwaManagerTest, MatchPinsUntilResultDies) {
    BlockPool pool(8);
    SwaManager mgr(4, 4);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    const std::int32_t b0 = CacheOnePage(pool, h0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    std::vector<std::string> keys{h0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(m.blocks.front().use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
    m = {};
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(SwaManagerTest, ClaimHitBlocksSkipsNullHoles) {
    BlockPool pool(16);
    SwaManager mgr(4, 10);  // pages_needed = 3
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    const std::int32_t b1 = CacheOnePage(pool, h1);
    const std::int32_t b2 = CacheOnePage(pool, h2);
    const std::int32_t b3 = CacheOnePage(pool, h3);
    std::int32_t free_before = pool.NumFreeBlocks();

    std::vector<std::string> keys{h0, h1, h2, h3};
    PrefixMatch m = mgr.Match(pool, keys, 0, 4);
    ASSERT_EQ(m.blocks.size(), 4u);
    ASSERT_FALSE(m.blocks[0]);
    ASSERT_EQ(m.num_hit_blocks, 3);

    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));

    // The null hole is preserved to keep logical-page slot alignment.
    EXPECT_EQ(table.NumBlocks(), 4);
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_EQ(table.Blocks()[1].use_count(), 1);
    EXPECT_EQ(table.Blocks()[2].use_count(), 1);
    EXPECT_EQ(table.Blocks()[3].use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 3);
}

TEST(SwaManagerTest, InheritedAcquireAndFreeWork) {
    BlockPool pool(8);
    SwaManager mgr(4, 10);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), 5);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(SwaManagerTest, InheritedCacheFullBlocksMakesPagesHittable) {
    BlockPool pool(8);
    SwaManager mgr(4, 4);  // pages_needed = 1
    std::string h0 = RealKey({0, 0, 0, 0}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{h0});

    std::vector<std::string> keys{h0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(m.blocks.back()->BlockId(), a.Blocks()[0]->BlockId());
}

TEST(BlockTableTest, EvictToNullReturnsOldBlockAndPunchesHole) {
    BlockPool pool(8);
    SwaManager mgr(4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 real pages
    ASSERT_EQ(table.NumBlocks(), 2);
    const std::int32_t page0 = table.Blocks()[0]->BlockId();

    BlockRef old = table.EvictToNull(0);
    EXPECT_EQ(old->BlockId(), page0);  // returns the displaced ownership
    EXPECT_FALSE(table.Blocks()[0]);   // slot is now a null hole
    EXPECT_EQ(table.NumBlocks(), 2);   // length unchanged (no shrink)
}

TEST(BlockTableTest, EvictToNullIsIdempotentOnNullSlot) {
    BlockPool pool(8);
    SwaManager mgr(4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4));  // 1 real page
    table.EvictToNull(0).reset();              // first: punches hole
    BlockRef again = table.EvictToNull(0);
    EXPECT_FALSE(again);  // empty on already-null
    EXPECT_FALSE(table.Blocks()[0]);
}

TEST(SwaManagerTest, ReclaimExpiredMirrorsVllmBoundarySequence) {
    // Mirrors vLLM test_sliding_window_remove_skipped_blocks.
    // skipped = max(0, n - 4 + 1); skipped_blocks = skipped / 2.
    BlockPool pool(32);
    SwaManager mgr(/*block_size=*/2, /*sliding_window=*/4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 10));  // 5 real pages (10 tokens / page 2)
    ASSERT_EQ(table.NumBlocks(), 5);
    const std::int32_t p0 = table.Blocks()[0]->BlockId();
    const std::int32_t p1 = table.Blocks()[1]->BlockId();
    const std::int32_t p2 = table.Blocks()[2]->BlockId();
    const std::int32_t p3 = table.Blocks()[3]->BlockId();
    const std::int32_t p4 = table.Blocks()[4]->BlockId();

    // n=0: skipped 0 -> nothing freed.
    mgr.ReclaimExpired(pool, table, 0);
    EXPECT_TRUE(table.Blocks()[0]);

    // n=4: skipped 1, blocks 0 -> page 0 still holds an in-window token, no free.
    mgr.ReclaimExpired(pool, table, 4);
    EXPECT_TRUE(table.Blocks()[0]);

    // n=5: skipped 2, blocks 1 -> page 0 fully out -> punched to null.
    std::int32_t free_before5 = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 5);
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_TRUE(table.Blocks()[1]);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before5 + 1);  // p0 returned

    // n=6: skipped 3, blocks 1 -> page 1 still in window; no change.
    mgr.ReclaimExpired(pool, table, 6);
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_TRUE(table.Blocks()[1]);

    // n=7: skipped 4, blocks 2 -> page 1 punched; page 0 already null -> break.
    std::int32_t free_before7 = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 7);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_TRUE(table.Blocks()[2]);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before7 + 1);  // only p1 returned

    // n=11: skipped 8, blocks 4 -> pages 2 and 3 punched; page 4 stays.
    std::int32_t free_before11 = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 11);
    EXPECT_FALSE(table.Blocks()[2]);
    EXPECT_FALSE(table.Blocks()[3]);
    EXPECT_TRUE(table.Blocks()[4]);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before11 + 2);  // p2, p3 returned
    EXPECT_EQ(table.NumBlocks(), 5);                     // length never shrinks

    (void)p0;
    (void)p1;
    (void)p2;
    (void)p3;
    (void)p4;
}

TEST(SwaManagerTest, ReclaimExpiredEarlyReturnInsideWindow) {
    BlockPool pool(32);
    SwaManager mgr(4, 16);  // big window
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages, 8 tokens <= window
    std::int32_t free_before = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 8);  // skipped = 8 - 16 + 1 < 0 -> early return
    EXPECT_TRUE(table.Blocks()[0]);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(SwaManagerTest, ReclaimExpiredCapsToAllocatedBlocks) {
    BlockPool pool(32);
    SwaManager mgr(4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages
    // skipped_blocks would exceed NumBlocks(); must cap, not go out of bounds.
    mgr.ReclaimExpired(pool, table, 1000);
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_EQ(table.NumBlocks(), 2);  // still 2 slots, both null
}

TEST(SwaManagerTest, ReclaimExpiredEvictsFirstSlidOutFirst) {
    // Pool sized to exactly the 4 acquired pages (+1 null): the free list is empty
    // after Acquire, so the next allocation must expose FIFO order among the freed batch.
    BlockPool pool(5);
    SwaManager mgr(2, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 4 pages
    const std::int32_t p0 = table.Blocks()[0]->BlockId();
    const std::int32_t p1 = table.Blocks()[1]->BlockId();
    mgr.ReclaimExpired(pool, table, 8);  // skipped 5, blocks 2 -> free pages 0,1
    ASSERT_FALSE(table.Blocks()[0]);
    ASSERT_FALSE(table.Blocks()[1]);

    BlockRef reused = pool.AcquireBlock();
    ASSERT_TRUE(reused);
    EXPECT_EQ(reused->BlockId(), p0);
    (void)p1;
}

TEST(SwaManagerTest, ReclaimExpiredFreedCachedPageStaysPrefixReusable) {
    BlockPool pool(32);
    SwaManager mgr(2, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 4 pages
    const std::string h0 = RealKey({1, 1}, 0);
    mgr.CacheFullBlocks(pool, table, std::vector<std::string>{h0});
    const std::int32_t p0 = table.Blocks()[0]->BlockId();
    EXPECT_TRUE(table.Blocks()[0]->IsCached());

    mgr.ReclaimExpired(pool, table, 8);  // frees pages 0,1; p0 returns with hash intact
    EXPECT_FALSE(table.Blocks()[0]);
    BlockRef hit = pool.AcquireCachedBlock(h0);
    EXPECT_EQ(hit->BlockId(), p0);
}

TEST(SwaManagerTest, ReclaimExpiredLeavesTailAvailUnchanged) {
    BlockPool pool(32);
    SwaManager mgr(4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 10));  // 3 pages, last partial: tail_avail = 2
    EXPECT_EQ(table.TailAvailableTokens(), 2);
    mgr.ReclaimExpired(pool, table, 10);  // skipped 7, blocks 1 -> frees front full page
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_EQ(table.TailAvailableTokens(), 2);  // tail untouched
}

TEST(SwaManagerTest, AcquireAdvancePairingKeepsPhysicalPagesBounded) {
    // Steady state: active pages stay bounded near ceil(window/block_size) = 2.
    BlockPool pool(64);
    SwaManager mgr(2, 4);
    BlockTable table;
    std::int32_t n = 0;
    std::int32_t baseline_free = pool.NumFreeBlocks();
    for (int step = 0; step < 20; ++step) {
        n += 2;  // two new tokens -> one new page
        ASSERT_TRUE(mgr.Acquire(pool, table, 2));
        mgr.ReclaimExpired(pool, table, n);
    }
    std::int32_t active = baseline_free - pool.NumFreeBlocks();
    EXPECT_LE(active, 3);
    // The table itself grows (holes accumulate), but physical pages are bounded.
    EXPECT_GT(table.NumBlocks(), 3);
}

TEST(SwaManagerTest, BoundedMatchReturnsOnlyTrailingRunWithAbsoluteBase) {
    BlockPool pool(16);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/10, KvTableLayout::kBoundedWindow);
    std::vector<std::string> keys{
        RealKey({0, 0, 0, 0}, 0), RealKey({1, 1, 1, 1}, 0), RealKey({2, 2, 2, 2}, 0),
        RealKey({3, 3, 3, 3}, 0), RealKey({4, 4, 4, 4}, 0),
    };
    const std::int32_t b2 = CacheOnePage(pool, keys[2]);
    const std::int32_t b3 = CacheOnePage(pool, keys[3]);
    const std::int32_t b4 = CacheOnePage(pool, keys[4]);

    PrefixMatch match = mgr.Match(pool, keys, /*begin_blocks=*/0, /*max_blocks=*/5);
    EXPECT_EQ(match.base_logical_page, 2);
    EXPECT_EQ(match.LogicalEnd(), 5);
    EXPECT_EQ(BlockIds(match.blocks), (std::vector<std::int32_t>{b2, b3, b4}));
    EXPECT_EQ(match.num_hit_blocks, 3);
}

TEST(SwaManagerTest, BoundedWindowOneRepresentsValidEmptyRangeAtBoundary) {
    BlockPool pool(8);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/1, KvTableLayout::kBoundedWindow);
    const std::vector<std::string> keys{"k0", "k1", "k2"};

    PrefixProbe probe = mgr.Probe(pool, keys, /*begin_blocks=*/0, /*max_blocks=*/3);
    EXPECT_TRUE(probe.hits.empty());
    EXPECT_EQ(probe.base_logical_page, 3);
    EXPECT_EQ(probe.LogicalEnd(), 3);

    PrefixMatch match = mgr.AcquireMatchedBlocks(pool, keys, 0, probe);
    EXPECT_TRUE(match.blocks.empty());
    EXPECT_EQ(match.base_logical_page, 3);
    EXPECT_EQ(match.LogicalEnd(), 3);
}

TEST(SwaManagerTest, ExactBoundaryDoesNotFallBackToEarlierResumableEndpoint) {
    BlockPool pool(16);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/10, KvTableLayout::kBoundedWindow);
    std::vector<std::string> keys{
        RealKey({0, 0, 0, 0}, 0),
        RealKey({1, 1, 1, 1}, 0),
        RealKey({2, 2, 2, 2}, 0),
        RealKey({3, 3, 3, 3}, 0),
    };
    CacheOnePage(pool, keys[0]);
    CacheOnePage(pool, keys[1]);
    CacheOnePage(pool, keys[2]);

    PrefixProbe earlier = mgr.Probe(pool, keys, 0, 4);
    EXPECT_EQ(earlier.LogicalEnd(), 3);
    EXPECT_FALSE(mgr.ProbeExactBoundary(pool, keys, 0, 4).has_value());
}

TEST(SwaManagerTest, BoundedReclaimAdvancesLogicalBaseAndShrinksStorage) {
    BlockPool pool(16);
    SwaManager mgr(/*block_size=*/2, /*sliding_window=*/4, KvTableLayout::kBoundedWindow);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 10));
    ASSERT_EQ(table.BaseLogicalPage(), 0);
    ASSERT_EQ(table.LogicalEnd(), 5);
    const std::int32_t free_before = pool.NumFreeBlocks();

    mgr.ReclaimExpired(pool, table, /*num_computed_tokens=*/5);
    EXPECT_EQ(table.BaseLogicalPage(), 1);
    EXPECT_EQ(table.LiveSize(), 4);
    EXPECT_EQ(table.LogicalEnd(), 5);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 1);

    mgr.ReclaimExpired(pool, table, /*num_computed_tokens=*/11);
    EXPECT_EQ(table.BaseLogicalPage(), 4);
    EXPECT_EQ(table.LiveSize(), 1);
    EXPECT_EQ(table.LogicalEnd(), 5);
}

}  // namespace
}  // namespace tokenspeed::test
