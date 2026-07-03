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

using token_span = std::span<const std::int32_t>;

// A real BlockHashWithGroupId key for a single page (matches test_block_pool /
// test_full_attn_manager style).
std::string RealKey(const std::vector<std::int32_t>& tokens, uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> keys = ComputePagedHashesWithGroup(pages, "", group_id);
    return keys.front();
}

// Cache one full page directly through the pool and return to the free list so a
// later MatchPrefix can prefix-hit it. Returns the physical block.
CacheBlock* CacheOnePage(BlockPool& pool, const std::string& key) {
    std::vector<CacheBlock*> got = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(got.front(), key);
    pool.FreeBlocks(got);
    return got.front();
}

TEST(SwaManagerTest, ConstructsWithWindow) {
    BlockPool pool(8);
    SwaManager mgr(pool, /*page_size=*/4, /*sliding_window=*/10);
    BlockTable table;
    EXPECT_EQ(table.NumBlocks(), 0);
}

TEST(SwaManagerTest, MatchAllMissReturnsEmpty) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 10);
    std::vector<std::string> hashes = {RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)};
    PrefixMatch m = mgr.MatchPrefix(hashes);
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_TRUE(m.blocks.empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // read-only, nothing claimed
}

TEST(SwaManagerTest, MatchStopsAfterContiguousNeededFromRight) {
    // page_size 4, window 10 -> contiguous_needed = ceil(9/4) = 3.
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    CacheBlock* b1 = CacheOnePage(pool, h1);
    CacheBlock* b2 = CacheOnePage(pool, h2);
    CacheBlock* b3 = CacheOnePage(pool, h3);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0, h1, h2, h3, h4});
    // Right->left: h4 miss; h3,h2,h1 hit -> run reaches 3, stop. run_end = 3.
    // keep [0..3] -> [NULL, b1, b2, b3]; num_hit_blocks = 3.
    ASSERT_EQ(m.blocks.size(), 4u);
    EXPECT_TRUE(m.blocks[0]->IsNull());
    EXPECT_EQ(m.blocks[1]->BlockId(), b1->BlockId());
    EXPECT_EQ(m.blocks[2]->BlockId(), b2->BlockId());
    EXPECT_EQ(m.blocks[3]->BlockId(), b3->BlockId());
    EXPECT_EQ(m.num_hit_blocks, 3);
}

TEST(SwaManagerTest, BoundedMatchEnforcesRunAgainstBoundedEnd) {
    // page_size 4, window 10 -> contiguous_needed 3. Cached: h2,h3,h4 (a tail
    // 3-run). Unbounded: keep 5 = [NULL,NULL,b2,b3,b4]. Bounded to 4 the run
    // {2,3} is too short and pages 0,1 are holes -> NO valid match of length
    // <= 4 exists, so the bounded overload returns empty (it must re-run the
    // scan on the first 4 hashes, never chop the unbounded result).
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    CacheOnePage(pool, h2);
    CacheOnePage(pool, h3);
    CacheOnePage(pool, h4);
    std::vector<std::string> hashes{h0, h1, h2, h3, h4};

    PrefixMatch unbounded = mgr.MatchPrefix(hashes, /*max_blocks=*/5);
    EXPECT_EQ(unbounded.blocks.size(), 5u);
    EXPECT_EQ(unbounded.num_hit_blocks, 3);

    PrefixMatch bounded = mgr.MatchPrefix(hashes, /*max_blocks=*/4);
    EXPECT_TRUE(bounded.blocks.empty());
    EXPECT_EQ(bounded.num_hit_blocks, 0);
}

TEST(SwaManagerTest, MatchTrimsTailAfterWindow) {
    // contiguous_needed = ceil((4-1)/4) = 1 -> any single hit (from the right) suffices.
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 4);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    CacheBlock* b0 = CacheOnePage(pool, h0);
    CacheBlock* b2 = CacheOnePage(pool, h2);  // h1 left uncached

    // Right->left: h2 hits first, run reaches 1 (>=1), stop at run_end = 2.
    // keep [0..2]: blocks[0]=NULL, blocks[1]=NULL(h1 miss), blocks[2]=b2.
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0, h1, h2});
    ASSERT_EQ(m.blocks.size(), 3u);
    EXPECT_TRUE(m.blocks[0]->IsNull());
    EXPECT_TRUE(m.blocks[1]->IsNull());
    EXPECT_EQ(m.blocks[2]->BlockId(), b2->BlockId());
    EXPECT_EQ(m.num_hit_blocks, 1);
    (void)b0;
}

TEST(SwaManagerTest, MatchAcceptsRunShorterThanContiguousNeeded) {
    // window 10 -> contiguous_needed 3, but prompt is only 2 pages, both cached.
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    CacheBlock* b0 = CacheOnePage(pool, h0);
    CacheBlock* b1 = CacheOnePage(pool, h1);

    // Right->left: h1 hit (run 1), h0 hit (run 2), reach left end without hitting 3.
    // run > 0 -> accept. run_end = 1, keep [0..1] = [b0, b1].
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0, h1});
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), b0->BlockId());
    EXPECT_EQ(m.blocks[1]->BlockId(), b1->BlockId());
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(SwaManagerTest, MatchRequiresContiguityNotAnyHit) {
    // contiguous_needed 3 (window 10, ps 4). Layout: h0,h1 cached, h2 miss,
    // h3,h4 cached. Right->left: h4(run1),h3(run2),h2 miss-> reset; h1(run1),
    // h0(run2), end. No run reaches 3, so the surviving run is the LEFT one
    // (h0,h1): run_end = 1, keep [0..1] = [b0, b1], num_hit_blocks = 2.
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 10);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    std::string h4 = RealKey({4, 4, 4, 4}, 0);
    CacheBlock* b0 = CacheOnePage(pool, h0);
    CacheBlock* b1 = CacheOnePage(pool, h1);
    CacheOnePage(pool, h3);
    CacheOnePage(pool, h4);  // h2 left uncached

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0, h1, h2, h3, h4});
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), b0->BlockId());
    EXPECT_EQ(m.blocks[1]->BlockId(), b1->BlockId());
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(SwaManagerTest, MatchDoesNotChangeRefCount) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 4);
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    CacheBlock* b0 = CacheOnePage(pool, h0);
    EXPECT_EQ(b0->RefCount(), 0);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0});
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(b0->RefCount(), 0);          // read-only
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(SwaManagerTest, ClaimHitBlocksSkipsNullHoles) {
    // Build an SWA hit [NULL, b1, b2, b3] and claim it.
    BlockPool pool(16);
    SwaManager mgr(pool, 4, 10);  // contiguous_needed = 3
    std::string h0 = RealKey({0, 0, 0, 0}, 0);
    std::string h1 = RealKey({1, 1, 1, 1}, 0);
    std::string h2 = RealKey({2, 2, 2, 2}, 0);
    std::string h3 = RealKey({3, 3, 3, 3}, 0);
    CacheBlock* b1 = CacheOnePage(pool, h1);
    CacheBlock* b2 = CacheOnePage(pool, h2);
    CacheBlock* b3 = CacheOnePage(pool, h3);
    std::int32_t free_before = pool.NumFreeBlocks();

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0, h1, h2, h3});
    ASSERT_EQ(m.blocks.size(), 4u);
    ASSERT_TRUE(m.blocks[0]->IsNull());
    ASSERT_EQ(m.num_hit_blocks, 3);

    BlockTable table;
    ASSERT_TRUE(mgr.ClaimHitBlocks(table, m));

    // Table keeps all 4 slots (hole preserved for logical-page alignment).
    EXPECT_EQ(table.NumBlocks(), 4);
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    // Only the 3 real blocks were claimed.
    EXPECT_EQ(b1->RefCount(), 1);
    EXPECT_EQ(b2->RefCount(), 1);
    EXPECT_EQ(b3->RefCount(), 1);
    // Free count dropped by exactly the 3 real hits, not the hole.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 3);
}

TEST(SwaManagerTest, InheritedAcquireAndFreeWork) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 10);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(table, 8));   // 2 pages
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), 5);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(SwaManagerTest, InheritedCacheFullBlocksMakesPagesHittable) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 4);  // contiguous_needed = 1
    std::string h0 = RealKey({0, 0, 0, 0}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 4));
    mgr.CacheFullBlocks(a, std::vector<std::string>{h0}, 1);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{h0});
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(m.blocks.back()->BlockId(), a.Blocks()[0]->BlockId());
}

TEST(BlockTableTest, EvictToNullReturnsOldBlockAndPunchesHole) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));  // 2 real pages
    ASSERT_EQ(table.NumBlocks(), 2);
    CacheBlock* page0 = table.Blocks()[0];
    ASSERT_FALSE(page0->IsNull());

    CacheBlock* old = table.EvictToNull(0, pool.NullBlock());
    EXPECT_EQ(old, page0);                       // returns the displaced block
    EXPECT_TRUE(table.Blocks()[0]->IsNull());    // slot is now a null hole
    EXPECT_EQ(table.NumBlocks(), 2);             // length unchanged (no shrink)
}

TEST(BlockTableTest, EvictToNullIsIdempotentOnNullSlot) {
    BlockPool pool(8);
    SwaManager mgr(pool, 4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 4));  // 1 real page
    table.EvictToNull(0, pool.NullBlock());                 // first: punches hole
    CacheBlock* again = table.EvictToNull(0, pool.NullBlock());  // second: already null
    EXPECT_EQ(again, nullptr);                   // returns nullptr on already-null
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
}

TEST(SwaManagerTest, AdvanceWindowMirrorsVllmBoundarySequence) {
    // Mirrors vLLM test_sliding_window_remove_skipped_blocks: block_size=2,
    // sliding_window=4. skipped = max(0, n - 4 + 1) = max(0, n - 3);
    // skipped_blocks = skipped / 2.
    BlockPool pool(32);
    SwaManager mgr(pool, /*page_size=*/2, /*sliding_window=*/4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 10));  // 5 real pages (10 tokens / page 2)
    ASSERT_EQ(table.NumBlocks(), 5);
    CacheBlock* p0 = table.Blocks()[0];
    CacheBlock* p1 = table.Blocks()[1];
    CacheBlock* p2 = table.Blocks()[2];
    CacheBlock* p3 = table.Blocks()[3];
    CacheBlock* p4 = table.Blocks()[4];

    // n=0: skipped 0 -> nothing freed.
    mgr.AdvanceWindow(table, 0);
    EXPECT_FALSE(table.Blocks()[0]->IsNull());

    // n=4: skipped 1, blocks 0 -> page 0 still holds an in-window token, no free.
    mgr.AdvanceWindow(table, 4);
    EXPECT_FALSE(table.Blocks()[0]->IsNull());

    // n=5: skipped 2, blocks 1 -> page 0 fully out -> punched to null.
    std::int32_t free_before5 = pool.NumFreeBlocks();
    mgr.AdvanceWindow(table, 5);
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    EXPECT_FALSE(table.Blocks()[1]->IsNull());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before5 + 1);  // p0 returned

    // n=6: skipped 3, blocks 1 -> page 1 still has an in-window token; page 0
    // already null -> no change.
    mgr.AdvanceWindow(table, 6);
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    EXPECT_FALSE(table.Blocks()[1]->IsNull());

    // n=7: skipped 4, blocks 2 -> page 1 punched; page 0 already null -> break.
    std::int32_t free_before7 = pool.NumFreeBlocks();
    mgr.AdvanceWindow(table, 7);
    EXPECT_TRUE(table.Blocks()[1]->IsNull());
    EXPECT_FALSE(table.Blocks()[2]->IsNull());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before7 + 1);  // only p1 returned

    // n=11: skipped 8, blocks 4 -> pages 2 and 3 punched; page 4 stays.
    std::int32_t free_before11 = pool.NumFreeBlocks();
    mgr.AdvanceWindow(table, 11);
    EXPECT_TRUE(table.Blocks()[2]->IsNull());
    EXPECT_TRUE(table.Blocks()[3]->IsNull());
    EXPECT_FALSE(table.Blocks()[4]->IsNull());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before11 + 2);  // p2, p3 returned
    EXPECT_EQ(table.NumBlocks(), 5);  // length never shrinks

    (void)p0; (void)p1; (void)p2; (void)p3; (void)p4;
}

TEST(SwaManagerTest, AdvanceWindowEarlyReturnInsideWindow) {
    BlockPool pool(32);
    SwaManager mgr(pool, 4, 16);  // big window
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));  // 2 pages, 8 tokens <= window
    std::int32_t free_before = pool.NumFreeBlocks();
    mgr.AdvanceWindow(table, 8);  // skipped = 8 - 16 + 1 < 0 -> early return
    EXPECT_FALSE(table.Blocks()[0]->IsNull());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(SwaManagerTest, AdvanceWindowCapsToAllocatedBlocks) {
    BlockPool pool(32);
    SwaManager mgr(pool, 4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));  // 2 pages
    // Huge num_computed_tokens -> skipped_blocks would exceed NumBlocks(); must
    // cap and not go out of bounds.
    mgr.AdvanceWindow(table, 1000);
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    EXPECT_TRUE(table.Blocks()[1]->IsNull());
    EXPECT_EQ(table.NumBlocks(), 2);  // still 2 slots, both null
}

TEST(SwaManagerTest, AdvanceWindowEvictsFirstSlidOutFirst) {
    // page_size 2, window 4. Free pages 0 and 1, then the next allocation should
    // reuse page 0 (the first slid out) before page 1.
    // Pool is sized to exactly fit the 4 acquired pages (+1 null block) so the
    // free list is empty after Acquire; the only blocks the next AllocateBlocks
    // can hand back are the just-freed pages. This exposes the FIFO order *among*
    // the freed batch -- with a larger pool, pre-existing free blocks at the LRU
    // head would be handed out first and the FIFO-among-freed property would be
    // unobservable (freed blocks return to the free-list tail, reused last).
    BlockPool pool(5);
    SwaManager mgr(pool, 2, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));  // 4 pages
    CacheBlock* p0 = table.Blocks()[0];
    CacheBlock* p1 = table.Blocks()[1];
    mgr.AdvanceWindow(table, 8);  // skipped 5, blocks 2 -> free pages 0,1
    ASSERT_TRUE(table.Blocks()[0]->IsNull());
    ASSERT_TRUE(table.Blocks()[1]->IsNull());

    // Next single allocation reuses the first-slid-out page (p0).
    std::vector<CacheBlock*> reused = pool.AllocateBlocks(1);
    ASSERT_EQ(reused.size(), 1u);
    EXPECT_EQ(reused.front()->BlockId(), p0->BlockId());
    (void)p1;
}

TEST(SwaManagerTest, AdvanceWindowFreedCachedPageStaysPrefixReusable) {
    BlockPool pool(32);
    SwaManager mgr(pool, 2, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));  // 4 pages
    const std::string h0 = RealKey({1, 1}, 0);
    // Register page 0's content so it carries a hash.
    mgr.CacheFullBlocks(table, std::vector<std::string>{h0}, 1);
    CacheBlock* p0 = table.Blocks()[0];
    EXPECT_TRUE(p0->IsCached());

    mgr.AdvanceWindow(table, 8);  // frees pages 0,1; p0 returns with hash intact
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    // The freed-but-cached page is still prefix-hittable.
    EXPECT_EQ(pool.GetCachedBlock(h0), p0);
}

TEST(SwaManagerTest, AdvanceWindowLeavesTailAvailUnchanged) {
    BlockPool pool(32);
    SwaManager mgr(pool, 4, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 10));  // 3 pages, last partial: tail_avail = 2
    EXPECT_EQ(table.TailAvailableTokens(), 2);
    mgr.AdvanceWindow(table, 10);  // skipped 7, blocks 1 -> frees front full page
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    EXPECT_EQ(table.TailAvailableTokens(), 2);  // tail untouched
}

TEST(SwaManagerTest, AcquireAdvancePairingKeepsPhysicalPagesBounded) {
    // Steady state: repeatedly Acquire one page worth of tokens then AdvanceWindow.
    // page_size 2, window 4 -> active physical pages should stay bounded around
    // ceil(window/page_size)=2, not grow without limit.
    BlockPool pool(64);
    SwaManager mgr(pool, 2, 4);
    BlockTable table;
    std::int32_t n = 0;
    std::int32_t baseline_free = pool.NumFreeBlocks();
    for (int step = 0; step < 20; ++step) {
        n += 2;                          // two new tokens -> one new page
        ASSERT_TRUE(mgr.Acquire(table, 2));
        mgr.AdvanceWindow(table, n);
    }
    // Active (non-free) physical pages = baseline_free - current_free. With a
    // 4-token window over page_size 2, at most ~2-3 pages are ever live.
    std::int32_t active = baseline_free - pool.NumFreeBlocks();
    EXPECT_LE(active, 3);
    // The table itself grows (holes accumulate), but physical pages are bounded.
    EXPECT_GT(table.NumBlocks(), 3);
}

}  // namespace
}  // namespace tokenspeed::test
