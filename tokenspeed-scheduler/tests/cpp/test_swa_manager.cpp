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

}  // namespace
}  // namespace tokenspeed::test
