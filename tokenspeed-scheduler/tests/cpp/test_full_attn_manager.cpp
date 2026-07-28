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
#include "cache/full_attn_manager.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

// A real key from page_hasher.h, not a synthetic placeholder.
std::string RealKey(const std::vector<std::int32_t>& tokens, uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> keys = ComputePagedHashesWithGroup(pages, "", group_id);
    return keys.front();
}

TEST(FullAttnManagerTest, ConstructsWithPageSize) {
    BlockPool pool(8);
    FullAttnManager mgr(/*block_size=*/4);
    BlockTable table;
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
}

TEST(FullAttnManagerTest, MatchEmptyListReturnsNoHit) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    std::vector<std::string> empty_hashes;
    PrefixMatch m = mgr.Match(pool, empty_hashes, 0, static_cast<std::int32_t>(empty_hashes.size()));
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_TRUE(m.blocks.empty());
}

TEST(FullAttnManagerTest, MatchAllMissReturnsNoHitAndDoesNotChangeRefs) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    std::vector<std::string> hashes = {RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)};
    PrefixMatch m = mgr.Match(pool, hashes, 0, static_cast<std::int32_t>(hashes.size()));
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // nothing claimed
}

TEST(FullAttnManagerTest, MatchStopsAtFirstMiss) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);
    const std::string k2 = RealKey({9, 9, 9, 9}, 0);

    BlockRef a = pool.AcquireBlock();
    const std::int32_t a_id = a->BlockId();
    pool.CacheFullBlock(a, k0);
    BlockRef b = pool.AcquireBlock();
    const std::int32_t b_id = b->BlockId();
    pool.CacheFullBlock(b, k1);
    a.reset();
    b.reset();

    std::vector<std::string> keys{k0, k1, k2};
    PrefixMatch m = mgr.Match(pool, keys, 0, 3);
    EXPECT_EQ(m.num_hit_blocks, 2);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), a_id);
    EXPECT_EQ(m.blocks[1]->BlockId(), b_id);
}

TEST(FullAttnManagerTest, MatchPinsUntilResultDies) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    BlockRef a = pool.AcquireBlock();
    pool.CacheFullBlock(a, k0);
    a.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(m.blocks.front().use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
    m = {};
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(FullAttnManagerTest, ClaimHitBlocksClaimsAndAppends) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    BlockRef a = pool.AcquireBlock();
    const std::int32_t id = a->BlockId();
    pool.CacheFullBlock(a, k0);
    a.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));

    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.Blocks()[0]->BlockId(), id);
    EXPECT_EQ(table.Blocks()[0].use_count(), 1);  // match ownership transferred
    EXPECT_EQ(pool.NumFreeBlocks(), 6);           // pulled out of free list
    EXPECT_EQ(table.TailAvailableTokens(), 0);    // hit pages are full
}

TEST(FullAttnManagerTest, ClaimNoHitsIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    PrefixMatch empty;
    mgr.ClaimHitBlocks(table, std::move(empty));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(FullAttnManagerTest, AcquireFillsTailBeforeAllocating) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(FullAttnManagerTest, AcquirePartialPageLeavesTailRoom) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 1);
}

TEST(FullAttnManagerTest, AcquireUsesTailRoomWithoutNewPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));  // 1 page, tail_avail 1
    ASSERT_TRUE(mgr.Acquire(pool, table, 1));  // fits in tail -> no new page
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(FullAttnManagerTest, AcquireSpillsAcrossMultiplePages) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 2));  // 1 page, tail_avail 2
    // 7 more tokens: 2 fill the tail, 5 remaining -> ceil(5/4) = 2 new pages.
    ASSERT_TRUE(mgr.Acquire(pool, table, 7));
    EXPECT_EQ(table.NumBlocks(), 3);
    // over = 7 - 2 = 5; used_in_tail = 5 % 4 = 1; tail_avail = 4 - 1 = 3.
    EXPECT_EQ(table.TailAvailableTokens(), 3);
}

TEST(FullAttnManagerTest, AcquireZeroTokensIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 0));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(FullAttnManagerTest, AcquireAllOrNothingOnShortage) {
    BlockPool pool(3);  // 2 usable blocks after null reservation
    FullAttnManager mgr(4);
    BlockTable table;

    // Need ceil(12/4) = 3 pages but only 2 free -> must fail and roll back.
    EXPECT_FALSE(mgr.Acquire(pool, table, 12));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 2);  // nothing consumed
}

TEST(FullAttnManagerTest, CacheFullBlocksMakesPagesPrefixHittable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 8));
    ASSERT_EQ(a.NumBlocks(), 2);
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0, k1});

    std::vector<std::string> keys{k0, k1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    EXPECT_EQ(m.num_hit_blocks, 2);
    EXPECT_EQ(m.blocks[0]->BlockId(), a.Blocks()[0]->BlockId());
    EXPECT_EQ(m.blocks[1]->BlockId(), a.Blocks()[1]->BlockId());
}

TEST(FullAttnManagerTest, CacheFullBlocksSkipsTailPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);

    // 6 tokens -> 2 pages, second page is a partial tail (only 2 of 4 used).
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 6));
    ASSERT_EQ(a.NumBlocks(), 2);
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0});

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_TRUE(a.Blocks()[0]->IsCached());
    EXPECT_FALSE(a.Blocks()[1]->IsCached());
}

TEST(FullAttnManagerTest, CacheFullBlocksIsIdempotentAcrossCalls) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0});      // page 0 cached
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));                            // grow to page 1
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0, k1});  // must skip already-cached page 0

    EXPECT_TRUE(a.Blocks()[0]->IsCached());
    EXPECT_TRUE(a.Blocks()[1]->IsCached());
    std::vector<std::string> keys{k0, k1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(FullAttnManagerTest, FreeReturnsPagesAndClearsTable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages
    EXPECT_EQ(pool.NumFreeBlocks(), 5);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // all returned
}

TEST(FullAttnManagerTest, FreedCachedPageStaysPrefixReusable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0});
    mgr.Free(a);

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
}

TEST(FullAttnManagerTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(16);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    // Request A: cold.
    {
        std::vector<std::string> keys{k0, k1};
        PrefixMatch m = mgr.Match(pool, keys, 0, 2);
        EXPECT_EQ(m.num_hit_blocks, 0);
        BlockTable a;
        mgr.ClaimHitBlocks(a, std::move(m));
        ASSERT_TRUE(mgr.Acquire(pool, a, 8));
        mgr.CacheFullBlocks(pool, a, std::vector<std::string>{k0, k1});
        mgr.Free(a);
    }

    // Request B: shares the prefix.
    {
        std::vector<std::string> keys{k0, k1};
        PrefixMatch m = mgr.Match(pool, keys, 0, 2);
        EXPECT_EQ(m.num_hit_blocks, 2);
        BlockTable b;
        mgr.ClaimHitBlocks(b, std::move(m));
        EXPECT_EQ(b.NumBlocks(), 2);
        std::int32_t free_before = pool.NumFreeBlocks();
        ASSERT_TRUE(mgr.Acquire(pool, b, 0));  // no new tokens beyond the hit prefix
        EXPECT_EQ(pool.NumFreeBlocks(), free_before);
        mgr.Free(b);
    }
}

TEST(FullAttnManagerTest, GroupIdIsolatesContent) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string g0 = RealKey({1, 2, 3, 4}, 0);
    const std::string g1 = RealKey({1, 2, 3, 4}, 1);  // same tokens, group 1
    ASSERT_NE(g0, g1);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<std::string>{g0});

    std::vector<std::string> keys_g0{g0};
    std::vector<std::string> keys_g1{g1};
    EXPECT_EQ(mgr.Match(pool, keys_g0, 0, 1).num_hit_blocks, 1);
    EXPECT_EQ(mgr.Match(pool, keys_g1, 0, 1).num_hit_blocks, 0);  // group 1 not cached
}

// Claimed full pages carry tail_avail_ 0: the next Acquire must start a fresh
// page, not consume phantom tail room.
TEST(FullAttnManagerTest, ClaimThenAcquireStartsFreshPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    BlockRef a = pool.AcquireBlock();
    pool.CacheFullBlock(a, k0);
    a.reset();

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));
    ASSERT_EQ(table.NumBlocks(), 1);
    ASSERT_EQ(table.TailAvailableTokens(), 0);

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(table.TailAvailableTokens(), 1);
}

TEST(FullAttnManagerTest, CacheFullBlocksZeroIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    std::vector<std::string> no_hashes;
    mgr.CacheFullBlocks(pool, a, no_hashes);  // nothing to register
    EXPECT_FALSE(a.Blocks()[0]->IsCached());
}

TEST(FullAttnManagerTest, ClaimHitBlocksOnNonEmptyTableAsserts) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4));  // table now non-empty
    PrefixMatch empty;
    EXPECT_THROW(mgr.ClaimHitBlocks(table, std::move(empty)), std::runtime_error);
}

// The chain links each page's key to the prior page's hash: an identical second
// page after a different first page yields a different key.
TEST(FullAttnManagerTest, ChainedPriorPreventsSecondPageCollision) {
    BlockPool pool(8);
    FullAttnManager mgr(4);

    std::vector<std::int32_t> p_a = {1, 2, 3, 4};
    std::vector<std::int32_t> p_b = {9, 9, 9, 9};
    std::vector<std::int32_t> q = {5, 6, 7, 8};  // shared second page

    std::vector<token_span> pages_a = {token_span(p_a.data(), p_a.size()), token_span(q.data(), q.size())};
    std::vector<token_span> pages_b = {token_span(p_b.data(), p_b.size()), token_span(q.data(), q.size())};
    std::vector<std::string> keys_a = ComputePagedHashesWithGroup(pages_a, "", 0);
    std::vector<std::string> keys_b = ComputePagedHashesWithGroup(pages_b, "", 0);
    ASSERT_EQ(keys_a.size(), 2u);
    ASSERT_EQ(keys_b.size(), 2u);
    EXPECT_NE(keys_a[1], keys_b[1]);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 8));
    mgr.CacheFullBlocks(pool, a, keys_a);

    PrefixMatch miss = mgr.Match(pool, keys_b, 0, static_cast<std::int32_t>(keys_b.size()));
    EXPECT_EQ(miss.num_hit_blocks, 0);

    PrefixMatch hit = mgr.Match(pool, keys_a, 0, static_cast<std::int32_t>(keys_a.size()));
    EXPECT_EQ(hit.num_hit_blocks, 2);
}

}  // namespace
}  // namespace tokenspeed::test
