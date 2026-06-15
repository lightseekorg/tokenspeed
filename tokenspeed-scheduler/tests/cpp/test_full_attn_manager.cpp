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

#include "block_pool/block_pool.h"
#include "full_attn_manager/full_attn_manager.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

// A real BlockHashWithGroupId key for a single page, so the manager is exercised
// with the exact string page_hasher.h emits (matches test_block_pool.cpp style).
std::string RealKey(const std::vector<std::int32_t>& tokens, uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> keys = ComputePagedHashesWithGroup(pages, "", group_id);
    return keys.front();
}

TEST(FullAttnManagerTest, ConstructsWithPageSize) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, /*page_size=*/4);
    BlockTable table;
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
}

TEST(FullAttnManagerTest, MatchEmptyListReturnsNoHit) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    std::vector<std::string> empty_hashes;
    PrefixMatch m = mgr.MatchPrefix(empty_hashes);
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_TRUE(m.blocks.empty());
}

TEST(FullAttnManagerTest, MatchAllMissReturnsNoHitAndDoesNotChangeRefs) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    std::vector<std::string> hashes = {RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)};
    PrefixMatch m = mgr.MatchPrefix(hashes);
    EXPECT_EQ(m.num_hit_blocks, 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // nothing claimed
}

TEST(FullAttnManagerTest, MatchStopsAtFirstMiss) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);
    const std::string k2 = RealKey({9, 9, 9, 9}, 0);

    // Cache page 0 and page 1 directly through the pool.
    auto a = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(a.front(), k0);
    auto b = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(b.front(), k1);
    pool.FreeBlocks(a);
    pool.FreeBlocks(b);

    // k2 is never cached -> match must stop after the two hits.
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0, k1, k2});
    EXPECT_EQ(m.num_hit_blocks, 2);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->BlockId(), a.front()->BlockId());
    EXPECT_EQ(m.blocks[1]->BlockId(), b.front()->BlockId());
}

TEST(FullAttnManagerTest, MatchDoesNotChangeRefCount) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    auto a = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(a.front(), k0);
    pool.FreeBlocks(a);
    EXPECT_EQ(a.front()->RefCount(), 0);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0});
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(a.front()->RefCount(), 0);  // read-only: still zero
    EXPECT_EQ(pool.NumFreeBlocks(), 7);   // still free
}

TEST(FullAttnManagerTest, ClaimHitBlocksClaimsAndAppends) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    auto a = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(a.front(), k0);
    pool.FreeBlocks(a);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0});
    BlockTable table;
    ASSERT_TRUE(mgr.ClaimHitBlocks(table, m));

    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.Blocks()[0]->BlockId(), a.front()->BlockId());
    EXPECT_EQ(a.front()->RefCount(), 1);          // claimed
    EXPECT_EQ(pool.NumFreeBlocks(), 6);           // pulled out of free list
    EXPECT_EQ(table.TailAvailableTokens(), 0);    // hit pages are full
}

TEST(FullAttnManagerTest, ClaimNoHitsIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;
    PrefixMatch empty;
    EXPECT_TRUE(mgr.ClaimHitBlocks(table, empty));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(FullAttnManagerTest, AcquireFillsTailBeforeAllocating) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;

    // First acquire of 4 tokens with page_size 4 -> exactly 1 page, tail full.
    ASSERT_TRUE(mgr.Acquire(table, 4));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(FullAttnManagerTest, AcquirePartialPageLeavesTailRoom) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;

    // 3 tokens -> 1 page, 1 token of tail room left.
    ASSERT_TRUE(mgr.Acquire(table, 3));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 1);
}

TEST(FullAttnManagerTest, AcquireUsesTailRoomWithoutNewPage) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(table, 3));   // 1 page, tail_avail 1
    ASSERT_TRUE(mgr.Acquire(table, 1));   // fits in tail -> no new page
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(FullAttnManagerTest, AcquireSpillsAcrossMultiplePages) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(table, 2));   // 1 page, tail_avail 2
    // 7 more tokens: 2 fill the tail, 5 remaining -> ceil(5/4) = 2 new pages.
    ASSERT_TRUE(mgr.Acquire(table, 7));
    EXPECT_EQ(table.NumBlocks(), 3);
    // over = 7 - 2 = 5; used_in_tail = 5 % 4 = 1; tail_avail = 4 - 1 = 3.
    EXPECT_EQ(table.TailAvailableTokens(), 3);
}

TEST(FullAttnManagerTest, AcquireZeroTokensIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 0));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(FullAttnManagerTest, AcquireAllOrNothingOnShortage) {
    BlockPool pool(3);   // 2 usable blocks after null reservation
    FullAttnManager mgr(pool, 4);
    BlockTable table;

    // Need ceil(12/4) = 3 pages but only 2 free -> must fail and roll back.
    EXPECT_FALSE(mgr.Acquire(table, 12));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 2);   // nothing consumed
}

TEST(FullAttnManagerTest, CacheFullBlocksMakesPagesPrefixHittable) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    // Request A: allocate 2 full pages worth of tokens, then cache them.
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 8));
    ASSERT_EQ(a.NumBlocks(), 2);
    mgr.CacheFullBlocks(a, std::vector<std::string>{k0, k1}, /*num_full_blocks=*/2);

    // Request B sees both as a prefix hit.
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0, k1});
    EXPECT_EQ(m.num_hit_blocks, 2);
    EXPECT_EQ(m.blocks[0]->BlockId(), a.Blocks()[0]->BlockId());
    EXPECT_EQ(m.blocks[1]->BlockId(), a.Blocks()[1]->BlockId());
}

TEST(FullAttnManagerTest, CacheFullBlocksSkipsTailPage) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);

    // 6 tokens -> 2 pages, second page is a partial tail (only 2 of 4 used).
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 6));
    ASSERT_EQ(a.NumBlocks(), 2);
    // Only the first (full) page is cached; num_full_blocks = 1.
    mgr.CacheFullBlocks(a, std::vector<std::string>{k0}, /*num_full_blocks=*/1);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0});
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_TRUE(a.Blocks()[0]->IsCached());
    EXPECT_FALSE(a.Blocks()[1]->IsCached());
}

TEST(FullAttnManagerTest, CacheFullBlocksIsIdempotentAcrossCalls) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 4));
    mgr.CacheFullBlocks(a, std::vector<std::string>{k0}, 1);        // page 0 cached
    ASSERT_TRUE(mgr.Acquire(a, 4));         // grow to page 1
    mgr.CacheFullBlocks(a, std::vector<std::string>{k0, k1}, 2);    // must skip already-cached page 0

    EXPECT_TRUE(a.Blocks()[0]->IsCached());
    EXPECT_TRUE(a.Blocks()[1]->IsCached());
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0, k1});
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(FullAttnManagerTest, FreeReturnsPagesAndClearsTable) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 8));   // 2 pages
    EXPECT_EQ(pool.NumFreeBlocks(), 5);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 7);   // all returned
}

TEST(FullAttnManagerTest, FreedCachedPageStaysPrefixReusable) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 4));
    mgr.CacheFullBlocks(a, std::vector<std::string>{k0}, 1);
    mgr.Free(a);

    // Content survives free -> still prefix-hittable.
    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0});
    EXPECT_EQ(m.num_hit_blocks, 1);
}

TEST(FullAttnManagerTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(16);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);

    // Request A: cold. No hit, allocate 2 pages, cache them, then free.
    {
        PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0, k1});
        EXPECT_EQ(m.num_hit_blocks, 0);
        BlockTable a;
        ASSERT_TRUE(mgr.ClaimHitBlocks(a, m));
        ASSERT_TRUE(mgr.Acquire(a, 8));
        mgr.CacheFullBlocks(a, std::vector<std::string>{k0, k1}, 2);
        mgr.Free(a);
    }

    // Request B: shares the [k0, k1] prefix -> 2 hits, claim them, no new alloc.
    {
        PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0, k1});
        EXPECT_EQ(m.num_hit_blocks, 2);
        BlockTable b;
        ASSERT_TRUE(mgr.ClaimHitBlocks(b, m));
        EXPECT_EQ(b.NumBlocks(), 2);
        std::int32_t free_before = pool.NumFreeBlocks();
        ASSERT_TRUE(mgr.Acquire(b, 0));  // no new tokens beyond the hit prefix
        EXPECT_EQ(pool.NumFreeBlocks(), free_before);
        mgr.Free(b);
    }
}

TEST(FullAttnManagerTest, GroupIdIsolatesContent) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string g0 = RealKey({1, 2, 3, 4}, 0);
    const std::string g1 = RealKey({1, 2, 3, 4}, 1);  // same tokens, group 1
    ASSERT_NE(g0, g1);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 4));
    mgr.CacheFullBlocks(a, std::vector<std::string>{g0}, 1);

    EXPECT_EQ(mgr.MatchPrefix(std::vector<std::string>{g0}).num_hit_blocks, 1);
    EXPECT_EQ(mgr.MatchPrefix(std::vector<std::string>{g1}).num_hit_blocks, 0);  // group 1 not cached
}

// After claiming full hit pages, tail_avail_ is 0, so the next Acquire(N>0) must
// start a FRESH page rather than consuming phantom tail room. This is the core
// claim->acquire interaction the prefill path depends on.
TEST(FullAttnManagerTest, ClaimThenAcquireStartsFreshPage) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    auto a = pool.AllocateBlocks(1);
    pool.CacheFullBlocks(a.front(), k0);
    pool.FreeBlocks(a);

    PrefixMatch m = mgr.MatchPrefix(std::vector<std::string>{k0});
    BlockTable table;
    ASSERT_TRUE(mgr.ClaimHitBlocks(table, m));
    ASSERT_EQ(table.NumBlocks(), 1);
    ASSERT_EQ(table.TailAvailableTokens(), 0);

    // 3 fresh tokens -> one NEW page (not packed into the full hit page).
    ASSERT_TRUE(mgr.Acquire(table, 3));
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(table.TailAvailableTokens(), 1);
}

TEST(FullAttnManagerTest, CacheFullBlocksZeroIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 4));
    std::vector<std::string> no_hashes;
    mgr.CacheFullBlocks(a, no_hashes, 0);   // nothing to register
    EXPECT_FALSE(a.Blocks()[0]->IsCached());
}

TEST(FullAttnManagerTest, ClaimHitBlocksOnNonEmptyTableAsserts) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(table, 4));     // table now non-empty
    PrefixMatch empty;
    EXPECT_THROW(mgr.ClaimHitBlocks(table, empty), std::runtime_error);
}

// The page-hash chain links each page's key to the prior page's hash. Two
// prefixes that share an identical SECOND page but differ in the first must
// therefore produce different second-page keys -- a request with a different
// page 1 cannot prefix-hit a cached page 2. This guards against a cross-prefix
// cache collision the single-page RealKey tests cannot reach.
TEST(FullAttnManagerTest, ChainedPriorPreventsSecondPageCollision) {
    BlockPool pool(8);
    FullAttnManager mgr(pool, 4);

    std::vector<std::int32_t> p_a = {1, 2, 3, 4};
    std::vector<std::int32_t> p_b = {9, 9, 9, 9};
    std::vector<std::int32_t> q = {5, 6, 7, 8};  // shared second page

    std::vector<token_span> pages_a = {token_span(p_a.data(), p_a.size()), token_span(q.data(), q.size())};
    std::vector<token_span> pages_b = {token_span(p_b.data(), p_b.size()), token_span(q.data(), q.size())};
    std::vector<std::string> keys_a = ComputePagedHashesWithGroup(pages_a, "", 0);
    std::vector<std::string> keys_b = ComputePagedHashesWithGroup(pages_b, "", 0);
    ASSERT_EQ(keys_a.size(), 2u);
    ASSERT_EQ(keys_b.size(), 2u);
    // Same second-page tokens, different prior -> different chained key.
    EXPECT_NE(keys_a[1], keys_b[1]);

    // Request A caches both of its pages.
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(a, 8));
    mgr.CacheFullBlocks(a, keys_a, 2);

    // Request B shares the second page's tokens but has a different first page;
    // page 1 misses, so the walk stops at zero -- it must not reach page 2.
    PrefixMatch miss = mgr.MatchPrefix(keys_b);
    EXPECT_EQ(miss.num_hit_blocks, 0);

    // Sanity: the identical prefix still hits both pages.
    PrefixMatch hit = mgr.MatchPrefix(keys_a);
    EXPECT_EQ(hit.num_hit_blocks, 2);
}

}  // namespace
}  // namespace tokenspeed::test
