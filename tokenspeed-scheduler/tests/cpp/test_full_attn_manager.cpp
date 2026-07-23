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

class FullAttnManager : public ::tokenspeed::FullAttnManager {
public:
    using ::tokenspeed::FullAttnManager::CacheBlock;
    using ::tokenspeed::FullAttnManager::CacheFullBlocks;
    using ::tokenspeed::FullAttnManager::FullAttnManager;
    using ::tokenspeed::FullAttnManager::Match;

    PrefixMatch Match(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) {
        return ::tokenspeed::FullAttnManager::Match(pool, keys, begin_blocks, max_blocks, recency_);
    }
    void CacheBlock(BlockPool& pool, CacheBlockRef& block, const std::string& key) {
        ::tokenspeed::FullAttnManager::CacheBlock(pool, block, key, recency_);
    }
    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const std::string> keys,
                         std::int32_t first_slot = 0) {
        ::tokenspeed::FullAttnManager::CacheFullBlocks(pool, table, keys, recency_, first_slot);
    }

private:
    std::uint64_t recency_{0};
};

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
    EXPECT_EQ(pool.NumFreeBlocks(), 8);  // nothing claimed
}

TEST(FullAttnManagerTest, MatchStopsAtFirstMiss) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({5, 6, 7, 8}, 0);
    const std::string k2 = RealKey({9, 9, 9, 9}, 0);

    CacheBlockRef a = pool.AcquireBlock();
    const std::int32_t a_id = a->Location().lcm_block_id;
    mgr.CacheBlock(pool, a, k0);
    CacheBlockRef b = pool.AcquireBlock();
    const std::int32_t b_id = b->Location().lcm_block_id;
    mgr.CacheBlock(pool, b, k1);
    a.reset();
    b.reset();

    std::vector<std::string> keys{k0, k1, k2};
    PrefixMatch m = mgr.Match(pool, keys, 0, 3);
    EXPECT_EQ(m.num_hit_blocks, 2);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->Location().lcm_block_id, a_id);
    EXPECT_EQ(m.blocks[1]->Location().lcm_block_id, b_id);
}

TEST(FullAttnManagerTest, MatchPinsUntilResultDies) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    CacheBlockRef a = pool.AcquireBlock();
    mgr.CacheBlock(pool, a, k0);
    a.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.num_hit_blocks, 1);
    EXPECT_EQ(m.blocks.front().use_count(), 2);  // Manager cache owner + match
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
    m = {};
    EXPECT_EQ(pool.NumFreeBlocks(), 7);  // Manager cache owner retains the LCM block
}

TEST(FullAttnManagerTest, ClaimHitBlocksClaimsAndAppends) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    CacheBlockRef a = pool.AcquireBlock();
    const std::int32_t id = a->Location().lcm_block_id;
    mgr.CacheBlock(pool, a, k0);
    a.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), 7);

    std::vector<std::string> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));

    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.Blocks()[0]->Location().lcm_block_id, id);
    EXPECT_EQ(table.Blocks()[0].use_count(), 2);  // Manager cache owner + request
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
    EXPECT_EQ(table.TailAvailableTokens(), 0);  // hit pages are full
}

TEST(FullAttnManagerTest, ClaimNoHitsIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    PrefixMatch empty;
    mgr.ClaimHitBlocks(table, std::move(empty));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 8);
}

TEST(FullAttnManagerTest, AcquireFillsTailBeforeAllocating) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
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
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
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
    EXPECT_EQ(pool.NumFreeBlocks(), 8);
}

TEST(FullAttnManagerTest, AcquireAllOrNothingOnShortage) {
    BlockPool pool(2);
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
    EXPECT_EQ(m.blocks[0]->Location().lcm_block_id, a.Blocks()[0]->Location().lcm_block_id);
    EXPECT_EQ(m.blocks[1]->Location().lcm_block_id, a.Blocks()[1]->Location().lcm_block_id);
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
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[0]->Location()));
    EXPECT_FALSE(mgr.ContainsCachedBlock(pool, a.Blocks()[1]->Location()));
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

    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[0]->Location()));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[1]->Location()));
    std::vector<std::string> keys{k0, k1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    EXPECT_EQ(m.num_hit_blocks, 2);
}

TEST(FullAttnManagerTest, FreeReturnsPagesAndClearsTable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages
    EXPECT_EQ(pool.NumFreeBlocks(), 6);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 8);  // all returned
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
    CacheBlockRef a = pool.AcquireBlock();
    mgr.CacheBlock(pool, a, k0);
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
    EXPECT_FALSE(mgr.ContainsCachedBlock(pool, a.Blocks()[0]->Location()));
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

TEST(FullAttnManagerLcmTest, ManagerOnlyCacheOwnerRetainsChild) {
    BlockPool pool(1);
    FullAttnManager mgr(/*cache_block_tokens=*/4, /*cache_blocks_per_lcm_block=*/2, /*group_id=*/0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    const CacheBlockLocation location = table.Blocks().front()->Location();
    const std::string key = RealKey({1, 2, 3, 4}, 0);
    std::uint64_t recency = 0;
    mgr.CacheFullBlocks(pool, table, std::vector<std::string>{key}, recency);

    mgr.Free(table);

    EXPECT_TRUE(pool.IsOccupied(location));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, key));
    EXPECT_TRUE(mgr.IsCachedBlockFree(pool, key));
}

TEST(FullAttnManagerLcmTest, RequestOnlyUniqueChildIsNotCacheEvictable) {
    BlockPool pool(1);
    FullAttnManager mgr(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    ASSERT_TRUE(table.Blocks().front().unique());

    EXPECT_TRUE(mgr.EvictableBlockLocations(pool).empty());
}

TEST(FullAttnManagerLcmTest, ChildEvictionLeavesSiblingLocationValid) {
    BlockPool pool(1);
    FullAttnManager mgr(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));
    const std::string first_key = RealKey({1, 2, 3, 4}, 0);
    const std::string second_key = RealKey({5, 6, 7, 8}, 0);
    const CacheBlockLocation sibling = table.Blocks()[1]->Location();
    std::uint64_t recency = 0;
    mgr.CacheFullBlocks(pool, table, std::vector<std::string>{first_key, second_key}, recency);
    mgr.Free(table);

    EXPECT_TRUE(mgr.EvictCachedBlock(pool, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));

    EXPECT_FALSE(mgr.ContainsCachedBlock(pool, first_key));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, second_key));
    EXPECT_TRUE(pool.IsOccupied(sibling));
}

TEST(FullAttnManagerLcmTest, PinnedChildBlocksWholeParentEviction) {
    BlockPool pool(1);
    FullAttnManager mgr(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));
    std::uint64_t recency = 0;
    mgr.CacheFullBlocks(pool, table, std::vector<std::string>{RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)},
                        recency);

    EXPECT_FALSE(mgr.ParentIsFullyEvictable(pool, 1));
    EXPECT_FALSE(mgr.EvictParent(pool, 1));
    mgr.Free(table);
    EXPECT_TRUE(mgr.ParentIsFullyEvictable(pool, 1));
}

TEST(FullAttnManagerLcmTest, CrossGroupRebindRequiresErasingEveryChildEntry) {
    BlockPool pool(1);
    FullAttnManager first_group(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(first_group.Acquire(pool, table, 8));
    std::uint64_t recency = 0;
    first_group.CacheFullBlocks(pool, table,
                                std::vector<std::string>{RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)}, recency);
    first_group.Free(table);

    ASSERT_TRUE(first_group.EvictParent(pool, 1));
    ASSERT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/8);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<GroupId>{1});
}

TEST(FullAttnManagerLcmTest, ParentVictimsUseMaxChildRecencyThenFewerChildren) {
    BlockPool pool(3);
    FullAttnManager mgr(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 20));
    const std::vector<std::string> keys{
        RealKey({1, 1, 1, 1}, 0), RealKey({2, 2, 2, 2}, 0), RealKey({3, 3, 3, 3}, 0),
        RealKey({4, 4, 4, 4}, 0), RealKey({5, 5, 5, 5}, 0),
    };
    std::uint64_t recency = 0;
    mgr.CacheFullBlocks(pool, table, keys, recency);
    mgr.Free(table);

    std::vector<ParentEvictionCandidate> candidates = mgr.CollectEvictableParents(pool);
    ASSERT_EQ(candidates.size(), 3u);
    EXPECT_EQ(candidates[0].lcm_block_id, 1);
    EXPECT_EQ(candidates[0].last_access, 2);
    EXPECT_EQ(candidates[1].lcm_block_id, 2);
    EXPECT_EQ(candidates[1].last_access, 4);
    EXPECT_EQ(candidates[2].lcm_block_id, 3);
    EXPECT_EQ(candidates[2].last_access, 5);
}

TEST(FullAttnManagerLcmTest, ParentVictimTiesUseFewerChildrenThenParentId) {
    std::vector<ParentEvictionCandidate> candidates{
        {.lcm_block_id = 2, .last_access = 10, .occupied_count = 2},
        {.lcm_block_id = 3, .last_access = 10, .occupied_count = 1},
        {.lcm_block_id = 1, .last_access = 10, .occupied_count = 2},
    };
    std::ranges::sort(candidates);

    EXPECT_EQ(candidates[0].lcm_block_id, 3);
    EXPECT_EQ(candidates[1].lcm_block_id, 1);
    EXPECT_EQ(candidates[2].lcm_block_id, 2);
}

TEST(FullAttnManagerLcmTest, DuplicateRegistrationCanonicalizesAndTouchesEntry) {
    BlockPool pool(2);
    FullAttnManager mgr(4, 2, 0);
    BlockTable first;
    BlockTable other;
    BlockTable duplicate;
    ASSERT_TRUE(mgr.Acquire(pool, first, 4));
    ASSERT_TRUE(mgr.Acquire(pool, other, 4));
    ASSERT_TRUE(mgr.Acquire(pool, duplicate, 4));
    const std::string key = RealKey({1, 2, 3, 4}, 0);
    const std::string other_key = RealKey({5, 6, 7, 8}, 0);
    std::uint64_t recency = 0;
    mgr.CacheFullBlocks(pool, first, std::vector<std::string>{key}, recency);
    mgr.CacheFullBlocks(pool, other, std::vector<std::string>{other_key}, recency);
    const CacheBlockLocation first_location = first.Blocks()[0]->Location();
    const CacheBlockLocation other_location = other.Blocks()[0]->Location();

    mgr.CacheFullBlocks(pool, duplicate, std::vector<std::string>{key}, recency);
    mgr.Free(first);
    mgr.Free(other);
    mgr.Free(duplicate);

    EXPECT_EQ(mgr.NumCachedBlocks(pool), 2);
    EXPECT_EQ(mgr.EvictableBlockLocations(pool), (std::vector<CacheBlockLocation>{other_location, first_location}));
    EXPECT_EQ(pool.NumOccupiedSlots(), 2);
}

TEST(FullAttnManagerLcmTest, LocationBasedEvictionIsScopedToItsPool) {
    BlockPool device_pool(1);
    BlockPool host_pool(1);
    FullAttnManager mgr(4, 1, 0);
    CacheBlockRef device = device_pool.AcquireBlock();
    CacheBlockRef host = host_pool.AcquireBlock();
    const CacheBlockLocation shared_location = device->Location();
    ASSERT_EQ(host->Location(), shared_location);
    const std::string device_key = RealKey({1, 2, 3, 4}, 0);
    const std::string host_key = RealKey({5, 6, 7, 8}, 0);
    std::uint64_t recency = 0;
    mgr.CacheBlock(device_pool, device, device_key, recency);
    mgr.CacheBlock(host_pool, host, host_key, recency);
    device.reset();
    host.reset();

    EXPECT_TRUE(mgr.EvictCachedBlock(device_pool, shared_location));
    EXPECT_FALSE(mgr.ContainsCachedBlock(device_pool, device_key));
    EXPECT_TRUE(mgr.ContainsCachedBlock(host_pool, host_key));
    EXPECT_TRUE(host_pool.IsOccupied(shared_location));
}

}  // namespace
}  // namespace tokenspeed::test
