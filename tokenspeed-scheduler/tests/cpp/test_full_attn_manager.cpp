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

#include "cache/core/block_pool.h"
#include "cache/manager/full_attn_manager.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

using token_span = std::span<const std::int32_t>;

// A real key from page_hasher.h, not a synthetic placeholder.
CacheKey RealKey(const std::vector<std::int32_t>& tokens, std::uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> hashes = ComputePagedHashes(pages, "");
    return CacheKey{.group_id = group_id, .content_hash = std::move(hashes.front())};
}

class FullAttnManager : public ::tokenspeed::FullAttnManager {
public:
    using ::tokenspeed::FullAttnManager::CacheFullBlocks;
    using ::tokenspeed::FullAttnManager::FullAttnManager;
    using ::tokenspeed::FullAttnManager::RegisterCachedBlock;

    PrefixMatch Match(BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks) {
        return AcquireMatchedBlocks(pool, keys, begin_blocks, Probe(pool, keys, begin_blocks, max_blocks),
                                    ++next_access_epoch_);
    }
    void RegisterCachedBlock(BlockPool& pool, CacheBlockRef& block, const CacheKey& key) {
        ::tokenspeed::FullAttnManager::RegisterCachedBlock(pool, block, key, ++next_access_epoch_);
    }
    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const CacheKey> keys,
                         std::int32_t first_slot = 0) {
        ::tokenspeed::FullAttnManager::CacheFullBlocks(pool, table, keys, ++next_access_epoch_, first_slot);
    }

private:
    std::uint64_t next_access_epoch_{0};
};

TEST(FullAttnManagerTest, ConstructsWithPageSize) {
    BlockPool pool(8);
    FullAttnManager mgr(/*block_size=*/4);
    BlockTable table;
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.AvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
}

TEST(FullAttnManagerTest, MatchEmptyListReturnsNoHit) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    std::vector<CacheKey> empty_hashes;
    PrefixMatch m = mgr.Match(pool, empty_hashes, 0, static_cast<std::int32_t>(empty_hashes.size()));
    EXPECT_EQ(m.NumHitBlocks(), 0);
    EXPECT_TRUE(m.blocks.empty());
}

TEST(FullAttnManagerTest, MatchAllMissReturnsNoHitAndDoesNotChangeRefs) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    std::vector<CacheKey> hashes = {RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)};
    PrefixMatch m = mgr.Match(pool, hashes, 0, static_cast<std::int32_t>(hashes.size()));
    EXPECT_EQ(m.NumHitBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);  // nothing claimed
}

TEST(FullAttnManagerTest, ProbeAcceptsTypedCacheKeys) {
    BlockPool pool(8);
    ::tokenspeed::FullAttnManager mgr(4);
    const std::vector<CacheKey> keys{
        CacheKey{.group_id = 0, .content_hash = "hash"},
    };

    const GroupPrefixProbe probe = mgr.Probe(pool, keys, /*begin_blocks=*/0, /*max_blocks=*/1);
    EXPECT_TRUE(probe.hits.empty());
}

TEST(FullAttnManagerTest, MatchStopsAtFirstMiss) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    const CacheKey k1 = RealKey({5, 6, 7, 8}, 0);
    const CacheKey k2 = RealKey({9, 9, 9, 9}, 0);

    CacheBlockRef a = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const std::int32_t a_id = a->Location().lcm_block_id;
    mgr.RegisterCachedBlock(pool, a, k0);
    CacheBlockRef b = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const std::int32_t b_id = b->Location().lcm_block_id;
    mgr.RegisterCachedBlock(pool, b, k1);
    a.reset();
    b.reset();

    std::vector<CacheKey> keys{k0, k1, k2};
    PrefixMatch m = mgr.Match(pool, keys, 0, 3);
    EXPECT_EQ(m.NumHitBlocks(), 2);
    ASSERT_EQ(m.blocks.size(), 2u);
    EXPECT_EQ(m.blocks[0]->Location().lcm_block_id, a_id);
    EXPECT_EQ(m.blocks[1]->Location().lcm_block_id, b_id);
}

TEST(FullAttnManagerTest, MatchPinsUntilResultDies) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    CacheBlockRef a = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    mgr.RegisterCachedBlock(pool, a, k0);
    a.reset();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);

    std::vector<CacheKey> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.NumHitBlocks(), 1);
    EXPECT_EQ(m.blocks.front().use_count(), 2);  // Manager cache owner + match
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);
    m = {};
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);  // Manager cache owner retains the LCM block
}

TEST(FullAttnManagerTest, ClaimHitBlocksClaimsAndAppends) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    CacheBlockRef a = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const std::int32_t id = a->Location().lcm_block_id;
    mgr.RegisterCachedBlock(pool, a, k0);
    a.reset();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);

    std::vector<CacheKey> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));

    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.Blocks()[0]->Location().lcm_block_id, id);
    EXPECT_EQ(table.Blocks()[0].use_count(), 2);  // Manager cache owner + request
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);
    EXPECT_EQ(table.AvailableTokens(), 0);  // hit pages are full
}

TEST(FullAttnManagerTest, ClaimNoHitsIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    PrefixMatch empty;
    mgr.ClaimHitBlocks(table, std::move(empty));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
}

TEST(FullAttnManagerTest, AcquireFillsTailBeforeAllocating) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.AvailableTokens(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);
}

TEST(FullAttnManagerTest, AcquirePartialPageLeavesTailRoom) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.AvailableTokens(), 1);
}

TEST(FullAttnManagerTest, AcquireCanReserveFutureTokens) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, /*num_tokens=*/1, /*reserve_tokens=*/8));
    EXPECT_EQ(table.NumBlocks(), 3);
    EXPECT_EQ(table.AvailableTokens(), 11);

    mgr.ConsumeReservedTokens(table, /*num_tokens=*/8);
    EXPECT_EQ(table.AvailableTokens(), 3);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 5);
}

TEST(FullAttnManagerTest, AcquireUsesTailRoomWithoutNewPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));  // 1 page, tail_avail 1
    ASSERT_TRUE(mgr.Acquire(pool, table, 1));  // fits in tail -> no new page
    EXPECT_EQ(table.NumBlocks(), 1);
    EXPECT_EQ(table.AvailableTokens(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 7);
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
    EXPECT_EQ(table.AvailableTokens(), 3);
}

TEST(FullAttnManagerTest, AcquireZeroTokensIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 0));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
}

TEST(FullAttnManagerTest, AcquireAllOrNothingOnShortage) {
    BlockPool pool(2);
    FullAttnManager mgr(4);
    BlockTable table;

    // Need ceil(12/4) = 3 pages but only 2 free -> must fail and roll back.
    EXPECT_FALSE(mgr.Acquire(pool, table, 12));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.AvailableTokens(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);  // nothing consumed
}

TEST(FullAttnManagerTest, CacheFullBlocksMakesPagesPrefixHittable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    const CacheKey k1 = RealKey({5, 6, 7, 8}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 8));
    ASSERT_EQ(a.NumBlocks(), 2);
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0, k1});

    std::vector<CacheKey> keys{k0, k1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    EXPECT_EQ(m.NumHitBlocks(), 2);
    EXPECT_EQ(m.blocks[0]->Location().lcm_block_id, a.Blocks()[0]->Location().lcm_block_id);
    EXPECT_EQ(m.blocks[1]->Location().lcm_block_id, a.Blocks()[1]->Location().lcm_block_id);
}

TEST(FullAttnManagerTest, CacheFullBlocksSkipsTailPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);

    // 6 tokens -> 2 pages, second page is a partial tail (only 2 of 4 used).
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 6));
    ASSERT_EQ(a.NumBlocks(), 2);
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0});

    std::vector<CacheKey> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.NumHitBlocks(), 1);
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[0]->Location()));
    EXPECT_FALSE(mgr.ContainsCachedBlock(pool, a.Blocks()[1]->Location()));
}

TEST(FullAttnManagerTest, CacheFullBlocksIsIdempotentAcrossCalls) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    const CacheKey k1 = RealKey({5, 6, 7, 8}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0});      // page 0 cached
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));                         // grow to page 1
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0, k1});  // must skip already-cached page 0

    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[0]->Location()));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, a.Blocks()[1]->Location()));
    std::vector<CacheKey> keys{k0, k1};
    PrefixMatch m = mgr.Match(pool, keys, 0, 2);
    EXPECT_EQ(m.NumHitBlocks(), 2);
}

TEST(FullAttnManagerTest, FreeReturnsPagesAndClearsTable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));  // 2 pages
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 6);

    mgr.Free(table);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.AvailableTokens(), 0);
    EXPECT_TRUE(table.Blocks().empty());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);  // all returned
}

TEST(FullAttnManagerTest, FreedCachedPageStaysPrefixReusable) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0});
    mgr.Free(a);

    std::vector<CacheKey> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    EXPECT_EQ(m.NumHitBlocks(), 1);
}

TEST(FullAttnManagerTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(16);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    const CacheKey k1 = RealKey({5, 6, 7, 8}, 0);

    // Request A: cold.
    {
        std::vector<CacheKey> keys{k0, k1};
        PrefixMatch m = mgr.Match(pool, keys, 0, 2);
        EXPECT_EQ(m.NumHitBlocks(), 0);
        BlockTable a;
        mgr.ClaimHitBlocks(a, std::move(m));
        ASSERT_TRUE(mgr.Acquire(pool, a, 8));
        mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{k0, k1});
        mgr.Free(a);
    }

    // Request B: shares the prefix.
    {
        std::vector<CacheKey> keys{k0, k1};
        PrefixMatch m = mgr.Match(pool, keys, 0, 2);
        EXPECT_EQ(m.NumHitBlocks(), 2);
        BlockTable b;
        mgr.ClaimHitBlocks(b, std::move(m));
        EXPECT_EQ(b.NumBlocks(), 2);
        std::int32_t free_before = pool.NumEmptyLcmBlocks();
        ASSERT_TRUE(mgr.Acquire(pool, b, 0));  // no new tokens beyond the hit prefix
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), free_before);
        mgr.Free(b);
    }
}

TEST(FullAttnManagerTest, RejectsKeyForAnotherGroup) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey g0 = RealKey({1, 2, 3, 4}, 0);
    const CacheKey g1 = RealKey({1, 2, 3, 4}, 1);  // same tokens, group 1
    ASSERT_NE(g0, g1);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    mgr.CacheFullBlocks(pool, a, std::vector<CacheKey>{g0});

    std::vector<CacheKey> keys_g0{g0};
    std::vector<CacheKey> keys_g1{g1};
    EXPECT_EQ(mgr.Match(pool, keys_g0, 0, 1).NumHitBlocks(), 1);
    EXPECT_THROW(mgr.Match(pool, keys_g1, 0, 1), std::runtime_error);
}

// Claimed full pages carry no available capacity: the next Acquire must start a fresh
// page, not consume phantom tail room.
TEST(FullAttnManagerTest, ClaimThenAcquireStartsFreshPage) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    const CacheKey k0 = RealKey({1, 2, 3, 4}, 0);
    CacheBlockRef a = pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    mgr.RegisterCachedBlock(pool, a, k0);
    a.reset();

    std::vector<CacheKey> keys{k0};
    PrefixMatch m = mgr.Match(pool, keys, 0, 1);
    BlockTable table;
    mgr.ClaimHitBlocks(table, std::move(m));
    ASSERT_EQ(table.NumBlocks(), 1);
    ASSERT_EQ(table.AvailableTokens(), 0);

    ASSERT_TRUE(mgr.Acquire(pool, table, 3));
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(table.AvailableTokens(), 1);
}

TEST(FullAttnManagerTest, CacheFullBlocksZeroIsNoOp) {
    BlockPool pool(8);
    FullAttnManager mgr(4);
    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 4));
    std::vector<CacheKey> no_hashes;
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
    std::vector<CacheKey> keys_a;
    for (std::string& hash : ComputePagedHashes(pages_a, "")) {
        keys_a.push_back(CacheKey{.group_id = 0, .content_hash = std::move(hash)});
    }
    std::vector<CacheKey> keys_b;
    for (std::string& hash : ComputePagedHashes(pages_b, "")) {
        keys_b.push_back(CacheKey{.group_id = 0, .content_hash = std::move(hash)});
    }
    ASSERT_EQ(keys_a.size(), 2u);
    ASSERT_EQ(keys_b.size(), 2u);
    EXPECT_NE(keys_a[1], keys_b[1]);

    BlockTable a;
    ASSERT_TRUE(mgr.Acquire(pool, a, 8));
    mgr.CacheFullBlocks(pool, a, keys_a);

    PrefixMatch miss = mgr.Match(pool, keys_b, 0, static_cast<std::int32_t>(keys_b.size()));
    EXPECT_EQ(miss.NumHitBlocks(), 0);

    PrefixMatch hit = mgr.Match(pool, keys_a, 0, static_cast<std::int32_t>(keys_a.size()));
    EXPECT_EQ(hit.NumHitBlocks(), 2);
}

TEST(FullAttnManagerLcmTest, ManagerOnlyCacheOwnerRetainsChild) {
    BlockPool pool(1);
    FullAttnManager mgr(/*cache_block_tokens=*/4, /*cache_blocks_per_lcm_block=*/2, /*group_id=*/0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4));
    const CacheBlockLocation location = table.Blocks().front()->Location();
    const CacheKey key = RealKey({1, 2, 3, 4}, 0);
    const std::uint64_t access_epoch = 1;
    mgr.CacheFullBlocks(pool, table, std::vector<CacheKey>{key}, access_epoch);

    mgr.Free(table);

    EXPECT_TRUE(pool.IsOccupied(location));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, key));
    EXPECT_EQ(mgr.EvictableBlockLocations(pool), std::vector<CacheBlockLocation>{location});
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
    const CacheKey first_key = RealKey({1, 2, 3, 4}, 0);
    const CacheKey second_key = RealKey({5, 6, 7, 8}, 0);
    const CacheBlockLocation sibling = table.Blocks()[1]->Location();
    const std::uint64_t access_epoch = 1;
    mgr.CacheFullBlocks(pool, table, std::vector<CacheKey>{first_key, second_key}, access_epoch);
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
    const std::uint64_t access_epoch = 1;
    mgr.CacheFullBlocks(pool, table, std::vector<CacheKey>{RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)},
                        access_epoch);

    EXPECT_FALSE(mgr.ParentIsFullyEvictable(pool, 1));
    mgr.Free(table);
    EXPECT_TRUE(mgr.ParentIsFullyEvictable(pool, 1));
}

TEST(FullAttnManagerLcmTest, CrossGroupRebindRequiresErasingEveryChildEntry) {
    BlockPool pool(1);
    FullAttnManager first_group(4, 2, 0);
    BlockTable table;
    ASSERT_TRUE(first_group.Acquire(pool, table, 8));
    const std::uint64_t access_epoch = 1;
    first_group.CacheFullBlocks(pool, table, std::vector<CacheKey>{RealKey({1, 2, 3, 4}, 0), RealKey({5, 6, 7, 8}, 0)},
                                access_epoch);
    first_group.Free(table);

    ASSERT_TRUE(first_group.EvictCachedBlock(pool, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));
    ASSERT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    ASSERT_TRUE(first_group.EvictCachedBlock(pool, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 1}));
    ASSERT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/8);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{1});
}

TEST(FullAttnManagerLcmTest, DuplicateRegistrationUpdatesEpochWithoutReorderingEntries) {
    BlockPool pool(2);
    FullAttnManager mgr(4, 2, 0);
    BlockTable first;
    BlockTable other;
    BlockTable duplicate;
    ASSERT_TRUE(mgr.Acquire(pool, first, 4));
    ASSERT_TRUE(mgr.Acquire(pool, other, 4));
    ASSERT_TRUE(mgr.Acquire(pool, duplicate, 4));
    const CacheKey key = RealKey({1, 2, 3, 4}, 0);
    const CacheKey other_key = RealKey({5, 6, 7, 8}, 0);
    std::uint64_t next_access_epoch = 0;
    mgr.CacheFullBlocks(pool, first, std::vector<CacheKey>{key}, ++next_access_epoch);
    mgr.CacheFullBlocks(pool, other, std::vector<CacheKey>{other_key}, ++next_access_epoch);
    const CacheBlockLocation first_location = first.Blocks()[0]->Location();
    const CacheBlockLocation other_location = other.Blocks()[0]->Location();

    mgr.CacheFullBlocks(pool, duplicate, std::vector<CacheKey>{key}, ++next_access_epoch);
    mgr.Free(first);
    mgr.Free(other);
    mgr.Free(duplicate);

    EXPECT_EQ(mgr.NumCachedBlocks(pool), 2);
    EXPECT_EQ(mgr.EvictableBlockLocations(pool), (std::vector<CacheBlockLocation>{first_location, other_location}));
    const std::optional<KvCacheManager::CachedBlockMetadata> metadata =
        mgr.CachedBlockMetadataFor(pool, first_location);
    ASSERT_TRUE(metadata);
    EXPECT_EQ(metadata->last_access_epoch, next_access_epoch);
    EXPECT_EQ(pool.NumOccupiedSlots(), 2);
}

TEST(FullAttnManagerLcmTest, NamespaceIsPartOfPrefixIndex) {
    BlockPool pool(2);
    FullAttnManager mgr(4, 1, 0);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8));
    const CacheKey first{.namespace_id = 1, .group_id = 0, .content_hash = "shared-content"};
    const CacheKey second{.namespace_id = 2, .group_id = 0, .content_hash = "shared-content"};
    const std::uint64_t access_epoch = 1;

    mgr.CacheFullBlocks(pool, table, std::vector<CacheKey>{first, second}, access_epoch);

    EXPECT_EQ(mgr.NumCachedBlocks(pool), 2);
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, first));
    EXPECT_TRUE(mgr.ContainsCachedBlock(pool, second));
    EXPECT_NE(table.Blocks()[0]->Location(), table.Blocks()[1]->Location());
}

TEST(FullAttnManagerLcmTest, LocationBasedEvictionIsScopedToItsPool) {
    BlockPool device_pool(1);
    BlockPool host_pool(1);
    FullAttnManager mgr(4, 1, 0);
    CacheBlockRef device = device_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    CacheBlockRef host = host_pool.AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
    const CacheBlockLocation shared_location = device->Location();
    ASSERT_EQ(host->Location(), shared_location);
    const CacheKey device_key = RealKey({1, 2, 3, 4}, 0);
    const CacheKey host_key = RealKey({5, 6, 7, 8}, 0);
    std::uint64_t next_access_epoch = 0;
    mgr.RegisterCachedBlock(device_pool, device, device_key, ++next_access_epoch);
    mgr.RegisterCachedBlock(host_pool, host, host_key, ++next_access_epoch);
    device.reset();
    host.reset();

    EXPECT_TRUE(mgr.EvictCachedBlock(device_pool, shared_location));
    EXPECT_FALSE(mgr.ContainsCachedBlock(device_pool, device_key));
    EXPECT_TRUE(mgr.ContainsCachedBlock(host_pool, host_key));
    EXPECT_TRUE(host_pool.IsOccupied(shared_location));
}

}  // namespace
}  // namespace tokenspeed::test
