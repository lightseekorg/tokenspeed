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

// ---- construction / null block -----------------------------------------

TEST(BlockPoolTest, ReservesNullBlockAndCountsFree) {
    BlockPool pool(8);
    EXPECT_EQ(pool.TotalBlocks(), 8);
    // block 0 is reserved as the null placeholder, so 7 are free.
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
    ASSERT_NE(pool.NullBlock(), nullptr);
    EXPECT_TRUE(pool.NullBlock()->IsNull());
    EXPECT_EQ(pool.NullBlock()->BlockId(), 0);
}

// ---- allocate / free lifecycle -----------------------------------------

TEST(BlockPoolTest, AllocateClaimsBlocksWithRefOne) {
    BlockPool pool(8);
    auto blocks = pool.AllocateBlocks(3);
    ASSERT_EQ(blocks.size(), 3u);
    for (CacheBlock* b : blocks) {
        EXPECT_EQ(b->RefCount(), 1);
        EXPECT_FALSE(b->IsNull());
    }
    EXPECT_EQ(pool.NumFreeBlocks(), 4);  // 7 free - 3 claimed
}

TEST(BlockPoolTest, AllocateFailsWhenCapacityShort) {
    BlockPool pool(4);  // 3 free after null reservation
    auto blocks = pool.AllocateBlocks(4);
    EXPECT_TRUE(blocks.empty());  // all-or-nothing
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

TEST(BlockPoolTest, FreeReturnsBlocksToPool) {
    BlockPool pool(8);
    auto blocks = pool.AllocateBlocks(3);
    EXPECT_EQ(pool.NumFreeBlocks(), 4);
    pool.FreeBlocks(blocks);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
    for (CacheBlock* b : blocks) {
        EXPECT_EQ(b->RefCount(), 0);
    }
}

TEST(BlockPoolTest, RefCountReachesZeroOnlyAfterAllRefsReleased) {
    BlockPool pool(8);
    auto blocks = pool.AllocateBlocks(1);
    CacheBlock* b = blocks.front();
    pool.TouchBlock(b);  // second reference
    EXPECT_EQ(b->RefCount(), 2);
    pool.FreeBlocks({b});
    EXPECT_EQ(b->RefCount(), 1);  // still referenced -> not back in free queue
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
    pool.FreeBlocks({b});
    EXPECT_EQ(b->RefCount(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

// ---- prefix caching: the three-state lifecycle -------------------------

TEST(BlockPoolTest, CachedFreeBlockSurvivesAndIsReusable) {
    BlockPool pool(8);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    auto blocks = pool.AllocateBlocks(1);
    CacheBlock* b = blocks.front();
    pool.CacheFullBlock(b, key);
    EXPECT_TRUE(b->IsCached());

    pool.FreeBlocks({b});
    EXPECT_EQ(b->RefCount(), 0);
    EXPECT_TRUE(b->IsCached());
    EXPECT_EQ(pool.NumCachedFreeBlocks(), 1);

    CacheBlock* hit = pool.GetCachedBlock(key);
    ASSERT_EQ(hit, b);

    // TouchBlock revives it out of the free queue.
    pool.TouchBlock(hit);
    EXPECT_EQ(hit->RefCount(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(BlockPoolTest, MissReturnsNull) {
    BlockPool pool(8);
    EXPECT_EQ(pool.GetCachedBlock(RealKey({9, 9}, 0)), nullptr);
}

TEST(BlockPoolTest, CachingDisabledNeverHits) {
    BlockPool pool(8, /*enable_caching=*/false);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    auto blocks = pool.AllocateBlocks(1);
    CacheBlock* b = blocks.front();
    pool.CacheFullBlock(b, key);  // no-op when caching is disabled
    EXPECT_FALSE(b->IsCached());
    EXPECT_EQ(pool.GetCachedBlock(key), nullptr);  // lookups always miss
}

TEST(BlockPoolTest, GroupIdDistinguishesSameContent) {
    BlockPool pool(8);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({1, 2, 3, 4}, 1);
    ASSERT_NE(k0, k1);  // same content, different group -> different key

    auto a = pool.AllocateBlocks(1);
    pool.CacheFullBlock(a.front(), k0);
    EXPECT_EQ(pool.GetCachedBlock(k0), a.front());
    EXPECT_EQ(pool.GetCachedBlock(k1), nullptr);  // group 1 not cached
}

TEST(BlockPoolTest, EvictionDropsCachedContentWhenReused) {
    // 1 usable block: reusing it must evict its old cached content from the map.
    BlockPool pool(2);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    auto first = pool.AllocateBlocks(1);
    CacheBlock* b = first.front();
    pool.CacheFullBlock(b, key);
    pool.FreeBlocks({b});  // cached + free
    EXPECT_EQ(pool.GetCachedBlock(key), b);

    auto second = pool.AllocateBlocks(1);
    EXPECT_EQ(second.front(), b);  // same physical block reused
    EXPECT_FALSE(b->IsCached());
    EXPECT_EQ(pool.GetCachedBlock(key), nullptr);  // content gone from the map
}

// ---- LRU ordering -------------------------------------------------------

TEST(BlockPoolTest, EvictionPrefersLeastRecentlyFreed) {
    BlockPool pool(4);  // 3 usable blocks
    auto blocks = pool.AllocateBlocks(3);
    CacheBlock* b0 = blocks[0];
    CacheBlock* b1 = blocks[1];
    CacheBlock* b2 = blocks[2];

    pool.FreeBlocks({b0});
    pool.FreeBlocks({b1});
    pool.FreeBlocks({b2});

    auto next = pool.AllocateBlocks(1);
    EXPECT_EQ(next.front(), b0);
}

TEST(BlockPoolTest, AllocateZeroBlocksReturnsEmpty) {
    BlockPool pool(4);
    EXPECT_TRUE(pool.AllocateBlocks(0).empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

}  // namespace
}  // namespace tokenspeed::test
