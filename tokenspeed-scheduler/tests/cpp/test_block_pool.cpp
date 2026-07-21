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

TEST(BlockPoolTest, AcquireReturnsOwningRefs) {
    BlockPool pool(8);
    auto blocks = pool.AcquireBlocks(3);
    ASSERT_EQ(blocks.size(), 3u);
    for (const BlockRef& block : blocks) {
        EXPECT_EQ(block.use_count(), 1);
        EXPECT_FALSE(block->IsNull());
    }
    EXPECT_EQ(pool.NumFreeBlocks(), 4);  // 7 free - 3 claimed
}

TEST(BlockPoolTest, AcquireFailsWhenCapacityShort) {
    BlockPool pool(4);  // 3 free after null reservation
    auto blocks = pool.AcquireBlocks(4);
    EXPECT_TRUE(blocks.empty());  // all-or-nothing
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

TEST(BlockPoolTest, ResetReturnsBlocksToPool) {
    BlockPool pool(8);
    auto blocks = pool.AcquireBlocks(3);
    EXPECT_EQ(pool.NumFreeBlocks(), 4);
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
        it->reset();
    }
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(BlockPoolTest, BlockReturnsOnlyAfterLastRefResets) {
    BlockPool pool(8);
    BlockRef first = pool.AcquireBlock();
    BlockRef second = first;
    EXPECT_EQ(first.use_count(), 2);
    first.reset();
    EXPECT_EQ(second.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
    second.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

// ---- prefix caching: the three-state lifecycle -------------------------

TEST(BlockPoolTest, CachedFreeBlockSurvivesAndIsReusable) {
    BlockPool pool(8);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef block = pool.AcquireBlock();
    CacheBlock* b = block.get();
    pool.CacheFullBlock(b, key);
    EXPECT_TRUE(b->IsCached());

    block.reset();
    EXPECT_TRUE(b->IsCached());
    EXPECT_EQ(pool.NumCachedFreeBlocks(), 1);

    BlockRef hit = pool.FindCachedBlock(key);
    ASSERT_EQ(hit.get(), b);
    EXPECT_EQ(hit.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(BlockPoolTest, MissReturnsNull) {
    BlockPool pool(8);
    EXPECT_FALSE(pool.FindCachedBlock(RealKey({9, 9}, 0)));
    EXPECT_FALSE(pool.ContainsCachedBlock(RealKey({9, 9}, 0)));
}

TEST(BlockPoolTest, CachingDisabledNeverHits) {
    BlockPool pool(8, /*enable_caching=*/false);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef block = pool.AcquireBlock();
    CacheBlock* b = block.get();
    pool.CacheFullBlock(b, key);  // no-op when caching is disabled
    EXPECT_FALSE(b->IsCached());
    EXPECT_FALSE(pool.FindCachedBlock(key));  // lookups always miss
    EXPECT_FALSE(pool.ContainsCachedBlock(key));
}

TEST(BlockPoolTest, GroupIdDistinguishesSameContent) {
    BlockPool pool(8);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({1, 2, 3, 4}, 1);
    ASSERT_NE(k0, k1);  // same content, different group -> different key

    BlockRef a = pool.AcquireBlock();
    pool.CacheFullBlock(a.get(), k0);
    EXPECT_TRUE(pool.ContainsCachedBlock(k0));
    EXPECT_FALSE(pool.ContainsCachedBlock(k1));  // group 1 not cached
}

TEST(BlockPoolTest, EvictionDropsCachedContentWhenReused) {
    // 1 usable block: reusing it must evict its old cached content from the map.
    BlockPool pool(2);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef first = pool.AcquireBlock();
    CacheBlock* b = first.get();
    pool.CacheFullBlock(b, key);
    first.reset();  // cached + free
    EXPECT_TRUE(pool.ContainsCachedBlock(key));

    BlockRef second = pool.AcquireBlock();
    EXPECT_EQ(second.get(), b);  // same physical block reused
    EXPECT_FALSE(b->IsCached());
    EXPECT_FALSE(pool.ContainsCachedBlock(key));  // content gone from the map
}

// ---- LRU ordering -------------------------------------------------------

TEST(BlockPoolTest, EvictionPrefersLeastRecentlyFreed) {
    BlockPool pool(4);  // 3 usable blocks
    auto blocks = pool.AcquireBlocks(3);
    CacheBlock* b0 = blocks[0].get();
    CacheBlock* b1 = blocks[1].get();
    CacheBlock* b2 = blocks[2].get();

    blocks[0].reset();
    blocks[1].reset();
    blocks[2].reset();

    BlockRef next = pool.AcquireBlock();
    EXPECT_EQ(next.get(), b0);
}

TEST(BlockPoolTest, AcquireZeroBlocksReturnsEmpty) {
    BlockPool pool(4);
    EXPECT_TRUE(pool.AcquireBlocks(0).empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

}  // namespace
}  // namespace tokenspeed::test
