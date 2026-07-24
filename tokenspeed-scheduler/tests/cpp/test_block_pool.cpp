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

#include <memory>
#include <span>
#include <string>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>

#include "cache/block_pool.h"
#include "scheduler/page_hasher.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasNullBlock = requires(T& value) { value.NullBlock(); };

template <class T>
concept HasNullBlockRef = requires(T& value) { value.NullBlockRef(); };

static_assert(!HasNullBlock<BlockPool>);
static_assert(!HasNullBlockRef<BlockPool>);

using token_span = std::span<const std::int32_t>;

// A real key from page_hasher.h, not a synthetic placeholder.
std::string RealKey(const std::vector<std::int32_t>& tokens, uint32_t group_id) {
    std::vector<token_span> pages = {token_span(tokens.data(), tokens.size())};
    std::vector<std::string> keys = ComputePagedHashesWithGroup(pages, "", group_id);
    return keys.front();
}

TEST(BlockPoolTest, ReservesPageIdZeroAndCountsFree) {
    BlockPool pool(8);
    EXPECT_EQ(pool.TotalBlocks(), 8);
    EXPECT_EQ(pool.NumFreeBlocks(), 7);
}

TEST(BlockPoolTest, DestroyWithLiveReferenceReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            auto pool = std::make_unique<BlockPool>(2);
            BlockRef ref = pool->AcquireBlock();
            pool.reset();
        },
        "BlockPool destroyed with live block references");
}

TEST(BlockPoolTest, AcquireReturnsOwningRefs) {
    BlockPool pool(8);
    auto blocks = pool.AcquireBlocks(3);
    ASSERT_EQ(blocks.size(), 3u);
    for (const BlockRef& block : blocks) {
        EXPECT_EQ(block.use_count(), 1);
        EXPECT_TRUE(block);
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

TEST(BlockPoolTest, BlockReturnsWhenOwnerCountReachesZero) {
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

TEST(BlockPoolTest, CachedFreeBlockSurvivesAndIsReusable) {
    BlockPool pool(8);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef block = pool.AcquireBlock();
    const std::int32_t block_id = block->BlockId();
    pool.CacheFullBlock(block, key);
    EXPECT_TRUE(block->IsCached());

    block.reset();
    EXPECT_EQ(pool.NumCachedFreeBlocks(), 1);

    BlockRef hit = pool.AcquireCachedBlock(key);
    ASSERT_EQ(hit->BlockId(), block_id);
    EXPECT_TRUE(hit->IsCached());
    EXPECT_EQ(hit.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(BlockPoolTest, ActiveCachedBlockCanBeShared) {
    BlockPool pool(4);
    const std::string key = RealKey({1, 2, 3, 4}, 0);
    BlockRef first = pool.AcquireBlock();
    pool.CacheFullBlock(first, key);

    BlockRef second = pool.AcquireCachedBlock(key);

    EXPECT_EQ(second, first);
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), 2);
}

TEST(BlockPoolTest, DuplicateHashesKeepDistinctBlocksIndexed) {
    BlockPool pool(3);
    const std::string key = RealKey({1, 2, 3, 4}, 0);
    BlockRef first = pool.AcquireBlock();
    BlockRef second = pool.AcquireBlock();
    const std::int32_t first_id = first->BlockId();
    pool.CacheFullBlock(first, key);
    pool.CacheFullBlock(second, key);
    first.reset();
    second.reset();

    EXPECT_EQ(pool.NumCachedBlocks(), 2);
    BlockRef hit = pool.AcquireCachedBlock(key);
    EXPECT_EQ(hit->BlockId(), first_id);
    EXPECT_EQ(pool.NumCachedBlocks(), 2);
}

TEST(BlockPoolTest, MissReturnsNull) {
    BlockPool pool(8);
    EXPECT_FALSE(pool.AcquireCachedBlock(RealKey({9, 9}, 0)));
    EXPECT_FALSE(pool.ContainsCachedBlock(RealKey({9, 9}, 0)));
}

TEST(BlockPoolTest, CachingDisabledNeverHits) {
    BlockPool pool(8, /*enable_caching=*/false);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef block = pool.AcquireBlock();
    pool.CacheFullBlock(block, key);  // no-op when caching is disabled
    EXPECT_FALSE(block->IsCached());
    EXPECT_FALSE(pool.AcquireCachedBlock(key));  // lookups always miss
    EXPECT_FALSE(pool.ContainsCachedBlock(key));
}

TEST(BlockPoolTest, GroupIdDistinguishesSameContent) {
    BlockPool pool(8);
    const std::string k0 = RealKey({1, 2, 3, 4}, 0);
    const std::string k1 = RealKey({1, 2, 3, 4}, 1);
    ASSERT_NE(k0, k1);  // same content, different group -> different key

    BlockRef a = pool.AcquireBlock();
    pool.CacheFullBlock(a, k0);
    EXPECT_TRUE(pool.ContainsCachedBlock(k0));
    EXPECT_FALSE(pool.ContainsCachedBlock(k1));  // group 1 not cached
}

TEST(BlockPoolTest, EvictionDropsCachedContentWhenReused) {
    // 1 usable block: reusing it must evict its old cached content from the map.
    BlockPool pool(2);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef first = pool.AcquireBlock();
    const std::int32_t block_id = first->BlockId();
    pool.CacheFullBlock(first, key);
    first.reset();  // cached + free
    EXPECT_TRUE(pool.ContainsCachedBlock(key));

    BlockRef second = pool.AcquireBlock();
    EXPECT_EQ(second->BlockId(), block_id);  // same physical block reused
    EXPECT_FALSE(second->IsCached());
    EXPECT_FALSE(pool.ContainsCachedBlock(key));  // content gone from the map
}

TEST(BlockPoolTest, BatchAcquireDropsCachedContentWhenReused) {
    BlockPool pool(3);
    const std::string key = RealKey({1, 2, 3, 4}, 0);

    BlockRef cached = pool.AcquireBlock();
    pool.CacheFullBlock(cached, key);
    cached.reset();
    ASSERT_TRUE(pool.ContainsCachedBlock(key));

    std::vector<BlockRef> blocks = pool.AcquireBlocks(2);

    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_FALSE(pool.ContainsCachedBlock(key));
}

TEST(BlockPoolTest, CacheFullBlockRejectsEmptyOrForeignReference) {
    BlockPool pool(8);
    BlockPool other_pool(8);
    const std::string key = RealKey({1, 2, 3, 4}, 0);
    BlockRef foreign = other_pool.AcquireBlock();
    BlockRef local = pool.AcquireBlock();

    EXPECT_THROW(pool.CacheFullBlock(BlockRef{}, key), std::runtime_error);
    EXPECT_THROW(pool.CacheFullBlock(foreign, key), std::runtime_error);
    EXPECT_THROW(pool.CacheFullBlock(local, ""), std::runtime_error);
    EXPECT_FALSE(local->IsCached());
}

TEST(BlockPoolTest, EvictionPrefersLeastRecentlyFreed) {
    BlockPool pool(4);  // 3 usable blocks
    auto blocks = pool.AcquireBlocks(3);
    const std::int32_t first_id = blocks[0]->BlockId();

    blocks[0].reset();
    blocks[1].reset();
    blocks[2].reset();

    BlockRef next = pool.AcquireBlock();
    EXPECT_EQ(next->BlockId(), first_id);
}

TEST(BlockPoolTest, AcquireZeroBlocksReturnsEmpty) {
    BlockPool pool(4);
    EXPECT_TRUE(pool.AcquireBlocks(0).empty());
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

}  // namespace
}  // namespace tokenspeed::test
