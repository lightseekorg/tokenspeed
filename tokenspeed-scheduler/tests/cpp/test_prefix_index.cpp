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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed::test {
namespace {

constexpr std::uint32_t kGroupId = 0;

CacheKey KeyOf(const std::string& content_hash) {
    return CacheKey{.group_id = kGroupId, .content_hash = content_hash, .page_offset = 0};
}

// Registers one freshly acquired block and drops the local reference, leaving
// the index as its only owner so the entry is evictable.
CacheBlockLocation Cache(PrefixCacheIndex& index, BlockPool& pool, const std::string& content_hash,
                         std::uint64_t access_epoch) {
    CacheBlockRef block = pool.AcquireBlock(kGroupId, /*cache_blocks_per_lcm_block=*/1);
    EXPECT_TRUE(block);
    const CacheBlockLocation location = block->Location();
    index.Register(pool, block, KeyOf(content_hash), access_epoch, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
    return location;
}

// Walks the whole eviction order one access epoch at a time, exactly the way
// admission consumes it.
std::vector<CacheBlockLocation> DrainEvictionOrder(const PrefixCacheIndex& index, const BlockPool& pool) {
    std::vector<CacheBlockLocation> order;
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    while (index.NextEvictionEpoch(pool, cursor, batch)) {
        for (const PrefixCacheIndex::EvictionCandidate& candidate : batch) {
            order.push_back(candidate.location);
        }
        batch.clear();
    }
    return order;
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversTheOldestAccessEpochFirst) {
    BlockPool pool(3);
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation newest = Cache(index, pool, "newest", /*access_epoch=*/30);
    const CacheBlockLocation oldest = Cache(index, pool, "oldest", /*access_epoch=*/10);
    const CacheBlockLocation middle = Cache(index, pool, "middle", /*access_epoch=*/20);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{oldest, middle, newest}));
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversOneEpochPerBatchSortedByLocation) {
    BlockPool pool(4);
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "later", /*access_epoch=*/9);
    const CacheBlockLocation first = Cache(index, pool, "same-a", /*access_epoch=*/7);
    const CacheBlockLocation second = Cache(index, pool, "same-b", /*access_epoch=*/7);
    const CacheBlockLocation third = Cache(index, pool, "same-c", /*access_epoch=*/7);

    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, batch));

    ASSERT_EQ(batch.size(), 3u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    EXPECT_EQ(batch[2].location, third);
    EXPECT_TRUE(std::ranges::all_of(batch, [](const PrefixCacheIndex::EvictionCandidate& candidate) {
        return candidate.metadata.last_access_epoch == 7;
    }));
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEntriesWithoutLosingTheRest) {
    BlockPool pool(3);
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation unpinned = Cache(index, pool, "unpinned", /*access_epoch=*/20);

    CacheBlockRef pinned = pool.AcquireBlock(kGroupId, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(pinned);
    index.Register(pool, pinned, KeyOf("pinned"), /*access_epoch=*/10, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{unpinned}));

    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 2u);
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEpochsAndReturnsTheWholeNextEpoch) {
    BlockPool pool(5);
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (std::uint64_t epoch : {1u, 2u, 3u}) {
        CacheBlockRef block = pool.AcquireBlock(kGroupId, /*cache_blocks_per_lcm_block=*/1);
        index.Register(pool, block, KeyOf("pinned-" + std::to_string(epoch)), epoch,
                       /*logical_block_index=*/-1, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
        pinned.push_back(std::move(block));
    }
    const CacheBlockLocation first = Cache(index, pool, "first", /*access_epoch=*/3);
    const CacheBlockLocation second = Cache(index, pool, "second", /*access_epoch=*/3);
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ExhaustsTheMaximumEpochWithoutWrapping) {
    BlockPool pool(1);
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "last", std::numeric_limits<std::uint64_t>::max());
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, batch));
    ASSERT_EQ(batch.size(), 1u);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ReRegisteringAtAnOlderEpochMovesTheEntryEarlier) {
    BlockPool pool(3);
    PrefixCacheIndex index(kGroupId);
    CacheBlockRef refreshed = pool.AcquireBlock(kGroupId, /*cache_blocks_per_lcm_block=*/1);
    ASSERT_TRUE(refreshed);
    const CacheBlockLocation moved = refreshed->Location();
    index.Register(pool, refreshed, KeyOf("moved"), /*access_epoch=*/30, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
    const CacheBlockLocation settled = Cache(index, pool, "settled", /*access_epoch=*/20);

    // A request that continues under its original epoch re-registers its pages
    // at an epoch older than entries published since, so the order cannot be
    // maintained by append alone.
    index.Register(pool, refreshed, KeyOf("moved"), /*access_epoch=*/5, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
    refreshed.reset();

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{moved, settled}));
}

TEST(PrefixCacheIndexEvictionOrderTest, AcquiringAMatchMovesTheEntryToTheRequestEpoch) {
    BlockPool pool(3);
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation matched = Cache(index, pool, "matched", /*access_epoch=*/10);
    const CacheBlockLocation untouched = Cache(index, pool, "untouched", /*access_epoch=*/20);

    const std::vector<CacheKey> keys{KeyOf("matched")};
    const GroupPrefixProbe probe{.hits = {1}};
    PrefixMatch match = index.AcquireMatched(pool, keys, /*begin_blocks=*/0, probe, /*access_epoch=*/40);
    ASSERT_EQ(match.blocks.size(), 1u);
    match.blocks.clear();

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{untouched, matched}));
}

TEST(PrefixCacheIndexEvictionOrderTest, EvictedEntriesLeaveTheOrder) {
    BlockPool pool(3);
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation evicted = Cache(index, pool, "evicted", /*access_epoch=*/10);
    const CacheBlockLocation kept = Cache(index, pool, "kept", /*access_epoch=*/20);

    ASSERT_TRUE(index.Evict(pool, evicted).has_value());

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{kept}));
    EXPECT_EQ(index.NumEntries(pool), 1);
}

TEST(PrefixCacheIndexEvictionOrderTest, MatchesTheEvictableSetSeenByAFullScan) {
    BlockPool pool(64);
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (int i = 0; i < 40; ++i) {
        // Interleave epochs so registration order and eviction order differ.
        const std::uint64_t access_epoch = static_cast<std::uint64_t>((i * 17) % 23 + 1);
        if (i % 5 == 0) {
            CacheBlockRef block = pool.AcquireBlock(kGroupId, /*cache_blocks_per_lcm_block=*/1);
            ASSERT_TRUE(block);
            index.Register(pool, block, KeyOf("pinned-" + std::to_string(i)), access_epoch,
                           /*logical_block_index=*/-1, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
            pinned.push_back(std::move(block));
            continue;
        }
        Cache(index, pool, "entry-" + std::to_string(i), access_epoch);
    }

    std::vector<CacheBlockLocation> streamed = DrainEvictionOrder(index, pool);
    std::vector<CacheBlockLocation> scanned;
    for (const PrefixCacheIndex::EvictionCandidate& candidate : index.EvictableCandidates(pool)) {
        scanned.push_back(candidate.location);
    }
    ASSERT_EQ(streamed.size(), scanned.size());

    const auto by_location = [](CacheBlockLocation lhs, CacheBlockLocation rhs) {
        return lhs.lcm_block_id != rhs.lcm_block_id ? lhs.lcm_block_id < rhs.lcm_block_id
                                                    : lhs.slot_index < rhs.slot_index;
    };
    std::ranges::sort(streamed, by_location);
    std::ranges::sort(scanned, by_location);
    EXPECT_EQ(streamed, scanned);

    pinned.clear();
}

}  // namespace
}  // namespace tokenspeed::test
