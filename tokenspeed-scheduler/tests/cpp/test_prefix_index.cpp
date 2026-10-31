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
#include <memory>
#include <random>
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
    CacheBlockRef block = pool.AcquireBlock(kGroupId);
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
    while (index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch)) {
        for (const PrefixCacheIndex::EvictionCandidate& candidate : batch) {
            order.push_back(candidate.location);
        }
        batch.clear();
    }
    return order;
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversTheOldestAccessEpochFirst) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation newest = Cache(index, pool, "newest", /*access_epoch=*/30);
    const CacheBlockLocation oldest = Cache(index, pool, "oldest", /*access_epoch=*/10);
    const CacheBlockLocation middle = Cache(index, pool, "middle", /*access_epoch=*/20);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{oldest, middle, newest}));
}

TEST(PrefixCacheIndexInspectionTest, ReportsCurrentOwnershipWithoutPinningOrTouchingTheEntry) {
    BlockPool pool(2, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation location = Cache(index, pool, "entry", /*access_epoch=*/10);
    const CacheKey key = KeyOf("entry");
    const auto initial = index.Inspect(pool, key);
    ASSERT_TRUE(initial);
    EXPECT_EQ(initial->location, location);
    EXPECT_EQ(initial->metadata.last_access_epoch, 10);
    EXPECT_FALSE(initial->metadata.was_acquired);
    EXPECT_FALSE(initial->pinned);

    auto holder = index.Find(pool, key);
    ASSERT_TRUE(holder);
    EXPECT_EQ(holder.use_count(), 2);
    ASSERT_TRUE(index.Inspect(pool, key));
    EXPECT_TRUE(index.Inspect(pool, key)->pinned);
    EXPECT_EQ(holder.use_count(), 2);
    index.Register(pool, holder, key, /*access_epoch=*/30, /*logical_block_index=*/-1, CacheBoundaryKind::kChunk,
                   /*newly_cached=*/nullptr);
    EXPECT_EQ(index.Inspect(pool, key)->metadata.last_access_epoch, 30);
    holder = {};
    EXPECT_FALSE(index.Inspect(pool, key)->pinned);

    // Keeping a snapshot alive must not prevent actual eviction.
    EXPECT_EQ(index.Evict(pool, location), key);
    EXPECT_FALSE(index.Inspect(pool, key));
}

TEST(PrefixCacheIndexInspectionTest, KeepsPoolsSeparateAndDoesNotClaimMissingKeys) {
    BlockPool device(1, {1});
    BlockPool host(1, {1});
    PrefixCacheIndex index(kGroupId);
    const auto location = Cache(index, device, "entry", /*access_epoch=*/7);
    ASSERT_TRUE(index.Inspect(device, KeyOf("entry")));
    EXPECT_EQ(index.Inspect(device, KeyOf("entry"))->location, location);
    EXPECT_FALSE(index.Inspect(host, KeyOf("entry")));
    EXPECT_FALSE(index.Inspect(device, KeyOf("absent")));
}

TEST(PrefixCacheIndexEvictionOrderTest, DeliversOneEpochPerBatchSortedByLocation) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "later", /*access_epoch=*/9);
    const CacheBlockLocation first = Cache(index, pool, "same-a", /*access_epoch=*/7);
    const CacheBlockLocation second = Cache(index, pool, "same-b", /*access_epoch=*/7);
    const CacheBlockLocation third = Cache(index, pool, "same-c", /*access_epoch=*/7);

    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));

    ASSERT_EQ(batch.size(), 3u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    EXPECT_EQ(batch[2].location, third);
    EXPECT_TRUE(std::ranges::all_of(batch, [](const PrefixCacheIndex::EvictionCandidate& candidate) {
        return candidate.metadata.last_access_epoch == 7;
    }));
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEntriesWithoutLosingTheRest) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation unpinned = Cache(index, pool, "unpinned", /*access_epoch=*/20);

    CacheBlockRef pinned = pool.AcquireBlock(kGroupId);
    ASSERT_TRUE(pinned);
    index.Register(pool, pinned, KeyOf("pinned"), /*access_epoch=*/10, /*logical_block_index=*/-1,
                   CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{unpinned}));

    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool).size(), 2u);
}

TEST(PrefixCacheIndexEvictionOrderTest, ReleasedPinnedEntryUsesItsLatestPublishedEpoch) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const auto middle = Cache(index, pool, "middle", 20);
    const auto last = Cache(index, pool, "last", 30);
    auto pinned = pool.AcquireBlock(kGroupId);
    const auto location = pinned->Location();
    index.Register(pool, pinned, KeyOf("pinned"), 40, 0, CacheBoundaryKind::kEndpoint, nullptr);
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{middle, last}));
    index.Register(pool, pinned, KeyOf("pinned"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr);
    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{location, middle, last}));
    auto repinned = index.Find(pool, KeyOf("pinned"));
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{middle, last}));
    repinned.reset();
    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{location, middle, last}));
}

TEST(PrefixCacheIndexEvictionOrderTest, MovingIndexKeepsReleasedEntriesEvictable) {
    BlockPool pool(2, {1});
    PrefixCacheIndex source(kGroupId);
    auto pinned = pool.AcquireBlock(kGroupId);
    const auto location = pinned->Location();
    source.Register(pool, pinned, KeyOf("pinned"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr);
    EXPECT_TRUE(DrainEvictionOrder(source, pool).empty());
    PrefixCacheIndex moved(std::move(source));
    PrefixCacheIndex assigned(kGroupId);
    Cache(assigned, pool, "discarded", 1);
    assigned = std::move(moved);
    pinned.reset();
    EXPECT_EQ(DrainEvictionOrder(assigned, pool), (std::vector<CacheBlockLocation>{location}));
    EXPECT_TRUE(assigned.Evict(pool, location));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
}

TEST(PrefixCacheIndexEvictionOrderTest, SharedIndexesRestoreTheRemainingCacheOwner) {
    BlockPool pool(1, {1});
    auto first = std::make_unique<PrefixCacheIndex>(kGroupId);
    PrefixCacheIndex second(kGroupId);
    auto block = pool.AcquireBlock(kGroupId);
    const auto location = block->Location();
    first->Register(pool, block, KeyOf("shared"), 10, 0, CacheBoundaryKind::kEndpoint, nullptr);
    second.Register(pool, block, KeyOf("shared"), 20, 0, CacheBoundaryKind::kEndpoint, nullptr);
    block.reset();
    EXPECT_TRUE(DrainEvictionOrder(*first, pool).empty());
    EXPECT_TRUE(DrainEvictionOrder(second, pool).empty());
    first.reset();
    EXPECT_EQ(DrainEvictionOrder(second, pool), (std::vector<CacheBlockLocation>{location}));
}

TEST(PrefixCacheIndexEvictionOrderTest, RepeatedOwnershipAndEpochChangesMatchFullScan) {
    BlockPool pool(64, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pins(64);
    for (int i = 0; i < 64; ++i) {
        Cache(index, pool, std::to_string(i), i % 7);
    }
    std::mt19937 random(20261030);
    for (int iteration = 0; iteration < 600; ++iteration) {
        const auto slot = random() % pins.size();
        const auto key = KeyOf(std::to_string(slot));
        if (pins[slot]) {
            pins[slot].reset();
        } else {
            pins[slot] = index.Find(pool, key);
        }
        if (iteration % 3 == 0) {
            auto block = index.Find(pool, key);
            index.Register(pool, block, key, random() % 13, static_cast<std::int32_t>(slot),
                           CacheBoundaryKind::kEndpoint, nullptr);
        }
        auto expected = index.EvictableCandidates(pool);
        std::ranges::sort(expected, [](const auto& lhs, const auto& rhs) {
            if (lhs.metadata.last_access_epoch != rhs.metadata.last_access_epoch) {
                return lhs.metadata.last_access_epoch < rhs.metadata.last_access_epoch;
            }
            if (lhs.location.lcm_block_id != rhs.location.lcm_block_id) {
                return lhs.location.lcm_block_id < rhs.location.lcm_block_id;
            }
            return lhs.location.slot_index < rhs.location.slot_index;
        });
        std::vector<CacheBlockLocation> expected_locations;
        for (const auto& candidate : expected) {
            expected_locations.push_back(candidate.location);
        }
        EXPECT_EQ(DrainEvictionOrder(index, pool), expected_locations) << iteration;
    }
}

TEST(PrefixCacheIndexEvictionOrderTest, BoundedTraversalLeavesNewerEpochsForTheNextCall) {
    BlockPool pool(4, {1});
    PrefixCacheIndex index(kGroupId);
    CacheBlockRef pinned = pool.AcquireBlock(kGroupId);
    index.Register(pool, pinned, KeyOf("pinned"), 1, -1, CacheBoundaryKind::kChunk, nullptr);
    const CacheBlockLocation first = Cache(index, pool, "first", 4);
    const CacheBlockLocation second = Cache(index, pool, "same-epoch", 4);
    const CacheBlockLocation later = Cache(index, pool, "later", 9);

    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 1u);
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 1u);
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, 3, batch));
    EXPECT_TRUE(batch.empty());
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 4u);
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 4, batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 9u);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, 8, batch));
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 9u);
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 9, batch));
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].location, later);
    EXPECT_FALSE(index.PeekEvictionEpoch(pool, cursor));
}

TEST(PrefixCacheIndexEvictionOrderTest, AZeroEpochAndAnAbsentTierAreDistinct) {
    BlockPool pool(1, {1});
    BlockPool empty_pool(1, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation first = Cache(index, pool, "superseded", 0);
    PrefixCacheIndex::EvictionCursor cursor;
    EXPECT_EQ(index.PeekEvictionEpoch(pool, cursor), 0u);
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, 0, batch));
    ASSERT_EQ(batch.size(), 1u);
    EXPECT_EQ(batch[0].location, first);
    PrefixCacheIndex::EvictionCursor empty_cursor;
    EXPECT_FALSE(index.PeekEvictionEpoch(empty_pool, empty_cursor));
    EXPECT_FALSE(index.NextEvictionEpoch(empty_pool, empty_cursor, 0, batch));
    EXPECT_EQ(batch.size(), 1u) << "an exhausted tier does not clear prior output";
}

TEST(PrefixCacheIndexEvictionOrderTest, SkipsPinnedEpochsAndReturnsTheWholeNextEpoch) {
    BlockPool pool(5, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (std::uint64_t epoch : {1u, 2u, 3u}) {
        CacheBlockRef block = pool.AcquireBlock(kGroupId);
        index.Register(pool, block, KeyOf("pinned-" + std::to_string(epoch)), epoch,
                       /*logical_block_index=*/-1, CacheBoundaryKind::kChunk, /*newly_cached=*/nullptr);
        pinned.push_back(std::move(block));
    }
    const CacheBlockLocation first = Cache(index, pool, "first", /*access_epoch=*/3);
    const CacheBlockLocation second = Cache(index, pool, "second", /*access_epoch=*/3);
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 2u);
    EXPECT_EQ(batch[0].location, first);
    EXPECT_EQ(batch[1].location, second);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ExhaustsTheMaximumEpochWithoutWrapping) {
    BlockPool pool(1, {1});
    PrefixCacheIndex index(kGroupId);
    Cache(index, pool, "last", std::numeric_limits<std::uint64_t>::max());
    PrefixCacheIndex::EvictionCursor cursor;
    std::vector<PrefixCacheIndex::EvictionCandidate> batch;
    ASSERT_TRUE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    ASSERT_EQ(batch.size(), 1u);
    batch.clear();
    EXPECT_FALSE(index.NextEvictionEpoch(pool, cursor, std::numeric_limits<std::uint64_t>::max(), batch));
    EXPECT_TRUE(batch.empty());
}

TEST(PrefixCacheIndexEvictionOrderTest, ReRegisteringAtAnOlderEpochMovesTheEntryEarlier) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    CacheBlockRef refreshed = pool.AcquireBlock(kGroupId);
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
    BlockPool pool(3, {1});
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
    BlockPool pool(3, {1});
    PrefixCacheIndex index(kGroupId);
    const CacheBlockLocation evicted = Cache(index, pool, "evicted", /*access_epoch=*/10);
    const CacheBlockLocation kept = Cache(index, pool, "kept", /*access_epoch=*/20);

    ASSERT_TRUE(index.Evict(pool, evicted).has_value());

    EXPECT_EQ(DrainEvictionOrder(index, pool), (std::vector<CacheBlockLocation>{kept}));
    EXPECT_EQ(index.NumEntries(pool), 1);
}

TEST(PrefixCacheIndexEvictionOrderTest, MatchesTheEvictableSetSeenByAFullScan) {
    BlockPool pool(64, {1});
    PrefixCacheIndex index(kGroupId);
    std::vector<CacheBlockRef> pinned;
    for (int i = 0; i < 40; ++i) {
        // Interleave epochs so registration order and eviction order differ.
        const std::uint64_t access_epoch = static_cast<std::uint64_t>((i * 17) % 23 + 1);
        if (i % 5 == 0) {
            CacheBlockRef block = pool.AcquireBlock(kGroupId);
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
