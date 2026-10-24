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

#include <array>
#include <limits>
#include <random>
#include <stdexcept>

#include <gtest/gtest.h>

#include "host_allocation_reference.h"

namespace tokenspeed::test {
namespace {

struct HostWorld {
    BlockPool device;
    BlockPool host;
    CacheCoordinator coordinator;
    std::vector<CacheBlockRef> held;

    HostWorld(std::int32_t parents, const std::vector<CacheGroupSpec>& specs, const std::vector<std::int32_t>& packing)
        : device{parents, packing},
          host{parents, packing},
          coordinator{MakeCoordinator(specs, 4, device, false, &host, nullptr, false)} {}
};

void ExpectSame(const HostWorld& actual, const HostWorld& reference, const std::vector<std::int32_t>& packing) {
    EXPECT_EQ(actual.host.NumEmptyLcmBlocks(), reference.host.NumEmptyLcmBlocks());
    for (std::int32_t parent = 1; parent <= actual.host.NumLcmBlocks(); ++parent) {
        const auto group = actual.host.BoundGroup(parent);
        ASSERT_EQ(group, reference.host.BoundGroup(parent));
        if (!group) continue;
        for (std::int32_t slot = 0; slot < packing[*group]; ++slot) {
            const CacheBlockLocation location{parent, slot};
            EXPECT_EQ(actual.host.IsOccupied(location), reference.host.IsOccupied(location));
            const auto& ai = actual.coordinator.GroupPrefixIndex(*group);
            const auto& ri = reference.coordinator.GroupPrefixIndex(*group);
            const auto am = ai.MetadataFor(actual.host, location);
            const auto rm = ri.MetadataFor(reference.host, location);
            ASSERT_EQ(am.has_value(), rm.has_value());
            if (am) {
                EXPECT_EQ(am->was_acquired, rm->was_acquired);
                EXPECT_EQ(am->last_access_epoch, rm->last_access_epoch);
                EXPECT_EQ(am->boundary_kind, rm->boundary_kind);
            }
        }
    }
}

TEST(HostRetentionTest, IndexesExistingPublicationsWhenAttached) {
    BlockPool pool(3, {1});
    PrefixCacheIndex index(0);
    auto blocks = pool.AcquireBlocks(0, 3);
    const auto expected = blocks[1]->Location();
    std::array<CacheKey, 3> keys;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = {.group_id = 0, .content_hash = std::to_string(i), .page_offset = 0};
        index.Register(pool, blocks[i], keys[i], i, -1, CacheBoundaryKind::kChunk, nullptr);
    }
    auto hit = index.AcquireMatched(pool, std::span(keys).first(1), 0, GroupPrefixProbe{.hits = {1}}, 0);
    hit.blocks.clear();
    blocks.clear();
    index.EnableHostRetention(pool, std::make_shared<PrefixCacheIndex::HostParentIndex>(), 1, 1);
    std::optional<PrefixCacheIndex::HostRetentionOrder> cursor;
    const auto victim = index.NextHostVictim(pool, 0, cursor);
    ASSERT_TRUE(victim);
    EXPECT_EQ(*victim, expected);
    EXPECT_EQ(index.Evict(pool, *victim), keys[1]);
    EXPECT_TRUE(index.Contains(pool, keys[0]));
}

TEST(HostRetentionTest, RejectedAttachmentPreservesExistingObserver) {
    const CacheGroupSpec spec{
        .attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4, .shard_count = 1};
    HostWorld world(1, {spec, spec}, {1, 1});
    const CacheKey key{.group_id = 0, .content_hash = "beam-observer-attachment", .page_offset = 0};
    auto block = world.coordinator.AcquireHostBlock(0, 0);
    world.coordinator.CacheHostBlock(block, key);
    block.reset();
    {
        PrefixCacheIndex rejected(0);
        EXPECT_THROW(
            rejected.EnableHostRetention(world.host, std::make_shared<PrefixCacheIndex::HostParentIndex>(), 1, 1),
            std::runtime_error);
    }
    auto pin = world.coordinator.FindHostCachedBlock(key);
    EXPECT_NO_THROW(EXPECT_FALSE(world.coordinator.AcquireHostBlock(0, 0)));
    EXPECT_NO_THROW(EXPECT_FALSE(world.coordinator.AcquireHostBlock(1, 0)));
    pin.reset();
    EXPECT_TRUE(world.coordinator.AcquireHostBlock(1, 0));
}

TEST(HostRetentionTest, RandomizedTracesMatchCollectSortVictimsAndPlacement) {
    // Small saturated pools exercise parent rebinding and partial batches often.
    // Mixed packing, shards, old/equal/max epochs, pins, unpublished transfer
    // destinations, duplicate publication, and successful prefix acquisitions.
    for (std::uint32_t seed = 0; seed < 40; ++seed) {
        const std::int32_t shards = seed % 2 ? 2 : 1;
        const std::vector<std::int32_t> packing{shards, 2 * shards, 4 * shards};
        std::vector<CacheGroupSpec> specs;
        for (const auto count : packing) {
            specs.push_back({.attn_kind = AttnKind::kFull,
                             .cache_blocks_per_lcm_block = count,
                             .block_granularity = 4,
                             .shard_count = shards});
        }
        HostWorld actual(13, specs, packing), reference(13, specs, packing);
        std::mt19937 random(seed);
        std::vector<CacheKey> keys;
        for (std::int32_t step = 0; step < 600; ++step) {
            SCOPED_TRACE(::testing::Message() << "seed=" << seed << " step=" << step);
            // Burst pin/unpin churn, including multiple refs per publication.
            // Keep most old entries pinned between bursts to stress eligibility.
            if (step % 7 == 0) {
                for (const auto& key : keys) {
                    if (random() % 10 == 0) continue;
                    auto a = actual.coordinator.FindHostCachedBlock(key);
                    auto r = reference.coordinator.FindHostCachedBlock(key);
                    ASSERT_EQ(static_cast<bool>(a), static_cast<bool>(r));
                    if (a) {
                        actual.held.push_back(a);
                        reference.held.push_back(r);
                        actual.held.push_back(std::move(a));
                        reference.held.push_back(std::move(r));
                    }
                }
            }
            // Publish an in-flight destination while its transfer still owns it.
            if (!actual.held.empty() && random() % 2 == 0) {
                const auto i = random() % actual.held.size();
                auto& a = actual.held[i];
                auto& r = reference.held[i];
                const auto group = *actual.host.BoundGroup(a->Location().lcm_block_id);
                if (!actual.coordinator.GroupPrefixIndex(group).Contains(actual.host, a->Location())) {
                    const CacheKey key{
                        .group_id = group, .content_hash = "transfer-" + std::to_string(step), .page_offset = 0};
                    actual.coordinator.GroupPrefixIndex(group).Register(actual.host, a, key, step, -1,
                                                                        CacheBoundaryKind::kChunk, nullptr);
                    reference.coordinator.GroupPrefixIndex(group).Register(reference.host, r, key, step, -1,
                                                                           CacheBoundaryKind::kChunk, nullptr);
                    keys.push_back(key);
                }
            }
            // Drop pins independently of publication, as transfer ACKs do.
            if (!actual.held.empty() && random() % 3 == 0) {
                const auto i = random() % actual.held.size();
                actual.held.erase(actual.held.begin() + i);
                reference.held.erase(reference.held.begin() + i);
            }
            if (!keys.empty()) {
                const auto& key = keys[random() % keys.size()];
                auto a = actual.coordinator.FindHostCachedBlock(key);
                auto r = reference.coordinator.FindHostCachedBlock(key);
                ASSERT_EQ(static_cast<bool>(a), static_cast<bool>(r));
                if (a) {
                    EXPECT_EQ(a->Location(), r->Location());
                    const auto epoch = random() % 2 ? actual.coordinator.GroupPrefixIndex(key.group_id)
                                                          .MetadataFor(actual.host, a->Location())
                                                          ->last_access_epoch
                                                    : random() % 9;
                    if (random() % 2) {
                        const std::array one{key};
                        const GroupPrefixProbe hit{.hits = {1}};
                        auto ah = actual.coordinator.GroupPrefixIndex(key.group_id)
                                      .AcquireMatched(actual.host, one, 0, hit, epoch);
                        auto rh = reference.coordinator.GroupPrefixIndex(key.group_id)
                                      .AcquireMatched(reference.host, one, 0, hit, epoch);
                        actual.held.push_back(std::move(ah.blocks[0]));
                        reference.held.push_back(std::move(rh.blocks[0]));
                    } else {
                        actual.coordinator.GroupPrefixIndex(key.group_id)
                            .Register(actual.host, a, key, epoch, -1, CacheBoundaryKind::kChunk, nullptr);
                        reference.coordinator.GroupPrefixIndex(key.group_id)
                            .Register(reference.host, r, key, epoch, -1, CacheBoundaryKind::kChunk, nullptr);
                    }
                }
            }
            const std::size_t count = 1 + random() % 12;
            std::vector<std::uint32_t> groups(count);
            std::vector<std::int32_t> buckets(count);
            for (std::size_t i = 0; i < count; ++i) {
                groups[i] = random() % specs.size();
                buckets[i] = random() % shards;
            }
            auto a = actual.coordinator.AcquireHostBlocks(groups, buckets);
            auto r = AllocateHostReference(reference.coordinator, &reference.host, groups, buckets);
            ASSERT_EQ(a.stats.allocated, r.stats.allocated);
            for (std::size_t i = 0; i < count; ++i) {
                ASSERT_EQ(static_cast<bool>(a.blocks[i]), static_cast<bool>(r.blocks[i]));
                if (!a.blocks[i]) continue;
                EXPECT_EQ(a.blocks[i]->Location(), r.blocks[i]->Location());
                const auto action = random() % 6;
                if (action != 0) {
                    CacheKey key{
                        .group_id = groups[i], .content_hash = std::to_string(step * 12 + i), .page_offset = 0};
                    // Dedupe a newly allocated location into an existing canonical entry.
                    if (!keys.empty() && action == 1) {
                        const auto& prior = keys[random() % keys.size()];
                        if (prior.group_id == groups[i]) key = prior;
                    }
                    const std::uint64_t epoch =
                        random() % 7 == 0 ? std::numeric_limits<std::uint64_t>::max() : random() % 11;
                    const auto kind = action % 2 ? CacheBoundaryKind::kEndpoint : CacheBoundaryKind::kChunk;
                    actual.coordinator.GroupPrefixIndex(groups[i]).Register(actual.host, a.blocks[i], key, epoch, -1,
                                                                            kind, nullptr);
                    reference.coordinator.GroupPrefixIndex(groups[i]).Register(reference.host, r.blocks[i], key, epoch,
                                                                               -1, kind, nullptr);
                    keys.push_back(key);
                }
                if (action <= 2) {
                    actual.held.push_back(std::move(a.blocks[i]));
                    reference.held.push_back(std::move(r.blocks[i]));
                }
            }
            a.blocks.clear();
            r.blocks.clear();
            ExpectSame(actual, reference, packing);
            // Compare identities too: identical occupancy alone could hide a wrong victim.
            for (const auto& key : keys) {
                EXPECT_EQ(actual.coordinator.ContainsHostCachedBlock(key),
                          reference.coordinator.ContainsHostCachedBlock(key));
            }
            if (step % 13 == 12) {
                actual.held.clear();
                reference.held.clear();
                // Bound trace bookkeeping, keeping live identities only.
                std::erase_if(keys, [&](const auto& key) { return !actual.coordinator.ContainsHostCachedBlock(key); });
            }
        }
    }
}

TEST(HostRetentionTest, OccupyingAnEligibleParentsHoleExcludesItUntilPublicationAndUnpin) {
    const std::vector<CacheGroupSpec> specs{
        {.cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    HostWorld world(1, specs, {2, 1});
    auto first = world.host.AcquireBlock(0);
    world.coordinator.CacheHostBlock(first, {.group_id = 0, .content_hash = "first", .page_offset = 0});
    first.reset();
    auto transfer = world.host.AcquireBlock(0);
    EXPECT_FALSE(world.coordinator.AcquireHostBlock(1, 0));
    world.coordinator.CacheHostBlock(transfer, {.group_id = 0, .content_hash = "second", .page_offset = 0});
    auto another_pin = transfer;
    transfer.reset();
    EXPECT_FALSE(world.coordinator.AcquireHostBlock(1, 0));
    another_pin.reset();
    EXPECT_TRUE(world.coordinator.AcquireHostBlock(1, 0));
}

TEST(HostRetentionTest, ObserverSurvivesIndexMoveAndDetachesBeforeIndexDestruction) {
    BlockPool pool(1, {1});
    auto pin = pool.AcquireBlock(0);
    const CacheKey key{.group_id = 0, .content_hash = "move", .page_offset = 0};
    {
        PrefixCacheIndex index(0);
        index.Register(pool, pin, key, 0, -1, CacheBoundaryKind::kChunk, nullptr);
        index.EnableHostRetention(pool, std::make_shared<PrefixCacheIndex::HostParentIndex>(), 1, 1);
        PrefixCacheIndex moved(std::move(index));
        pin.reset();
        std::optional<PrefixCacheIndex::HostRetentionOrder> cursor;
        EXPECT_TRUE(moved.NextHostVictim(pool, 0, cursor));
        pin = moved.Find(pool, key);
        cursor.reset();
        EXPECT_FALSE(moved.NextHostVictim(pool, 0, cursor));
    }
    // The remaining owner may be copied and released after its index is gone.
    auto copy = pin;
    pin.reset();
    copy.reset();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(HostRetentionTest, SameEpochAcquisitionAndReleasedTransferPinChangeVictim) {
    const std::vector<CacheGroupSpec> specs{
        {.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    HostWorld world(3, specs, {1});
    std::array<CacheKey, 3> keys;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = {.group_id = 0, .content_hash = std::to_string(i), .page_offset = 0};
        auto block = world.coordinator.AcquireHostBlock(0, 0);
        world.coordinator.GroupPrefixIndex(0).Register(world.host, block, keys[i], 7, -1, CacheBoundaryKind::kChunk,
                                                       nullptr);
    }
    auto acquired = world.coordinator.GroupPrefixIndex(0).AcquireMatched(world.host, std::span(keys).first(1), 0,
                                                                         GroupPrefixProbe{.hits = {1}}, 7);
    acquired.blocks.clear();
    auto pin = world.coordinator.FindHostCachedBlock(keys[1]);
    auto first = world.coordinator.AcquireHostBlock(0, 0);
    EXPECT_TRUE(world.coordinator.ContainsHostCachedBlock(keys[0]));
    EXPECT_TRUE(world.coordinator.ContainsHostCachedBlock(keys[1]));
    EXPECT_FALSE(world.coordinator.ContainsHostCachedBlock(keys[2]));
    pin.reset();
    auto second = world.coordinator.AcquireHostBlock(0, 0);
    EXPECT_TRUE(world.coordinator.ContainsHostCachedBlock(keys[0]));
    EXPECT_FALSE(world.coordinator.ContainsHostCachedBlock(keys[1]));
    EXPECT_TRUE(first);
    EXPECT_TRUE(second);
}

TEST(HostRetentionTest, UnpublishedSiblingExcludesParentUntilTransferReleasesIt) {
    const std::vector<CacheGroupSpec> specs{
        {.cache_blocks_per_lcm_block = 2, .block_granularity = 4},
        {.cache_blocks_per_lcm_block = 1, .block_granularity = 4},
    };
    HostWorld world(2, specs, {2, 1});
    auto blocks = world.host.AcquireBlocks(0, 4);
    ASSERT_EQ(blocks.size(), 4);
    const auto private_parent = blocks[0]->Location().lcm_block_id;
    const auto eligible_parent = blocks[2]->Location().lcm_block_id;
    for (const std::size_t i : {0U, 2U, 3U}) {
        world.coordinator.GroupPrefixIndex(0).Register(
            world.host, blocks[i], CacheKey{.group_id = 0, .content_hash = std::to_string(i), .page_offset = 0}, i, -1,
            CacheBoundaryKind::kChunk, nullptr);
        blocks[i].reset();
    }
    // The oldest parent has an unpublished in-flight destination in slot 1.
    auto first = world.coordinator.AcquireHostBlock(1, 0);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->Location().lcm_block_id, eligible_parent);
    EXPECT_FALSE(world.coordinator.AcquireHostBlock(1, 0));
    blocks[1].reset();
    auto second = world.coordinator.AcquireHostBlock(1, 0);
    ASSERT_TRUE(second);
    EXPECT_EQ(second->Location().lcm_block_id, private_parent);
}

}  // namespace
}  // namespace tokenspeed::test
