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

#include <array>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/checkpoint_tree.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"
#include "scheduler/types.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kPage = 4;

// A State checkpoint at the end of `tokens`, hashed as a request publishes it.
struct Checkpoint {
    std::vector<std::int32_t> tokens;
    std::vector<ContentHash> hashes;
    std::optional<TailPublication> tail;

    explicit Checkpoint(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior);
        }
        if (tokens.size() % kPage != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
        }
    }
    ContentHash Hash() const { return tail ? tail->hash : hashes.back(); }
    CompletedPages Publication() const {
        return CompletedPages{.tokens = tokens,
                              .prefix_hashes = hashes,
                              .tail = tail,
                              .publish_state = true,
                              .boundary_kind = CacheBoundaryKind::kEndpoint};
    }
};

std::vector<std::int32_t> Sequence(std::int32_t count) {
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

std::vector<std::int32_t> Concat(std::vector<std::int32_t> head, std::initializer_list<std::int32_t> rest) {
    head.insert(head.end(), rest);
    return head;
}

constexpr CheckpointTree::Residency kOnDevice{.complete = {true, false}, .held = true};
constexpr CheckpointTree::Residency kOnHost{.complete = {false, true}, .held = true};
constexpr CheckpointTree::Residency kOnBoth{.complete = {true, true}, .held = true};
constexpr CheckpointTree::Residency kIncomplete{.held = true};
constexpr CheckpointTree::Residency kGone{};

void Record(CheckpointTree& tree, const Checkpoint& point, CheckpointTree::Residency residency) {
    tree.Observe(point.tokens, point.hashes, point.tail);
    tree.Update(point.Hash(), residency);
}

TEST(CheckpointTree, DistanceCountsCheckpointsToTheNearestLeaf) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)}, d{Sequence(16)};
    for (const Checkpoint* point : {&a, &b, &c, &d}) Record(tree, *point, kOnDevice);
    EXPECT_EQ(tree.Distance(a.Hash()), 3);
    EXPECT_EQ(tree.Distance(b.Hash()), 2);
    EXPECT_EQ(tree.Distance(c.Hash()), 1);
    EXPECT_EQ(tree.Distance(d.Hash()), 0);
    // A branch after B: B and A are now one checkpoint nearer a leaf.
    const Checkpoint e{Concat(Sequence(8), {99, 98, 97, 96})};
    Record(tree, e, kOnDevice);
    EXPECT_EQ(tree.Distance(a.Hash()), 2);
    EXPECT_EQ(tree.Distance(b.Hash()), 1);
    EXPECT_EQ(tree.Distance(e.Hash()), 0);
    EXPECT_EQ(tree.Distance(c.Hash()), 1);
}

TEST(CheckpointTree, DistanceStopsAtTheCap) {
    CheckpointTree tree{kPage};
    std::vector<Checkpoint> chain;
    for (std::int32_t pages = 1; pages <= 10; ++pages) chain.emplace_back(Sequence(pages * kPage));
    for (const Checkpoint& point : chain) Record(tree, point, kOnDevice);
    EXPECT_EQ(tree.Distance(chain[0].Hash()), CheckpointTree::kMaxDistance);
    EXPECT_EQ(tree.Distance(chain[4].Hash()), CheckpointTree::kMaxDistance);
    EXPECT_EQ(tree.Distance(chain[5].Hash()), 4);
    // The leaf leaving moves each checkpoint one nearer, up to the cap.
    tree.Update(chain[9].Hash(), kGone);
    EXPECT_EQ(tree.Distance(chain[8].Hash()), 0);
    EXPECT_EQ(tree.Distance(chain[4].Hash()), 4);
    EXPECT_EQ(tree.Distance(chain[3].Hash()), CheckpointTree::kMaxDistance);
}

TEST(CheckpointTree, CheckpointsInsideAPageOrderByTokens) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(5)}, b{Sequence(9)}, c{Sequence(13)};
    for (const Checkpoint* point : {&a, &b, &c}) Record(tree, *point, kOnDevice);
    EXPECT_EQ(tree.Distance(a.Hash()), 2);
    EXPECT_EQ(tree.Distance(b.Hash()), 1);
    // Same first page, different token where A's tail is: not on A's path.
    const Checkpoint other{{1, 2, 3, 4, 99, 6, 7, 8, 9}};
    Record(tree, other, kOnDevice);
    EXPECT_EQ(tree.Distance(a.Hash()), 2);
    EXPECT_EQ(tree.Distance(other.Hash()), 0);
}

TEST(CheckpointTree, ShorterCheckpointPublishedLaterJoinsThePath) {
    CheckpointTree tree{kPage};
    const Checkpoint longest{Sequence(12)}, inside_page{Sequence(5)}, page_aligned{Sequence(8)};
    Record(tree, longest, kOnDevice);
    Record(tree, inside_page, kOnDevice);
    EXPECT_EQ(tree.Distance(inside_page.Hash()), 1);
    Record(tree, page_aligned, kOnDevice);
    EXPECT_EQ(tree.Distance(inside_page.Hash()), 2);
    EXPECT_EQ(tree.Distance(page_aligned.Hash()), 1);
    EXPECT_EQ(tree.Distance(longest.Hash()), 0);
}

TEST(CheckpointTree, IncompleteCheckpointsDoNotCountAsHops) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const Checkpoint* point : {&a, &b, &c}) Record(tree, *point, kOnDevice);
    tree.Update(b.Hash(), kIncomplete);
    EXPECT_EQ(tree.Distance(b.Hash()), std::nullopt);
    EXPECT_EQ(tree.Distance(a.Hash()), 1);
    tree.Update(c.Hash(), kGone);
    EXPECT_EQ(tree.Distance(a.Hash()), 0);
    EXPECT_EQ(tree.Size(), 2u);
    tree.Update(b.Hash(), kGone);
    EXPECT_EQ(tree.Size(), 1u);
}

TEST(CheckpointTree, NonLeafDistanceIsPerTierAndCountsBothTiers) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    Record(tree, a, kOnDevice);
    Record(tree, b, kOnHost);
    Record(tree, c, kOnHost);
    EXPECT_EQ(tree.NonLeafDistance(a.Hash(), CacheTier::kDevice), 2);
    EXPECT_EQ(tree.NonLeafDistance(a.Hash(), CacheTier::kHost), 0);
    EXPECT_EQ(tree.NonLeafDistance(b.Hash(), CacheTier::kHost), 1);
    EXPECT_EQ(tree.NonLeafDistance(c.Hash(), CacheTier::kHost), 0);
    std::vector<ContentHash> visited;
    tree.ForEachNonLeaf(CacheTier::kDevice, 2, [&](const ContentHash& hash) { visited.push_back(hash); });
    EXPECT_EQ(visited, std::vector<ContentHash>{a.Hash()});
}

TEST(CheckpointTree, JunctionsKeepTheirRank) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(4)}, junction{Sequence(8)}, b{Sequence(12)};
    const Checkpoint c{Concat(Sequence(8), {99, 98, 97, 96})};
    for (const Checkpoint* point : {&a, &junction, &b, &c}) Record(tree, *point, kOnDevice);
    EXPECT_EQ(tree.Distance(junction.Hash()), 1);
    EXPECT_EQ(tree.NonLeafDistance(junction.Hash(), CacheTier::kDevice), 0);
    EXPECT_EQ(tree.NonLeafDistance(a.Hash(), CacheTier::kDevice), 2);
    // One branch gone: an ordinary non-leaf again.
    tree.Update(c.Hash(), kGone);
    EXPECT_EQ(tree.NonLeafDistance(junction.Hash(), CacheTier::kDevice), 1);
}

TEST(CheckpointTree, KeepsDescendantSeesThePlannedRemovals) {
    CheckpointTree tree{kPage};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)};
    Record(tree, a, kOnDevice);
    Record(tree, b, kOnDevice);
    const auto none = [](const ContentHash&) { return false; };
    const auto b_taken = [&](const ContentHash& hash) { return hash == b.Hash(); };
    EXPECT_TRUE(tree.KeepsDescendant(a.Hash(), CacheTier::kDevice, none));
    EXPECT_FALSE(tree.KeepsDescendant(a.Hash(), CacheTier::kDevice, b_taken));
    // A Host copy stays when the Device's leaves.
    tree.Update(b.Hash(), kOnBoth);
    EXPECT_TRUE(tree.KeepsDescendant(a.Hash(), CacheTier::kDevice, b_taken));
}

TEST(CheckpointTree, LeavingCheckpointsReleaseSharedPrefixPages) {
    CheckpointTree tree{kPage};
    std::vector<Checkpoint> chain;
    for (std::int32_t pages = 1; pages <= 100; ++pages) chain.emplace_back(Sequence(pages * kPage));
    for (const Checkpoint& point : chain) Record(tree, point, kOnDevice);
    EXPECT_EQ(tree.PrefixSegments(), 100u);
    for (std::size_t i = 0; i + 1 < chain.size(); ++i) tree.Update(chain[i].Hash(), kGone);
    EXPECT_EQ(tree.Size(), 1u);
    EXPECT_EQ(tree.PrefixSegments(), 100u);
    tree.Update(chain.back().Hash(), kGone);
    EXPECT_EQ(tree.Size(), 0u);
    EXPECT_EQ(tree.PrefixSegments(), 0u);
}

std::vector<CacheGroupSpec> StateSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

// One new request that publishes `point` at its end.
void Publish(CacheCoordinator& cache, const Checkpoint& point) {
    std::vector<BlockTable> tables(2);
    auto admission = AdmitForTest(cache, tables, 0);
    ASSERT_TRUE(admission);
    cache.CacheCompletedBlocks(tables,
                               RequestProgress{.completed_pages = point.Publication(),
                                               .num_computed_tokens = static_cast<std::int32_t>(point.tokens.size())},
                               admission->access_epoch);
    cache.Free(tables);
}

bool Holds(const CacheCoordinator& cache, const BlockPool& pool, const Checkpoint& point) {
    return cache.GroupPrefixIndex(0).Contains(pool, CacheKey{.group_id = 0, .content_hash = point.Hash()}) &&
           cache.GroupPrefixIndex(1).Contains(pool, CacheKey{.group_id = 1, .content_hash = point.Hash()});
}

// Admits a new request into a full Device: it takes two fresh blocks per
// State group, so two checkpoints leave. Returns which of `points` stayed.
std::vector<bool> AdmitIntoFullDevice(CacheCoordinator& cache, BlockPool& device,
                                      std::initializer_list<const Checkpoint*> points) {
    std::vector<CacheBlockRef> pins;
    while (device.NumEmptyLcmBlocks() > 0) pins.push_back(device.AcquireBlock(0));
    std::vector<BlockTable> incoming(2);
    EXPECT_TRUE(AdmitForTest(cache, incoming, 0));
    EXPECT_EQ(incoming[0].NumBlocks(), 2);
    std::vector<bool> kept;
    for (const Checkpoint* point : points) kept.push_back(Holds(cache, device, *point));
    cache.Free(incoming);
    return kept;
}

TEST(CheckpointRetention, DeviceEvictsNonLeafCheckpointsBeforeOlderSessions) {
    for (const std::int64_t retention : {std::int64_t{0}, std::int64_t{1000}}) {
        SCOPED_TRACE(retention);
        BlockPool device(16, {1, 1});
        auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false, retention);
        const Checkpoint other{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
        for (const Checkpoint* point : {&other, &a, &b, &c}) Publish(cache, *point);
        const std::vector<bool> kept = AdmitIntoFullDevice(cache, device, {&other, &a, &b, &c});
        if (retention == 0) {
            EXPECT_EQ(kept, (std::vector<bool>{false, false, true, true})) << "plain LRU";
        } else {
            EXPECT_EQ(kept, (std::vector<bool>{true, false, false, true})) << "the larger distance first";
        }
    }
}

TEST(CheckpointRetention, StaleSessionsLeaveFirstAndRecentOnesOutlastNonLeafCheckpoints) {
    BlockPool device(16, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false,
                                 /*checkpoint_retention_requests=*/4);
    const Checkpoint stale{{90, 91, 92, 93}}, recent{{80, 81, 82, 83}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const Checkpoint* point : {&stale, &recent, &a, &b, &c}) Publish(cache, *point);
    // The last four admissions began with `recent`'s: `stale` is older, and A
    // and B rank as if used then, ahead of `recent` itself.
    EXPECT_EQ(AdmitIntoFullDevice(cache, device, {&stale, &recent, &a, &b, &c}),
              (std::vector<bool>{false, true, false, true, true}));
}

TEST(CheckpointRetention, PlannedRemovalOfTheOnlyDescendantRestoresTheLeafRank) {
    BlockPool device(16, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false,
                                 /*checkpoint_retention_requests=*/1000);
    const Checkpoint older{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)};
    for (const Checkpoint* point : {&older, &a, &b}) Publish(cache, *point);
    // B ranks first, at epoch 0; once the plan takes it, A is a leaf again
    // and ranks by its own use, after the older session.
    for (std::int32_t group : {0, 1}) {
        const CacheKey key{.group_id = static_cast<std::uint32_t>(group), .content_hash = b.Hash()};
        CacheBlockRef block = cache.GroupPrefixIndex(group).Find(device, key);
        ASSERT_TRUE(block);
        cache.GroupPrefixIndex(group).Register(device, block, key, /*access_epoch=*/0, /*logical_block_index=*/1,
                                               CacheBoundaryKind::kEndpoint, /*newly_cached=*/nullptr,
                                               /*session=*/nullptr);
    }
    EXPECT_EQ(AdmitIntoFullDevice(cache, device, {&older, &a, &b}), (std::vector<bool>{false, true, false}));
}

TEST(CheckpointRetention, HostPlacementRanksNonLeafCheckpoints) {
    for (const std::int64_t retention : {std::int64_t{0}, std::int64_t{1000}}) {
        for (const int placement : {0, 1, 2}) {
            SCOPED_TRACE(testing::Message() << "retention " << retention << ", placement " << placement);
            BlockPool device(16, {1, 1});
            BlockPool host(16, {1, 1});
            auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, &host, nullptr, false, false, retention);
            const Checkpoint other{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
            for (const Checkpoint* point : {&other, &a, &b, &c}) {
                Publish(cache, *point);
                for (std::uint32_t group : {0u, 1u}) {
                    CacheBlockRef block = cache.AcquireHostBlock(group, 0);
                    ASSERT_TRUE(block);
                    cache.CacheHostBlock(block, CacheKey{.group_id = group, .content_hash = point->Hash()},
                                         std::nullopt);
                }
            }
            ASSERT_TRUE(cache.ClearDeviceCache());
            std::vector<CacheBlockRef> pins;
            while (host.NumEmptyLcmBlocks() > 0) pins.push_back(host.AcquireBlock(0));
            const std::array<std::uint32_t, 1> groups{0};
            const std::array<std::int32_t, 1> buckets{0};
            const std::array boundaries{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
            const auto result = placement == 0   ? cache.AcquireHostBlocks(groups, buckets)
                                : placement == 1 ? cache.AcquireHostBoundaries(groups, buckets, boundaries)
                                                 : cache.AcquireHostSegments(groups, buckets, boundaries);
            ASSERT_EQ(result.stats.allocated, 1u);
            const std::vector<bool> kept{Holds(cache, host, other), Holds(cache, host, a), Holds(cache, host, b),
                                         Holds(cache, host, c)};
            if (retention == 0) {
                EXPECT_EQ(kept, (std::vector<bool>{false, true, true, true}));
            } else {
                EXPECT_EQ(kept, (std::vector<bool>{true, false, true, true}));
            }
        }
    }
}

TEST(CheckpointRetention, HostPlacementTakesTheOlderOfEqualDistances) {
    for (const int placement : {0, 1, 2}) {
        SCOPED_TRACE(placement);
        BlockPool device(16, {1, 1});
        BlockPool host(16, {1, 1});
        auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, &host, nullptr, false, false,
                                     /*checkpoint_retention_requests=*/1000);
        // A is older and deeper than B; each has one newer checkpoint after it.
        const Checkpoint a{{10, 11, 12, 13, 14, 15, 16, 17}}, a_next{{10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}};
        const Checkpoint b{{50, 51, 52, 53}}, b_next{{50, 51, 52, 53, 54, 55, 56, 57}};
        for (const Checkpoint* point : {&a, &a_next, &b, &b_next}) {
            Publish(cache, *point);
            for (std::uint32_t group : {0u, 1u}) {
                CacheBlockRef block = cache.AcquireHostBlock(group, 0);
                ASSERT_TRUE(block);
                cache.CacheHostBlock(block, CacheKey{.group_id = group, .content_hash = point->Hash()}, std::nullopt);
            }
        }
        ASSERT_TRUE(cache.ClearDeviceCache());
        std::vector<CacheBlockRef> pins;
        while (host.NumEmptyLcmBlocks() > 0) pins.push_back(host.AcquireBlock(0));
        const std::array<std::uint32_t, 1> groups{0};
        const std::array<std::int32_t, 1> buckets{0};
        const std::array boundaries{CacheCoordinator::HostSegmentRows{.first_row = 0, .past_end_row = 1}};
        const auto result = placement == 0   ? cache.AcquireHostBlocks(groups, buckets)
                            : placement == 1 ? cache.AcquireHostBoundaries(groups, buckets, boundaries)
                                             : cache.AcquireHostSegments(groups, buckets, boundaries);
        ASSERT_EQ(result.stats.allocated, 1u);
        EXPECT_FALSE(Holds(cache, host, a));
        EXPECT_TRUE(Holds(cache, host, b));
    }
}

TEST(CheckpointRetention, ResumeGapCountsNewRequestsSinceTheCheckpointsLastUse) {
    BlockPool device(32, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false,
                                 /*checkpoint_retention_requests=*/0);
    const Checkpoint a{Sequence(4)};
    Publish(cache, a);
    for (const std::int32_t first : {100, 200, 300})
        Publish(cache, Checkpoint{{first, first + 1, first + 2, first + 3}});
    std::vector<BlockTable> tables(2);
    auto admission = AdmitForTest(
        cache, tables,
        cache.ProbePromptPrefix(CacheCoordinator::PromptPrefix{
            .prefix_hashes = a.hashes, .tokens = a.tokens, .max_tokens = static_cast<std::int32_t>(a.tokens.size())}),
        GroupDemand{.extent = DenseGrowth{0}});
    ASSERT_TRUE(admission);
    EXPECT_EQ(admission->device_prefix_tokens, 4);
    // Three new requests came between: index 2 counts [2, 4).
    std::vector<std::uint64_t> expected(cache.CheckpointResumeGaps().size(), 0);
    expected[2] = 1;
    EXPECT_EQ(cache.CheckpointResumeGaps(), expected);
    cache.Free(tables);
}

TEST(CheckpointRetention, DefaultIsSixteenBatchesOfNewRequests) {
    SchedulerConfig config;
    config.max_batch_size = 8;
    EXPECT_EQ(config.CheckpointRetentionRequests(), 128);
    config.checkpoint_retention_requests = 0;
    EXPECT_EQ(config.CheckpointRetentionRequests(), 0);
    config.max_batch_size = 1 << 20;
    config.checkpoint_retention_requests.reset();
    EXPECT_EQ(config.CheckpointRetentionRequests(), kMaxCheckpointRetentionRequests);
}

}  // namespace
}  // namespace tokenspeed::test
