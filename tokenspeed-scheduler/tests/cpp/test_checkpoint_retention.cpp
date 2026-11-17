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
    std::vector<std::string> hashes;
    std::optional<TailPublication> tail;

    explicit Checkpoint(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + kPage <= tokens.size(); i += kPage) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, kPage), prior);
            hashes.push_back(prior.ToString());
        }
        if (tokens.size() % kPage != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % kPage),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * kPage), prior)};
        }
    }
    ContentHash Hash() const { return tail ? tail->hash : ContentHash{hashes.back()}; }
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

// Publishes `point` from the request of `access_epoch`.
void RecordFor(CheckpointTree& tree, const Checkpoint& point, CheckpointTree::Residency residency,
               std::uint64_t access_epoch) {
    tree.Observe(point.tokens, point.hashes, point.tail, access_epoch);
    tree.Update(point.Hash(), residency);
}

// Publishes `point` from a new request.
void Record(CheckpointTree& tree, const Checkpoint& point, CheckpointTree::Residency residency) {
    static std::uint64_t last_epoch = 0;
    RecordFor(tree, point, residency, ++last_epoch);
}

TEST(CheckpointTree, DistanceCountsRequestsToTheNearestLeaf) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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

TEST(CheckpointTree, AgedDistanceIsPerTierAndCountsBothTiers) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    Record(tree, a, kOnDevice);
    Record(tree, b, kOnHost);
    Record(tree, c, kOnHost);
    EXPECT_EQ(tree.AgedDistance(a.Hash(), CacheTier::kDevice), 2);
    EXPECT_EQ(tree.AgedDistance(a.Hash(), CacheTier::kHost), 0);
    EXPECT_EQ(tree.AgedDistance(b.Hash(), CacheTier::kHost), 1);
    EXPECT_EQ(tree.AgedDistance(c.Hash(), CacheTier::kHost), 0);
    std::vector<ContentHash> visited;
    tree.ForEachAged(CacheTier::kDevice, 2, [&](const ContentHash& hash) { visited.push_back(hash); });
    EXPECT_EQ(visited, std::vector<ContentHash>{a.Hash()});
}

TEST(CheckpointTree, JunctionsKeepTheirRank) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint a{Sequence(4)}, junction{Sequence(8)}, b{Sequence(12)};
    const Checkpoint c{Concat(Sequence(8), {99, 98, 97, 96})};
    for (const Checkpoint* point : {&a, &junction, &b, &c}) Record(tree, *point, kOnDevice);
    EXPECT_EQ(tree.Distance(junction.Hash()), 1);
    EXPECT_EQ(tree.AgedDistance(junction.Hash(), CacheTier::kDevice), 0);
    EXPECT_EQ(tree.AgedDistance(a.Hash(), CacheTier::kDevice), 2);
    // One branch gone: aged again.
    tree.Update(c.Hash(), kGone);
    EXPECT_EQ(tree.AgedDistance(junction.Hash(), CacheTier::kDevice), 1);
}

TEST(CheckpointTree, StaysAgedSeesThePlannedRemovals) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)};
    Record(tree, a, kOnDevice);
    Record(tree, b, kOnDevice);
    const auto none = [](const ContentHash&) { return false; };
    const auto b_taken = [&](const ContentHash& hash) { return hash == b.Hash(); };
    EXPECT_TRUE(tree.StaysAged(a.Hash(), CacheTier::kDevice, none));
    EXPECT_FALSE(tree.StaysAged(a.Hash(), CacheTier::kDevice, b_taken));
    // A Host copy stays when the Device's leaves.
    tree.Update(b.Hash(), kOnBoth);
    EXPECT_TRUE(tree.StaysAged(a.Hash(), CacheTier::kDevice, b_taken));
}

TEST(CheckpointTree, LeafDistanceRanksNearbyCheckpointsAsLeaves) {
    CheckpointTree tree{kPage, /*leaf_distance=*/1};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)}, d{Sequence(16)};
    for (const Checkpoint* point : {&a, &b, &c, &d}) Record(tree, *point, kOnDevice);
    EXPECT_EQ(tree.Distance(c.Hash()), 1);
    EXPECT_EQ(tree.AgedDistance(c.Hash(), CacheTier::kDevice), 0);
    EXPECT_EQ(tree.AgedDistance(b.Hash(), CacheTier::kDevice), 2);
    EXPECT_EQ(tree.AgedDistance(a.Hash(), CacheTier::kDevice), 3);
    std::vector<ContentHash> visited;
    tree.ForEachAged(CacheTier::kDevice, 2, [&](const ContentHash& hash) { visited.push_back(hash); });
    EXPECT_EQ(visited, std::vector<ContentHash>{b.Hash()});
    // With D taken, C is the leaf: B comes within the leaf distance, A stays out.
    const auto d_taken = [&](const ContentHash& hash) { return hash == d.Hash(); };
    EXPECT_FALSE(tree.StaysAged(b.Hash(), CacheTier::kDevice, d_taken));
    EXPECT_TRUE(tree.StaysAged(a.Hash(), CacheTier::kDevice, d_taken));
}

// One turn publishes an in-forward checkpoint (on the Host only, below the
// Device copies of the rest), its prompt end and its end state; the next turn
// extends the end state.
TEST(CheckpointTree, OneRequestsCheckpointsAreNoStepApart) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint in_forward{Sequence(8)}, prompt_end{Sequence(10)}, end_state{Sequence(14)};
    RecordFor(tree, in_forward, kOnHost, 1);
    RecordFor(tree, prompt_end, kOnDevice, 1);
    RecordFor(tree, end_state, kOnDevice, 1);
    EXPECT_EQ(tree.AgedDistance(in_forward.Hash(), CacheTier::kHost), 0);
    for (const Checkpoint* point : {&in_forward, &prompt_end, &end_state}) {
        EXPECT_EQ(tree.Distance(point->Hash()), 0);
    }
    EXPECT_EQ(tree.AgedDistance(prompt_end.Hash(), CacheTier::kDevice), 0);
    const Checkpoint next_prompt_end{Sequence(18)}, next_end_state{Sequence(22)};
    RecordFor(tree, next_prompt_end, kOnDevice, 2);
    RecordFor(tree, next_end_state, kOnDevice, 2);
    EXPECT_EQ(tree.AgedDistance(in_forward.Hash(), CacheTier::kHost), 1);
    for (const Checkpoint* point : {&prompt_end, &end_state}) {
        EXPECT_EQ(tree.AgedDistance(point->Hash(), CacheTier::kDevice), 1);
    }
    EXPECT_EQ(tree.Distance(next_prompt_end.Hash()), 0);
}

// A checkpoint compares its request with the nearest present checkpoint
// above it, which changes as checkpoints between them come and go.
TEST(CheckpointTree, ACheckpointBetweenTwoOfOneRequestsSplitsThem) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint first{Sequence(4)}, between{Sequence(8)}, last{Sequence(12)};
    RecordFor(tree, first, kOnDevice, 1);
    RecordFor(tree, last, kOnDevice, 1);
    EXPECT_EQ(tree.Distance(first.Hash()), 0);
    RecordFor(tree, between, kIncomplete, 2);
    EXPECT_EQ(tree.Distance(first.Hash()), 0);
    tree.Update(between.Hash(), kOnDevice);
    EXPECT_EQ(tree.Distance(between.Hash()), 1);
    EXPECT_EQ(tree.Distance(first.Hash()), 2);
    tree.Update(between.Hash(), kIncomplete);
    EXPECT_EQ(tree.Distance(first.Hash()), 0);
    tree.Update(between.Hash(), kOnDevice);
    tree.Update(between.Hash(), kGone);
    EXPECT_EQ(tree.Distance(first.Hash()), 0);
    EXPECT_EQ(tree.AgedDistance(first.Hash(), CacheTier::kDevice), 0);
}

TEST(CheckpointTree, StaysAgedCountsRequestsNotCheckpoints) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
    const Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    RecordFor(tree, a, kOnDevice, 1);
    RecordFor(tree, b, kOnDevice, 1);
    RecordFor(tree, c, kOnDevice, 2);
    EXPECT_EQ(tree.AgedDistance(a.Hash(), CacheTier::kDevice), 1);
    const auto b_taken = [&](const ContentHash& hash) { return hash == b.Hash(); };
    const auto c_taken = [&](const ContentHash& hash) { return hash == c.Hash(); };
    // Without C, A and B are the newest request's: neither stays aged.
    EXPECT_FALSE(tree.StaysAged(a.Hash(), CacheTier::kDevice, c_taken));
    // Without B, C is still a later request than A.
    EXPECT_TRUE(tree.StaysAged(a.Hash(), CacheTier::kDevice, b_taken));
}

TEST(CheckpointTree, LeavingCheckpointsReleaseSharedPrefixPages) {
    CheckpointTree tree{kPage, /*leaf_distance=*/0};
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

// DeviceEvictsNonLeafCheckpointsBeforeOlderSessions with K = 1: B ranks as a
// leaf, so LRU takes the other session instead.
TEST(CheckpointRetention, LeafDistanceKeepsThePreviousTurnsCheckpoint) {
    BlockPool device(16, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false,
                                 /*checkpoint_retention_requests=*/1000, /*checkpoint_retention_leaf_distance=*/1);
    const Checkpoint other{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const Checkpoint* point : {&other, &a, &b, &c}) Publish(cache, *point);
    EXPECT_EQ(AdmitIntoFullDevice(cache, device, {&other, &a, &b, &c}), (std::vector<bool>{false, false, true, true}));
}

TEST(CheckpointRetention, LeafDistanceAppliesToEveryHostPlacement) {
    for (const std::int32_t leaf_distance : {0, 1}) {
        for (const int placement : {0, 1, 2}) {
            SCOPED_TRACE(testing::Message() << "leaf distance " << leaf_distance << ", placement " << placement);
            BlockPool device(16, {1, 1});
            BlockPool host(16, {1, 1});
            auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, &host, nullptr, false, false,
                                         /*checkpoint_retention_requests=*/1000, leaf_distance);
            const Checkpoint other{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)};
            for (const Checkpoint* point : {&other, &a, &b}) {
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
            // A, at distance 1, is aged only with leaf distance 0.
            EXPECT_EQ(Holds(cache, host, other), leaf_distance == 0);
            EXPECT_EQ(Holds(cache, host, a), leaf_distance == 1);
        }
    }
}

TEST(CheckpointRetention, PlannedRemovalCanBringACheckpointWithinTheLeafDistance) {
    BlockPool device(16, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), kPage, device, false, nullptr, nullptr, false, false,
                                 /*checkpoint_retention_requests=*/1000, /*checkpoint_retention_leaf_distance=*/1);
    const Checkpoint older{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const Checkpoint* point : {&older, &a, &b, &c}) Publish(cache, *point);
    // C ranks first, at epoch 0; once the plan takes it, B is the leaf and A
    // comes within the leaf distance, ranking by its own use after older.
    for (std::int32_t group : {0, 1}) {
        const CacheKey key{.group_id = static_cast<std::uint32_t>(group), .content_hash = c.Hash()};
        CacheBlockRef block = cache.GroupPrefixIndex(group).Find(device, key);
        ASSERT_TRUE(block);
        cache.GroupPrefixIndex(group).Register(device, block, key, /*access_epoch=*/0, /*logical_block_index=*/2,
                                               CacheBoundaryKind::kEndpoint, /*newly_cached=*/nullptr,
                                               /*session=*/nullptr);
    }
    EXPECT_EQ(AdmitIntoFullDevice(cache, device, {&older, &a, &b, &c}), (std::vector<bool>{false, true, true, false}));
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
    config.max_resident_requests = config.max_batch_size;
    EXPECT_EQ(config.CheckpointRetentionRequests(), 128);
    EXPECT_EQ(config.checkpoint_retention_leaf_distance, 0);
    config.checkpoint_retention_requests = 0;
    EXPECT_EQ(config.CheckpointRetentionRequests(), 0);
    config.max_batch_size = 1 << 20;
    config.max_resident_requests = config.max_batch_size;
    config.checkpoint_retention_requests.reset();
    EXPECT_EQ(config.CheckpointRetentionRequests(), kMaxCheckpointRetentionRequests);
}

}  // namespace
}  // namespace tokenspeed::test
