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
#include <numeric>
#include <unordered_set>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/prefix/checkpoint_tree.h"
#include "cache/prefix/prefix_hasher.h"
#include "cache_test_access.h"

namespace tokenspeed::test {
namespace {

struct Checkpoint {
    std::vector<std::int32_t> tokens;
    std::vector<ContentHash> hashes;
    std::optional<TailPublication> tail;

    explicit Checkpoint(std::vector<std::int32_t> input) : tokens{std::move(input)} {
        ContentHash prior;
        for (std::size_t i = 0; i + 4 <= tokens.size(); i += 4) {
            prior = HashPrefixPage(std::span{tokens}.subspan(i, 4), prior);
            hashes.push_back(prior);
        }
        if (tokens.size() % 4 != 0) {
            tail = TailPublication{.tail_tokens = static_cast<std::int32_t>(tokens.size() % 4),
                                   .hash = HashPrefixPage(std::span{tokens}.subspan(hashes.size() * 4), prior)};
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
    std::vector<std::int32_t> tokens(count);
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

TEST(CheckpointRetention, RequiresTwoCurrentlyResidentDescendants) {
    CheckpointTree tree{4};
    Checkpoint a{Sequence(5)}, b{Sequence(9)}, c{Sequence(13)};
    std::unordered_set<ContentHash> resident{a.Hash(), b.Hash(), c.Hash()};
    const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
    tree.Observe(a.Publication());
    tree.Observe(b.Publication());
    EXPECT_FALSE(tree.Demoted(a.Hash(), exists));
    tree.Observe(c.Publication());
    EXPECT_TRUE(tree.Demoted(a.Hash(), exists));
    EXPECT_FALSE(tree.Demoted(b.Hash(), exists));
    resident.erase(b.Hash());
    EXPECT_FALSE(tree.Demoted(a.Hash(), exists));
    resident.insert(b.Hash());
    resident.erase(c.Hash());
    EXPECT_FALSE(tree.Demoted(a.Hash(), exists));
}

TEST(CheckpointRetention, RewindInsertsEarlierCheckpointAndBranchesRemainProtected) {
    CheckpointTree tree{4};
    Checkpoint a{Sequence(3)}, b{Sequence(7)}, c{Sequence(11)}, rewind{Sequence(5)};
    for (const auto* point : {&a, &b, &c}) tree.Observe(point->Publication());
    const auto resident = [](const ContentHash&) { return true; };
    EXPECT_TRUE(tree.Demoted(a.Hash(), resident));
    tree.Observe(rewind.Publication());
    EXPECT_TRUE(tree.Demoted(rewind.Hash(), resident));
    auto branch_tokens = Sequence(5);
    branch_tokens.push_back(99);
    Checkpoint branch{branch_tokens};
    tree.Observe(branch.Publication());
    EXPECT_FALSE(tree.Demoted(rewind.Hash(), resident));
    EXPECT_FALSE(tree.Demoted(a.Hash(), resident));
}

TEST(CheckpointRetention, RewindToPreviouslyUnpublishedForkRemainsProtected) {
    for (bool with_parent : {false, true}) {
        SCOPED_TRACE(with_parent);
        CheckpointTree tree{4};
        Checkpoint parent{Sequence(2)}, x{Sequence(12)}, y{{1, 2, 3, 4, 99, 98, 97, 96}}, z{Sequence(16)},
            rewind{Sequence(4)};
        if (with_parent) tree.Observe(parent.Publication());
        for (const auto* point : {&x, &y, &z}) tree.Observe(point->Publication());
        tree.Prune([&](const ContentHash& hash) { return hash != y.Hash(); });
        tree.Observe(rewind.Publication());
        EXPECT_FALSE(tree.Demoted(rewind.Hash(), [](const ContentHash&) { return true; }));
    }
}

TEST(CheckpointRetention, DivergentTailIsNotAContinuation) {
    CheckpointTree tree{4};
    Checkpoint a{{1, 2, 3, 4, 5}}, b{{1, 2, 3, 4, 99, 6, 7, 8, 9}}, c{{1, 2, 3, 4, 99, 6, 7, 8, 9, 10, 11}};
    for (const auto* point : {&a, &b, &c}) tree.Observe(point->Publication());
    EXPECT_FALSE(tree.Demoted(a.Hash(), [](const ContentHash&) { return true; }));
}

TEST(CheckpointRetention, PruningPreservesForkBarrierAndClearForgetsHistory) {
    CheckpointTree tree{4};
    Checkpoint a{Sequence(4)}, fork{Sequence(8)}, b{Sequence(12)}, c{Sequence(16)};
    auto alternate = Sequence(8);
    alternate.insert(alternate.end(), {99, 98, 97, 96});
    Checkpoint branch{alternate};
    for (const auto* point : {&a, &fork, &b, &c, &branch}) tree.Observe(point->Publication());
    const std::unordered_set<ContentHash> resident{a.Hash(), b.Hash(), c.Hash()};
    const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
    tree.Prune(exists);
    EXPECT_FALSE(tree.Demoted(a.Hash(), exists));
    EXPECT_LE(tree.Size(), resident.size() * 2);
    tree.Clear();
    EXPECT_EQ(tree.Size(), 0);
    EXPECT_FALSE(tree.Demoted(a.Hash(), exists));
}

TEST(CheckpointRetention, PruningPreservesEveryUnpublishedFork) {
    for (const auto rewinds : {std::array{4, 8}, std::array{8, 4}}) {
        SCOPED_TRACE(rewinds[0]);
        for (bool with_parent : {false, true}) {
            SCOPED_TRACE(with_parent);
            CheckpointTree tree{4};
            Checkpoint parent{Sequence(2)}, a{Sequence(16)}, b{Sequence(20)}, c{Sequence(24)};
            Checkpoint first_branch{{1, 2, 3, 4, 99, 98, 97, 96}};
            Checkpoint second_branch{{1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97, 96}};
            std::unordered_set<ContentHash> resident{a.Hash(), b.Hash(), c.Hash()};
            if (with_parent) {
                tree.Observe(parent.Publication());
                resident.insert(parent.Hash());
            }
            for (const auto* point : {&a, &first_branch, &second_branch, &b, &c}) {
                tree.Observe(point->Publication());
            }
            const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
            tree.Prune(exists);
            for (const auto length : rewinds) {
                SCOPED_TRACE(length);
                Checkpoint rewind{Sequence(length)};
                tree.Observe(rewind.Publication());
                resident.insert(rewind.Hash());
                EXPECT_FALSE(tree.Demoted(rewind.Hash(), exists));
            }
        }
    }
}

TEST(CheckpointRetention, InsertedCheckpointBetweenPrunedForksRemainsProtected) {
    CheckpointTree tree{4};
    Checkpoint parent{Sequence(4)}, longest{Sequence(16)};
    Checkpoint first_branch{{1, 2, 3, 4, 5, 6, 99, 98}};
    Checkpoint second_branch{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 98}};
    for (const auto* point : {&parent, &longest, &first_branch, &second_branch}) {
        tree.Observe(point->Publication());
    }
    std::unordered_set<ContentHash> resident{parent.Hash(), longest.Hash()};
    const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
    tree.Prune(exists);
    Checkpoint earlier{Sequence(8)}, later{Sequence(12)};
    for (const auto* point : {&earlier, &later}) {
        tree.Observe(point->Publication());
        resident.insert(point->Hash());
    }
    EXPECT_FALSE(tree.Demoted(earlier.Hash(), exists));
}

TEST(CheckpointRetention, PrunedLeafPassesOlderForksToSurvivingSibling) {
    CheckpointTree tree{4};
    Checkpoint parent{Sequence(4)}, longest{Sequence(12)};
    Checkpoint first_branch{{1, 2, 3, 4, 5, 6, 99, 98}};
    for (const auto* point : {&parent, &longest, &first_branch}) {
        tree.Observe(point->Publication());
    }
    std::unordered_set<ContentHash> resident{parent.Hash(), longest.Hash()};
    const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
    tree.Prune(exists);
    Checkpoint sibling{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 76}};
    tree.Observe(sibling.Publication());
    resident = {parent.Hash(), sibling.Hash()};
    tree.Prune(exists);
    Checkpoint earlier{Sequence(8)}, later{Sequence(9)};
    for (const auto* point : {&earlier, &later}) {
        tree.Observe(point->Publication());
        resident.insert(point->Hash());
    }
    EXPECT_FALSE(tree.Demoted(parent.Hash(), exists));
    EXPECT_FALSE(tree.Demoted(earlier.Hash(), exists));
}

TEST(CheckpointRetention, PruningDiscardsForksBeyondRetainedPrefixes) {
    CheckpointTree tree{4};
    Checkpoint parent{Sequence(4)}, longest{Sequence(12)};
    Checkpoint branch{{1, 2, 3, 4, 5, 6, 99, 98}};
    for (const auto* point : {&parent, &longest, &branch}) tree.Observe(point->Publication());
    std::unordered_set<ContentHash> resident{parent.Hash()};
    const auto exists = [&](const ContentHash& hash) { return resident.contains(hash); };
    tree.Prune(exists);
    Checkpoint earlier{Sequence(8)}, later{Sequence(9)};
    for (const auto* point : {&earlier, &later}) {
        tree.Observe(point->Publication());
        resident.insert(point->Hash());
    }
    EXPECT_TRUE(tree.Demoted(parent.Hash(), exists));
}

std::vector<CacheGroupSpec> StateSpecs() {
    return {CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
            CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
}

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

TEST(CheckpointRetention, DeviceEvictsSupersededCheckpointBeforeOlderUnrelatedCheckpoint) {
    BlockPool device(10, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), 4, device, false, nullptr, nullptr, false);
    Checkpoint unrelated{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const auto* point : {&unrelated, &a, &b, &c}) Publish(cache, *point);
    std::vector<BlockTable> incoming(2);
    ASSERT_TRUE(AdmitForTest(cache, incoming, 0));
    for (std::uint32_t group : {0u, 1u}) {
        EXPECT_TRUE(cache.GroupPrefixIndex(group).Contains(
            device, CacheKey{.group_id = group, .content_hash = unrelated.Hash()}));
        EXPECT_FALSE(
            cache.GroupPrefixIndex(group).Contains(device, CacheKey{.group_id = group, .content_hash = a.Hash()}));
    }
    cache.Free(incoming);
}

TEST(CheckpointRetention, MissingOneStateGroupPreventsDemotion) {
    BlockPool device(10, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), 4, device, false, nullptr, nullptr, false);
    Checkpoint unrelated{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const auto* point : {&unrelated, &a, &b, &c}) Publish(cache, *point);
    CacheBlockLocation missing;
    {
        auto ref = cache.GroupPrefixIndex(1).Find(device, CacheKey{.group_id = 1, .content_hash = c.Hash()});
        ASSERT_TRUE(ref);
        missing = ref->Location();
    }
    ASSERT_TRUE(cache.GroupPrefixIndex(1).Evict(device, missing));
    auto pinned = device.AcquireBlock(1);
    std::vector<BlockTable> incoming(2);
    ASSERT_TRUE(AdmitForTest(cache, incoming, 0));
    for (std::uint32_t group : {0u, 1u}) {
        EXPECT_TRUE(
            cache.GroupPrefixIndex(group).Contains(device, CacheKey{.group_id = group, .content_hash = a.Hash()}));
        EXPECT_FALSE(cache.GroupPrefixIndex(group).Contains(
            device, CacheKey{.group_id = group, .content_hash = unrelated.Hash()}));
    }
    cache.Free(incoming);
}

TEST(CheckpointRetention, HostUsesItsOwnCompleteDescendants) {
    BlockPool device(16, {1, 1});
    BlockPool host(8, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), 4, device, false, &host, nullptr, false);
    Checkpoint unrelated{{90, 91, 92, 93}}, a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)};
    for (const auto* point : {&unrelated, &a, &b, &c}) {
        Publish(cache, *point);
        for (std::uint32_t group : {0u, 1u}) {
            auto block = cache.AcquireHostBlock(group, 0);
            ASSERT_TRUE(block);
            cache.CacheHostBlock(block, CacheKey{.group_id = group, .content_hash = point->Hash()});
        }
    }
    auto fresh = cache.AcquireHostBlock(0, 0);
    ASSERT_TRUE(fresh);
    EXPECT_TRUE(cache.GroupPrefixIndex(0).Contains(host, CacheKey{.group_id = 0, .content_hash = unrelated.Hash()}));
    EXPECT_FALSE(cache.GroupPrefixIndex(0).Contains(host, CacheKey{.group_id = 0, .content_hash = a.Hash()}));
}

TEST(CheckpointRetention, CacheClearForgetsPreviousForks) {
    BlockPool device(10, {1, 1});
    auto cache = MakeCoordinator(StateSpecs(), 4, device, false, nullptr, nullptr, false);
    Checkpoint a{Sequence(4)}, b{Sequence(8)}, c{Sequence(12)}, branch{{1, 2, 3, 4, 99, 98, 97, 96}};
    for (const auto* point : {&a, &b, &branch}) Publish(cache, *point);
    ASSERT_TRUE(cache.ClearCache());
    Checkpoint unrelated{{90, 91, 92, 93}};
    for (const auto* point : {&a, &unrelated, &b, &c}) Publish(cache, *point);
    {
        auto hit = MatchPrefixForTest(cache, a.hashes);
        EXPECT_EQ(hit.device.num_common_tokens, 4);
    }
    std::vector<BlockTable> incoming(2);
    ASSERT_TRUE(AdmitForTest(cache, incoming, 0));
    for (std::uint32_t group : {0u, 1u}) {
        EXPECT_TRUE(cache.GroupPrefixIndex(group).Contains(
            device, CacheKey{.group_id = group, .content_hash = unrelated.Hash()}));
        EXPECT_FALSE(
            cache.GroupPrefixIndex(group).Contains(device, CacheKey{.group_id = group, .content_hash = a.Hash()}));
    }
    cache.Free(incoming);
}

TEST(CheckpointRetention, PruningBoundsLongConversationMetadata) {
    CheckpointTree tree{4};
    const Checkpoint root{Sequence(1)};
    tree.Observe(root.Publication());
    std::unordered_set<ContentHash> resident{root.Hash()};
    for (std::int32_t length = 2; length < 100; ++length) {
        Checkpoint point{Sequence(length)};
        tree.Observe(point.Publication());
        resident = {root.Hash(), point.Hash()};
        tree.Prune([&](const ContentHash& hash) { return resident.contains(hash); });
        EXPECT_LE(tree.Size(), 2);
        EXPECT_FALSE(tree.Demoted(root.Hash(), [&](const ContentHash& hash) { return resident.contains(hash); }));
    }
}

TEST(CheckpointRetention, HybridTailEvictionPreservesLongerReusableCheckpoints) {
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 8, .block_granularity = 4},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0},
        CacheGroupSpec{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0}};
    BlockPool device(16, {8, 1, 1});
    auto cache = MakeCoordinator(specs, 4, device, false, nullptr, nullptr, false);
    Checkpoint unrelated{{90, 91, 92, 93}}, a{Sequence(5)}, b{Sequence(9)}, c{Sequence(13)};
    for (const auto* point : {&unrelated, &a, &b, &c}) {
        std::vector<BlockTable> tables(3);
        auto admitted = AdmitForTest(cache, tables, static_cast<std::int32_t>(point->tokens.size()));
        ASSERT_TRUE(admitted);
        cache.CacheCompletedBlocks(
            tables,
            RequestProgress{.completed_pages = point->Publication(),
                            .num_computed_tokens = static_cast<std::int32_t>(point->tokens.size())},
            admitted->access_epoch);
        cache.Free(tables);
    }
    std::vector<CacheBlockRef> pins;
    while (device.NumEmptyLcmBlocks() > 2) pins.push_back(device.AcquireBlock(1));
    std::vector<BlockTable> incoming(3);
    ASSERT_TRUE(AdmitForTest(cache, incoming, 0));
    for (std::uint32_t group : {0u, 1u, 2u}) {
        EXPECT_FALSE(
            cache.GroupPrefixIndex(group).Contains(device, CacheKey{.group_id = group, .content_hash = a.Hash()}));
        EXPECT_TRUE(cache.GroupPrefixIndex(group).Contains(
            device, CacheKey{.group_id = group, .content_hash = unrelated.Hash()}));
    }
    for (const auto* point : {&b, &c}) {
        auto probe = cache.ProbePromptPrefix(
            CacheCoordinator::PromptPrefix{.prefix_hashes = point->hashes,
                                           .tokens = point->tokens,
                                           .max_tokens = static_cast<std::int32_t>(point->tokens.size())});
        EXPECT_EQ(probe.device.num_common_tokens, point->tokens.size());
    }
    EXPECT_EQ(cache.CheckpointRetentionStats().device_demoted_victims, 2);
    cache.Free(incoming);
    pins.clear();
    ASSERT_TRUE(cache.ClearCache());
    EXPECT_EQ(cache.CheckpointRetentionStats().device_demoted_victims, 0);
    EXPECT_EQ(cache.CheckpointRetentionStats().checkpoint_nodes, 0);
    EXPECT_EQ(cache.CheckpointRetentionStats().prefix_segments, 0);
}

TEST(CheckpointRetention, CheckpointsSharePrefixMetadata) {
    CheckpointTree tree{4};
    for (std::int32_t length = 4; length <= 400; length += 4) {
        Checkpoint point{Sequence(length)};
        tree.Observe(point.Publication());
    }
    EXPECT_EQ(tree.Size(), 100);
    EXPECT_EQ(tree.PrefixSegments(), 100);
    Checkpoint last{Sequence(400)};
    tree.Prune([&](const ContentHash& hash) { return hash == last.Hash(); });
    EXPECT_EQ(tree.Size(), 1);
    EXPECT_EQ(tree.PrefixSegments(), 100);
    tree.Prune([](const ContentHash&) { return false; });
    EXPECT_EQ(tree.PrefixSegments(), 0);
}

TEST(CheckpointRetention, DeviceHitRefreshesExistingHostCopy) {
    BlockPool device(8, {1});
    BlockPool host(2, {1});
    const std::array specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 4}};
    auto cache = MakeCoordinator(specs, 4, device, false, &host, nullptr, false);
    const CacheKey a{.group_id = 0, .content_hash = "a"}, b{.group_id = 0, .content_hash = "b"};
    for (const auto& key : {a, b}) {
        auto gpu = device.AcquireBlock(0);
        cache.GroupPrefixIndex(0).Register(device, gpu, key, CacheCoordinatorTestAccess::NextAccessEpoch(cache), 0,
                                           CacheBoundaryKind::kEndpoint, nullptr);
        auto cpu = cache.AcquireHostBlock(0, 0);
        cache.CacheHostBlock(cpu, key);
    }
    const std::array hashes{a.content_hash};
    {
        auto hit = MatchPrefixForTest(cache, hashes);
        EXPECT_EQ(hit.device.num_common_tokens, 4);
    }
    auto fresh = cache.AcquireHostBlock(0, 0);
    ASSERT_TRUE(fresh);
    EXPECT_TRUE(cache.GroupPrefixIndex(0).Contains(host, a));
    EXPECT_FALSE(cache.GroupPrefixIndex(0).Contains(host, b));
}

}  // namespace
}  // namespace tokenspeed::test
