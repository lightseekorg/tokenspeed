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

#include "scheduler/kv_cache_events.h"
#include "integration_test_helper.h"
#include <array>
#include <limits>

#include <gtest/gtest.h>
#include <optional>
#include <array>
#include <random>
#include <set>
#include <vector>

namespace tokenspeed::test {
namespace {
KvCacheEvent Store(std::uint64_t hash, std::optional<std::uint64_t> parent, std::vector<std::int32_t> tokens) {
    return KvBlockStoredEvent{
        .block_hashes = {hash}, .parent_block_hash = parent, .token_ids = std::move(tokens), .block_size = 2};
}
KvCacheEvent Remove(std::uint64_t hash) {
    return KvBlockRemovedEvent{.block_hashes = {hash}};
}
std::vector<std::uint64_t> Hashes(const std::vector<KvCacheEvent>& events, bool stored) {
    std::vector<std::uint64_t> hashes;
    for (const auto& event : events) {
        EXPECT_EQ(std::holds_alternative<KvBlockStoredEvent>(event), stored);
        std::visit(
            [&](const auto& data) { hashes.insert(hashes.end(), data.block_hashes.begin(), data.block_hashes.end()); },
            event);
    }
    return hashes;
}
}  // namespace

TEST(KvEventPrefixView, MiddleEvictionWithdrawsAndRestoresResidentDescendants) {
    KvEventPrefixView view;
    EXPECT_EQ(Hashes(view.Apply({Store(1, {}, {1, 2}), Store(2, 1, {3, 4}), Store(3, 2, {5, 6})}), true),
              (std::vector<std::uint64_t>{1, 2, 3}));
    EXPECT_EQ(Hashes(view.Apply({Remove(2)}), false), (std::vector<std::uint64_t>{3, 2}));
    const auto restored = view.Apply({Store(2, 1, {3, 4})});
    EXPECT_EQ(Hashes(restored, true), (std::vector<std::uint64_t>{2, 3}));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(restored.back()).token_ids, (std::vector<std::int32_t>{5, 6}));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(restored.back()).parent_block_hash, 2u);
    EXPECT_EQ(Hashes(view.Apply({Store(4, 3, {7, 8})}), true), (std::vector<std::uint64_t>{4}));
}

TEST(KvEventPrefixView, MissingAncestorHidesNewBranchesAndShortTailsUntilRestore) {
    KvEventPrefixView view;
    view.Apply({Store(1, {}, {1, 2}), Store(2, 1, {3, 4}), Store(3, 2, {5})});
    EXPECT_EQ(Hashes(view.Apply({Remove(1)}), false), (std::vector<std::uint64_t>{3, 2, 1}));
    EXPECT_TRUE(view.Apply({Store(4, 2, {6, 7}), Remove(3)}).empty());
    EXPECT_EQ(Hashes(view.Apply({Store(1, {}, {1, 2})}), true), (std::vector<std::uint64_t>{1, 2, 4}));
    const auto tail = view.Apply({Store(3, 2, {5})});
    EXPECT_EQ(std::get<KvBlockStoredEvent>(tail.front()).token_ids, (std::vector<std::int32_t>{5}));
}

TEST(KvEventPrefixView, CoalescesNetChangesAndWaitsForEveryMissingAncestor) {
    KvEventPrefixView view;
    view.Apply({Store(1, {}, {1, 2}), Store(2, 1, {3, 4}), Store(3, 2, {5, 6})});
    EXPECT_TRUE(view.Apply({Remove(2), Store(2, 1, {3, 4})}).empty());
    EXPECT_EQ(Hashes(view.Apply({Remove(1), Remove(2)}), false), (std::vector<std::uint64_t>{3, 2, 1}));
    EXPECT_TRUE(view.Apply({Store(2, 1, {3, 4})}).empty());
    EXPECT_EQ(Hashes(view.Apply({Store(1, {}, {1, 2})}), true), (std::vector<std::uint64_t>{1, 2, 3}));
}

TEST(KvEventPrefixView, AcceptsChildrenBeforeParentsAndForgetsFullyRemovedChains) {
    KvEventPrefixView view;
    EXPECT_TRUE(view.Apply({Store(3, 2, {5, 6})}).empty());
    EXPECT_TRUE(view.Apply({Store(2, 1, {3, 4})}).empty());
    EXPECT_EQ(Hashes(view.Apply({Store(1, {}, {1, 2})}), true), (std::vector<std::uint64_t>{1, 2, 3}));
    EXPECT_EQ(Hashes(view.Apply({Remove(1), Remove(2), Remove(3)}), false), (std::vector<std::uint64_t>{3, 2, 1}));
    EXPECT_TRUE(view.Apply({Remove(3)}).empty());
    EXPECT_EQ(Hashes(view.Apply({Store(1, {}, {1, 2})}), true), (std::vector<std::uint64_t>{1}));
}
TEST(KvEventPrefixView, RandomBatchesMatchResidentAncestorClosure) {
    KvEventPrefixView view;
    std::array<bool, 64> resident{};
    std::set<std::uint64_t> visible;
    std::mt19937 random(42);
    for (int step = 0; step < 500; ++step) {
        std::vector<KvCacheEvent> batch;
        for (int change = 0; change < 8; ++change) {
            const std::uint64_t hash = 1 + random() % 63;
            resident[hash] = random() % 2;
            if (resident[hash]) {
                batch.push_back(Store(hash, hash == 1 ? std::nullopt : std::optional<std::uint64_t>(hash / 2),
                                      {static_cast<std::int32_t>(hash), 0}));
            } else {
                batch.push_back(Remove(hash));
            }
        }
        for (const auto& event : view.Apply(std::move(batch))) {
            if (const auto* stored = std::get_if<KvBlockStoredEvent>(&event)) {
                if (stored->parent_block_hash) EXPECT_TRUE(visible.contains(*stored->parent_block_hash));
                EXPECT_TRUE(visible.insert(stored->block_hashes.front()).second);
            } else {
                const auto hash = std::get<KvBlockRemovedEvent>(event).block_hashes.front();
                EXPECT_FALSE(visible.contains(hash * 2));
                EXPECT_FALSE(visible.contains(hash * 2 + 1));
                EXPECT_EQ(visible.erase(hash), 1u);
            }
        }
        std::set<std::uint64_t> expected;
        for (std::uint64_t hash = 1; hash < resident.size(); ++hash) {
            if (resident[hash] && (hash == 1 || expected.contains(hash / 2))) expected.insert(hash);
        }
        EXPECT_EQ(visible, expected) << "batch " << step;
    }
}
TEST(KvEventBatch, PreservesMultiHashPayloadsParentZeroAndEmptyBatches) {
    const std::vector<KvCacheEvent> original{
        KvBlockStoredEvent{.block_hashes = {0, std::numeric_limits<std::uint64_t>::max()},
                           .parent_block_hash = 0,
                           .token_ids = {0, 2147483647},
                           .block_size = 2},
        KvBlockRemovedEvent{.block_hashes = {17, 29}}};
    const auto batch = KvEventBatch::FromEvents(original);
    const auto restored = batch.ToEvents();
    ASSERT_EQ(restored.size(), 2u);
    const auto& stored = std::get<KvBlockStoredEvent>(restored[0]);
    EXPECT_EQ(stored.block_hashes, (std::vector<std::uint64_t>{0, std::numeric_limits<std::uint64_t>::max()}));
    EXPECT_EQ(stored.parent_block_hash, 0u);
    EXPECT_EQ(stored.token_ids, (std::vector<std::int32_t>{0, 2147483647}));
    EXPECT_EQ(std::get<KvBlockRemovedEvent>(restored[1]).block_hashes, (std::vector<std::uint64_t>{17, 29}));
    EXPECT_TRUE(KvEventBatch{}.ToEvents().empty());
    KvEventBatch empty;
    EXPECT_THROW(empty.Reserve(std::numeric_limits<std::size_t>::max(), 0, 0), std::length_error);
    EXPECT_TRUE(empty.Empty());
}

TEST(KvEventPrefixView, SnapshotsSurviveFurtherMutationsAndTheRoutingView) {
    KvEventBatch snapshot;
    {
        KvEventPrefixView view;
        const auto parent = std::make_shared<const KvEventDescriptor>(KvEventDescriptor{1, {}, {1, 2}, 2});
        const auto child = std::make_shared<const KvEventDescriptor>(KvEventDescriptor{2, 1, {3}, 2});
        const std::array initial{KvEventMutation{2, child}, KvEventMutation{1, parent}};
        snapshot = view.ApplyMutations(initial);
        const std::array removed{KvEventMutation{1, {}}, KvEventMutation{2, {}}};
        EXPECT_EQ(Hashes(view.ApplyMutations(removed).ToEvents(), false), (std::vector<std::uint64_t>{2, 1}));
    }
    const auto events = snapshot.ToEvents();
    EXPECT_EQ(Hashes(events, true), (std::vector<std::uint64_t>{1, 2}));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[0]).token_ids, (std::vector<std::int32_t>{1, 2}));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(events[1]).token_ids, (std::vector<std::int32_t>{3}));
}

class KvEventSnapshotTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.disable_l2_cache = true;
        config.enable_l3_storage = false;
        config.enable_kv_cache_events = true;
        return config;
    }
};

TEST_F(KvEventSnapshotTestSuite, ADrainedBatchOwnsNoCacheOrRequestResources) {
    Submit(RequestSpec{.request_id = "r", .tokens = {1, 2, 3, 4}});
    PlanOnce();
    SendForwardDone("r", {5});
    SendFinish("r");
    PlanOnce();
    const auto snapshot = scheduler_->DrainKvEventBatch();
    ASSERT_EQ(snapshot.Size(), 2u);
    EXPECT_TRUE(scheduler_->ClearCache());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    const auto removed = scheduler_->DrainKvEventBatch().ToEvents();
    ASSERT_EQ(removed.size(), 2u);

    Submit(RequestSpec{.request_id = "r", .tokens = {11, 12, 13, 14}});
    PlanOnce();
    SendForwardDone("r", {15});
    SendFinish("r");
    PlanOnce();
    scheduler_->DrainKvEventBatch();
    EXPECT_TRUE(scheduler_->ClearCache());
    const auto before = snapshot.ToEvents();
    EXPECT_EQ(std::get<KvBlockStoredEvent>(before[0]).token_ids, (std::vector<std::int32_t>{1, 2}));
    EXPECT_EQ(std::get<KvBlockStoredEvent>(before[1]).token_ids, (std::vector<std::int32_t>{3, 4}));
}

}  // namespace tokenspeed::test
