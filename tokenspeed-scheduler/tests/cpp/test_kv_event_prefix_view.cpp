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
}  // namespace tokenspeed::test
