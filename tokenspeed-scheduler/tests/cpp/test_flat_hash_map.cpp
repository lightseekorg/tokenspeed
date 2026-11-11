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
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "cache/coordinator/flat_hash_map.h"
#include "cache/core/planning_scratch.h"

namespace tokenspeed::test {
namespace {

TEST(FlatHashSet, StridedLocationsStayDistinctAcrossGrowthClearAndReusedScratch) {
    PlanningScratch storage;
    for (const std::int32_t stride : {1, 4096, 104729}) {
        const auto location = [&](std::int32_t i, std::int32_t slot) {
            return CacheBlockLocation{.lcm_block_id = 1 + i * stride, .slot_index = slot};
        };
        for (int round = 0; round < 3; ++round) {
            auto lease = storage.Acquire();
            FlatHashSet<CacheBlockLocation> locations{&lease};
            // A reused bank still holds the last round's slots.
            EXPECT_FALSE(locations.contains(location(round, 0)));
            if (round == 1) {
                locations.reserve(4096);
            }
            for (std::int32_t i = 0; i < 16384; ++i) {
                if (i % 3 == round) {
                    continue;
                }
                EXPECT_TRUE(locations.insert(location(i, 0)));
                EXPECT_TRUE(locations.insert(location(i, 3)));
                EXPECT_FALSE(locations.insert(location(i, 0)));
            }
            EXPECT_EQ(locations.size(), 2u * (16384 - (16384 - round + 2) / 3));
            for (std::int32_t i = 0; i < 16384; ++i) {
                EXPECT_EQ(locations.contains(location(i, 0)), i % 3 != round);
                EXPECT_EQ(locations.contains(location(i, 3)), i % 3 != round);
                EXPECT_FALSE(locations.contains(location(i, 1)));
            }
            locations.clear();
            EXPECT_TRUE(locations.empty());
            EXPECT_FALSE(locations.contains(location(round == 0 ? 1 : 0, 0)));
            EXPECT_TRUE(locations.insert(location(round, 1)));
            EXPECT_TRUE(locations.contains(location(round, 1)));
            EXPECT_EQ(locations.size(), 1u);
        }
    }
}

TEST(FlatHashMap, MatchesAReferenceMapThroughCollisionsErasesAndWraparound) {
    PlanningScratch storage;
    std::mt19937 random{20261112};
    for (const std::int32_t universe : {24, 600}) {
        auto lease = storage.Acquire();
        FlatHashMap<CacheBlockLocation, std::int64_t> values{&lease};
        std::unordered_map<std::int64_t, std::int64_t> expected;
        std::uniform_int_distribution<std::int32_t> pick(0, universe - 1);
        std::uniform_int_distribution<int> operation(0, 9);
        const auto location = [](std::int32_t key) {
            return CacheBlockLocation{.lcm_block_id = 1 + (key / 2) * 4096, .slot_index = key % 2};
        };
        for (int step = 0; step < 200000; ++step) {
            const std::int32_t key = pick(random);
            const int op = operation(random);
            if (op < 4) {
                const auto [value, inserted] = values.TryEmplace(location(key), step);
                const auto [reference, reference_inserted] = expected.try_emplace(key, step);
                EXPECT_EQ(inserted, reference_inserted);
                EXPECT_EQ(*value, reference->second);
            } else if (op < 6) {
                ++values[location(key)];
                ++expected[key];
            } else if (op < 9) {
                EXPECT_EQ(values.Erase(location(key)), expected.erase(key) == 1);
            } else if (step % 1000 == 9) {
                values.clear();
                expected.clear();
            }
            ASSERT_EQ(values.size(), expected.size());
            if (step % 997 == 0) {
                for (std::int32_t k = 0; k < universe; ++k) {
                    const std::int64_t* value = values.Find(location(k));
                    const auto reference = expected.find(k);
                    ASSERT_EQ(value != nullptr, reference != expected.end());
                    if (value != nullptr) {
                        EXPECT_EQ(*value, reference->second);
                    }
                }
            }
        }
    }
}

TEST(FlatHashMap, PooledNodeAddressesKeepTheirCountsAcrossEraseAndRegrowth) {
    struct Node {
        std::array<std::byte, 256> bytes;
    };
    const std::vector<Node> nodes(16384);
    PlanningScratch storage;
    for (const std::size_t stride : {1u, 16u}) {
        auto lease = storage.Acquire();
        FlatHashMap<const Node*, std::int32_t> counts{&lease};
        for (std::size_t i = 0; i < nodes.size(); i += stride) {
            for (std::size_t n = 0; n <= i % 3; ++n) {
                ++counts[&nodes[i]];
            }
        }
        for (std::size_t i = 0; i < nodes.size(); i += 2 * stride) {
            EXPECT_TRUE(counts.Erase(&nodes[i]));
            EXPECT_FALSE(counts.Erase(&nodes[i]));
        }
        for (std::size_t i = 0; i < nodes.size(); i += stride) {
            const std::int32_t* count = counts.Find(&nodes[i]);
            if (i % (2 * stride) == 0) {
                EXPECT_EQ(count, nullptr);
                EXPECT_TRUE(counts.TryEmplace(&nodes[i], -1).second);
            } else {
                ASSERT_NE(count, nullptr);
                EXPECT_EQ(*count, static_cast<std::int32_t>(i % 3) + 1);
            }
        }
        for (std::size_t i = 0; i < nodes.size(); i += stride) {
            EXPECT_EQ(*counts.Find(&nodes[i]), i % (2 * stride) == 0 ? -1 : static_cast<std::int32_t>(i % 3) + 1);
        }
        EXPECT_EQ(counts.size(), (nodes.size() + stride - 1) / stride);
        EXPECT_EQ(counts.contains(&nodes[1]), stride == 1);
    }
}

TEST(FlatHashSet, TablesPastTheBankAndAFourthLeaseKeepTheirOwnKeys) {
    PlanningScratch storage;
    for (int round = 0; round < 2; ++round) {
        auto first = storage.Acquire();
        auto second = storage.Acquire();
        auto third = storage.Acquire();
        // No bank is left for this one.
        auto fourth = storage.Acquire();
        const std::array<PlanningScratch::Lease*, 4> leases{&first, &second, &third, &fourth};
        std::vector<FlatHashSet<CacheBlockLocation>> sets;
        sets.reserve(leases.size());
        for (std::int32_t s = 0; s < 4; ++s) {
            sets.emplace_back(leases[s]);
            // The first table outgrows its 256 KiB bank.
            for (std::int32_t i = 0; i < (s == 0 ? 50000 : 1000); ++i) {
                sets.back().insert(CacheBlockLocation{.lcm_block_id = 1 + i, .slot_index = s});
            }
        }
        for (std::int32_t s = 0; s < 4; ++s) {
            EXPECT_EQ(sets[s].size(), s == 0 ? 50000u : 1000u);
            for (std::int32_t i = 0; i < 1000; ++i) {
                for (std::int32_t other = 0; other < 4; ++other) {
                    EXPECT_EQ(sets[s].contains(CacheBlockLocation{.lcm_block_id = 1 + i, .slot_index = other}),
                              other == s);
                }
            }
        }
    }
}

// Backward-shift erase on a cluster that wraps past the end of the table.
// The keys are picked by the table's mix (the splitmix64 finalizer) so their
// homes are the last two slots of a fresh 16-slot table; what is checked holds
// for any hash.
TEST(FlatHashMap, EraseInTheLastSlotShiftsAClusterWrappedToTheFront) {
    const auto home = [](CacheBlockLocation location) {
        std::uint64_t bits = static_cast<std::uint32_t>(location.lcm_block_id) |
                             static_cast<std::uint64_t>(static_cast<std::uint32_t>(location.slot_index)) << 32U;
        bits ^= bits >> 30U;
        bits *= 0xbf58476d1ce4e5b9ULL;
        bits ^= bits >> 27U;
        bits *= 0x94d049bb133111ebULL;
        bits ^= bits >> 31U;
        return bits & 15U;
    };
    std::vector<CacheBlockLocation> home_14;
    std::vector<CacheBlockLocation> home_15;
    for (std::int32_t id = 1; home_14.size() < 2 || home_15.size() < 2; ++id) {
        const CacheBlockLocation location{.lcm_block_id = id, .slot_index = 0};
        if (home(location) == 14 && home_14.size() < 2) home_14.push_back(location);
        if (home(location) == 15 && home_15.size() < 2) home_15.push_back(location);
    }
    PlanningScratch storage;
    auto lease = storage.Acquire();
    FlatHashMap<CacheBlockLocation, std::int32_t> values{&lease};
    // Slots 14 and 15, then 0 and 1 past the end of the table.
    const std::array<CacheBlockLocation, 4> keys{home_14[0], home_15[0], home_14[1], home_15[1]};
    for (std::int32_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(values.TryEmplace(keys[i], i).second);
    }
    // The key in the last slot leaves; the two that wrapped move back.
    EXPECT_TRUE(values.Erase(keys[1]));
    EXPECT_EQ(values.size(), 3u);
    EXPECT_EQ(values.Find(keys[1]), nullptr);
    for (const std::int32_t i : {0, 2, 3}) {
        const std::int32_t* value = values.Find(keys[i]);
        ASSERT_NE(value, nullptr) << i;
        EXPECT_EQ(*value, i);
        const auto [existing, inserted] = values.TryEmplace(keys[i], -1);
        EXPECT_FALSE(inserted) << i;
        EXPECT_EQ(*existing, i);
    }
    const auto [value, inserted] = values.TryEmplace(keys[1], 7);
    EXPECT_TRUE(inserted);
    EXPECT_EQ(*value, 7);
    EXPECT_EQ(values.size(), 4u);
}

}  // namespace
}  // namespace tokenspeed::test
