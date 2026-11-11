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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <list>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "cache/cache_group.h"
#include "cache/coordinator/cache_coordinator.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/parent_occupancy.h"
#include "cache/coordinator/shadow_occupancy.h"

namespace tokenspeed::test {
namespace {

TEST(ParentOccupancy, SparseParentsKeepZeroAndRestoredCountsAcrossGrowthAndUnwind) {
    for (const std::int32_t stride : {4096, 104729}) {
        PlanningScratch storage;
        for (int round = 0; round < 3; ++round) {
            try {
                auto lease = storage.Acquire();
                ParentOccupancy counts{&lease};
                EXPECT_EQ(counts.Find(1), nullptr);
                for (int i = 1; i <= 16384; ++i) {
                    const auto parent = static_cast<std::int32_t>(static_cast<std::int64_t>(i) * stride);
                    counts.GetOrInsert(parent, i % 4 + 1) -= 1;
                    EXPECT_EQ(counts.GetOrInsert(parent, 99), i % 4);
                }
                EXPECT_EQ(counts.size(), 16384u);
                for (int i = 16384; i > 0; --i) {
                    const auto parent = static_cast<std::int32_t>(static_cast<std::int64_t>(i) * stride);
                    auto* occupied = counts.Find(parent);
                    ASSERT_NE(occupied, nullptr);
                    EXPECT_EQ(*occupied, i % 4);
                    ++*occupied;
                    EXPECT_EQ(counts.GetOrInsert(parent, 99), i % 4 + 1);
                }
                EXPECT_EQ(counts.Find(1), nullptr);
                throw 1;
            } catch (int) {
            }
        }
    }
}

TEST(CacheEvictionCandidates, PrescanReplaysPolicyOrderAcrossPinnedEpochsAndSnapshotLimit) {
    for (const std::size_t threshold : {16u, 9000u, 12000u}) {
        BlockPool pool(10032, {1, 1});
        std::vector<CacheGroup> groups;
        std::vector<CacheBlockRef> pinned;
        for (std::uint32_t group = 0; group < 2; ++group) {
            const CacheGroupSpec spec{.cache_blocks_per_lcm_block = 1, .block_granularity = 4};
            groups.emplace_back(spec, std::make_unique<GroupAllocator>(1, group, 1), MakePrefixMatcher(spec));
            for (int i = 0; i < (group == 0 ? 10000 : 32); ++i) {
                auto block = pool.AcquireBlock(group);
                const CacheKey key{.group_id = group, .content_hash = std::to_string(i)};
                groups.back().Index().Register(pool, block, key, static_cast<std::uint64_t>(i / 16 + 1), i,
                                               CacheBoundaryKind::kChunk, nullptr, nullptr);
                if (i >= 16 && i < 32) pinned.push_back(block);
            }
        }
        CacheEvictionCandidates replay{groups, pool, nullptr, nullptr, CheckpointAging{}, nullptr};
        std::size_t seen = 0;
        const bool fits = replay.PreScanUntil([&](std::uint32_t, auto entries) {
            seen += entries.size();
            return seen >= threshold;
        });
        EXPECT_EQ(fits, threshold <= 10000);
        CacheEvictionCandidates direct{groups, pool, nullptr, nullptr, CheckpointAging{}, nullptr};
        using Key = std::tuple<std::uint32_t, std::int32_t, std::int32_t, std::uint64_t, std::int64_t>;
        const auto drain = [](CacheEvictionCandidates& candidates) {
            std::vector<Key> result;
            while (const auto candidate = candidates.Next(std::numeric_limits<std::uint64_t>::max())) {
                result.emplace_back(candidate->group_id, candidate->location.lcm_block_id,
                                    candidate->location.slot_index, candidate->rank_epoch, candidate->position_rank);
            }
            return result;
        };
        const auto expected = drain(direct);
        EXPECT_EQ(expected.size(), 10000u);
        EXPECT_EQ(drain(replay), expected);
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
    }
}

// Whatever physical pages an epoch's entries sit on, they leave in the policy
// order: equal-age prefix pages from the suffix, unproven chunk boundaries
// before established ones, then by group and physical location.
TEST(CacheEvictionCandidates, EpochsLeaveInPolicyOrderOnAnyPhysicalLayout) {
    constexpr int shuffled_pages = 1100;
    constexpr int ascending_pages = 600;
    BlockPool pool(shuffled_pages + ascending_pages, {1, 1});
    std::vector<CacheGroup> groups;
    const CacheGroupSpec full{.cache_blocks_per_lcm_block = 1, .block_granularity = 4};
    const CacheGroupSpec window{.attn_kind = AttnKind::kSlidingWindow,
                                .sliding_window = 8,
                                .cache_blocks_per_lcm_block = 1,
                                .block_granularity = 4};
    groups.emplace_back(full, std::make_unique<GroupAllocator>(1, 0, 1), MakePrefixMatcher(full));
    groups.emplace_back(window, std::make_unique<GroupAllocator>(1, 1, 1), MakePrefixMatcher(window));
    {
        // The ascending parents are handed out first, then these in a random order.
        auto blocks = pool.AcquireBlocks(0, shuffled_pages);
        std::mt19937 random(8731);
        std::shuffle(blocks.begin(), blocks.end(), random);
        for (CacheBlockRef& block : blocks) block.reset();
    }

    struct Entry {
        std::uint32_t group;
        std::int32_t parent;
        std::int32_t position;
        CacheBoundaryKind kind;
    };
    std::vector<std::vector<Entry>> epochs;
    int next_key = 0;
    const auto add = [&](std::uint32_t group, std::int32_t position, CacheBoundaryKind kind) {
        CacheBlockRef block = pool.AcquireBlock(group);
        groups[group].Index().Register(pool, block,
                                       CacheKey{.group_id = group, .content_hash = std::to_string(next_key++)},
                                       epochs.size(), position, kind, nullptr, nullptr);
        epochs.back().push_back(Entry{group, block->Location().lcm_block_id, position, kind});
    };
    const auto start_epoch = [&] { epochs.emplace_back(); };
    // Ascending pages, the prefix written forward and then backward.
    start_epoch();
    for (int i = 0; i < 300; ++i) add(0, i, CacheBoundaryKind::kChunk);
    start_epoch();
    for (int i = 299; i >= 0; --i) add(0, i, CacheBoundaryKind::kChunk);
    // Shuffled pages: one prefix, two prefixes sharing positions, a short one.
    start_epoch();
    for (int i = 0; i < 300; ++i) add(0, i, CacheBoundaryKind::kChunk);
    start_epoch();
    for (int i = 0; i < 300; ++i) add(0, i % 150, CacheBoundaryKind::kChunk);
    start_epoch();
    for (int i = 0; i < 20; ++i) add(0, i, CacheBoundaryKind::kChunk);
    // Both groups in one epoch, the window group with both boundary tiers.
    start_epoch();
    for (int i = 0; i < 300; ++i) {
        add(1, i, i % 2 == 0 ? CacheBoundaryKind::kChunk : CacheBoundaryKind::kEndpoint);
        if (i < 100) add(0, i, CacheBoundaryKind::kChunk);
    }

    std::vector<std::int32_t> expected;
    for (std::vector<Entry>& epoch : epochs) {
        // Probationary window chunks by position, established window
        // boundaries by page, then full-history pages from the suffix.
        const auto rank = [](const Entry& entry) {
            if (entry.group == 0) return std::tuple{2, -entry.position};
            return entry.kind == CacheBoundaryKind::kChunk ? std::tuple{0, entry.position} : std::tuple{1, 0};
        };
        std::ranges::sort(epoch, [&](const Entry& lhs, const Entry& rhs) {
            return std::tuple{rank(lhs), lhs.parent} < std::tuple{rank(rhs), rhs.parent};
        });
        for (const Entry& entry : epoch) expected.push_back(entry.parent);
    }
    EXPECT_EQ(expected.size(), static_cast<std::size_t>(shuffled_pages + ascending_pages - 80));

    // Read from the index directly, and replayed from a pre-scan's snapshots.
    for (const bool prescan : {false, true}) {
        CacheEvictionCandidates candidates{groups, pool, nullptr, nullptr, CheckpointAging{}, nullptr};
        if (prescan) {
            EXPECT_FALSE(candidates.PreScanUntil([](std::uint32_t, auto) { return false; }));
        }
        std::vector<std::int32_t> order;
        while (const auto candidate = candidates.Next(std::numeric_limits<std::uint64_t>::max())) {
            order.push_back(candidate->location.lcm_block_id);
        }
        EXPECT_EQ(order, expected);
    }
}

// Sort reaches the evictionKey order from any input: an epoch already in that
// order or reversed, one in page order with scattered ranks, and inputs one
// loaded epoch never forms (mixed epochs, groups, aged ranks, repeated pages).
TEST(CacheEvictionCandidates, SortReachesTheKeyOrderFromAnyInput) {
    using Candidate = CacheEvictionCandidates::Candidate;
    using Tier = CacheEvictionCandidates::BoundaryTier;
    std::mt19937 random(97);
    const auto epoch = [](std::size_t size, auto&& tier_and_rank) {
        std::vector<Candidate> candidates;
        for (std::size_t i = 0; i < size; ++i) {
            const auto [tier, rank] = tier_and_rank(i);
            candidates.push_back(Candidate{.last_access_epoch = 9,
                                           .position_rank = rank,
                                           .location = {.lcm_block_id = static_cast<std::int32_t>(i / 2 + 1),
                                                        .slot_index = static_cast<std::int32_t>(i % 2)},
                                           .group_id = 1,
                                           .boundary_tier = tier,
                                           .rank_epoch = 9});
        }
        return candidates;
    };
    std::vector<std::vector<Candidate>> inputs;
    for (const std::size_t size : {std::size_t{2}, std::size_t{40}, std::size_t{300}, std::size_t{5000}}) {
        const auto n = static_cast<std::int64_t>(size);
        std::vector<std::int64_t> shuffled(size);
        std::iota(shuffled.begin(), shuffled.end(), 0);
        std::shuffle(shuffled.begin(), shuffled.end(), random);
        std::uniform_int_distribution<std::int32_t> any_rank(-std::numeric_limits<std::int32_t>::max(),
                                                             std::numeric_limits<std::int32_t>::max());
        std::uniform_int_distribution<int> any_tier(0, 3);
        std::vector<std::pair<Tier, std::int64_t>> wide(size);
        for (auto& [tier, rank] : wide) {
            tier = static_cast<Tier>(any_tier(random));
            rank = any_rank(random);
        }
        const auto closed = [](std::int64_t rank) { return std::pair{Tier::kClosedPrefix, rank}; };
        inputs.push_back(epoch(size, [&](std::size_t i) { return closed(static_cast<std::int64_t>(i) - n); }));
        inputs.push_back(epoch(size, [&](std::size_t i) { return closed(-static_cast<std::int64_t>(i)); }));
        inputs.push_back(epoch(size, [&](std::size_t i) { return closed(-shuffled[i]); }));
        inputs.push_back(epoch(size, [&](std::size_t i) { return closed(-(shuffled[i] % 7)); }));
        inputs.push_back(epoch(size, [&](std::size_t i) { return wide[i]; }));
        inputs.push_back(epoch(size, [&](std::size_t i) {
            return std::pair{shuffled[i] % 3 == 0 ? Tier::kEstablished : Tier::kProbationary, shuffled[i] % 50};
        }));
        // Not one loaded epoch: each breaks a different shared field or the page order.
        const std::vector<Candidate> base = inputs.back();
        inputs.push_back(base);
        inputs.back()[size / 2].last_access_epoch = inputs.back()[size / 2].rank_epoch = 8;
        inputs.push_back(base);
        inputs.back().back().group_id = 0;
        inputs.push_back(base);
        inputs.back().front().aged_distance = 2;
        inputs.back().front().rank_epoch = 10;
        inputs.push_back(base);
        inputs.back().back().position_rank = std::int64_t{1} << 40;
        inputs.push_back(base);
        inputs.back().insert(inputs.back().end(), base.begin(), base.begin() + static_cast<std::ptrdiff_t>(size / 2));
    }

    const auto same = [](const Candidate& lhs, const Candidate& rhs) {
        const auto fields = [](const Candidate& c) {
            return std::tuple{c.last_access_epoch,   c.position_rank, c.location.lcm_block_id,
                              c.location.slot_index, c.group_id,      c.boundary_tier,
                              c.rank_epoch,          c.aged_distance};
        };
        return fields(lhs) == fields(rhs);
    };
    std::vector<Candidate> scratch(3, Candidate{.last_access_epoch = 1, .group_id = 7});
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<Candidate>& input = inputs[i];
        SCOPED_TRACE(testing::Message() << "input " << i << ", size " << input.size());
        std::vector<Candidate> expected = input;
        std::stable_sort(expected.begin(), expected.end(), CacheEvictionCandidates::Before);
        std::vector<Candidate> sorted = input;
        CacheEvictionCandidates::Sort(sorted);
        EXPECT_TRUE(std::ranges::equal(sorted, expected, same));
        sorted = input;
        CacheEvictionCandidates::Sort(sorted, scratch);
        EXPECT_TRUE(std::ranges::equal(sorted, expected, same));
    }
}

TEST(FixedSlotResource, AlignedNodesKeepTheirAddressAcrossGrowthAndReuse) {
    struct alignas(256) Value {
        int value;
    };
    FixedSlotResource storage{16};
    std::pmr::list<Value> values{&storage};
    values.push_back(Value{73});
    const auto* anchor = &values.front();
    for (int round = 0; round < 4; ++round) {
        for (int i = 0; i < 4096; ++i) values.push_back(Value{i});
        EXPECT_EQ(&values.front(), anchor);
        EXPECT_EQ(anchor->value, 73);
        auto it = std::next(values.begin());
        for (int i = 0; i < 4096; ++i, ++it) {
            EXPECT_EQ(it->value, i);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&*it) % alignof(Value), 0);
        }
        values.erase(std::next(values.begin()), values.end());
    }
}

TEST(ShadowOccupancy, ConcurrentPlansAndReusedScratchHaveIndependentCapacity) {
    const CacheGroupSpec history{.cache_blocks_per_lcm_block = 2, .block_granularity = 4};
    std::vector<CacheGroup> groups;
    groups.emplace_back(history, std::make_unique<GroupAllocator>(2, 0, 1), MakePrefixMatcher(history));
    BlockPool pool{2, {2}};
    pool.RegisterGroup(0, 2, 1);
    const auto held = pool.AcquireBlocks(0, 4);
    const std::vector<std::vector<std::int64_t>> needed{{2}};
    const std::vector<std::int64_t> zero{0};
    for (int round = 0; round < 4; ++round) {
        ShadowOccupancy first{pool, groups};
        ShadowOccupancy second{pool, groups};
        ShadowOccupancy third{pool, groups};
        EXPECT_FALSE(first.Fits(needed, zero, zero));
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        EXPECT_FALSE(third.Fits(needed, zero, zero));
        first.Remove(0, held[0]->Location());
        first.Remove(0, held[1]->Location());
        EXPECT_TRUE(first.Fits(needed, zero, zero));
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        second.Remove(0, held[2]->Location());
        EXPECT_FALSE(second.Fits(needed, zero, zero));
        second.Remove(0, held[3]->Location());
        EXPECT_TRUE(second.Fits(needed, zero, zero));
        first.Restore(0, held[0]->Location());
        EXPECT_FALSE(first.Fits(needed, zero, zero));
        EXPECT_TRUE(second.Fits(needed, zero, zero));
        EXPECT_FALSE(third.Fits(needed, zero, zero));
        EXPECT_EQ(pool.NumOccupiedSlots(), 4);
    }
}

// Two groups on one pool, some parents part full and some empty: a 4-slot
// group in two buckets and a 2-slot group.
struct MixedPackingPool {
    MixedPackingPool() : pool{12, {4, 2}} {
        const CacheGroupSpec sharded{.cache_blocks_per_lcm_block = 4, .block_granularity = 4};
        const CacheGroupSpec single{.cache_blocks_per_lcm_block = 2, .block_granularity = 4};
        groups.emplace_back(sharded, std::make_unique<GroupAllocator>(4, 0, 2), MakePrefixMatcher(sharded));
        groups.emplace_back(single, std::make_unique<GroupAllocator>(2, 1, 1), MakePrefixMatcher(single));
        pool.RegisterGroup(0, 4, 2);
        pool.RegisterGroup(1, 2, 1);
        const std::vector<std::int32_t> buckets{0, 0, 0, 1, 1, 0, 1, 0, 0};
        for (CacheBlockRef& block : pool.AcquireBlocksInBuckets(0, buckets)) held.emplace_back(0, std::move(block));
        for (CacheBlockRef& block : pool.AcquireBlocks(1, 5)) held.emplace_back(1, std::move(block));
    }

    BlockPool pool;
    std::vector<CacheGroup> groups;
    std::vector<std::pair<std::uint32_t, CacheBlockRef>> held;
};

// Per group: blocks by bucket, dense blocks, pairs.
struct MixedDemand {
    std::vector<std::vector<std::int64_t>> need;
    std::vector<std::int64_t> dense;
    std::vector<std::int64_t> pairs;
};

// MixedDemands' entries whose fit is known without the formula.
constexpr std::size_t kMoreThanThePool = 4;
constexpr std::size_t kFitsAsThePoolIs = 5;

std::vector<MixedDemand> MixedDemands() {
    return {
        {{{3, 1}, {1}}, {14, 6}, {0, 0}},  // bucket and dense blocks
        {{{0, 0}, {0}}, {20, 9}, {0, 0}},  // dense blocks only
        {{{6, 2}, {0}}, {9, 0}, {3, 2}},   // pairs
        {{{6, 6}, {3}}, {4, 5}, {1, 1}},   // all three
        {{{0, 0}, {0}}, {60, 0}, {0, 0}},  // more than the whole pool
        {{{1, 0}, {0}}, {0, 0}, {0, 0}},   // fits as the pool is
    };
}

// A Remove lowers the shortfall by at most one parent and a Restore never
// lowers it.
TEST(ShadowOccupancy, ARemoveClosesAtMostOneParentOfTheShortfall) {
    MixedPackingPool setup;
    std::size_t removes_closing_one_parent = 0;
    const std::vector<MixedDemand> demands = MixedDemands();
    for (std::size_t d = 0; d < demands.size(); ++d) {
        const MixedDemand& demand = demands[d];
        for (std::uint32_t seed = 0; seed < 4; ++seed) {
            SCOPED_TRACE(testing::Message() << "demand " << d << ", seed " << seed);
            std::vector<std::size_t> order(setup.held.size());
            std::iota(order.begin(), order.end(), 0);
            std::mt19937 random(seed);
            std::ranges::shuffle(order, random);
            ShadowOccupancy shadow{setup.pool, setup.groups};
            const std::int64_t initial = shadow.ParentShortfall(demand.need, demand.dense, demand.pairs);
            if (d == kFitsAsThePoolIs) {
                // An empty parent takes the one bucket-0 block.
                EXPECT_EQ(initial, 0);
            }
            std::int64_t shortfall = initial;
            for (const std::size_t i : order) {
                shadow.Remove(setup.held[i].first, setup.held[i].second->Location());
                const std::int64_t next = shadow.ParentShortfall(demand.need, demand.dense, demand.pairs);
                ASSERT_GE(next, std::max<std::int64_t>(shortfall - 1, 0));
                removes_closing_one_parent += next == shortfall - 1 ? 1 : 0;
                shortfall = next;
            }
            if (d == kMoreThanThePool) {
                // 60 blocks of a 4-slot group need 15 parents; the pool has 12.
                EXPECT_EQ(shortfall, 3);
            }
            std::ranges::shuffle(order, random);
            for (const std::size_t i : order) {
                shadow.Restore(setup.held[i].first, setup.held[i].second->Location());
                const std::int64_t next = shadow.ParentShortfall(demand.need, demand.dense, demand.pairs);
                ASSERT_GE(next, shortfall);
                shortfall = next;
            }
            // Every block is back: the pool's own shortfall again.
            EXPECT_EQ(shortfall, initial);
        }
    }
    EXPECT_GT(removes_closing_one_parent, 0u);
    EXPECT_EQ(setup.pool.NumOccupiedSlots(), 14);
}

// Selection stops after the first selection the demand fits after -- one
// that removes no block, one or several -- and fails once they run out.
TEST(CacheEvictionPlan, SelectionStopsAtTheFirstSelectionAfterWhichTheDemandFits) {
    MixedPackingPool setup;
    const std::vector<std::vector<std::size_t>> scripts{
        {1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1},
        {1, 0, 3, 1, 2, 0, 1, 4, 1, 1},
    };
    std::set<std::optional<std::size_t>> stops;
    for (int permutation = 0; permutation < 4; ++permutation) {
        // Fixed removal orders: as held, reversed, rotated and interleaved.
        std::vector<std::size_t> order(setup.held.size());
        std::iota(order.begin(), order.end(), 0);
        if (permutation == 1) std::ranges::reverse(order);
        if (permutation == 2) std::ranges::rotate(order, order.begin() + static_cast<std::ptrdiff_t>(order.size() / 2));
        if (permutation == 3) std::ranges::stable_partition(order, [](std::size_t i) { return i % 2 == 0; });
        for (const std::vector<std::size_t>& removed_per_selection : scripts) {
            for (const MixedDemand& demand : MixedDemands()) {
                SCOPED_TRACE(permutation);
                // Reference: ask the fit after every selection.
                ShadowOccupancy reference{setup.pool, setup.groups};
                std::optional<std::size_t> fits_after;
                std::size_t next_block = 0;
                for (std::size_t selection = 0;; ++selection) {
                    if (reference.Fits(demand.need, demand.dense, demand.pairs)) {
                        fits_after = selection;
                        break;
                    }
                    if (selection == removed_per_selection.size()) break;
                    for (std::size_t k = 0; k < removed_per_selection[selection]; ++k, ++next_block) {
                        const auto& [group_id, block] = setup.held[order[next_block]];
                        reference.Remove(group_id, block->Location());
                    }
                }
                stops.insert(fits_after);
                CacheEvictionPlan plan{setup.pool, setup.groups};
                std::size_t selections = 0;
                next_block = 0;
                const bool fits = plan.SelectUntilFits(demand.need, demand.dense, demand.pairs, [&] {
                    if (selections == removed_per_selection.size()) return false;
                    for (std::size_t k = 0; k < removed_per_selection[selections]; ++k, ++next_block) {
                        const auto& [group_id, block] = setup.held[order[next_block]];
                        plan.Remove(group_id, block->Location());
                    }
                    ++selections;
                    return true;
                });
                EXPECT_EQ(fits, fits_after.has_value());
                EXPECT_EQ(selections, fits_after.value_or(removed_per_selection.size()));
                EXPECT_EQ(plan.Fits(demand.need, demand.dense, demand.pairs), fits);
            }
        }
    }
    EXPECT_GT(stops.size(), 4u) << "the demands stop at many different selections";
}

TEST(PlanningScratch, AlignedOverflowAndNestedLeasesKeepTheirOwnValues) {
    struct alignas(512) Value {
        int value;
    };
    PlanningScratch scratch;
    for (int round = 0; round < 3; ++round) {
        auto first = scratch.Acquire();
        auto second = scratch.Acquire();
        auto third = scratch.Acquire();
        std::pmr::vector<Value> large{&first};
        std::pmr::vector<Value> small{&second};
        std::pmr::vector<Value> nested{&third};
        for (int i = 0; i < 2048; ++i) large.push_back(Value{i});
        small.push_back(Value{42});
        nested.push_back(Value{99});
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(large.data()) % alignof(Value), 0);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small.data()) % alignof(Value), 0);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(nested.data()) % alignof(Value), 0);
        for (int i = 0; i < 2048; ++i) EXPECT_EQ(large[i].value, i);
        EXPECT_EQ(small[0].value, 42);
        EXPECT_EQ(nested[0].value, 99);
    }
}

TEST(ShadowOccupancy, AReleasedHostParentCannotAlsoLeaveHistorySlotsBehind) {
    const CacheGroupSpec history{.cache_blocks_per_lcm_block = 6, .block_granularity = 4};
    const CacheGroupSpec state{.attn_kind = AttnKind::kState, .cache_blocks_per_lcm_block = 1, .block_granularity = 0};
    std::vector<CacheGroup> groups;
    groups.emplace_back(history, std::make_unique<GroupAllocator>(6, 0, 1), MakePrefixMatcher(history));
    groups.emplace_back(state, std::make_unique<GroupAllocator>(1, 1, 1), MakePrefixMatcher(state));
    BlockPool host{1, {8, 1}};
    host.RegisterGroup(0, 8, 1);
    host.RegisterGroup(1, 1, 1);
    const CacheBlockRef held = host.AcquireBlock(0);
    ASSERT_TRUE(held);
    ShadowOccupancy shadow{host, groups};
    shadow.Remove(0, held->Location());

    const std::vector<std::int64_t> dense{0, 0};
    const std::vector<std::int64_t> pairs{0, 0};
    const std::vector<std::vector<std::int64_t>> mixed{{2}, {1}};
    const std::vector<std::vector<std::int64_t>> history_only{{8}, {0}};
    const std::vector<std::vector<std::int64_t>> state_only{{0}, {1}};
    EXPECT_FALSE(shadow.Fits(mixed, dense, pairs));
    EXPECT_TRUE(shadow.Fits(history_only, dense, pairs));
    EXPECT_TRUE(shadow.Fits(state_only, dense, pairs));
    EXPECT_EQ(host.NumEmptyLcmBlocks(), 0);  // The plan has not mutated storage.
}

TEST(HostPacking, ABlockIdUsesTheOwningPoolsPacking) {
    GroupAllocator allocator{4, 0, 2};
    BlockPool device{2, {4}};
    BlockPool host{2, {6}};
    device.RegisterGroup(0, 4, 2);
    host.RegisterGroup(0, 6, 2);
    const std::vector<std::int32_t> buckets{0, 1, 0, 1, 0, 1, 0, 1};
    const auto device_blocks = device.AcquireBlocksInBuckets(0, std::span{buckets}.first(6));
    const auto host_blocks = host.AcquireBlocksInBuckets(0, buckets);
    ASSERT_EQ(device_blocks.size(), 6);
    ASSERT_EQ(host_blocks.size(), 8);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*device_blocks.back()), 6);
    EXPECT_EQ(allocator.ResolveCacheBlockId(*host_blocks.back()), 8);
    EXPECT_EQ(host_blocks.back()->Location().lcm_block_id, 2);
    EXPECT_EQ(host_blocks.back()->Location().slot_index, 1);
    EXPECT_EQ((allocator.ResolveCacheBlockId(*host_blocks.back()) - 1) % 2, 1);

    // Both pools use parent 2, but their first child IDs differ. Holes and
    // a return to the Device pool must preserve every logical table slot.
    BlockTable table = BlockTable::FromBlocks(
        /*group_id=*/0, {device_blocks[4], device_blocks[5], {}, host_blocks[6], host_blocks[7], device_blocks[5]}, 0);
    EXPECT_EQ(allocator.BlockTablePageIds(table), (std::vector<std::int32_t>{5, 6, 0, 7, 8, 6}));
    allocator.ReclaimExpired(device, table, 2);
    EXPECT_EQ(allocator.BlockTablePageIds(table), (std::vector<std::int32_t>{0, 0, 0, 7, 8, 6}));
}

TEST(BlockTableAccounting, ReleasedPrefixesAndSharedParentsCountOnce) {
    BlockPool device{4, {4}};
    const std::vector specs{CacheGroupSpec{.attn_kind = AttnKind::kSlidingWindow,
                                           .sliding_window = 4,
                                           .cache_blocks_per_lcm_block = 4,
                                           .block_granularity = 4}};
    auto coordinator = MakeCoordinator(specs, 4, device, /*enable_l3_storage=*/false, /*host_pool=*/nullptr,
                                       /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                                       /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
    const auto blocks = device.AcquireBlocks(0, 8);
    ASSERT_EQ(blocks.size(), 8u);
    std::vector<BlockTable> first{BlockTable::FromBlocks(/*group_id=*/0, {blocks.begin(), blocks.begin() + 7}, 0)};
    std::vector<BlockTable> second{
        BlockTable::FromBlocks(/*group_id=*/0, {blocks[0], {}, {}, {}, {}, {}, {}, blocks[7]}, 0)};
    const std::vector<std::span<const BlockTable>> requests{first, second};
    coordinator.Allocator(0).ReclaimExpired(device, first[0], 6);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 2);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 7);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, first[0], 7);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 1);
    coordinator.Allocator(0).ReclaimExpired(device, second[0], 8);
    EXPECT_EQ(coordinator.NumActiveLcmBlocks(requests), 0);
    EXPECT_EQ(device.NumEmptyLcmBlocks(), 2) << "the retained copies still occupy the pool";
}

TEST(HostPacking, SnapshotAdmissionCountsTheSnapshotPoolsChildren) {
    for (const std::int32_t host_packing : {2, 6}) {
        BlockPool device{2, {4}};
        BlockPool snapshots{1, {host_packing}};
        const std::vector specs{CacheGroupSpec{.cache_blocks_per_lcm_block = 4, .block_granularity = 2}};
        auto coordinator = MakeCoordinator(specs, /*prefix_granularity=*/2, device, /*enable_l3_storage=*/false,
                                           /*host_pool=*/nullptr, &snapshots, /*stream_device_cache_to_host=*/false,
                                           /*write_back_evictions=*/false, /*checkpoint_retention_requests=*/0);
        const std::int32_t blocks = host_packing == 2 ? 3 : 5;
        const std::int32_t tokens = blocks * 2;
        std::vector<BlockTable> tables{BlockTable::FromBlocks(/*group_id=*/0, device.AcquireBlocks(0, blocks), 0)};
        const std::vector<std::vector<ImageSlot>> host_slots(1);
        const bool fits = blocks <= host_packing;
        EXPECT_EQ(coordinator.SnapshotPoolHolds(tables, tokens, host_slots), fits);
        EXPECT_EQ(snapshots.NumEmptyLcmBlocks(), 1);
        auto image = coordinator.TakeImage(tables, tokens, host_slots);
        ASSERT_EQ(image.has_value(), fits);
        if (fits) {
            EXPECT_EQ(image->store_pairs.size(), blocks);
            EXPECT_EQ(snapshots.NumOccupiedSlots(), blocks);
        } else {
            EXPECT_EQ(snapshots.NumEmptyLcmBlocks(), 1);
            EXPECT_EQ(snapshots.NumOccupiedSlots(), 0);
        }
    }
}

}  // namespace
}  // namespace tokenspeed::test
