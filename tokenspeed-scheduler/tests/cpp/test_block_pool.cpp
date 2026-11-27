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
#include <array>
#include <deque>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>

#include "cache/core/block_placement_demand.h"
#include "cache/core/block_pool.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasCacheIndex = requires(T& value) { value.ContainsCachedBlock("key"); };

static_assert(!HasCacheIndex<BlockPool>);

// Reference implementations of what the pool's occupancy index answers in
// constant time, written the way the pool used to: by walking every LCM block.
// The pool must agree with them after any sequence of acquires and releases.
std::int32_t FreeSlotsInGroupByScan(const BlockPool& pool, std::uint32_t group_id, std::int32_t packing) {
    std::int32_t free_slots = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool.NumLcmBlocks(); ++parent_id) {
        if (pool.BoundGroup(parent_id) == group_id) {
            free_slots += packing - pool.OccupiedCount(parent_id);
        }
    }
    return free_slots;
}

std::int32_t OccupiedSlotsByScan(const BlockPool& pool) {
    std::int32_t occupied = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool.NumLcmBlocks(); ++parent_id) {
        occupied += pool.OccupiedCount(parent_id);
    }
    return occupied;
}

std::int32_t MinimumParentOccupancyByScan(const BlockPool& pool, std::uint32_t group_id, std::int32_t packing) {
    std::int32_t minimum = packing;
    for (std::int32_t parent_id = 1; parent_id <= pool.NumLcmBlocks(); ++parent_id) {
        if (pool.BoundGroup(parent_id) == group_id) {
            minimum = std::min(minimum, pool.OccupiedCount(parent_id));
        }
    }
    return minimum;
}

// The placement order the removed full-pool scan produced: parents already
// bound to the group, densest first with the lowest id breaking a tie, each
// contributing its free slots in ascending slot order.
std::vector<CacheBlockLocation> PlanInBoundParentsByScan(const BlockPool& pool, std::uint32_t group_id,
                                                         std::int32_t packing, std::size_t count) {
    std::vector<std::int32_t> parent_ids;
    for (std::int32_t parent_id = 1; parent_id <= pool.NumLcmBlocks(); ++parent_id) {
        if (pool.BoundGroup(parent_id) == group_id && pool.OccupiedCount(parent_id) < packing) {
            parent_ids.push_back(parent_id);
        }
    }
    std::ranges::sort(parent_ids, [&pool](std::int32_t lhs, std::int32_t rhs) {
        const std::int32_t lhs_occupied = pool.OccupiedCount(lhs);
        const std::int32_t rhs_occupied = pool.OccupiedCount(rhs);
        return lhs_occupied != rhs_occupied ? lhs_occupied > rhs_occupied : lhs < rhs;
    });

    std::vector<CacheBlockLocation> locations;
    for (std::int32_t parent_id : parent_ids) {
        for (std::int32_t slot = 0; slot < packing && locations.size() < count; ++slot) {
            const CacheBlockLocation location{.lcm_block_id = parent_id, .slot_index = slot};
            if (!pool.IsOccupied(location)) {
                locations.push_back(location);
            }
        }
        if (locations.size() == count) {
            break;
        }
    }
    return locations;
}

std::vector<CacheBlockLocation> LocationsOf(const std::vector<CacheBlockRef>& blocks) {
    std::vector<CacheBlockLocation> locations;
    locations.reserve(blocks.size());
    for (const CacheBlockRef& block : blocks) {
        locations.push_back(block->Location());
    }
    return locations;
}

TEST(BlockPoolTest, ConstructsExactlyRequestedLcmBlocks) {
    BlockPool pool(8, {4});
    EXPECT_EQ(pool.NumLcmBlocks(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 0);
}

TEST(BlockPoolTest, GroupWithoutPlacementReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            BlockPool pool(1, {1});
            (void)pool.AcquireBlock(/*group_id=*/1);
        },
        "group id has no placement in this pool");
}

TEST(BlockPoolTest, MaintainsFreeSlotsAcrossOccupancyTransitions) {
    BlockPool pool(2, {3});

    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(first);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 2);

    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(second);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 1);

    first.reset();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 2);

    second.reset();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 0);
}

TEST(BlockPoolTest, DestroyWithLiveReferenceReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            auto pool = std::make_unique<BlockPool>(1, std::vector<std::int32_t>{1});
            CacheBlockRef ref = pool->AcquireBlock(/*group_id=*/0);
            pool.reset();
        },
        "BlockPool destroyed with live block references");
}

TEST(BlockPoolTest, KOneBatchAcquireIsAllOrNothing) {
    BlockPool pool(3, {1});
    auto blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/4);
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);

    blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    EXPECT_EQ(blocks.size(), 3u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, EmptyParentBindsToGroupOnFirstChild) {
    BlockPool pool(3, {2});
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/0);

    ASSERT_TRUE(first);
    EXPECT_EQ(first->Location(), (CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
}

TEST(BlockPoolLcmPlacementTest, RejectsInvalidPackingAtConstruction) {
    EXPECT_THROW((void)BlockPool(1, {0}), std::runtime_error);
    EXPECT_THROW((void)BlockPool(2, {std::numeric_limits<std::int32_t>::max()}), std::runtime_error);
}

TEST(BlockPoolLcmPlacementTest, NormalCapacityShortfallDoesNotMutatePartialParent) {
    BlockPool pool(1, {2});
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(existing);
    const CacheBlockLocation location = existing->Location();

    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/2);

    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{0});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
    EXPECT_TRUE(pool.IsOccupied(location));
}

TEST(BlockPoolLcmPlacementTest, ParentRebindsOnlyAfterLastChildReleases) {
    BlockPool pool(1, {2, 8});
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/0);

    EXPECT_FALSE(pool.AcquireBlock(/*group_id=*/1));
    child.reset();
    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/1);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(rebound->Location().slot_index, 0);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{1});
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsRestoreBatchCapacity) {
    BlockPool pool(2, {1, 1});
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/2);
    ASSERT_EQ(blocks.size(), 2u);
    blocks[0].reset();
    blocks[1].reset();

    std::vector<CacheBlockRef> reused = pool.AcquireBlocks(/*group_id=*/1, /*num=*/2);

    EXPECT_EQ(reused.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsAreReusedInReleaseOrder) {
    BlockPool pool(4, {1, 1});
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/4);
    ASSERT_EQ(blocks.size(), 4u);
    const CacheBlockLocation first_released = blocks[0]->Location();
    const CacheBlockLocation second_released = blocks[2]->Location();
    blocks[0].reset();
    blocks[2].reset();

    CacheBlockRef first_reused = pool.AcquireBlock(/*group_id=*/1);
    CacheBlockRef second_reused = pool.AcquireBlock(/*group_id=*/1);

    ASSERT_TRUE(first_reused);
    ASSERT_TRUE(second_reused);
    EXPECT_EQ(first_reused->Location(), first_released);
    EXPECT_EQ(second_reused->Location(), second_released);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, IndependentChildrenShareOneParent) {
    BlockPool pool(1, {2});
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/0);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/0);

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->Location().lcm_block_id, second->Location().lcm_block_id);
    EXPECT_NE(first->Location().slot_index, second->Location().slot_index);
    EXPECT_EQ(pool.OccupiedCount(1), 2);
}

TEST(BlockPoolLcmPlacementTest, ReleasingOneChildKeepsSiblingAndReusesOnlyItsSlot) {
    BlockPool pool(1, {2});
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/0);
    CacheBlockRef sibling = pool.AcquireBlock(/*group_id=*/0);
    const CacheBlockLocation released = first->Location();
    const CacheBlockLocation retained = sibling->Location();

    first.reset();
    EXPECT_FALSE(pool.IsOccupied(released));
    EXPECT_TRUE(pool.IsOccupied(retained));

    CacheBlockRef replacement = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), released);
    EXPECT_TRUE(pool.IsOccupied(retained));
}

TEST(BlockPoolLcmPlacementTest, FillsMostOccupiedPartialParentBeforeFreeParent) {
    BlockPool pool(3, {3});
    auto first_parent = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(first_parent.size(), 3u);
    CacheBlockRef second_parent = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(second_parent);
    first_parent.back().reset();

    CacheBlockRef fills_more_occupied_parent = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(fills_more_occupied_parent);
    EXPECT_EQ(fills_more_occupied_parent->Location().lcm_block_id, first_parent.front()->Location().lcm_block_id);

    CacheBlockRef fills_second_parent = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(fills_second_parent);
    CacheBlockRef fills_second_parent_again = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(fills_second_parent_again);
    CacheBlockRef uses_free_parent = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(uses_free_parent);
    EXPECT_EQ(uses_free_parent->Location().lcm_block_id, 3);
}

TEST(BlockPoolLcmPlacementTest, LastReleaseImmediatelyClearsParentBinding) {
    BlockPool pool(1, {8});
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/0);
    child.reset();

    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(BlockPoolLcmPlacementTest, WideParentTracksSlotsAcrossWordBoundaries) {
    constexpr std::int32_t kPacking = 130;
    BlockPool pool(2, {kPacking});
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/kPacking);
    ASSERT_EQ(blocks.size(), static_cast<std::size_t>(kPacking));
    for (std::int32_t slot = 0; slot < kPacking; ++slot) {
        ASSERT_EQ(blocks[static_cast<std::size_t>(slot)]->Location(),
                  (CacheBlockLocation{.lcm_block_id = 1, .slot_index = slot}));
    }
    EXPECT_FALSE(pool.IsOccupied(CacheBlockLocation{.lcm_block_id = 1, .slot_index = kPacking}));

    const std::vector<std::int32_t> released{129, 64, 0, 127, 63, 128};
    for (const std::int32_t slot : released) {
        blocks[static_cast<std::size_t>(slot)].reset();
    }
    std::vector<CacheBlockLocation> expected;
    for (std::int32_t slot = 0; slot < kPacking; ++slot) {
        const bool occupied = std::ranges::find(released, slot) == released.end();
        EXPECT_EQ(pool.IsOccupied(CacheBlockLocation{.lcm_block_id = 1, .slot_index = slot}), occupied) << slot;
        if (occupied) {
            expected.push_back(CacheBlockLocation{.lcm_block_id = 1, .slot_index = slot});
        }
    }
    EXPECT_EQ(pool.OccupiedLocations(1), expected);
    EXPECT_EQ(pool.OccupiedCount(1), kPacking - 6);

    // The lowest free slot first, found across word boundaries.
    std::vector<CacheBlockRef> refilled = pool.AcquireBlocks(/*group_id=*/0, /*num=*/4);
    EXPECT_EQ(LocationsOf(refilled), (std::vector<CacheBlockLocation>{{1, 0}, {1, 63}, {1, 64}, {1, 127}}));
    CacheBlockRef placed =
        pool.AcquireBlockAt(/*group_id=*/0, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 129});
    EXPECT_TRUE(pool.IsOccupied(CacheBlockLocation{.lcm_block_id = 1, .slot_index = 129}));
    EXPECT_FALSE(pool.IsOccupied(CacheBlockLocation{.lcm_block_id = 1, .slot_index = 128}));
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/0), 1);
}

TEST(BlockPoolLcmPlacementTest, ParentRebindsBetweenInlineAndWideOccupancy) {
    // Group 0 fits its occupancy inline, group 1 exactly fills the inline
    // word, group 2 needs wide words.
    BlockPool pool(1, {3, 64, 100});
    const auto occupied_slots = [&pool](std::int32_t packing) {
        std::vector<std::int32_t> slots;
        for (std::int32_t slot = 0; slot < 2 * packing; ++slot) {
            if (pool.IsOccupied(CacheBlockLocation{.lcm_block_id = 1, .slot_index = slot})) {
                slots.push_back(slot);
            }
        }
        return slots;
    };

    std::vector<CacheBlockRef> wide = pool.AcquireBlocks(/*group_id=*/2, /*num=*/100);
    ASSERT_EQ(wide.size(), 100u);
    wide.clear();
    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);

    // No slot of the wide binding survives into the next one.
    std::vector<CacheBlockRef> narrow = pool.AcquireBlocks(/*group_id=*/0, /*num=*/2);
    EXPECT_EQ(occupied_slots(3), (std::vector<std::int32_t>{0, 1}));
    narrow.clear();

    std::vector<CacheBlockRef> full_word = pool.AcquireBlocks(/*group_id=*/1, /*num=*/64);
    ASSERT_EQ(full_word.size(), 64u);
    EXPECT_EQ(pool.OccupiedLocations(1).size(), 64u);
    EXPECT_FALSE(pool.AcquireBlock(/*group_id=*/1));
    full_word.clear();

    // The wide words were cleared slot by slot; rebinding sees none occupied.
    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/2);
    ASSERT_TRUE(rebound);
    CacheBlockRef far = pool.AcquireBlockAt(/*group_id=*/2, CacheBlockLocation{.lcm_block_id = 1, .slot_index = 99});
    EXPECT_EQ(occupied_slots(100), (std::vector<std::int32_t>{0, 99}));
    EXPECT_EQ(pool.NumFreeSlots(/*group_id=*/2), 98);
}

TEST(BlockPoolTest, AcquireUpToBlocksReturnsAvailableCompatiblePlacements) {
    BlockPool pool(2, {2});
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/0, /*max_num=*/3);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 4);
}

TEST(BlockPoolTest, AcquireUpToBlocksWithPackingOneReturnsPartialCapacity) {
    BlockPool pool(2, {1, 1});
    CacheBlockRef occupied = pool.AcquireBlock(/*group_id=*/0);
    ASSERT_TRUE(occupied);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/1, /*max_num=*/2);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.BoundGroup(partial.front()->Location().lcm_block_id), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 2);
}

TEST(BlockPoolTest, ExactAcquireBlocksRemainsAllOrNothing) {
    BlockPool pool(2, {2});
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*num=*/2).empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 3);
}

// A deliberately exhaustive oracle: rank every compatible free slot from
// scratch. It shares neither the availability indices nor the capacity
// counters of BlockPool, and tracks physical lifetime via reference counts
// plus each parent's last occupancy-change sequence to match the LIFO head
// order of BlockPool's intrusive per-occupancy bucket lists.
class PlacementOracle {
public:
    explicit PlacementOracle(std::int32_t parents) : parents_(static_cast<std::size_t>(parents)) {
        for (std::int32_t id = 1; id <= parents; ++id) {
            empty_.push_back(id);
        }
    }

    std::vector<CacheBlockLocation> Acquire(std::uint32_t group, std::int32_t packing, std::int32_t count,
                                            std::vector<std::int32_t> loads, bool partial,
                                            std::int32_t shard_count = 1) {
        auto planned = parents_;
        auto empty = empty_;
        auto seq = next_seq_;
        const std::int32_t buckets = loads.empty() ? shard_count : static_cast<std::int32_t>(loads.size());
        std::vector<CacheBlockLocation> result;
        for (std::int32_t i = 0; i < count; ++i) {
            // Mirror BlockPool::nextLocation: each bucket picks the LIFO head
            // of its highest non-empty occupancy list, then buckets compete on
            // (load, bucket, -occupied, parent_id, slot).
            using Priority = std::tuple<std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t>;
            std::optional<Priority> best;
            CacheBlockLocation chosen;
            for (std::int32_t bucket = 0; bucket < buckets; ++bucket) {
                std::optional<std::tuple<std::int32_t, std::uint64_t, std::int32_t, std::int32_t>> bucket_head;
                for (std::size_t p = 0; p < planned.size(); ++p) {
                    const Parent& parent = planned[p];
                    if (parent.group != group) {
                        continue;
                    }
                    const auto occupied = static_cast<std::int32_t>(
                        std::ranges::count_if(parent.refs, [](int refs) { return refs > 0; }));
                    if (occupied == packing) {
                        continue;
                    }
                    std::int32_t first_free = packing;
                    for (std::int32_t slot = bucket; slot < packing; slot += buckets) {
                        if (parent.refs[static_cast<std::size_t>(slot)] == 0) {
                            first_free = slot;
                            break;
                        }
                    }
                    if (first_free == packing) {
                        continue;
                    }
                    const auto rank =
                        std::tuple{occupied, parent.touched_seq, static_cast<std::int32_t>(p + 1), first_free};
                    if (!bucket_head || std::tie(std::get<0>(rank), std::get<1>(rank)) >
                                            std::tie(std::get<0>(*bucket_head), std::get<1>(*bucket_head))) {
                        bucket_head = rank;
                    }
                }
                if (!bucket_head) {
                    continue;
                }
                const auto [occupied, _, id, slot] = *bucket_head;
                const Priority priority{loads.empty() ? 0 : loads[static_cast<std::size_t>(bucket)],
                                        loads.empty() ? 0 : bucket, -occupied, id, slot};
                if (!best || priority < *best) {
                    best = priority;
                    chosen = CacheBlockLocation{id, slot};
                }
            }
            if (!best) {
                if (empty.empty()) {
                    if (!partial) {
                        return {};
                    }
                    break;
                }
                const auto slot =
                    loads.empty() ? 0 : static_cast<std::int32_t>(std::ranges::min_element(loads) - loads.begin());
                chosen = CacheBlockLocation{empty.front(), slot};
                empty.pop_front();
                Parent& parent = planned[static_cast<std::size_t>(chosen.lcm_block_id - 1)];
                parent.group = group;
                parent.refs.assign(static_cast<std::size_t>(packing), 0);
            }
            Parent& updated = planned[static_cast<std::size_t>(chosen.lcm_block_id - 1)];
            updated.refs[static_cast<std::size_t>(chosen.slot_index)] = 1;
            updated.touched_seq = ++seq;
            if (!loads.empty()) {
                ++loads[static_cast<std::size_t>(chosen.slot_index) % loads.size()];
            }
            result.push_back(chosen);
        }
        parents_ = std::move(planned);
        empty_ = std::move(empty);
        next_seq_ = seq;
        return result;
    }

    void Retain(CacheBlockLocation location) {
        ++parents_[static_cast<std::size_t>(location.lcm_block_id - 1)]
              .refs[static_cast<std::size_t>(location.slot_index)];
    }

    void Release(CacheBlockLocation location) {
        Parent& parent = parents_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        if (--parent.refs[static_cast<std::size_t>(location.slot_index)] > 0) {
            return;
        }
        if (std::ranges::all_of(parent.refs, [](int refs) { return refs == 0; })) {
            parent.group.reset();
            parent.refs.clear();
            parent.touched_seq = 0;
            empty_.push_back(location.lcm_block_id);
        } else {
            parent.touched_seq = ++next_seq_;
        }
    }

    void Check(const BlockPool& pool) const {
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), static_cast<std::int32_t>(empty_.size()));
        for (std::size_t p = 0; p < parents_.size(); ++p) {
            const Parent& parent = parents_[p];
            const auto id = static_cast<std::int32_t>(p + 1);
            EXPECT_EQ(pool.BoundGroup(id), parent.group);
            EXPECT_EQ(pool.OccupiedCount(id), std::ranges::count_if(parent.refs, [](int refs) { return refs > 0; }));
            for (std::size_t slot = 0; slot < parent.refs.size(); ++slot) {
                EXPECT_EQ(pool.IsOccupied(CacheBlockLocation{id, static_cast<std::int32_t>(slot)}),
                          parent.refs[slot] > 0);
            }
        }
    }

private:
    struct Parent {
        std::optional<std::uint32_t> group;
        std::vector<int> refs;
        std::uint64_t touched_seq{0};
    };
    std::vector<Parent> parents_;
    std::deque<std::int32_t> empty_;
    std::uint64_t next_seq_{0};
};

// Random acquires (ordinary, balanced and partial), shared references and
// releases against the oracle, dropping every reference each clear_every
// steps (never when 0); returns how often a parent bound to a group wider
// than 64 slots was rebound to another group.
int ExpectPlacementMatchesOracle(std::uint32_t seed, std::int32_t parents, const std::vector<std::int32_t>& packing,
                                 const std::vector<std::int32_t>& shards, std::int32_t max_count, int clear_every,
                                 int steps) {
    std::mt19937 random(seed);
    BlockPool pool(parents, packing);
    PlacementOracle oracle(parents);
    std::vector<CacheBlockRef> held;
    for (std::uint32_t group = 0; group < packing.size(); ++group) {
        pool.RegisterGroup(group, packing[group], shards[group]);
    }
    std::vector<std::optional<std::uint32_t>> bound(static_cast<std::size_t>(parents));
    int wide_rebinds = 0;
    for (int step = 0; step < steps; ++step) {
        SCOPED_TRACE(step);
        const auto operation = random() % 10;
        if (operation < 5 || held.empty()) {
            const auto group = random() % packing.size();
            const auto count = static_cast<std::int32_t>(1 + random() % max_count);
            const bool partial = random() % 4 == 0;
            std::vector<std::int32_t> loads;
            if (!partial && shards[group] > 1 && random() % 4 != 0) {
                loads.resize(static_cast<std::size_t>(shards[group]));
                for (auto& load : loads) {
                    load = static_cast<std::int32_t>(random() % 31);
                }
            }
            const auto saved_loads = loads;
            const auto expected = oracle.Acquire(group, packing[group], count, loads, partial, shards[group]);
            auto actual = partial ? pool.AcquireUpToBlocks(group, count) : pool.AcquireBlocks(group, count, loads);
            EXPECT_EQ(actual.size(), expected.size());
            if (actual.size() != expected.size()) {
                return wide_rebinds;
            }
            EXPECT_EQ(loads, saved_loads);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_EQ(actual[i]->Location(), expected[i]);
                held.push_back(std::move(actual[i]));
            }
        } else if (operation == 9) {
            const auto index = random() % held.size();
            oracle.Retain(held[index]->Location());
            CacheBlockRef copy = held[index];
            held.push_back(std::move(copy));
        } else {
            const auto index = random() % held.size();
            oracle.Release(held[index]->Location());
            held[index].reset();
            held[index] = std::move(held.back());
            held.pop_back();
        }
        if (clear_every > 0 && step % clear_every == clear_every - 1) {
            while (!held.empty()) {
                oracle.Release(held.back()->Location());
                held.pop_back();
            }
        }
        oracle.Check(pool);
        for (std::int32_t id = 1; id <= parents; ++id) {
            std::optional<std::uint32_t>& last = bound[static_cast<std::size_t>(id - 1)];
            const std::optional<std::uint32_t> now = pool.BoundGroup(id);
            if (now && last && *now != *last && packing[*last] > 64) {
                ++wide_rebinds;
            }
            if (now) {
                last = now;
            }
        }
    }
    while (!held.empty()) {
        oracle.Release(held.back()->Location());
        held.pop_back();
    }
    oracle.Check(pool);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), pool.NumLcmBlocks());
    return wide_rebinds;
}

TEST(BlockPoolBucketTest, InterleavedLifetimesMatchExhaustivePlacementOracle) {
    for (std::uint32_t seed : {0u, 17u, 2026u}) {
        SCOPED_TRACE(seed);
        (void)ExpectPlacementMatchesOracle(seed, /*parents=*/24, /*packing=*/{8, 16, 1}, /*shards=*/{2, 8, 1},
                                           /*max_count=*/20, /*clear_every=*/0, /*steps=*/1200);
    }
}

// Parents up to 64 slots keep their occupancy inline, wider ones in words of
// their own; parents move between both kinds as groups rebind them.
TEST(BlockPoolBucketTest, WideAndInlineOccupancyMatchExhaustivePlacementOracle) {
    for (std::uint32_t seed : {0u, 17u, 2026u}) {
        SCOPED_TRACE(seed);
        const int wide_rebinds =
            ExpectPlacementMatchesOracle(seed, /*parents=*/6, /*packing=*/{130, 64, 65, 1}, /*shards=*/{2, 4, 5, 1},
                                         /*max_count=*/90, /*clear_every=*/40, /*steps=*/1500);
        EXPECT_GT(wide_rebinds, 0) << "the workload must rebind wide parents to other groups";
    }
}

TEST(BlockPoolOccupancyIndexTest, PlacementFillsTheDensestParentThenTheLowestId) {
    constexpr std::int32_t kPacking = 4;
    BlockPool pool(3, {kPacking});
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3 * kPacking);
    ASSERT_EQ(blocks.size(), 12u);

    // Parent 1 keeps one hole, parent 3 keeps two, parent 2 stays full.
    blocks[3].reset();
    blocks[10].reset();
    blocks[11].reset();

    std::vector<CacheBlockRef> refilled = pool.AcquireBlocks(/*group_id=*/0, /*num=*/3);
    ASSERT_EQ(refilled.size(), 3u);
    EXPECT_EQ(LocationsOf(refilled), (std::vector<CacheBlockLocation>{{.lcm_block_id = 1, .slot_index = 3},
                                                                      {.lcm_block_id = 3, .slot_index = 2},
                                                                      {.lcm_block_id = 3, .slot_index = 3}}));
}

TEST(BlockPoolOccupancyIndexTest, AgreesWithAFullScanUnderRandomChurn) {
    constexpr std::int32_t kPoolSize = 40;
    constexpr std::array<std::int32_t, 3> kPackings{1, 3, 4};
    BlockPool pool(kPoolSize, {1, 3, 4});
    std::vector<CacheBlockRef> held;
    std::mt19937 rng(20260912);

    for (int step = 0; step < 4000; ++step) {
        const std::uint32_t group_id = std::uniform_int_distribution<std::uint32_t>(0, 2)(rng);
        const std::int32_t packing = kPackings[group_id];
        if (held.empty() || std::uniform_int_distribution<int>(0, 2)(rng) != 0) {
            const std::int32_t demand = std::uniform_int_distribution<std::int32_t>(1, 6)(rng);
            // Each block landing in a bound parent must pick a parent with the
            // maximum occupancy among partial parents (densest first) and that
            // parent's lowest free slot.
            const std::int32_t bound_free = pool.NumFreeSlots(group_id);
            const std::size_t in_bound_parents =
                std::min(static_cast<std::size_t>(demand), static_cast<std::size_t>(bound_free));
            std::int32_t max_partial_occ = 0;
            for (std::int32_t p = 1; p <= pool.NumLcmBlocks(); ++p) {
                if (pool.BoundGroup(p) == group_id && pool.OccupiedCount(p) < packing) {
                    max_partial_occ = std::max(max_partial_occ, pool.OccupiedCount(p));
                }
            }
            std::vector<std::int32_t> occ_before(static_cast<std::size_t>(kPoolSize + 1), 0);
            std::vector<std::int32_t> first_free_before(static_cast<std::size_t>(kPoolSize + 1), packing);
            for (std::int32_t p = 1; p <= pool.NumLcmBlocks(); ++p) {
                if (pool.BoundGroup(p) == group_id) {
                    occ_before[static_cast<std::size_t>(p)] = pool.OccupiedCount(p);
                    for (std::int32_t s = 0; s < packing; ++s) {
                        if (!pool.IsOccupied(CacheBlockLocation{p, s})) {
                            first_free_before[static_cast<std::size_t>(p)] = s;
                            break;
                        }
                    }
                }
            }

            std::vector<CacheBlockRef> acquired = pool.AcquireUpToBlocks(group_id, demand);
            if (in_bound_parents > 0) {
                const CacheBlockLocation first = acquired[0]->Location();
                EXPECT_EQ(occ_before[static_cast<std::size_t>(first.lcm_block_id)], max_partial_occ) << "step " << step;
                EXPECT_EQ(first.slot_index, first_free_before[static_cast<std::size_t>(first.lcm_block_id)])
                    << "step " << step;
            }
            for (CacheBlockRef& block : acquired) {
                held.push_back(std::move(block));
            }
        } else {
            const std::size_t victim = std::uniform_int_distribution<std::size_t>(0, held.size() - 1)(rng);
            std::swap(held[victim], held.back());
            held.pop_back();
        }

        ASSERT_EQ(pool.NumOccupiedSlots(), OccupiedSlotsByScan(pool)) << "step " << step;
        for (std::uint32_t id = 0; id < kPackings.size(); ++id) {
            ASSERT_EQ(pool.NumFreeSlots(id), FreeSlotsInGroupByScan(pool, id, kPackings[id]))
                << "step " << step << " group " << id;
            ASSERT_EQ(pool.MinimumParentOccupancy(id), MinimumParentOccupancyByScan(pool, id, kPackings[id]))
                << "step " << step << " group " << id;
        }
    }
    held.clear();
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), kPoolSize);
    for (std::uint32_t id = 0; id < kPackings.size(); ++id) {
        EXPECT_EQ(pool.NumFreeSlots(id), 0);
        EXPECT_EQ(pool.MinimumParentOccupancy(id), kPackings[id]);
    }
}

TEST(BlockPoolOccupancyIndexTest, MinimumOccupancyIncludesParentsWithHolesInOnlyOneBucket) {
    BlockPool pool(3, {4});
    pool.RegisterGroup(0, 4, 2);
    auto blocks = pool.AcquireBlocksInBuckets(0, std::array<std::int32_t, 6>{0, 0, 1, 1, 0, 0});
    ASSERT_EQ(blocks.size(), 6u);
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 2);
    blocks[0].reset();
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 2);
    blocks[4].reset();
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 1);
    blocks[5].reset();
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 3);
    blocks.clear();
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 4);
}

TEST(BlockPoolOccupancyIndexTest, FreeSlotsAreScopedToTheOwningGroup) {
    BlockPool pool(4, {1, 4, 2});
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/1, /*num=*/5);
    ASSERT_EQ(first.size(), 5u);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/2);
    ASSERT_TRUE(second);

    // Group 1 owns two parents holding five of eight slots; group 2 owns one
    // parent holding one of two. The remaining empty parent belongs to neither.
    EXPECT_EQ(pool.NumFreeSlots(1), 3);
    EXPECT_EQ(pool.NumFreeSlots(2), 1);
    EXPECT_EQ(pool.NumFreeSlots(0), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);

    first.clear();
    second.reset();
}

// ---------------------------------------------------------------------------
// Bucket-constrained acquisition: a retraction snapshot and its restore keep
// every block in the bucket (slot % shard_count) of the block it images.
// ---------------------------------------------------------------------------

TEST(BlockPlacementDemandTest, TotalBlocksIncludesBothBlocksOfEachPair) {
    BlockPlacementDemand block_demand{.blocks_by_bucket = {2, 3}, .any_bucket_blocks = 4, .same_bucket_pairs = 2};
    EXPECT_EQ(block_demand.TotalBlocks(), 13);

    block_demand.same_bucket_pairs = std::numeric_limits<std::int32_t>::max();
    EXPECT_EQ(block_demand.TotalBlocks(), 9 + 2 * std::int64_t{std::numeric_limits<std::int32_t>::max()});
}

TEST(BlockPoolBucketAcquireTest, ParentsNeededSharesOpenedParentsAcrossBuckets) {
    // packing 4, two buckets: an opened parent gives each bucket two slots.
    const std::array<std::int64_t, 2> no_holes{0, 0};
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{3, 3}, no_holes, /*any_bucket_blocks=*/0,
                                      /*packing=*/4),
              2)
        << "both buckets are served by the same two parents";
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{5, 0}, no_holes, 0, 4), 3)
        << "one bucket alone forces ceil(5 / 2) parents";
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{1, 1}, std::array<std::int64_t, 2>{0, 1}, 0, 4), 1)
        << "a hole in one bucket does not serve the other";
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{1, 1}, std::array<std::int64_t, 2>{1, 1}, 0, 4), 0);
    // Dense blocks take the holes the bucket demands leave: four slots per
    // parent minus the two bucket blocks leaves two, so three dense blocks
    // need a second parent.
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{1, 1}, no_holes, /*any_bucket_blocks=*/2, 4), 1);
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{1, 1}, no_holes, /*any_bucket_blocks=*/3, 4), 2);
    // One bucket is ordinary placement: holes plus parents times packing.
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 1>{0}, std::array<std::int64_t, 1>{1}, 5, 2), 2);
    // One slot per parent: every block the holes miss opens a parent.
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 1>{5}, std::array<std::int64_t, 1>{2}, 0, 1), 3);
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 1>{5}, std::array<std::int64_t, 1>{2}, 4, 1), 7);
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 1>{1}, std::array<std::int64_t, 1>{3}, 2, 1), 0);
    // One slot per bucket (packing 2, two buckets): the shortest bucket opens
    // a parent per block, while the total still counts two slots a parent.
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{3, 1}, std::array<std::int64_t, 2>{0, 1}, 0, 2), 3);
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{1, 1}, std::array<std::int64_t, 2>{0, 1}, 5, 2), 3);
}

// A pair of blocks that must share a bucket, either bucket: each pair takes
// the bucket with the most holes its demand leaves, and that choice needs no
// more parents than any other.
TEST(BlockPoolBucketAcquireTest, APairTakesTheBucketWithTheMostHolesLeft) {
    using Buckets = std::vector<std::int32_t>;
    EXPECT_EQ(AssignPairBuckets(BlockPlacementDemand{.blocks_by_bucket = {2, 0}, .same_bucket_pairs = 1},
                                std::array<std::int64_t, 2>{3, 2}),
              Buckets{1})
        << "bucket 0 has more holes but fewer left after its demand";
    EXPECT_EQ(AssignPairBuckets(BlockPlacementDemand{.blocks_by_bucket = {0, 0}, .same_bucket_pairs = 2},
                                std::array<std::int64_t, 2>{2, 2}),
              (Buckets{0, 1}))
        << "the lower bucket on a tie, then the one the first pair left emptier";
    EXPECT_EQ(AssignPairBuckets(BlockPlacementDemand{.blocks_by_bucket = {0, 0}, .same_bucket_pairs = 2},
                                std::array<std::int64_t, 2>{5, 0}),
              (Buckets{0, 0}));
    EXPECT_EQ(AssignPairBuckets(BlockPlacementDemand{.blocks_by_bucket = {3}, .same_bucket_pairs = 2},
                                std::array<std::int64_t, 1>{0}),
              (Buckets{0, 0}))
        << "one bucket takes every pair";

    // packing 4, two buckets. The pair fits bucket 1's two holes; bucket 0
    // would open a parent for it.
    std::array<std::int64_t, 2> scratch{};
    EXPECT_EQ(ParentsNeededForBucketsAndPairs(
                  BlockPlacementDemand{.blocks_by_bucket = {2, 0}, .any_bucket_blocks = 0, .same_bucket_pairs = 1},
                  std::array<std::int64_t, 2>{3, 2}, /*packing=*/4, scratch),
              0);
    EXPECT_EQ(ParentsNeededForBuckets(std::array<std::int64_t, 2>{4, 0}, std::array<std::int64_t, 2>{3, 2}, 0, 4), 1);
    // A pair needs two slots of one bucket: with one slot per bucket in a
    // parent (packing 2), two parents.
    EXPECT_EQ(ParentsNeededForBucketsAndPairs(
                  BlockPlacementDemand{.blocks_by_bucket = {0, 0}, .any_bucket_blocks = 0, .same_bucket_pairs = 1},
                  std::array<std::int64_t, 2>{0, 0}, /*packing=*/2, scratch),
              2);
    // Dense blocks take the holes the pairs leave.
    EXPECT_EQ(ParentsNeededForBucketsAndPairs(
                  BlockPlacementDemand{.blocks_by_bucket = {0, 0}, .any_bucket_blocks = 2, .same_bucket_pairs = 1},
                  std::array<std::int64_t, 2>{0, 0}, /*packing=*/4, scratch),
              1);
    EXPECT_EQ(ParentsNeededForBucketsAndPairs(
                  BlockPlacementDemand{.blocks_by_bucket = {0, 0}, .any_bucket_blocks = 3, .same_bucket_pairs = 1},
                  std::array<std::int64_t, 2>{0, 0}, /*packing=*/4, scratch),
              2);
}

TEST(BlockPoolBucketAcquireTest, PlacementIsDeterministicAndOpensOneParentForBothBuckets) {
    BlockPool pool(2, {4});
    pool.RegisterGroup(/*group_id=*/0, /*packing=*/4, /*shard_count=*/2);

    // Bucket 1 has no hole: the FIFO's first parent opens at slot 1. Its slot
    // 3 serves the second bucket-1 block and slot 0 the bucket-0 block.
    const std::array<std::int32_t, 3> buckets{1, 1, 0};
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocksInBuckets(0, buckets);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(LocationsOf(blocks), (std::vector<CacheBlockLocation>{{1, 1}, {1, 3}, {1, 0}}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[0], 1) << "slot 2 is the one hole left, in bucket 0";
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[1], 0);

    // The hole is used before a second parent opens; bucket 1 then opens the
    // next FIFO parent at its own slot.
    const std::array<std::int32_t, 2> more{0, 1};
    std::vector<CacheBlockRef> second = pool.AcquireBlocksInBuckets(0, more);
    ASSERT_EQ(second.size(), 2u);
    EXPECT_EQ(LocationsOf(second), (std::vector<CacheBlockLocation>{{1, 2}, {2, 1}}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolBucketAcquireTest, FailedAcquisitionLeavesIndicesUntouched) {
    BlockPool pool(2, {4});
    pool.RegisterGroup(/*group_id=*/0, /*packing=*/4, /*shard_count=*/2);
    CacheBlockRef seed = pool.AcquireBlock(0);  // opens parent 1 at slot 0
    ASSERT_TRUE(seed);
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[0], 1);
    ASSERT_EQ(pool.FreeSlotsByBucket(0)[1], 2);

    // Bucket 0 needs four with one hole and two slots per opened parent:
    // two parents, but only one is empty.
    const std::array<std::int32_t, 4> too_many{0, 0, 0, 0};
    EXPECT_TRUE(pool.AcquireBlocksInBuckets(0, too_many).empty());
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(pool.NumOccupiedSlots(), 1);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[0], 1);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[1], 2);

    // The same pool still serves a demand that fits, in the untouched order:
    // bucket 1's two holes in parent 1 first, then parent 2 at slot 1.
    const std::array<std::int32_t, 3> fits{1, 1, 1};
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocksInBuckets(0, fits);
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(LocationsOf(blocks), (std::vector<CacheBlockLocation>{{1, 1}, {1, 3}, {2, 1}}));
}

TEST(BlockPoolBucketAcquireTest, ReleaseRestoresPerBucketHolesAndUnbindsWholeParents) {
    BlockPool pool(1, {4});
    pool.RegisterGroup(/*group_id=*/0, /*packing=*/4, /*shard_count=*/2);
    const std::array<std::int32_t, 4> all{0, 1, 0, 1};
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocksInBuckets(0, all);
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[0], 0);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[1], 0);

    blocks[1].reset();  // slot 1, bucket 1
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[0], 0);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[1], 1);
    EXPECT_EQ(pool.NumFreeSlots(0), 1) << "the group total is the sum over buckets";

    blocks.clear();  // the parent unbinds: its slots belong to the FIFO, not the group
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[0], 0);
    EXPECT_EQ(pool.FreeSlotsByBucket(0)[1], 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(BlockPoolBucketAcquireTest, OneBucketIsOrdinaryPlacement) {
    BlockPool pool(2, {2});
    CacheBlockRef seed = pool.AcquireBlock(0);  // parent 1, slot 0; slot 1 is a hole
    ASSERT_TRUE(seed);
    const std::array<std::int32_t, 2> zeros{0, 0};
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocksInBuckets(0, zeros);
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(LocationsOf(blocks), (std::vector<CacheBlockLocation>{{1, 1}, {2, 0}}))
        << "the hole in the bound parent first, then the next FIFO parent";
}

TEST(BlockPoolRequestAwareTest, IsolatesInterleavedRequestsAndReclaimsWholeParents) {
    for (const std::int32_t packing : {12, 6}) {
        SCOPED_TRACE(::testing::Message() << "packing=" << packing);
        BlockPool pool(/*num_lcm_blocks=*/4, {packing, 1});
        std::vector<CacheBlockRef> req_a;
        std::vector<CacheBlockRef> req_b;

        for (std::int32_t step = 0; step < packing; ++step) {
            const std::optional<std::int32_t> pref_a =
                req_a.empty() ? std::nullopt : std::optional<std::int32_t>{req_a.back()->Location().lcm_block_id};
            auto a = pool.AcquireBlocksWithAffinity(
                /*group_id=*/0, /*num=*/1, /*bucket_loads=*/{},
                ParentAffinity{.preferred_parent = pref_a, .reserved_empty_parents = 0});
            ASSERT_EQ(a.size(), 1u);
            req_a.push_back(std::move(a.front()));

            const std::optional<std::int32_t> pref_b =
                req_b.empty() ? std::nullopt : std::optional<std::int32_t>{req_b.back()->Location().lcm_block_id};
            auto b = pool.AcquireBlocksWithAffinity(
                /*group_id=*/0, /*num=*/1, /*bucket_loads=*/{},
                ParentAffinity{.preferred_parent = pref_b, .reserved_empty_parents = 0});
            ASSERT_EQ(b.size(), 1u);
            req_b.push_back(std::move(b.front()));
        }

        const std::int32_t parent_a = req_a.front()->Location().lcm_block_id;
        const std::int32_t parent_b = req_b.front()->Location().lcm_block_id;
        EXPECT_NE(parent_a, parent_b);
        for (const CacheBlockRef& block : req_a) {
            EXPECT_EQ(block->Location().lcm_block_id, parent_a);
        }
        for (const CacheBlockRef& block : req_b) {
            EXPECT_EQ(block->Location().lcm_block_id, parent_b);
        }
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

        // Releasing Request A returns its entire LCM parent at once.
        req_a.clear();
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);
        EXPECT_FALSE(pool.BoundGroup(parent_a).has_value());
        EXPECT_EQ(pool.OccupiedCount(parent_b), packing);
    }
}

TEST(BlockPoolRequestAwareTest, RespectsReservedEmptyParentsAndReusesPreferredParentAtFloor) {
    BlockPool pool(/*num_lcm_blocks=*/5, {6, 1});

    // Reserve 3 empty parents (e.g. for later packing=1 groups): A opens
    // parent 1 (5 -> 4), B opens parent 2 (4 -> 3 == reserved).
    auto a = pool.AcquireBlocksWithAffinity(
        0, 2, {}, ParentAffinity{.preferred_parent = std::nullopt, .reserved_empty_parents = 3});
    ASSERT_EQ(a.size(), 2u);
    EXPECT_EQ(a[0]->Location().lcm_block_id, 1);
    EXPECT_EQ(a[1]->Location().lcm_block_id, 1);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 4);

    auto b = pool.AcquireBlocksWithAffinity(
        0, 1, {}, ParentAffinity{.preferred_parent = std::nullopt, .reserved_empty_parents = 3});
    ASSERT_EQ(b.size(), 1u);
    EXPECT_EQ(b[0]->Location().lcm_block_id, 2);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);

    // At the reserve floor, a new request C cannot open parent 3: it falls
    // back to a hole in the densest bound parent (parent 1) without crossing
    // the reserve floor.
    auto c = pool.AcquireBlocksWithAffinity(
        0, 1, {}, ParentAffinity{.preferred_parent = std::nullopt, .reserved_empty_parents = 3});
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0]->Location().lcm_block_id, 1);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);

    // Meanwhile B still reuses its own preferred parent 2 even at the floor.
    auto b_more =
        pool.AcquireBlocksWithAffinity(0, 5, {}, ParentAffinity{.preferred_parent = 2, .reserved_empty_parents = 3});
    ASSERT_EQ(b_more.size(), 5u);
    for (const CacheBlockRef& block : b_more) {
        EXPECT_EQ(block->Location().lcm_block_id, 2);
    }
    EXPECT_EQ(pool.OccupiedCount(2), 6);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);

    // Once parent 2 is full, requesting another block for B at the floor falls
    // back to parent 1's remaining holes rather than stealing reserved parents.
    auto b_overflow =
        pool.AcquireBlocksWithAffinity(0, 1, {}, ParentAffinity{.preferred_parent = 2, .reserved_empty_parents = 3});
    ASSERT_EQ(b_overflow.size(), 1u);
    EXPECT_EQ(b_overflow[0]->Location().lcm_block_id, 1);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);
}

TEST(BlockPoolRequestAwareTest, BalancesKvParallelShardsWhileClusteringPerRequest) {
    for (const std::int32_t shards : {2, 4}) {
        SCOPED_TRACE(::testing::Message() << "shards=" << shards);
        constexpr std::int32_t kPacking = 12;
        BlockPool pool(/*num_lcm_blocks=*/6, {kPacking, 1});
        pool.RegisterGroup(/*group_id=*/0, kPacking, shards);

        std::vector<CacheBlockRef> req_a;
        std::vector<CacheBlockRef> req_b;
        std::vector<std::int32_t> loads_a(static_cast<std::size_t>(shards), 0);
        std::vector<std::int32_t> loads_b(static_cast<std::size_t>(shards), 0);

        for (std::int32_t step = 0; step < kPacking; ++step) {
            const std::optional<std::int32_t> pref_a =
                req_a.empty() ? std::nullopt : std::optional<std::int32_t>{req_a.back()->Location().lcm_block_id};
            auto a = pool.AcquireBlocksWithAffinity(
                0, 1, loads_a, ParentAffinity{.preferred_parent = pref_a, .reserved_empty_parents = 1});
            ASSERT_EQ(a.size(), 1u);
            ++loads_a[static_cast<std::size_t>(a.front()->Location().slot_index % shards)];
            req_a.push_back(std::move(a.front()));

            const std::optional<std::int32_t> pref_b =
                req_b.empty() ? std::nullopt : std::optional<std::int32_t>{req_b.back()->Location().lcm_block_id};
            auto b = pool.AcquireBlocksWithAffinity(
                0, 1, loads_b, ParentAffinity{.preferred_parent = pref_b, .reserved_empty_parents = 1});
            ASSERT_EQ(b.size(), 1u);
            ++loads_b[static_cast<std::size_t>(b.front()->Location().slot_index % shards)];
            req_b.push_back(std::move(b.front()));
        }

        const std::int32_t parent_a = req_a.front()->Location().lcm_block_id;
        const std::int32_t parent_b = req_b.front()->Location().lcm_block_id;
        EXPECT_NE(parent_a, parent_b);
        for (const CacheBlockRef& block : req_a) {
            EXPECT_EQ(block->Location().lcm_block_id, parent_a);
        }
        for (const CacheBlockRef& block : req_b) {
            EXPECT_EQ(block->Location().lcm_block_id, parent_b);
        }
        for (std::int32_t bucket = 0; bucket < shards; ++bucket) {
            EXPECT_EQ(loads_a[static_cast<std::size_t>(bucket)], kPacking / shards);
            EXPECT_EQ(loads_b[static_cast<std::size_t>(bucket)], kPacking / shards);
        }
    }

    // Unbalanced AcquireBlocksWithAffinity (empty bucket_loads) on a sharded
    // group still scans every bucket of preferred_parent rather than stopping
    // once bucket 0 is full.
    BlockPool sharded(/*num_lcm_blocks=*/3, {4});
    sharded.RegisterGroup(/*group_id=*/0, /*packing=*/4, /*shard_count=*/2);
    const std::array<std::int32_t, 2> b0_only{0, 0};
    auto filled_b0 = sharded.AcquireBlocksInBuckets(0, b0_only);
    ASSERT_EQ(filled_b0.size(), 2u);
    ASSERT_EQ(filled_b0[0]->Location(), (CacheBlockLocation{1, 0}));
    ASSERT_EQ(filled_b0[1]->Location(), (CacheBlockLocation{1, 2}));
    auto unbalanced = sharded.AcquireBlocksWithAffinity(
        0, 2, /*bucket_loads=*/{}, ParentAffinity{.preferred_parent = 1, .reserved_empty_parents = 0});
    ASSERT_EQ(unbalanced.size(), 2u);
    EXPECT_EQ(LocationsOf(unbalanced), (std::vector<CacheBlockLocation>{{1, 1}, {1, 3}}));
    EXPECT_EQ(sharded.NumEmptyLcmBlocks(), 2);
}

TEST(BlockPoolRequestAwareTest, AcquireBlocksInBucketsWithAffinityClustersAndBypassesPackingOne) {
    BlockPool pool(/*num_lcm_blocks=*/4, {6, 1});
    pool.RegisterGroup(/*group_id=*/0, /*packing=*/6, /*shard_count=*/2);
    pool.RegisterGroup(/*group_id=*/1, /*packing=*/1, /*shard_count=*/1);

    // Seed parent 1 with one block in bucket 0, leaving holes in both buckets.
    CacheBlockRef seed = pool.AcquireBlock(0);
    ASSERT_TRUE(seed);
    ASSERT_EQ(seed->Location().lcm_block_id, 1);
    ASSERT_EQ(pool.NumEmptyLcmBlocks(), 3);

    // With empty (3) > reserved (1) and no preferred parent, bucket acquisition
    // opens parent 2 and clusters both requested buckets into parent 2.
    const std::array<std::int32_t, 2> pair{0, 1};
    auto first = pool.AcquireBlocksInBucketsWithAffinity(
        0, pair, ParentAffinity{.preferred_parent = std::nullopt, .reserved_empty_parents = 1});
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(LocationsOf(first), (std::vector<CacheBlockLocation>{{2, 0}, {2, 1}}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

    // At empty == reserved (2), preferred_parent = 2 still clusters inside
    // parent 2 without opening parent 3.
    auto cont = pool.AcquireBlocksInBucketsWithAffinity(
        0, pair, ParentAffinity{.preferred_parent = 2, .reserved_empty_parents = 2});
    ASSERT_EQ(cont.size(), 2u);
    EXPECT_EQ(LocationsOf(cont), (std::vector<CacheBlockLocation>{{2, 2}, {2, 3}}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

    // At empty == reserved (2) with preferred_parent = nullopt, bucket
    // acquisition falls back to existing holes without touching empty parents.
    auto fallback = pool.AcquireBlocksInBucketsWithAffinity(
        0, pair, ParentAffinity{.preferred_parent = std::nullopt, .reserved_empty_parents = 2});
    ASSERT_EQ(fallback.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

    // packing == 1 bypasses affinity even when empty <= reserved_empty_parents.
    const std::array<std::int32_t, 2> state_buckets{0, 0};
    auto state = pool.AcquireBlocksInBucketsWithAffinity(
        1, state_buckets, ParentAffinity{.preferred_parent = 1, .reserved_empty_parents = 2});
    ASSERT_EQ(state.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolRequestAwareTest, WidePackingAndWideShardsWithAffinityMaintainIntrusiveListInvariants) {
    constexpr std::int32_t kParents = 8;
    constexpr std::int32_t kWidePacking = 130;  // > 64 inline occupancy mask
    constexpr std::int32_t kWideShards = 5;     // multi-shard bucket_links_ stride
    BlockPool pool(kParents, {kWidePacking, 1});
    pool.RegisterGroup(/*group_id=*/0, kWidePacking, kWideShards);
    pool.RegisterGroup(/*group_id=*/1, /*packing=*/1, /*shard_count=*/1);

    std::mt19937 rng(0x5A17C0DEu);
    std::vector<CacheBlockRef> held_g0;
    std::vector<CacheBlockRef> held_g1;

    const auto check_invariants = [&] {
        std::int32_t empty = 0;
        std::array<std::int64_t, kWideShards> free_by_bucket{};
        for (std::int32_t p = 1; p <= kParents; ++p) {
            const auto bound = pool.BoundGroup(p);
            if (!bound) {
                ++empty;
                EXPECT_EQ(pool.OccupiedCount(p), 0);
                continue;
            }
            const std::int32_t packing = *bound == 0u ? kWidePacking : 1;
            std::int32_t occ = 0;
            for (std::int32_t slot = 0; slot < packing; ++slot) {
                if (pool.IsOccupied(CacheBlockLocation{p, slot})) {
                    ++occ;
                } else if (*bound == 0u) {
                    ++free_by_bucket[static_cast<std::size_t>(slot % kWideShards)];
                }
            }
            EXPECT_EQ(pool.OccupiedCount(p), occ) << "parent " << p;
        }
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), empty);
        const auto actual = pool.FreeSlotsByBucket(0);
        ASSERT_EQ(actual.size(), static_cast<std::size_t>(kWideShards));
        std::int64_t total_free = 0;
        for (std::size_t b = 0; b < static_cast<std::size_t>(kWideShards); ++b) {
            EXPECT_EQ(actual[b], free_by_bucket[b]) << "bucket " << b;
            total_free += free_by_bucket[b];
        }
        EXPECT_EQ(pool.NumFreeSlots(0), total_free);
    };

    for (int step = 0; step < 300; ++step) {
        const std::uint32_t op = rng() % 10;
        if (op < 4) {
            const auto count = static_cast<std::int32_t>(1 + rng() % 17);
            const auto reserved = static_cast<std::int64_t>(rng() % 4);
            std::optional<std::int32_t> pref;
            if (!held_g0.empty() && (rng() % 2 == 0)) {
                pref = held_g0[rng() % held_g0.size()]->Location().lcm_block_id;
            }
            std::vector<std::int32_t> loads;
            if (rng() % 2 == 0) {
                loads.resize(kWideShards);
                for (std::int32_t& load : loads) {
                    load = static_cast<std::int32_t>(rng() % 10);
                }
            }
            auto blocks = pool.AcquireBlocksWithAffinity(
                0, count, loads, ParentAffinity{.preferred_parent = pref, .reserved_empty_parents = reserved});
            for (CacheBlockRef& block : blocks) {
                held_g0.push_back(std::move(block));
            }
        } else if (op < 6) {
            std::vector<std::int32_t> buckets(1 + rng() % 6);
            for (std::int32_t& b : buckets) {
                b = static_cast<std::int32_t>(rng() % kWideShards);
            }
            std::optional<std::int32_t> pref;
            if (!held_g0.empty() && (rng() % 2 == 0)) {
                pref = held_g0[rng() % held_g0.size()]->Location().lcm_block_id;
            }
            auto blocks = pool.AcquireBlocksInBucketsWithAffinity(
                0, buckets,
                ParentAffinity{
                    .preferred_parent = pref,
                    .reserved_empty_parents = static_cast<std::int64_t>(rng() % 3),
                });
            for (CacheBlockRef& block : blocks) {
                held_g0.push_back(std::move(block));
            }
        } else if (op == 6) {
            auto blocks = pool.AcquireBlocks(1, static_cast<std::int32_t>(1 + rng() % 3));
            for (CacheBlockRef& block : blocks) {
                held_g1.push_back(std::move(block));
            }
        } else if (op < 9 && !held_g0.empty()) {
            const std::size_t drop = std::min<std::size_t>(held_g0.size(), 1 + rng() % 25);
            for (std::size_t d = 0; d < drop; ++d) {
                const std::size_t idx = rng() % held_g0.size();
                held_g0[idx] = std::move(held_g0.back());
                held_g0.pop_back();
            }
        } else if (!held_g1.empty()) {
            held_g1.pop_back();
        }
        if (step % 60 == 59) {
            // Exercise full unbind and rebind across wide group 0 and group 1.
            held_g0.clear();
            held_g1.clear();
        }
        check_invariants();
    }
}

TEST(BlockPoolRequestAwareTest, LateWiderShardRegistrationPreservesLiveIntrusiveLinks) {
    // Group 0 registers with 1 bucket and populates multiple partial parents at
    // the same occupancy level before group 1 registers with 4 buckets, forcing
    // the flat bucket_links_ table to widen while group 0's lists are live.
    BlockPool pool(/*num_lcm_blocks=*/6, {8, 12, 1});
    pool.RegisterGroup(/*group_id=*/2, /*packing=*/1, /*shard_count=*/1);
    pool.RegisterGroup(/*group_id=*/0, /*packing=*/8, /*shard_count=*/1);

    // Put parents 1, 2, 3 into group 0's occupancy-1 list (order: 3 -> 2 -> 1).
    auto p1 = pool.AcquireBlocksWithAffinity(0, 1, {}, ParentAffinity{.preferred_parent = std::nullopt});
    auto p2 = pool.AcquireBlocksWithAffinity(0, 1, {}, ParentAffinity{.preferred_parent = std::nullopt});
    auto p3 = pool.AcquireBlocksWithAffinity(0, 1, {}, ParentAffinity{.preferred_parent = std::nullopt});
    ASSERT_EQ(p1[0]->Location().lcm_block_id, 1);
    ASSERT_EQ(p2[0]->Location().lcm_block_id, 2);
    ASSERT_EQ(p3[0]->Location().lcm_block_id, 3);
    ASSERT_EQ(pool.MinimumParentOccupancy(0), 1);

    // Late registration widens link_stride_ from 1 to 4 while 3 -> 2 -> 1 is live.
    pool.RegisterGroup(/*group_id=*/1, /*packing=*/12, /*shard_count=*/4);
    const std::array<std::int32_t, 4> loads{0, 0, 0, 0};
    auto g1 = pool.AcquireBlocksWithAffinity(1, 5, loads, ParentAffinity{.preferred_parent = std::nullopt});
    ASSERT_EQ(g1.size(), 5u);
    EXPECT_EQ(g1[0]->Location().lcm_block_id, 4);

    // Unlink middle element (parent 2), then advance head (parent 3) and tail
    // (parent 1) via hole-first acquisition to prove migrated prev/next pointers.
    p2.clear();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 3);
    CacheBlockRef next_hole = pool.AcquireBlock(0);
    ASSERT_TRUE(next_hole);
    EXPECT_EQ(next_hole->Location().lcm_block_id, 3) << "head of group 0's occupancy-1 list survived stride widening";
    EXPECT_EQ(pool.MinimumParentOccupancy(0), 1) << "parent 1 remains reachable behind parent 3";

    p3.clear();
    next_hole.reset();
    p1.clear();
    g1.clear();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 6);
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);
}

}  // namespace
}  // namespace tokenspeed::test
