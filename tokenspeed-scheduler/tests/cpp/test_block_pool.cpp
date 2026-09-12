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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>

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
    BlockPool pool(8);
    EXPECT_EQ(pool.NumLcmBlocks(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
}

TEST(BlockPoolTest, DestroyWithLiveReferenceReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            auto pool = std::make_unique<BlockPool>(1);
            CacheBlockRef ref = pool->AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
            pool.reset();
        },
        "BlockPool destroyed with live block references");
}

TEST(BlockPoolTest, KOneBatchAcquireIsAllOrNothing) {
    BlockPool pool(3);
    auto blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);

    blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/3);
    EXPECT_EQ(blocks.size(), 3u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, EmptyParentBindsToGroupOnFirstChild) {
    BlockPool pool(3);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    EXPECT_EQ(first->Location(), (CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
}

TEST(BlockPoolLcmPlacementTest, RejectsPackingChangeWhileParentIsOccupied) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_THROW((void)pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/4), std::runtime_error);
}

TEST(BlockPoolLcmPlacementTest, NormalCapacityShortfallDoesNotMutatePartialParent) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(existing);
    const CacheBlockLocation location = existing->Location();

    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2, /*num=*/2);

    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
    EXPECT_TRUE(pool.IsOccupied(location));
}

TEST(BlockPoolLcmPlacementTest, ParentRebindsOnlyAfterLastChildReleases) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_FALSE(pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8));
    child.reset();
    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(rebound->Location().slot_index, 0);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{2});
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsRestoreBatchCapacity) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);
    ASSERT_EQ(blocks.size(), 2u);
    blocks[0].reset();
    blocks[1].reset();

    std::vector<CacheBlockRef> reused = pool.AcquireBlocks(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);

    EXPECT_EQ(reused.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsAreReusedInReleaseOrder) {
    BlockPool pool(4);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    ASSERT_EQ(blocks.size(), 4u);
    const CacheBlockLocation first_released = blocks[0]->Location();
    const CacheBlockLocation second_released = blocks[2]->Location();
    blocks[0].reset();
    blocks[2].reset();

    CacheBlockRef first_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);
    CacheBlockRef second_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);

    ASSERT_TRUE(first_reused);
    ASSERT_TRUE(second_reused);
    EXPECT_EQ(first_reused->Location(), first_released);
    EXPECT_EQ(second_reused->Location(), second_released);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, IndependentChildrenShareOneParent) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->Location().lcm_block_id, second->Location().lcm_block_id);
    EXPECT_NE(first->Location().slot_index, second->Location().slot_index);
    EXPECT_EQ(pool.OccupiedCount(1), 2);
}

TEST(BlockPoolLcmPlacementTest, ReleasingOneChildKeepsSiblingAndReusesOnlyItsSlot) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef sibling = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    const CacheBlockLocation released = first->Location();
    const CacheBlockLocation retained = sibling->Location();

    first.reset();
    EXPECT_FALSE(pool.IsOccupied(released));
    EXPECT_TRUE(pool.IsOccupied(retained));

    CacheBlockRef replacement = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), released);
    EXPECT_TRUE(pool.IsOccupied(retained));
}

TEST(BlockPoolLcmPlacementTest, FillsMostOccupiedPartialParentBeforeFreeParent) {
    BlockPool pool(3);
    auto first_parent = pool.AcquireBlocks(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3, /*num=*/3);
    ASSERT_EQ(first_parent.size(), 3u);
    CacheBlockRef second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(second_parent);
    first_parent.back().reset();

    CacheBlockRef fills_more_occupied_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_more_occupied_parent);
    EXPECT_EQ(fills_more_occupied_parent->Location().lcm_block_id, first_parent.front()->Location().lcm_block_id);

    CacheBlockRef fills_second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent);
    CacheBlockRef fills_second_parent_again = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent_again);
    CacheBlockRef uses_free_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(uses_free_parent);
    EXPECT_EQ(uses_free_parent->Location().lcm_block_id, 3);
}

TEST(BlockPoolLcmPlacementTest, LastReleaseImmediatelyClearsParentBinding) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/4, /*cache_blocks_per_lcm_block=*/8);
    child.reset();

    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(BlockPoolTest, AcquireUpToBlocksReturnsAvailableCompatiblePlacements) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/0, /*packing=*/2, /*max_num=*/3);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 4);
}

TEST(BlockPoolTest, AcquireUpToBlocksWithPackingOneReturnsPartialCapacity) {
    BlockPool pool(2);
    CacheBlockRef occupied = pool.AcquireBlock(/*group_id=*/0, /*packing=*/1);
    ASSERT_TRUE(occupied);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/1, /*packing=*/1, /*max_num=*/2);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.BoundGroup(partial.front()->Location().lcm_block_id), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 2);
}

TEST(BlockPoolTest, ExactAcquireBlocksRemainsAllOrNothing) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/2).empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 3);
}

TEST(BlockPoolOccupancyIndexTest, PlacementFillsTheDensestParentThenTheLowestId) {
    constexpr std::int32_t kPacking = 4;
    BlockPool pool(3);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/0, kPacking, /*num=*/3 * kPacking);
    ASSERT_EQ(blocks.size(), 12u);

    // Parent 1 keeps one hole, parent 3 keeps two, parent 2 stays full.
    blocks[3].reset();
    blocks[10].reset();
    blocks[11].reset();

    std::vector<CacheBlockRef> refilled = pool.AcquireBlocks(/*group_id=*/0, kPacking, /*num=*/3);
    ASSERT_EQ(refilled.size(), 3u);
    EXPECT_EQ(LocationsOf(refilled), (std::vector<CacheBlockLocation>{{.lcm_block_id = 1, .slot_index = 3},
                                                                      {.lcm_block_id = 3, .slot_index = 2},
                                                                      {.lcm_block_id = 3, .slot_index = 3}}));
}

TEST(BlockPoolOccupancyIndexTest, AgreesWithAFullScanUnderRandomChurn) {
    constexpr std::int32_t kPoolSize = 40;
    constexpr std::array<std::int32_t, 3> kPackings{1, 3, 4};
    BlockPool pool(kPoolSize);
    std::vector<CacheBlockRef> held;
    std::mt19937 rng(20260912);

    for (int step = 0; step < 4000; ++step) {
        const std::uint32_t group_id = std::uniform_int_distribution<std::uint32_t>(0, 2)(rng);
        const std::int32_t packing = kPackings[group_id];
        if (held.empty() || std::uniform_int_distribution<int>(0, 2)(rng) != 0) {
            const std::int32_t demand = std::uniform_int_distribution<std::int32_t>(1, 6)(rng);
            // The plan must match the order the removed scan produced, for the
            // part that lands in parents this group already owns.
            const std::int32_t bound_free = pool.NumFreeSlotsInGroup(group_id);
            const std::size_t in_bound_parents =
                std::min(static_cast<std::size_t>(demand), static_cast<std::size_t>(bound_free));
            const std::vector<CacheBlockLocation> expected =
                PlanInBoundParentsByScan(pool, group_id, packing, in_bound_parents);

            std::vector<CacheBlockRef> acquired = pool.AcquireUpToBlocks(group_id, packing, demand);
            std::vector<CacheBlockLocation> actual = LocationsOf(acquired);
            actual.resize(std::min(actual.size(), in_bound_parents));
            EXPECT_EQ(actual, expected) << "step " << step;
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
            ASSERT_EQ(pool.NumFreeSlotsInGroup(id), FreeSlotsInGroupByScan(pool, id, kPackings[id]))
                << "step " << step << " group " << id;
        }
    }
    held.clear();
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), kPoolSize);
    for (std::uint32_t id = 0; id < kPackings.size(); ++id) {
        EXPECT_EQ(pool.NumFreeSlotsInGroup(id), 0);
    }
}

TEST(BlockPoolOccupancyIndexTest, GroupMayChangePackingOnceItHoldsNoParent) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> wide = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/4, /*num=*/2);
    ASSERT_EQ(wide.size(), 2u);
    EXPECT_EQ(pool.NumFreeSlotsInGroup(0), 2);
    wide.clear();

    std::vector<CacheBlockRef> narrow = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/1);
    ASSERT_EQ(narrow.size(), 1u);
    EXPECT_EQ(pool.NumFreeSlotsInGroup(0), 1);
    narrow.clear();
}

TEST(BlockPoolOccupancyIndexTest, FreeSlotsAreScopedToTheOwningGroup) {
    BlockPool pool(4);
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/1, /*packing=*/4, /*num=*/5);
    ASSERT_EQ(first.size(), 5u);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/2, /*packing=*/2);
    ASSERT_TRUE(second);

    // Group 1 owns two parents holding five of eight slots; group 2 owns one
    // parent holding one of two. The remaining empty parent belongs to neither.
    EXPECT_EQ(pool.NumFreeSlotsInGroup(1), 3);
    EXPECT_EQ(pool.NumFreeSlotsInGroup(2), 1);
    EXPECT_EQ(pool.NumFreeSlotsInGroup(0), 0);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);

    first.clear();
    second.reset();
}

}  // namespace
}  // namespace tokenspeed::test
