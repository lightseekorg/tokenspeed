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

#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

#include <spdlog/sinks/stdout_color_sinks.h>

#include "cache/core/block_pool.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasCacheIndex = requires(T& value) { value.ContainsCachedBlock("key"); };

static_assert(!HasCacheIndex<BlockPool>);

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

}  // namespace
}  // namespace tokenspeed::test
