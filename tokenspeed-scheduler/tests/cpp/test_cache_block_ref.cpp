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

#include <type_traits>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_block_ref.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasGet = requires(const T& value) { value.get(); };

static_assert(!HasGet<CacheBlockRef>);
static_assert(sizeof(CacheBlockRef) == sizeof(void*));
static_assert(!std::is_constructible_v<CacheBlockRef, internal_cache_block_ref::CacheBlockControl&>);
static_assert(!std::is_copy_constructible_v<internal_cache_block_ref::CacheBlockControl>);
static_assert(!std::is_move_constructible_v<internal_cache_block_ref::CacheBlockControl>);
static_assert(std::is_copy_constructible_v<CacheBlockRef>);
static_assert(std::is_copy_assignable_v<CacheBlockRef>);
static_assert(std::is_nothrow_move_constructible_v<CacheBlockRef>);
static_assert(std::is_same_v<decltype(std::declval<const CacheBlockRef&>().operator->()), const CacheBlock*>);

TEST(CacheBlockRefTest, AcquireReturnsUniqueOwningHandle) {
    BlockPool pool(/*num_lcm_blocks=*/4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    CacheBlockRef ref = pool.AcquireBlock();

    ASSERT_TRUE(ref);
    EXPECT_TRUE(ref);
    EXPECT_EQ(ref.use_count(), 1);
    EXPECT_TRUE(ref.unique());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(CacheBlockRefTest, CopySharesControlAndLastOwnerReturnsBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    CacheBlockRef first = pool.AcquireBlock();
    const std::int32_t block_id = first->Location().lcm_block_id;

    {
        CacheBlockRef second = first;
        EXPECT_EQ(second, first);
        EXPECT_EQ(second->Location().lcm_block_id, block_id);
        EXPECT_EQ(first.use_count(), 2);
        EXPECT_EQ(second.use_count(), 2);
        EXPECT_FALSE(first.unique());

        first.reset();
        EXPECT_FALSE(first);
        EXPECT_EQ(second.use_count(), 1);
        EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
    }

    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CacheBlockRefTest, CopyAssignmentReleasesPreviousBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    CacheBlockRef first = pool.AcquireBlock();
    CacheBlockRef second = pool.AcquireBlock();
    const std::int32_t first_id = first->Location().lcm_block_id;

    second = first;

    EXPECT_EQ(second, first);
    EXPECT_EQ(second->Location().lcm_block_id, first_id);
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(CacheBlockRefTest, MoveTransfersWithoutChangingCount) {
    BlockPool pool(4);
    CacheBlockRef source = pool.AcquireBlock();
    const std::int32_t block_id = source->Location().lcm_block_id;

    CacheBlockRef target = std::move(source);

    EXPECT_FALSE(source);
    EXPECT_EQ(target->Location().lcm_block_id, block_id);
    EXPECT_EQ(target.use_count(), 1);
}

TEST(CacheBlockRefTest, MoveAssignmentReleasesPreviousBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    CacheBlockRef holder = pool.AcquireBlock();
    CacheBlockRef incoming = pool.AcquireBlock();
    const std::int32_t incoming_id = incoming->Location().lcm_block_id;

    holder = std::move(incoming);

    EXPECT_EQ(holder->Location().lcm_block_id, incoming_id);
    EXPECT_FALSE(incoming);
    EXPECT_EQ(holder.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(CacheBlockRefTest, EmptyRefHasSharedPtrNullSemantics) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    CacheBlockRef first;
    CacheBlockRef second = first;

    EXPECT_FALSE(first);
    EXPECT_FALSE(second);
    EXPECT_EQ(first.use_count(), 0);
    EXPECT_FALSE(first.unique());
    first.reset();
    second.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CacheBlockRefTest, SwapExchangesOwnershipWithoutChangingCounts) {
    BlockPool pool(4);
    CacheBlockRef first = pool.AcquireBlock();
    CacheBlockRef second = pool.AcquireBlock();
    const std::int32_t first_id = first->Location().lcm_block_id;
    const std::int32_t second_id = second->Location().lcm_block_id;

    swap(first, second);

    EXPECT_EQ(first->Location().lcm_block_id, second_id);
    EXPECT_EQ(second->Location().lcm_block_id, first_id);
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(second.use_count(), 1);
}

TEST(CacheBlockRefTest, SelfAssignmentKeepsOwnership) {
    BlockPool pool(4);
    CacheBlockRef ref = pool.AcquireBlock();
    const std::int32_t block_id = ref->Location().lcm_block_id;

    ref = ref;
    ref = std::move(ref);

    EXPECT_EQ(ref->Location().lcm_block_id, block_id);
    EXPECT_EQ(ref.use_count(), 1);
}

TEST(CacheBlockRefTest, VectorCopiesKeepBlockPinnedUntilLastCopyDies) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    CacheBlockRef original = pool.AcquireBlock();
    std::vector<CacheBlockRef> refs(8, original);
    EXPECT_EQ(original.use_count(), 9);

    original.reset();
    EXPECT_EQ(refs.front().use_count(), 8);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);

    refs.clear();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CacheBlockRefTest, LastOwnerDestroysDynamicBlockAndReleasesExactSlot) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/4, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef last = first;
    const CacheBlockLocation location = first->Location();

    first.reset();
    EXPECT_TRUE(pool.IsOccupied(location));
    last.reset();
    EXPECT_FALSE(pool.IsOccupied(location));
    EXPECT_EQ(pool.BoundGroup(location.lcm_block_id), std::nullopt);
}

}  // namespace
}  // namespace tokenspeed::test
