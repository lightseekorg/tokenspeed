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
#include "cache/block_ref.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasGet = requires(const T& value) { value.get(); };

static_assert(!HasGet<BlockRef>);
static_assert(sizeof(BlockRef) == sizeof(void*));
static_assert(!std::is_constructible_v<BlockRef, internal_block_ref::BlockControl&>);
static_assert(!std::is_copy_constructible_v<internal_block_ref::BlockControl>);
static_assert(!std::is_move_constructible_v<internal_block_ref::BlockControl>);
static_assert(std::is_copy_constructible_v<BlockRef>);
static_assert(std::is_copy_assignable_v<BlockRef>);
static_assert(std::is_nothrow_move_constructible_v<BlockRef>);
static_assert(std::is_same_v<decltype(std::declval<const BlockRef&>().operator->()), const CacheBlock*>);

TEST(BlockRefTest, AcquireReturnsUniqueOwningHandle) {
    BlockPool pool(/*total_num_blocks=*/4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    BlockRef ref = pool.AcquireBlock();

    ASSERT_TRUE(ref);
    EXPECT_TRUE(ref);
    EXPECT_EQ(ref.use_count(), 1);
    EXPECT_TRUE(ref.unique());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, CopySharesControlAndLastOwnerReturnsBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef first = pool.AcquireBlock();
    const std::int32_t block_id = first->BlockId();

    {
        BlockRef second = first;
        EXPECT_EQ(second, first);
        EXPECT_EQ(second->BlockId(), block_id);
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

TEST(BlockRefTest, CopyAssignmentReleasesPreviousBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef first = pool.AcquireBlock();
    BlockRef second = pool.AcquireBlock();
    const std::int32_t first_id = first->BlockId();

    second = first;

    EXPECT_EQ(second, first);
    EXPECT_EQ(second->BlockId(), first_id);
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, MoveTransfersWithoutChangingCount) {
    BlockPool pool(4);
    BlockRef source = pool.AcquireBlock();
    const std::int32_t block_id = source->BlockId();

    BlockRef target = std::move(source);

    EXPECT_FALSE(source);
    EXPECT_EQ(target->BlockId(), block_id);
    EXPECT_EQ(target.use_count(), 1);
}

TEST(BlockRefTest, MoveAssignmentReleasesPreviousBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef holder = pool.AcquireBlock();
    BlockRef incoming = pool.AcquireBlock();
    const std::int32_t incoming_id = incoming->BlockId();

    holder = std::move(incoming);

    EXPECT_EQ(holder->BlockId(), incoming_id);
    EXPECT_FALSE(incoming);
    EXPECT_EQ(holder.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, EmptyRefHasSharedPtrNullSemantics) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    BlockRef first;
    BlockRef second = first;

    EXPECT_FALSE(first);
    EXPECT_FALSE(second);
    EXPECT_EQ(first.use_count(), 0);
    EXPECT_FALSE(first.unique());
    first.reset();
    second.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(BlockRefTest, SwapExchangesOwnershipWithoutChangingCounts) {
    BlockPool pool(4);
    BlockRef first = pool.AcquireBlock();
    BlockRef second = pool.AcquireBlock();
    const std::int32_t first_id = first->BlockId();
    const std::int32_t second_id = second->BlockId();

    swap(first, second);

    EXPECT_EQ(first->BlockId(), second_id);
    EXPECT_EQ(second->BlockId(), first_id);
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(second.use_count(), 1);
}

TEST(BlockRefTest, SelfAssignmentKeepsOwnership) {
    BlockPool pool(4);
    BlockRef ref = pool.AcquireBlock();
    const std::int32_t block_id = ref->BlockId();

    ref = ref;
    ref = std::move(ref);

    EXPECT_EQ(ref->BlockId(), block_id);
    EXPECT_EQ(ref.use_count(), 1);
}

TEST(BlockRefTest, VectorCopiesKeepBlockPinnedUntilLastCopyDies) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef original = pool.AcquireBlock();
    std::vector<BlockRef> refs(8, original);
    EXPECT_EQ(original.use_count(), 9);

    original.reset();
    EXPECT_EQ(refs.front().use_count(), 8);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);

    refs.clear();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

}  // namespace
}  // namespace tokenspeed::test
