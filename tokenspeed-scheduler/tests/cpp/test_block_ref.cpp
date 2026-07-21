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

static_assert(sizeof(BlockRef) == sizeof(void*));
static_assert(std::is_copy_constructible_v<BlockRef>);
static_assert(std::is_copy_assignable_v<BlockRef>);

TEST(BlockRefTest, AcquireReturnsUniqueOwningHandle) {
    BlockPool pool(/*total_num_blocks=*/4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    BlockRef ref = pool.AcquireBlock();

    ASSERT_NE(ref.get(), nullptr);
    EXPECT_FALSE(ref->IsNull());
    EXPECT_EQ(ref.use_count(), 1);
    EXPECT_TRUE(ref.unique());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, CopySharesControlAndLastOwnerReturnsBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef first = pool.AcquireBlock();
    CacheBlock* raw = first.get();

    {
        BlockRef second = first;
        EXPECT_EQ(second.get(), raw);
        EXPECT_EQ(first.use_count(), 2);
        EXPECT_EQ(second.use_count(), 2);
        EXPECT_FALSE(first.unique());

        first.reset();
        EXPECT_EQ(first.get(), nullptr);
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
    CacheBlock* first_raw = first.get();

    second = first;

    EXPECT_EQ(second.get(), first_raw);
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, MoveTransfersWithoutChangingCount) {
    BlockPool pool(4);
    BlockRef source = pool.AcquireBlock();
    CacheBlock* raw = source.get();

    BlockRef target = std::move(source);

    EXPECT_EQ(source.get(), nullptr);
    EXPECT_EQ(target.get(), raw);
    EXPECT_EQ(target.use_count(), 1);
}

TEST(BlockRefTest, MoveAssignmentReleasesPreviousBlock) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    BlockRef holder = pool.AcquireBlock();
    BlockRef incoming = pool.AcquireBlock();
    CacheBlock* incoming_raw = incoming.get();

    holder = std::move(incoming);

    EXPECT_EQ(holder.get(), incoming_raw);
    EXPECT_EQ(incoming.get(), nullptr);
    EXPECT_EQ(holder.use_count(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
}

TEST(BlockRefTest, NullBlockRefIsTruthyButUncounted) {
    BlockPool pool(4);
    const std::int32_t free_before = pool.NumFreeBlocks();

    BlockRef first = pool.NullBlockRef();
    BlockRef second = first;

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_TRUE(first->IsNull());
    EXPECT_EQ(first.use_count(), 0);
    EXPECT_FALSE(first.unique());
    first.reset();
    second.reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
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
