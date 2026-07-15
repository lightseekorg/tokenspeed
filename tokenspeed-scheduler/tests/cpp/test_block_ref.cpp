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

#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"

namespace tokenspeed::test {
namespace {

TEST(BlockRefTest, AdoptTakesOverExistingRefWithoutTouching) {
    BlockPool pool(/*total_num_blocks=*/4);
    const std::int32_t free_before = pool.NumFreeBlocks();
    std::vector<CacheBlock*> allocated = pool.AllocateBlocks(1);
    ASSERT_EQ(allocated.size(), 1u);
    CacheBlock* raw = allocated.front();
    EXPECT_EQ(raw->RefCount(), 1);
    {
        BlockRef ref = BlockRef::Adopt(pool, raw);
        EXPECT_EQ(ref.Get(), raw);
        EXPECT_EQ(raw->RefCount(), 1) << "Adopt must not add a ref";
        EXPECT_EQ(pool.NumFreeBlocks(), free_before - 1);
    }
    // Stray destruction released the adopted ref back to the pool.
    EXPECT_EQ(raw->RefCount(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(BlockRefTest, ShareClaimsAdditionalRef) {
    BlockPool pool(4);
    std::vector<CacheBlock*> allocated = pool.AllocateBlocks(1);
    CacheBlock* raw = allocated.front();
    ASSERT_EQ(raw->RefCount(), 1);
    {
        BlockRef ref = BlockRef::Share(pool, raw);
        EXPECT_EQ(raw->RefCount(), 2) << "Share must TouchBlock";
    }
    EXPECT_EQ(raw->RefCount(), 1) << "destroying the share drops only its own ref";
    pool.FreeBlocks({raw});
}

TEST(BlockRefTest, MoveTransfersOwnershipAndEmptiesSource) {
    BlockPool pool(4);
    CacheBlock* raw = pool.AllocateBlocks(1).front();
    BlockRef a = BlockRef::Adopt(pool, raw);
    BlockRef b = std::move(a);
    EXPECT_EQ(a.Get(), nullptr) << "moved-from ref must be empty";
    EXPECT_EQ(b.Get(), raw);
    EXPECT_EQ(raw->RefCount(), 1) << "moving must not change ref counts";
    a = BlockRef{};  // assigning over an empty ref frees nothing
    EXPECT_EQ(raw->RefCount(), 1);
}

TEST(BlockRefTest, MoveAssignReleasesCurrentRefFirst) {
    BlockPool pool(4);
    std::vector<CacheBlock*> allocated = pool.AllocateBlocks(2);
    CacheBlock* first = allocated[0];
    CacheBlock* second = allocated[1];
    BlockRef holder = BlockRef::Adopt(pool, first);
    BlockRef incoming = BlockRef::Adopt(pool, second);
    const std::int32_t free_before = pool.NumFreeBlocks();
    holder = std::move(incoming);
    EXPECT_EQ(first->RefCount(), 0) << "assignment must release the previously held block";
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 1);
    EXPECT_EQ(holder.Get(), second);
    EXPECT_EQ(incoming.Get(), nullptr);
}

TEST(BlockRefTest, ReleaseSurrendersWithoutFreeing) {
    BlockPool pool(4);
    CacheBlock* raw = pool.AllocateBlocks(1).front();
    const std::int32_t free_held = pool.NumFreeBlocks();
    CacheBlock* surrendered = nullptr;
    {
        BlockRef ref = BlockRef::Adopt(pool, raw);
        surrendered = ref.Release();
        EXPECT_EQ(ref.Get(), nullptr);
    }
    // The destructor ran on an empty ref: the block is still held by the caller.
    EXPECT_EQ(surrendered, raw);
    EXPECT_EQ(raw->RefCount(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_held);
    pool.FreeBlocks({surrendered});
    EXPECT_EQ(pool.NumFreeBlocks(), free_held + 1);
}

TEST(BlockRefTest, NullBlockIsHeldButNeverCounted) {
    BlockPool pool(4);
    CacheBlock* null_block = pool.NullBlock();
    const std::int32_t free_before = pool.NumFreeBlocks();
    {
        BlockRef ref = BlockRef::Share(pool, null_block);
        EXPECT_EQ(ref.Get(), null_block);
        EXPECT_EQ(null_block->RefCount(), 0) << "Share must not count the null block";
        EXPECT_EQ(pool.NumFreeBlocks(), free_before);
    }
    EXPECT_EQ(null_block->RefCount(), 0) << "destruction must not count the null block either";
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(BlockRefTest, VectorReallocationPreservesRefs) {
    BlockPool pool(16);
    std::vector<CacheBlock*> allocated = pool.AllocateBlocks(8);
    std::vector<BlockRef> refs;  // no reserve: force reallocation moves
    for (CacheBlock* raw : allocated) {
        refs.push_back(BlockRef::Adopt(pool, raw));
    }
    for (std::size_t i = 0; i < allocated.size(); ++i) {
        EXPECT_EQ(refs[i].Get(), allocated[i]);
        EXPECT_EQ(allocated[i]->RefCount(), 1) << "reallocation moves must not touch ref counts";
    }
    const std::int32_t free_before = pool.NumFreeBlocks();
    refs.clear();
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 8);
}

}  // namespace
}  // namespace tokenspeed::test
