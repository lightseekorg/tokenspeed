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

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_ref.h"
#include "cache/cache_types.h"
#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"

namespace tokenspeed::test {
namespace {

std::vector<BlockRef> Adopt(BlockPool& pool, const std::vector<CacheBlock*>& blocks) {
    std::vector<BlockRef> refs;
    refs.reserve(blocks.size());
    for (CacheBlock* block : blocks) {
        refs.push_back(BlockRef::Adopt(pool, block));
    }
    return refs;
}

TEST(BlockTableLogicalRangeTest, DefaultTableKeepsLegacyZeroBasedEmptySemantics) {
    const BlockTable table;

    EXPECT_EQ(table.BaseLogicalPage(), 0);
    EXPECT_EQ(table.LiveSize(), 0);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.LogicalEnd(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_FALSE(table.ContainsLogical(0));
    EXPECT_TRUE(table.Blocks().empty());
}

TEST(BlockTableLogicalRangeTest, InitRangeMapsAbsolutePagesToCompactLocalSlots) {
    BlockPool pool(/*total_num_blocks=*/8);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    std::vector<CacheBlock*> blocks = pool.AllocateBlocks(3);
    ASSERT_EQ(blocks.size(), 3u);

    BlockTable table;
    table.InitRange(/*base_logical_page=*/7, Adopt(pool, blocks));

    EXPECT_EQ(table.BaseLogicalPage(), 7);
    EXPECT_EQ(table.LiveSize(), 3);
    EXPECT_EQ(table.NumBlocks(), 3) << "legacy NumBlocks remains the live vector size";
    EXPECT_EQ(table.LogicalEnd(), 10);
    EXPECT_FALSE(table.ContainsLogical(6));
    EXPECT_TRUE(table.ContainsLogical(7));
    EXPECT_TRUE(table.ContainsLogical(9));
    EXPECT_FALSE(table.ContainsLogical(10));
    EXPECT_EQ(table.ToLocal(7), 0);
    EXPECT_EQ(table.ToLocal(9), 2);
    EXPECT_EQ(table.AtLogical(7).Get(), blocks[0]);
    EXPECT_EQ(table.AtLogical(9).Get(), blocks[2]);

    const BlockTable& const_table = table;
    EXPECT_EQ(const_table.AtLogical(8).Get(), blocks[1]);

    table.Reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, InitRangeRequiresFreshTableAndRepresentableRange) {
    BlockPool pool(/*total_num_blocks=*/4);
    BlockTable table;
    table.InitRange(/*base_logical_page=*/2, Adopt(pool, pool.AllocateBlocks(1)));

    EXPECT_THROW(table.InitRange(/*base_logical_page=*/0, {}), std::runtime_error);
    EXPECT_EQ(table.BaseLogicalPage(), 2);
    EXPECT_EQ(table.LiveSize(), 1);

    BlockTable negative;
    EXPECT_THROW(negative.InitRange(/*base_logical_page=*/-1, {}), std::runtime_error);
    EXPECT_EQ(negative.BaseLogicalPage(), 0);

    BlockTable overflow;
    EXPECT_THROW(overflow.InitRange(std::numeric_limits<std::int32_t>::max(), Adopt(pool, pool.AllocateBlocks(1))),
                 std::runtime_error);
    EXPECT_EQ(overflow.BaseLogicalPage(), 0);
    EXPECT_EQ(overflow.LiveSize(), 0);
}

TEST(BlockTableLogicalRangeTest, CheckedLogicalLookupRejectsPagesOutsideLiveRange) {
    BlockPool pool(/*total_num_blocks=*/4);
    BlockTable table;
    table.InitRange(/*base_logical_page=*/11, Adopt(pool, pool.AllocateBlocks(2)));

    EXPECT_THROW(table.ToLocal(10), std::runtime_error);
    EXPECT_THROW(table.ToLocal(13), std::runtime_error);
    EXPECT_THROW(table.AtLogical(10), std::runtime_error);
    EXPECT_THROW(table.AtLogical(13), std::runtime_error);
}

TEST(BlockTableLogicalRangeTest, DropBeforeReleasesOnlyTheLivePrefixAndAdvancesBase) {
    BlockPool pool(/*total_num_blocks=*/8);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    std::vector<CacheBlock*> blocks = pool.AllocateBlocks(4);
    ASSERT_EQ(blocks.size(), 4u);
    ASSERT_EQ(pool.NumFreeBlocks(), free_baseline - 4);

    BlockTable table;
    table.InitRange(/*base_logical_page=*/10, Adopt(pool, blocks));
    table.DropBefore(/*abs_page=*/12);

    EXPECT_EQ(table.BaseLogicalPage(), 12);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_EQ(table.LogicalEnd(), 14);
    EXPECT_EQ(table.AtLogical(12).Get(), blocks[2]);
    EXPECT_EQ(table.AtLogical(13).Get(), blocks[3]);
    EXPECT_EQ(blocks[0]->RefCount(), 0);
    EXPECT_EQ(blocks[1]->RefCount(), 0);
    EXPECT_EQ(blocks[2]->RefCount(), 1);
    EXPECT_EQ(blocks[3]->RefCount(), 1);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);

    table.DropBefore(/*abs_page=*/12);  // dropping an empty prefix is a no-op
    EXPECT_EQ(table.BaseLogicalPage(), 12);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);

    table.Reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, DropBeforeRejectsBackwardAndPastEndWithoutMutation) {
    BlockPool pool(/*total_num_blocks=*/6);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    std::vector<CacheBlock*> blocks = pool.AllocateBlocks(2);
    BlockTable table;
    table.InitRange(/*base_logical_page=*/5, Adopt(pool, blocks));
    const std::int32_t free_held = pool.NumFreeBlocks();

    EXPECT_THROW(table.DropBefore(/*abs_page=*/4), std::runtime_error);
    EXPECT_THROW(table.DropBefore(/*abs_page=*/8), std::runtime_error);
    EXPECT_EQ(table.BaseLogicalPage(), 5);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.AtLogical(5).Get(), blocks[0]);
    EXPECT_EQ(table.AtLogical(6).Get(), blocks[1]);
    EXPECT_EQ(pool.NumFreeBlocks(), free_held);

    table.Reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, DropAllAndResetClearTailAndRestorePoolBaseline) {
    BlockPool pool(/*total_num_blocks=*/6);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    FullAttnManager manager(/*block_size=*/4);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/5));
    ASSERT_EQ(table.NumBlocks(), 2);
    ASSERT_EQ(table.TailAvailableTokens(), 3);

    table.DropBefore(/*abs_page=*/1);
    ASSERT_EQ(table.BaseLogicalPage(), 1);
    ASSERT_EQ(table.LiveSize(), 1);
    ASSERT_EQ(table.TailAvailableTokens(), 3) << "the retained tail page keeps its availability";

    table.DropBefore(table.LogicalEnd());
    EXPECT_EQ(table.BaseLogicalPage(), 2);
    EXPECT_EQ(table.LiveSize(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0) << "an empty table cannot retain tail capacity";
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);

    table.Reset();
    EXPECT_EQ(table.BaseLogicalPage(), 0);
    EXPECT_EQ(table.LiveSize(), 0);
    EXPECT_EQ(table.LogicalEnd(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, ResetReleasesCompactRangeAndRestoresZeroOrigin) {
    BlockPool pool(/*total_num_blocks=*/6);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    std::vector<CacheBlock*> blocks = pool.AllocateBlocks(3);
    BlockTable table;
    table.InitRange(/*base_logical_page=*/20, Adopt(pool, blocks));

    table.Reset();

    EXPECT_EQ(table.BaseLogicalPage(), 0);
    EXPECT_EQ(table.LiveSize(), 0);
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(table.LogicalEnd(), 0);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    for (CacheBlock* block : blocks) {
        EXPECT_EQ(block->RefCount(), 0);
    }
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, LegacyAbsoluteHolesRemainZeroBasedAndUncompacted) {
    BlockPool pool(/*total_num_blocks=*/5);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    FullAttnManager manager(/*block_size=*/2);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/4));
    ASSERT_EQ(table.NumBlocks(), 2);
    CacheBlock* second = table.Blocks()[1];

    CacheBlock* evicted = table.EvictToNull(/*index=*/0, pool.NullBlock());
    ASSERT_NE(evicted, nullptr);
    pool.FreeBlock(evicted);

    EXPECT_EQ(table.BaseLogicalPage(), 0);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.LogicalEnd(), 2);
    EXPECT_EQ(table.NumBlocks(), 2);
    EXPECT_TRUE(table.Blocks()[0]->IsNull());
    EXPECT_EQ(table.Blocks()[1], second);
    EXPECT_EQ(BlockTablePageIds(table), (std::vector<std::int32_t>{0, second->BlockId()}));

    table.Reset();
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, BoundedSlidingReclaimCompactsDescriptorsAndUsesAbsoluteHashSlots) {
    BlockPool pool(/*total_num_blocks=*/8);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    SwaManager manager(/*block_size=*/2, /*sliding_window=*/4, KvTableLayout::kBoundedWindow);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/6));
    ASSERT_EQ(table.BaseLogicalPage(), 0);
    ASSERT_EQ(table.LogicalEnd(), 3);

    EXPECT_EQ(manager.BlocksReclaimableAt(table, /*num_computed_tokens=*/5, /*count_uncached=*/true), 1);
    manager.ReclaimExpired(pool, table, /*num_computed_tokens=*/5);

    EXPECT_EQ(table.BaseLogicalPage(), 1);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.LogicalEnd(), 3);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);
    for (CacheBlock* block : table.Blocks()) {
        EXPECT_FALSE(block->IsNull()) << "bounded tables must not preserve expired front holes";
    }

    const std::vector<std::string> hashes{"logical-page-1", "logical-page-2"};
    manager.CacheFullBlocks(pool, table, hashes, /*first_slot=*/1);
    EXPECT_TRUE(table.AtLogical(1)->IsCached());
    EXPECT_TRUE(table.AtLogical(2)->IsCached());

    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/2));
    EXPECT_EQ(table.BaseLogicalPage(), 1);
    EXPECT_EQ(table.LiveSize(), 3);
    EXPECT_EQ(table.LogicalEnd(), 4);

    manager.Free(pool, table);
    EXPECT_EQ(table.BaseLogicalPage(), 0);
    EXPECT_EQ(table.LiveSize(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, RewindKeepsSharedAcceptedPageAndReusesItsRejectedRows) {
    BlockPool pool(/*total_num_blocks=*/8);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    FullAttnManager manager(/*block_size=*/4);
    BlockTable table;
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/16));
    ASSERT_EQ(table.LiveSize(), 4);

    manager.RewindTail(pool, table, /*accepted_raw_end=*/5, /*retain_raw_end=*/5);

    EXPECT_EQ(table.LogicalEnd(), 2);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.TailAvailableTokens(), 3);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);

    const std::vector<std::int32_t> retained_ids = BlockTablePageIds(table);
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/3));
    EXPECT_EQ(BlockTablePageIds(table), retained_ids);
    EXPECT_EQ(table.TailAvailableTokens(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);

    manager.Free(pool, table);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

TEST(BlockTableLogicalRangeTest, RewindRespectsNonzeroBoundedBase) {
    BlockPool pool(/*total_num_blocks=*/8);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    FullAttnManager manager(/*block_size=*/4);
    BlockTable table;
    table.InitRange(/*base_logical_page=*/10, Adopt(pool, pool.AllocateBlocks(4)));

    manager.RewindTail(pool, table, /*accepted_raw_end=*/45, /*retain_raw_end=*/45);

    EXPECT_EQ(table.BaseLogicalPage(), 10);
    EXPECT_EQ(table.LogicalEnd(), 12);
    EXPECT_EQ(table.LiveSize(), 2);
    EXPECT_EQ(table.TailAvailableTokens(), 3);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline - 2);
    ASSERT_TRUE(manager.Acquire(pool, table, /*num_tokens=*/3));
    EXPECT_EQ(table.LogicalEnd(), 12);

    manager.Free(pool, table);
    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
}

}  // namespace
}  // namespace tokenspeed::test
