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

// GroupGeometry's token -> block arithmetic for a demand whose extent ends
// within one block of INT32_MAX: rounded up to whole blocks, such an extent
// spans 2^31 tokens of capacity, one past the int32 range.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cache/core/block_table.h"
#include "cache/core/cache_types.h"
#include "cache/coordinator/group_geometry.h"

namespace tokenspeed::test {
namespace {

constexpr std::int32_t kInt32Max = std::numeric_limits<std::int32_t>::max();
constexpr std::int32_t kBlock = 1024;
// 2^31 / 1024: the blocks that hold an extent of INT32_MAX - 1000 tokens.
constexpr std::int32_t kBlocksToTheLimit = 2097152;

TEST(GroupGeometryTest, RoundsAnExtentNearInt32MaxUpToWholeBlocks) {
    const GroupGeometry geometry{kBlock};
    EXPECT_EQ(geometry.BlocksNeededBeyond(0, kInt32Max), kBlocksToTheLimit);
    // 2^31 - 1023 tokens still reach into the last block; 2^31 - 1024 fill
    // the one before it exactly.
    EXPECT_EQ(geometry.BlocksNeededBeyond(0, kInt32Max - kBlock + 2), kBlocksToTheLimit);
    EXPECT_EQ(geometry.BlocksNeededBeyond(0, kInt32Max - kBlock + 1), kBlocksToTheLimit - 1);
    EXPECT_EQ(geometry.BlocksNeededBeyond(7, kInt32Max), kBlocksToTheLimit);
}

TEST(GroupGeometryTest, PlansADenseExtentNearInt32Max) {
    const GroupGeometry geometry{kBlock};
    const BlockTable table;
    const AcquirePlan plan = geometry.PlanAcquire(table, kInt32Max - 1000, /*reserve_tokens=*/0);
    EXPECT_EQ(plan.num_blocks, kBlocksToTheLimit);
    EXPECT_EQ(plan.table_blocks_after, kBlocksToTheLimit);
    EXPECT_EQ(plan.available_tokens_after, 1001) << "2^31 tokens of capacity less the extent";

    const AcquirePlan reserved = geometry.PlanAcquire(table, kInt32Max - 1000, /*reserve_tokens=*/500);
    EXPECT_EQ(reserved.num_blocks, kBlocksToTheLimit);
    EXPECT_EQ(reserved.available_tokens_after, 1001) << "the reserve stays inside the last block";
}

TEST(GroupGeometryTest, PlansASparseSuffixNearInt32Max) {
    const GroupGeometry geometry{kBlock};
    const BlockTable table;
    const GroupDemand demand{
        .table = nullptr,
        .extent = SparseSuffix{.extent_tokens = kInt32Max - 1000, .first_block = kBlocksToTheLimit - 2},
    };
    const AcquirePlan plan = geometry.PlanAcquire(table, demand);
    EXPECT_EQ(plan.num_blocks, 2);
    EXPECT_EQ(plan.suffix_start, kBlocksToTheLimit - 2);
    EXPECT_EQ(plan.table_blocks_after, kBlocksToTheLimit);
    EXPECT_EQ(plan.available_tokens_after, 1001);
}

TEST(GroupGeometryTest, RefusesADenseDemandPastInt32Positions) {
    // The request token limit keeps every demand inside the range; one past
    // it is a broken invariant, not a wrapped block count.
    const GroupGeometry geometry{kBlock};
    const BlockTable table;
    EXPECT_THROW((void)geometry.PlanAcquire(table, kInt32Max - 10, /*reserve_tokens=*/11), std::runtime_error);
    EXPECT_THROW(
        (void)geometry.BlocksNeededFor(table, GroupDemand{.extent = DenseGrowth{kInt32Max - 10}, .reserve_tokens = 11}),
        std::runtime_error);
}

}  // namespace
}  // namespace tokenspeed::test
