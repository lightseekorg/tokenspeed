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
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/block_pool_set.h"

namespace tokenspeed::test {
namespace {

enum class InvalidPoolDemandArithmetic { kShapeMismatch, kUnderflow };

TEST(PoolDemandTest, InvalidArithmeticDoesNotPartiallyMutate) {
    PoolDemand valid{1, 2, 3};
    valid.AddInPlace(PoolDemand{4, 0, 1});
    valid.SubtractInPlace(PoolDemand{2, 2, 1});
    EXPECT_EQ(valid, (PoolDemand{3, 0, 3}));

    struct Case {
        const char* name;
        InvalidPoolDemandArithmetic operation;
    };
    const Case cases[] = {
        {"shape mismatch", InvalidPoolDemandArithmetic::kShapeMismatch},
        {"underflow", InvalidPoolDemandArithmetic::kUnderflow},
    };

    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        PoolDemand demand{3, 0, 3};
        const PoolDemand before = demand;
        switch (test_case.operation) {
            case InvalidPoolDemandArithmetic::kShapeMismatch:
                EXPECT_THROW(demand.AddInPlace(PoolDemand{1}), std::invalid_argument);
                break;
            case InvalidPoolDemandArithmetic::kUnderflow:
                EXPECT_THROW(demand.SubtractInPlace(PoolDemand{4, 0, 0}), std::underflow_error);
                break;
        }
        EXPECT_EQ(demand, before);
    }
}

TEST(BlockPoolSetTest, CanonicalizesIdsAndKeepsPoolAddressesStable) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "v4.swa", .total_blocks = 8, .bytes_per_block = 64},
        FlatBlockPoolConfig{.pool_id = "v4.c4.history", .total_blocks = 4, .bytes_per_block = 256},
    });

    ASSERT_EQ(pools.Size(), 2u);
    EXPECT_EQ(pools.PoolId(0), "v4.c4.history");
    EXPECT_EQ(pools.PoolId(1), "v4.swa");

    BlockPool* first = &pools.Pool(pools.IndexOf("v4.c4.history"));
    BlockPool* second = &pools.Pool(pools.IndexOf("v4.swa"));
    EXPECT_NE(first, second);
    EXPECT_EQ(&pools.Pool(0), first);
    EXPECT_EQ(&pools.Pool(1), second);

    auto first_page = first->AllocateBlocks(1);
    auto second_page = second->AllocateBlocks(1);
    ASSERT_EQ(first_page.size(), 1u);
    ASSERT_EQ(second_page.size(), 1u);
    EXPECT_EQ(first_page.front()->BlockId(), 1);
    EXPECT_EQ(second_page.front()->BlockId(), 1);
    EXPECT_NE(first_page.front(), second_page.front());
}

TEST(BlockPoolSetTest, SnapshotReportsIndependentPerPoolCapacityAndUsage) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 5, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 2, .bytes_per_block = 32},
    });

    const std::vector<BlockPoolSnapshot> initial = pools.Snapshot();
    ASSERT_EQ(initial.size(), 2u);
    EXPECT_EQ(initial[0].pool_id, "history");
    EXPECT_EQ(initial[0].usable_blocks, 4);
    EXPECT_EQ(initial[0].free_blocks, 4);
    EXPECT_EQ(initial[1].pool_id, "state");
    EXPECT_EQ(initial[1].usable_blocks, 1);
    EXPECT_EQ(initial[1].free_blocks, 1);

    auto state = pools.Pool(pools.IndexOf("state")).AllocateBlocks(1);
    ASSERT_EQ(state.size(), 1u);
    const std::vector<BlockPoolSnapshot> allocated = pools.Snapshot();
    ASSERT_EQ(allocated.size(), initial.size());
    EXPECT_EQ(allocated[0].free_blocks, initial[0].free_blocks);
    EXPECT_EQ(allocated[1].free_blocks, initial[1].free_blocks - 1);
}

TEST(BlockPoolSetTest, RejectsInvalidOrDuplicateConfiguration) {
    struct Case {
        std::string name;
        std::vector<FlatBlockPoolConfig> configs;
    };
    const std::vector<Case> cases{
        {.name = "empty pool set", .configs = {}},
        {.name = "empty pool id",
         .configs = {FlatBlockPoolConfig{.pool_id = "", .total_blocks = 2, .bytes_per_block = 1}}},
        {.name = "reserved null page only",
         .configs = {FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 1, .bytes_per_block = 1}}},
        {.name = "duplicate pool id",
         .configs =
             {
                 FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 2, .bytes_per_block = 1},
                 FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 3, .bytes_per_block = 1},
             }},
    };

    for (const Case& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        EXPECT_THROW(
            {
                BlockPoolSet invalid(test_case.configs);
                static_cast<void>(invalid);
            },
            std::invalid_argument);
    }
}

TEST(BlockPoolSetTest, QuiescentResetAtomicallyInvalidatesCachedContentAndAdvancesGeneration) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 8},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 3, .bytes_per_block = 16},
    });
    BlockPool& history = pools.Pool(pools.IndexOf("history"));
    BlockPool& state = pools.Pool(pools.IndexOf("state"));
    CacheBlock* history_block = history.AllocateBlock();
    CacheBlock* state_block = state.AllocateBlock();
    ASSERT_NE(history_block, nullptr);
    ASSERT_NE(state_block, nullptr);
    history.CacheFullBlock(history_block, "history-key");
    state.CacheFullBlock(state_block, "state-key");

    EXPECT_FALSE(pools.IsQuiescent());
    EXPECT_THROW(pools.ResetQuiescent(), std::logic_error);
    EXPECT_EQ(pools.Generation(), 0u);
    EXPECT_NE(history.GetCachedBlock("history-key"), nullptr);
    EXPECT_NE(state.GetCachedBlock("state-key"), nullptr);

    history.FreeBlock(history_block);
    state.FreeBlock(state_block);
    ASSERT_TRUE(pools.IsQuiescent());
    EXPECT_EQ(pools.ResetQuiescent(), 1u);
    EXPECT_EQ(pools.Generation(), 1u);
    EXPECT_EQ(history.GetCachedBlock("history-key"), nullptr);
    EXPECT_EQ(state.GetCachedBlock("state-key"), nullptr);
    EXPECT_EQ(history.NumFreeBlocks(), history.TotalBlocks() - 1);
    EXPECT_EQ(state.NumFreeBlocks(), state.TotalBlocks() - 1);
    EXPECT_EQ(pools.ResetQuiescent(), 2u);
}

}  // namespace
}  // namespace tokenspeed::test
