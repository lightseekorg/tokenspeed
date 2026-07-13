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
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "cache/block_pool_set.h"

namespace tokenspeed::test {
namespace {

TEST(PoolDemandTest, CheckedArithmeticIsComponentWise) {
    PoolDemand demand{1, 2, 3};
    demand.AddInPlace(PoolDemand{4, 0, 1});
    EXPECT_EQ(demand, (PoolDemand{5, 2, 4}));

    demand.SubtractInPlace(PoolDemand{2, 2, 1});
    EXPECT_EQ(demand, (PoolDemand{3, 0, 3}));
}

TEST(PoolDemandTest, RejectsShapeMismatchAndNegativeResult) {
    PoolDemand demand{1, 2};
    EXPECT_THROW(demand.AddInPlace(PoolDemand{1}), std::invalid_argument);
    EXPECT_THROW(demand.SubtractInPlace(PoolDemand{2, 0}), std::underflow_error);
    EXPECT_EQ(demand, (PoolDemand{1, 2}));
}

TEST(PoolDemandTest, ArithmeticFailureDoesNotPartiallyMutate) {
    PoolDemand demand{7, std::numeric_limits<std::int32_t>::max()};
    EXPECT_THROW(demand.AddInPlace(PoolDemand{1, 1}), std::overflow_error);
    EXPECT_EQ(demand, (PoolDemand{7, std::numeric_limits<std::int32_t>::max()}));
}

TEST(PoolDemandTest, InlineCapacityCoversV4AndPreservesValueSemantics) {
    static_assert(PoolDemand::kInlineCapacity >= 6);
    static_assert(std::is_nothrow_move_constructible_v<PoolDemand>);
    static_assert(std::is_nothrow_move_assignable_v<PoolDemand>);

    PoolDemand original(PoolDemand::kInlineCapacity, 0);
    for (PoolIndex i = 0; i < original.Size(); ++i) {
        original[i] = static_cast<std::int32_t>(i + 1);
    }

    PoolDemand copy = original;
    copy[0] = 99;
    EXPECT_EQ(original[0], 1);
    EXPECT_EQ(copy[0], 99);

    PoolDemand moved = std::move(copy);
    EXPECT_TRUE(copy.Empty());
    EXPECT_EQ(moved.Size(), PoolDemand::kInlineCapacity);
    EXPECT_EQ(moved[0], 99);
    EXPECT_EQ(moved[moved.Size() - 1], static_cast<std::int32_t>(moved.Size()));
}

TEST(PoolDemandTest, HeapFallbackPreservesArithmeticAndStrongFailureGuarantee) {
    const std::size_t size = PoolDemand::kInlineCapacity + 3;
    PoolDemand demand(size, 1);
    PoolDemand delta(size, 2);

    demand.AddInPlace(delta);
    EXPECT_EQ(demand.Size(), size);
    EXPECT_EQ(demand[0], 3);
    EXPECT_EQ(demand[size - 1], 3);

    PoolDemand copy = demand;
    copy.SubtractInPlace(delta);
    EXPECT_EQ(copy[0], 1);
    EXPECT_EQ(demand[0], 3);

    demand[size - 1] = std::numeric_limits<std::int32_t>::max();
    PoolDemand overflow(size, 0);
    overflow[size - 1] = 1;
    EXPECT_THROW(demand.AddInPlace(overflow), std::overflow_error);
    EXPECT_EQ(demand[0], 3);
    EXPECT_EQ(demand[size - 1], std::numeric_limits<std::int32_t>::max());

    PoolDemand moved;
    moved = std::move(copy);
    EXPECT_TRUE(copy.Empty());
    EXPECT_EQ(moved.Size(), size);
    EXPECT_EQ(moved[0], 1);
    EXPECT_EQ(moved[size - 1], 1);
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
}

TEST(BlockPoolSetTest, LocalPageIdsAreIndependentAcrossPools) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 3, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 3, .bytes_per_block = 32},
    });

    auto history = pools.Pool(pools.IndexOf("history")).AllocateBlocks(1);
    auto state = pools.Pool(pools.IndexOf("state")).AllocateBlocks(1);
    ASSERT_EQ(history.size(), 1u);
    ASSERT_EQ(state.size(), 1u);
    EXPECT_EQ(history.front()->BlockId(), 1);
    EXPECT_EQ(state.front()->BlockId(), 1);
    EXPECT_NE(history.front(), state.front());
}

TEST(BlockPoolSetTest, AdmissionChecksEveryPoolWithoutCrossPoolBorrowing) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 5, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 2, .bytes_per_block = 32},
    });

    EXPECT_TRUE(pools.CanSatisfy(PoolDemand{4, 1}));
    EXPECT_FALSE(pools.CanSatisfy(PoolDemand{1, 2}));

    auto state = pools.Pool(pools.IndexOf("state")).AllocateBlocks(1);
    ASSERT_EQ(state.size(), 1u);
    EXPECT_TRUE(pools.CanSatisfy(PoolDemand{4, 0}));
    EXPECT_FALSE(pools.CanSatisfy(PoolDemand{0, 1}));
}

TEST(BlockPoolSetTest, ExposesPerPoolCountsAndByteDemand) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 5, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 3, .bytes_per_block = 32},
    });

    EXPECT_EQ(pools.TotalUsableBlocks(), (PoolDemand{4, 2}));
    EXPECT_EQ(pools.FreeBlocks(), (PoolDemand{4, 2}));
    EXPECT_EQ(pools.BytesFor(PoolDemand{2, 1}), 288);

    BlockPool& history_pool = pools.Pool(pools.IndexOf("history"));
    auto history = history_pool.AllocateBlocks(1);
    ASSERT_EQ(history.size(), 1u);
    history_pool.CacheFullBlock(history.front(), "history-key");
    auto snapshots = pools.Snapshot();
    ASSERT_EQ(snapshots.size(), 2u);
    EXPECT_EQ(snapshots[0].pool_id, "history");
    EXPECT_EQ(snapshots[0].active_blocks, 1);
    EXPECT_EQ(snapshots[0].free_blocks, 3);
    EXPECT_EQ(snapshots[0].cached_evictable_blocks, 0);
    EXPECT_EQ(snapshots[0].pinned_cached_blocks, 1);
    EXPECT_EQ(snapshots[1].pool_id, "state");
    EXPECT_EQ(snapshots[1].active_blocks, 0);

    history_pool.FreeBlocks(history);
    snapshots = pools.Snapshot();
    EXPECT_EQ(snapshots[0].active_blocks, 0);
    EXPECT_EQ(snapshots[0].free_blocks, 4);
    EXPECT_EQ(snapshots[0].cached_evictable_blocks, 1);
    EXPECT_EQ(snapshots[0].pinned_cached_blocks, 0);
}

TEST(BlockPoolSetTest, RejectsByteDemandOverflow) {
    BlockPoolSet pools({FlatBlockPoolConfig{
        .pool_id = "huge",
        .total_blocks = 3,
        .bytes_per_block = std::numeric_limits<std::int64_t>::max() / 2 + 1,
    }});
    EXPECT_THROW(pools.BytesFor(PoolDemand{2}), std::overflow_error);
}

TEST(BlockPoolSetTest, RejectsInvalidOrDuplicateConfiguration) {
    EXPECT_THROW(BlockPoolSet({}), std::invalid_argument);
    EXPECT_THROW(BlockPoolSet({FlatBlockPoolConfig{.pool_id = "", .total_blocks = 2, .bytes_per_block = 1}}),
                 std::invalid_argument);
    EXPECT_THROW(BlockPoolSet({FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 1, .bytes_per_block = 1}}),
                 std::invalid_argument);
    EXPECT_THROW(BlockPoolSet({
                     FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 2, .bytes_per_block = 1},
                     FlatBlockPoolConfig{.pool_id = "p", .total_blocks = 3, .bytes_per_block = 1},
                 }),
                 std::invalid_argument);
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

TEST(BlockPoolSetThreadDomainTest, SharesOneDomainAndKeepsImmutableSchemaCrossThreadReadable) {
    SchedulerThreadMutationDomain domain;
    BlockPoolSet pools(
        {
            FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 5, .bytes_per_block = 128},
            FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 3, .bytes_per_block = 32},
        },
        domain);

    EXPECT_EQ(&pools.MutationDomain(), &domain);
    EXPECT_EQ(&pools.Pool(0).MutationDomain(), &domain);
    EXPECT_EQ(&pools.Pool(1).MutationDomain(), &domain);

    std::exception_ptr generation_error;
    std::exception_ptr snapshot_error;
    std::exception_ptr schema_error;
    std::size_t schema_size = 0;
    PoolIndex history_index = 0;
    std::string state_pool_id;
    std::int32_t history_total_blocks = 0;
    PoolDemand total_usable;
    std::int64_t bytes = 0;
    std::thread worker([&] {
        try {
            (void)pools.Generation();
        } catch (...) {
            generation_error = std::current_exception();
        }
        try {
            (void)pools.FreeBlocks();
        } catch (...) {
            snapshot_error = std::current_exception();
        }
        try {
            schema_size = pools.Size();
            history_index = pools.IndexOf("history");
            state_pool_id = pools.PoolId(1);
            history_total_blocks = pools.Config(0).total_blocks;
            total_usable = pools.TotalUsableBlocks();
            bytes = pools.BytesFor(PoolDemand{1, 1});
        } catch (...) {
            schema_error = std::current_exception();
        }
    });
    worker.join();

    ASSERT_NE(generation_error, nullptr);
    ASSERT_NE(snapshot_error, nullptr);
    EXPECT_THROW(std::rethrow_exception(generation_error), std::logic_error);
    EXPECT_THROW(std::rethrow_exception(snapshot_error), std::logic_error);
    EXPECT_EQ(schema_error, nullptr);
    EXPECT_EQ(schema_size, 2u);
    EXPECT_EQ(history_index, 0u);
    EXPECT_EQ(state_pool_id, "state");
    EXPECT_EQ(history_total_blocks, 5);
    EXPECT_EQ(total_usable, (PoolDemand{4, 2}));
    EXPECT_EQ(bytes, 160);
}

}  // namespace
}  // namespace tokenspeed::test
