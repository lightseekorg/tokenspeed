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

// Live-tail allocation reference geometry: Inkling hiddenconv (block 8, window 12, alignment 256) -> 2 of every 32
// slots stay real once slid out.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_types.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/swa_manager.h"

namespace tokenspeed::test {
namespace {

std::int32_t CountReal(const BlockTable& table) {
    std::int32_t real = 0;
    for (CacheBlock* b : table.Blocks()) {
        real += b->IsNull() ? 0 : 1;
    }
    return real;
}

std::int32_t HighestRealBelow(const BlockTable& table, std::int32_t bound_slots) {
    std::int32_t highest = -1;
    for (std::int32_t s = 0; s < std::min(bound_slots, table.NumBlocks()); ++s) {
        if (!table.Blocks()[s]->IsNull()) {
            highest = s;
        }
    }
    return highest;
}

TEST(SwaLiveAllocTest, FirstChunkAllocatesTailAndCheckpointsOnly) {
    // 8192-token chunk: slid bound (8192-12+1)/8=1022; trailing {1022,1023} + s%32 in {30,31} -> 64 real pages.
    BlockPool pool(2048);
    SwaManager mgr(/*block_size=*/8, /*sliding_window=*/12, /*live_alloc_alignment=*/256);
    BlockTable table;

    const std::int32_t before = pool.NumFreeBlocks();
    EXPECT_EQ(mgr.BlocksNeededFor(table, 8192), 64);
    ASSERT_TRUE(mgr.Acquire(pool, table, 8192));
    EXPECT_EQ(table.NumBlocks(), 1024);
    EXPECT_EQ(CountReal(table), 64);
    EXPECT_EQ(before - pool.NumFreeBlocks(), 64);

    for (std::int32_t s = 0; s < table.NumBlocks(); ++s) {
        const bool trailing = s >= 1022;
        const bool checkpoint = s % 32 >= 30;
        EXPECT_EQ(!table.Blocks()[s]->IsNull(), trailing || checkpoint) << "slot " << s;
    }
}

TEST(SwaLiveAllocTest, BlocksNeededForMirrorsAcquireAcrossUnalignedChunks) {
    BlockPool pool(4096);
    SwaManager mgr(8, 12, 256);
    BlockTable table;

    // Odd chunk sizes force tail_avail continuation and mid-page frontiers.
    for (std::int32_t chunk : {777, 1, 7, 8192, 640, 3, 2048, 12345}) {
        const std::int32_t predicted = mgr.BlocksNeededFor(table, chunk);
        const std::int32_t before = pool.NumFreeBlocks();
        ASSERT_TRUE(mgr.Acquire(pool, table, chunk)) << "chunk " << chunk;
        EXPECT_EQ(before - pool.NumFreeBlocks(), predicted) << "chunk " << chunk;
    }
}

TEST(SwaLiveAllocTest, ZeroAlignmentMatchesBaseAllocation) {
    BlockPool pool_live(4096);
    BlockPool pool_base(4096);
    SwaManager live(8, 12, /*live_alloc_alignment=*/0);
    SwaManager base(8, 12);
    BlockTable t_live;
    BlockTable t_base;

    for (std::int32_t chunk : {777, 8192, 3}) {
        ASSERT_TRUE(live.Acquire(pool_live, t_live, chunk));
        ASSERT_TRUE(base.Acquire(pool_base, t_base, chunk));
        EXPECT_EQ(t_live.NumBlocks(), t_base.NumBlocks());
        EXPECT_EQ(CountReal(t_live), CountReal(t_base));
        EXPECT_EQ(pool_live.NumFreeBlocks(), pool_base.NumFreeBlocks());
    }
    EXPECT_EQ(CountReal(t_live), t_live.NumBlocks());  // no holes without live mode
}

TEST(SwaLiveAllocTest, WindowCoveringChunkAllocatesEverything) {
    // Window larger than the total extent: nothing slid out, no holes.
    BlockPool pool(64);
    SwaManager mgr(8, /*sliding_window=*/512, /*live_alloc_alignment=*/256);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 256));
    EXPECT_EQ(CountReal(table), table.NumBlocks());
}

TEST(SwaLiveAllocTest, ResumeSpanCoveringAlignmentDegeneratesToFull) {
    // kvconv geometry: block 128, window 132 -> every slot is a resume page.
    BlockPool pool(256);
    SwaManager mgr(128, 132, 256);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8192));
    EXPECT_EQ(CountReal(table), table.NumBlocks());
}

TEST(SwaLiveAllocTest, AcquireShortfallLeavesTableUntouched) {
    BlockPool pool(16);  // far fewer than the 64 live pages an 8192 chunk needs
    SwaManager mgr(8, 12, 256);
    BlockTable table;
    const std::int32_t before = pool.NumFreeBlocks();
    EXPECT_FALSE(mgr.Acquire(pool, table, 8192));
    EXPECT_EQ(table.NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), before);
}

TEST(SwaLiveAllocTest, ReclaimPunchesRealPagesBelowInterleavedHoles) {
    // Chunk-1 checkpoint pages sit below interleaved holes: ReclaimExpired must not early-break and must free every
    // real page under the bound.
    BlockPool pool(2048);
    SwaManager mgr(8, 12, 256);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 4096));
    ASSERT_TRUE(mgr.Acquire(pool, table, 4096));
    const std::int32_t real_before = CountReal(table);
    ASSERT_GT(real_before, 2);

    const std::int32_t free_before = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, /*num_computed_tokens=*/8192);
    // Slid bound (8192-12+1)/8=1022 -> everything below is punched, trailing 2 stay.
    EXPECT_EQ(HighestRealBelow(table, 1022), -1);
    EXPECT_EQ(CountReal(table), 2);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + (real_before - 2));
}

TEST(SwaLiveAllocTest, ReclaimableCountsRealPagesBelowInterleavedHoles) {
    BlockPool pool(2048);
    SwaManager mgr(8, 12, 256);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8192));
    const std::int32_t real = CountReal(table);
    // All but the 2 trailing-live pages are reclaimable at the frontier.
    EXPECT_EQ(mgr.BlocksReclaimableAt(table, 8192, /*count_uncached=*/true), real - 2);
}

TEST(SwaLiveAllocTest, CheckpointPagesLandExactlyBeforeAlignedBoundaries) {
    // Checkpoint pages for an aligned boundary must be real while being written, even past the frontier.
    BlockPool pool(2048);
    SwaManager mgr(8, 12, 256);
    BlockTable table;
    // 300 tokens: slots 0..37, slid bound (300-12+1)/8=36; live = trailing {36,37} + checkpoint {30,31}.
    ASSERT_TRUE(mgr.Acquire(pool, table, 300));
    ASSERT_EQ(table.NumBlocks(), 38);
    for (std::int32_t s = 0; s < 38; ++s) {
        const bool live = s >= 36 || (s % 32 >= 30);
        EXPECT_EQ(!table.Blocks()[s]->IsNull(), live) << "slot " << s;
    }
}

TEST(SwaLiveAllocTest, SequentialChargeCoversPeakAcquireDemand) {
    // BlocksNeededFor is non-monotonic (frontier 19 holes slot 0): the combined query
    // under-charges the chunk-then-reserve sequence that admission actually performs.
    BlockPool pool(64);
    SwaManager mgr(8, 12, 256);
    const BlockTable fresh;
    EXPECT_EQ(mgr.BlocksNeededFor(fresh, 17), 3);
    EXPECT_EQ(mgr.BlocksNeededFor(fresh, 19), 2);  // the drop the combined gate would trust
    EXPECT_EQ(mgr.BlocksNeededForSequential(fresh, 17, 2), 3);

    // The sequential charge equals what the two real acquires pin at their peak.
    BlockTable table;
    std::int32_t before = pool.NumFreeBlocks();
    ASSERT_TRUE(mgr.Acquire(pool, table, 17));
    EXPECT_EQ(before - pool.NumFreeBlocks(), 3);
    before = pool.NumFreeBlocks();
    ASSERT_TRUE(mgr.Acquire(pool, table, 2));
    EXPECT_EQ(before - pool.NumFreeBlocks(), 0);

    // A reserve crossing into new pages charges both batches (A=3 at frontier 17, B=1 at 27).
    EXPECT_EQ(mgr.BlocksNeededForSequential(fresh, 17, 10), 4);
    BlockTable table2;
    before = pool.NumFreeBlocks();
    ASSERT_TRUE(mgr.Acquire(pool, table2, 17));
    ASSERT_TRUE(mgr.Acquire(pool, table2, 10));
    EXPECT_EQ(before - pool.NumFreeBlocks(), 4);
}

TEST(SwaLiveAllocTest, SequentialReserveMatchesSecondAcquireAndIsNonNegative) {
    // reserve = Sequential(chunk, extra) - BlocksNeededFor(chunk) must equal the second
    // acquire's real demand (the old combined form went negative, e.g. chunk=17/extra=2).
    for (const auto [chunk, extra] : {std::pair{17, 2}, {18, 2}, {16, 3}, {17, 10}, {24, 2}, {8192, 16}, {3, 4}}) {
        BlockPool pool(2048);
        SwaManager mgr(8, 12, 256);
        const BlockTable fresh;
        const std::int32_t reserve =
            mgr.BlocksNeededForSequential(fresh, chunk, extra) - mgr.BlocksNeededFor(fresh, chunk);
        ASSERT_GE(reserve, 0) << "chunk " << chunk << " extra " << extra;
        BlockTable table;
        ASSERT_TRUE(mgr.Acquire(pool, table, chunk));
        const std::int32_t before = pool.NumFreeBlocks();
        ASSERT_TRUE(mgr.Acquire(pool, table, extra));
        EXPECT_EQ(before - pool.NumFreeBlocks(), reserve) << "chunk " << chunk << " extra " << extra;
    }
}

TEST(SwaLiveAllocTest, ReclaimFloorFreesNewBandOnlyAndStaysExact) {
    // Repeated slides must free exactly the newly-expired real pages; the floor never
    // hides a real page and a same-bound re-reclaim frees nothing.
    BlockPool pool(2048);
    SwaManager mgr(8, 12, 256);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, 8192));
    mgr.ReclaimExpired(pool, table, 8192);  // bound 1022, floor advances
    ASSERT_EQ(CountReal(table), 2);

    // Decode 16 more tokens -> bound moves to 1024; slots 1022/1023 expire, 1024/1025 are the new tail.
    ASSERT_TRUE(mgr.Acquire(pool, table, 16));
    EXPECT_EQ(mgr.BlocksReclaimableAt(table, 8208, /*count_uncached=*/true), 2);
    std::int32_t free_before = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 8208);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 2);
    EXPECT_EQ(HighestRealBelow(table, 1024), -1);
    EXPECT_EQ(CountReal(table), 2);

    // Same bound again: nothing left in the band.
    free_before = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 8208);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
    EXPECT_EQ(mgr.BlocksReclaimableAt(table, 8208, /*count_uncached=*/true), 0);
}

TEST(SwaLiveAllocTest, CoordinatorSequentialSumsPerGroupPeaks) {
    const std::vector<KvCacheSpec> specs = {
        {.kind = AttnKind::kFull, .block_size = 256, .sliding_window = 0},
        {.kind = AttnKind::kSlidingWindow, .block_size = 8, .sliding_window = 12, .live_tail_alloc = true},
    };
    BlockPool pool(4096);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    // full: 1 page either way; conv live: sequential 3 vs combined 2.
    EXPECT_EQ(coord.BlocksNeededFor(19), 1 + 2);
    EXPECT_EQ(coord.BlocksNeededForSequential(17, 2), 1 + 3);
}

TEST(SwaLiveAllocTest, CoordinatorWiresGroupLcmAsAlignment) {
    // Inkling-shaped hybrid: full 256 + SWA 128 (no live) + conv 8 (live), lcm=256.
    const std::vector<KvCacheSpec> specs = {
        {.kind = AttnKind::kFull, .block_size = 256, .sliding_window = 0},
        {.kind = AttnKind::kSlidingWindow, .block_size = 128, .sliding_window = 511},
        {.kind = AttnKind::kSlidingWindow, .block_size = 8, .sliding_window = 12, .live_tail_alloc = true},
    };
    BlockPool pool(4096);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(specs.size());

    // 8192 tokens: full 32 + swa 64 (full alloc) + conv 64 (live) pages.
    EXPECT_EQ(coord.BlocksNeededFor(tables, 8192), 32 + 64 + 64);
    const std::int32_t before = pool.NumFreeBlocks();
    ASSERT_TRUE(coord.Acquire(tables, 8192));
    EXPECT_EQ(before - pool.NumFreeBlocks(), 32 + 64 + 64);
    EXPECT_EQ(CountReal(tables[0]), 32);
    EXPECT_EQ(CountReal(tables[1]), 64);
    EXPECT_EQ(CountReal(tables[2]), 64);
    EXPECT_EQ(tables[2].NumBlocks(), 1024);  // holes keep slot alignment
}

}  // namespace
}  // namespace tokenspeed::test
