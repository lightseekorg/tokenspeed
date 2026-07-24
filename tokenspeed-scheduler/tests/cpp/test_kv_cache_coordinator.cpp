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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "cache/block_pool.h"
#include "cache/block_pool_set.h"
#include "cache/cache_group.h"
#include "cache/kv_cache_coordinator.h"
#include "cache/cache_types.h"
#include "cache/full_attn_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"

#if TOKENSPEED_FLAT_KVCACHE
#include "fsm/forward_events.h"
#include "resource/allocator/req_pool_allocator.h"
#endif

namespace tokenspeed::test {
namespace {

std::vector<std::int32_t> BlockIds(const std::vector<BlockRef>& refs) {
    std::vector<std::int32_t> ids;
    ids.reserve(refs.size());
    for (const BlockRef& ref : refs) {
        ids.push_back(ref ? ref->BlockId() : 0);
    }
    return ids;
}

using token_span = std::span<const std::int32_t>;

std::vector<std::string> ContentHashes(const std::vector<std::vector<std::int32_t>>& pages) {
    std::vector<token_span> spans;
    spans.reserve(pages.size());
    for (const auto& p : pages) {
        spans.emplace_back(p.data(), p.size());
    }
    return ComputePagedHashes(spans, "");
}

KvCacheCoordinator::AdmissionMatch ProbeAndAcquirePrefix(KvCacheCoordinator& coordinator,
                                                         std::span<const std::string> content_hashes) {
    return coordinator.AcquirePrefix(coordinator.ProbePrefix(content_hashes));
}

// Cache then free, so the block is prefix-hittable via the admission path.
std::int32_t CacheForGroup(BlockPool& pool, const std::string& content_hash, std::uint32_t group_id) {
    std::string key = MakeKeyWithGroupId(content_hash, group_id);
    BlockRef got = pool.AcquireBlock();
    const std::int32_t id = got->BlockId();
    pool.CacheFullBlock(got, key);
    got.reset();
    return id;
}

void CacheKeys(BlockPool& pool, std::span<const std::string> keys) {
    for (const std::string& key : keys) {
        BlockRef block = pool.AcquireBlock();
        pool.CacheFullBlock(block, key);
    }
}

// Asserts no null hole inside the last min(len, pages_needed) blocks.
void ExpectSwaWindowIntact(const PrefixMatch& m, std::int32_t window, std::int32_t block_size) {
    std::int32_t len = static_cast<std::int32_t>(m.blocks.size());
    std::int32_t pages_needed = (window - 1 + block_size - 1) / block_size;
    std::int32_t need = std::min(len, pages_needed);
    for (std::int32_t i = len - need; i < len; ++i) {
        EXPECT_TRUE(m.blocks[static_cast<std::size_t>(i)])
            << "null hole inside the last window at slot " << i << " of " << len;
    }
}

TEST(CacheGroupTest, HoldsOnlySchemaIndexManagerAndPoolObserver) {
    BlockPool pool(8);
    auto mgr = std::make_unique<FullAttnManager>(4);
    CacheGroup g(/*schema_index=*/3, std::move(mgr), pool);
    EXPECT_EQ(g.SchemaIndex(), 3u);
    EXPECT_EQ(&g.Pool(), &pool);
}

TEST(MakeCoordinatorTest, OwnsNormalizedSchemaAndPreservesLegacyThreeFieldAggregate) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        KvCacheSpec{
            .kind = AttnKind::kSlidingWindow,
            .block_size = 4,
            .sliding_window = 10,
            .group_id = "window",
            .rows_per_page = 2,
            .entry_stride_tokens = 2,
        },
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    EXPECT_EQ(coord.NumGroups(), 2);
    ASSERT_EQ(coord.Schema().size(), 2u);
    EXPECT_EQ(coord.Schema()[0].group_id, "0");
    EXPECT_EQ(coord.Schema()[0].rows_per_page, 4);
    EXPECT_EQ(coord.Schema()[0].entry_stride_tokens, 1);
    EXPECT_EQ(coord.Schema()[1].group_id, "window");
    EXPECT_EQ(coord.Schema()[1].rows_per_page, 2);
    EXPECT_EQ(coord.Schema()[1].entry_stride_tokens, 2);
    EXPECT_EQ(coord.FindSchemaIndex("0"), 0u);
    EXPECT_EQ(coord.FindSchemaIndex("window"), 1u);
    EXPECT_FALSE(coord.FindSchemaIndex("missing").has_value());
}

TEST(KvCacheCoordinatorFreeTest, IsNoexceptAndPreservesTailFirstRecyclingOrder) {
    BlockPool pool(/*total_num_blocks=*/4);  // three usable blocks
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 1, 0}};
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(static_cast<std::size_t>(coordinator.NumGroups()));
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/3));
    static_assert(noexcept(std::declval<KvCacheCoordinator&>().Free(std::declval<std::span<BlockTable>>())));

    coordinator.Free(tables);
    std::vector<BlockRef> recycled = pool.AcquireBlocks(3);
    ASSERT_EQ(recycled.size(), 3u);
    EXPECT_EQ(recycled[0]->BlockId(), 3);
    EXPECT_EQ(recycled[1]->BlockId(), 2);
    EXPECT_EQ(recycled[2]->BlockId(), 1);
    recycled.clear();
}

TEST(MakeCoordinatorTest, AcceptsDivisibleBlockSizesAndFoldsGcdLcm) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        // Per-group block_size is a multiple of the base.
        {AttnKind::kSlidingWindow, 8, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    EXPECT_EQ(coord.BaseBlockSize(), 4);           // gcd(4,8)
    EXPECT_EQ(coord.HistoryAlignmentTokens(), 8);  // history lcm(4,8)
}

TEST(CoordinatorMatchTest, BothGroupsAllMiss) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(CoordinatorMatchTest, CommonIsMinCoverageFullDeeperThanSwa) {
    // full caches 4 contiguous pages; swa (window 10 -> pages_needed 3)
    // caches only the last 3. Common = min(4, 3) = 3.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    // swa front 3-run (a TAIL run would null-pad back to index 0 -> coverage 4).
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 12);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 3u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 3u);
    // Full had 4 real hits, truncated to 3 -> num_hit recomputed to 3.
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
}

TEST(CoordinatorMatchTest, TrimmedFullHitsDoNotRefreshEvictionOrder) {
    BlockPool pool(11);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    const std::int32_t trimmed = CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    const std::string unrelated = ContentHashes({{9, 9, 9, 9}}).front();
    CacheForGroup(pool, unrelated, 99);

    CoordinatorMatch match = ProbeAndAcquirePrefix(coord, ch).device;
    ASSERT_EQ(match.num_common_tokens, 12);

    std::vector<BlockRef> acquired = pool.AcquireBlocks(3);
    ASSERT_EQ(acquired.size(), 3u);
    EXPECT_EQ(acquired.back()->BlockId(), trimmed);
}

TEST(CoordinatorMatchTest, AdmissionProbeDoesNotRefreshEvictionOrder) {
    BlockPool pool(3);  // 2 usable blocks, both cached below.
    const std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    const std::int32_t oldest = CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    KvCacheCoordinator::AdmissionProbe probe = coord.ProbePrefix(std::span<const std::string>{ch}.first(1));
    EXPECT_EQ(probe.device.num_common_tokens, 4);
    EXPECT_EQ(probe.device.free_hit_blocks_by_pool, (PoolDemand{1}));
    EXPECT_EQ(pool.NumFreeBlocks(), 2);

    BlockRef acquired = pool.AcquireBlock();
    ASSERT_TRUE(acquired);
    EXPECT_EQ(acquired->BlockId(), oldest);
}

TEST(CoordinatorMatchTest, SwaMissForcesZeroCommon) {
    // full caches 2 pages, swa caches nothing -> common = min(2, 0) = 0.
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    EXPECT_EQ(m.per_group[0].blocks.size(), 0u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 0u);
}

TEST(CoordinatorAllocTest, ColdStartAllocatesAlignedPages) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CoordinatorMatch hit = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(hit.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, std::move(hit));  // no hits -> no-op
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/8));
    // 8 tokens / page 4 = 2 pages in EACH group; tables aligned.
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorAllocTest, ClaimsCommonPrefixThenAllocatesRemainder) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    // swa window 4 -> pages_needed 1, so a single cached front page is a hit.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);

    CoordinatorMatch hit = ProbeAndAcquirePrefix(coord, ch).device;
    ASSERT_EQ(hit.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    // 8 tokens total, 1 page (4 tokens) common -> 4 uncached tokens -> +1 page each.
    const std::int32_t num_common_tokens = hit.num_common_tokens;
    coord.ClaimCommonPrefix(tables, std::move(hit));  // claim the 1 cached page each
    ASSERT_TRUE(coord.Acquire(tables, 8 - num_common_tokens));
    EXPECT_EQ(tables[0].NumBlocks(), 2);  // 1 claimed + 1 allocated
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorPreparedPrefixTest, CommitsClaimAndFreshTailAsOneOwnedResult) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}, {5, 6, 7, 8}});
    CacheForGroup(pool, hashes[0], /*group_id=*/0);

    KvCacheCoordinator::PreparedPrefix prepared = coord.PreparePrefix(hashes);
    ASSERT_EQ(prepared.HitTokens(), 4);
    ASSERT_EQ(prepared.ClaimDemand(), (PoolDemand{1}));

    KvCacheCoordinator::FreshAllocationPlan allocation =
        coord.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    std::optional<KvCacheCoordinator::CommittedPrefix> committed =
        coord.CommitPreparedPrefix(std::move(prepared), std::move(allocation));

    ASSERT_TRUE(committed.has_value());
    ASSERT_EQ(committed->hit_tokens, 4);
    ASSERT_EQ(committed->tables.size(), 1u);
    EXPECT_EQ(committed->tables[0].NumBlocks(), 2);
    coord.Free(committed->tables);
    EXPECT_EQ(pool.NumFreeBlocks(), pool.TotalBlocks() - 1);
}

static_assert(!std::is_copy_constructible_v<KvCacheCoordinator::AdmissionProbe>);
static_assert(std::is_move_constructible_v<KvCacheCoordinator::AdmissionProbe>);
static_assert(!std::is_copy_constructible_v<KvCacheCoordinator::PreparedPrefix>);
static_assert(std::is_move_constructible_v<KvCacheCoordinator::PreparedPrefix>);

TEST(CoordinatorPreparedPrefixTest, AdmissionProbeCanBeConsumedOnlyOnce) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(pool, hashes[0], /*group_id=*/0);

    KvCacheCoordinator::AdmissionProbe probe = coord.ProbePrefix(hashes);
    KvCacheCoordinator::AdmissionMatch acquired = coord.AcquirePrefix(std::move(probe));
    const std::int32_t free_after_acquire = pool.NumFreeBlocks();

    EXPECT_THROW(coord.AcquirePrefix(std::move(probe)), std::logic_error);
    EXPECT_EQ(pool.NumFreeBlocks(), free_after_acquire);
}

TEST(CoordinatorPreparedPrefixTest, CapacityConsumedAfterPrepareRejectsBeforeAcquire) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    CacheForGroup(pool, hashes[0], /*group_id=*/0);

    KvCacheCoordinator::PreparedPrefix prepared = coord.PreparePrefix(hashes);
    std::vector<BlockRef> unrelated = pool.AcquireBlocks(pool.NumFreeBlocks() - 1);
    ASSERT_EQ(pool.NumFreeBlocks(), 1);
    const std::int32_t free_before_commit = pool.NumFreeBlocks();

    KvCacheCoordinator::FreshAllocationPlan allocation =
        coord.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    std::optional<KvCacheCoordinator::CommittedPrefix> committed =
        coord.CommitPreparedPrefix(std::move(prepared), std::move(allocation));

    EXPECT_FALSE(committed.has_value());
    EXPECT_EQ(pool.NumFreeBlocks(), free_before_commit);
    unrelated.clear();
    const std::int32_t free_before_retry = pool.NumFreeBlocks();

    KvCacheCoordinator::FreshAllocationPlan retry_allocation =
        coord.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    EXPECT_THROW(coord.CommitPreparedPrefix(std::move(prepared), std::move(retry_allocation)), std::logic_error);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before_retry);

    KvCacheCoordinator::PreparedPrefix fresh = coord.PreparePrefix(hashes);
    KvCacheCoordinator::FreshAllocationPlan fresh_allocation =
        coord.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    committed = coord.CommitPreparedPrefix(std::move(fresh), std::move(fresh_allocation));
    ASSERT_TRUE(committed.has_value());
    coord.Free(committed->tables);
    EXPECT_EQ(pool.NumFreeBlocks(), pool.TotalBlocks() - 1);
}

#if TOKENSPEED_FLAT_KVCACHE
TEST(CoordinatorPreparedPrefixTest, AbandonedCommittedEventRollsBackSlotPagesAndReservation) {
    BlockPool pool(8);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool(1);
    const std::int32_t free_baseline = pool.NumFreeBlocks();
    const std::int32_t slots_baseline = req_pool.AvailableSlots();

    KvCacheCoordinator::PreparedPrefix prepared = coord.PreparePrefix({});
    KvCacheCoordinator::FreshAllocationPlan allocation =
        coord.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    std::optional<KvCacheCoordinator::CommittedPrefix> committed =
        coord.CommitPreparedPrefix(std::move(prepared), std::move(allocation));
    ASSERT_TRUE(committed.has_value());
    ASSERT_LT(pool.NumFreeBlocks(), free_baseline);

    FlatReservationTracker reservations(/*pool_count=*/1);
    auto reservation = reservations.MakeAccount();
    reservation.Set(PoolDemand{1});
    {
        fsm::FlatCommittedAdmission admission{std::make_unique<ReqPoolIndex>(req_pool.Allocate()),
                                              std::move(committed->tables), committed->hit_tokens, &reservation};
        fsm::ScheduleFlatPrefillFirstChunkEvent abandoned{/*tokens_this_round=*/4,
                                                          /*decode_input_tokens=*/0, Role::kFused,
                                                          std::move(admission)};
    }

    EXPECT_EQ(pool.NumFreeBlocks(), free_baseline);
    EXPECT_EQ(req_pool.AvailableSlots(), slots_baseline);
    EXPECT_TRUE(reservations.Empty());
    EXPECT_EQ(reservations.Total(), (PoolDemand{0}));
}
#endif

TEST(CoordinatorAllocTest, CrossGroupShortfallAllocatesNothing) {
    BlockPool pool(5);  // 4 usable after null reservation
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CoordinatorMatch hit = ProbeAndAcquirePrefix(coord, ch).device;  // all miss, common 0
    ASSERT_EQ(hit.num_common_tokens, 0);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    coord.ClaimCommonPrefix(tables, std::move(hit));  // no hits -> no-op
    // 12 tokens -> 3 pages per group = 6 needed, only 4 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 12));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);  // untouched, not rolled back
}

TEST(CoordinatorStepTest, AcquireKeepsGroupsAligned) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));  // 1 page each
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);
    ASSERT_TRUE(coord.Acquire(tables, 4));  // 1 more each
    EXPECT_EQ(tables[0].NumBlocks(), 2);
    EXPECT_EQ(tables[1].NumBlocks(), 2);
}

TEST(CoordinatorStepTest, AcquireShortfallAllocatesNothing) {
    BlockPool pool(3);  // 2 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    std::int32_t free_before = pool.NumFreeBlocks();
    // 2 pages per group (8 tokens) = 4 blocks, only 2 free -> fail, nothing taken.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(CoordinatorStepTest, CacheFullBlocksThenMatchHits) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 4));  // 1 page each
    coord.CacheFullBlocks(tables, ch);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
}

TEST(CoordinatorStepTest, FreeReturnsAllGroups) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 8));  // 2 pages each = 4 blocks
    std::int32_t free_mid = pool.NumFreeBlocks();
    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_mid + 4);
}

TEST(CoordinatorStepTest, EndToEndTwoRequestsSharePrefix) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});

    // Request A: cold, allocate 2 pages each, cache both, free.
    {
        CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
        EXPECT_EQ(m.num_common_tokens, 0);
        std::vector<BlockTable> a(2);
        coord.ClaimCommonPrefix(a, std::move(m));
        ASSERT_TRUE(coord.Acquire(a, 8));
        coord.CacheFullBlocks(a, ch);
        coord.Free(a);
    }
    // Request B: shares the prefix -> common 2 pages in both groups.
    {
        CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
        EXPECT_EQ(m.num_common_tokens, 8);
        std::vector<BlockTable> b(2);
        const std::int32_t num_common_tokens = m.num_common_tokens;
        coord.ClaimCommonPrefix(b, std::move(m));
        ASSERT_TRUE(coord.Acquire(b, 8 - num_common_tokens));
        EXPECT_EQ(b[0].NumBlocks(), 2);
        EXPECT_EQ(b[1].NumBlocks(), 2);
        coord.Free(b);
    }
}

TEST(CoordinatorStepTest, CacheFullBlocksAtSlotOffsetExtendsPrefix) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 24));                 // 6 pages each
    coord.CacheFullBlocks(tables, std::span(ch).first(4));  // prefill path: slots 0..3
    coord.CacheFullBlocks(tables, std::span(ch).subspan(4), /*first_slot=*/4);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 24);
    ASSERT_EQ(m.per_group.size(), 2u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 6u);
    for (std::size_t s = 0; s < 6; ++s) {
        EXPECT_EQ(m.per_group[0].blocks[s], tables[0].Blocks()[s]) << "slot " << s;
    }
    // swa window 4 -> pages_needed 1: tail run maps to the offset-registered slot-5 block.
    ASSERT_EQ(m.per_group[1].blocks.size(), 6u);
    EXPECT_EQ(m.per_group[1].blocks[5], tables[1].Blocks()[5]);
}

TEST(CoordinatorStepTest, CacheFullBlocksAtOffsetSkipsSwaHoles) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 24));  // 6 pages each
    // num_computed=20 -> swa skipped = 20-4+1 = 17 -> 17/4 = 4 pages punched:
    // swa slots 0..3 are null holes.
    for (std::int32_t g = 0; g < coord.NumGroups(); ++g) {
        coord.GroupManager(g).ReclaimExpired(pool, tables[static_cast<std::size_t>(g)], /*num_computed_tokens=*/20);
    }
    ASSERT_FALSE(tables[1].Blocks()[3]);
    ASSERT_TRUE(tables[1].Blocks()[4]);

    coord.CacheFullBlocks(tables, std::span(ch).subspan(2), /*first_slot=*/2);
    for (std::size_t s = 2; s < 6; ++s) {
        EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[s], 0))) << "full slot " << s;
    }
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[2], 1)));
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[3], 1)));
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[4], 1)));
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[5], 1)));
}

TEST(CoordinatorStepTest, CacheFullBlocksRejectsOutOfRangeFirstSlot) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{7, 7, 7, 7}});
    std::vector<BlockTable> tables(2);
    ASSERT_TRUE(coord.Acquire(tables, 8));  // 2 pages each
    EXPECT_THROW(coord.CacheFullBlocks(tables, ch, /*first_slot=*/2), std::runtime_error);
    EXPECT_THROW(coord.CacheFullBlocks(tables, ch, /*first_slot=*/-1), std::runtime_error);
}

TEST(CoordinatorMatchTest, SwaRunCutByFullBoundDropsToNoValidMatch) {
    // full covers 4; swa's tail run {2,3,4} bounded to 4 leaves run {2,3} <
    // pages_needed 3 with holes at 0,1 -> no valid swa match, common = 0.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_size=*/4);
}

TEST(CoordinatorMatchTest, FullShorterThanSwaBoundsSwaWithRunIntact) {
    // full covers 4; swa caches 1..4. Bounded to 4 the run {1,2,3} still reaches
    // pages_needed 3, so common stays 4 -- hole only OUTSIDE the last window.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_TRUE(m.per_group[1].blocks[1]);
    EXPECT_TRUE(m.per_group[1].blocks[2]);
    EXPECT_TRUE(m.per_group[1].blocks[3]);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_size=*/4);
}

TEST(CoordinatorMatchTest, SwaShorterThanFullTruncatesFull) {
    // swa's best valid match is 4 blocks [null, b1, b2, b3]; full truncates 5 -> 4.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    ASSERT_EQ(m.per_group[1].blocks.size(), 4u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/10, /*block_size=*/4);
}

TEST(CoordinatorMatchTest, TwoSwaGroupsSharedBoundaryMatches) {
    // pages_needed 3. full: 5; both SWA groups cache {1,2,3}, so each accepts
    // the SAME boundary 4 in the single sweep -- no cascade, full truncates 5 -> 4.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    for (std::uint32_t g : {1u, 2u}) {
        CacheForGroup(pool, ch[1], g);
        CacheForGroup(pool, ch[2], g);
        CacheForGroup(pool, ch[3], g);
    }

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    for (std::size_t i = 1; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 4u) << "group " << i;
        EXPECT_FALSE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_EQ(m.per_group[i].num_hit_blocks, 3) << "group " << i;
        ExpectSwaWindowIntact(m.per_group[i], /*window=*/10, /*block_size=*/4);
    }
}

TEST(CoordinatorMatchTest, TwoSwaGroupsCascadingShrinkConverges) {
    // swaA first accepts boundary 5 (run {2,3,4}), then swaB shrinks the bound to 4 (run {1,2,3})
    // UNDER swaA's match; re-matching swaA at 4 fails its window ({1} missing) and cascades both
    // groups down to boundary 1 -- the greatest boundary ALL groups support (block 0 shared).
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);
    CacheForGroup(pool, ch[2], 2);
    CacheForGroup(pool, ch[3], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 1u) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_EQ(m.per_group[i].num_hit_blocks, 1) << "group " << i;
    }
}

TEST(CoordinatorMatchTest, SwaGroupOrderDoesNotChangeConvergedCommon) {
    // The cascade fixture above with the two window groups swapped: convergence must land on the
    // same greatest common boundary regardless of sweep order among non-closed groups.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kSlidingWindow, 4, 10},  // = swaB above
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},  // = swaA above
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch =
        ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}, {5, 5, 5, 5}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 1);
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[2], 2);
    CacheForGroup(pool, ch[3], 2);
    CacheForGroup(pool, ch[4], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 4);
}

TEST(CoordinatorMatchTest, MultiWindowThreeGroupsSharedBoundary) {
    // Mixed window sizes on one pool (the W=128 / W=4 / full shape, scaled to P=2: W=6 needs a
    // 3-page tail, W=2 needs 1): all three groups support boundary 5 and the pool stays unified.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 2, 0},
        {AttnKind::kSlidingWindow, 2, 6},
        {AttnKind::kSlidingWindow, 2, 2},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);
    CacheForGroup(pool, ch[4], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 10);
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 5);
    EXPECT_EQ(m.per_group[1].blocks.size(), 5u);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 3);  // holes at 0,1
    EXPECT_EQ(m.per_group[2].blocks.size(), 5u);
    EXPECT_EQ(m.per_group[2].num_hit_blocks, 1);  // holes at 0..3
    ExpectSwaWindowIntact(m.per_group[1], /*window=*/6, /*block_size=*/2);
    ExpectSwaWindowIntact(m.per_group[2], /*window=*/2, /*block_size=*/2);
}

TEST(CoordinatorMatchTest, MultiWindowCascadeToZero) {
    // W=2's only cached page (3) forces boundary 4, where W=6 cannot cover {1,2,3} -> its
    // re-match collapses to 0 and drags the small window with it: no boundary works, common = 0.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 2, 0},
        {AttnKind::kSlidingWindow, 2, 6},
        {AttnKind::kSlidingWindow, 2, 2},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[2], 1);
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[4], 1);
    CacheForGroup(pool, ch[3], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0);
    for (const PrefixMatch& g : m.per_group) {
        EXPECT_TRUE(g.blocks.empty());
    }
}

TEST(CoordinatorMatchTest, DeepCascadeRequiresSecondConvergeSweep) {
    // Forces the converge loop through a SECOND productive sweep: swaA re-matches to boundary 3,
    // then swaB's re-match lands at 2 UNDER swaA's already re-matched boundary, so swaA must
    // re-match again. Hand-computed greatest common boundary: 2 pages (A@3 valid but B@3 lacks
    // block 2; at 2 both windows clamp to begin over {0,1}).
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},  // swaA, needed = 3
        {AttnKind::kSlidingWindow, 4, 10},  // swaB, needed = 3
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0},
                                                 {1, 1, 1, 1},
                                                 {2, 2, 2, 2},
                                                 {3, 3, 3, 3},
                                                 {4, 4, 4, 4},
                                                 {5, 5, 5, 5},
                                                 {6, 6, 6, 6},
                                                 {7, 7, 7, 7}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    for (int j : {0, 1, 2, 4, 5, 6}) CacheForGroup(pool, ch[static_cast<std::size_t>(j)], 1);  // swaA
    for (int j : {0, 1, 3, 4, 5}) CacheForGroup(pool, ch[static_cast<std::size_t>(j)], 2);     // swaB

    // Sweep: A {4,5,6} -> 7; B {3,4,5} -> 6. Converge pass 1: A@6 -> run {0,1,2} -> 3;
    // B@3 -> bottoming run {0,1} -> 2. Pass 2: A@2 -> bottoming {0,1} -> 2. Stable.
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 2u) << "group " << i;
        EXPECT_EQ(m.per_group[i].num_hit_blocks, 2) << "group " << i;
    }
}

TEST(CoordinatorMatchTest, MultiWindowSlideCreditSumsPerWindow) {
    // Retention is per-window: at 10 computed tokens W=6 has slid past 2 pages, W=2 past 4,
    // full past none -- the gate credit is their sum over the shared pool.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 2, 0},
        {AttnKind::kSlidingWindow, 2, 6},
        {AttnKind::kSlidingWindow, 2, 2},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/10));

    std::int32_t credit = 0;
    for (std::int32_t i = 0; i < coord.NumGroups(); ++i) {
        credit += coord.GroupManager(i).BlocksReclaimableAt(tables[static_cast<std::size_t>(i)],
                                                            /*num_computed_tokens=*/10,
                                                            /*count_uncached=*/true);
    }
    EXPECT_EQ(credit, 6);  // 0 (full) + 2 (W=6) + 4 (W=2)

    const std::int32_t free_before = pool.NumFreeBlocks();
    for (std::int32_t i = 0; i < coord.NumGroups(); ++i) {
        coord.GroupManager(i).ReclaimExpired(pool, tables[static_cast<std::size_t>(i)], 10);
    }
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 6) << "reclaim must deliver exactly the credited pages";
    coord.Free(tables);
}

TEST(CoordinatorMatchTest, AllFullGroupsMinTruncationUnchanged) {
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SingleFullGroupUnchanged) {
    BlockPool pool(16);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8);
    ASSERT_EQ(m.per_group.size(), 1u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, SwaOnlyConfigKeepsTailRunWithLeadingHoles) {
    // No full bound: tail run {2,3,4} covers the window; leading holes null-pad to page 0.
    BlockPool pool(32);
    std::vector<KvCacheSpec> specs = {{AttnKind::kSlidingWindow, 4, 10}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}});
    CacheForGroup(pool, ch[2], 0);
    CacheForGroup(pool, ch[3], 0);
    CacheForGroup(pool, ch[4], 0);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 20);
    ASSERT_EQ(m.per_group.size(), 1u);
    ASSERT_EQ(m.per_group[0].blocks.size(), 5u);
    EXPECT_FALSE(m.per_group[0].blocks[0]);
    EXPECT_FALSE(m.per_group[0].blocks[1]);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 3);
    ExpectSwaWindowIntact(m.per_group[0], /*window=*/10, /*block_size=*/4);
}

TEST(CoordinatorAllocTest, AcquireShortfallLeavesClaimedPrefixForCallerToFree) {
    BlockPool pool(6);  // 5 usable
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[0], 1);
    std::int32_t free_before = pool.NumFreeBlocks();  // after caching, before claim

    CoordinatorMatch hit = ProbeAndAcquirePrefix(coord, ch).device;
    ASSERT_EQ(hit.num_common_tokens, 4);

    std::vector<BlockTable> tables(2);
    coord.ClaimCommonPrefix(tables, std::move(hit));  // transfer 1 cached page per group
    EXPECT_EQ(tables[0].NumBlocks(), 1);
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    // Uncached 8 tokens -> 2 pages/group = 4 needed; 5 usable - 2 claimed = 3 free -> fail.
    EXPECT_FALSE(coord.Acquire(tables, 8));
    EXPECT_EQ(tables[0].NumBlocks(), 1);  // claimed prefix still there
    EXPECT_EQ(tables[1].NumBlocks(), 1);

    coord.Free(tables);
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(KvCacheCoordinatorReclaimExpired, OnlySlidingWindowGroupEvicts) {
    BlockPool pool(/*total_num_blocks=*/32, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    // 6 tokens -> 3 pages per group.
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));
    ASSERT_EQ(tables[0].NumBlocks(), 3);
    ASSERT_EQ(tables[1].NumBlocks(), 3);

    const std::vector<std::int32_t> full_snapshot = BlockTablePageIds(tables[0]);

    // num_computed_tokens=5 -> swa skipped=5-4+1=2 -> skipped_blocks=2/2=1 -> page 0 evicted.
    for (std::int32_t g = 0; g < coordinator.NumGroups(); ++g) {
        coordinator.GroupManager(g).ReclaimExpired(pool, tables[static_cast<std::size_t>(g)],
                                                   /*num_computed_tokens=*/5);
    }

    ASSERT_EQ(tables[0].NumBlocks(), 3);
    const auto full_after = tables[0].Blocks();
    for (std::int32_t i = 0; i < tables[0].NumBlocks(); ++i) {
        ASSERT_TRUE(full_after[i]) << "full group got a null hole at " << i;
        EXPECT_EQ(full_after[i]->BlockId(), full_snapshot[i]) << "full group block " << i << " changed";
    }

    ASSERT_EQ(tables[1].NumBlocks(), 3);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_TRUE(tables[1].Blocks()[1]);
    EXPECT_TRUE(tables[1].Blocks()[2]);
}

TEST(CoordinatorMatchTest, ThreeGroupsCommonIsMinCoverageAcrossAll) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    // Shortest window group first in index order: group 2's deeper match trims to
    // group 1's bound inside the sweep (the reverse order would be a cascade).
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[0], 2);
    CacheForGroup(pool, ch[1], 2);
    CacheForGroup(pool, ch[2], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "common = min(4, 2, 3) pages, in tokens (P=4)";
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[2].blocks.size(), 2u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[2].num_hit_blocks, 2);
}

TEST(CoordinatorMatchTest, ThreeGroupsOneAllMissForcesZeroCommon) {
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 40},
        {AttnKind::kSlidingWindow, 4, 40},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    // Groups 0 and 2 fully cache both pages; group 1 caches nothing. The all-miss
    // group zeroes the bound before group 2 matches (the reverse would be a cascade).
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    for (const std::string& h : ch) CacheForGroup(pool, h, 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 0) << "one group all-miss -> common 0";
}

TEST(KvCacheCoordinatorStoreCandidates, CollectsPinnedNewRegistrations) {
    BlockPool pool(/*total_num_blocks=*/16, /*enable_caching=*/true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/4},
    };
    BlockPool host_pool(4);
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool, &host_pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    const std::int32_t free_before = pool.NumFreeBlocks();

    coordinator.CacheFullBlocks(tables, hashes, /*first_slot=*/0);
    std::vector<KvCacheCoordinator::StoreCandidate> pending = coordinator.TakePendingStores();

    ASSERT_EQ(pending.size(), 4u);  // 2 pages x 2 groups, group-wrapped keys
    EXPECT_EQ(pool.NumFreeBlocks(), free_before) << "pinning ref'd blocks must not touch the free list";
    for (const auto& c : pending) {
        EXPECT_GE(c.block.use_count(), 2) << "pinned on top of the table ref";
    }
    // Keys are group-wrapped and distinct across groups for the same content hash.
    std::set<std::string> keys;
    for (const auto& c : pending) keys.insert(c.key);
    EXPECT_EQ(keys.size(), 4u);
    // Collection is group-major: 2 pages for group 0, then 2 for group 1 (group lives in the key).
    for (std::size_t i = 0; i < pending.size(); ++i) {
        EXPECT_EQ(pending[i].key, MakeKeyWithGroupId(hashes[i % 2], static_cast<std::uint32_t>(i / 2)))
            << "candidate " << i;
    }

    // Re-registering the same hashes yields nothing new (IsCached skip).
    coordinator.CacheFullBlocks(tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());

    // Unpin restores balance.
    pending.clear();  // dropping the candidates releases the pins
    coordinator.Free(tables);
    EXPECT_EQ(pool.NumFreeBlocks(), 15);  // all but null block 0
}

TEST(KvCacheCoordinatorStoreCandidates, DisabledByDefaultCollectsNothing) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*block_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, 4));
    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    coordinator.CacheFullBlocks(tables, hashes, 0);
    EXPECT_TRUE(coordinator.TakePendingStores().empty());
    coordinator.Free(tables);
}

// Caller-side slide credit (the scheduler's flat gates run this exact per-group loop).
std::int32_t SlideCredit(const KvCacheCoordinator& coord, std::span<const BlockTable> tables,
                         std::int32_t num_computed_tokens) {
    std::int32_t total_freed = 0;
    for (std::int32_t i = 0; i < coord.NumGroups(); ++i) {
        total_freed +=
            coord.GroupManager(i).BlocksReclaimableAt(tables[static_cast<std::size_t>(i)], num_computed_tokens,
                                                      /*count_uncached=*/!coord.HasHostTier());
    }
    return total_freed;
}

// The exact slide-credit rule: collection-on credits a slide-out block only when it is CACHED
// (an uncached one is this op's own registration and gets pinned before the slide); collection-off
// keeps the unique-owner rule so uncached same-round registrations still free.
TEST(KvCacheCoordinatorStoreCandidates, SlideCreditExcludesUncachedOnlyWhenCollecting) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kSlidingWindow, /*block_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator off = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(off.NumGroups());
    ASSERT_TRUE(off.Acquire(tables, /*num_tokens=*/8));  // pages 0..3; N=8 slides out pages 0,1
    EXPECT_EQ(SlideCredit(off, tables, 8), 2) << "collection-off counts uncached ref-1 blocks";
    off.Free(tables);

    BlockPool host_pool(4);
    KvCacheCoordinator on = MakeCoordinator(specs, pool, &host_pool);
    std::vector<BlockTable> tables2(on.NumGroups());
    ASSERT_TRUE(on.Acquire(tables2, 8));
    EXPECT_EQ(SlideCredit(on, tables2, 8), 0) << "collection-on excludes uncached slide-out blocks";

    std::vector<std::string> hashes = ContentHashes({{1, 2}, {3, 4}});
    on.CacheFullBlocks(tables2, hashes, 0);  // registers + pins pages 0,1
    EXPECT_EQ(SlideCredit(on, tables2, 8), 0) << "cached but still pinned: ref 2";
    on.TakePendingStores();  // unpin, as WriteBackDone would: dropping the candidates releases the pins
    EXPECT_EQ(SlideCredit(on, tables2, 8), 2) << "cached and unpinned: exact credit restored";
    on.Free(tables2);
}

// Publish a host page for (hash, group) directly (the scheduler's store path minus the
// D2H write): allocate -> hash -> free leaves it cached-and-evictable, like a committed store.
std::int32_t HostPut(BlockPool& host_pool, const std::string& content_hash, std::uint32_t gid) {
    std::string key = MakeKeyWithGroupId(content_hash, gid);
    BlockRef block = host_pool.AcquireBlock();
    const std::int32_t id = block->BlockId();
    host_pool.CacheFullBlock(block, key);
    block.reset();
    return id;
}

// Cache slots [0, blocks) in the DEVICE pool for every group, so the merged admission match's
// device boundary lands exactly there (SWA's bottom-clamped run accepts any such floor).
void SeedDeviceFloor(BlockPool& pool, const KvCacheCoordinator& coord, std::span<const std::string> ch,
                     std::int32_t blocks) {
    for (std::int32_t g = 0; g < coord.NumGroups(); ++g) {
        for (std::int32_t j = 0; j < blocks; ++j) {
            (void)HostPut(pool, ch[static_cast<std::size_t>(j)], static_cast<std::uint32_t>(g));
        }
    }
}

// Fixture constants: full P=2 + SWA W=4 -> pages_needed = (4-1+2-1)/2 = 2.
std::vector<KvCacheSpec> HostExtSpecs() {
    return {KvCacheSpec{AttnKind::kFull, 2, 0}, KvCacheSpec{AttnKind::kSlidingWindow, 2, 4}};
}

TEST(KvCacheCoordinatorHostExtension, BothGroupsFullyPresent) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    std::vector<std::int32_t> fp, sp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 2; j <= 3; ++j) sp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);  // boundary 4 blocks * P=2 (floor 1 + extension 3)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2]}));
    // SWA tail at boundary 4 needs blocks [2, 4); extension slot for block 1 is a hole.
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{0, sp[0], sp[1]}));
    EXPECT_EQ(m.per_group[0].num_hit_blocks + m.per_group[1].num_hit_blocks, 5);
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 5);
}

TEST(KvCacheCoordinatorHostExtension, SwaTailMissShrinksBoundary) {
    // swa misses block 3 -> boundary shrinks to 3; tail at 3 = blocks [1, 3), which hits.
    // ext = 2, swa start = max(1, 3-2) = 1 = dev -> no holes; full's block-3 page stays unpinned.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    std::vector<std::int32_t> fp, sp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 1; j <= 2; ++j) sp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 6);  // boundary 3 blocks * P=2 (floor 1 + extension 2)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1]}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{sp[0], sp[1]}));
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 4);
}

TEST(KvCacheCoordinatorHostExtension, FullGapCapsExtension) {
    // full misses block 2 -> boundary 2; swa tail at 2: start = max(1, 2-2) = 1 -> needs block 1 only.
    // ext = 1, both groups = {block-1 page}, 2 pins -- swa's deeper blocks 2..3 stay unused.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    const std::int32_t fp1 = HostPut(host_pool, ch[1], 0);
    (void)HostPut(host_pool, ch[3], 0);  // gap at block 2
    std::vector<std::int32_t> sp;
    for (int j = 1; j <= 3; ++j) sp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1));

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 4);  // boundary 2 blocks * P=2 (floor 1 + extension 1)
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp1}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{sp[0]}));
}

TEST(KvCacheCoordinatorHostExtension, EmptyStoreZeroExtension) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(5);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 2) << "no extension: boundary stays at the device floor";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_TRUE(m.per_group[0].blocks.empty());
    EXPECT_TRUE(m.per_group[1].blocks.empty());
}

TEST(KvCacheCoordinatorHostExtension, DeviceBoundaryRespected) {
    // Host holds only blocks 0..1 (below dev=2): zero extension, and those entries stay unpinned.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(5);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    for (int j = 0; j <= 1; ++j) {
        (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0);
        (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1);
    }

    SeedDeviceFloor(pool, coord, ch, 2);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 4) << "below-floor host pages extend nothing";
    // The below-dev entries were never probed: all four stay evictable.
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 0);
    EXPECT_TRUE(host_pool.AcquireBlock());
}

TEST(KvCacheCoordinatorHostExtension, MatchPinsPagesUntilResultDies) {
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = HostExtSpecs();
    BlockPool host_pool(6);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}});

    for (int j = 1; j <= 3; ++j) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j = 2; j <= 3; ++j) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 5);
    EXPECT_EQ(host_pool.NumFreeBlocks(), 0);
    EXPECT_FALSE(host_pool.AcquireBlock()) << "matched pages must not be evicted before claim";
    m = {};
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 0);
    EXPECT_EQ(host_pool.NumFreeBlocks(), 5);
}

TEST(KvCacheCoordinatorHostExtension, DeepCascadeConverges) {
    // Host twin of the deep cascade (dev = 0 so the block math mirrors the device test):
    // convergence must land on extension 2 and take no refs along the way.
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 10},
        {AttnKind::kSlidingWindow, 4, 10},
    };
    BlockPool host_pool(32);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0},
                                                 {1, 1, 1, 1},
                                                 {2, 2, 2, 2},
                                                 {3, 3, 3, 3},
                                                 {4, 4, 4, 4},
                                                 {5, 5, 5, 5},
                                                 {6, 6, 6, 6},
                                                 {7, 7, 7, 7}});

    for (int j = 0; j <= 7; ++j) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j : {0, 1, 2, 4, 5, 6}) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1);
    for (int j : {0, 1, 3, 4, 5}) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 2);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 8);  // boundary 2 blocks * P=4 (floor 0)
    ASSERT_EQ(m.per_group.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        ASSERT_EQ(m.per_group[i].blocks.size(), 2u) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[0]) << "group " << i;
        EXPECT_TRUE(m.per_group[i].blocks[1]) << "group " << i;
    }
    // Abandoned intermediate matches release; only the final 2 pages/group stay pinned.
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 6);
}

TEST(KvCacheCoordinatorHostExtension, MultiWindowGroupsExtendTogether) {
    // Host-tier twin of the multi-window device case (P=2, full + W=6 + W=2, dev=1): every
    // group supports host boundary 5, holes pad below each window's tail run.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 2, 0},
        {AttnKind::kSlidingWindow, 2, 6},
        {AttnKind::kSlidingWindow, 2, 2},
    };
    BlockPool host_pool(16);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}});

    std::vector<std::int32_t> fp, ap;
    for (int j = 1; j <= 4; ++j) fp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0));
    for (int j = 2; j <= 4; ++j) ap.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1));
    const std::int32_t bp4 = HostPut(host_pool, ch[4], 2);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 10);  // boundary 5 blocks * P=2 (floor 1 + extension 4)
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2], fp[3]}));
    const std::int32_t hole = 0;
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{hole, ap[0], ap[1], ap[2]}));
    EXPECT_EQ(BlockIds(m.per_group[2].blocks), (std::vector<std::int32_t>{hole, hole, hole, bp4}));
    EXPECT_EQ(m.per_group[0].num_hit_blocks + m.per_group[1].num_hit_blocks + m.per_group[2].num_hit_blocks,
              8);  // 4 full + 3 W=6 + 1 W=2
}

TEST(KvCacheCoordinatorHostExtension, MultiWindowCascadeConvergesToZeroExtension) {
    // W=2's only host page (block 3) caps the boundary at 4, where W=6 cannot cover {1,2,3};
    // its re-match collapses to the device boundary and the extension converges to zero with
    // nothing left pinned.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 2, 0},
        {AttnKind::kSlidingWindow, 2, 6},
        {AttnKind::kSlidingWindow, 2, 2},
    };
    BlockPool host_pool(16);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}});

    for (int j = 1; j <= 4; ++j) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0);
    for (int j = 2; j <= 4; ++j) (void)HostPut(host_pool, ch[static_cast<std::size_t>(j)], 1);
    (void)HostPut(host_pool, ch[3], 2);

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 2) << "extension converges to zero: boundary = device floor";
    for (const PrefixMatch& g : m.per_group) {
        EXPECT_TRUE(g.blocks.empty());
    }
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 0);
}

// ---------------------------------------------------------------------------
// Mamba-analog semantics: vLLM reduces a mamba/linear-attention group to the
// paged machinery via (a) hit = ONE aligned state snapshot found right-to-left,
// padded with nulls ([null]*i + [state]); (b) retention = only the last token's
// state lives (skipped = n-1, exactly our W=2 slide rule); (c) L2 = sliding
// window of one block. These tests pin that our SwaManager with a one-page
// window already produces those exact shapes -- the machinery a MambaManager
// (AttnKind::kMambaState) would reuse unchanged.
// ---------------------------------------------------------------------------

TEST(MambaAnalogTest, HitIsSingleSnapshotPlusLeadingHoles) {
    // needed = ceil((5-1)/4) = 1: the match is the RIGHTMOST cached block with
    // null holes below -- byte-identical to MambaManager.find_longest_cache_hit.
    BlockPool pool(16);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/5);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[3], 0);  // rightmost snapshot; the gap at 2 is irrelevant

    std::vector<std::string> keys;  // direct manager call: wrap keys as the coordinator would
    for (const std::string& h : ch) keys.push_back(MakeKeyWithGroupId(h, 0));
    PrefixMatch m = mgr.Match(pool, keys, 0, /*max_blocks=*/4);
    ASSERT_EQ(m.blocks.size(), 4u);
    EXPECT_FALSE(m.blocks[0]);
    EXPECT_FALSE(m.blocks[1]);
    EXPECT_FALSE(m.blocks[2]);
    EXPECT_TRUE(m.blocks[3]);
    EXPECT_EQ(m.num_hit_blocks, 1);
}

TEST(MambaAnalogTest, RetentionKeepsOnlyTheLastStateBlock) {
    // vLLM: get_num_skipped_tokens = n-1 (only the last token's state matters).
    // Our slide rule skips n-W+1 tokens, so W=2 IS that policy: at 16 computed
    // tokens pages 0..2 free and only the tail state page survives.
    BlockPool pool(16);
    SwaManager mgr(/*block_size=*/4, /*sliding_window=*/2);
    BlockTable table;
    ASSERT_TRUE(mgr.Acquire(pool, table, /*num_tokens=*/16));
    ASSERT_EQ(table.NumBlocks(), 4);

    EXPECT_EQ(mgr.BlocksReclaimableAt(table, /*num_computed_tokens=*/16, /*count_uncached=*/true), 3);
    const std::int32_t free_before = pool.NumFreeBlocks();
    mgr.ReclaimExpired(pool, table, 16);
    EXPECT_EQ(pool.NumFreeBlocks(), free_before + 3);
    ASSERT_EQ(table.NumBlocks(), 4) << "holes preserve slot alignment";
    EXPECT_FALSE(table.Blocks()[0]);
    EXPECT_FALSE(table.Blocks()[1]);
    EXPECT_FALSE(table.Blocks()[2]);
    EXPECT_TRUE(table.Blocks()[3]) << "the live state block";
    mgr.Free(table);
}

TEST(MambaAnalogTest, HybridFullSwaMambaComposesUnderOnePool) {
    // Task-4 shape: full + real window (W=8) + mamba-analog (W=5, needed=1),
    // three groups sharing ONE BlockPool; common = min over all three.
    BlockPool pool(64);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 8},
        {AttnKind::kSlidingWindow, 4, 5},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& h : ch) CacheForGroup(pool, h, 0);
    CacheForGroup(pool, ch[2], 1);  // W=8 needs a 2-page tail run
    CacheForGroup(pool, ch[3], 1);
    CacheForGroup(pool, ch[3], 2);  // mamba-analog: one snapshot at the boundary

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 3u);
    EXPECT_EQ(m.per_group[0].num_hit_blocks, 4);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 2);
    EXPECT_EQ(m.per_group[2].num_hit_blocks, 1);

    // Claim + acquire keeps the pool unified and balanced across all three.
    std::vector<BlockTable> tables(coord.NumGroups());
    coord.ClaimCommonPrefix(tables, std::move(m));
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/4));
    coord.Free(tables);
}

TEST(MambaAnalogTest, HostTierStoresAndMatchesTheSnapshotOnly) {
    // vLLM offloading treats a mamba group as sliding_window_size_in_blocks=1:
    // the host index needs only the boundary snapshot; earlier keys stay holes.
    BlockPool pool(16, true);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kSlidingWindow, 4, 5},
    };
    BlockPool host_pool(8);
    KvCacheCoordinator coord = MakeCoordinator(specs, pool, &host_pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});

    std::vector<std::int32_t> fp;
    for (int j = 1; j <= 3; ++j) fp.push_back(HostPut(host_pool, ch[static_cast<std::size_t>(j)], 0));
    const std::int32_t snapshot = HostPut(host_pool, ch[3], 1);  // ONLY the boundary snapshot

    SeedDeviceFloor(pool, coord, ch, 1);
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).host;
    EXPECT_EQ(m.num_common_tokens, 16);  // boundary 4 blocks * P=4 (floor 1 + extension 3)
    EXPECT_EQ(BlockIds(m.per_group[0].blocks), (std::vector<std::int32_t>{fp[0], fp[1], fp[2]}));
    EXPECT_EQ(BlockIds(m.per_group[1].blocks), (std::vector<std::int32_t>{0, 0, snapshot}));
    EXPECT_EQ(m.per_group[0].num_hit_blocks + m.per_group[1].num_hit_blocks, 4);
}

// kMambaState is the named form of the analog pinned above: MakeCoordinator maps it to the
// W=2 machinery, so a mixed full+state model converges with single-snapshot state semantics.
TEST(MambaStateKindTest, FactoryMapsStateKindToAlignSemantics) {
    BlockPool pool(32, true);
    std::vector<KvCacheSpec> specs = {
        {AttnKind::kFull, 4, 0},
        {AttnKind::kMambaState, 4, 0},
    };
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    for (int j = 0; j <= 2; ++j) CacheForGroup(pool, ch[static_cast<std::size_t>(j)], 0);
    CacheForGroup(pool, ch[2], 1);  // ONLY the boundary snapshot for the state group

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 12);  // full covers 3 pages; state resumes off snapshot @2
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[1].num_hit_blocks, 1);
    ASSERT_EQ(m.per_group[1].blocks.size(), 3u);
    EXPECT_FALSE(m.per_group[1].blocks[0]);
    EXPECT_FALSE(m.per_group[1].blocks[1]);
    EXPECT_TRUE(m.per_group[1].blocks[2]);  // [null, null, snapshot]
}

TEST(MambaStateKindTest, StateGroupRetentionKeepsOnlyLastPage) {
    BlockPool pool(32, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kMambaState, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/16));  // 4 pages
    coord.ReclaimExpired(tables, /*num_computed_tokens=*/16);
    EXPECT_FALSE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[0].Blocks()[1]);
    EXPECT_FALSE(tables[0].Blocks()[2]);
    EXPECT_TRUE(tables[0].Blocks()[3]);  // skipped = n-1: only the live state page
    coord.Free(tables);
}

// State snapshots are only boundary-correct where a forward call ended page-aligned: the
// coordinator narrows a state group's registration to the final full page of an aligned range.
TEST(MambaStateRegistrationTest, AlignedEndRegistersOnlyFinalPage) {
    BlockPool pool(32, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kMambaState, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/12));  // 3 pages
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/12);
    // full group: all 3 registered; state group: only page 2 (the aligned chunk end)
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[0], 0)));
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[2], 0)));
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[0], 1)));
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[1], 1)));
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[2], 1)));
    coord.Free(tables);
}

TEST(MambaStateRegistrationTest, UnalignedEndRegistersNoStatePages) {
    BlockPool pool(32, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kMambaState, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/14));  // 3 full pages + partial tail
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/14);  // 14 % 4 != 0
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[2], 0)));     // full group unaffected
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(ch[2], 1)));    // state group skipped
    coord.Free(tables);
}

// ---- Hetero block_size (folded per-group granularity) -------------------

// {4,8}: base=4, group1 folds every 2 base pages into one 8-token coarse block. The
// converged boundary must be the cross-group min expressed in TOKENS (Step C token bound).
TEST(HeteroFoldedMatchTest, ConvergedBoundaryIsCrossGroupMinInTokens) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 8, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ASSERT_EQ(coord.BaseBlockSize(), 4);

    // 4 base pages (16 tokens). group0 caches 3 base blocks (12 tokens of coverage);
    // group1 caches only its first folded coarse block (base pages [0,1] -> 8 tokens).
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    CacheForGroup(pool, ch[0], 0);
    CacheForGroup(pool, ch[1], 0);
    CacheForGroup(pool, ch[2], 0);
    std::vector<std::string> folded_g1 = FoldBaseHashes(ch, /*first_base=*/0, /*m=*/2);
    ASSERT_EQ(folded_g1.size(), 2u);
    CacheForGroup(pool, folded_g1[0], 1);  // gid 1, coarse block 0

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "min(12 tokens, 8 tokens) in TOKENS, not blocks";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);  // 8 tokens / 4 = 2 base blocks (truncated from 3)
    EXPECT_EQ(m.per_group[1].blocks.size(), 1u);  // 8 tokens / 8 = 1 coarse block
}

// {8,4}: groups_[0].block_size(8) > base(=gcd(8,4)=4). The Step C token bound seeds
// SweepThenConverge with num_base_pages * base_block_size_; the pre-fix code used
// num_base_pages * schema[0].block_size, which here is DOUBLE the real token count.
// That wrong bound is only an initial UPPER cap on the converge seed, and every group's
// Match self-caps at min(keys.size(), max_blocks) and breaks on the first miss -- and
// keys.size() derives from the request's own content_hashes, so cached availability can
// never exceed the true request length. The over-loose bound is therefore non-binding in
// every reachable state and unobservable through prefix admission (with {8,4} just as with the
// {4,8} case above, where buggy == fixed exactly). This is a positive guard: it asserts the
// CORRECT cross-group-min-in-TOKENS boundary for a group whose block_size exceeds base, so
// the fold/bound wiring stays correct-by-construction even though the unit bug is unreachable.
TEST(HeteroFoldedMatchTest, ConvergedBoundaryWithGroup0LargerThanBase) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 8, 0}, {AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ASSERT_EQ(coord.BaseBlockSize(), 4);  // gcd(8,4); groups_[0].block_size(8) is DOUBLE base

    // 4 base pages (16 tokens). group0 (block_size 8) caches only its first folded coarse
    // block (base pages [0,1] -> 8 tokens); group1 (block_size 4) caches all 4 base blocks.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    std::vector<std::string> folded_g0 = FoldBaseHashes(ch, /*first_base=*/0, /*m=*/2);
    ASSERT_EQ(folded_g0.size(), 2u);
    CacheForGroup(pool, folded_g0[0], 0);  // gid 0, coarse block 0 only (block 1 misses)
    for (const std::string& h : ch) CacheForGroup(pool, h, 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "min(8 tokens, 16 tokens) in TOKENS; base-unit bound, not 8*num";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 1u);  // 8 tokens / 8 = 1 coarse block
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);  // 8 tokens / 4 = 2 base blocks (truncated from 4)
}

// {8,4} MIRROR: the coarse closed group covers MORE (16) than the fine group (12). Without
// lcm alignment the boundary settles at 12 while the coarse group floor-truncates to 8,
// leaving [8,12) neither cached nor recomputed; the boundary must fall to the lcm (8).
TEST(HeteroFoldedMatchTest, CoarseCoversMoreThanFineAlignsBoundaryToLcm) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 8, 0}, {AttnKind::kFull, 4, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ASSERT_EQ(coord.BaseBlockSize(), 4);
    ASSERT_EQ(coord.HistoryAlignmentTokens(), 8);

    // group0 (bs 8) caches both coarse blocks (16 tokens); group1 (bs 4) caches blocks 0..2 (12).
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    std::vector<std::string> folded_g0 = FoldBaseHashes(ch, /*first_base=*/0, /*m=*/2);
    ASSERT_EQ(folded_g0.size(), 2u);
    CacheForGroup(pool, folded_g0[0], 0);
    CacheForGroup(pool, folded_g0[1], 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "12 is not a whole coarse block; boundary aligns down to lcm";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 1u);  // coarse coverage == boundary, no phantom [8,12)
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);  // fine truncated to the boundary
}

// Two WINDOW groups of different granularity: the fine group drags the aligned bound under
// the coarse group's first match; the converge loop re-matches BOTH at the lcm-aligned bound.
TEST(HeteroFoldedMatchTest, CoarseWindowGroupRematchesUnderAlignedBound) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kSlidingWindow, 8, 8}, {AttnKind::kSlidingWindow, 4, 4}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ASSERT_EQ(coord.HistoryAlignmentTokens(), 8);

    // group0 (coarse swa) caches 16 tokens; group1 (fine swa) caches 12 -> aligned bound 8.
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    std::vector<std::string> folded_g0 = FoldBaseHashes(ch, /*first_base=*/0, /*m=*/2);
    CacheForGroup(pool, folded_g0[0], 0);
    CacheForGroup(pool, folded_g0[1], 0);
    CacheForGroup(pool, ch[0], 1);
    CacheForGroup(pool, ch[1], 1);
    CacheForGroup(pool, ch[2], 1);

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 8) << "12 aligns down to 8; both window groups re-match there";
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 1u);  // re-matched at 8: one coarse block
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);  // re-matched at 8: two fine blocks
}

// {4,8}: two full groups over 16 tokens register at their OWN folded granularity, and a
// second prefix admission hits every registered block.
TEST(HeteroFoldedRegistrationTest, EachGroupRegistersAtOwnGranularityThenHits) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 8, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/16));
    EXPECT_EQ(tables[0].NumBlocks(), 4);  // 16 / 4
    EXPECT_EQ(tables[1].NumBlocks(), 2);  // 16 / 8

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/16);

    // group0: 4 base coarse blocks; group1: 2 folded coarse blocks.
    std::vector<std::string> folded_g1 = FoldBaseHashes(ch, 0, 2);
    for (const std::string& h : ch) EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(h, 0)));
    for (const std::string& h : folded_g1) EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(h, 1)));

    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, ch).device;
    EXPECT_EQ(m.num_common_tokens, 16);
    ASSERT_EQ(m.per_group.size(), 2u);
    EXPECT_EQ(m.per_group[0].blocks.size(), 4u);
    EXPECT_EQ(m.per_group[1].blocks.size(), 2u);
    coord.Free(tables);
}

// A mamba (RegistersAlignedFinalPageOnly) group with m>1: only the FINAL aligned folded
// coarse block registers; the earlier coarse block stays unregistered.
TEST(HeteroFoldedRegistrationTest, MambaGroupRegistersOnlyFinalFoldedBlock) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kMambaState, 8, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    ASSERT_EQ(coord.BaseBlockSize(), 4);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/16));
    EXPECT_EQ(tables[1].NumBlocks(), 2);  // 16 / 8, two folded coarse blocks

    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/16);  // 16 % 8 == 0

    std::vector<std::string> folded_g1 = FoldBaseHashes(ch, 0, 2);
    ASSERT_EQ(folded_g1.size(), 2u);
    EXPECT_FALSE(pool.ContainsCachedBlock(MakeKeyWithGroupId(folded_g1[0], 1)));  // interior: skipped
    EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(folded_g1[1], 1)));   // aligned end: registered
    // full group unaffected: all 4 base blocks registered.
    for (const std::string& h : ch) EXPECT_TRUE(pool.ContainsCachedBlock(MakeKeyWithGroupId(h, 0)));
    coord.Free(tables);
}

// End-to-end: a second request sharing only a prefix converges BOTH groups to the same
// token boundary despite their different block sizes.
TEST(HeteroFoldedRegistrationTest, PartialPrefixConvergesInTokens) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 8, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/16));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/16);

    // Second request shares base pages 0,1 (tokens 0..8) then diverges.
    std::vector<std::string> other = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {9, 9, 9, 9}, {8, 8, 8, 8}});
    CoordinatorMatch m = ProbeAndAcquirePrefix(coord, other).device;
    EXPECT_EQ(m.num_common_tokens, 8);            // 2 base pages of 4 tokens
    EXPECT_EQ(m.per_group[0].blocks.size(), 2u);  // 8 / 4
    EXPECT_EQ(m.per_group[1].blocks.size(), 1u);  // 8 / 8, one folded coarse block
    coord.Free(tables);
}

// End-to-end: after a full acquire/register/free cycle the pool returns to baseline
// (no leaked device blocks across heterogeneous group granularities).
TEST(HeteroFoldedRegistrationTest, FullCycleRestoresPoolBaseline) {
    BlockPool pool(64, true);
    std::vector<KvCacheSpec> specs = {{AttnKind::kFull, 4, 0}, {AttnKind::kFull, 8, 0}};
    KvCacheCoordinator coord = MakeCoordinator(specs, pool);
    const std::int32_t free_before = pool.NumFreeBlocks();
    std::vector<BlockTable> tables(coord.NumGroups());
    ASSERT_TRUE(coord.Acquire(tables, /*num_tokens=*/16));
    std::vector<std::string> ch = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    coord.CacheFullBlocks(tables, ch, /*first_slot=*/0, /*end_tokens=*/16);
    coord.Free(tables);
    // Cached-but-free blocks stay in the free list (evictable), so the count returns to baseline.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before);
}

TEST(HeterogeneousPoolCoordinatorTest, DemandAndAllocationUseCanonicalPoolIndices) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 4, .bytes_per_block = 32},
    });
    const PoolIndex history = pools.IndexOf("history");
    const PoolIndex state = pools.IndexOf("state");
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{AttnKind::kFull, 4, 0, history},
        KvCacheSpec{AttnKind::kFull, 4, 0, state},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);

    KvCacheCoordinator::FreshDemandPlan demand =
        coordinator.PlanFreshDemand(/*chunk_tokens=*/4, /*protected_tokens=*/8);
    EXPECT_EQ(demand.ChunkDemand(), (PoolDemand{1, 1}));
    EXPECT_EQ(demand.ProtectedDemand(), (PoolDemand{2, 2}));
    EXPECT_EQ(demand.ReservationDemand(), (PoolDemand{1, 1}));

    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/4));
    ASSERT_EQ(tables[0].NumBlocks(), 1);
    ASSERT_EQ(tables[1].NumBlocks(), 1);
    const BlockRef& history_block = tables[0].Blocks().front();
    const BlockRef& state_block = tables[1].Blocks().front();
    EXPECT_EQ(history_block->BlockId(), 1);
    EXPECT_EQ(state_block->BlockId(), 1);
    EXPECT_NE(&*history_block, &*state_block);

    EXPECT_EQ(coordinator.BlocksNeededByPool(tables, /*num_tokens=*/4), (PoolDemand{1, 1}));
    coordinator.Free(tables);
}

TEST(HeterogeneousPoolCoordinatorTest, ShortPoolMakesCrossPoolAcquireAtomic) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 2, .bytes_per_block = 32},
    });
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("history")},
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("state")},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    const PoolDemand free_before = pools.FreeBlocks();

    EXPECT_EQ(coordinator.BlocksNeededByPool(tables, /*num_tokens=*/8), (PoolDemand{2, 2}));
    EXPECT_FALSE(coordinator.Acquire(tables, /*num_tokens=*/8));
    EXPECT_EQ(tables[0].NumBlocks(), 0);
    EXPECT_EQ(tables[1].NumBlocks(), 0);
    EXPECT_EQ(pools.FreeBlocks(), free_before);
}

TEST(HeterogeneousPoolCoordinatorTest, PrefixClaimAndFreeReturnEveryPoolToBaseline) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 4, .bytes_per_block = 32},
    });
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("history")},
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("state")},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);
    const PoolDemand baseline = pools.FreeBlocks();
    std::vector<BlockTable> producer(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(producer, /*num_tokens=*/4));
    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    coordinator.CacheFullBlocks(producer, hashes, /*first_slot=*/0, /*end_tokens=*/4);
    coordinator.Free(producer);
    EXPECT_EQ(pools.FreeBlocks(), baseline);

    CoordinatorMatch hit = ProbeAndAcquirePrefix(coordinator, hashes).device;
    ASSERT_EQ(hit.num_common_tokens, 4);

    std::vector<BlockTable> consumer(coordinator.NumGroups());
    coordinator.ClaimCommonPrefix(consumer, std::move(hit));
    EXPECT_EQ(pools.FreeBlocks(), (PoolDemand{baseline[0] - 1, baseline[1] - 1}));
    coordinator.Free(consumer);
    EXPECT_EQ(pools.FreeBlocks(), baseline);
}

TEST(HeterogeneousPoolCoordinatorTest, FreeReliefCountsOnlyCapacityMadeImmediatelyReusable) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 4, .bytes_per_block = 32},
    });
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("history")},
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("state")},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);
    const PoolDemand baseline = pools.FreeBlocks();
    std::vector<BlockTable> producer(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(producer, /*num_tokens=*/4));
    EXPECT_EQ(coordinator.BlocksReleasedByFreeByPool(producer), (PoolDemand{1, 1}));

    const std::vector<std::string> hashes = ContentHashes({{1, 2, 3, 4}});
    coordinator.CacheFullBlocks(producer, hashes, /*first_slot=*/0, /*end_tokens=*/4);
    CoordinatorMatch hit = ProbeAndAcquirePrefix(coordinator, hashes).device;
    std::vector<BlockTable> consumer(coordinator.NumGroups());
    coordinator.ClaimCommonPrefix(consumer, std::move(hit));

    // Either request dropping its shared refs leaves refcount one, so neither
    // page becomes free yet and neither can relieve a bottleneck by itself.
    EXPECT_EQ(coordinator.BlocksReleasedByFreeByPool(producer), (PoolDemand{0, 0}));
    EXPECT_EQ(coordinator.BlocksReleasedByFreeByPool(consumer), (PoolDemand{0, 0}));

    coordinator.Free(consumer);
    EXPECT_EQ(coordinator.BlocksReleasedByFreeByPool(producer), (PoolDemand{1, 1}));
    coordinator.Free(producer);
    EXPECT_EQ(pools.FreeBlocks(), baseline);
}

TEST(HeterogeneousPoolCoordinatorTest, ReclaimExpiredUsesTheGroupsOwnPool) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 5, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 5, .bytes_per_block = 32},
    });
    const PoolIndex history = pools.IndexOf("history");
    const PoolIndex state = pools.IndexOf("state");
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{AttnKind::kFull, 2, 0, history},
        KvCacheSpec{AttnKind::kSlidingWindow, 2, 4, state},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);
    const PoolDemand baseline = pools.FreeBlocks();
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/6));

    coordinator.ReclaimExpired(tables, /*num_computed_tokens=*/5);

    EXPECT_TRUE(tables[0].Blocks()[0]);
    EXPECT_FALSE(tables[1].Blocks()[0]);
    EXPECT_EQ(pools.FreeBlocks(), (PoolDemand{baseline[0] - 3, baseline[1] - 2}));

    coordinator.Free(tables);
    EXPECT_EQ(pools.FreeBlocks(), baseline);
}

TEST(HeterogeneousPoolCoordinatorTest, RejectsInvalidPoolIndexAndSharedHostTier) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 4, .bytes_per_block = 128},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 4, .bytes_per_block = 32},
    });
    BlockPool host_pool(8);
    std::vector<KvCacheSpec> invalid = {
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.Size()},
    };
    EXPECT_THROW(MakeCoordinator(invalid, pools), std::out_of_range);

    std::vector<KvCacheSpec> heterogeneous = {
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("history")},
        KvCacheSpec{AttnKind::kFull, 4, 0, pools.IndexOf("state")},
    };
    EXPECT_THROW(MakeCoordinator(heterogeneous, pools, &host_pool), std::invalid_argument);
}

TEST(V4ContinuationPrefixTest, DeepestHistoryRequiresExactTerminalContinuationBundle) {
    BlockPool pool(/*total_num_blocks=*/32);
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{.kind = AttnKind::kFull,
                    .block_size = 8,
                    .sliding_window = 0,
                    .prefix_role = KvPrefixRole::kHistoryAnchor,
                    .table_layout = KvTableLayout::kAbsolute},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    const std::vector<std::string> history_keys = MakeFoldedGroupKeys(hashes, /*schema_index=*/0, /*fold=*/2);
    CacheKeys(pool, history_keys);
    CacheForGroup(pool, hashes[2], /*group_id=*/1);
    CacheForGroup(pool, hashes[3], /*group_id=*/1);

    CoordinatorMatch hit = ProbeAndAcquirePrefix(coordinator, hashes).device;
    ASSERT_EQ(hit.num_common_tokens, 16);
    ASSERT_EQ(hit.per_group.size(), 2u);
    EXPECT_EQ(hit.per_group[0].base_logical_page, 0);
    EXPECT_EQ(hit.per_group[0].blocks.size(), 2u);
    EXPECT_EQ(hit.per_group[1].base_logical_page, 2);
    ASSERT_EQ(hit.per_group[1].blocks.size(), 2u);
    EXPECT_TRUE(hit.per_group[1].blocks[0]);
    EXPECT_TRUE(hit.per_group[1].blocks[1]);

    std::vector<BlockTable> tables(coordinator.NumGroups());
    coordinator.ClaimCommonPrefix(tables, std::move(hit));
    EXPECT_EQ(tables[0].BaseLogicalPage(), 0);
    EXPECT_EQ(tables[0].LogicalEnd(), 2);
    EXPECT_EQ(tables[1].BaseLogicalPage(), 2);
    EXPECT_EQ(tables[1].LogicalEnd(), 4);
    coordinator.Free(tables);

    KvCacheCoordinator::PreparedPrefix prepared = coordinator.PreparePrefix(hashes);
    KvCacheCoordinator::FreshAllocationPlan allocation =
        coordinator.PlanFreshAllocation(/*chunk_tokens=*/4, /*protected_tokens=*/4);
    std::optional<KvCacheCoordinator::CommittedPrefix> committed =
        coordinator.CommitPreparedPrefix(std::move(prepared), std::move(allocation));
    ASSERT_TRUE(committed.has_value());
    ASSERT_EQ(committed->hit_tokens, 16);
    EXPECT_EQ(committed->tables[0].BaseLogicalPage(), 0);
    EXPECT_EQ(committed->tables[0].LogicalEnd(), 3);
    EXPECT_EQ(committed->tables[1].BaseLogicalPage(), 2);
    EXPECT_EQ(committed->tables[1].LogicalEnd(), 5);
    coordinator.Free(committed->tables);
}

TEST(V4ContinuationPrefixTest, PublicationSkipsExpiredCompactColumnsAndRemainsExactlyMatchable) {
    BlockPool pool(/*total_num_blocks=*/32);
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{
            .kind = AttnKind::kFull, .block_size = 4, .sliding_window = 0, .prefix_role = KvPrefixRole::kHistoryAnchor},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    std::vector<BlockTable> producer(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(producer, /*num_tokens=*/16));
    coordinator.ReclaimExpired(producer, /*num_computed_tokens=*/16);
    ASSERT_EQ(producer[0].BaseLogicalPage(), 0);
    ASSERT_EQ(producer[1].BaseLogicalPage(), 2);

    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    EXPECT_NO_THROW(coordinator.CacheFullBlocks(producer, hashes, /*first_slot=*/0, /*end_tokens=*/16));
    coordinator.Free(producer);

    const CoordinatorMatch hit = ProbeAndAcquirePrefix(coordinator, hashes).device;
    EXPECT_EQ(hit.num_common_tokens, 16);
    EXPECT_EQ(hit.per_group[1].base_logical_page, 2);
    EXPECT_EQ(hit.per_group[1].blocks.size(), 2u);
}

TEST(CoordinatorReadyPublicationTest, PartialCrossPoolPublicationCannotCreateCommonHit) {
    BlockPoolSet pools({
        FlatBlockPoolConfig{.pool_id = "history", .total_blocks = 16, .bytes_per_block = 64},
        FlatBlockPoolConfig{.pool_id = "state", .total_blocks = 16, .bytes_per_block = 32},
    });
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{.kind = AttnKind::kFull,
                    .block_size = 4,
                    .sliding_window = 0,
                    .pool_index = 0,
                    .prefix_role = KvPrefixRole::kHistoryAnchor,
                    .table_layout = KvTableLayout::kAbsolute},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .pool_index = 1,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pools);
    std::vector<BlockTable> tables(coordinator.NumGroups());
    ASSERT_TRUE(coordinator.Acquire(tables, /*num_tokens=*/16));
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});

    const std::vector<std::int32_t> initial_ends{0, 0};
    const std::vector<std::int32_t> history_ready{16, 0};
    coordinator.PublishReadyBlocks(tables, hashes, initial_ends, history_ready);
    EXPECT_EQ(ProbeAndAcquirePrefix(coordinator, hashes).device.num_common_tokens, 0);

    const std::vector<std::int32_t> all_ready{16, 16};
    coordinator.PublishReadyBlocks(tables, hashes, history_ready, all_ready);
    EXPECT_EQ(ProbeAndAcquirePrefix(coordinator, hashes).device.num_common_tokens, 16);
}

TEST(V4ContinuationPrefixTest, MissingDeepestContinuationFallsBackToRootNotShallowerSnapshot) {
    BlockPool pool(/*total_num_blocks=*/32);
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{
            .kind = AttnKind::kFull, .block_size = 4, .sliding_window = 0, .prefix_role = KvPrefixRole::kHistoryAnchor},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}});
    for (const std::string& hash : hashes) {
        CacheForGroup(pool, hash, /*group_id=*/0);
    }
    // Boundary 12 would be resumable from pages [1, 3), but the deepest
    // history boundary 16 additionally needs page 3. Exact continuation must
    // reject the whole hit instead of silently searching left.
    CacheForGroup(pool, hashes[1], /*group_id=*/1);
    CacheForGroup(pool, hashes[2], /*group_id=*/1);

    const CoordinatorMatch miss = ProbeAndAcquirePrefix(coordinator, hashes).device;
    EXPECT_EQ(miss.num_common_tokens, 0);
    ASSERT_EQ(miss.per_group.size(), 2u);
    EXPECT_TRUE(miss.per_group[0].blocks.empty());
    EXPECT_TRUE(miss.per_group[1].blocks.empty());
}

TEST(V4ContinuationPrefixTest, NonParticipatingGroupDoesNotConstrainHistoryBoundary) {
    BlockPool pool(/*total_num_blocks=*/24);
    std::vector<KvCacheSpec> specs = {
        KvCacheSpec{
            .kind = AttnKind::kFull, .block_size = 4, .sliding_window = 0, .prefix_role = KvPrefixRole::kHistoryAnchor},
        KvCacheSpec{.kind = AttnKind::kFull, .block_size = 4, .sliding_window = 0, .prefix_role = KvPrefixRole::kNone},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    const std::vector<std::string> hashes = ContentHashes({{0, 0, 0, 0}, {1, 1, 1, 1}});
    CacheForGroup(pool, hashes[0], /*group_id=*/0);
    CacheForGroup(pool, hashes[1], /*group_id=*/0);

    const CoordinatorMatch hit = ProbeAndAcquirePrefix(coordinator, hashes).device;
    EXPECT_EQ(hit.num_common_tokens, 8);
    EXPECT_EQ(hit.per_group[0].blocks.size(), 2u);
    EXPECT_TRUE(hit.per_group[1].blocks.empty());
}

TEST(V4ContinuationPrefixTest, RejectsContinuationWithoutAlignedHistoryAnchor) {
    BlockPool pool(/*total_num_blocks=*/16);
    std::vector<KvCacheSpec> no_history = {
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    EXPECT_THROW(MakeCoordinator(no_history, pool), std::invalid_argument);

    std::vector<KvCacheSpec> misaligned = {
        KvCacheSpec{
            .kind = AttnKind::kFull, .block_size = 6, .sliding_window = 0, .prefix_role = KvPrefixRole::kHistoryAnchor},
        KvCacheSpec{.kind = AttnKind::kSlidingWindow,
                    .block_size = 4,
                    .sliding_window = 8,
                    .prefix_role = KvPrefixRole::kContinuationState,
                    .table_layout = KvTableLayout::kBoundedWindow},
    };
    EXPECT_THROW(MakeCoordinator(misaligned, pool), std::invalid_argument);
}

}  // namespace
}  // namespace tokenspeed::test
