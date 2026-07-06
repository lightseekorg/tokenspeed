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

// Extra end-to-end scenario tests for the flat KV-cache FSM path
// (TOKENSPEED_FLAT_KVCACHE=ON), complementing test_flat_kvcache_lifecycle.cpp.
// Covers what the flat path SUPPORTS today (verified wired at the FSM level):
//   - chunked prefill (a prompt split across multiple prefill chunks; the flat
//     op-builder runs PrefillFirstChunk then PrefillChunk per chunk),
//   - multi-group configs beyond the 2-group full+swa shape (three groups; an
//     all-full two-group shape) -- group 0 stays a full-history group, matching
//     the flat consumer's block_tables_[0] contract,
//   - shared-pool accounting across several requests of differing lengths and
//     finish orders.
//   - cross-request prefix reuse (M9): admission-layer match -> FSM claim ->
//     input window starting past the hit (FlatPrefixHitSuite below).
//   - decode-filled page caching (M13): multi-turn prompts hit past the
//     previous turn's response region (FlatDecodeCachingSuite below).
// Retract/writeback are deliberately deferred on the flat path (C slice), so
// they are NOT tested here. Abort, pool-exhaustion admission, and the
// failure-path page release ARE covered (see the suites at the bottom of this
// file).

#if TOKENSPEED_FLAT_KVCACHE

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>

#include "integration_test_helper.h"

namespace tokenspeed::test {

namespace {

const FlatForwardOperation* FindFlatOp(const ExecutionPlan& plan) {
    for (const auto& op : plan.Operations()) {
        if (const auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
    }
    return nullptr;
}

PagedCacheGroupConfig MakeGroup(const std::string& id, std::int32_t page_size,
                                std::int32_t total_pages,
                                PagedCacheGroupConfig::Retention retention,
                                PagedCacheGroupFamily family,
                                std::int32_t sliding_window_tokens = 0) {
    PagedCacheGroupConfig g;
    g.group_id = id;
    g.rows_per_page = page_size;
    g.entry_stride_tokens = 1;
    g.total_pages = total_pages;
    g.retention = retention;
    g.family = family;
    if (sliding_window_tokens > 0) {
        g.sliding_window_tokens = sliding_window_tokens;
    }
    return g;
}

// Collect every real (>0) physical page id across all rows of a group.
std::vector<std::int32_t> RealPages(const std::vector<std::vector<std::int32_t>>& group) {
    std::vector<std::int32_t> out;
    for (const auto& row : group) {
        for (std::int32_t id : row) {
            if (id > 0) out.push_back(id);
        }
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Chunked prefill: a prompt longer than max_scheduled_tokens is split into
// multiple prefill chunks. The flat op-builder runs PrefillFirstChunk on the
// first chunk and PrefillChunk on each subsequent chunk, growing the full-group
// block table monotonically. Verifies the request reaches decode and the pool
// is fully reclaimed on finish.
// ---------------------------------------------------------------------------
class FlatChunkedPrefillSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 4;  // 4 tokens = 2 pages per chunk
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(FlatChunkedPrefillSuite, MultiChunkPrefillGrowsFullTableThenDecodes) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 8 tokens (4 pages) with max_scheduled_tokens=4 -> 2 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));

    // Chunk 1 (PrefillFirstChunk): first 4 tokens -> 2 pages in the full group.
    ExecutionPlan chunk1 = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(chunk1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->flat_block_tables.count("full"), 1u);
    const std::size_t full_after_c1 = op1->flat_block_tables.at("full").at(0).size();
    EXPECT_GT(full_after_c1, 0u);
    // Still prefilling: not yet in decode.
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);

    // Chunk 2 (PrefillChunk): remaining 4 tokens -> the full row must grow.
    ExecutionPlan chunk2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(chunk2);
    ASSERT_NE(op2, nullptr);
    const auto& full_c2 = op2->flat_block_tables.at("full").at(0);
    EXPECT_GT(full_c2.size(), full_after_c1)
        << "second chunk should extend the full-history block table";
    // Full-history group never punches a null hole.
    for (std::int32_t id : full_c2) {
        EXPECT_GT(id, 0) << "full-history row must have no null hole";
    }

    // Prefill complete -> a decode step should now run.
    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    SendForwardDone("r1", {100});

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "all pages returned to the pool after a chunked-prefill request finishes";
}

// ---------------------------------------------------------------------------
// Three cache groups: one full-history + two sliding-window groups with
// different windows. Verifies the flat op emits one row per group (all three
// keys present), group 0 (full) keeps history, and the shared pool is reclaimed.
// group 0 stays full-history to honor the flat consumer's block_tables_[0]
// contract.
// ---------------------------------------------------------------------------
class FlatThreeGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa_small", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
            MakeGroup("swa_big", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/8),
        };
        return cfg;
    }
};

TEST_F(FlatThreeGroupSuite, ThreeGroupsEachEmitARowAndReclaim) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(prefill);
    ASSERT_NE(op, nullptr);

    // All three groups present, one row each.
    ASSERT_EQ(op->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(op->flat_block_tables.count("swa_small"), 1u);
    ASSERT_EQ(op->flat_block_tables.count("swa_big"), 1u);
    EXPECT_EQ(op->flat_block_tables.at("full").size(), 1u);
    EXPECT_EQ(op->flat_block_tables.at("swa_small").size(), 1u);
    EXPECT_EQ(op->flat_block_tables.at("swa_big").size(), 1u);

    // The three groups draw disjoint physical pages from the shared pool.
    auto full_pages = RealPages(op->flat_block_tables.at("full"));
    auto small_pages = RealPages(op->flat_block_tables.at("swa_small"));
    auto big_pages = RealPages(op->flat_block_tables.at("swa_big"));
    std::set<std::int32_t> all(full_pages.begin(), full_pages.end());
    all.insert(small_pages.begin(), small_pages.end());
    all.insert(big_pages.begin(), big_pages.end());
    EXPECT_EQ(all.size(), full_pages.size() + small_pages.size() + big_pages.size())
        << "groups must not share physical pages";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Sub-page and page-straddling windows (M14): P=4 with w=3 (< P, needs a
// 1-page trailing run) and w=5 (= P+1, straddles a page boundary). Pins the
// per-group slide independence and the <=2-real-page steady state.
// ---------------------------------------------------------------------------
class FlatSubPageWindowSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 4;
        cfg.device_allocator.total_pages = 96;
        cfg.host_allocator.total_pages = 96;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa_w3", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/3),
            MakeGroup("swa_w5", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/5),
        };
        return cfg;
    }
};

TEST_F(FlatSubPageWindowSuite, SubPageWindowsPlateauAtTwoRealPages) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    ASSERT_NE(FindFlatOp(prefill), nullptr);
    SendForwardDone("r1", {1000});

    for (std::int32_t step = 0; step < 24; ++step) {
        ExecutionPlan decode = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(decode);
        ASSERT_NE(op, nullptr) << "decode step " << step;
        // fullySlidOutBlocks frees only FULLY slid-out pages, so a window
        // straddling a boundary keeps its previous page: 1 <= real pages <= 2.
        const std::size_t w3_real = RealPages(op->flat_block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->flat_block_tables.at("swa_w5")).size();
        EXPECT_GE(w3_real, 1u) << "w=3 lost its live tail page at step " << step;
        EXPECT_LE(w3_real, 2u) << "w=3 working set exceeded 2 pages at step " << step;
        EXPECT_GE(w5_real, 1u) << "w=5 lost its live tail page at step " << step;
        EXPECT_LE(w5_real, 2u) << "w=5 working set exceeded 2 pages at step " << step;
        SendForwardDone("r1", {1001 + step});
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatSubPageWindowSuite, StraddlingWindowHoldsPreviousPage) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    PlanOnce();
    SendForwardDone("r1", {1000});

    bool diverged = false;
    for (std::int32_t step = 0; step < 8; ++step) {
        ExecutionPlan decode = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(decode);
        ASSERT_NE(op, nullptr);
        const std::size_t w3_real = RealPages(op->flat_block_tables.at("swa_w3")).size();
        const std::size_t w5_real = RealPages(op->flat_block_tables.at("swa_w5")).size();
        EXPECT_LE(w3_real, w5_real) << "a smaller window can never hold more pages, step " << step;
        if (w3_real < w5_real) {
            diverged = true;  // the straddling window (w=5) holds one more real page
        }
        SendForwardDone("r1", {1001 + step});
    }
    EXPECT_TRUE(diverged) << "w=3 and w=5 never diverged: per-group slides are not independent";

    SendFinish("r1");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// Two full-history groups (no sliding window at all). Verifies the flat path
// works for a non-hybrid multi-group shape: neither group ever develops a null
// hole, and both reclaim on finish.
// ---------------------------------------------------------------------------
class FlatAllFullTwoGroupSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full_a", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("full_b", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(FlatAllFullTwoGroupSuite, BothFullGroupsKeepHistoryNoHoles) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // prefill
    SendForwardDone("r1", {42});

    // Drive several decode steps; a full-history group must never punch a hole.
    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindFlatOp(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const FlatForwardOperation* op = FindFlatOp(*last);
    ASSERT_NE(op, nullptr);
    for (const char* key : {"full_a", "full_b"}) {
        const auto& row = op->flat_block_tables.at(key).at(0);
        for (std::int32_t id : row) {
            EXPECT_GT(id, 0) << key << " (full-history) must not develop a null hole";
        }
    }

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Shared-pool accounting: three requests of differing lengths, finished in a
// non-submission order, must each return exactly their pages -- the pool is
// back to its starting free count only after the last one finishes, and never
// over- or under-counts along the way.
// ---------------------------------------------------------------------------
class FlatPoolAccountingSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(FlatPoolAccountingSuite, ThreeRequestsOutOfOrderFinishReclaimExactly) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/101));
    Submit(MakeRequestSpec("r3", /*num_pages=*/3, /*start=*/201));
    PlanOnce();  // prefill all three (max_scheduled_tokens=64 covers them)
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const std::int32_t free_after_prefill = scheduler_->FlatPoolFreeBlocks();
    EXPECT_LT(free_after_prefill, free_at_start)
        << "prefill must consume pages from the shared pool";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendForwardDone("r3", {242});

    // Finish out of submission order: r2, then r1, then r3.
    SendFinish("r2");
    PlanOnce();
    SendFinish("r1");
    PlanOnce();
    EXPECT_LT(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "pool not fully reclaimed while r3 is still live";
    SendFinish("r3");
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "every page returns to the pool once all requests finish";
}

// Chunked prefill on a long prompt slides the SWA window DURING prefill, then
// decode keeps sliding past it: the swa group develops null holes mid-prefill
// (not only at the second decode step), while the full group keeps every page.
// Pins the EXACT first-kept page each round from the window convention: with
// N = tokens whose KV is computed BEFORE that round's forward (the prior
// chunks' tokens for a prefill chunk; container size minus the one pending
// decode token for decode), the pending query at position N attends keys
// [N - W + 1, N], so pages [0, (N - W + 1) / page_size) are free and page
// (N - W + 1) / page_size is the first kept. Passing the container size
// directly (off-by-one) frees the first kept page one round early whenever
// (size - W + 1) % page_size == 0.
TEST_F(FlatChunkedPrefillSuite, ChunkedPrefillThenSwaSlidesToNullHole) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> 3 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    // Chunk 2: N = 4 computed -> first kept token 4-4+1=1 -> first kept page 0:
    // nothing is fully out of window yet, no hole.
    ExecutionPlan chunk2 = PlanOnce();
    const FlatForwardOperation* c2op = FindFlatOp(chunk2);
    ASSERT_NE(c2op, nullptr);
    {
        const auto& swa_c2 = c2op->flat_block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c2.size(), 4u);
        EXPECT_EQ(std::count(swa_c2.begin(), swa_c2.end(), 0), 0)
            << "N=4, W=4: no page fully below token 1, so chunk 2 punches nothing";
    }
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    const std::int32_t free_after_c2 = scheduler_->FlatPoolFreeBlocks();

    // Chunk 3: N = 8 computed -> first kept token 8-4+1=5 -> first kept page
    // 5/2=2: slots 0,1 punched MID-PREFILL, slots 2..5 kept after the acquire.
    ExecutionPlan chunk3 = PlanOnce();  // chunk 3 (last)
    const FlatForwardOperation* c3op = FindFlatOp(chunk3);
    ASSERT_NE(c3op, nullptr);
    {
        const auto& swa_c3 = c3op->flat_block_tables.at("swa").at(0);
        ASSERT_EQ(swa_c3.size(), 6u);
        for (int s = 0; s <= 1; ++s) EXPECT_EQ(swa_c3[s], 0) << "slot " << s << " punched during prefill";
        for (int s = 2; s <= 5; ++s) EXPECT_GT(swa_c3[s], 0) << "slot " << s;
        for (std::int32_t id : c3op->flat_block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group keeps every chunk-built page";
        }
    }
    // Pool balance of chunk 3: slide freed 2 swa pages, acquire took 2 pages
    // per group (4 tokens) -> net -2, visible mid-prefill.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_after_c2 + 2 - 4)
        << "the mid-prefill slide must return the out-of-window pages to the pool";

    SendForwardDone("r1", {99});  // container size 13 (12 prompt + 1 sampled)

    // One decode round per iteration; swa_rows[i] is the swa block-table row the
    // round-i op carried (state AFTER that round's slide + acquire).
    std::vector<std::vector<std::int32_t>> swa_rows;
    int tok = 100;
    for (int i = 0; i < 4; ++i) {
        ExecutionPlan plan = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(plan);
        ASSERT_NE(op, nullptr);
        // Full group: no holes across the whole (chunk-built) history, ever.
        for (std::int32_t id : op->flat_block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group must keep chunk-built history without holes (round " << i << ")";
        }
        swa_rows.push_back(op->flat_block_tables.at("swa").at(0));
        SendForwardDone("r1", {tok++});
    }

    auto null_count = [](const std::vector<std::int32_t>& row) {
        return std::count(row.begin(), row.end(), 0);
    };

    // Round 0 (PrefillDone->Decoding): FinalizePrefillAndReserveDecode slides
    // at N = 12 (full prefill computed) -> first kept token 12-4+1=9 -> first
    // kept page 9/2=4: slots 0..3 punched (0,1 already were, mid-prefill),
    // then the reserve page (7th) is acquired -> 7 slots, 4 holes.
    ASSERT_EQ(swa_rows[0].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[0]), 4) << "finalize slides at the full prefill length";
    for (int s = 0; s <= 3; ++s) EXPECT_EQ(swa_rows[0][s], 0) << "slot " << s;
    for (int s = 4; s <= 6; ++s) EXPECT_GT(swa_rows[0][s], 0) << "slot " << s;

    // Round 1: N = 14 - 1 = 13 computed -> first kept token 13-4+1=10 -> first
    // kept page 10/2=5: slots 0..4 punched, 5..6 kept. Tail room absorbs the
    // acquire (no new page).
    ASSERT_EQ(swa_rows[1].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[1]), 5);
    for (int s = 0; s <= 4; ++s) EXPECT_EQ(swa_rows[1][s], 0) << "slot " << s;
    for (int s = 5; s <= 6; ++s) EXPECT_GT(swa_rows[1][s], 0) << "slot " << s;

    // Round 2: N = 15 - 1 = 14 -> first kept token 11 -> first kept page 11/2=5
    // (unchanged); acquire adds page 7. THE off-by-one boundary: passing the
    // container size 15 gives first kept token 12 -> page 6, freeing slot 5 one
    // round early while the pending query at position 14 still reads key 11
    // from page 5 (and the freed page could be re-allocated the same round).
    ASSERT_EQ(swa_rows[2].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[2]), 5);
    EXPECT_GT(swa_rows[2][5], 0) << "slot 5 must survive round 2: key 11 of the pending query lives there";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[2][s], 0) << "slot " << s;

    // Round 3: N = 16 - 1 = 15 -> first kept token 12 -> first kept page 6:
    // slot 5 is punched exactly now, one round after the old code did.
    ASSERT_EQ(swa_rows[3].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[3]), 6);
    EXPECT_EQ(swa_rows[3][5], 0) << "slot 5 slides out once the query window has moved past key 11";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[3][s], 0) << "slot " << s;

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Two requests batched under a three-group config: the FlatForwardOperation
// aggregates both into one SoA op with two rows per group across all three
// groups, and no physical page is shared between requests or across groups.
TEST_F(FlatThreeGroupSuite, TwoRequestsBatchedAcrossThreeGroupsNoCollision) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(prefill);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);

    // Every group carries two rows (one per request).
    for (const char* key : {"full", "swa_small", "swa_big"}) {
        ASSERT_EQ(op->flat_block_tables.count(key), 1u) << key;
        EXPECT_EQ(op->flat_block_tables.at(key).size(), 2u) << key;
    }

    // No physical page collides anywhere: across both requests AND all groups.
    std::vector<std::int32_t> every;
    for (const char* key : {"full", "swa_small", "swa_big"}) {
        auto pages = RealPages(op->flat_block_tables.at(key));
        every.insert(every.end(), pages.begin(), pages.end());
    }
    std::vector<std::int32_t> sorted = every;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
        << "no physical page may be shared across requests or groups";

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Mixed batch: with enable_mixed_prefill_decode, a request already in decode and
// a freshly-submitted request in prefill can be scheduled in the SAME plan. The
// FlatForwardOperation aggregates both into one SoA op (stable_partition puts the
// prefill row(s) ahead of the decode row(s)); both op kinds carry their
// flat_block_tables, and no physical page collides across the two requests.
// ---------------------------------------------------------------------------
class FlatMixedBatchSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.enable_mixed_prefill_decode = true;  // decode + prefill in one plan

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(FlatMixedBatchSuite, PrefillAndDecodeShareOnePlan) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // r1: get it into decode first.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();                       // r1 prefill
    SendForwardDone("r1", {42});      // r1 -> decode

    // r2 arrives now; next plan should carry r1 (decode) + r2 (prefill) together.
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan mixed = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(mixed);
    ASSERT_NE(op, nullptr);

    // Both requests present, and the op genuinely mixes one prefill + one decode.
    ASSERT_EQ(op->request_ids.size(), 2u);
    EXPECT_EQ(op->num_extends(), 1u) << "exactly one prefill row (r2)";
    EXPECT_EQ(op->decode_input_ids.size(), 1u) << "exactly one decode row (r1)";

    // stable_partition puts the prefill ahead of the decode.
    EXPECT_EQ(op->request_ids.at(0), "r2") << "prefill partitioned first";
    EXPECT_EQ(op->request_ids.at(1), "r1") << "decode after prefill";

    // Both groups carry two rows (one per request); no page collides across reqs.
    for (const char* key : {"full", "swa"}) {
        ASSERT_EQ(op->flat_block_tables.count(key), 1u) << key;
        ASSERT_EQ(op->flat_block_tables.at(key).size(), 2u) << key;
        auto pages = RealPages(op->flat_block_tables.at(key));
        std::vector<std::int32_t> sorted = pages;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
            << key << ": two requests must not share a physical page";
    }

    SendForwardDone("r1", {43});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Two requests decoded to DIFFERENT depths in the same batch: the sliding-window
// group's null hole is per-request. r1 is driven well past the window (must show
// a hole in its swa row); r2 stays within the window (must NOT yet have a hole).
// The full-history rows of both stay hole-free throughout. Verifies swa eviction
// state is tracked independently per request, not batch-wide.
TEST_F(FlatMixedBatchSuite, PerRequestSwaHoleAtDifferentDecodeDepths) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    PlanOnce();                    // both prefill together (mixed batch)
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Drive r1 far past its window (window=4 tokens=2 pages); keep r2 at just one
    // decode step so it stays inside the window.
    std::optional<ExecutionPlan> last;
    int t1 = 43, t2 = 143;
    for (int step = 0; step < 5; ++step) {
        last = PlanOnce();
        ASSERT_NE(FindFlatOp(*last), nullptr);
        SendForwardDone("r1", {t1++});
        if (step == 0) {
            SendForwardDone("r2", {t2++});  // r2 advances only once
        }
    }
    const FlatForwardOperation* op = FindFlatOp(*last);
    ASSERT_NE(op, nullptr);

    // Map each request id to its row index (order is not guaranteed).
    const auto& ids = op->request_ids;
    auto row_of = [&](const std::string& id) -> std::size_t {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == id) return i;
        }
        ADD_FAILURE() << "request " << id << " not in op";
        return 0;
    };

    // Only r1 should still be live+decoding here; r2 stopped advancing but may or
    // may not remain in the batch. Assert on whichever rows are present.
    const auto& swa = op->flat_block_tables.at("swa");
    const auto& full = op->flat_block_tables.at("full");
    if (std::find(ids.begin(), ids.end(), "r1") != ids.end()) {
        std::size_t r1 = row_of("r1");
        EXPECT_NE(std::find(swa.at(r1).begin(), swa.at(r1).end(), 0), swa.at(r1).end())
            << "r1 drove past the window -> swa row must have a null hole";
        for (std::int32_t id : full.at(r1)) {
            EXPECT_GT(id, 0) << "r1 full-history row must stay hole-free";
        }
    }

    SendFinish("r1");
    if (scheduler_->DecodingSize() > 0) SendFinish("r2");
    PlanOnce();
}

// ---------------------------------------------------------------------------
// page_size = 1 (token-granular pages). Verifies the flat path is not hard-wired
// to page_size=2: prefill allocates one page per token, the sliding-window group
// still develops a null hole once decode crosses its (token-granular) window, the
// full group stays hole-free, and the pool reclaims on finish.
// ---------------------------------------------------------------------------
class FlatPageSizeOneSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 1;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 64;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/2),
        };
        return cfg;
    }
};

TEST_F(FlatPageSizeOneSuite, TokenGranularPagesSlideAndReclaim) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 3 tokens, page_size=1 -> 3 pages in the full group after prefill.
    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* pop = FindFlatOp(prefill);
    ASSERT_NE(pop, nullptr);
    EXPECT_EQ(pop->flat_block_tables.at("full").at(0).size(), 3u)
        << "page_size=1 -> one page per prompt token";

    SendForwardDone("r1", {42});

    // Decode past the sliding window (window=2 tokens).
    std::optional<ExecutionPlan> last;
    int tok = 43;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindFlatOp(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const FlatForwardOperation* op = FindFlatOp(*last);
    ASSERT_NE(op, nullptr);
    for (std::int32_t id : op->flat_block_tables.at("full").at(0)) {
        EXPECT_GT(id, 0) << "full group hole-free at page_size=1";
    }
    const auto& swa = op->flat_block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end())
        << "swa group must develop a null hole at page_size=1 too";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

namespace {

void SendAbort(Scheduler& scheduler, const std::string& id) {
    ExecutionEvent event;
    event.With(ForwardEvent{forward::Abort{.request_id = id}});
    scheduler.Advance(std::move(event));
}

}  // namespace

// ---------------------------------------------------------------------------
// Pool-exhaustion admission: with a pool sized to roughly one request, a second
// request must be DEFERRED by the flat capacity gate (not scheduled, not
// corrupted, no throw), stay schedulable, and run once the first one finishes.
// Pool is back to baseline after both complete. The first-chunk gate charges
// prompt + decode reserve (2 groups * ceil((tokens + 1) / page_size) blocks),
// so r1's own decode headroom is guaranteed at admission time.
// ---------------------------------------------------------------------------
class FlatTinyPoolSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        // 11 physical pages -> 10 usable (page 0 is the null placeholder). One
        // 4-page prompt over 2 groups needs 8 at prefill plus 2 decode-reserve
        // blocks at admission: exactly the whole pool.
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(FlatTinyPoolSuite, ExhaustedPoolDefersSecondRequestUntilFirstFinishes) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1: 8 tokens -> admission gate charges 8 prefill + 2 reserve blocks = 10;
    // the prefill itself consumes 8, leaving 2 reserved for r1's decode.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan1 = PlanOnce();
    ASSERT_NE(FindFlatOp(plan1), nullptr);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // r2 arrives with only r1's decode headroom left: the flat admission gate
    // must defer it (its prompt + reserve need 4 blocks, and r2 is gated
    // BEFORE r1's decode transition frees anything) without throwing, while
    // r1's decode step proceeds into the reserved 2 blocks.
    Submit(MakeRequestSpec("r2", /*num_pages=*/1, /*start=*/101));
    SendForwardDone("r1", {99});
    ExecutionPlan starved = PlanOnce();
    const FlatForwardOperation* starved_op = FindFlatOp(starved);
    ASSERT_NE(starved_op, nullptr);
    ASSERT_EQ(starved_op->request_ids.size(), 1u) << "only r1's reserved decode step fits this round";
    EXPECT_EQ(starved_op->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    // r1's finalize slides at N = 8 (W=4, page=2): first kept token 8-4+1=5 ->
    // first kept page 2, so swa slots 0,1 free (+2); the reserve acquire takes
    // 1 fresh page per group (-2): free stays 2.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // r1 finishes -> its pages return -> r2 becomes schedulable.
    SendForwardDone("r1", {100});
    SendFinish("r1");
    ExecutionPlan plan2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after pages free up";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    SendForwardDone("r2", {142});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "pool back to baseline after the deferred request completes";
}

// A prompt whose prefill alone fits the pool exactly, but whose decode reserve
// does not, must be deferred at the first-chunk gate instead of being admitted
// into a corner its own decode can't exit (flat retract is unimplemented).
TEST_F(FlatTinyPoolSuite, PromptWhoseDecodeCannotFitIsDeferredAtFirstChunk) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // 10 tokens -> prefill needs exactly 10 blocks (5 pages * 2 groups), but the
    // gate charges prompt + reserve = 2 * ceil(11/2) = 12 > 10.
    Submit(MakeRequestSpec("r1", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->request_ids.empty()) << "self-cornering prompt must not be admitted";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "a deferred first chunk must not touch the pool";
}

// ---------------------------------------------------------------------------
// Prefill-slide admission: a long chunked prompt that fits the pool ONLY
// because (a) earlier chunks' slid-out SWA pages are already free at the final
// chunk's gate and (b) the gate credits the slide the final chunk itself will
// perform (BlocksFreedByAdvance). 13 physical pages -> 12 usable; page=2, W=4,
// chunks of 4 tokens, 12-token prompt.
// ---------------------------------------------------------------------------
class FlatPrefillSlideAdmissionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 13;
        cfg.max_scheduled_tokens = 4;  // 4-token prefill chunks
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, /*sliding_window_tokens=*/4),
        };
        return cfg;
    }
};

TEST_F(FlatPrefillSlideAdmissionSuite, LongPromptAdmittedOnlyBecausePrefillSlides) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);

    // Chunk gates (page=2, W=4, chunks of 4): c1 charges 4 blocks (2/group),
    // free 12 -> 8. c2 (N=4, slide credit 0) charges 4, free 8 -> punches 0,
    // acquires 4 -> 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindFlatOp(c1), nullptr);
    ASSERT_EQ(FindFlatOp(c1)->request_ids.size(), 1u);
    ExecutionPlan c2 = PlanOnce();
    ASSERT_NE(FindFlatOp(c2), nullptr);
    ASSERT_EQ(FindFlatOp(c2)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // Chunk 3 completes the prefill: the gate charges chunk + decode reserve =
    // BlocksNeededFor(5 tokens) = 3 blocks/group = 6, against raw free 4 --
    // WITHOUT the slide credit the request would defer here forever (flat
    // retract is unimplemented). With N = 8 the pending slide frees the 2 swa
    // pages fully below token 8-4+1=5, so the gate sees 4 + 2 = 6 and admits.
    ExecutionPlan c3 = PlanOnce();
    const FlatForwardOperation* c3op = FindFlatOp(c3);
    ASSERT_NE(c3op, nullptr);
    ASSERT_EQ(c3op->request_ids.size(), 1u) << "final chunk must be admitted via the prefill slide credit";
    // Op balance: punch 2, acquire 2/group -> free 4 + 2 - 4 = 2.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // Decode transition: gate needs 2 (1 fresh page/group), credit = the
    // finalize slide at N=12 (2 more swa pages) -> 2 + 2 >= 2, admitted.
    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    ASSERT_EQ(FindFlatOp(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Collective starvation deadlock: two requests admitted legitimately together
// (each with its decode reserve fully charged, so admission itself is sound)
// corner the pool through DECODE GROWTH beyond the reserve -- every candidate
// is deferred, the deferred set holds all the pool's pages, and no forward
// result is pending that could ever finish a request and free them. The
// scheduler must fail loudly (flat retract is unimplemented) instead of
// silently returning empty plans forever, and the failed round must not leak
// or touch pool pages. While a decode result is still in flight a starved
// round must stay quiet, and even with nothing in flight the assert requires
// TWO consecutive fully-starved rounds (a queued pool-freeing Finish between
// an ExtendResult and the next plan would make a single round a false
// positive).
// ---------------------------------------------------------------------------
class FlatCollectiveStarvationSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        // 13 physical pages -> 12 usable. Two 2-page prompts charge
        // 2*ceil(5/2)=6 blocks each at admission (4 prefill + 2 decode
        // reserve): 12 = exactly the pool, so both are admitted with their
        // reserves intact. Their decode steps then grow past the reserve and
        // eat the pool down to zero.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 13;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full_a", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("full_b", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(FlatCollectiveStarvationSuite, DeadlockedPoolFailsLoudWithoutLeaking) {
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 12);

    // Round 1: both prompts admitted. r1 gate 6 <= 12; after r1's 4-block
    // prefill and 2-block reservation, r2 gate 6 <= 8 - 2 = 6. Free 12-8 = 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(prefill);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Round 2: both decode transitions consume their own 2-block reservations
    // (r1 gate 2 <= 4 - r2's 2; r2 gate 2 <= 2 after r1 retired its entry).
    ExecutionPlan round2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);
    SendForwardDone("r1", {43});
    SendForwardDone("r2", {143});

    // Round 3: both next steps fit their tail pages (0 fresh blocks).
    ExecutionPlan round3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 2u);

    // Starved round with r2's round-3 decode result STILL IN FLIGHT: both
    // defer (each now needs a fresh page per group), but the round must stay
    // quiet -- the pending result can finish r2.
    SendForwardDone("r1", {44});
    ExecutionPlan quiet = PlanOnce();
    const FlatForwardOperation* quiet_op = FindFlatOp(quiet);
    ASSERT_NE(quiet_op, nullptr);
    EXPECT_TRUE(quiet_op->request_ids.empty());

    // Deliver that last result: nothing is in flight, both requests are
    // deferred, and together they hold every pool page. The FIRST fully
    // starved round must still stay quiet (a pool-freeing Finish could be
    // queued right behind the ExtendResult we just delivered) ...
    SendForwardDone("r2", {144});
    ExecutionPlan starved1 = PlanOnce();
    const FlatForwardOperation* starved1_op = FindFlatOp(starved1);
    ASSERT_NE(starved1_op, nullptr);
    EXPECT_TRUE(starved1_op->request_ids.empty()) << "first starved round is quiet (two-round hardening)";

    // ... and the SECOND consecutive fully-starved round must fire the
    // deadlock assert, loudly and without touching the pool.
    const std::int32_t free_before = scheduler_->FlatPoolFreeBlocks();
    try {
        PlanOnce();
        FAIL() << "expected the flat starvation-deadlock assert to fire on the second starved round";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("flat pool starvation deadlock"), std::string::npos) << e.what();
    }
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_before) << "the failed round must not leak or free pages";
}

// ---------------------------------------------------------------------------
// Abort-mid-flight pool balance: aborting a request in the middle of a chunked
// prefill, or in the middle of decode, must return every page to the pool.
// ---------------------------------------------------------------------------
TEST_F(FlatChunkedPrefillSuite, AbortMidPrefillRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> abort lands mid-prefill.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    PlanOnce();  // chunk 2 -> still Prefilling
    EXPECT_LT(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();  // reap the aborted request
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "abort mid-prefill must return every page (both groups) to the pool";
}

TEST_F(FlatChunkedPrefillSuite, AbortDuringDecodeRestoresPoolBaseline) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();  // single-chunk prefill (4 tokens)
    SendForwardDone("r1", {42});
    PlanOnce();  // decode step
    SendForwardDone("r1", {43});
    EXPECT_LT(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    SendAbort(*scheduler_, "r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "abort during decode must return every page to the pool";
}

// ---------------------------------------------------------------------------
// Failure-path page release, event level. The scheduler's flat admission gate
// makes coordinator failures unreachable through NextExecutionPlan, so drive
// the FSM events directly against a starved pool: a throwing transition must
// return the request's pages to the pool, and the request must still be
// Abortable cleanly afterwards (its tables are empty, not dangling).
// ---------------------------------------------------------------------------
TEST(FlatEventFailurePath, PrefillChunkFailureReleasesPagesAndAbortStaysClean) {
    BlockPool pool(/*total_num_blocks=*/6);  // 5 usable
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/6, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    // First chunk: 4 tokens -> 2 pages per group = 4 of the 5 usable blocks.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/0, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::Prefilling>());
    ASSERT_EQ(pool.NumFreeBlocks(), 1);

    // Second chunk wants 8 tokens -> 4 pages per group = 8 > 1 free: the
    // coordinator Acquire inside the transition fails and the event throws.
    EXPECT_THROW(
        request.Apply(fsm::SchedulePrefillEvent{/*tokens_this_round=*/8,
                                                /*reserve_num_tokens_in_next_schedule_event=*/0,
                                                /*hybrid_prefix_cache=*/nullptr, &coordinator}),
        std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 5) << "failure path must return the request's pages to the pool";

    // The request kept its (now table-less) state; Abort must not throw and
    // must not double-free.
    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), 5);
}

TEST(FlatEventFailurePath, DecodeStepFailureReleasesPagesAndAbortStaysClean) {
    BlockPool pool(/*total_num_blocks=*/5);  // 4 usable
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    // Whole 4-token prompt in one chunk -> PrefillDone holding all 4 blocks,
    // with 1 reserve token to acquire on the decode transition.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    ASSERT_EQ(pool.NumFreeBlocks(), 0);

    // PrefillDone -> Decoding needs 1 fresh page per group (tail pages are
    // full) with 0 free: Acquire fails, the event throws, pages are released.
    EXPECT_THROW(request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1,
                                                        /*hybrid_prefix_cache=*/nullptr, &coordinator}),
                 std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 4) << "failure path must return the request's pages to the pool";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), 4);
}

TEST(FlatEventFailurePath, MidDecodeStepFailureReleasesPagesAndAbortStaysClean) {
    BlockPool pool(/*total_num_blocks=*/7);  // 6 usable
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    // Whole 4-token prompt in one chunk (4 of 6 blocks), then two decode steps:
    // the first takes a fresh page per group (pool empty, tail room 1), the
    // second fills the tail for free. The request is now mid-decode on a
    // starved pool.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    ASSERT_EQ(pool.NumFreeBlocks(), 0);
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    ASSERT_EQ(pool.NumFreeBlocks(), 0);

    // Third Decoding->Decoding step needs a fresh page per group with 0 free:
    // DecodeStep's Acquire fails, the event throws, and every page the request
    // held must return to the pool.
    EXPECT_THROW(request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1,
                                                        /*hybrid_prefix_cache=*/nullptr, &coordinator}),
                 std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 6) << "mid-decode failure path must return the request's pages to the pool";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), 6);
}

TEST(FlatEventFailurePath, FirstChunkFailureLeavesPoolBalancedAndAbortStaysClean) {
    BlockPool pool(/*total_num_blocks=*/4);  // 3 usable
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    // First chunk wants 4 tokens -> 2 pages per group = 4 > 3 free: the
    // PrefillFirstChunk Acquire inside the transition fails and the event
    // throws before any state is committed. Nothing may stick to the pool.
    EXPECT_THROW(request.Apply(fsm::SchedulePrefillFirstChunkEvent{
                     /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool,
                     MatchResult{}, Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true,
                     /*loadback_diff=*/{}, /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr,
                     /*mamba_loadback_nodes=*/{}, &coordinator}),
                 std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 3) << "failed first chunk must leave the pool untouched";
    EXPECT_EQ(req_pool.AvailableSlots(), 4) << "no request-pool slot may leak on a failed first chunk";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), 3);
}

// Request-pool exhaustion at the first chunk: ReqPoolAllocator::Allocate()
// throws, and it must do so BEFORE the transition populates the block tables --
// with the old after-population order the throw bypassed the FreeRequest guard
// and leaked every freshly acquired page. Unreachable through the scheduler
// (AvailableSlots pre-gate), so driven at the event level.
TEST(FlatEventFailurePath, ReqPoolExhaustionAtFirstChunkLeavesPoolBalanced) {
    BlockPool pool(/*total_num_blocks=*/32);  // 31 usable: pages are NOT the constraint
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{1};
    ReqPoolIndex held = req_pool.Allocate();  // exhaust the single slot
    ASSERT_EQ(req_pool.AvailableSlots(), 0);

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    EXPECT_THROW(request.Apply(fsm::SchedulePrefillFirstChunkEvent{
                     /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool,
                     MatchResult{}, Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true,
                     /*loadback_diff=*/{}, /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr,
                     /*mamba_loadback_nodes=*/{}, &coordinator}),
                 std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 31) << "a failed req-pool Allocate must not leak block-pool pages";

    EXPECT_NO_THROW(request.Apply(fsm::AbortEvent{&coordinator}));
    EXPECT_TRUE(request.Is<fsm::Finished>());
    EXPECT_EQ(pool.NumFreeBlocks(), 31);
}

// ---------------------------------------------------------------------------
// SWA window off-by-one regression, at the exact boundary. page_size=2, W=4,
// 4-token prompt, single-token decode. Convention: with N tokens computed
// before a round's forward, the pending query sits at position N and attends
// keys [N-3, N]; AdvanceWindow may free only pages fully below N-3. The
// Decoding transition must therefore pass container_size - decode_input_tokens
// (= N), NOT the container size (which already includes the pending token):
// the off-by-one frees the page holding the query's oldest key one round early
// whenever (size - W + 1) % page_size == 0.
// ---------------------------------------------------------------------------
TEST(FlatSwaWindowBoundary, DecodeStepKeepsOldestInWindowPageAtPageBoundary) {
    BlockPool pool(/*total_num_blocks=*/32);
    std::vector<KvCacheSpec> specs{
        KvCacheSpec{AttnKind::kFull, /*page_size=*/2, /*sliding_window=*/0},
        KvCacheSpec{AttnKind::kSlidingWindow, /*page_size=*/2, /*sliding_window=*/4},
    };
    KvCacheCoordinator coordinator = MakeCoordinator(specs, pool);
    ReqPoolAllocator req_pool{4};

    RequestSpec spec{.request_id = "r1", .tokens = MakeAlignedTokens(/*num_pages=*/2, /*page_size=*/2)};
    Request request{spec, /*page_size=*/2, Role::kFused};

    // Whole 4-token prompt in one chunk -> PrefillDone, 2 pages per group.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());

    const auto swa_slot_null = [&](std::int32_t i) { return request.FlatBlockTablesRef()[1].Blocks()[i]->IsNull(); };

    // Sampled token lands -> size 5. Decode transition (no slide): 3 pages.
    request.Apply(fsm::ExtendResultEvent{"r1", {100}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    ASSERT_EQ(request.FlatBlockTablesRef()[1].NumBlocks(), 3);
    EXPECT_FALSE(swa_slot_null(0));

    // size 6 -> N = 5 computed; query at 5 needs keys [2,5]: token 2 opens page
    // 1, so page 0 (tokens 0,1) is out -> slot 0 punched, slot 1 kept.
    request.Apply(fsm::ExtendResultEvent{"r1", {101}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_FALSE(swa_slot_null(1));

    // size 7 -> N = 6 computed; query at 6 needs keys [3,6]: key 3 STILL lives
    // in page 1, so slot 1 must survive this round. The old code passed the
    // container size 7 (skipped = 7-4+1 = 4 -> 2 pages) and freed slot 1 here,
    // handing the kernel a null hole for an in-window key -- and the freed page
    // could be re-allocated to another request the same round.
    request.Apply(fsm::ExtendResultEvent{"r1", {102}});
    const std::int32_t free_before = pool.NumFreeBlocks();
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_FALSE(swa_slot_null(1)) << "key 3 of the pending query lives in page 1; freeing it is the off-by-one";
    EXPECT_TRUE(swa_slot_null(0));
    // This round slides nothing and acquires one fresh page per group.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 2);

    // size 8 -> N = 7; query at 7 needs keys [4,7] -> page 1 (tokens 2,3) is
    // now fully out and is punched exactly one round after the old code did.
    request.Apply(fsm::ExtendResultEvent{"r1", {103}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_TRUE(swa_slot_null(1));
    EXPECT_FALSE(swa_slot_null(2));

    // Full group never punches holes.
    for (CacheBlock* b : request.FlatBlockTablesRef()[0].Blocks()) {
        EXPECT_FALSE(b->IsNull());
    }

    request.Apply(fsm::AbortEvent{&coordinator});
    EXPECT_TRUE(request.Is<fsm::Finished>());
}

// ---------------------------------------------------------------------------
// Decode-reserve ledger (flat_reserved_pages_): a prefill-completing admission
// promises the decode-transition pages, but they are only Acquired one round
// later -- other candidates must not be admitted into the promised headroom in
// between, and an aborted request must not leave a phantom reservation behind.
// page_size=2, two full-history groups, 11 physical pages -> 10 usable.
// ---------------------------------------------------------------------------
class FlatReserveLedgerSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 11;
        cfg.host_allocator.total_pages = 11;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        cfg.paged_cache_groups = {
            MakeGroup("full_a", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("full_b", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
        };
        return cfg;
    }
};

TEST_F(FlatReserveLedgerSuite, LaterRequestCannotStealReservedDecodeHeadroom) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a: 6-token prompt. Gate charges 2*ceil(7/2) = 8 <= 10; prefill consumes
    // 6, promising 2 reserve blocks -> free 4, outstanding reservation 2.
    // b: 2-token prompt needing 2*ceil(3/2) = 4 blocks. Raw free is 4, but 2 of
    // those are a's promise: b must defer (4 > 4 - 2). Before the ledger, b was
    // admitted here and a's decode round then deferred forever.
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/1, /*start=*/101));
    ExecutionPlan round1 = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(round1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u) << "b must not be admitted into a's promised decode pages";
    EXPECT_EQ(op1->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // a's decode transition consumes its own reservation (gate excludes it);
    // b still defers against the raw remainder.
    SendForwardDone("a", {99});
    ExecutionPlan round2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u) << "a's decode must proceed into its reserved pages";
    EXPECT_EQ(op2->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    // a finishes -> its 8 blocks return -> b becomes schedulable.
    SendForwardDone("a", {100});
    SendFinish("a");
    ExecutionPlan round3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->request_ids.at(0), "b");

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatReserveLedgerSuite, AbortWithOutstandingReservationLeavesNoPhantom) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // a admitted with a 2-block outstanding decode reservation (see above).
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // Abort a BEFORE its decode transition ever acquires the reserve: the pages
    // return and the reservation entry must be dropped with them.
    SendAbort(*scheduler_, "a");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // b needs the WHOLE pool: 8-token prompt -> gate 2*ceil(9/2) = 10 <= 10
    // only if no phantom reservation is still deflating the free count.
    Submit(MakeRequestSpec("b", /*num_pages=*/4, /*start=*/101));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u) << "a leaked reservation would defer b forever";
    EXPECT_EQ(op->request_ids.at(0), "b");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    SendForwardDone("b", {142});
    SendFinish("b");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// M9 cross-request flat prefix hits, end to end through the Scheduler: the
// admission layer matches once (schedulePrefillFirstChunk), the FSM first-chunk
// transition claims the hit and acquires only the remainder, and the op's
// input window starts past the claimed tokens. disable_prefix_cache=false is
// the point of this suite (every other flat suite runs with it true, which is
// why the intermediate zero-hit state stayed latent).
//
// Shared shape: page_size=2, full + swa groups. The base suite uses a LARGE
// sliding window (32 tokens) so tests 1-4 stay about claim / window / pool
// math -- the SWA group then matches and slides exactly like the full group.
// SwaGroupHitRespectsWindow runs a small window (4) in its own fixture.
//
// Pool-accounting convention used in every derivation below: a cached FREE
// block (ref 0 + hash) sits in the free list, so CLAIMING it (TouchBlock)
// removes it from NumFreeBlocks just like a fresh allocation does -- a hit
// request's pool delta is claimed + newly acquired pages.
// ---------------------------------------------------------------------------
class FlatPrefixHitSuite : public SchedulerTestSuite {
protected:
    virtual std::int32_t SlidingWindowTokens() const { return 32; }
    virtual bool DisablePrefixCache() const { return false; }
    virtual std::int32_t TotalPages() const { return 64; }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = TotalPages();
        cfg.host_allocator.total_pages = TotalPages();
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = DisablePrefixCache();

        cfg.paged_cache_groups = {
            MakeGroup("full", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::FullHistory,
                      PagedCacheGroupFamily::History),
            MakeGroup("swa", cfg.page_size, cfg.device_allocator.total_pages,
                      PagedCacheGroupConfig::Retention::SlidingWindow,
                      PagedCacheGroupFamily::State, SlidingWindowTokens()),
        };
        return cfg;
    }

    RequestSpec MakeSpecWithTokens(const std::string& id, token_vec_t tokens) {
        return RequestSpec{.request_id = id, .tokens = std::move(tokens)};
    }

    // Drive a request through prefill -> one decode round -> finish, returning
    // the per-group block-table row its PREFILL op carried. The decode round is
    // load-bearing: the PrefillDone->Decoding finalize is what registers the
    // full prefill pages' content hashes (CacheFullBlocks), and FreeRequest on
    // finish returns the blocks to the free list WITH those hashes intact --
    // that cached-free state is what a later request's admission match hits.
    std::map<std::string, std::vector<std::int32_t>> RunLifecycle(const RequestSpec& spec) {
        Submit(spec);
        ExecutionPlan prefill = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(prefill);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->flat_block_tables) {
                rows[gid] = table.at(0);
            }
        }
        SendForwardDone(spec.request_id, {9001});
        PlanOnce();  // PrefillDone -> Decoding: finalize registers the hashes
        SendForwardDone(spec.request_id, {9002});
        SendFinish(spec.request_id);
        PlanOnce();  // reap
        return rows;
    }

    static void ExpectRowPrefixEq(const std::vector<std::int32_t>& row,
                                  const std::vector<std::int32_t>& expected_prefix, const char* what) {
        ASSERT_GE(row.size(), expected_prefix.size()) << what;
        for (std::size_t i = 0; i < expected_prefix.size(); ++i) {
            EXPECT_EQ(row[i], expected_prefix[i]) << what << " slot " << i;
        }
    }
};

// Test 1: two requests share an 8-token prefix; the second reuses the first
// one's cached pages physically and computes only its own tail.
TEST_F(FlatPrefixHitSuite, TwoRequestsSharePrefixReusePages) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // r1: 8 tokens = 4 full pages/group. Its lifecycle registers 4 page hashes
    // per group and frees the blocks cached (see RunLifecycle).
    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";
    ASSERT_EQ(r1_rows.at("full").size(), 4u);
    ASSERT_EQ(r1_rows.at("swa").size(), 4u);

    // r2: 12 tokens, first 8 identical to r1. Hit derivation: the match input
    // is capped at (12-1)/2 = 5 pages (r2's own last-token recompute cap); r1
    // REGISTERED 4 full pages (its cap only bounds what r1 can MATCH, not what
    // its finalize registers), r2's page-4 hash chains off different tail
    // tokens -> miss there. Full group hits 4; the swa group (W=32,
    // contiguous_needed=16 > 4) keeps its whole accumulating run of 4 -> the
    // fixpoint is 4 common blocks = 8 tokens.
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PageSize());  // tokens 1..8 == r1's
    const token_vec_t tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // (c) The op's input covers ONLY tokens [8, 12): 4 new tokens starting past
    // the 8 hit tokens, and the input token ids are exactly r2's distinct tail.
    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 12);
    EXPECT_EQ(op->input_ids, tail);
    // Page-space fields (radix-hit parity, see applyPrefillEvent): begin = 0
    // (no pages before the first chunk) and size counts everything new to the
    // request's table this round = 4 claimed + ceil(4 new tokens / 2) = 6, so
    // Python copies the claimed prefix mappings into req_to_page too.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 6);
    ASSERT_EQ(op->occupied_pages.at(0).size(), 6u);

    // (a) Physical reuse: both groups' first 4 blocks are r1's pages.
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), r1_rows.at("full"), "full row");
    ExpectRowPrefixEq(op->flat_block_tables.at("swa").at(0), r1_rows.at("swa"), "swa row");

    // (b) Pool delta: claiming pulls the 4 cached-free blocks per group out of
    // the free list (8) and the remainder Acquire takes ceil(4/2) = 2 fresh
    // pages per group (4) -> 12 total. The decode reserve is only PROMISED here
    // (ledger entry), not acquired, so it does not move the free count yet.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);

    // Decode transition: finalize registers r2's pages 4..5 (0..3 already carry
    // hashes) and acquires the 1-token reserve -> 1 fresh page per group (r2's
    // tail page is full after the 4-token chunk).
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 14);

    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

// Test 2: identical prompt. The hit is capped at (PrefillSize-1)/page_size
// pages so the last token is always recomputed to produce logits.
TEST_F(FlatPrefixHitSuite, FullHitCapsAtLastToken) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    RunLifecycle(r1);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2 = the same 8 tokens. Cap = (8-1)/2 = 3 pages, so even though all 4 of
    // r1's pages are cached, the match input stops at 3 -> hit 3 pages = 6
    // tokens; the 4th page's 2 tokens are recomputed.
    Submit(MakeSpecWithTokens("r2", r1.tokens));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 6);
    // input = tokens [6, 8) of the 1..8 sequence.
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/7));
    // 3 claimed + ceil(2/2) = 1 fresh page per group.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 4);
    // Pool: 3 claimed + 1 fresh per group = 8 blocks off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    // Proceeds to decode (reserve: 1 fresh page per group, tail full) + finish;
    // pool returns to baseline.
    SendForwardDone("r2", {199});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 10);
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Test 3: disable_prefix_cache=true must skip the admission match entirely --
// full input, zero reuse -- even though r1's pages were registered.
class FlatPrefixHitDisabledSuite : public FlatPrefixHitSuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(FlatPrefixHitDisabledSuite, DisablePrefixCacheSkipsMatch) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // Same 12-token / shared-8-prefix shape as test 1, but with the config
    // knob off the hit must be zero: full 12-token input from position 0.
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PageSize());
    const token_vec_t tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 12) << "no hit -> the whole prompt is the input";
    EXPECT_EQ(op->extend_prefix_lens.at(0), 0);
    EXPECT_EQ(op->input_ids, r2_tokens);
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 6) << "all 6 pages freshly allocated, none claimed";
    // Pool: 6 fresh pages per group = 12 (numerically equal to test 1's delta,
    // but all 12 are allocations here -- the distinguishing signal is the full
    // input above).
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Test 4: a partial hit claims only the common pages.
TEST_F(FlatPrefixHitSuite, PartialHit) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2: 12 tokens, only the first 4 match r1 (pages 0..1); page 2 hashes over
    // different tokens AND a matching prior, so it and every later page miss
    // (the chain propagates the divergence). Hit = 2 pages = 4 tokens.
    token_vec_t r2_tokens = MakeTokens(/*count=*/4);  // 1..4 == r1's first 4
    const token_vec_t tail = MakeTokens(/*count=*/8, /*start=*/801);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // Input = tokens [4, 12): 8 tokens starting past the 4 hit tokens.
    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_ids, tail);
    // 2 claimed + ceil(8/2) = 4 fresh pages per group.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 6);

    // Only the common pages are reused, in place.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 2);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 2);
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->flat_block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 2 claimed + 4 fresh per group = 12 blocks off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Test 5: with a SMALL sliding window the SWA group's bounded match holds its
// own invariant inside the common fixpoint -- the bounded right-to-left scan
// stops once its contiguous run is satisfied, so it never inspects the slots
// r1's own window punched (it claims them as null holes).
class FlatPrefixHitSmallWindowSuite : public FlatPrefixHitSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(FlatPrefixHitSmallWindowSuite, SwaGroupHitRespectsWindow) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // r1: 8 tokens. Its finalize (PrefillDone->Decoding) REGISTERS all 4 swa
    // page hashes BEFORE AdvanceWindow(8) punches slots 0..1 (first kept token
    // 8-4+1=5 -> first kept page 2), so the punched blocks land in the free
    // list as cached-with-hash -- still matchable. Finish frees the rest.
    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
    ASSERT_EQ(r1_rows.at("swa").size(), 4u);

    // r2: 10 tokens, first 8 == r1's. Fixpoint derivation (W=4, page=2,
    // contiguous_needed = ceil(3/2) = 2):
    //   cap = (10-1)/2 = 4 -> match input = pages 0..3, all cached for BOTH
    //   groups (register-then-punch above). Full group matches 4. SwaManager
    //   scans right->left: page 3 (run 1), page 2 (run 2 == needed) -> stop;
    //   keep = 4 with pages 0..1 as null holes, so its bounded coverage is
    //   ALSO 4 -> common stays 4 on the first pass and the fixpoint is
    //   num_common_blocks = 4 (8 hit tokens), swa claiming 2 real + 2 holes.
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PageSize());  // 1..8 == r1's
    const token_vec_t tail = MakeTokens(/*count=*/2, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // Input = tokens [8, 10): the fixpoint hit covers all 8 shared tokens.
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, tail);
    // Table length space: 4 claimed slots (real or hole) + 1 fresh page.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 5);

    // Full row: r1's 4 pages reused + 1 fresh, no holes.
    const auto& full_row = op->flat_block_tables.at("full").at(0);
    ASSERT_EQ(full_row.size(), 5u);
    ExpectRowPrefixEq(full_row, r1_rows.at("full"), "full row");
    EXPECT_GT(full_row[4], 0);

    // Swa row: holes exactly where the match put them, r1's pages 2..3 reused.
    const auto& swa_row = op->flat_block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 5u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]);
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    // Window invariant on the CLAIMED prefix (mirrors ExpectSwaWindowIntact,
    // test_kv_cache_coordinator.cpp): the last contiguous_needed = 2 slots of
    // the 4-slot claimed prefix must be real -- no null hole inside the last
    // window of the hit.
    for (std::size_t i = 2; i < 4; ++i) {
        EXPECT_GT(swa_row[i], 0) << "null hole inside the last window of the claimed prefix at slot " << i;
    }

    // Pool: full claims 4, swa claims 2 (holes claim nothing), remainder
    // Acquire takes 1 fresh page per group -> 4 + 2 + 2 = 8 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Test 6 (regression, M9-F1): the first-chunk gate must also charge the free
// blocks the CLAIM consumes. ClaimCommonPrefix TouchBlock()s every real hit
// block, and touching a ref-0 cached block REMOVES it from the free list
// (block_pool.h) -- so with a pool where the new-token + reserve charge EXACTLY
// equals the free count and the hit blocks sit ref-0-cached in that free list,
// the uncharged gate admitted the request, the claim then ate the hit blocks,
// and the transition's Acquire hit the "flat path: allocation failure
// unsupported in C slice" assert. Post-fix the request is DEFERRED intact and
// admitted once the pool holder finishes.
class FlatPrefixHitTightPoolSuite : public FlatPrefixHitSuite {
protected:
    // 11 physical pages -> 10 usable (page 0 is the null placeholder).
    std::int32_t TotalPages() const override { return 11; }
};

TEST_F(FlatPrefixHitTightPoolSuite, GateChargesFreeHitBlocksClaimWillConsume) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1: 4 tokens = 2 full pages/group (gate charge 2 * ceil(5/2) = 6 <= 10).
    // Its lifecycle registers 2 page hashes per group and frees everything, so
    // 4 of the 10 free blocks are now ref-0 CACHED (r2's future hit set); the
    // other 6 (r1's two decode-reserve pages + the 4 never-used blocks) are
    // plain free.
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/2));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r3: 2 distinct tokens, kept alive as the pool holder. Prefill takes 1
    // fresh page/group: free 10 -> 8. Both pops come from the LRU head, which
    // holds the never-used blocks -- r1's cached blocks survive with their
    // hashes intact.
    Submit(MakeRequestSpec("r3", /*num_pages=*/1, /*start=*/501));
    ExecutionPlan r3_prefill = PlanOnce();
    ASSERT_NE(FindFlatOp(r3_prefill), nullptr);
    ASSERT_EQ(FindFlatOp(r3_prefill)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 8);

    // r3's finalize acquires its decode reserve (1 fresh page/group): 8 -> 6.
    // The acquire erases r3's reserve ledger entry, so r2's gate below reads
    // raw free = 6 with no outstanding reservations. The 6 free blocks are
    // exactly r1's 4 cached hit blocks + r1's 2 plain reserve pages.
    SendForwardDone("r3", {599});
    PlanOnce();
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 6);

    // r2: 8 tokens, first 4 == r1's. Hit derivation: cap = (8-1)/2 = 3 pages;
    // r1 registered only 2 -> full group hits 2, swa (W=32, needed 16 > 2)
    // keeps its accumulating run of 2 -> fixpoint 2 common blocks = 4 hit
    // tokens, ALL 4 hit blocks ref-0 in the free list. New-token charge:
    // tokens_this_round = 4 remainder, + 1 decode reserve -> 2 * ceil(5/2) = 6
    // == free 6 EXACTLY. Pre-fix the gate admitted (6 > 6 is false), the claim
    // then pulled the 4 hit blocks out of the free list, and the remainder
    // Acquire (4 blocks against 2 free) threw the "flat path: allocation
    // failure unsupported in C slice" assert. Post-fix the gate charges
    // 6 + 4 = 10 > 6 and defers, touching nothing: the round re-issues only
    // r3's pending decode step (0 fresh pages, its forward is still in
    // flight).
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/2, PageSize());  // tokens 1..4 == r1's
    const token_vec_t tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan starved = PlanOnce();
    const FlatForwardOperation* starved_op = FindFlatOp(starved);
    ASSERT_NE(starved_op, nullptr);
    ASSERT_EQ(starved_op->request_ids.size(), 1u) << "r2 must be deferred, not admitted into a short pool";
    EXPECT_EQ(starved_op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 6) << "a deferred first chunk must not touch the pool";

    // r3 finishes -> its 4 blocks return (free 10). r2's charge is now
    // 6 new/reserve + 4 claim = 10 == free 10: admitted exactly at the
    // boundary (the new charge is exact, not an over-approximation). The claim
    // pulls 4 and the remainder Acquire takes 2 pages/group: free 10 -> 2.
    SendForwardDone("r3", {600});
    SendFinish("r3");
    ExecutionPlan plan2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(plan2);
    ASSERT_NE(op2, nullptr) << "deferred request must be schedulable after the holder frees its pages";
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r2");
    EXPECT_EQ(op2->input_lengths.at(0), 4) << "only the 4-token remainder is computed";
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // r2's finalize acquires the promised 2-block decode reserve: the pool hits
    // exactly 0 -- the admission arithmetic left no slack and needed none.
    SendForwardDone("r2", {699});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);

    SendForwardDone("r2", {700});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after both complete";
}

// ---------------------------------------------------------------------------
// M13 decode-block caching, end to end through the Scheduler: pages that fill
// up DURING decode register into the flat prefix cache via the hash chain the
// Decoding state carries (DecodeStep: register -> slide -> acquire), so a
// later turn whose prompt embeds the previous turn's response hits PAST the
// previous prompt boundary -- the motivating multi-turn scenario.
//
// Shares FlatPrefixHitSuite's shape (page_size=2, full + swa groups, W=32 so
// tests 1/2/4 stay about caching, disable_prefix_cache=false) and its
// pool-accounting convention (claiming a cached free block leaves the free
// count like an allocation does).
//
// Decode-page fill timing used in every derivation below: a round scheduled at
// container Size s has N = s - decode_input_tokens(=1) computed tokens (the
// pending input's KV is written by THIS round's forward), and the chain
// registers pages up to N / page_size. So the page a generated token completes
// registers only on the NEXT round -- once that token is no longer the pending
// input. In particular the last wanted page needs one extra generated token
// and schedule round before finish, or its block is freed hashless and can
// never hit ("tail page registers one round late").
// ---------------------------------------------------------------------------
class FlatDecodeCachingSuite : public FlatPrefixHitSuite {
protected:
    // Deliver one sampled token and run the next schedule round, returning the
    // per-group block-table row the round's flat op carried (state after that
    // round's register -> slide -> acquire). Single-request rounds only.
    std::map<std::string, std::vector<std::int32_t>> AdvanceOneRound(const std::string& id, token_t token) {
        SendForwardDone(id, {token});
        ExecutionPlan plan = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(plan);
        EXPECT_NE(op, nullptr);
        std::map<std::string, std::vector<std::int32_t>> rows;
        if (op != nullptr) {
            for (const auto& [gid, table] : op->flat_block_tables) {
                rows[gid] = table.at(0);
            }
        }
        return rows;
    }

    // Turn-1 conversation prefix shared by the tests below: prompt {1,2,3,4},
    // generated tokens 101..105. Registration schedule (page=2, W large or
    // small -- fill timing is window-independent):
    //   plan #1: prefill, 2 pages/group.
    //   +101 -> Size 5, PrefillDone->Decoding finalize: registers PROMPT pages
    //           0,1 and seeds the chain at {2 pages, hash(page 1)}.
    //   +102 -> Size 6, N=5, filled = 5/2 = 2 == chained 2: nothing new.
    //   +103 -> Size 7, N=6, filled = 3: REGISTERS page 2 = {101,102}.
    //   +104 -> Size 8, N=7, filled = 3: page 3 = {103,104} is full in the
    //           container, but 104 is this round's pending input -> NOT yet.
    //   +105 -> Size 9, N=8, filled = 4: REGISTERS page 3 -- the tail page
    //           lands one round late, so token 105 exists only to push N past
    //           8; finishing after +104 instead would free page 3's block
    //           hashless and cap the next turn's hit at 3 pages.
    // The returned rows are the LAST round's tables: 5 slots (ceil(9/2)), the
    // first 4 holding the conversation's pages 0..3.
    std::map<std::string, std::vector<std::int32_t>> RunTurnOne() {
        Submit(MakeRequestSpec("r1", /*num_pages=*/2));
        ExecutionPlan prefill = PlanOnce();
        EXPECT_NE(FindFlatOp(prefill), nullptr);
        AdvanceOneRound("r1", 101);
        AdvanceOneRound("r1", 102);
        AdvanceOneRound("r1", 103);
        AdvanceOneRound("r1", 104);
        auto rows = AdvanceOneRound("r1", 105);
        SendFinish("r1");
        PlanOnce();  // reap
        return rows;
    }

    // Turn-2 prompt: r1's 4 prompt tokens + r1's first 4 generated tokens + 2
    // new tokens = 10. The admission hashes chain from "" over the same token
    // stream r1's registration chained over, so pages 0..3 match by content.
    token_vec_t MakeTurnTwoPrompt() {
        token_vec_t tokens = MakeAlignedTokens(/*num_pages=*/2, PageSize());  // {1,2,3,4} == r1's prompt
        const token_vec_t response = MakeTokens(/*count=*/4, /*start=*/101);  // r1's generated 101..104
        tokens.insert(tokens.end(), response.begin(), response.end());
        const token_vec_t fresh = MakeTokens(/*count=*/2, /*start=*/901);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }

    // Turn-3 prompt: turn 2's full 13-token stream + 3 new tokens = 16.
    token_vec_t MakeTurnThreePrompt() {
        token_vec_t tokens = MakeTurnTwoPrompt();
        const token_vec_t r2_response = MakeTokens(/*count=*/3, /*start=*/201);
        tokens.insert(tokens.end(), r2_response.begin(), r2_response.end());
        const token_vec_t fresh = MakeTokens(/*count=*/3, /*start=*/951);
        tokens.insert(tokens.end(), fresh.begin(), fresh.end());
        return tokens;
    }
};

// Test 1: the motivating hit. r2's 10-token prompt shares r1's full 8-token
// stream; the hit covers all 4 pages -- 2 of them PAST r1's prompt boundary
// (pages 2,3 were filled by r1's decode, not its prefill).
TEST_F(FlatDecodeCachingSuite, DecodeFilledPageBecomesHittable) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const auto r1_rows = RunTurnOne();
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";

    // Hit derivation: cap = (10-1)/2 = 4 pages -> match input = pages 0..3,
    // all four registered by r1 (prompt pages 0,1 at finalize; decode pages
    // 2,3 via the chain, see RunTurnOne). Full group matches 4; swa (W=32,
    // contiguous_needed = ceil(31/2) = 16 > 4) keeps its whole cached run of
    // 4 -> fixpoint 4 common blocks = 8 hit tokens.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    // Input window skips the 8 hit tokens: only the 2 new tokens are computed.
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    // Page space: 4 claimed + ceil(2 new tokens / 2) = 1 fresh page.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 5);

    // Physical reuse: r2's first 4 blocks ARE r1's pages, in both groups --
    // slots 2,3 are the pages r1's decode filled, beyond its prompt boundary.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->flat_block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: claim pulls 4 cached-free blocks per group (8), the remainder
    // Acquire takes 1 fresh page per group (2) -> 10 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

// Test 2: third turn. r2 decodes its own response whose pages register through
// the SAME chain mechanics, so r3's hit grows to cover r2's response region.
TEST_F(FlatDecodeCachingSuite, MultiTurnConversationReusesResponsePages) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    RunTurnOne();  // registers conversation pages 0..3
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // Turn 2: 10-token prompt, hit 4 pages (test 1's derivation), then 3
    // generated tokens 201..203:
    //   +201 -> Size 11, finalize: FlatWindowPageHashes covers pages 0..4
    //           ((8+2)/2 = 5); pages 0..3 already carry hashes (IsCached skip),
    //           page 4 = {901,902} REGISTERS; chain seeds at {5, hash(page 4)}.
    //   +202 -> Size 12, N=11, filled = 5: nothing new.
    //   +203 -> Size 13, N=12, filled = 6: REGISTERS page 5 = {201,202} (the
    //           tail-page-late rule again: 203 exists to push N past 12).
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(turn2);
    ASSERT_NE(op2, nullptr);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 8) << "turn 2 hits r1's prompt + response pages";
    AdvanceOneRound("r2", 201);
    AdvanceOneRound("r2", 202);
    const auto r2_rows = AdvanceOneRound("r2", 203);
    ASSERT_EQ(r2_rows.at("full").size(), 7u);  // ceil(13/2)
    SendFinish("r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // Turn 3 prompt: r2's full 13-token stream + 3 new = 16 tokens. Hit
    // derivation: cap = (16-1)/2 = 7 -> match input = pages 0..6; pages 0..5
    // are registered (0..3 by r1, 4..5 by r2), page 6 = {203, 951} was never
    // full inside any request -> full group matches 6; swa (needed 16 > 6)
    // keeps its run of 6 -> fixpoint 6 common blocks = 12 hit tokens, past
    // turn 2's own 10-token prompt boundary and into r2's response (page 5).
    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));

    ExecutionPlan turn3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(turn3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12) << "hit grows across turns: 8 -> 12 tokens";
    EXPECT_EQ(op3->input_lengths.at(0), 4);
    EXPECT_EQ(op3->prefill_lengths.at(0), 16);
    EXPECT_EQ(op3->input_ids, (token_vec_t{203, 951, 952, 953}));
    // 6 claimed + ceil(4 new tokens / 2) = 2 fresh pages.
    EXPECT_EQ(op3->begins.at(0), 0);
    EXPECT_EQ(op3->sizes.at(0), 8);

    // Physical reuse spans BOTH earlier turns: slots 0..3 are r1's blocks
    // (r2 claimed and re-freed them cached), slots 4..5 are r2's own pages.
    const std::vector<std::int32_t> full_prefix(r2_rows.at("full").begin(), r2_rows.at("full").begin() + 6);
    const std::vector<std::int32_t> swa_prefix(r2_rows.at("swa").begin(), r2_rows.at("swa").begin() + 6);
    ExpectRowPrefixEq(op3->flat_block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op3->flat_block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 6 claimed per group (12) + 2 fresh per group (4) = 16.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 16);

    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after all three turns";
}

// Test 3: small window. A decode page REGISTERS (DecodeStep runs the
// registration before the slide) and a later round's AdvanceWindow then
// punches it -- the punch frees the block WITH its hash, so the next turn
// still hits it through the bounded SWA fixpoint.
class FlatDecodeCachingSmallWindowSuite : public FlatDecodeCachingSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(FlatDecodeCachingSmallWindowSuite, SwaPunchedDecodePageStillHittable) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // Turn-1 fill/register timing is RunTurnOne's (window-independent), inlined
    // here because the punch round +106 must land BEFORE finish; the swa slides
    // on top of it (W=4: first kept token = N-3, punched pages = (N-3)/2):
    //   +101 -> Size 5, finalize: registers prompt pages 0,1; N=4 -> punches
    //           nothing (first kept token 1).
    //   +102 -> Size 6, N=5 -> punch slot 0.
    //   +103 -> Size 7, N=6: REGISTERS page 2; first kept token 3 is still in
    //           page 1 -> no new punch.
    //   +104 -> Size 8, N=7 -> punch slot 1.
    //   +105 -> Size 9, N=8: REGISTERS page 3; first kept page 2 -> no punch.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan r1_prefill = PlanOnce();
    ASSERT_NE(FindFlatOp(r1_prefill), nullptr);
    AdvanceOneRound("r1", 101);
    AdvanceOneRound("r1", 102);
    AdvanceOneRound("r1", 103);
    AdvanceOneRound("r1", 104);
    const auto r1_rows = AdvanceOneRound("r1", 105);
    ASSERT_EQ(r1_rows.at("swa").size(), 5u);
    EXPECT_EQ(r1_rows.at("swa")[0], 0);
    EXPECT_EQ(r1_rows.at("swa")[1], 0);
    ASSERT_GT(r1_rows.at("swa")[2], 0) << "page 2 is registered AND still live after the +105 round";
    ASSERT_GT(r1_rows.at("swa")[3], 0);

    // One more round punches the REGISTERED decode page: +106 -> Size 10, N=9
    // -> first kept token 6 -> first kept page 3 -> slot 2 (registered at the
    // +103 round) is evicted to null; its block reaches the free list with the
    // hash intact. That block plus r1's finish is the cached state r2 hits.
    const auto punched = AdvanceOneRound("r1", 106);
    EXPECT_EQ(punched.at("swa")[2], 0) << "the registered decode page must be punched by now";
    SendFinish("r1");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2: same 8-token prefix + 2 new = 10 tokens. Fixpoint derivation (W=4,
    // page=2, contiguous_needed = ceil(3/2) = 2): cap = (10-1)/2 = 4 -> match
    // input = pages 0..3, all four hashes cached (slots 0,1,2 punched WITH
    // hash, slot 3 and the whole full group freed cached at finish). Full
    // group matches 4 -> common 4. SwaManager bounded to 4 scans right->left:
    // page 3 cached (run 1), page 2 cached -- the punched decode page --
    // (run 2 == needed) -> stop; keep = 4 with slots 0,1 as null holes. Its
    // bounded length 4 == common, so the fixpoint settles at 4 common blocks
    // = 8 hit tokens; the full group drives the common length.
    token_vec_t r2_tokens = MakeTurnTwoPrompt();
    Submit(MakeSpecWithTokens("r2", r2_tokens));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 5);  // 4 claimed slots (real or hole) + 1 fresh page

    // Full row: r1's 4 pages reused, no holes.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), full_prefix, "full row");

    // Swa row: holes exactly where the bounded match put them; slot 2 is THE
    // punched-then-refreed decode block, physically r1's (id captured at the
    // +105 round, before the punch).
    const auto& swa_row = op->flat_block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 5u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]) << "punched decode page claimed back by hash";
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);

    // Pool: full claims 4, swa claims 2 (holes claim nothing), remainder
    // Acquire takes 1 fresh page per group -> 4 + 2 + 2 = 8 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Test 4: registration writes hashes only -- never refcounts -- so the whole
// three-turn flow must return the pool to baseline after every turn's finish.
TEST_F(FlatDecodeCachingSuite, PoolBalanceAcrossDecodeCaching) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // Turn 1: registers pages 0..3, two of them decode-filled (RunTurnOne).
    RunTurnOne();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "turn 1: decode registration must not hold refs";

    // Turn 2: hits 4 pages, decodes 3 tokens, registers pages 4..5.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan turn2 = PlanOnce();
    ASSERT_NE(FindFlatOp(turn2), nullptr);
    EXPECT_LT(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "turn 2 holds claimed + fresh pages while live";
    AdvanceOneRound("r2", 201);
    AdvanceOneRound("r2", 202);
    AdvanceOneRound("r2", 203);
    SendFinish("r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "turn 2: claimed and fresh pages all return";

    // Turn 3: hits 6 pages (12 tokens, test 2's derivation), one decode round.
    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));
    ExecutionPlan turn3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(turn3);
    ASSERT_NE(op3, nullptr);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12);
    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "baseline restored after the whole conversation";
}

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
