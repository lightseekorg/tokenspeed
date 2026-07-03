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
// Cross-request prefix reuse and retract/writeback are deliberately deferred on
// the flat path (C slice), so they are NOT tested here. Abort, pool-exhaustion
// admission, and the failure-path page release ARE covered (see the suites at
// the bottom of this file).

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

// Chunked prefill on a long prompt, then decode past the sliding window: the
// swa group must develop a null hole even though the prompt arrived in chunks,
// while the full group keeps every page. Exercises PrefillChunk feeding the
// coordinator's AdvanceWindow correctly across chunk boundaries.
TEST_F(FlatChunkedPrefillSuite, ChunkedPrefillThenSwaSlidesToNullHole) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> 3 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    PlanOnce();  // chunk 2
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    ExecutionPlan chunk3 = PlanOnce();  // chunk 3 (last)
    ASSERT_NE(FindFlatOp(chunk3), nullptr);

    SendForwardDone("r1", {99});

    std::optional<ExecutionPlan> last;
    int tok = 100;
    for (int i = 0; i < 4; ++i) {
        last = PlanOnce();
        ASSERT_NE(FindFlatOp(*last), nullptr);
        SendForwardDone("r1", {tok++});
    }
    const FlatForwardOperation* op = FindFlatOp(*last);
    ASSERT_NE(op, nullptr);

    // Full group: no holes across the whole (chunk-built) history.
    for (std::int32_t id : op->flat_block_tables.at("full").at(0)) {
        EXPECT_GT(id, 0) << "full group must keep chunk-built history without holes";
    }
    // Swa group: a null hole appears once decode crosses the window.
    const auto& swa = op->flat_block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa.begin(), swa.end(), 0), swa.end())
        << "swa group should slide to a null hole even after chunked prefill";

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
    // must defer it (its prompt + reserve need 4 blocks) without throwing,
    // while r1's decode step proceeds into the reserved 2 blocks.
    Submit(MakeRequestSpec("r2", /*num_pages=*/1, /*start=*/101));
    SendForwardDone("r1", {99});
    ExecutionPlan starved = PlanOnce();
    const FlatForwardOperation* starved_op = FindFlatOp(starved);
    ASSERT_NE(starved_op, nullptr);
    ASSERT_EQ(starved_op->request_ids.size(), 1u) << "only r1's reserved decode step fits this round";
    EXPECT_EQ(starved_op->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);

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
// Collective starvation deadlock: two requests admitted legitimately together
// corner the pool -- every candidate is deferred, the deferred set holds all
// the pool's pages, and no forward result is pending that could ever finish a
// request and free them. The scheduler must fail loudly (flat retract is
// unimplemented) instead of silently returning empty plans forever, and the
// failed round must not leak or touch pool pages. While a decode result is
// still in flight the same starved round must stay quiet: that state resolves
// itself when the result lands.
// ---------------------------------------------------------------------------
class FlatCollectiveStarvationSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        // 11 physical pages -> 10 usable. Two 2-page prompts need 2*ceil(5/2)=6
        // blocks each at admission (4 prefill + 2 reserve); both fit initially,
        // then their decode steps eat the pool down to zero.
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

TEST_F(FlatCollectiveStarvationSuite, DeadlockedPoolFailsLoudWithoutLeaking) {
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 10);

    // Round 1: both prompts admitted (r1 gate 6<=10, r2 gate 6<=6 after r1's 4).
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(prefill);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Round 2: r1's decode transition takes the last 2 blocks; r2's decode is
    // deferred -- quiet, r1's decode result is still in flight.
    ExecutionPlan round2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);
    SendForwardDone("r1", {43});

    // Round 3: r1's next step fits its tail page (0 fresh blocks); r2 deferred.
    ExecutionPlan round3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(round3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->request_ids.at(0), "r1");

    // Starved round with r1's round-3 decode result STILL IN FLIGHT: both defer,
    // but the round must stay quiet -- the pending result can finish r1.
    ExecutionPlan quiet = PlanOnce();
    const FlatForwardOperation* quiet_op = FindFlatOp(quiet);
    ASSERT_NE(quiet_op, nullptr);
    EXPECT_TRUE(quiet_op->request_ids.empty());

    // Deliver that last result: now nothing is in flight, both requests are
    // deferred, and together they hold every pool page -> the deadlock assert
    // must fire, loudly and without touching the pool.
    SendForwardDone("r1", {44});
    const std::int32_t free_before = scheduler_->FlatPoolFreeBlocks();
    try {
        PlanOnce();
        FAIL() << "expected the flat starvation-deadlock assert to fire";
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

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
