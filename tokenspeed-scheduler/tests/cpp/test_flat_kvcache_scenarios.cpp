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

// End-to-end scenario tests for the flat KV-cache FSM path
// (TOKENSPEED_FLAT_KVCACHE=ON), complementing test_flat_kvcache_lifecycle.cpp.
// Retract/writeback are deliberately deferred on the flat path (C slice), so
// they are NOT tested here.

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
// Chunked prefill: PrefillFirstChunk then PrefillChunk per chunk.
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

    ExecutionPlan chunk1 = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(chunk1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->flat_block_tables.count("full"), 1u);
    const std::size_t full_after_c1 = op1->flat_block_tables.at("full").at(0).size();
    EXPECT_GT(full_after_c1, 0u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);

    ExecutionPlan chunk2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(chunk2);
    ASSERT_NE(op2, nullptr);
    const auto& full_c2 = op2->flat_block_tables.at("full").at(0);
    EXPECT_GT(full_c2.size(), full_after_c1)
        << "second chunk should extend the full-history block table";
    for (std::int32_t id : full_c2) {
        EXPECT_GT(id, 0) << "full-history row must have no null hole";
    }

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
// Three cache groups: full + two sliding windows. Group 0 stays full-history
// to honor the flat consumer's block_tables_[0] contract.
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

    ASSERT_EQ(op->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(op->flat_block_tables.count("swa_small"), 1u);
    ASSERT_EQ(op->flat_block_tables.count("swa_big"), 1u);
    EXPECT_EQ(op->flat_block_tables.at("full").size(), 1u);
    EXPECT_EQ(op->flat_block_tables.at("swa_small").size(), 1u);
    EXPECT_EQ(op->flat_block_tables.at("swa_big").size(), 1u);

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
// Sub-page (w=3 < P=4) and page-straddling (w=5 = P+1) windows (M14): pins
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
        // fullySlidOutBlocks frees only FULLY slid-out pages: 1 <= real pages <= 2.
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
// Two full-history groups (no sliding window at all).
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
// Shared-pool accounting: out-of-order finishes each return exactly their pages.
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

// Chunked prefill slides the SWA window DURING prefill, then decode keeps
// sliding. Window convention used below: with N = tokens computed BEFORE a
// round's forward, the pending query at N attends keys [N-W+1, N], so the
// first kept page is (N-W+1)/page_size and everything below it is freed.
TEST_F(FlatChunkedPrefillSuite, ChunkedPrefillThenSwaSlidesToNullHole) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // 12 tokens (6 pages), max_scheduled_tokens=4 -> 3 prefill chunks.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    PlanOnce();  // chunk 1
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    // Chunk 2: N=4 -> first kept token 4-4+1=1 -> first kept page 0: no hole.
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

    // Chunk 3: N=8 -> first kept token 5 -> page 5/2=2: slots 0,1 punched MID-PREFILL.
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
    // Chunk-3 balance: slide freed 2 swa pages, acquire took 2/group -> net -2.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_after_c2 + 2 - 4)
        << "the mid-prefill slide must return the out-of-window pages to the pool";

    SendForwardDone("r1", {99});  // container size 13 (12 prompt + 1 sampled)

    // swa_rows[i] = the swa row round i's op carried (after slide + acquire).
    std::vector<std::vector<std::int32_t>> swa_rows;
    int tok = 100;
    for (int i = 0; i < 4; ++i) {
        ExecutionPlan plan = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(plan);
        ASSERT_NE(op, nullptr);
        for (std::int32_t id : op->flat_block_tables.at("full").at(0)) {
            EXPECT_GT(id, 0) << "full group must keep chunk-built history without holes (round " << i << ")";
        }
        swa_rows.push_back(op->flat_block_tables.at("swa").at(0));
        SendForwardDone("r1", {tok++});
    }

    auto null_count = [](const std::vector<std::int32_t>& row) {
        return std::count(row.begin(), row.end(), 0);
    };

    // Round 0 (finalize): N=12 -> first kept page 4; + reserve page -> 7 slots, 4 holes.
    ASSERT_EQ(swa_rows[0].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[0]), 4) << "finalize slides at the full prefill length";
    for (int s = 0; s <= 3; ++s) EXPECT_EQ(swa_rows[0][s], 0) << "slot " << s;
    for (int s = 4; s <= 6; ++s) EXPECT_GT(swa_rows[0][s], 0) << "slot " << s;

    // Round 1: N=13 -> first kept page 5; tail room absorbs the acquire.
    ASSERT_EQ(swa_rows[1].size(), 7u);
    EXPECT_EQ(null_count(swa_rows[1]), 5);
    for (int s = 0; s <= 4; ++s) EXPECT_EQ(swa_rows[1][s], 0) << "slot " << s;
    for (int s = 5; s <= 6; ++s) EXPECT_GT(swa_rows[1][s], 0) << "slot " << s;

    // Round 2: N=14 -> first kept token 11 -> page 5 (unchanged); acquire adds
    // page 7. Sliding at the container size 15 instead would free slot 5 early.
    ASSERT_EQ(swa_rows[2].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[2]), 5);
    EXPECT_GT(swa_rows[2][5], 0) << "slot 5 must survive round 2: key 11 of the pending query lives there";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[2][s], 0) << "slot " << s;

    // Round 3: N=15 -> first kept token 12 -> first kept page 6.
    ASSERT_EQ(swa_rows[3].size(), 8u);
    EXPECT_EQ(null_count(swa_rows[3]), 6);
    EXPECT_EQ(swa_rows[3][5], 0) << "slot 5 slides out once the query window has moved past key 11";
    for (int s = 6; s <= 7; ++s) EXPECT_GT(swa_rows[3][s], 0) << "slot " << s;

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatThreeGroupSuite, TwoRequestsBatchedAcrossThreeGroupsNoCollision) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(prefill);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 2u);

    for (const char* key : {"full", "swa_small", "swa_big"}) {
        ASSERT_EQ(op->flat_block_tables.count(key), 1u) << key;
        EXPECT_EQ(op->flat_block_tables.at(key).size(), 2u) << key;
    }

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
// Mixed batch: with enable_mixed_prefill_decode a decode and a prefill share
// one SoA op; stable_partition puts prefill rows ahead of decode rows.
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

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();                       // r1 prefill
    SendForwardDone("r1", {42});      // r1 -> decode

    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan mixed = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(mixed);
    ASSERT_NE(op, nullptr);

    ASSERT_EQ(op->request_ids.size(), 2u);
    EXPECT_EQ(op->num_extends(), 1u) << "exactly one prefill row (r2)";
    EXPECT_EQ(op->decode_input_ids.size(), 1u) << "exactly one decode row (r1)";

    EXPECT_EQ(op->request_ids.at(0), "r2") << "prefill partitioned first";
    EXPECT_EQ(op->request_ids.at(1), "r1") << "decode after prefill";

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

// Swa eviction state is tracked independently per request, not batch-wide.
TEST_F(FlatMixedBatchSuite, PerRequestSwaHoleAtDifferentDecodeDepths) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    PlanOnce();                    // both prefill together (mixed batch)
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // r1 goes well past the window (W=4 = 2 pages); r2 advances once, staying inside it.
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

    // Row order within the op is not guaranteed.
    const auto& ids = op->request_ids;
    auto row_of = [&](const std::string& id) -> std::size_t {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == id) return i;
        }
        ADD_FAILURE() << "request " << id << " not in op";
        return 0;
    };

    // r2 may or may not remain in the batch; assert only on rows present.
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
// page_size = 1: the flat path is not hard-wired to page_size=2.
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

    Submit(MakeRequestSpec("r1", /*num_pages=*/3));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* pop = FindFlatOp(prefill);
    ASSERT_NE(pop, nullptr);
    EXPECT_EQ(pop->flat_block_tables.at("full").at(0).size(), 3u)
        << "page_size=1 -> one page per prompt token";

    SendForwardDone("r1", {42});

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
// Pool-exhaustion admission. The first-chunk gate charges prompt + decode
// reserve = groups * ceil((tokens + 1) / page_size) blocks.
// ---------------------------------------------------------------------------
class FlatTinyPoolSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        // 11 physical pages -> 10 usable (page 0 is the null placeholder):
        // one 4-page prompt over 2 groups (8 prefill + 2 reserve) = the pool.
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

    // r1 gate: 8 prefill + 2 reserve = 10; prefill consumes 8 -> free 2.
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan1 = PlanOnce();
    ASSERT_NE(FindFlatOp(plan1), nullptr);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // r2 needs 4 blocks against only r1's 2-block decode headroom: deferred.
    Submit(MakeRequestSpec("r2", /*num_pages=*/1, /*start=*/101));
    SendForwardDone("r1", {99});
    ExecutionPlan starved = PlanOnce();
    const FlatForwardOperation* starved_op = FindFlatOp(starved);
    ASSERT_NE(starved_op, nullptr);
    ASSERT_EQ(starved_op->request_ids.size(), 1u) << "only r1's reserved decode step fits this round";
    EXPECT_EQ(starved_op->request_ids.at(0), "r1");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r2 stays intact in the waiting set";
    // r1 finalize at N=8 (W=4, page=2): first kept token 5 -> page 2 frees 2
    // swa pages; the reserve acquire takes 1 page/group: free stays 2.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

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

TEST_F(FlatTinyPoolSuite, PromptWhoseDecodeCannotFitIsDeferredAtFirstChunk) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // 10 tokens: prefill alone fits (10 blocks), but the gate charges
    // prompt + reserve = 2 * ceil(11/2) = 12 > 10.
    Submit(MakeRequestSpec("r1", /*num_pages=*/5));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->request_ids.empty()) << "self-cornering prompt must not be admitted";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "a deferred first chunk must not touch the pool";
}

// ---------------------------------------------------------------------------
// Prefill-slide admission: a long chunked prompt fits ONLY because the gate
// credits the slide the chunk itself performs (BlocksFreedByAdvance).
// ---------------------------------------------------------------------------
class FlatPrefillSlideAdmissionSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 14;  // 13 usable + the null placeholder (page 0)
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

    // page=2, W=4, 4-token chunks: c1 charges 4 blocks (2/group), 12 -> 8;
    // c2 (slide credit 0) charges 4, acquires 4 -> free 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindFlatOp(c1), nullptr);
    ASSERT_EQ(FindFlatOp(c1)->request_ids.size(), 1u);
    ExecutionPlan c2 = PlanOnce();
    ASSERT_NE(FindFlatOp(c2), nullptr);
    ASSERT_EQ(FindFlatOp(c2)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // c3 gate: chunk + reserve = 3 blocks/group = 6 vs raw free 4; the pending
    // slide at N=8 frees the 2 swa pages below token 5 -> 4 + 2 = 6, admitted.
    ExecutionPlan c3 = PlanOnce();
    const FlatForwardOperation* c3op = FindFlatOp(c3);
    ASSERT_NE(c3op, nullptr);
    ASSERT_EQ(c3op->request_ids.size(), 1u) << "final chunk must be admitted via the prefill slide credit";
    // Op balance: punch 2, acquire 2/group -> free 4 + 2 - 4 = 2.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    // Decode transition: gate needs 2, finalize-slide credit at N=12 gives 2.
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

TEST_F(FlatPrefillSlideAdmissionSuite, SinkPinsDeferAdmissionUntilWriteBackDone) {
    // Sink ON over the LongPromptAdmittedOnlyBecausePrefillSlides math: device 13 -> 12 usable, c1+c2
    // charge 8, c3 needs 6 = free 4 + slide credit 2 (pins only delay frees, so no extra device
    // headroom); host 12 usable (+null page 0) = op1 committed 4 + op2 in-flight 4 + op3 in-flight 4 at peak.
    config_.disable_l2_cache = false;
    config_.host_allocator.total_pages = 13;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);

    Submit(MakeRequestSpec("r1", /*num_pages=*/6));
    ExecutionPlan c1 = PlanOnce();
    ASSERT_NE(FindFlatOp(c1), nullptr);
    ASSERT_EQ(FindFlatOp(c1)->request_ids.size(), 1u);

    ExecutionPlan c2 = PlanOnce();  // registers pages 0,1 both groups: 4 pins + streaming op1
    ASSERT_NE(FindFlatOp(c2), nullptr);
    ASSERT_EQ(FindFlatOp(c2)->request_ids.size(), 1u);
    auto wb1 = ExtractCacheOpsOfKind<FlatWriteBackOperation>(c2);
    ASSERT_EQ(wb1.size(), 1u);
    const auto op1 = std::get<FlatWriteBackOperation>(wb1.front());
    ASSERT_EQ(op1.op_ids.size(), 1u);
    EXPECT_EQ(op1.src_pages.at(0).size(), 4u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // c3 needs 6 > free 4 + credit 0: the slide-out swa pages stay pinned by op1, so the chunk is
    // DEFERRED; a second starved round must NOT trip the deadlock assert while the store is in flight.
    ExecutionPlan d1 = PlanOnce();
    ASSERT_NE(FindFlatOp(d1), nullptr);
    EXPECT_TRUE(FindFlatOp(d1)->request_ids.empty());
    ExecutionPlan d2 = PlanOnce();
    ASSERT_NE(FindFlatOp(d2), nullptr);
    EXPECT_TRUE(FindFlatOp(d2)->request_ids.empty());
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    SendWriteBackDone(op1.op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 4);

    ExecutionPlan c3 = PlanOnce();  // unpinned + cached -> credit 2 restored: admitted; emits op2
    ASSERT_NE(FindFlatOp(c3), nullptr);
    ASSERT_EQ(FindFlatOp(c3)->request_ids.size(), 1u);
    auto wb2 = ExtractCacheOpsOfKind<FlatWriteBackOperation>(c3);
    ASSERT_EQ(wb2.size(), 1u);
    const auto op2 = std::get<FlatWriteBackOperation>(wb2.front());
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();  // finalize registers pages 4,5: emits op3
    ASSERT_NE(FindFlatOp(decode), nullptr);
    ASSERT_EQ(FindFlatOp(decode)->request_ids.size(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    auto wb3 = ExtractCacheOpsOfKind<FlatWriteBackOperation>(decode);
    ASSERT_EQ(wb3.size(), 1u);
    const auto op3 = std::get<FlatWriteBackOperation>(wb3.front());
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();  // reap: op2 + op3 pins (8 blocks) stay off the free list
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendWriteBackDone(op2.op_ids.at(0), /*success=*/true);
    SendWriteBackDone(op3.op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 12);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Pool 17 -> 16 usable: swa at full prompt length would need 10+10+2 = 22
// (infeasible); the plateau ceil((chunk+W-1)/P) = ceil(7/2) = 4 keeps the peak
// at full 10 + swa 4 + reserve 2 = 16 (exact fit) -- the flat-swa-alloc contract.
class FlatPrefillPlateauSuite : public FlatPrefillSlideAdmissionSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatPrefillSlideAdmissionSuite::MakeConfig();
        cfg.device_allocator.total_pages = 17;
        cfg.host_allocator.total_pages = 17;
        return cfg;
    }
};

TEST_F(FlatPrefillPlateauSuite, SwaWorkingSetPlateausWhileFullGrowsToPromptLength) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 16);

    Submit(MakeRequestSpec("r1", /*num_pages=*/10));  // 20 tokens, 5 chunks of 4
    std::size_t swa_peak = 0;
    std::size_t full_last = 0;
    for (std::int32_t chunk = 0; chunk < 5; ++chunk) {
        ExecutionPlan plan = PlanOnce();
        const FlatForwardOperation* op = FindFlatOp(plan);
        ASSERT_NE(op, nullptr) << "chunk " << chunk;
        ASSERT_EQ(op->request_ids.size(), 1u) << "chunk " << chunk << " must be admitted";
        const std::size_t swa_real = RealPages(op->flat_block_tables.at("swa")).size();
        const std::size_t full_real = RealPages(op->flat_block_tables.at("full")).size();
        EXPECT_LE(swa_real, 4u) << "swa exceeded the plateau at chunk " << chunk;
        EXPECT_GE(full_real, full_last) << "full group must grow monotonically, chunk " << chunk;
        swa_peak = std::max(swa_peak, swa_real);
        full_last = full_real;
    }
    EXPECT_EQ(swa_peak, 4u) << "the plateau bound must be reached, not just respected";
    EXPECT_EQ(full_last, 10u);

    SendForwardDone("r1", {99});
    ExecutionPlan decode = PlanOnce();
    ASSERT_NE(FindFlatOp(decode), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    SendForwardDone("r1", {100});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// ---------------------------------------------------------------------------
// Collective starvation deadlock: the scheduler must fail loudly, but only on
// the SECOND consecutive fully-starved round with nothing in flight (a queued
// Finish could make a single round a false positive).
// ---------------------------------------------------------------------------
class FlatCollectiveStarvationSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        // 13 physical pages -> 12 usable: two 2-page prompts charge
        // 2*ceil(5/2) = 6 blocks each at admission = exactly the pool.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 14;  // 13 usable + the null placeholder (page 0)
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

    // Round 1: both admitted (r1 gate 6 <= 12, r2 gate 6 <= 8 - 2); free 4.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/101));
    ExecutionPlan prefill = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(prefill);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 2u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});

    // Round 2: both decode transitions consume their 2-block reservations.
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

    // A starved round with r2's decode result STILL IN FLIGHT must stay quiet.
    SendForwardDone("r1", {44});
    ExecutionPlan quiet = PlanOnce();
    const FlatForwardOperation* quiet_op = FindFlatOp(quiet);
    ASSERT_NE(quiet_op, nullptr);
    EXPECT_TRUE(quiet_op->request_ids.empty());

    // Nothing in flight now: the FIRST fully starved round still stays quiet.
    SendForwardDone("r2", {144});
    ExecutionPlan starved1 = PlanOnce();
    const FlatForwardOperation* starved1_op = FindFlatOp(starved1);
    ASSERT_NE(starved1_op, nullptr);
    EXPECT_TRUE(starved1_op->request_ids.empty()) << "first starved round is quiet (two-round hardening)";

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
// Abort-mid-flight pool balance: abort mid-chunked-prefill or mid-decode must
// return every page to the pool.
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
// Failure-path page release, event level: the admission gate makes coordinator
// failures unreachable via NextExecutionPlan, so drive the FSM events directly.
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

    // Second chunk: 8 tokens -> 8 blocks > 1 free: the Acquire throws.
    EXPECT_THROW(
        request.Apply(fsm::SchedulePrefillEvent{/*tokens_this_round=*/8,
                                                /*reserve_num_tokens_in_next_schedule_event=*/0,
                                                /*hybrid_prefix_cache=*/nullptr, &coordinator}),
        std::runtime_error);
    EXPECT_EQ(pool.NumFreeBlocks(), 5) << "failure path must return the request's pages to the pool";

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

    // Whole 4-token prompt in one chunk -> PrefillDone holding all 4 blocks.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());
    ASSERT_EQ(pool.NumFreeBlocks(), 0);

    // Decode transition needs 1 fresh page per group (tails full) with 0 free.
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

    // Prefill takes 4 of 6 blocks; decode step 1 takes a fresh page per group
    // (pool empty), step 2 fills the tail free -> mid-decode on a starved pool.
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

    // Third step needs a fresh page per group with 0 free.
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

    // First chunk needs 4 blocks > 3 free: throws before any state commits.
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

// ReqPoolAllocator::Allocate() throws before tables are populated; with BlockRef
// RAII the pool balances either way -- this pins the balance, not the order.
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
// SWA off-by-one regression: the Decoding transition must slide at
// N = container_size - decode_input_tokens, NOT the container size.
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

    // 4-token prompt in one chunk (page=2, W=4) -> PrefillDone, 2 pages/group.
    request.Apply(fsm::SchedulePrefillFirstChunkEvent{
        /*tokens_this_round=*/4, /*decode_input_tokens=*/1, /*device_allocator=*/nullptr, &req_pool, MatchResult{},
        Role::kFused, /*kv_prefix_cache=*/nullptr, /*disable_l2_cache=*/true, /*loadback_diff=*/{},
        /*hybrid_prefix_cache=*/nullptr, /*mamba_allocator=*/nullptr, /*mamba_loadback_nodes=*/{}, &coordinator});
    ASSERT_TRUE(request.Is<fsm::PrefillDone>());

    const auto swa_slot_null = [&](std::int32_t i) { return request.FlatBlockTablesRef()[1].Blocks()[i]->IsNull(); };

    // Size 5, decode transition (no slide): 3 pages.
    request.Apply(fsm::ExtendResultEvent{"r1", {100}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    ASSERT_TRUE(request.Is<fsm::Decoding>());
    ASSERT_EQ(request.FlatBlockTablesRef()[1].NumBlocks(), 3);
    EXPECT_FALSE(swa_slot_null(0));

    // Size 6 -> N=5; keys [2,5] -> page 0 out: slot 0 punched, slot 1 kept.
    request.Apply(fsm::ExtendResultEvent{"r1", {101}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_TRUE(swa_slot_null(0));
    EXPECT_FALSE(swa_slot_null(1));

    // Size 7 -> N=6; keys [3,6]: key 3 still lives in page 1, so slot 1 must
    // survive (sliding at the container size 7 would free it here).
    request.Apply(fsm::ExtendResultEvent{"r1", {102}});
    const std::int32_t free_before = pool.NumFreeBlocks();
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_FALSE(swa_slot_null(1)) << "key 3 of the pending query lives in page 1; freeing it is the off-by-one";
    EXPECT_TRUE(swa_slot_null(0));
    // This round slides nothing and acquires one fresh page per group.
    EXPECT_EQ(pool.NumFreeBlocks(), free_before - 2);

    // Size 8 -> N=7; keys [4,7] -> page 1 fully out, punched exactly now.
    request.Apply(fsm::ExtendResultEvent{"r1", {103}});
    request.Apply(fsm::ScheduleDecodeEvent{/*decode_input_tokens=*/1, /*hybrid_prefix_cache=*/nullptr, &coordinator});
    EXPECT_TRUE(swa_slot_null(1));
    EXPECT_FALSE(swa_slot_null(2));

    for (CacheBlock* b : request.FlatBlockTablesRef()[0].Blocks()) {
        EXPECT_FALSE(b->IsNull());
    }

    request.Apply(fsm::AbortEvent{&coordinator});
    EXPECT_TRUE(request.Is<fsm::Finished>());
}

// ---------------------------------------------------------------------------
// Decode-reserve ledger (flat_reserved_pages_): promised decode pages are only
// Acquired one round later; nobody may be admitted into them in between.
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

    // a: gate 2*ceil(7/2) = 8 <= 10; prefill consumes 6 -> free 4, promise 2.
    // b: needs 2*ceil(3/2) = 4 > 4 - a's promised 2 -> must defer.
    Submit(MakeRequestSpec("a", /*num_pages=*/3));
    Submit(MakeRequestSpec("b", /*num_pages=*/1, /*start=*/101));
    ExecutionPlan round1 = PlanOnce();
    const FlatForwardOperation* op1 = FindFlatOp(round1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u) << "b must not be admitted into a's promised decode pages";
    EXPECT_EQ(op1->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 4);

    // a's decode transition consumes its own reservation (the gate excludes it).
    SendForwardDone("a", {99});
    ExecutionPlan round2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(round2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u) << "a's decode must proceed into its reserved pages";
    EXPECT_EQ(op2->request_ids.at(0), "a");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 2);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

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

    // Abort BEFORE the reserve is acquired: the ledger entry must drop too.
    SendAbort(*scheduler_, "a");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // b needs the whole pool: gate 2*ceil(9/2) = 10 <= 10 only without a phantom.
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
// M9 cross-request prefix hits, end to end: admission match -> FSM claim ->
// input window starts past the hit (disable_prefix_cache=false, W=32).
// Pool convention: claiming a cached free block (TouchBlock) removes it from
// NumFreeBlocks like an allocation -- a hit's delta = claimed + acquired pages.
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

    // Prefill -> one decode round -> finish; returns the PREFILL op's per-group
    // rows. The decode round is load-bearing: the finalize registers the page
    // hashes, and finish frees the blocks WITH hashes intact (still matchable).
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

TEST_F(FlatPrefixHitSuite, TwoRequestsSharePrefixReusePages) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";
    ASSERT_EQ(r1_rows.at("full").size(), 4u);
    ASSERT_EQ(r1_rows.at("swa").size(), 4u);

    // r2: 12 tokens, first 8 == r1's. Hit: cap = (12-1)/2 = 5 pages; r1
    // registered 4, r2's page-4 hash chains off different tail tokens -> full
    // hits 4; swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 tokens.
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PageSize());  // tokens 1..8 == r1's
    const token_vec_t tail = MakeTokens(/*count=*/4, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 4);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 12);
    EXPECT_EQ(op->input_ids, tail);
    // Page-space fields (radix-hit parity): sizes counts everything new to the
    // request's table this round = 4 claimed + ceil(4/2) = 6.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 6);
    ASSERT_EQ(op->occupied_pages.at(0).size(), 6u);

    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), r1_rows.at("full"), "full row");
    ExpectRowPrefixEq(op->flat_block_tables.at("swa").at(0), r1_rows.at("swa"), "swa row");

    // Pool: claim 4/group (8) + acquire ceil(4/2) = 2/group (4) = 12. The
    // decode reserve is only PROMISED here (ledger), not acquired.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);

    // Finalize registers pages 4..5 and acquires the reserve: 1 fresh page/group.
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

// The hit is capped at (PrefillSize-1)/page_size pages so the last token is
// always recomputed to produce logits.
TEST_F(FlatPrefixHitSuite, FullHitCapsAtLastToken) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const RequestSpec r1 = MakeRequestSpec("r1", /*num_pages=*/4);  // 8 tokens
    RunLifecycle(r1);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2 = the same 8 tokens: cap = (8-1)/2 = 3 pages -> hit 3 = 6 tokens.
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

    // Reserve: 1 fresh page per group (tail full).
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

class FlatPrefixHitDisabledSuite : public FlatPrefixHitSuite {
protected:
    bool DisablePrefixCache() const override { return true; }
};

TEST_F(FlatPrefixHitDisabledSuite, DisablePrefixCacheSkipsMatch) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

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
    // Pool: 6 fresh pages per group = 12.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatPrefixHitSuite, PartialHit) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));  // tokens 1..8
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2: 12 tokens, only the first 4 match r1 (pages 0..1); the hash chain
    // propagates the divergence to every later page. Hit = 2 pages = 4 tokens.
    token_vec_t r2_tokens = MakeTokens(/*count=*/4);  // 1..4 == r1's first 4
    const token_vec_t tail = MakeTokens(/*count=*/8, /*start=*/801);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 8);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 4);
    EXPECT_EQ(op->input_ids, tail);
    // 2 claimed + ceil(8/2) = 4 fresh pages per group.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 6);

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

// Small window: the SWA group's bounded right-to-left scan stops once its
// contiguous run is satisfied, claiming r1's punched slots as null holes.
class FlatPrefixHitSmallWindowSuite : public FlatPrefixHitSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(FlatPrefixHitSmallWindowSuite, SwaGroupHitRespectsWindow) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // r1's finalize REGISTERS all 4 swa hashes BEFORE ReclaimExpired(8) punches
    // slots 0,1 -- punched blocks reach the free list with hashes, matchable.
    const auto r1_rows = RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
    ASSERT_EQ(r1_rows.at("swa").size(), 4u);

    // r2: 10 tokens, first 8 == r1's. Fixpoint (W=4, page=2, contiguous_needed
    // = ceil(3/2) = 2): cap = (10-1)/2 = 4, full matches 4; swa scan stops at
    // run 2 -> keep 4 with 2 holes -> common stays 4 = 8 hit tokens.
    token_vec_t r2_tokens = MakeAlignedTokens(/*num_pages=*/4, PageSize());  // 1..8 == r1's
    const token_vec_t tail = MakeTokens(/*count=*/2, /*start=*/901);
    r2_tokens.insert(r2_tokens.end(), tail.begin(), tail.end());
    Submit(MakeSpecWithTokens("r2", r2_tokens));

    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->input_ids, tail);
    // 4 claimed slots (real or hole) + 1 fresh page.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 5);

    const auto& full_row = op->flat_block_tables.at("full").at(0);
    ASSERT_EQ(full_row.size(), 5u);
    ExpectRowPrefixEq(full_row, r1_rows.at("full"), "full row");
    EXPECT_GT(full_row[4], 0);

    const auto& swa_row = op->flat_block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 5u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]);
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);
    // Window invariant (mirrors ExpectSwaWindowIntact): the last
    // contiguous_needed = 2 slots of the claimed prefix must be real.
    for (std::size_t i = 2; i < 4; ++i) {
        EXPECT_GT(swa_row[i], 0) << "null hole inside the last window of the claimed prefix at slot " << i;
    }

    // Pool: full claims 4 + swa claims 2 (holes claim nothing) + 1 fresh
    // page/group = 8 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Regression (M9-F1): the first-chunk gate must also charge the free blocks
// the CLAIM consumes (TouchBlock removes a ref-0 cached block from the free list).
class FlatPrefixHitTightPoolSuite : public FlatPrefixHitSuite {
protected:
    std::int32_t TotalPages() const override { return 11; }
};

TEST_F(FlatPrefixHitTightPoolSuite, GateChargesFreeHitBlocksClaimWillConsume) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 10);

    // r1: 2 pages/group registered then freed cached -> 4 of the 10 free
    // blocks are ref-0 CACHED (r2's future hit set), the other 6 plain free.
    RunLifecycle(MakeRequestSpec("r1", /*num_pages=*/2));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r3 (pool holder): 1 fresh page/group, free 10 -> 8. Pops come from the
    // LRU head (never-used blocks) -- r1's cached blocks survive with hashes.
    Submit(MakeRequestSpec("r3", /*num_pages=*/1, /*start=*/501));
    ExecutionPlan r3_prefill = PlanOnce();
    ASSERT_NE(FindFlatOp(r3_prefill), nullptr);
    ASSERT_EQ(FindFlatOp(r3_prefill)->request_ids.size(), 1u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 8);

    // r3's finalize acquires its decode reserve (1 fresh page/group): 8 -> 6,
    // erasing its ledger entry: r2's gate below reads raw free 6, no reserves.
    SendForwardDone("r3", {599});
    PlanOnce();
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 6);

    // r2: 8 tokens, first 4 == r1's. Hit: cap = (8-1)/2 = 3, r1 registered 2
    // -> fixpoint 2 blocks = 4 tokens, all 4 hit blocks ref-0 free. Gate:
    // new/reserve 2*ceil(5/2) = 6 + claim 4 = 10 > free 6 -> defers untouched.
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

    // r3 finishes -> free 10; r2's charge 6 + 4 = 10 == 10: admitted exactly
    // at the boundary. Claim pulls 4, Acquire takes 2/group: free 10 -> 2.
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

    // r2's finalize acquires the promised 2-block reserve: pool hits exactly 0.
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
// M13 decode-block caching: pages filled DURING decode register via the hash
// chain (DecodeStep: register -> slide -> acquire), so a later turn hits PAST
// the previous prompt boundary. Fill timing: a round at container Size s has
// N = s - 1 computed and registers pages up to N/page_size -- a tail page
// registers one round late (finishing earlier frees its block hashless).
// ---------------------------------------------------------------------------
class FlatDecodeCachingSuite : public FlatPrefixHitSuite {
protected:
    // Deliver one sampled token and run the next schedule round, returning the
    // per-group rows the round's op carried. Single-request rounds only.
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

    // Turn 1: prompt {1,2,3,4}, generated 101..105 (page=2). Finalize registers
    // prompt pages 0,1; +103 (N=6) registers page 2; +105 (N=8) registers page
    // 3 (tail one round late: 105 exists only to push N past 8). Returns the
    // last round's rows: 5 slots, the first 4 = the conversation's pages 0..3.
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

    // Turn-2 prompt: r1's 4 prompt tokens + first 4 generated + 2 new = 10;
    // pages 0..3 match r1's registration by content.
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

TEST_F(FlatDecodeCachingSuite, DecodeFilledPageBecomesHittable) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    const auto r1_rows = RunTurnOne();
    ASSERT_EQ(r1_rows.at("full").size(), 5u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "r1 must fully reclaim before r2 runs";

    // Hit: cap = (10-1)/2 = 4 -> pages 0..3, all registered by r1 (RunTurnOne);
    // swa (W=32, needed 16 > 4) keeps 4 -> fixpoint 4 blocks = 8 hit tokens.
    Submit(MakeSpecWithTokens("r2", MakeTurnTwoPrompt()));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);

    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op->prefill_lengths.at(0), 10);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/901));
    // 4 claimed + ceil(2/2) = 1 fresh page.
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 5);

    // Slots 2,3 are the pages r1's decode filled, beyond its prompt boundary.
    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    const std::vector<std::int32_t> swa_prefix(r1_rows.at("swa").begin(), r1_rows.at("swa").begin() + 4);
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op->flat_block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: claim 4/group (8) + 1 fresh/group (2) = 10 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 10);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after r2 finishes";
}

TEST_F(FlatDecodeCachingSuite, MultiTurnConversationReusesResponsePages) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    RunTurnOne();  // registers conversation pages 0..3
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // Turn 2: hit 4 pages, then decode 201..203: +201 finalize registers page
    // 4 = {901,902}; +203 (N=12) registers page 5 (tail one round late).
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

    // Turn 3 hit: cap = (16-1)/2 = 7; pages 0..5 registered (0..3 by r1, 4..5
    // by r2), page 6 never full in any request -> fixpoint 6 blocks = 12 hit
    // tokens, into r2's response (page 5).
    Submit(MakeSpecWithTokens("r3", MakeTurnThreePrompt()));

    ExecutionPlan turn3 = PlanOnce();
    const FlatForwardOperation* op3 = FindFlatOp(turn3);
    ASSERT_NE(op3, nullptr);
    ASSERT_EQ(op3->request_ids.size(), 1u);
    EXPECT_EQ(op3->extend_prefix_lens.at(0), 12) << "hit grows across turns: 8 -> 12 tokens";
    EXPECT_EQ(op3->input_lengths.at(0), 4);
    EXPECT_EQ(op3->prefill_lengths.at(0), 16);
    EXPECT_EQ(op3->input_ids, (token_vec_t{203, 951, 952, 953}));
    // 6 claimed + ceil(4/2) = 2 fresh pages.
    EXPECT_EQ(op3->begins.at(0), 0);
    EXPECT_EQ(op3->sizes.at(0), 8);

    // Slots 0..3 are r1's blocks (re-freed cached by r2), 4..5 r2's own pages.
    const std::vector<std::int32_t> full_prefix(r2_rows.at("full").begin(), r2_rows.at("full").begin() + 6);
    const std::vector<std::int32_t> swa_prefix(r2_rows.at("swa").begin(), r2_rows.at("swa").begin() + 6);
    ExpectRowPrefixEq(op3->flat_block_tables.at("full").at(0), full_prefix, "full row");
    ExpectRowPrefixEq(op3->flat_block_tables.at("swa").at(0), swa_prefix, "swa row");

    // Pool: 6 claimed/group (12) + 2 fresh/group (4) = 16.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 16);

    SendForwardDone("r3", {299});
    PlanOnce();
    SendForwardDone("r3", {300});
    SendFinish("r3");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool back to baseline after all three turns";
}

// A decode page REGISTERS (DecodeStep registers before the slide) and a later
// ReclaimExpired punches it: the punch frees the block WITH its hash intact.
class FlatDecodeCachingSmallWindowSuite : public FlatDecodeCachingSuite {
protected:
    std::int32_t SlidingWindowTokens() const override { return 4; }
};

TEST_F(FlatDecodeCachingSmallWindowSuite, SwaPunchedDecodePageStillHittable) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // RunTurnOne's fill timing, inlined because the punch round +106 must land
    // BEFORE finish. W=4 slides on top (punched pages = (N-3)/2): +102 punches
    // slot 0, +103 registers page 2, +104 punches slot 1, +105 registers page 3.
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

    // +106 -> N=9 -> first kept page 3: slot 2 (REGISTERED at +103) is punched;
    // its block reaches the free list with the hash intact.
    const auto punched = AdvanceOneRound("r1", 106);
    EXPECT_EQ(punched.at("swa")[2], 0) << "the registered decode page must be punched by now";
    SendFinish("r1");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    // r2: same 8-token prefix + 2 new. Fixpoint (W=4, needed 2): cap =
    // (10-1)/2 = 4, all four hashes cached (0,1,2 punched WITH hash); full
    // matches 4, swa bounded scan keeps 4 (2 holes) -> common 4 = 8 hit tokens.
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

    const std::vector<std::int32_t> full_prefix(r1_rows.at("full").begin(), r1_rows.at("full").begin() + 4);
    ExpectRowPrefixEq(op->flat_block_tables.at("full").at(0), full_prefix, "full row");

    // Slot 2's expected id was captured at the +105 round, before the punch.
    const auto& swa_row = op->flat_block_tables.at("swa").at(0);
    ASSERT_EQ(swa_row.size(), 5u);
    EXPECT_EQ(swa_row[0], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[1], 0) << "out-of-window slot claimed as a null hole";
    EXPECT_EQ(swa_row[2], r1_rows.at("swa")[2]) << "punched decode page claimed back by hash";
    EXPECT_EQ(swa_row[3], r1_rows.at("swa")[3]);
    EXPECT_GT(swa_row[4], 0);

    // Pool: full claims 4 + swa claims 2 + 1 fresh/group = 8 off the free count.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendForwardDone("r2", {199});
    PlanOnce();
    SendForwardDone("r2", {200});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Registration writes hashes only -- never refcounts.
TEST_F(FlatDecodeCachingSuite, PoolBalanceAcrossDecodeCaching) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    RunTurnOne();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "turn 1: decode registration must not hold refs";

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

// ---------------------------------------------------------------------------
// M15 streaming L2 sink: pages registered by a planning round batch into ONE
// D2H write-back; WriteBackDone commits/aborts the host index and unpins the
// pinned source blocks. Byte movement itself is Phase D.
// ---------------------------------------------------------------------------
class FlatStreamingSinkSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 9;  // 8 usable + the null placeholder (page 0, device convention)
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;
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

    // Prefill -> finalize; the finalize round registers the prompt's page
    // hashes, so it is the round whose plan carries the streaming write-back.
    ExecutionPlan RunToFinalize(const RequestSpec& spec) {
        Submit(spec);
        PlanOnce();  // prefill
        SendForwardDone(spec.request_id, {9001});
        return PlanOnce();  // PrefillDone -> Decoding: registration + drain
    }

    void FinishAndReap(const std::string& id) {
        SendForwardDone(id, {9002});
        SendFinish(id);
        PlanOnce();  // reap
    }

    static std::optional<FlatWriteBackOperation> FindFlatWriteBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<FlatWriteBackOperation>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged write-back list";
        return std::get<FlatWriteBackOperation>(ops.front());
    }
};

TEST_F(FlatStreamingSinkSuite, RegisteredPagesEmitWriteBackAndIndexOnDone) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb = FindFlatWriteBack(finalize);
    ASSERT_TRUE(wb.has_value()) << "finalize-registered pages must emit a streaming write-back";
    ASSERT_EQ(wb->op_ids.size(), 1u);
    EXPECT_EQ(wb->src_pages.at(0).size(), 8u) << "4 registered pages x 2 groups = 8 D2H pairs";
    EXPECT_EQ(wb->dst_pages.at(0).size(), 8u);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 0) << "nothing indexed until WriteBackDone";
    EXPECT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 0) << "all 8 host pages held in flight";

    FinishAndReap("r1");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8)
        << "the 8 pinned sources stay off the free list past request finish";
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 0);

    SendWriteBackDone(wb->op_ids.at(0), /*success=*/true);
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "commit unpins every source block";
}

TEST_F(FlatStreamingSinkSuite, DuplicateRegistrationsAreDroppedAtDrain) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb1 = FindFlatWriteBack(finalize1);
    ASSERT_TRUE(wb1.has_value());
    FinishAndReap("r1");
    SendWriteBackDone(wb1->op_ids.at(0), /*success=*/true);
    PlanOnce();
    ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4));  // identical tokens
    EXPECT_FALSE(FindFlatWriteBack(finalize2).has_value())
        << "already-indexed keys must not re-emit a write-back";
    FinishAndReap("r2");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start)
        << "duplicate candidates are unpinned at drain, pool back to baseline";
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
}

TEST_F(FlatStreamingSinkSuite, FailedWriteBackAbortsAndUnpins) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb = FindFlatWriteBack(finalize);
    ASSERT_TRUE(wb.has_value());
    FinishAndReap("r1");
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);

    SendWriteBackDone(wb->op_ids.at(0), /*success=*/false);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 0) << "a failed transfer must not be indexed";
    EXPECT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 8) << "aborted host pages return to the host pool";
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "abort still unpins the sources";
}

TEST_F(FlatStreamingSinkSuite, HostPoolExhaustionSkipsSilently) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    ExecutionPlan finalize1 = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb1 = FindFlatWriteBack(finalize1);
    ASSERT_TRUE(wb1.has_value());
    FinishAndReap("r1");
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8);
    ASSERT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 0) << "r1 holds all 8 host pages in flight";

    ExecutionPlan finalize2 = RunToFinalize(MakeRequestSpec("r2", /*num_pages=*/4, /*start=*/501));
    EXPECT_FALSE(FindFlatWriteBack(finalize2).has_value())
        << "a fully-consumed host pool drops every candidate: no op at all";
    FinishAndReap("r2");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8)
        << "r2's candidates unpinned at drain; only r1's 8 pins remain";

    SendWriteBackDone(wb1->op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "everything balances after r1's commit";
}

TEST_F(FlatStreamingSinkSuite, SameRoundDuplicateKeysDedupeAtDrain) {
    // Host pool with headroom (16 usable) so duplicates are dropped by the drain's batch
    // dedupe, NOT by pool exhaustion: two IDENTICAL prompts registering in one round drain
    // 16 candidates into 8 pairs.
    config_.host_allocator.total_pages = 17;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    PlanOnce();  // both prefill (batch 2 <= max_batch_size 8, 16 tokens <= budget 64)
    SendForwardDone("r1", {9001});
    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();  // both register, one merged drain
    auto wb = FindFlatWriteBack(finalize);
    ASSERT_TRUE(wb.has_value());
    ASSERT_EQ(wb->op_ids.size(), 1u);
    EXPECT_EQ(wb->src_pages.at(0).size(), 8u) << "each key must be emitted at most once across both requests";
    EXPECT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 8) << "duplicates must not consume host pages";
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 0);

    FinishAndReap("r1");
    FinishAndReap("r2");
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 8)
        << "only the emitted op's 8 pins survive; the duplicate candidates unpinned at drain";

    SendWriteBackDone(wb->op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    EXPECT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 16) << "published pages are free-and-cached";
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

TEST_F(FlatStreamingSinkSuite, MidDrainPoolFillEmitsPartialOp) {
    // 4 usable host pages against 8 candidates: the drain emits the 4 that fit and drops the
    // rest -- a partial op IS the contract when the pool fills mid-batch.
    config_.host_allocator.total_pages = 5;
    scheduler_ = std::make_unique<Scheduler>(config_);
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb = FindFlatWriteBack(finalize);
    ASSERT_TRUE(wb.has_value());
    EXPECT_EQ(wb->src_pages.at(0).size(), 4u) << "4 of 8 candidates fit";
    EXPECT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 0);

    FinishAndReap("r1");
    SendWriteBackDone(wb->op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 4);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "dropped candidates unpinned at drain";
}

TEST_F(FlatStreamingSinkSuite, DuplicateWriteBackDoneIsIgnored) {
    ExecutionPlan finalize = RunToFinalize(MakeRequestSpec("r1", /*num_pages=*/4));
    auto wb = FindFlatWriteBack(finalize);
    ASSERT_TRUE(wb.has_value());
    FinishAndReap("r1");
    const std::int32_t free_after_reap = scheduler_->FlatPoolFreeBlocks();

    SendWriteBackDone(wb->op_ids.at(0), /*success=*/true);
    ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    const std::int32_t free_after_ack = scheduler_->FlatPoolFreeBlocks();
    EXPECT_EQ(free_after_ack, free_after_reap + 8);

    // A replayed ack must be a no-op (the ledger already retired the op).
    SendWriteBackDone(wb->op_ids.at(0), /*success=*/true);
    EXPECT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_after_ack);
}

// ---------------------------------------------------------------------------
// M15 host-hit load-back: an admission whose device match ends inside the host
// index extends it from the host tier; the plan carries one H2D load-back and
// LoadBackDone releases the host load pins and the destination-page pins.
// ---------------------------------------------------------------------------
class FlatHostHitSuite : public FlatStreamingSinkSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatStreamingSinkSuite::MakeConfig();
        cfg.disable_prefix_cache = false;
        // 13 device pages -> 12 free (page 0 is null): the 5-page churn request's peak
        // (10 prefill + 2 reserve) spans the whole free list, recycling r1's 8 cached pages.
        cfg.device_allocator.total_pages = 13;
        cfg.host_allocator.total_pages = 33;  // ample (+null page 0): r1's 8 + the churn's 10 entries fit un-evicted
        for (auto& g : cfg.paged_cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }

    static std::optional<FlatLoadBackOperation> FindFlatLoadBack(const ExecutionPlan& plan) {
        auto ops = ExtractCacheOpsOfKind<FlatLoadBackOperation>(plan);
        if (ops.empty()) {
            return std::nullopt;
        }
        EXPECT_EQ(ops.size(), 1u) << "the plan must carry at most one merged load-back list";
        return std::get<FlatLoadBackOperation>(ops.front());
    }

    // Full sink lifecycle: prefill -> finalize (registration + drain) -> reap;
    // returns the write-back the finalize emitted.
    std::optional<FlatWriteBackOperation> RunSinkLifecycle(const RequestSpec& spec) {
        ExecutionPlan finalize = RunToFinalize(spec);
        FinishAndReap(spec.request_id);
        return FindFlatWriteBack(finalize);
    }

    // r1 (tokens 1..8) indexes 8 host entries (4 pages x 2 groups); the churn request
    // then floods the free list so r1's pages survive ONLY on the host tier.
    void SeedHostThenEvictDevice() {
        auto wb1 = RunSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4));
        ASSERT_TRUE(wb1.has_value());
        SendWriteBackDone(wb1->op_ids.at(0), /*success=*/true);
        ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
        // Published pages return to the free list cached-and-evictable (device convention),
        // so the full 32 usable pages stay allocatable while 8 of them are hittable.
        ASSERT_EQ(scheduler_->FlatHostPoolFreeBlocks(), 32);

        auto wb3 = RunSinkLifecycle(MakeRequestSpec("churn", /*num_pages=*/5, /*start=*/501));
        ASSERT_TRUE(wb3.has_value());
        SendWriteBackDone(wb3->op_ids.at(0), /*success=*/true);
        ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 18);
        ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 12) << "both seeding requests fully retired";
    }
};

TEST_F(FlatHostHitSuite, HostHitLoadsBackAfterDeviceEviction) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 12);
    SeedHostThenEvictDevice();

    // r2 == r1's tokens: hash cap = (8-1)/2 = 3 pages, device common 0 (all recycled) ->
    // host extension 3 blocks; real pages = full 3 + swa tail ceil((W-1)/P) = 2 -> 5 pairs.
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindFlatLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->op_ids.size(), 1u);
    ASSERT_EQ(lb->src_pages.at(0).size(), 5u);
    ASSERT_EQ(lb->dst_pages.at(0).size(), 5u);

    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    // The input window skips the 6 host-hit tokens exactly as a device hit would.
    EXPECT_EQ(op->input_lengths.at(0), 2);
    EXPECT_EQ(op->extend_prefix_lens.at(0), 6);
    EXPECT_EQ(op->prefill_lengths.at(0), 8);
    EXPECT_EQ(op->input_ids, MakeTokens(/*count=*/2, /*start=*/7));
    EXPECT_EQ(op->begins.at(0), 0);
    EXPECT_EQ(op->sizes.at(0), 4) << "3 extension + 1 fresh page, all new to the table";

    // Wire pairs are group-major: full ext slots 0..2, then swa slots 1..2 (slot 0
    // is a pre-window hole = the null page 0).
    const auto& full_row = op->flat_block_tables.at("full").at(0);
    const auto& swa_row = op->flat_block_tables.at("swa").at(0);
    ASSERT_EQ(full_row.size(), 4u);
    ASSERT_EQ(swa_row.size(), 4u);
    const auto& dst = lb->dst_pages.at(0);
    EXPECT_EQ(dst.at(0), full_row.at(0));
    EXPECT_EQ(dst.at(1), full_row.at(1));
    EXPECT_EQ(dst.at(2), full_row.at(2));
    EXPECT_EQ(swa_row.at(0), 0) << "swa slot 0 is the pre-window hole";
    EXPECT_EQ(dst.at(3), swa_row.at(1));
    EXPECT_EQ(dst.at(4), swa_row.at(2));

    // The 5 matched host entries stay load-pinned until LoadBackDone retires the op.
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 5);
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);

    // r2 holds full 3 ext + 1 fresh and swa 2 ext + 1 fresh = 7 blocks.
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 7);

    SendForwardDone("r2", {9001});
    PlanOnce();  // finalize: page-3 keys are already indexed, so no new write-back pins
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool balances after the host-hit request";
}

TEST_F(FlatHostHitSuite, EmptyHostIndexEmitsNoLoadBack) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    ASSERT_NE(FindFlatOp(plan), nullptr);
    EXPECT_FALSE(FindFlatLoadBack(plan).has_value()) << "an empty host index must emit no load-back";
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);
}

TEST_F(FlatHostHitSuite, AbandonedAdmissionUnpins) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    SeedHostThenEvictDevice();

    // Filler: 5 pages -> 10 prefill + 2 reserve = the whole pool while it decodes.
    Submit(MakeRequestSpec("filler", /*num_pages=*/5, /*start=*/701));
    PlanOnce();
    SendForwardDone("filler", {9001});
    ExecutionPlan filler_finalize = PlanOnce();  // acquires the reserve: free = 0
    auto filler_wb = FindFlatWriteBack(filler_finalize);
    ASSERT_TRUE(filler_wb.has_value());
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);

    // r2's host match takes 5 pins, but the gate needs 4 + 5 ext > 0 free: the
    // abandoning return must give the pins back.
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    ExecutionPlan starved = PlanOnce();
    EXPECT_FALSE(FindFlatLoadBack(starved).has_value());
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0) << "an abandoned admission must unpin its host match";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    // Free the filler (its write-back pins included) -> r2 admits with the load-back.
    SendForwardDone("filler", {9002});
    SendFinish("filler");
    SendWriteBackDone(filler_wb->op_ids.at(0), /*success=*/true);
    ExecutionPlan plan = PlanOnce();
    auto lb = FindFlatLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    EXPECT_EQ(lb->src_pages.at(0).size(), 5u);
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 5);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);
    SendForwardDone("r2", {9001});
    PlanOnce();
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool balances after the deferred host hit";
}

TEST_F(FlatHostHitSuite, AbortDuringLoadKeepsPagesPinned) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindFlatLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->dst_pages.at(0).size(), 5u);
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 7);

    // Abort while the H2D copy is in flight: the reap returns only the 2 fresh pages;
    // the 5 load destinations must stay off the free list until LoadBackDone.
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 5)
        << "in-flight load destinations must not be reusable";
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 5) << "the host sources stay pinned too";

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "LoadBackDone releases the destinations";
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);
}

// An abort-during-load leaves pages held ONLY by the load ledger; the starvation
// deadlock check must count that as in-flight (LoadBackDone will free them), not
// crash a candidate that double-starves against the ticket-held pages.
TEST_F(FlatHostHitSuite, StarvationWaitsForInFlightLoads) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    SeedHostThenEvictDevice();

    // Same shape as AbortDuringLoadKeepsPagesPinned: 5 destinations stay ticket-held.
    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindFlatLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    SendAbort(*scheduler_, "r2");
    PlanOnce();  // reap
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 5);

    // r3 (fresh tokens, no host hit) charges 8 prefill + 2 reserve = 10 > 7 free: deferred.
    Submit(MakeRequestSpec("r3", /*num_pages=*/4, /*start=*/901));
    ExecutionPlan starved1 = PlanOnce();
    ASSERT_NE(FindFlatOp(starved1), nullptr);
    EXPECT_TRUE(FindFlatOp(starved1)->request_ids.empty());
    // The SECOND consecutive starved round is where the deadlock assert would
    // fire; the in-flight load ledger must keep the starvation counter quiet.
    ExecutionPlan starved2 = PlanOnce();
    ASSERT_NE(FindFlatOp(starved2), nullptr);
    EXPECT_TRUE(FindFlatOp(starved2)->request_ids.empty());
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "deferred r3 stays intact in the waiting set";

    // LoadBackDone frees the 5 destinations: r3's 10-block gate now clears.
    SendLoadBackDone(lb->op_ids.at(0));
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
    ExecutionPlan admitted = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(admitted);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->request_ids.size(), 1u);
    EXPECT_EQ(op->request_ids.at(0), "r3");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// A LoadBackDone whose op_id was already retired must hit the silent-ignore arm:
// no crash, no double UnpinLoad, no double-free of the destination pages.
TEST_F(FlatHostHitSuite, DuplicateLoadBackDoneIsIgnored) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    SeedHostThenEvictDevice();

    Submit(MakeRequestSpec("r2", /*num_pages=*/4));
    ExecutionPlan plan = PlanOnce();
    auto lb = FindFlatLoadBack(plan);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 7);

    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);
    const std::int32_t free_after_first = scheduler_->FlatPoolFreeBlocks();
    EXPECT_EQ(free_after_first, free_at_start - 7) << "destinations still table-held: no free-list change";

    SendLoadBackDone(lb->op_ids.at(0));  // duplicate
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_after_first) << "a duplicate Done must not double-free";
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    PlanOnce();
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool balances despite the duplicate event";
}

// ---------------------------------------------------------------------------
// M15 host hit + chunked prefill: later chunks must count the host extension as
// computed tokens, and an SWA slide that punches a still-loading destination
// page must leave it ticket-protected until LoadBackDone.
// ---------------------------------------------------------------------------
class FlatChunkedHostHitSuite : public FlatHostHitSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatHostHitSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;  // 4-token prefill chunks
        // 21 -> 20 free: r2's first chunk holds 10 (6 ext + 4 fresh) and its second
        // chunk charges 6 with zero slide credit (ticket-held punches don't count).
        cfg.device_allocator.total_pages = 21;
        for (auto& g : cfg.paged_cache_groups) {
            g.total_pages = cfg.device_allocator.total_pages;
        }
        return cfg;
    }

    void AckWriteBacks(const ExecutionPlan& plan) {
        for (const CacheOperation& op : ExtractCacheOpsOfKind<FlatWriteBackOperation>(plan)) {
            for (cache_op_id id : std::get<FlatWriteBackOperation>(op).op_ids) {
                SendWriteBackDone(id, /*success=*/true);
            }
        }
    }

    // Chunked twin of RunSinkLifecycle: drives prefill round by round, acking every
    // streaming write-back so no sink pin outlives the seeding.
    void RunChunkedSinkLifecycle(const RequestSpec& spec, std::int32_t prefill_rounds) {
        Submit(spec);
        for (std::int32_t i = 0; i < prefill_rounds; ++i) {
            ExecutionPlan plan = PlanOnce();
            ASSERT_NE(FindFlatOp(plan), nullptr) << "chunk " << i;
            ASSERT_EQ(FindFlatOp(plan)->request_ids.size(), 1u) << "chunk " << i << " must be admitted";
            AckWriteBacks(plan);
        }
        SendForwardDone(spec.request_id, {9001});
        AckWriteBacks(PlanOnce());  // finalize: registration + drain
        FinishAndReap(spec.request_id);
    }

    // r1 (4 pages) indexes 8 host entries over 2 chunks; the churn request must pop
    // 22 free-list entries (full 10 + swa 10 + reserve 2) = the 12 fresh blocks
    // still unused plus ALL 10 of r1's cached blocks, so r1 survives host-only.
    void SeedHostThenEvictDeviceChunked() {
        RunChunkedSinkLifecycle(MakeRequestSpec("r1", /*num_pages=*/4), /*prefill_rounds=*/2);
        ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 8);
        RunChunkedSinkLifecycle(MakeRequestSpec("churn", /*num_pages=*/10, /*start=*/501), /*prefill_rounds=*/5);
        ASSERT_EQ(scheduler_->FlatHostPoolCachedBlocks(), 28);
        ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), 20) << "both seeding requests fully retired";
    }
};

TEST_F(FlatChunkedHostHitSuite, ChunkedPrefillAfterHostHit) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();
    ASSERT_EQ(free_at_start, 20);
    SeedHostThenEvictDeviceChunked();

    // r2: 16 tokens sharing r1's first 8. Host extension = 4 blocks (full run 0..3;
    // swa tail ceil((W-1)/P)=2 at the boundary) -> real pages full 4 + swa 2 = 6.
    Submit(MakeRequestSpec("r2", /*num_pages=*/8));
    ExecutionPlan c1 = PlanOnce();
    auto lb = FindFlatLoadBack(c1);
    ASSERT_TRUE(lb.has_value());
    ASSERT_EQ(lb->src_pages.at(0).size(), 6u);
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 6);

    const FlatForwardOperation* op1 = FindFlatOp(c1);
    ASSERT_NE(op1, nullptr);
    ASSERT_EQ(op1->request_ids.size(), 1u);
    // First chunk: the 8 host-hit tokens are computed; chunk covers tokens [8,12).
    EXPECT_EQ(op1->extend_prefix_lens.at(0), 8);
    EXPECT_EQ(op1->input_lengths.at(0), 4);
    EXPECT_EQ(op1->prefill_lengths.at(0), 16);
    // 6 ext + 2 fresh/group: 10 blocks held.
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 10);

    // Chunk 2 completes prefill; its slide at num_computed=12 punches swa ext slots
    // 2,3 = LOADED destinations mid-copy. The ticket must keep them off the free list.
    ExecutionPlan c2 = PlanOnce();
    const FlatForwardOperation* op2 = FindFlatOp(c2);
    ASSERT_NE(op2, nullptr);
    ASSERT_EQ(op2->request_ids.size(), 1u);
    EXPECT_EQ(op2->extend_prefix_lens.at(0), 12) << "chunk 2 must see ext(8) + chunk1(4) as computed";
    EXPECT_EQ(op2->input_lengths.at(0), 4);
    AckWriteBacks(c2);  // pages 4,5 registered this round; ack so only the ticket pins remain
    // Punched destinations withheld: chunk 2 acquired 4, punched 2 stay ticket-held.
    ASSERT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 14);
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 6) << "the copy is still in flight";

    // LoadBackDone releases exactly the 2 punched destinations (the other 4 stay table-held).
    SendLoadBackDone(lb->op_ids.at(0));
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start - 12);
    EXPECT_EQ(scheduler_->FlatHostPoolPinnedBlocks(), 0);

    SendForwardDone("r2", {9001});
    ExecutionPlan finalize = PlanOnce();
    ASSERT_NE(FindFlatOp(finalize), nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    AckWriteBacks(finalize);
    SendForwardDone("r2", {9002});
    SendFinish("r2");
    PlanOnce();  // reap
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start) << "pool balances after the chunked host hit";
}

// ---------------------------------------------------------------------------
// M15-C3 review: until the emission takes the HostMatch pins, the first-chunk
// EVENT owns them (BlockRef RAII) -- destroying it unconsumed (throwing
// transition) must release them exactly once, and moving must not double-release.
// ---------------------------------------------------------------------------
TEST(FlatFirstChunkEventPinOwnership, DtorReleasesUnconsumedPinsExactlyOnce) {
    BlockPool host_pool(/*total_num_blocks=*/2);
    CacheBlock* page = host_pool.AllocateBlocks(1).front();
    host_pool.CacheFullBlock(page, "k");
    host_pool.FreeBlocks({page});
    {
        HostMatch host;
        host.num_extension_blocks = 1;
        host.pinned.push_back(BlockRef::Share(host_pool, host_pool.GetCachedBlock("k")));
        ASSERT_EQ(host_pool.NumPinnedCachedBlocks(), 1);
        fsm::SchedulePrefillFirstChunkEvent ev{/*tokens_this_round=*/0,
                                               /*decode_input_tokens=*/0,
                                               /*device_allocator=*/nullptr,
                                               /*req_pool_allocator=*/nullptr,
                                               MatchResult{},
                                               Role::kFused,
                                               /*kv_prefix_cache=*/nullptr,
                                               /*disable_l2_cache=*/false,
                                               /*loadback_diff=*/{},
                                               /*hybrid_prefix_cache=*/nullptr,
                                               /*mamba_allocator=*/nullptr,
                                               /*mamba_loadback_nodes=*/{},
                                               /*coordinator=*/nullptr,
                                               CoordinatorMatch{},
                                               std::move(host),
                                               /*flat_ext_hashes=*/{}};
        // Vector move must leave the source's pins empty: a double-release would
        // underflow the ref count and throw from the second dtor.
        fsm::SchedulePrefillFirstChunkEvent moved{std::move(ev)};
        EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 1) << "construction and move must not unpin";
    }
    EXPECT_EQ(host_pool.NumPinnedCachedBlocks(), 0) << "the unconsumed event must release its pins";
}

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
