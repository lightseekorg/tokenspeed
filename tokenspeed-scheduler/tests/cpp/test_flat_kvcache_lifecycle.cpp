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

// End-to-end lifecycle test for the flat KV-cache FSM path
// (TOKENSPEED_FLAT_KVCACHE=ON). The whole file compiles to nothing when the
// flag is off. Models a minimal GPT-OSS-shaped config: two paged-cache groups
// (one full-attention, one sliding-window), no L2/L3 cache. Drives
// Submit -> prefill -> decode -> finish and asserts the flat per-group block
// tables (full-history keeps all pages; sliding-window develops a null hole)
// plus pool reclamation on finish.

#if TOKENSPEED_FLAT_KVCACHE

#include <algorithm>
#include <optional>

#include "integration_test_helper.h"

namespace tokenspeed::test {

// Minimal GPT-OSS-shaped flat config: full + sliding-window groups, no L2/L3.
class FlatKvCacheLifecycleTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.page_size = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        PagedCacheGroupConfig full_grp;
        full_grp.group_id = "full";
        full_grp.rows_per_page = cfg.page_size;
        full_grp.entry_stride_tokens = 1;
        full_grp.total_pages = cfg.device_allocator.total_pages;
        full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
        full_grp.family = PagedCacheGroupFamily::History;

        PagedCacheGroupConfig swa_grp;
        swa_grp.group_id = "swa";
        swa_grp.rows_per_page = cfg.page_size;
        swa_grp.entry_stride_tokens = 1;
        swa_grp.total_pages = cfg.device_allocator.total_pages;
        swa_grp.retention = PagedCacheGroupConfig::Retention::SlidingWindow;
        swa_grp.sliding_window_tokens = 4;
        swa_grp.family = PagedCacheGroupFamily::State;

        cfg.paged_cache_groups = {full_grp, swa_grp};
        return cfg;
    }

    static const FlatForwardOperation* FindFlatOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (const auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

// Smoke: with a GPT-OSS-shaped flat config, the Scheduler constructs (building
// a two-group KvCacheCoordinator from paged_cache_groups) and accepts a request
// into the waiting queue. This is the part of the flat path that works today:
// construction + Submit touch no per-request coordinator state.
TEST_F(FlatKvCacheLifecycleTestSuite, Construct_AndSubmit_Waiting) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

// Full lifecycle: Submit -> prefill -> decode -> finish, with the flat
// coordinator driving allocation. Each plan emits a single FlatForwardOperation
// whose flat_block_tables carry one row per request per group. Asserts the
// retention contract: the full-history group keeps every page (no null holes),
// while the sliding-window group evicts old pages (a null hole, id 0, appears
// once decoding crosses the window). Finishing returns all pages to the pool.
TEST_F(FlatKvCacheLifecycleTestSuite, SingleRequest_PrefillDecodeFinish) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->flat_block_tables.count("swa"), 1u);
    EXPECT_EQ(prefill->flat_block_tables.at("full").size(), 1u);
    EXPECT_EQ(prefill->flat_block_tables.at("swa").size(), 1u);

    SendForwardDone("r1", {42});
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    // Drive enough decode steps that the sliding-window row is guaranteed to
    // have evicted past its window. The swa row first shows a null hole (id 0)
    // at decode step 1 (window=4 tokens=2 pages); 4 steps is comfortably past.
    // Keep the last plan alive: the FlatForwardOperation is owned by its plan.
    std::optional<ExecutionPlan> last_plan;
    int tok = 43;
    for (int step = 0; step < 4; ++step) {
        last_plan = PlanOnce();
        ASSERT_NE(FindFlatOp(*last_plan), nullptr);
        EXPECT_EQ(scheduler_->DecodingSize(), 1u);
        SendForwardDone("r1", {tok++});
    }
    const FlatForwardOperation* last_decode = FindFlatOp(*last_plan);
    ASSERT_NE(last_decode, nullptr);

    // Full-attention group keeps all history: one request -> one row, no -1
    // padding and no null holes.
    const auto& full_row = last_decode->flat_block_tables.at("full").at(0);
    for (std::int32_t id : full_row) {
        EXPECT_GT(id, 0) << "full row should keep history with no null/padding hole";
    }
    // Sliding-window group evicts old pages: the row carries at least one null
    // hole (id 0) once decoding crosses the window.
    const auto& swa_row = last_decode->flat_block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa_row.begin(), swa_row.end(), 0), swa_row.end())
        << "swa row should contain a null hole after the sliding window slides";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// Two requests in one batch: the FlatForwardOperation must aggregate both into
// one SoA op. Exercises the multi-request path the single-request test cannot:
// two rows per group, rectangular (-1 padded) tables, and physical pages that
// never collide across the two requests (they draw from the same shared pool).
TEST_F(FlatKvCacheLifecycleTestSuite, TwoRequests_BatchedFlatBlockTables) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    // Different prompt lengths so the full-history rows differ in page count and
    // the batch genuinely needs -1 padding to become rectangular.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids.size(), 2u);

    // Both groups present, one row per request.
    ASSERT_EQ(prefill->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->flat_block_tables.count("swa"), 1u);
    const auto& full = prefill->flat_block_tables.at("full");
    const auto& swa = prefill->flat_block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    // Tables are rectangular: every row in a group shares one column count.
    EXPECT_EQ(full.at(0).size(), full.at(1).size());
    EXPECT_EQ(swa.at(0).size(), swa.at(1).size());
    // The shorter prompt's full row is -1 padded (real pages are > 0, holes are
    // 0, padding is -1) -- so the two requests' page counts genuinely differ.
    const bool any_pad = std::any_of(full.at(0).begin(), full.at(0).end(),
                                     [](std::int32_t id) { return id == -1; }) ||
                         std::any_of(full.at(1).begin(), full.at(1).end(),
                                     [](std::int32_t id) { return id == -1; });
    EXPECT_TRUE(any_pad) << "unequal prompt lengths should force -1 padding in one full row";

    // Physical pages must not be shared between the two requests in either group:
    // collect every real page id (id > 0) across both rows of a group and assert
    // no duplicates.
    auto assert_no_page_collision = [](const std::vector<std::vector<std::int32_t>>& group) {
        std::vector<std::int32_t> real;
        for (const auto& row : group) {
            for (std::int32_t id : row) {
                if (id > 0) real.push_back(id);
            }
        }
        std::vector<std::int32_t> sorted = real;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
            << "two requests must not be handed the same physical page";
    };
    assert_no_page_collision(full);
    assert_no_page_collision(swa);

    // Finishing both requests returns every page to the pool.
    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
