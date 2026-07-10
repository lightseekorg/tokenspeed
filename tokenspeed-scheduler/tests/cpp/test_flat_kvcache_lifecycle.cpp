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

// End-to-end lifecycle tests for the flat KV-cache FSM path
// (TOKENSPEED_FLAT_KVCACHE=ON; the whole file compiles to nothing otherwise).
// Config: two paged-cache groups (full + sliding-window), no L2/L3.

#if TOKENSPEED_FLAT_KVCACHE

#include <algorithm>
#include <optional>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class FlatKvCacheLifecycleTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        PagedCacheGroupConfig full_grp;
        full_grp.group_id = "full";
        full_grp.rows_per_page = cfg.block_size;
        full_grp.entry_stride_tokens = 1;
        full_grp.total_pages = cfg.device_allocator.total_pages;
        full_grp.retention = PagedCacheGroupConfig::Retention::FullHistory;
        full_grp.family = PagedCacheGroupFamily::History;

        PagedCacheGroupConfig swa_grp;
        swa_grp.group_id = "swa";
        swa_grp.rows_per_page = cfg.block_size;
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

TEST_F(FlatKvCacheLifecycleTestSuite, Construct_AndSubmit_Waiting) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

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

    // Swa null hole first appears at decode step 1 (window=4 tokens = 2 pages).
    // last_plan must outlive the loop: the FlatForwardOperation is owned by its plan.
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

    const auto& full_row = last_decode->flat_block_tables.at("full").at(0);
    for (std::int32_t id : full_row) {
        EXPECT_GT(id, 0) << "full row should keep history with no null/padding hole";
    }
    const auto& swa_row = last_decode->flat_block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa_row.begin(), swa_row.end(), 0), swa_row.end())
        << "swa row should contain a null hole after the sliding window slides";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// AvailableKvPages() must report the flat shared BlockPool, not the radix
// device_allocator_. TODO(radix-removal): collapses to the only accessor.
TEST_F(FlatKvCacheLifecycleTestSuite, AvailableKvPagesReportsFlatPool) {
    const std::size_t idle = scheduler_->AvailableKvPages();
    EXPECT_EQ(idle, static_cast<std::size_t>(scheduler_->FlatPoolFreeBlocks()));
    // 32 total pages, block 0 is the never-allocated null placeholder.
    EXPECT_EQ(idle, 31u);

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), static_cast<std::size_t>(scheduler_->FlatPoolFreeBlocks()));
    EXPECT_LT(scheduler_->AvailableKvPages(), idle)
        << "prefill draws from the flat pool and the bound accessor must see it";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), idle);
}

TEST_F(FlatKvCacheLifecycleTestSuite, TwoRequests_BatchedFlatBlockTables) {
    const std::int32_t free_at_start = scheduler_->FlatPoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const FlatForwardOperation* prefill = FindFlatOp(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids.size(), 2u);

    ASSERT_EQ(prefill->flat_block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->flat_block_tables.count("swa"), 1u);
    const auto& full = prefill->flat_block_tables.at("full");
    const auto& swa = prefill->flat_block_tables.at("swa");
    ASSERT_EQ(full.size(), 2u);
    ASSERT_EQ(swa.size(), 2u);

    EXPECT_EQ(full.at(0).size(), full.at(1).size());
    EXPECT_EQ(swa.at(0).size(), swa.at(1).size());
    const bool any_pad = std::any_of(full.at(0).begin(), full.at(0).end(), [](std::int32_t id) { return id == -1; }) ||
                         std::any_of(full.at(1).begin(), full.at(1).end(), [](std::int32_t id) { return id == -1; });
    EXPECT_TRUE(any_pad) << "unequal prompt lengths should force -1 padding in one full row";

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
