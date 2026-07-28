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
#include <array>
#include <optional>
#include <set>

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
        full_grp.cache_blocks_per_lcm_block = 2;
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
    ASSERT_EQ(prefill->occupied_pages.size(), 1u);
    const auto& full_prefill_row = prefill->flat_block_tables.at("full").at(0);
    ASSERT_GE(full_prefill_row.size(), 2u);
    EXPECT_NE(full_prefill_row[0], full_prefill_row[1])
        << "two child slots in one LCM block need distinct kernel page ids";
    EXPECT_EQ(prefill->occupied_pages.at(0), full_prefill_row)
        << "the legacy single-table output must carry resolved kernel page ids, not LCM parent ids";
    ASSERT_EQ(prefill_plan.flat_pages_to_zero.count("full"), 1u);
    ASSERT_EQ(prefill_plan.flat_pages_to_zero.count("swa"), 1u);
    EXPECT_EQ(prefill_plan.flat_pages_to_zero.at("full"), full_prefill_row);
    EXPECT_EQ(prefill_plan.flat_pages_to_zero.at("swa"), prefill->flat_block_tables.at("swa").at(0));

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

    // Two requests must not be handed the same physical page within a group.
    std::set<std::int32_t> full_pages;
    CollectDisjointRealPages(full, full_pages, "full");
    std::set<std::int32_t> swa_pages;
    CollectDisjointRealPages(swa, swa_pages, "swa");

    SendForwardDone("r1", {42});
    SendForwardDone("r2", {142});
    SendFinish("r1");
    SendFinish("r2");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), free_at_start);
}

// FlatKimiFourGroupSuite (integration_test_helper.h) is shared with the
// four-group scenario tests in test_flat_kvcache_scenarios.cpp.

TEST_F(FlatKimiFourGroupSuite, FourTablesUseDisjointGlobalPages) {
    const std::size_t before = scheduler_->AvailableKvPages();
    ASSERT_EQ(before, 32u);
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->flat_block_tables.size(), GroupIds().size());

    std::set<std::int32_t> all_real;
    std::size_t fresh_positive_entries = 0;
    for (const auto& group_id : GroupIds()) {
        ASSERT_EQ(op->flat_block_tables.count(group_id), 1u) << group_id;
        ASSERT_EQ(op->flat_block_tables.at(group_id).size(), 1u) << group_id;
        std::set<std::int32_t> group_real;
        const std::size_t group_entries =
            CollectDisjointRealPages(op->flat_block_tables.at(group_id), group_real, group_id);
        EXPECT_GT(group_entries, 0u) << group_id;
        fresh_positive_entries += group_entries;
        for (std::int32_t page_id : group_real) {
            EXPECT_TRUE(all_real.insert(page_id).second) << "physical page " << page_id << " reused across groups";
        }
        ASSERT_EQ(plan.flat_pages_to_zero.count(group_id), 1u) << group_id;
        EXPECT_EQ(
            std::set<std::int32_t>(plan.flat_pages_to_zero.at(group_id).begin(),
                                   plan.flat_pages_to_zero.at(group_id).end()),
            group_real)
            << "every freshly-owned physical page must be sanitized in its cache group";
    }
    EXPECT_EQ(all_real.size(), fresh_positive_entries);

    AbortRequest("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
}

TEST_F(FlatKimiFourGroupSuite, FinishAndAbortRestoreAllUsablePages) {
    const std::size_t before = scheduler_->AvailableKvPages();
    ASSERT_EQ(before, 32u);

    Submit(MakeRequestSpec("finished", /*num_pages=*/2));
    ASSERT_NE(FindFlatOp(PlanOnce()), nullptr);
    SendForwardDone("finished", {42});
    SendFinish("finished");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);

    Submit(MakeRequestSpec("aborted", /*num_pages=*/2, /*start=*/101));
    ASSERT_NE(FindFlatOp(PlanOnce()), nullptr);
    AbortRequest("aborted");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
}

TEST_F(FlatKimiFourGroupSuite, OomDoesNotPartiallyCommitTables) {
    const std::size_t before = scheduler_->AvailableKvPages();
    ASSERT_EQ(before, 32u);
    Submit(MakeRequestSpec("oom", /*num_pages=*/8));

    ExecutionPlan deferred = PlanOnce();
    const FlatForwardOperation* deferred_op = FindFlatOp(deferred);
    ASSERT_NE(deferred_op, nullptr);
    EXPECT_TRUE(deferred_op->request_ids.empty());
    EXPECT_TRUE(deferred.flat_oom_request_ids.empty());
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);

    ExecutionPlan rejected = PlanOnce();
    const FlatForwardOperation* op = FindFlatOp(rejected);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->request_ids.empty());
    for (const auto& [group_id, table] : op->flat_block_tables) {
        EXPECT_TRUE(table.empty()) << group_id;
    }
    EXPECT_EQ(rejected.flat_oom_request_ids, std::vector<std::string>{"oom"});
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

}  // namespace tokenspeed::test

#endif  // TOKENSPEED_FLAT_KVCACHE
