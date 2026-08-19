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

// End-to-end lifecycle tests for the two-level KV-cache FSM path.
// Config: two cache groups (full + sliding-window), no L2/L3.

#include <algorithm>
#include <array>
#include <optional>
#include <set>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class KvCacheLifecycleTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
        cfg.device_allocator.total_pages = 32;
        cfg.host_allocator.total_pages = 32;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        CacheGroupConfig full_grp;
        full_grp.group_id = "full";
        full_grp.rows_per_page = cfg.prefix_granularity;
        full_grp.entry_stride_tokens = 1;
        full_grp.total_pages = cfg.device_allocator.total_pages;
        full_grp.cache_blocks_per_lcm_block = 2;
        full_grp.retention = CacheGroupConfig::Retention::FullHistory;
        full_grp.family = CacheGroupFamily::History;

        CacheGroupConfig swa_grp;
        swa_grp.group_id = "swa";
        swa_grp.rows_per_page = cfg.prefix_granularity;
        swa_grp.entry_stride_tokens = 1;
        swa_grp.total_pages = cfg.device_allocator.total_pages;
        swa_grp.retention = CacheGroupConfig::Retention::SlidingWindow;
        swa_grp.sliding_window_tokens = 4;
        swa_grp.family = CacheGroupFamily::State;

        cfg.cache_groups = {full_grp, swa_grp};
        return cfg;
    }
};

TEST_F(KvCacheLifecycleTestSuite, Construct_AndSubmit_Waiting) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

TEST_F(KvCacheLifecycleTestSuite, SingleRequest_PrefillDecodeFinish) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    const ForwardBatch* prefill = FindForwardBatch(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->block_tables.count("swa"), 1u);
    EXPECT_EQ(prefill->block_tables.at("full").size(), 1u);
    EXPECT_EQ(prefill->block_tables.at("swa").size(), 1u);
    const auto& full_prefill_row = prefill->block_tables.at("full").at(0);
    ASSERT_GE(full_prefill_row.size(), 2u);
    EXPECT_NE(full_prefill_row[0], full_prefill_row[1])
        << "two child slots in one LCM block need distinct kernel page ids";
    ASSERT_EQ(prefill_plan.pages_to_zero.count("full"), 1u);
    ASSERT_EQ(prefill_plan.pages_to_zero.count("swa"), 1u);
    EXPECT_EQ(prefill_plan.pages_to_zero.at("full"), full_prefill_row);
    EXPECT_EQ(prefill_plan.pages_to_zero.at("swa"), prefill->block_tables.at("swa").at(0));

    SendForwardDone("r1", {42});
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    // Swa null hole first appears at decode step 1 (window=4 tokens = 2 pages).
    // last_plan must outlive the loop: the ForwardBatch is owned by its plan.
    std::optional<ExecutionPlan> last_plan;
    int tok = 43;
    for (int step = 0; step < 4; ++step) {
        last_plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(*last_plan), nullptr);
        EXPECT_EQ(scheduler_->DecodingSize(), 1u);
        SendForwardDone("r1", {tok++});
    }
    const ForwardBatch* last_decode = FindForwardBatch(*last_plan);
    ASSERT_NE(last_decode, nullptr);

    const auto& full_row = last_decode->block_tables.at("full").at(0);
    for (std::int32_t id : full_row) {
        EXPECT_GT(id, 0) << "full row should keep history with no null/padding hole";
    }
    const auto& swa_row = last_decode->block_tables.at("swa").at(0);
    EXPECT_NE(std::find(swa_row.begin(), swa_row.end(), 0), swa_row.end())
        << "swa row should contain a null hole after the sliding window slides";

    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// AvailableKvPages() reports the shared BlockPool.
TEST_F(KvCacheLifecycleTestSuite, AvailableKvPagesReportsSharedPool) {
    const std::size_t idle = scheduler_->AvailableKvPages();
    EXPECT_EQ(idle, static_cast<std::size_t>(scheduler_->PoolFreeBlocks()));
    // 32 total pages, block 0 is the never-allocated null placeholder.
    EXPECT_EQ(idle, 31u);

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), static_cast<std::size_t>(scheduler_->PoolFreeBlocks()));
    EXPECT_LT(scheduler_->AvailableKvPages(), idle)
        << "prefill draws from the shared pool and the bound accessor must see it";

    SendForwardDone("r1", {42});
    SendFinish("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), idle);
}

TEST_F(KvCacheLifecycleTestSuite, TwoRequestsBatchBlockTables) {
    const std::int32_t free_at_start = scheduler_->PoolFreeBlocks();

    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    Submit(MakeRequestSpec("r2", /*num_pages=*/3, /*start=*/101));
    ExecutionPlan prefill_plan = PlanOnce();
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const ForwardBatch* prefill = FindForwardBatch(prefill_plan);
    ASSERT_NE(prefill, nullptr);
    ASSERT_EQ(prefill->request_ids.size(), 2u);

    ASSERT_EQ(prefill->block_tables.count("full"), 1u);
    ASSERT_EQ(prefill->block_tables.count("swa"), 1u);
    const auto& full = prefill->block_tables.at("full");
    const auto& swa = prefill->block_tables.at("swa");
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
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), free_at_start);
}

// KimiFourGroupSuite (integration_test_helper.h) is shared with the
// four-group scenario tests in test_cache_kvcache_scenarios.cpp.

TEST_F(KimiFourGroupSuite, FourTablesUseDisjointGlobalPages) {
    const std::size_t before = scheduler_->AvailableKvPages();
    ASSERT_EQ(before, 32u);
    Submit(MakeRequestSpec("r1", /*num_pages=*/2));
    ExecutionPlan plan = PlanOnce();
    const ForwardBatch* op = FindForwardBatch(plan);
    ASSERT_NE(op, nullptr);
    ASSERT_EQ(op->block_tables.size(), GroupIds().size());

    std::set<std::int32_t> all_real;
    std::size_t fresh_positive_entries = 0;
    for (const auto& group_id : GroupIds()) {
        ASSERT_EQ(op->block_tables.count(group_id), 1u) << group_id;
        ASSERT_EQ(op->block_tables.at(group_id).size(), 1u) << group_id;
        std::set<std::int32_t> group_real;
        const std::size_t group_entries = CollectDisjointRealPages(op->block_tables.at(group_id), group_real, group_id);
        EXPECT_GT(group_entries, 0u) << group_id;
        fresh_positive_entries += group_entries;
        for (std::int32_t page_id : group_real) {
            EXPECT_TRUE(all_real.insert(page_id).second) << "physical page " << page_id << " reused across groups";
        }
        ASSERT_EQ(plan.pages_to_zero.count(group_id), 1u) << group_id;
        EXPECT_EQ(
            std::set<std::int32_t>(plan.pages_to_zero.at(group_id).begin(), plan.pages_to_zero.at(group_id).end()),
            group_real)
            << "every freshly-owned physical page must be sanitized in its cache group";
    }
    EXPECT_EQ(all_real.size(), fresh_positive_entries);

    AbortRequest("r1");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
}

TEST_F(KimiFourGroupSuite, FinishAndAbortRestoreAllUsablePages) {
    const std::size_t before = scheduler_->AvailableKvPages();
    ASSERT_EQ(before, 32u);

    Submit(MakeRequestSpec("finished", /*num_pages=*/2));
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    SendForwardDone("finished", {42});
    SendFinish("finished");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);

    Submit(MakeRequestSpec("aborted", /*num_pages=*/2, /*start=*/101));
    ASSERT_NE(FindForwardBatch(PlanOnce()), nullptr);
    AbortRequest("aborted");
    PlanOnce();
    EXPECT_EQ(scheduler_->AvailableKvPages(), before);
}

}  // namespace tokenspeed::test
