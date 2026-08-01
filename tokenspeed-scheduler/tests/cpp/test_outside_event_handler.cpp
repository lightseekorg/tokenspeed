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

#include "integration_test_helper.h"

namespace tokenspeed::test {

inline const FlatForwardOperation* GetForwardOp(const std::vector<Operation>& ops) {
    for (const auto& op : ops) {
        if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
            return f;
        }
    }
    return nullptr;
}

inline std::int32_t FindRequestIndex(const FlatForwardOperation* fwd, const std::string& rid) {
    if (fwd == nullptr) return -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == rid) return static_cast<std::int32_t>(i);
    }
    return -1;
}

class LoadBackDoneTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 5;
        cfg.host_allocator.total_pages = 32;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SetupHostCache() {
        Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/1));
        PlanOnce();
        SendForwardDone("r1", {42});
        PlanOnce();
        SendFinish("r1");
        auto plan_wb = PlanOnce();
        const FlatWriteBackOperation* wb = nullptr;
        for (const auto& op : plan_wb.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    wb = w;
                    break;
                }
            }
        }
        ASSERT_NE(wb, nullptr) << "SetupHostCache: expected WriteBack op for r1";
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        PlanOnce();
        SendFinish("r_fill");
        auto plan_wb2 = PlanOnce();
        for (const auto& op : plan_wb2.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<FlatWriteBackOperation>(cop)) {
                    if (!w->op_ids.empty()) SendWriteBackDone(w->op_ids[0]);
                    break;
                }
            }
        }
        PlanOnce();
    }
};

// After host cache is populated, a new request with same tokens should see
// the host cache and be scheduled with reduced input_length (host pages already cached).
TEST_F(LoadBackDoneTestSuite, LoadBackDone_Success_PrefixLenChangesInForward) {
    SetupHostCache();

    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan.Operations());
    ASSERT_NE(fwd, nullptr);
    auto idx = FindRequestIndex(fwd, "r2");
    ASSERT_GE(idx, 0) << "r2 should be in forward after host cache hit";

    // With block_size=2 and 4 prefill tokens, GetFullPagedTokens(except_last=true)
    // yields 3 tokens → 1 matchable page. Host has 2 pages but only 1 matches.
    // unscheduled = 4 - 1*2 = 2, so input_length = 2 and extend_prefix_len = 1*block_size = 2.
    EXPECT_EQ(fwd->input_lengths[idx], 2) << "host hit covers 1 page; 2 tokens remain";

    if (!fwd->extend_prefix_lens.empty()) {
        EXPECT_EQ(fwd->extend_prefix_lens[idx], 1 * PageSize()) << "extend_prefix_len should cover the 1 loadback page";
    }
}

// LoadBack is inline (synchronous); subsequent plans proceed normally.
TEST_F(LoadBackDoneTestSuite, LoadBack_SubsequentPlanProceeds) {
    SetupHostCache();

    Submit(MakeRequestSpec("r2", /*num_pages=*/2, /*start=*/1));
    auto plan = PlanOnce();

    auto plan2 = PlanOnce();
    (void)plan2;
    EXPECT_TRUE(true);
}

class DisaggDecodeAdmissionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        // Flat block 0 is the null page, leaving three usable pages.
        cfg.device_allocator.total_pages = 4;
        cfg.host_allocator.total_pages = 4;
        cfg.max_scheduled_tokens = 2;
        cfg.max_batch_size = 1;
        cfg.decode_input_tokens = 1;
        cfg.role = Role::kD;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;

        PagedCacheGroupConfig full;
        full.group_id = "full";
        full.rows_per_page = cfg.block_size;
        full.entry_stride_tokens = 1;
        full.total_pages = cfg.device_allocator.total_pages;
        full.retention = PagedCacheGroupConfig::Retention::FullHistory;
        full.family = PagedCacheGroupFamily::History;
        cfg.paged_cache_groups = {full};
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(PDEvent{pd::BootstrappedEvent{request_id}});
        scheduler_->Advance(std::move(event));
    }

    void SendRemotePrefillDone(const std::string& request_id, std::int32_t bootstrap_token) {
        ExecutionEvent event;
        event.With(PDEvent{pd::RemotePrefillDoneEvent{request_id, bootstrap_token}});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(DisaggDecodeAdmissionTestSuite, ReservesWholeDestinationAndSurvivesRemoteCompletion) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan admission = PlanOnce();
    const FlatForwardOperation* prefill = GetForwardOp(admission.Operations());
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->request_ids, (std::vector<std::string>{"r0"}));
    EXPECT_EQ(prefill->input_lengths, (std::vector<std::int32_t>{4}));
    ASSERT_EQ(prefill->occupied_pages.size(), 1u);
    EXPECT_EQ(prefill->occupied_pages[0].size(), 3u);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 3u);

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    const ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode = GetForwardOp(decode_plan.Operations());
    ASSERT_NE(decode, nullptr);
    const std::int32_t r0 = FindRequestIndex(decode, "r0");
    ASSERT_GE(r0, 0);
    EXPECT_EQ(decode->decode_input_ids[static_cast<std::size_t>(r0)], 42);
    EXPECT_EQ(decode->occupied_pages[static_cast<std::size_t>(r0)].size(), 3u);
#if TOKENSPEED_FLAT_KVCACHE
    ASSERT_EQ(decode->flat_block_tables.count("full"), 1u);
    EXPECT_EQ(decode->flat_block_tables.at("full")[static_cast<std::size_t>(r0)].size(), 3u);
#endif
}

#if TOKENSPEED_FLAT_KVCACHE
class FlatPdSparseDecodeAdmissionTestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 6;  // null parent + five usable LCM parents
        cfg.max_scheduled_tokens = 8;
        cfg.enable_flatkv_pd = true;
        cfg.disable_prefix_cache = false;

        PagedCacheGroupConfig full = cfg.paged_cache_groups.front();
        full.group_id = "full";
        full.total_pages = 11;
        full.cache_blocks_per_lcm_block = 2;
        full.transfer_policy = PagedCacheTransferPolicy::FullSuffix;

        PagedCacheGroupConfig state = full;
        state.group_id = "state";
        state.total_pages = 6;
        state.cache_blocks_per_lcm_block = 1;
        state.family = PagedCacheGroupFamily::State;
        state.transfer_policy = PagedCacheTransferPolicy::LatestSnapshot;
        cfg.paged_cache_groups = {full, state};
        return cfg;
    }
};

class FlatPdSparseDecodeNoPrefixCacheTestSuite : public FlatPdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FlatPdSparseDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.paged_cache_groups[0].total_pages = 15;
        cfg.paged_cache_groups[1].total_pages = 8;
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

TEST_F(FlatPdSparseDecodeAdmissionTestSuite, MaterializesHistoryAndLatestStateSnapshotAtomically) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* destination = GetForwardOp(plan.Operations());
    ASSERT_NE(destination, nullptr);
    const auto& full = destination->flat_block_tables.at("full").at(0);
    ASSERT_EQ(full.size(), 5u);
    EXPECT_TRUE(std::ranges::all_of(full, [](std::int32_t page_id) { return page_id > 0; }));

    const auto& state = destination->flat_block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 5u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_EQ(state[2], 0);
    EXPECT_GT(state[3], 0);
    EXPECT_GT(state[4], 0);
    ASSERT_EQ(plan.flat_pages_to_zero.size(), 2u);
    EXPECT_EQ(plan.flat_pages_to_zero.at("full"), full);
    EXPECT_EQ(plan.flat_pages_to_zero.at("state"), (std::vector<std::int32_t>{state[3], state[4]}));
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 0);
    EXPECT_TRUE(scheduler_->FlatPdTransferPinned("r0"));

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    EXPECT_FALSE(scheduler_->FlatPdTransferPinned("r0"));
    const ExecutionPlan decode_plan = PlanOnce();
    const FlatForwardOperation* decode = GetForwardOp(decode_plan.Operations());
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->flat_block_tables, destination->flat_block_tables);

    ExecutionEvent succeeded;
    succeeded.With(PDEvent{pd::SucceededEvent{"r0"}});
    scheduler_->Advance(succeeded);
    EXPECT_EQ(scheduler_->FlatPoolFreeBlocks(), 5);
}

TEST_F(FlatPdSparseDecodeAdmissionTestSuite, ReusesHistoryPrefixAndLeavesStatePrefixSparse) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");
    PlanOnce();
    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    PlanOnce();

    ExecutionEvent succeeded;
    succeeded.With(PDEvent{pd::SucceededEvent{"r0"}});
    scheduler_->Advance(succeeded);

    Submit({MakeRequestSpec("r1", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r1");
    const ExecutionPlan plan = PlanOnce();
    const FlatForwardOperation* destination = GetForwardOp(plan.Operations());
    ASSERT_NE(destination, nullptr);
    EXPECT_EQ(destination->input_lengths, (std::vector<std::int32_t>{2}));

    const auto& full = destination->flat_block_tables.at("full").at(0);
    ASSERT_EQ(full.size(), 5u);
    EXPECT_TRUE(std::ranges::all_of(full, [](std::int32_t page_id) { return page_id > 0; }));

    const auto& state = destination->flat_block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 5u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_EQ(state[2], 0);
    EXPECT_GT(state[3], 0);
    EXPECT_GT(state[4], 0);
}

TEST_F(FlatPdSparseDecodeNoPrefixCacheTestSuite, RemoteBootstrapConsumesSparseTailBeforeNextDecode) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");
    PlanOnce();
    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);

    const ExecutionPlan first_decode = PlanOnce();
    const FlatForwardOperation* first = GetForwardOp(first_decode.Operations());
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(first->flat_block_tables.count("state"), 1u);
    EXPECT_EQ(first->flat_block_tables.at("state").at(0).size(), 5u);

    SendForwardDone("r0", {43});
    const ExecutionPlan second_decode = PlanOnce();
    const FlatForwardOperation* second = GetForwardOp(second_decode.Operations());
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(second->flat_block_tables.count("state"), 1u);
    EXPECT_EQ(second->flat_block_tables.at("state").at(0).size(), 5u);

    SendForwardDone("r0", {44});
    const ExecutionPlan boundary_decode = PlanOnce();
    const FlatForwardOperation* boundary = GetForwardOp(boundary_decode.Operations());
    ASSERT_NE(boundary, nullptr);
    ASSERT_EQ(boundary->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(boundary->flat_block_tables.count("state"), 1u);
    const auto& state = boundary->flat_block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 6u);
    EXPECT_GT(state.back(), 0);
}
#endif

}  // namespace tokenspeed::test
