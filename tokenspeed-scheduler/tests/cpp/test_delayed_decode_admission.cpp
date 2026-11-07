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

class IndependentAdmissionTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kP;
        cfg.decode_allocation_policy = SchedulerConfig::DecodeAllocationPolicy::kIndependent;
        cfg.disable_l2_cache = true;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 4;
        cfg.decode_input_tokens = 4;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "state",
            .block_granularity = 0,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kState,
            .transfer_policy = CacheTransferPolicy::LatestSnapshot,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SubmitReady(const std::string& id, int length, int start = 1) {
        auto request = MakeRequestSpec(id, (length + 1) / 2, start);
        request.tokens.resize(length);
        request.max_new_tokens = 64;
        Submit(request);
    }

    // Every Decode rank allocated the destination.
    void Grant(const std::string& id) { scheduler_->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id})); }

    void Ack(const std::string& id) { scheduler_->Advance(ExecutionEvent{}.With(pd::SucceededEvent{id})); }

    // The transfer failed or, cancelled, retired, and stopped: no reader or
    // writer touches the request's pages any more.
    void Fail(const std::string& id) {
        scheduler_->Advance(ExecutionEvent{}.With(pd::FailedEvent{id}).With(pd::TransferStoppedEvent{id}));
    }
};

TEST_F(IndependentAdmissionTest, FinalResultRetainsEveryGroupUntilGrant) {
    SubmitReady("r", 6);
    EXPECT_EQ(scheduler_->BootstrappingSize(), 0u);
    const auto first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first)->input_lengths, std::vector<std::int32_t>{4});
    SendForwardDone("r");
    const auto last = PlanOnce();
    ASSERT_EQ(FindForwardBatch(last)->input_lengths, std::vector<std::int32_t>{2});
    const auto source_blocks = FindForwardBatch(last)->BlockTables().at("full_attention").front();
    EXPECT_FALSE(last.remote_decode.has_value());
    PlanOnce();

    scheduler_->Advance(ExecutionEvent{}.With(
        forward::ExtendResult{.request_id = "r", .tokens = {42}, .spec_candidate_ids = {42, 43, 44}}));
    const auto ready = PlanOnce();
    EXPECT_FALSE(ready.remote_decode.has_value());
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 1u);
    EXPECT_EQ(scheduler_->AwaitingDestinationLcmBlocks(), scheduler_->ActiveLcmBlocks());
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    EXPECT_FALSE(scheduler_->ClearCache());
    const auto held = scheduler_->ActiveLcmBlocks();
    for (int i = 0; i < 3; ++i) {
        const auto wait = PlanOnce();
        EXPECT_FALSE(wait.remote_decode.has_value());
        EXPECT_EQ(scheduler_->ActiveLcmBlocks(), held);
    }

    Grant("r");
    Grant("r");
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"r"});
    EXPECT_EQ(send.remote_decode->decode_input_ids, std::vector<std::int32_t>{42});
    EXPECT_EQ(send.remote_decode->spec_candidate_ids.front(), (std::vector<std::int32_t>{42, 43, 44}));
    EXPECT_EQ(send.remote_decode->BlockTables().at("full_attention").front(), source_blocks);
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    for (const auto& [group, blocks] : send.pages_to_zero) EXPECT_TRUE(blocks.empty());
    Ack("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_TRUE(scheduler_->ClearCache());
}

// A client abort cancels the transfer, whose retirement fails the request.
TEST_F(IndependentAdmissionTest, CancellationIgnoresLateGrant) {
    SubmitReady("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    Fail("r");
    Grant("r");
    const auto cancelled = PlanOnce();
    EXPECT_FALSE(cancelled.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
}

// A forward abort on the P role waits for the transfer to retire.
TEST_F(IndependentAdmissionTest, CancellingWhileWaitingReleasesResourcesWithoutTransfer) {
    SubmitReady("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    PlanOnce();
    const auto held = scheduler_->ActiveLcmBlocks();
    SendAbortEvent("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), held);
    Fail("r");
    const auto cancelled = PlanOnce();
    EXPECT_FALSE(cancelled.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    Grant("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
}

// The cancellation retires while the final forward is still out: the pages
// stay until its result lands, and a grant in between sends nothing.
TEST_F(IndependentAdmissionTest, CancellingDuringTheFinalForwardReleasesOnceItsResultLands) {
    SubmitReady("r", 4);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"r"});
    Fail("r");
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    Grant("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    SendForwardDone("r", {42});
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
}

TEST_F(IndependentAdmissionTest, EarlyGrantIsRememberedWithoutSendingUncomputedKv) {
    SubmitReady("r", 6);
    Grant("r");
    Grant("r");
    const auto first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first)->input_lengths, std::vector<std::int32_t>{4});
    EXPECT_FALSE(first.remote_decode.has_value());
    SendForwardDone("r");
    const auto last = PlanOnce();
    ASSERT_EQ(FindForwardBatch(last)->input_lengths, std::vector<std::int32_t>{2});
    EXPECT_FALSE(last.remote_decode.has_value());
    SendForwardDone("r", {42});
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"r"});
    Ack("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

class IndependentAdmissionFullPoolTest : public IndependentAdmissionTest, public ::testing::WithParamInterface<bool> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = IndependentAdmissionTest::MakeConfig();
        cfg.cache_groups.resize(GetParam() ? 2 : 1);
        cfg.device_allocator.total_pages = GetParam() ? 11 : 7;
        for (auto& group : cfg.cache_groups) group.total_pages = cfg.device_allocator.total_pages;
        cfg.disable_prefix_cache = true;
        cfg.max_scheduled_tokens = 8;
        return cfg;
    }
};

TEST_P(IndependentAdmissionFullPoolTest, GrantDrainsCompletedPromptAndUnblocksPartialPromptInAFullPool) {
    SubmitReady("first", 4);
    SubmitReady("partial", 6, 100);
    const auto compute = PlanOnce();
    ASSERT_EQ(FindForwardBatch(compute)->request_ids, (std::vector<std::string>{"first", "partial"}));
    EXPECT_EQ(FindForwardBatch(compute)->input_lengths, (std::vector<std::int32_t>{4, 4}));
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 0);
    SendForwardDone("first", {42});
    SendForwardDone("partial");
    const auto ready = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(ready)->request_ids.empty());
    Grant("first");
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"first"});
    EXPECT_TRUE(send.pages_to_zero.at("full_attention").empty());
    Ack("first");
    const auto resume = PlanOnce();
    ASSERT_EQ(FindForwardBatch(resume)->request_ids, std::vector<std::string>{"partial"});
    EXPECT_EQ(FindForwardBatch(resume)->input_lengths, std::vector<std::int32_t>{2});
    SendForwardDone("partial", {43});
    PlanOnce();
    Grant("partial");
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("partial");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

INSTANTIATE_TEST_SUITE_P(HistoryAndState, IndependentAdmissionFullPoolTest, ::testing::Bool());

TEST_F(IndependentAdmissionTest, GrantDuringFinalForwardSendsAsSoonAsItsResultLands) {
    SubmitReady("r", 4);
    PlanOnce();
    Grant("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    SendForwardDone("r", {42});
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->decode_input_ids, std::vector<std::int32_t>{42});
    Ack("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(IndependentAdmissionTest, DecodeFailureWhileAwaitingGrantReleasesAllSourceGroups) {
    SubmitReady("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    PlanOnce();
    Fail("r");
    const auto failed = PlanOnce();
    EXPECT_FALSE(failed.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(IndependentAdmissionTest, EarlyPolicySendsWithoutASeparateDestinationGrant) {
    auto cfg = MakeConfig();
    cfg.decode_allocation_policy = SchedulerConfig::DecodeAllocationPolicy::kEarly;
    Scheduler early{WithHostPackingFromDevice(cfg)};
    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
    early.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2, 3, 4}, .max_new_tokens = 8}});
    early.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"r"}));
    early.NextExecutionPlan(options);
    early.Advance(ExecutionEvent{}.With(forward::ExtendResult{.request_id = "r", .tokens = {42}}));
    const auto send = early.NextExecutionPlan(options);
    EXPECT_TRUE(send.remote_decode.has_value());
}

TEST_F(IndependentAdmissionTest, DecodeAllocatesWithoutWaitingForPrefillCompletion) {
    auto cfg = MakeConfig();
    cfg.role = Role::kD;
    cfg.decode_allocation_policy.reset();
    SetTestSnapshotPool(cfg);
    Scheduler decode{WithHostPackingFromDevice(cfg)};
    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
    decode.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2, 3, 4}, .max_new_tokens = 8}});
    for (int i = 0; i < 3; ++i) {
        const auto plan = decode.NextExecutionPlan(options);
        EXPECT_FALSE(plan.remote_prefill.has_value());
        EXPECT_EQ(decode.ActiveLcmBlocks(), 0);
        EXPECT_EQ(decode.BootstrappingSize(), 1u);
    }
    decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"r"}));
    const auto receive = decode.NextExecutionPlan(options);
    ASSERT_TRUE(receive.remote_prefill.has_value());
    EXPECT_GT(decode.ActiveLcmBlocks(), 0);
    EXPECT_EQ(decode.RemotePrefillSize(), 1u);
}

TEST_F(IndependentAdmissionTest, DecodeAdmissionPressureDoesNotBlockPrefillComputation) {
    auto cfg = MakeConfig();
    cfg.role = Role::kD;
    cfg.decode_allocation_policy.reset();
    cfg.max_batch_size = 1;
    SetTestSnapshotPool(cfg);
    Scheduler decode{WithHostPackingFromDevice(cfg)};
    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
    decode.SubmitRequests({RequestSpec{.request_id = "first", .tokens = {1, 2, 3, 4}, .max_new_tokens = 8},
                           RequestSpec{.request_id = "later", .tokens = {101, 102, 103, 104}, .max_new_tokens = 8}});
    decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"first"}).With(pd::BootstrappedEvent{"later"}));
    const auto first = decode.NextExecutionPlan(options);
    ASSERT_TRUE(first.remote_prefill.has_value());
    EXPECT_EQ(first.remote_prefill->request_ids, std::vector<std::string>{"first"});
    EXPECT_FALSE(decode.NextExecutionPlan(options).remote_prefill.has_value());
    EXPECT_EQ(decode.WaitingSize(), 1u);

    SubmitReady("later", 4, 100);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->input_lengths, std::vector<std::int32_t>{4});
    SendForwardDone("later", {43});
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 1u);
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    EXPECT_FALSE(decode.NextExecutionPlan(options).remote_prefill.has_value());

    decode.Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{"first", 42}));
    decode.Advance(ExecutionEvent{}.With(forward::Finish{.request_id = "first"}));
    const auto later = decode.NextExecutionPlan(options);
    ASSERT_TRUE(later.remote_prefill.has_value());
    EXPECT_EQ(later.remote_prefill->request_ids, std::vector<std::string>{"later"});
    Grant("later");
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("later");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(IndependentAdmissionTest, AllocationPolicyIsExplicitOnlyForPrefill) {
    auto prefill = MakeConfig();
    prefill.decode_allocation_policy.reset();
    EXPECT_THROW(Scheduler{WithHostPackingFromDevice(prefill)}, std::invalid_argument);
    for (const auto role : {Role::kD, Role::kFused}) {
        auto cfg = MakeConfig();
        cfg.role = role;
        SetTestSnapshotPool(cfg);
        EXPECT_THROW(Scheduler{WithHostPackingFromDevice(cfg)}, std::invalid_argument);
        cfg.decode_allocation_policy.reset();
        EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(cfg)});
    }
}

}  // namespace tokenspeed::test
