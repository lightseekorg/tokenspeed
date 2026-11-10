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

// A computed prompt waiting for its destination holds its pages and request
// slot, as one awaiting its transfer does: both count as running.
TEST_F(IndependentAdmissionTest, AComputedPromptWaitingForItsDestinationIsRunning) {
    SubmitReady("r", 4);
    EXPECT_EQ(scheduler_->RunningSize(), 0u);
    PlanOnce();
    SendForwardDone("r", {42});
    ASSERT_EQ(scheduler_->AwaitingDestinationSize(), 1u);
    EXPECT_EQ(scheduler_->RunningSize(), 1u);
    Grant("r");
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    EXPECT_EQ(scheduler_->RunningSize(), 1u);
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("r");
    EXPECT_EQ(scheduler_->RunningSize(), 0u);
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

// The P/D circular wait: a decode Z leaves the Decode no room for a large
// prompt X, so it allocates the destinations of the small prompts Y1 and Y2
// queued behind X instead. Their KV must still reach it, whatever the policy:
// a computed X waiting for its destination must not keep the prefill pool
// full while Y1 and Y2 hold the Decode pages X needs. Real P and D
// schedulers; the harness carries what the runtime does: a D remote prefill
// (an allocation) bootstraps the request on P, a P remote decode lands the
// KV and first token on D and succeeds, every forward lands in its round,
// and a scheduler abort fails the request's transfer on both nodes.
class PrefillDecodeConvoyTest : public SchedulerTestSuite,
                                public ::testing::WithParamInterface<SchedulerConfig::DecodeAllocationPolicy> {
protected:
    // The P role: 11 usable pages, which X and its first token fill.
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kP;
        cfg.decode_allocation_policy = GetParam();
        cfg.disable_l2_cache = true;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_prefix_cache = true;
        cfg.device_allocator.total_pages = 12;
        cfg.cache_groups.front().total_pages = 12;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_P(PrefillDecodeConvoyTest, GrantedPromptsReachTheDecodeBehindAPromptItCannotFit) {
    auto decode_config = MakeConfig();
    decode_config.role = Role::kD;
    decode_config.decode_allocation_policy.reset();
    decode_config.device_allocator.total_pages = 15;
    decode_config.cache_groups.front().total_pages = 15;
    SetTestSnapshotPool(decode_config);
    Scheduler decode{WithHostPackingFromDevice(decode_config)};
    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
    // Z decodes on D with 9 of its 14 pages until round 5.
    auto z = MakeRequestSpec("Z", 4, 1000);
    z.max_new_tokens = 10;
    decode.SubmitRequests({z});
    decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"Z"}));
    ASSERT_TRUE(decode.NextExecutionPlan(options).remote_prefill.has_value());
    decode.Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{"Z", 7}));

    // With its first token X takes 11 pages on either node, Y1 and Y2 4 each.
    std::vector<RequestSpec> specs{MakeRequestSpec("X", 10, 1), MakeRequestSpec("Y1", 3, 300),
                                   MakeRequestSpec("Y2", 3, 600)};
    for (auto& spec : specs) spec.max_new_tokens = 2;
    decode.SubmitRequests(specs);
    Submit(specs);
    for (const auto& spec : specs) decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{spec.request_id}));

    std::set<std::string> decoded;
    std::set<std::string> aborted;
    int round = 0;
    // Until Z has finished too, so a fix that settles X, Y1 and Y2 sooner
    // still leaves the Decode empty.
    for (; round < 200 && (decoded.size() + aborted.size() < specs.size() || round <= 5); ++round) {
        if (round == 5) {
            decode.Advance(ExecutionEvent{}.With(forward::Finish{.request_id = "Z"}));
        }
        const auto prefill = PlanOnce();
        // Every prompt fits one chunk: each forward is final.
        for (const auto& id : FindForwardBatch(prefill)->request_ids) SendForwardDone(id, {42});
        if (prefill.remote_decode) {
            for (const auto& id : prefill.remote_decode->request_ids) {
                decode.Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{id, 42}));
                scheduler_->Advance(ExecutionEvent{}.With(pd::SucceededEvent{id}));
            }
        }
        const auto decoding = decode.NextExecutionPlan(options);
        if (decoding.remote_prefill) {
            for (const auto& id : decoding.remote_prefill->request_ids) {
                scheduler_->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id}));
            }
        }
        for (const auto& id : FindForwardBatch(decoding)->request_ids) {
            decode.Advance(ExecutionEvent{}.With(forward::ExtendResult{.request_id = id, .tokens = {8}}));
            if (id != "Z") {
                decode.Advance(ExecutionEvent{}.With(forward::Finish{.request_id = id}));
                decoded.insert(id);
            }
        }
        for (const auto* plan : {&prefill, &decoding}) {
            for (const auto& abort : plan->aborts) {
                aborted.insert(abort.request_id);
                scheduler_->Advance(ExecutionEvent{}
                                        .With(pd::FailedEvent{abort.request_id})
                                        .With(pd::TransferStoppedEvent{abort.request_id}));
                decode.Advance(ExecutionEvent{}
                                   .With(pd::FailedEvent{abort.request_id})
                                   .With(pd::TransferStoppedEvent{abort.request_id}));
            }
        }
    }
    EXPECT_TRUE(decoded.contains("Y1") && decoded.contains("Y2"))
        << "stuck after " << round << " rounds: P waits for " << scheduler_->AwaitingDestinationSize()
        << " destinations with " << scheduler_->EmptyLcmBlocks() << " empty pages, D holds "
        << decode.RemotePrefillSize() << " allocated destinations";
    EXPECT_EQ(decoded.size() + aborted.size(), specs.size());
    if (GetParam() == SchedulerConfig::DecodeAllocationPolicy::kEarly) {
        EXPECT_TRUE(aborted.empty());
    }
    PlanOnce();
    decode.NextExecutionPlan(options);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(decode.ActiveLcmBlocks(), 0);
}

INSTANTIATE_TEST_SUITE_P(Policies, PrefillDecodeConvoyTest,
                         ::testing::Values(SchedulerConfig::DecodeAllocationPolicy::kEarly,
                                           SchedulerConfig::DecodeAllocationPolicy::kIndependent),
                         [](const auto& info) {
                             return info.param == SchedulerConfig::DecodeAllocationPolicy::kEarly ? "Early"
                                                                                                  : "Independent";
                         });

// The same wait while small prompts keep flowing through both nodes: D
// allocates the destination of a prompt G but cannot fit W's beside it, while
// under independent admission P computed W first and cannot fit G beside it.
// A small prompt S fits beside W every round, so no round lacks prefill
// progress; G must still reach the Decode, and its KV free D for W, while the
// stream flows. 13 usable pages on P: W and two S.
class PrefillDecodeStreamTest : public PrefillDecodeConvoyTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillDecodeConvoyTest::MakeConfig();
        cfg.device_allocator.total_pages = 14;
        cfg.cache_groups.front().total_pages = 14;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_P(PrefillDecodeStreamTest, GrantedPromptsReachTheDecodeWhileSmallPromptsFlow) {
    auto decode_config = MakeConfig();
    decode_config.role = Role::kD;
    decode_config.decode_allocation_policy.reset();
    decode_config.device_allocator.total_pages = 15;  // 14 usable: G and W do not fit together
    decode_config.cache_groups.front().total_pages = 15;
    SetTestSnapshotPool(decode_config);
    Scheduler decode{WithHostPackingFromDevice(decode_config)};
    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};

    // With its first token W takes 9 pages on either node, G 6 and each S 2.
    auto w = MakeRequestSpec("W", 8, 1);
    auto g = MakeRequestSpec("G", 5, 300);
    w.max_new_tokens = 2;
    g.max_new_tokens = 2;
    std::set<std::string> decoded;
    std::set<std::string> aborted;
    int smalls = 0;
    const auto submit_small = [&] {
        auto s = MakeRequestSpec("S" + std::to_string(smalls), 1, 1000 + 10 * smalls);
        s.max_new_tokens = 2;
        ++smalls;
        decode.SubmitRequests({s});
        Submit(s);
        decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{s.request_id}));
    };
    const auto prefill_round = [&] {
        const auto prefill = PlanOnce();
        for (const auto& id : FindForwardBatch(prefill)->request_ids) SendForwardDone(id, {42});
        if (prefill.remote_decode) {
            for (const auto& id : prefill.remote_decode->request_ids) {
                decode.Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{id, 42}));
                scheduler_->Advance(ExecutionEvent{}.With(pd::SucceededEvent{id}));
            }
        }
        for (const auto& abort : prefill.aborts) aborted.insert(abort.request_id);
    };
    const auto decode_round = [&] {
        const auto decoding = decode.NextExecutionPlan(options);
        if (decoding.remote_prefill) {
            for (const auto& id : decoding.remote_prefill->request_ids) {
                scheduler_->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id}));
            }
        }
        for (const auto& id : FindForwardBatch(decoding)->request_ids) {
            decode.Advance(ExecutionEvent{}.With(forward::ExtendResult{.request_id = id, .tokens = {8}}));
            decode.Advance(ExecutionEvent{}.With(forward::Finish{.request_id = id}));
            decoded.insert(id);
        }
        for (const auto& abort : decoding.aborts) aborted.insert(abort.request_id);
    };

    // D learns of G first and allocates it, then S0: W does not fit beside G.
    Submit(std::vector<RequestSpec>{w, g});
    prefill_round();
    decode.SubmitRequests({g, w});
    decode.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"G"}).With(pd::BootstrappedEvent{"W"}));
    submit_small();
    decode_round();
    decode_round();
    int round = 0;
    for (; round < 60 && !(decoded.contains("G") && decoded.contains("W")); ++round) {
        submit_small();
        prefill_round();
        decode_round();
    }
    EXPECT_TRUE(decoded.contains("G") && decoded.contains("W"))
        << "stuck while the stream flows: P waits for " << scheduler_->AwaitingDestinationSize()
        << " destinations, D holds " << decode.RemotePrefillSize() << " allocated destinations";
    // The stream stops; both nodes drain.
    for (int drain = 0; drain < 20; ++drain) {
        prefill_round();
        decode_round();
    }
    EXPECT_TRUE(aborted.empty());
    EXPECT_EQ(decoded.size(), static_cast<std::size_t>(smalls) + 2);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_EQ(decode.ActiveLcmBlocks(), 0);
}

INSTANTIATE_TEST_SUITE_P(Policies, PrefillDecodeStreamTest,
                         ::testing::Values(SchedulerConfig::DecodeAllocationPolicy::kEarly,
                                           SchedulerConfig::DecodeAllocationPolicy::kIndependent),
                         [](const auto& info) {
                             return info.param == SchedulerConfig::DecodeAllocationPolicy::kEarly ? "Early"
                                                                                                  : "Independent";
                         });

// Granted work the pool or the request slots block takes them from idle
// prompts without a destination, which are requeued and computed again. One
// history group with prefix caching, 8 usable pages, 10-token chunks, a
// one-token decode slot and three request slots.
class IndependentAdmissionRequeueTest : public IndependentAdmissionTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = IndependentAdmissionTest::MakeConfig();
        cfg.cache_groups.resize(1);
        cfg.device_allocator.total_pages = 9;
        cfg.cache_groups.front().total_pages = 9;
        cfg.max_scheduled_tokens = 10;
        cfg.decode_input_tokens = 1;
        cfg.max_batch_size = 3;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

// A computed prompt waiting for its destination gives its pages to a granted
// prompt that does not fit beside it. It had no transfer: nothing was sent and
// nothing pins it once requeued, a failure would release nothing more, and a
// grant first computes it again, from the prefix its pages left in the cache,
// before its KV goes out with the token its new result reports.
TEST_F(IndependentAdmissionRequeueTest, GrantedPromptRequeuesAComputedPromptWaitingForItsDestination) {
    SubmitReady("waiting", 13);
    const auto first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first)->input_lengths, std::vector<std::int32_t>{10});
    SendForwardDone("waiting");
    const auto last = PlanOnce();
    ASSERT_EQ(FindForwardBatch(last)->input_lengths, std::vector<std::int32_t>{3});
    SendForwardDone("waiting", {42});
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 1u);
    EXPECT_EQ(scheduler_->RequestTokenSize("waiting"), 14);
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 1);

    SubmitReady("granted", 5, 100);
    Grant("granted");
    const auto requeue = PlanOnce();
    ASSERT_EQ(FindForwardBatch(requeue)->request_ids, std::vector<std::string>{"granted"});
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    EXPECT_FALSE(scheduler_->PdTransferPinned("waiting"));
    EXPECT_EQ(scheduler_->RequestTokenSize("waiting"), 13);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->Requeues(), 1u);
    EXPECT_EQ(scheduler_->RequeuedTokens(), 13u);
    for (const auto* plan : {&first, &last, &requeue}) EXPECT_FALSE(plan->remote_decode.has_value());
    SendForwardDone("granted", {7});
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"granted"});
    Ack("granted");

    // Granted work no longer blocks: the prompt computes again, resuming at
    // the prefix the granted prompt left cached.
    const auto again = PlanOnce();
    EXPECT_FALSE(again.remote_decode.has_value());
    ASSERT_EQ(FindForwardBatch(again)->request_ids, std::vector<std::string>{"waiting"});
    EXPECT_EQ(FindForwardBatch(again)->extend_prefix_lens, std::vector<std::int32_t>{10});
    Grant("waiting");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    SendForwardDone("waiting", {42});
    EXPECT_EQ(scheduler_->RequestTokenSize("waiting"), 14);
    const auto sent = PlanOnce();
    ASSERT_TRUE(sent.remote_decode.has_value());
    EXPECT_EQ(sent.remote_decode->decode_input_ids, std::vector<std::int32_t>{42});
    Ack("waiting");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

// A requeued prompt whose room fails before it computes again ends with
// nothing to release.
TEST_F(IndependentAdmissionRequeueTest, ARequeuedPromptsFailureReleasesNothingMore) {
    SubmitReady("waiting", 13);
    PlanOnce();
    SendForwardDone("waiting");
    PlanOnce();
    SendForwardDone("waiting", {42});
    SubmitReady("granted", 5, 100);
    Grant("granted");
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"granted"});
    const auto held = scheduler_->ActiveLcmBlocks();
    Fail("waiting");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), held);
    SendForwardDone("granted", {7});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    EXPECT_EQ(scheduler_->RequestTokenSize("waiting"), -1);
    Ack("granted");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

// The runtime sends a requeued prompt's first token, which its first
// computation sampled, with the KV of its next one: no flush -- a weight
// update -- comes between them, though the prompt holds no pages. Once the
// next computation's token landed, the transfer's ACK lets the flush through.
TEST_F(IndependentAdmissionRequeueTest, NoFlushBetweenARequeueAndTheNextComputation) {
    SubmitReady("waiting", 13);
    PlanOnce();
    SendForwardDone("waiting");
    PlanOnce();
    SendForwardDone("waiting", {42});
    SubmitReady("granted", 5, 100);
    Grant("granted");
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"granted"});
    SendForwardDone("granted", {7});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("granted");
    EXPECT_EQ(scheduler_->PdTransferSize(), 0u);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_FALSE(scheduler_->CanClearCache());
    EXPECT_FALSE(scheduler_->ClearL1Cache());
    EXPECT_FALSE(scheduler_->ClearCache());

    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"waiting"});
    Grant("waiting");
    SendForwardDone("waiting", {42});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("waiting");
    EXPECT_TRUE(scheduler_->CanClearCache());
    EXPECT_TRUE(scheduler_->ClearCache());
}

// An aborted holder -- here the runtime found a NaN in its final forward --
// parks in PdTransferring until its transfer stopped: it no longer waits for
// a destination, so it neither stops the prompts behind a granted one that
// does not fit beside it nor is requeued, and its pages free once the
// transport reports the transfer stopped.
TEST_F(IndependentAdmissionRequeueTest, AnAbortedHolderNeitherStopsLaterPromptsNorIsRequeued) {
    SubmitReady("aborted", 7);  // with its decode reserve, 4 of the 8 pages
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"aborted"});
    SendForwardDone("aborted", {42});
    scheduler_->Advance(ExecutionEvent{}.With(forward::Abort{"aborted"}));
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    EXPECT_TRUE(scheduler_->PdTransferPinned("aborted"));
    SubmitReady("granted", 9, 100);  // 5 pages
    Grant("granted");
    std::vector<std::string> admitted;
    const auto round = [&](int k) {
        const std::string small = "small" + std::to_string(k);
        SubmitReady(small, 1, 1000 + 10 * k);  // 1 page
        Grant(small);
        const auto plan = PlanOnce();
        if (plan.remote_decode) {
            for (const auto& id : plan.remote_decode->request_ids) {
                EXPECT_NE(id, "aborted");
                Ack(id);
            }
        }
        for (const auto& id : FindForwardBatch(plan)->request_ids) {
            admitted.push_back(id);
            SendForwardDone(id, {7});
        }
    };
    for (int k = 0; k < 4; ++k) round(k);
    EXPECT_EQ(admitted, (std::vector<std::string>{"small0", "small1", "small2", "small3"}));
    EXPECT_TRUE(scheduler_->PdTransferPinned("aborted"));
    EXPECT_EQ(scheduler_->RequestTokenSize("aborted"), 8);

    Fail("aborted");
    round(4);
    EXPECT_NE(std::ranges::find(admitted, std::string{"granted"}), admitted.end());
    EXPECT_EQ(scheduler_->RequestTokenSize("aborted"), -1);
}

// Waiting prompts hold every request slot: a granted prompt that fits the
// pool takes the newest one's slot.
TEST_F(IndependentAdmissionRequeueTest, GrantedPromptTakesTheRequestSlotOfTheNewestWaitingPrompt) {
    SubmitReady("oldest", 3);
    SubmitReady("older", 3, 100);
    SubmitReady("newest", 3, 200);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, (std::vector<std::string>{"oldest", "older", "newest"}));
    SendForwardDone("oldest", {40});
    SendForwardDone("older", {41});
    SendForwardDone("newest", {42});
    EXPECT_EQ(scheduler_->FreeRequestSlots(), 0);
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 2);

    SubmitReady("granted", 3, 300);
    Grant("granted");
    const auto requeue = PlanOnce();
    EXPECT_FALSE(requeue.remote_decode.has_value());
    ASSERT_EQ(FindForwardBatch(requeue)->request_ids, std::vector<std::string>{"granted"});
    EXPECT_FALSE(scheduler_->PdTransferPinned("newest"));
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 2u);
    SendForwardDone("granted", {7});
    Grant("oldest");
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, (std::vector<std::string>{"oldest", "granted"}));
}

// Smaller granted prompts that would fit do not pass a granted prompt that a
// computed prompt without a destination blocks: admitted, one every round,
// they would keep every round in prefill progress, the computed prompt would
// never be requeued and the granted prompt would wait for its destination.
// The first round admits none of them, so it requeues the computed prompt;
// the stream follows the granted prompt.
TEST_F(IndependentAdmissionRequeueTest, AStreamOfSmallerGrantedPromptsWaitsBehindABlockedGrantedPrompt) {
    SubmitReady("waiting", 7);  // with its decode reserve, 4 of the 8 pages
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"waiting"});
    SendForwardDone("waiting", {42});
    SubmitReady("granted", 9, 100);  // 5 pages
    Grant("granted");
    std::vector<std::string> admitted;
    for (int k = 0; k < 8; ++k) {
        const std::string small = "small" + std::to_string(k);
        SubmitReady(small, 1, 1000 + 10 * k);  // 1 page
        Grant(small);
        const auto plan = PlanOnce();
        if (plan.remote_decode) {
            for (const auto& id : plan.remote_decode->request_ids) Ack(id);
        }
        for (const auto& id : FindForwardBatch(plan)->request_ids) {
            admitted.push_back(id);
            SendForwardDone(id, {7});
        }
        if (k == 0) {
            EXPECT_EQ(admitted, std::vector<std::string>{"granted"});
            EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
            EXPECT_FALSE(scheduler_->PdTransferPinned("waiting"));
            EXPECT_EQ(scheduler_->Requeues(), 1u);
        }
    }
    for (int k = 0; k < 7; ++k) {
        EXPECT_NE(std::ranges::find(admitted, "small" + std::to_string(k)), admitted.end()) << "small" << k;
    }
}

// Under short-prefill-first ordering a granted prompt a requeue lets in is a
// first admission like any other: it counts toward the oldest waiting
// prompt's bypass allowance. Seven short prompts pass the old one, an eighth
// after a requeue, so the old prompt goes before a later short one.
TEST_F(IndependentAdmissionRequeueTest, ARequeuedAdmissionCountsTowardTheOldestPromptsBypasses) {
    auto config = MakeConfig();
    config.enable_short_prefill_first = true;
    config.device_allocator.total_pages = 12;
    config.cache_groups.front().total_pages = 12;
    config.max_batch_size = 10;
    SetTestSnapshotPool(config);
    config_ = WithHostPackingFromDevice(config);
    scheduler_ = std::make_unique<Scheduler>(config_);
    SubmitReady("waiting", 7);  // with its decode reserve, 4 of the 11 pages
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"waiting"});
    SendForwardDone("waiting", {42});

    SubmitReady("old", 9, 100);
    Grant("old");
    std::vector<std::string> shorts;
    for (int k = 0; k < 7; ++k) {
        shorts.push_back("short" + std::to_string(k));
        SubmitReady(shorts.back(), 1, 1000 + 10 * k);  // 1 page
        Grant(shorts.back());
    }
    // The short prompts fill the pool; the old one does not fit beside them.
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, shorts);
    for (const auto& id : shorts) SendForwardDone(id, {7});

    SubmitReady("requeued", 1, 2000);
    Grant("requeued");
    const auto requeue = PlanOnce();
    ASSERT_EQ(FindForwardBatch(requeue)->request_ids, std::vector<std::string>{"requeued"});
    ASSERT_EQ(scheduler_->Requeues(), 1u);
    ASSERT_TRUE(requeue.remote_decode.has_value());
    for (const auto& id : requeue.remote_decode->request_ids) Ack(id);
    SendForwardDone("requeued", {7});

    SubmitReady("late", 1, 3000);
    Grant("late");
    EXPECT_EQ(FindForwardBatch(PlanOnce())->request_ids, (std::vector<std::string>{"old", "late"}));
}

// With no computed prompt waiting for its destination, the pages a granted
// prompt needs come back at transfer ACKs, as under early admission, and a
// smaller granted prompt that fits still goes ahead of it.
TEST_F(IndependentAdmissionRequeueTest, WithoutAWaitingPromptASmallerGrantedPromptPassesABlockedOne) {
    SubmitReady("sending", 7);  // with its decode reserve, 4 of the 8 pages
    Grant("sending");
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"sending"});
    SendForwardDone("sending", {42});
    SubmitReady("granted", 9, 100);  // 5 pages
    Grant("granted");
    SubmitReady("small", 1, 1000);
    Grant("small");
    const auto passed = PlanOnce();
    ASSERT_TRUE(passed.remote_decode.has_value());
    EXPECT_EQ(FindForwardBatch(passed)->request_ids, std::vector<std::string>{"small"});
    Ack("sending");
    SendForwardDone("small", {7});
    EXPECT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"granted"});
}

// A head of line without a destination whose next chunk does not fit keeps
// the head of line, until a granted prompt waits behind it: then it is
// requeued, before any computed prompt, but only once its forward landed.
// Meanwhile no prompt without a destination is admitted, though one would
// fit.
TEST_F(IndependentAdmissionRequeueTest, AHeadOfLineWithoutADestinationIsRequeuedOnceItsForwardLanded) {
    SubmitReady("parked", 3);
    SubmitReady("long", 13, 100);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->input_lengths, (std::vector<std::int32_t>{3, 7}));
    SendForwardDone("parked", {41});
    SubmitReady("small", 1, 300);
    EXPECT_TRUE(FindForwardBatch(PlanOnce())->request_ids.empty());
    SubmitReady("granted", 3, 200);
    Grant("granted");
    EXPECT_TRUE(FindForwardBatch(PlanOnce())->request_ids.empty());
    EXPECT_TRUE(scheduler_->PdTransferPinned("long"));
    SendForwardDone("long");
    const auto requeue = PlanOnce();
    ASSERT_EQ(FindForwardBatch(requeue)->request_ids, std::vector<std::string>{"granted"});
    EXPECT_FALSE(scheduler_->PdTransferPinned("long"));
    EXPECT_TRUE(scheduler_->PdTransferPinned("parked"));
    EXPECT_EQ(scheduler_->Requeues(), 1u);
    EXPECT_EQ(scheduler_->RequeuedTokens(), 7u);
    // Once the granted prompt's and the parked prompt's KV went out, the
    // requeued prompt resumes from what is left of its chunk in the cache.
    SendForwardDone("granted", {7});
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("granted");
    Grant("parked");
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    Ack("parked");
    const auto again = PlanOnce();
    ASSERT_EQ(FindForwardBatch(again)->request_ids, std::vector<std::string>{"long"});
    EXPECT_EQ(FindForwardBatch(again)->extend_prefix_lens, std::vector<std::int32_t>{6});
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
