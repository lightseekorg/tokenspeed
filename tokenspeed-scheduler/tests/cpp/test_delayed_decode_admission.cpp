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

#include <map>
#include <memory>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class DeferredAllocationTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kP;
        cfg.decode_allocation_policy = SchedulerConfig::DecodeAllocationPolicy::kDeferred;
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

    void SubmitPrompt(const std::string& id, int length, int start = 1) {
        auto request = MakeRequestSpec(id, (length + 1) / 2, start);
        request.tokens.resize(length);
        request.max_new_tokens = 64;
        Submit(request);
    }

    // Every Decode rank allocated the destination.
    void Grant(const std::string& id) { scheduler_->Advance(ExecutionEvent{}.With(pd::DestinationReadyEvent{id})); }

    void Ack(const std::string& id) { scheduler_->Advance(ExecutionEvent{}.With(pd::SucceededEvent{id})); }

    // The transfer failed or, cancelled, retired, and stopped: no reader or
    // writer touches the request's pages any more.
    void Fail(const std::string& id) {
        scheduler_->Advance(ExecutionEvent{}.With(pd::FailedEvent{id}).With(pd::TransferStoppedEvent{id}));
    }
};

TEST_F(DeferredAllocationTest, APromptComputesWithoutPeerAdmission) {
    SubmitPrompt("r", 4);
    EXPECT_EQ(scheduler_->BootstrappingSize(), 0u);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"r"});
    SendForwardDone("r", {42});
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 1u);
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());

    Grant("r");
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"r"});
    Ack("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(DeferredAllocationTest, FinalResultRetainsEveryGroupUntilGrant) {
    SubmitPrompt("r", 6);
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
    // The remote decode takes no block: the Decode node allocated on that.
    EXPECT_LE(scheduler_->ActiveLcmBlocks(), held);
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    for (const auto& [group, blocks] : send.pages_to_zero) EXPECT_TRUE(blocks.empty());
    Ack("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_TRUE(scheduler_->ClearCache());
}

// A computed prompt waiting for its destination holds its pages and request
// slot, as one awaiting its transfer does: both count as running.
TEST_F(DeferredAllocationTest, AComputedPromptWaitingForItsDestinationIsRunning) {
    SubmitPrompt("r", 4);
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

// A prompt that holds no pages yet, admitted or not, ends at once.
TEST_F(DeferredAllocationTest, CancellingBeforeTheFirstChunkHoldsNothing) {
    SubmitPrompt("held", 4);
    SubmitPrompt("admitted", 4, 100);
    SendAbortEvent("held");
    SendAbortEvent("admitted");
    EXPECT_EQ(scheduler_->BootstrappingSize(), 0u);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    Grant("held");
    Grant("admitted");
    const auto cancelled = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(cancelled)->request_ids.empty());
    EXPECT_FALSE(cancelled.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

// A client abort cancels the transfer, whose retirement fails the request.
TEST_F(DeferredAllocationTest, CancellationIgnoresLateGrant) {
    SubmitPrompt("r", 4);
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
TEST_F(DeferredAllocationTest, CancellingWhileWaitingReleasesResourcesWithoutTransfer) {
    SubmitPrompt("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    PlanOnce();
    const auto held = scheduler_->ActiveLcmBlocks();
    SendAbortEvent("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), held);
    EXPECT_EQ(scheduler_->AwaitingDestinationSize(), 0u);
    Fail("r");
    const auto cancelled = PlanOnce();
    EXPECT_FALSE(cancelled.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    Grant("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
}

// The cancellation retires while the final forward is still out: the pages
// stay until its result lands, and a grant in between sends nothing.
TEST_F(DeferredAllocationTest, CancellingDuringTheFinalForwardReleasesOnceItsResultLands) {
    SubmitPrompt("r", 4);
    ASSERT_EQ(FindForwardBatch(PlanOnce())->request_ids, std::vector<std::string>{"r"});
    Fail("r");
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    Grant("r");
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
    SendForwardDone("r", {42});
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
    EXPECT_FALSE(PlanOnce().remote_decode.has_value());
}

// Once its KV is being sent, a cancelled prompt keeps its pages until the
// transfer stopped, as under early allocation.
TEST_F(DeferredAllocationTest, CancellingAGrantedPromptWaitsForItsTransferToStop) {
    SubmitPrompt("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    Grant("r");
    ASSERT_TRUE(PlanOnce().remote_decode.has_value());
    const auto held = scheduler_->ActiveLcmBlocks();
    SendAbortEvent("r");
    EXPECT_TRUE(scheduler_->PdTransferPinned("r"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), held);
    SendTransferStopped("r");
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_F(DeferredAllocationTest, EarlyGrantIsRememberedWithoutSendingUncomputedKv) {
    SubmitPrompt("r", 6);
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

class DeferredAllocationFullPoolTest : public DeferredAllocationTest, public ::testing::WithParamInterface<bool> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = DeferredAllocationTest::MakeConfig();
        cfg.cache_groups.resize(GetParam() ? 2 : 1);
        cfg.device_allocator.total_pages = GetParam() ? 11 : 7;
        for (auto& group : cfg.cache_groups) group.total_pages = cfg.device_allocator.total_pages;
        cfg.disable_prefix_cache = true;
        cfg.max_scheduled_tokens = 8;
        return cfg;
    }
};

TEST_P(DeferredAllocationFullPoolTest, GrantDrainsCompletedPromptAndUnblocksPartialPromptInAFullPool) {
    SubmitPrompt("first", 4);
    SubmitPrompt("partial", 6, 100);
    const auto compute = PlanOnce();
    ASSERT_EQ(FindForwardBatch(compute)->request_ids, (std::vector<std::string>{"first", "partial"}));
    EXPECT_EQ(FindForwardBatch(compute)->input_lengths, (std::vector<std::int32_t>{4, 4}));
    EXPECT_EQ(scheduler_->EmptyLcmBlocks(), 0);
    SendForwardDone("first", {42});
    SendForwardDone("partial");
    const auto ready = PlanOnce();
    EXPECT_TRUE(FindForwardBatch(ready)->request_ids.empty());
    Grant("first");
    const auto held = scheduler_->ActiveLcmBlocks();
    const auto send = PlanOnce();
    ASSERT_TRUE(send.remote_decode.has_value());
    EXPECT_EQ(send.remote_decode->request_ids, std::vector<std::string>{"first"});
    EXPECT_TRUE(send.pages_to_zero.at("full_attention").empty());
    EXPECT_LE(scheduler_->ActiveLcmBlocks(), held);
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

INSTANTIATE_TEST_SUITE_P(HistoryAndState, DeferredAllocationFullPoolTest, ::testing::Bool());

TEST_F(DeferredAllocationTest, GrantDuringFinalForwardSendsAsSoonAsItsResultLands) {
    SubmitPrompt("r", 4);
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

TEST_F(DeferredAllocationTest, DecodeFailureWhileAwaitingGrantReleasesAllSourceGroups) {
    SubmitPrompt("r", 4);
    PlanOnce();
    SendForwardDone("r", {42});
    PlanOnce();
    Fail("r");
    const auto failed = PlanOnce();
    EXPECT_FALSE(failed.remote_decode.has_value());
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

// Under early allocation the one event admits the prompt with its
// destination allocated: its KV is sent as soon as it is computed, and the
// destination's own event changes nothing. A config without a policy, as a
// caller built before the policy existed sends it, is early.
TEST_F(DeferredAllocationTest, EarlyPolicySendsWithoutASeparateDestinationGrant) {
    using Policy = SchedulerConfig::DecodeAllocationPolicy;
    for (const std::optional<Policy> policy : {std::optional{Policy::kEarly}, std::optional<Policy>{}}) {
        auto cfg = MakeConfig();
        cfg.decode_allocation_policy = policy;
        Scheduler early{WithHostPackingFromDevice(cfg)};
        const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
        early.SubmitRequests({RequestSpec{.request_id = "r", .tokens = {1, 2, 3, 4}, .max_new_tokens = 8}});
        early.Advance(ExecutionEvent{}.With(pd::DestinationReadyEvent{"r"}));
        EXPECT_TRUE(FindForwardBatch(early.NextExecutionPlan(options))->request_ids.empty());
        EXPECT_EQ(early.BootstrappingSize(), 1u);
        early.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{"r"}));
        ASSERT_EQ(FindForwardBatch(early.NextExecutionPlan(options))->request_ids, std::vector<std::string>{"r"});
        early.Advance(ExecutionEvent{}.With(forward::ExtendResult{.request_id = "r", .tokens = {42}}));
        EXPECT_EQ(early.AwaitingDestinationSize(), 0u);
        const auto send = early.NextExecutionPlan(options);
        EXPECT_TRUE(send.remote_decode.has_value());
    }
}

TEST_F(DeferredAllocationTest, DecodeAllocatesWithoutWaitingForPrefillCompletion) {
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

TEST_F(DeferredAllocationTest, DecodeAdmissionPressureDoesNotBlockPrefillComputation) {
    auto cfg = MakeConfig();
    cfg.role = Role::kD;
    cfg.decode_allocation_policy.reset();
    cfg.max_batch_size = 1;
    cfg.max_resident_requests = cfg.max_batch_size;
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

    SubmitPrompt("later", 4, 100);
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

TEST_F(DeferredAllocationTest, AllocationPolicyIsForPrefillOnly) {
    for (const auto role : {Role::kD, Role::kFused}) {
        auto cfg = MakeConfig();
        cfg.role = role;
        SetTestSnapshotPool(cfg);
        EXPECT_THROW(Scheduler{WithHostPackingFromDevice(cfg)}, std::invalid_argument);
        cfg.decode_allocation_policy.reset();
        EXPECT_NO_THROW(Scheduler{WithHostPackingFromDevice(cfg)});
    }
}

// A real P scheduler under deferred allocation and real D schedulers, run
// round by round. Run() does what the transfer peers do between them:
//   - a prompt reaches P and its destination's D scheduler in its arrival
//     round;
//   - P's peer admits it (pd::BootstrappedEvent on P) at once;
//   - D's peer lets D allocate the destination (pd::BootstrappedEvent on D)
//     when P admitted the prompt's last chunk, so that P holds every page the
//     prompt needs, or, with kAtArrival, at once: the order that deadlocks;
//   - D's allocation is P's pd::DestinationReadyEvent, and P's remote decode
//     lands the KV and first token on D and succeeds on P;
//   - every forward lands in the round that ran it, and a prompt finishes
//     after `decode_rounds` forwards on D.
class PrefillDecodeOrderTest : public SchedulerTestSuite {
protected:
    enum class Notice { kAtArrival, kPagesComplete };

    struct Setup {
        std::int32_t prefill_pages;              // P's pool, the null page included
        std::vector<std::int32_t> decode_pages;  // one D scheduler each, likewise
        std::int32_t decode_requests;            // each D's request slots
        std::int32_t chunk_tokens;               // P's token budget per round
        std::int32_t yield_tokens;               // P's prefill_chunk_yield_tokens
        Notice notice;
    };

    struct Prompt {
        std::string id;
        std::int32_t tokens;
        std::int32_t max_new_tokens;
        int decode_rounds;
        int arrival;
        std::size_t destination;
    };

    // Runs until every prompt finished or `rounds` passed. Returns the round
    // each prompt finished in; a prompt that did not finish is absent.
    std::map<std::string, int> Run(const Setup& setup, const std::vector<Prompt>& prompts, int rounds) {
        auto config = MakeConfig();
        config.disable_l2_cache = true;
        config.host_allocator.total_pages = 0;
        config.disable_prefix_cache = true;
        config.max_scheduled_tokens = setup.chunk_tokens;
        config.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        auto prefill_config = config;
        prefill_config.role = Role::kP;
        prefill_config.decode_allocation_policy = SchedulerConfig::DecodeAllocationPolicy::kDeferred;
        prefill_config.prefill_chunk_yield_tokens = setup.yield_tokens;
        prefill_config.device_allocator.total_pages = setup.prefill_pages;
        prefill_config.cache_groups.front().total_pages = setup.prefill_pages;
        SetTestSnapshotPool(prefill_config);
        Scheduler prefill{WithHostPackingFromDevice(prefill_config)};
        std::vector<std::unique_ptr<Scheduler>> decodes;
        for (const std::int32_t pages : setup.decode_pages) {
            auto decode_config = config;
            decode_config.role = Role::kD;
            decode_config.max_batch_size = setup.decode_requests;
            decode_config.max_resident_requests = decode_config.max_batch_size;
            decode_config.device_allocator.total_pages = pages;
            decode_config.cache_groups.front().total_pages = pages;
            SetTestSnapshotPool(decode_config);
            decodes.push_back(std::make_unique<Scheduler>(WithHostPackingFromDevice(decode_config)));
        }
        const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};

        std::map<std::string, const Prompt*> by_id;
        std::map<std::string, int> decoded;
        std::map<std::string, int> finished;
        for (int round = 0; round < rounds && finished.size() < prompts.size(); ++round) {
            for (const Prompt& prompt : prompts) {
                if (prompt.arrival != round) {
                    continue;
                }
                by_id[prompt.id] = &prompt;
                const RequestSpec spec{.request_id = prompt.id,
                                       .tokens = MakeTokens(prompt.tokens, 1000 * static_cast<int>(by_id.size())),
                                       .max_new_tokens = prompt.max_new_tokens};
                prefill.SubmitRequests({spec});
                decodes[prompt.destination]->SubmitRequests({spec});
                if (setup.notice == Notice::kAtArrival) {
                    decodes[prompt.destination]->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{prompt.id}));
                }
            }

            const auto prefill_plan = prefill.NextExecutionPlan(options);
            EXPECT_TRUE(prefill_plan.aborts.empty());
            const ForwardBatch& chunks = *FindForwardBatch(prefill_plan);
            for (std::size_t row = 0; row < chunks.request_ids.size(); ++row) {
                const std::string& id = chunks.request_ids[row];
                const bool last_chunk = chunks.CompletesPrompt(row);
                prefill.Advance(ExecutionEvent{}.With(forward::ExtendResult{
                    .request_id = id,
                    .tokens = last_chunk ? std::vector<std::int32_t>{42} : std::vector<std::int32_t>{}}));
                if (last_chunk && setup.notice == Notice::kPagesComplete) {
                    decodes[by_id.at(id)->destination]->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id}));
                }
            }
            if (prefill_plan.remote_decode) {
                for (const std::string& id : prefill_plan.remote_decode->request_ids) {
                    decodes[by_id.at(id)->destination]->Advance(
                        ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{id, 42}));
                    prefill.Advance(ExecutionEvent{}.With(pd::SucceededEvent{id}));
                }
            }

            for (std::size_t destination = 0; destination < decodes.size(); ++destination) {
                Scheduler& decode = *decodes[destination];
                const auto decode_plan = decode.NextExecutionPlan(options);
                EXPECT_TRUE(decode_plan.aborts.empty());
                // A retraction's image and its restore land at once.
                for (const CacheOperation& op : ExtractCacheOpsOfKind<SnapshotStoreBatch>(decode_plan)) {
                    for (const std::uint32_t op_id : std::get<SnapshotStoreBatch>(op).op_ids) {
                        decode.Advance(ExecutionEvent{}.With(cache::SnapshotDone{.op_id = op_id}));
                    }
                }
                for (const CacheOperation& op : ExtractCacheOpsOfKind<SnapshotRestoreBatch>(decode_plan)) {
                    for (const std::uint32_t op_id : std::get<SnapshotRestoreBatch>(op).op_ids) {
                        decode.Advance(ExecutionEvent{}.With(cache::RestoreDone{.op_id = op_id}));
                    }
                }
                if (decode_plan.remote_prefill) {
                    for (const std::string& id : decode_plan.remote_prefill->request_ids) {
                        prefill.Advance(ExecutionEvent{}.With(pd::DestinationReadyEvent{id}));
                    }
                }
                for (const std::string& id : FindForwardBatch(decode_plan)->request_ids) {
                    decode.Advance(ExecutionEvent{}.With(forward::ExtendResult{.request_id = id, .tokens = {8}}));
                    if (++decoded[id] == by_id.at(id)->decode_rounds) {
                        decode.Advance(ExecutionEvent{}.With(forward::Finish{.request_id = id}));
                        finished[id] = round;
                    }
                }
            }
        }
        if (finished.size() == prompts.size()) {
            prefill.NextExecutionPlan(options);
            EXPECT_EQ(prefill.ActiveLcmBlocks(), 0);
            for (const auto& decode : decodes) {
                decode->NextExecutionPlan(options);
                EXPECT_EQ(decode->ActiveLcmBlocks(), 0);
            }
        }
        return finished;
    }
};

// The convoy. Z decodes on D and holds 9 of its 14 pages when X, which fills
// P's 11 pages, and Y1 and Y2 arrive. D cannot fit X beside Z, but can fit
// Y1. If D may allocate at arrival it takes pages for Y1 and later Y2, which
// P cannot compute beside X, and never has the 11 pages X waits for: each
// node waits for the other. If D learns of a prompt only once P holds all
// its pages, D allocates only what P can send.
TEST_F(PrefillDecodeOrderTest, TheConvoyCompletesWhenDecodeAllocatesOnlyPromptsWhosePagesPrefillHolds) {
    const std::vector<Prompt> prompts{
        {.id = "Z", .tokens = 8, .max_new_tokens = 10, .decode_rounds = 8, .arrival = 0, .destination = 0},
        {.id = "X", .tokens = 20, .max_new_tokens = 2, .decode_rounds = 1, .arrival = 3, .destination = 0},
        {.id = "Y1", .tokens = 6, .max_new_tokens = 2, .decode_rounds = 1, .arrival = 3, .destination = 0},
        {.id = "Y2", .tokens = 6, .max_new_tokens = 2, .decode_rounds = 1, .arrival = 3, .destination = 0},
    };
    Setup setup{.prefill_pages = 12,
                .decode_pages = {15},
                .decode_requests = 8,
                .chunk_tokens = 64,
                .yield_tokens = 0,
                .notice = Notice::kPagesComplete};
    EXPECT_EQ(Run(setup, prompts, 200).size(), prompts.size());

    setup.notice = Notice::kAtArrival;
    const auto stalled = Run(setup, prompts, 200);
    EXPECT_TRUE(stalled.contains("Z"));
    EXPECT_FALSE(stalled.contains("X"));
    EXPECT_FALSE(stalled.contains("Y1"));
}

// A long prompt of several chunks and short prompts P carries beside them.
// If D may allocate at arrival, the long prompt's destination takes D's pool
// while the short prompts, computed, fill P's: the long prompt's last chunk
// and the short prompts' destinations wait for each other.
TEST_F(PrefillDecodeOrderTest, ALongPromptAndCarriedShortPromptsComplete) {
    std::vector<Prompt> prompts{
        {.id = "Z", .tokens = 40, .max_new_tokens = 4, .decode_rounds = 2, .arrival = 0, .destination = 0}};
    for (int i = 0; i < 6; ++i) {
        prompts.push_back({.id = "X" + std::to_string(i),
                           .tokens = 4,
                           .max_new_tokens = 4,
                           .decode_rounds = 2,
                           .arrival = 0,
                           .destination = 0});
    }
    Setup setup{.prefill_pages = 26,
                .decode_pages = {24},
                .decode_requests = 8,
                .chunk_tokens = 16,
                .yield_tokens = 8,
                .notice = Notice::kPagesComplete};
    EXPECT_EQ(Run(setup, prompts, 200).size(), prompts.size());

    setup.notice = Notice::kAtArrival;
    EXPECT_FALSE(Run(setup, prompts, 200).contains("Z"));
}

TEST(MaxRemoteAdmissionsTest, DecodeAdmitsMultipleSubmittedRequestsUpToConfiguredLimit) {
    auto base_cfg = [] {
        SchedulerConfig cfg{};
        cfg.role = Role::kD;
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 64;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        cfg.max_scheduled_tokens = 64;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = 8;
        cfg.max_remote_admissions_per_round = 3;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full",
            .block_granularity = 4,
            .total_pages = 64,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    };

    SchedulerConfig bad_zero = base_cfg();
    bad_zero.max_remote_admissions_per_round = 0;
    EXPECT_THROW(bad_zero.Validate(), std::invalid_argument);
    SchedulerConfig bad_role = base_cfg();
    bad_role.role = Role::kFused;
    EXPECT_THROW(bad_role.Validate(), std::invalid_argument);

    Scheduler scheduler{WithHostPackingFromDevice(base_cfg())};
    for (int i = 0; i < 4; ++i) {
        const std::string rid = "r" + std::to_string(i);
        scheduler.SubmitRequests({RequestSpec{
            .request_id = rid,
            .tokens = MakeTokens(4, 100 * (i + 1)),
            .max_new_tokens = 4,
        }});
        scheduler.Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{rid}));
    }

    const Scheduler::PlanOptions options{.allow_prefill = true, .apply_prefill_watermark = false};
    const ExecutionPlan p1 = scheduler.NextExecutionPlan(options);
    ASSERT_TRUE(p1.remote_prefill.has_value());
    EXPECT_EQ(p1.remote_prefill->request_ids, (std::vector<std::string>{"r0", "r1", "r2"}));

    const ExecutionPlan p2 = scheduler.NextExecutionPlan(options);
    ASSERT_TRUE(p2.remote_prefill.has_value());
    EXPECT_EQ(p2.remote_prefill->request_ids, (std::vector<std::string>{"r3"}));
}

}  // namespace tokenspeed::test
