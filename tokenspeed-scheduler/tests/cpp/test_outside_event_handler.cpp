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

inline const ForwardBatch* FindForwardBatch(const std::vector<Operation>& operations) {
    for (const auto& operation : operations) {
        if (auto* batch = std::get_if<ForwardBatch>(&operation)) {
            return batch;
        }
    }
    return nullptr;
}

inline std::int32_t FindRequestIndex(const ForwardBatch* fwd, const std::string& rid) {
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
        auto plan_wb = PlanOnce();
        SendFinish("r1");
        const WriteBackBatch* wb = nullptr;
        for (const auto& op : plan_wb.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<WriteBackBatch>(cop)) {
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
        auto plan_wb2 = PlanOnce();
        SendFinish("r_fill");
        for (const auto& op : plan_wb2.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* w = std::get_if<WriteBackBatch>(cop)) {
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
    auto* fwd = FindForwardBatch(plan.Operations());
    ASSERT_NE(fwd, nullptr);
    auto idx = FindRequestIndex(fwd, "r2");
    ASSERT_GE(idx, 0) << "r2 should be in forward after host cache hit";

    // With block_size=2 and 4 prefill tokens, FullPagedTokens(except_last=true)
    // yields 3 tokens → 1 matchable page. Host has 2 pages but only 1 matches.
    // unscheduled = 4 - 1*2 = 2, so input_length = 2 and extend_prefix_len = 1*block_size = 2.
    EXPECT_EQ(fwd->input_lengths[idx], 2) << "host hit covers 1 page; 2 tokens remain";

    if (!fwd->extend_prefix_lens.empty()) {
        EXPECT_EQ(fwd->extend_prefix_lens[idx], 1 * PageSize()) << "extend_prefix_len should cover the 1 loadback page";
    }
}

class DisaggDecodeAdmissionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.block_size = 2;
        // Cache block 0 is the null page, leaving three usable pages.
        cfg.device_allocator.total_pages = 4;
        cfg.host_allocator.total_pages = 4;
        cfg.max_scheduled_tokens = 2;
        cfg.max_batch_size = 1;
        cfg.decode_input_tokens = 1;
        cfg.role = Role::kD;
        cfg.enable_l3_storage = false;
        cfg.disable_l2_cache = false;
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
    const ForwardBatch* prefill = FindForwardBatch(admission.Operations());
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->request_ids, (std::vector<std::string>{"r0"}));
    EXPECT_EQ(prefill->input_lengths, (std::vector<std::int32_t>{4}));
    ASSERT_EQ(prefill->block_tables.count("full"), 1u);
    EXPECT_EQ(prefill->block_tables.at("full").at(0).size(), 3u);
    EXPECT_EQ(scheduler_->ActiveKvPages(), 3u);

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    const ExecutionPlan decode_plan = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(decode_plan.Operations());
    ASSERT_NE(decode, nullptr);
    const std::int32_t r0 = FindRequestIndex(decode, "r0");
    ASSERT_GE(r0, 0);
    EXPECT_EQ(decode->decode_input_ids[static_cast<std::size_t>(r0)], 42);
    ASSERT_EQ(decode->block_tables.count("full"), 1u);
    EXPECT_EQ(decode->block_tables.at("full")[static_cast<std::size_t>(r0)].size(), 3u);
}

class DecodeRetractionL2TestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        return cfg;
    }
};

class DecodeRetractionCapacityTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 16;
        cfg.host_allocator.total_pages = 6;  // null parent + five retraction parents
        cfg.max_batch_size = 3;
        cfg.paged_cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        return cfg;
    }
};

class DecodeRetractionVictimSelectionTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 6;  // null parent + five Device parents
        cfg.host_allocator.total_pages = 8;    // null parent + seven retraction parents
        cfg.max_batch_size = 3;
        cfg.paged_cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        return cfg;
    }
};

class DecodeRetractionRecoveryAdmissionTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 7;  // null parent + six Device parents
        cfg.host_allocator.total_pages = 20;
        cfg.max_batch_size = 3;
        cfg.enable_mixed_prefill_decode = false;
        cfg.paged_cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        return cfg;
    }
};

class DecodeRetractionRequestSlotTestSuite : public DecodeRetractionRecoveryAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionRecoveryAdmissionTestSuite::MakeConfig();
        cfg.max_batch_size = 2;
        return cfg;
    }
};

TEST_F(DecodeRetractionCapacityTestSuite, AdmissionDoesNotReserveFutureRetractionCapacity) {
    RequestSpec longest = MakeRequestSpec("a", /*num_pages=*/2, /*start=*/1);
    longest.max_new_tokens = 6;  // five-parent maximum retraction state
    RequestSpec short_b = MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101);
    short_b.max_new_tokens = 2;  // three-parent maximum retraction state
    RequestSpec short_c = MakeRequestSpec("c", /*num_pages=*/2, /*start=*/201);
    short_c.max_new_tokens = 2;
    Submit({longest, short_b, short_c});
    SendBootstrapped("a");
    SendBootstrapped("b");
    SendBootstrapped("c");

    const ExecutionPlan first_plan = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(first_plan.Operations());
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->request_ids, (std::vector<std::string>{"a"}));
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_plan.Operations());
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->request_ids, (std::vector<std::string>{"b"}));
    const ExecutionPlan third_plan = PlanOnce();
    const ForwardBatch* third = FindForwardBatch(third_plan.Operations());
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->request_ids, (std::vector<std::string>{"c"}));
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

TEST_F(DecodeRetractionVictimSelectionTestSuite, AbortsLargestVictimWhenItsRetractionStateDoesNotFit) {
    Submit({MakeRequestSpec("parked", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("parked");
    PlanOnce();
    SendRemotePrefillDone("parked", /*bootstrap_token=*/42);
    PlanOnce();
    SendForwardDone("parked", {43});

    Submit({MakeRequestSpec("long", /*num_pages=*/2, /*start=*/101)});
    SendBootstrapped("long");
    std::vector<CacheOperation> parked_writebacks;
    for (std::int32_t token = 44; token < 52 && parked_writebacks.empty(); ++token) {
        const ExecutionPlan plan = PlanOnce();
        parked_writebacks = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const ForwardBatch* forward = FindForwardBatch(plan.Operations());
        if (parked_writebacks.empty() && forward != nullptr && !forward->request_ids.empty()) {
            ASSERT_EQ(forward->request_ids, (std::vector<std::string>{"parked"}));
            SendForwardDone("parked", {token});
        }
    }
    ASSERT_EQ(parked_writebacks.size(), 1u);
    const auto& parked_writeback = std::get<WriteBackBatch>(parked_writebacks.front());
    SendWriteBackDone(parked_writeback.op_ids.front());

    const ExecutionPlan long_admission = PlanOnce();
    ASSERT_EQ(FindForwardBatch(long_admission.Operations())->request_ids,
              (std::vector<std::string>{"long"}));
    SendRemotePrefillDone("long", /*bootstrap_token=*/142);

    Submit({MakeRequestSpec("short", /*num_pages=*/1, /*start=*/201)});
    SendBootstrapped("short");
    const ExecutionPlan short_admission = PlanOnce();
    ASSERT_EQ(FindForwardBatch(short_admission.Operations())->request_ids,
              (std::vector<std::string>{"short"}));
    SendRemotePrefillDone("short", /*bootstrap_token=*/242);

    std::optional<std::string> aborted_request_id;
    std::int32_t long_token = 143;
    std::int32_t short_token = 243;
    for (std::int32_t round = 0; round < 8 && !aborted_request_id; ++round) {
        const ExecutionPlan plan = PlanOnce();
        aborted_request_id = plan.aborted_request_id;
        if (const ForwardBatch* forward = FindForwardBatch(plan.Operations());
            !aborted_request_id && forward != nullptr) {
            for (const std::string& request_id : forward->request_ids) {
                if (request_id == "long") {
                    SendForwardDone(request_id, {long_token++});
                } else if (request_id == "short") {
                    SendForwardDone(request_id, {short_token++});
                }
            }
        }
    }

    EXPECT_EQ(aborted_request_id, std::optional<std::string>{"long"});
    PlanOnce();
    EXPECT_EQ(scheduler_->RequestTokenSize("long"), -1);
}

TEST_F(DecodeRetractionL2TestSuite, InitialAdmissionAndDecodeDoNotUsePrefixL2) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan admission = PlanOnce();
    EXPECT_TRUE(ExtractCacheOps(admission).empty());
    ASSERT_NE(FindForwardBatch(admission.Operations()), nullptr);

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    const ExecutionPlan decode = PlanOnce();
    EXPECT_TRUE(ExtractCacheOps(decode).empty());
    ASSERT_NE(FindForwardBatch(decode.Operations()), nullptr);
}

TEST_F(DecodeRetractionL2TestSuite, RetractionLetsBlockedAdmissionRun) {
    Submit({MakeRequestSpec("running", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("running");
    PlanOnce();
    SendRemotePrefillDone("running", /*bootstrap_token=*/42);
    PlanOnce();
    SendForwardDone("running", {43});

    Submit({MakeRequestSpec("blocked", /*num_pages=*/2, /*start=*/101)});
    SendBootstrapped("blocked");
    ExecutionPlan retract;
    std::vector<CacheOperation> write_back_ops;
    for (std::int32_t token = 44; token < 48 && write_back_ops.empty(); ++token) {
        retract = PlanOnce();
        write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(retract);
        const ForwardBatch* forward = FindForwardBatch(retract.Operations());
        if (write_back_ops.empty() && forward != nullptr && !forward->request_ids.empty()) {
            ASSERT_EQ(forward->request_ids, (std::vector<std::string>{"running"}));
            SendForwardDone("running", {token});
        }
    }
    ASSERT_EQ(write_back_ops.size(), 1u);
    ASSERT_NE(FindForwardBatch(retract.Operations()), nullptr);
    EXPECT_TRUE(FindForwardBatch(retract.Operations())->request_ids.empty());
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);

    SendWriteBackDone(write_back.op_ids.front());
    SendWriteBackDone(write_back.op_ids.front());  // Duplicate ACK is ignored.
    EXPECT_GT(scheduler_->HostPoolCachedBlocks(), 0)
        << "retraction ACK must publish completed Host boundaries for future reuse";
    const ExecutionPlan admitted = PlanOnce();
    const ForwardBatch* blocked = FindForwardBatch(admitted.Operations());
    ASSERT_NE(blocked, nullptr);
    EXPECT_EQ(blocked->request_ids, (std::vector<std::string>{"blocked"}));
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(admitted).empty())
        << "recovering immediately would consume the capacity retraction just released";

    SendRemotePrefillDone("blocked", /*bootstrap_token=*/142);
    SendFinish("blocked");
}

TEST_F(DecodeRetractionRecoveryAdmissionTestSuite, RecoveryStartsDecodeWithoutWaitingForLoadBackAck) {
    Submit({MakeRequestSpec("running", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("running");
    PlanOnce();
    SendRemotePrefillDone("running", /*bootstrap_token=*/42);
    PlanOnce();
    SendForwardDone("running", {43});
    ASSERT_EQ(scheduler_->ActiveKvPages(), 3u);

    RequestSpec filler = MakeRequestSpec("filler", /*num_pages=*/1, /*start=*/101);
    filler.tokens.resize(1);
    Submit({filler});
    SendBootstrapped("filler");
    const ExecutionPlan filler_admission = PlanOnce();
    const ForwardBatch* filler_forward = FindForwardBatch(filler_admission.Operations());
    ASSERT_GE(FindRequestIndex(filler_forward, "filler"), 0);
    if (FindRequestIndex(filler_forward, "running") >= 0) {
        SendForwardDone("running", {44});
    }
    SendRemotePrefillDone("filler", /*bootstrap_token=*/142);

    RequestSpec blocked = MakeRequestSpec("blocked", /*num_pages=*/1, /*start=*/201);
    blocked.tokens.resize(1);
    Submit({blocked});
    SendBootstrapped("blocked");
    std::vector<CacheOperation> write_back_ops;
    std::int32_t running_token = 45;
    std::int32_t filler_token = 143;
    for (std::int32_t round = 0; round < 12 && write_back_ops.empty(); ++round) {
        const ExecutionPlan plan = PlanOnce();
        write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const ForwardBatch* forward = FindForwardBatch(plan.Operations());
        if (write_back_ops.empty() && FindRequestIndex(forward, "running") >= 0) {
            SendForwardDone("running", {running_token++});
        }
        if (write_back_ops.empty() && FindRequestIndex(forward, "filler") >= 0) {
            SendForwardDone("filler", {filler_token++});
        }
    }
    ASSERT_EQ(write_back_ops.size(), 1u);
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);
    SendWriteBackDone(write_back.op_ids.front());
    ASSERT_EQ(scheduler_->WaitingSize(), 0u);
    SendRemotePrefillDone("blocked", /*bootstrap_token=*/242);
    SendFinish("filler");
    SendFinish("blocked");

    const ExecutionPlan recover = PlanOnce();
    const auto load_back_ops = ExtractCacheOpsOfKind<LoadBackBatch>(recover);
    ASSERT_EQ(load_back_ops.size(), 1u);
    const ForwardBatch* forward = FindForwardBatch(recover.Operations());
    ASSERT_NE(forward, nullptr);
    EXPECT_EQ(forward->request_ids, (std::vector<std::string>{"running"}));
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

TEST_F(DecodeRetractionRequestSlotTestSuite, RecoveryWaitsForAvailableRequestPoolSlot) {
    Submit({MakeRequestSpec("a", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("a");
    PlanOnce();
    SendRemotePrefillDone("a", /*bootstrap_token=*/42);

    Submit({MakeRequestSpec("holder", /*num_pages=*/2, /*start=*/101)});
    SendBootstrapped("holder");
    PlanOnce();
    SendRemotePrefillDone("holder", /*bootstrap_token=*/142);
    PlanOnce();
    SendForwardDone("a", {43});
    SendForwardDone("holder", {143});

    RequestSpec first_small = MakeRequestSpec("first_small", /*num_pages=*/1, /*start=*/201);
    first_small.tokens.resize(1);
    Submit({first_small});
    SendBootstrapped("first_small");

    std::vector<CacheOperation> write_back_ops;
    std::int32_t a_token = 44;
    std::int32_t holder_token = 144;
    for (std::int32_t round = 0; round < 12 && write_back_ops.empty(); ++round) {
        const ExecutionPlan plan = PlanOnce();
        write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const ForwardBatch* forward = FindForwardBatch(plan.Operations());
        if (write_back_ops.empty() && FindRequestIndex(forward, "a") >= 0) {
            SendForwardDone("a", {a_token++});
        }
        if (write_back_ops.empty() && FindRequestIndex(forward, "holder") >= 0) {
            SendForwardDone("holder", {holder_token++});
        }
    }
    ASSERT_EQ(write_back_ops.size(), 1u);
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);
    SendWriteBackDone(write_back.op_ids.front());

    const ExecutionPlan first_small_admission = PlanOnce();
    ASSERT_GE(FindRequestIndex(FindForwardBatch(first_small_admission.Operations()), "first_small"), 0);
    SendRemotePrefillDone("first_small", /*bootstrap_token=*/242);

    RequestSpec second_small = MakeRequestSpec("second_small", /*num_pages=*/1, /*start=*/301);
    second_small.tokens.resize(1);
    Submit({second_small});
    SendBootstrapped("second_small");
    SendFinish("holder");

    ExecutionPlan second_small_admission;
    EXPECT_NO_THROW(second_small_admission = PlanOnce());
    ASSERT_GE(FindRequestIndex(FindForwardBatch(second_small_admission.Operations()), "second_small"), 0);
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(second_small_admission).empty())
        << "recovery must wait while both request-pool slots are occupied";
}

TEST_F(DecodeRetractionRecoveryAdmissionTestSuite, SubmittedAdmissionUsesIndependentCapacityBeforeRetractionAck) {
    Submit({MakeRequestSpec("a", /*num_pages=*/2, /*start=*/1),
            MakeRequestSpec("b", /*num_pages=*/2, /*start=*/101)});
    SendBootstrapped("a");
    SendBootstrapped("b");
    PlanOnce();
    PlanOnce();
    SendRemotePrefillDone("a", /*bootstrap_token=*/42);
    SendRemotePrefillDone("b", /*bootstrap_token=*/142);

    Submit({MakeRequestSpec("blocked", /*num_pages=*/2, /*start=*/201)});
    SendBootstrapped("blocked");
    std::vector<CacheOperation> write_back_ops;
    std::int32_t a_token = 43;
    std::int32_t b_token = 143;
    for (std::int32_t round = 0; round < 12 && write_back_ops.empty(); ++round) {
        const ExecutionPlan plan = PlanOnce();
        write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const ForwardBatch* forward = FindForwardBatch(plan.Operations());
        if (write_back_ops.empty() && FindRequestIndex(forward, "a") >= 0) {
            SendForwardDone("a", {a_token++});
        }
        if (write_back_ops.empty() && FindRequestIndex(forward, "b") >= 0) {
            SendForwardDone("b", {b_token++});
        }
    }
    ASSERT_EQ(write_back_ops.size(), 1u);

    // The victim's Device pages remain pinned until the D2H ACK. Free the
    // other request so Device capacity alone would allow a new admission.
    SendAbortEvent("b");
    const ExecutionPlan while_retracting = PlanOnce();
    const ForwardBatch* forward = FindForwardBatch(while_retracting.Operations());
    ASSERT_NE(forward, nullptr);
    EXPECT_GE(FindRequestIndex(forward, "blocked"), 0)
        << "in-flight retraction pins only its own blocks and must not impose a global admission gate";
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);

    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);
    SendWriteBackDone(write_back.op_ids.front());
}

TEST_F(DecodeRetractionL2TestSuite, AbortDuringRetractionWriteBackKeepsTransferBlocksPinnedUntilAck) {
    Submit({MakeRequestSpec("running", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("running");
    PlanOnce();
    SendRemotePrefillDone("running", /*bootstrap_token=*/42);
    PlanOnce();
    SendForwardDone("running", {43});

    Submit({MakeRequestSpec("blocked", /*num_pages=*/2, /*start=*/101)});
    SendBootstrapped("blocked");
    std::vector<CacheOperation> write_back_ops;
    for (std::int32_t token = 44; token < 48 && write_back_ops.empty(); ++token) {
        const ExecutionPlan plan = PlanOnce();
        write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(plan);
        const ForwardBatch* forward = FindForwardBatch(plan.Operations());
        if (write_back_ops.empty() && forward != nullptr && !forward->request_ids.empty()) {
            SendForwardDone("running", {token});
        }
    }
    ASSERT_EQ(write_back_ops.size(), 1u);
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);

    SendAbortEvent("running");
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0)
        << "the in-flight D2H operation must outlive request-owned Device tables";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 0)
        << "the in-flight D2H operation must keep its Host destinations pinned";

    SendWriteBackDone(write_back.op_ids.front());
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 3);
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 3);
}

class PdSparseDecodeAdmissionTestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 6;  // null parent + five usable LCM parents
        cfg.host_allocator.total_pages = 32;   // Keep retraction capacity outside these placement tests.
        cfg.max_scheduled_tokens = 8;
        cfg.enable_pd_cache = true;
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

class PdSparseDecodeNoPrefixCacheTestSuite : public PdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PdSparseDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.paged_cache_groups[0].total_pages = 15;
        cfg.paged_cache_groups[1].total_pages = 8;
        cfg.disable_prefix_cache = true;
        return cfg;
    }
};

class PdSmallStatePagesTestSuite : public PdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PdSparseDecodeAdmissionTestSuite::MakeConfig();
        auto& state = cfg.paged_cache_groups[1];
        state.rows_per_page = 1;
        state.total_pages = 11;
        state.cache_blocks_per_lcm_block = 2;
        return cfg;
    }
};

TEST_F(PdSparseDecodeAdmissionTestSuite, MaterializesHistoryAndLatestStateSnapshotAtomically) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindForwardBatch(plan.Operations());
    ASSERT_NE(destination, nullptr);
    const auto& full = destination->block_tables.at("full").at(0);
    ASSERT_EQ(full.size(), 5u);
    EXPECT_TRUE(std::ranges::all_of(full, [](std::int32_t page_id) { return page_id > 0; }));

    const auto& state = destination->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 5u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_EQ(state[2], 0);
    EXPECT_GT(state[3], 0);
    EXPECT_GT(state[4], 0);
    ASSERT_EQ(plan.pages_to_zero.size(), 2u);
    EXPECT_EQ(plan.pages_to_zero.at("full"), full);
    EXPECT_EQ(plan.pages_to_zero.at("state"), (std::vector<std::int32_t>{state[3], state[4]}));
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 0);
    EXPECT_TRUE(scheduler_->PdTransferPinned("r0"));

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    EXPECT_FALSE(scheduler_->PdTransferPinned("r0"));
    const ExecutionPlan decode_plan = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(decode_plan.Operations());
    ASSERT_NE(decode, nullptr);
    EXPECT_EQ(decode->block_tables, destination->block_tables);

    ExecutionEvent succeeded;
    succeeded.With(PDEvent{pd::SucceededEvent{"r0"}});
    scheduler_->Advance(succeeded);
    EXPECT_EQ(scheduler_->PoolFreeBlocks(), 5);
}

TEST_F(PdSmallStatePagesTestSuite, LatestSnapshotUsesTheStateGroupsPageSize) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindForwardBatch(plan.Operations());
    ASSERT_NE(destination, nullptr);

    const auto& state = destination->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 9u);
    EXPECT_TRUE(std::ranges::all_of(state.begin(), state.end() - 2, [](std::int32_t page_id) { return page_id == 0; }));
    EXPECT_GT(state[state.size() - 2], 0);
    EXPECT_GT(state.back(), 0);
}

TEST_F(PdSparseDecodeAdmissionTestSuite, ReusesHistoryPrefixAndLeavesStatePrefixSparse) {
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
    const ForwardBatch* destination = FindForwardBatch(plan.Operations());
    ASSERT_NE(destination, nullptr);
    EXPECT_EQ(destination->input_lengths, (std::vector<std::int32_t>{2}));

    const auto& full = destination->block_tables.at("full").at(0);
    ASSERT_EQ(full.size(), 5u);
    EXPECT_TRUE(std::ranges::all_of(full, [](std::int32_t page_id) { return page_id > 0; }));

    const auto& state = destination->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 5u);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);
    EXPECT_EQ(state[2], 0);
    EXPECT_GT(state[3], 0);
    EXPECT_GT(state[4], 0);
}

TEST_F(PdSparseDecodeNoPrefixCacheTestSuite, RemoteBootstrapConsumesSparseTailBeforeNextDecode) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");
    PlanOnce();
    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);

    const ExecutionPlan first_decode = PlanOnce();
    const ForwardBatch* first = FindForwardBatch(first_decode.Operations());
    ASSERT_NE(first, nullptr);
    ASSERT_EQ(first->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(first->block_tables.count("state"), 1u);
    EXPECT_EQ(first->block_tables.at("state").at(0).size(), 5u);

    SendForwardDone("r0", {43});
    const ExecutionPlan second_decode = PlanOnce();
    const ForwardBatch* second = FindForwardBatch(second_decode.Operations());
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(second->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(second->block_tables.count("state"), 1u);
    EXPECT_EQ(second->block_tables.at("state").at(0).size(), 5u);

    SendForwardDone("r0", {44});
    const ExecutionPlan boundary_decode = PlanOnce();
    const ForwardBatch* boundary = FindForwardBatch(boundary_decode.Operations());
    ASSERT_NE(boundary, nullptr);
    ASSERT_EQ(boundary->request_ids, (std::vector<std::string>{"r0"}));
    ASSERT_EQ(boundary->block_tables.count("state"), 1u);
    const auto& state = boundary->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 6u);
    EXPECT_GT(state.back(), 0);
}

}  // namespace tokenspeed::test
