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

TEST(ExecutionEventTest, StoresConcreteEventsInInsertionOrder) {
    ExecutionEvent event;
    event.With(cache::WriteBackDone{.op_id = 7})
        .With(forward::Abort{.request_id = "r0"})
        .With(pd::BootstrappedEvent{"r1"});

    ASSERT_EQ(event.Events().size(), 3u);
    EXPECT_TRUE(std::holds_alternative<cache::WriteBackDone>(event.Events()[0]));
    EXPECT_TRUE(std::holds_alternative<forward::Abort>(event.Events()[1]));
    EXPECT_TRUE(std::holds_alternative<pd::BootstrappedEvent>(event.Events()[2]));
}

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

class FinishOnlyWriteBackTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.host_allocator.total_pages = 8;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(FinishOnlyWriteBackTestSuite, PrefillStreamsMlaPagesDecodeDefersUntilFinish) {
    Submit(MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1));
    const ExecutionPlan prefill = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(prefill).empty())
        << "the first prefill admit has no completed pages yet";
    SendForwardDone("r0", {42});

    const ExecutionPlan after_prefill = PlanOnce();
    std::vector<CacheOperation> prefill_stores = ExtractCacheOpsOfKind<WriteBackBatch>(after_prefill);
    ASSERT_EQ(prefill_stores.size(), 1u) << "completed prompt pages stream when leaving prefill";
    const auto& prefill_write_back = std::get<WriteBackBatch>(prefill_stores.front());
    ASSERT_EQ(prefill_write_back.op_ids.size(), 1u);
    ASSERT_EQ(prefill_write_back.group_ids.size(), 1u);
    EXPECT_EQ(prefill_write_back.group_ids.front(), (std::vector<std::uint32_t>{0, 0}));
    SendWriteBackDone(prefill_write_back.op_ids.front());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 2);

    SendForwardDone("r0", {43});
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty());
    SendForwardDone("r0", {44});
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty())
        << "ordinary decode must not stream a newly completed MLA page";
    SendForwardDone("r0", {45});
    SendFinish("r0");

    const ExecutionPlan finish_plan = PlanOnce();
    const std::vector<CacheOperation> finish_stores = ExtractCacheOpsOfKind<WriteBackBatch>(finish_plan);
    ASSERT_EQ(finish_stores.size(), 1u);
    const auto& finish_write_back = std::get<WriteBackBatch>(finish_stores.front());
    ASSERT_EQ(finish_write_back.group_ids.size(), 1u);
    EXPECT_EQ(finish_write_back.group_ids.front(), (std::vector<std::uint32_t>{0}))
        << "finish writes only the new decode MLA page";
    SendWriteBackDone(finish_write_back.op_ids.front());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3);
}

TEST_F(FinishOnlyWriteBackTestSuite, FinishWithoutEligiblePageEmitsNoStore) {
    Submit(RequestSpec{.request_id = "r0", .tokens = {1}});
    PlanOnce();
    SendForwardDone("r0", {42});
    SendFinish("r0");

    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty());
}

TEST_F(FinishOnlyWriteBackTestSuite, AbortEmitsNoStore) {
    Submit(MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1));
    PlanOnce();
    SendForwardDone("r0", {42});
    PlanOnce();
    SendForwardDone("r0", {43});
    SendAbortEvent("r0");

    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty());
}

class FinishOnlyHybridWriteBackTestSuite : public FinishOnlyWriteBackTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = FinishOnlyWriteBackTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 16;
        cfg.host_allocator.total_pages = 16;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        for (std::int32_t i = 0; i < 3; ++i) {
            CacheGroupConfig state = cfg.cache_groups.front();
            state.group_id = "state_" + std::to_string(i);
            state.family = CacheGroupFamily::State;
            cfg.cache_groups.push_back(std::move(state));
        }
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(FinishOnlyHybridWriteBackTestSuite, FinishStoresDecodeMlaPagesWithoutNewKdaSnapshot) {
    Submit(MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1));
    const ExecutionPlan prefill = PlanOnce();
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(prefill).empty());
    SendForwardDone("r0", {42});

    const ExecutionPlan after_prefill = PlanOnce();
    std::vector<CacheOperation> prefill_stores = ExtractCacheOpsOfKind<WriteBackBatch>(after_prefill);
    ASSERT_EQ(prefill_stores.size(), 1u);
    const auto& prefill_write_back = std::get<WriteBackBatch>(prefill_stores.front());
    ASSERT_EQ(prefill_write_back.group_ids.size(), 1u);
    EXPECT_EQ(prefill_write_back.group_ids.front(), (std::vector<std::uint32_t>{0, 0, 1, 2, 3}))
        << "prefill publication streams every MLA page plus one snapshot for each of three KDA groups";
    SendWriteBackDone(prefill_write_back.op_ids.front());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 5);

    SendForwardDone("r0", {43});
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty());
    SendForwardDone("r0", {44});
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce()).empty())
        << "ordinary decode must not stream the new MLA page or KDA snapshots";
    SendForwardDone("r0", {45});
    SendFinish("r0");
    const std::vector<CacheOperation> finish_stores = ExtractCacheOpsOfKind<WriteBackBatch>(PlanOnce());
    ASSERT_EQ(finish_stores.size(), 1u);
    const auto& finish_write_back = std::get<WriteBackBatch>(finish_stores.front());
    ASSERT_EQ(finish_write_back.group_ids.size(), 1u);
    EXPECT_EQ(finish_write_back.group_ids.front(), (std::vector<std::uint32_t>{0}))
        << "finish writes decode MLA pages but no decode KDA snapshot";
    SendWriteBackDone(finish_write_back.op_ids.front());
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 6);
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
        const ExecutionPlan seed_stream = PlanOnce();
        ASSERT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(seed_stream).empty())
            << "SetupHostCache: expected WriteBack op for r1";
        AckWriteBacks(seed_stream);
        SendFinish("r1");
        AckWriteBacks(PlanOnce());
        PlanOnce();

        Submit(MakeRequestSpec("r_fill", /*num_pages=*/3, /*start=*/100));
        PlanOnce();
        SendForwardDone("r_fill", {200});
        AckWriteBacks(PlanOnce());
        SendFinish("r_fill");
        AckWriteBacks(PlanOnce());
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

    // With prefix_granularity=2 and 4 prefill tokens, FullPrefixPages(except_last=true)
    // yields 3 tokens → 1 matchable page. Host has 2 pages but only 1 matches.
    // unscheduled = 4 - 1*2 = 2, so input_length = 2 and extend_prefix_len = 1*prefix_granularity = 2.
    EXPECT_EQ(fwd->input_lengths[idx], 2) << "host hit covers 1 page; 2 tokens remain";

    if (!fwd->extend_prefix_lens.empty()) {
        EXPECT_EQ(fwd->extend_prefix_lens[idx], 1 * PrefixGranularity())
            << "extend_prefix_len should cover the 1 loadback page";
    }
}

class DisaggDecodeAdmissionTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.prefix_granularity = 2;
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

        CacheGroupConfig full;
        full.group_id = "full";
        full.block_granularity = cfg.prefix_granularity;
        full.total_pages = cfg.device_allocator.total_pages;
        full.retention = CacheGroupConfig::Retention::FullHistory;
        full.family = CacheGroupFamily::History;
        full.transfer_policy = CacheTransferPolicy::FullSuffix;
        cfg.cache_groups = {full};
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void SendBootstrapped(const std::string& request_id) {
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{request_id});
        scheduler_->Advance(std::move(event));
    }

    void SendRemotePrefillDone(const std::string& request_id, std::int32_t bootstrap_token) {
        ExecutionEvent event;
        event.With(pd::RemotePrefillDoneEvent{request_id, bootstrap_token});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(DisaggDecodeAdmissionTestSuite, ReservesWholeDestinationAndSurvivesRemoteCompletion) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan admission = PlanOnce();
    const ForwardBatch* prefill = FindRemoteAdmission(admission);
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->request_ids, (std::vector<std::string>{"r0"}));
    EXPECT_EQ(prefill->input_lengths, (std::vector<std::int32_t>{4}));
    ASSERT_EQ(prefill->block_tables.count("full"), 1u);
    EXPECT_EQ(prefill->block_tables.at("full").at(0).size(), 3u);
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 3u);

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

class DisaggDecodePriorityTestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 5;
        cfg.max_batch_size = 2;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(DisaggDecodePriorityTestSuite, PrefillDoneDoesNotMixWithAnotherSubmittedRequest) {
    Submit(MakeRequestSpec("a", /*num_pages=*/1));
    SendBootstrapped("a");

    const ExecutionPlan admission = PlanOnce();
    ASSERT_NE(FindRemoteAdmission(admission), nullptr);
    ASSERT_EQ(FindRemoteAdmission(admission)->request_ids, (std::vector<std::string>{"a"}));
    SendRemotePrefillDone("a", /*bootstrap_token=*/42);
    Submit(MakeRequestSpec("b", /*num_pages=*/1, /*start=*/101));
    SendBootstrapped("b");

    const ExecutionPlan next = PlanOnce();
    const ForwardBatch* forward = FindForwardBatch(next.Operations());
    ASSERT_NE(forward, nullptr);
    EXPECT_EQ(forward->request_ids, (std::vector<std::string>{"a"})) << "a's bootstrap decode runs";
    // The remote admission rides plan.remote_prefill beside the decode
    // batch: it consumes no token budget and no batch slot, so there is
    // nothing to defer for.
    const ForwardBatch* beside = FindRemoteAdmission(next);
    ASSERT_NE(beside, nullptr) << "a remote admission rides beside the decode batch";
    EXPECT_EQ(beside->request_ids, (std::vector<std::string>{"b"}));
    EXPECT_EQ(beside->NumExtends(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

// D-role retraction: two request slots, six usable Device parents. "running"
// (2-page prompt, 3 blocks) decodes beside "other" (2-page prompt, 3 blocks)
// whose remote prefill never completes until the test says so -- a
// RemotePrefilling request is PD-pinned, never a victim, and holds its pages
// without growing. A third prompt, "blocked", then finds no room.
class DecodeRetractionL2TestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.disable_l2_cache = false;
        cfg.disable_prefix_cache = false;
        cfg.device_allocator.total_pages = 7;
        cfg.host_allocator.total_pages = 7;
        cfg.max_batch_size = 2;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Admits "running" and "other" remotely, lands "running" and lets it
    // decode until it needs a fourth page while "blocked" waits: that round
    // retracts "running" (the blocker itself; its pages serve the waiting
    // prompt) and admits "blocked" on them. Returns the retraction round.
    // Post: running is Retracted at 7 tokens (6 computed, 3 pages), other is
    // RemotePrefilling, blocked is RemotePrefilling, the Device pool is full.
    void DriveRunningToRetraction(ExecutionPlan& retract) {
        Submit({MakeRequestSpec("running", /*num_pages=*/2, /*start=*/1),
                MakeRequestSpec("other", /*num_pages=*/2, /*start=*/201)});
        SendBootstrapped("running");
        SendBootstrapped("other");
        ExecutionPlan first = PlanOnce();
        ASSERT_NE(FindRemoteAdmission(first), nullptr);
        ASSERT_EQ(FindRemoteAdmission(first)->request_ids, (std::vector<std::string>{"running"}));
        ExecutionPlan second = PlanOnce();
        ASSERT_NE(FindRemoteAdmission(second), nullptr);
        ASSERT_EQ(FindRemoteAdmission(second)->request_ids, (std::vector<std::string>{"other"}));
        SendRemotePrefillDone("running", /*bootstrap_token=*/42);
        ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);

        Submit({MakeRequestSpec("blocked", /*num_pages=*/2, /*start=*/101)});
        SendBootstrapped("blocked");
        // Two decodes fit the reserve (tokens 5 and 6); the third needs a page.
        for (const std::int32_t token : {43, 44}) {
            ExecutionPlan decode = PlanOnce();
            const ForwardBatch* forward = FindForwardBatch(decode.Operations());
            ASSERT_NE(forward, nullptr);
            ASSERT_EQ(forward->request_ids, (std::vector<std::string>{"running"}));
            ASSERT_EQ(FindRemoteAdmission(decode), nullptr) << "blocked does not fit";
            ASSERT_EQ(scheduler_->RetractedSize(), 0u) << "a request with a forward out is never retracted";
            SendForwardDone("running", {token});
        }
        retract = PlanOnce();
        ASSERT_EQ(scheduler_->RetractedSize(), 1u) << "running gave way";
        ASSERT_TRUE(FindForwardBatch(retract.Operations())->request_ids.empty());
        ASSERT_NE(FindRemoteAdmission(retract), nullptr);
        ASSERT_EQ(FindRemoteAdmission(retract)->request_ids, (std::vector<std::string>{"blocked"}))
            << "the freed pages serve the waiting prompt in the same round";
        ASSERT_EQ(scheduler_->RequestTokenSize("running"), 7);
    }

    void FinishRemote(const std::string& request_id, std::int32_t bootstrap_token) {
        SendRemotePrefillDone(request_id, bootstrap_token);
        SendAbortEvent(request_id);
    }
};

class DecodeRetractionCapacityTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 16;
        cfg.host_allocator.total_pages = 6;  // null parent + five retraction parents
        cfg.max_batch_size = 3;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

// Eight usable parents and three slots: room for a restore beside a decoding
// request and a remote admission in the same round once a large pinned
// neighbour leaves.
class DecodeRetractionMixedTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;
        cfg.cache_groups.front().total_pages = cfg.device_allocator.total_pages;
        cfg.max_batch_size = 3;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class DecodeRetractionWithoutL2TestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        cfg.host_allocator.total_pages = 0;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class DecodeRetractionNoPrefixCacheTestSuite : public DecodeRetractionL2TestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeRetractionL2TestSuite::MakeConfig();
        cfg.disable_prefix_cache = true;
        SetTestSnapshotPool(cfg);
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

    // One remote admission per round: each one reserves a whole prompt, so
    // admitting a queue's worth at once would drain the pool before any of
    // their KV arrives.
    const ExecutionPlan first_plan = PlanOnce();
    const ForwardBatch* first = FindRemoteAdmission(first_plan);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->request_ids, (std::vector<std::string>{"a"}));
    const ExecutionPlan second_plan = PlanOnce();
    const ForwardBatch* second = FindRemoteAdmission(second_plan);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->request_ids, (std::vector<std::string>{"b"}));
    const ExecutionPlan third_plan = PlanOnce();
    const ForwardBatch* third = FindRemoteAdmission(third_plan);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->request_ids, (std::vector<std::string>{"c"}));
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
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

TEST_F(DecodeRetractionL2TestSuite, AnAbortedVictimStopsQualifyingForReadmission) {
    // Readmission order is read off the Retracted states themselves, so a
    // victim that dies while retracted simply stops qualifying -- there is no
    // separate queue that could still name it -- and its image dies with it.
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    ASSERT_NE(FindSnapshotStore(retract), nullptr);
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 6) << "6 computed tokens end on a page boundary: no tail page";
    AckImageStores(retract);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 3) << "the image pins its three published pages on Host";

    // The victim dies before it is ever restored.
    SendAbortEvent("running");
    EXPECT_EQ(scheduler_->WaitingSize(), 0u) << "the aborted victim leaves the waiting set";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 6) << "its image is released";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "and its Host L2 pins; the entries stay cached";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3);

    FinishRemote("blocked", /*bootstrap_token=*/142);
    FinishRemote("other", /*bootstrap_token=*/242);
    const ExecutionPlan after_abort = PlanOnce();
    EXPECT_EQ(FindRestore(after_abort), nullptr) << "an aborted victim must not be restored";
}

TEST_F(DecodeRetractionL2TestSuite, RetractionLetsBlockedAdmissionRunAndRestoresTheVictimLater) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);

    // The image's L2 leg is today's stream-ordered store of the published
    // pages; its tail leg -- here only the slot-state blob, since 6 computed
    // tokens end on a page boundary -- is the snapshot store.
    const auto write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(retract);
    ASSERT_EQ(write_back_ops.size(), 1u);
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);
    EXPECT_EQ(write_back.source_pinned, std::vector<bool>{false})
        << "the victim's pages are granted away this round; the runtime must order the copy ahead of reuse";
    EXPECT_EQ(write_back.src_pages.at(0).size(), 3u);
    const SnapshotStoreBatch* store = FindSnapshotStore(retract);
    ASSERT_NE(store, nullptr) << "the tail leg always rides the plan: it carries the slot-state blob";
    EXPECT_EQ(store->request_ids, (std::vector<std::string>{"running"}));
    EXPECT_TRUE(store->src_pages.at(0).empty());
    EXPECT_EQ(store->snapshot_slots.at(0), 1) << "one blob slot, numbered from 1";
    EXPECT_GE(store->request_pool_indices.at(0), 1) << "the victim's slot, to export from";
    EXPECT_EQ(FindRestore(retract), nullptr) << "nothing is restored while its image is still in flight";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u) << "the Retracted request remains visible as scheduler pressure";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0) << "other and blocked hold the pool";

    SendWriteBackDone(write_back.op_ids.front());
    SendWriteBackDone(write_back.op_ids.front());  // Duplicate ACK is ignored.
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3) << "the L2 leg's ACK publishes the Host entries";
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 3) << "and the suspended request pins every one of them";
    SendSnapshotDone(store->op_ids.front());
    SendSnapshotDone(store->op_ids.front());  // Duplicate ACK is ignored.

    // The restore (3 imaged pages + a decode slot) does not fit while other
    // and blocked hold the pool: it waits and retracts nobody.
    FinishRemote("blocked", /*bootstrap_token=*/142);
    const ExecutionPlan waiting = PlanOnce();
    EXPECT_EQ(FindRestore(waiting), nullptr);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_TRUE(scheduler_->PdTransferPinned("other")) << "other is never a victim";
    FinishRemote("other", /*bootstrap_token=*/242);

    // The restore rides beside an empty decode batch: no local prefill, no
    // load-back, and the request is schedulable only after the ACK.
    const ExecutionPlan recovery = PlanOnce();
    ASSERT_TRUE(FindForwardBatch(recovery.Operations())->request_ids.empty());
    EXPECT_TRUE(ExtractCacheOpsOfKind<LoadBackBatch>(recovery).empty());
    const SnapshotRestoreBatch* restore = FindRestore(recovery);
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->request_ids, (std::vector<std::string>{"running"}));
    EXPECT_EQ(restore->snapshot_slots.at(0), 1);
    EXPECT_GE(restore->request_pool_indices.at(0), 1) << "the new slot, to import into";
    EXPECT_EQ(restore->src_pages.at(0).size(), 3u) << "the published pages come back from L2";
    for (const std::uint8_t tier : restore->source_tiers.at(0)) {
        EXPECT_EQ(tier, static_cast<std::uint8_t>(HostTier::kL2));
    }
    EXPECT_EQ(scheduler_->DecodingSize(), 0u) << "a retracted request returns to Decode only after the ACK";
    EXPECT_EQ(scheduler_->WaitingSize(), 1u);
    SendRestoreDone(restore->op_ids.front());
    SendRestoreDone(restore->op_ids.front());  // Duplicate ACK is ignored.
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 0) << "the restore released the pins";
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3) << "the entries stay cached";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 6);
    EXPECT_EQ(scheduler_->RequestTokenSize("running"), 7) << "the same tokens, nothing rebased";

    const ExecutionPlan decode = PlanOnce();
    const ForwardBatch* resumed = FindForwardBatch(decode.Operations());
    ASSERT_NE(resumed, nullptr);
    EXPECT_EQ(resumed->request_ids, (std::vector<std::string>{"running"}));
    EXPECT_EQ(resumed->NumExtends(), 0u) << "the D role runs no prefill of any kind";
    EXPECT_EQ(resumed->decode_input_ids.front(), -1) << "a decoding victim resumes an ordinary decode";
    EXPECT_EQ(resumed->block_tables.at("full").at(0).size(), 4u) << "3 restored pages and the decode slot";
}

TEST_F(DecodeRetractionNoPrefixCacheTestSuite, RestoreUsesItsOwnImageWithPrefixCachingDisabled) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    // The victim's own pages are published for its image whatever the prefix
    // cache setting: disabling ordinary request-to-request reuse only stops
    // the admission probe.
    EXPECT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(retract).empty())
        << "disabling ordinary prefix caching must not hide a request's own image";
    AckImageStores(retract);
    FinishRemote("blocked", /*bootstrap_token=*/142);
    FinishRemote("other", /*bootstrap_token=*/242);

    const ExecutionPlan recovery = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(recovery);
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->src_pages.at(0).size(), 3u) << "the image comes back by copy";
    AckRestores(recovery);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

TEST_F(DecodeRetractionL2TestSuite, RemotePrefillInFlightStallsAdditionalAdmission) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    AckImageStores(retract);
    Submit({MakeRequestSpec("blocked-b", /*num_pages=*/2, /*start=*/301)});
    SendBootstrapped("blocked-b");

    const ExecutionPlan stalled = PlanOnce();
    EXPECT_EQ(FindRemoteAdmission(stalled), nullptr)
        << "another admission must wait while the first remote prefill can still make progress";
    const ForwardBatch* forward = FindForwardBatch(stalled.Operations());
    ASSERT_NE(forward, nullptr);
    EXPECT_TRUE(forward->request_ids.empty());
    EXPECT_EQ(FindRestore(stalled), nullptr) << "the restore waits for capacity and a request slot";
}

TEST_F(DecodeRetractionL2TestSuite, PrefillDoneRunsBeforeRetractedRecovery) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    AckImageStores(retract);
    SendRemotePrefillDone("blocked", /*bootstrap_token=*/142);

    // Both request slots are taken (other, blocked): the ready decode runs
    // and the restore waits for a slot without stalling it.
    const ExecutionPlan next = PlanOnce();
    const ForwardBatch* forward = FindForwardBatch(next.Operations());
    ASSERT_NE(forward, nullptr);
    ASSERT_FALSE(forward->request_ids.empty());
    EXPECT_EQ(forward->request_ids.front(), "blocked")
        << "a ready Decode request must run before recovery can consume its capacity";
    EXPECT_EQ(forward->decode_input_ids.front(), 142);
    EXPECT_EQ(FindRestore(next), nullptr);
}

TEST_F(DecodeRetractionMixedTestSuite, ARestoreRidesBesideTheDecodeBatchAndARemoteAdmission) {
    // Eight usable parents: running (3) and other (a 4-page prompt, 5) fill
    // the pool; "blocked-a" (a 1-page prompt, 2 blocks) does not fit, and
    // running gives way at its third decode.
    Submit({MakeRequestSpec("running", /*num_pages=*/2, /*start=*/1),
            MakeRequestSpec("other", /*num_pages=*/4, /*start=*/201)});
    SendBootstrapped("running");
    SendBootstrapped("other");
    PlanOnce();
    PlanOnce();
    ASSERT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    SendRemotePrefillDone("running", /*bootstrap_token=*/42);
    Submit({MakeRequestSpec("blocked-a", /*num_pages=*/1, /*start=*/101)});
    SendBootstrapped("blocked-a");
    ExecutionPlan retract;
    for (std::int32_t token = 43; token < 60 && scheduler_->RetractedSize() == 0u; ++token) {
        retract = PlanOnce();
        if (FindRequestIndex(FindForwardBatch(retract.Operations()), "running") >= 0) {
            SendForwardDone("running", {token});
        }
    }
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);
    ASSERT_NE(FindRemoteAdmission(retract), nullptr);
    ASSERT_EQ(FindRemoteAdmission(retract)->request_ids, (std::vector<std::string>{"blocked-a"}));
    ASSERT_EQ(scheduler_->RequestTokenSize("running"), 7);
    AckImageStores(retract);
    SendRemotePrefillDone("blocked-a", /*bootstrap_token=*/142);
    FinishRemote("other", /*bootstrap_token=*/242);
    Submit({MakeRequestSpec("blocked-b", /*num_pages=*/1, /*start=*/301)});
    SendBootstrapped("blocked-b");

    // One round: running's restore (a cache op: 3 pages + a decode slot),
    // blocked-a's first decode (the batch) and blocked-b's remote admission
    // (the remote stream) all ride together; nothing claims the round.
    const ExecutionPlan together = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(together);
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->request_ids, (std::vector<std::string>{"running"}));
    const ForwardBatch* forward = FindForwardBatch(together.Operations());
    ASSERT_NE(forward, nullptr);
    EXPECT_EQ(forward->request_ids, (std::vector<std::string>{"blocked-a"}));
    EXPECT_EQ(forward->decode_input_ids.front(), 142);
    ASSERT_NE(FindRemoteAdmission(together), nullptr);
    EXPECT_EQ(FindRemoteAdmission(together)->request_ids, (std::vector<std::string>{"blocked-b"}));
    EXPECT_FALSE(scheduler_->PdTransferPinned("running"))
        << "a restore is no PD transfer: nothing waits for a nonexistent completion";
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    AckRestores(together);
    SendForwardDone("blocked-a", {143});

    const ExecutionPlan both = PlanOnce();
    const ForwardBatch* decodes = FindForwardBatch(both.Operations());
    ASSERT_NE(decodes, nullptr);
    EXPECT_EQ(decodes->request_ids, (std::vector<std::string>{"running", "blocked-a"}));
    EXPECT_EQ(decodes->NumExtends(), 0u);
}

TEST_F(DecodeRetractionWithoutL2TestSuite, WithoutHostCacheTheWholeImageRidesTheSnapshotPool) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    EXPECT_TRUE(ExtractCacheOpsOfKind<WriteBackBatch>(retract).empty()) << "no Host cache, no L2 leg";
    const SnapshotStoreBatch* store = FindSnapshotStore(retract);
    ASSERT_NE(store, nullptr);
    EXPECT_EQ(store->src_pages.at(0).size(), 3u) << "every data page goes to the snapshot pool";
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 6 - 3);
    AckImageStores(retract);
    FinishRemote("blocked", /*bootstrap_token=*/142);
    FinishRemote("other", /*bootstrap_token=*/242);

    const ExecutionPlan recovery = PlanOnce();
    const SnapshotRestoreBatch* restore = FindRestore(recovery);
    ASSERT_NE(restore, nullptr);
    EXPECT_EQ(restore->src_pages.at(0).size(), 3u);
    for (const std::uint8_t tier : restore->source_tiers.at(0)) {
        EXPECT_EQ(tier, static_cast<std::uint8_t>(HostTier::kSnapshotPool));
    }
    for (const std::string& hash : restore->content_hashes.at(0)) {
        EXPECT_TRUE(hash.empty()) << "pool rows carry no key";
    }
    EXPECT_TRUE(FindForwardBatch(recovery.Operations())->request_ids.empty()) << "no recompute of any suffix";
    AckRestores(recovery);
    EXPECT_EQ(scheduler_->RequestTokenSize("running"), 7);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->SnapshotPoolFreeBlocks(), 6);
}

TEST_F(DecodeRetractionL2TestSuite, WriteBackAckPublishesTheImagesHostEntries) {
    ExecutionPlan retract;
    DriveRunningToRetraction(retract);
    const auto write_back_ops = ExtractCacheOpsOfKind<WriteBackBatch>(retract);
    ASSERT_EQ(write_back_ops.size(), 1u);
    const auto& write_back = std::get<WriteBackBatch>(write_back_ops.front());
    ASSERT_EQ(write_back.op_ids.size(), 1u);

    // The victim's pages were freed and immediately granted to the blocked
    // admission in the same round -- no D2H source pin holds them (the
    // execution stream orders the copy ahead of the granted request's use).
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 6 - 3)
        << "the in-flight D2H operation must keep its Host destinations pinned";

    SendWriteBackDone(write_back.op_ids.front());
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 0);
    EXPECT_EQ(scheduler_->HostPoolCachedBlocks(), 3);
    EXPECT_EQ(scheduler_->HostPoolPinnedBlocks(), 3) << "published AND pinned by the suspended request";
    EXPECT_EQ(scheduler_->HostPoolFreeBlocks(), 6 - 3);
}

class PdSparseDecodeAdmissionTestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 7;  // null parent + six usable LCM parents
        cfg.host_allocator.total_pages = 32;   // Keep retraction capacity outside these placement tests.
        cfg.max_scheduled_tokens = 2;
        cfg.overlap_schedule_depth = 0;
        cfg.disable_prefix_cache = false;

        CacheGroupConfig full = cfg.cache_groups.front();
        full.group_id = "full";
        full.total_pages = 13;
        full.cache_blocks_per_lcm_block = 2;
        full.transfer_policy = CacheTransferPolicy::FullSuffix;

        CacheGroupConfig state = full;
        state.group_id = "state";
        state.total_pages = 7;
        state.cache_blocks_per_lcm_block = 1;
        state.family = CacheGroupFamily::State;
        state.transfer_policy = CacheTransferPolicy::LatestSnapshot;
        cfg.cache_groups = {full, state};
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class PdSparseDecodeNoPrefixCacheTestSuite : public PdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PdSparseDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 8;
        cfg.cache_groups[0].total_pages = 15;
        cfg.cache_groups[1].total_pages = 8;
        cfg.disable_prefix_cache = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class PdSmallStatePagesTestSuite : public PdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PdSparseDecodeAdmissionTestSuite::MakeConfig();
        auto& state = cfg.cache_groups[1];
        state.block_granularity = 1;
        state.total_pages = 13;
        state.cache_blocks_per_lcm_block = 2;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class PdDecodeCapacityTestSuite : public PdSparseDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = PdSparseDecodeAdmissionTestSuite::MakeConfig();
        cfg.device_allocator.total_pages = 9;  // null parent + eight usable parents
        cfg.max_scheduled_tokens = 8;
        cfg.cache_groups[0].total_pages = 17;
        cfg.cache_groups[1].total_pages = 9;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

class PdSlidingSparseDecodeAdmissionTestSuite : public DisaggDecodeAdmissionTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DisaggDecodeAdmissionTestSuite::MakeConfig();
        cfg.prefix_granularity = 4;
        cfg.device_allocator.total_pages = 8;
        cfg.host_allocator.total_pages = 0;
        cfg.max_scheduled_tokens = 16;
        cfg.disable_prefix_cache = false;

        CacheGroupConfig sliding;
        sliding.group_id = "sliding";
        sliding.block_granularity = 2;
        sliding.total_pages = cfg.device_allocator.total_pages;
        sliding.retention = CacheGroupConfig::Retention::SlidingWindow;
        sliding.sliding_window_tokens = 4;
        sliding.family = CacheGroupFamily::History;
        sliding.transfer_policy = CacheTransferPolicy::FullSuffix;
        cfg.cache_groups = {sliding};
        SetTestSnapshotPool(cfg);
        return cfg;
    }
};

TEST_F(PdDecodeCapacityTestSuite, SingleRequestCapacityChargesTheLandingShapeOnly) {
    // Full KV uses one 2-token page per two tokens, two pages per parent. A
    // State group lands its endpoint snapshot and banks one growth block: two
    // parents, whatever the prompt length, since the D role never prefills
    // locally -- a retracted request comes back by restoring that same shape.
    // Eight usable parents therefore admit 24 total tokens: 12 pages = 6
    // full-KV parents plus 2 state parents, where 25 would need a seventh.
    EXPECT_EQ(scheduler_->MaxSingleRequestTokens(), 24);
}

TEST_F(PdSparseDecodeAdmissionTestSuite, MaterializesHistoryAndLatestStateSnapshotAtomically) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindRemoteAdmission(plan);
    ASSERT_NE(destination, nullptr);
    const auto& full = destination->block_tables.at("full").at(0);
    ASSERT_EQ(full.size(), 5u);
    EXPECT_TRUE(std::ranges::all_of(full, [](std::int32_t page_id) { return page_id > 0; }));

    // Endpoint snapshot lands in slot 3; slot 4 is the pre-reserved growth block
    // so the first boundary crossing never needs a fresh empty parent.
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
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 1);
    EXPECT_TRUE(scheduler_->PdTransferPinned("r0"));

    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    EXPECT_FALSE(scheduler_->PdTransferPinned("r0"));
    const ExecutionPlan decode_plan = PlanOnce();
    const ForwardBatch* decode = FindForwardBatch(decode_plan.Operations());
    ASSERT_NE(decode, nullptr);
    const auto& decode_state = decode->block_tables.at("state").at(0);
    ASSERT_EQ(decode_state.size(), 5u);
    EXPECT_EQ(decode_state[3], state[3]);
    EXPECT_EQ(decode_state[4], state[4]);  // first decode consumes the owned growth block in place

    ExecutionEvent succeeded;
    succeeded.With(pd::SucceededEvent{"r0"});
    scheduler_->Advance(succeeded);
    EXPECT_EQ(scheduler_->AvailableLcmBlocks(), 6);
}

TEST_F(PdSmallStatePagesTestSuite, LatestSnapshotUsesTheStateGroupsBlockGranularity) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");

    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindRemoteAdmission(plan);
    ASSERT_NE(destination, nullptr);

    const auto& state = destination->block_tables.at("state").at(0);
    ASSERT_EQ(state.size(), 9u);  // endpoint snapshot slot 7 plus one growth block at the group's own granularity
    EXPECT_TRUE(std::ranges::all_of(state.begin(), state.end() - 2, [](std::int32_t page_id) { return page_id == 0; }));
    EXPECT_GT(state[7], 0);
    EXPECT_GT(state[8], 0);
}

TEST_F(PdSparseDecodeAdmissionTestSuite, ReusesHistoryPrefixAndLeavesStatePrefixSparse) {
    Submit({MakeRequestSpec("r0", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r0");
    PlanOnce();
    SendRemotePrefillDone("r0", /*bootstrap_token=*/42);
    PlanOnce();

    ExecutionEvent succeeded;
    succeeded.With(pd::SucceededEvent{"r0"});
    scheduler_->Advance(succeeded);

    Submit({MakeRequestSpec("r1", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("r1");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindRemoteAdmission(plan);
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

TEST_F(PdSlidingSparseDecodeAdmissionTestSuite, KeepsCachedPrefixIslandWhileMaterializingRemoteTail) {
    // Seed a resumable W=4 lookback at raw-token boundary 8. At q=2 the
    // cached island occupies slots [2, 4); earlier slots are null holes.
    Submit({MakeRequestSpec("seed", /*num_pages=*/2, /*start=*/1)});
    SendBootstrapped("seed");
    const ExecutionPlan seed_admission = PlanOnce();
    const ForwardBatch* seed_destination = FindRemoteAdmission(seed_admission);
    ASSERT_NE(seed_destination, nullptr);
    const auto seed_row = seed_destination->block_tables.at("sliding").at(0);
    ASSERT_EQ(seed_row.size(), 5u);
    EXPECT_EQ(seed_row[0], 0);
    EXPECT_EQ(seed_row[1], 0);
    EXPECT_GT(seed_row[2], 0);
    EXPECT_GT(seed_row[3], 0);
    EXPECT_GT(seed_row[4], 0);

    SendRemotePrefillDone("seed", /*bootstrap_token=*/42);
    const ExecutionPlan seed_decode = PlanOnce();
    ASSERT_NE(FindForwardBatch(seed_decode.Operations()), nullptr);
    ExecutionEvent seed_succeeded;
    seed_succeeded.With(pd::SucceededEvent{"seed"});
    scheduler_->Advance(std::move(seed_succeeded));

    // The longer request hits through token 8. Its retained remote tail begins
    // at floor((16 - W + 1) / q)=6, leaving [4, 6) sparse while preserving the
    // cached island and materializing the phase page for decode reserve.
    Submit({MakeRequestSpec("long", /*num_pages=*/4, /*start=*/1)});
    SendBootstrapped("long");
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* destination = FindRemoteAdmission(plan);
    ASSERT_NE(destination, nullptr);
    EXPECT_EQ(destination->extend_prefix_lens, (std::vector<std::int32_t>{8}));
    EXPECT_EQ(destination->input_lengths, (std::vector<std::int32_t>{8}));

    const auto& row = destination->block_tables.at("sliding").at(0);
    ASSERT_EQ(row.size(), 9u);
    EXPECT_EQ(row[0], 0);
    EXPECT_EQ(row[1], 0);
    EXPECT_EQ(row[2], seed_row[2]);
    EXPECT_EQ(row[3], seed_row[3]);
    EXPECT_EQ(row[4], 0);
    EXPECT_EQ(row[5], 0);
    EXPECT_GT(row[6], 0);
    EXPECT_GT(row[7], 0);
    EXPECT_GT(row[8], 0);
    EXPECT_EQ(plan.pages_to_zero.at("sliding"), (std::vector<std::int32_t>{row[6], row[7], row[8]}));
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
