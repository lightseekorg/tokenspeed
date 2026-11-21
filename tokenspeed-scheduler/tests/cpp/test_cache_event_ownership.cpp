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

#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "cache_event_test_sink.h"
#include "integration_test_helper.h"

namespace tokenspeed {

// Place the real owner at the wire rollover boundary without issuing 2^32
// transfers. The test below still creates and retires its actual tickets.
struct TierTransferManagerTestAccess {
    static void SetNextOperation(TierTransferManager& manager, std::uint64_t value) { manager.next_op_id_ = value; }
};

}  // namespace tokenspeed

namespace tokenspeed::test {

TEST(CacheRestoreIdentityTest, WireIdRolloverPreservesGenerationAndTicketOwnership) {
    BlockPool device_pool{8, {1}};
    BlockPool host_pool{8, {1}};
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, device_pool, false, &host_pool, nullptr, false, false, 0);
    ReqPoolAllocator request_slots(2);
    SnapshotSlotAllocator blob_slots(2);
    TierTransferManager manager(coordinator);
    TierTransferManagerTestAccess::SetNextOperation(manager, std::numeric_limits<std::uint32_t>::max());
    auto first_slot = request_slots.Allocate();
    first_slot.BindOwner("a", 11);
    const auto first =
        manager.StartSnapshotRestore("a", std::move(first_slot), DecodeSlotIndex{},
                                     std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), {}, {}, 0);
    auto second_slot = request_slots.Allocate();
    second_slot.BindOwner("b", 22);
    const auto second =
        manager.StartSnapshotRestore("b", std::move(second_slot), DecodeSlotIndex{},
                                     std::make_shared<SnapshotSlotIndex>(blob_slots.Allocate()), {}, {}, 0);
    EXPECT_EQ(first.op_id, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(first.telemetry.op_generation, 0u);
    EXPECT_EQ(second.op_id, 0u);
    EXPECT_EQ(second.telemetry.op_generation, 1u);
    EXPECT_EQ(first.telemetry.slot_generation, 1u);
    EXPECT_EQ(second.telemetry.slot_generation, 1u);
    EXPECT_EQ(request_slots.AvailableSlots(), 0);
    {
        auto released = manager.CompleteSnapshotRestore(first.op_id, false);
        EXPECT_TRUE(released.request_pool_index.valid());
        EXPECT_EQ(released.request_pool_index.generation(), first.telemetry.slot_generation);
        EXPECT_EQ(request_slots.AvailableSlots(), 0);
    }
    EXPECT_EQ(request_slots.AvailableSlots(), 1);
    manager.CompleteSnapshotRestore(second.op_id, false);
    EXPECT_EQ(request_slots.AvailableSlots(), 2);
    EXPECT_EQ(blob_slots.AvailableSlots(), 2);
    EXPECT_FALSE(manager.CompleteSnapshotRestore(first.op_id, false).request_pool_index.valid());
}

TEST(CacheRestoreIdentityTest, PrefixLoadWireIdsAlsoCarryTheirOwnersWrapGeneration) {
    BlockPool device_pool{8, {1}};
    BlockPool host_pool{8, {1}};
    const std::vector<CacheGroupSpec> specs{
        CacheGroupSpec{.attn_kind = AttnKind::kFull, .cache_blocks_per_lcm_block = 1, .block_granularity = 2}};
    auto coordinator = MakeCoordinator(specs, 2, device_pool, false, &host_pool, nullptr, false, false, 0);
    const std::array<std::int32_t, 1> buckets{0};
    auto device = device_pool.AcquireBlocksInBuckets(0, buckets);
    auto host = host_pool.AcquireBlocksInBuckets(0, buckets);
    const CacheKey key{.group_id = 0, .content_hash = ContentHash{"prefix"}};
    coordinator.Host().CacheHostBlock(host.front(), key, std::nullopt);
    TierTransferManager manager(coordinator);
    TierTransferManagerTestAccess::SetNextOperation(manager, std::numeric_limits<std::uint32_t>::max());
    const BlockTransfer row{.group_id = 0, .source = host.front(), .destination = device.front(), .key = key};
    const auto first = manager.StartPrefixLoad({row});
    EXPECT_EQ(first.op_id, std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(first.telemetry.op_generation, 0u);
    manager.CompleteLoadBack(first.op_id);
    const auto second = manager.StartPrefixLoad({row});
    EXPECT_EQ(second.op_id, 0u);
    EXPECT_EQ(second.telemetry.op_generation, 1u);
    manager.CompleteLoadBack(second.op_id);
}

class CacheEventOwnershipTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.max_batch_size = 1;
        config.max_resident_requests = 1;
        config.disable_prefix_cache = true;
        config.disable_l2_cache = true;
        config.debug_force_retraction_interval = 3;
        SetTestSnapshotPool(config);
        return config;
    }
    void SetUp() override {
        SchedulerTestSuite::SetUp();
        scheduler_->SetCacheEventSink(sink_);
    }
    void TearDown() override { scheduler_.reset(); }

    void DriveToRestoring() {
        auto first = MakeRequestSpec("a", 2);
        first.admission_id = 11;
        Submit(first);
        PlanOnce();
        SendForwardDone("a", {42});
        AckImageStores(PlanOnce());
        SendForwardDone("a", {43});
        const auto retraction = PlanOnce();
        ASSERT_NE(FindSnapshotStore(retraction), nullptr);
        ASSERT_EQ(sink_->slots.size(), 2u);
        EXPECT_FALSE(sink_->slots.back().assigned);
        EXPECT_TRUE(sink_->restores.empty());

        // The old row really belongs to another admission while the image's
        // store is still pending. Its release lets the restore reuse that row.
        auto other = MakeRequestSpec("b", 1, 100);
        other.admission_id = 22;
        Submit(other);
        const auto admit_other = PlanOnce();
        ASSERT_NE(FindForwardBatch(admit_other), nullptr);
        ASSERT_EQ(FindForwardBatch(admit_other)->request_ids, std::vector<std::string>{"b"});
        SendForwardDone("b", {142});
        SendAbortEvent("b");
        const auto still_pending = PlanOnce();
        EXPECT_EQ(FindRestore(still_pending), nullptr);
        EXPECT_TRUE(sink_->restores.empty());
        AckImageStores(retraction);
        restore_plan_ = PlanOnce();
        ASSERT_NE(FindRestore(restore_plan_), nullptr);
        ASSERT_EQ(sink_->restores.size(), 1u);
    }

    std::shared_ptr<RecordingCacheEventSink> sink_{std::make_shared<RecordingCacheEventSink>()};
    ExecutionPlan restore_plan_;
};

TEST_F(CacheEventOwnershipTest, RestoreJoinsTheActualPriorAndNewOwnersAndFrozenFrontiers) {
    DriveToRestoring();
    ASSERT_EQ(sink_->restores.size(), 1u);
    const auto& op = sink_->restores.front();
    const auto& context = op.telemetry;
    EXPECT_EQ(context.admission_id, 11u);
    EXPECT_EQ(context.previous_slot, op.request_pool_index);
    EXPECT_EQ(context.previous_slot_generation, 1u);
    EXPECT_EQ(context.slot_generation, 3u);
    EXPECT_EQ(context.slot_previous_admission_id, 22u);
    EXPECT_EQ(context.restore_sequence, 1u);
    EXPECT_EQ(context.committed_tokens, 6);
    EXPECT_EQ(context.computed_tokens, 5);
    EXPECT_EQ(context.prompt_tokens, 4);
    EXPECT_EQ(context.op_generation, 0u);
    ASSERT_EQ(sink_->slots.size(), 5u);
    EXPECT_EQ(sink_->slots[2].admission_id, 22u);
    EXPECT_EQ(sink_->slots[2].generation, 2u);
    EXPECT_EQ(sink_->slots[2].previous_admission_id, 11u);
    EXPECT_EQ(FindRestore(restore_plan_)->telemetry.at(0), context);
    EXPECT_EQ(FindRestore(restore_plan_)->op_ids.at(0), op.op_id);
    EXPECT_EQ(scheduler_->RequestTokenSize("a"), 6);
    ASSERT_NE(FindForwardBatch(restore_plan_), nullptr);
    EXPECT_TRUE(FindForwardBatch(restore_plan_)->request_ids.empty());
    AckRestores(restore_plan_);
    EXPECT_EQ(sink_->slots.size(), 5u) << "ACK transfers the handle, not its ownership";
    const auto resumed = PlanOnce();
    ASSERT_NE(FindForwardBatch(resumed), nullptr);
    EXPECT_EQ(FindForwardBatch(resumed)->request_ids, std::vector<std::string>{"a"});
}

TEST_F(CacheEventOwnershipTest, CancellationCannotReleaseTheRestoringOwnerBeforeAck) {
    DriveToRestoring();
    ASSERT_EQ(sink_->restores.size(), 1u);
    const auto assigned = sink_->slots.back();
    SendAbortEvent("a");
    EXPECT_EQ(scheduler_->FreeRequestSlots(), 0);
    EXPECT_EQ(sink_->slots.size(), 5u);
    AckRestores(restore_plan_);
    ASSERT_EQ(sink_->slots.size(), 6u);
    EXPECT_FALSE(sink_->slots.back().assigned);
    EXPECT_EQ(sink_->slots.back().admission_id, assigned.admission_id);
    EXPECT_EQ(sink_->slots.back().generation, assigned.generation);
    EXPECT_EQ(scheduler_->FreeRequestSlots(), 1);
    AckRestores(restore_plan_);
    EXPECT_EQ(sink_->slots.size(), 6u) << "duplicate ACK cannot release twice";
}

TEST_F(CacheEventOwnershipTest, SchedulerKeepsTheSinkThroughPendingTicketDestruction) {
    auto retired = std::make_shared<std::vector<RecordedSlotOwnership>>();
    sink_ =
        std::shared_ptr<RecordingCacheEventSink>(new RecordingCacheEventSink, [retired](RecordingCacheEventSink* sink) {
            *retired = std::move(sink->slots);
            delete sink;
        });
    scheduler_->SetCacheEventSink(sink_);
    DriveToRestoring();
    SendAbortEvent("a");
    ASSERT_EQ(sink_->slots.size(), 5u);
    std::weak_ptr<RecordingCacheEventSink> weak = sink_;
    sink_.reset();
    EXPECT_FALSE(weak.expired()) << "the caller no longer has to keep the sink alive";
    scheduler_.reset();
    EXPECT_TRUE(weak.expired());
    ASSERT_EQ(retired->size(), 6u);
    EXPECT_FALSE(retired->back().assigned);
    EXPECT_EQ(retired->back().admission_id, 11u);
    EXPECT_EQ(retired->back().generation, 3u);
}

TEST_F(CacheEventOwnershipTest, EachSuccessfulRestoreAdvancesSequenceWithinTheSameAdmission) {
    DriveToRestoring();
    AckRestores(restore_plan_);
    PlanOnce();
    SendForwardDone("a", {44});
    PlanOnce();
    SendForwardDone("a", {45});
    const auto retraction = PlanOnce();
    ASSERT_NE(FindSnapshotStore(retraction), nullptr);
    AckImageStores(retraction);
    const auto restored = PlanOnce();
    ASSERT_NE(FindRestore(restored), nullptr);
    ASSERT_EQ(sink_->restores.size(), 2u);
    const auto& context = sink_->restores.back().telemetry;
    EXPECT_EQ(context.admission_id, 11u);
    EXPECT_EQ(context.restore_sequence, 2u);
    EXPECT_EQ(context.previous_slot_generation, 3u);
    EXPECT_EQ(context.slot_generation, 4u);
    EXPECT_EQ(context.slot_previous_admission_id, 11u);
    EXPECT_EQ(context.committed_tokens, 8);
    EXPECT_EQ(context.computed_tokens, 7);
}

TEST_F(CacheEventOwnershipTest, AReusedExternalRequestIdHasANewAdmissionOwner) {
    auto first = MakeRequestSpec("reused", 1);
    first.admission_id = 11;
    Submit(first);
    PlanOnce();
    SendForwardDone("reused", {42});
    SendAbortEvent("reused");
    PlanOnce();
    first.admission_id = 22;
    Submit(first);
    PlanOnce();
    ASSERT_EQ(sink_->slots.size(), 3u);
    EXPECT_EQ(sink_->slots.back().request_id, "reused");
    EXPECT_EQ(sink_->slots.back().admission_id, 22u);
    EXPECT_EQ(sink_->slots.back().previous_admission_id, 11u);
    EXPECT_EQ(sink_->slots.back().generation, 2u);
}

TEST_F(CacheEventOwnershipTest, EnablingObservationRequiresTheExistingAdmissionIdentity) {
    auto request = MakeRequestSpec("a", 1);
    EXPECT_THROW(Submit(request), std::invalid_argument);
    EXPECT_TRUE(sink_->slots.empty());
    request.admission_id = 11;
    Submit(request);
    EXPECT_THROW(scheduler_->SetCacheEventSink(nullptr), std::logic_error);
}

TEST_F(CacheEventOwnershipTest, ObservationDoesNotChangePlansOrResourceCounts) {
    Scheduler unobserved(Config());
    const auto options = Scheduler::PlanOptions{.allow_prefill = true, .apply_prefill_watermark = false};
    auto request = MakeRequestSpec("a", 2);
    request.admission_id = 11;
    Submit(request);
    unobserved.SubmitRequests({request});
    for (int round = 0; round < 5; ++round) {
        const auto observed_plan = PlanOnce();
        EXPECT_EQ(observed_plan, unobserved.NextExecutionPlan(options));
        EXPECT_EQ(scheduler_->FreeRequestSlots(), unobserved.FreeRequestSlots());
        ExecutionEvent event;
        if (const auto* batch = FindForwardBatch(observed_plan); batch && !batch->request_ids.empty()) {
            event.With(forward::ExtendResult{.request_id = "a", .tokens = {42 + round}});
        }
        if (const auto* store = FindSnapshotStore(observed_plan)) {
            for (auto id : store->op_ids) event.With(cache::SnapshotDone{.op_id = id});
        }
        if (const auto* restore = FindRestore(observed_plan)) {
            for (auto id : restore->op_ids) event.With(cache::RestoreDone{.op_id = id});
        }
        scheduler_->Advance(event);
        unobserved.Advance(std::move(event));
    }
    EXPECT_EQ(sink_->restores.size(), 1u);
}

class CachePrefixOwnershipTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.max_batch_size = 1;
        config.max_resident_requests = 1;
        SetTestSnapshotPool(config);
        return config;
    }
    void SetUp() override {
        SchedulerTestSuite::SetUp();
        scheduler_->SetCacheEventSink(sink_);
    }
    std::shared_ptr<RecordingCacheEventSink> sink_{std::make_shared<RecordingCacheEventSink>()};
};

TEST_F(CachePrefixOwnershipTest, PrefixAttributionSurvivesCancellationAndRequestIdReuse) {
    auto request = MakeRequestSpec("reused", 4);
    request.admission_id = 11;
    Submit(request);
    AckWriteBacks(PlanOnce());
    SendForwardDone("reused", {42});
    SendFinish("reused");
    AckWriteBacks(PlanOnce());
    ASSERT_TRUE(scheduler_->ClearL1Cache());

    request.admission_id = 22;
    Submit(request);
    const auto load_plan = PlanOnce();
    const auto loads = ExtractCacheOpsOfKind<LoadBackBatch>(load_plan);
    ASSERT_EQ(loads.size(), 1u);
    const auto& batch = std::get<LoadBackBatch>(loads.front());
    ASSERT_EQ(batch.op_ids.size(), 1u);
    EXPECT_EQ(batch.request_ids, std::vector<std::string>{"reused"});
    ASSERT_EQ(sink_->prefix_loads.size(), 1u);
    const auto scheduled = sink_->prefix_loads.front();
    EXPECT_EQ(scheduled.op_id, batch.op_ids.front());
    EXPECT_EQ(scheduled.telemetry, batch.telemetry.front());
    EXPECT_EQ(scheduled.telemetry.admission_id, 22u);
    EXPECT_EQ(scheduled.telemetry.slot_generation, 2u);
    EXPECT_EQ(scheduled.telemetry.op_generation, 0u);
    EXPECT_EQ(scheduled.telemetry.request_pool_index, FindForwardBatch(load_plan)->request_pool_indices.front());
    EXPECT_FALSE(scheduled.transfers.empty());

    SendAbortEvent("reused");
    DrainForwardResults("reused");
    PlanOnce();
    auto successor = MakeRequestSpec("reused", 1, 100);
    successor.admission_id = 33;
    Submit(successor);
    PlanOnce();
    EXPECT_EQ(sink_->slots.back().admission_id, 33u);
    EXPECT_EQ(sink_->slots.back().generation, 3u);
    EXPECT_EQ(batch.telemetry.front().admission_id, 22u);
    EXPECT_EQ(batch.telemetry.front().slot_generation, 2u);
    SendLoadBackDone(scheduled.op_id);
    EXPECT_EQ(scheduler_->FreeRequestSlots(), 0) << "a late copy ACK does not release the successor's row";
}

class CacheEventMidPrefillTest : public CacheEventOwnershipTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = CacheEventOwnershipTest::MakeConfig();
        config.max_scheduled_tokens = 4;
        config.debug_force_retraction_interval = -2;
        return config;
    }
};

TEST_F(CacheEventMidPrefillTest, FullPromptCommitmentIsDistinctFromTheComputedChunkFrontier) {
    auto request = MakeRequestSpec("prompt", 6);
    request.admission_id = 11;
    Submit(request);
    const auto prefill = PlanOnce();
    ASSERT_NE(FindForwardBatch(prefill), nullptr);
    SendForwardDone("prompt");
    const auto retraction = PlanOnce();
    ASSERT_NE(FindSnapshotStore(retraction), nullptr);
    AckImageStores(retraction);
    const auto restored = PlanOnce();
    ASSERT_NE(FindRestore(restored), nullptr);
    ASSERT_EQ(sink_->restores.size(), 1u);
    const auto& context = sink_->restores.back().telemetry;
    EXPECT_EQ(context.prompt_tokens, 12);
    EXPECT_EQ(context.committed_tokens, 12);
    EXPECT_EQ(context.computed_tokens, 4);
    EXPECT_EQ(context.previous_slot_generation, 1u);
    EXPECT_EQ(context.slot_generation, 2u);
}

}  // namespace tokenspeed::test
