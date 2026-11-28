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

// Per-row execution slots (docs/design/scheduler.md, Execution slots): the
// decode permit on the D role, the request row elsewhere, -1 for a row the
// transfer peer runs; fresh on the request's first forward on the slot; and
// a retraction's state folds name the victim's slot before releasing it.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

std::vector<bool> Fresh(const ForwardBatch& batch) {
    return {batch.execution_slot_fresh.begin(), batch.execution_slot_fresh.end()};
}

TEST(ForwardBatchExecutionSlots, StayAlignedWithRequestRowsWhenPrefillRowsMoveFirst) {
    std::vector<ForwardOperation> ops;
    DecodeOperation decode;
    decode.request_id = "decode";
    decode.request_pool_index = 5;
    decode.input_length = 1;
    decode.execution_slot = 2;
    decode.execution_slot_fresh = false;
    ops.emplace_back(std::move(decode));
    PrefillOperation prefill;
    prefill.request_id = "prefill";
    prefill.request_pool_index = 3;
    prefill.execution_slot = 1;
    prefill.execution_slot_fresh = true;
    ops.emplace_back(std::move(prefill));

    const ForwardBatch batch{std::move(ops)};
    EXPECT_EQ(batch.request_ids, (std::vector<std::string>{"prefill", "decode"}));
    EXPECT_EQ(batch.request_pool_indices, (std::vector<std::int32_t>{3, 5}));
    EXPECT_EQ(batch.execution_slots, (std::vector<std::int32_t>{1, 2}));
    EXPECT_EQ(Fresh(batch), (std::vector<bool>{true, false}));
}

// The fused role: the request row is the execution slot.
class FusedExecutionSlotSuite : public SchedulerTestSuite {};

TEST_F(FusedExecutionSlotSuite, TheRequestRowIsTheSlotFreshOnlyOnTheFirstForward) {
    RequestSpec spec = MakeRequestSpec("r", /*num_pages=*/2);
    spec.max_new_tokens = 16;
    Submit(spec);
    const ExecutionPlan prefill = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(prefill);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"r"});
    const std::int32_t row = batch->request_pool_indices.at(0);
    EXPECT_GE(row, 1);
    EXPECT_EQ(batch->execution_slots, std::vector<std::int32_t>{row});
    EXPECT_EQ(Fresh(*batch), std::vector<bool>{true});

    for (const std::int32_t token : {42, 43}) {
        SendForwardDone("r", {token});
        const ExecutionPlan decode = PlanOnce();
        batch = FindForwardBatch(decode);
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->execution_slots, std::vector<std::int32_t>{row});
        EXPECT_EQ(Fresh(*batch), std::vector<bool>{false});
    }
}

// The D role with more resident requests than decode permits, and a State
// group that defers its decode state to the execution slot's workspace.
class DecodeExecutionSlotSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = SchedulerTestSuite::MakeConfig();
        cfg.role = Role::kD;
        cfg.max_batch_size = 2;
        cfg.max_resident_requests = 4;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "state",
            .block_granularity = 0,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kState,
            .transfer_policy = CacheTransferPolicy::LatestSnapshot,
            .deferred_decode_state = true,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    // Submits and remotely admits `ids` (one admission per plan); a landing
    // row holds no execution slot.
    void Admit(const std::vector<std::string>& ids) {
        std::int32_t start = 1;
        for (const std::string& id : ids) {
            RequestSpec spec = MakeRequestSpec(id, /*num_pages=*/1, start);
            spec.max_new_tokens = 16;
            start += 100;
            Submit(spec);
            scheduler_->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id}));
        }
        for (const std::string& id : ids) {
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* landing = FindRemoteAdmission(plan);
            ASSERT_NE(landing, nullptr);
            EXPECT_EQ(landing->request_ids, std::vector<std::string>{id});
            EXPECT_EQ(landing->execution_slots, std::vector<std::int32_t>{-1});
            EXPECT_EQ(Fresh(*landing), std::vector<bool>{false});
        }
    }

    void Land(const std::string& id) { scheduler_->Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{id, 42})); }

    // request id -> (execution slot, fresh) of the plan's batch.
    static std::map<std::string, std::pair<std::int32_t, bool>> Slots(const ExecutionPlan& plan) {
        std::map<std::string, std::pair<std::int32_t, bool>> slots;
        const ForwardBatch* batch = FindForwardBatch(plan);
        if (batch == nullptr) {
            return slots;
        }
        EXPECT_EQ(batch->execution_slots.size(), batch->request_ids.size());
        EXPECT_EQ(batch->execution_slot_fresh.size(), batch->request_ids.size());
        for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
            slots[batch->request_ids[row]] = {batch->execution_slots[row], batch->execution_slot_fresh[row]};
        }
        return slots;
    }
};

TEST_F(DecodeExecutionSlotSuite, PermitsAreDistinctStableAndFreshOnlyOnTheFirstDecode) {
    Admit({"a", "b", "c"});
    Land("b");
    Land("c");
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    // b and c hold request rows 2 and 3, past the two decode permits.
    std::vector<std::int32_t> rows = batch->request_pool_indices;
    std::ranges::sort(rows);
    EXPECT_EQ(rows, (std::vector<std::int32_t>{2, 3}));
    const auto slots = Slots(first);
    ASSERT_EQ(slots.size(), 2u);
    EXPECT_NE(slots.at("b").first, slots.at("c").first);
    for (const auto& [id, slot] : slots) {
        EXPECT_GE(slot.first, 1) << id;
        EXPECT_LE(slot.first, 2) << id;
        EXPECT_TRUE(slot.second) << id;
    }

    for (const std::int32_t token : {43, 44}) {
        SendForwardDone("b", {token});
        SendForwardDone("c", {token});
        const auto next = Slots(PlanOnce());
        ASSERT_EQ(next.size(), 2u);
        for (const auto& [id, slot] : next) {
            EXPECT_EQ(slot, std::make_pair(slots.at(id).first, false)) << id;
        }
    }

    // b's permit passes to a, fresh there, once b's last result landed.
    Land("a");
    SendForwardDone("b", {45});
    SendForwardDone("c", {45});
    SendFinish("b");
    const auto handed = Slots(PlanOnce());
    ASSERT_EQ(handed.size(), 2u);
    EXPECT_EQ(handed.at("a"), std::make_pair(slots.at("b").first, true));
    EXPECT_EQ(handed.at("c"), std::make_pair(slots.at("c").first, false));
}

// The resident admitted first holds request row 1 but never lands, so the
// victim's row (2) differs from its permit (1): the fold must name the permit.
TEST_F(DecodeExecutionSlotSuite, ARetractionFoldsFromTheVictimsPermitBeforeReleasingIt) {
    Admit({"waiting", "victim"});
    Land("victim");
    PlanOnce();
    SendForwardDone("victim", {43});
    const ExecutionPlan decode = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(decode);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, std::vector<std::string>{"victim"});
    const StateFold expected{
        .request_slot = batch->request_pool_indices.at(0),
        .page = batch->BlockTables().at("state").at(0).at(1),
        .execution_slot = batch->execution_slots.at(0),
    };
    EXPECT_EQ(expected.execution_slot, 1);
    EXPECT_EQ(expected.request_slot, 2);
    EXPECT_NE(expected.request_slot, expected.execution_slot);
    SendForwardDone("victim", {44});
    ASSERT_EQ(scheduler_->ActiveDecodeSlots(), 1);

    ExecutionPlan plan;
    SchedulerTestAccess::RetractVictim(*scheduler_, "victim", plan);
    EXPECT_EQ(plan.state_folds, (std::map<std::string, std::vector<StateFold>>{{"state", {expected}}}));
    EXPECT_EQ(scheduler_->ActiveDecodeSlots(), 0);
}

// The knob retracts the decoding victim at plan 5; its image is restored and
// it is retracted again, directly, before or after its first resumed forward.
// Without an L2 cache the image rides the snapshot pool whole and the restore
// claims no indexed block, so the Device cache can be cleared after it.
class DecodeRestoreFoldSuite : public DecodeExecutionSlotSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg = DecodeExecutionSlotSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        cfg.debug_force_retraction_interval = 5;
        cfg.debug_force_retraction_max_per_request = 1;
        return cfg;
    }

    static StateFold LiveFold(const ForwardBatch& batch) {
        return StateFold{
            .request_slot = batch.request_pool_indices.at(0),
            .page = batch.BlockTables().at("state").at(0).at(1),
            .execution_slot = batch.execution_slots.at(0),
        };
    }

    // Decodes the victim (plans 2-4), lets the knob retract it (plan 5) and
    // restores it (plan 6). Returns the folds the second retraction lists.
    std::map<std::string, std::vector<StateFold>> RetractAgainAfterRestore(bool resumed_forward_first,
                                                                           StateFold& resumed) {
        Admit({"victim"});  // plan 1
        Land("victim");
        const ExecutionPlan first = PlanOnce();  // 2
        const ForwardBatch* batch = FindForwardBatch(first);
        EXPECT_NE(batch, nullptr);
        if (batch == nullptr) {
            return {};
        }
        EXPECT_EQ(Fresh(*batch), std::vector<bool>{true});
        StateFold live{};
        for (const std::int32_t token : {43, 44}) {
            SendForwardDone("victim", {token});
            const ExecutionPlan decode = PlanOnce();  // 3, 4
            batch = FindForwardBatch(decode);
            EXPECT_NE(batch, nullptr);
            if (batch == nullptr) {
                return {};
            }
            live = LiveFold(*batch);
        }
        SendForwardDone("victim", {45});

        const ExecutionPlan retraction = PlanOnce();  // 5
        const ForwardBatch* idle = FindForwardBatch(retraction);
        EXPECT_TRUE(idle == nullptr || idle->request_ids.empty()) << "the knob keeps the victim out";
        EXPECT_EQ(scheduler_->RetractedSize(), 1u);
        EXPECT_GE(live.execution_slot, 1);
        EXPECT_EQ(retraction.state_folds, (std::map<std::string, std::vector<StateFold>>{{"state", {live}}}))
            << "the first retraction folds the records of its decodes";
        AckImageStores(retraction);
        const ExecutionPlan readmit = PlanOnce();  // 6
        EXPECT_NE(FindRestore(readmit), nullptr);
        AckRestores(readmit);
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
        EXPECT_EQ(scheduler_->ActiveDecodeSlots(), 1) << "the restore reinstalled a permit";
        // Drops the entries the first retraction published. Otherwise the
        // second retraction's publication swaps the live block for that equal
        // entry, and no fold is listed whatever the slot holds.
        EXPECT_TRUE(scheduler_->ClearL1Cache());

        if (resumed_forward_first) {
            const ExecutionPlan resume = PlanOnce();  // 7
            batch = FindForwardBatch(resume);
            EXPECT_NE(batch, nullptr);
            if (batch == nullptr) {
                return {};
            }
            EXPECT_EQ(batch->request_ids, std::vector<std::string>{"victim"});
            EXPECT_EQ(Fresh(*batch), std::vector<bool>{true});
            resumed = LiveFold(*batch);
            SendForwardDone("victim", {46});
        }

        ExecutionPlan plan;
        EXPECT_FALSE(SchedulerTestAccess::RetractVictim(*scheduler_, "victim", plan).has_value());
        EXPECT_EQ(scheduler_->RetractedSize(), 1u) << "the victim is retracted with its image either way";
        EXPECT_EQ(scheduler_->ActiveDecodeSlots(), 0);
        return plan.state_folds;
    }
};

// Restored but not yet forwarded: its live block holds the folded state, and
// its new permit's workspace holds no record of its own.
TEST_F(DecodeRestoreFoldSuite, ARestoredDecodeRetractedBeforeItsFirstForwardFoldsNothing) {
    StateFold unused{};
    EXPECT_TRUE(RetractAgainAfterRestore(/*resumed_forward_first=*/false, unused).empty());
}

TEST_F(DecodeRestoreFoldSuite, ARestoredDecodeFoldsAgainOnceItsFirstResumedForwardRan) {
    StateFold resumed{};
    const auto folds = RetractAgainAfterRestore(/*resumed_forward_first=*/true, resumed);
    EXPECT_GE(resumed.execution_slot, 1);
    EXPECT_EQ(folds, (std::map<std::string, std::vector<StateFold>>{{"state", {resumed}}}));
}

}  // namespace
}  // namespace tokenspeed::test
