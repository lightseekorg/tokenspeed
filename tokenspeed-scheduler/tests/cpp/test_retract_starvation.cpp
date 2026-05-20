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

#include "integration_test_helper.h"

namespace tokenspeed::test {

// Retracted resource accounting: the device tail page must be released to
// the pool at retract time. Otherwise N retractions accumulate N permanently
// pinned device pages and admission deadlocks under sustained pressure.

class RetractLeakTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // decode_input_tokens > 0 forces LocalKVAllocator to hold a real
        // tail page (one extra page beyond the radix-tree-inserted full
        // pages) — the leak target.
        cfg.decode_input_tokens = 2;
        cfg.page_size = 2;
        // total=6 → 5 usable pages. Need headroom for post-recovery decode:
        // recovery consumes loadback + partial_tail + decode_reserve, and
        // the PrefillDone → Decoding transition needs another reserve page.
        // Tighter pools would cascade into a second retract during recovery
        // and mask the test's intent.
        cfg.device_allocator.total_pages = 6;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        cfg.enable_mixed_prefill_decode = false;
        return cfg;
    }

    void SendReserveNumTokens(const std::string& id, std::int32_t n) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = id,
            .reserve_num_tokens_in_next_schedule_event = n,
        }});
        scheduler_->Advance(std::move(event));
    }

    static const FlatWriteBackOperation* GetWriteBack(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* cop = std::get_if<CacheOperation>(&op)) {
                if (auto* wb = std::get_if<FlatWriteBackOperation>(cop)) {
                    return wb;
                }
            }
        }
        return nullptr;
    }

    static const FlatForwardOperation* GetForward(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
                return f;
            }
        }
        return nullptr;
    }
};

// A single retract releases the tail page back to the pool. Pre-fix this
// would be a no-op (tail page stayed inside Retracted::local_kv_allocator_).
TEST_F(RetractLeakTestSuite, RetractReleasesTailPageToDevicePool) {
    // Bring r1 to a known Decoding state.
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);
    const std::size_t avail_decoding = scheduler_->AvailableKvPages();

    // Force retract.
    SendReserveNumTokens("r1", 5);
    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    // Full pages move into the radix tree (device-occupied but evictable, not
    // counted in AvailableKvPages which only tracks the free list). The
    // LocalKVAllocator's leftover — partial tail + Acquire'd decode-reserve
    // pages — must come back to the free list.
    EXPECT_GT(scheduler_->AvailableKvPages(), avail_decoding)
        << "retract should release the tail and decode-reserve pages to the pool";
}

// Invariant: across N retracts (none recovered), AvailableKvPages must not
// monotonically shrink — i.e. no per-retract device-page leak.
TEST_F(RetractLeakTestSuite, ManyRetractsDoNotLeakDevicePages) {
    const std::size_t avail_initial = scheduler_->AvailableKvPages();

    constexpr int kNumRetracts = 5;
    for (int i = 0; i < kNumRetracts; ++i) {
        const std::string id = "r_" + std::to_string(i);
        const token_t start = static_cast<token_t>(10'000 + 1'000 * i);
        Submit(MakeRequestSpec(id, /*num_pages=*/1, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
        SendReserveNumTokens(id, 5);
        auto plan = PlanOnce();
        const auto* wb = GetWriteBack(plan);
        ASSERT_NE(wb, nullptr) << "device pool deadlocked at iter " << i
                               << "; AvailableKvPages=" << scheduler_->AvailableKvPages();
        ASSERT_FALSE(wb->op_ids.empty());
        SendWriteBackDone(wb->op_ids[0]);
    }

    EXPECT_EQ(scheduler_->RetractedSize(), kNumRetracts);

    // Worst-case bound: no more than 1 page net consumed per retract (any
    // bigger regression = leak). Tree-evictable full pages don't count
    // against this — only pages stuck outside both the free list and the
    // evictable tree do.
    EXPECT_GE(scheduler_->AvailableKvPages() + kNumRetracts, avail_initial);
}

// Two-step recovery: Retracted → PrefillDone (PrefillOperation covering the
// partial tail) → Decoding (ScheduleDecodeEvent on the next plan). The
// partial-tail re-prefill needs PrefillOperation semantics because
// DecodeOperation only carries a single decode_input_id.
TEST_F(RetractLeakTestSuite, RecoveryAfterTailPageRelease) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));
    PlanOnce();
    SendForwardDone("r1", {42});
    PlanOnce();
    SendReserveNumTokens("r1", 5);

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr);
    bool found = false;
    for (const auto& id : fwd->request_ids) {
        if (id == "r1") { found = true; break; }
    }
    ASSERT_TRUE(found);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_EQ(static_cast<std::size_t>(fwd->num_extends()), 1u)
        << "recovery's first plan emits PrefillOperation (re-prefill chunk)";

    SendForwardDone("r1", {99});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

// Re-prefill window can extend past PrefillSize() — it covers generated
// tokens whose KV was dropped at retract. Verifies ComputeShiftedInputIds
// caps at Size(), not PrefillSize(); otherwise the drafter sees -1 sentinels
// at the partial-tail positions and spec acceptance collapses for one iter.
TEST_F(RetractLeakTestSuite, RecoveryShiftedInputIdsCoversGeneratedTail) {
    Submit(MakeRequestSpec("r1", /*num_pages=*/1, /*start=*/1));  // PrefillSize=2
    PlanOnce();
    SendForwardDone("r1", {42});  // Size=3 > PrefillSize=2
    PlanOnce();
    SendReserveNumTokens("r1", 5);

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(static_cast<std::size_t>(fwd->num_extends()), 1u);

    // partial_tail = {token 42 at position 2}; shifted at last position is
    // the canonical -1 (no next token yet).
    ASSERT_EQ(fwd->input_ids.size(), 1u);
    EXPECT_EQ(fwd->input_ids[0], 42);
    ASSERT_EQ(fwd->shifted_input_ids.size(), 1u);
    EXPECT_EQ(fwd->shifted_input_ids[0], -1);

    SendForwardDone("r1", {99});
    PlanOnce();
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

}  // namespace tokenspeed::test
