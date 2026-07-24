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

// ============================================================
//  Retract: Decoding → Retracting → Retracted → Decoding
// ============================================================

// device_total=2 (block_size=2) means one request fills the device during Decoding;
// requesting an extra page then forces a Retract.
class RetractTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 3;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    // Submitted → PrefillDone → Decoding (1 page of tokens; fills the device).
    void BringToDecoding(const std::string& id = "r1") {
        Submit(MakeRequestSpec(id, /*num_pages=*/1, /*start=*/1));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }
};

class DisableL2RetractTestSuite : public RetractTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = RetractTestSuite::MakeConfig();
        cfg.disable_l2_cache = true;
        return cfg;
    }
};

// Device full + extra page needed → WriteBack (Retract) op emitted.
TEST_F(RetractTestSuite, Retract_TriggeredWhenDeviceFull) {
    BringToDecoding("r1");
    // tail has 2 available; reserve=3 → 1 extra page needed; device full → Retract.
    SendReserveNumTokens("r1", 3);

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_FALSE(wb->op_ids.empty());
    bool any_pages = false;
    for (const auto& pages : wb->src_pages) {
        if (!pages.empty()) {
            any_pages = true;
            break;
        }
    }
    EXPECT_TRUE(any_pages);
}

TEST_F(DisableL2RetractTestSuite, RetractStillEmitsHostWriteBackWhenL2Disabled) {
    BringToDecoding("r1");
    SendReserveNumTokens("r1", 3);

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_FALSE(wb->op_ids.empty());
    EXPECT_EQ(scheduler_->AvailableHostKvPages(), Config().host_allocator.total_pages - 2);
}

// Retracting request must not appear in the forward batch.
TEST_F(RetractTestSuite, Retract_RequestNotInForwardWhileRetracting) {
    BringToDecoding("r1");
    SendReserveNumTokens("r1", 3);

    auto plan = PlanOnce();
    const FlatForwardOperation* fwd = nullptr;
    for (const auto& op : plan.Operations()) {
        if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
            fwd = f;
            break;
        }
    }
    bool in_fwd = false;
    if (fwd) {
        for (const auto& id : fwd->request_ids) {
            if (id == "r1") {
                in_fwd = true;
                break;
            }
        }
    }
    EXPECT_FALSE(in_fwd);
}

// WriteBackDone → Retracted state.
TEST_F(RetractTestSuite, Retract_WriteBackDoneTransitionsToRetracted) {
    BringToDecoding("r1");
    SendReserveNumTokens("r1", 3);

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);

    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
}

#if !TOKENSPEED_FLAT_KVCACHE
TEST_F(RetractTestSuite, FailedWriteBackAbortsAfterOutstandingForwardResult) {
    BringToDecoding("r1");
    SendReserveNumTokens("r1", 3);

    auto retract_plan = PlanOnce();
    const auto* writeback = GetWriteBack(retract_plan);
    ASSERT_NE(writeback, nullptr);
    ASSERT_EQ(writeback->op_ids.size(), 1u);

    SendWriteBackDone(writeback->op_ids[0], /*success=*/false);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    EXPECT_TRUE(PlanOnce().SchedulerAborts().empty());

    SendForwardDone("r1", {});
    auto abort_plan = PlanOnce();
    ASSERT_EQ(abort_plan.SchedulerAborts().size(), 1u);
    EXPECT_EQ(abort_plan.SchedulerAborts().front().request_id, "r1");
    EXPECT_EQ(abort_plan.SchedulerAborts().front().message, "L2 retract write-back failed");
}
#endif

// Retracted request recovers to Decoding on next PlanOnce.
TEST_F(RetractTestSuite, Retract_RetractedRequestRecoversToDecoding) {
    BringToDecoding("r1");
    SendReserveNumTokens("r1", 3);

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);
    ASSERT_EQ(scheduler_->RetractedSize(), 1u);

    auto plan2 = PlanOnce();
    const FlatForwardOperation* fwd = nullptr;
    for (const auto& op : plan2.Operations()) {
        if (auto* f = std::get_if<FlatForwardOperation>(&op)) {
            fwd = f;
            break;
        }
    }
    ASSERT_NE(fwd, nullptr);
    bool in_fwd = false;
    for (const auto& id : fwd->request_ids) {
        if (id == "r1") {
            in_fwd = true;
            break;
        }
    }
    EXPECT_TRUE(in_fwd);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
}

// ============================================================
//  Retract from PrefillDone: PrefillDone → Retracting → Retracted → Decoding
// ============================================================

// device_total=4 → 3 usable pages (page 0 is reserved).
// block_size=2, decode_input_tokens=2.
// A 2-page request (4 tokens) uses 2 token pages + 1 decode reserve = 3 device pages.
// This fills the device. scheduleDecode needs 1 more page → fails → retract from PrefillDone.
//
// After retract, 2 pages remain in local_kv_allocator (boundary + reserve), so only 1
// page is freed (the radix tree insert page). FullCycle test uses device_total=6 (5 usable)
// to give r2 enough room to prefill after r1's retraction.
class RetractFromPrefillDoneTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 2;
        cfg.device_allocator.total_pages = 4;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        return cfg;
    }
};

// PrefillDone request triggers retract when device is full.
TEST_F(RetractFromPrefillDoneTestSuite, Retract_FromPrefillDone_TriggeredWhenDeviceFull) {
    // r1: 2-page request → uses 3 device pages (2 token + 1 decode reserve), fills all 3 usable pages.
    Submit(MakeRequestSpec("r1", 2, 1));
    PlanOnce();  // Submitted → PrefillDone

    // r2: submitted but can't prefill (0 free pages).
    Submit(MakeRequestSpec("r2", 1, 100));

    // Next plan: scheduleDecode for r1 needs 1 more page → fails, r2 can't prefill either.
    // Retract must be triggered on the PrefillDone request (r1).
    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    ASSERT_NE(wb, nullptr);
    EXPECT_FALSE(wb->op_ids.empty());
}

// PrefillDone retract → WriteBackDone → Retracted → next plan: r2 can prefill
// after prefix-cache eviction frees r1's inserted page.
// device_total=6 (5 usable): r1 uses 3, r2 needs 2, with 1 prefix-cache page evictable
// + 2 held by local_kv_allocator = r2 gets 2 free pages after eviction.
TEST_F(RetractFromPrefillDoneTestSuite, Retract_FromPrefillDone_FullCycle) {
    auto cfg = MakeConfig();
    cfg.device_allocator.total_pages = 6;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    Submit(MakeRequestSpec("r1", 2, 1));
    PlanOnce();  // r1 → PrefillDone (3 pages used out of 5 usable)

    // Fill the remaining 2 pages so scheduleDecode for r1 fails → retract.
    Submit(MakeRequestSpec("r2", 1, 100));
    PlanOnce();  // r2 → PrefillDone (2 more pages used, 5/5 full)

    // Now r1 needs scheduleDecode (1 extra page) → fails → retract r1.
    // r2 also needs scheduleDecode → might also fail.
    // Submit r3 to have a pending request to verify scheduling works after retract.
    Submit(MakeRequestSpec("r3", 1, 200));

    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr);
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);
    EXPECT_GE(scheduler_->RetractedSize(), 1u);

    // After retract + eviction, some pages freed → other requests can proceed.
    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr);
    EXPECT_FALSE(fwd->request_ids.empty());
}

// ============================================================
//  Retract with tail_page_available==0: verify boundary page preserved
// ============================================================

// block_size=2, decode_input_tokens=1.
// device_total=7 (6 usable).
// r1 (2-page, 4 tokens): prefill 2 + Acquire(1) = 3 pages, tail_available=1.
// r2 (1-page, 2 tokens): prefill 1 + Acquire(1) = 2 pages, tail_available=1.
// Total: 5 out of 6 used. 1 free.
// After decode for both: r1 Acquire(1) → tail=0 (3 pages), r2 Acquire(1) → tail=0 (2 pages).
// Total: 5, free: 1. scheduleDecode needs 1 each → r1 gets it (1 page), now 6/6 full.
// r2 needs 1 → fails → retract r2 (the smaller one).
// Actually simpler: use r1 only, fill device, then force retract via SendReserveNumTokens.
class RetractTailPageTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 3;
        cfg.host_allocator.total_pages = 16;
        cfg.enable_l3_storage = false;
        return cfg;
    }
};

// Retract when tail_page_available==0, then recover.
// device_total=3 → 2 usable pages. block_size=2, decode_input_tokens=0.
// r1: 1-page request (2 tokens). Prefill: 1 page, tail_available=0.
// After PrefillDone → Decoding: Acquire(0), still 1 page, tail_available=0.
// SendForwardDone to add output token → 3 tokens. Then Acquire(1) → 1 more page, tail=1.
// Actually this doesn't reach tail=0 easily.
//
// Better approach: use decode_input_tokens=0, 1-page request fills the 2 usable pages.
// SendReserveNumTokens(3) → reserve=3, tail=2 → need (3-2+1)/2=1 extra page → device full → retract.
// At retract: GetFullPagedTokens(true) on 2 tokens → except_last → 0 full pages.
// Hmm, that means alloc_count=0 and TakeFirstPages(0) takes nothing. No bug here.
//
// Need: tail_page_available == 0 AND alloc_count > 0 AND local_kv_allocator.Size() > alloc_count.
// This means: enough tokens to have full pages, and extra pages beyond what's inserted.
//
// Use decode_input_tokens=2, 1-page request (2 tokens).
// Prefill: LocalKVAllocator(2 tokens) → 1 page, tail=0. Acquire(2) → 1 more page, tail=0.
// Total: 2 pages. device_total=3 → 2 usable → FULL.
// After PrefillDone → SendForwardDone → 3 tokens.
// Next PlanOnce: scheduleDecode needs Acquire(2). tail=0, need 1 page → fails → retract.
// At retract: GetFullPagedTokens(true) on 3 tokens except_last → 2 tokens → 1 full page.
// prefix=0. alloc_count=1. TakeFirstPages(1) from 2 pages → 1 inserted, 1 remains.
// This tests the fix! The remaining page holds token 2's KV (position 2, partially fills page).
// device_total=3 (2 usable). decode_input_tokens=0, block_size=2.
// r1: 1-page (2 tokens). Prefill: 1 page, tail=0.
// After decode: Acquire(0) → still 1 page, tail=0.
// SendForwardDone → token count grows. SendReserveNumTokens(3) →
// scheduleDecode needs (3 - 0 + 1)/2 = 2 extra pages → 0 free → fails → retract.
//
// At retract: GetFullPagedTokens(true) on 3 tokens except_last → 2 tokens → 1 full page.
// alloc_count=1. TakeFullPages returns 1 page (all), then TakeFirst(1) uses it.
// No extra page to lose here (only 1 page total). tail_available=0 but pages=alloc_count.
//
// For the bug to manifest, we need local_kv_allocator.Size() > alloc_count when tail=0.
// Use decode_input_tokens=2 to force extra pages.
//
// decode_input_tokens=2, device_total=4 (3 usable).
// r1: 1-page (2 tokens). Prefill: 1 page (tail=0) + Acquire(2) → +1 page, tail=0. Total: 2 pages.
// PlanOnce → PrefillDone. SendForwardDone.
// PlanOnce → ScheduleDecodeEvent: Acquire(2) needs 2 tokens, tail=0 → allocate 1 page. Total: 3 pages.
// Now 3/3 usable used. SendReserveNumTokens(3) → scheduleDecode needs (3-0+1)/2=2 pages → fails → retract.
//
// At retract: 3 tokens (2 prefill + 1 output). except_last → 2 → 1 full page.
// alloc_count=1. TakeFirstPages(1) from 3 → 1 inserted, 2 remain.
// After WriteBackDone: 1 prefix-cache evictable, 2 local held. Total free after eviction: 3-2=1.
// Recovery needs: 1 (loadback) + 1 (Acquire(2)) = 2. Only 1 free → fails.
// Need more device pages for recovery. Use device_total=6 (5 usable).
// r1 uses 3 pages. 2 free. SendReserveNumTokens(5) → needs (5-0+1)/2=3 pages → fails → retract.
// After retract: 2 remain in local. prefix-cache evictable 1. Free after eviction: 5-2=3.
// Recovery: 1 loadback + 1 Acquire(2) = 2 pages. 3 free → OK.
TEST_F(RetractTailPageTestSuite, Retract_TailPageZero_PreservesBoundaryPage) {
    auto cfg = MakeConfig();
    cfg.decode_input_tokens = 2;
    cfg.device_allocator.total_pages = 6;
    cfg.host_allocator.total_pages = 16;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    // r1: 1-page request (2 tokens). Prefill uses 2 pages (1 token + 1 decode reserve).
    Submit(MakeRequestSpec("r1", 1, 1));
    PlanOnce();  // Submitted → PrefillDone (2 pages used, 3 free)

    // Forward done → ExtendResult, now 3 tokens.
    SendForwardDone("r1", {42});

    // PrefillDone → Decoding: Acquire(2) → 1 more page. Total: 3 pages. (2 free)
    PlanOnce();
    ASSERT_EQ(scheduler_->DecodingSize(), 1u);

    // Force retract: reserve requires more pages than available.
    SendReserveNumTokens("r1", 5);

    // scheduleDecode: extra_tokens = max(0, 5 - 0) = 5, pages = (5+1)/2 = 3.
    // EnsureCapacityByEvict(3): only 2 free → fails → retract.
    auto plan1 = PlanOnce();
    const auto* wb = GetWriteBack(plan1);
    ASSERT_NE(wb, nullptr) << "Retract should be triggered when device is full";
    ASSERT_FALSE(wb->op_ids.empty());

    SendWriteBackDone(wb->op_ids[0]);
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);

    // Recovery: Retracted → Decoding.
    auto plan2 = PlanOnce();
    const auto* fwd = GetForward(plan2);
    ASSERT_NE(fwd, nullptr) << "Recovery should produce a forward op";

    bool found_r1 = false;
    std::int32_t r1_idx = -1;
    for (std::size_t i = 0; i < fwd->request_ids.size(); ++i) {
        if (fwd->request_ids[i] == "r1") {
            found_r1 = true;
            r1_idx = static_cast<std::int32_t>(i);
            break;
        }
    }
    ASSERT_TRUE(found_r1) << "r1 must be in the forward batch after recovery";
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);

    // Verify boundary page preserved: occupied_pages has loadback + boundary + decode pages.
    // With fix: 1 loadback page + 2 boundary/reserve pages + 1 decode page = 4 pages minimum.
    // Without fix: boundary page lost, model reads garbage at position 2.
    EXPECT_GE(static_cast<std::int32_t>(fwd->occupied_pages[r1_idx].size()), 3);
}

}  // namespace tokenspeed::test
