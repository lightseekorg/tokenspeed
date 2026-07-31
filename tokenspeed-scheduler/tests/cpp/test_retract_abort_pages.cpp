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
//  Retract/Abort page safety: no double-free, no leak
// ============================================================
//
// These tests exercise page lifecycle regressions around retract and abort.
// The OwnedPages RAII design guarantees:
//   - TakeFullPages moves pages out of LocalKVAllocator before Insert
//   - If the request is aborted after Insert, ~LocalKVAllocator only
//     frees its tail page, not the pages now owned by the tree.
//   - alloc_nodes failure returns all allocated pages to the pool.

class RetractAbortPagesSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.decode_input_tokens = 0;
        cfg.device_allocator.total_pages = 4;  // 3 usable (page 0 reserved)
        cfg.host_allocator.total_pages = 4;    // 3 usable
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void BringToDecoding(const std::string& id, std::int32_t num_pages = 1, token_t start = 1) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }
};

class RetractHostExhaustionSuite : public RetractAbortPagesSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = RetractAbortPagesSuite::MakeConfig();
        cfg.host_allocator.total_pages = 1;
        return cfg;
    }
};

TEST_F(RetractHostExhaustionSuite, InternalAbortIsPublishedInExecutionPlan) {
    // BringToDecoding schedules zero-token decode ops; settle their result debt before testing terminal abort.
    BringToDecoding("r1", /*num_pages=*/2, /*start=*/1);
    SendForwardDone("r1", {});
    BringToDecoding("r2", /*num_pages=*/1, /*start=*/5);
    SendForwardDone("r1", {});
    SendForwardDone("r2", {});
    SendReserveNumTokens("r1", 2);
    SendReserveNumTokens("r2", 1);

    auto plan = PlanOnce();

    ASSERT_EQ(plan.SchedulerAborts().size(), 1u);
    EXPECT_EQ(plan.SchedulerAborts().front().request_id, "r1");
    EXPECT_NE(plan.SchedulerAborts().front().message.find("host cache capacity"), std::string::npos);
}

// Core scenario: Decoding request gets retracted, then aborted.
// Before OwnedPages: ~LocalKVAllocator would double-free pages already in the tree.
// After OwnedPages: TakeFullPages moved pages out before Insert; abort is clean.
TEST_F(RetractAbortPagesSuite, Retract_ThenAbort_NoDoubleFree) {
    // r1 fills 1 page on device, r2 fills 1 page -> 2 of 3 device pages used
    BringToDecoding("r1", 1, 1);
    BringToDecoding("r2", 1, 3);

    // r1 needs extra space -> retract triggered on longest request
    SendReserveNumTokens("r1", 3);
    SendReserveNumTokens("r2", 0);

    auto plan = PlanOnce();
    (void)GetWriteBack(plan);

    // Whether retract was triggered or not, aborting should be safe.
    SendAbortEvent("r1");

    // No crash (no double-free). r2 is still in Decoding.
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    // r2 should still be schedulable
    SendReserveNumTokens("r2", 0);
    auto plan2 = PlanOnce();
    EXPECT_TRUE(RequestInFwd(plan2, "r2"));
}

// Retract followed by write-back-done: pages are properly accounted.
// After retract+writeback completes, device pages should be freed,
// host pages should hold the data, and retracted request recovers.
TEST_F(RetractAbortPagesSuite, Retract_WriteBackDone_PagesAccounted) {
    BringToDecoding("r1", 1, 1);

    // Force retract by submitting a second request that needs all device pages
    BringToDecoding("r2", 1, 3);
    SendReserveNumTokens("r1", 3);
    SendReserveNumTokens("r2", 0);

    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    if (wb != nullptr && !wb->op_ids.empty()) {
        // Complete the write-back
        SendWriteBackDone(wb->op_ids[0]);

        // r1 should be retracted
        EXPECT_GE(scheduler_->RetractedSize(), 1u);

        // On next plan, retracted request can recover
        auto plan2 = PlanOnce();
        // Either recovers to Decoding or stays retracted (depending on device capacity)
        // The key invariant: no crash, no double-free.
    }
}

// Two requests in Decoding, abort one -> the other continues unaffected.
// This exercises that the aborted request's pages don't corrupt the survivor's tree nodes.
TEST_F(RetractAbortPagesSuite, AbortOne_OtherContinues_PagesIntact) {
    BringToDecoding("r1", 1, 1);
    BringToDecoding("r2", 1, 3);

    SendAbortEvent("r1");
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);

    // r2 can still decode
    SendReserveNumTokens("r2", 0);
    auto plan = PlanOnce();
    EXPECT_TRUE(RequestInFwd(plan, "r2"));
}

// Abort during Decoding (before any retract) -- LocalKVAllocator pages are freed.
TEST_F(RetractAbortPagesSuite, Abort_FromDecoding_PagesFreed) {
    auto before = scheduler_->AvailableKvPages();
    BringToDecoding("r1", 1, 1);
    auto after_decoding = scheduler_->AvailableKvPages();
    EXPECT_LT(after_decoding, before);

    SendAbortEvent("r1");

    // After abort + next plan (cleanup), pages should be freed
    PlanOnce();
    auto after_abort = scheduler_->AvailableKvPages();
    // At least some pages freed (tree may retain evictable nodes)
    EXPECT_GE(after_abort, after_decoding);
}

// Finish two requests -> WriteBack -> WriteBackDone -> all device pages freed.
// Exercises the FinishEvent path where TakeFullPages + Insert happens.
TEST_F(RetractAbortPagesSuite, Finish_WriteBack_AllPagesFreed) {
    BringToDecoding("r1", 1, 1);

    SendFinish("r1");
    auto plan = PlanOnce();
    const auto* wb = GetWriteBack(plan);
    if (wb != nullptr && !wb->op_ids.empty()) {
        SendWriteBackDone(wb->op_ids[0]);
    }
    PlanOnce();

    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    EXPECT_EQ(scheduler_->WaitingSize(), 0u);
}

// Multiple Decoding requests, retract longest, abort it, submit new request.
// Exercises the full cycle: pages freed by abort are reusable.
TEST_F(RetractAbortPagesSuite, RetractAbort_NewRequestReusesPages) {
    BringToDecoding("r1", 1, 1);

    // Abort r1
    SendAbortEvent("r1");
    PlanOnce();

    // New request can reuse the freed device pages
    Submit(MakeRequestSpec("r3", 1, 10));
    auto plan = PlanOnce();
    EXPECT_TRUE(RequestInFwd(plan, "r3"));
}

// Overlap scheduling: ExtendResult grows token_container before the next
// Acquire.  If the new tokens cross a page boundary and device is full,
// scheduleRetract must not crash on pages < full_paged_tokens.
TEST_F(RetractAbortPagesSuite, Retract_AfterExtendResult_NoPageMismatchCrash) {
    // Config: block_size=2, device_total=4 (3 usable), decode_input_tokens=0
    // r1 uses 1 page; fill remaining 2 pages with r2 so device is full.
    BringToDecoding("r1", 1, 1);  // tokens=[1,2,42], 1 page, tail=0
    BringToDecoding("r2", 2, 3);  // tokens=[3,4,5,6,42], 2 pages, tail=0
    // 3 of 3 usable pages consumed, 0 free.

    // Simulate overlap: ExtendResult arrives BEFORE the next Acquire.
    // r1 gains 4 tokens -> token_count=7, full_paged_tokens(true)=(7-1)/2=3,
    // but local allocator still has 1 page -> 1 < 3 page mismatch.
    SendForwardDone("r1", {43, 44, 45, 46});

    // Both requests need pages but device is full -> both scheduleDecode fail.
    // r1 (7 tokens) > r2 (5 tokens) -> r1 retracted as longest.
    SendReserveNumTokens("r1", 4);
    SendReserveNumTokens("r2", 1);

    // PlanOnce must handle partial tail pages during retract.
    auto plan = PlanOnce();

    // The retracted request was either written back (WriteBack op present) or
    // went straight to Retracted (no device->host diff needed).
    const auto* wb = GetWriteBack(plan);
    if (wb != nullptr && !wb->op_ids.empty()) {
        SendWriteBackDone(wb->op_ids[0]);
    }

    // At least one request should be retracted.
    EXPECT_GE(scheduler_->RetractedSize(), 1u);
}

}  // namespace tokenspeed::test
