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

class AbortTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void SendUpdateReserveNumTokens(const std::string& id, std::int32_t n) {
        ExecutionEvent event;
        event.With(ForwardEvent{forward::UpdateReserveNumTokens{
            .request_id = id,
            .reserve_num_tokens_in_next_schedule_event = n,
        }});
        scheduler_->Advance(std::move(event));
    }

    void BringToDecoding(const std::string& id, std::int32_t num_pages = 2, token_t start = 1) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }
};

TEST_F(AbortTestSuite, Abort_FromSubmitted) {
    Submit(MakeRequestSpec("r1", 2));
    SendAbortEvent("r1");
    auto plan = PlanOnce();
    EXPECT_FALSE(RequestInFwd(plan, "r1"));
}

TEST_F(AbortTestSuite, Abort_FromDecoding) {
    BringToDecoding("r1");
    SendAbortEvent("r1");
    EXPECT_EQ(scheduler_->DecodingSize(), 0u);
    auto plan = PlanOnce();
    EXPECT_FALSE(RequestInFwd(plan, "r1"));
}

TEST_F(AbortTestSuite, Abort_UnknownRequestDoesNotThrow) {
    EXPECT_NO_THROW(SendAbortEvent("nonexistent"));
}

TEST_F(AbortTestSuite, Abort_DoesNotAffectOtherRequests) {
    BringToDecoding("r1");
    BringToDecoding("r2", 2, 50);
    SendAbortEvent("r1");
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

// Regression: late forward::* events that arrive after the request transitioned
// to Finished (e.g. retract->host exhausted->AbortEvent race) are terminal
// no-ops, not invalid transitions.
//
// Real-world trigger: scheduler/operations/forward.cpp newRetractOperation
// falls back to fsm::AbortEvent when host buffer is exhausted, leaving the
// request Finished. In-flight forward events for that request may still
// arrive via overlap scheduling and must not hit FSM InvalidTransitionHandler.
//
// Coverage: all three forward::* outside events (ExtendResult, Finish,
// UpdateReserveNumTokens) have different accepted-state sets, but share the
// same async-race semantics.
TEST_F(AbortTestSuite, ExtendResult_AfterAbort_DroppedNotThrown) {
    BringToDecoding("r1");
    SendAbortEvent("r1");
    EXPECT_NO_THROW(SendForwardDone("r1", {99}));
}

TEST_F(AbortTestSuite, UpdateReserveNumTokens_AfterAbort_DroppedNotThrown) {
    BringToDecoding("r1");
    SendAbortEvent("r1");
    EXPECT_NO_THROW(SendUpdateReserveNumTokens("r1", 1));
}

TEST_F(AbortTestSuite, Finish_AfterAbort_DroppedNotThrown) {
    BringToDecoding("r1");
    SendAbortEvent("r1");
    EXPECT_NO_THROW(SendFinish("r1"));
}

TEST_F(AbortTestSuite, InvalidForwardEvent_BeforeTerminal_StillThrows) {
    Submit(MakeRequestSpec("r1", 2));
    EXPECT_THROW(SendUpdateReserveNumTokens("r1", 1), std::logic_error);
}

}  // namespace tokenspeed::test
