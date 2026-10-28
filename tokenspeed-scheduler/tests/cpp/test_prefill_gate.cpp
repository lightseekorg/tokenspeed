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

// The prefill-delay gate's scheduler half (docs/design/prefill-delay.md):
// ProbePrefillGate's answers, and PlanOptions::allow_prefill holding the fused
// role's prefill work while decode continues.

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "integration_test_helper.h"

namespace tokenspeed::test {
namespace {

class PrefillGateSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        // Decode and prefill share a plan, so "held" and "scheduled" are
        // distinguishable within one round. A small token budget leaves a
        // prompt resident across rounds, which the gate must hold too.
        cfg.enable_mixed_prefill_decode = true;
        cfg.max_scheduled_tokens = 8;
        return cfg;
    }

    ExecutionPlan Plan(bool allow_prefill) {
        return scheduler_->NextExecutionPlan(Scheduler::PlanOptions{.allow_prefill = allow_prefill});
    }

    static std::optional<ForwardBatch> Batch(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (const auto* batch = std::get_if<ForwardBatch>(&op)) {
                return *batch;
            }
        }
        return std::nullopt;
    }

    static bool InBatch(const std::optional<ForwardBatch>& batch, const std::string& request_id) {
        if (!batch) {
            return false;
        }
        for (const auto& id : batch->request_ids) {
            if (id == request_id) {
                return true;
            }
        }
        return false;
    }
};

TEST_F(PrefillGateSuite, ProbeReportsTheRanksOwnWork) {
    // A submitted prompt, nothing decoding.
    Submit(MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1));
    Scheduler::PrefillGateProbe probe = scheduler_->ProbePrefillGate();
    EXPECT_TRUE(probe.wants_prefill);
    EXPECT_FALSE(probe.has_decode_work);

    // Its prompt lands: the request owes a first decode, nothing to prefill.
    Plan(true);
    SendForwardDone("r0", {42});
    probe = scheduler_->ProbePrefillGate();
    EXPECT_FALSE(probe.wants_prefill);
    EXPECT_TRUE(probe.has_decode_work);

    // A second prompt arrives beside the decode: the rank wants both.
    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/100));
    probe = scheduler_->ProbePrefillGate();
    EXPECT_TRUE(probe.wants_prefill);
    EXPECT_TRUE(probe.has_decode_work);
}

TEST_F(PrefillGateSuite, ClosedGateKeepsDecodingAndHoldsPrefill) {
    Submit(MakeRequestSpec("r0", /*num_pages=*/2, /*start=*/1));
    Plan(true);
    SendForwardDone("r0", {42});

    Submit(MakeRequestSpec("r1", /*num_pages=*/2, /*start=*/100));

    // A closed gate schedules the running decode and nothing else: the prompt
    // is neither admitted nor allowed to displace the decode batch.
    ExecutionPlan held = Plan(false);
    std::optional<ForwardBatch> held_batch = Batch(held);
    EXPECT_TRUE(InBatch(held_batch, "r0"));
    EXPECT_FALSE(InBatch(held_batch, "r1"));

    // The next open round admits it.
    ExecutionPlan open = Plan(true);
    std::optional<ForwardBatch> open_batch = Batch(open);
    EXPECT_TRUE(InBatch(open_batch, "r1"));
    ASSERT_TRUE(open_batch.has_value());
    EXPECT_EQ(open_batch->NumExtends(), 1u);
}

TEST_F(PrefillGateSuite, ClosedGateHoldsAResidentChunkToo) {
    // A prompt longer than the token budget (8) leaves a resident Prefilling
    // request after its first chunk; the gate holds its next chunk as well,
    // so a long prompt's chunks cluster instead of spreading over rounds.
    Submit(MakeRequestSpec("long", /*num_pages=*/6, /*start=*/1));
    ExecutionPlan first = Plan(true);
    ASSERT_TRUE(Batch(first).has_value());
    EXPECT_EQ(Batch(first)->NumExtends(), 1u);

    ExecutionPlan held = Plan(false);
    EXPECT_FALSE(InBatch(Batch(held), "long"));

    ExecutionPlan resumed = Plan(true);
    EXPECT_TRUE(InBatch(Batch(resumed), "long"));
}

}  // namespace
}  // namespace tokenspeed::test
