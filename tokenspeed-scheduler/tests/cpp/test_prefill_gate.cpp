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
        return scheduler_->NextExecutionPlan(
            Scheduler::PlanOptions{.allow_prefill = allow_prefill, .apply_prefill_watermark = true});
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

class PrefillWatermarkSuite : public PrefillGateSuite, public ::testing::WithParamInterface<bool> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillGateSuite::MakeConfig();
        cfg.enable_mixed_prefill_decode = GetParam();
        return cfg;
    }
};

TEST_P(PrefillWatermarkSuite, ResidentReservationCannotBlockItsOwnChunks) {
    auto spec = MakeRequestSpec("long", /*num_pages=*/28);
    spec.max_new_tokens = 2;
    Submit(spec);
    ASSERT_TRUE(scheduler_->ProbePrefillGate().wants_prefill);
    auto first = Batch(Plan(true));
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->input_lengths, std::vector<std::int32_t>{8});
    SendForwardDone("long");
    // 56 prompt tokens plus two decode slots occupy 29 of 31 usable pages.
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 29);
    EXPECT_FALSE(scheduler_->ProbePrefillGate().has_decode_work);
    EXPECT_FALSE(InBatch(Batch(Plan(false)), "long"));

    for (std::int32_t prefix = 8; prefix < 56; prefix += 8) {
        SCOPED_TRACE(prefix);
        const auto probe = scheduler_->ProbePrefillGate();
        ASSERT_TRUE(probe.wants_prefill);
        auto chunk = Batch(Plan(probe.wants_prefill));
        ASSERT_TRUE(chunk.has_value());
        EXPECT_EQ(chunk->request_ids, std::vector<std::string>{"long"});
        EXPECT_EQ(chunk->extend_prefix_lens, std::vector<std::int32_t>{prefix});
        EXPECT_EQ(chunk->input_lengths, std::vector<std::int32_t>{8});
        SendForwardDone("long", prefix == 48 ? std::vector<std::int32_t>{42} : std::vector<std::int32_t>{});
    }
    EXPECT_FALSE(scheduler_->ProbePrefillGate().wants_prefill);
    EXPECT_TRUE(scheduler_->ProbePrefillGate().has_decode_work);
    auto decode = Batch(Plan(false));
    ASSERT_TRUE(decode.has_value());
    EXPECT_EQ(decode->NumExtends(), 0u);
    SendForwardDone("long", {43});
    SendFinish("long");
    AckWriteBacks(Plan(false));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 0);
}

TEST_P(PrefillWatermarkSuite, ResidentFinalChunkDoesNotAdmitNewPromptsAboveWatermark) {
    auto spec = MakeRequestSpec("long", /*num_pages=*/26);
    spec.max_new_tokens = 4;
    auto queued = MakeRequestSpec("queued", /*num_pages=*/1, /*start=*/100);
    queued.max_new_tokens = 1;
    Submit({spec, queued});
    for (std::int32_t prefix = 0; prefix < 52; prefix += 8) {
        SCOPED_TRACE(prefix);
        ASSERT_TRUE(scheduler_->ProbePrefillGate().wants_prefill);
        auto plan = Plan(true);
        auto batch = Batch(plan);
        ASSERT_TRUE(batch.has_value());
        EXPECT_EQ(batch->request_ids, std::vector<std::string>{"long"});
        EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{prefix});
        EXPECT_EQ(batch->input_lengths, std::vector<std::int32_t>{std::min(52 - prefix, 8)});
        EXPECT_TRUE(plan.aborts.empty());
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
        SendForwardDone("long", prefix == 48 ? std::vector<std::int32_t>{42} : std::vector<std::int32_t>{});
    }
    // The final chunk leaves budget for "queued", and its two-page demand
    // fits the three free pages, but 28/31 is already above the watermark.
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 28);
    EXPECT_FALSE(scheduler_->ProbePrefillGate().wants_prefill);
    auto decode = Batch(Plan(true));  // A peer can open the group's window.
    ASSERT_TRUE(decode.has_value());
    EXPECT_EQ(decode->request_ids, std::vector<std::string>{"long"});
    EXPECT_EQ(decode->NumExtends(), 0u);
    SendForwardDone("long", {43});
    SendFinish("long");
    AckWriteBacks(Plan(false));
    EXPECT_TRUE(scheduler_->ProbePrefillGate().wants_prefill);
    EXPECT_TRUE(InBatch(Batch(Plan(true)), "queued"));
}

TEST_P(PrefillWatermarkSuite, WatermarkIsInclusiveAndRecheckedBetweenNewPrompts) {
    auto cfg = Config();
    cfg.device_allocator.total_pages = 31;
    cfg.cache_groups[0].total_pages = 31;
    SetTestSnapshotPool(cfg);
    scheduler_ = std::make_unique<Scheduler>(cfg);
    auto spec = MakeRequestSpec("long", /*num_pages=*/25);
    spec.max_new_tokens = 4;
    Submit(spec);
    for (std::int32_t prefix = 0; prefix < 50; prefix += 8) {
        ASSERT_TRUE(InBatch(Batch(Plan(true)), "long"));
        SendForwardDone("long", prefix == 48 ? std::vector<std::int32_t>{42} : std::vector<std::int32_t>{});
    }
    ASSERT_EQ(scheduler_->ActiveLcmBlocks(), 27);  // Exactly 90% of 30 usable pages.
    auto first = MakeRequestSpec("first", /*num_pages=*/1, /*start=*/100);
    auto second = MakeRequestSpec("second", /*num_pages=*/1, /*start=*/200);
    first.max_new_tokens = second.max_new_tokens = 1;
    Submit({first, second});
    EXPECT_TRUE(scheduler_->ProbePrefillGate().wants_prefill);
    auto batch = Batch(Plan(true));
    EXPECT_TRUE(InBatch(batch, "first"));
    EXPECT_FALSE(InBatch(batch, "second"));
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 29);
}

TEST_P(PrefillWatermarkSuite, DisabledDelayAdmitsNewPromptsAboveWatermark) {
    auto spec = MakeRequestSpec("long", /*num_pages=*/26);
    spec.max_new_tokens = 4;
    auto queued = MakeRequestSpec("queued", /*num_pages=*/1, /*start=*/100);
    queued.max_new_tokens = 1;
    Submit({spec, queued});
    for (std::int32_t prefix = 0; prefix < 48; prefix += 8) {
        ASSERT_TRUE(InBatch(Batch(Plan(true)), "long"));
        SendForwardDone("long");
    }
    EXPECT_EQ(scheduler_->ActiveLcmBlocks(), 28);
    // Ordinary plans can batch the finishing chunk and the fresh prompt.
    auto batch = Batch(PlanOnce());
    EXPECT_TRUE(InBatch(batch, "long"));
    EXPECT_TRUE(InBatch(batch, "queued"));
}

TEST_P(PrefillWatermarkSuite, CapacityRetryDoesNotBypassThePrefillGate) {
    auto cfg = Config();
    cfg.device_allocator.total_pages = 8;
    cfg.cache_groups[0].total_pages = 8;
    cfg.disable_l2_cache = true;
    SetTestSnapshotPool(cfg);
    scheduler_ = std::make_unique<Scheduler>(cfg);
    Submit(MakeRequestSpec("decode", /*num_pages=*/1));  // No declared generation reserve.
    ASSERT_TRUE(InBatch(Batch(Plan(true)), "decode"));
    SendForwardDone("decode", {42});

    // Fill the pool until the decode itself needs a page that cannot fit.
    bool blocked = false;
    for (std::int32_t step = 0; step < 16; ++step) {
        auto plan = Plan(false);
        if (!InBatch(Batch(plan), "decode")) {
            blocked = true;
            break;
        }
        SendForwardDone("decode", {1000 + step});
    }
    ASSERT_TRUE(blocked);
    ASSERT_EQ(scheduler_->ActiveLcmBlocks(), 7);
    auto queued = MakeRequestSpec("queued", /*num_pages=*/1, /*start=*/100);
    queued.max_new_tokens = 1;
    Submit(queued);
    for (const bool allow_prefill : {false, true}) {
        auto plan = Plan(allow_prefill);
        EXPECT_FALSE(InBatch(Batch(plan), "queued"));
        EXPECT_TRUE(plan.aborts.empty());
        EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    }
    // The ordinary recovery policy can retract the grower for the newcomer;
    // neither a closed window nor a pressure-held admission may use it.
    auto plan = PlanOnce();
    EXPECT_TRUE(InBatch(Batch(plan), "queued"));
    EXPECT_EQ(scheduler_->RetractedSize(), 1u);
}

INSTANTIATE_TEST_SUITE_P(MixedAndSeparate, PrefillWatermarkSuite, ::testing::Bool());

}  // namespace
}  // namespace tokenspeed::test
