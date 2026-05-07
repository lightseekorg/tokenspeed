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

class ChunkedPrefillTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_scheduled_tokens = 4;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    static const FlatForwardOperation* GetForwardOp(const ExecutionPlan& plan) {
        for (const auto& op : plan.Operations()) {
            if (auto* f = std::get_if<FlatForwardOperation>(&op)) return f;
        }
        return nullptr;
    }
};

TEST_F(ChunkedPrefillTestSuite, ChunkedPrefill_SplitsAcrossPlans) {
    // page_size=2, max_scheduled_tokens=4 → each chunk handles 4 tokens (2 pages).
    // 8 tokens = 2 chunks.
    Submit(MakeRequestSpec("r1", 4));  // 4 pages = 8 tokens
    auto plan1 = PlanOnce();
    auto* fwd1 = GetForwardOp(plan1);
    ASSERT_NE(fwd1, nullptr);
    EXPECT_EQ(fwd1->input_lengths[0], 4);
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    auto plan2 = PlanOnce();
    auto* fwd2 = GetForwardOp(plan2);
    ASSERT_NE(fwd2, nullptr);
    EXPECT_EQ(fwd2->input_lengths[0], 4);
    EXPECT_EQ(scheduler_->PrefillSize(), 1u);

    // Third plan transitions to Decoding.
    auto plan3 = PlanOnce();
    auto* fwd3 = GetForwardOp(plan3);
    ASSERT_NE(fwd3, nullptr);
    EXPECT_EQ(scheduler_->DecodingSize(), 1u);
}

TEST_F(ChunkedPrefillTestSuite, ExtendPrefixLen_GrowsPerChunk) {
    Submit(MakeRequestSpec("r1", 4));  // 8 tokens
    auto plan1 = PlanOnce();
    auto* fwd1 = GetForwardOp(plan1);
    ASSERT_NE(fwd1, nullptr);
    EXPECT_EQ(fwd1->extend_prefix_lens[0], 0);

    auto plan2 = PlanOnce();
    auto* fwd2 = GetForwardOp(plan2);
    ASSERT_NE(fwd2, nullptr);
    EXPECT_EQ(fwd2->extend_prefix_lens[0], 4);
}

TEST_F(ChunkedPrefillTestSuite, PrefillFirst_ContinuesPrefillBeforeNewSubmitted) {
    Submit(MakeRequestSpec("r1", 4));  // 8 tokens, needs 2 chunks
    PlanOnce();                        // r1 chunk 1

    Submit(MakeRequestSpec("r2", 2, 50));  // arrives during r1's prefill
    auto plan = PlanOnce();                // should continue r1, not start r2
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r1");
}

TEST_F(ChunkedPrefillTestSuite, CompletedChunk_IsVisibleToPrefixCacheWithoutHybridCache) {
    Submit(MakeRequestSpec("r1", 4));  // 8 tokens, needs 2 chunks
    PlanOnce();                        // r1 chunk 1

    Submit(MakeRequestSpec("r2", 4));  // same prefix as r1
    PlanOnce();                        // r1 chunk 2; inserts chunk 1 into KV prefix cache

    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_ids.size(), 1u);
    EXPECT_EQ(fwd->request_ids[0], "r2");
    EXPECT_EQ(fwd->extend_prefix_lens[0], 4);
}

TEST_F(ChunkedPrefillTestSuite, InputIds_CorrectPerChunk) {
    Submit(MakeRequestSpec("r1", 3));  // 6 tokens: [1,2,3,4,5,6]
    auto plan1 = PlanOnce();
    auto* fwd1 = GetForwardOp(plan1);
    ASSERT_NE(fwd1, nullptr);
    // First chunk: tokens [1,2,3,4]
    EXPECT_EQ(fwd1->input_ids.size(), 4u);
    EXPECT_EQ(fwd1->input_ids[0], 1);
    EXPECT_EQ(fwd1->input_ids[3], 4);

    auto plan2 = PlanOnce();
    auto* fwd2 = GetForwardOp(plan2);
    ASSERT_NE(fwd2, nullptr);
    // Second chunk: tokens [5,6]
    EXPECT_EQ(fwd2->input_ids.size(), 2u);
    EXPECT_EQ(fwd2->input_ids[0], 5);
    EXPECT_EQ(fwd2->input_ids[1], 6);
}

}  // namespace tokenspeed::test
