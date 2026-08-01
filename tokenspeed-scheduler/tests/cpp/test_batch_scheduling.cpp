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

class BatchSchedulingTestSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_batch_size = 3;
        cfg.enable_l3_storage = false;
        return cfg;
    }

    void BringToDecoding(const std::string& id, std::int32_t num_pages = 2, token_t start = 1) {
        Submit(MakeRequestSpec(id, num_pages, start));
        PlanOnce();
        SendForwardDone(id, {42});
        PlanOnce();
    }
};

TEST_F(BatchSchedulingTestSuite, MaxBatchSize_LimitsScheduledRequests) {
    for (int i = 0; i < 5; ++i) {
        Submit(MakeRequestSpec("r" + std::to_string(i), 2, static_cast<token_t>(i * 100 + 1)));
    }
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    EXPECT_LE(fwd->request_ids.size(), 3u);
}

TEST_F(BatchSchedulingTestSuite, MultipleDecoding_AllInSameBatch) {
    BringToDecoding("r1", 2, 1);
    BringToDecoding("r2", 2, 50);

    SendReserveNumTokens("r1", 0);
    SendReserveNumTokens("r2", 0);
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    EXPECT_EQ(fwd->request_ids.size(), 2u);
}

TEST_F(BatchSchedulingTestSuite, PrefillBlocksDecode) {
    BringToDecoding("r1", 2, 1);

    Submit(MakeRequestSpec("r2", 2, 50));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);

    // r2 should be prefilling; r1 should NOT be in batch (prefill blocks decode).
    bool r1_found = false, r2_found = false;
    for (const auto& rid : fwd->request_ids) {
        if (rid == "r1") r1_found = true;
        if (rid == "r2") r2_found = true;
    }
    EXPECT_TRUE(r2_found);
    EXPECT_FALSE(r1_found) << "Decode should be blocked when prefill is scheduled";
}

TEST_F(BatchSchedulingTestSuite, DistinctPoolIndices) {
    Submit(MakeRequestSpec("r1", 2, 1));
    Submit(MakeRequestSpec("r2", 2, 50));
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    ASSERT_EQ(fwd->request_pool_indices.size(), 2u);
    EXPECT_NE(fwd->request_pool_indices[0], fwd->request_pool_indices[1]);
}

TEST_F(BatchSchedulingTestSuite, TokenBudget_LimitsScheduledTokens) {
    auto cfg = MakeConfig();
    cfg.max_scheduled_tokens = 4;
    cfg.max_batch_size = 8;
    scheduler_ = std::make_unique<Scheduler>(cfg);

    Submit(MakeRequestSpec("r1", 2, 1));   // 4 tokens
    Submit(MakeRequestSpec("r2", 2, 50));  // 4 tokens
    auto plan = PlanOnce();
    auto* fwd = GetForwardOp(plan);
    ASSERT_NE(fwd, nullptr);
    // Only r1 fits in budget of 4.
    EXPECT_EQ(fwd->request_ids.size(), 1u);
}

}  // namespace tokenspeed::test
