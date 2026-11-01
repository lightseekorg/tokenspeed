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
namespace {

class PendingPrefixStoreTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig config = SchedulerTestSuite::MakeConfig();
        config.prefix_granularity = 4;
        config.device_allocator.total_pages = 8;
        config.host_allocator.total_pages = 65;
        config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        config.cache_groups.front().block_granularity = 4;
        config.cache_groups.front().total_pages = 8;
        SetTestSnapshotPool(config);
        return config;
    }

    ExecutionPlan StartSeedEviction() {
        Submit(RequestSpec{.request_id = "seed", .tokens = MakeTokens(8), .max_new_tokens = 1});
        const ExecutionPlan seed = PlanOnce();
        EXPECT_NE(FindForwardBatch(seed), nullptr);
        SendForwardDone("seed", {900});
        SendFinish("seed");

        Submit(RequestSpec{.request_id = "pressure", .tokens = MakeTokens(23, 101), .max_new_tokens = 1});
        ExecutionPlan pressure = PlanOnce();
        EXPECT_NE(FindForwardBatch(pressure), nullptr);
        EXPECT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(pressure).empty());
        SendForwardDone("pressure", {900});
        SendFinish("pressure");
        return pressure;
    }
};

TEST_F(PendingPrefixStoreTest, WaitsForItsCopyWhileAnUnrelatedPromptRuns) {
    const ExecutionPlan store = StartSeedEviction();
    Submit({RequestSpec{.request_id = "hit", .tokens = MakeTokens(9), .max_new_tokens = 1},
            RequestSpec{.request_id = "other", .tokens = MakeTokens(4, 501), .max_new_tokens = 1}});
    const ExecutionPlan waiting = PlanOnce();
    const ForwardBatch* other = FindForwardBatch(waiting);
    ASSERT_NE(other, nullptr);
    EXPECT_EQ(other->request_ids, std::vector<std::string>{"other"});
    EXPECT_TRUE(waiting.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    SendForwardDone("other", {900});
    SendFinish("other");

    AckWriteBacks(waiting);
    AckWriteBacks(store);
    const ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* hit = FindForwardBatch(resumed);
    ASSERT_NE(hit, nullptr);
    ASSERT_EQ(hit->request_ids, std::vector<std::string>{"hit"});
    EXPECT_EQ(hit->extend_prefix_lens, std::vector<std::int32_t>{8});
    EXPECT_FALSE(ExtractCacheOpsOfKind<LoadBackBatch>(resumed).empty());
}

TEST_F(PendingPrefixStoreTest, AnUnrelatedCopyDoesNotDelayAdmission) {
    const ExecutionPlan store = StartSeedEviction();
    Submit(RequestSpec{.request_id = "other", .tokens = MakeTokens(9, 501), .max_new_tokens = 1});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"other"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{0});
    AckWriteBacks(store);
}

TEST_F(PendingPrefixStoreTest, ARequestThatForbidsReuseDoesNotWaitForItsCopy) {
    const ExecutionPlan store = StartSeedEviction();
    RequestSpec request{.request_id = "uncached", .tokens = MakeTokens(9), .max_new_tokens = 1};
    request.max_cached_prefix_tokens = 0;
    Submit(request);
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"uncached"});
    EXPECT_EQ(batch->extend_prefix_lens, std::vector<std::int32_t>{0});
    AckWriteBacks(store);
}

// A prior branch leaves a longer checkpoint beside the one hit will reuse.
// Pressure evicts both in its admission; later prompts in that same plan
// must see the recorded store before the plan drains it into a copy command.
class SamePlanPrefixStoreTest : public PendingPrefixStoreTest {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig config = PendingPrefixStoreTest::MakeConfig();
        config.device_allocator.total_pages = 24;
        config.cache_groups.front().total_pages = 24;
        config.cache_groups.push_back(CacheGroupConfig{
            .group_id = "state", .block_granularity = 0, .total_pages = 24, .attn_kind = AttnKind::kState});
        SetTestSnapshotPool(config);
        return config;
    }

    void SeedBranches() {
        for (const auto& [id, length] :
             std::vector<std::pair<std::string, std::int32_t>>{{"seed", 8}, {"branch", 16}}) {
            Submit(RequestSpec{.request_id = id, .tokens = MakeTokens(length), .max_new_tokens = 1});
            const ExecutionPlan plan = PlanOnce();
            const ForwardBatch* batch = FindForwardBatch(plan);
            ASSERT_NE(batch, nullptr);
            ASSERT_EQ(batch->request_ids, std::vector<std::string>{id});
            ASSERT_EQ(batch->extend_prefix_lens.at(0) + batch->input_lengths.at(0), length);
            AckWriteBacks(plan);
            SendForwardDone(id, {900});
            SendFinish(id);
        }
    }
};

TEST_F(SamePlanPrefixStoreTest, WaitsForAnEarlierAdmissionInTheSamePlan) {
    SeedBranches();
    Submit({RequestSpec{.request_id = "pressure", .tokens = MakeTokens(60, 101), .max_new_tokens = 1},
            RequestSpec{.request_id = "hit", .tokens = MakeTokens(9), .max_new_tokens = 1},
            RequestSpec{.request_id = "other", .tokens = MakeTokens(4, 501), .max_new_tokens = 1}});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"pressure", "other"}));
    EXPECT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(plan).empty());
    EXPECT_TRUE(plan.aborts.empty());
    EXPECT_EQ(scheduler_->RetractedSize(), 0u);
    // Only the copy changes hit visibility. Keep pressure's allocation alive.
    AckWriteBacks(plan);
    SendForwardDone("pressure", {900});
    SendForwardDone("other", {900});
    SendFinish("other");
    const ExecutionPlan resumed = PlanOnce();
    const ForwardBatch* hit = FindForwardBatch(resumed);
    ASSERT_NE(hit, nullptr);
    ASSERT_GE(hit->NumExtends(), 1u);
    EXPECT_EQ(hit->request_ids.front(), "hit");
    EXPECT_EQ(hit->extend_prefix_lens.front(), 8);
    EXPECT_FALSE(ExtractCacheOpsOfKind<LoadBackBatch>(resumed).empty());
}

TEST_F(SamePlanPrefixStoreTest, ARecordedStoreDoesNotDelayAPromptThatForbidsReuse) {
    SeedBranches();
    RequestSpec uncached{.request_id = "uncached", .tokens = MakeTokens(9), .max_new_tokens = 1};
    uncached.max_cached_prefix_tokens = 0;
    Submit({RequestSpec{.request_id = "pressure", .tokens = MakeTokens(60, 101), .max_new_tokens = 1}, uncached});
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"pressure", "uncached"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{0, 0}));
    EXPECT_FALSE(ExtractCacheOpsOfKind<WriteBackBatch>(plan).empty());
}

}  // namespace
}  // namespace tokenspeed::test
