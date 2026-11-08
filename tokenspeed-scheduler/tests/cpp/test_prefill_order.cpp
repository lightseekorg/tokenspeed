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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace tokenspeed::test {
namespace {

class PrefillOrderTest : public SchedulerTestSuite, public ::testing::WithParamInterface<Role> {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.role = GetParam();
        config.prefix_granularity = 4;
        config.max_scheduled_tokens = 64;
        config.max_batch_size = 32;
        config.device_allocator.total_pages = 2049;
        config.host_allocator.total_pages = 4097;
        config.host_write_policy = SchedulerConfig::HostWritePolicy::kWriteBack;
        config.cache_groups.front().block_granularity = 4;
        config.cache_groups.front().total_pages = 2049;
        config.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        SetTestSnapshotPool(config);
        return config;
    }

    void Start(const std::vector<RequestSpec>& requests) {
        Submit(requests);
        if (GetParam() == Role::kP) {
            for (const auto& request : requests) {
                ExecutionEvent event;
                event.With(pd::BootstrappedEvent{request.request_id});
                scheduler_->Advance(std::move(event));
            }
        }
    }

    void Land(const ExecutionPlan& plan) {
        for (const auto& operation : ExtractCacheOpsOfKind<LoadBackBatch>(plan)) {
            for (const auto id : std::get<LoadBackBatch>(operation).op_ids) {
                SendLoadBackDone(id);
            }
        }
        if (const auto* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const auto& id = batch->request_ids[row];
                const bool complete =
                    batch->extend_prefix_lens[row] + batch->input_lengths[row] == batch->prefill_lengths[row];
                SendForwardDone(id, complete ? std::vector<std::int32_t>{42} : std::vector<std::int32_t>{});
                if (complete && GetParam() == Role::kFused) {
                    SendFinish(id);
                }
            }
        }
        if (plan.remote_decode) {
            for (const auto& id : plan.remote_decode->request_ids) {
                ExecutionEvent event;
                event.With(pd::SucceededEvent{id});
                scheduler_->Advance(std::move(event));
            }
        }
        AckWriteBacks(plan);
    }

    void Seed(const RequestSpec& request) {
        Start({request});
        for (int round = 0; round < 16 && scheduler_->RequestTokenSize(request.request_id) != -1; ++round) {
            Land(PlanOnce());
        }
        ASSERT_EQ(scheduler_->RequestTokenSize(request.request_id), -1);
    }
};

class ShortPrefillOrderTest : public PrefillOrderTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = PrefillOrderTest::MakeConfig();
        config.enable_short_prefill_first = true;
        return config;
    }
};

class ShortPrefillYieldOrderTest : public ShortPrefillOrderTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ShortPrefillOrderTest::MakeConfig();
        config.prefill_chunk_yield_tokens = 32;
        return config;
    }
};

INSTANTIATE_TEST_SUITE_P(PrefillRole, ShortPrefillYieldOrderTest, ::testing::Values(Role::kP));

INSTANTIATE_TEST_SUITE_P(LocalRoles, PrefillOrderTest, ::testing::Values(Role::kP, Role::kFused));
INSTANTIATE_TEST_SUITE_P(LocalRoles, ShortPrefillOrderTest, ::testing::Values(Role::kP, Role::kFused));

TEST_P(PrefillOrderTest, DefaultKeepsSubmissionOrderDespiteCachedWork) {
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    Start({RequestSpec{.request_id = "cold", .tokens = MakeTokens(16, 101), .max_new_tokens = 1},
           RequestSpec{.request_id = "hot", .tokens = MakeTokens(65), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"cold", "hot"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{0, 64}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{16, 1}));
}

TEST_P(PrefillOrderTest, DefaultStartsAnEarlierLongPromptBeforeAShortOne) {
    Start({RequestSpec{.request_id = "long", .tokens = MakeTokens(128), .max_new_tokens = 1},
           RequestSpec{.request_id = "short", .tokens = MakeTokens(1, 501), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"long"}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{64}));
}

TEST_P(ShortPrefillOrderTest, CachedWorkPrecedesAnEarlierShorterColdPrompt) {
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    Start({RequestSpec{.request_id = "cold", .tokens = MakeTokens(16, 101), .max_new_tokens = 1},
           RequestSpec{.request_id = "hot", .tokens = MakeTokens(65), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"hot", "cold"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{64, 0}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{1, 16}));
}

TEST_P(ShortPrefillOrderTest, AHostPrefixAlsoReducesTheRemainingWork) {
    auto config = MakeConfig();
    config.device_allocator.total_pages = 17;
    config.cache_groups.front().total_pages = 17;
    SetTestSnapshotPool(config);
    config_ = WithHostPackingFromDevice(config);
    scheduler_ = std::make_unique<Scheduler>(config_);
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(32), .max_new_tokens = 1});
    Seed(RequestSpec{.request_id = "pressure", .tokens = MakeTokens(55, 101), .max_new_tokens = 1});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    Start({RequestSpec{.request_id = "cold", .tokens = MakeTokens(16, 501), .max_new_tokens = 1},
           RequestSpec{.request_id = "hot", .tokens = MakeTokens(33), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_GE(batch->NumExtends(), 1u);
    EXPECT_EQ(batch->request_ids.front(), "hot");
    EXPECT_EQ(batch->extend_prefix_lens.front(), 32);
    EXPECT_EQ(batch->input_lengths.front(), 1);
    EXPECT_FALSE(ExtractCacheOpsOfKind<LoadBackBatch>(plan).empty());
}

TEST_P(ShortPrefillOrderTest, ForbiddenReuseDoesNotImproveTheRank) {
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    RequestSpec uncached{.request_id = "uncached", .tokens = MakeTokens(65), .max_new_tokens = 1};
    uncached.max_cached_prefix_tokens = 0;
    Start({uncached, RequestSpec{.request_id = "cold", .tokens = MakeTokens(16, 101), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids, (std::vector<std::string>{"cold", "uncached"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{0, 0}));
}

TEST_P(ShortPrefillOrderTest, EqualWorkPreservesSubmissionOrder) {
    Start({RequestSpec{.request_id = "first", .tokens = MakeTokens(8, 101), .max_new_tokens = 1},
           RequestSpec{.request_id = "second", .tokens = MakeTokens(8, 201), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"first", "second"}));
}

TEST_P(ShortPrefillOrderTest, BypassLimitIsAppliedWithinOneBatch) {
    std::vector<RequestSpec> requests{{.request_id = "old-long", .tokens = MakeTokens(128, 101), .max_new_tokens = 1}};
    for (int i = 0; i < 20; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "short" + std::to_string(i), .tokens = MakeTokens(1, 1001 + i), .max_new_tokens = 1});
    }
    Start(requests);
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    const auto old = std::find(batch->request_ids.begin(), batch->request_ids.end(), "old-long");
    ASSERT_NE(old, batch->request_ids.end());
    EXPECT_EQ(old - batch->request_ids.begin(), 8);
    EXPECT_EQ(batch->request_ids.back(), "old-long");
    EXPECT_EQ(batch->input_lengths.back(), 56);
}

TEST_P(ShortPrefillOrderTest, EachOldestPromptGetsItsOwnBypassAllowance) {
    std::vector<RequestSpec> requests;
    for (int i = 0; i < 2; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "long" + std::to_string(i), .tokens = MakeTokens(128, 101 + i * 200), .max_new_tokens = 1});
    }
    for (int i = 0; i < 17; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "short" + std::to_string(i), .tokens = MakeTokens(1, 1001 + i), .max_new_tokens = 1});
    }
    Start(requests);
    std::vector<std::string> first_admissions;
    int tokens = 0;
    for (int round = 0; round < 16; ++round) {
        const auto plan = PlanOnce();
        if (const auto* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const auto& id = batch->request_ids[row];
                if (std::find(first_admissions.begin(), first_admissions.end(), id) == first_admissions.end()) {
                    first_admissions.push_back(id);
                }
                tokens += batch->input_lengths[row];
            }
        }
        Land(plan);
    }
    std::vector<std::string> expected;
    for (int i = 0; i < 17; ++i) {
        expected.push_back("short" + std::to_string(i));
        if (i == 7 || i == 15) {
            expected.push_back("long" + std::to_string(i / 8));
        }
    }
    EXPECT_EQ(first_admissions, expected);
    EXPECT_EQ(tokens, 2 * 128 + 17);
    for (const auto& request : requests) {
        EXPECT_EQ(scheduler_->RequestTokenSize(request.request_id), -1);
    }
}

TEST_P(ShortPrefillOrderTest, ContinuousShortArrivalsAdvanceEveryEarlierLongPrompt) {
    auto config = MakeConfig();
    config.max_batch_size = 2;
    config_ = WithHostPackingFromDevice(config);
    scheduler_ = std::make_unique<Scheduler>(config_);
    for (int i = 0; i < 3; ++i) {
        Start({RequestSpec{
            .request_id = "long" + std::to_string(i), .tokens = MakeTokens(128, 101 + i * 200), .max_new_tokens = 1}});
    }
    int admitted_shorts = 0;
    std::vector<int> overtakes(3, -1);
    std::vector<int> long_tokens(3, 0);
    for (int round = 0; round < 64; ++round) {
        for (int i = 0; i < 2; ++i) {
            const int index = 2 * round + i;
            Start({RequestSpec{.request_id = "short" + std::to_string(index),
                               .tokens = MakeTokens(32, 1001 + 100 * index),
                               .max_new_tokens = 1}});
        }
        const auto plan = PlanOnce();
        if (const auto* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const auto& id = batch->request_ids[row];
                if (id.starts_with("long")) {
                    const int index = id.back() - '0';
                    if (overtakes[index] == -1) {
                        overtakes[index] = admitted_shorts;
                    }
                    long_tokens[index] += batch->input_lengths[row];
                } else {
                    ++admitted_shorts;
                }
            }
        }
        Land(plan);
    }
    EXPECT_EQ(overtakes.front(), 8);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(overtakes[i], 0);
        EXPECT_LE(overtakes[i], 8 * (i + 1));
        EXPECT_EQ(long_tokens[i], 128);
        EXPECT_EQ(scheduler_->RequestTokenSize("long" + std::to_string(i)), -1);
    }
}

TEST_P(ShortPrefillOrderTest, ResidentPrefillKeepsItsPlaceWhenAShortPromptArrives) {
    Start({RequestSpec{.request_id = "resident", .tokens = MakeTokens(128, 101), .max_new_tokens = 1}});
    const auto first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    ASSERT_EQ(FindForwardBatch(first)->input_lengths, (std::vector<std::int32_t>{64}));
    Land(first);
    Start({RequestSpec{.request_id = "short", .tokens = MakeTokens(1, 501), .max_new_tokens = 1}});
    const auto next = PlanOnce();
    ASSERT_NE(FindForwardBatch(next), nullptr);
    EXPECT_EQ(FindForwardBatch(next)->request_ids, (std::vector<std::string>{"resident"}));
    EXPECT_EQ(FindForwardBatch(next)->input_lengths, (std::vector<std::int32_t>{64}));
}

TEST_P(ShortPrefillYieldOrderTest, CarriedFirstAdmissionsProtectTheOldestWaitingPrompt) {
    Start({RequestSpec{.request_id = "resident", .tokens = MakeTokens(224, 101), .max_new_tokens = 1}});
    Land(PlanOnce());
    Start({RequestSpec{.request_id = "old-long", .tokens = MakeTokens(128, 501), .max_new_tokens = 1}});
    for (int round = 0; round < 2; ++round) {
        std::vector<RequestSpec> shorts;
        std::vector<std::string> expected;
        for (int i = 0; i < 4; ++i) {
            const int index = 4 * round + i;
            const auto id = "short" + std::to_string(index);
            shorts.push_back(
                RequestSpec{.request_id = id, .tokens = MakeTokens(8, 1001 + 100 * index), .max_new_tokens = 1});
            expected.push_back(id);
        }
        Start(shorts);
        const auto plan = PlanOnce();
        const auto* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        expected.push_back("resident");
        EXPECT_EQ(batch->request_ids, expected);
        EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{8, 8, 8, 8, 32}));
        Land(plan);
    }
    Land(PlanOnce());  // The resident has 32 tokens left after this full chunk.
    Start({RequestSpec{.request_id = "late-short", .tokens = MakeTokens(33, 3001), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    // Eight carried first admissions protect the oldest waiter. The resident
    // finishes first; the protected prompt starts before the later short one.
    // The later prompt cannot fit whole beside either chunk.
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"resident", "old-long"}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{32, 32}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{192, 0}));
}

}  // namespace
}  // namespace tokenspeed::test
