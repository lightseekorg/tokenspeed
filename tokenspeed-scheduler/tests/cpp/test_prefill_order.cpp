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
#include <memory>
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
        config.max_resident_requests = config.max_batch_size;
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

// Short-prefill-first with configurable batch rows, chunk yield and a replayable group.
class ShortPrefillAgingTest : public PrefillOrderTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = PrefillOrderTest::MakeConfig();
        config.enable_short_prefill_first = true;
        config.max_batch_size = batch_;
        config.max_resident_requests = batch_;
        config.prefill_chunk_yield_tokens = yield_;
        if (replay_) {
            config.cache_groups.push_back(CacheGroupConfig{
                .group_id = "replayable",
                .block_granularity = 4,
                .total_pages = 2049,
                .attn_kind = AttnKind::kSlidingWindow,
                .sliding_window_tokens = 8,
                .replayable = true,
                .transfer_policy = CacheTransferPolicy::FullSuffix,
            });
        }
        return config;
    }

    void Rebuild() {
        config_ = WithHostPackingFromDevice(MakeConfig());
        scheduler_ = std::make_unique<Scheduler>(config_);
    }

    // The next plan with forward rows, landing the plans before it.
    ExecutionPlan NextForward() {
        for (int i = 0; i < 4; ++i) {
            auto plan = PlanOnce();
            const auto* batch = FindForwardBatch(plan);
            if (batch != nullptr && !batch->request_ids.empty()) {
                return plan;
            }
            Land(plan);
        }
        return PlanOnce();
    }

    // A prompt resuming at a seeded 64-token prefix with `uncached` tokens of its own.
    static RequestSpec Hit(const std::string& id, std::int32_t uncached, std::int32_t unique) {
        auto tokens = MakeTokens(64);
        for (std::int32_t i = 0; i < uncached; ++i) {
            tokens.push_back(100000 + unique * 16 + i);
        }
        return RequestSpec{.request_id = id, .tokens = tokens, .max_new_tokens = 1};
    }

    // The round a 128-token prompt first runs while `per_round` hit-heavy prompts arrive each round.
    int RowBoundStart(std::int32_t uncached, std::int32_t per_round) {
        Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
        Start({RequestSpec{.request_id = "old", .tokens = MakeTokens(128, 5001), .max_new_tokens = 1}});
        for (int round = 0; round < 64; ++round) {
            std::vector<RequestSpec> hits;
            for (int i = 0; i < per_round; ++i) {
                hits.push_back(Hit("h" + std::to_string(round) + "_" + std::to_string(i), uncached, round * 8 + i));
            }
            Start(hits);
            const auto plan = PlanOnce();
            const auto* batch = FindForwardBatch(plan);
            if (batch != nullptr &&
                std::find(batch->request_ids.begin(), batch->request_ids.end(), "old") != batch->request_ids.end()) {
                return round;
            }
            Land(plan);
        }
        return -1;
    }

    std::int32_t batch_{32};
    std::int32_t yield_{0};
    bool replay_{false};
};

INSTANTIATE_TEST_SUITE_P(PrefillRole, ShortPrefillYieldOrderTest, ::testing::Values(Role::kP));
INSTANTIATE_TEST_SUITE_P(LocalRoles, ShortPrefillAgingTest, ::testing::Values(Role::kP, Role::kFused));

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

TEST_P(ShortPrefillOrderTest, ReplayWorkRanksAfterASmallerColdPrompt) {
    auto config = MakeConfig();
    config.max_scheduled_tokens = 16;
    config.cache_groups.push_back(CacheGroupConfig{
        .group_id = "replayable",
        .block_granularity = 4,
        .total_pages = 2049,
        .attn_kind = AttnKind::kSlidingWindow,
        .sliding_window_tokens = 8,
        .replayable = true,
        .transfer_policy = CacheTransferPolicy::FullSuffix,
    });
    config_ = WithHostPackingFromDevice(config);
    scheduler_ = std::make_unique<Scheduler>(config_);
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // The hit leaves one new token but also replays eight: nine tokens of
    // work must rank after the eight-token cold prompt.
    Start({RequestSpec{.request_id = "cold", .tokens = MakeTokens(8, 501), .max_new_tokens = 1},
           RequestSpec{.request_id = "hit", .tokens = MakeTokens(65), .max_new_tokens = 1}});
    const auto first = PlanOnce();
    const auto* batch = FindForwardBatch(first);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"cold"}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{8}));
    Land(first);
    const auto next = PlanOnce();
    batch = FindForwardBatch(next);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"hit"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{56}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{9}));
}

TEST_P(ShortPrefillOrderTest, EqualWorkPreservesSubmissionOrder) {
    Start({RequestSpec{.request_id = "first", .tokens = MakeTokens(8, 101), .max_new_tokens = 1},
           RequestSpec{.request_id = "second", .tokens = MakeTokens(8, 201), .max_new_tokens = 1}});
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"first", "second"}));
}

// The oldest prompt's 16 tokens is four 4-token prompts' work, so it ages within one batch.
TEST_P(ShortPrefillOrderTest, AgingIsAppliedWithinOneBatch) {
    std::vector<RequestSpec> requests{{.request_id = "old-long", .tokens = MakeTokens(16, 101), .max_new_tokens = 1}};
    for (int i = 0; i < 20; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "short" + std::to_string(i), .tokens = MakeTokens(4, 1001 + 10 * i), .max_new_tokens = 1});
    }
    Start(requests);
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    const auto old = std::find(batch->request_ids.begin(), batch->request_ids.end(), "old-long");
    ASSERT_NE(old, batch->request_ids.end());
    EXPECT_EQ(old - batch->request_ids.begin(), 4);
    EXPECT_EQ(batch->input_lengths.at(static_cast<std::size_t>(old - batch->request_ids.begin())), 16);
    EXPECT_EQ(batch->request_ids.size(), 13u) << "eight more short prompts fill the 64-token budget";
}

// Each oldest prompt's allowance counts the work that went ahead of it from when it became the oldest.
TEST_P(ShortPrefillOrderTest, EachOldestPromptGetsItsOwnShareOfTheWork) {
    std::vector<RequestSpec> requests;
    for (int i = 0; i < 2; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "long" + std::to_string(i), .tokens = MakeTokens(16, 101 + i * 200), .max_new_tokens = 1});
    }
    for (int i = 0; i < 17; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "short" + std::to_string(i), .tokens = MakeTokens(4, 1001 + 10 * i), .max_new_tokens = 1});
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
        if (i == 3 || i == 7) {
            expected.push_back("long" + std::to_string(i / 4));
        }
    }
    EXPECT_EQ(first_admissions, expected);
    EXPECT_EQ(tokens, 2 * 16 + 17 * 4);
    for (const auto& request : requests) {
        EXPECT_EQ(scheduler_->RequestTokenSize(request.request_id), -1);
    }
}

TEST_P(ShortPrefillOrderTest, OneTokenPromptsDoNotAgeALongPrompt) {
    std::vector<RequestSpec> requests{{.request_id = "old-long", .tokens = MakeTokens(128, 101), .max_new_tokens = 1}};
    for (int i = 0; i < 20; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "short" + std::to_string(i), .tokens = MakeTokens(1, 1001 + i), .max_new_tokens = 1});
    }
    Start(requests);
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->request_ids.size(), 21u);
    EXPECT_EQ(batch->request_ids.back(), "old-long") << "20 tokens went ahead of 128: not aged, so last";
    EXPECT_EQ(batch->input_lengths.back(), 44);
}

TEST_P(ShortPrefillOrderTest, ContinuousShortArrivalsAdvanceEveryEarlierLongPrompt) {
    auto config = MakeConfig();
    config.max_batch_size = 2;
    config.max_resident_requests = config.max_batch_size;
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
    EXPECT_EQ(overtakes.front(), 4);
    for (int i = 0; i < 3; ++i) {
        EXPECT_GE(overtakes[i], 0);
        EXPECT_LE(overtakes[i], 4 * (i + 1));
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
    Start({RequestSpec{.request_id = "resident", .tokens = MakeTokens(256, 101), .max_new_tokens = 1}});
    Land(PlanOnce());
    // 64 tokens: too long to carry in the 32 yielded tokens; it ages after 64 of the 96 carried tokens (not at twice).
    Start({RequestSpec{.request_id = "old-long", .tokens = MakeTokens(64, 501), .max_new_tokens = 1}});
    for (int round = 0; round < 3; ++round) {
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
    // Twelve carried first admissions protect the oldest waiter. The resident
    // finishes first; the protected prompt starts before the later short one.
    // The later prompt cannot fit whole beside either chunk.
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"resident", "old-long"}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{32, 32}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{224, 0}));
}

// Two rows a round of 1-token hits: each counts a row's share (32 of 64), so the 128-token prompt runs by round 4.
TEST_P(ShortPrefillAgingTest, RowBoundOneTokenHitsStillAgeTheOldest) {
    batch_ = 2;
    Rebuild();
    const int started = RowBoundStart(/*uncached=*/1, /*per_round=*/2);
    EXPECT_GE(started, 0);
    EXPECT_LE(started, 4);
}

// Four rows a round of 4-token hits: each counts a row's share (16 of 64), so the 128-token prompt runs by round 4.
TEST_P(ShortPrefillAgingTest, RowBoundFourTokenHitsStillAgeTheOldest) {
    batch_ = 4;
    Rebuild();
    const int started = RowBoundStart(/*uncached=*/4, /*per_round=*/4);
    EXPECT_GE(started, 0);
    EXPECT_LE(started, 4);
}

// Bypassers that replay an 8-token window count the 9 tokens they are ranked by: a 20-token prompt waits for three.
TEST_P(ShortPrefillAgingTest, ABypassersReplayCountsTowardTheAllowance) {
    replay_ = true;
    Rebuild();
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    Start({RequestSpec{.request_id = "old", .tokens = MakeTokens(20, 5001), .max_new_tokens = 1}});
    std::vector<RequestSpec> hits;
    for (int i = 0; i < 40; ++i) {
        hits.push_back(Hit("h" + std::to_string(i), 1, i));
    }
    Start(hits);
    int before = 0;
    bool started = false;
    for (int round = 0; round < 20 && !started; ++round) {
        const auto plan = PlanOnce();
        if (const auto* batch = FindForwardBatch(plan)) {
            for (const auto& id : batch->request_ids) {
                started = started || id == "old";
                before += started ? 0 : 1;
            }
        }
        Land(plan);
    }
    EXPECT_TRUE(started);
    EXPECT_EQ(before, 3);
}

// The carried and the new-prompt paths count the same 9-token bypassers alike: five of them age a 40-token prompt.
TEST_P(ShortPrefillAgingTest, CarriedAndNewPromptBypassersCountTheSameWork) {
    if (GetParam() != Role::kP) {
        GTEST_SKIP() << "only the P role carries prompts";
    }
    for (const bool via_carry : {true, false}) {
        SCOPED_TRACE(via_carry ? "carry" : "new prompt");
        batch_ = via_carry ? 32 : 5;
        yield_ = 32;
        replay_ = true;
        Rebuild();
        Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
        if (via_carry) {
            Start({RequestSpec{.request_id = "resident", .tokens = MakeTokens(256, 9001), .max_new_tokens = 1}});
            Land(PlanOnce());
        }
        Start({RequestSpec{.request_id = "old", .tokens = MakeTokens(40, 5001), .max_new_tokens = 1}});
        for (int round = 0; round < (via_carry ? 2 : 1); ++round) {
            std::vector<RequestSpec> hits;
            for (int i = 0; i < (via_carry ? (round == 0 ? 3 : 2) : 5); ++i) {
                hits.push_back(Hit("h" + std::to_string(round) + "_" + std::to_string(i), 1, round * 8 + i));
            }
            Start(hits);
            Land(PlanOnce());
        }
        if (via_carry) {
            SendAbortEvent("resident");
        }
        Start({RequestSpec{.request_id = "late", .tokens = MakeTokens(33, 7001), .max_new_tokens = 1}});
        const auto plan = NextForward();
        const auto* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr);
        EXPECT_EQ(batch->request_ids.front(), "old");
    }
}

// A chunked bypasser counts its whole prompt less hit at its first admission, not the chunk it took.
TEST_P(ShortPrefillAgingTest, AChunkedBypasserCountsItsWholeWork) {
    Rebuild();
    Start({RequestSpec{.request_id = "old", .tokens = MakeTokens(128, 101), .max_new_tokens = 1},
           RequestSpec{.request_id = "mid", .tokens = MakeTokens(100, 501), .max_new_tokens = 1}});
    const auto first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    EXPECT_EQ(FindForwardBatch(first)->request_ids, (std::vector<std::string>{"mid"}));
    Land(first);
    std::vector<RequestSpec> shorts;
    for (int i = 0; i < 20; ++i) {
        shorts.push_back(RequestSpec{
            .request_id = "s" + std::to_string(i), .tokens = MakeTokens(4, 2001 + 10 * i), .max_new_tokens = 1});
    }
    Start(shorts);
    const auto second = PlanOnce();
    ASSERT_NE(FindForwardBatch(second), nullptr);
    EXPECT_EQ(FindForwardBatch(second)->request_ids.size(), 8u);
    Land(second);
    const auto third = PlanOnce();
    ASSERT_NE(FindForwardBatch(third), nullptr);
    EXPECT_EQ(FindForwardBatch(third)->request_ids.front(), "old") << "100 + 7 * 4 = 128 went ahead";
}

// The oldest prompt's own share is its work past its hit (40 of 104 tokens), not its prompt length.
TEST_P(ShortPrefillAgingTest, TheOldestsOwnHitShortensItsShare) {
    Rebuild();
    Seed(RequestSpec{.request_id = "seed", .tokens = MakeTokens(64), .max_new_tokens = 1});
    auto old_tokens = MakeTokens(64);
    for (int i = 0; i < 40; ++i) {
        old_tokens.push_back(70000 + i);
    }
    std::vector<RequestSpec> requests{{.request_id = "old", .tokens = old_tokens, .max_new_tokens = 1}};
    for (int i = 0; i < 20; ++i) {
        requests.push_back(RequestSpec{
            .request_id = "s" + std::to_string(i), .tokens = MakeTokens(4, 1001 + 10 * i), .max_new_tokens = 1});
    }
    Start(requests);
    const auto plan = PlanOnce();
    const auto* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    const auto old = std::find(batch->request_ids.begin(), batch->request_ids.end(), "old");
    ASSERT_NE(old, batch->request_ids.end());
    EXPECT_EQ(old - batch->request_ids.begin(), 10) << "ten 4-token prompts are its 40 uncached tokens";
}

}  // namespace
}  // namespace tokenspeed::test
