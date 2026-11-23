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

#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <string>

#include "integration_test_helper.h"

namespace tokenspeed::test {

class ForcedRetractionLimitTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = SchedulerTestSuite::MakeConfig();
        config.device_allocator.total_pages = 129;
        config.host_allocator.total_pages = 129;
        config.cache_groups.front().total_pages = 129;
        config.max_batch_size = 2;
        config.max_resident_requests = 2;
        config.disable_prefix_cache = true;
        config.disable_l2_cache = true;
        config.debug_force_retraction_interval = 2;
        config.debug_force_retraction_max_per_request = 2;
        SetTestSnapshotPool(config);
        return config;
    }

    struct Counts {
        int stores{0};
        int restores{0};
        int generated{0};
        int after_restore{0};
    };
    std::map<std::string, Counts> counts_;

    void SubmitBounded(const std::string& id, std::uint64_t admission, std::int32_t pages) {
        auto request = MakeRequestSpec(id, pages, admission == 1 ? 1 : 101);
        request.admission_id = admission;
        // The whole budget is below the safe-step window and fits on Device:
        // ordinary capacity recovery has no reason to retract these requests.
        request.max_new_tokens = 96;
        Submit(request);
        Bootstrap(id);
    }

    void Bootstrap(const std::string& id) {
        if (Config().role == Role::kD) {
            ExecutionEvent event;
            event.With(pd::BootstrappedEvent{id});
            scheduler_->Advance(event);
        }
    }

    void RecordTransfers(const ExecutionPlan& plan) {
        if (const auto* store = FindSnapshotStore(plan)) {
            for (const auto& id : store->request_ids) ++counts_[id].stores;
        }
        if (const auto* restore = FindRestore(plan)) {
            for (const auto& id : restore->request_ids) {
                ++counts_[id].restores;
                counts_[id].after_restore = 0;
            }
        }
    }

    void CompleteForwards(const ExecutionPlan& plan) {
        if (plan.remote_prefill) {
            for (const auto& id : plan.remote_prefill->request_ids) {
                auto& count = counts_[id];
                ExecutionEvent event;
                event.With(pd::RemotePrefillDoneEvent{id, 1000 + count.generated++});
                scheduler_->Advance(event);
            }
        }
        if (const auto* batch = FindForwardBatch(plan)) {
            for (std::size_t row = 0; row < batch->request_ids.size(); ++row) {
                const auto& id = batch->request_ids[row];
                auto& count = counts_[id];
                if (row >= batch->NumExtends() || batch->CompletesPrompt(row)) {
                    SendForwardDone(id, {1000 + count.generated++});
                    if (count.restores > 0) ++count.after_restore;
                } else {
                    SendForwardDone(id);
                }
            }
        }
    }

    void Step() {
        const auto plan = PlanOnce();
        ASSERT_TRUE(plan.aborts.empty());
        RecordTransfers(plan);
        // Substitute only the physical copy/forward boundary: the real FSM
        // and transfer-ticket owners perform every allocation and transition.
        AckImageStores(plan);
        AckRestores(plan);
        CompleteForwards(plan);
    }

    void ReachRestores(const std::string& id, int target) {
        for (int round = 0; round < 200 && counts_[id].restores < target; ++round) Step();
        ASSERT_EQ(counts_[id].restores, target);
    }

    void VerifyTail(const std::string& id, int expected_restores) {
        const auto frontier = scheduler_->RequestTokenSize(id);
        for (int round = 0; round < 100 && counts_[id].after_restore < 32; ++round) {
            Step();
            ASSERT_EQ(counts_[id].stores, expected_restores);
            ASSERT_EQ(counts_[id].restores, expected_restores);
        }
        EXPECT_GE(counts_[id].after_restore, 32);
        EXPECT_GE(scheduler_->RequestTokenSize(id) - frontier, 32);
    }
};

class ForcedRetractionLimitRoles : public ForcedRetractionLimitTest, public ::testing::WithParamInterface<Role> {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ForcedRetractionLimitTest::MakeConfig();
        config.role = GetParam();
        config.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        return config;
    }
};

TEST_P(ForcedRetractionLimitRoles, ForceTwiceThenCommitAtLeastThirtyTwoTokensWithoutFurtherRetractions) {
    SubmitBounded("r", 1, 2);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("r", 2));
    ASSERT_NO_FATAL_FAILURE(VerifyTail("r", 2));
}

TEST_P(ForcedRetractionLimitRoles, ReusingAnExternalRequestIdStartsANewForcedBudget) {
    SubmitBounded("reused", 1, 2);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("reused", 2));
    ASSERT_NO_FATAL_FAILURE(VerifyTail("reused", 2));
    SendFinish("reused");
    Step();
    SubmitBounded("reused", 2, 2);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("reused", 4));
    ASSERT_NO_FATAL_FAILURE(VerifyTail("reused", 4));
}

INSTANTIATE_TEST_SUITE_P(Roles, ForcedRetractionLimitRoles, ::testing::Values(Role::kFused, Role::kD));

TEST_F(ForcedRetractionLimitTest, ACappedOldestRequestDoesNotPreventForcingTheNextRequest) {
    SubmitBounded("a", 1, 2);
    SubmitBounded("b", 2, 2);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("a", 2));
    ASSERT_NO_FATAL_FAILURE(ReachRestores("b", 2));
    EXPECT_EQ(counts_["a"].stores, 2);
    EXPECT_EQ(counts_["b"].stores, 2);
}

class UnlimitedForcedRetractionTest : public ForcedRetractionLimitTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ForcedRetractionLimitTest::MakeConfig();
        config.debug_force_retraction_max_per_request.reset();
        return config;
    }
};

TEST_F(UnlimitedForcedRetractionTest, OmissionKeepsForcingPastTheOptionalCap) {
    SubmitBounded("r", 1, 2);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("r", 5));
    EXPECT_EQ(counts_["r"].stores, 5);
}

class MidPrefillForcedRetractionLimitTest : public ForcedRetractionLimitTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ForcedRetractionLimitTest::MakeConfig();
        config.debug_force_retraction_interval = -2;
        config.max_scheduled_tokens = 4;
        return config;
    }
};

TEST_F(MidPrefillForcedRetractionLimitTest, TheCapAlsoLetsAnIncompletePromptFinishAndContinue) {
    SubmitBounded("r", 1, 10);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("r", 2));
    ASSERT_NO_FATAL_FAILURE(VerifyTail("r", 2));
}

class RefusedForcedRetractionLimitTest : public ForcedRetractionLimitTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ForcedRetractionLimitTest::MakeConfig();
        config.debug_force_retraction_max_per_request = 1;
        config.max_retracted_requests = 1;
        return config;
    }
};

TEST_F(RefusedForcedRetractionLimitTest, ARefusedImageDoesNotConsumeTheForcedBudget) {
    SubmitBounded("a", 1, 2);
    SubmitBounded("b", 2, 2);
    ExecutionPlan held_store;
    for (int round = 0; round < 20; ++round) {
        auto plan = PlanOnce();
        RecordTransfers(plan);
        CompleteForwards(plan);
        if (FindSnapshotStore(plan)) {
            held_store = std::move(plan);
            break;
        }
        AckImageStores(plan);
        AckRestores(plan);
    }
    ASSERT_NE(FindSnapshotStore(held_store), nullptr);
    ASSERT_EQ(FindSnapshotStore(held_store)->request_ids, std::vector<std::string>{"a"});
    SendAbortEvent("a");
    // The export ticket still owns the only blob slot. b is eligible at
    // several debug intervals, but each image-fit refusal must remain free.
    for (int round = 0; round < 6; ++round) Step();
    EXPECT_EQ(counts_["b"].stores, 0);
    AckImageStores(held_store);
    ASSERT_NO_FATAL_FAILURE(ReachRestores("b", 1));
    ASSERT_NO_FATAL_FAILURE(VerifyTail("b", 1));
}

class CapacityAfterForcedLimitTest : public ForcedRetractionLimitTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto config = ForcedRetractionLimitTest::MakeConfig();
        config.device_allocator.total_pages = 17;
        config.cache_groups.front().total_pages = 17;
        config.debug_force_retraction_max_per_request = 1;
        SetTestSnapshotPool(config);
        return config;
    }
};

TEST_F(CapacityAfterForcedLimitTest, CapacityRecoveryCanRetractAnAlreadyCappedRequest) {
    // No declared generation budget: the ordinary reserve exemption does
    // not protect these requests as they outgrow this deliberately small pool.
    Submit({MakeRequestSpec("a", 2), MakeRequestSpec("b", 2, 101)});
    for (int round = 0; round < 100 && counts_["a"].stores < 2 && counts_["b"].stores < 2; ++round) Step();
    EXPECT_TRUE(counts_["a"].stores >= 2 || counts_["b"].stores >= 2)
        << "the per-request forced cap must not cap ordinary capacity recovery";
}

TEST_F(ForcedRetractionLimitTest, ConfigRejectsNonpositiveCapsAndDisabledForcing) {
    auto config = Config();
    for (const auto cap : {std::numeric_limits<std::int32_t>::min(), -1, 0}) {
        config.debug_force_retraction_max_per_request = cap;
        EXPECT_THROW(config.Validate(), std::invalid_argument);
    }
    config.debug_force_retraction_max_per_request = 1;
    config.debug_force_retraction_interval = 0;
    try {
        config.Validate();
        FAIL() << "a supplied cap requires active forcing";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string{error.what()}.find("--debug-force-retraction-max-per-request"), std::string::npos);
    }
    config.debug_force_retraction_max_per_request.reset();
    EXPECT_NO_THROW(config.Validate());
    for (const auto interval : {2, -2}) {
        config.debug_force_retraction_interval = interval;
        config.debug_force_retraction_max_per_request = std::numeric_limits<std::int32_t>::max();
        EXPECT_NO_THROW(config.Validate());
    }
    config.snapshot_allocator.total_pages = 1;
    config.max_retracted_requests = 0;
    EXPECT_THROW(config.Validate(), std::invalid_argument);
}

}  // namespace tokenspeed::test
