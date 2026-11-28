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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "integration_test_helper.h"
#include "scheduler/prefill_policy.h"

namespace tokenspeed::test {
namespace {

struct PrefillSlice {
    std::string id;
    std::int32_t start;
    std::int32_t end;
    bool operator==(const PrefillSlice&) const = default;
};

std::ostream& operator<<(std::ostream& out, const PrefillSlice& row) {
    return out << row.id << "[" << row.start << ", " << row.end << ")";
}

class PrefillPolicyTest : public SchedulerTestSuite, public ::testing::WithParamInterface<std::tuple<Role, bool>> {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.role = std::get<0>(GetParam());
        cfg.enable_short_prefill_first = std::get<1>(GetParam());
        cfg.prefix_granularity = 128;
        cfg.device_allocator.total_pages = 513;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 8192;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = 8;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Prompt(const std::string& id, std::int32_t tokens, std::int32_t first) {
        Submit(RequestSpec{.request_id = id, .tokens = MakeTokens(tokens, first), .max_new_tokens = 1});
        if (config_.role == Role::kP) {
            ExecutionEvent event;
            event.With(pd::BootstrappedEvent{id});
            scheduler_->Advance(std::move(event));
        }
    }

    void ExpectPromptOrder(bool short_first) {
        Prompt("A", 16384, 10000);
        Prompt("B", 8192, 100000);
        if (short_first) {
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"B", 0, 8192}}));
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 0, 8192}}));
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 8192, 16384}}));
        } else {
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 0, 8192}}));
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 8192, 16384}}));
            EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"B", 0, 8192}}));
        }
    }

    std::vector<PrefillSlice> Round() {
        ExecutionPlan plan = PlanOnce();
        std::vector<PrefillSlice> rows;
        if (const ForwardBatch* batch = FindForwardBatch(plan)) {
            for (std::size_t i = 0; i < batch->NumExtends(); ++i) {
                rows.push_back({batch->request_ids[i], batch->extend_prefix_lens[i] + batch->extend_replay_lens[i],
                                batch->extend_prefix_lens[i] + batch->input_lengths[i]});
                const bool complete = rows.back().end == batch->prefill_lengths[i];
                SendForwardDone(rows.back().id,
                                complete ? std::vector<std::int32_t>{900} : std::vector<std::int32_t>{});
                if (complete && config_.role == Role::kFused) SendFinish(rows.back().id);
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
        return rows;
    }
};

TEST_P(PrefillPolicyTest, ChoosesPromptOrderWhenSchedulerIsConstructed) {
    ExpectPromptOrder(config_.enable_short_prefill_first);
}

TEST_P(PrefillPolicyTest, IncompletePrefillKeepsItsPlaceWhenAShortPromptArrives) {
    Prompt("A", 16384, 10000);
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 0, 8192}}));
    Prompt("B", 1024, 100000);
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 8192, 16384}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"B", 0, 1024}}));
}

class ExplicitPrefillPolicyTest : public PrefillPolicyTest {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = PrefillPolicyTest::MakeConfig();
        cfg.prefill_policy =
            cfg.enable_short_prefill_first ? PrefillPolicyKind::kShortPrefillFirst : PrefillPolicyKind::kFifo;
        cfg.enable_short_prefill_first = false;
        return cfg;
    }
};

TEST_P(ExplicitPrefillPolicyTest, ChoosesPromptOrderWithTheExplicitSelector) {
    ExpectPromptOrder(config_.prefill_policy == PrefillPolicyKind::kShortPrefillFirst);
}

TEST_P(ExplicitPrefillPolicyTest, RejectsTheLegacySelectorAlongsideAnExplicitPolicy) {
    auto cfg = config_;
    ASSERT_NO_THROW(cfg.Validate());
    cfg.enable_short_prefill_first = true;
    EXPECT_THROW(Scheduler{cfg}, std::invalid_argument);
}

TEST_P(ExplicitPrefillPolicyTest, RejectsAnUnknownExplicitPolicy) {
    auto cfg = config_;
    ASSERT_NO_THROW(cfg.Validate());
    cfg.prefill_policy = static_cast<PrefillPolicyKind>(99);
    EXPECT_THROW(cfg.Validate(), std::invalid_argument);
}

TEST_P(ExplicitPrefillPolicyTest, CapacityValidationKeepsPriorityOverAnInvalidPolicy) {
    auto cfg = config_;
    cfg.max_scheduled_tokens = 0;
    cfg.prefill_policy = static_cast<PrefillPolicyKind>(99);
    std::string validation_error;
    try {
        cfg.Validate();
        FAIL() << "expected the invalid capacity to be rejected";
    } catch (const std::invalid_argument& error) {
        validation_error = error.what();
    }
    try {
        Scheduler scheduler{cfg};
        FAIL() << "expected the invalid scheduler configuration to be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_EQ(error.what(), validation_error);
    }
}

class PrefillPeerCarryTest : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.role = Role::kP;
        cfg.prefix_granularity = 128;
        cfg.device_allocator.total_pages = 1025;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 512;
        cfg.max_batch_size = 16;
        cfg.max_resident_requests = 16;
        cfg.prefill_chunk_yield_tokens = 256;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        cfg.peer_config = PeerConfig{.name = "reader:1",
                                     .incarnation = 7,
                                     .max_leases = 8,
                                     .soft_pin_us = 1000000,
                                     .hard_pin_us = 1000000,
                                     .lookup_timeout_us = 1000000,
                                     .min_fetch_tokens = cfg.prefix_granularity,
                                     .landing_tier = PeerLandingTier::kDevice,
                                     .max_host_landing_blocks = 0,
                                     .host_ready_timeout_us = 0};
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Prompt(const std::string& id, std::int32_t tokens, std::int32_t first, const std::string& holder) {
        Submit(RequestSpec{.request_id = id, .peer_holder = holder, .tokens = MakeTokens(tokens, first)});
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(PrefillPeerCarryTest, PeerLandingDoesNotConsumeTheCarryProbeWindow) {
    ASSERT_TRUE(scheduler_->EnableAsyncDevicePeer(true));
    Prompt("peer-A", 8193, 10000, "holder:1");
    EXPECT_TRUE(PlanOnce().peer_fetches.empty());
    const auto lookups = scheduler_->TakePeerLookups();
    ASSERT_EQ(lookups.size(), 1u);

    Prompt("holder-B", 2048, 100000, "");
    // The carry's eight probes start after the local holder. Counting B
    // while selecting A's peer landing would hide the short eighth prompt.
    for (std::int32_t i = 0; i < 7; ++i) {
        Prompt("long-" + std::to_string(i), 2048, 200000 + i * 10000, "");
    }
    Prompt("short-S", 128, 500000, "");
    ExecutionEvent answer;
    answer.With(cache::PeerLookupAnswered{.lease = lookups.front().lease, .num_tokens = 8192, .source_mask = 1});
    scheduler_->Advance(std::move(answer));

    const ExecutionPlan plan = PlanOnce();
    ASSERT_EQ(plan.peer_fetches.size(), 1u);
    EXPECT_EQ(plan.peer_fetches.front().request_id, "peer-A");
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->NumExtends(), 2u);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"short-S", "holder-B"}));
    EXPECT_EQ(batch->extend_prefix_lens, (std::vector<std::int32_t>{0, 0}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{128, 384}));
    // Only the final row remains incomplete; the carried prompt runs whole.
    EXPECT_EQ(batch->input_lengths.front(), batch->prefill_lengths.front());
    EXPECT_LT(batch->input_lengths.back(), batch->prefill_lengths.back());
}

class AlternatingQuotaPrefillPolicy final : public PrefillPolicy {
public:
    AlternatingQuotaPrefillPolicy() : next_quota_{4096} {}

    void Schedule(PrefillSchedulingContext& context) override {
        const auto candidates = context.Candidates();
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            Request* request = candidates[index];
            if (!context.Eligible(*request)) continue;
            const auto admitted = context.Admit(PrefillAllocation{.candidate = {index, request},
                                                                  .max_tokens = next_quota_,
                                                                  .yield_tokens = 0,
                                                                  .completion = PrefillCompletion::kAllowPartial});
            if (admitted.status == PrefillAdmissionStatus::kScheduled) {
                next_quota_ = next_quota_ == 4096 ? 2048 : 4096;
            }
            return;
        }
    }

    void FirstAdmissionSucceeded(std::span<Request* const>, const Request&) override {}

private:
    std::int32_t next_quota_;
};

class EagerQuotaPrefillPolicy final : public PrefillPolicy {
public:
    void Schedule(PrefillSchedulingContext& context) override {
        // A policy may compute its candidate order before admitting anything.
        // Admission must still enforce the existing incomplete-prefill barrier.
        std::vector<PrefillCandidate> pending;
        const auto candidates = context.Candidates();
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            Request* request = candidates[index];
            if (context.Eligible(*request)) pending.push_back({index, request});
        }
        for (const PrefillCandidate candidate : pending) {
            context.Admit(PrefillAllocation{.candidate = candidate,
                                            .max_tokens = 4096,
                                            .yield_tokens = 0,
                                            .completion = PrefillCompletion::kAllowPartial});
        }
    }

    void FirstAdmissionSucceeded(std::span<Request* const>, const Request&) override {}
};

class InjectedPrefillPolicyTest : public PrefillPolicyTest {
protected:
    void SetUp() override {
        config_ = WithHostPackingFromDevice(MakeConfig());
        scheduler_ = std::make_unique<Scheduler>(config_, std::make_unique<AlternatingQuotaPrefillPolicy>());
    }
};

class CheckingPrefillPolicy final : public PrefillPolicy {
public:
    explicit CheckingPrefillPolicy(std::function<void(PrefillSchedulingContext&)> check) : check_{std::move(check)} {}

    void Schedule(PrefillSchedulingContext& context) override { check_(context); }
    void FirstAdmissionSucceeded(std::span<Request* const>, const Request&) override {}

private:
    std::function<void(PrefillSchedulingContext&)> check_;
};

void ExpectSameBudget(const PrefillBudget& actual, const PrefillBudget& expected) {
    EXPECT_EQ(actual.tokens, expected.tokens);
    EXPECT_EQ(actual.kv_pages, expected.kv_pages);
    EXPECT_EQ(actual.request_slots, expected.request_slots);
    EXPECT_EQ(actual.batch_slots, expected.batch_slots);
    EXPECT_EQ(actual.stopped, expected.stopped);
}

class PrefillCandidateMembershipTest : public PrefillPolicyTest {};

TEST_P(PrefillCandidateMembershipTest, RejectsInvalidCandidatesWithoutSpendingAdmissionBudget) {
    Request outside{RequestSpec{.request_id = "outside", .tokens = MakeTokens(128, 500000)}, 128, false};
    scheduler_ = std::make_unique<Scheduler>(
        config_, std::make_unique<CheckingPrefillPolicy>([&](PrefillSchedulingContext& context) {
            const auto candidates = context.Candidates();
            ASSERT_EQ(candidates.size(), 2u);
            const PrefillBudget before = context.Budget();
            const std::vector<PrefillCandidate> invalid{
                {0, nullptr},
                {0, &outside},
                {candidates.size(), candidates.front()},
                {std::numeric_limits<std::size_t>::max(), candidates.front()},
                {1, candidates.front()},
                {0, candidates.back()},
            };
            for (const PrefillCandidate candidate : invalid) {
                SCOPED_TRACE(::testing::Message()
                             << "candidate index " << candidate.index << ", request " << candidate.request);
                EXPECT_THROW(context.Admit(PrefillAllocation{.candidate = candidate,
                                                             .max_tokens = 128,
                                                             .yield_tokens = 0,
                                                             .completion = PrefillCompletion::kAllowPartial}),
                             std::invalid_argument);
                EXPECT_THROW(context.TryPeerLanding(candidate), std::invalid_argument);
                ExpectSameBudget(context.Budget(), before);
            }
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                const PrefillCandidate candidate{index, candidates[index]};
                EXPECT_FALSE(context.TryPeerLanding(candidate));
                const auto admitted = context.Admit(PrefillAllocation{.candidate = candidate,
                                                                      .max_tokens = 128,
                                                                      .yield_tokens = 0,
                                                                      .completion = PrefillCompletion::kAllowPartial});
                EXPECT_EQ(admitted.status, PrefillAdmissionStatus::kScheduled);
            }
        }));
    Prompt("A", 128, 10000);
    Prompt("B", 128, 100000);
    const ExecutionPlan plan = PlanOnce();
    const ForwardBatch* batch = FindForwardBatch(plan);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, (std::vector<std::string>{"A", "B"}));
    EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{128, 128}));
}

class PrefillPreviousRoundCandidateTest : public PrefillPolicyTest {};

TEST_P(PrefillPreviousRoundCandidateTest, RejectsALiveRequestMissingFromTheCurrentRound) {
    PrefillCandidate previous{0, nullptr};
    scheduler_ = std::make_unique<Scheduler>(
        config_, std::make_unique<CheckingPrefillPolicy>([&](PrefillSchedulingContext& context) {
            const auto candidates = context.Candidates();
            ASSERT_EQ(candidates.size(), 1u);
            const PrefillCandidate candidate{0, candidates.front()};
            if (previous.request != nullptr) {
                ASSERT_TRUE(previous.request->Is<fsm::PrefillAwaitingResult>());
                const PrefillBudget before = context.Budget();
                EXPECT_THROW(context.Admit(PrefillAllocation{.candidate = previous,
                                                             .max_tokens = 128,
                                                             .yield_tokens = 0,
                                                             .completion = PrefillCompletion::kAllowPartial}),
                             std::invalid_argument);
                EXPECT_THROW(context.TryPeerLanding(previous), std::invalid_argument);
                ExpectSameBudget(context.Budget(), before);
            } else {
                previous = candidate;
            }
            EXPECT_FALSE(context.TryPeerLanding(candidate));
            const auto admitted = context.Admit(PrefillAllocation{.candidate = candidate,
                                                                  .max_tokens = 128,
                                                                  .yield_tokens = 0,
                                                                  .completion = PrefillCompletion::kAllowPartial});
            EXPECT_EQ(admitted.status, PrefillAdmissionStatus::kScheduled);
        }));
    Prompt("A", 128, 10000);
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* first_batch = FindForwardBatch(first);
    ASSERT_NE(first_batch, nullptr);
    EXPECT_EQ(first_batch->request_ids, (std::vector<std::string>{"A"}));
    // Keep A alive with its result in flight while B becomes the next candidate.
    Prompt("B", 128, 100000);
    const ExecutionPlan second = PlanOnce();
    const ForwardBatch* second_batch = FindForwardBatch(second);
    ASSERT_NE(second_batch, nullptr);
    EXPECT_EQ(second_batch->request_ids, (std::vector<std::string>{"B"}));
    EXPECT_EQ(second_batch->input_lengths, (std::vector<std::int32_t>{128}));
}

class PrefillDeepQueueCapacityTest : public SchedulerTestSuite {
protected:
    static void ExpectNoPrefill(const ExecutionPlan& plan) {
        const ForwardBatch* batch = FindForwardBatch(plan);
        EXPECT_TRUE(batch == nullptr || batch->NumExtends() == 0);
    }

    SchedulerConfig MakeConfig() override {
        SchedulerConfig cfg{};
        cfg.role = Role::kP;
        cfg.prefix_granularity = 128;
        cfg.device_allocator.total_pages = 5;
        cfg.disable_l2_cache = true;
        cfg.max_scheduled_tokens = 512;
        cfg.max_batch_size = 8;
        cfg.max_resident_requests = 8;
        cfg.prefill_policy = PrefillPolicyKind::kFifo;
        cfg.cache_groups.push_back(CacheGroupConfig{
            .group_id = "full_attention",
            .block_granularity = cfg.prefix_granularity,
            .total_pages = cfg.device_allocator.total_pages,
            .attn_kind = AttnKind::kFull,
            .transfer_policy = CacheTransferPolicy::FullSuffix,
        });
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Prompt(const std::string& id, std::int32_t first) {
        Submit(RequestSpec{.request_id = id, .tokens = MakeTokens(128, first)});
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{id});
        scheduler_->Advance(std::move(event));
    }

    void AckTransfer(const std::string& id) {
        ExecutionEvent event;
        event.With(pd::SucceededEvent{id});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(PrefillDeepQueueCapacityTest, TransferAcknowledgementsResumeADeepQueueInFifoOrder) {
    Prompt("holder-A", 10000);
    Prompt("holder-B", 20000);
    const ExecutionPlan holders = PlanOnce();
    const ForwardBatch* holders_batch = FindForwardBatch(holders);
    ASSERT_NE(holders_batch, nullptr);
    ASSERT_EQ(holders_batch->request_ids, (std::vector<std::string>{"holder-A", "holder-B"}));
    SendForwardDone("holder-A", {900});
    SendForwardDone("holder-B", {900});
    const ExecutionPlan transfers = PlanOnce();
    ASSERT_TRUE(transfers.remote_decode);
    EXPECT_EQ(transfers.remote_decode->request_ids, (std::vector<std::string>{"holder-A", "holder-B"}));

    constexpr std::int32_t kQueueSize = 257;
    for (std::int32_t i = 0; i < kQueueSize; ++i) {
        Prompt("queued-" + std::to_string(i), 100000 + i * 256);
    }
    ExpectNoPrefill(PlanOnce());
    AckTransfer("holder-A");
    const ExecutionPlan first = PlanOnce();
    const ForwardBatch* first_batch = FindForwardBatch(first);
    ASSERT_NE(first_batch, nullptr);
    ASSERT_EQ(first_batch->request_ids, (std::vector<std::string>{"queued-0"}));
    SendForwardDone("queued-0", {900});
    const ExecutionPlan first_transfer = PlanOnce();
    ASSERT_TRUE(first_transfer.remote_decode);
    EXPECT_EQ(first_transfer.remote_decode->request_ids, (std::vector<std::string>{"queued-0"}));
    ExpectNoPrefill(first_transfer);
    AckTransfer("holder-B");
    AckTransfer("queued-0");

    for (std::int32_t next = 1; next < kQueueSize;) {
        const ExecutionPlan plan = PlanOnce();
        const ForwardBatch* batch = FindForwardBatch(plan);
        ASSERT_NE(batch, nullptr) << "next queued request " << next;
        ASSERT_EQ(batch->NumExtends(), 2u);
        EXPECT_EQ(batch->request_ids,
                  (std::vector<std::string>{"queued-" + std::to_string(next), "queued-" + std::to_string(next + 1)}));
        EXPECT_EQ(batch->input_lengths, (std::vector<std::int32_t>{128, 128}));
        for (const std::string& id : batch->request_ids) SendForwardDone(id, {900});
        const ExecutionPlan transfer = PlanOnce();
        ASSERT_TRUE(transfer.remote_decode);
        EXPECT_EQ(transfer.remote_decode->request_ids, batch->request_ids);
        ExpectNoPrefill(transfer);
        for (const std::string& id : transfer.remote_decode->request_ids) AckTransfer(id);
        next += 2;
    }
    ExpectNoPrefill(PlanOnce());
}

TEST_P(InjectedPrefillPolicyTest, AStatefulPolicyKeepsItsQuotaAcrossSerialRounds) {
    Prompt("A", 16384, 10000);
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 0, 4096}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 4096, 6144}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 6144, 10240}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 10240, 12288}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 12288, 16384}}));
}

TEST_P(InjectedPrefillPolicyTest, AnIncompletePrefillStillPreventsASecondPartialAdmission) {
    scheduler_ = std::make_unique<Scheduler>(config_, std::make_unique<EagerQuotaPrefillPolicy>());
    Prompt("A", 16384, 10000);
    Prompt("B", 8192, 100000);
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 0, 4096}}));
    EXPECT_EQ(Round(), (std::vector<PrefillSlice>{{"A", 4096, 8192}}));
}

INSTANTIATE_TEST_SUITE_P(LocalRoles, PrefillPolicyTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(LocalRoles, ExplicitPrefillPolicyTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(LocalRoles, InjectedPrefillPolicyTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Values(false)));
INSTANTIATE_TEST_SUITE_P(LocalRoles, PrefillCandidateMembershipTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Values(false)));
INSTANTIATE_TEST_SUITE_P(PrefillWorker, PrefillPreviousRoundCandidateTest,
                         ::testing::Combine(::testing::Values(Role::kP), ::testing::Values(false)));

}  // namespace
}  // namespace tokenspeed::test
