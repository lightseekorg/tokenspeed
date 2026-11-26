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

#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "integration_test_helper.h"

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

INSTANTIATE_TEST_SUITE_P(LocalRoles, PrefillPolicyTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Bool()));
INSTANTIATE_TEST_SUITE_P(LocalRoles, ExplicitPrefillPolicyTest,
                         ::testing::Combine(::testing::Values(Role::kP, Role::kFused), ::testing::Bool()));

}  // namespace
}  // namespace tokenspeed::test
