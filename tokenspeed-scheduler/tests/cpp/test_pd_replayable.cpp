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
#include <cstdint>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "integration_test_helper.h"

namespace tokenspeed::test {
struct LandingCase {
    std::string name;
    bool v41;
    bool replayable;
    bool disable_prefix_cache;
    std::int32_t hit;
    std::int32_t prompt;
    std::int32_t decode_width;
    std::int32_t sink_tokens{0};
};

void PrintTo(const LandingCase& c, std::ostream* out) {
    *out << c.name;
}

class PdReplayableLandingTestSuite : public SchedulerTestSuite, public ::testing::WithParamInterface<LandingCase> {
protected:
    SchedulerConfig MakeConfig() override {
        const auto& c = GetParam();
        SchedulerConfig cfg{};
        cfg.role = Role::kD;
        cfg.prefix_granularity = c.v41 ? 256 : 8;
        cfg.device_allocator.total_pages = 4097;
        cfg.host_allocator.total_pages = 0;
        cfg.disable_l2_cache = true;
        cfg.snapshot_allocator.total_pages = 1;
        cfg.max_retracted_requests = 0;
        cfg.max_scheduled_tokens = 1024;
        cfg.max_batch_size = 4;
        cfg.max_resident_requests = cfg.max_batch_size;
        cfg.decode_input_tokens = c.decode_width;
        cfg.disable_prefix_cache = c.disable_prefix_cache;
        const auto add = [&](std::string id, std::int32_t grain, AttnKind kind, std::optional<std::int32_t> window,
                             bool replayable) {
            cfg.cache_groups.push_back(CacheGroupConfig{
                .group_id = std::move(id),
                .block_granularity = grain,
                .total_pages = cfg.device_allocator.total_pages,
                .attn_kind = kind,
                .sliding_window_tokens = window,
                .replayable = replayable,
                .transfer_policy = CacheTransferPolicy::FullSuffix,
            });
        };
        add("full", c.v41 ? 64 : 4, AttnKind::kFull, std::nullopt, false);
        add("swa", c.v41 ? 64 : 8, AttnKind::kSlidingWindow, c.v41 ? 128 : 16, c.replayable);
        cfg.cache_groups[1].sink_tokens = c.sink_tokens;
        if (c.v41) {
            add("global_r2", 128, AttnKind::kFull, std::nullopt, false);
            add("compressor_tail", 2, AttnKind::kSlidingWindow, 2, true);
        }
        return cfg;
    }

    void SubmitPrompt(const std::string& id, std::int32_t size) {
        std::vector<std::int32_t> tokens(size);
        std::iota(tokens.begin(), tokens.end(), 10);
        Submit(RequestSpec{.request_id = id, .tokens = tokens, .max_new_tokens = 16});
        scheduler_->Advance(ExecutionEvent{}.With(pd::BootstrappedEvent{id}));
    }

    void Land(const std::string& id) {
        scheduler_->Advance(ExecutionEvent{}.With(pd::RemotePrefillDoneEvent{id, 30000}));
    }
};

TEST_P(PdReplayableLandingTestSuite, RetainedWindowHasRealLandingPages) {
    const auto& c = GetParam();
    if (c.hit > 0) {
        SubmitPrompt("warm", c.hit);
        const auto warm = PlanOnce();
        ASSERT_TRUE(warm.remote_prefill.has_value());
        EXPECT_EQ(warm.remote_prefill->extend_prefix_lens.at(0), 0);
        Land("warm");
        SendFinish("warm");
        PlanOnce();
    }
    SubmitPrompt("reuse", c.prompt);
    SCOPED_TRACE(c.name);
    ExecutionPlan plan;
    ASSERT_NO_THROW(plan = PlanOnce());
    ASSERT_TRUE(plan.remote_prefill.has_value());
    const auto& remote = *plan.remote_prefill;
    ASSERT_EQ(remote.extend_prefix_lens.at(0), c.disable_prefix_cache ? 0 : c.hit);
    EXPECT_EQ(remote.extend_replay_lens.at(0), 0);
    for (const auto& group : config_.cache_groups) {
        if (group.attn_kind == AttnKind::kFull) {
            const auto full_row = remote.BlockTables().at(group.group_id).at(0);
            const auto end = (c.prompt + group.block_granularity - 1) / group.block_granularity;
            ASSERT_GE(full_row.size(), static_cast<std::size_t>(end));
            for (auto slot = 0; slot < end; ++slot) {
                EXPECT_GT(full_row.at(slot), 0) << group.group_id << " required logical slot " << slot;
            }
            continue;
        }
        const auto row = remote.BlockTables().at(group.group_id).at(0);
        const auto first = std::max(0, c.prompt - *group.sliding_window_tokens + 1) / group.block_granularity;
        const auto end = (c.prompt + c.decode_width + group.block_granularity - 1) / group.block_granularity;
        ASSERT_GE(row.size(), static_cast<std::size_t>(end));
        for (std::int32_t slot = 0; slot < (group.sink_tokens + group.block_granularity - 1) / group.block_granularity;
             ++slot) {
            EXPECT_GT(row.at(slot), 0) << "sink logical slot " << slot;
        }
        if (group.replayable) {
            for (std::int32_t slot = 0; slot < first; ++slot) {
                EXPECT_EQ(row.at(slot), 0) << group.group_id << " expired logical slot " << slot;
            }
        }
        for (auto slot = first; slot < end; ++slot) {
            EXPECT_GT(row.at(slot), 0) << group.group_id << " required logical slot " << slot;
        }
    }
    if (::testing::Test::HasFailure()) return;
    Land("reuse");
    const auto decode = PlanOnce();
    const auto* batch = FindForwardBatch(decode);
    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->request_ids, std::vector<std::string>{"reuse"});
    SendForwardDone("reuse", {30001});
    SendFinish("reuse");
    PlanOnce();
}

INSTANTIATE_TEST_SUITE_P(Cases, PdReplayableLandingTestSuite,
                         ::testing::Values(LandingCase{"sink_cold", false, false, false, 0, 128, 1, 2},
                                           LandingCase{"sink_overlap", false, false, false, 0, 6, 1, 2},
                                           LandingCase{"sink_hit", false, false, false, 16, 128, 1, 2},
                                           LandingCase{"sink_tail_hit", false, false, false, 13, 128, 1, 2},
                                           LandingCase{"sink_disabled", false, false, true, 0, 128, 4, 2},
                                           LandingCase{"cold", false, true, false, 0, 6, 1},
                                           LandingCase{"tail_same_block", false, true, false, 5, 6, 1},
                                           LandingCase{"tail_cross_block", false, true, false, 5, 9, 1},
                                           LandingCase{"aligned_hit", false, true, false, 8, 9, 1},
                                           LandingCase{"window_past_hit", false, true, false, 5, 32, 1},
                                           LandingCase{"prefix_disabled", false, true, true, 5, 6, 1},
                                           LandingCase{"cached_window_control", false, false, false, 13, 14, 1},
                                           LandingCase{"v41_tail_exception", true, true, false, 257, 258, 1},
                                           LandingCase{"v41_tail_hole", true, true, false, 257, 320, 1},
                                           LandingCase{"v41_aligned_hit", true, true, false, 256, 320, 1},
                                           LandingCase{"v41_spec_width4", true, true, false, 257, 260, 4},
                                           LandingCase{"v41_window_past_hit", true, true, false, 256, 512, 1}),
                         [](const ::testing::TestParamInfo<LandingCase>& info) { return info.param.name; });
}  // namespace tokenspeed::test
