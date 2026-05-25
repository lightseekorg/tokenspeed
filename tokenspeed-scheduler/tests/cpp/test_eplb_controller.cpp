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

#include "scheduler/eplb/eplb_controller.h"

#include <gtest/gtest.h>

#include <optional>
#include <variant>
#include <vector>

namespace tokenspeed::test {
namespace {

EplbControllerConfig MakeConfig() {
    EplbControllerConfig cfg;
    cfg.enabled = true;
    cfg.warmup_steps = 0;
    cfg.interval = 2;
    cfg.max_layers_per_step = 0;
    cfg.max_rebalance_period_ms = 0;
    cfg.planner_timeout_ms = 5000;
    cfg.max_consecutive_failures = 3;
    return cfg;
}

template <typename T>
const T* Get(const std::optional<EplbOperation>& op) {
    if (!op.has_value()) return nullptr;
    return std::get_if<T>(&*op);
}

std::optional<EplbOperation> Pop(EplbController& ctl) {
    return ctl.NextOperation();
}

void DriveStep(EplbController& ctl, int count = 1) {
    for (int i = 0; i < count; ++i) ctl.OnStepCompleted(/*wall_clock_ms=*/0);
}

}  // namespace

TEST(EplbControllerTest, WarmupSkipped) {
    auto cfg = MakeConfig();
    cfg.warmup_steps = 100;
    cfg.interval = 1;
    EplbController ctl(cfg);

    DriveStep(ctl, 100);
    EXPECT_FALSE(Pop(ctl).has_value());
    EXPECT_EQ(ctl.State(), EplbState::kWarmingUp);

    DriveStep(ctl);
    auto op = Pop(ctl);
    ASSERT_NE(Get<EplbCollectStatsOperation>(op), nullptr);
    EXPECT_EQ(ctl.State(), EplbState::kCollecting);
}

TEST(EplbControllerTest, HappyPath) {
    auto cfg = MakeConfig();
    cfg.interval = 2;
    EplbController ctl(cfg);

    DriveStep(ctl);
    EXPECT_FALSE(Pop(ctl).has_value());
    DriveStep(ctl);
    auto collect = Pop(ctl);
    ASSERT_NE(Get<EplbCollectStatsOperation>(collect), nullptr);
    EXPECT_EQ(Get<EplbCollectStatsOperation>(collect)->op_id, 1);

    ctl.Advance(eplb::StatsCollected{.op_id = 1, .stats_handle = 11, .total_count = 42.0});
    auto plan = Pop(ctl);
    ASSERT_NE(Get<EplbPlanOperation>(plan), nullptr);
    EXPECT_EQ(Get<EplbPlanOperation>(plan)->stats_handle, 11);
    EXPECT_EQ(ctl.State(), EplbState::kPlanning);

    ctl.Advance(eplb::PlanDone{.op_id = 2,
                               .plan_handle = 22,
                               .layers_changed = {5, 1, 3},
                               .balancedness_before = 0.5,
                               .balancedness_pred = 0.9});
    auto relocate = Pop(ctl);
    ASSERT_NE(Get<EplbRelocateOperation>(relocate), nullptr);
    EXPECT_EQ(Get<EplbRelocateOperation>(relocate)->plan_handle, 22);
    EXPECT_EQ(Get<EplbRelocateOperation>(relocate)->layer_ids, (std::vector<std::int32_t>{1, 3, 5}));
    EXPECT_EQ(ctl.State(), EplbState::kRelocating);

    ctl.Advance(eplb::RelocateDone{.op_id = 3, .layer_ids = {1, 3, 5}});
    auto swap = Pop(ctl);
    ASSERT_NE(Get<EplbSwapOperation>(swap), nullptr);
    EXPECT_EQ(Get<EplbSwapOperation>(swap)->plan_handle, 22);
    EXPECT_EQ(Get<EplbSwapOperation>(swap)->layer_ids, (std::vector<std::int32_t>{1, 3, 5}));
    EXPECT_EQ(ctl.State(), EplbState::kSwapping);

    ctl.Advance(eplb::SwapDone{.op_id = 4, .layer_ids = {1, 3, 5}, .blocked_us = 7});
    EXPECT_EQ(ctl.State(), EplbState::kIdle);
    EXPECT_FALSE(Pop(ctl).has_value());
}

TEST(EplbControllerTest, EmptyAndIdenticalReturnIdle) {
    EplbController ctl(MakeConfig());
    DriveStep(ctl, 2);
    ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::StatsEmpty{.op_id = 1});
    EXPECT_EQ(ctl.State(), EplbState::kIdle);

    DriveStep(ctl, 2);
    ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::StatsCollected{.op_id = 2, .stats_handle = 13, .total_count = 1.0});
    ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::PlanIdentical{.op_id = 3});
    EXPECT_EQ(ctl.State(), EplbState::kIdle);
    EXPECT_FALSE(Pop(ctl).has_value());
}

TEST(EplbControllerTest, PlanFailedCountingDisables) {
    EplbController ctl(MakeConfig());
    for (int i = 0; i < 3; ++i) {
        DriveStep(ctl, 2);
        ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
        ctl.Advance(eplb::StatsCollected{.op_id = 10 + i, .stats_handle = 20 + i, .total_count = 1.0});
        ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
        ctl.Advance(eplb::PlanFailed{.op_id = 30 + i, .reason = "timeout"});
    }
    EXPECT_EQ(ctl.State(), EplbState::kDisabled);
    DriveStep(ctl, 20);
    EXPECT_FALSE(Pop(ctl).has_value());
}

TEST(EplbControllerTest, GateFailureNotCounted) {
    EplbController ctl(MakeConfig());
    DriveStep(ctl, 2);
    ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::StatsCollected{.op_id = 1, .stats_handle = 2, .total_count = 1.0});
    ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::PlanDone{
        .op_id = 2, .plan_handle = 3, .layers_changed = {1}, .balancedness_before = 0.4, .balancedness_pred = 0.8});
    ASSERT_NE(Get<EplbRelocateOperation>(Pop(ctl)), nullptr);
    for (int i = 0; i < 100; ++i) {
        ctl.Advance(eplb::RelocateFailed{.op_id = 4 + i, .layer_id = 1, .reason = "gate_prefetch_busy"});
        EXPECT_EQ(ctl.State(), EplbState::kIdle);
        DriveStep(ctl, 2);
        ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
        ctl.Advance(eplb::StatsCollected{.op_id = 200 + i, .stats_handle = 300 + i, .total_count = 1.0});
        ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
        ctl.Advance(eplb::PlanDone{.op_id = 400 + i,
                                   .plan_handle = 500 + i,
                                   .layers_changed = {1},
                                   .balancedness_before = 0.4,
                                   .balancedness_pred = 0.8});
        ASSERT_NE(Get<EplbRelocateOperation>(Pop(ctl)), nullptr);
    }
    EXPECT_EQ(ctl.State(), EplbState::kRelocating);
    EXPECT_EQ(ctl.ConsecutiveFailures(), 0);
}

TEST(EplbControllerTest, RollingLayers) {
    auto cfg = MakeConfig();
    cfg.max_layers_per_step = 4;
    EplbController ctl(cfg);
    DriveStep(ctl, 2);
    ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::StatsCollected{.op_id = 1, .stats_handle = 2, .total_count = 1.0});
    ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::PlanDone{.op_id = 2,
                               .plan_handle = 7,
                               .layers_changed = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
                               .balancedness_before = 0.2,
                               .balancedness_pred = 0.8});

    auto r1 = Pop(ctl);
    ASSERT_NE(Get<EplbRelocateOperation>(r1), nullptr);
    EXPECT_EQ(Get<EplbRelocateOperation>(r1)->layer_ids, (std::vector<std::int32_t>{1, 2, 3, 4}));
    ctl.Advance(eplb::RelocateDone{.op_id = 3, .layer_ids = {1, 2, 3, 4}});
    auto s1 = Pop(ctl);
    ASSERT_NE(Get<EplbSwapOperation>(s1), nullptr);
    EXPECT_EQ(Get<EplbSwapOperation>(s1)->layer_ids, (std::vector<std::int32_t>{1, 2, 3, 4}));

    ctl.Advance(eplb::SwapDone{.op_id = 4, .layer_ids = {1, 2, 3, 4}, .blocked_us = 0});
    auto r2 = Pop(ctl);
    ASSERT_NE(Get<EplbRelocateOperation>(r2), nullptr);
    EXPECT_EQ(Get<EplbRelocateOperation>(r2)->layer_ids, (std::vector<std::int32_t>{5, 6, 7, 8}));

    ctl.Advance(eplb::RelocateDone{.op_id = 5, .layer_ids = {5, 6, 7, 8}});
    ASSERT_NE(Get<EplbSwapOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::SwapDone{.op_id = 6, .layer_ids = {5, 6, 7, 8}, .blocked_us = 0});
    auto r3 = Pop(ctl);
    ASSERT_NE(Get<EplbRelocateOperation>(r3), nullptr);
    EXPECT_EQ(Get<EplbRelocateOperation>(r3)->layer_ids, (std::vector<std::int32_t>{9, 10}));
    ctl.Advance(eplb::RelocateDone{.op_id = 7, .layer_ids = {9, 10}});
    ASSERT_NE(Get<EplbSwapOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::SwapDone{.op_id = 8, .layer_ids = {9, 10}, .blocked_us = 0});
    EXPECT_EQ(ctl.State(), EplbState::kIdle);
}

TEST(EplbControllerTest, DisabledAtConstruction) {
    auto cfg = MakeConfig();
    cfg.enabled = false;
    EplbController ctl(cfg);
    EXPECT_EQ(ctl.State(), EplbState::kDisabled);
    DriveStep(ctl, 100);
    EXPECT_FALSE(Pop(ctl).has_value());
}

TEST(EplbControllerTest, ConcurrencyGuard) {
    EplbController ctl(MakeConfig());
    DriveStep(ctl, 2);
    ASSERT_NE(Get<EplbCollectStatsOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::StatsCollected{.op_id = 1, .stats_handle = 2, .total_count = 1.0});
    ASSERT_NE(Get<EplbPlanOperation>(Pop(ctl)), nullptr);
    ctl.Advance(eplb::PlanDone{
        .op_id = 2, .plan_handle = 3, .layers_changed = {1}, .balancedness_before = 0.2, .balancedness_pred = 0.8});
    ASSERT_NE(Get<EplbRelocateOperation>(Pop(ctl)), nullptr);
    DriveStep(ctl, 100);
    EXPECT_FALSE(Pop(ctl).has_value());
    EXPECT_EQ(ctl.State(), EplbState::kRelocating);
}

}  // namespace tokenspeed::test
