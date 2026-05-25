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

#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "fsm/eplb_states.h"
#include "scheduler/eplb/config.h"
#include "scheduler/operations/eplb.h"
#include "scheduler/outside_events/eplb.h"

namespace tokenspeed {

class EplbController {
public:
    EplbController();
    explicit EplbController(EplbControllerConfig config);

    void Configure(EplbControllerConfig config);
    void OnStepCompleted(std::int64_t wall_clock_ms);
    void Advance(const eplb::EplbEvent& event);
    std::optional<EplbOperation> NextOperation();

    EplbState State() const { return state_; }
    std::int64_t StepCounter() const { return step_counter_; }
    std::int32_t ConsecutiveFailures() const { return consecutive_failures_; }

private:
    void OnStatsCollected(const eplb::StatsCollected& event);
    void OnStatsEmpty(const eplb::StatsEmpty& event);
    void OnPlanDone(eplb::PlanDone event);
    void OnPlanFailed(const eplb::PlanFailed& event);
    void OnPlanIdentical(const eplb::PlanIdentical& event);
    void OnRelocateDone(const eplb::RelocateDone& event);
    void OnRelocateFailed(const eplb::RelocateFailed& event);
    void OnSwapDone(const eplb::SwapDone& event);
    void OnStatsCollected(const eplb::StatsEmpty& event);
    void OnStatsCollected(eplb::PlanDone event);
    void OnStatsCollected(const eplb::PlanFailed& event);
    void OnStatsCollected(const eplb::PlanIdentical& event);
    void OnStatsCollected(const eplb::RelocateDone& event);
    void OnStatsCollected(const eplb::RelocateFailed& event);
    void OnStatsCollected(const eplb::SwapDone& event);

    void Emit(EplbOperation op);
    void EmitCollect();
    void EmitRelocateNextSlice();
    void RegisterFailure();
    void ResetFailureCount();
    bool IsGateFailure(const std::string& reason) const;
    bool ShouldTriggerByStep() const;
    bool ShouldTriggerByTime(std::int64_t wall_clock_ms) const;

    EplbControllerConfig config_{};
    EplbState state_{EplbState::kDisabled};
    std::int64_t step_counter_{0};
    std::int64_t last_rebalance_ms_{0};
    std::int32_t next_op_id_{1};
    std::int32_t last_stats_handle_{-1};
    std::int32_t last_plan_handle_{-1};
    std::int32_t consecutive_failures_{0};
    std::deque<std::int32_t> layers_pending_;
    std::deque<EplbOperation> pending_ops_;
};

}  // namespace tokenspeed
