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

#include <algorithm>
#include <utility>
#include <variant>

namespace tokenspeed {

EplbController::EplbController() : EplbController(EplbControllerConfig{}) {}

EplbController::EplbController(EplbControllerConfig config) {
    Configure(config);
}

void EplbController::Configure(EplbControllerConfig config) {
    config_ = config;
    step_counter_ = 0;
    last_rebalance_ms_ = 0;
    next_op_id_ = 1;
    last_stats_handle_ = -1;
    last_plan_handle_ = -1;
    consecutive_failures_ = 0;
    layers_pending_.clear();
    pending_ops_.clear();
    state_ =
        !config_.enabled ? EplbState::kDisabled : (config_.warmup_steps > 0 ? EplbState::kWarmingUp : EplbState::kIdle);
}

void EplbController::OnStepCompleted(std::int64_t wall_clock_ms) {
    if (state_ == EplbState::kDisabled) return;
    if (state_ != EplbState::kIdle && state_ != EplbState::kWarmingUp) return;

    ++step_counter_;
    if (step_counter_ <= config_.warmup_steps) {
        state_ = EplbState::kWarmingUp;
        return;
    }
    if (state_ == EplbState::kWarmingUp) {
        state_ = EplbState::kIdle;
    }
    if (!ShouldTriggerByStep() && !ShouldTriggerByTime(wall_clock_ms)) return;
    last_rebalance_ms_ = wall_clock_ms;
    EmitCollect();
}

void EplbController::Advance(const eplb::EplbEvent& event) {
    std::visit([this](auto inner) { this->OnStatsCollected(inner); }, event);
}

std::optional<EplbOperation> EplbController::NextOperation() {
    if (pending_ops_.empty()) return std::nullopt;
    auto op = std::move(pending_ops_.front());
    pending_ops_.pop_front();
    return op;
}

void EplbController::OnStatsCollected(const eplb::StatsCollected& event) {
    if (state_ != EplbState::kCollecting) return;
    last_stats_handle_ = event.stats_handle;
    Emit(EplbPlanOperation{.op_id = next_op_id_++, .stats_handle = event.stats_handle});
    state_ = EplbState::kPlanning;
}

void EplbController::OnStatsEmpty(const eplb::StatsEmpty&) {
    if (state_ == EplbState::kCollecting) {
        ResetFailureCount();
        state_ = EplbState::kIdle;
    }
}

void EplbController::OnPlanDone(eplb::PlanDone event) {
    if (state_ != EplbState::kPlanning) return;
    ResetFailureCount();
    last_plan_handle_ = event.plan_handle;
    std::sort(event.layers_changed.begin(), event.layers_changed.end());
    layers_pending_.assign(event.layers_changed.begin(), event.layers_changed.end());
    if (layers_pending_.empty()) {
        state_ = EplbState::kIdle;
        return;
    }
    EmitRelocateNextSlice();
}

void EplbController::OnPlanFailed(const eplb::PlanFailed&) {
    if (state_ == EplbState::kPlanning) {
        RegisterFailure();
    }
}

void EplbController::OnPlanIdentical(const eplb::PlanIdentical&) {
    if (state_ == EplbState::kPlanning) {
        ResetFailureCount();
        state_ = EplbState::kIdle;
    }
}

void EplbController::OnRelocateDone(const eplb::RelocateDone& event) {
    if (state_ != EplbState::kRelocating) return;
    ResetFailureCount();
    Emit(EplbSwapOperation{.op_id = next_op_id_++, .plan_handle = last_plan_handle_, .layer_ids = event.layer_ids});
    state_ = EplbState::kSwapping;
}

void EplbController::OnRelocateFailed(const eplb::RelocateFailed& event) {
    if (state_ != EplbState::kRelocating) return;
    if (!IsGateFailure(event.reason)) {
        RegisterFailure();
        if (state_ == EplbState::kDisabled) return;
    }
    if (!layers_pending_.empty()) {
        EmitRelocateNextSlice();
    } else {
        state_ = EplbState::kIdle;
    }
}

void EplbController::OnSwapDone(const eplb::SwapDone&) {
    if (state_ != EplbState::kSwapping) return;
    ResetFailureCount();
    if (!layers_pending_.empty()) {
        EmitRelocateNextSlice();
    } else {
        state_ = EplbState::kIdle;
    }
}

void EplbController::Emit(EplbOperation op) {
    pending_ops_.push_back(std::move(op));
}

void EplbController::EmitCollect() {
    Emit(EplbCollectStatsOperation{.op_id = next_op_id_++});
    state_ = EplbState::kCollecting;
}

void EplbController::EmitRelocateNextSlice() {
    std::vector<std::int32_t> layer_ids;
    const std::int32_t limit = config_.max_layers_per_step > 0 ? config_.max_layers_per_step
                                                               : static_cast<std::int32_t>(layers_pending_.size());
    while (!layers_pending_.empty() && static_cast<std::int32_t>(layer_ids.size()) < limit) {
        layer_ids.push_back(layers_pending_.front());
        layers_pending_.pop_front();
    }
    Emit(EplbRelocateOperation{
        .op_id = next_op_id_++, .plan_handle = last_plan_handle_, .layer_ids = std::move(layer_ids)});
    state_ = EplbState::kRelocating;
}

void EplbController::RegisterFailure() {
    ++consecutive_failures_;
    if (config_.max_consecutive_failures > 0 && consecutive_failures_ >= config_.max_consecutive_failures) {
        state_ = EplbState::kDisabled;
        layers_pending_.clear();
        pending_ops_.clear();
    } else {
        state_ = EplbState::kIdle;
    }
}

void EplbController::ResetFailureCount() {
    consecutive_failures_ = 0;
}

bool EplbController::IsGateFailure(const std::string& reason) const {
    return reason.rfind("gate_", 0) == 0;
}

bool EplbController::ShouldTriggerByStep() const {
    const std::int32_t interval = std::max(config_.interval, 1);
    return ((step_counter_ - config_.warmup_steps) % interval) == 0;
}

bool EplbController::ShouldTriggerByTime(std::int64_t wall_clock_ms) const {
    return config_.max_rebalance_period_ms > 0 && wall_clock_ms > 0 &&
           (wall_clock_ms - last_rebalance_ms_) >= config_.max_rebalance_period_ms;
}

void EplbController::OnStatsCollected(const eplb::StatsEmpty& event) {
    OnStatsEmpty(event);
}
void EplbController::OnStatsCollected(eplb::PlanDone event) {
    OnPlanDone(std::move(event));
}
void EplbController::OnStatsCollected(const eplb::PlanFailed& event) {
    OnPlanFailed(event);
}
void EplbController::OnStatsCollected(const eplb::PlanIdentical& event) {
    OnPlanIdentical(event);
}
void EplbController::OnStatsCollected(const eplb::RelocateDone& event) {
    OnRelocateDone(event);
}
void EplbController::OnStatsCollected(const eplb::RelocateFailed& event) {
    OnRelocateFailed(event);
}
void EplbController::OnStatsCollected(const eplb::SwapDone& event) {
    OnSwapDone(event);
}

}  // namespace tokenspeed
