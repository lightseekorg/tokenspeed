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

#include "scheduler/scheduler.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include <spdlog/spdlog.h>

#include "fsm/forward_states.h"
#include "utils.h"

namespace tokenspeed {

Scheduler::PrefillContext::PrefillContext(Scheduler& scheduler, AdmissionFeedback& feedback, PlanBuild& build,
                                          std::span<Request* const> candidates, bool new_prompts_sealed,
                                          std::int32_t decode_reserve)
    : scheduler_{scheduler},
      feedback_{feedback},
      build_{build},
      candidates_{candidates},
      new_prompts_sealed_{new_prompts_sealed},
      decode_reserve_{decode_reserve} {}

std::span<Request* const> Scheduler::PrefillContext::Candidates() const {
    return candidates_;
}
const SchedulerConfig& Scheduler::PrefillContext::Config() const {
    return scheduler_.config_;
}

PrefillBudget Scheduler::PrefillContext::Budget() const {
    return {.tokens = build_.token_budget,
            .kv_pages = build_.kv_page_budget,
            .request_slots = scheduler_.req_pool_allocator_.AvailableSlots(),
            .batch_slots = Config().max_batch_size - CheckedInt32(build_.operations.size()),
            .stopped = build_.Full(Config().max_batch_size) || scheduler_.lone_requests_.ActedInPlan()};
}

bool Scheduler::PrefillContext::Eligible(const Request& request) const {
    if (build_.Scheduled(request)) return false;
    if (request.Is<fsm::Prefilling>()) return true;
    return request.Is<fsm::Submitted>() && !new_prompts_sealed_ && Budget().request_slots > 0;
}

bool Scheduler::PrefillContext::containsCandidate(PrefillCandidate candidate) const {
    return candidate.request != nullptr && candidate.index < candidates_.size() &&
           candidates_[candidate.index] == candidate.request;
}

bool Scheduler::PrefillContext::TryPeerLanding(PrefillCandidate candidate) {
    if (!containsCandidate(candidate)) {
        throw std::invalid_argument("peer landing must name a round candidate");
    }
    Request& request = *candidate.request;
    if (Budget().stopped || !Eligible(request) || serial_holder_ ||
        (blocked_resident_ && blocked_resident_ != &request)) {
        return true;
    }
    const auto* submitted = request.GetIf<fsm::Submitted>();
    if (submitted == nullptr || !submitted->admission.peer_hit) return false;
    if (!scheduler_.adaptive_device_peer_) return true;
    Request* request_ptr = &request;
    scheduler_.schedulePeerLandings(feedback_, build_, std::span(&request_ptr, 1), new_prompts_sealed_);
    return build_.Scheduled(request);
}

bool Scheduler::PrefillContext::PrefillAllowed(const Request& request) const {
    return scheduler_.prefillAllowed(request, build_);
}

std::int32_t Scheduler::PrefillContext::EstimateRemaining(Request& request) {
    if (request.Is<fsm::Prefilling>()) return request.UnscheduledPrefillSize();
    const auto match = scheduler_.matchPrefixAtAdmission(&request);
    if (match.waits_for_eviction_store) return std::numeric_limits<std::int32_t>::max();
    const auto hit = std::max(match.probe.device.num_common_tokens, match.probe.host.num_common_tokens);
    return CheckedInt32(std::int64_t{request.PrefillSize()} - hit + scheduler_.coordinator_.ReplayTokens(hit));
}

std::optional<std::int32_t> Scheduler::PrefillContext::WholePromptTokens(Request& request, Request* resident_holder) {
    bool has_state_group = false;
    for (std::int32_t i = 0; i < scheduler_.coordinator_.NumGroups(); ++i) {
        has_state_group = has_state_group || scheduler_.coordinator_.GroupIsState(static_cast<std::size_t>(i));
    }
    return scheduler_.wholePromptTokens(request, resident_holder, has_state_group);
}

PrefillAdmission Scheduler::PrefillContext::Admit(const PrefillAllocation& allocation) {
    if (!containsCandidate(allocation.candidate)) {
        throw std::invalid_argument("prefill allocation must name a round candidate");
    }
    if (allocation.max_tokens < 0 || allocation.max_tokens > build_.token_budget || allocation.yield_tokens < 0) {
        throw std::invalid_argument("prefill allocation must fit the remaining token budget");
    }
    if (allocation.completion != PrefillCompletion::kAllowPartial &&
        allocation.completion != PrefillCompletion::kRequireComplete) {
        throw std::invalid_argument("unknown prefill completion constraint");
    }
    if (Budget().stopped) return {PrefillAdmissionStatus::kStopped};
    Request& request = *allocation.candidate.request;
    if (!Eligible(request) || allocation.max_tokens == 0) return {PrefillAdmissionStatus::kUnavailable};
    const bool first = request.Is<fsm::Submitted>();
    const bool must_complete = allocation.completion == PrefillCompletion::kRequireComplete;
    if (!first && must_complete) return {PrefillAdmissionStatus::kUnavailable};
    // A failed resident owns the capacity priority. A retry of that same
    // request may still proceed without letting another request overtake it.
    if (blocked_resident_ && blocked_resident_ != &request) return {PrefillAdmissionStatus::kStopped};
    if (serial_holder_ && !(Config().role == Role::kP && first && must_complete)) {
        return {PrefillAdmissionStatus::kStopped};
    }
    if (!scheduler_.prefillAllowed(request, build_)) return {PrefillAdmissionStatus::kStopped};
    feedback_.admission_failed = false;
    feedback_.chunk_gave_way = false;
    auto operation = scheduler_.schedulePrefillCandidate(build_.plan, feedback_, &request, allocation.max_tokens,
                                                         decode_reserve_, build_.load_backs, build_.kv_page_budget,
                                                         allocation.yield_tokens, must_complete);
    if (!operation) {
        if (!feedback_.admission_failed) return {PrefillAdmissionStatus::kUnavailable};
        if (!must_complete) spdlog::debug("[Scheduler] prefill {} blocked on capacity", request.Id());
        feedback_.NoteCapacityBlocked(&request);
        if (!first) blocked_resident_ = &request;
        return {PrefillAdmissionStatus::kCapacityBlocked};
    }
    if (first) scheduler_.prefill_policy_->FirstAdmissionSucceeded(candidates_, request);
    if (!must_complete) {
        spdlog::debug("[Scheduler] prefill {} scheduled ({} tokens)", request.Id(), operation->input_length);
    }
    const PrefillAdmission result{.status = PrefillAdmissionStatus::kScheduled,
                                  .input_tokens = operation->input_length,
                                  .incomplete = request.Is<fsm::Prefilling>(),
                                  .chunk_gave_way = feedback_.chunk_gave_way};
    scheduler_.pushOperation(build_, request, std::move(*operation));
    request.TrackScheduledForward();
    if (serial_holder_) {
        // P carry/fill admits the partial holder first, but executes complete
        // prompts before it, preserving the existing output prefix contract.
        const auto holder = build_.operations.begin() + static_cast<std::ptrdiff_t>(serial_holder_row_);
        std::rotate(holder, holder + 1, build_.operations.end());
        ++serial_holder_row_;
    }
    if (result.incomplete) {
        serial_holder_ = &request;
        serial_holder_row_ = build_.operations.size() - 1;
    }
    return result;
}

}  // namespace tokenspeed
