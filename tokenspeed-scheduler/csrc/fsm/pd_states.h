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
#include <memory>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "core/token_container.h"
#include "fsm/forward_states.h"
#include "resource/allocator/req_pool_allocator.h"
#include "scheduler/request_spec.h"

// States that exist only under PD disaggregation: a fused engine never
// enters any of them. The universal lifecycle lives in forward_states.h;
// these extend it with the P and D roles' parking states.

namespace tokenspeed {
namespace fsm {
struct Bootstrapping {
    Bootstrapping(TokenContainer* _token_container, std::int32_t _prefix_granularity)
        : admission{_token_container, _prefix_granularity} {}
    explicit Bootstrapping(AdmissionResources resources) : admission{std::move(resources)} {}

    AdmissionResources admission;
};

// The peer node is prefilling this prompt: this engine admitted and holds
// the destination pages, and nothing here is schedulable work -- the state
// advances only when RemotePrefillDoneEvent lands with the bootstrap token.
// A state of its own, not Prefilling with a source flag: the two share the
// data layout (hence the inheritance) but no lifecycle.
struct RemotePrefilling : public Prefilling {
    using Prefilling::Prefilling;
};

// P role: every chunk is scheduled, but the FINAL chunk's result has not
// landed yet -- and the remote decode that hands this prompt to the peer
// needs the bootstrap token that arrives with it. Nothing is schedulable
// here; the only event this state accepts is that result, which turns it
// into PrefillDone.
//
// A sibling of PrefillDone, not a subclass: `Is<PrefillDone>()` must mean
// "schedulable" with no exceptions, and inheritance would make the answer
// depend on how the caller dispatches (holds_alternative vs derived_from vs
// an overload set).
struct PrefillAwaitingResult {
    PrefillAwaitingResult(ForwardResources resources, TokenContainer::Window window,
                          std::int32_t reserve_num_tokens_in_next_schedule_event)
        : resources{std::move(resources)},
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    PrefillInfo CurrentPrefillInfo() const { return MakePrefillInfo(resources, window); }
    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { resources.ExtendTokens(result_tokens); }

    ForwardResources resources;
    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

// P role, independent admission: a computed prompt whose Decode destination
// is not allocated yet (no pd::BootstrappedEvent). It keeps the resources of
// PrefillAwaitingResult, pinned like every page-holding P state, and is never
// scheduled; pd::BootstrappedEvent moves it to PrefillDone, whose remote
// decode sends the KV.
struct PrefillAwaitingDestination {
    PrefillAwaitingDestination(ForwardResources resources, TokenContainer::Window window,
                               std::int32_t reserve_num_tokens_in_next_schedule_event)
        : resources{std::move(resources)},
          window{window},
          reserve_num_tokens_in_next_schedule_event_{reserve_num_tokens_in_next_schedule_event} {}

    PrefillInfo CurrentPrefillInfo() const { return MakePrefillInfo(resources, window); }
    std::int32_t ReserveNumTokensInNextScheduleEvent() const { return reserve_num_tokens_in_next_schedule_event_; }
    void ExtendResultTokens(const std::vector<std::int32_t>& result_tokens) { resources.ExtendTokens(result_tokens); }

    ForwardResources resources;
    TokenContainer::Window window{};

private:
    std::int32_t reserve_num_tokens_in_next_schedule_event_{};
};

// The request ended on this engine -- aborted, or its transfer failed --
// while a PD transfer may still read or write its pages: the peer's decode
// reading a P request's sources, the peer's prefill writing a D request's
// destinations. The transport was told to stop; the bundle stays whole,
// pages and all, until it reports that every reader and writer on every rank
// stopped (TransferStoppedEvent). Nothing is scheduled or published here, and
// it is no retraction victim.
struct PdTransferring {
    explicit PdTransferring(ForwardResources resources) : resources{std::move(resources)} {}

    ForwardResources resources;
    // A transport stop may arrive before the last local forward result.
    bool transfer_stopped{false};
};

}  // namespace fsm
}  // namespace tokenspeed
