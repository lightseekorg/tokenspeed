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

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "fsm/base_event.h"
#include "fsm/forward_states.h"
#include "fsm/pd_states.h"

namespace tokenspeed {
class CacheCoordinator;

namespace fsm {

struct Bootstrapping;
struct RemotePrefilling;
struct Prefetching;
struct PrefillAwaitingDestination;

struct BootstrappedEvent : InvalidTransitionHandler<BootstrappedEvent> {
    using InvalidTransitionHandler<BootstrappedEvent>::operator();

    Submitted operator()(Bootstrapping&& state);
    Prefetching operator()(Prefetching&& state);
};

// P role, independent decode allocation: a computed prompt that waited for
// its destination sends its KV.
struct DestinationReadyEvent : InvalidTransitionHandler<DestinationReadyEvent> {
    using InvalidTransitionHandler<DestinationReadyEvent>::operator();

    PrefillDone operator()(PrefillAwaitingDestination&& state);
};

struct SucceededEvent : InvalidTransitionHandler<SucceededEvent> {
    using InvalidTransitionHandler<SucceededEvent>::operator();

    Finished operator()(Decoding&& state);
};

struct RemotePrefillDoneEvent : InvalidTransitionHandler<RemotePrefillDoneEvent> {
    using InvalidTransitionHandler<RemotePrefillDoneEvent>::operator();

    std::int32_t bootstrap_token{-1};

    explicit RemotePrefillDoneEvent(std::int32_t token) : bootstrap_token(token) {}

    PrefillDone operator()(RemotePrefilling&& state);
};

// The request ends while its PD transfer may still touch its pages: it parks
// in PdTransferring with its bundle. Every page-holding state but Finishing
// can, the rule the P role pins by (Scheduler::pdTransferInFlight), so a
// pinned request can always end. Ending it again changes nothing.
struct StopPdTransferEvent : InvalidTransitionHandler<StopPdTransferEvent> {
    using InvalidTransitionHandler<StopPdTransferEvent>::operator();

    template <typename State>
        requires HoldsForwardResources<std::remove_cvref_t<State>> &&
                 (!std::same_as<std::remove_cvref_t<State>, Finishing>)
    PdTransferring operator()(State&& state) {
        return PdTransferring{std::move(state.resources)};
    }
    PdTransferring operator()(PdTransferring&& state) { return std::move(state); }
};

// Every reader and writer of a parked request's transfer stopped: its pages
// return to the pool. Nothing of it is published.
struct TransferStoppedEvent : InvalidTransitionHandler<TransferStoppedEvent> {
    using InvalidTransitionHandler<TransferStoppedEvent>::operator();

    explicit TransferStoppedEvent(CacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Finished operator()(PdTransferring&& state);

private:
    CacheCoordinator* coordinator_{};
};

}  // namespace fsm
}  // namespace tokenspeed
