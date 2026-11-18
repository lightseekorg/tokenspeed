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
#include <optional>
#include <utility>

#include "cache/coordinator/cache_coordinator.h"
#include "fsm/base_event.h"
#include "fsm/peer_states.h"

namespace tokenspeed::fsm {

// A waiting prompt asks its holder: once per request.
struct StartPeerLookupEvent : InvalidTransitionHandler<StartPeerLookupEvent> {
    using InvalidTransitionHandler<StartPeerLookupEvent>::operator();

    StartPeerLookupEvent(PeerLeaseId lease, std::int64_t deadline_us)
        : lease_{std::move(lease)}, deadline_us_{deadline_us} {}

    PeerLooking operator()(Submitted&& state) {
        state.admission.peer_prepared = true;
        return PeerLooking{std::move(state.admission), std::move(lease_), deadline_us_};
    }

private:
    PeerLeaseId lease_;
    std::int64_t deadline_us_;
};

// The holder answered, or the wait ended: admission takes the hit, if any.
struct FinishPeerLookupEvent : InvalidTransitionHandler<FinishPeerLookupEvent> {
    using InvalidTransitionHandler<FinishPeerLookupEvent>::operator();

    explicit FinishPeerLookupEvent(std::optional<PeerHit> hit) : hit_{std::move(hit)} {}

    Submitted operator()(PeerLooking&& state) {
        state.admission.peer_hit = std::move(hit_);
        return Submitted{std::move(state.admission)};
    }

private:
    std::optional<PeerHit> hit_;
};

// The whole landed prefix is the request's own: it prefills on from there.
struct PeerLandedEvent : InvalidTransitionHandler<PeerLandedEvent> {
    using InvalidTransitionHandler<PeerLandedEvent>::operator();

    Prefilling operator()(PeerLanding&& state) {
        const std::int32_t reserve = state.ReserveNumTokensInNextScheduleEvent();
        return Prefilling{std::move(state.resources), state.window, reserve};
    }
};

// The landing fell short: its pages go back and the request is admitted
// again, as a prompt that asked its holder already, from its local hit.
struct PeerLandingFailedEvent : InvalidTransitionHandler<PeerLandingFailedEvent> {
    using InvalidTransitionHandler<PeerLandingFailedEvent>::operator();

    explicit PeerLandingFailedEvent(CacheCoordinator* coordinator) : coordinator_{coordinator} {}

    Submitted operator()(PeerLanding&& state);

private:
    CacheCoordinator* coordinator_{};
};

}  // namespace tokenspeed::fsm
