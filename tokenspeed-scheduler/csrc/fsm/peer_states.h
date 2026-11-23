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
#include <utility>

#include "cache/tier/peer.h"
#include "fsm/forward_states.h"

// States of a reader under peer prefix sharing
// (docs/rfcs/0004-peer-prefix-sharing.md); an engine without it never
// enters them.

namespace tokenspeed::fsm {

// A lookup is out to the holder the request's hint names. Nothing is held
// here; the answer, or the deadline, returns the request to Submitted.
struct PeerLooking {
    PeerLooking(AdmissionResources admission, PeerLeaseId lease, std::int64_t deadline_us)
        : admission{std::move(admission)}, lease{std::move(lease)}, deadline_us{deadline_us} {}

    AdmissionResources admission;
    PeerLeaseId lease;
    std::int64_t deadline_us;
};

// The admitted prompt's prefix, its window, is landing from the holder into
// the request's pages. Nothing here is schedulable and no forward reads the
// pages until the fetch reports (PeerFetchDone). A state of its own, as
// RemotePrefilling is: the two share Prefilling's data, not its lifecycle.
struct PeerLanding : public Prefilling {
    using Prefilling::Prefilling;
};

}  // namespace tokenspeed::fsm
