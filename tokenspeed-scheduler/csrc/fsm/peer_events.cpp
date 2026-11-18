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

#include "fsm/peer_events.h"

#include "scheduler/operations/cache.h"
#include "utils.h"

namespace tokenspeed::fsm {

Submitted PeerLandingFailedEvent::operator()(PeerLanding&& state) {
    _assert(coordinator_ != nullptr, "PeerLandingFailedEvent requires a cache coordinator");
    _assert(state.resources.results_in_flight == 0, "no forward reads a landing's pages");
    FreeRequest(*coordinator_, state.resources.block_tables);
    Submitted submitted{state.resources.token_container, state.resources.prefix_granularity};
    // It considered both once already, before this admission.
    submitted.admission.storage_prepared = true;
    submitted.admission.peer_prepared = true;
    return submitted;
}

}  // namespace tokenspeed::fsm
