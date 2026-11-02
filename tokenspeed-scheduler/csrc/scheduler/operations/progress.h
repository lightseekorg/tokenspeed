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

#include "cache/core/cache_types.h"
#include "fsm/forward_states.h"
#include "scheduler/request.h"

namespace tokenspeed {

// What a request's computed frontier is to the prefix cache at the moment an
// admission, a finish or a retraction reads it.
enum class PublicationPoint {
    // An ordinary chunk end or decode step: only newly completed full pages
    // are published; the partial page stays request-private.
    kChunk,
    // The frontier is final for the owner (finish, retraction): the partial
    // page is published in place as a tail, and so is the state.
    kFinal,
    // The owner keeps writing past the frontier (a prompt end followed by
    // decode): the partial page is published as a tail and the admission
    // swaps a fresh block into the owner's table, copying the published rows.
    kOwnerContinues,
};

// Hashes the full prefix pages num_computed_tokens has filled since the
// previous publication, and at a non-chunk point the partial page beyond
// them, and states the request's progress for the coordinator. A non-chunk
// point may publish the live state only when allow_state_checkpoint is true.
// A retained in-forward checkpoint is published whenever its frontier has been
// reached, independently of the live-state policy. The returned spans view
// cache_progress, which must outlive their use.
RequestProgress AdvanceRequestProgress(const Request& request, fsm::CacheProgress& cache_progress,
                                       std::int32_t num_computed_tokens, std::int32_t prefix_granularity,
                                       bool stream_completed_to_host, PublicationPoint point,
                                       bool allow_state_checkpoint);

}  // namespace tokenspeed
