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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cache/core/cache_types.h"
#include "fsm/forward_states.h"
#include "scheduler/request.h"
#include "utils.h"

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
    // swaps into the owner's table the copy-in block the request holds for
    // each block it registered, copying the published rows
    // (OwnerTail::kCopiedIn).
    kOwnerContinues,
    // A chunk end cut at a junction, or at the in-forward State checkpoint's
    // page boundary: the state is published there -- under the tail's hash
    // at a junction -- and the partial page's KV, which a junction has cached
    // already, stays the owner's, unpublished (OwnerTail::kKept).
    kStateCheckpoint,
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
                                       bool allow_state_checkpoint, bool preserve_source_blocks);

// A scheduling attempt advances its request's cache progress in place; unless
// the attempt commits, the advance is undone when this goes out of scope. The
// hash chain only grows during an attempt (AdvanceRequestProgress appends the
// completed pages' hashes), so undoing truncates it: an attempt no longer
// works on a copy of the whole chain, one string per page. The other fields
// are restored from copies, except the retained input checkpoint (admission
// may release that optional hold for capacity, and the release stands when
// the attempt fails anyway).
class StagedCacheProgress {
public:
    explicit StagedCacheProgress(fsm::CacheProgress& live)
        : live_{live},
          prefix_pages_{live.prefix_hashes.size()},
          access_epoch_{live.access_epoch},
          junction_tokens_{live.junction_tokens},
          resumed_tail_hash_{live.resumed_tail_hash},
          state_checkpoint_tokens_{live.state_checkpoint_tokens},
          state_checkpoint_blocks_{live.state_checkpoint_blocks},
          published_state_checkpoint_blocks_{live.published_state_checkpoint_blocks},
          published_state_checkpoint_tokens_{live.published_state_checkpoint_tokens} {}
    StagedCacheProgress(const StagedCacheProgress&) = delete;
    StagedCacheProgress& operator=(const StagedCacheProgress&) = delete;
    ~StagedCacheProgress() {
        if (committed_) {
            return;
        }
        FatalCheck(live_.prefix_hashes.size() >= prefix_pages_, "a scheduling attempt only appends to the hash chain");
        live_.prefix_hashes.erase(live_.prefix_hashes.begin() + static_cast<std::ptrdiff_t>(prefix_pages_),
                                  live_.prefix_hashes.end());
        live_.access_epoch = access_epoch_;
        live_.junction_tokens = junction_tokens_;
        live_.resumed_tail_hash = std::move(resumed_tail_hash_);
        live_.state_checkpoint_tokens = state_checkpoint_tokens_;
        live_.state_checkpoint_blocks = std::move(state_checkpoint_blocks_);
        live_.published_state_checkpoint_blocks = std::move(published_state_checkpoint_blocks_);
        live_.published_state_checkpoint_tokens = published_state_checkpoint_tokens_;
    }

    void Commit() noexcept { committed_ = true; }

private:
    fsm::CacheProgress& live_;
    std::size_t prefix_pages_;
    std::uint64_t access_epoch_;
    std::int32_t junction_tokens_;
    std::optional<std::string> resumed_tail_hash_;
    std::int32_t state_checkpoint_tokens_;
    std::vector<CacheBlockRef> state_checkpoint_blocks_;
    std::vector<CacheBlockRef> published_state_checkpoint_blocks_;
    std::int32_t published_state_checkpoint_tokens_;
    bool committed_{false};
};

}  // namespace tokenspeed
