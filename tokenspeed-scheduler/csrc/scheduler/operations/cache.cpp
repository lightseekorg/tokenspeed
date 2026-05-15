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

#include "scheduler/operations/cache.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fsm/cache_events.h"
#include "fsm/forward_states.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/types.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/scheduler.h"
#include "scheduler/types.h"

namespace tokenspeed {

std::optional<fsm::SchedulePrefetchEvent> Scheduler::schedulePrefetch(Request* request, const MatchResult& match) {
    const auto& storage = request->GetStorageInfo();
    if (config_.disable_prefix_cache || !config_.enable_l3_storage || !request->Is<fsm::Submitted>() ||
        storage.hit_pages <= config_.prefetch_threshold) {
        return {};
    }

    const std::int32_t num_pages_to_fetch = storage.hit_pages;
    if (!kv_prefix_cache_.EnsureCapacityByEvict<ResourceType::Host>(num_pages_to_fetch)) {
        return {};
    }

    std::vector<std::string> hashes(storage.rolling_hashes.begin(),
                                    storage.rolling_hashes.begin() + num_pages_to_fetch);

    return fsm::SchedulePrefetchEvent{num_pages_to_fetch, std::move(hashes), &host_allocator_, match.host.last_node};
}

PrefetchOperation Scheduler::applyEventAndGenerateOp(Request* request, fsm::SchedulePrefetchEvent event) {
    // Save rolling hashes BEFORE Apply (event will be moved into the state transition).
    auto rolling_hashes = event.TakeRollingPageHashes();

    // Apply event: Submitted → Prefetching (host pages allocated inside the state transition).
    request->Apply(event);

    // After Apply, request is in Prefetching state; read back the allocated host pages.
    cache_op_id op_id = kv_prefix_cache_.AllocateCacheOpId();

    CacheOpSpec spec;
    spec.request_id = request->Id();
    cache_op_tracker_[op_id] = std::move(spec);

    PrefetchOperation prefetch_op;
    prefetch_op.op_id = op_id;
    prefetch_op.dst_pages = request->GetHostPageIds();
    prefetch_op.request_id = request->Id();
    prefetch_op.rolling_page_hashes = std::move(rolling_hashes);
    return prefetch_op;
}

}  // namespace tokenspeed
