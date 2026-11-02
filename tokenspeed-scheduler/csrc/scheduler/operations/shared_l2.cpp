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
#include <stdexcept>
#include <vector>

namespace tokenspeed {

void Scheduler::handleEvent(const cache::SharedL2WriteDone& event) {
    if (!shared_l2_) throw std::invalid_argument("shared L2 completion without a configured pool");
    shared_l2_->Complete(event);
}

std::vector<SharedL2Lookup> Scheduler::SharedL2Lookups() {
    std::vector<SharedL2Lookup> lookups;
    if (!shared_l2_ || config_.disable_prefix_cache) return lookups;
    for (const auto& request : requests_) {
        auto* admission = request->AdmissionResourcesPtr();
        if (admission == nullptr || admission->pool_lookup_asked) continue;
        admission->pool_lookup_asked = true;
        const auto max_tokens = cacheablePrefixTokens(*request);
        const auto hashes = request->AdmissionPrefixHashes().first(
            std::min(request->AdmissionPrefixHashes().size(),
                     static_cast<std::size_t>(max_tokens / config_.prefix_granularity)));
        const CacheCoordinator::PromptPrefix prompt{
            .prefix_hashes = hashes, .tokens = request->TokenSlice(0, max_tokens), .max_tokens = max_tokens};
        if (max_tokens == 0 || coordinator_.ProbePromptPrefix(prompt).device.num_common_tokens >= max_tokens) {
            // Nothing the pool could add.
            shared_l2_->ApplyLookup(SharedL2LookupResult{.request_id = request->Id()}, ContentHash{});
            continue;
        }
        lookups.push_back(SharedL2Lookup{.request_id = request->Id(),
                                         .prefix_hashes = {hashes.begin(), hashes.end()},
                                         .tokens = {prompt.tokens.begin(), prompt.tokens.end()},
                                         .max_tokens = max_tokens});
    }
    return lookups;
}

std::vector<SharedL2ProxyPage> Scheduler::ApplySharedL2Lookups(const std::vector<SharedL2LookupResult>& results) {
    _assert(shared_l2_ != nullptr, "shared L2 lookups need a configured pool");
    std::vector<SharedL2ProxyPage> pages;
    for (const SharedL2LookupResult& result : results) {
        Request* request = findRequest(result.request_id);
        auto* admission = request != nullptr ? request->AdmissionResourcesPtr() : nullptr;
        _assert(admission != nullptr && admission->pool_lookup_asked && !shared_l2_->LookedUp(result.request_id),
                "a shared L2 lookup answers a waiting request that asked");
        admission->pool_kv_tokens = result.kv_tokens;
        ContentHash prior;
        if (result.tail) {
            const auto full_pages = (result.num_tokens - result.tail->tail_tokens) / config_.prefix_granularity;
            if (full_pages > 0) prior = request->AdmissionPrefixHashes()[static_cast<std::size_t>(full_pages - 1)];
        }
        std::vector<SharedL2ProxyPage> created = shared_l2_->ApplyLookup(result, prior);
        pages.insert(pages.end(), created.begin(), created.end());
    }
    return pages;
}

}  // namespace tokenspeed
