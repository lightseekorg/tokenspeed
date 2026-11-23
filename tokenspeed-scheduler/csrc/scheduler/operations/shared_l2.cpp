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

#include <stdexcept>
#include <vector>

namespace tokenspeed {

void Scheduler::handleEvent(const cache::SharedL2WriteDone& event) {
    tier_transfers_.CompleteSharedL2Write(event);
}

std::vector<SharedL2Lookup> Scheduler::StartSharedL2Lookups() {
    std::vector<SharedL2Lookup> lookups;
    if (!shared_l2_proxy_) {
        return lookups;
    }
    for (const auto& request : requests_) {
        auto* admission = request->AdmissionResourcesPtr();
        if (admission == nullptr || admission->shared_l2_lookup_state != fsm::SharedL2LookupState::kNotAsked) {
            continue;
        }
        const std::int32_t max_tokens = cacheablePrefixTokens(*request);
        const std::span<const std::string> hashes =
            request->PromptPrefixHashes(max_tokens / config_.prefix_granularity);
        const PromptPrefix prompt{
            .prefix_hashes = hashes, .tokens = request->TokenSlice(0, max_tokens), .max_tokens = max_tokens};
        // The Device alone: a proxy entry another prompt's lookup holds may
        // be gone by this prompt's admission.
        if (max_tokens == 0 || coordinator_.Prober().ProbePromptPrefix(prompt).device_only_hit_tokens >= max_tokens) {
            // Nothing the pool could add.
            admission->shared_l2_lookup_state = fsm::SharedL2LookupState::kSettled;
            continue;
        }
        admission->shared_l2_lookup_state = fsm::SharedL2LookupState::kAsked;
        lookups.push_back(SharedL2Lookup{.request_id = request->Id(),
                                         .prefix_hashes = {hashes.begin(), hashes.end()},
                                         .tokens = {prompt.tokens.begin(), prompt.tokens.end()},
                                         .max_tokens = max_tokens});
    }
    return lookups;
}

std::vector<SharedL2ProxyBlock> Scheduler::ApplySharedL2Answers(const std::vector<SharedL2Answer>& answers) {
    if (!shared_l2_proxy_) {
        throw std::invalid_argument("shared L2 lookup answers without a configured pool");
    }
    std::vector<SharedL2ProxyBlock> proxy_blocks;
    for (const SharedL2Answer& answer : answers) {
        // A lookup waits in its request's pre-admission state.
        Request* request = findRequest(answer.request_id);
        auto* admission = request != nullptr ? request->AdmissionResourcesPtr() : nullptr;
        if (admission == nullptr || admission->shared_l2_lookup_state != fsm::SharedL2LookupState::kAsked) {
            throw std::invalid_argument("a shared L2 answer names no lookup that waits for it");
        }
        const std::int32_t tail_tokens = answer.tail ? answer.tail->tail_tokens : 0;
        // The lookup asked up to the prompt's cacheable prefix
        // (StartSharedL2Lookups): neither the hit nor the pool's KV reach
        // lies past it.
        const std::int32_t max_tokens = cacheablePrefixTokens(*request);
        if (answer.num_tokens < 0 || answer.num_tokens > max_tokens || answer.kv_tokens < 0 ||
            answer.kv_tokens > max_tokens || tail_tokens < 0 || tail_tokens >= config_.prefix_granularity ||
            (answer.num_tokens - tail_tokens) % config_.prefix_granularity != 0) {
            throw std::invalid_argument("a shared L2 answer lies outside the prompt it answers");
        }
        ContentHash tail_prior;
        if (const std::int32_t full_pages = (answer.num_tokens - tail_tokens) / config_.prefix_granularity;
            answer.tail && full_pages > 0) {
            tail_prior = ContentHash{request->PromptPrefixHashes()[static_cast<std::size_t>(full_pages - 1)]};
        }
        std::vector<SharedL2ProxyBlock> created =
            shared_l2_proxy_->MirrorAnswer(answer, tail_prior, admission->shared_l2_hit_blocks);
        admission->shared_l2_kv_tokens = answer.kv_tokens;
        admission->shared_l2_lookup_state = fsm::SharedL2LookupState::kSettled;
        proxy_blocks.insert(proxy_blocks.end(), created.begin(), created.end());
    }
    return proxy_blocks;
}

}  // namespace tokenspeed
