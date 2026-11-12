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

SharedL2Stats Scheduler::GetSharedL2Stats() const {
    SharedL2Stats stats = shared_l2_ ? shared_l2_->Stats() : SharedL2Stats{};
    stats.writes_in_flight = tier_transfers_.NumPoolWritesInFlight();
    stats.failed_writes = tier_transfers_.NumFailedPoolWrites();
    stats.failed_write_blocks = tier_transfers_.NumFailedPoolWriteBlocks();
    stats.lease_expired_hits = shared_l2_lease_expired_hits_;
    return stats;
}

void Scheduler::handleEvent(const cache::SharedL2WriteDone& event) {
    if (!shared_l2_) {
        throw std::invalid_argument("shared L2 completion without a configured pool");
    }
    tier_transfers_.CompletePoolWrite(event);
}

std::vector<SharedL2Lookup> Scheduler::SharedL2Lookups() {
    std::vector<SharedL2Lookup> lookups;
    if (!shared_l2_ || config_.disable_prefix_cache) {
        return lookups;
    }
    for (const auto& request : requests_) {
        auto* admission = request->AdmissionResourcesPtr();
        if (admission == nullptr || admission->pool_settled || admission->pool_lookup) {
            continue;
        }
        const std::int32_t max_tokens = cacheablePrefixTokens(*request);
        const std::span<const std::string> prompt_hashes = request->PromptPrefixHashes();
        const std::span<const std::string> hashes = prompt_hashes.first(
            std::min(prompt_hashes.size(), static_cast<std::size_t>(max_tokens / config_.prefix_granularity)));
        const CacheCoordinator::PromptPrefix prompt{
            .prefix_hashes = hashes, .tokens = request->TokenSlice(0, max_tokens), .max_tokens = max_tokens};
        // The Device alone: a proxy entry another prompt's lookup holds may
        // be gone by this prompt's admission.
        if (max_tokens == 0 || coordinator_.ProbePromptPrefix(prompt).device_only_hit_tokens >= max_tokens) {
            // Nothing the pool could add.
            admission->pool_settled = true;
            continue;
        }
        admission->pool_lookup = shared_l2_->StartLookup(request->Id());
        lookups.push_back(SharedL2Lookup{.lookup_id = admission->pool_lookup.LookupId(),
                                         .request_id = request->Id(),
                                         .prefix_hashes = {hashes.begin(), hashes.end()},
                                         .tokens = {prompt.tokens.begin(), prompt.tokens.end()},
                                         .max_tokens = max_tokens});
    }
    return lookups;
}

std::vector<SharedL2ProxyPage> Scheduler::ApplySharedL2Lookups(const std::vector<SharedL2LookupResult>& results) {
    if (!shared_l2_) {
        throw std::invalid_argument("shared L2 lookup answers without a configured pool");
    }
    std::vector<SharedL2ProxyPage> pages;
    for (const SharedL2LookupResult& result : results) {
        const std::string* request_id = shared_l2_->PendingRequest(result.lookup_id);
        if (request_id == nullptr) {
            shared_l2_->DropAnswer(result);
            continue;
        }
        // A pending lookup's hold lives in its request's pre-admission state.
        Request* request = findRequest(*request_id);
        auto* admission = request != nullptr ? request->AdmissionResourcesPtr() : nullptr;
        FatalCheck(admission != nullptr && admission->pool_lookup.LookupId() == result.lookup_id,
                   "a pending shared L2 lookup lost its waiting request");
        const std::int32_t tail_tokens = result.tail ? result.tail->tail_tokens : 0;
        if (result.num_tokens < 0 || result.num_tokens > cacheablePrefixTokens(*request) || result.kv_tokens < 0 ||
            tail_tokens < 0 || tail_tokens >= config_.prefix_granularity ||
            (result.num_tokens - tail_tokens) % config_.prefix_granularity != 0) {
            throw std::invalid_argument("a shared L2 answer lies outside the prompt it answers");
        }
        ContentHash tail_prior;
        if (const std::int32_t full_pages = (result.num_tokens - tail_tokens) / config_.prefix_granularity;
            result.tail && full_pages > 0) {
            tail_prior = ContentHash{request->PromptPrefixHashes()[static_cast<std::size_t>(full_pages - 1)]};
        }
        std::vector<SharedL2ProxyPage> created = shared_l2_->ApplyAnswer(result, tail_prior, admission->pool_lookup);
        admission->pool_kv_tokens = result.kv_tokens;
        admission->pool_settled = true;
        pages.insert(pages.end(), created.begin(), created.end());
    }
    return pages;
}

void Scheduler::AbandonSharedL2Lookups(const std::vector<std::uint64_t>& lookup_ids) {
    if (!shared_l2_) {
        throw std::invalid_argument("shared L2 lookups abandoned without a configured pool");
    }
    for (const std::uint64_t lookup_id : lookup_ids) {
        // One already answered, abandoned or whose request left needs nothing.
        const std::string* request_id = shared_l2_->PendingRequest(lookup_id);
        if (request_id == nullptr) {
            continue;
        }
        Request* request = findRequest(*request_id);
        auto* admission = request != nullptr ? request->AdmissionResourcesPtr() : nullptr;
        FatalCheck(admission != nullptr && admission->pool_lookup.LookupId() == lookup_id,
                   "a pending shared L2 lookup lost its waiting request");
        // The prompt computes its prefix; dropping the hold abandons the
        // lookup at the owner.
        admission->pool_lookup.reset();
        admission->pool_settled = true;
    }
}

void Scheduler::SetSharedL2LeaseLost(bool lost) {
    if (!shared_l2_) {
        throw std::invalid_argument("shared L2 lease state without a configured pool");
    }
    shared_l2_->SetLeaseLost(lost);
    if (!lost) {
        return;
    }
    for (const auto& request : requests_) {
        // A lookup still pending keeps waiting for its answer.
        if (auto* admission = request->AdmissionResourcesPtr();
            admission != nullptr && admission->pool_lookup.HoldsHit()) {
            admission->pool_lookup.reset();
            // Nothing of the answer is used: no junction is cut at its reach.
            admission->pool_kv_tokens = 0;
            ++shared_l2_lease_expired_hits_;
        }
    }
    // At once rather than at the end of the next plan, whose commands then
    // already release the tickets left without an entry.
    shared_l2_->ReleaseUnheld();
}

}  // namespace tokenspeed
