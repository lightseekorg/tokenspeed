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

#include <stdexcept>

#include "resource/allocator/owned_pages.h"
#include "fsm/states.h"
#include "scheduler/outside_events/inc.h"
#include "scheduler/page_hasher.h"
#include "scheduler/scheduler.h"

#include "fsm/forward_events.h"
#include "fsm/pd_events.h"

namespace tokenspeed {

void Scheduler::handleEvent(const cache::PrefetchDone& event) {
    // Remove from op tracker (regardless of success).
    cache_op_tracker_.erase(event.op_id);

    auto req_iter = requests_.find(event.request_id);
    if (req_iter == requests_.end()) {
        return;
    }

    Request* req = req_iter->second.get();
    if (!req->Is<fsm::Prefetching>() && !req->Is<fsm::Aborting>()) {
        return;
    }

    std::int32_t completed = 0;
    std::int32_t inserted = 0;

    if (req->Is<fsm::Prefetching>() && event.success && event.completed_pages > 0) {
        // Insert completed host pages into the KVPrefixCache so that future Match() calls
        // see them in the host side and can generate LoadBack ops.
        auto token_pages = req->GetFullPagedTokens(false);
        auto all_host_pages = req->GetHostPageIds();

        std::int32_t n = std::min(event.completed_pages, static_cast<std::int32_t>(all_host_pages.size()));
        std::int32_t n_tokens = std::min(n, static_cast<std::int32_t>(token_pages.size()));

        if (n_tokens > 0) {
            std::vector<std::span<const std::int32_t>> insert_token_pages(token_pages.begin(),
                                                                          token_pages.begin() + n_tokens);
            std::vector<std::int32_t> insert_pages(all_host_pages.begin(), all_host_pages.begin() + n);

            // Storage hashes for L3 backup (optional).
            const auto& storage = req->GetStorageInfo();
            std::vector<std::string> page_hashes;
            if (!storage.rolling_hashes.empty()) {
                std::int32_t nh = std::min(n_tokens, static_cast<std::int32_t>(storage.rolling_hashes.size()));
                page_hashes.assign(storage.rolling_hashes.begin(), storage.rolling_hashes.begin() + nh);
            }

            // Insert into host side; InsertHost returns how many were actually inserted
            // (0 for pages that already existed — "overlapping").
            auto insert_result = kv_prefix_cache_.Insert<ResourceType::Host>(insert_token_pages, insert_pages,
                                                                             OwnedPages{}, page_hashes);
            completed = n;
            inserted = insert_result.inserted_num_pages;
        }
    }

    fsm::PrefetchDoneEvent fsm_event{completed, inserted};
    req->Apply(fsm_event);
}

void Scheduler::handleEvent(const pd::BootstrappedEvent& event) {
    requests_.at(event.request_id)->Apply(fsm::BootstrappedEvent{});
}

void Scheduler::handleEvent(const pd::FailedEvent& event) {}

void Scheduler::handleEvent(const pd::SucceededEvent& event) {
    std::vector<std::string> page_hashes;
    auto& req = requests_.at(event.request_id);
    req->Apply(fsm::FinishEvent{&kv_prefix_cache_, &host_allocator_, std::move(page_hashes), config_.disable_l2_cache,
                                hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr, req->LoraId()});
}

void Scheduler::handleEvent(const pd::RemotePrefillDoneEvent& event) {
    requests_.at(event.request_id)->Apply(fsm::RemotePrefillDoneEvent{event.bootstrap_token});
}

void Scheduler::handleEvent(const forward::Finish& event) {
    if (auto req = find_request(event.request_id)) {
        // except_last=true: exclude the tail page, matching FinishEvent's InsertDevice behavior
        auto token_pages = req->GetFullPagedTokens(true);

        // page_hashes are only needed for L3 storage (BackUp ops).
        // Without L3, pass empty to avoid any size-mismatch bugs.
        std::vector<std::string> page_hashes;
        if (config_.enable_l3_storage) {
            page_hashes = req->GetStorageInfo().rolling_hashes;
            if (page_hashes.size() != token_pages.size()) {
                page_hashes = ComputePagedHashes(token_pages, "");
            }
        }
        req->Apply(fsm::FinishEvent{&kv_prefix_cache_, &host_allocator_, std::move(page_hashes),
                                    config_.disable_l2_cache, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
                                    req->LoraId()});
    }
}

void Scheduler::handleEvent(const forward::UpdateReserveNumTokens& event) {
    if (auto req = find_request(event.request_id)) {
        req->Apply(fsm::UpdateReserveNumTokensEvent{event.reserve_num_tokens_in_next_schedule_event});
    }
}
void Scheduler::handleEvent(const forward::ExtendResult& event) {
    if (auto req = find_request(event.request_id)) {
        req->Apply(fsm::ExtendResultEvent{event.tokens});
    }
}

void Scheduler::handleEvent(const forward::Abort& event) {
    auto iter = requests_.find(event.request_id);
    if (iter == requests_.end()) {
        return;
    }

    Request* req = iter->second.get();
    req->Apply(fsm::AbortEvent{});
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
    auto it = cache_op_tracker_.find(event.op_id);
    if (it == cache_op_tracker_.end()) {
        return;
    }

    auto spec = std::move(it->second);
    cache_op_tracker_.erase(it);

    auto now = std::chrono::steady_clock::now();
    for (TreeNode* n : spec.nodes) n->Touch(now);

    if (!spec.request_id.empty()) {
        if (auto* req = find_request(spec.request_id)) {
            req->Apply(fsm::WriteBackDoneEvent{});
        }
    }
}

}  // namespace tokenspeed
