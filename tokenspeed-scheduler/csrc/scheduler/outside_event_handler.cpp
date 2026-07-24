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

#if TOKENSPEED_FLAT_KVCACHE
namespace {

// Release explicitly in reverse; vector destruction order is not our eviction policy.
void FreeAll(std::vector<BlockRef>&& refs) {
    for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
        it->reset();
    }
}

}  // namespace
#endif

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
    pending_forward_results_.erase(event.request_id);
#if TOKENSPEED_FLAT_KVCACHE
    flat_reserved_pages_.erase(event.request_id);
#endif
    std::vector<std::string> page_hashes;
    requests_.at(event.request_id)
        ->Apply(fsm::FinishEvent{&kv_prefix_cache_, &host_allocator_, std::move(page_hashes), config_.disable_l2_cache,
                                 hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                 ,
                                 &coordinator_
#endif
        });
    tryFinalizeDeferredAbort(event.request_id);
}

void Scheduler::handleEvent(const pd::RemotePrefillDoneEvent& event) {
    requests_.at(event.request_id)->Apply(fsm::RemotePrefillDoneEvent{event.bootstrap_token});
}

void Scheduler::handleEvent(const forward::Finish& event) {
    pending_forward_results_.erase(event.request_id);
#if TOKENSPEED_FLAT_KVCACHE
    flat_reserved_pages_.erase(event.request_id);
#endif
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
                                    config_.disable_l2_cache, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr
#if TOKENSPEED_FLAT_KVCACHE
                                    ,
                                    &coordinator_
#endif
        });
    }
    tryFinalizeDeferredAbort(event.request_id);
}

void Scheduler::handleEvent(const forward::UpdateReserveNumTokens& event) {
    if (auto req = find_request(event.request_id)) {
        req->Apply(fsm::UpdateReserveNumTokensEvent{event.reserve_num_tokens_in_next_schedule_event});
    }
}
void Scheduler::handleEvent(const forward::ExtendResult& event) {
    // One owed forward result delivered (see pending_forward_results_).
    if (auto it = pending_forward_results_.find(event.request_id); it != pending_forward_results_.end()) {
        if (--it->second <= 0) {
            pending_forward_results_.erase(it);
        }
    }
    if (auto req = find_request(event.request_id)) {
        const std::int32_t protected_tail_tokens = config_.overlap_schedule_depth * config_.decode_input_tokens;
        req->Apply(fsm::ExtendResultEvent{event.request_id, event.tokens,
                                          hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
                                          protected_tail_tokens});
    }
    tryFinalizeDeferredAbort(event.request_id);
}

void Scheduler::handleEvent(const forward::Abort& event) {
    // Terminal for this request's forward stream: late results are ignored.
    pending_forward_results_.erase(event.request_id);
#if TOKENSPEED_FLAT_KVCACHE
    // Drop any decode reservation it never consumed -- an abort between the
    // prefill-completing admission and the PrefillDone->Decoding transition must
    // not leave a permanent phantom reservation deflating every later gate.
    flat_reserved_pages_.erase(event.request_id);
#endif
    Request* req = find_request(event.request_id);
    if (req == nullptr) {
        return;
    }

    // L3 prefetch owns a separate abort handoff: the request remains Aborting
    // until PrefetchDone releases its host-page transfer.
    if (req->Is<fsm::Prefetching>() || req->Is<fsm::Aborting>()) {
        req->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
            &coordinator_
#endif
        });
        return;
    }

    deferAbort(event.request_id, /*discard_writeback=*/true);
    tryFinalizeDeferredAbort(event.request_id);
}

void Scheduler::handleEvent(const cache::WriteBackDone& event) {
#if TOKENSPEED_FLAT_KVCACHE
    if (std::vector<FlatStoreTicket> tickets = flat_store_ops_.Retire(event.op_id); !tickets.empty()) {
        // Publish-at-ack: hashing the host block makes it hittable; either way it returns to the
        // host free list (hash-intact = reusable, unhashed = plain recycling). Batched frees in
        // ticket order keep both pools' recycling order deterministic.
        for (FlatStoreTicket& t : tickets) {
            if (event.success) {
                flat_host_pool_.CacheFullBlock(t.host_block, t.key);
            }
        }
        for (auto it = tickets.rbegin(); it != tickets.rend(); ++it) {
            it->device_block.reset();
            it->host_block.reset();
        }
        return;
    }
#endif
    auto it = cache_op_tracker_.find(event.op_id);
    if (it == cache_op_tracker_.end()) {
        return;
    }

    auto spec = std::move(it->second);
    cache_op_tracker_.erase(it);

    const auto deferred_it = deferred_aborts_.find(spec.request_id);
    const bool discard_writeback = deferred_it != deferred_aborts_.end() && deferred_it->second.discard_writeback;
    const bool effective_success = event.success && !discard_writeback;

    auto now = std::chrono::steady_clock::now();
    for (TreeNode* n : spec.nodes) n->Touch(now);

    if (!spec.request_id.empty()) {
        if (auto* req = find_request(spec.request_id)) {
            const bool was_writing_back = req->Is<fsm::WritingBack>();
            const bool was_retracting = req->Is<fsm::Retracting>();
            // A finishing request no longer needs its paged-cache table once D2H completes.
            // A successful retract keeps its table for recovery; the deferred-abort finalizer
            // releases failed or explicitly aborted retracts.
            if (hybrid_prefix_cache_ && was_writing_back) {
                hybrid_prefix_cache_->ReleaseRequest(spec.request_id);
            }
            req->Apply(fsm::WriteBackDoneEvent{
                &kv_prefix_cache_, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr, effective_success});
            if (was_retracting && !event.success && !discard_writeback) {
                deferAbort(spec.request_id, /*discard_writeback=*/false, "L2 retract write-back failed");
            }
            tryFinalizeDeferredAbort(spec.request_id);
            return;
        }
    }
    if (hybrid_prefix_cache_ && !spec.paged_cache_nodes.empty()) {
        hybrid_prefix_cache_->OnPagedCacheHostWriteBackDone(spec.paged_cache_nodes, effective_success);
    }
    if (!spec.request_id.empty()) {
        tryFinalizeDeferredAbort(spec.request_id);
    }
}

void Scheduler::handleEvent(const cache::LoadBackDone& event) {
#if TOKENSPEED_FLAT_KVCACHE
    if (auto flat_it = flat_load_ops_.find(event.op_id); flat_it != flat_load_ops_.end()) {
        // The loaded device pages are already claimed as computed KV: a failed copy
        // means the request would decode over garbage bytes -- fail loud.
        _assert(event.success, "flat host loadback failed: host bytes integrity");
        FreeAll(std::move(flat_it->second.host_pins));
        FreeAll(std::move(flat_it->second.device_blocks));
        flat_load_ops_.erase(flat_it);
        return;
    }
#endif
    auto it = cache_op_tracker_.find(event.op_id);
    if (it == cache_op_tracker_.end() || !it->second.paged_loadback_host_pin) {
        return;
    }
    _assert(event.success, "radix paged-cache host loadback failed");
    const std::string request_id = it->second.request_id;
    cache_op_tracker_.erase(it);
    tryFinalizeDeferredAbort(request_id);
}

}  // namespace tokenspeed
