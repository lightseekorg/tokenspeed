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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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
            auto insert_result = radixPrefixCache().Insert<ResourceType::Host>(insert_token_pages, insert_pages,
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
#if TOKENSPEED_FLAT_KVCACHE
    if (deferFlatTerminal(event.request_id, FlatPendingTerminal::kFinish)) {
        return;
    }
    applyFlatFinish(event.request_id);
#else
    requests_.at(event.request_id)
        ->Apply(fsm::FinishEvent{&radixPrefixCache(),
                                 &host_allocator_,
                                 {},
                                 config_.disable_l2_cache,
                                 hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
#endif
}

void Scheduler::handleEvent(const pd::RemotePrefillDoneEvent& event) {
    requests_.at(event.request_id)->Apply(fsm::RemotePrefillDoneEvent{event.bootstrap_token});
}

void Scheduler::handleEvent(const forward::Finish& event) {
#if TOKENSPEED_FLAT_KVCACHE
    if (deferFlatTerminal(event.request_id, FlatPendingTerminal::kFinish)) {
        return;
    }
    applyFlatFinish(event.request_id);
#else
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
        req->Apply(fsm::FinishEvent{&radixPrefixCache(), &host_allocator_, std::move(page_hashes),
                                    config_.disable_l2_cache, hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
    }
#endif
}

#if TOKENSPEED_FLAT_KVCACHE
bool Scheduler::deferFlatTerminal(const std::string& request_id, FlatPendingTerminal terminal) {
    Request* request = find_request(request_id);
    if (!config_.UsesExplicitFlatPools() || request == nullptr || !request->FlatCompletionState().HasOutstanding()) {
        return false;
    }
    request->DeferFlatTerminal(terminal);
    (void)request->FlatCompletionState().CancelOutstanding();
    request->FlatReservation().Clear();
    return true;
}

void Scheduler::applyFlatFinish(const std::string& request_id) {
    pending_forward_results_.erase(request_id);
    if (auto req = find_request(request_id)) {
        req->FlatReservation().Clear();
        req->Apply(fsm::FinishEvent{&coordinator_});
    }
}

void Scheduler::applyFlatAbort(const std::string& request_id) {
    pending_forward_results_.erase(request_id);
    if (Request* req = find_request(request_id); req != nullptr) {
        req->FlatReservation().Clear();
        req->Apply(fsm::AbortEvent{&coordinator_});
    }
}

bool Scheduler::applyPendingFlatTerminal(Request* request) {
    if (request == nullptr || !request->HasPendingFlatTerminal()) {
        return false;
    }
    _assert(request->FlatCompletionState().Snapshot().outstanding_count == 1,
            "flat terminal requires the final completion fence");
    std::optional<FlatPendingTerminal> terminal = request->TakePendingFlatTerminal();
    _assert(terminal.has_value(), "flat terminal disappeared while applying it");
    if (*terminal == FlatPendingTerminal::kAbort) {
        applyFlatAbort(request->Id());
    } else {
        applyFlatFinish(request->Id());
    }
    return true;
}
#endif

void Scheduler::handleEvent(const forward::UpdateReserveNumTokens& event) {
    if (auto req = find_request(event.request_id)) {
        req->Apply(fsm::UpdateReserveNumTokensEvent{event.reserve_num_tokens_in_next_schedule_event});
    }
}
void Scheduler::handleEvent(const forward::ExtendResult& event) {
#if TOKENSPEED_FLAT_KVCACHE
    if (config_.UsesExplicitFlatPools()) {
        if (!event.flat_kv_completion.has_value()) {
            throw std::invalid_argument("explicit flat KV ExtendResult requires completion metadata");
        }
        Request* request = find_request(event.request_id);
        if (request == nullptr) {
            return;
        }
        FlatKVCompletionPrepareResult fifo_prepared = flat_completion_ledger_.Prepare(
            request->FlatCompletionState(), event.request_id, *event.flat_kv_completion, event.tokens);
        if (fifo_prepared.disposition == FlatKVCompletionDisposition::kStale) {
            return;
        }
        _assert(fifo_prepared.ticket.has_value(), "non-stale flat completion has no retirement ticket");
        const bool final_fence = request->FlatCompletionState().Snapshot().outstanding_count == 1;
        if (fifo_prepared.disposition == FlatKVCompletionDisposition::kApplied) {
            _assert(fifo_prepared.ready.has_value(), "applied flat completion has no ready view");
            ApplyFlatCompletion(*fifo_prepared.ready, request);
        } else {
            _assert(!fifo_prepared.ready.has_value(), "canceled flat completion unexpectedly has a ready view");
        }

        // The current event is the physical execution fence for the validated
        // FIFO front. When it is also the last fence, finish every fallible
        // terminal/reclaim/reservation update before retiring the ticket.
        // Failures propagate to the event-loop boundary; this scheduler does
        // not attempt to resume from the partially applied completion.
        if (final_fence) {
            if (!applyPendingFlatTerminal(request)) {
                FinalizeFlatQuiescentState(request);
            }
        }
        (void)flat_completion_ledger_.Commit(request->FlatCompletionState(), *fifo_prepared.ticket);
        return;
    }
    std::size_t retired = 0;
    if (pending_forward_results_.find(event.request_id) != pending_forward_results_.end()) {
        // Preserve the legacy ABI's permissive late-result behavior:
        // decrement only when a debt entry is still live.
        retired = 1;
    }
#endif
    if (auto req = find_request(event.request_id)) {
        const std::int32_t protected_tail_tokens = config_.overlap_schedule_depth * config_.decode_input_tokens;
        req->Apply(fsm::ExtendResultEvent{event.request_id, event.tokens,
                                          hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr,
                                          protected_tail_tokens});
#if TOKENSPEED_FLAT_KVCACHE
        consumeFlatCompletionDebt(event.request_id, retired);
#endif
    }
}

#if TOKENSPEED_FLAT_KVCACHE
void Scheduler::ApplyFlatCompletion(const FlatKVReadyCompletion& ready, Request* request) {
    _assert(request != nullptr && request->Id() == ready.request_id, "active flat KV completion lost its request");
    bool append_tokens = false;
    if (ready.apply_fsm_result) {
        append_tokens = request->PrepareFlatResultAppend(ready.tokens.size());
    }

    std::optional<FlatKVWriteProgress>& progress_slot = request->FlatWriteProgress();
    const bool inserted_progress = !progress_slot.has_value();
    if (!progress_slot.has_value()) {
        FlatKVWriteProgress initial;
        initial.table_generation = ready.table_generation;
        initial.published_raw_ends.reserve(coordinator_.Schema().size());
        for (const KvCacheGroupSchema& group : coordinator_.Schema()) {
            initial.published_raw_ends.push_back(ready.dispatch_raw_start / group.entry_stride_tokens *
                                                 group.entry_stride_tokens);
        }
        progress_slot.emplace(std::move(initial));
    }
    FlatKVWriteProgress& progress = *progress_slot;
    const std::size_t original_base_hash_count = progress.base_hashes.size();

    try {
        _assert(progress.table_generation == ready.table_generation,
                "flat write progress generation disagrees with completion ledger");
        _assert(progress.published_raw_ends.size() == ready.ready_raw_ends.size(),
                "flat write progress group count disagrees with completion");

        std::int32_t max_hash_raw_end = 0;
        for (std::size_t i = 0; i < ready.ready_raw_ends.size(); ++i) {
            const std::int32_t ready_raw_end = ready.ready_raw_ends[i];
            _assert(ready_raw_end >= progress.published_raw_ends[i],
                    "flat completion would move an aligned ready end backward");
            if (coordinator_.GroupSchema(i).prefix_role != KvPrefixRole::kNone) {
                max_hash_raw_end = std::max(max_hash_raw_end, ready_raw_end);
            }
        }

        if (!config_.disable_prefix_cache && max_hash_raw_end > 0) {
            const std::int32_t base_block_size = coordinator_.BaseBlockSize();
            const std::int32_t required_base_pages = max_hash_raw_end / base_block_size;
            _assert(required_base_pages >= 0, "flat completion produced a negative base-page count");
            _assert(progress.base_hashes.size() <= static_cast<std::size_t>(required_base_pages),
                    "flat completion accepted end moved behind the hashed prefix");
            if (progress.base_hashes.size() < static_cast<std::size_t>(required_base_pages)) {
                const std::size_t first_fresh_page = progress.base_hashes.size();
                const std::size_t required_pages = static_cast<std::size_t>(required_base_pages);
                const std::size_t page_size = static_cast<std::size_t>(base_block_size);
                const std::size_t first_token = first_fresh_page * page_size;
                const std::size_t past_last_token = required_pages * page_size;
                const std::size_t stable_tokens = static_cast<std::size_t>(request->TokenSize());
                const std::size_t prospective_tokens = stable_tokens + (append_tokens ? ready.tokens.size() : 0);
                _assert(past_last_token <= prospective_tokens,
                        "accepted flat KV pages exceed prospective stable request tokens");

                std::span<const std::int32_t> stable_fresh_tokens;
                const std::size_t stable_end = std::min(past_last_token, stable_tokens);
                if (first_token < stable_end) {
                    stable_fresh_tokens = request->GetTokenSlice(TokenContainer::Window{
                        .begin = static_cast<std::int32_t>(first_token),
                        .size = static_cast<std::int32_t>(stable_end - first_token),
                    });
                }

                std::span<const std::int32_t> appended_fresh_tokens;
                if (past_last_token > stable_tokens) {
                    const std::size_t appended_begin = std::max(first_token, stable_tokens) - stable_tokens;
                    const std::size_t appended_count = past_last_token - std::max(first_token, stable_tokens);
                    _assert(appended_begin + appended_count <= ready.tokens.size(),
                            "flat completion result tokens do not cover the hash interval");
                    appended_fresh_tokens =
                        std::span<const std::int32_t>{ready.tokens}.subspan(appended_begin, appended_count);
                }
                _assert(stable_fresh_tokens.size() + appended_fresh_tokens.size() == past_last_token - first_token,
                        "flat completion hash spans have the wrong combined size");

                const std::size_t fresh_page_count = required_pages - first_fresh_page;
                progress.base_hashes.reserve(required_pages);
                const std::string empty_prior;
                const std::string* prior = progress.base_hashes.empty() ? &empty_prior : &progress.base_hashes.back();
                for (std::size_t page_index = 0; page_index < fresh_page_count; ++page_index) {
                    const std::size_t offset = page_index * page_size;
                    const std::size_t past_page = offset + page_size;
                    const std::size_t stable_part_begin = std::min(offset, stable_fresh_tokens.size());
                    const std::size_t stable_part_end = std::min(past_page, stable_fresh_tokens.size());
                    const std::span<const std::int32_t> stable_part =
                        stable_fresh_tokens.subspan(stable_part_begin, stable_part_end - stable_part_begin);
                    const std::size_t appended_part_begin =
                        offset > stable_fresh_tokens.size() ? offset - stable_fresh_tokens.size() : 0;
                    const std::size_t appended_part_end =
                        past_page > stable_fresh_tokens.size() ? past_page - stable_fresh_tokens.size() : 0;
                    const std::span<const std::int32_t> appended_part =
                        appended_fresh_tokens.subspan(appended_part_begin, appended_part_end - appended_part_begin);
                    _assert(stable_part.size() + appended_part.size() == page_size,
                            "flat completion hash page is not fully covered by stable and appended spans");
                    progress.base_hashes.push_back(HashPageSegments(stable_part, appended_part, *prior));
                    prior = &progress.base_hashes.back();
                }
                _assert(progress.base_hashes.size() == required_pages,
                        "flat completion hash count disagrees with its page interval");
            }
            coordinator_.PublishReadyBlocks(request->FlatBlockTablesRef(), progress.base_hashes,
                                            progress.published_raw_ends, ready.ready_raw_ends);
        }
    } catch (...) {
        if (inserted_progress) {
            progress_slot.reset();
        } else {
            while (progress.base_hashes.size() > original_base_hash_count) {
                progress.base_hashes.pop_back();
            }
        }
        throw;
    }

    request->CommitFlatResultAppend(ready.tokens, append_tokens);
    for (std::size_t i = 0; i < ready.ready_raw_ends.size(); ++i) {
        progress.published_raw_ends[i] = ready.ready_raw_ends[i];
    }
    progress.accepted_raw_end = ready.accepted_raw_end;

    if (ready.accepted_raw_end == ready.dispatch_raw_end) {
        coordinator_.ReclaimExpired(request->FlatBlockTablesRef(), ready.accepted_raw_end);
    }
}

void Scheduler::FinalizeFlatQuiescentState(Request* request) {
    if (request == nullptr || request->FlatBlockTablesEmpty()) {
        return;
    }
    _assert(request->FlatCompletionState().Snapshot().outstanding_count == 1,
            "flat reclaim/rewind requires the final completion fence");
    std::optional<FlatKVWriteProgress>& progress_slot = request->FlatWriteProgress();
    if (!progress_slot.has_value()) {
        return;
    }
    FlatKVWriteProgress& progress = *progress_slot;
    // All exported table consumers are now fenced. Full-accept callbacks may
    // already have reclaimed expired sliding fronts; only quiescence permits
    // rejected/protected tail rewind and reuse.
    coordinator_.ReclaimExpired(request->FlatBlockTablesRef(), progress.accepted_raw_end);
    coordinator_.RewindTail(request->FlatBlockTablesRef(), progress.accepted_raw_end,
                            /*retain_raw_end=*/progress.accepted_raw_end);

    // Rewind may change the tail capacity that the request's existing account
    // was computed from. Keep only the future slots promised beyond the next
    // schedulable decode; PrefillDone still owes that first decode as well.
    std::size_t reserved_slots = 0;
    if (request->Is<fsm::PrefillDone>()) {
        reserved_slots = static_cast<std::size_t>(config_.overlap_schedule_depth) + 1;
    } else if (request->Is<fsm::Decoding>()) {
        reserved_slots = static_cast<std::size_t>(config_.overlap_schedule_depth);
    }
    if (reserved_slots == 0) {
        request->FlatReservation().Clear();
        return;
    }
    const std::int32_t reserved_tokens =
        decodeReservationTokensForSlots(request->GetReserveNumTokensInNextScheduleEvent(), reserved_slots);
    request->FlatReservation().Set(coordinator_.BlocksNeededByPool(request->FlatBlockTablesRef(), reserved_tokens));
}
#endif

void Scheduler::handleEvent(const forward::Abort& event) {
#if TOKENSPEED_FLAT_KVCACHE
    if (deferFlatTerminal(event.request_id, FlatPendingTerminal::kAbort)) {
        return;
    }
    applyFlatAbort(event.request_id);
#else
    auto iter = requests_.find(event.request_id);
    if (iter != requests_.end()) {
        iter->second->Apply(fsm::AbortEvent{});
    }
#endif
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

    auto now = std::chrono::steady_clock::now();
    for (TreeNode* n : spec.nodes) n->Touch(now);

    if (!spec.request_id.empty()) {
        if (auto* req = find_request(spec.request_id)) {
            req->Apply(
                fsm::WriteBackDoneEvent{&radixPrefixCache(), hybrid_prefix_cache_ ? &*hybrid_prefix_cache_ : nullptr});
        }
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
    // Radix loadbacks emit no LoadBackDone today: unknown op_ids are silently ignored.
}

}  // namespace tokenspeed
