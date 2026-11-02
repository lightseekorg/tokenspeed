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
#include <limits>
#include <stdexcept>

#include "cache/prefix/prefix_hasher.h"
#include "fsm/storage_events.h"

namespace tokenspeed {

StorageStats Scheduler::GetStorageStats() const {
    StorageStats stats;
    if (!storage_transfers_) return stats;
    stats.operations = storage_transfers_->NumOperations();
    stats.held_blocks = storage_transfers_->NumHeldBlocks();
    stats.quarantined = storage_transfers_->NumQuarantined();
    stats.negative_keys = storage_transfers_->NumNegativeKeys();
    stats.backup_candidates = storage_transfers_->NumBackupCandidates();
    stats.wait_timeouts = storage_wait_timeouts_;
    stats.ready_expirations = storage_ready_expirations_;
    stats.pressure_gate_closed = storage_transfers_->PressureGateClosed();
    stats.pressure_bypasses = storage_pressure_bypasses_;
    stats.pressure_dropped_backup_keys = storage_transfers_->NumPressureDroppedBackups();
    stats.overflow_dropped_backup_keys = storage_transfers_->NumOverflowDroppedBackups();
    stats.unavailable_dropped_backup_keys = storage_transfers_->NumUnavailableDroppedBackups();
    stats.disabled_dropped_backup_keys = storage_transfers_->NumDisabledDroppedBackups();
    stats.cleanup_dropped_backup_keys = storage_transfers_->NumCleanupDroppedBackups();
    for (const auto& request : requests_) {
        stats.waiting_requests += request->Is<fsm::Prefetching>();
        const auto* admission = request->AdmissionResourcesPtr();
        stats.ready_requests += admission != nullptr && admission->storage_hold.has_value();
    }
    return stats;
}

void Scheduler::expireStorage(Request& request) {
    if (request.Is<fsm::Prefetching>()) {
        ++storage_wait_timeouts_;
    } else if (const auto* admission = request.AdmissionResourcesPtr();
               admission != nullptr && admission->storage_hold) {
        ++storage_ready_expirations_;
    }
    releaseStorage(request);
}

std::int32_t Scheduler::cacheablePrefixTokens(const Request& request) const {
    const auto replay = std::max(config_.prefix_replay_tokens, 1);
    return std::max(std::min(request.PrefillSize() - replay, request.MaxCachedPrefixTokens()), 0);
}

void Scheduler::handleEvent(const cache::StorageTick& event) {
    if (event.now_us < storage_now_us_) throw std::invalid_argument("L3 control clock moved backwards");
    if (storage_transfers_) storage_transfers_->UpdatePressure(event.pressure);
    storage_now_us_ = event.now_us;
    storage_allowed_ = event.allow_storage;
    forward_allowed_ = event.allow_forward;
    for (const auto& request : requests_) {
        const auto* prefetch = request->GetIf<fsm::Prefetching>();
        const auto* admission = request->AdmissionResourcesPtr();
        if ((prefetch != nullptr && storage_now_us_ >= prefetch->deadline_us) ||
            (admission != nullptr && admission->storage_hold &&
             storage_now_us_ >= admission->storage_hold->deadline_us)) {
            expireStorage(*request);
        }
    }
}

void Scheduler::handleEvent(const cache::StorageDone& event) {
    if (!storage_transfers_) throw std::invalid_argument("storage completion without configured L3");
    storage_transfers_->Complete(event);
}

void Scheduler::handleEvent(const cache::StorageDetach& event) {
    if (auto* request = findRequest(event.request_id)) releaseStorage(*request);
}

void Scheduler::releaseStorage(Request& request) {
    if (!storage_transfers_) return;
    if (const auto* prefetch = request.GetIf<fsm::Prefetching>()) {
        storage_transfers_->Detach(prefetch->operation_id);
        request.Apply(fsm::FinishPrefetchEvent{std::nullopt});
    }
    if (auto* admission = request.AdmissionResourcesPtr()) {
        if (admission->storage_hold) {
            storage_transfers_->Detach(admission->storage_hold->operation_id);
            admission->storage_hold.reset();
        }
        admission->storage_prepared = true;
    }
}

void Scheduler::prepareStorage(Request& request) {
    if (!storage_transfers_) return;
    const auto& limits = storage_transfers_->Config();
    if (auto* state = request.GetIf<fsm::Prefetching>()) {
        if (storage_transfers_->Completed(state->operation_id) && !storage_transfers_->IsQuery(state->operation_id)) {
            const auto deadline =
                storage_now_us_ +
                std::min(limits.ready_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
            request.Apply(fsm::FinishPrefetchEvent{fsm::StorageReadyHold{state->operation_id, deadline}});
            return;
        }
        if (storage_now_us_ >= state->deadline_us) {
            expireStorage(request);
            return;
        }
        const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = state->hashes,
                                                    .tokens = request.TokenSlice(0, state->max_tokens),
                                                    .max_tokens = state->max_tokens,
                                                    .tail = state->tail};
        // Another request's D2H/GET may have fulfilled this dependency. This
        // bounded local check does not rehash the prompt while it waits.
        const auto probe = coordinator_.ProbePromptPrefix(prompt);
        if (std::max(probe.device.num_common_tokens, probe.host.num_common_tokens) >= state->max_tokens) {
            releaseStorage(request);
            return;
        }
        // Congestion skips a new GET; an existing GET keeps its original
        // deadline and Host references until native retirement.
        if (storage_transfers_->IsQuery(state->operation_id) && storage_transfers_->PressureGateClosed()) {
            ++storage_pressure_bypasses_;
            releaseStorage(request);
            return;
        }
        if (!storage_transfers_->Completed(state->operation_id) || !storage_allowed_) return;
        const auto hits = storage_transfers_->QueryHits(state->operation_id);
        storage_transfers_->Detach(state->operation_id);
        state->operation_id = storage_transfers_->StartRead(prompt, hits);
        if (!state->operation_id) releaseStorage(request);
        return;
    }

    auto* admission = request.AdmissionResourcesPtr();
    if (admission == nullptr || admission->storage_prepared || !storage_allowed_) return;
    admission->storage_prepared = true;
    // Decode uses its ordinary remote landing. No extra L3 handshake here.
    if (config_.role == Role::kD || config_.disable_prefix_cache) return;
    if (!storage_transfers_->CanStart()) {
        if (storage_transfers_->PressureGateClosed()) ++storage_pressure_bypasses_;
        return;
    }
    const auto grain = config_.prefix_granularity;
    const auto max_tokens = cacheablePrefixTokens(request);
    auto pages = request.FullPrefixPages(false);
    pages.resize(std::min(pages.size(), static_cast<std::size_t>(max_tokens / grain)));
    auto hashes = ComputePrefixHashes(pages, "");
    std::int32_t bound = max_tokens;
    std::optional<TailPublication> tail;
    const auto setBound = [&request, &hashes, &bound, &tail, grain](std::int32_t value) {
        bound = value;
        hashes.resize(static_cast<std::size_t>(bound / grain));
        tail.reset();
        if (const auto tail_tokens = bound % grain; tail_tokens > 0) {
            const ContentHash prior = hashes.empty() ? ContentHash{} : hashes.back();
            tail = TailPublication{.tail_tokens = tail_tokens,
                                   .hash = HashPrefixPage(request.TokenSlice(bound - tail_tokens, tail_tokens), prior)};
        }
    };
    const auto prompt = [&request, &hashes, &bound, &tail]() {
        return CacheCoordinator::PromptPrefix{
            .prefix_hashes = hashes, .tokens = request.TokenSlice(0, bound), .max_tokens = bound, .tail = tail};
    };
    setBound(max_tokens);
    const auto probe = coordinator_.ProbePromptPrefix(prompt());
    const auto local = std::max(probe.device.num_common_tokens, probe.host.num_common_tokens);
    if (max_tokens - local < limits.min_prefetch_tokens) return;
    const auto end = std::min<std::int64_t>(max_tokens, static_cast<std::int64_t>(local) + limits.max_prefetch_tokens);
    setBound(static_cast<std::int32_t>(end));
    auto query = coordinator_.MissingStorageKeys(prompt(), limits.max_keys_per_operation);
    setBound(query.max_tokens);
    auto keys = std::move(query.keys);
    if (bound - local < limits.min_prefetch_tokens || keys.empty()) return;
    // A known failed immutable object remains unavailable for this attempt.
    std::erase_if(keys, [this](const auto& key) { return storage_transfers_->IsNegative(key); });
    const auto operation = storage_transfers_->StartQuery(std::move(keys));
    if (!operation) return;
    const auto deadline =
        storage_now_us_ + std::min(limits.wait_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
    request.Apply(fsm::StartPrefetchEvent{operation, deadline, std::move(hashes), bound, std::move(tail)});
}
}  // namespace tokenspeed
