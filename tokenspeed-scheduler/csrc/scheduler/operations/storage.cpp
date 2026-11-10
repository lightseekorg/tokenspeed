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

bool Scheduler::SharedPrefixStorageEnabled() const {
    return storage_transfers_ && coordinator_.HasHostPool() && config_.role != Role::kD &&
           !config_.disable_prefix_cache && storage_transfers_->Config().max_operations >= 2 &&
           storage_transfers_->Config().max_host_blocks >= 2;
}

Scheduler::SharedPrefixPublication Scheduler::GetSharedPrefixPublication(std::span<const std::int32_t> tokens,
                                                                         std::int32_t max_prefix_tokens) const {
    if (max_prefix_tokens <= 0 || static_cast<std::size_t>(max_prefix_tokens) > tokens.size() ||
        tokens.size() > 1048576)
        throw std::invalid_argument("invalid bounded shared prefix publication query");
    if (!SharedPrefixStorageEnabled()) return {.status = "unsupported"};
    SharedPrefixPublication result{.status = "pending", .cache_epoch = storage_transfers_->PublicationEpoch()};
    const auto grain = coordinator_.PrefixGranularity();
    const auto input = tokens.first(static_cast<std::size_t>(max_prefix_tokens));
    std::vector<std::span<const std::int32_t>> pages;
    for (std::int32_t i = 0; i + grain <= max_prefix_tokens; i += grain) pages.push_back(input.subspan(i, grain));
    const auto hashes = ComputePrefixHashes(pages, "");
    const auto tail_tokens = max_prefix_tokens % grain;
    const auto tail =
        tail_tokens
            ? std::optional{TailPublication{
                  .tail_tokens = tail_tokens,
                  .hash = HashPrefixPage(input.last(tail_tokens), hashes.empty() ? ContentHash{} : hashes.back())}}
            : std::nullopt;
    const auto dependencies = coordinator_.SharedPrefixKeys(
        {.prefix_hashes = hashes, .tokens = input, .max_tokens = max_prefix_tokens, .tail = tail},
        static_cast<std::size_t>(storage_transfers_->Config().metadata_entries));
    if (dependencies.max_tokens == 0)
        result.status = "unsupported";
    else if (dependencies.max_tokens < 0)
        result.status = "too_large";
    else if (dependencies.max_tokens > 0 && !dependencies.keys.empty() &&
             std::ranges::all_of(dependencies.keys,
                                 [&](const auto& key) { return storage_transfers_->WasPublished(key); })) {
        result.status = "published";
        result.confirmed_prefix_tokens = max_prefix_tokens;
    } else if (storage_transfers_->PressureGateClosed() || storage_transfers_->NumQuarantined() > 0) {
        result.status = "pressure";
    }
    return result;
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
    // The request gives L3 up and keeps its place: the lease on the operation
    // it waits for, or on its Ready data, dies with the wait.
    if (request.Is<fsm::Prefetching>()) {
        request.Apply(fsm::FinishPrefetchEvent{std::nullopt});
    }
    if (auto* admission = request.AdmissionResourcesPtr()) {
        admission->storage_hold.reset();
        admission->storage_prepared = true;
    }
}

void Scheduler::prepareStorage(Request& request) {
    if (!storage_transfers_) return;
    const auto& limits = storage_transfers_->Config();
    if (auto* state = request.GetIf<fsm::Prefetching>()) {
        const std::uint64_t operation = state->lease.OperationId();
        if (storage_transfers_->Completed(operation) && !storage_transfers_->IsQuery(operation)) {
            const auto deadline =
                storage_now_us_ +
                std::min(limits.ready_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
            request.Apply(fsm::FinishPrefetchEvent{deadline});
            return;
        }
        if (storage_now_us_ >= state->deadline_us) {
            expireStorage(request);
            return;
        }
        const CacheCoordinator::PromptPrefix prompt{.prefix_hashes = state->hashes,
                                                    .tokens = request.TokenSlice(0, state->max_tokens),
                                                    .max_tokens = state->max_tokens,
                                                    .tail = state->tail,
                                                    .shared_prefix_tokens = request.SharedPrefixTokens()};
        // Another request's D2H/GET may have fulfilled this dependency. This
        // bounded local check does not rehash the prompt while it waits.
        const auto probe = coordinator_.ProbePromptPrefix(prompt);
        if (std::max(probe.device.num_common_tokens, probe.host.num_common_tokens) >= state->max_tokens) {
            releaseStorage(request);
            return;
        }
        // Congestion skips a new GET; an existing GET keeps its original
        // deadline and Host references until native retirement.
        if (storage_transfers_->IsQuery(operation) && storage_transfers_->PressureGateClosed()) {
            ++storage_pressure_bypasses_;
            releaseStorage(request);
            return;
        }
        if (!storage_transfers_->Completed(operation) || !storage_allowed_) return;
        const auto hits = storage_transfers_->QueryHits(operation);
        // The finished query retires before the GET starts: CanStart counts
        // its ticket.
        state->lease.reset();
        state->lease = storage_transfers_->StartRead(prompt, hits);
        if (!state->lease) releaseStorage(request);
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
        return CacheCoordinator::PromptPrefix{.prefix_hashes = hashes,
                                              .tokens = request.TokenSlice(0, bound),
                                              .max_tokens = bound,
                                              .tail = tail,
                                              .shared_prefix_tokens = request.SharedPrefixTokens()};
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
    auto key_tokens = std::move(query.key_tokens);
    if (bound - local < limits.min_prefetch_tokens || keys.empty()) return;
    _assert(key_tokens.size() == keys.size(), "Each storage query key needs a token span");
    // A known failed immutable object remains unavailable for this attempt.
    std::size_t kept = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (storage_transfers_->IsNegative(keys[i])) continue;
        keys[kept] = keys[i];
        key_tokens[kept] = key_tokens[i];
        ++kept;
    }
    keys.resize(kept);
    key_tokens.resize(kept);
    StorageLease query_lease = storage_transfers_->StartQuery(std::move(keys), std::move(key_tokens));
    if (!query_lease) return;
    const auto deadline =
        storage_now_us_ + std::min(limits.wait_timeout_us, std::numeric_limits<std::int64_t>::max() - storage_now_us_);
    request.Apply(fsm::StartPrefetchEvent{std::move(query_lease), deadline, std::move(hashes), bound, std::move(tail)});
}
}  // namespace tokenspeed
