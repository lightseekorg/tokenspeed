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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "core/token_container.h"
#include "cache/tier/transfer.h"
#include "fsm/forward_states.h"
#include "scheduler/capacity_model.h"
#include "scheduler/operations/forward.h"
#include "scheduler/operations/group_demands.h"
#include "cache/prefix/prefix_hasher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int32_t hostPoolBlocks(const SchedulerConfig& config) {
    return config.HasHostCache() ? config.host_allocator.NumUsableBlocks() : 0;
}

std::vector<std::int32_t> slotsPerParentByGroup(const SchedulerConfig& config) {
    std::vector<std::int32_t> slots_per_group;
    slots_per_group.reserve(config.cache_groups.size());
    for (const CacheGroupConfig& group : config.cache_groups) {
        slots_per_group.push_back(group.cache_blocks_per_lcm_block);
    }
    return slots_per_group;
}

// config_ is the first member, so routing it through this helper validates the
// configuration before any pool or the coordinator is built off it.
SchedulerConfig validated(SchedulerConfig config) {
    config.Validate();
    return config;
}

CacheKey eventKey(const CacheKey& key) {
    // External events describe one scheduler-level boundary. Fold every
    // group/child offset behind that boundary into the same accounting key.
    return CacheKey{
        .namespace_id = key.namespace_id,
        .group_id = 0,
        .content_hash = key.content_hash,
    };
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_{validated(std::move(config))},
      req_pool_allocator_{config_.max_batch_size},
      block_pool_{config_.device_allocator.NumUsableBlocks(), slotsPerParentByGroup(config_)},
      host_pool_{hostPoolBlocks(config_), slotsPerParentByGroup(config_)},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_), config_.prefix_granularity, block_pool_,
                                   config_.enable_l3_storage, hostPoolBlocks(config_) > 0 ? &host_pool_ : nullptr,
                                   config_.StreamsDeviceCacheToHost())},
      tier_transfers_{coordinator_} {
    // config_.Validate() already ran; the body only derives state from it.
    cache_group_ids_.reserve(config_.cache_groups.size());
    for (const CacheGroupConfig& group : config_.cache_groups) {
        cache_group_ids_.push_back(group.group_id);
    }
    max_single_request_tokens_ = CapacityModel{config_}.MaxSingleRequestTokens(coordinator_.TotalLcmBlocks());

    if (config_.enable_kv_cache_events) {
        coordinator_.SetCacheMutationSink([this](const CacheKey& key, CacheCoordinator::CacheMutation) {
            markKvEventBoundaryForReconcile(eventKey(key));
        });
    }

    if (const char* level = std::getenv("SPDLOG_LEVEL")) {
        spdlog::set_level(spdlog::level::from_str(level));
    }
}

Request* Scheduler::findRequest(const std::string& request_id) {
    const auto it = requests_by_id_.find(request_id);
    return it == requests_by_id_.end() ? nullptr : it->second;
}

bool Scheduler::pdTransferInFlight(const Request& request) const {
    switch (config_.role) {
        case Role::kD:
            return request.Is<fsm::RemotePrefilling>();
        case Role::kP:
            return request.HoldsPages();
        case Role::kFused:
            return false;
    }
    return false;
}

bool Scheduler::PdTransferPinned(const std::string& request_id) const {
    const auto it = requests_by_id_.find(request_id);
    return it != requests_by_id_.end() && pdTransferInFlight(*it->second);
}

std::size_t Scheduler::groupIndex(const std::string& group_id) const {
    const auto it = std::ranges::find(cache_group_ids_, group_id);
    if (it == cache_group_ids_.end()) {
        throw std::out_of_range("Scheduler: unknown cache group '" + group_id + "'");
    }
    return static_cast<std::size_t>(std::distance(cache_group_ids_.begin(), it));
}

std::vector<KvCacheEvent> Scheduler::DrainKvEvents() {
    // An event reports the net change since the last drain: a boundary is
    // published exactly while every child block of it is cached. Mutations
    // mark boundaries in whatever order the coordinator touches pages, so the
    // batch is sorted by chain depth: removals leaf-first, then Stored
    // parent-first, because consumers resolve a Stored event's
    // parent_block_hash against what they have already received.
    std::vector<std::pair<std::int32_t, KvCacheEvent>> removed;
    std::vector<std::pair<std::int32_t, KvCacheEvent>> stored;
    for (const CacheKey& boundary : std::exchange(kv_event_boundaries_to_reconcile_, {})) {
        const auto it = kv_event_boundaries_.find(boundary);
        FatalCheck(it != kv_event_boundaries_.end(),
                   "KV event boundary marked for reconcile lost its token descriptor");
        KvEventBoundary& event_boundary = it->second;
        event_boundary.needs_reconcile = false;
        const CacheCoordinator::BoundaryResidency residency = coordinator_.DeviceBoundaryResidency(boundary);
        const bool complete = residency == CacheCoordinator::BoundaryResidency::kComplete;
        if (complete && !event_boundary.published) {
            stored.emplace_back(event_boundary.depth, event_boundary.stored);
        } else if (!complete && event_boundary.published) {
            removed.emplace_back(event_boundary.depth,
                                 KvBlockRemovedEvent{.block_hashes = event_boundary.stored.block_hashes});
        }
        event_boundary.published = complete;
        if (residency == CacheCoordinator::BoundaryResidency::kNone) {
            kv_event_boundaries_.erase(it);
        }
    }
    std::stable_sort(removed.begin(), removed.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::stable_sort(stored.begin(), stored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<KvCacheEvent> events;
    events.reserve(removed.size() + stored.size());
    for (auto& [depth, event] : removed) {
        events.push_back(std::move(event));
    }
    for (auto& [depth, event] : stored) {
        events.push_back(std::move(event));
    }
    return events;
}

bool Scheduler::ClearL1Cache() {
    return clearCache(false);
}

bool Scheduler::ClearCache() {
    return clearCache(true);
}

bool Scheduler::CanClearCache() const {
    return cacheIsClearable(true);
}

bool Scheduler::cacheIsClearable(bool include_host) const {
    const bool has_pd_transfers = std::ranges::any_of(
        requests_, [this](const std::unique_ptr<Request>& request) { return pdTransferInFlight(*request); });
    const bool has_tier_transfers = tier_transfers_.HasAnyInFlight();
    if (has_pd_transfers || has_tier_transfers) {
        return false;
    }
    return coordinator_.CacheIsClearable(include_host);
}

bool Scheduler::clearCache(bool include_host) {
    // A live request's pages are protected by their pins, and the coordinator
    // completes its pin check before mutating anything -- so residency is not
    // this function's business. What IS its business are the writers the pins
    // do not cover: an asynchronous transfer still landing into a cached
    // block would race a clear that succeeded on the pin check alone.
    const bool has_pd_transfers = std::ranges::any_of(
        requests_, [this](const std::unique_ptr<Request>& request) { return pdTransferInFlight(*request); });
    const bool has_tier_transfers = tier_transfers_.HasAnyInFlight();
    if (has_pd_transfers || has_tier_transfers) {
        spdlog::info("[Scheduler] flush L1 cache rejected: pd_transfers={} tier_transfers={}", has_pd_transfers,
                     has_tier_transfers);
        return false;
    }
    const bool cleared = include_host ? coordinator_.ClearCache() : coordinator_.ClearDeviceCache();
    if (!cleared) {
        spdlog::info("[Scheduler] flush {}cache rejected: cached blocks are still pinned", include_host ? "" : "L1 ");
        return false;
    }
    spdlog::info("[Scheduler] flush {}cache completed", include_host ? "" : "L1 ");
    return true;
}

void Scheduler::registerKvEventPrefixPages(const Request& request, std::span<const std::string> prefix_hashes,
                                           std::int32_t first_page) {
    if (!config_.enable_kv_cache_events) {
        return;
    }
    _assert(first_page >= 0 && static_cast<std::size_t>(first_page) <= prefix_hashes.size(),
            "KV event page range is invalid");
    const std::vector<std::span<const std::int32_t>> token_pages = request.FullPrefixPages(false);
    _assert(prefix_hashes.size() <= token_pages.size(), "KV event hashes exceed the request's complete pages");

    KvEventHashProgress& progress = kv_event_hash_progress_[request.Id()];
    for (std::size_t i = progress.block_hashes.size(); i < prefix_hashes.size(); ++i) {
        const std::optional<std::uint64_t> parent_hash =
            i == 0 ? std::nullopt : std::optional<std::uint64_t>{progress.block_hashes[i - 1]};
        progress.block_hashes.push_back(HashKvBlock(token_pages[i], parent_hash));
    }

    for (std::size_t i = static_cast<std::size_t>(first_page); i < prefix_hashes.size(); ++i) {
        CacheKey key{.content_hash = prefix_hashes[i]};
        const std::optional<std::uint64_t> parent_hash =
            i == 0 ? std::nullopt : std::optional<std::uint64_t>{progress.block_hashes[i - 1]};
        KvBlockStoredEvent event{
            .block_hashes = {progress.block_hashes[i]},
            .parent_block_hash = parent_hash,
            .token_ids = std::vector<std::int32_t>(token_pages[i].begin(), token_pages[i].end()),
            .block_size = config_.prefix_granularity,
        };
        const auto [it, inserted] = kv_event_boundaries_.try_emplace(
            key, KvEventBoundary{.stored = std::move(event), .depth = static_cast<std::int32_t>(i)});
        FatalCheck(inserted || (it->second.stored.block_hashes.front() == progress.block_hashes[i] &&
                                it->second.depth == static_cast<std::int32_t>(i)),
                   "one cache content hash mapped to different KV event blocks");
        // The next drain drops the descriptor if the publication never lands.
        markKvEventBoundaryForReconcile(key);
    }
}

void Scheduler::markKvEventBoundaryForReconcile(const CacheKey& boundary) {
    // Descriptors are erased only by DrainKvEvents, so every cache mutation of
    // a registered boundary finds one here.
    const auto it = kv_event_boundaries_.find(boundary);
    FatalCheck(it != kv_event_boundaries_.end(), "cache mutation on a KV event boundary with no token descriptor");
    if (!it->second.needs_reconcile) {
        it->second.needs_reconcile = true;
        kv_event_boundaries_to_reconcile_.push_back(boundary);
    }
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
    std::unordered_set<std::string> request_ids;
    request_ids.reserve(request_specs.size());
    std::vector<std::unique_ptr<Request>> pending_requests;
    pending_requests.reserve(request_specs.size());
    for (const RequestSpec& spec : request_specs) {
        if (spec.tokens.empty()) {
            throw std::invalid_argument("Scheduler: request tokens must be non-empty");
        }
        if (requests_by_id_.contains(spec.request_id) || !request_ids.insert(spec.request_id).second) {
            throw std::invalid_argument("Scheduler: duplicate request id '" + spec.request_id + "'");
        }
        if (spec.max_new_tokens < 0) {
            throw std::invalid_argument("Scheduler: max_new_tokens must be non-negative");
        }
        const std::int64_t generation_reserve =
            config_.role == Role::kP ? 0 : std::max<std::int64_t>(spec.max_new_tokens, config_.decode_input_tokens);
        const std::int64_t token_limit = static_cast<std::int64_t>(spec.tokens.size()) + generation_reserve;
        if (token_limit > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Scheduler: request token limit exceeds int32 range");
        }
        if (token_limit > max_single_request_tokens_) {
            throw std::invalid_argument("Scheduler: request token limit exceeds cache capacity");
        }
        pending_requests.push_back(std::make_unique<Request>(spec, config_.prefix_granularity, config_.role));
    }

    requests_.reserve(requests_.size() + pending_requests.size());
    for (std::size_t i = 0; i < request_specs.size(); ++i) {
        const bool inserted = requests_by_id_.emplace(request_specs[i].request_id, pending_requests[i].get()).second;
        FatalCheck(inserted, "validated request id became duplicate before insertion");
        requests_.push_back(std::move(pending_requests[i]));
    }
}

std::size_t Scheduler::BootstrappingSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(
        requests_, [](const std::unique_ptr<Request>& request) { return request->Is<fsm::Bootstrapping>(); }));
}

std::size_t Scheduler::WaitingSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const std::unique_ptr<Request>& request) {
        return request->IsAnyOf<fsm::Submitted, fsm::Retracted>();
    }));
}

std::size_t Scheduler::DecodingSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(
        requests_, [](const std::unique_ptr<Request>& request) { return request->Is<fsm::Decoding>(); }));
}

std::size_t Scheduler::PrefillSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const std::unique_ptr<Request>& request) {
        return request->IsAnyOf<fsm::Prefilling, fsm::RemotePrefilling, fsm::PrefillAwaitingResult, fsm::PrefillDone>();
    }));
}

std::size_t Scheduler::RemotePrefillSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(
        requests_, [](const std::unique_ptr<Request>& request) { return request->Is<fsm::RemotePrefilling>(); }));
}

std::size_t Scheduler::PdTransferSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(
        requests_, [this](const std::unique_ptr<Request>& request) { return pdTransferInFlight(*request); }));
}

std::int32_t Scheduler::ActiveLcmBlocks() const {
    std::vector<std::span<const BlockTable>> request_tables;
    request_tables.reserve(requests_.size());
    for (const auto& request : requests_) {
        if (!request->HoldsPages()) {
            continue;
        }
        request_tables.emplace_back(request->BlockTablesRef());
    }
    return coordinator_.NumActiveLcmBlocks(request_tables);
}

std::int32_t Scheduler::CacheGroupTotalPages(const std::string& group_id) const {
    return config_.cache_groups[groupIndex(group_id)].total_pages;
}

std::int32_t Scheduler::CacheGroupAvailablePages(const std::string& group_id) const {
    return coordinator_.GroupAvailablePages(static_cast<std::int32_t>(groupIndex(group_id)));
}

std::vector<std::string> Scheduler::PrefixHashesForTokens(const std::vector<std::int32_t>& tokens) const {
    TokenContainer container(tokens);
    std::vector<std::span<const std::int32_t>> prefix_pages =
        container.FullPrefixPages(config_.prefix_granularity, false);
    // The last prompt token is always recomputed. Admission probes the same
    // (n - 1) / prefix_granularity candidate pages.
    const std::int32_t candidate_prefix_pages =
        std::max((static_cast<std::int32_t>(tokens.size()) - 1) / config_.prefix_granularity, 0);
    prefix_pages.resize(std::min(prefix_pages.size(), static_cast<std::size_t>(candidate_prefix_pages)));
    return ComputePrefixHashes(prefix_pages, "");
}

std::vector<std::string> Scheduler::WaitingPrefixHashes() const {
    std::vector<Request*> candidates;
    candidates.reserve(requests_.size());
    for (const auto& request : requests_) {
        candidates.push_back(request.get());
    }
    Request* readmission = nextReadmission(candidates);

    bool hol_blocks_new_prompts = false;
    for (const auto& request : requests_) {
        const auto* prefilling = request->GetIf<fsm::Prefilling>();
        if (prefilling != nullptr) {
            hol_blocks_new_prompts = true;
            break;
        }
    }
    const std::int32_t occupied = static_cast<std::int32_t>(PrefillSize() + DecodingSize());
    const std::int32_t free_slots = config_.max_batch_size - occupied;
    if (free_slots <= 0 && readmission == nullptr) {
        return {};
    }

    std::vector<std::string> hashes;
    std::unordered_set<std::string> seen;
    const auto append_hashes = [&](const Request& request) {
        std::vector<std::span<const std::int32_t>> prefix_pages = request.FullPrefixPages(/*except_last=*/false);
        const std::int32_t candidate_prefix_pages =
            std::max((request.PrefillSize() - 1) / config_.prefix_granularity, 0);
        prefix_pages.resize(std::min(prefix_pages.size(), static_cast<std::size_t>(candidate_prefix_pages)));
        for (std::string& content_hash : ComputePrefixHashes(prefix_pages, "")) {
            if (seen.insert(content_hash).second) {
                hashes.push_back(std::move(content_hash));
            }
        }
    };
    if (readmission != nullptr) {
        append_hashes(*readmission);
    }
    if (free_slots <= 0 || hol_blocks_new_prompts || AvailableLcmBlocks() <= 0) {
        return hashes;
    }
    std::int32_t remaining = free_slots;
    if (readmission != nullptr) {
        remaining = std::max(remaining - 1, 0);
    }
    for (Request* request : candidates) {
        if (remaining <= 0) {
            break;
        }
        if (request == readmission) {
            continue;
        }
        if (request->Is<fsm::Submitted>()) {
            append_hashes(*request);
            --remaining;
            continue;
        }
        const auto* retracted = request->GetIf<fsm::Retracted>();
        if (retracted != nullptr && !retracted->HasRecoverableSnapshot()) {
            append_hashes(*request);
            --remaining;
        }
    }
    return hashes;
}

std::int32_t Scheduler::RequestTokenSize(const std::string& id) const {
    const auto it = requests_by_id_.find(id);
    return it == requests_by_id_.end() ? -1 : it->second->TokenSize();
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    std::vector<WriteBackOperation> write_back_operations = std::exchange(pending_write_back_operations_, {});
    std::erase_if(requests_, [this](const auto& request) {
        if (!request->template Is<fsm::Finished>()) {
            return false;
        }
        kv_event_hash_progress_.erase(request->Id());
        requests_by_id_.erase(request->Id());
        return true;
    });

    std::vector<Request*> candidates;
    candidates.reserve(requests_.size());
    for (const auto& request : requests_) {
        if (request->IsAnyOf<fsm::Submitted, fsm::Prefilling, fsm::RemotePrefilling, fsm::PrefillDone, fsm::Decoding,
                             fsm::Retracted>()) {
            candidates.push_back(request.get());
        }
    }
    ExecutionPlan plan;
    auto [forward_operations, load_back_operations] =
        buildForwardOperations(plan, std::move(candidates), write_back_operations);

    plan.With(ForwardBatch{std::move(forward_operations)});

    if (config_.StreamsDeviceCacheToHost()) {
        // Boundary publications of live requests: their owners hold the pages,
        // and the ticket pins them until the ACK, so the copy stays off the
        // forward's critical path.
        if (auto store = tier_transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck)) {
            write_back_operations.push_back(std::move(*store));
        }
    }

    if (!write_back_operations.empty()) {
        plan.With(CacheOperation{WriteBackBatch{write_back_operations}});
    }
    if (!load_back_operations.empty()) {
        plan.With(CacheOperation{LoadBackBatch{load_back_operations}});
    }
    return plan;
}

void Scheduler::Advance(const ExecutionEvent& event) {
    for (const auto& item : event.Events()) {
        std::visit([this](const auto& inner) { handleEvent(inner); }, item);
    }
}

}  // namespace tokenspeed
