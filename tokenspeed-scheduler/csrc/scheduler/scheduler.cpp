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
#include <bit>
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

// Stalled rounds (Scheduler::trackStalledRounds) before the first warning.
// Later warnings come at every doubling of the streak, so a stall that never
// ends still logs only a few dozen lines.
constexpr std::int64_t kStalledRoundsBeforeWarning = 1024;

std::int32_t hostPoolBlocks(const SchedulerConfig& config) {
    return config.HasHostCache() ? config.host_allocator.NumUsableBlocks() : 0;
}

std::int32_t snapshotPoolBlocks(const SchedulerConfig& config) {
    return config.HasSnapshotPool() ? config.snapshot_allocator.NumUsableBlocks() : 0;
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
      snapshot_slots_{config_.max_retracted_requests},
      block_pool_{config_.device_allocator.NumUsableBlocks(), slotsPerParentByGroup(config_)},
      host_pool_{hostPoolBlocks(config_), config_.host_group_packing},
      snapshot_pool_{snapshotPoolBlocks(config_), config_.host_group_packing},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_), config_.prefix_granularity, block_pool_,
                                   config_.enable_l3_storage, hostPoolBlocks(config_) > 0 ? &host_pool_ : nullptr,
                                   snapshotPoolBlocks(config_) > 0 ? &snapshot_pool_ : nullptr,
                                   config_.StreamsDeviceCacheToHost(),
                                   config_.host_write_policy == SchedulerConfig::HostWritePolicy::kWriteBack,
                                   config_.CheckpointRetentionRequests(), config_.checkpoint_retention_leaf_distance)},
      tier_transfers_{coordinator_},
      storage_transfers_{config_.enable_l3_storage
                             ? std::make_unique<StorageTransferManager>(coordinator_, *config_.storage_config)
                             : nullptr} {
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

    // Validate requires in-forward checkpoints, and with them prefix caching,
    // outside the D role, which computes no prompt; without a State group
    // there is no checkpoint to demote.
    bool has_state_group = false;
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        has_state_group = has_state_group || coordinator_.GroupIsState(static_cast<std::size_t>(i));
    }
    if (config_.demote_superseded_state_checkpoints && config_.role != Role::kD && has_state_group) {
        checkpoint_lineage_.emplace();
    }
    coordinator_.SetCheckpointDemotion(checkpoint_lineage_.has_value());

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
            return request.IsAnyOf<fsm::RemotePrefilling, fsm::PdTransferring>();
        case Role::kP:
            // Every page-holding state but Finishing, PdTransferring
            // included: SucceededEvent moves a request to Finishing once the
            // peer has read its pages.
            return request.HoldsPages() && !request.Is<fsm::Finishing>();
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
        // A tail's descriptor carries fewer tokens than a page.
        const auto descriptor_tokens = static_cast<std::int32_t>(event_boundary.stored.token_ids.size());
        const std::int32_t tail_tokens = descriptor_tokens < config_.prefix_granularity ? descriptor_tokens : 0;
        const CacheCoordinator::BoundaryResidency residency =
            coordinator_.DeviceBoundaryResidency(boundary, tail_tokens);
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
    const bool has_tier_transfers = tier_transfers_.HasAnyInFlight() || (include_host && StoragePending());
    if (has_pd_transfers || has_tier_transfers || RetractedSize() > 0) {
        return false;
    }
    return coordinator_.CacheIsClearable(include_host);
}

bool Scheduler::clearCache(bool include_host) {
    // A live request's pages are protected by their pins, and the coordinator
    // completes its pin check before mutating anything -- so residency is not
    // this function's business. What IS its business are the writers the pins
    // do not cover: an asynchronous transfer still landing into a cached
    // block would race a clear that succeeded on the pin check alone. And the
    // images the pins do not show: a request suspended with its KV on Host
    // continues from those bytes once restored, so a flush (a weight update)
    // under it would resume old-weight KV under new weights. Its L2 leg is
    // pinned and would refuse the Host clear by itself; its snapshot-pool leg
    // is not indexed anywhere, so the suspended set is checked directly.
    const bool has_pd_transfers = std::ranges::any_of(
        requests_, [this](const std::unique_ptr<Request>& request) { return pdTransferInFlight(*request); });
    const bool has_tier_transfers = tier_transfers_.HasAnyInFlight() || (include_host && StoragePending());
    const std::size_t suspended = RetractedSize();
    if (has_pd_transfers || has_tier_transfers || suspended > 0) {
        spdlog::info("[Scheduler] flush {}cache rejected: pd_transfers={} tier_transfers={} suspended_requests={}",
                     include_host ? "" : "L1 ", has_pd_transfers, has_tier_transfers, suspended);
        return false;
    }
    const bool cleared = include_host ? coordinator_.ClearCache() : coordinator_.ClearDeviceCache();
    if (!cleared) {
        spdlog::info("[Scheduler] flush {}cache rejected: cached blocks are still pinned", include_host ? "" : "L1 ");
        return false;
    }
    if (include_host && storage_transfers_) storage_transfers_->ClearMetadata();
    spdlog::info("[Scheduler] flush {}cache completed", include_host ? "" : "L1 ");
    return true;
}

void Scheduler::registerKvEventPrefixPages(Request& request, std::span<const std::string> prefix_hashes,
                                           std::int32_t first_page) {
    if (!config_.enable_kv_cache_events) {
        return;
    }
    _assert(first_page >= 0 && static_cast<std::size_t>(first_page) <= prefix_hashes.size(),
            "KV event page range is invalid");
    const std::vector<std::span<const std::int32_t>> token_pages =
        request.PrefixPages(first_page, static_cast<std::int32_t>(prefix_hashes.size()));
    const std::span<const std::uint64_t> block_hashes = request.KvEventBlockHashes(prefix_hashes.size());

    for (std::size_t i = static_cast<std::size_t>(first_page); i < prefix_hashes.size(); ++i) {
        const std::span<const std::int32_t> tokens = token_pages[i - static_cast<std::size_t>(first_page)];
        CacheKey key{.content_hash = prefix_hashes[i]};
        const std::optional<std::uint64_t> parent_hash =
            i == 0 ? std::nullopt : std::optional<std::uint64_t>{block_hashes[i - 1]};
        KvBlockStoredEvent event{
            .block_hashes = {block_hashes[i]},
            .parent_block_hash = parent_hash,
            .token_ids = std::vector<std::int32_t>(tokens.begin(), tokens.end()),
            .block_size = config_.prefix_granularity,
        };
        const auto [it, inserted] = kv_event_boundaries_.try_emplace(
            key, KvEventBoundary{.stored = std::move(event), .depth = static_cast<std::int32_t>(i)});
        FatalCheck(inserted || (it->second.stored.block_hashes.front() == block_hashes[i] &&
                                it->second.depth == static_cast<std::int32_t>(i)),
                   "one cache content hash mapped to different KV event blocks");
        // The next drain drops the descriptor if the publication never lands.
        markKvEventBoundaryForReconcile(key);
    }
}

void Scheduler::registerKvEventPublication(Request& request, std::span<const std::string> prefix_hashes,
                                           const CompletedPages& completed) {
    if (!config_.enable_kv_cache_events) {
        return;
    }
    // Pages below the new ones keep their descriptors while the request's own
    // published blocks hold them on the Device -- not after a restore brought
    // the request back on unpublished copies -- so the range starts at the
    // first page the publication registers an entry under.
    registerKvEventPrefixPages(request, prefix_hashes, coordinator_.FirstPublishedPage(completed));
    if (completed.tail) {
        registerKvEventTail(request, prefix_hashes, *completed.tail);
    }
}

std::vector<std::int32_t> Scheduler::kvEventTailsOf(const RetractionImage& image) const {
    std::vector<std::int32_t> tails;
    if (!config_.enable_kv_cache_events) {
        return tails;
    }
    for (const ImageTable& table : image.tables) {
        for (const ImageSlot& slot : table.slots) {
            if (!slot.InHostCache()) {
                continue;
            }
            // A published slot's boundary is on the Device, so it has its
            // descriptor; one with fewer tokens than a page is a tail's.
            const auto it = kv_event_boundaries_.find(eventKey(slot.key));
            FatalCheck(it != kv_event_boundaries_.end(), "an imaged published slot has no KV event descriptor");
            const auto tail_tokens = static_cast<std::int32_t>(it->second.stored.token_ids.size());
            const std::int32_t boundary = it->second.depth * config_.prefix_granularity + tail_tokens;
            if (tail_tokens < config_.prefix_granularity && std::ranges::find(tails, boundary) == tails.end()) {
                tails.push_back(boundary);
            }
        }
    }
    return tails;
}

void Scheduler::registerKvEventRestore(Request& request, std::span<const std::string> prefix_hashes,
                                       std::span<const std::int32_t> tails) {
    registerKvEventPrefixPages(request, prefix_hashes, 0);
    for (const std::int32_t boundary : tails) {
        const auto full_pages = static_cast<std::size_t>(boundary / config_.prefix_granularity);
        _assert(full_pages <= prefix_hashes.size(), "an imaged tail lies inside the request's computed pages");
        const std::span<const std::string> pages = prefix_hashes.first(full_pages);
        const std::optional<TailPublication> tail =
            request.TailPublicationAt(boundary, pages.empty() ? std::string{} : pages.back());
        _assert(tail.has_value(), "an imaged tail lies strictly inside a page");
        registerKvEventTail(request, pages, *tail);
    }
}

void Scheduler::registerKvEventTail(Request& request, std::span<const std::string> prefix_hashes,
                                    const TailPublication& tail) {
    if (!config_.enable_kv_cache_events) {
        return;
    }
    const auto full_pages = static_cast<std::int32_t>(prefix_hashes.size());
    const std::span<const std::uint64_t> block_hashes = request.KvEventBlockHashes(prefix_hashes.size());
    const std::optional<std::uint64_t> parent_hash =
        full_pages == 0 ? std::nullopt : std::optional<std::uint64_t>{block_hashes[full_pages - 1]};
    const std::span<const std::int32_t> tail_tokens =
        request.TokenSlice(full_pages * config_.prefix_granularity, tail.tail_tokens);
    const CacheKey key{.content_hash = tail.hash};
    KvBlockStoredEvent event{
        .block_hashes = {HashKvBlock(tail_tokens, parent_hash)},
        .parent_block_hash = parent_hash,
        .token_ids = std::vector<std::int32_t>(tail_tokens.begin(), tail_tokens.end()),
        .block_size = config_.prefix_granularity,
    };
    const auto [it, inserted] =
        kv_event_boundaries_.try_emplace(key, KvEventBoundary{.stored = std::move(event), .depth = full_pages});
    FatalCheck(inserted || (it->second.stored.block_hashes.front() == HashKvBlock(tail_tokens, parent_hash) &&
                            it->second.depth == full_pages),
               "one cache content hash mapped to different KV event blocks");
    markKvEventBoundaryForReconcile(key);
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
        if (config_.role != Role::kD && config_.max_kv_page_num > 0 &&
            (static_cast<std::int64_t>(spec.tokens.size()) + config_.prefix_granularity - 1) /
                    config_.prefix_granularity >
                config_.max_kv_page_num) {
            throw std::invalid_argument("Scheduler: prompt exceeds max_kv_page_num");
        }
        if (requests_by_id_.contains(spec.request_id) || !request_ids.insert(spec.request_id).second) {
            throw std::invalid_argument("Scheduler: duplicate request id '" + spec.request_id + "'");
        }
        if (spec.max_new_tokens < 0) {
            throw std::invalid_argument("Scheduler: max_new_tokens must be non-negative");
        }
        if (spec.shared_prefix_tokens.size() > 64 || std::ranges::any_of(spec.shared_prefix_tokens, [&](auto boundary) {
                return boundary <= 0 || static_cast<std::size_t>(boundary) >= spec.tokens.size();
            }))
            throw std::invalid_argument("Scheduler: invalid shared prefix boundaries");
        if (spec.max_cached_prefix_tokens < 0) {
            throw std::invalid_argument("Scheduler: max_cached_prefix_tokens must be non-negative");
        }
        const std::int64_t generation_reserve =
            config_.role == Role::kP ? 0 : std::max<std::int64_t>(spec.max_new_tokens, config_.decode_input_tokens);
        const std::int64_t token_limit = static_cast<std::int64_t>(spec.tokens.size()) + generation_reserve;
        if (token_limit > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Scheduler: request token limit exceeds int32 range");
        }
        if (token_limit > max_single_request_tokens_) {
            throw std::invalid_argument("Scheduler: request token limit " + std::to_string(token_limit) +
                                        " exceeds cache capacity " + std::to_string(max_single_request_tokens_));
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
        requests_, [](const std::unique_ptr<Request>& request) { return request->AwaitsBootstrap(); }));
}

std::size_t Scheduler::WaitingSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const std::unique_ptr<Request>& request) {
        return request->IsAnyOf<fsm::Submitted, fsm::Prefetching, fsm::Retracted, fsm::Restoring>() &&
               !request->AwaitsBootstrap();
    }));
}

std::size_t Scheduler::RetractedSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const std::unique_ptr<Request>& request) {
        return request->IsAnyOf<fsm::Retracted, fsm::Restoring>();
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

std::size_t Scheduler::RunningSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const std::unique_ptr<Request>& request) {
        return request->IsAnyOf<fsm::Prefilling, fsm::PrefillAwaitingResult, fsm::PrefillDone, fsm::Decoding>();
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

HostCacheStats Scheduler::GetHostCacheStats() const {
    return HostCacheStats{
        .prefix_hits = prefix_hit_stats_,
        .storage_query_hit_tokens = storage_transfers_ ? storage_transfers_->QueryHitTokens() : std::uint64_t{0},
        .publication_stores = tier_transfers_.WriteBackStatsFor(HostStoreCause::kPublication),
        .finish_stores = tier_transfers_.WriteBackStatsFor(HostStoreCause::kFinish),
        .retraction_stores = tier_transfers_.WriteBackStatsFor(HostStoreCause::kRetraction),
        .eviction_stores = tier_transfers_.WriteBackStatsFor(HostStoreCause::kEviction),
        .prefix_loads = tier_transfers_.PrefixLoadStats(),
        .restore_loads = tier_transfers_.RestoreLoadStats(),
        .host_evictions = coordinator_.HostEvictions(),
        .store_candidates_dropped_by_flush = coordinator_.StoreCandidatesDroppedByFlush(),
        .dead_entries_not_written_back = coordinator_.DeadEntriesNotWrittenBack(),
        .demoted_entries_not_written_back = coordinator_.DemotedEntriesNotWrittenBack(),
        .device_evictions = coordinator_.DeviceEvictions(),
        .checkpoint_resume_gaps = coordinator_.CheckpointResumeGaps(),
    };
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

std::int32_t Scheduler::RequestTokenSize(const std::string& id) const {
    const auto it = requests_by_id_.find(id);
    return it == requests_by_id_.end() ? -1 : it->second->TokenSize();
}

bool Scheduler::CanAbortBeforeAdmission(const std::string& id) const {
    const auto it = requests_by_id_.find(id);
    return config_.role == Role::kFused && it != requests_by_id_.end() &&
           it->second->IsAnyOf<fsm::Submitted, fsm::Prefetching>();
}

std::vector<std::int64_t> Scheduler::AdmissionCapacityCurve(std::int32_t token_step) const {
    _assert(token_step > 0, "capacity curve step must be positive");
    CapacityModel model(config_);
    std::vector<std::int64_t> curve;
    for (std::int64_t tokens = 0; tokens < MaxSingleRequestTokens(); tokens += token_step) {
        const auto upper =
            static_cast<std::int32_t>(std::min<std::int64_t>(tokens + token_step, MaxSingleRequestTokens()));
        curve.push_back(model.LcmBlocksNeededFor(model.SingleRequestGroupPages(upper)));
    }
    return curve;
}

Scheduler::PrefillGateProbe Scheduler::ProbePrefillGate() const {
    PrefillGateProbe probe{.wants_prefill = false, .has_decode_work = false};
    if (config_.role != Role::kFused) {
        // Only the fused role has both kinds of work to choose between: a P
        // node is all prefill and a D node's prompts are its peer's.
        return probe;
    }
    bool pending_prefill = false;
    bool resident_prefill = false;
    for (const auto& request : requests_) {
        // Storage preparation has not admitted Device pages yet. Resident
        // chunks, however, must be able to consume their own reservations.
        if (request->Is<fsm::Prefilling>()) {
            resident_prefill = true;
        } else if (request->IsAnyOf<fsm::Submitted, fsm::Prefetching>()) {
            pending_prefill = true;
        } else if (request->IsAnyOf<fsm::PrefillDone, fsm::Decoding>()) {
            probe.has_decode_work = true;
        }
    }
    probe.wants_prefill = resident_prefill || (pending_prefill && belowPrefillWatermark());
    return probe;
}

bool Scheduler::belowPrefillWatermark() const {
    const std::int32_t total_pages = coordinator_.TotalLcmBlocks();
    const float used_ratio =
        total_pages > 0 ? static_cast<float>(ActiveLcmBlocks()) / static_cast<float>(total_pages) : 0.0F;
    return used_ratio <= kPrefillWatermark;
}

ExecutionPlan Scheduler::NextExecutionPlan(PlanOptions options) {
    std::vector<WriteBackOperation> write_back_operations = std::exchange(pending_write_back_operations_, {});
    // Dropping a finished request drops what it still holds elsewhere -- its
    // session's registration under its access epoch -- with it.
    std::erase_if(requests_, [this](const auto& request) {
        if (!request->template Is<fsm::Finished>()) {
            return false;
        }
        requests_by_id_.erase(request->Id());
        return true;
    });

    if (storage_transfers_) {
        for (const auto& request : requests_) prepareStorage(*request);
    }
    std::vector<Request*> candidates;
    candidates.reserve(requests_.size());
    for (const auto& request : requests_) {
        if (request->IsAnyOf<fsm::Submitted, fsm::Prefilling, fsm::RemotePrefilling, fsm::PrefillDone, fsm::Decoding,
                             fsm::Retracted>()) {
            candidates.push_back(request.get());
        }
    }
    ++plan_calls_;
    const std::size_t num_live_requests = candidates.size();
    ExecutionPlan plan;
    for (const std::string& group_id : cache_group_ids_) {
        plan.pages_to_zero[group_id];
        plan.pages_to_copy[group_id];
    }
    BuiltOperations built;
    bool builds_work = false;
    if (forward_allowed_) {
        built = buildForwardOperations(plan, std::move(candidates), write_back_operations, options);
        builds_work = !built.forward.empty() || plan.remote_decode.has_value() || plan.remote_prefill.has_value();
        plan.With(ForwardBatch{std::move(built.forward)});
    }
    // What the build's admissions and restores evicted under the write-back
    // policy goes to the Host in this plan: their pages may already be granted
    // again, so the copy is stream-ordered and the runtime runs it ahead of
    // the plan's page reuse. Every eviction record is taken here, never in a
    // later plan. A no-op with nothing recorded.
    if (auto store = tier_transfers_.StartEvictionStores()) {
        write_back_operations.push_back(std::move(*store));
    }
    // The prefix entries the build's evictions (and everything since the last
    // plan) left dead on both tiers. Nothing is queued unless the Host
    // write-back policy evicts by segments.
    coordinator_.CollectDeadEntries();
    std::vector<StorageOperation> storage_operations;
    if (storage_transfers_) {
        if (storage_allowed_)
            storage_transfers_->StartBackups();
        else
            storage_transfers_->DropBackups();
        storage_operations = storage_transfers_->TakeCommands();
    }

    if (forward_allowed_ && config_.StreamsDeviceCacheToHost()) {
        // Boundary publications of live requests: their owners hold the pages,
        // and the ticket pins them until the ACK, so the copy stays off the
        // forward's critical path.
        if (auto store =
                tier_transfers_.StartPendingStores(StoreSourceGuard::kPinnedUntilAck, HostStoreCause::kPublication)) {
            write_back_operations.push_back(std::move(*store));
        }
    }

    // Store legs precede loads: the runtime fences page reuse behind the
    // stream-ordered stores. Host-only storage ops keep their own completion
    // protocol; their position does not add ordering against Device copies.
    if (!write_back_operations.empty()) {
        plan.With(CacheOperation{WriteBackBatch{write_back_operations}});
    }
    if (!built.snapshot_stores.empty()) {
        plan.With(CacheOperation{SnapshotStoreBatch{built.snapshot_stores}});
    }
    if (!built.load_backs.empty()) {
        plan.With(CacheOperation{LoadBackBatch{built.load_backs}});
    }
    if (!built.snapshot_restores.empty()) {
        plan.With(CacheOperation{SnapshotRestoreBatch{built.snapshot_restores}});
    }
    const bool has_storage_work = !storage_operations.empty();
    for (auto& operation : storage_operations) {
        plan.With(CacheOperation{std::move(operation)});
    }
    trackStalledRounds(num_live_requests,
                       /*plan_is_empty=*/!builds_work && write_back_operations.empty() &&
                           built.snapshot_stores.empty() && !has_storage_work && built.load_backs.empty() &&
                           built.snapshot_restores.empty() && plan.aborts.empty());
    return plan;
}

void Scheduler::trackStalledRounds(std::size_t num_live_requests, bool plan_is_empty) {
    // With nothing out, only a new submission or an abort can change the next
    // plan.
    const bool stalled = forward_allowed_ && (!storage_transfers_ || !storage_transfers_->HasPending()) &&
                         num_live_requests > 0 && plan_is_empty && !tier_transfers_.HasAnyInFlight() &&
                         std::ranges::none_of(requests_, [this](const std::unique_ptr<Request>& request) {
                             return request->ResultsInFlight() > 0 || pdTransferInFlight(*request);
                         });
    if (!stalled) {
        if (stalled_rounds_ >= kStalledRoundsBeforeWarning) {
            spdlog::info("[Scheduler] scheduling resumed after {} stalled rounds", stalled_rounds_);
        }
        stalled_rounds_ = 0;
        return;
    }
    ++stalled_rounds_;
    if (stalled_rounds_ >= kStalledRoundsBeforeWarning &&
        std::has_single_bit(static_cast<std::uint64_t>(stalled_rounds_))) {
        spdlog::warn(
            "[Scheduler] no work scheduled for {} consecutive rounds with {} live requests and nothing in flight "
            "(waiting={}, prefilling={}, decoding={}; device LCM blocks active={}, empty={}, total={}); "
            "they cannot progress until a request is aborted",
            stalled_rounds_, num_live_requests, WaitingSize(), PrefillSize(), DecodingSize(), ActiveLcmBlocks(),
            EmptyLcmBlocks(), coordinator_.TotalLcmBlocks());
    }
}

void Scheduler::Advance(const ExecutionEvent& event) {
    for (const auto& item : event.Events()) {
        std::visit([this](const auto& inner) { handleEvent(inner); }, item);
    }
    // Finishing is entered and forwards land only here, so this is the one
    // place a finished request can become settleable.
    settleFinishedRequests();
}

}  // namespace tokenspeed
