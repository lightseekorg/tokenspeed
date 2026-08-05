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

#include "cache/tier/transfer.h"
#include "fsm/forward_states.h"
#include "scheduler/operations/forward.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) {
    _assert(value >= 0 && divisor > 0, "ceilDiv requires non-negative value and positive divisor");
    return (value + divisor - 1) / divisor;
}

CacheKey eventKey(const CacheKey& key) {
    return CacheKey{
        .namespace_id = key.namespace_id,
        .group_id = 0,
        .content_hash = key.content_hash,
    };
}

std::vector<CacheBlockLocation> snapshotLocations(const std::vector<BlockTable>& tables) {
    std::vector<CacheBlockLocation> locations;
    for (const BlockTable& table : tables) {
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            if (block_ref) {
                _assert(block_ref.unique(), "snapshot Host block has an owner outside its request table");
                locations.push_back(block_ref->Location());
            }
        }
    }
    return locations;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_{std::move(config)},
      req_pool_allocator_{config_.max_batch_size},
      block_pool_{config_.device_allocator.total_pages - 1},
      host_pool_{(config_.OrdinaryL2Enabled() || config_.DecodeSnapshotEnabled())
                     ? config_.host_allocator.total_pages - 1
                     : 0},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_), config_.block_size, block_pool_,
                                   config_.OrdinaryL2Enabled() ? &host_pool_ : nullptr)},
      tier_transfers_{coordinator_, host_pool_} {
    if (config_.block_size <= 0) {
        throw std::invalid_argument("Scheduler: block_size must be > 0");
    }
    if (config_.device_allocator.total_pages <= 1) {
        throw std::invalid_argument("Scheduler: device cache must contain a null page and usable capacity");
    }
    if (config_.paged_cache_groups.empty()) {
        throw std::invalid_argument("Scheduler: at least one cache group is required");
    }
    if (config_.decode_input_tokens < 0) {
        throw std::invalid_argument("Scheduler: decode_input_tokens must be >= 0");
    }
    if (config_.max_scheduled_tokens <= 0) {
        throw std::invalid_argument("Scheduler: max_scheduled_tokens must be > 0");
    }
    if (config_.overlap_schedule_depth < 0 || config_.overlap_schedule_depth > 1) {
        throw std::invalid_argument("Scheduler: overlap_schedule_depth must be 0 or 1");
    }
    if (config_.overlap_schedule_depth > 0 && config_.decode_input_tokens == 0) {
        throw std::invalid_argument("Scheduler: overlapped decode requires decode_input_tokens > 0");
    }
    if (config_.enable_l3_storage) {
        throw std::invalid_argument("Scheduler: L3 storage is not supported by the cache coordinator");
    }
    if (coordinator_.HasMambaStateGroup() && config_.max_scheduled_tokens < coordinator_.CacheBlockTokens()) {
        throw std::invalid_argument("Scheduler: Mamba max_scheduled_tokens must cover one cache block");
    }

    cache_group_ids_.reserve(config_.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config_.paged_cache_groups) {
        group.Validate();
        cache_group_ids_.push_back(group.group_id);
    }
    max_single_request_tokens_ = calculateMaxSingleRequestTokens(block_pool_.NumLcmBlocks());
    if (config_.DecodeSnapshotEnabled()) {
        max_host_snapshot_tokens_ = calculateMaxSingleRequestTokens(host_pool_.NumLcmBlocks());
        max_single_request_tokens_ = std::min(max_single_request_tokens_, max_host_snapshot_tokens_);
    }

    if (config_.enable_kv_cache_events) {
        coordinator_.SetCacheMutationSink([this](const CacheKey& key, KvCacheCoordinator::CacheMutation mutation) {
            handleCacheMutation(key, mutation);
        });
    }

    if (const char* level = std::getenv("SPDLOG_LEVEL")) {
        spdlog::set_level(spdlog::level::from_str(level));
    }
}

std::int64_t Scheduler::singleRequestParentsRequired(std::int32_t token_limit) const {
    _assert(token_limit >= 0, "single-request token limit must be non-negative");
    const std::int64_t decode_width = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
    // An overlapped forward protects one additional decode reservation that
    // cannot yet be reclaimed from the request table.
    const std::int64_t protected_tokens = static_cast<std::int64_t>(config_.overlap_schedule_depth) * decode_width;
    // The largest accepted prompt must still leave the first decode/MTP
    // reservation inside token_limit.
    const std::int64_t max_prompt_tokens =
        std::max<std::int64_t>(static_cast<std::int64_t>(token_limit) - decode_width, 0);
    const std::int64_t chunk_tokens = config_.max_scheduled_tokens;

    std::int64_t required_parents = 0;
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        const KvCacheManager& manager = coordinator_.GroupManager(i);
        const std::int64_t page_tokens = manager.CacheBlockTokens();
        std::int64_t child_pages = 0;
        if (manager.MatchIsPrefixClosed()) {
            child_pages = ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, page_tokens);
        } else if (config_.role == Role::kD) {
            const bool latest_snapshot =
                config_.enable_pd_cache && config_.paged_cache_groups[static_cast<std::size_t>(i)].transfer_policy ==
                                               PagedCacheTransferPolicy::LatestSnapshot;
            if (latest_snapshot) {
                // The final prompt page may be full, so a non-zero decode
                // reservation can span one more page than its own page count.
                const std::int64_t reserved_tokens = decode_width + protected_tokens;
                child_pages = reserved_tokens == 0 ? 1 : 1 + ceilDiv(reserved_tokens, page_tokens);
            } else {
                // Decode-only restores its destination in one admission, so a
                // non-sparse group cannot slide old prompt pages first.
                child_pages = ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, page_tokens);
            }
        } else {
            // Across every prompt up to max_prompt_tokens, retain the largest
            // resident window seen by either the first chunk or a later chunk.
            const std::int64_t first_prompt = std::min(max_prompt_tokens, chunk_tokens);
            child_pages = ceilDiv(first_prompt + decode_width + protected_tokens, page_tokens);
            if (max_prompt_tokens > chunk_tokens) {
                const std::int64_t later_prompt = std::min(max_prompt_tokens - chunk_tokens, chunk_tokens);
                const std::int64_t lookback_pages = manager.BoundaryLookbackBlocks();
                child_pages = std::max(child_pages, lookback_pages + ceilDiv(chunk_tokens, page_tokens));
                child_pages = std::max(
                    child_pages, lookback_pages + ceilDiv(later_prompt + decode_width + protected_tokens, page_tokens));
            }
        }
        required_parents += ceilDiv(child_pages, manager.CacheBlocksPerLcmBlock());
    }
    return required_parents;
}

std::int32_t Scheduler::calculateMaxSingleRequestTokens(std::int64_t usable_parents) const {
    std::int64_t low = 0;
    std::int64_t high = std::numeric_limits<std::int32_t>::max();
    while (low < high) {
        const std::int64_t candidate = low + (high - low + 1) / 2;
        if (singleRequestParentsRequired(static_cast<std::int32_t>(candidate)) <= usable_parents) {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    return static_cast<std::int32_t>(low);
}

Request* Scheduler::findRequest(const std::string& request_id) {
    const auto it = requests_.find(request_id);
    return it == requests_.end() ? nullptr : it->second.get();
}

std::size_t Scheduler::groupIndex(const std::string& group_id) const {
    const auto it = std::ranges::find(cache_group_ids_, group_id);
    if (it == cache_group_ids_.end()) {
        throw std::out_of_range("Scheduler: unknown cache group '" + group_id + "'");
    }
    return static_cast<std::size_t>(std::distance(cache_group_ids_.begin(), it));
}

std::vector<KvCacheEvent> Scheduler::DrainKvEvents() {
    return std::exchange(kv_events_, {});
}

bool Scheduler::ClearL1Cache() {
    const bool has_live_request = std::ranges::any_of(
        requests_, [](const auto& item) { return !item.second->template Is<fsm::Finished>(); });
    if (has_live_request || !pending_forward_results_.empty() || !pd_transfer_pins_.empty() ||
        tier_transfers_.HasAnyInFlight()) {
        return false;
    }
    return coordinator_.ClearDeviceCache();
}

std::vector<CacheKey> Scheduler::registerKvEventPages(const Request& request, std::span<const std::string> page_hashes,
                                                      std::int32_t first_page) {
    if (!config_.enable_kv_cache_events) {
        return {};
    }
    _assert(first_page >= 0 && static_cast<std::size_t>(first_page) <= page_hashes.size(),
            "KV event page range is invalid");
    const std::vector<std::span<const std::int32_t>> token_pages = request.FullPagedTokens(false);
    _assert(page_hashes.size() <= token_pages.size(), "KV event hashes exceed the request's complete pages");

    KvEventHashProgress& progress = kv_event_hash_progress_[request.Id()];
    for (std::size_t i = progress.block_hashes.size(); i < page_hashes.size(); ++i) {
        const std::optional<std::uint64_t> parent_hash =
            i == 0 ? std::nullopt : std::optional<std::uint64_t>{progress.block_hashes[i - 1]};
        progress.block_hashes.push_back(HashKvBlock(token_pages[i], parent_hash));
    }

    std::vector<CacheKey> registered_keys;
    registered_keys.reserve(page_hashes.size() - static_cast<std::size_t>(first_page));
    for (std::size_t i = static_cast<std::size_t>(first_page); i < page_hashes.size(); ++i) {
        CacheKey key{.content_hash = page_hashes[i]};
        const std::optional<std::uint64_t> parent_hash =
            i == 0 ? std::nullopt : std::optional<std::uint64_t>{progress.block_hashes[i - 1]};
        KvBlockStoredEvent event{
            .block_hashes = {progress.block_hashes[i]},
            .parent_block_hash = parent_hash,
            .token_ids = std::vector<std::int32_t>(token_pages[i].begin(), token_pages[i].end()),
            .block_size = config_.block_size,
        };
        const auto [it, inserted] = kv_event_pages_.try_emplace(key, std::move(event));
        FatalCheck(inserted || it->second.block_hashes.front() == progress.block_hashes[i],
                   "one cache content hash mapped to different KV event blocks");
        registered_keys.push_back(std::move(key));
    }
    return registered_keys;
}

void Scheduler::discardUncachedKvEventPages(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) {
        if (!cached_event_group_counts_.contains(key)) {
            kv_event_pages_.erase(key);
        }
    }
}

void Scheduler::handleCacheMutation(const CacheKey& key, KvCacheCoordinator::CacheMutation mutation) {
    const CacheKey prefix_key = eventKey(key);
    if (mutation == KvCacheCoordinator::CacheMutation::kStored) {
        std::int32_t& group_count = cached_event_group_counts_[prefix_key];
        FatalCheck(group_count < coordinator_.NumGroups(), "duplicate group entry for one KV event boundary");
        ++group_count;
        if (group_count == coordinator_.NumGroups()) {
            const auto page_it = kv_event_pages_.find(prefix_key);
            FatalCheck(page_it != kv_event_pages_.end(), "cached KV event boundary has no token descriptor");
            kv_events_.emplace_back(page_it->second);
        }
        return;
    }

    auto count_it = cached_event_group_counts_.find(prefix_key);
    FatalCheck(count_it != cached_event_group_counts_.end() && count_it->second > 0,
               "removed KV event boundary was not registered");
    if (count_it->second == coordinator_.NumGroups()) {
        const auto page_it = kv_event_pages_.find(prefix_key);
        FatalCheck(page_it != kv_event_pages_.end(), "removed KV event boundary has no token descriptor");
        kv_events_.emplace_back(KvBlockRemovedEvent{.block_hashes = page_it->second.block_hashes});
    }
    if (--count_it->second == 0) {
        cached_event_group_counts_.erase(count_it);
        kv_event_pages_.erase(prefix_key);
    }
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
    for (const RequestSpec& spec : request_specs) {
        if (spec.max_new_tokens < 0) {
            throw std::invalid_argument("Scheduler: max_new_tokens must be non-negative");
        }
        const std::int64_t minimum_generation_reserve = config_.role == Role::kP ? 0 : config_.decode_input_tokens;
        const std::int64_t token_limit =
            static_cast<std::int64_t>(spec.tokens.size()) + std::max<std::int64_t>(spec.max_new_tokens,
                                                                                  minimum_generation_reserve);
        if (token_limit > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Scheduler: request token limit exceeds int32 range");
        }
        if (token_limit > max_single_request_tokens_) {
            throw std::invalid_argument("Scheduler: request token limit exceeds paged-cache capacity");
        }
        std::int32_t max_snapshot_parents = 0;
        if (config_.DecodeSnapshotEnabled()) {
            const std::int64_t required = singleRequestParentsRequired(static_cast<std::int32_t>(token_limit));
            max_snapshot_parents = static_cast<std::int32_t>(required);
        }
        auto request = std::make_unique<Request>(spec, config_.block_size, config_.role, max_snapshot_parents);
        const bool inserted = requests_.emplace(spec.request_id, std::move(request)).second;
        if (!inserted) {
            throw std::invalid_argument("Scheduler: duplicate request id '" + spec.request_id + "'");
        }
    }
}

bool Scheduler::preservesRetractionCapacityAfterAdmission(const Request& candidate) const {
    if (!config_.DecodeSnapshotEnabled()) {
        return true;
    }
    // A request in snapshot transfer is absent from the active-state sum
    // below. Its post-ACK state depends on transfer success, so do not admit
    // against that temporary undercount.
    if (tier_transfers_.HasSnapshotsInFlight()) {
        return false;
    }
    const std::int64_t occupied_parents = host_pool_.NumLcmBlocks() - host_pool_.NumEmptyLcmBlocks();
    // Keep enough Host parents to retract every active request except the
    // largest one, which may remain on Device: P + sum(S) - max(S) <= H.
    std::int64_t sum = candidate.MaxSnapshotParents();
    std::int64_t largest = candidate.MaxSnapshotParents();
    for (const auto& [_, request] : requests_) {
        if (request.get() == &candidate ||
            (!request->Is<fsm::Prefilling>() && !request->Is<fsm::PrefillDone>() &&
             !request->Is<fsm::Decoding>())) {
            continue;
        }
        const std::int64_t bound = request->MaxSnapshotParents();
        sum += bound;
        largest = std::max(largest, bound);
    }
    return occupied_parents + sum - largest <= host_pool_.NumLcmBlocks();
}

bool Scheduler::preservesRetractionCapacityAfterRetraction(const Request& candidate) const {
    if (!config_.DecodeSnapshotEnabled()) {
        return true;
    }
    // PrepareOffload already holds the candidate's Host destinations, so
    // current occupancy is exactly P after the proposed migration.
    const std::int64_t occupied_parents = host_pool_.NumLcmBlocks() - host_pool_.NumEmptyLcmBlocks();
    std::int64_t sum = 0;
    std::int64_t largest = 0;
    for (const auto& [_, request] : requests_) {
        if (request.get() == &candidate ||
            (!request->Is<fsm::Prefilling>() && !request->Is<fsm::PrefillDone>() &&
             !request->Is<fsm::Decoding>())) {
            continue;
        }
        const std::int64_t bound = request->MaxSnapshotParents();
        sum += bound;
        largest = std::max(largest, bound);
    }
    return occupied_parents + sum - largest <= host_pool_.NumLcmBlocks();
}

bool Scheduler::preservesRetractionCapacityAfterRecovery(const Request& request) const {
    if (!config_.DecodeSnapshotEnabled()) {
        return true;
    }
    const fsm::RetractionSnapshot& snapshot = request.RetractionSnapshotRef();
    const std::vector<CacheBlockLocation> snapshot_locations = snapshotLocations(snapshot.host_tables);
    const std::int64_t released_parents = host_pool_.NumParentsFreedBy(snapshot_locations);
    const std::int64_t occupied_parents = host_pool_.NumLcmBlocks() - host_pool_.NumEmptyLcmBlocks();
    _assert(released_parents <= occupied_parents, "snapshot releases more than the occupied Host parents");

    std::int64_t sum = request.MaxSnapshotParents();
    std::int64_t largest = request.MaxSnapshotParents();
    for (const auto& [_, active] : requests_) {
        if (!active->Is<fsm::Prefilling>() && !active->Is<fsm::PrefillDone>() &&
            !active->Is<fsm::Decoding>()) {
            continue;
        }
        const std::int64_t bound = active->MaxSnapshotParents();
        sum += bound;
        largest = std::max(largest, bound);
    }
    return occupied_parents - released_parents + sum - largest <= host_pool_.NumLcmBlocks();
}

std::size_t Scheduler::WaitingSize() const {
    return static_cast<std::size_t>(
        std::ranges::count_if(requests_, [](const auto& item) { return item.second->template Is<fsm::Submitted>(); }));
}

std::size_t Scheduler::DecodingSize() const {
    return static_cast<std::size_t>(
        std::ranges::count_if(requests_, [](const auto& item) { return item.second->template Is<fsm::Decoding>(); }));
}

std::size_t Scheduler::PrefillSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const auto& item) {
        return item.second->template Is<fsm::Prefilling>() || item.second->template Is<fsm::PrefillDone>();
    }));
}

std::size_t Scheduler::AvailableKvPages() const {
    return static_cast<std::size_t>(coordinator_.NumAvailableLcmBlocks());
}

std::size_t Scheduler::ActiveKvPages() const {
    std::unordered_set<std::int32_t> active_lcm_blocks;
    for (const auto& [_, request] : requests_) {
        if (!request->Is<fsm::Prefilling>() && !request->Is<fsm::PrefillDone>() && !request->Is<fsm::Decoding>()) {
            continue;
        }
        for (std::int32_t block_id : request->ActiveLcmBlockIds()) {
            if (block_id > 0) {
                active_lcm_blocks.insert(block_id);
            }
        }
    }
    return active_lcm_blocks.size();
}

std::int32_t Scheduler::PagedCacheGroupTotalPages(const std::string& group_id) const {
    return config_.paged_cache_groups[groupIndex(group_id)].total_pages;
}

std::int32_t Scheduler::PagedCacheGroupAvailablePages(const std::string& group_id) const {
    const std::size_t index = groupIndex(group_id);
    const std::int32_t slots_per_parent = config_.paged_cache_groups[index].cache_blocks_per_lcm_block;
    std::int32_t available = block_pool_.NumEmptyLcmBlocks() * slots_per_parent;
    for (std::int32_t id = 1; id <= block_pool_.NumLcmBlocks(); ++id) {
        if (block_pool_.BoundGroup(id) == static_cast<std::uint32_t>(index)) {
            available += slots_per_parent - block_pool_.OccupiedCount(id);
        }
    }
    return available;
}

std::int32_t Scheduler::RequestTokenSize(const std::string& id) const {
    const auto it = requests_.find(id);
    return it == requests_.end() ? -1 : it->second->TokenSize();
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    std::erase_if(requests_, [this](const auto& item) {
        if (!item.second->template Is<fsm::Finished>()) {
            return false;
        }
        kv_event_hash_progress_.erase(item.first);
        return true;
    });

    std::vector<Request*> candidates;
    candidates.reserve(requests_.size());
    for (auto& [_, request] : requests_) {
        if (request->Is<fsm::Submitted>() || request->Is<fsm::Prefilling>() || request->Is<fsm::PrefillDone>() ||
            request->Is<fsm::Decoding>() || request->Is<fsm::Retracted>()) {
            candidates.push_back(request.get());
        }
    }

    std::vector<WriteBackOperation> write_back_operations;
    auto [forward_operations, load_back_operations] =
        buildForwardOperations(std::move(candidates), write_back_operations);

    ExecutionPlan plan;
    plan.With(ForwardBatch{std::move(forward_operations)});

    if (config_.OrdinaryL2Enabled()) {
        if (auto store = tier_transfers_.StartPendingStores()) {
            write_back_operations.push_back(std::move(*store));
        }
    }
    if (!write_back_operations.empty()) {
        plan.With(CacheOperation{WriteBackBatch{write_back_operations}});
    }
    if (!load_back_operations.empty()) {
        plan.With(CacheOperation{LoadBackBatch{load_back_operations}});
    }
    plan.pages_to_zero = std::exchange(new_page_ids_, {});
    return plan;
}

void Scheduler::Advance(const ExecutionEvent& event) {
    const auto dispatch = [this](const auto& inner) { handleEvent(inner); };
    for (const auto& item : event.Events()) {
        std::visit([&](const auto& outer) { std::visit(dispatch, outer); }, item);
    }
}

}  // namespace tokenspeed
