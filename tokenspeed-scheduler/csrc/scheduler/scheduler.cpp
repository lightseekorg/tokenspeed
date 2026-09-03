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
#include "scheduler/operations/forward.h"
#include "cache/prefix/prefix_hasher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) {
    _assert(value >= 0 && divisor > 0, "ceilDiv requires non-negative value and positive divisor");
    return (value + divisor - 1) / divisor;
}

std::int32_t hostPoolBlocks(const SchedulerConfig& config) {
    return config.HasHostCache() ? config.host_allocator.NumUsableBlocks() : 0;
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
      block_pool_{config_.device_allocator.NumUsableBlocks()},
      host_pool_{hostPoolBlocks(config_)},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_), config_.prefix_granularity, block_pool_,
                                   hostPoolBlocks(config_) > 0 ? &host_pool_ : nullptr,
                                   config_.StreamsDeviceCacheToHost(), config_.enable_l3_storage)},
      tier_transfers_{coordinator_} {
    // config_.Validate() already ran; the body only derives state from it.
    cache_group_ids_.reserve(config_.cache_groups.size());
    for (const CacheGroupConfig& group : config_.cache_groups) {
        cache_group_ids_.push_back(group.group_id);
        const std::int32_t child_entries = config_.prefix_granularity / group.BlockGranularity();
        if (cache_entries_per_event_boundary_ > std::numeric_limits<std::int32_t>::max() - child_entries) {
            throw std::invalid_argument("Scheduler: cache entries per event boundary exceed int32 range");
        }
        cache_entries_per_event_boundary_ += child_entries;
    }
    max_single_request_tokens_ = calculateMaxSingleRequestTokens(coordinator_.TotalLcmBlocks());

    if (config_.enable_kv_cache_events) {
        coordinator_.SetCacheMutationSink([this](const CacheKey& key, CacheCoordinator::CacheMutation mutation) {
            handleCacheMutation(key, mutation);
        });
    }

    if (const char* level = std::getenv("SPDLOG_LEVEL")) {
        spdlog::set_level(spdlog::level::from_str(level));
    }
}

std::int64_t Scheduler::singleRequestLcmBlocksRequired(std::int32_t token_limit) const {
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
    const std::int64_t prefix_granularity = config_.prefix_granularity;
    // A final sub-page tail can follow the first aligned body, or a later body
    // that also retains an input checkpoint. Bound both cases independently.
    const bool splits_final_state_checkpoint = config_.role != Role::kD && !config_.disable_prefix_cache;
    const auto max_split_tail_after = [&](std::int64_t minimum_body_end) {
        return splits_final_state_checkpoint
                   ? std::max<std::int64_t>(0, std::min({prefix_granularity - 1, chunk_tokens - prefix_granularity,
                                                         max_prompt_tokens - minimum_body_end}))
                   : 0;
    };
    const std::int64_t max_first_chunk_tail_tokens = max_split_tail_after(prefix_granularity);
    const std::int64_t max_later_chunk_tail_tokens = max_split_tail_after(2 * prefix_granularity);

    std::vector<std::int64_t> group_pages(static_cast<std::size_t>(coordinator_.NumGroups()));
    for (std::int32_t i = 0; i < coordinator_.NumGroups(); ++i) {
        const std::int64_t block_granularity = coordinator_.GroupBlockGranularity(i);
        const CacheGroupConfig& group = config_.cache_groups[static_cast<std::size_t>(i)];
        const auto local_prefill_peak = [&] {
            if (group.IsSnapshotStateGroup()) {
                if (token_limit == 0) return std::int64_t{0};
                const std::int64_t input_lookback =
                    max_prompt_tokens > chunk_tokens ? coordinator_.GroupBoundaryLookbackPages(i) : 0;
                const std::int64_t first_split_checkpoint_peak =
                    max_first_chunk_tail_tokens == 0 ? 0 : 1 + ceilDiv(max_first_chunk_tail_tokens, block_granularity);
                const std::int64_t later_split_checkpoint_peak =
                    max_later_chunk_tail_tokens == 0 ? 0
                                                     : coordinator_.GroupBoundaryLookbackPages(i) + 1 +
                                                           ceilDiv(max_later_chunk_tail_tokens, block_granularity);
                return std::max(
                    {std::int64_t{2}, input_lookback + 1, first_split_checkpoint_peak, later_split_checkpoint_peak});
            }
            // Across every prompt up to max_prompt_tokens, retain the largest
            // resident window seen by either the first chunk or a later chunk.
            const std::int64_t first_prompt = std::min(max_prompt_tokens, chunk_tokens);
            std::int64_t pages = ceilDiv(first_prompt + decode_width + protected_tokens, block_granularity);
            if (max_prompt_tokens > chunk_tokens) {
                const std::int64_t later_prompt = std::min(max_prompt_tokens - chunk_tokens, chunk_tokens);
                const std::int64_t lookback_pages = coordinator_.GroupBoundaryLookbackPages(i);
                pages = std::max(pages, lookback_pages + ceilDiv(chunk_tokens, block_granularity));
                pages = std::max(
                    pages, lookback_pages + ceilDiv(later_prompt + decode_width + protected_tokens, block_granularity));
            }
            return pages;
        };
        std::int64_t child_pages = 0;
        if (coordinator_.GroupIsPrefixClosed(i)) {
            child_pages = ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, block_granularity);
        } else if (config_.role == Role::kD) {
            if (group.transfer_policy == CacheTransferPolicy::LatestSnapshot) {
                const std::int64_t snapshot_pages = token_limit == 0 ? 0 : 1;
                // A retracted Decode request may recover by locally
                // recomputing its suffix. Old State checkpoints are
                // evictable, but one recovery chunk and its lookback must fit.
                child_pages = std::max(snapshot_pages, local_prefill_peak());
            } else if (group.retention == CacheGroupConfig::Retention::SlidingWindow) {
                const std::int64_t dense_pages =
                    ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, block_granularity);
                const std::int64_t window_pages = ceilDiv(static_cast<std::int64_t>(*group.sliding_window_tokens - 1) +
                                                              decode_width + protected_tokens + block_granularity - 1,
                                                          block_granularity);
                // A sliding prefix probe can retain one older lookback island
                // across null holes while the remote prompt tail is restored at
                // absolute slots. Bound both intervals, capped by a dense table.
                child_pages =
                    std::min<std::int64_t>(dense_pages, coordinator_.GroupBoundaryLookbackPages(i) + window_pages);
            } else {
                // Decode-only restores its destination in one admission, so a
                // non-sparse group cannot slide old prompt pages first.
                child_pages = ceilDiv(static_cast<std::int64_t>(token_limit) + protected_tokens, block_granularity);
            }
        } else {
            child_pages = local_prefill_peak();
        }
        group_pages[static_cast<std::size_t>(i)] = child_pages;
    }
    return coordinator_.LcmBlocksNeededFor(group_pages);
}

std::int32_t Scheduler::calculateMaxSingleRequestTokens(std::int64_t usable_lcm_blocks) const {
    std::int64_t low = 0;
    std::int64_t high = std::numeric_limits<std::int32_t>::max();
    while (low < high) {
        const std::int64_t candidate = low + (high - low + 1) / 2;
        if (singleRequestLcmBlocksRequired(static_cast<std::int32_t>(candidate)) <= usable_lcm_blocks) {
            low = candidate;
        } else {
            high = candidate - 1;
        }
    }
    return static_cast<std::int32_t>(low);
}

Request* Scheduler::findRequest(const std::string& request_id) {
    const auto it = requests_by_id_.find(request_id);
    return it == requests_by_id_.end() ? nullptr : it->second;
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
    return clearCache(false);
}

bool Scheduler::ClearCache() {
    return clearCache(true);
}

bool Scheduler::clearCache(bool include_host) {
    // A live request's pages are protected by their pins, and the coordinator
    // completes its pin check before mutating anything -- so residency is not
    // this function's business. What IS its business are the writers the pins
    // do not cover: an asynchronous transfer still landing into a cached
    // block would race a clear that succeeded on the pin check alone.
    const bool has_pd_transfers = !pd_transfer_pins_.empty();
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

std::vector<CacheKey> Scheduler::registerKvEventPrefixPages(const Request& request,
                                                            std::span<const std::string> prefix_hashes,
                                                            std::int32_t first_page) {
    if (!config_.enable_kv_cache_events) {
        return {};
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

    std::vector<CacheKey> registered_keys;
    registered_keys.reserve(prefix_hashes.size() - static_cast<std::size_t>(first_page));
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
        const auto [it, inserted] = kv_event_pages_.try_emplace(key, std::move(event));
        FatalCheck(inserted || it->second.block_hashes.front() == progress.block_hashes[i],
                   "one cache content hash mapped to different KV event blocks");
        registered_keys.push_back(std::move(key));
    }
    return registered_keys;
}

void Scheduler::discardUncachedKvEventPages(std::span<const CacheKey> keys) {
    for (const CacheKey& key : keys) {
        if (!cached_event_child_counts_.contains(key)) {
            kv_event_pages_.erase(key);
        }
    }
}

void Scheduler::handleCacheMutation(const CacheKey& key, CacheCoordinator::CacheMutation mutation) {
    const CacheKey prefix_key = eventKey(key);
    if (mutation == CacheCoordinator::CacheMutation::kStored) {
        std::int32_t& child_count = cached_event_child_counts_[prefix_key];
        FatalCheck(child_count < cache_entries_per_event_boundary_, "duplicate child entry for one KV event boundary");
        ++child_count;
        if (child_count == cache_entries_per_event_boundary_) {
            const auto page_it = kv_event_pages_.find(prefix_key);
            FatalCheck(page_it != kv_event_pages_.end(), "cached KV event boundary has no token descriptor");
            kv_events_.emplace_back(page_it->second);
        }
        return;
    }

    auto count_it = cached_event_child_counts_.find(prefix_key);
    FatalCheck(count_it != cached_event_child_counts_.end() && count_it->second > 0,
               "removed KV event boundary was not registered");
    if (count_it->second == cache_entries_per_event_boundary_) {
        const auto page_it = kv_event_pages_.find(prefix_key);
        FatalCheck(page_it != kv_event_pages_.end(), "removed KV event boundary has no token descriptor");
        kv_events_.emplace_back(KvBlockRemovedEvent{.block_hashes = page_it->second.block_hashes});
    }
    if (--count_it->second == 0) {
        cached_event_child_counts_.erase(count_it);
        kv_event_pages_.erase(prefix_key);
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

std::size_t Scheduler::WaitingSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const auto& request) {
        return request->template Is<fsm::Submitted>() || request->template Is<fsm::Retracted>();
    }));
}

std::size_t Scheduler::DecodingSize() const {
    return static_cast<std::size_t>(
        std::ranges::count_if(requests_, [](const auto& request) { return request->template Is<fsm::Decoding>(); }));
}

std::size_t Scheduler::PrefillSize() const {
    return static_cast<std::size_t>(std::ranges::count_if(requests_, [](const auto& request) {
        return request->template Is<fsm::Prefilling>() || request->template Is<fsm::RemotePrefilling>() ||
               request->template Is<fsm::PrefillAwaitingResult>() || request->template Is<fsm::PrefillDone>();
    }));
}

std::size_t Scheduler::AvailableKvPages() const {
    return static_cast<std::size_t>(coordinator_.NumAvailableLcmBlocks());
}

std::size_t Scheduler::ActiveKvPages() const {
    std::vector<std::span<const BlockTable>> request_tables;
    request_tables.reserve(requests_.size());
    for (const auto& request : requests_) {
        if (!request->Is<fsm::Prefilling>() && !request->Is<fsm::RemotePrefilling>() &&
            !request->Is<fsm::PrefillAwaitingResult>() && !request->Is<fsm::PrefillDone>() &&
            !request->Is<fsm::Decoding>()) {
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
    const std::int32_t candidate_prefix_pages =
        std::max((static_cast<std::int32_t>(tokens.size()) - 1) / config_.prefix_granularity, 0);
    prefix_pages.resize(std::min(prefix_pages.size(), static_cast<std::size_t>(candidate_prefix_pages)));
    return ComputePrefixHashes(prefix_pages, "");
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
        if (request->Is<fsm::Submitted>() || request->Is<fsm::Prefilling>() || request->Is<fsm::RemotePrefilling>() ||
            request->Is<fsm::PrefillDone>() || request->Is<fsm::Decoding>() || request->Is<fsm::Retracted>()) {
            candidates.push_back(request.get());
        }
    }
    ExecutionPlan plan;
    auto [forward_operations, load_back_operations] =
        buildForwardOperations(plan, std::move(candidates), write_back_operations);

    plan.With(ForwardBatch{std::move(forward_operations)});

    if (config_.StreamsDeviceCacheToHost()) {
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
    return plan;
}

void Scheduler::Advance(const ExecutionEvent& event) {
    for (const auto& item : event.Events()) {
        std::visit([this](const auto& inner) { handleEvent(inner); }, item);
    }
}

}  // namespace tokenspeed
