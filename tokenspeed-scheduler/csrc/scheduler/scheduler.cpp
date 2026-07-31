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

#include "fsm/forward_states.h"
#include "scheduler/operations/cache.h"
#include "scheduler/operations/forward.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int64_t ceilDiv(std::int64_t value, std::int64_t divisor) {
    _assert(value >= 0 && divisor > 0, "ceilDiv requires non-negative value and positive divisor");
    return (value + divisor - 1) / divisor;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_{std::move(config)},
      req_pool_allocator_{config_.max_batch_size},
      block_pool_{config_.device_allocator.total_pages - 1},
      host_pool_{config_.StreamingSinkEnabled() ? config_.host_allocator.total_pages - 1 : 0},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_), config_.block_size, block_pool_,
                                   config_.StreamingSinkEnabled() ? &host_pool_ : nullptr)} {
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
    max_single_request_tokens_ = calculateMaxSingleRequestTokens();

    if (const char* level = std::getenv("SPDLOG_LEVEL")) {
        spdlog::set_level(spdlog::level::from_str(level));
    }
}

std::int64_t Scheduler::singleRequestParentsRequired(std::int32_t token_limit) const {
    _assert(token_limit >= 0, "single-request token limit must be non-negative");
    const std::int64_t page_tokens = coordinator_.CacheBlockTokens();
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

std::int32_t Scheduler::calculateMaxSingleRequestTokens() const {
    const std::int64_t usable_parents = block_pool_.NumLcmBlocks();
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

void Scheduler::StoreLedger::Add(cache_op_id id, std::vector<StoreTicket> tickets) {
    for (const StoreTicket& ticket : tickets) {
        keys_.insert(ticket.key);
    }
    const bool inserted = ops_.emplace(id, std::move(tickets)).second;
    _assert(inserted, "duplicate store op id");
}

std::vector<Scheduler::StoreTicket> Scheduler::StoreLedger::Retire(cache_op_id id) {
    auto it = ops_.find(id);
    if (it == ops_.end()) {
        return {};
    }
    for (const StoreTicket& ticket : it->second) {
        keys_.erase(ticket.key);
    }
    std::vector<StoreTicket> tickets = std::move(it->second);
    ops_.erase(it);
    return tickets;
}

Request* Scheduler::find_request(const std::string& request_id) {
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

std::vector<std::string> Scheduler::CalcRollingHash(const std::vector<std::int32_t>& input_tokens) {
    const std::size_t num_pages = input_tokens.size() / static_cast<std::size_t>(config_.block_size);
    std::vector<std::span<const std::int32_t>> token_pages;
    token_pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        token_pages.emplace_back(input_tokens.data() + i * static_cast<std::size_t>(config_.block_size),
                                 static_cast<std::size_t>(config_.block_size));
    }
    return ComputePagedHashes(token_pages, "");
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
    for (const RequestSpec& spec : request_specs) {
        auto request = std::make_unique<Request>(spec, config_.block_size, config_.role);
        const bool inserted = requests_.emplace(spec.request_id, std::move(request)).second;
        if (!inserted) {
            throw std::invalid_argument("Scheduler: duplicate request id '" + spec.request_id + "'");
        }
    }
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
        if (block_pool_.BoundGroup(id) == static_cast<GroupId>(index)) {
            available += slots_per_parent - block_pool_.OccupiedCount(id);
        }
    }
    return available;
}

std::int64_t Scheduler::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    (void)groupIndex(group_id);
    return 0;
}

std::vector<std::int32_t> Scheduler::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                 const std::string& group_id) const {
    const auto request_it = requests_.find(request_id);
    if (request_it == requests_.end() ||
        (!request_it->second->Is<fsm::Prefilling>() && !request_it->second->Is<fsm::PrefillDone>() &&
         !request_it->second->Is<fsm::Decoding>())) {
        return {};
    }
    const std::size_t index = groupIndex(group_id);
    const auto& tables = request_it->second->BlockTablesRef();
    return coordinator_.GroupManager(static_cast<std::int32_t>(index)).BlockTablePageIds(tables[index]);
}

std::int32_t Scheduler::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                            const std::string& group_id) const {
    (void)request_id;
    (void)groupIndex(group_id);
    return 0;
}

std::int32_t Scheduler::GetRequestTokenSize(const std::string& id) const {
    const auto it = requests_.find(id);
    return it == requests_.end() ? -1 : it->second->TokenSize();
}

void Scheduler::emitPendingStores(std::vector<WriteBackOperation>& write_back_ops) {
    if (!config_.StreamingSinkEnabled()) {
        return;
    }

    std::vector<TransferPair> transfers;
    std::vector<StoreTicket> tickets;
    std::unordered_set<CacheKey, CacheKeyHash> batch_keys;
    for (auto& candidate : coordinator_.TakePendingStores()) {
        if (coordinator_.ContainsHostCachedBlock(candidate.key) || store_ops_.InFlight(candidate.key) ||
            !batch_keys.insert(candidate.key).second) {
            candidate.block_ref.reset();
            continue;
        }

        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(candidate.key.group_id));
        CacheBlockRef host_block_ref =
            host_pool_.AcquireBlock(candidate.key.group_id, manager.CacheBlocksPerLcmBlock());
        if (!host_block_ref) {
            candidate.block_ref.reset();
            continue;
        }
        transfers.push_back(TransferPair{
            CacheKind::kKV,
            manager.ResolveKernelPageId(candidate.block_ref->Location()),
            manager.ResolveKernelPageId(host_block_ref->Location()),
        });
        tickets.push_back(StoreTicket{
            std::move(candidate.key),
            std::move(candidate.block_ref),
            std::move(host_block_ref),
        });
    }

    if (transfers.empty()) {
        return;
    }
    const cache_op_id op_id = allocateCacheOpId();
    store_ops_.Add(op_id, std::move(tickets));
    write_back_ops.emplace_back(op_id, std::move(transfers));
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    std::erase_if(requests_, [](const auto& item) { return item.second->template Is<fsm::Finished>(); });

    std::vector<Request*> candidates;
    candidates.reserve(requests_.size());
    for (auto& [_, request] : requests_) {
        if (request->Is<fsm::Submitted>() || request->Is<fsm::Prefilling>() || request->Is<fsm::PrefillDone>() ||
            request->Is<fsm::Decoding>()) {
            candidates.push_back(request.get());
        }
    }

    auto [forward_operations, loadback_operations] = newForwardOperation(std::move(candidates));

    ExecutionPlan plan;
    plan.With(ForwardBatch{std::move(forward_operations)});

    std::vector<WriteBackOperation> write_back_operations;
    emitPendingStores(write_back_operations);
    if (!write_back_operations.empty()) {
        plan.With(CacheOperation{WriteBackBatch{write_back_operations}});
    }
    if (!loadback_operations.empty()) {
        plan.With(CacheOperation{LoadBackBatch{loadback_operations}});
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
