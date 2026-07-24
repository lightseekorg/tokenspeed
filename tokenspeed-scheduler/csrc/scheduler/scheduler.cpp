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
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "cache/forward_cache_ops.h"
#include "fsm/cache_states.h"
#include "fsm/forward_events.h"
#include "fsm/forward_states.h"
#include "resource/kv_prefix_cache/kv_prefix_cache.h"
#include "resource/radix_tree/radix_tree.h"
#include "resource/radix_tree/tree_node.h"
#include "scheduler/execution_event.h"
#include "scheduler/operations/cache.h"
#include "scheduler/page_hasher.h"
#include "scheduler/request.h"
#include "scheduler/request_spec.h"
#include "scheduler/types.h"

namespace tokenspeed {
namespace {

SchedulerConfig ValidateFlatConfigBeforeResources(SchedulerConfig config) {
#if TOKENSPEED_FLAT_KVCACHE
    if (!config.flat_block_pools.empty() && config.device_allocator.total_pages != 0) {
        throw std::invalid_argument(
            "explicit flat_block_pools require device_allocator.total_pages=0; "
            "per-pool total_blocks is the only device page authority");
    }
#endif
    if (!config.UsesExplicitFlatPools()) {
        return config;
    }
    if (config.role != Role::kFused) {
        throw std::invalid_argument(
            "Scheduler: explicit flat KV does not support PD roles; group-aware PD transfer is required");
    }
    if (config.enable_mamba && config.mamba_pool_total_chunks > 0) {
        throw std::invalid_argument(
            "Scheduler: explicit flat KV does not support the radix-owned Mamba adjunct; "
            "model Mamba state as a coordinator-native cache group");
    }
    if (!config.disable_l2_cache && config.host_allocator.total_pages > 1) {
        throw std::invalid_argument("Scheduler: explicit flat KV is device-only; group-aware host offload is required");
    }
    if (config.enable_l3_storage) {
        throw std::invalid_argument("Scheduler: explicit flat KV does not support radix-owned L3 storage");
    }
    return config;
}

#if TOKENSPEED_FLAT_KVCACHE
std::size_t FlatCompletionDepth(const SchedulerConfig& config) {
    if (config.overlap_schedule_depth < 0 || config.overlap_schedule_depth > 1) {
        throw std::invalid_argument("Scheduler: overlap_schedule_depth must be 0 or 1");
    }
    return static_cast<std::size_t>(config.overlap_schedule_depth) + 1;
}

#endif

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_{ValidateFlatConfigBeforeResources(std::move(config))},
      device_allocator_{config_.block_size, config_.device_allocator.total_pages},
      host_allocator_{config_.block_size, config_.host_allocator.total_pages},
      mamba_allocator_{},
      req_pool_allocator_{config_.max_batch_size}
#if TOKENSPEED_FLAT_KVCACHE
      ,
      block_pools_{MakeFlatBlockPoolConfigs(config_)},
      block_pool_{block_pools_.Pool(0)},
      flat_host_pool_{config_.FlatStreamingSinkEnabled() ? config_.host_allocator.total_pages : 1},
      coordinator_{MakeCoordinator(MakeSpecsFromConfig(config_, block_pools_), block_pools_,
                                   config_.FlatStreamingSinkEnabled() ? &flat_host_pool_ : nullptr)},
      flat_completion_ledger_{FlatCompletionDepth(config_), coordinator_.Schema()},
      flat_reserved_pages_{block_pools_.Size()}
#endif
{
#if TOKENSPEED_FLAT_KVCACHE
    coordinator_.SetCompletionFencedPublication(config_.UsesExplicitFlatPools());
#endif
    const bool uses_explicit_flat_pools = config_.UsesExplicitFlatPools();
    if (!uses_explicit_flat_pools) {
        kv_prefix_cache_.emplace(&device_allocator_, &host_allocator_, config_.enable_l3_storage,
                                 config_.disable_prefix_cache);
    }
    if (config_.decode_input_tokens < 0) {
        throw std::invalid_argument("Scheduler: decode_input_tokens must be >= 0");
    }
    if (config_.overlap_schedule_depth < 0 || config_.overlap_schedule_depth > 1) {
        throw std::invalid_argument("Scheduler: overlap_schedule_depth must be 0 or 1");
    }
    if (config_.overlap_schedule_depth > 0 && config_.decode_input_tokens == 0) {
        throw std::invalid_argument("Scheduler: overlapped decode requires decode_input_tokens > 0");
    }
#if !TOKENSPEED_FLAT_KVCACHE
    radix_page_table_emissions_.resize(static_cast<std::size_t>(config_.max_batch_size) + 1);
#endif
    if (auto* env = std::getenv("SPDLOG_LEVEL")) {
        std::string level_str{env};
        spdlog::level::level_enum level = spdlog::level::from_str(level_str);
        spdlog::set_level(level);
    }

    if (config_.enable_kv_cache_events && kv_prefix_cache_.has_value()) {
        radixPrefixCache().SetKvEventSink([this](KvCacheEvent event) { kv_events_.push_back(std::move(event)); });
    }
    const bool has_mamba_pool = config_.enable_mamba && config_.mamba_pool_total_chunks > 0;
    if (has_mamba_pool) {
        mamba_allocator_.emplace(config_.mamba_pool_total_chunks);
    }
    const bool has_mamba_l2_pool = has_mamba_pool && config_.enable_mamba_l2 && config_.mamba_l2_host_slots > 0;
    if (has_mamba_l2_pool) {
        mamba_host_allocator_.emplace(config_.mamba_l2_host_slots);
    }

    // Construct HybridPrefixCache when any adjunct/paged-cache feature is configured.
    // Role::kD skips Mamba but still participates in paged-cache transport.
    const bool has_mamba_adjunct = has_mamba_pool && config_.role != Role::kD;
    // In explicit-pool mode, BlockPoolSet/KvCacheCoordinator is
    // the sole page owner. Paged group configs remain scheduling geometry only;
    // do not mirror their capacities into HybridPrefixCache allocators or its
    // radix snapshot adjunct. The legacy prefix_cache_adjunct may still arrive
    // from a model-specific Python config, but prefix_role is the
    // coordinator's sole policy
    // authority. Radix-owned Mamba was rejected above.
    const bool has_prefix_cache_adjunct = config_.prefix_cache_adjunct.has_value() && !uses_explicit_flat_pools;
    const bool has_paged_cache_groups = !config_.paged_cache_groups.empty() && !uses_explicit_flat_pools;
    if (has_mamba_adjunct || has_prefix_cache_adjunct || has_paged_cache_groups) {
        MambaChunkAllocator* mamba_ptr = has_mamba_adjunct ? &*mamba_allocator_ : nullptr;
        MambaHostAllocator* mamba_host_ptr = has_mamba_l2_pool ? &*mamba_host_allocator_ : nullptr;
        hybrid_prefix_cache_.emplace(radixPrefixCache(), mamba_ptr, config_.mamba_cache_chunk_size, mamba_host_ptr);
        radixPrefixCache().GetDeviceManager().SetEvictionCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnKVEvict(node); });
        radixPrefixCache().GetHostManager().SetEvictionCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnKVHostEvict(node); });
        // Prune frees TreeNodes (including empty ancestors) outside the per-tier
        // eviction callbacks; un-register them from the adjunct sets before the
        // node is destroyed so mamba_leaves_ / paged-cache membership never
        // dangles.
        radixPrefixCache().GetRadixTree().SetNodeDestroyCallback(
            [this](TreeNode* node) { hybrid_prefix_cache_->OnNodeDestroyed(node); });

        if (has_paged_cache_groups) {
            for (const auto& cfg : config_.paged_cache_groups) {
                PagedCacheGroupConfig copy = cfg;
                copy.Validate();
                hybrid_prefix_cache_->RegisterPagedCacheGroup(
                    std::make_unique<PagedCacheGroupAllocator>(std::move(copy)));
            }
        }

        if (has_prefix_cache_adjunct) {
            const auto& spec = *config_.prefix_cache_adjunct;
            if (spec.required_groups.empty()) {
                throw std::invalid_argument("Scheduler: prefix_cache_adjunct.required_groups must be non-empty");
            }
            // HybridPrefixCache derives history alignment from the registered
            // group configs; we still build the sliding-window map here.
            std::unordered_map<std::string, std::int32_t> sliding_window_per_group;
            for (const auto& gid : spec.required_groups) {
                const PagedCacheGroupConfig* cfg = nullptr;
                for (const auto& g : config_.paged_cache_groups) {
                    if (g.group_id == gid) {
                        cfg = &g;
                        break;
                    }
                }
                if (cfg == nullptr) {
                    throw std::invalid_argument("Scheduler: prefix_cache_adjunct required group_id '" + gid +
                                                "' not found in paged_cache_groups");
                }
                if (cfg->retention == PagedCacheGroupConfig::Retention::SlidingWindow) {
                    if (!cfg->sliding_window_tokens.has_value() || *cfg->sliding_window_tokens <= 0) {
                        throw std::invalid_argument("Scheduler: prefix_cache_adjunct sliding group '" + gid +
                                                    "' must declare positive sliding_window_tokens");
                    }
                    sliding_window_per_group.emplace(gid, *cfg->sliding_window_tokens);
                }
            }
            hybrid_prefix_cache_->EnablePagedCacheAdjunct(spec.required_groups, std::move(sliding_window_per_group));
        }
    }
}

#if TOKENSPEED_FLAT_KVCACHE
std::uint64_t Scheduler::FlatKVGeneration() const {
    return block_pools_.Generation();
}

bool Scheduler::FlatKVQuiescent() const {
    return requests_.empty() && pending_forward_results_.empty() && flat_reserved_pages_.Empty() &&
           cache_op_tracker_.empty() && flat_store_ops_.Empty() && flat_load_ops_.empty() && block_pools_.IsQuiescent();
}

std::uint64_t Scheduler::ResetFlatKVCache() {
    if (!config_.UsesExplicitFlatPools()) {
        throw std::logic_error("flat KV cache reset requires explicit block pools");
    }
    if (!FlatKVQuiescent()) {
        throw std::logic_error("cannot reset flat KV cache while scheduler state is not quiescent");
    }
    return block_pools_.ResetQuiescent();
}

void Scheduler::consumeFlatCompletionDebt(const std::string& request_id, std::size_t count) {
    if (count == 0) {
        return;
    }
    auto it = pending_forward_results_.find(request_id);
    _assert(it != pending_forward_results_.end(), "flat completion debt is absent");
    _assert(count <= static_cast<std::size_t>(it->second), "flat completion debt underflow");
    it->second -= static_cast<std::int32_t>(count);
    if (it->second == 0) {
        pending_forward_results_.erase(it);
    }
}

#endif

std::vector<KvCacheEvent> Scheduler::DrainKvEvents() {
    std::vector<KvCacheEvent> events;
    events.swap(kv_events_);
    return events;
}

std::vector<std::string> Scheduler::CalcRollingHash(const std::vector<std::int32_t>& input_tokens, bool apply_match) {
    const std::int32_t block_size =
#if TOKENSPEED_FLAT_KVCACHE
        coordinator_.BaseBlockSize();
#else
        config_.block_size;
#endif
    const std::size_t num_pages = input_tokens.size() / block_size;
    std::vector<std::span<const std::int32_t>> token_pages;
    token_pages.reserve(num_pages);
    for (std::size_t i = 0; i < num_pages; ++i) {
        token_pages.emplace_back(input_tokens.data() + i * block_size, block_size);
    }
    if (!apply_match) {
        return ComputePagedHashes(token_pages, "");
    }
    KVPrefixCache& prefix_cache = radixPrefixCache();
    MatchResult result = prefix_cache.Match(token_pages);
    const std::int32_t host_matched = result.host.DepthInPage();
    if (host_matched >= static_cast<std::int32_t>(num_pages)) {
        return {};
    }
    const auto& hashes = result.host.last_node->PageHashes();
    std::string prior = hashes.empty() ? std::string{} : hashes.back();

    return ComputePagedHashes(
        std::vector<std::span<const std::int32_t>>(token_pages.begin() + host_matched, token_pages.end()), prior);
}

void Scheduler::SubmitRequests(const std::vector<RequestSpec>& request_specs) {
#if TOKENSPEED_FLAT_KVCACHE
    std::size_t max_request_id_size = flat_max_request_id_size_;
    for (const RequestSpec& spec : request_specs) {
        max_request_id_size = std::max(max_request_id_size, spec.request_id.size());
    }
    // Prepare all terminal bookkeeping before any request in this admission
    // batch becomes visible. These capacities are then reused across plans.
    flat_oom_request_ids_.reserve(1);
    flat_starved_request_id_.reserve(max_request_id_size);
    flat_oom_terminal_scratch_.reserve(max_request_id_size);
    flat_max_request_id_size_ = max_request_id_size;
#endif
    if (request_specs.size() > requests_.max_size() - requests_.size()) {
        throw std::length_error("request batch exceeds scheduler map max_size");
    }
    requests_.reserve(requests_.size() + request_specs.size());
#if TOKENSPEED_FLAT_KVCACHE
    // The content-hash chain and base-slot indexing run at base (GCD) granularity; the
    // coordinator folds base pages up to each group's block_size. Uniform block_size =>
    // base == block_size.
    const std::int32_t page_size = coordinator_.BaseBlockSize();
#else
    const std::int32_t page_size = config_.block_size;
#endif
    for (const auto& spec : request_specs) {
        auto req = std::make_unique<Request>(spec, page_size, config_.role);
#if TOKENSPEED_FLAT_KVCACHE
        req->AttachFlatReservation(flat_reserved_pages_);
#endif
        requests_.emplace(spec.request_id, std::move(req));
    }
}

std::size_t Scheduler::WaitingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Submitted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::DecodingSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Decoding>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::PrefillSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::RetractedSize() const {
    std::size_t count = 0;
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Retracting>() || req->Is<fsm::Retracted>()) {
            count++;
        }
    }
    return count;
}

std::size_t Scheduler::AvailableKvPages() const {
#if TOKENSPEED_FLAT_KVCACHE
    // Legacy scalar metric: aggregate cardinality across independent local-ID
    // domains. Admission and pressure never consume this lossy aggregate.
    std::size_t free_blocks = 0;
    for (PoolIndex i = 0; i < block_pools_.Size(); ++i) {
        free_blocks += static_cast<std::size_t>(block_pools_.Pool(i).NumFreeBlocks());
    }
    return free_blocks;
#else
    return device_allocator_.AvailablePages();
#endif
}

std::size_t Scheduler::ActiveKvPages() const {
#if TOKENSPEED_FLAT_KVCACHE
    // Flat page ids are local to each independent pool. Count distinct live
    // physical blocks per pool, then sum the cardinalities; a single set would
    // incorrectly deduplicate equal local ids from heterogeneous pools.
    std::size_t active_blocks = 0;
    for (PoolIndex i = 0; i < block_pools_.Size(); ++i) {
        const BlockPool& pool = block_pools_.Pool(i);
        active_blocks += static_cast<std::size_t>((pool.TotalBlocks() - 1) - pool.NumFreeBlocks());
    }
    return active_blocks;
#else
    // Radix page ids share one allocator domain. Scan every paged-cache group
    // and deduplicate pages shared between requests through prefix hits.
    std::unordered_set<std::int32_t> active_pages;
    for (const auto& [_, req] : requests_) {
        if (req->Is<fsm::Prefilling>() || req->Is<fsm::PrefillDone>() || req->Is<fsm::Decoding>()) {
            for (std::int32_t page : req->GetOccupiedPagesAllGroups()) {
                active_pages.insert(page);
            }
        }
    }
    return active_pages.size();
#endif
}

#if TOKENSPEED_FLAT_KVCACHE
PoolDemand Scheduler::FlatReservedBlocksByPool() const {
    return flat_reserved_pages_.Total();
}

FlatPoolAggregate Scheduler::FlatPoolAggregateStats() const {
    const PoolDemand& reserved = flat_reserved_pages_.Total();
    _assert(reserved.Size() == block_pools_.Size(), "flat pool aggregate/reservation shape mismatch");

    FlatPoolAggregate aggregate;
    for (PoolIndex i = 0; i < block_pools_.Size(); ++i) {
        const FlatBlockPoolConfig& config = block_pools_.Config(i);
        const std::int32_t usable = config.total_blocks - 1;
        const std::int32_t free = block_pools_.Pool(i).NumFreeBlocks();
        const std::int32_t active = usable - free;
        const std::int64_t bytes_per_block = config.bytes_per_block;
        if (bytes_per_block != 0 && usable > std::numeric_limits<std::int64_t>::max() / bytes_per_block) {
            throw std::overflow_error("flat pool aggregate capacity bytes overflow");
        }
        const std::int64_t pool_capacity = static_cast<std::int64_t>(usable) * bytes_per_block;
        const std::int64_t pool_active = static_cast<std::int64_t>(active) * bytes_per_block;
        if (aggregate.capacity_bytes > std::numeric_limits<std::int64_t>::max() - pool_capacity ||
            aggregate.active_bytes > std::numeric_limits<std::int64_t>::max() - pool_active) {
            throw std::overflow_error("flat pool aggregate bytes overflow");
        }
        aggregate.capacity_bytes += pool_capacity;
        aggregate.active_bytes += pool_active;

        const std::int32_t available_unreserved = std::clamp(free - reserved[i], 0, usable);
        const std::int32_t used_or_reserved = usable - available_unreserved;
        if (static_cast<std::int64_t>(used_or_reserved) * aggregate.pressure_denominator >
            static_cast<std::int64_t>(aggregate.pressure_numerator) * usable) {
            aggregate.pressure_numerator = used_or_reserved;
            aggregate.pressure_denominator = usable;
        }
    }
    return aggregate;
}

std::vector<BlockPoolSnapshot> Scheduler::FlatPoolSnapshots() const {
    std::vector<BlockPoolSnapshot> snapshots = block_pools_.Snapshot();
    const PoolDemand reserved = FlatReservedBlocksByPool();
    _assert(snapshots.size() == reserved.Size(), "flat pool snapshot/reservation shape mismatch");
    for (PoolIndex i = 0; i < snapshots.size(); ++i) {
        snapshots[i].reserved_blocks = reserved[i];
    }
    return snapshots;
}
#endif

std::vector<std::string> Scheduler::PagedCacheGroupIds() const {
#if TOKENSPEED_FLAT_KVCACHE
    if (!coordinator_.Schema().empty()) {
        std::vector<std::string> group_ids;
        group_ids.reserve(coordinator_.Schema().size());
        for (const KvCacheGroupSchema& group : coordinator_.Schema()) {
            group_ids.push_back(group.group_id);
        }
        return group_ids;
    }
#endif
    if (!hybrid_prefix_cache_) return {};
    return hybrid_prefix_cache_->PagedCacheGroupIds();
}

std::int32_t Scheduler::PagedCacheGroupTotalPages(const std::string& group_id) const {
#if TOKENSPEED_FLAT_KVCACHE
    if (const std::optional<std::size_t> schema_index = coordinator_.FindSchemaIndex(group_id)) {
        return block_pools_.Config(coordinator_.GroupSchema(*schema_index).pool_index).total_blocks;
    }
#endif
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupTotalPages: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupTotalPages(group_id);
}

std::int32_t Scheduler::PagedCacheGroupAvailablePages(const std::string& group_id) const {
#if TOKENSPEED_FLAT_KVCACHE
    if (const std::optional<std::size_t> schema_index = coordinator_.FindSchemaIndex(group_id)) {
        return block_pools_.Pool(coordinator_.GroupSchema(*schema_index).pool_index).NumFreeBlocks();
    }
#endif
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupAvailablePages: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupAvailablePages(group_id);
}

std::int64_t Scheduler::PagedCacheGroupFailedAllocCount(const std::string& group_id) const {
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::PagedCacheGroupFailedAllocCount: group_id not configured");
    }
    return hybrid_prefix_cache_->PagedCacheGroupFailedAllocCount(group_id);
}

std::vector<std::int32_t> Scheduler::GetRequestPagedCachePageIds(const std::string& request_id,
                                                                 const std::string& group_id) const {
#if TOKENSPEED_FLAT_KVCACHE
    if (const std::optional<std::size_t> schema_index = coordinator_.FindSchemaIndex(group_id)) {
        const auto request = requests_.find(request_id);
        if (request == requests_.end() || request->second->FlatBlockTablesEmpty()) {
            return {};
        }
        const std::vector<BlockTable>& tables = request->second->FlatBlockTablesRef();
        _assert(*schema_index < tables.size(), "flat request table/group schema mismatch");
        return BlockTablePageIds(tables[*schema_index]);
    }
#endif
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::GetRequestPagedCachePageIds: group_id not configured");
    }
    return hybrid_prefix_cache_->GetRequestPagedCachePageIds(request_id, group_id);
}

std::int32_t Scheduler::GetRequestPagedCacheBaseLogicalPage(const std::string& request_id,
                                                            const std::string& group_id) const {
#if TOKENSPEED_FLAT_KVCACHE
    if (const std::optional<std::size_t> schema_index = coordinator_.FindSchemaIndex(group_id)) {
        const auto request = requests_.find(request_id);
        if (request == requests_.end() || request->second->FlatBlockTablesEmpty()) {
            return 0;
        }
        const std::vector<BlockTable>& tables = request->second->FlatBlockTablesRef();
        _assert(*schema_index < tables.size(), "flat request table/group schema mismatch");
        return tables[*schema_index].BaseLogicalPage();
    }
#endif
    if (!hybrid_prefix_cache_) {
        throw std::out_of_range("Scheduler::GetRequestPagedCacheBaseLogicalPage: group_id not configured");
    }
    return hybrid_prefix_cache_->GetRequestPagedCacheBaseLogicalPage(request_id, group_id);
}

std::int32_t Scheduler::GetRequestTokenSize(const std::string& id) const {
    auto it = requests_.find(id);
    if (it == requests_.end()) {
        return -1;
    }
    return it->second->TokenSize();
}

std::vector<WriteBackOperation> Scheduler::newWriteBackOperation(
    std::unordered_map<std::string, std::unique_ptr<Request>>& requests) {
    std::vector<WriteBackOperation> ops;
    if (config_.disable_l2_cache) {
        return ops;
    }
    for (auto& [id, req] : requests) {
        if (!req->Is<fsm::Draining>()) continue;
        const auto& pages_to_transfer = req->GetPagesToTransfer<fsm::Draining>();

        if (!pages_to_transfer.empty()) {
            cache_op_id op_id = radixPrefixCache().AllocateCacheOpId();
            CacheOpSpec spec;
            spec.request_id = id;
            cache_op_tracker_[op_id] = std::move(spec);
            ops.push_back(WriteBackOperation{
                op_id, std::vector<TransferPair>(pages_to_transfer.begin(), pages_to_transfer.end())});
            req->Apply(fsm::CommitDrainingEvent{});
        } else {
            req->Apply(fsm::AbortEvent{
#if TOKENSPEED_FLAT_KVCACHE
                &coordinator_
#endif
            });
        }
    }
    return ops;
}

ExecutionPlan Scheduler::NextExecutionPlan() {
    ExecutionPlan plan;
    // One forward batch plus at most one write-back and one load-back batch.
    // Reserve before any scheduler/FSM mutation so final publication consists
    // solely of no-throw Operation moves.
    plan.ReserveOperations(3);
#if TOKENSPEED_FLAT_KVCACHE
    if (!requests_.empty()) {
        if (flat_oom_request_ids_.size() == flat_oom_request_ids_.max_size()) {
            throw std::length_error("flat OOM outbox exceeds max_size");
        }
        flat_oom_request_ids_.reserve(flat_oom_request_ids_.size() + 1);
        flat_starved_request_id_.reserve(flat_max_request_id_size_);
        flat_oom_terminal_scratch_.clear();
        flat_oom_terminal_scratch_.reserve(flat_max_request_id_size_);
    }
#endif

    std::vector<WriteBackOperation> write_back_ops;
    write_back_ops = std::move(newWriteBackOperation(requests_));

    if (hybrid_prefix_cache_) {
        for (const auto& [id, req] : requests_) {
            if (req->Is<fsm::Finished>()) {
                hybrid_prefix_cache_->ReleaseRequest(id);
            }
        }
    }
#if TOKENSPEED_FLAT_KVCACHE
    for (const auto& [id, req] : requests_) {
        if (req->Is<fsm::Finished>()) {
            pending_forward_results_.erase(id);
            req->FlatReservation().Clear();
        }
    }
#endif
    std::erase_if(requests_, [](const auto& req) { return req.second->template Is<fsm::Finished>(); });

    std::vector<Request*> candidates;
    for (auto& [id, req] : requests_) {
        if (!req->Is<fsm::Draining>() && !req->Is<fsm::Prefetching>() && !req->Is<fsm::Retracting>() &&
            !req->Is<fsm::WritingBack>()) {
            candidates.push_back(req.get());
        }
    }

    auto [fwd_ops, cache_ops] = newForwardOperation(candidates);
    // Flat table rectangles and all SoA storage are fully materialized before
    // completion debt or execution-plan publication. The production row view
    // copies each table cell exactly once into this owner. Any failure after
    // newForwardOperation propagates to the event-loop boundary; the scheduler
    // does not attempt to resume from a partially mutated FSM.
    FlatForwardOperation flat_forward_op{std::move(fwd_ops)};
#if TOKENSPEED_FLAT_KVCACHE
    flat_forward_op.cache_generation = block_pools_.Generation();
#endif

    // Merge retract write-backs (if any) into the Draining write-back list, then emit once.
    if (auto* wb = std::get_if<std::vector<WriteBackOperation>>(&cache_ops)) {
        write_back_ops.insert(write_back_ops.end(), std::make_move_iterator(wb->begin()),
                              std::make_move_iterator(wb->end()));
    }
#if TOKENSPEED_FLAT_KVCACHE
    if (config_.FlatStreamingSinkEnabled()) {
        // Streaming L2 sink: batch this round's newly-registered pages into one D2H op.
        std::vector<TransferPair> pairs;
        std::vector<FlatStoreTicket> tickets;
        // Same-round twins register the same key twice (batch_keys catches them). Cross-round
        // recurrence is rare but real: a device match can settle below a still-in-flight page after
        // earlier chain / SWA-neighbor ops retire, and the request re-registers a key whose store is
        // in flight. InFlight() drops it (load-bearing: else a key sits in two ops and Retire
        // corrupts the ledger's key set).
        std::unordered_set<std::string> batch_keys;
        for (auto& cand : coordinator_.TakePendingStores()) {
            if (flat_host_pool_.ContainsCachedBlock(cand.key) || flat_store_ops_.InFlight(cand.key) ||
                !batch_keys.insert(cand.key).second) {
                cand.block.reset();  // duplicate: drop + unpin
                continue;
            }
            BlockRef host_block = flat_host_pool_.AcquireBlock();
            if (!host_block) {
                cand.block.reset();  // host full: drop + unpin
                continue;
            }
            pairs.push_back(TransferPair{CacheKind::kKV, cand.block->BlockId(), host_block->BlockId()});
            tickets.push_back(FlatStoreTicket{std::move(cand.key), std::move(cand.block), std::move(host_block)});
        }
        if (!pairs.empty()) {
            const cache_op_id id = radixPrefixCache().AllocateCacheOpId();
            flat_store_ops_.Add(id, std::move(tickets));
            write_back_ops.push_back(WriteBackOperation{id, std::move(pairs)});
        }
    }
#endif
    std::optional<CacheOperation> write_back_plan_op;
    if (!write_back_ops.empty()) {
        write_back_plan_op.emplace(FlatWriteBackOperation{write_back_ops});
    }
    std::optional<CacheOperation> load_back_plan_op;
    if (auto* lb = std::get_if<std::vector<LoadBackOperation>>(&cache_ops)) {
        if (!lb->empty()) {
            load_back_plan_op.emplace(FlatLoadBackOperation{*lb});
        }
    }
#if TOKENSPEED_FLAT_KVCACHE
    // Materialize legacy debt nodes and reserve the destination buckets before
    // publishing either one. The final node-handle merge below is
    // allocation-free and occurs once, immediately before return.
    std::unordered_map<std::string, std::int32_t> legacy_debt;
    if (config_.UsesExplicitFlatPools()) {
        flat_forward_op.flat_kv_completion_inputs.reserve(flat_forward_op.request_ids.size());
    } else {
        legacy_debt.reserve(flat_forward_op.request_ids.size());
        for (const std::string& request_id : flat_forward_op.request_ids) {
            Request* request = find_request(request_id);
            _assert(request != nullptr, "flat forward request disappeared before completion publication");
            // Legacy flat ABI emits ExtendResult only for prefill-completing
            // and decode operations; mid-prefill chunks owe no result.
            if (!request->Is<fsm::Prefilling>()) {
                legacy_debt.emplace(request_id, 1);
            }
        }
        pending_forward_results_.reserve(pending_forward_results_.size() + legacy_debt.size());
    }
#endif
    if (std::getenv("DEBUG_MEM")) {
        check_device_mem();
    }
#if TOKENSPEED_FLAT_KVCACHE
    if (config_.UsesExplicitFlatPools()) {
        // All fallible plan storage is materialized above, and the
        // request-owned FIFO is now the single authority for the dispatches
        // being returned.
        for (std::size_t row = 0; row < flat_forward_op.request_ids.size(); ++row) {
            Request* request = find_request(flat_forward_op.request_ids[row]);
            _assert(request != nullptr, "flat forward request disappeared before completion attachment");
            attachFlatKVCompletionInput(request, flat_forward_op, row,
                                        /*apply_fsm_result=*/!request->Is<fsm::Prefilling>());
        }
    }
#endif

    plan.With(std::move(flat_forward_op));
    if (write_back_plan_op.has_value()) {
        plan.With(std::move(*write_back_plan_op));
    }
    if (load_back_plan_op.has_value()) {
        plan.With(std::move(*load_back_plan_op));
    }

#if TOKENSPEED_FLAT_KVCACHE
    while (!legacy_debt.empty()) {
        auto node = legacy_debt.extract(legacy_debt.begin());
        auto existing = pending_forward_results_.find(node.key());
        if (existing != pending_forward_results_.end()) {
            existing->second += node.mapped();
            continue;
        }
        const auto inserted = pending_forward_results_.insert(std::move(node));
        _assert(inserted.inserted, "legacy flat completion debt publication failed");
    }
    // OOM notifications stay in the scheduler outbox across every
    // materialization step and transfer only after publication.
    if (!flat_oom_request_ids_.empty()) {
        plan.TakeFlatOomRequestIds(flat_oom_request_ids_);
    }
#endif
    return plan;
}

void Scheduler::check_device_mem() {
    if (!kv_prefix_cache_.has_value()) {
        return;
    }
    bool ok = true;
    const std::int32_t total_device = device_allocator_.TotalPages() - 1;
    std::unordered_map<std::string, std::vector<std::int32_t>> req_pages_map;
    // page_id → (owner_req_id, state_name) for duplicate tail-page reporting
    std::unordered_map<std::int32_t, std::pair<std::string, std::string>> page_owner;

    for (auto& [id, req] : requests_) {
        std::string state = req->StateName();
        std::vector<std::int32_t> pages = req->GetLocalAllocatorPages();
        if (pages.empty()) continue;
        req_pages_map[id] = pages;

        for (std::int32_t p : pages) {
            auto [it, inserted] = page_owner.emplace(p, std::make_pair(id, state));
            if (!inserted) {
                spdlog::error("[check_mem] DEVICE TAIL PAGE OVERLAP: page={}  req1={}({})  req2={}({})", p,
                              it->second.first, it->second.second, id, state);
                ok = false;
            }
        }
    }

    // ── 2. Collect pages in radix tree ───────────────────────────────────────
    auto tree_device_pages = radixPrefixCache().CollectAllPages<ResourceType::Device>();

    // 2a. Check for duplicate page_ids inside the tree itself
    for (auto& [page, cnt] : tree_device_pages) {
        if (cnt > 1) {
            spdlog::error("[check_mem] DEVICE TREE DUPLICATE: page={} appears {} times in radix tree", page, cnt);
            ok = false;
        }
    }

    std::int32_t tree_device_total = static_cast<std::int32_t>(tree_device_pages.size());

    std::int32_t req_device_total = 0;
    for (auto& [id, pages] : req_pages_map) req_device_total += static_cast<std::int32_t>(pages.size());

    std::int32_t free_device = device_allocator_.AvailablePages();

    if (tree_device_total + req_device_total + free_device != total_device) {
        spdlog::error("[check_mem] DEVICE PAGE ACCOUNTING MISMATCH: tree={} req={} free={} sum={} total={}",
                      tree_device_total, req_device_total, free_device,
                      tree_device_total + req_device_total + free_device, total_device);
        ok = false;
    }

    // ── 4. Per-request: page ids must be in [1, total] ────────────────────
    // PageAllocator starts from page id 1 (0 is reserved as invalid/null).
    for (auto& [id, pages] : req_pages_map) {
        for (std::int32_t p : pages) {
            if (p <= 0 || p > total_device) {
                spdlog::error("[check_mem] INVALID DEVICE PAGE id={} for req={} (valid range [1,{}])", p, id,
                              total_device);
                ok = false;
            }
        }
    }
    for (auto& [p, cnt] : tree_device_pages) {
        if (p <= 0 || p > total_device) {
            spdlog::error("[check_mem] INVALID DEVICE PAGE id={} in radix tree (valid range [1,{}])", p, total_device);
            ok = false;
        }
    }

    // ── 5. Summary ────────────────────────────────────────────────────────────
    if (!ok) {
        throw std::runtime_error("Scheduler::CheckMem: device page accounting check failed");
    }
}

void Scheduler::Advance(const ExecutionEvent& event) {
    auto dispatch = [this](const auto& inner) { handleEvent(inner); };
    for (const auto& item : event.Events()) {
        std::visit([&](const auto& outer) { std::visit(dispatch, outer); }, item);
    }
}

}  // namespace tokenspeed
