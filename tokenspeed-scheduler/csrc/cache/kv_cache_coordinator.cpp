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

#include "cache/kv_cache_coordinator.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::vector<BlockPool*> CollectPools(BlockPoolSet& pools) {
    std::vector<BlockPool*> out;
    out.reserve(pools.Size());
    for (PoolIndex i = 0; i < pools.Size(); ++i) {
        out.push_back(&pools.Pool(i));
    }
    return out;
}

struct BlockGeometry {
    std::int32_t base;
    std::int32_t history_alignment;
};

std::vector<KvCacheGroupSchema> NormalizeSchema(std::span<const KvCacheSpec> specs, std::size_t pool_count) {
    if (specs.empty()) {
        throw std::invalid_argument("MakeCoordinator requires at least one spec");
    }
    if (specs.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("flat KV group schema exceeds the uint32 hash-domain index space");
    }

    std::vector<KvCacheGroupSchema> schema;
    schema.reserve(specs.size());
    std::unordered_set<std::string> group_ids;
    group_ids.reserve(specs.size());
    for (std::size_t schema_index = 0; schema_index < specs.size(); ++schema_index) {
        const KvCacheSpec& spec = specs[schema_index];
        if (spec.block_size <= 0) {
            throw std::invalid_argument("flat KV schema block_size must be positive");
        }
        if (spec.pool_index >= pool_count) {
            throw std::out_of_range("KvCacheSpec pool_index is outside the configured BlockPoolSet");
        }

        const std::int32_t rows_per_page = spec.rows_per_page > 0 ? spec.rows_per_page : spec.block_size;
        const std::int32_t entry_stride_tokens = spec.rows_per_page > 0 ? spec.entry_stride_tokens : std::int32_t{1};
        if (entry_stride_tokens <= 0 ||
            static_cast<std::int64_t>(rows_per_page) * entry_stride_tokens != spec.block_size) {
            throw std::invalid_argument("flat KV schema block_size must equal rows_per_page * entry_stride_tokens");
        }
        std::string group_id = spec.group_id.empty() ? std::to_string(schema_index) : spec.group_id;
        if (!group_ids.emplace(group_id).second) {
            throw std::invalid_argument("duplicate flat KV group_id '" + group_id + "'");
        }
        schema.push_back(KvCacheGroupSchema{
            .group_id = std::move(group_id),
            .kind = spec.kind,
            .block_size = spec.block_size,
            .rows_per_page = rows_per_page,
            .entry_stride_tokens = entry_stride_tokens,
            .sliding_window = spec.sliding_window,
            .pool_index = spec.pool_index,
            .prefix_role = spec.prefix_role,
            .table_layout = spec.table_layout,
            .owner_mask = spec.owner_mask,
        });
    }
    return schema;
}

BlockGeometry ComputeBlockGeometry(std::span<const KvCacheGroupSchema> schema) {
    _assert(!schema.empty(), "MakeCoordinator requires at least one schema entry");
    std::int32_t base = schema[0].block_size;
    std::int32_t history_lcm = 0;
    for (const KvCacheGroupSchema& group : schema) {
        base = std::gcd(base, group.block_size);
        if (group.prefix_role == KvPrefixRole::kHistoryAnchor) {
            history_lcm = history_lcm == 0 ? group.block_size : std::lcm(history_lcm, group.block_size);
        }
    }
    return BlockGeometry{base, history_lcm == 0 ? base : history_lcm};
}

std::unique_ptr<KvCacheManager> MakeManager(const KvCacheGroupSchema& group) {
    if (group.kind == AttnKind::kFull) {
        return std::make_unique<FullAttnManager>(group.block_size);
    }
    if (group.kind == AttnKind::kMambaState) {
        return std::make_unique<MambaStateManager>(group.block_size, group.table_layout);
    }
    return std::make_unique<SwaManager>(group.block_size, group.sliding_window, group.table_layout);
}

std::vector<CacheGroup> BuildGroups(std::span<const KvCacheGroupSchema> schema, std::span<BlockPool* const> pools,
                                    std::int32_t base_block_size) {
    std::vector<CacheGroup> groups;
    groups.reserve(schema.size());
    for (std::size_t schema_index = 0; schema_index < schema.size(); ++schema_index) {
        const KvCacheGroupSchema& group = schema[schema_index];
        _assert(group.block_size % base_block_size == 0, "group block_size must be a multiple of base");
        groups.emplace_back(static_cast<std::uint32_t>(schema_index), MakeManager(group), *pools[group.pool_index]);
    }
    return groups;
}

void AddDemand(PoolDemand& demand, PoolIndex pool_index, std::int32_t blocks) {
    _assert(blocks >= 0, "pool demand must be non-negative");
    _assert(demand[pool_index] <= std::numeric_limits<std::int32_t>::max() - blocks, "pool demand overflows int32");
    demand[pool_index] += blocks;
}

template <typename MatchGroup, typename ExtentTokens>
std::int32_t SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                               std::int32_t bound_tokens, std::int32_t align_tokens, const MatchGroup& match,
                               const ExtentTokens& extent) {
    const auto align_down = [align_tokens](std::int32_t tokens) { return tokens - tokens % align_tokens; };
    bound_tokens = align_down(bound_tokens);
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, align_down(extent(i)));
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Manager().MatchIsPrefixClosed() || extent(i) <= bound_tokens) {
                continue;
            }
            match(i, bound_tokens);
            bound_tokens = std::min(bound_tokens, align_down(extent(i)));
            changed = true;
        }
    }
    return bound_tokens;
}

}  // namespace

KvCacheCoordinator::KvCacheCoordinator(std::vector<KvCacheGroupSchema> schema, std::vector<CacheGroup> groups,
                                       BlockPool& pool, BlockPool* host_pool, std::int32_t base_block_size,
                                       std::int32_t history_alignment_tokens)
    : schema_{std::move(schema)},
      groups_{std::move(groups)},
      pools_{&pool},
      host_pool_{host_pool},
      base_block_size_{base_block_size},
      history_alignment_tokens_{history_alignment_tokens} {
    Initialize();
}

KvCacheCoordinator::KvCacheCoordinator(std::vector<KvCacheGroupSchema> schema, std::vector<CacheGroup> groups,
                                       BlockPoolSet& pools, BlockPool* host_pool, std::int32_t base_block_size,
                                       std::int32_t history_alignment_tokens)
    : schema_{std::move(schema)},
      groups_{std::move(groups)},
      pools_{CollectPools(pools)},
      host_pool_{host_pool},
      base_block_size_{base_block_size},
      history_alignment_tokens_{history_alignment_tokens} {
    Initialize();
}

void KvCacheCoordinator::Initialize() {
    if (pools_.empty()) {
        throw std::invalid_argument("KvCacheCoordinator requires at least one device pool");
    }
    if (host_pool_ != nullptr && pools_.size() > 1) {
        throw std::invalid_argument(
            "heterogeneous flat KV pools do not support a shared host tier; "
            "group-aware offload is required");
    }
    _assert(base_block_size_ > 0 && history_alignment_tokens_ > 0,
            "coordinator needs positive base and history alignment");

    if (schema_.size() != groups_.size()) {
        throw std::invalid_argument("flat KV schema/runtime group count mismatch");
    }
    schema_index_by_group_id_.reserve(schema_.size());
    match_order_.reserve(groups_.size());
    bool has_history_anchor = false;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (groups_[i].SchemaIndex() != i) {
            throw std::logic_error("CacheGroup order does not match its canonical schema index");
        }
        const KvCacheGroupSchema& spec = schema_[i];
        if (!schema_index_by_group_id_.emplace(spec.group_id, i).second) {
            throw std::invalid_argument("duplicate flat KV group_id '" + spec.group_id + "'");
        }
        const PoolIndex pool_index = spec.pool_index;
        if (pool_index >= pools_.size()) {
            throw std::out_of_range("CacheGroup pool_index is outside the coordinator pool registry");
        }
        if (&groups_[i].Pool() != pools_[pool_index]) {
            throw std::logic_error("CacheGroup pool binding does not match its canonical pool_index");
        }
        has_history_anchor = has_history_anchor || spec.prefix_role == KvPrefixRole::kHistoryAnchor;
        has_continuation_state_ = has_continuation_state_ || spec.prefix_role == KvPrefixRole::kContinuationState;
        if (spec.prefix_role == KvPrefixRole::kHistoryAnchor && groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (schema_[i].prefix_role == KvPrefixRole::kHistoryAnchor && !groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    if (has_continuation_state_ && !has_history_anchor) {
        throw std::invalid_argument("continuation-state flat groups require at least one history anchor");
    }
    if (has_continuation_state_ && host_pool_ != nullptr) {
        throw std::invalid_argument(
            "exact continuation-state prefix matching is device-only; "
            "group-aware host transfer is required");
    }
    if (has_continuation_state_) {
        for (const KvCacheGroupSchema& spec : schema_) {
            if (spec.prefix_role != KvPrefixRole::kContinuationState) {
                continue;
            }
            if (history_alignment_tokens_ % spec.block_size != 0) {
                throw std::invalid_argument("history alignment must be divisible by every continuation block size");
            }
            if (spec.sliding_window <= 0 || spec.sliding_window % spec.block_size != 0) {
                throw std::invalid_argument(
                    "continuation sliding window must be positive and aligned to its block size");
            }
            if (spec.table_layout != KvTableLayout::kBoundedWindow) {
                throw std::invalid_argument("continuation-state flat groups require bounded-window tables");
            }
        }
    }
}

std::optional<std::size_t> KvCacheCoordinator::FindSchemaIndex(std::string_view group_id) const noexcept {
    const auto found = schema_index_by_group_id_.find(group_id);
    return found == schema_index_by_group_id_.end() ? std::nullopt : std::optional<std::size_t>{found->second};
}

std::vector<std::string> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                          std::uint32_t schema_index, std::int32_t group_block_size,
                                                          std::int32_t first_base) const {
    const std::int32_t fold = group_block_size / base_block_size_;
    // Hash namespace is deliberately the stable numeric schema index. External
    // group_id labels are an ABI key and must not alter cache-key identity.
    return MakeFoldedGroupKeys(content_hashes, schema_index, fold, first_base);
}

std::vector<std::vector<std::string>> KvCacheCoordinator::buildGroupKeys(
    std::span<const std::string> content_hashes) const {
    std::vector<std::vector<std::string>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].SchemaIndex(), schema_[i].block_size, /*first_base=*/0);
    }
    return group_keys;
}

KvCacheCoordinator::CoordinatorProbe KvCacheCoordinator::probeTierWithKeys(
    const BlockPool* shared_pool, std::span<const std::vector<std::string>> group_keys, std::int32_t num_base_pages,
    std::int32_t floor_tokens) const {
    CoordinatorProbe out;
    out.free_hit_blocks_by_pool = PoolDemand(pools_.size(), 0);
    out.per_group.resize(groups_.size());
    if (groups_.empty() || match_order_.empty()) {
        return out;
    }

    const std::int32_t boundary_tokens = SweepThenConverge(
        match_order_, groups_, num_base_pages * base_block_size_, history_alignment_tokens_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t block_size = schema_[i].block_size;
            const BlockPool& pool = shared_pool != nullptr ? *shared_pool : groups_[i].Pool();
            out.per_group[i] =
                groups_[i].Manager().Probe(pool, group_keys[i], floor_tokens / block_size, bound_tokens / block_size);
        },
        [&](std::size_t i) {
            const PrefixProbe& probe = out.per_group[i];
            return probe.LogicalEnd() * schema_[i].block_size;
        });

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (schema_[i].prefix_role != KvPrefixRole::kHistoryAnchor) {
            continue;
        }
        const std::int32_t block_size = schema_[i].block_size;
        PrefixProbe& probe = out.per_group[i];
        if (probe.LogicalEnd() * block_size <= boundary_tokens) {
            continue;
        }
        _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged boundary");
        const std::int32_t keep = boundary_tokens / block_size - probe.base_logical_page;
        _assert(keep >= 0, "history probe base is beyond the converged boundary");
        probe.hits.resize(static_cast<std::size_t>(keep));
    }

    if (has_continuation_state_ && boundary_tokens > floor_tokens) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (schema_[i].prefix_role != KvPrefixRole::kContinuationState) {
                continue;
            }
            const std::int32_t block_size = schema_[i].block_size;
            const BlockPool& pool = shared_pool != nullptr ? *shared_pool : groups_[i].Pool();
            std::optional<PrefixProbe> exact = groups_[i].Manager().ProbeExactBoundary(
                pool, group_keys[i], floor_tokens / block_size, boundary_tokens / block_size);
            if (!exact.has_value()) {
                CoordinatorProbe root;
                root.free_hit_blocks_by_pool = PoolDemand(pools_.size(), 0);
                root.per_group.resize(groups_.size());
                return root;
            }
            out.per_group[i] = std::move(*exact);
        }
    }

    out.num_common_tokens = boundary_tokens;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const PrefixProbe& probe = out.per_group[i];
        for (CachedBlockState state : probe.hits) {
            if (state == CachedBlockState::kFreeHit) {
                AddDemand(out.free_hit_blocks_by_pool, schema_[i].pool_index, 1);
            }
        }
    }
    return out;
}

CoordinatorMatch KvCacheCoordinator::acquireTierWithKeys(BlockPool* shared_pool,
                                                         std::span<const std::vector<std::string>> group_keys,
                                                         std::int32_t floor_tokens, CoordinatorProbe&& probe) {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_blocks = floor_tokens / schema_[i].block_size;
        BlockPool& pool = shared_pool != nullptr ? *shared_pool : groups_[i].Pool();
        out.per_group[i] =
            groups_[i].Manager().AcquireMatchedBlocks(pool, group_keys[i], floor_blocks, probe.per_group[i]);
    }
    return out;
}

KvCacheCoordinator::AdmissionProbe KvCacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    const std::int32_t num_base_pages = static_cast<std::int32_t>(content_hashes.size());
    AdmissionProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    out.device = probeTierWithKeys(/*shared_pool=*/nullptr, out.group_keys, num_base_pages,
                                   /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys(host_pool_, out.group_keys, num_base_pages, out.device.num_common_tokens);
    } else {
        out.host.free_hit_blocks_by_pool = PoolDemand(pools_.size(), 0);
        out.host.per_group.resize(groups_.size());
    }
    out.consumable_ = true;
    return out;
}

KvCacheCoordinator::AdmissionMatch KvCacheCoordinator::AcquirePrefix(AdmissionProbe probe) {
    if (!probe.Consume()) {
        throw std::logic_error("prefix admission probe was already consumed");
    }
    _assert(probe.group_keys.size() == groups_.size(), "prefix probe group-key shape mismatch");
    AdmissionMatch out;
    out.device = acquireTierWithKeys(/*shared_pool=*/nullptr, probe.group_keys,
                                     /*floor_tokens=*/0, std::move(probe.device));
    if (host_pool_ != nullptr) {
        out.host =
            acquireTierWithKeys(host_pool_, probe.group_keys, out.device.num_common_tokens, std::move(probe.host));
    }
    return out;
}

KvCacheCoordinator::PreparedPrefix KvCacheCoordinator::PreparePrefix(
    std::span<const std::string> content_hashes) const {
    if (host_pool_ != nullptr) {
        throw std::logic_error("PreparedPrefix is device-only; legacy host extension uses legacy admission");
    }
    AdmissionProbe probe = ProbePrefix(content_hashes);
    return PreparedPrefix{std::move(probe)};
}

KvCacheCoordinator::FreshDemandPlan KvCacheCoordinator::planFreshDemand(
    std::int32_t chunk_tokens, std::int32_t protected_tokens, detail::KvCacheGroupCounts* fresh_counts) const {
    if (chunk_tokens < 0 || protected_tokens < chunk_tokens) {
        throw std::invalid_argument("fresh allocation requires 0 <= chunk_tokens <= protected_tokens");
    }
    if (fresh_counts != nullptr && fresh_counts->Size() != groups_.size()) {
        throw std::invalid_argument("fresh allocation group-count output has the wrong shape");
    }
    const BlockTable empty;
    PoolDemand chunk_demand(pools_.size(), 0);
    PoolDemand protected_demand(pools_.size(), 0);
    PoolDemand reservation_demand(pools_.size(), 0);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t fresh_count = groups_[i].Manager().BlocksNeededFor(empty, chunk_tokens);
        const std::int32_t protected_count = protected_tokens == chunk_tokens
                                                 ? fresh_count
                                                 : groups_[i].Manager().BlocksNeededFor(empty, protected_tokens);
        if (fresh_counts != nullptr) {
            (*fresh_counts)[i] = fresh_count;
        }
        AddDemand(chunk_demand, schema_[i].pool_index, fresh_count);
        AddDemand(protected_demand, schema_[i].pool_index, protected_count);
        AddDemand(reservation_demand, schema_[i].pool_index, protected_count - fresh_count);
    }
    return FreshDemandPlan{
        std::move(chunk_demand),
        std::move(protected_demand),
        std::move(reservation_demand),
    };
}

KvCacheCoordinator::FreshDemandPlan KvCacheCoordinator::PlanFreshDemand(std::int32_t chunk_tokens,
                                                                        std::int32_t protected_tokens) const {
    return planFreshDemand(chunk_tokens, protected_tokens, nullptr);
}

KvCacheCoordinator::FreshAllocationPlan KvCacheCoordinator::PlanFreshAllocation(std::int32_t chunk_tokens,
                                                                                std::int32_t protected_tokens) const {
    detail::KvCacheGroupCounts fresh_counts(groups_.size(), 0);
    FreshDemandPlan demand = planFreshDemand(chunk_tokens, protected_tokens, &fresh_counts);
    return FreshAllocationPlan{chunk_tokens, std::move(fresh_counts), std::move(demand)};
}

std::optional<KvCacheCoordinator::CommittedPrefix> KvCacheCoordinator::CommitPreparedPrefix(
    PreparedPrefix prepared, FreshAllocationPlan allocation) {
    if (!prepared.probe_.Consumable()) {
        throw std::logic_error("prepared prefix was already consumed");
    }
    _assert(prepared.probe_.device.per_group.size() == groups_.size(), "PreparedPrefix group shape mismatch");
    _assert(allocation.fresh_counts_.Size() == groups_.size(), "fresh allocation group shape mismatch");
    _assert(allocation.demand_.chunk_demand_.Size() == pools_.size(), "fresh allocation pool shape mismatch");
    PoolDemand total_demand = std::move(allocation.demand_.chunk_demand_);
    total_demand.AddInPlace(prepared.ClaimDemand());
    for (PoolIndex i = 0; i < pools_.size(); ++i) {
        if (total_demand[i] > pools_[i]->NumFreeBlocks()) {
            return std::nullopt;
        }
    }

    std::vector<BlockTable> tables(groups_.size());
    std::vector<std::vector<BlockRef>> staged(groups_.size());
    std::vector<std::vector<BlockRef>> fresh(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::size_t prefix_count = prepared.probe_.device.per_group[i].hits.size();
        const std::int32_t fresh_count = allocation.fresh_counts_[i];
        staged[i].reserve(prefix_count + static_cast<std::size_t>(fresh_count));
        fresh[i].resize(static_cast<std::size_t>(fresh_count));
    }

    AdmissionMatch acquired = AcquirePrefix(std::move(prepared.probe_));
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        PrefixMatch& match = acquired.device.per_group[i];
        for (BlockRef& block : match.blocks) {
            staged[i].push_back(std::move(block));
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const bool ok = groups_[i].Pool().AcquireBlocksInto(fresh[i]);
        _assert(ok, "prepared prefix aggregate precheck must cover every fresh allocation");
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        tables[i].InitRange(acquired.device.per_group[i].base_logical_page, std::move(staged[i]));
        groups_[i].Manager().CommitAcquire(groups_[i].Pool(), tables[i], allocation.chunk_tokens_, fresh[i]);
    }
    return CommittedPrefix{
        .hit_tokens = acquired.device.num_common_tokens,
        .tables = std::move(tables),
    };
}

void KvCacheCoordinator::ClaimCommonPrefix(std::span<BlockTable> tables, CoordinatorMatch&& hit) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (hit.per_group.empty()) {
        _assert(hit.num_common_tokens == 0, "empty per_group with nonzero num_common_tokens");
        return;
    }
    _assert(hit.per_group.size() == groups_.size(), "hit/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ClaimHitBlocks(tables[i], std::move(hit.per_group[i]));
    }
}

std::vector<BlockTransfer> KvCacheCoordinator::LoadHostExtension(std::span<BlockTable> tables,
                                                                 CoordinatorMatch&& host) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    std::vector<BlockTransfer> transfers;
    if (host.per_group.empty()) {
        return transfers;
    }
    _assert(host.per_group.size() == groups_.size(), "host match/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().AppendHostExtension(groups_[i].Pool(), tables[i], std::move(host.per_group[i].blocks),
                                                 transfers);
    }
    return transfers;
}

PoolDemand KvCacheCoordinator::BlocksNeededByPool(std::span<const BlockTable> tables, std::int32_t num_tokens) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    PoolDemand demand(pools_.size(), 0);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        AddDemand(demand, schema_[i].pool_index, groups_[i].Manager().BlocksNeededFor(tables[i], num_tokens));
    }
    return demand;
}

PoolDemand KvCacheCoordinator::BlocksReclaimableByPool(std::span<const BlockTable> tables,
                                                       std::int32_t num_computed_tokens, bool count_uncached) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    PoolDemand demand(pools_.size(), 0);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        AddDemand(demand, schema_[i].pool_index,
                  groups_[i].Manager().BlocksReclaimableAt(tables[i], num_computed_tokens, count_uncached));
    }
    return demand;
}

PoolDemand KvCacheCoordinator::BlocksReleasedByFreeByPool(std::span<const BlockTable> tables) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    PoolDemand demand(pools_.size(), 0);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t released = static_cast<std::int32_t>(
            std::ranges::count_if(tables[i].Blocks(), [](const BlockRef& block) { return block && block.unique(); }));
        AddDemand(demand, schema_[i].pool_index, released);
    }
    return demand;
}

bool KvCacheCoordinator::Acquire(std::span<BlockTable> tables, std::int32_t num_tokens) {
    const PoolDemand demand = BlocksNeededByPool(tables, num_tokens);
    for (PoolIndex i = 0; i < pools_.size(); ++i) {
        if (demand[i] > pools_[i]->NumFreeBlocks()) {
            return false;
        }
    }

    std::vector<std::vector<BlockRef>> fresh(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t need = groups_[i].Manager().BlocksNeededFor(tables[i], num_tokens);
        fresh[i].resize(static_cast<std::size_t>(need));
        tables[i].ReserveLiveSize(tables[i].LiveSize() + need);
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const bool ok = groups_[i].Pool().AcquireBlocksInto(fresh[i]);
        _assert(ok, "component-wise prechecked AcquireBlocksInto must succeed");
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().CommitAcquire(groups_[i].Pool(), tables[i], num_tokens, fresh[i]);
    }
    return true;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                         std::int32_t first_slot, std::int32_t end_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;
    }
    _assert(first_slot >= 0, "CacheFullBlocks first_slot must be non-negative");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (schema_[i].prefix_role == KvPrefixRole::kNone) {
            continue;
        }
        const std::int32_t block_size = schema_[i].block_size;
        const std::int32_t fold = block_size / base_block_size_;
        std::vector<std::string> keys = keysForGroup(content_hashes, groups_[i].SchemaIndex(), block_size, first_slot);
        std::int32_t group_first_slot = (first_slot + fold - 1) / fold;
        std::span<const std::string> group_keys = keys;
        if (groups_[i].Manager().RegistersAlignedFinalPageOnly()) {
            if (end_tokens < 0 || end_tokens % block_size != 0 || keys.empty()) {
                continue;
            }
            const std::int32_t past_end_slot = group_first_slot + static_cast<std::int32_t>(keys.size());
            group_first_slot = past_end_slot - 1;
            group_keys = group_keys.last(1);
            _assert(past_end_slot == end_tokens / block_size,
                    "state registration range must end at the aligned boundary");
        }
        const std::int32_t live_base = tables[i].BaseLogicalPage();
        if (group_first_slot < live_base) {
            const std::int32_t skip =
                std::min(live_base - group_first_slot, static_cast<std::int32_t>(group_keys.size()));
            group_first_slot += skip;
            group_keys = group_keys.subspan(static_cast<std::size_t>(skip));
        }
        if (group_keys.empty()) {
            continue;
        }

        std::vector<std::pair<std::string, BlockRef>> newly_cached;
        groups_[i].Manager().CacheFullBlocks(groups_[i].Pool(), tables[i], group_keys, group_first_slot,
                                             host_pool_ != nullptr ? &newly_cached : nullptr);
        for (auto& [key, block] : newly_cached) {
            pending_stores_.push_back(StoreCandidate{std::move(key), std::move(block)});
        }
    }
}

bool KvCacheCoordinator::AdvanceRequest(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                        std::int32_t first_page_slot, std::int32_t num_new_tokens,
                                        std::int32_t num_computed_tokens) {
    CacheFullBlocks(tables, content_hashes, first_page_slot, num_computed_tokens);
    ReclaimExpired(tables, num_computed_tokens);
    return Acquire(tables, num_new_tokens);
}

void KvCacheCoordinator::PublishReadyBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                            std::span<const std::int32_t> previous_raw_ends,
                                            std::span<const std::int32_t> ready_raw_ends) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(previous_raw_ends.size() == groups_.size(), "previous progress/groups size mismatch");
    _assert(ready_raw_ends.size() == groups_.size(), "ready progress/groups size mismatch");

    const std::int64_t hash_raw_end =
        static_cast<std::int64_t>(content_hashes.size()) * static_cast<std::int64_t>(base_block_size_);

    std::size_t pending_store_capacity = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t previous_raw_end = previous_raw_ends[i];
        const std::int32_t ready_raw_end = ready_raw_ends[i];
        _assert(previous_raw_end >= 0 && ready_raw_end >= previous_raw_end,
                "ready group progress must be non-negative and monotonic");
        _assert(static_cast<std::int64_t>(ready_raw_end) <= hash_raw_end + base_block_size_ - 1,
                "ready group progress exceeds the available base hash chain");
        if (schema_[i].prefix_role == KvPrefixRole::kNone || ready_raw_end == previous_raw_end) {
            continue;
        }
        const std::int32_t block_size = schema_[i].block_size;
        std::size_t max_bindings = 0;
        if (groups_[i].Manager().RegistersAlignedFinalPageOnly()) {
            max_bindings = ready_raw_end % block_size == 0 ? 1 : 0;
        } else {
            const std::int32_t first_complete = previous_raw_end / block_size;
            const std::int32_t past_last_complete = ready_raw_end / block_size;
            max_bindings = static_cast<std::size_t>(std::max(0, past_last_complete - first_complete));
        }
        if (host_pool_ != nullptr) {
            if (max_bindings > pending_stores_.max_size() - pending_store_capacity) {
                throw std::length_error("ready publication exceeds pending-store max_size");
            }
            pending_store_capacity += max_bindings;
        }
    }
    if (pending_store_capacity > pending_stores_.max_size() - pending_stores_.size()) {
        throw std::length_error("ready publication exceeds pending-store max_size");
    }
    pending_stores_.reserve(pending_stores_.size() + pending_store_capacity);
    static_assert(std::is_nothrow_move_constructible_v<StoreCandidate>);

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t previous_raw_end = previous_raw_ends[i];
        const std::int32_t ready_raw_end = ready_raw_ends[i];
        if (schema_[i].prefix_role == KvPrefixRole::kNone || ready_raw_end == previous_raw_end) {
            continue;
        }

        const std::int32_t block_size = schema_[i].block_size;
        const std::int32_t fold = block_size / base_block_size_;
        std::int32_t first_group_slot = 0;
        std::int32_t first_base_slot = 0;
        std::int32_t num_base_hashes = 0;
        if (groups_[i].Manager().RegistersAlignedFinalPageOnly()) {
            if (ready_raw_end % block_size != 0) {
                continue;
            }
            first_group_slot = ready_raw_end / block_size - 1;
            if (first_group_slot < 0 || first_group_slot < previous_raw_end / block_size) {
                continue;
            }
            first_base_slot = first_group_slot * fold;
            num_base_hashes = fold;
        } else {
            const std::int32_t first_complete = previous_raw_end / block_size;
            const std::int32_t past_last_complete = ready_raw_end / block_size;
            if (past_last_complete <= first_complete) {
                continue;
            }
            first_group_slot = first_complete;
            first_base_slot = first_group_slot * fold;
            num_base_hashes = (past_last_complete - first_group_slot) * fold;
        }

        _assert(first_base_slot >= 0 && num_base_hashes > 0,
                "ready publication produced an invalid base hash interval");
        _assert(static_cast<std::size_t>(first_base_slot + num_base_hashes) <= content_hashes.size(),
                "ready publication requires unavailable base hashes");
        const std::span<const std::string> base_slice = content_hashes.subspan(
            static_cast<std::size_t>(first_base_slot), static_cast<std::size_t>(num_base_hashes));
        std::vector<std::string> keys = keysForGroup(base_slice, groups_[i].SchemaIndex(), block_size, first_base_slot);
        _assert(!keys.empty(), "aligned ready publication must produce a group key");

        const std::int32_t live_base = tables[i].BaseLogicalPage();
        std::size_t first_key_index = 0;
        if (first_group_slot < live_base) {
            const std::int32_t skip = std::min(live_base - first_group_slot, static_cast<std::int32_t>(keys.size()));
            first_group_slot += skip;
            first_key_index = static_cast<std::size_t>(skip);
        }
        if (first_key_index == keys.size()) {
            continue;
        }
        const std::size_t remaining_keys = keys.size() - first_key_index;
        _assert(first_group_slot + static_cast<std::int32_t>(remaining_keys) <= tables[i].LogicalEnd(),
                "ready publication exceeds the live block table");

        for (std::size_t relative = 0; relative < remaining_keys; ++relative) {
            const std::size_t key_index = first_key_index + relative;
            const std::int32_t logical_slot = first_group_slot + static_cast<std::int32_t>(relative);
            BlockRef& block = tables[i].AtLogical(logical_slot);
            if (!block || block->IsCached()) {
                continue;
            }
            if (host_pool_ != nullptr) {
                // Materialize the host-store record before mutating the pool.
                // Capacity was reserved above, so publishing the record after
                // CacheFullBlock is allocation-free.
                StoreCandidate store{.key = keys[key_index], .block = block};
                groups_[i].Pool().CacheFullBlock(block, keys[key_index]);
                pending_stores_.push_back(std::move(store));
            } else {
                groups_[i].Pool().CacheFullBlock(block, keys[key_index]);
            }
        }
    }
}

void KvCacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) noexcept {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ReclaimExpired(groups_[i].Pool(), tables[i], num_computed_tokens);
    }
}

void KvCacheCoordinator::RewindTail(std::span<BlockTable> tables, std::int32_t accepted_raw_end,
                                    std::int32_t retain_raw_end) noexcept {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().RewindTail(groups_[i].Pool(), tables[i], accepted_raw_end, retain_raw_end);
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) noexcept {
    if (tables.empty()) {
        return;
    }
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(groups_[i].Pool(), tables[i]);
    }
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPool& pool, BlockPool* host_pool) {
    std::vector<KvCacheGroupSchema> schema = NormalizeSchema(specs, /*pool_count=*/1);
    const BlockGeometry geometry = ComputeBlockGeometry(schema);
    std::vector<BlockPool*> pools{&pool};
    std::vector<CacheGroup> groups = BuildGroups(schema, pools, geometry.base);
    return KvCacheCoordinator{std::move(schema), std::move(groups), pool,
                              host_pool,         geometry.base,     geometry.history_alignment};
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, BlockPoolSet& pools, BlockPool* host_pool) {
    std::vector<KvCacheGroupSchema> schema = NormalizeSchema(specs, pools.Size());
    const BlockGeometry geometry = ComputeBlockGeometry(schema);
    std::vector<BlockPool*> pool_ptrs = CollectPools(pools);
    std::vector<CacheGroup> groups = BuildGroups(schema, pool_ptrs, geometry.base);
    return KvCacheCoordinator{std::move(schema), std::move(groups), pools,
                              host_pool,         geometry.base,     geometry.history_alignment};
}

}  // namespace tokenspeed
