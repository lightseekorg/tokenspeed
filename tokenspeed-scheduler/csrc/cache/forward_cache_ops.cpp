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

#include "cache/forward_cache_ops.h"

#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "resource/allocator/paged_cache_group.h"
#include "scheduler/types.h"

namespace tokenspeed {

namespace {

constexpr std::string_view kLegacyFlatPoolId = "default";

std::string_view EffectivePoolId(const PagedCacheGroupConfig& group) {
    return group.pool_id.empty() ? kLegacyFlatPoolId : std::string_view{group.pool_id};
}

bool HasExplicitFlatContract(const SchedulerConfig& config, const PagedCacheGroupConfig& group) {
    return !config.flat_block_pools.empty() || !group.pool_id.empty() ||
           group.prefix_role != PrefixRole::HistoryAnchor || group.table_layout != TableLayout::Absolute ||
           group.owner_mask != 0;
}

void ValidateGroupForFlatScheduler(const SchedulerConfig& config, const PagedCacheGroupConfig& group) {
    if (HasExplicitFlatContract(config, group)) {
        group.ValidateFlatBlockGeometry();
    }
}

KvCacheSpec MakeSpec(const SchedulerConfig& config, const PagedCacheGroupConfig& group, PoolIndex pool_index) {
    ValidateGroupForFlatScheduler(config, group);
    const std::int32_t block_size = group.block_size > 0 ? group.block_size : config.block_size;
    std::int32_t rows_per_page = block_size;
    std::int32_t entry_stride_tokens = 1;
    const std::int64_t configured_span =
        static_cast<std::int64_t>(group.rows_per_page) * static_cast<std::int64_t>(group.entry_stride_tokens);
    if (group.rows_per_page > 0 && group.entry_stride_tokens > 0 && configured_span == block_size) {
        rows_per_page = group.rows_per_page;
        entry_stride_tokens = group.entry_stride_tokens;
    }
    const KvPrefixRole prefix_role = [&] {
        if (config.disable_prefix_cache && !config.FlatStreamingSinkEnabled() &&
            group.prefix_role == PrefixRole::ContinuationState) {
            return KvPrefixRole::kNone;
        }
        switch (group.prefix_role) {
            case PrefixRole::HistoryAnchor:
                return KvPrefixRole::kHistoryAnchor;
            case PrefixRole::ContinuationState:
                return KvPrefixRole::kContinuationState;
            case PrefixRole::None:
                return KvPrefixRole::kNone;
        }
        throw std::logic_error("unreachable PagedCacheGroupConfig prefix role");
    }();
    const KvTableLayout table_layout =
        group.table_layout == TableLayout::BoundedWindow ? KvTableLayout::kBoundedWindow : KvTableLayout::kAbsolute;
    if (group.family == PagedCacheGroupFamily::State &&
        group.retention != PagedCacheGroupConfig::Retention::SlidingWindow) {
        return KvCacheSpec{
            .kind = AttnKind::kMambaState,
            .block_size = block_size,
            .sliding_window = 0,
            .pool_index = pool_index,
            .prefix_role = prefix_role,
            .table_layout = table_layout,
            .owner_mask = group.owner_mask,
            .group_id = group.group_id,
            .rows_per_page = rows_per_page,
            .entry_stride_tokens = entry_stride_tokens,
        };
    }
    const bool is_swa = group.retention == PagedCacheGroupConfig::Retention::SlidingWindow;
    return KvCacheSpec{
        .kind = is_swa ? AttnKind::kSlidingWindow : AttnKind::kFull,
        .block_size = block_size,
        .sliding_window = is_swa ? group.sliding_window_tokens.value_or(0) : 0,
        .pool_index = pool_index,
        .prefix_role = prefix_role,
        .table_layout = table_layout,
        .owner_mask = group.owner_mask,
        .group_id = group.group_id,
        .rows_per_page = rows_per_page,
        .entry_stride_tokens = entry_stride_tokens,
    };
}

}  // namespace

bool PrefillFirstChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables, CoordinatorMatch&& hit,
                       std::int32_t num_new_tokens) {
    coordinator.ClaimCommonPrefix(tables, std::move(hit));
    return coordinator.Acquire(tables, num_new_tokens);
}

std::vector<BlockTransfer> LoadHostExtension(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                             CoordinatorMatch&& host) {
    return coordinator.LoadHostExtension(tables, std::move(host));
}

bool PrefillChunk(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                  std::span<const std::string> content_hashes, std::int32_t num_tokens,
                  std::int32_t num_computed_tokens) {
    return DecodeStep(coordinator, tables, content_hashes, /*first_page_slot=*/0, num_tokens, num_computed_tokens);
}

bool DecodeStep(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                std::span<const std::string> content_hashes, std::int32_t first_page_slot, std::int32_t num_tokens,
                std::int32_t num_computed_tokens) {
    // CacheFullBlocks before ReclaimExpired: registration skips null holes, so
    // the reverse order would lose the punched pages' hashes forever.
    // ReclaimExpired before Acquire: the slide's freed pages fund this chunk
    // (admission gates credit them via BlocksReclaimableAt in lockstep).
    // num_computed_tokens is the chunk end: state groups register only its aligned final page.
    coordinator.CacheFullBlocks(tables, content_hashes, first_page_slot, num_computed_tokens);
    coordinator.ReclaimExpired(tables, num_computed_tokens);
    return coordinator.Acquire(tables, num_tokens);
}

bool FinalizePrefillAndReserveDecode(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables,
                                     std::span<const std::string> content_hashes, std::int32_t reserve_tokens,
                                     std::int32_t num_computed_tokens) {
    return PrefillChunk(coordinator, tables, content_hashes, reserve_tokens, num_computed_tokens);
}

std::vector<FlatBlockPoolConfig> MakeFlatBlockPoolConfigs(const SchedulerConfig& config) {
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        ValidateGroupForFlatScheduler(config, group);
    }
    if (!config.flat_block_pools.empty()) {
        if (config.device_allocator.total_pages != 0) {
            throw std::invalid_argument("explicit flat_block_pools require device_allocator.total_pages=0");
        }
        std::unordered_map<std::string, std::int32_t> totals;
        totals.reserve(config.flat_block_pools.size());
        for (const FlatBlockPoolConfig& pool : config.flat_block_pools) {
            if (pool.pool_id.empty() || pool.bytes_per_block <= 0) {
                throw std::invalid_argument("explicit flat pools require non-empty ids and positive bytes_per_block");
            }
            if (!totals.emplace(pool.pool_id, pool.total_blocks).second) {
                throw std::invalid_argument("duplicate explicit flat block pool_id '" + pool.pool_id + "'");
            }
        }
        for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
            const auto it = totals.find(group.pool_id);
            if (group.pool_id.empty() || it == totals.end()) {
                throw std::invalid_argument("explicit flat group '" + group.group_id +
                                            "' must bind a configured pool_id");
            }
            if (group.total_pages != it->second) {
                throw std::invalid_argument("explicit flat group '" + group.group_id +
                                            "' total_pages disagrees with its pool");
            }
        }
        return config.flat_block_pools;
    }
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        if (EffectivePoolId(group) != kLegacyFlatPoolId) {
            throw std::invalid_argument("flat group '" + group.group_id +
                                        "' binds a pool but flat_block_pools is empty");
        }
    }
    return {FlatBlockPoolConfig{
        .pool_id = std::string{kLegacyFlatPoolId},
        .total_blocks = config.device_allocator.total_pages,
        .bytes_per_block = 0,
    }};
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config) {
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        specs.push_back(MakeSpec(config, group, /*pool_index=*/0));
    }
    return specs;
}

std::vector<KvCacheSpec> MakeSpecsFromConfig(const SchedulerConfig& config, const BlockPoolSet& pools) {
    std::vector<KvCacheSpec> specs;
    specs.reserve(config.paged_cache_groups.size());
    for (const PagedCacheGroupConfig& group : config.paged_cache_groups) {
        specs.push_back(MakeSpec(config, group, pools.IndexOf(EffectivePoolId(group))));
    }
    return specs;
}

void FreeRequest(KvCacheCoordinator& coordinator, std::vector<BlockTable>& tables) noexcept {
    if (tables.empty()) {
        return;  // request never got tables, or a failure path already released them
    }
    coordinator.Free(tables);
}

}  // namespace tokenspeed
