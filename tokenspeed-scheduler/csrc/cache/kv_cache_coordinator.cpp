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
#include <optional>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "utils.h"

namespace tokenspeed {

KvCacheCoordinator::KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                                       BlockPool* host_pool)
    : groups_{std::move(groups)}, pool_{pool}, host_pool_{host_pool}, cache_block_tokens_{cache_block_tokens} {
    _assert(cache_block_tokens_ > 0, "coordinator needs positive cache_block_tokens");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Manager().CacheBlockTokens() == cache_block_tokens_,
                "every cache manager must use the domain cache_block_tokens");
        _assert(groups_[i].Manager().CacheBlocksPerLcmBlock() == groups_[i].Spec().cache_blocks_per_lcm_block,
                "cache manager packing must match its group spec");
        if (groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Manager().MatchIsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
}

std::vector<CacheKey> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                       GroupId group_id) const {
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size());
    for (const std::string& content_hash : content_hashes) {
        keys.push_back(CacheKey{.group_id = group_id, .content_hash = content_hash});
    }
    return keys;
}

namespace {

// Shared match skeleton: one ordered sweep (closed groups first), then re-match any window
// group left above the settled bound -- with 2+ window groups a later group can shrink the
// bound UNDER an earlier one's boundary-dependent match. A re-matched group lands at or
// under the current bound and only a further bound drop can lift it back above, so
// re-matches are finite; the result is the greatest boundary every group supports.
//
// Bounds align down to the shared logical CacheBlock granularity P.
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

std::vector<std::vector<CacheKey>> KvCacheCoordinator::buildGroupKeys(
    std::span<const std::string> content_hashes) const {
    std::vector<std::vector<CacheKey>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].Id());
    }
    return group_keys;
}

// The one tier matcher: slots below floor_tokens are assumed valid in a lower tier; per_group
// blocks are relative to the floor, num_common_tokens is the absolute converged boundary (in
// TOKENS). num_cache_blocks = content_hashes.size(); every group has one key per
// logical CacheBlock.
KvCacheCoordinator::PrefixProbe::Tier KvCacheCoordinator::probeTierWithKeys(
    const BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys, std::int32_t num_cache_blocks,
    std::int32_t floor_tokens) const {
    PrefixProbe::Tier out;
    out.per_group.resize(groups_.size());
    if (groups_.empty()) {
        return out;
    }
    const std::int32_t boundary_tokens = SweepThenConverge(
        match_order_, groups_, num_cache_blocks * cache_block_tokens_, cache_block_tokens_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            out.per_group[i] = groups_[i].Manager().Probe(pool, group_keys[i], floor_tokens / cache_block_tokens_,
                                                          bound_tokens / cache_block_tokens_);
        },
        [&](std::size_t i) {
            return (floor_tokens / cache_block_tokens_ + static_cast<std::int32_t>(out.per_group[i].hits.size())) *
                   cache_block_tokens_;
        });

    // Truncate closed probes to the converged boundary.
    // Non-closed groups were re-probed against the settled bound and are already at or below it.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        GroupPrefixProbe& probe = out.per_group[i];
        const std::int32_t floor_blocks = floor_tokens / cache_block_tokens_;
        if ((floor_blocks + static_cast<std::int32_t>(probe.hits.size())) * cache_block_tokens_ > boundary_tokens) {
            _assert(groups_[i].Manager().MatchIsPrefixClosed(), "window group left above the converged boundary");
            probe.hits.resize(static_cast<std::size_t>(boundary_tokens / cache_block_tokens_ - floor_blocks));
        }
    }
    out.num_common_tokens = boundary_tokens;
    return out;
}

CoordinatorMatch KvCacheCoordinator::acquireTierWithKeys(BlockPool& pool,
                                                         std::span<const std::vector<CacheKey>> group_keys,
                                                         std::int32_t floor_tokens, PrefixProbe::Tier&& probe) {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_blocks = floor_tokens / cache_block_tokens_;
        out.per_group[i] = groups_[i].Manager().AcquireMatchedBlocks(pool, group_keys[i], floor_blocks,
                                                                     probe.per_group[i], next_recency_);
    }
    return out;
}

KvCacheCoordinator::PrefixProbe KvCacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / cache_block_tokens_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_cache_blocks = static_cast<std::int32_t>(content_hashes.size());
    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    out.device = probeTierWithKeys(pool_, out.group_keys, num_cache_blocks, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys(*host_pool_, out.group_keys, num_cache_blocks,
                                     /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::AcquiredPrefix KvCacheCoordinator::acquirePrefix(PrefixProbe&& probe) {
    AcquiredPrefix out;
    out.device = acquireTierWithKeys(pool_, probe.group_keys, /*floor_tokens=*/0, std::move(probe.device));
    if (host_pool_ != nullptr) {
        out.host =
            acquireTierWithKeys(*host_pool_, probe.group_keys, out.device.num_common_tokens, std::move(probe.host));
    }
    return out;
}

std::int32_t KvCacheCoordinator::BlocksNeededFor(std::span<const BlockTable> tables, std::int32_t num_tokens) const {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    std::int32_t total_needed = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        total_needed += groups_[i].Manager().BlocksNeededFor(tables[i], num_tokens);
    }
    return total_needed;
}

std::int32_t KvCacheCoordinator::BlocksNeededFor(std::int32_t num_tokens) const {
    const BlockTable fresh;
    std::int32_t total_needed = 0;
    for (const CacheGroup& group : groups_) {
        total_needed += group.Manager().BlocksNeededFor(fresh, num_tokens);
    }
    return total_needed;
}

std::int32_t KvCacheCoordinator::NumAvailableLcmBlocks() const {
    std::int32_t available = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
        const std::optional<GroupId> group_id = pool_.BoundGroup(parent_id);
        if (!group_id || groups_[*group_id].Manager().ParentIsFullyEvictable(pool_, parent_id)) {
            ++available;
        }
    }
    return available;
}

void KvCacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                         std::int32_t first_slot, std::int32_t end_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        cacheFullBlocksForGroup(i, tables[i], content_hashes, first_slot, end_tokens);
    }
}

void KvCacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                 std::span<const std::string> content_hashes, std::int32_t first_slot,
                                                 std::int32_t end_tokens) {
    std::vector<CacheKey> keys = keysForGroup(content_hashes, groups_[group_index].Id());
    std::int32_t group_first_slot = first_slot;
    std::span<const CacheKey> group_keys = keys;
    if (groups_[group_index].Manager().RegistersAlignedFinalPageOnly()) {
        if (end_tokens < 0 || end_tokens % cache_block_tokens_ != 0 || keys.empty()) {
            return;
        }
        const std::int32_t past_end_slot = group_first_slot + static_cast<std::int32_t>(keys.size());
        group_first_slot = past_end_slot - 1;
        group_keys = group_keys.last(1);
        _assert(past_end_slot == end_tokens / cache_block_tokens_,
                "state registration range must end at the aligned boundary");
    }
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    groups_[group_index].Manager().CacheFullBlocks(pool_, table, group_keys, next_recency_, group_first_slot,
                                                   host_pool_ != nullptr ? &newly_cached : nullptr);
    for (auto& [key, block_ref] : newly_cached) {
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
            .block_ref = std::move(block_ref),
        });
    }
}

void KvCacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ReclaimExpired(pool_, tables[i], num_computed_tokens);
    }
}

void KvCacheCoordinator::ConsumeAvailable(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ConsumeAvailable(tables[i], num_tokens);
    }
}

void KvCacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().Free(tables[i]);
    }
}

bool KvCacheCoordinator::ContainsHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Manager().ContainsCachedBlock(*host_pool_, key);
}

bool KvCacheCoordinator::IsHostCachedBlock(CacheBlockLocation location) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    return std::ranges::any_of(
        groups_, [&](const CacheGroup& group) { return group.Manager().ContainsCachedBlock(*host_pool_, location); });
}

std::int32_t KvCacheCoordinator::NumHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Manager().NumCachedBlocks(*host_pool_);
    }
    return count;
}

std::int32_t KvCacheCoordinator::NumPinnedHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Manager().NumPinnedCachedBlocks(*host_pool_);
    }
    return count;
}

void KvCacheCoordinator::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < groups_.size(), "CacheHostBlock group id out of range");
    groups_[key.group_id].Manager().CacheBlock(*host_pool_, block_ref, key, next_recency_);
}

KvCacheCoordinator MakeCoordinator(std::span<const KvCacheSpec> specs, std::int32_t cache_block_tokens, BlockPool& pool,
                                   BlockPool* host_pool) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    _assert(cache_block_tokens > 0, "cache_block_tokens must be > 0");
    _assert(specs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "number of cache groups exceeds int32 range");
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const KvCacheSpec& spec = specs[i];
        const GroupId group_id = static_cast<GroupId>(i);
        _assert(spec.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        std::unique_ptr<KvCacheManager> manager;
        if (spec.kind == AttnKind::kFull) {
            manager = std::make_unique<FullAttnManager>(cache_block_tokens, spec.cache_blocks_per_lcm_block, group_id);
        } else if (spec.kind == AttnKind::kMambaState) {
            manager =
                std::make_unique<MambaStateManager>(cache_block_tokens, spec.cache_blocks_per_lcm_block, group_id);
        } else {
            manager = std::make_unique<SwaManager>(cache_block_tokens, spec.cache_blocks_per_lcm_block,
                                                   spec.sliding_window, group_id);
        }
        groups.emplace_back(spec, std::move(manager));
    }
    return KvCacheCoordinator{std::move(groups), cache_block_tokens, pool, host_pool};
}

}  // namespace tokenspeed
