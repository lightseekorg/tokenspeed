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
#include <tuple>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "utils.h"

namespace tokenspeed {

namespace {

std::int32_t EffectiveCacheBlockTokens(const KvCacheSpec& spec, std::int32_t coordinator_cache_block_tokens) {
    return spec.cache_block_tokens > 0 ? spec.cache_block_tokens : coordinator_cache_block_tokens;
}

}  // namespace

KvCacheCoordinator::KvCacheCoordinator(std::vector<CacheGroup> groups, std::int32_t cache_block_tokens, BlockPool& pool,
                                       BlockPool* host_pool)
    : groups_{std::move(groups)}, pool_{pool}, host_pool_{host_pool}, cache_block_tokens_{cache_block_tokens} {
    _assert(cache_block_tokens_ > 0, "coordinator needs positive cache_block_tokens");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Id() == static_cast<GroupId>(i), "cache manager group id must equal its group index");
        const std::int32_t group_cache_block_tokens = EffectiveCacheBlockTokens(groups_[i].Spec(), cache_block_tokens_);
        _assert(group_cache_block_tokens > 0 && cache_block_tokens_ % group_cache_block_tokens == 0,
                "manager cache block tokens must divide the coordinator domain");
        _assert(groups_[i].Manager().CacheBlockTokens() == group_cache_block_tokens,
                "cache manager block tokens must match its group spec");
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

bool KvCacheCoordinator::HasMambaStateGroup() const {
    return std::ranges::any_of(groups_,
                               [](const CacheGroup& group) { return group.Spec().kind == AttnKind::kMambaState; });
}

bool KvCacheCoordinator::ClearDeviceCache() {
    std::vector<std::pair<GroupId, CacheBlockLocation>> cached_locations;
    for (const CacheGroup& group : groups_) {
        const KvCacheManager& manager = group.Manager();
        std::vector<CacheBlockLocation> group_locations = manager.EvictableBlockLocations(pool_);
        if (static_cast<std::int32_t>(group_locations.size()) != manager.NumCachedBlocks(pool_)) {
            return false;
        }
        for (CacheBlockLocation location : group_locations) {
            cached_locations.emplace_back(group.Id(), location);
        }
    }

    pending_stores_.clear();
    for (const auto& [group_id, location] : cached_locations) {
        _assert(evictCachedBlock(group_id, location), "clearable Device cache entry disappeared");
    }
    return true;
}

std::vector<CacheKey> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                       GroupId group_id) const {
    _assert(group_id < groups_.size(), "cache key group id out of range");
    const std::int32_t group_cache_block_tokens = groups_[group_id].Manager().CacheBlockTokens();
    const std::int32_t cache_blocks_per_hash = cache_block_tokens_ / group_cache_block_tokens;
    _assert(content_hashes.size() <=
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(cache_blocks_per_hash),
            "expanded cache key count exceeds size_t range");
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size() * static_cast<std::size_t>(cache_blocks_per_hash));
    for (const std::string& content_hash : content_hashes) {
        for (std::int32_t offset = 0; offset < cache_blocks_per_hash; ++offset) {
            keys.push_back(CacheKey{
                .group_id = group_id,
                .content_hash = content_hash,
                .cache_block_offset = offset,
            });
        }
    }
    return keys;
}

namespace {

struct ConvergedBoundary {
    std::int32_t common_tokens{0};
    std::int32_t prefix_closed_tokens{0};
};

// Shared match skeleton: one ordered sweep (closed groups first), then re-match any window
// group left above the settled bound -- with 2+ window groups a later group can shrink the
// bound UNDER an earlier one's boundary-dependent match. A re-matched group lands at or
// under the current bound and only a further bound drop can lift it back above, so
// re-matches are finite; the result is the greatest boundary every group supports.
//
// Bounds align down to the shared logical CacheBlock granularity P.
template <typename MatchGroup, typename ExtentTokens>
ConvergedBoundary SweepThenConverge(std::span<const std::size_t> order, const std::vector<CacheGroup>& groups,
                                    std::int32_t bound_tokens, std::int32_t align_tokens, const MatchGroup& match,
                                    const ExtentTokens& extent) {
    const auto align_down = [align_tokens](std::int32_t tokens) { return tokens - tokens % align_tokens; };
    bound_tokens = align_down(bound_tokens);
    std::int32_t prefix_closed_tokens = 0;
    for (std::size_t i : order) {
        match(i, bound_tokens);
        bound_tokens = std::min(bound_tokens, align_down(extent(i)));
        if (groups[i].Manager().MatchIsPrefixClosed()) {
            prefix_closed_tokens = bound_tokens;
        }
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
    return {
        .common_tokens = bound_tokens,
        .prefix_closed_tokens = prefix_closed_tokens,
    };
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
// TOKENS). num_cache_blocks = content_hashes.size() in coordinator P-pages;
// each group key vector is expanded to that manager's CacheBlock granularity.
KvCacheCoordinator::PrefixProbe::Tier KvCacheCoordinator::probeTierWithKeys(
    const BlockPool& pool, std::span<const std::vector<CacheKey>> group_keys, std::span<const std::size_t> match_order,
    std::int32_t num_cache_blocks, std::int32_t floor_tokens) const {
    PrefixProbe::Tier out;
    out.per_group.resize(groups_.size());
    if (match_order.empty()) {
        return out;
    }
    const ConvergedBoundary boundary = SweepThenConverge(
        match_order, groups_, num_cache_blocks * cache_block_tokens_, cache_block_tokens_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_cache_block_tokens = groups_[i].Manager().CacheBlockTokens();
            out.per_group[i] = groups_[i].Manager().Probe(pool, group_keys[i], floor_tokens / group_cache_block_tokens,
                                                          bound_tokens / group_cache_block_tokens);
        },
        [&](std::size_t i) {
            return floor_tokens +
                   static_cast<std::int32_t>(out.per_group[i].hits.size()) * groups_[i].Manager().CacheBlockTokens();
        });

    // A manager can find a q-aligned resume point above the final P-aligned
    // boundary. Only acquire the CacheBlocks covered by the shared boundary.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        GroupPrefixProbe& probe = out.per_group[i];
        const std::int32_t group_cache_block_tokens = groups_[i].Manager().CacheBlockTokens();
        const std::int32_t covered_cache_blocks = (boundary.common_tokens - floor_tokens) / group_cache_block_tokens;
        if (static_cast<std::int32_t>(probe.hits.size()) > covered_cache_blocks) {
            probe.hits.resize(static_cast<std::size_t>(covered_cache_blocks));
        }
    }
    out.num_common_tokens = boundary.common_tokens;
    out.prefix_closed_tokens = boundary.prefix_closed_tokens;
    return out;
}

CoordinatorMatch KvCacheCoordinator::acquireTierWithKeys(BlockPool& pool,
                                                         std::span<const std::vector<CacheKey>> group_keys,
                                                         std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                         std::uint64_t access_epoch) {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_cache_blocks = floor_tokens / groups_[i].Manager().CacheBlockTokens();
        out.per_group[i] = groups_[i].Manager().AcquireMatchedBlocks(pool, group_keys[i], floor_cache_blocks,
                                                                     probe.per_group[i], access_epoch);
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
    out.device = probeTierWithKeys(pool_, out.group_keys, match_order_, num_cache_blocks, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys(*host_pool_, out.group_keys, match_order_, num_cache_blocks,
                                     /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::PrefixProbe KvCacheCoordinator::ProbeDecodeDestinationPrefix(
    std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / cache_block_tokens_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_cache_blocks = static_cast<std::int32_t>(content_hashes.size());
    std::vector<std::size_t> history_match_order;
    history_match_order.reserve(match_order_.size());
    for (std::size_t group_index : match_order_) {
        if (groups_[group_index].Spec().kind != AttnKind::kMambaState) {
            history_match_order.push_back(group_index);
        }
    }

    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    const auto probe_tier = [&](const BlockPool& pool, std::int32_t floor_tokens) {
        PrefixProbe::Tier tier =
            probeTierWithKeys(pool, out.group_keys, history_match_order, num_cache_blocks, floor_tokens);
        const std::int64_t covered_tokens =
            static_cast<std::int64_t>(tier.num_common_tokens) - static_cast<std::int64_t>(floor_tokens);
        _assert(covered_tokens >= 0, "decode destination state coverage is negative");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].Spec().kind == AttnKind::kMambaState) {
                const std::int64_t num_holes = covered_tokens / groups_[i].Manager().CacheBlockTokens();
                _assert(num_holes <= static_cast<std::int64_t>(out.group_keys[i].size()),
                        "decode destination state hole count is outside the probed range");
                const std::size_t hole_count = static_cast<std::size_t>(num_holes);
                tier.per_group[i].hits.resize(hole_count);
            }
        }
        return tier;
    };
    out.device = probe_tier(pool_, /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probe_tier(*host_pool_, /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

KvCacheCoordinator::AcquiredPrefix KvCacheCoordinator::acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch) {
    AcquiredPrefix out;
    out.device =
        acquireTierWithKeys(pool_, probe.group_keys, /*floor_tokens=*/0, std::move(probe.device), access_epoch);
    if (host_pool_ != nullptr) {
        out.host = acquireTierWithKeys(*host_pool_, probe.group_keys, out.device.num_common_tokens,
                                       std::move(probe.host), access_epoch);
    }
    return out;
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
                                         std::uint64_t access_epoch, std::int32_t first_slot,
                                         CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheKey> keys = keysForGroup(content_hashes, groups_[i].Id());
        const std::int32_t cache_blocks_per_hash = cache_block_tokens_ / groups_[i].Manager().CacheBlockTokens();
        cacheFullBlocksForGroup(i, tables[i], keys, first_slot * cache_blocks_per_hash, access_epoch, boundary_kind);
    }
}

void KvCacheCoordinator::CacheCompletedBlocks(std::span<BlockTable> tables, std::span<const std::string> page_hashes,
                                              std::uint64_t access_epoch, std::int32_t first_new_page,
                                              std::int32_t num_computed_tokens, CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(first_new_page >= 0 && static_cast<std::size_t>(first_new_page) < page_hashes.size(),
            "completed page range must be non-empty");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand demand{
            .table = &tables[i],
            .page_hashes = page_hashes,
            .new_page_hash_begin = first_new_page,
            .completed_boundary_kind = boundary_kind,
            .num_computed_tokens = num_computed_tokens,
        };
        cacheCompletedBlocksForGroup(i, demand, access_epoch);
    }
}

void KvCacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                                 std::span<const CacheKey> keys, std::int32_t first_cache_block,
                                                 std::uint64_t access_epoch, CacheBoundaryKind boundary_kind) {
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    groups_[group_index].Manager().CacheFullBlocks(
        pool_, table, keys, access_epoch, first_cache_block, boundary_kind,
        host_pool_ != nullptr || cache_mutation_sink_ ? &newly_cached : nullptr);
    for (auto& [key, block_ref] : newly_cached) {
        if (cache_mutation_sink_) {
            cache_mutation_sink_(key, CacheMutation::kStored);
        }
        if (host_pool_ == nullptr) {
            continue;
        }
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
        });
    }
}

CacheBlockRef KvCacheCoordinator::AcquireDeviceCachedBlock(const CacheKey& key) const {
    if (key.group_id >= groups_.size()) {
        return {};
    }
    return groups_[key.group_id].Manager().AcquireCachedBlock(pool_, key);
}

CacheBlockRef KvCacheCoordinator::AcquireHostBlockForStore(GroupId group_id) {
    _assert(host_pool_ != nullptr, "AcquireHostBlockForStore requires a host pool");
    _assert(group_id < groups_.size(), "Host store group id out of range");
    KvCacheManager& target = groups_[group_id].Manager();
    const std::int32_t packing = target.CacheBlocksPerLcmBlock();
    if (CacheBlockRef block_ref = host_pool_->AcquireBlock(group_id, packing)) {
        return block_ref;
    }

    const auto value = [&](GroupId candidate_group, CacheBlockLocation location) {
        const auto metadata = groups_[candidate_group].Manager().CachedBlockMetadataFor(*host_pool_, location);
        _assert(metadata.has_value(), "evictable Host block has no cache metadata");
        return std::tuple{metadata->was_acquired,
                          metadata->last_access_epoch,
                          candidate_group,
                          location.lcm_block_id,
                          location.slot_index};
    };
    using HostCacheValue = decltype(value(group_id, CacheBlockLocation{}));

    // Reusing one child of an already-bound parent destroys less cache than
    // rebinding a complete parent from another group.
    std::vector<CacheBlockLocation> local_victims = target.EvictableBlockLocations(*host_pool_);
    if (!local_victims.empty()) {
        const auto victim = std::ranges::min_element(
            local_victims, {}, [&](CacheBlockLocation location) { return value(group_id, location); });
        _assert(target.EvictCachedBlock(*host_pool_, *victim).has_value(), "selected Host child is not evictable");
        CacheBlockRef block_ref = host_pool_->AcquireBlock(group_id, packing);
        _assert(static_cast<bool>(block_ref), "evicting a same-group Host child did not free a placement");
        return block_ref;
    }

    std::optional<std::int32_t> victim_parent;
    std::optional<HostCacheValue> victim_value;
    for (std::int32_t parent_id = 1; parent_id <= host_pool_->NumLcmBlocks(); ++parent_id) {
        const std::optional<GroupId> bound_group = host_pool_->BoundGroup(parent_id);
        if (!bound_group || !groups_[*bound_group].Manager().ParentIsFullyEvictable(*host_pool_, parent_id)) {
            continue;
        }
        std::optional<HostCacheValue> parent_value;
        for (std::int32_t slot = 0; slot < groups_[*bound_group].Manager().CacheBlocksPerLcmBlock(); ++slot) {
            const CacheBlockLocation location{.lcm_block_id = parent_id, .slot_index = slot};
            if (!host_pool_->IsOccupied(location)) {
                continue;
            }
            const auto child_value = value(*bound_group, location);
            parent_value = parent_value ? std::max(*parent_value, child_value) : child_value;
        }
        _assert(parent_value.has_value(), "evictable Host parent has no children");
        if (!victim_value || *parent_value < *victim_value) {
            victim_parent = parent_id;
            victim_value = *parent_value;
        }
    }
    if (!victim_parent) {
        return {};
    }

    const GroupId bound_group = *host_pool_->BoundGroup(*victim_parent);
    KvCacheManager& manager = groups_[bound_group].Manager();
    for (std::int32_t slot = 0; slot < manager.CacheBlocksPerLcmBlock(); ++slot) {
        const CacheBlockLocation location{.lcm_block_id = *victim_parent, .slot_index = slot};
        if (host_pool_->IsOccupied(location)) {
            _assert(manager.EvictCachedBlock(*host_pool_, location).has_value(),
                    "selected Host parent changed before eviction");
        }
    }
    CacheBlockRef block_ref = host_pool_->AcquireBlock(group_id, packing);
    _assert(static_cast<bool>(block_ref), "evicting a Host parent did not free a placement");
    return block_ref;
}

bool KvCacheCoordinator::evictCachedBlock(GroupId group_id, CacheBlockLocation location) {
    std::optional<CacheKey> removed = groups_[group_id].Manager().EvictCachedBlock(pool_, location);
    if (!removed) {
        return false;
    }
    if (cache_mutation_sink_) {
        cache_mutation_sink_(*removed, CacheMutation::kRemoved);
    }
    return true;
}

void KvCacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                                      std::uint64_t access_epoch) {
    const KvCacheManager& manager = groups_[group_index].Manager();
    const std::int32_t cache_blocks_per_hash = cache_block_tokens_ / manager.CacheBlockTokens();
    if (manager.MatchIsPrefixClosed()) {
        std::vector<CacheKey> keys =
            keysForGroup(demand.page_hashes.subspan(static_cast<std::size_t>(demand.new_page_hash_begin)),
                         groups_[group_index].Id());
        cacheFullBlocksForGroup(group_index, *demand.table, keys, demand.new_page_hash_begin * cache_blocks_per_hash,
                                access_epoch, *demand.completed_boundary_kind);
        return;
    }
    if (demand.num_computed_tokens < 0) {
        return;
    }
    // Mamba can publish only a state checkpoint that the kernel materialized
    // exactly at this boundary. SWA pages are ordinary KV, so an unaligned
    // endpoint can still publish its trailing complete-page boundary.
    if (groups_[group_index].Spec().kind == AttnKind::kMambaState &&
        demand.num_computed_tokens % cache_block_tokens_ != 0) {
        return;
    }

    const std::int32_t boundary_cache_block =
        static_cast<std::int32_t>(demand.page_hashes.size()) * cache_blocks_per_hash;
    const std::int32_t lookback = std::min(manager.BoundaryLookbackBlocks(), boundary_cache_block);
    if (lookback == 0) {
        return;
    }
    const std::int32_t first_cache_block = boundary_cache_block - lookback;
    std::vector<CacheKey> keys = keysForGroup(demand.page_hashes, groups_[group_index].Id());
    cacheFullBlocksForGroup(group_index, *demand.table,
                            std::span<const CacheKey>{keys}.subspan(static_cast<std::size_t>(first_cache_block)),
                            first_cache_block, access_epoch, *demand.completed_boundary_kind);
}

void KvCacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ReclaimExpired(pool_, tables[i], num_computed_tokens);
    }
}

void KvCacheCoordinator::ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().ConsumeReservedTokens(tables[i], num_tokens);
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
    groups_[key.group_id].Manager().RegisterCachedBlock(*host_pool_, block_ref, key, ++next_access_epoch_);
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
        const std::int32_t group_cache_block_tokens = EffectiveCacheBlockTokens(spec, cache_block_tokens);
        _assert(group_cache_block_tokens > 0 && cache_block_tokens % group_cache_block_tokens == 0,
                "group cache block tokens must divide the coordinator domain");
        std::unique_ptr<KvCacheManager> manager;
        if (spec.kind == AttnKind::kFull) {
            manager =
                std::make_unique<FullAttnManager>(group_cache_block_tokens, spec.cache_blocks_per_lcm_block, group_id);
        } else if (spec.kind == AttnKind::kMambaState) {
            manager = std::make_unique<MambaStateManager>(group_cache_block_tokens, spec.cache_blocks_per_lcm_block,
                                                          group_id);
        } else {
            manager = std::make_unique<SwaManager>(group_cache_block_tokens, spec.cache_blocks_per_lcm_block,
                                                   spec.sliding_window, group_id);
        }
        groups.emplace_back(spec, std::move(manager));
    }
    return KvCacheCoordinator{std::move(groups), cache_block_tokens, pool, host_pool};
}

}  // namespace tokenspeed
