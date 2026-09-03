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

#include "cache/coordinator/cache_coordinator.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <iterator>

#include "cache/prefix/prefix_matcher.h"
#include "utils.h"

namespace tokenspeed {

CacheCoordinator::CacheCoordinator(std::vector<CacheGroup> groups, std::int32_t prefix_granularity, BlockPool& pool,
                                   BlockPool* host_pool, bool stream_device_cache_to_host, bool enable_l3_storage)
    : groups_{std::move(groups)},
      pool_{pool},
      host_pool_{host_pool},
      stream_device_cache_to_host_{stream_device_cache_to_host && host_pool != nullptr},
      enable_l3_storage_{enable_l3_storage && host_pool != nullptr},
      prefix_granularity_{prefix_granularity} {
    _assert(prefix_granularity_ > 0, "coordinator needs positive prefix_granularity");
    geometry_.reserve(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(groups_[i].Id() == static_cast<std::uint32_t>(i), "cache group id must equal its group index");
        const std::int32_t group_block_granularity = groups_[i].Spec().block_granularity;
        _assert(group_block_granularity > 0 && prefix_granularity_ % group_block_granularity == 0,
                "group block_granularity must be a positive divisor of the prefix granularity");
        _assert(groups_[i].Allocator().CacheBlocksPerLcmBlock() == groups_[i].Spec().cache_blocks_per_lcm_block,
                "group allocator packing must match its group spec");
        geometry_.emplace_back(group_block_granularity);
        if (groups_[i].Matcher().IsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!groups_[i].Matcher().IsPrefixClosed()) {
            match_order_.push_back(i);
        }
    }
}

bool CacheCoordinator::HasMambaStateGroup() const {
    return std::ranges::any_of(groups_,
                               [](const CacheGroup& group) { return group.Spec().kind == AttnKind::kMambaState; });
}

bool CacheCoordinator::ClearDeviceCache() {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> cached_locations;
    for (const CacheGroup& group : groups_) {
        const PrefixCacheIndex& index = group.Index();
        std::vector<CacheBlockLocation> group_locations = index.EvictableLocations(pool_);
        if (static_cast<std::int32_t>(group_locations.size()) != index.NumEntries(pool_)) {
            return false;
        }
        for (CacheBlockLocation location : group_locations) {
            cached_locations.emplace_back(group.Id(), location);
        }
    }

    pending_stores_.clear();
    storage_keys_.clear();
    for (const auto& [group_id, location] : cached_locations) {
        _assert(evictCachedBlock(group_id, location), "clearable Device cache entry disappeared");
    }
    return true;
}

bool CacheCoordinator::ClearCache() {
    if (host_pool_ == nullptr) {
        return ClearDeviceCache();
    }

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> host_locations;
    for (const CacheGroup& group : groups_) {
        const PrefixCacheIndex& index = group.Index();
        std::vector<CacheBlockLocation> group_locations = index.EvictableLocations(*host_pool_);
        if (static_cast<std::int32_t>(group_locations.size()) != index.NumEntries(*host_pool_)) {
            return false;
        }
        for (CacheBlockLocation location : group_locations) {
            host_locations.emplace_back(group.Id(), location);
        }
    }

    // ClearDeviceCache performs its complete pin check before mutation. Since
    // Host was checked above, a false return leaves both tiers unchanged.
    if (!ClearDeviceCache()) {
        return false;
    }
    for (const auto& [group_id, location] : host_locations) {
        _assert(groups_[group_id].Index().Evict(*host_pool_, location).has_value(),
                "clearable Host cache entry disappeared");
    }
    return true;
}

std::vector<CacheKey> CacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                     std::uint32_t group_id) const {
    _assert(group_id < groups_.size(), "cache key group id out of range");
    const std::int32_t group_block_granularity = geometry_[group_id].BlockGranularity();
    const std::int32_t pages_per_prefix_hash = prefix_granularity_ / group_block_granularity;
    _assert(content_hashes.size() <=
                std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(pages_per_prefix_hash),
            "expanded cache key count exceeds size_t range");
    std::vector<CacheKey> keys;
    keys.reserve(content_hashes.size() * static_cast<std::size_t>(pages_per_prefix_hash));
    for (const std::string& content_hash : content_hashes) {
        for (std::int32_t offset = 0; offset < pages_per_prefix_hash; ++offset) {
            keys.push_back(CacheKey{
                .group_id = group_id,
                .content_hash = content_hash,
                .page_offset = offset,
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
// Bounds align down to the shared prefix granularity.
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
        if (groups[i].Matcher().IsPrefixClosed()) {
            prefix_closed_tokens = bound_tokens;
        }
    }
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i : order) {
            if (groups[i].Matcher().IsPrefixClosed() || extent(i) <= bound_tokens) {
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

std::vector<std::vector<CacheKey>> CacheCoordinator::buildGroupKeys(std::span<const std::string> content_hashes) const {
    std::vector<std::vector<CacheKey>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].Id());
    }
    return group_keys;
}

template <CacheTier Tier>
BlockPool& CacheCoordinator::tierPool() {
    if constexpr (Tier == CacheTier::kDevice) {
        return pool_;
    }
    FatalCheck(host_pool_ != nullptr, "Host cache tier is not configured");
    return *host_pool_;
}

template <CacheTier Tier>
const BlockPool& CacheCoordinator::tierPool() const {
    if constexpr (Tier == CacheTier::kDevice) {
        return pool_;
    }
    FatalCheck(host_pool_ != nullptr, "Host cache tier is not configured");
    return *host_pool_;
}

// The one tier matcher: slots below floor_tokens are assumed valid in a lower tier; per_group
// blocks are relative to the floor, num_common_tokens is the absolute converged boundary (in
// TOKENS). num_prefix_pages = content_hashes.size() in prefix pages;
// each group key vector is expanded to that group's block_granularity.
template <CacheTier Tier>
CacheCoordinator::PrefixProbe::Tier CacheCoordinator::probeTierWithKeys(
    std::span<const std::vector<CacheKey>> group_keys, std::span<const std::size_t> match_order,
    std::int32_t num_prefix_pages, std::int32_t floor_tokens) const {
    const BlockPool& pool = tierPool<Tier>();
    PrefixProbe::Tier out;
    out.per_group.resize(groups_.size());
    if (match_order.empty()) {
        return out;
    }
    const ConvergedBoundary boundary = SweepThenConverge(
        match_order, groups_, num_prefix_pages * prefix_granularity_, prefix_granularity_,
        [&](std::size_t i, std::int32_t bound_tokens) {
            const std::int32_t group_block_granularity = geometry_[i].BlockGranularity();
            const std::unordered_set<CacheKey, CacheKeyHash>* extra_hits = nullptr;
            if constexpr (Tier == CacheTier::kHost) {
                extra_hits = enable_l3_storage_ ? &storage_keys_ : nullptr;
            }
            out.per_group[i] = groups_[i].Matcher().Probe(groups_[i].Index(), pool, group_keys[i],
                                                          floor_tokens / group_block_granularity,
                                                          bound_tokens / group_block_granularity, extra_hits);
        },
        [&](std::size_t i) {
            return floor_tokens +
                   static_cast<std::int32_t>(out.per_group[i].hits.size()) * geometry_[i].BlockGranularity();
        });

    // A matcher can find a q-aligned resume point above the final P-aligned
    // boundary. Only acquire the CacheBlocks covered by the shared boundary.
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        GroupPrefixProbe& probe = out.per_group[i];
        const std::int32_t group_block_granularity = geometry_[i].BlockGranularity();
        const std::int32_t covered_pages = (boundary.common_tokens - floor_tokens) / group_block_granularity;
        if (static_cast<std::int32_t>(probe.hits.size()) > covered_pages) {
            probe.hits.resize(static_cast<std::size_t>(covered_pages));
        }
    }
    out.num_common_tokens = boundary.common_tokens;
    out.prefix_closed_tokens = boundary.prefix_closed_tokens;
    return out;
}

template <CacheTier Tier>
CoordinatorMatch CacheCoordinator::acquireTierWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                                       std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                       std::uint64_t access_epoch) {
    BlockPool& pool = tierPool<Tier>();
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_pages = floor_tokens / geometry_[i].BlockGranularity();
        out.per_group[i] =
            groups_[i].Index().AcquireMatched(pool, group_keys[i], floor_pages, probe.per_group[i], access_epoch);
    }
    return out;
}

CoordinatorMatch CacheCoordinator::acquireHostWithKeys(std::span<const std::vector<CacheKey>> group_keys,
                                                       std::int32_t floor_tokens, PrefixProbe::Tier&& probe,
                                                       std::uint64_t access_epoch) {
    CoordinatorMatch out;
    out.num_common_tokens = probe.num_common_tokens;
    out.per_group.resize(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::int32_t floor_pages = floor_tokens / geometry_[i].BlockGranularity();
        const GroupPrefixProbe& group_probe = probe.per_group[i];
        PrefixMatch& match = out.per_group[i];
        match.blocks.resize(group_probe.hits.size());
        PrefixCacheIndex& index = groups_[i].Index();
        for (std::size_t hit_index = 0; hit_index < group_probe.hits.size(); ++hit_index) {
            if (group_probe.hits[hit_index] == 0) {
                continue;
            }
            const CacheKey& key = group_keys[i][static_cast<std::size_t>(floor_pages) + hit_index];
            CacheBlockRef host_block_ref = index.Find(*host_pool_, key);
            if (host_block_ref) {
                PrefixMatch acquired =
                    index.AcquireMatched(*host_pool_, group_keys[i], floor_pages + static_cast<std::int32_t>(hit_index),
                                         GroupPrefixProbe{.hits = {1}}, access_epoch);
                match.blocks[hit_index] = std::move(acquired.blocks.front());
                continue;
            }
            _assert(storage_keys_.contains(key), "Host probe hit without a Host or L3 entry");
            host_block_ref = AcquireHostBlock(groups_[i].Id());
            if (!host_block_ref) {
                // Host pool is pinned; drop this and later hits so admission can
                // recompute instead of FatalCheck-ing on a missing prefetch page.
                match.blocks.resize(hit_index);
                out.num_common_tokens =
                    std::min(out.num_common_tokens,
                             floor_tokens + static_cast<std::int32_t>(hit_index) * geometry_[i].BlockGranularity());
                break;
            }
            match.blocks[hit_index] = std::move(host_block_ref);
        }
    }
    return out;
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbePrefix(std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / prefix_granularity_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_prefix_pages = static_cast<std::int32_t>(content_hashes.size());
    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    out.device = probeTierWithKeys<CacheTier::kDevice>(out.group_keys, match_order_, num_prefix_pages,
                                                       /*floor_tokens=*/0);
    if (host_pool_ != nullptr) {
        out.host = probeTierWithKeys<CacheTier::kHost>(out.group_keys, match_order_, num_prefix_pages,
                                                       /*floor_tokens=*/out.device.num_common_tokens);
    }
    return out;
}

CacheCoordinator::PrefixProbe CacheCoordinator::ProbeDecodeDevicePrefix(
    std::span<const std::string> content_hashes) const {
    _assert(content_hashes.size() <=
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() / prefix_granularity_),
            "prefix length exceeds int32 token range");
    const std::int32_t num_prefix_pages = static_cast<std::int32_t>(content_hashes.size());
    std::vector<std::size_t> history_match_order;
    history_match_order.reserve(match_order_.size());
    for (std::size_t group_index : match_order_) {
        if (groups_[group_index].Spec().kind != AttnKind::kMambaState) {
            history_match_order.push_back(group_index);
        }
    }

    PrefixProbe out;
    out.group_keys = buildGroupKeys(content_hashes);
    const auto probe_device = [&](std::int32_t floor_tokens) {
        PrefixProbe::Tier tier =
            probeTierWithKeys<CacheTier::kDevice>(out.group_keys, history_match_order, num_prefix_pages, floor_tokens);
        const std::int64_t covered_tokens =
            static_cast<std::int64_t>(tier.num_common_tokens) - static_cast<std::int64_t>(floor_tokens);
        _assert(covered_tokens >= 0, "decode destination state coverage is negative");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (groups_[i].Spec().kind == AttnKind::kMambaState) {
                const std::int64_t num_holes = covered_tokens / geometry_[i].BlockGranularity();
                _assert(num_holes <= static_cast<std::int64_t>(out.group_keys[i].size()),
                        "decode destination state hole count is outside the probed range");
                const std::size_t hole_count = static_cast<std::size_t>(num_holes);
                tier.per_group[i].hits.resize(hole_count);
            }
        }
        return tier;
    };
    out.device = probe_device(/*floor_tokens=*/0);
    return out;
}

CacheCoordinator::AcquiredPrefix CacheCoordinator::acquirePrefix(PrefixProbe&& probe, std::uint64_t access_epoch) {
    AcquiredPrefix out;
    out.device = acquireTierWithKeys<CacheTier::kDevice>(probe.group_keys, /*floor_tokens=*/0, std::move(probe.device),
                                                         access_epoch);
    if (host_pool_ != nullptr && !probe.host.per_group.empty()) {
        out.host =
            acquireHostWithKeys(probe.group_keys, out.device.num_common_tokens, std::move(probe.host), access_epoch);
    }
    return out;
}

std::int32_t CacheCoordinator::NumAvailableLcmBlocks() const {
    std::int32_t available = 0;
    for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
        const std::optional<std::uint32_t> group_id = pool_.BoundGroup(parent_id);
        if (!group_id || groups_[*group_id].Index().ParentIsFullyEvictable(
                             pool_, parent_id, groups_[*group_id].Allocator().CacheBlocksPerLcmBlock())) {
            ++available;
        }
    }
    return available;
}

std::int64_t CacheCoordinator::LcmBlocksNeededFor(std::span<const std::int64_t> group_pages) const {
    _assert(group_pages.size() == groups_.size(), "page demand requires one entry per cache group");
    std::int64_t prefix_blocks = 0;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(group_pages[i] >= 0, "group page demand must be non-negative");
        const std::int64_t packing = groups_[i].Allocator().CacheBlocksPerLcmBlock();
        prefix_blocks += (group_pages[i] + packing - 1) / packing;
    }
    return prefix_blocks;
}

std::size_t CacheCoordinator::NumActiveLcmBlocks(std::span<const std::span<const BlockTable>> request_tables) const {
    std::unordered_set<std::int32_t> active;
    for (std::span<const BlockTable> tables : request_tables) {
        for (const BlockTable& table : tables) {
            for (const CacheBlockRef& block_ref : table.Blocks()) {
                if (block_ref) {
                    active.insert(block_ref->Location().lcm_block_id);
                }
            }
        }
    }
    return active.size();
}

std::int32_t CacheCoordinator::GroupAvailablePages(std::int32_t group_index) const {
    _assert(group_index >= 0 && static_cast<std::size_t>(group_index) < groups_.size(),
            "cache group index out of range");
    const std::int32_t slots_per_parent =
        groups_[static_cast<std::size_t>(group_index)].Allocator().CacheBlocksPerLcmBlock();
    std::int32_t available = pool_.NumEmptyLcmBlocks() * slots_per_parent;
    for (std::int32_t id = 1; id <= pool_.NumLcmBlocks(); ++id) {
        if (pool_.BoundGroup(id) == static_cast<std::uint32_t>(group_index)) {
            available += slots_per_parent - pool_.OccupiedCount(id);
        }
    }
    return available;
}

std::int32_t CacheCoordinator::NumNewlyReleasableLcmBlocks(std::span<const BlockTable> tables) const {
    _assert(tables.size() == groups_.size(), "release estimate requires one table per cache group");

    struct ReleasedRefs {
        const CacheBlockRef* block_ref{};
        std::uint32_t count{0};
    };
    std::vector<std::unordered_map<CacheBlockLocation, ReleasedRefs, CacheBlockLocationHash>> released_by_group(
        groups_.size());
    std::unordered_set<std::int32_t> referenced_parents;
    for (std::size_t group_id = 0; group_id < tables.size(); ++group_id) {
        for (const CacheBlockRef& block_ref : tables[group_id].Blocks()) {
            if (!block_ref) {
                continue;
            }
            const CacheBlockLocation location = block_ref->Location();
            ReleasedRefs& refs = released_by_group[group_id][location];
            refs.block_ref = &block_ref;
            ++refs.count;
            referenced_parents.insert(location.lcm_block_id);
        }
    }

    std::int32_t count = 0;
    for (std::int32_t parent_id : referenced_parents) {
        const std::optional<std::uint32_t> group_id = pool_.BoundGroup(parent_id);
        _assert(group_id.has_value(), "request table references an unbound LCM block");
        const PrefixCacheIndex& index = groups_[*group_id].Index();
        bool parent_becomes_reclaimable = true;
        for (CacheBlockLocation location : pool_.OccupiedLocations(parent_id)) {
            const auto released = released_by_group[*group_id].find(location);
            if (released == released_by_group[*group_id].end()) {
                parent_becomes_reclaimable = false;
                break;
            }
            const std::uint32_t owners = released->second.block_ref->use_count();
            _assert(owners >= released->second.count, "request-owned reference count exceeds total owners");
            const std::uint32_t retained_owners = owners - released->second.count;
            const std::uint32_t allowed_owners = index.Contains(pool_, location) ? 1U : 0U;
            if (retained_owners != allowed_owners) {
                parent_becomes_reclaimable = false;
                break;
            }
        }
        if (parent_becomes_reclaimable) {
            ++count;
        }
    }
    return count;
}

void CacheCoordinator::CacheFullBlocks(std::span<BlockTable> tables, std::span<const std::string> content_hashes,
                                       std::uint64_t access_epoch, std::int32_t first_slot,
                                       CacheBoundaryKind boundary_kind) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    if (content_hashes.empty()) {
        return;  // hot decode rounds usually fill no page
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheKey> keys = keysForGroup(content_hashes, groups_[i].Id());
        const std::int32_t pages_per_prefix_hash = prefix_granularity_ / geometry_[i].BlockGranularity();
        cacheFullBlocksForGroup<CacheTier::kDevice>(i, tables[i], keys, first_slot * pages_per_prefix_hash,
                                                    access_epoch, boundary_kind);
    }
}

void CacheCoordinator::QueueCachedBlocksForStore(std::span<const std::string> prefix_hashes) {
    if (host_pool_ == nullptr) {
        return;
    }
    for (const CacheGroup& group : groups_) {
        if (group.Spec().kind == AttnKind::kMambaState) {
            continue;
        }
        for (CacheKey& key : keysForGroup(prefix_hashes, group.Id())) {
            if (group.Index().Contains(pool_, key)) {
                pending_stores_.push_back(StoreCandidate{.key = std::move(key)});
            }
        }
    }
}

void CacheCoordinator::QueueLatestSnapshotBlocksForStore(std::span<const std::string> prefix_hashes) {
    if (host_pool_ == nullptr) {
        return;
    }
    for (const CacheGroup& group : groups_) {
        if (group.Spec().kind != AttnKind::kMambaState) {
            continue;
        }
        std::vector<CacheKey> keys = keysForGroup(prefix_hashes, group.Id());
        for (auto key = keys.rbegin(); key != keys.rend(); ++key) {
            if (group.Index().Contains(pool_, *key)) {
                pending_stores_.push_back(StoreCandidate{.key = std::move(*key)});
                break;
            }
        }
    }
}

void CacheCoordinator::CacheCompletedBlocks(std::span<BlockTable> tables, std::span<const std::string> prefix_hashes,
                                            std::uint64_t access_epoch, std::int32_t first_new_prefix_page,
                                            std::int32_t num_computed_tokens, CacheBoundaryKind boundary_kind,
                                            bool stream_completed_to_host) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    _assert(first_new_prefix_page >= 0 && static_cast<std::size_t>(first_new_prefix_page) < prefix_hashes.size(),
            "completed page range must be non-empty");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand demand{
            .table = &tables[i],
            .prefix_hashes = prefix_hashes,
            .new_prefix_hash_begin = first_new_prefix_page,
            .completed_boundary_kind = boundary_kind,
            .num_computed_tokens = num_computed_tokens,
            .stream_completed_to_host = stream_completed_to_host,
        };
        cacheDeviceCompletedBlocksForGroup(i, demand, access_epoch);
    }
}

template <CacheTier Tier>
void CacheCoordinator::cacheFullBlocksForGroup(std::size_t group_index, BlockTable& table,
                                               std::span<const CacheKey> keys, std::int32_t first_cache_block,
                                               std::uint64_t access_epoch, CacheBoundaryKind boundary_kind,
                                               bool stream_completed_to_host) {
    std::vector<std::pair<CacheKey, CacheBlockRef>> newly_cached;
    const bool automatically_streams_to_host =
        stream_device_cache_to_host_ &&
        (groups_[group_index].Spec().kind == AttnKind::kSlidingWindow || stream_completed_to_host);
    auto* inserted = [&]() -> std::vector<std::pair<CacheKey, CacheBlockRef>>* {
        if constexpr (Tier == CacheTier::kDevice) {
            return automatically_streams_to_host || cache_mutation_sink_ ? &newly_cached : nullptr;
        }
        return nullptr;
    }();
    groups_[group_index].Index().RegisterFullBlocks(tierPool<Tier>(), table, keys, access_epoch, first_cache_block,
                                                    boundary_kind, inserted);
    if constexpr (Tier == CacheTier::kHost) {
        return;
    }
    for (auto& [key, block_ref] : newly_cached) {
        if (cache_mutation_sink_) {
            cache_mutation_sink_(key, CacheMutation::kStored);
        }
        if (!automatically_streams_to_host) {
            continue;
        }
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
        });
    }
}

CacheBlockRef CacheCoordinator::AcquireDeviceCachedBlock(const CacheKey& key) const {
    if (key.group_id >= groups_.size()) {
        return {};
    }
    return groups_[key.group_id].Index().Find(pool_, key);
}

CacheCoordinator::HostAllocationBatch CacheCoordinator::AcquireHostBlocks(std::span<const std::uint32_t> group_ids) {
    _assert(host_pool_ != nullptr, "AcquireHostBlocks requires a host pool");
    HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    _assert(group_ids.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "Host allocation batch request exceeds int32 range");

    std::vector<std::uint32_t> group_order;
    group_order.reserve(groups_.size());
    std::vector<bool> seen_group(groups_.size(), false);
    for (std::size_t i = 0; i < group_ids.size(); ++i) {
        _assert(group_ids[i] < groups_.size(), "Host block group id out of range");
        if (!seen_group[group_ids[i]]) {
            seen_group[group_ids[i]] = true;
            group_order.push_back(group_ids[i]);
        }
    }
    std::vector<std::vector<std::size_t>> unresolved_by_group(groups_.size());
    const auto assign = [&](std::span<const std::size_t> indices, std::vector<CacheBlockRef> refs) {
        _assert(refs.size() <= indices.size(), "Host allocation returned too many blocks");
        for (std::size_t i = 0; i < refs.size(); ++i) {
            batch.blocks[indices[i]] = std::move(refs[i]);
        }
        return indices.subspan(refs.size());
    };

    std::vector<std::int32_t> cache_blocks_per_group;
    cache_blocks_per_group.reserve(groups_.size());
    for (const CacheGroup& group : groups_) {
        cache_blocks_per_group.push_back(group.Allocator().CacheBlocksPerLcmBlock());
    }
    batch.blocks = host_pool_->AcquireAvailableBlocksInOrder(group_ids, cache_blocks_per_group);
    for (std::size_t i = 0; i < batch.blocks.size(); ++i) {
        if (!batch.blocks[i]) {
            unresolved_by_group[group_ids[i]].push_back(i);
        }
    }
    const auto value = [&](std::uint32_t candidate_group, CacheBlockLocation location) {
        const auto metadata = groups_[candidate_group].Index().MetadataFor(*host_pool_, location);
        _assert(metadata.has_value(), "evictable Host block has no cache metadata");
        return std::tuple{metadata->was_acquired, metadata->last_access_epoch, candidate_group, location.lcm_block_id,
                          location.slot_index};
    };
    using HostCacheValue = decltype(value(std::uint32_t{}, CacheBlockLocation{}));

    for (std::uint32_t group_id : group_order) {
        std::vector<std::size_t>& unresolved = unresolved_by_group[group_id];
        if (unresolved.empty()) {
            continue;
        }
        ++batch.stats.same_group_scans;
        std::vector<CacheBlockLocation> local_victims = groups_[group_id].Index().EvictableLocations(*host_pool_);
        std::ranges::sort(local_victims, {}, [&](CacheBlockLocation location) { return value(group_id, location); });
        const std::size_t victim_count = std::min(unresolved.size(), local_victims.size());
        for (std::size_t i = 0; i < victim_count; ++i) {
            _assert(groups_[group_id].Index().Evict(*host_pool_, local_victims[i]).has_value(),
                    "selected Host child is not evictable");
        }
        const std::int32_t packing = groups_[group_id].Allocator().CacheBlocksPerLcmBlock();
        std::vector<CacheBlockRef> refs =
            host_pool_->AcquireUpToBlocks(group_id, packing, static_cast<std::int32_t>(unresolved.size()));
        const std::span<const std::size_t> remaining = assign(unresolved, std::move(refs));
        unresolved.erase(unresolved.begin(), unresolved.end() - static_cast<std::ptrdiff_t>(remaining.size()));
    }
    const bool has_unresolved = std::ranges::any_of(
        unresolved_by_group, [](const std::vector<std::size_t>& unresolved) { return !unresolved.empty(); });
    if (has_unresolved) {
        ++batch.stats.cross_group_scans;
        std::vector<std::pair<HostCacheValue, std::int32_t>> victim_parents;
        victim_parents.reserve(static_cast<std::size_t>(host_pool_->NumLcmBlocks()));
        for (std::int32_t parent_id = 1; parent_id <= host_pool_->NumLcmBlocks(); ++parent_id) {
            const std::optional<std::uint32_t> bound_group = host_pool_->BoundGroup(parent_id);
            if (!bound_group ||
                !groups_[*bound_group].Index().ParentIsFullyEvictable(
                    *host_pool_, parent_id, groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock())) {
                continue;
            }
            std::optional<HostCacheValue> parent_value;
            for (std::int32_t slot = 0; slot < groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock(); ++slot) {
                const CacheBlockLocation location{.lcm_block_id = parent_id, .slot_index = slot};
                if (!host_pool_->IsOccupied(location)) {
                    continue;
                }
                const auto child_value = value(*bound_group, location);
                parent_value = parent_value ? std::max(*parent_value, child_value) : child_value;
            }
            _assert(parent_value.has_value(), "evictable Host parent has no children");
            victim_parents.emplace_back(*parent_value, parent_id);
        }
        std::ranges::sort(victim_parents);

        std::size_t victim_index = 0;
        for (std::size_t result_index = 0; result_index < group_ids.size() && victim_index < victim_parents.size();
             ++result_index) {
            if (batch.blocks[result_index]) {
                continue;
            }
            const std::uint32_t target_group = group_ids[result_index];
            std::vector<std::size_t>& unresolved = unresolved_by_group[target_group];
            while (!unresolved.empty() && victim_index < victim_parents.size()) {
                const std::int32_t victim_parent = victim_parents[victim_index++].second;
                const std::optional<std::uint32_t> bound_group = host_pool_->BoundGroup(victim_parent);
                if (!bound_group ||
                    !groups_[*bound_group].Index().ParentIsFullyEvictable(
                        *host_pool_, victim_parent, groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock())) {
                    continue;
                }
                const std::int32_t victim_packing = groups_[*bound_group].Allocator().CacheBlocksPerLcmBlock();
                for (std::int32_t slot = 0; slot < victim_packing; ++slot) {
                    const CacheBlockLocation location{.lcm_block_id = victim_parent, .slot_index = slot};
                    if (host_pool_->IsOccupied(location)) {
                        _assert(groups_[*bound_group].Index().Evict(*host_pool_, location).has_value(),
                                "selected Host parent changed before eviction");
                    }
                }

                const std::int32_t target_packing = groups_[target_group].Allocator().CacheBlocksPerLcmBlock();
                std::vector<CacheBlockRef> refs = host_pool_->AcquireUpToBlocksFromEmptyParent(
                    target_group, target_packing, victim_parent, static_cast<std::int32_t>(unresolved.size()));
                _assert(!refs.empty(), "evicting a Host parent did not free a placement");
                const std::span<const std::size_t> remaining = assign(unresolved, std::move(refs));
                unresolved.erase(unresolved.begin(), unresolved.end() - static_cast<std::ptrdiff_t>(remaining.size()));
                break;
            }
        }
    }
    batch.stats.allocated = static_cast<std::size_t>(
        std::ranges::count_if(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    batch.stats.unallocated = batch.stats.requested - batch.stats.allocated;
    return batch;
}

CacheBlockRef CacheCoordinator::AcquireHostBlock(std::uint32_t group_id) {
    const std::array groups{group_id};
    HostAllocationBatch batch = AcquireHostBlocks(groups);
    return batch.blocks.empty() ? CacheBlockRef{} : std::move(batch.blocks.front());
}

bool CacheCoordinator::evictCachedBlock(std::uint32_t group_id, CacheBlockLocation location) {
    std::optional<CacheKey> removed = groups_[group_id].Index().Evict(pool_, location);
    if (!removed) {
        return false;
    }
    if (cache_mutation_sink_) {
        cache_mutation_sink_(*removed, CacheMutation::kRemoved);
    }
    return true;
}

template <CacheTier Tier>
void CacheCoordinator::cacheCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                                    std::uint64_t access_epoch) {
    const std::int32_t pages_per_prefix_hash = prefix_granularity_ / geometry_[group_index].BlockGranularity();
    if (groups_[group_index].Matcher().IsPrefixClosed()) {
        std::vector<CacheKey> keys =
            keysForGroup(demand.prefix_hashes.subspan(static_cast<std::size_t>(demand.new_prefix_hash_begin)),
                         groups_[group_index].Id());
        cacheFullBlocksForGroup<Tier>(group_index, *demand.table, keys,
                                      demand.new_prefix_hash_begin * pages_per_prefix_hash, access_epoch,
                                      *demand.completed_boundary_kind, demand.stream_completed_to_host);
        return;
    }
    if (demand.num_computed_tokens < 0) {
        return;
    }
    // Mamba can publish only a state checkpoint that the kernel materialized
    // exactly at this boundary. SWA pages are ordinary KV, so an unaligned
    // endpoint can still publish its trailing complete-page boundary.
    if (groups_[group_index].Spec().kind == AttnKind::kMambaState &&
        demand.num_computed_tokens % prefix_granularity_ != 0) {
        return;
    }

    const std::int32_t boundary_cache_block =
        static_cast<std::int32_t>(demand.prefix_hashes.size()) * pages_per_prefix_hash;
    const std::int32_t lookback =
        std::min(groups_[group_index].Matcher().BoundaryLookbackPages(), boundary_cache_block);
    if (lookback == 0) {
        return;
    }
    const std::int32_t first_cache_block = boundary_cache_block - lookback;
    std::vector<CacheKey> keys = keysForGroup(demand.prefix_hashes, groups_[group_index].Id());
    cacheFullBlocksForGroup<Tier>(group_index, *demand.table,
                                  std::span<const CacheKey>{keys}.subspan(static_cast<std::size_t>(first_cache_block)),
                                  first_cache_block, access_epoch, *demand.completed_boundary_kind,
                                  demand.stream_completed_to_host);
}

void CacheCoordinator::cacheDeviceCompletedBlocksForGroup(std::size_t group_index, const GroupDemand& demand,
                                                          std::uint64_t access_epoch) {
    cacheCompletedBlocksForGroup<CacheTier::kDevice>(group_index, demand, access_epoch);
}

void CacheCoordinator::ReclaimExpired(std::span<BlockTable> tables, std::int32_t num_computed_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().ReclaimExpired(pool_, tables[i],
                                              groupExpiredBlocksAt(static_cast<std::int32_t>(i), num_computed_tokens));
    }
}

void CacheCoordinator::ConsumeReservedTokens(std::span<BlockTable> tables, std::int32_t num_tokens) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().ConsumeReservedTokens(tables[i], num_tokens);
    }
}

void CacheCoordinator::Free(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Allocator().Free(tables[i]);
    }
}

bool CacheCoordinator::ContainsHostCachedBlock(const CacheKey& key) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    _assert(key.group_id < groups_.size(), "host cache key group id out of range");
    return groups_[key.group_id].Index().Contains(*host_pool_, key);
}

bool CacheCoordinator::IsHostCachedBlock(CacheBlockLocation location) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    return std::ranges::any_of(groups_,
                               [&](const CacheGroup& group) { return group.Index().Contains(*host_pool_, location); });
}

std::int32_t CacheCoordinator::NumHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Index().NumEntries(*host_pool_);
    }
    return count;
}

std::int32_t CacheCoordinator::NumPinnedHostCachedBlocks() const {
    if (host_pool_ == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : groups_) {
        count += group.Index().NumPinnedEntries(*host_pool_);
    }
    return count;
}

void CacheCoordinator::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < groups_.size(), "CacheHostBlock group id out of range");
    groups_[key.group_id].Index().Register(*host_pool_, block_ref, key, ++next_access_epoch_);
    if (enable_l3_storage_) {
        storage_keys_.insert(key);
    }
}

void CacheCoordinator::RegisterStorageKeys(std::span<const CacheKey> keys) {
    if (!enable_l3_storage_) {
        return;
    }
    for (const CacheKey& key : keys) {
        _assert(key.group_id < groups_.size(), "storage key group id out of range");
        storage_keys_.insert(key);
    }
}

std::vector<CacheKey> CacheCoordinator::ExpandPrefixKeys(std::span<const std::string> content_hashes) const {
    std::vector<CacheKey> keys;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        std::vector<CacheKey> group_keys = keysForGroup(content_hashes, groups_[i].Id());
        keys.insert(keys.end(), std::make_move_iterator(group_keys.begin()), std::make_move_iterator(group_keys.end()));
    }
    return keys;
}

CacheCoordinator MakeCoordinator(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                                 BlockPool& pool, BlockPool* host_pool, bool stream_device_cache_to_host,
                                 bool enable_l3_storage) {
    _assert(!specs.empty(), "MakeCoordinator requires at least one spec");
    _assert(prefix_granularity > 0, "prefix_granularity must be > 0");
    _assert(specs.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "number of cache groups exceeds int32 range");
    std::vector<CacheGroup> groups;
    groups.reserve(specs.size());
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const CacheGroupSpec& spec = specs[i];
        const std::uint32_t group_id = static_cast<std::uint32_t>(i);
        _assert(spec.cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        const std::int32_t group_block_granularity = spec.block_granularity;
        _assert(group_block_granularity > 0 && prefix_granularity % group_block_granularity == 0,
                "group block_granularity must be a positive divisor of the prefix granularity");
        auto allocator = std::make_unique<GroupAllocator>(spec.cache_blocks_per_lcm_block, group_id);
        std::unique_ptr<PrefixMatcher> matcher;
        switch (spec.kind) {
            case AttnKind::kFull:
                matcher = std::make_unique<FullAttnMatcher>();
                break;
            case AttnKind::kMambaState:
                matcher = std::make_unique<SwaMatcher>(group_block_granularity, GroupGeometry::kMambaStateWindow);
                break;
            case AttnKind::kSlidingWindow:
                _assert(spec.sliding_window > 0, "sliding window group requires a positive window");
                matcher = std::make_unique<SwaMatcher>(group_block_granularity, spec.sliding_window);
                break;
            default:
                FatalCheck(false, "unknown AttnKind in coordinator group spec");
                break;
        }
        groups.emplace_back(spec, std::move(allocator), std::move(matcher));
    }
    return CacheCoordinator{std::move(groups), prefix_granularity,          pool,
                            host_pool,         stream_device_cache_to_host, enable_l3_storage};
}

}  // namespace tokenspeed
