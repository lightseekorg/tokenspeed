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
#include <queue>
#include <tuple>
#include <unordered_set>

#include "cache/full_attn_manager.h"
#include "cache/mamba_state_manager.h"
#include "cache/swa_manager.h"
#include "scheduler/page_hasher.h"
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

std::vector<std::string> KvCacheCoordinator::keysForGroup(std::span<const std::string> content_hashes,
                                                          GroupId group_id) const {
    std::vector<std::string> keys;
    keys.reserve(content_hashes.size());
    for (const std::string& content_hash : content_hashes) {
        keys.push_back(MakeKeyWithGroupId(content_hash, group_id));
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

class AdmissionPlanner {
public:
    AdmissionPlanner(
        const std::vector<CacheGroup>& groups, const BlockPool& pool,
        const std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>& protected_locations,
        const std::vector<std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>>& reclaimable_locations,
        KvCacheCoordinator::AdmissionPlan& admission, bool pack_existing_parents)
        : groups_{groups},
          pool_{pool},
          protected_locations_{protected_locations},
          reclaimable_locations_{reclaimable_locations},
          admission_{admission},
          pack_existing_parents_{pack_existing_parents},
          parents_(static_cast<std::size_t>(pool.NumLcmBlocks()) + 1),
          free_slot_parents_(groups.size()) {
        initializeParents();
    }

    bool plan() {
        for (KvCacheCoordinator::AdmissionPlan::Group& group : admission_.per_group) {
            group.placements.clear();
        }
        admission_.victims.clear();

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (!placeGroup(static_cast<GroupId>(i))) {
                return false;
            }
        }
        return true;
    }

private:
    struct ParentState {
        std::optional<GroupId> group_id;
        std::vector<bool> occupied;
        std::vector<bool> planned;
    };
    using FreeParent = std::pair<std::size_t, std::int32_t>;
    using ParentCandidate = std::tuple<std::uint64_t, GroupId, std::int32_t>;

    void initializeParents() {
        for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
            ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
            parent.group_id = pool_.BoundGroup(parent_id);
            if (!parent.group_id) {
                empty_parents_.push_back(parent_id);
                continue;
            }
            _assert(*parent.group_id < groups_.size(), "LCM parent has invalid group binding");
            const std::int32_t slots = groups_[*parent.group_id].Manager().CacheBlocksPerLcmBlock();
            parent.occupied.resize(static_cast<std::size_t>(slots));
            parent.planned.resize(static_cast<std::size_t>(slots));
            for (CacheBlockLocation location : pool_.OccupiedLocations(parent_id)) {
                parent.occupied[static_cast<std::size_t>(location.slot_index)] = true;
            }
            const std::size_t occupied = static_cast<std::size_t>(std::ranges::count(parent.occupied, true));
            if (occupied < parent.occupied.size()) {
                free_slot_parents_[*parent.group_id].emplace(occupied, parent_id);
            }
        }
    }

    bool isEvictable(GroupId group_id, CacheBlockLocation location) const {
        return reclaimable_locations_[group_id].contains(location) ||
               groups_[group_id].Manager().IsCachedBlockEvictable(pool_, location);
    }

    const std::vector<CacheBlockLocation>& groupEvictable(GroupId group_id) {
        if (!group_evictable_) {
            group_evictable_ = groups_[group_id].Manager().EvictableBlockLocations(pool_);
            for (CacheBlockLocation location : reclaimable_locations_[group_id]) {
                if (std::ranges::find(*group_evictable_, location) == group_evictable_->end()) {
                    group_evictable_->push_back(location);
                }
            }
        }
        return *group_evictable_;
    }

    const std::vector<ParentCandidate>& parentCandidates() {
        if (evictable_parents_) {
            return *evictable_parents_;
        }
        evictable_parents_.emplace();
        std::unordered_set<std::int32_t> candidate_ids;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            for (const ParentEvictionCandidate& candidate : groups_[i].Manager().CollectEvictableParents(pool_)) {
                evictable_parents_->emplace_back(candidate.last_access, static_cast<GroupId>(i),
                                                 candidate.lcm_block_id);
                candidate_ids.insert(candidate.lcm_block_id);
            }
        }
        for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
            const ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
            if (!parent.group_id || candidate_ids.contains(parent_id)) {
                continue;
            }
            const GroupId owner = *parent.group_id;
            const bool all_evictable =
                std::ranges::all_of(pool_.OccupiedLocations(parent_id),
                                    [&](CacheBlockLocation location) { return isEvictable(owner, location); });
            if (all_evictable) {
                evictable_parents_->emplace_back(std::numeric_limits<std::uint64_t>::max(), owner, parent_id);
            }
        }
        std::ranges::sort(*evictable_parents_);
        return *evictable_parents_;
    }

    void place(GroupId group_id, CacheBlockLocation location) {
        admission_.per_group[group_id].placements.push_back(location);
        ParentState& parent = parents_[static_cast<std::size_t>(location.lcm_block_id)];
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        if (!parent.group_id) {
            const std::size_t slots =
                static_cast<std::size_t>(groups_[group_id].Manager().CacheBlocksPerLcmBlock());
            parent.group_id = group_id;
            parent.occupied.assign(slots, false);
            parent.planned.assign(slots, false);
        }
        _assert(parent.group_id == group_id && slot < parent.occupied.size() && !parent.occupied[slot],
                "admission planned an invalid LCM slot");
        parent.occupied[slot] = true;
        parent.planned[slot] = true;
        active_parent_ = location.lcm_block_id;
        const std::size_t occupied = static_cast<std::size_t>(std::ranges::count(parent.occupied, true));
        if (occupied < parent.occupied.size()) {
            free_slot_parents_[group_id].emplace(occupied, location.lcm_block_id);
        }
    }

    void replaceVictim(GroupId group_id, CacheBlockLocation location) {
        admission_.victims.emplace_back(group_id, location);
        parents_[static_cast<std::size_t>(location.lcm_block_id)]
            .occupied[static_cast<std::size_t>(location.slot_index)] = false;
        place(group_id, location);
    }

    std::optional<CacheBlockLocation> takeFreeSlot(GroupId group_id) {
        auto& candidates = free_slot_parents_[group_id];
        while (!candidates.empty()) {
            const auto [expected_occupied, parent_id] = candidates.top();
            ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
            const std::size_t occupied = static_cast<std::size_t>(std::ranges::count(parent.occupied, true));
            if (parent.group_id != group_id || occupied != expected_occupied || occupied == parent.occupied.size()) {
                candidates.pop();
                continue;
            }
            candidates.pop();
            const auto slot = std::ranges::find(parent.occupied, false);
            _assert(slot != parent.occupied.end(), "partially occupied parent has no free slot");
            return CacheBlockLocation{
                .lcm_block_id = parent_id,
                .slot_index = static_cast<std::int32_t>(slot - parent.occupied.begin()),
            };
        }
        return std::nullopt;
    }

    std::optional<CacheBlockLocation> takeActiveFreeSlot(GroupId group_id) {
        if (!active_parent_) {
            return std::nullopt;
        }
        const std::int32_t parent_id = *active_parent_;
        ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
        const auto slot = std::ranges::find(parent.occupied, false);
        if (slot == parent.occupied.end()) {
            return std::nullopt;
        }
        return CacheBlockLocation{
            .lcm_block_id = parent_id,
            .slot_index = static_cast<std::int32_t>(slot - parent.occupied.begin()),
        };
    }

    std::optional<CacheBlockLocation> takePackingVictim(GroupId group_id) {
        if (!active_parent_) {
            return std::nullopt;
        }
        const std::int32_t parent_id = *active_parent_;
        ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
        for (std::size_t slot = 0; slot < parent.occupied.size(); ++slot) {
            const CacheBlockLocation location{
                .lcm_block_id = parent_id,
                .slot_index = static_cast<std::int32_t>(slot),
            };
            if (parent.occupied[slot] && !parent.planned[slot] && isEvictable(group_id, location) &&
                !protected_locations_.contains(location)) {
                return location;
            }
        }
        return std::nullopt;
    }

    std::optional<std::int32_t> takeEmptyParent() {
        while (empty_parent_cursor_ < empty_parents_.size()) {
            const std::int32_t parent_id = empty_parents_[empty_parent_cursor_++];
            if (!parents_[static_cast<std::size_t>(parent_id)].group_id) {
                return parent_id;
            }
        }
        return std::nullopt;
    }

    std::optional<CacheBlockLocation> takeSameGroupVictim(GroupId group_id) {
        const std::vector<CacheBlockLocation>& candidates = groupEvictable(group_id);
        while (group_evictable_cursor_ < candidates.size()) {
            const CacheBlockLocation location = candidates[group_evictable_cursor_++];
            ParentState& parent = parents_[static_cast<std::size_t>(location.lcm_block_id)];
            const std::size_t slot = static_cast<std::size_t>(location.slot_index);
            if (parent.group_id == group_id && parent.occupied[slot] && !parent.planned[slot] &&
                !protected_locations_.contains(location)) {
                return location;
            }
        }
        return std::nullopt;
    }

    std::optional<std::pair<GroupId, std::int32_t>> takeForeignParent(GroupId group_id) {
        const std::vector<ParentCandidate>& candidates = parentCandidates();
        while (foreign_parent_cursor_ < candidates.size()) {
            const auto& [last_access, owner, parent_id] = candidates[foreign_parent_cursor_++];
            (void)last_access;
            const ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
            if (!parent.group_id || *parent.group_id != owner || owner == group_id) {
                continue;
            }
            bool fully_evictable = true;
            for (std::size_t slot = 0; slot < parent.occupied.size(); ++slot) {
                if (!parent.occupied[slot]) {
                    continue;
                }
                const CacheBlockLocation location{
                    .lcm_block_id = parent_id,
                    .slot_index = static_cast<std::int32_t>(slot),
                };
                if (parent.planned[slot] || protected_locations_.contains(location) ||
                    !isEvictable(owner, location)) {
                    fully_evictable = false;
                    break;
                }
            }
            if (fully_evictable) {
                return std::pair{owner, parent_id};
            }
        }
        return std::nullopt;
    }

    bool placeGroup(GroupId group_id) {
        active_parent_.reset();
        group_evictable_.reset();
        group_evictable_cursor_ = 0;
        foreign_parent_cursor_ = 0;

        const KvCacheCoordinator::AdmissionPlan::Group& group = admission_.per_group[group_id];
        const GroupDemand& demand = group.demand;
        const std::int32_t total_blocks =
            groups_[group_id].Manager().BlocksNeededFor(*demand.table, demand.num_tokens + demand.reserve_tokens);
        std::int32_t needed = group.host_placement_count + total_blocks;

        while (needed-- > 0) {
            if (pack_existing_parents_) {
                if (std::optional<CacheBlockLocation> free = takeActiveFreeSlot(group_id)) {
                    place(group_id, *free);
                    continue;
                }
                if (std::optional<CacheBlockLocation> victim = takePackingVictim(group_id)) {
                    replaceVictim(group_id, *victim);
                    continue;
                }
            }
            if (std::optional<CacheBlockLocation> free = takeFreeSlot(group_id)) {
                place(group_id, *free);
                continue;
            }
            if (!pack_existing_parents_) {
                if (std::optional<CacheBlockLocation> victim = takePackingVictim(group_id)) {
                    replaceVictim(group_id, *victim);
                    continue;
                }
            }
            active_parent_.reset();
            if (std::optional<std::int32_t> empty = takeEmptyParent()) {
                place(group_id, CacheBlockLocation{.lcm_block_id = *empty, .slot_index = 0});
                continue;
            }
            if (std::optional<CacheBlockLocation> victim = takeSameGroupVictim(group_id)) {
                replaceVictim(group_id, *victim);
                continue;
            }
            const std::optional<std::pair<GroupId, std::int32_t>> foreign_parent = takeForeignParent(group_id);
            if (!foreign_parent) {
                return false;
            }

            const auto [owner, parent_id] = *foreign_parent;
            ParentState& parent = parents_[static_cast<std::size_t>(parent_id)];
            for (std::size_t slot = 0; slot < parent.occupied.size(); ++slot) {
                if (parent.occupied[slot]) {
                    admission_.victims.emplace_back(
                        owner, CacheBlockLocation{
                                   .lcm_block_id = parent_id,
                                   .slot_index = static_cast<std::int32_t>(slot),
                               });
                }
            }
            parent = {};
            place(group_id, CacheBlockLocation{.lcm_block_id = parent_id, .slot_index = 0});
        }
        return true;
    }

    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    const std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>& protected_locations_;
    const std::vector<std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>>& reclaimable_locations_;
    KvCacheCoordinator::AdmissionPlan& admission_;
    bool pack_existing_parents_;

    std::vector<ParentState> parents_;
    std::vector<std::priority_queue<FreeParent>> free_slot_parents_;
    std::vector<std::int32_t> empty_parents_;
    std::size_t empty_parent_cursor_{0};
    std::optional<std::int32_t> active_parent_;
    std::optional<std::vector<CacheBlockLocation>> group_evictable_;
    std::size_t group_evictable_cursor_{0};
    std::optional<std::vector<ParentCandidate>> evictable_parents_;
    std::size_t foreign_parent_cursor_{0};
};

}  // namespace

std::vector<std::vector<std::string>> KvCacheCoordinator::buildGroupKeys(
    std::span<const std::string> content_hashes) const {
    std::vector<std::vector<std::string>> group_keys(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        group_keys[i] = keysForGroup(content_hashes, groups_[i].GroupId());
    }
    return group_keys;
}

// The one tier matcher: slots below floor_tokens are assumed valid in a lower tier; per_group
// blocks are relative to the floor, num_common_tokens is the absolute converged boundary (in
// TOKENS). num_cache_blocks = content_hashes.size(); every group has one key per
// logical CacheBlock.
KvCacheCoordinator::PrefixProbe::Tier KvCacheCoordinator::probeTierWithKeys(
    const BlockPool& pool, std::span<const std::vector<std::string>> group_keys, std::int32_t num_cache_blocks,
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
        tokenspeed::PrefixProbe& probe = out.per_group[i];
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
                                                         std::span<const std::vector<std::string>> group_keys,
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

std::optional<KvCacheCoordinator::AdmissionPlan> KvCacheCoordinator::ProbeAdmission(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands) const {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");

    AdmissionPlan admission;
    admission.prefix = std::move(prefix);
    admission.per_group.resize(groups_.size());

    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const std::vector<CacheBlockLocation> hits = groups_[i].Manager().MatchedBlockLocations(
            pool_, admission.prefix.group_keys[i], /*begin_blocks=*/0, admission.prefix.device.per_group[i]);
        protected_locations.insert(hits.begin(), hits.end());
    }

    std::vector<std::unordered_set<CacheBlockLocation, CacheBlockLocationHash>> reclaimable_locations(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        AdmissionPlan::Group& group = admission.per_group[i];
        group.demand = demands[i];
        const GroupDemand& demand = group.demand;
        _assert(demand.table != nullptr, "group demand requires a block table");
        const KvCacheManager& manager = groups_[i].Manager();
        if (demand.num_computed_tokens >= 0) {
            const std::vector<CacheBlockLocation> reclaimable =
                manager.ReclaimableBlockLocationsAt(*demand.table, demand.num_computed_tokens);
            reclaimable_locations[i].insert(reclaimable.begin(), reclaimable.end());
        }
        const std::int32_t host_blocks =
            admission.prefix.host.per_group.empty()
                ? 0
                : static_cast<std::int32_t>(
                      std::ranges::count(admission.prefix.host.per_group[i].hits, std::uint8_t{1}));
        group.host_placement_count = host_blocks;
    }

    // First preserve cached children. If that fragments free slots across too many
    // parents, retry by packing each group's demand into its current parent.
    AdmissionPlanner preserve_cache{groups_, pool_, protected_locations, reclaimable_locations, admission,
                                    /*pack_existing_parents=*/false};
    if (!preserve_cache.plan() &&
        !AdmissionPlanner{groups_, pool_, protected_locations, reclaimable_locations, admission,
                          /*pack_existing_parents=*/true}
             .plan()) {
        return std::nullopt;
    }
    return admission;
}

KvCacheCoordinator::AdmissionResult KvCacheCoordinator::Acquire(AdmissionPlan&& plan) {
    _assert(plan.per_group.size() == groups_.size(), "admission plan/groups size mismatch");

    AcquiredPrefix prefix = acquirePrefix(std::move(plan.prefix));
    AdmissionResult result{
        .device_prefix_tokens = prefix.device.num_common_tokens,
        .host_prefix_tokens = prefix.host.num_common_tokens,
    };
    std::size_t total_host_blocks = 0;
    for (const AdmissionPlan::Group& group : plan.per_group) {
        total_host_blocks += static_cast<std::size_t>(group.host_placement_count);
    }
    result.load_pairs.reserve(total_host_blocks);

    if (prefix.device.num_common_tokens > 0) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            groups_[i].Manager().ClaimHitBlocks(*plan.per_group[i].demand.table,
                                                std::move(prefix.device.per_group[i]));
        }
    }
    std::vector<std::pair<GroupId, CacheBlockLocation>> prospective_victims;
    prospective_victims.reserve(plan.victims.size());
    for (const auto& victim : plan.victims) {
        const auto& [group_id, location] = victim;
        if (!groups_[group_id].Manager().EvictCachedBlock(pool_, location)) {
            prospective_victims.push_back(victim);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = plan.per_group[i].demand;
        if (!demand.content_hashes.empty()) {
            cacheFullBlocksForGroup(i, *demand.table, demand.content_hashes, demand.first_page_slot,
                                    demand.num_computed_tokens);
        }
        if (demand.num_computed_tokens >= 0) {
            groups_[i].Manager().ReclaimExpired(pool_, *demand.table, demand.num_computed_tokens);
        }
    }
    for (const auto& [group_id, location] : prospective_victims) {
        if (!groups_[group_id].Manager().EvictCachedBlock(pool_, location)) {
            _assert(!pool_.IsOccupied(location), "admission victim changed before acquisition");
        }
    }

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        AdmissionPlan::Group& group = plan.per_group[i];
        const GroupDemand& demand = group.demand;
        std::vector<CacheBlockLocation>& placements = group.placements;
        const std::size_t host_count = static_cast<std::size_t>(group.host_placement_count);
        if (host_count > 0) {
            groups_[i].Manager().AppendHostExtensionAt(
                pool_, *demand.table, std::move(prefix.host.per_group[i].blocks),
                std::span<const CacheBlockLocation>{placements}.first(host_count), result.load_pairs);
        }
        groups_[i].Manager().AcquireAt(
            pool_, *demand.table, demand.num_tokens, demand.reserve_tokens,
            std::span<const CacheBlockLocation>{placements}.subspan(host_count));
    }
    return result;
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
    std::vector<BlockTransfer> pairs;
    if (host.per_group.empty()) {
        return pairs;
    }
    _assert(host.per_group.size() == groups_.size(), "host match/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        groups_[i].Manager().AppendHostExtension(pool_, tables[i], std::move(host.per_group[i].blocks), pairs);
    }
    return pairs;
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

bool KvCacheCoordinator::Acquire(std::span<BlockTable> tables, std::int32_t num_tokens) {
    // Check-then-act: no group is ever left in a partial/unaligned state.
    if (BlocksNeededFor(tables, num_tokens) > pool_.NumFreeBlocks()) {
        return false;
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const bool acquired = groups_[i].Manager().Acquire(pool_, tables[i], num_tokens);
        _assert(acquired, "pre-checked Acquire must succeed");
    }
    return true;
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
                                                 std::span<const std::string> content_hashes,
                                                 std::int32_t first_slot, std::int32_t end_tokens) {
    std::vector<std::string> keys = keysForGroup(content_hashes, groups_[group_index].GroupId());
    std::int32_t group_first_slot = first_slot;
    std::span<const std::string> group_keys = keys;
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
    std::vector<std::pair<std::string, CacheBlockRef>> newly_cached;
    groups_[group_index].Manager().CacheFullBlocks(pool_, table, group_keys, next_recency_, group_first_slot,
                                                   host_pool_ != nullptr ? &newly_cached : nullptr);
    for (auto& [key, block] : newly_cached) {
        pending_stores_.push_back(StoreCandidate{
            .key = std::move(key),
            .group_id = groups_[group_index].GroupId(),
            .block = std::move(block),
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

bool KvCacheCoordinator::ContainsHostCachedBlock(const std::string& key) const {
    if (host_pool_ == nullptr) {
        return false;
    }
    return std::ranges::any_of(
        groups_, [&](const CacheGroup& group) { return group.Manager().ContainsCachedBlock(*host_pool_, key); });
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

void KvCacheCoordinator::CacheHostBlock(GroupId group_id, CacheBlockRef& block, const std::string& key) {
    _assert(host_pool_ != nullptr, "CacheHostBlock requires a host pool");
    _assert(group_id < groups_.size(), "CacheHostBlock group id out of range");
    groups_[group_id].Manager().CacheBlock(*host_pool_, block, key, next_recency_);
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
