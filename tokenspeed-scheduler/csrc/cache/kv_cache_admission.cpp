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
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

#include "utils.h"

namespace tokenspeed {
namespace {

struct AdmissionPlan {
    KvCacheCoordinator::PrefixProbe prefix;
    std::vector<std::pair<GroupId, CacheBlockLocation>> victims;
};

class AdmissionPlanner {
public:
    AdmissionPlanner(const std::vector<CacheGroup>& groups, const BlockPool& pool, std::span<const GroupDemand> demands,
                     AdmissionPlan& admission)
        : groups_{groups},
          pool_{pool},
          demands_{demands},
          admission_{admission},
          remaining_occupied_(static_cast<std::size_t>(pool.NumLcmBlocks()) + 1),
          local_free_slots_(groups.size()),
          blocks_needed_(groups.size()) {}

    bool Plan() {
        admission_.victims.clear();
        initializeCapacity();

        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser.
        if (fits()) {
            return true;
        }

        collectCandidates();
        for (const VictimCandidate& candidate : victim_candidates_) {
            removeOccupant(candidate.group_id, candidate.location);
        }
        if (!fits()) {
            return false;
        }

        // Start from the all-evicted feasible state, then restore newest
        // entries whenever capacity still fits. The remaining victims are the
        // oldest cache entries that admission actually requires.
        for (std::size_t i = victim_candidates_.size(); i > 0; --i) {
            const VictimCandidate& candidate = victim_candidates_[i - 1];
            restoreOccupant(candidate.group_id, candidate.location);
            if (!fits()) {
                removeOccupant(candidate.group_id, candidate.location);
                admission_.victims.emplace_back(candidate.group_id, candidate.location);
            }
        }
        std::ranges::reverse(admission_.victims);
        return true;
    }

private:
    struct VictimCandidate {
        GroupId group_id;
        CacheBlockLocation location;
        std::uint64_t last_access_epoch;
        std::int32_t logical_block_index;
        bool is_full;
        bool is_state;
        bool was_prefix_hit;
    };

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const KvCacheManager& manager = groups_[i].Manager();
            const std::int32_t device_blocks =
                manager.BlocksNeededFor(*demand.table, demand.num_tokens + demand.reserve_tokens);
            const std::int32_t host_blocks = admission_.prefix.host.per_group.empty()
                                                 ? 0
                                                 : static_cast<std::int32_t>(std::ranges::count(
                                                       admission_.prefix.host.per_group[i].hits, std::uint8_t{1}));
            blocks_needed_[i] = static_cast<std::int64_t>(device_blocks) + host_blocks;
        }

        for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
            const std::optional<GroupId> group_id = pool_.BoundGroup(parent_id);
            if (!group_id) {
                ++empty_parent_count_;
                continue;
            }

            _assert(*group_id < groups_.size(), "LCM parent has invalid group binding");
            const std::int32_t occupied = pool_.OccupiedCount(parent_id);
            const std::int32_t slots = groups_[*group_id].Manager().CacheBlocksPerLcmBlock();
            _assert(0 < occupied && occupied <= slots, "bound LCM parent has invalid occupancy");
            remaining_occupied_[static_cast<std::size_t>(parent_id)] = occupied;
            local_free_slots_[*group_id] += slots - occupied;
        }
    }

    void collectCandidates() {
        std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Manager().MatchedBlockLocations(
                pool_, admission_.prefix.group_keys[i], /*begin_blocks=*/0, admission_.prefix.device.per_group[i]);
            protected_locations.insert(hits.begin(), hits.end());
        }

        std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> candidates;
        const auto add_candidate = [&](GroupId group_id, CacheBlockLocation location) {
            if (protected_locations.contains(location) || !candidates.insert(location).second) {
                return;
            }
            const std::uint64_t last_access_epoch = groups_[group_id]
                                                        .Manager()
                                                        .CachedBlockLastAccessEpoch(pool_, location)
                                                        .value_or(std::numeric_limits<std::uint64_t>::max());
            victim_candidates_.push_back(VictimCandidate{
                .group_id = group_id,
                .location = location,
                .last_access_epoch = last_access_epoch,
                .logical_block_index =
                    groups_[group_id].Manager().CachedBlockLogicalIndex(pool_, location).value_or(-1),
                .is_full = groups_[group_id].Spec().kind == AttnKind::kFull,
                .is_state = groups_[group_id].Spec().kind == AttnKind::kMambaState,
                .was_prefix_hit = groups_[group_id].Manager().CachedBlockWasPrefixHit(pool_, location),
            });
        };

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupId group_id = static_cast<GroupId>(i);
            const KvCacheManager& manager = groups_[i].Manager();
            for (CacheBlockLocation location : manager.EvictableBlockLocations(pool_)) {
                add_candidate(group_id, location);
            }
            if (demands_[i].num_computed_tokens >= 0) {
                for (CacheBlockLocation location :
                     manager.ReclaimableBlockLocationsAt(*demands_[i].table, demands_[i].num_computed_tokens)) {
                    add_candidate(group_id, location);
                }
            }
        }
        std::ranges::sort(victim_candidates_, [](const VictimCandidate& lhs, const VictimCandidate& rhs) {
            // A real request access is the primary cache-value signal. The
            // probationary flag only breaks ties inside one request epoch.
            if (lhs.last_access_epoch != rhs.last_access_epoch) {
                return lhs.last_access_epoch < rhs.last_access_epoch;
            }
            const auto eviction_class = [](const VictimCandidate& candidate) {
                if (!candidate.was_prefix_hit && candidate.is_state && candidate.logical_block_index >= 0) {
                    return 0;
                }
                if (candidate.is_full) {
                    return 1;
                }
                return candidate.was_prefix_hit ? 3 : 2;
            };
            const int lhs_class = eviction_class(lhs);
            const int rhs_class = eviction_class(rhs);
            if (lhs_class != rhs_class) {
                return lhs_class < rhs_class;
            }
            if (lhs_class == 0 && lhs.logical_block_index != rhs.logical_block_index) {
                // Dense State checkpoints are alternative resume points. Until
                // one proves useful, retain the longer checkpoint and evict an
                // earlier one instead of letting insertion order erase the
                // previous request's live frontier.
                return lhs.logical_block_index < rhs.logical_block_index;
            }
            if (lhs_class == 1 && lhs.logical_block_index != rhs.logical_block_index) {
                // Full KV is prefix-closed: after comparing request access
                // epochs, reclaim the suffix before an earlier page.
                return lhs.logical_block_index > rhs.logical_block_index;
            }
            if (lhs.group_id != rhs.group_id) {
                return lhs.group_id < rhs.group_id;
            }
            if (lhs.location.lcm_block_id != rhs.location.lcm_block_id) {
                return lhs.location.lcm_block_id < rhs.location.lcm_block_id;
            }
            return lhs.location.slot_index < rhs.location.slot_index;
        });
    }

    void removeOccupant(GroupId group_id, CacheBlockLocation location) {
        _assert(pool_.BoundGroup(location.lcm_block_id) == group_id,
                "released admission location belongs to another group");
        std::int32_t& occupied = remaining_occupied_[static_cast<std::size_t>(location.lcm_block_id)];
        _assert(occupied > 0, "admission released the same location twice");
        const std::int32_t slots = groups_[group_id].Manager().CacheBlocksPerLcmBlock();
        if (occupied == 1) {
            local_free_slots_[group_id] -= slots - 1;
            occupied = 0;
            ++empty_parent_count_;
        } else {
            --occupied;
            ++local_free_slots_[group_id];
        }
    }

    void restoreOccupant(GroupId group_id, CacheBlockLocation location) {
        std::int32_t& occupied = remaining_occupied_[static_cast<std::size_t>(location.lcm_block_id)];
        const std::int32_t slots = groups_[group_id].Manager().CacheBlocksPerLcmBlock();
        if (occupied == 0) {
            _assert(empty_parent_count_ > 0, "restoring an admission victim underflowed empty parents");
            --empty_parent_count_;
            occupied = 1;
            local_free_slots_[group_id] += slots - 1;
        } else {
            _assert(occupied < slots, "restoring an admission victim overflowed its parent");
            ++occupied;
            --local_free_slots_[group_id];
        }
    }

    bool fits() const {
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::int64_t remaining = std::max<std::int64_t>(blocks_needed_[i] - local_free_slots_[i], 0);
            const std::int64_t slots = groups_[i].Manager().CacheBlocksPerLcmBlock();
            parents_needed += (remaining + slots - 1) / slots;
        }
        return parents_needed <= empty_parent_count_;
    }

    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    std::span<const GroupDemand> demands_;
    AdmissionPlan& admission_;
    std::vector<std::int32_t> remaining_occupied_;
    std::vector<std::int64_t> local_free_slots_;
    std::vector<std::int64_t> blocks_needed_;
    std::int64_t empty_parent_count_{0};
    std::vector<VictimCandidate> victim_candidates_;
};

std::optional<AdmissionPlan> planAdmission(const std::vector<CacheGroup>& groups, const BlockPool& pool,
                                           KvCacheCoordinator::PrefixProbe&& prefix,
                                           std::span<const GroupDemand> demands) {
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    AdmissionPlan admission;
    admission.prefix = std::move(prefix);

    AdmissionPlanner planner{groups, pool, demands, admission};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return admission;
}

}  // namespace

std::optional<KvCacheCoordinator::AdmissionResult> KvCacheCoordinator::Admit(PrefixProbe&& prefix,
                                                                             std::span<const GroupDemand> demands,
                                                                             std::optional<std::uint64_t>
                                                                                 request_access_epoch) {
    std::optional<AdmissionPlan> candidate = planAdmission(groups_, pool_, std::move(prefix), demands);
    if (!candidate) {
        return std::nullopt;
    }
    AdmissionPlan plan = std::move(*candidate);

    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    for (const GroupDemand& demand : demands) {
        _assert(demand.table != nullptr, "group demand requires a block table");
    }

    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= next_access_epoch_,
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch =
        request_access_epoch.has_value() ? *request_access_epoch : ++next_access_epoch_;
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .access_epoch = access_epoch,
    };
    if (acquired_prefix.device.num_common_tokens > 0) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            groups_[i].Manager().ClaimHitBlocks(*demands[i].table, std::move(acquired_prefix.device.per_group[i]));
        }
    }
    std::vector<std::pair<GroupId, CacheBlockLocation>> prospective_victims;
    prospective_victims.reserve(plan.victims.size());
    // A reclaimable table block may still be pinned by both the request and
    // cache here. Evict what is already free, then slide the request tables and
    // retry the blocks whose request reference has just been released.
    for (const auto& victim : plan.victims) {
        const auto& [group_id, location] = victim;
        if (!groups_[group_id].Manager().EvictCachedBlock(pool_, location)) {
            prospective_victims.push_back(victim);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (!demand.page_hashes.empty()) {
            cacheCompletedBlocksForGroup(i, demand, access_epoch);
        }
        if (demand.num_computed_tokens >= 0) {
            groups_[i].Manager().ReclaimExpired(pool_, *demand.table, demand.num_computed_tokens);
        }
    }
    for (const auto& [group_id, location] : prospective_victims) {
        if (!groups_[group_id].Manager().EvictCachedBlock(pool_, location)) {
            FatalCheck(!pool_.IsOccupied(location), "admission victim changed before acquisition");
        }
    }

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            groups_[i].Manager().AppendHostExtension(
                pool_, *demand.table, std::move(acquired_prefix.host.per_group[i].blocks), result.load_pairs);
        }
        const bool acquired =
            groups_[i].Manager().Acquire(pool_, *demand.table, demand.num_tokens, demand.reserve_tokens);
        FatalCheck(acquired, "admission plan no longer fits the block pool");
    }
    return result;
}

}  // namespace tokenspeed
