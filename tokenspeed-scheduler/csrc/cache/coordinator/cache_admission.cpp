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

#include "cache/coordinator/group_geometry.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "utils.h"

namespace tokenspeed {
namespace {

struct AdmissionPlan {
    CacheCoordinator::PrefixProbe prefix;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
};

class AdmissionPlanner {
public:
    AdmissionPlanner(const std::vector<CacheGroup>& groups, std::span<const GroupGeometry> geometry,
                     const BlockPool& pool, std::span<const GroupDemand> demands,
                     const CacheCoordinator::PrefixProbe& prefix,
                     std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims)
        : groups_{groups},
          geometry_{geometry},
          pool_{pool},
          demands_{demands},
          prefix_{prefix},
          victims_{victims},
          local_free_slots_(groups.size()),
          blocks_needed_(groups.size()) {}

    bool Plan() {
        victims_.clear();
        initializeCapacity();

        // Existing local holes plus empty parents are the zero-eviction
        // capacity. Do not discard cache merely to make placement denser.
        if (fits()) {
            return true;
        }

        collectCandidates();
        while (!fits()) {
            const std::optional<VictimCandidate> candidate = nextVictimCandidate();
            if (!candidate) {
                return false;
            }
            removeOccupant(candidate->group_id, candidate->location);
            victims_.emplace_back(candidate->group_id, candidate->location);
        }

        // Restore newer blocks first, keeping each one unless its capacity is
        // still needed. All releases and restores affect only shadow occupancy.
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> required_victims;
        required_victims.reserve(victims_.size());
        for (std::size_t i = victims_.size(); i > 0; --i) {
            const auto& [group_id, location] = victims_[i - 1];
            restoreOccupant(group_id, location);
            if (!fits()) {
                removeOccupant(group_id, location);
                required_victims.emplace_back(group_id, location);
            }
        }
        std::ranges::reverse(required_victims);
        victims_ = std::move(required_victims);
        return true;
    }

private:
    // Current prefix hits are protected before candidates reach this policy.
    // A request-only block with no CacheEntry is reclaimed first. Cached
    // entries then compare request access epoch, followed within one epoch by
    // the tier order below. Position keeps the deeper unproven non-closed
    // boundary, while a closed prefix is reclaimed from its suffix.
    enum class EvictionTier {
        kUncached,  // physically allocated, but owned only by the request table
        kProbationaryBoundary,
        kEstablishedBoundary,
        kClosedPrefix,
    };

    struct VictimCandidate {
        std::uint32_t group_id;
        CacheBlockLocation location;
        std::uint64_t last_access_epoch;
        EvictionTier eviction_tier;
        std::int64_t position_rank;
    };

    static auto evictionKey(const VictimCandidate& candidate) {
        return std::tuple{candidate.last_access_epoch, candidate.eviction_tier,         candidate.position_rank,
                          candidate.group_id,          candidate.location.lcm_block_id, candidate.location.slot_index};
    }

    static bool shouldEvictFirst(const VictimCandidate& lhs, const VictimCandidate& rhs) {
        return evictionKey(lhs) < evictionKey(rhs);
    }

    struct CacheGroupVictimCandidates {
        explicit CacheGroupVictimCandidates(std::uint32_t id) : group_id{id} {}

        const VictimCandidate& CurrentCandidate() const { return candidates[next_candidate_index]; }

        std::uint32_t group_id;
        bool index_exhausted{false};
        PrefixCacheIndex::EvictionCursor index_cursor;
        std::vector<VictimCandidate> candidates;
        std::size_t next_candidate_index{0};
    };

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        empty_parent_count_ = pool_.NumEmptyLcmBlocks();
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const std::int32_t device_blocks = geometry_[i].BlocksNeededFor(*demand.table, demand);
            const std::int32_t host_blocks =
                prefix_.host.per_group.empty()
                    ? 0
                    : static_cast<std::int32_t>(std::ranges::count(prefix_.host.per_group[i].hits, std::uint8_t{1}));
            blocks_needed_[i] = static_cast<std::int64_t>(device_blocks) + host_blocks;
            local_free_slots_[i] = pool_.NumFreeSlotsInGroup(static_cast<std::uint32_t>(i));
        }
    }

    // Copy a parent's real occupancy only on first use. Subsequent reads must
    // preserve the simulated releases and restores in this plan.
    std::int32_t& remainingParentOccupancy(std::uint32_t group_id, std::int32_t parent_id) {
        const auto [it, inserted] = remaining_occupied_.try_emplace(parent_id, 0);
        if (inserted) {
            it->second = pool_.OccupiedCount(parent_id);
            _assert(0 < it->second && it->second <= groups_[group_id].Allocator().CacheBlocksPerLcmBlock(),
                    "bound LCM parent has invalid occupancy");
        }
        return it->second;
    }

    VictimCandidate makeVictimCandidate(std::uint32_t group_id, CacheBlockLocation location,
                                        const std::optional<PrefixCacheIndex::CachedBlockMetadata>& metadata) const {
        const std::uint64_t last_access_epoch = metadata ? metadata->last_access_epoch : 0;
        const std::int32_t logical_block_index = metadata ? metadata->logical_block_index : -1;
        const CacheBoundaryKind boundary_kind = metadata ? metadata->boundary_kind : CacheBoundaryKind::kChunk;
        const bool is_prefix_closed = groups_[group_id].Matcher().IsPrefixClosed();
        const bool is_probationary_boundary = !is_prefix_closed && boundary_kind == CacheBoundaryKind::kChunk &&
                                              !(metadata && metadata->was_acquired) && logical_block_index >= 0;
        const EvictionTier eviction_tier = [&] {
            if (last_access_epoch == 0) {
                return EvictionTier::kUncached;
            }
            if (is_probationary_boundary) {
                return EvictionTier::kProbationaryBoundary;
            }
            return is_prefix_closed ? EvictionTier::kClosedPrefix : EvictionTier::kEstablishedBoundary;
        }();
        std::int64_t position_rank = 0;
        if (is_probationary_boundary) {
            // Retain the longer unproven frontier.
            position_rank = logical_block_index;
        } else if (is_prefix_closed && logical_block_index >= 0) {
            // Reclaim a closed prefix from its suffix.
            position_rank = -static_cast<std::int64_t>(logical_block_index);
        }
        return VictimCandidate{
            .group_id = group_id,
            .location = location,
            // Access epochs start at one. Zero puts an uncached block ahead of
            // every reusable cache entry.
            .last_access_epoch = last_access_epoch,
            .eviction_tier = eviction_tier,
            .position_rank = position_rank,
        };
    }

    void collectCandidates() {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Index().MatchedLocations(
                pool_, prefix_.group_keys[i], /*begin_blocks=*/0, prefix_.device.per_group[i]);
            protected_locations_.insert(hits.begin(), hits.end());
        }

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            if (demands_[i].num_computed_tokens < 0) {
                continue;
            }
            const std::uint32_t group_id = static_cast<std::uint32_t>(i);
            const std::int32_t expired_blocks =
                geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), demands_[i].num_computed_tokens);
            for (CacheBlockLocation location : groups_[i].Allocator().ReclaimableBlockLocationsAt(
                     groups_[i].Index(), *demands_[i].table, expired_blocks)) {
                if (protected_locations_.contains(location)) {
                    continue;
                }
                const bool first_occurrence = request_reclaim_locations_.insert(location).second;
                if (!first_occurrence) {
                    continue;
                }
                request_reclaim_candidates_.push_back(
                    makeVictimCandidate(group_id, location, groups_[i].Index().MetadataFor(pool_, location)));
            }
        }
        std::ranges::sort(request_reclaim_candidates_, shouldEvictFirst);

        cache_group_candidates_.reserve(groups_.size());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            cache_group_candidates_.emplace_back(static_cast<std::uint32_t>(i));
        }
    }

    bool prepareNextGroupCandidate(CacheGroupVictimCandidates& group) {
        if (group.next_candidate_index < group.candidates.size()) {
            return true;
        }
        group.candidates.clear();
        group.next_candidate_index = 0;
        while (!group.index_exhausted) {
            epoch_scratch_.clear();
            const bool found_epoch =
                groups_[group.group_id].Index().NextEvictionEpoch(pool_, group.index_cursor, epoch_scratch_);
            if (!found_epoch) {
                group.index_exhausted = true;
                break;
            }
            for (const PrefixCacheIndex::EvictionCandidate& entry : epoch_scratch_) {
                if (protected_locations_.contains(entry.location) ||
                    request_reclaim_locations_.contains(entry.location)) {
                    continue;
                }
                group.candidates.push_back(makeVictimCandidate(group.group_id, entry.location, entry.metadata));
            }
            // Index order breaks epoch ties by location; eviction policy also
            // considers boundary tier and logical position, so sort the whole epoch.
            if (!group.candidates.empty()) {
                std::ranges::sort(group.candidates, shouldEvictFirst);
                return true;
            }
        }
        return false;
    }

    std::optional<VictimCandidate> nextVictimCandidate() {
        CacheGroupVictimCandidates* selected_group = nullptr;
        for (CacheGroupVictimCandidates& group : cache_group_candidates_) {
            if (!prepareNextGroupCandidate(group)) {
                continue;
            }
            if (selected_group == nullptr ||
                shouldEvictFirst(group.CurrentCandidate(), selected_group->CurrentCandidate())) {
                selected_group = &group;
            }
        }

        if (next_request_candidate_index_ < request_reclaim_candidates_.size()) {
            const VictimCandidate& request_candidate = request_reclaim_candidates_[next_request_candidate_index_];
            if (selected_group == nullptr || shouldEvictFirst(request_candidate, selected_group->CurrentCandidate())) {
                ++next_request_candidate_index_;
                return request_candidate;
            }
        }
        if (selected_group == nullptr) {
            return std::nullopt;
        }
        const VictimCandidate candidate = selected_group->CurrentCandidate();
        ++selected_group->next_candidate_index;
        return candidate;
    }

    void removeOccupant(std::uint32_t group_id, CacheBlockLocation location) {
        _assert(pool_.BoundGroup(location.lcm_block_id) == group_id,
                "released admission location belongs to another group");
        std::int32_t& occupied = remainingParentOccupancy(group_id, location.lcm_block_id);
        _assert(occupied > 0, "admission released the same location twice");
        const std::int32_t slots = groups_[group_id].Allocator().CacheBlocksPerLcmBlock();
        if (occupied == 1) {
            local_free_slots_[group_id] -= slots - 1;
            occupied = 0;
            ++empty_parent_count_;
        } else {
            --occupied;
            ++local_free_slots_[group_id];
        }
    }

    void restoreOccupant(std::uint32_t group_id, CacheBlockLocation location) {
        std::int32_t& occupied = remainingParentOccupancy(group_id, location.lcm_block_id);
        const std::int32_t slots = groups_[group_id].Allocator().CacheBlocksPerLcmBlock();
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
            const std::int64_t slots = groups_[i].Allocator().CacheBlocksPerLcmBlock();
            parents_needed += (remaining + slots - 1) / slots;
        }
        return parents_needed <= empty_parent_count_;
    }

    const std::vector<CacheGroup>& groups_;
    std::span<const GroupGeometry> geometry_;
    const BlockPool& pool_;
    std::span<const GroupDemand> demands_;
    const CacheCoordinator::PrefixProbe& prefix_;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims_;
    std::unordered_map<std::int32_t, std::int32_t> remaining_occupied_;
    std::vector<std::int64_t> local_free_slots_;
    std::vector<std::int64_t> blocks_needed_;
    std::int64_t empty_parent_count_{0};
    std::vector<VictimCandidate> request_reclaim_candidates_;
    std::size_t next_request_candidate_index_{0};
    std::vector<CacheGroupVictimCandidates> cache_group_candidates_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> request_reclaim_locations_;
    std::vector<PrefixCacheIndex::EvictionCandidate> epoch_scratch_;
};

std::optional<AdmissionPlan> planAdmission(const std::vector<CacheGroup>& groups,
                                           std::span<const GroupGeometry> geometry, const BlockPool& pool,
                                           CacheCoordinator::PrefixProbe&& prefix,
                                           std::span<const GroupDemand> demands) {
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    AdmissionPlanner planner{groups, geometry, pool, demands, prefix, victims};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix), .victims = std::move(victims)};
}

}  // namespace

std::int32_t CacheCoordinator::PromotionBoundaryTokens(const PrefixProbe& prefix) const {
    const std::int32_t matched_tokens = std::max(prefix.device.num_common_tokens, prefix.host.num_common_tokens);
    const std::int32_t prefix_closed_tokens =
        std::max(prefix.device.prefix_closed_tokens, prefix.host.prefix_closed_tokens);
    return prefix_closed_tokens > matched_tokens ? prefix_closed_tokens : 0;
}

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::Admit(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands, std::optional<std::uint64_t> request_access_epoch) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    for (const GroupDemand& demand : demands) {
        _assert(demand.table != nullptr, "group demand requires a block table");
        _assert(demand.new_prefix_hash_begin >= 0 &&
                    static_cast<std::size_t>(demand.new_prefix_hash_begin) <= demand.prefix_hashes.size(),
                "new page hash begin is outside the hash history");
        const bool has_new_prefix_hashes =
            static_cast<std::size_t>(demand.new_prefix_hash_begin) < demand.prefix_hashes.size();
        _assert(demand.completed_boundary_kind.has_value() == has_new_prefix_hashes,
                "completed boundary kind must match newly completed page hashes");
    }

    std::optional<AdmissionPlan> candidate = planAdmission(groups_, geometry_, pool_, std::move(prefix), demands);
    if (!candidate) {
        return std::nullopt;
    }
    AdmissionPlan plan = std::move(*candidate);

    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= next_access_epoch_,
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch = request_access_epoch.has_value() ? *request_access_epoch : ++next_access_epoch_;
    const std::int32_t promotion_boundary_tokens = PromotionBoundaryTokens(plan.prefix);
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .promotion_boundary_tokens = promotion_boundary_tokens,
        .access_epoch = access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
    };
    if (acquired_prefix.device.num_common_tokens > 0) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            groups_[i].Allocator().ClaimHitBlocks(*demands[i].table, std::move(acquired_prefix.device.per_group[i]));
        }
    }
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> prospective_victims;
    prospective_victims.reserve(plan.victims.size());
    // A reclaimable table block may still be pinned by both the request and
    // cache here. Evict what is already free, then slide the request tables and
    // retry the blocks whose request reference has just been released.
    for (const auto& victim : plan.victims) {
        const auto& [group_id, location] = victim;
        if (!evictCachedBlock(group_id, location)) {
            prospective_victims.push_back(victim);
        }
    }
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (demand.completed_boundary_kind) {
            cacheDeviceCompletedBlocksForGroup(i, demand, access_epoch);
        }
        if (demand.num_computed_tokens >= 0) {
            groups_[i].Allocator().ReclaimExpired(
                pool_, *demand.table, geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), demand.num_computed_tokens));
        }
    }
    for (const auto& [group_id, location] : prospective_victims) {
        if (!evictCachedBlock(group_id, location)) {
            FatalCheck(!pool_.IsOccupied(location), "admission victim changed before acquisition");
        }
    }

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const GroupDemand& demand = demands[i];
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            groups_[i].Allocator().AppendHostExtension(
                pool_, *demand.table, std::move(acquired_prefix.host.per_group[i].blocks), result.load_pairs);
        }
        const std::int32_t first_new_block = demand.table->NumBlocks();
        const bool acquired =
            groups_[i].Allocator().Acquire(pool_, *demand.table, geometry_[i].PlanAcquire(*demand.table, demand));
        FatalCheck(acquired, "admission plan no longer fits the block pool");
        const std::span<const CacheBlockRef> blocks = demand.table->Blocks();
        for (std::int32_t block = first_new_block; block < demand.table->NumBlocks(); ++block) {
            if (!blocks[static_cast<std::size_t>(block)]) {
                continue;
            }
            result.new_page_ids[i].push_back(
                groups_[i].Allocator().ResolveCacheBlockId(blocks[static_cast<std::size_t>(block)]->Location()));
        }
    }
    return result;
}

}  // namespace tokenspeed
