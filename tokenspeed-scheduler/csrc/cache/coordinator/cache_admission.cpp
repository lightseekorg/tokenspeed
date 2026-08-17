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
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases,
                     std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims)
        : groups_{groups},
          geometry_{geometry},
          pool_{pool},
          demands_{demands},
          prefix_{prefix},
          pending_store_releases_{pending_store_releases},
          victims_{victims},
          remaining_occupied_(static_cast<std::size_t>(pool.NumLcmBlocks()) + 1),
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
            if (victim_candidates_.empty()) {
                return false;
            }
            std::ranges::pop_heap(victim_candidates_, evictedAfter);
            const VictimCandidate candidate = victim_candidates_.back();
            victim_candidates_.pop_back();
            removeOccupant(candidate.group_id, candidate.location);
            victims_.emplace_back(candidate.group_id, candidate.location);
        }

        // Once removing an eviction prefix fits, keeping the entire unpopped
        // tail also fits. Tentatively restore the prefix newest-first using
        // only the planner's shadow occupancy. If a restore makes admission
        // infeasible, undo it and keep that block as a required victim.
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

    static bool evictedAfter(const VictimCandidate& lhs, const VictimCandidate& rhs) {
        return evictionKey(rhs) < evictionKey(lhs);
    }

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const std::int32_t device_blocks = geometry_[i].BlocksNeededFor(*demand.table, demand);
            const std::int32_t host_blocks =
                prefix_.host.per_group.empty()
                    ? 0
                    : static_cast<std::int32_t>(std::ranges::count(prefix_.host.per_group[i].hits, std::uint8_t{1}));
            blocks_needed_[i] = static_cast<std::int64_t>(device_blocks) + host_blocks;
        }

        for (std::int32_t parent_id = 1; parent_id <= pool_.NumLcmBlocks(); ++parent_id) {
            const std::optional<std::uint32_t> group_id = pool_.BoundGroup(parent_id);
            if (!group_id) {
                ++empty_parent_count_;
                continue;
            }

            _assert(*group_id < groups_.size(), "LCM parent has invalid group binding");
            const std::int32_t occupied = pool_.OccupiedCount(parent_id);
            const std::int32_t slots = groups_[*group_id].Allocator().CacheBlocksPerLcmBlock();
            _assert(0 < occupied && occupied <= slots, "bound LCM parent has invalid occupancy");
            remaining_occupied_[static_cast<std::size_t>(parent_id)] = occupied;
            local_free_slots_[*group_id] += slots - occupied;
        }
    }

    void collectCandidates() {
        std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Index().MatchedLocations(
                pool_, prefix_.group_keys[i], /*begin_blocks=*/0, prefix_.device.per_group[i]);
            protected_locations.insert(hits.begin(), hits.end());
        }

        std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> candidates;
        const auto add_candidate = [&](std::uint32_t group_id, CacheBlockLocation location) {
            if (protected_locations.contains(location) || !candidates.insert(location).second) {
                return;
            }
            const std::optional<PrefixCacheIndex::CachedBlockMetadata> metadata =
                groups_[group_id].Index().MetadataFor(pool_, location);
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
            victim_candidates_.push_back(VictimCandidate{
                .group_id = group_id,
                .location = location,
                // Access epochs start at one. Zero puts an uncached block
                // ahead of every reusable cache entry.
                .last_access_epoch = last_access_epoch,
                .eviction_tier = eviction_tier,
                .position_rank = position_rank,
            });
        };

        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::uint32_t group_id = static_cast<std::uint32_t>(i);
            std::vector<CacheBlockLocation> group_pending_store_releases;
            for (const auto& [released_group_id, location] : pending_store_releases_) {
                if (released_group_id == group_id) {
                    group_pending_store_releases.push_back(location);
                }
            }
            for (CacheBlockLocation location :
                 groups_[i].Index().EvictableLocationsAfterReleasing(pool_, group_pending_store_releases)) {
                add_candidate(group_id, location);
            }
            if (demands_[i].num_computed_tokens >= 0) {
                const std::int32_t expired_blocks =
                    geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), demands_[i].num_computed_tokens);
                for (CacheBlockLocation location : groups_[i].Allocator().ReclaimableBlockLocationsAt(
                         groups_[i].Index(), *demands_[i].table, expired_blocks, group_pending_store_releases)) {
                    add_candidate(group_id, location);
                }
            }
        }
        std::ranges::make_heap(victim_candidates_, evictedAfter);
    }

    void removeOccupant(std::uint32_t group_id, CacheBlockLocation location) {
        _assert(pool_.BoundGroup(location.lcm_block_id) == group_id,
                "released admission location belongs to another group");
        std::int32_t& occupied = remaining_occupied_[static_cast<std::size_t>(location.lcm_block_id)];
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
        std::int32_t& occupied = remaining_occupied_[static_cast<std::size_t>(location.lcm_block_id)];
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
    // These locations are still pinned by in-flight Store tickets. The planner only discounts those ticket refs to
    // decide whether waiting for Store ACK makes admission feasible; it never mutates the real pool.
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases_;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims_;
    std::vector<std::int32_t> remaining_occupied_;
    std::vector<std::int64_t> local_free_slots_;
    std::vector<std::int64_t> blocks_needed_;
    std::int64_t empty_parent_count_{0};
    std::vector<VictimCandidate> victim_candidates_;
};

std::optional<AdmissionPlan> planAdmission(const std::vector<CacheGroup>& groups,
                                           std::span<const GroupGeometry> geometry, const BlockPool& pool,
                                           CacheCoordinator::PrefixProbe&& prefix,
                                           std::span<const GroupDemand> demands) {
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    AdmissionPlanner planner{groups, geometry, pool, demands, prefix, {}, victims};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix), .victims = std::move(victims)};
}

}  // namespace

bool CacheCoordinator::CanAdmitAfterReleasing(
    const PrefixProbe& prefix, std::span<const GroupDemand> demands,
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> pending_store_releases) const {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    AdmissionPlanner planner{groups_, geometry_, pool_, demands, prefix, pending_store_releases, victims};
    return planner.Plan();
}

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
    std::vector<std::vector<CacheKey>> group_keys = plan.prefix.group_keys;
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
            PrefixMatch& host_match = acquired_prefix.host.per_group[i];
            const std::int32_t floor_pages = acquired_prefix.device.num_common_tokens / geometry_[i].BlockGranularity();
            std::vector<std::uint8_t> prefetch_flags(host_match.blocks.size());
            std::vector<CacheKey> host_keys(host_match.blocks.size());
            for (std::size_t hit_index = 0; hit_index < host_match.blocks.size(); ++hit_index) {
                if (!host_match.blocks[hit_index]) {
                    continue;
                }
                const CacheKey& key = group_keys[i][static_cast<std::size_t>(floor_pages) + hit_index];
                host_keys[hit_index] = key;
                prefetch_flags[hit_index] =
                    groups_[i].Index().Contains(*host_pool_, host_match.blocks[hit_index]->Location()) ? 0 : 1;
            }
            const std::size_t pair_begin = result.load_pairs.size();
            groups_[i].Allocator().AppendHostExtension(pool_, *demand.table, std::move(host_match.blocks),
                                                       result.load_pairs);
            std::size_t pair_index = pair_begin;
            for (std::size_t hit_index = 0; hit_index < prefetch_flags.size(); ++hit_index) {
                if (host_keys[hit_index].content_hash.empty()) {
                    continue;
                }
                if (pair_index >= result.load_pairs.size()) {
                    break;
                }
                BlockTransfer& transfer = result.load_pairs[pair_index++];
                transfer.key = std::move(host_keys[hit_index]);
                transfer.prefetch_from_storage = prefetch_flags[hit_index] != 0;
            }
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
