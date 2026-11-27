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

#pragma once

// TierEvictionPlanner: block-by-block placement on a lower tier (the Host's
// AcquireHostBlocks, the Disk's AcquireDiskBlocks). Free capacity first; then
// the tier's entries in eviction order (access epoch, an aged checkpoint at
// its retention rank, then boundary tier and position) are tried against a
// shadow of the pool until the demand fits, a State checkpoint taking its
// unpinned State siblings at the same boundary along, and Minimize restores
// every victim the demand can do without, newest first. Planning changes
// nothing; the caller evicts the victims and allocates.
// With Disk, Host placement first reclaims dead entries and usable Device
// duplicates, then safe Disk-clean copies, before ordinary dirty candidates.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <utility>
#include <vector>

#include "cache/cache_group.h"
#include "cache/coordinator/admission_history.h"
#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/clean_host_copies.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/core/block_placement_demand.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

class TierEvictionPlanner {
public:
    struct Member {
        std::uint32_t group_id;
        CacheBlockLocation location;
    };
    struct VictimSet {
        std::vector<Member> members;
    };
    struct Stats {
        // Prefix-index enumerations, including groups that can donate capacity.
        std::size_t group_scans{0};
        // Unpinned entries visited while collecting replacement candidates.
        std::size_t block_candidates{0};
        // Distinct parents whose occupancy was simulated during replacement.
        std::size_t parent_candidates{0};
    };

    // `pool` is `tier`'s pool; `tier` selects the checkpoint retention ranks.
    TierEvictionPlanner(const CacheStore& store, const AdmissionHistory& history, const BlockPool& pool, CacheTier tier)
        : store_{store},
          groups_{store.Groups()},
          pool_{pool},
          plan_{pool, store.Groups()},
          tier_{tier},
          candidates_{store.Groups(), pool, history.CheckpointAgingFor(tier)} {}
    TierEvictionPlanner(const TierEvictionPlanner&) = delete;
    TierEvictionPlanner& operator=(const TierEvictionPlanner&) = delete;

    bool Fits(std::span<const BlockPlacementDemand> block_demands) const { return plan_.Fits(block_demands); }

    // Counts an entry as gone in the shadow (a victim a caller chose).
    void Seed(std::uint32_t group_id, CacheBlockLocation location) {
        taken_.insert(location);
        plan_.Remove(group_id, location);
    }

    // Takes victims until the demand fits; false when the tier cannot fund it.
    bool Fit(std::span<const BlockPlacementDemand> block_demands) {
        const CacheEvictionCandidates::LocationSet no_request_reclaims{&scratch_};
        return plan_.SelectUntilFits(block_demands, [&] {
            if (!copies_initialized_) {
                if (tier_ == CacheTier::kHost) copies_ = HostCopiesBeforeDirty(store_);
                copies_initialized_ = true;
            }
            while (next_copy_ < copies_.size()) {
                VictimSet set;
                for (const auto& [group, location] : copies_[next_copy_++].members) {
                    if (taken_.contains(location)) continue;
                    Seed(group, location);
                    set.members.push_back({group, location});
                }
                if (set.members.empty()) continue;
                victims_.push_back(std::move(set));
                return true;
            }
            while (const auto candidate =
                       candidates_.Next({}, taken_, no_request_reclaims,
                                        [&](CacheBlockLocation location) { return taken_.contains(location); })) {
                if (taken_.contains(candidate->location)) continue;
                VictimSet set;
                const auto take = [&](std::uint32_t group_id, CacheBlockLocation location) {
                    Seed(group_id, location);
                    set.members.push_back({group_id, location});
                };
                take(candidate->group_id, candidate->location);
                // State eviction takes the other unpinned State groups at this boundary.
                if (store_.GroupIsState(candidate->group_id)) {
                    const auto key = groups_[candidate->group_id].Index().KeyOf(pool_, candidate->location);
                    store_.ForEachBoundaryKey(*key, /*tail_tokens=*/0, [&](std::size_t i, const CacheKey& sibling_key) {
                        if (!store_.GroupIsState(i)) return;
                        const auto sibling = groups_[i].Index().Inspect(pool_, sibling_key);
                        if (sibling && !sibling->pinned && !taken_.contains(sibling->location)) {
                            take(groups_[i].Id(), sibling->location);
                        }
                    });
                }
                victims_.push_back(std::move(set));
                return true;
            }
            return false;
        });
    }

    void Minimize(std::span<const BlockPlacementDemand> block_demands) {
        victims_ = plan_.Minimize(std::move(victims_), block_demands);
    }
    const std::vector<VictimSet>& Victims() const { return victims_; }
    std::vector<VictimSet> TakeVictims() { return std::move(victims_); }
    Stats GetStats() const {
        return {.group_scans = candidates_.GroupScans(),
                .block_candidates = candidates_.ScannedCandidates(),
                .parent_candidates = plan_.TouchedParents()};
    }

private:
    const CacheStore& store_;
    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    CacheEvictionPlan plan_;
    CacheTier tier_;
    bool copies_initialized_{false};
    std::vector<HostCopyVictim> copies_;
    std::size_t next_copy_{0};
    CacheEvictionCandidates candidates_;
    // Holds taken_'s slots until the planner dies, first in the planner's own
    // raw buffer (left uninitialized: only the arena writes it), so a small
    // placement calls no allocator for it.
    static constexpr std::size_t kInlineScratchBytes = 4 * 1024;
    alignas(std::max_align_t) std::array<std::byte, kInlineScratchBytes> scratch_buffer_;
    std::pmr::monotonic_buffer_resource scratch_{scratch_buffer_.data(), scratch_buffer_.size()};
    CacheEvictionCandidates::LocationSet taken_{&scratch_};
    std::vector<VictimSet> victims_;
};

// Per-group placement demands over each group's buckets, all zero.
inline std::vector<BlockPlacementDemand> EmptyBlockDemands(const CacheStore& store) {
    std::vector<BlockPlacementDemand> demands(store.Groups().size());
    for (std::size_t i = 0; i < store.Groups().size(); ++i) {
        demands[i].blocks_by_bucket.resize(store.Groups()[i].Allocator().ShardCount());
    }
    return demands;
}

}  // namespace tokenspeed
