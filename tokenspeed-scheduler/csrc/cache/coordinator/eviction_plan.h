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

#include <cstdint>
#include <span>
#include <vector>

#include "cache/coordinator/shadow_occupancy.h"
#include "cache/core/block_placement_demand.h"

namespace tokenspeed {

// Planning never changes the pool. Callers supply the blocks that must leave together.
class CacheEvictionPlan {
public:
    CacheEvictionPlan(const BlockPool& pool, const std::vector<CacheGroup>& groups) : shadow_{pool, groups} {}

    // A demand per group: blocks in named buckets, blocks in any bucket, and
    // same-bucket pairs whose bucket the fit chooses (ShadowOccupancy::Fits).
    bool Fits(std::span<const BlockPlacementDemand> block_demands) const { return shadow_.Fits(block_demands); }

    void Remove(std::uint32_t group_id, CacheBlockLocation location) {
        shadow_.Remove(group_id, location);
        ++removals_;
    }
    void Restore(std::uint32_t group_id, CacheBlockLocation location) { shadow_.Restore(group_id, location); }
    std::size_t TouchedParents() const { return shadow_.TouchedParents(); }
    std::span<const std::int64_t> HolesByBucket(std::size_t group_index) const {
        return shadow_.HolesByBucket(group_index);
    }

    // Stops at the first selection after which the demand fits. A demand
    // short by n parents cannot fit before n more Removes
    // (ShadowOccupancy::ParentShortfall), so the fit is asked only then.
    template <typename SelectNext>
    bool SelectUntilFits(std::span<const BlockPlacementDemand> block_demands, SelectNext&& select_next) const {
        std::int64_t shortfall = shadow_.ParentShortfall(block_demands);
        std::int64_t asked_at = removals_;
        while (shortfall > 0) {
            if (!select_next()) return false;
            if (removals_ - asked_at >= shortfall) {
                shortfall = shadow_.ParentShortfall(block_demands);
                asked_at = removals_;
            }
        }
        return true;
    }

    // Newer sets are returned first; a failed demand must not cost useful cache.
    template <typename VictimSet>
    std::vector<VictimSet> Minimize(std::vector<VictimSet> victims,
                                    std::span<const BlockPlacementDemand> block_demands) {
        for (auto it = victims.rbegin(); it != victims.rend(); ++it) {
            if (it->members.empty()) continue;
            for (const auto& member : it->members) Restore(member.group_id, member.location);
            if (!Fits(block_demands)) {
                for (const auto& member : it->members) Remove(member.group_id, member.location);
            } else {
                it->members.clear();
            }
        }
        // The sets kept stay in selection order.
        std::erase_if(victims, [](const auto& set) { return set.members.empty(); });
        return victims;
    }

private:
    ShadowOccupancy shadow_;
    // Removes so far; a Restore never brings a fit closer, so it is not
    // subtracted (SelectUntilFits).
    std::int64_t removals_{0};
};

}  // namespace tokenspeed
