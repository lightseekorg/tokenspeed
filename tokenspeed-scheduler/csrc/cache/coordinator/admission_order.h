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

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "cache/coordinator/folly_position_radix.h"

namespace tokenspeed {

enum class AdmissionEvictionTier {
    kUncached,  // physically allocated, but owned only by the request table
    kProbationaryBoundary,
    kEstablishedBoundary,
    kClosedPrefix,
};

struct AdmissionVictimCandidate {
    std::uint32_t group_id;
    CacheBlockLocation location;
    std::uint64_t last_access_epoch;
    AdmissionEvictionTier eviction_tier;
    std::int64_t position_rank;
};

inline auto AdmissionEvictionKey(const AdmissionVictimCandidate& candidate) {
    return std::tuple{candidate.last_access_epoch, candidate.eviction_tier,         candidate.position_rank,
                      candidate.group_id,          candidate.location.lcm_block_id, candidate.location.slot_index};
}

// Input is one group's single access epoch, in physical location order as
// emitted by PrefixCacheIndex::NextEvictionEpoch. Filtering preserves that
// order. Stable position sorting therefore preserves the remaining tuple ties.
inline void SortAdmissionGroupEpoch(std::vector<AdmissionVictimCandidate>& candidates) {
    if (candidates.size() < 128 || std::ranges::any_of(candidates, [](const auto& candidate) {
            return candidate.eviction_tier != AdmissionEvictionTier::kClosedPrefix;
        })) {
        std::ranges::sort(candidates, {}, AdmissionEvictionKey);
        return;
    }
    folly_radix::sortPosition(candidates);
}

}  // namespace tokenspeed
