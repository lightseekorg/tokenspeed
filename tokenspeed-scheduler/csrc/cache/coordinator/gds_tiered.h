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

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "cache/coordinator/admission_candidates.h"
#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/victim_set.h"

namespace tokenspeed {
// The complete transaction, including Host capacity needed before Device reuse.
// All locations and keys are non-owning until CacheEvictor prepares the plan.
struct TieredGdsPlan {
    struct Member {
        std::uint32_t group;
        CacheBlockLocation location;
        std::optional<CacheKey> key;
        bool request_reclaim;
    };
    std::vector<Member> device_removals;
    std::vector<Member> host_removals;
    std::vector<CacheKey> host_copies;
    std::vector<CacheKey> host_kept;
    std::vector<ContentHash> losses;
    std::vector<ContentHash> fresh_rejections;
    GdsEvictionStats stats;
    GdsTrace trace;
};
struct TieredHostAllocation {
    std::shared_ptr<const TieredGdsPlan> plan;
    std::vector<bool> accepted;
    std::size_t accepted_segments;
};
// A read-only policy view. Copying this closure neither admits retention nor
// changes a GDS watermark; the Disk manager applies ownership and I/O limits.
struct HostCleaningCandidate {
    ContentHash hash;
    double score;
    std::uint64_t sequence;
    std::vector<CacheKey> keys;
    std::vector<CacheKey> disk_dependencies;
};
// Stop at the write budget without expanding the remaining candidate paths.
void VisitGdsHostCleaningCandidates(const CacheStore& store, const std::function<bool(HostCleaningCandidate)>& visit);

std::vector<VictimSet> TieredVictims(std::shared_ptr<const TieredGdsPlan> plan, std::pmr::memory_resource* resource);
std::optional<std::vector<VictimSet>> PlanTieredDeviceEviction(const CacheStore& store,
                                                               const AdmissionCandidateConstraints& constraints,
                                                               std::span<const BlockPlacementDemand> demands,
                                                               std::pmr::memory_resource* resource);
// Compaction cannot choose an arbitrary logical victim to clear a fixed slot.
// Only a duplicate or an orphan may be removed at that destination.
std::optional<VictimSet> PlanTieredCompaction(const CacheStore& store, const AdmissionCandidateConstraints& constraints,
                                              const CacheKey& key, std::pmr::memory_resource* resource);
// A segment ends at the given exclusive row. With prefix=false a rejected
// segment does not prevent a later one, in another bucket, from being placed.
TieredHostAllocation PlanTieredHostAllocation(const CacheStore& store, std::span<const std::uint32_t> groups,
                                              std::span<const std::int32_t> buckets,
                                              std::span<const std::size_t> segment_ends, bool allow_logical_loss,
                                              bool prefix);
}  // namespace tokenspeed
