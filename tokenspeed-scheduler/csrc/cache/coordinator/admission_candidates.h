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
#include <memory_resource>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/eviction_candidates.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

// Request-specific exclusions and releases, built once and then read by
// feasibility checks, candidate selection and victim-set expansion.
struct AdmissionCandidateConstraints {
    explicit AdmissionCandidateConstraints(std::pmr::memory_resource* resource)
        : protected_locations{resource}, request_reclaim_locations{resource} {}

    CacheEvictionCandidates::LocationSet protected_locations;
    CacheEvictionCandidates::LocationSet request_reclaim_locations;
    std::vector<CacheEvictionCandidates::Candidate> request_reclaim_candidates;
};

// Read the request and cache without changing either. The returned sets use
// resource, which must outlive them; candidates hold locations, not block refs.
inline AdmissionCandidateConstraints BuildAdmissionCandidateConstraints(
    const CacheStore& store, std::span<const GroupDemand> demands, std::optional<std::int32_t> num_computed_tokens,
    const PrefixProbe& prefix, std::span<const std::pair<std::uint32_t, CacheBlockLocation>> planned_claim_locations,
    std::pmr::memory_resource* resource) {
    const auto& groups = store.Groups();
    const auto geometry = store.Geometry();
    const BlockPool& pool = store.DevicePool();
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");
    AdmissionCandidateConstraints constraints{resource};
    const auto protect = [&](std::span<const CacheBlockLocation> locations) {
        constraints.protected_locations.reserve(constraints.protected_locations.size() + locations.size());
        for (const CacheBlockLocation location : locations) {
            constraints.protected_locations.insert(location);
        }
    };
    for (std::size_t i = 0; i < groups.size(); ++i) {
        protect(groups[i].Index().MatchedLocations(pool, (*prefix.group_keys)[i], /*begin_block=*/0,
                                                   prefix.device.per_group[i]));
        if (const auto& tail = prefix.device.tail; tail && !tail->group_keys[i].empty()) {
            protect(
                groups[i].Index().MatchedLocations(pool, tail->group_keys[i], /*begin_block=*/0, tail->per_group[i]));
        }
        if (const auto& state = prefix.state_at_prompt_end; state && !state->group_keys[i].empty()) {
            protect(
                groups[i].Index().MatchedLocations(pool, state->group_keys[i], /*begin_block=*/0, state->per_group[i]));
        }
    }
    constraints.protected_locations.reserve(constraints.protected_locations.size() + planned_claim_locations.size());
    for (const auto& [group_id, location] : planned_claim_locations) {
        (void)group_id;
        constraints.protected_locations.insert(location);
    }

    // A released State input and expired window slots are still table-pinned.
    // Admission can reclaim them before evicting cached blocks.
    if (num_computed_tokens) {
        for (std::size_t i = 0; i < groups.size(); ++i) {
            std::vector<CacheBlockLocation> reclaimable;
            if (const auto* step = std::get_if<StateStep>(&demands[i].extent)) {
                if (const auto location =
                        groups[i].Allocator().ReleasableStateInput(groups[i].Index(), *demands[i].table, *step)) {
                    reclaimable.push_back(*location);
                }
            } else {
                const std::int32_t expired_block_end =
                    geometry[i].ExpiredBlockEndAt(groups[i].Spec(), *num_computed_tokens);
                reclaimable = groups[i].Allocator().ReclaimableBlockLocationsAt(groups[i].Index(), *demands[i].table,
                                                                                expired_block_end);
            }
            for (CacheBlockLocation location : reclaimable) {
                if (constraints.protected_locations.contains(location)) {
                    continue;
                }
                const bool first_occurrence = constraints.request_reclaim_locations.insert(location);
                if (!first_occurrence) {
                    continue;
                }
                constraints.request_reclaim_candidates.push_back(CacheEvictionCandidates::MakeCandidate(
                    groups[i], location, groups[i].Index().MetadataFor(pool, location)));
            }
        }
        CacheEvictionCandidates::Sort(constraints.request_reclaim_candidates);
    }
    return constraints;
}

}  // namespace tokenspeed
