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
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/shadow_occupancy.h"

namespace tokenspeed {

// Planning never changes the pool. Callers supply the blocks that must leave together.
class CacheEvictionPlan {
public:
    using Candidate = CacheEvictionCandidates::Candidate;
    using LocationSet = CacheEvictionCandidates::LocationSet;

    CacheEvictionPlan(const BlockPool& pool, const std::vector<CacheGroup>& groups)
        : pool_{pool}, groups_{groups}, shadow_{pool, groups} {}

    void StartCandidates(const LocationSet* protected_locations, const LocationSet* request_reclaim_locations,
                         CheckpointAging aging) {
        tracks_checkpoints_ = aging.tree != nullptr;
        candidates_.emplace(groups_, pool_, protected_locations, request_reclaim_locations, aging,
                            &removed_checkpoints_);
    }

    CacheEvictionCandidates& Candidates() { return *candidates_; }
    const CacheEvictionCandidates& Candidates() const { return *candidates_; }

    std::optional<Candidate> Next(std::span<const Candidate> request_candidates) {
        const Candidate* request =
            next_request_ < request_candidates.size() ? &request_candidates[next_request_] : nullptr;
        if (!next_cached_) {
            next_cached_ = candidates_->Next(request ? request->rank : std::numeric_limits<std::uint64_t>::max());
        }
        if (request && (!next_cached_ || CacheEvictionCandidates::Before(*request, *next_cached_))) {
            ++next_request_;
            return *request;
        }
        return std::exchange(next_cached_, std::nullopt);
    }

    // A demand per group: blocks in named buckets, blocks in any bucket, and
    // same-bucket pairs whose bucket the fit chooses (ShadowOccupancy::Fits).
    bool Fits(std::span<const std::vector<std::int64_t>> needed, std::span<const std::int64_t> dense,
              std::span<const std::int64_t> pairs) const {
        return shadow_.Fits(needed, dense, pairs);
    }

    void Remove(std::uint32_t group_id, CacheBlockLocation location) {
        shadow_.Remove(group_id, location);
        trackCheckpoint(group_id, location, 1);
    }
    void Restore(std::uint32_t group_id, CacheBlockLocation location) {
        shadow_.Restore(group_id, location);
        trackCheckpoint(group_id, location, -1);
    }
    std::size_t TouchedParents() const { return shadow_.TouchedParents(); }
    std::span<const std::int64_t> HolesByBucket(std::size_t group_index) const {
        return shadow_.HolesByBucket(group_index);
    }

    template <typename SelectNext>
    bool SelectUntilFits(std::span<const std::vector<std::int64_t>> needed, std::span<const std::int64_t> dense,
                         std::span<const std::int64_t> pairs, SelectNext&& select_next) {
        while (!Fits(needed, dense, pairs)) {
            if (!select_next()) return false;
        }
        return true;
    }

    // Newer sets are returned first; a failed demand must not cost useful cache.
    template <typename VictimSet>
    void Minimize(std::vector<VictimSet>& victims, std::span<const std::vector<std::int64_t>> needed,
                  std::span<const std::int64_t> dense, std::span<const std::int64_t> pairs) {
        std::vector<VictimSet> required;
        required.reserve(victims.size());
        for (auto it = victims.rbegin(); it != victims.rend(); ++it) {
            if (it->members.empty()) continue;
            for (const auto& member : it->members) Restore(member.group_id, member.location);
            if (!Fits(needed, dense, pairs)) {
                for (const auto& member : it->members) Remove(member.group_id, member.location);
                required.push_back(std::move(*it));
            }
        }
        std::ranges::reverse(required);
        victims = std::move(required);
    }

private:
    // The checkpoints whose State the plan took from this pool, for the aged
    // candidates' recheck.
    void trackCheckpoint(std::uint32_t group_id, CacheBlockLocation location, std::int32_t delta) {
        if (!tracks_checkpoints_ || groups_[group_id].Spec().attn_kind != AttnKind::kState) {
            return;
        }
        const std::optional<CacheKey> key = groups_[group_id].Index().KeyOf(pool_, location);
        if (!key) {
            return;
        }
        const auto it = removed_checkpoints_.try_emplace(key->content_hash, 0).first;
        it->second += delta;
        if (it->second <= 0) {
            removed_checkpoints_.erase(it);
        }
    }

    const BlockPool& pool_;
    const std::vector<CacheGroup>& groups_;
    ShadowOccupancy shadow_;
    bool tracks_checkpoints_{false};
    CacheEvictionCandidates::RemovedCheckpoints removed_checkpoints_;
    std::optional<CacheEvictionCandidates> candidates_;
    std::optional<Candidate> next_cached_;
    std::size_t next_request_{0};
};

}  // namespace tokenspeed
