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
#include <memory_resource>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "cache/cache_group.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {

// Non-owning candidate traversal. Finish it before changing the pool or indexes.
class CacheEvictionCandidates {
public:
    using LocationSet = std::pmr::unordered_set<CacheBlockLocation, CacheBlockLocationHash>;

    enum class BoundaryTier { kUncached, kProbationary, kEstablished, kClosedPrefix };

    struct Candidate {
        std::uint64_t last_access_epoch;
        std::int64_t position_rank{0};
        CacheBlockLocation location;
        std::uint32_t group_id;
        BoundaryTier boundary_tier{BoundaryTier::kUncached};
    };

    CacheEvictionCandidates(const std::vector<CacheGroup>& groups, const BlockPool& pool,
                            const LocationSet* protected_locations, const LocationSet* request_reclaim_locations)
        : groups_{groups},
          pool_{pool},
          protected_locations_{protected_locations},
          request_reclaim_locations_{request_reclaim_locations} {
        cursors_.reserve(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            cursors_.emplace_back(static_cast<std::uint32_t>(i));
        }
    }

    Candidate MakeCandidate(std::uint32_t group_id, CacheBlockLocation location,
                            const std::optional<PrefixCacheIndex::CachedBlockMetadata>& metadata) const {
        Candidate candidate{.last_access_epoch = metadata ? metadata->last_access_epoch : 0,
                            .location = location,
                            .group_id = group_id};
        const std::int32_t position = metadata ? metadata->logical_block_index : -1;
        const CacheBoundaryKind kind = metadata ? metadata->boundary_kind : CacheBoundaryKind::kChunk;
        const bool closed = groups_[group_id].Matcher().IsPrefixClosed();
        const bool probationary =
            !closed && kind == CacheBoundaryKind::kChunk && !(metadata && metadata->was_acquired) && position >= 0;
        if (candidate.last_access_epoch != 0) {
            candidate.boundary_tier = probationary
                                          ? BoundaryTier::kProbationary
                                          : (closed ? BoundaryTier::kClosedPrefix : BoundaryTier::kEstablished);
        }
        if (probationary) {
            candidate.position_rank = position;
        } else if (closed && position >= 0) {
            candidate.position_rank = -static_cast<std::int64_t>(position);
        }
        return candidate;
    }

    static bool Before(const Candidate& lhs, const Candidate& rhs) { return evictionKey(lhs) < evictionKey(rhs); }

    static void Sort(std::vector<Candidate>& candidates) {
        std::ranges::sort(candidates, {}, [](const Candidate& candidate) { return evictionKey(candidate); });
    }

    // A request release can precede the cache; never inspect a newer epoch.
    std::optional<Candidate> Next(std::uint64_t max_access_epoch) {
        while (true) {
            GroupCursor* selected = nullptr;
            GroupCursor* next_group = nullptr;
            std::optional<std::uint64_t> next_epoch;
            for (GroupCursor& group : cursors_) {
                if (group.next_candidate < group.candidates.size()) {
                    if (group.Current().last_access_epoch <= max_access_epoch &&
                        (selected == nullptr || Before(group.Current(), selected->Current()))) {
                        selected = &group;
                    }
                } else if (!group.index_exhausted) {
                    const auto epoch = groups_[group.group_id].Index().PeekEvictionEpoch(pool_, group.index_cursor);
                    if (!epoch) {
                        group.index_exhausted = true;
                    } else if (*epoch <= max_access_epoch && (!next_epoch || *epoch < *next_epoch)) {
                        next_epoch = epoch;
                        next_group = &group;
                    }
                }
            }
            // Read complete equal epochs before applying policy tie-breaks.
            if (next_group != nullptr &&
                (selected == nullptr || *next_epoch <= selected->Current().last_access_epoch)) {
                loadEpoch(*next_group, *next_epoch);
                continue;
            }
            if (selected == nullptr) {
                return std::nullopt;
            }
            return selected->candidates[selected->next_candidate++];
        }
    }

    std::size_t ScannedCandidates() const { return scanned_candidates_; }
    std::size_t GroupScans() const { return group_scans_; }

private:
    static std::tuple<std::uint64_t, BoundaryTier, std::int64_t, std::uint32_t, std::int32_t, std::int32_t> evictionKey(
        const Candidate& candidate) {
        return std::tuple{candidate.last_access_epoch, candidate.boundary_tier,         candidate.position_rank,
                          candidate.group_id,          candidate.location.lcm_block_id, candidate.location.slot_index};
    }

    bool excluded(CacheBlockLocation location) const {
        return (protected_locations_ != nullptr && protected_locations_->contains(location)) ||
               (request_reclaim_locations_ != nullptr && request_reclaim_locations_->contains(location));
    }

    struct GroupCursor {
        explicit GroupCursor(std::uint32_t id) : group_id{id} {}
        const Candidate& Current() const { return candidates[next_candidate]; }

        std::uint32_t group_id;
        bool index_exhausted{false};
        bool started{false};
        PrefixCacheIndex::EvictionCursor index_cursor;
        std::vector<Candidate> candidates;
        std::size_t next_candidate{0};
    };

    void loadEpoch(GroupCursor& group, std::uint64_t epoch) {
        if (!group.started) {
            group.started = true;
            ++group_scans_;
        }
        group.candidates.clear();
        group.next_candidate = 0;
        epoch_scratch_.clear();
        groups_[group.group_id].Index().NextEvictionEpoch(pool_, group.index_cursor, epoch, epoch_scratch_);
        scanned_candidates_ += epoch_scratch_.size();
        for (const auto& entry : epoch_scratch_) {
            if (excluded(entry.location)) {
                continue;
            }
            group.candidates.push_back(MakeCandidate(group.group_id, entry.location, entry.metadata));
        }
        Sort(group.candidates);
    }

    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    const LocationSet* protected_locations_;
    const LocationSet* request_reclaim_locations_;
    std::vector<GroupCursor> cursors_;
    std::vector<PrefixCacheIndex::EvictionCandidate> epoch_scratch_;
    std::size_t scanned_candidates_{0};
    std::size_t group_scans_{0};
};

}  // namespace tokenspeed
