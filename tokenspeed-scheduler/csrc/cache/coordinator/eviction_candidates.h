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
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/cache_group.h"
#include "cache/coordinator/flat_hash_map.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/checkpoint_tree.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {

// Non-owning candidate traversal. Finish it before changing the pool or indexes.
class CacheEvictionCandidates {
public:
    using LocationSet = FlatHashSet<CacheBlockLocation>;
    // Per checkpoint, the State entries the plan has taken from this tier.
    using RemovedCheckpoints = std::unordered_map<ContentHash, std::int32_t>;

    enum class BoundaryTier { kUncached, kProbationary, kEstablished, kClosedPrefix };

    struct Candidate {
        std::uint64_t last_access_epoch;
        std::int64_t position_rank{0};
        CacheBlockLocation location;
        std::uint32_t group_id;
        BoundaryTier boundary_tier{BoundaryTier::kUncached};
        // The epoch the candidate ranks at, and its distance when aged.
        std::uint64_t rank_epoch{0};
        std::int32_t aged_distance{0};
    };

    CacheEvictionCandidates(const std::vector<CacheGroup>& groups, const BlockPool& pool,
                            const LocationSet* protected_locations, const LocationSet* request_reclaim_locations,
                            CheckpointAging aging, const RemovedCheckpoints* removed_checkpoints)
        : groups_{groups},
          pool_{pool},
          protected_locations_{protected_locations},
          request_reclaim_locations_{request_reclaim_locations},
          aging_{aging},
          removed_checkpoints_{removed_checkpoints},
          aged_floor_{aging.tree != nullptr ? aging.tree->LeafDistance() : 0},
          next_aged_distance_{aging.tree != nullptr ? CheckpointTree::kMaxDistance : 0} {
        cursors_.reserve(groups.size());
        for (std::size_t i = 0; i < groups.size(); ++i) {
            cursors_.emplace_back(static_cast<std::uint32_t>(i));
        }
    }

    Candidate MakeCandidate(std::uint32_t group_id, CacheBlockLocation location,
                            const std::optional<PrefixCacheIndex::CachedBlockMetadata>& metadata) const {
        return makeCandidate(group_id, groups_[group_id].Matcher().IsPrefixClosed(), location,
                             metadata ? &metadata->eviction : nullptr);
    }

    static bool Before(const Candidate& lhs, const Candidate& rhs) { return evictionKey(lhs) < evictionKey(rhs); }

    // Every Candidate field is part of evictionKey, so candidates with equal
    // keys are identical and the sorted sequence is unique: each way below
    // reaches the same one. An epoch loaded in physical order is often
    // already in policy order, or reversed (a prefix written into ascending
    // pages leaves from the suffix).
    static void Sort(std::vector<Candidate>& candidates) {
        std::vector<Candidate> scratch;
        Sort(candidates, scratch);
    }
    static void Sort(std::vector<Candidate>& candidates, std::vector<Candidate>& scratch) {
        if (std::ranges::is_sorted(candidates, Before)) {
            return;
        }
        if (std::ranges::is_sorted(candidates, after)) {
            std::ranges::reverse(candidates);
            return;
        }
        if (!radixSortLoadedEpoch(candidates, scratch)) {
            std::ranges::sort(candidates, {}, [](const Candidate& candidate) { return evictionKey(candidate); });
        }
    }

    // Run the feasibility walk without policy reordering. Retain a bounded
    // prefix of raw epochs for the later policy walk, which still filters,
    // ranks and sorts them normally. Snapshots own policy metadata, never
    // blocks or sessions, and die with this planner. Impossible admissions
    // cannot retain the entire cache: the cap counts vector capacity, not
    // just live entries.
    template <typename EnoughCapacity>
    bool PreScanUntil(EnoughCapacity&& enough_capacity) {
        constexpr std::size_t kMaxRetainedEntries = 8192;
        std::size_t retained_entries = 0;
        std::vector<PrefixCacheIndex::EvictionCandidate> entries;
        for (GroupCursor& group : cursors_) {
            _assert(!group.started && group.snapshots.empty(), "pre-scan precedes candidate selection");
            const auto& index = groups_[group.group_id].Index();
            PrefixCacheIndex::EvictionCursor cursor;
            bool retaining = true;
            while (const auto epoch = index.PeekEvictionEpoch(pool_, cursor)) {
                entries.clear();
                index.NextEvictionEpoch(pool_, cursor, *epoch, entries);
                const bool fits =
                    enough_capacity(group.group_id, std::span<const PrefixCacheIndex::EvictionCandidate>{entries});
                if (retaining && !entries.empty()) {
                    if (entries.capacity() <= kMaxRetainedEntries - retained_entries) {
                        retained_entries += entries.capacity();
                        group.snapshots.push_back(EpochSnapshot{*epoch, std::move(entries)});
                    } else {
                        retaining = false;
                    }
                }
                if (fits) return true;
            }
        }
        return false;
    }

    // A request release can precede the cache; never inspect a newer epoch.
    std::optional<Candidate> Next(std::uint64_t max_rank_epoch) {
        while (true) {
            const Candidate* best = nullptr;
            GroupCursor* best_group = nullptr;
            enum class Source { kGroup, kAged, kRequeued } source{Source::kGroup};
            const auto consider = [&](const Candidate& candidate, Source from, GroupCursor* group) {
                if (candidate.rank_epoch <= max_rank_epoch && (best == nullptr || Before(candidate, *best))) {
                    best = &candidate;
                    source = from;
                    best_group = group;
                }
            };
            GroupCursor* next_group = nullptr;
            std::optional<std::uint64_t> next_epoch;
            for (GroupCursor& group : cursors_) {
                if (group.next_candidate < group.candidates.size()) {
                    consider(group.Current(), Source::kGroup, &group);
                } else if (!group.index_exhausted) {
                    const auto epoch = peekEpoch(group);
                    if (!epoch) {
                        group.index_exhausted = true;
                    } else if (*epoch <= max_rank_epoch && (!next_epoch || *epoch < *next_epoch)) {
                        next_epoch = epoch;
                        next_group = &group;
                    }
                }
            }
            if (next_aged_ < aged_.size()) {
                consider(aged_[next_aged_].first, Source::kAged, nullptr);
            }
            if (!requeued_.empty()) {
                consider(requeued_.front(), Source::kRequeued, nullptr);
            }
            // Read complete equal epochs before applying policy tie-breaks.
            if (next_group != nullptr && (best == nullptr || *next_epoch <= best->rank_epoch)) {
                loadEpoch(*next_group, *next_epoch);
                continue;
            }
            // Each distance's aged checkpoints are read once those of every
            // larger distance are gone.
            if (next_aged_distance_ > aged_floor_ && aging_.cutoff_epoch <= max_rank_epoch &&
                (best == nullptr || best->rank_epoch > aging_.cutoff_epoch ||
                 (best->rank_epoch == aging_.cutoff_epoch && best->aged_distance < next_aged_distance_))) {
                loadAgedDistance();
                continue;
            }
            if (best == nullptr) {
                return std::nullopt;
            }
            switch (source) {
                case Source::kGroup:
                    return best_group->candidates[best_group->next_candidate++];
                case Source::kRequeued: {
                    std::ranges::pop_heap(requeued_, after);
                    const Candidate candidate = requeued_.back();
                    requeued_.pop_back();
                    return candidate;
                }
                case Source::kAged: {
                    const auto [candidate, hash] = aged_[next_aged_++];
                    // Descendants the plan took can bring this within the leaf distance.
                    if (aging_.tree->StaysAged(*hash, aging_.tier, [&](const ContentHash& descendant) {
                            return removed_checkpoints_ != nullptr && removed_checkpoints_->contains(descendant);
                        })) {
                        return candidate;
                    }
                    Candidate own_rank = candidate;
                    own_rank.rank_epoch = own_rank.last_access_epoch;
                    own_rank.aged_distance = 0;
                    requeued_.push_back(own_rank);
                    std::ranges::push_heap(requeued_, after);
                    continue;
                }
            }
        }
    }

    std::size_t ScannedCandidates() const { return scanned_candidates_; }
    std::size_t GroupScans() const { return group_scans_; }

private:
    // The evictionKey fields before (boundary_tier, position_rank): the
    // radix path of Sort requires every candidate to share them.
    static std::tuple<std::uint64_t, std::int32_t, std::uint64_t> rankPrefix(const Candidate& candidate) {
        return std::tuple{candidate.rank_epoch, -candidate.aged_distance, candidate.last_access_epoch};
    }
    static std::tuple<std::uint64_t, std::int32_t, std::uint64_t, BoundaryTier, std::int64_t, std::uint32_t,
                      std::int32_t, std::int32_t>
    evictionKey(const Candidate& candidate) {
        const auto [rank_epoch, aged_order, last_access_epoch] = rankPrefix(candidate);
        return std::tuple{rank_epoch,
                          aged_order,
                          last_access_epoch,
                          candidate.boundary_tier,
                          candidate.position_rank,
                          candidate.group_id,
                          candidate.location.lcm_block_id,
                          candidate.location.slot_index};
    }
    static bool after(const Candidate& lhs, const Candidate& rhs) { return Before(rhs, lhs); }

    // A stable LSD radix sort by (boundary_tier, position_rank) for
    // candidates that share every evictionKey field before those (rankPrefix)
    // and the group, and arrive in strictly increasing location order, as an
    // epoch read from one index does: the location tie-break then holds
    // without comparing it. False, leaving the candidates untouched, otherwise.
    static bool radixSortLoadedEpoch(std::vector<Candidate>& candidates, std::vector<Candidate>& scratch) {
        // Smaller sets sort faster by comparison than by clearing the digit counts.
        constexpr std::size_t kMinRadixSortSize = 256;
        constexpr int kDigitBits = 11;
        if (candidates.size() < kMinRadixSortSize) {
            return false;
        }
        // The tier, then the rank with its sign bit flipped: order-preserving
        // for a rank within int32, as one from a logical block index is.
        const auto rank_key = [](const Candidate& candidate) {
            return static_cast<std::uint64_t>(candidate.boundary_tier) << 32 |
                   (static_cast<std::uint32_t>(candidate.position_rank) ^ 0x80000000U);
        };
        const Candidate& first = candidates.front();
        if (!std::in_range<std::int32_t>(first.position_rank)) {
            return false;
        }
        const auto shared_prefix = rankPrefix(first);
        std::uint64_t lowest = rank_key(first);
        std::uint64_t highest = lowest;
        for (std::size_t i = 1; i < candidates.size(); ++i) {
            const Candidate& candidate = candidates[i];
            const CacheBlockLocation previous = candidates[i - 1].location;
            if (rankPrefix(candidate) != shared_prefix || candidate.group_id != first.group_id ||
                !std::in_range<std::int32_t>(candidate.position_rank) ||
                std::pair{previous.lcm_block_id, previous.slot_index} >=
                    std::pair{candidate.location.lcm_block_id, candidate.location.slot_index}) {
                return false;
            }
            lowest = std::min(lowest, rank_key(candidate));
            highest = std::max(highest, rank_key(candidate));
        }
        scratch.resize(candidates.size());
        const int width = std::bit_width(highest - lowest);
        std::array<std::size_t, (1U << kDigitBits) + 1> starts;
        for (int shift = 0; shift < width; shift += kDigitBits) {
            const std::uint64_t mask = (std::uint64_t{1} << std::min(kDigitBits, width - shift)) - 1;
            const auto digit = [&](const Candidate& candidate) {
                return static_cast<std::size_t>((rank_key(candidate) - lowest) >> shift & mask);
            };
            std::fill_n(starts.begin(), mask + 2, 0);
            for (const Candidate& candidate : candidates) {
                ++starts[digit(candidate) + 1];
            }
            std::partial_sum(starts.begin(), starts.begin() + static_cast<std::ptrdiff_t>(mask + 2), starts.begin());
            for (const Candidate& candidate : candidates) {
                scratch[starts[digit(candidate)]++] = candidate;
            }
            candidates.swap(scratch);
        }
        return true;
    }

    bool excluded(CacheBlockLocation location) const {
        return (protected_locations_ != nullptr && protected_locations_->contains(location)) ||
               (request_reclaim_locations_ != nullptr && request_reclaim_locations_->contains(location));
    }

    // `closed`: the group is prefix-closed. `metadata` is the entry's, null
    // when no entry is at `location`.
    Candidate makeCandidate(std::uint32_t group_id, bool closed, CacheBlockLocation location,
                            const PrefixCacheIndex::EvictionMetadata* metadata) const {
        Candidate candidate{.last_access_epoch = metadata ? metadata->last_access_epoch : 0,
                            .location = location,
                            .group_id = group_id};
        candidate.rank_epoch = candidate.last_access_epoch;
        const std::int32_t position = metadata ? metadata->logical_block_index : -1;
        const CacheBoundaryKind kind = metadata ? metadata->boundary_kind : CacheBoundaryKind::kChunk;
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

    bool isState(std::uint32_t group_id) const { return groups_[group_id].Spec().attn_kind == AttnKind::kState; }

    // Whether an entry ranks among the aged checkpoints instead of its epoch.
    bool aged(std::uint32_t group_id, const PrefixCacheIndex::EvictionCandidate& entry) const {
        if (aging_.tree == nullptr || !isState(group_id) || entry.metadata.last_access_epoch < aging_.cutoff_epoch) {
            return false;
        }
        const std::optional<CacheKey> key = groups_[group_id].Index().KeyOf(pool_, entry.location);
        return key && aging_.tree->AgedDistance(key->content_hash, aging_.tier) != 0;
    }

    struct EpochSnapshot {
        std::uint64_t epoch;
        std::vector<PrefixCacheIndex::EvictionCandidate> entries;
    };

    struct GroupCursor {
        explicit GroupCursor(std::uint32_t id) : group_id{id} {}
        const Candidate& Current() const { return candidates[next_candidate]; }

        std::uint32_t group_id;
        bool index_exhausted{false};
        bool started{false};
        PrefixCacheIndex::EvictionCursor index_cursor;
        std::vector<Candidate> candidates;
        std::size_t next_candidate{0};
        std::vector<EpochSnapshot> snapshots;
        std::size_t next_snapshot{0};
    };

    std::optional<std::uint64_t> peekEpoch(GroupCursor& group) const {
        if (group.next_snapshot < group.snapshots.size()) return group.snapshots[group.next_snapshot].epoch;
        return groups_[group.group_id].Index().PeekEvictionEpoch(pool_, group.index_cursor);
    }

    void loadEpoch(GroupCursor& group, std::uint64_t epoch) {
        if (!group.started) {
            group.started = true;
            ++group_scans_;
        }
        group.candidates.clear();
        group.next_candidate = 0;
        epoch_scratch_.clear();
        const auto& index = groups_[group.group_id].Index();
        if (group.next_snapshot < group.snapshots.size()) {
            auto& snapshot = group.snapshots[group.next_snapshot++];
            _assert(snapshot.epoch == epoch, "the replay preserves complete epoch order");
            epoch_scratch_ = std::move(snapshot.entries);
            if (group.next_snapshot == group.snapshots.size()) {
                index.SeekAfterEvictionEpoch(pool_, group.index_cursor, epoch);
            }
        } else {
            index.NextEvictionEpoch(pool_, group.index_cursor, epoch, epoch_scratch_);
        }
        scanned_candidates_ += epoch_scratch_.size();
        group.candidates.reserve(epoch_scratch_.size());
        const bool closed = groups_[group.group_id].Matcher().IsPrefixClosed();
        for (const auto& entry : epoch_scratch_) {
            if (excluded(entry.location) || aged(group.group_id, entry)) {
                continue;
            }
            group.candidates.push_back(makeCandidate(group.group_id, closed, entry.location, &entry.metadata));
        }
        Sort(group.candidates, sort_scratch_);
    }

    // The unpinned State entries of the next distance's aged checkpoints,
    // as loadEpoch skips them.
    void loadAgedDistance() {
        const std::int32_t distance = next_aged_distance_--;
        aged_.clear();
        next_aged_ = 0;
        aging_.tree->ForEachAged(aging_.tier, distance, [&](const ContentHash& hash) {
            for (std::uint32_t group_id = 0; group_id < groups_.size(); ++group_id) {
                if (!isState(group_id)) {
                    continue;
                }
                const auto entry = groups_[group_id].Index().Inspect(
                    pool_, CacheKey{.group_id = groups_[group_id].Id(), .content_hash = hash});
                if (!entry || entry->pinned || excluded(entry->location) ||
                    entry->metadata.eviction.last_access_epoch < aging_.cutoff_epoch) {
                    continue;
                }
                Candidate candidate = makeCandidate(group_id, groups_[group_id].Matcher().IsPrefixClosed(),
                                                    entry->location, &entry->metadata.eviction);
                candidate.rank_epoch = aging_.cutoff_epoch;
                candidate.aged_distance = distance;
                aged_.emplace_back(candidate, &hash);
            }
        });
        scanned_candidates_ += aged_.size();
        std::ranges::sort(aged_, {}, [](const auto& entry) { return evictionKey(entry.first); });
    }

    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    const LocationSet* protected_locations_;
    const LocationSet* request_reclaim_locations_;
    CheckpointAging aging_;
    const RemovedCheckpoints* removed_checkpoints_;
    std::vector<GroupCursor> cursors_;
    std::vector<PrefixCacheIndex::EvictionCandidate> epoch_scratch_;
    // The other buffer of a loaded epoch's radix sort (Sort).
    std::vector<Candidate> sort_scratch_;
    std::int32_t aged_floor_;
    // The aged candidates of the distance being read, and the next distance
    // to read (the leaf distance once every aged distance was read).
    std::vector<std::pair<Candidate, const ContentHash*>> aged_;
    std::size_t next_aged_{0};
    std::int32_t next_aged_distance_;
    // Aged candidates that became leaves within the plan, at their own epoch.
    std::vector<Candidate> requeued_;
    std::size_t scanned_candidates_{0};
    std::size_t group_scans_{0};
};

}  // namespace tokenspeed
