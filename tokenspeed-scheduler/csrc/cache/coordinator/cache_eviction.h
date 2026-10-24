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
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/cache_group.h"
#include "cache/core/block_pool.h"
#include "cache/prefix/checkpoint_tree.h"
#include "utils.h"

namespace tokenspeed {

// Read-only eviction planning shared by Device admission and Host allocation.
// Owners commit its victims only after all index cursors have been destroyed.
class CacheEvictionPlan {
public:
    CacheEvictionPlan(const std::vector<CacheGroup>& groups, const BlockPool& pool, const CheckpointTree& checkpoints,
                      bool history_eviction_drops_checkpoint)
        : groups_{groups},
          pool_{pool},
          checkpoints_{checkpoints},
          history_eviction_drops_checkpoint_{history_eviction_drops_checkpoint},
          local_free_slots_by_bucket_(groups.size()),
          empty_parent_count_{pool.NumEmptyLcmBlocks()} {
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto holes = pool.FreeSlotsByBucket(static_cast<std::uint32_t>(i));
            auto& local = local_free_slots_by_bucket_[i];
            local.assign(holes.begin(), holes.end());
            if (local.empty()) local.assign(groups[i].Allocator().ShardCount(), 0);
        }
    }

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
        ContentHash hash;
    };

    void refreshDemotions() {
        if (demotions_version_ == checkpoint_priority_version_) return;
        demoted_checkpoints_ = checkpoints_.DemotedHashes([&](const ContentHash& hash) {
            if (removed_checkpoints_.contains(hash)) return false;
            for (const CacheGroup& group : groups_) {
                if (group.Spec().attn_kind == AttnKind::kState &&
                    !group.Index().Contains(pool_, CacheKey{.group_id = group.Id(), .content_hash = hash}))
                    return false;
            }
            return true;
        });
        demotions_version_ = checkpoint_priority_version_;
    }

    auto evictionKey(const VictimCandidate& candidate) const {
        const bool demoted = groups_[candidate.group_id].Spec().attn_kind == AttnKind::kState &&
                             demoted_checkpoints_.contains(candidate.hash);
        const int retention_class = candidate.last_access_epoch == 0 ? 0 : (demoted ? 1 : 2);
        return std::tuple{
            retention_class,    candidate.last_access_epoch,     candidate.eviction_tier,      candidate.position_rank,
            candidate.group_id, candidate.location.lcm_block_id, candidate.location.slot_index};
    }

    bool shouldEvictFirst(const VictimCandidate& lhs, const VictimCandidate& rhs) const {
        return evictionKey(lhs) < evictionKey(rhs);
    }

    struct CacheGroupVictimCandidates {
        explicit CacheGroupVictimCandidates(std::uint32_t id) : group_id{id} {}

        const VictimCandidate& CurrentCandidate() const { return candidates[next_candidate_index]; }

        std::uint32_t group_id;
        bool index_exhausted{false};
        bool started{false};
        PrefixCacheIndex::EvictionCursor index_cursor;
        std::vector<VictimCandidate> candidates;
        std::size_t next_candidate_index{0};
        std::uint64_t priority_version{0};
    };

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
            .hash = metadata ? metadata->content_hash : ContentHash{},
        };
    }

    bool ensureGroupCandidateAvailable(CacheGroupVictimCandidates& group) {
        if (groups_[group.group_id].Spec().attn_kind == AttnKind::kState) {
            if (!group.index_exhausted) {
                ++group_scans_;
                for (const auto& entry : groups_[group.group_id].Index().EvictableCandidates(pool_)) {
                    ++candidates_;

                    if (!protected_locations_.contains(entry.location) &&
                        !request_reclaim_locations_.contains(entry.location)) {
                        group.candidates.push_back(makeVictimCandidate(group.group_id, entry.location, entry.metadata));
                    }
                }
                group.index_exhausted = true;
            }
            if (group.next_candidate_index == group.candidates.size()) {
                return false;
            }
            if (group.priority_version != checkpoint_priority_version_) {
                auto first = group.candidates.begin() + group.next_candidate_index;
                std::sort(first, group.candidates.end(),
                          [&](const auto& a, const auto& b) { return shouldEvictFirst(a, b); });
                group.priority_version = checkpoint_priority_version_;
            }
            return true;
        }
        if (group.next_candidate_index < group.candidates.size()) {
            return true;
        }
        group.candidates.clear();
        group.next_candidate_index = 0;
        if (!group.started) {
            ++group_scans_;
            group.started = true;
        }
        while (!group.index_exhausted) {
            epoch_scratch_.clear();
            const bool found_epoch =
                groups_[group.group_id].Index().NextEvictionEpoch(pool_, group.index_cursor, epoch_scratch_);
            if (!found_epoch) {
                group.index_exhausted = true;
                break;
            }
            candidates_ += epoch_scratch_.size();
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
                std::ranges::sort(group.candidates,
                                  [&](const auto& a, const auto& b) { return shouldEvictFirst(a, b); });
                return true;
            }
        }
        return false;
    }

    std::optional<VictimCandidate> Next() {
        if (cache_group_candidates_.empty()) {
            for (std::size_t i = 0; i < groups_.size(); ++i)
                cache_group_candidates_.emplace_back(static_cast<std::uint32_t>(i));
        }
        refreshDemotions();
        CacheGroupVictimCandidates* selected_group = nullptr;
        for (CacheGroupVictimCandidates& group : cache_group_candidates_) {
            if (!ensureGroupCandidateAvailable(group)) {
                continue;
            }
            if (selected_group == nullptr ||
                shouldEvictFirst(group.CurrentCandidate(), selected_group->CurrentCandidate())) {
                selected_group = &group;
            }
        }

        if (next_request_candidate_index_ < request_reclaim_candidates_.size()) {
            auto first = request_reclaim_candidates_.begin() + next_request_candidate_index_;
            auto best = std::min_element(first, request_reclaim_candidates_.end(),
                                         [&](const auto& a, const auto& b) { return shouldEvictFirst(a, b); });
            std::iter_swap(first, best);
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

    // Shadow holes follow the pool's own bookkeeping: a parent whose last
    // occupant leaves takes its holes with it into the empty FIFO, and every
    // bucket owned packing / shard_count of them.
    void Remove(std::uint32_t group_id, CacheBlockLocation location) {
        if (checkpoints_.Size() != 0 &&
            (history_eviction_drops_checkpoint_ || groups_[group_id].Spec().attn_kind == AttnKind::kState)) {
            if (const auto metadata = groups_[group_id].Index().MetadataFor(pool_, location);
                metadata && checkpoints_.Contains(metadata->content_hash)) {
                if (++removed_checkpoints_[metadata->content_hash] == 1) {
                    ++checkpoint_priority_version_;
                }
            }
        }
        _assert(pool_.BoundGroup(location.lcm_block_id) == group_id,
                "released admission location belongs to another group");
        auto it =
            remaining_occupied_.try_emplace(location.lcm_block_id, pool_.OccupiedCount(location.lcm_block_id)).first;
        std::int32_t& occupied = it->second;
        _assert(occupied > 0, "admission released the same location twice");
        const GroupAllocator& allocator = groups_[group_id].Allocator();
        std::vector<std::int64_t>& holes = local_free_slots_by_bucket_[group_id];
        const auto bucket = static_cast<std::size_t>(allocator.BucketOf(location));
        if (occupied == 1) {
            const std::int64_t per_bucket = allocator.CacheBlocksPerLcmBlock() / allocator.ShardCount();
            for (std::int64_t& bucket_holes : holes) {
                bucket_holes -= per_bucket;
            }
            ++holes[bucket];
            occupied = 0;
            ++empty_parent_count_;
        } else {
            --occupied;
            ++holes[bucket];
        }
    }

    void Restore(std::uint32_t group_id, CacheBlockLocation location) {
        if (checkpoints_.Size() != 0 &&
            (history_eviction_drops_checkpoint_ || groups_[group_id].Spec().attn_kind == AttnKind::kState)) {
            if (const auto metadata = groups_[group_id].Index().MetadataFor(pool_, location);
                metadata && checkpoints_.Contains(metadata->content_hash)) {
                auto removed = removed_checkpoints_.find(metadata->content_hash);
                _assert(removed != removed_checkpoints_.end(), "restored checkpoint was not removed");
                if (--removed->second == 0) {
                    removed_checkpoints_.erase(removed);
                    ++checkpoint_priority_version_;
                }
            }
        }
        auto it = remaining_occupied_.find(location.lcm_block_id);
        _assert(it != remaining_occupied_.end(), "restored admission victim has no shadow occupancy");
        std::int32_t& occupied = it->second;
        const GroupAllocator& allocator = groups_[group_id].Allocator();
        const std::int32_t slots = allocator.CacheBlocksPerLcmBlock();
        std::vector<std::int64_t>& holes = local_free_slots_by_bucket_[group_id];
        const auto bucket = static_cast<std::size_t>(allocator.BucketOf(location));
        if (occupied == 0) {
            _assert(empty_parent_count_ > 0, "restoring an admission victim underflowed empty parents");
            --empty_parent_count_;
            occupied = 1;
            const std::int64_t per_bucket = slots / allocator.ShardCount();
            for (std::int64_t& bucket_holes : holes) {
                bucket_holes += per_bucket;
            }
            --holes[bucket];
        } else {
            _assert(occupied < slots, "restoring an admission victim overflowed its parent");
            ++occupied;
            --holes[bucket];
        }
    }

    void Protect(CacheBlockLocation location) { protected_locations_.insert(location); }
    bool IsProtected(CacheBlockLocation location) const { return protected_locations_.contains(location); }
    void AddRequestCandidate(std::uint32_t group, CacheBlockLocation location) {
        if (request_reclaim_locations_.insert(location).second)
            request_reclaim_candidates_.push_back(
                makeVictimCandidate(group, location, groups_[group].Index().MetadataFor(pool_, location)));
    }
    bool IsDemoted(const VictimCandidate& candidate) const {
        return candidate.last_access_epoch != 0 && groups_[candidate.group_id].Spec().attn_kind == AttnKind::kState &&
               demoted_checkpoints_.contains(candidate.hash);
    }
    bool Fits(std::span<const std::vector<std::int64_t>> buckets, std::span<const std::int64_t> dense) const {
        std::int64_t parents = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i)
            parents += ParentsNeededForBuckets(buckets[i], local_free_slots_by_bucket_[i], dense[i],
                                               groups_[i].Allocator().CacheBlocksPerLcmBlock());
        return parents <= empty_parent_count_;
    }
    std::size_t GroupScans() const { return group_scans_; }
    std::size_t Candidates() const { return candidates_; }

private:
    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    const CheckpointTree& checkpoints_;
    bool history_eviction_drops_checkpoint_;
    std::unordered_map<ContentHash, std::int32_t> removed_checkpoints_;
    std::uint64_t checkpoint_priority_version_{1};
    std::uint64_t demotions_version_{0};
    std::unordered_set<ContentHash> demoted_checkpoints_;
    std::unordered_map<std::int32_t, std::int32_t> remaining_occupied_;
    std::vector<std::vector<std::int64_t>> local_free_slots_by_bucket_;
    std::int64_t empty_parent_count_;
    std::vector<VictimCandidate> request_reclaim_candidates_;
    std::size_t next_request_candidate_index_{0};
    std::vector<CacheGroupVictimCandidates> cache_group_candidates_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> request_reclaim_locations_;
    std::vector<PrefixCacheIndex::EvictionCandidate> epoch_scratch_;
    std::size_t group_scans_{0};
    std::size_t candidates_{0};
};

}  // namespace tokenspeed
