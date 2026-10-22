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
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils.h"

namespace tokenspeed {
namespace {

struct AdmissionPlan {
    CacheCoordinator::PrefixProbe prefix;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
};

// The table shape a restore rebuilds before its dense demand is placed: how
// many blocks each bucket must receive and the tail capacity the rebuilt
// table then has. An admission plans against the live table instead and
// passes none.
struct PlannedShape {
    std::vector<std::int64_t> blocks_by_bucket;
    std::int32_t available_tokens{0};
};

// The table slots a tail ending at `tokens` occupies in a token-spanned
// group: the blocks of the page below the boundary, [first, past_end), and
// the capacity the last of them has left.
struct TailSlots {
    std::int32_t first{0};
    std::int32_t past_end{0};
    std::int32_t available_tokens{0};
};

TailSlots tailSlots(const GroupGeometry& geometry, std::int32_t prefix_granularity, std::int32_t tokens) {
    const std::int32_t block_granularity = geometry.BlockGranularity();
    const std::int32_t page_tokens = tokens - tokens % prefix_granularity;
    const std::int32_t past_end = (tokens + block_granularity - 1) / block_granularity;
    return TailSlots{.first = page_tokens / block_granularity,
                     .past_end = past_end,
                     .available_tokens = past_end * block_granularity - tokens};
}

class AdmissionPlanner {
public:
    AdmissionPlanner(const std::vector<CacheGroup>& groups, std::span<const GroupGeometry> geometry,
                     std::int32_t prefix_granularity, const BlockPool& pool, const BlockPool* host_pool,
                     std::span<const GroupDemand> demands, std::optional<std::int32_t> num_computed_tokens,
                     const CompletedPages* completed, const CacheCoordinator::PrefixProbe& prefix,
                     std::span<const PlannedShape> shapes,
                     std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed,
                     std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims)
        : groups_{groups},
          geometry_{geometry},
          prefix_granularity_{prefix_granularity},
          pool_{pool},
          host_pool_{host_pool},
          demands_{demands},
          num_computed_tokens_{num_computed_tokens},
          completed_{completed},
          prefix_{prefix},
          shapes_{shapes},
          claimed_{claimed},
          victims_{victims},
          local_free_slots_by_bucket_(groups.size()),
          shape_need_(groups.size()),
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
            const std::optional<VictimCandidate> candidate = nextVictimCandidate();
            if (!candidate) {
                return false;
            }
            removeOccupant(candidate->group_id, candidate->location);
            victims_.emplace_back(candidate->group_id, candidate->location);
        }

        // Restore newer blocks first, keeping each one unless its capacity is
        // still needed. All releases and restores affect only shadow occupancy.
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

    static bool shouldEvictFirst(const VictimCandidate& lhs, const VictimCandidate& rhs) {
        return evictionKey(lhs) < evictionKey(rhs);
    }

    struct CacheGroupVictimCandidates {
        explicit CacheGroupVictimCandidates(std::uint32_t id) : group_id{id} {}

        const VictimCandidate& CurrentCandidate() const { return candidates[next_candidate_index]; }

        std::uint32_t group_id;
        bool index_exhausted{false};
        PrefixCacheIndex::EvictionCursor index_cursor;
        std::vector<VictimCandidate> candidates;
        std::size_t next_candidate_index{0};
    };

    void initializeCapacity() {
        _assert(demands_.size() == groups_.size(), "demands/groups size mismatch");
        _assert(shapes_.empty() || shapes_.size() == groups_.size(), "shapes/groups size mismatch");
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const GroupDemand& demand = demands_[i];
            _assert(demand.table != nullptr, "group demand requires a block table");
            const auto buckets = static_cast<std::size_t>(groups_[i].Allocator().ShardCount());
            if (shapes_.empty()) {
                // An admission grows the live table densely; each Host hit
                // and each Device tail copy lands on a fresh Device block in
                // its source block's own bucket, exactly like a restored slot.
                shape_need_[i].assign(buckets, 0);
                blocks_needed_[i] = geometry_[i].BlocksNeededFor(*demand.table, demand) +
                                    tailBlocksNeeded(i, shape_need_[i]) + zeroStateBlocksNeeded(i);
                if (collectOwnerCopyBuckets(i, shape_need_[i])) {
                    owner_copy_changes_bucket_ = true;
                }
                if (!prefix_.host.per_group.empty()) {
                    collectHostHitBuckets(i, shape_need_[i]);
                }
            } else {
                // A restore rebuilds the recorded shape, bucket for bucket,
                // and grows the rebuilt table from its recorded tail.
                _assert(demand.table->NumBlocks() == 0, "a restore rebuilds into an empty table");
                _assert(shapes_[i].blocks_by_bucket.size() == buckets,
                        "restore shape buckets must match the shard count");
                if (const auto* step = std::get_if<StateStep>(&demand.extent)) {
                    _assert(step->kind == StateStep::Kind::kKeep, "a restore rebuilds a State table as imaged");
                    blocks_needed_[i] = 0;
                } else {
                    const auto* dense = std::get_if<DenseGrowth>(&demand.extent);
                    _assert(dense != nullptr, "a restore demand grows the rebuilt table densely");
                    blocks_needed_[i] = geometry_[i].BlocksNeededBeyond(shapes_[i].available_tokens,
                                                                        dense->num_tokens + demand.reserve_tokens);
                }
                shape_need_[i] = shapes_[i].blocks_by_bucket;
            }
            const std::span<const std::int64_t> holes = pool_.FreeSlotsByBucket(static_cast<std::uint32_t>(i));
            local_free_slots_by_bucket_[i].assign(holes.begin(), holes.end());
            if (local_free_slots_by_bucket_[i].empty()) {
                local_free_slots_by_bucket_[i].assign(buckets, 0);  // never allocated: no bound parent
            }
        }
        empty_parent_count_ = pool_.NumEmptyLcmBlocks();
    }

    // Blocks a tail costs this group beyond the growth demand. A table never
    // keeps a tail entry's block: the block holds the rows of a page that is
    // later published under the page's own hash, and one block carries one
    // key. So on a first admission every block of a Device tail becomes the
    // hitter's own (a copy of the shared block, or a fresh block where the
    // probe left a hole), and an owner that keeps writing past a published
    // tail moves every published tail block to a fresh one
    // (collectOwnerCopyBuckets). A copy takes its fresh block in its source's
    // bucket (SwapInFreshBlock), so it is added to `need` like a Host hit; a
    // hole's fresh block takes any bucket and is returned with the growth. A
    // dense growth starts from the capacity the last tail block leaves, which
    // the live table cannot show yet; a sparse suffix (the D role's landing
    // of a sliding window) starts at an absolute block past the tail and is
    // costed whole by BlocksNeededFor.
    std::int64_t tailBlocksNeeded(std::size_t i, std::vector<std::int64_t>& need) const {
        if (!geometry_[i].HasTokenSpan()) {
            return 0;  // a State group's step is costed by BlocksNeededFor
        }
        const std::int32_t block_granularity = geometry_[i].BlockGranularity();
        const GroupAllocator& allocator = groups_[i].Allocator();
        std::int64_t blocks = 0;
        if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
            const std::vector<CacheBlockLocation> copied =
                groups_[i].Index().MatchedLocations(pool_, tail->group_keys[i], /*begin_blocks=*/0, tail->per_group[i]);
            for (const CacheBlockLocation& source : copied) {
                ++need[static_cast<std::size_t>(allocator.BucketOf(source))];
            }
            blocks += static_cast<std::int64_t>(tail->group_keys[i].size()) - static_cast<std::int64_t>(copied.size()) -
                      FirstTailBlockNeeded(groups_[i].Spec(), block_granularity, tail->tail_tokens);
            if (const auto* dense = std::get_if<DenseGrowth>(&demands_[i].extent)) {
                const std::int32_t partial_rows = tail->tail_tokens % block_granularity;
                const std::int32_t available = partial_rows == 0 ? 0 : block_granularity - partial_rows;
                const std::int32_t growth = dense->num_tokens + demands_[i].reserve_tokens;
                blocks += geometry_[i].BlocksNeededBeyond(available, growth) -
                          geometry_[i].BlocksNeededFor(*demands_[i].table, growth);
            } else {
                const auto* sparse = std::get_if<SparseSuffix>(&demands_[i].extent);
                _assert(sparse != nullptr && sparse->first_block * block_granularity >= tail->tail_tokens,
                        "a sparse suffix lands past the hit's tail blocks");
            }
        }
        return blocks;
    }

    // Adds to `need`, in its bucket, each block Admit's copy-in reads for an
    // owner that keeps writing past the tail it publishes: the tail page's
    // slots whose retention has not expired (Admit reclaims the expired ones
    // first), each the canonical block when another request published the
    // tail first (the publication swaps it into the table), else the owner's
    // own. Returns whether a copy reads a canonical block in another bucket
    // than the owner's own block (protectDedupedTail).
    bool collectOwnerCopyBuckets(std::size_t i, std::vector<std::int64_t>& need) const {
        if (completed_ == nullptr || !completed_->owner_continues || !completed_->tail ||
            !geometry_[i].HasTokenSpan() || groups_[i].Spec().replayable) {
            return false;
        }
        const GroupAllocator& allocator = groups_[i].Allocator();
        const BlockTable& table = *demands_[i].table;
        const std::int32_t computed = *num_computed_tokens_;
        const TailSlots slots = tailSlots(geometry_[i], prefix_granularity_, computed);
        bool copy_changes_bucket = false;
        for (std::int32_t slot = std::max(slots.first, geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), computed));
             slot < std::min(slots.past_end, table.NumBlocks()); ++slot) {
            const CacheBlockRef& block = table.Blocks()[static_cast<std::size_t>(slot)];
            if (!block) {
                continue;
            }
            const CacheBlockRef canonical =
                groups_[i].Index().Find(pool_, CacheKey{.group_id = groups_[i].Id(),
                                                        .content_hash = completed_->tail->hash,
                                                        .page_offset = slot - slots.first});
            const std::int32_t bucket = allocator.BucketOf(canonical ? canonical->Location() : block->Location());
            ++need[static_cast<std::size_t>(bucket)];
            if (bucket != allocator.BucketOf(block->Location())) {
                copy_changes_bucket = true;
            }
        }
        return copy_changes_bucket;
    }

    // Whether the probe matched a checkpoint for State group i on either
    // tier, at a page boundary, a tail or (the D role) the prompt end.
    bool stateCheckpointMatched(std::size_t i) const {
        const auto any_hit = [](const GroupPrefixProbe& probe) {
            return std::ranges::any_of(probe.hits, [](std::uint8_t hit) { return hit != 0; });
        };
        const auto tier_hit = [&](const CacheCoordinator::PrefixProbe::Tier& tier) {
            return (!tier.per_group.empty() && any_hit(tier.per_group[i])) ||
                   (tier.tail && !tier.tail->per_group.empty() && any_hit(tier.tail->per_group[i]));
        };
        return tier_hit(prefix_.device) || tier_hit(prefix_.host) ||
               (prefix_.state_at_prompt_end && any_hit(prefix_.state_at_prompt_end->per_group[i]));
    }

    // A State group's first rotation without a checkpoint to read starts from
    // the zero state, held as a fresh zeroed block in slot 0: the table holds
    // its two blocks from the first admission on, and every later rotation
    // swaps the released input for the fresh live slot.
    std::int64_t zeroStateBlocksNeeded(std::size_t i) const {
        if (geometry_[i].HasTokenSpan() || demands_[i].table->NumBlocks() != 0) {
            return 0;
        }
        const auto* step = std::get_if<StateStep>(&demands_[i].extent);
        return step != nullptr && step->kind == StateStep::Kind::kRotate && !stateCheckpointMatched(i) ? 1 : 0;
    }

    // A Host hit's Device destination must share the Host block's bucket, so
    // the copy's two ends have one owner under page-cyclic sharding. A hit
    // with no Host block yet (an L3 prefetch, replicated groups only) takes
    // bucket 0, where its block will be allocated. A Host tail's blocks land
    // the same way, into the hitter's private pages.
    void collectHostHitBuckets(std::size_t group_index, std::vector<std::int64_t>& need) const {
        _assert(host_pool_ != nullptr, "Host prefix hits require a Host pool");
        const GroupPrefixProbe& probe = prefix_.host.per_group[group_index];
        const std::span<const CacheKey> keys = prefix_.group_keys[group_index];
        const std::int32_t floor_pages =
            prefix_.device.full_page_tokens /
            (geometry_[group_index].HasTokenSpan() ? geometry_[group_index].BlockGranularity() : prefix_granularity_);
        const GroupAllocator& allocator = groups_[group_index].Allocator();
        const auto count_hit = [&](const CacheKey& key) {
            const CacheBlockRef host_block = groups_[group_index].Index().Find(*host_pool_, key);
            const std::int32_t bucket = host_block ? allocator.BucketOf(host_block->Location()) : 0;
            ++need[static_cast<std::size_t>(bucket)];
        };
        for (std::size_t hit_index = 0; hit_index < probe.hits.size(); ++hit_index) {
            if (probe.hits[hit_index] == 0) {
                continue;
            }
            const std::size_t key_index = static_cast<std::size_t>(floor_pages) + hit_index;
            _assert(key_index < keys.size(), "host prefix hit is outside the probed key range");
            count_hit(keys[key_index]);
        }
        if (const auto& tail = prefix_.host.tail; tail && !tail->group_keys[group_index].empty()) {
            const GroupPrefixProbe& tail_probe = tail->per_group[group_index];
            for (std::size_t j = 0; j < tail_probe.hits.size(); ++j) {
                if (tail_probe.hits[j] != 0) {
                    count_hit(tail->group_keys[group_index][j]);
                }
            }
        }
    }

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
        };
    }

    void collectCandidates() {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::vector<CacheBlockLocation> hits = groups_[i].Index().MatchedLocations(
                pool_, prefix_.group_keys[i], /*begin_blocks=*/0, prefix_.device.per_group[i]);
            protected_locations_.insert(hits.begin(), hits.end());
            if (const auto& tail = prefix_.device.tail; tail && !tail->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> tail_hits = groups_[i].Index().MatchedLocations(
                    pool_, tail->group_keys[i], /*begin_blocks=*/0, tail->per_group[i]);
                protected_locations_.insert(tail_hits.begin(), tail_hits.end());
            }
            if (const auto& state = prefix_.state_at_prompt_end; state && !state->group_keys[i].empty()) {
                const std::vector<CacheBlockLocation> state_hits = groups_[i].Index().MatchedLocations(
                    pool_, state->group_keys[i], /*begin_blocks=*/0, state->per_group[i]);
                protected_locations_.insert(state_hits.begin(), state_hits.end());
            }
        }
        protectDedupedTail();
        for (const auto& [group_id, location] : claimed_) {
            (void)group_id;
            protected_locations_.insert(location);
        }

        if (num_computed_tokens_) {
            collectRequestReclaimCandidates(*num_computed_tokens_);
        }

        cache_group_candidates_.reserve(groups_.size());
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            cache_group_candidates_.emplace_back(static_cast<std::uint32_t>(i));
        }
    }

    // An owner's copy-in that reads the canonical block of a tail another
    // request published first keeps that block cached until the publication
    // dedupes onto it, and with it every other Device entry under the tail's
    // hash, State checkpoints included: evicting any one of them evicts the
    // whole tail (CacheCoordinator::evictCachedBlock), and the publication
    // would then keep the owner's own block and copy that one, in its own
    // bucket rather than the canonical block's. A canonical block in the
    // owner's own bucket needs no protection: either copy lands in the bucket
    // the plan charged.
    void protectDedupedTail() {
        if (!owner_copy_changes_bucket_) {
            return;
        }
        const TailPublication& tail = *completed_->tail;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            // A State group keys one checkpoint under the tail's hash.
            const std::int32_t blocks =
                geometry_[i].HasTokenSpan() ? geometry_[i].BlocksNeededBeyond(0, tail.tail_tokens) : 1;
            for (std::int32_t offset = 0; offset < blocks; ++offset) {
                const CacheBlockRef block = groups_[i].Index().Find(
                    pool_, CacheKey{.group_id = groups_[i].Id(), .content_hash = tail.hash, .page_offset = offset});
                if (block) {
                    protected_locations_.insert(block->Location());
                }
            }
        }
    }

    // The request's own slots that its retention has expired at this
    // progress, and a State table's slot 0 when this step releases it. They
    // are still pinned by the table here; Admit publishes and reclaims them
    // before it evicts, so they can fund this same admission.
    void collectRequestReclaimCandidates(std::int32_t num_computed_tokens) {
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const std::uint32_t group_id = static_cast<std::uint32_t>(i);
            std::vector<CacheBlockLocation> reclaimable;
            if (const auto* step = std::get_if<StateStep>(&demands_[i].extent)) {
                if (const auto location =
                        groups_[i].Allocator().ReleasableStateInput(groups_[i].Index(), *demands_[i].table, *step)) {
                    reclaimable.push_back(*location);
                }
            } else {
                const std::int32_t expired_blocks =
                    geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), num_computed_tokens);
                reclaimable = groups_[i].Allocator().ReclaimableBlockLocationsAt(groups_[i].Index(), *demands_[i].table,
                                                                                 expired_blocks);
            }
            for (CacheBlockLocation location : reclaimable) {
                if (protected_locations_.contains(location)) {
                    continue;
                }
                const bool first_occurrence = request_reclaim_locations_.insert(location).second;
                if (!first_occurrence) {
                    continue;
                }
                request_reclaim_candidates_.push_back(
                    makeVictimCandidate(group_id, location, groups_[i].Index().MetadataFor(pool_, location)));
            }
        }
        std::ranges::sort(request_reclaim_candidates_, shouldEvictFirst);
    }

    bool ensureGroupCandidateAvailable(CacheGroupVictimCandidates& group) {
        if (group.next_candidate_index < group.candidates.size()) {
            return true;
        }
        group.candidates.clear();
        group.next_candidate_index = 0;
        while (!group.index_exhausted) {
            epoch_scratch_.clear();
            const bool found_epoch =
                groups_[group.group_id].Index().NextEvictionEpoch(pool_, group.index_cursor, epoch_scratch_);
            if (!found_epoch) {
                group.index_exhausted = true;
                break;
            }
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
                std::ranges::sort(group.candidates, shouldEvictFirst);
                return true;
            }
        }
        return false;
    }

    std::optional<VictimCandidate> nextVictimCandidate() {
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
    void removeOccupant(std::uint32_t group_id, CacheBlockLocation location) {
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

    void restoreOccupant(std::uint32_t group_id, CacheBlockLocation location) {
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

    // One fit formula for both transactions: the bucket-constrained demand
    // shares the parents it opens, the dense demand takes any hole, and an
    // admission with no bucket demand reduces to holes plus parents times
    // packing.
    bool fits() const {
        std::int64_t parents_needed = 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            parents_needed += ParentsNeededForBuckets(shape_need_[i], local_free_slots_by_bucket_[i], blocks_needed_[i],
                                                      groups_[i].Allocator().CacheBlocksPerLcmBlock());
        }
        return parents_needed <= empty_parent_count_;
    }

    const std::vector<CacheGroup>& groups_;
    std::span<const GroupGeometry> geometry_;
    std::int32_t prefix_granularity_;
    const BlockPool& pool_;
    const BlockPool* host_pool_;
    std::span<const GroupDemand> demands_;
    std::optional<std::int32_t> num_computed_tokens_;
    // The publication this admission carries (nullptr when none): an owner
    // that keeps writing past a tail needs a fresh block for it.
    const CompletedPages* completed_;
    const CacheCoordinator::PrefixProbe& prefix_;
    std::span<const PlannedShape> shapes_;
    // Device-cached blocks a restore claims instead of copying: protected
    // from eviction like the probe's prefix hits.
    std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed_;
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>>& victims_;
    std::unordered_map<std::int32_t, std::int32_t> remaining_occupied_;
    // Per group: holes per bucket in bound parents, the bucket-constrained
    // blocks (a restore's rebuilt slots; an admission's Host-hit destinations
    // and tail copies), and the dense blocks wanted in any bucket.
    std::vector<std::vector<std::int64_t>> local_free_slots_by_bucket_;
    std::vector<std::vector<std::int64_t>> shape_need_;
    std::vector<std::int64_t> blocks_needed_;
    // Whether an owner's copy-in reads a canonical block in another bucket
    // than its own (collectOwnerCopyBuckets, protectDedupedTail).
    bool owner_copy_changes_bucket_{false};
    std::int64_t empty_parent_count_{0};
    std::vector<VictimCandidate> request_reclaim_candidates_;
    std::size_t next_request_candidate_index_{0};
    std::vector<CacheGroupVictimCandidates> cache_group_candidates_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> protected_locations_;
    std::unordered_set<CacheBlockLocation, CacheBlockLocationHash> request_reclaim_locations_;
    std::vector<PrefixCacheIndex::EvictionCandidate> epoch_scratch_;
};

std::optional<AdmissionPlan> planAdmission(const std::vector<CacheGroup>& groups,
                                           std::span<const GroupGeometry> geometry, std::int32_t prefix_granularity,
                                           const BlockPool& pool, const BlockPool* host_pool,
                                           CacheCoordinator::PrefixProbe&& prefix, std::span<const GroupDemand> demands,
                                           const RequestProgress& progress, std::span<const PlannedShape> shapes,
                                           std::span<const std::pair<std::uint32_t, CacheBlockLocation>> claimed) {
    _assert(demands.size() == groups.size(), "demands/groups size mismatch");

    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> victims;
    AdmissionPlanner planner{groups,
                             geometry,
                             prefix_granularity,
                             pool,
                             host_pool,
                             demands,
                             progress.num_computed_tokens,
                             progress.completed_pages ? &*progress.completed_pages : nullptr,
                             prefix,
                             shapes,
                             claimed,
                             victims};
    if (!planner.Plan()) {
        return std::nullopt;
    }
    return AdmissionPlan{.prefix = std::move(prefix), .victims = std::move(victims)};
}

// The same progress without the tail it would publish: the full pages still
// publish; the tail, the owner's move off it and the state checkpoint keyed
// by it are dropped. Nothing left to publish leaves no CompletedPages.
RequestProgress withoutTailPublication(const RequestProgress& progress) {
    RequestProgress out = progress;
    CompletedPages& completed = *out.completed_pages;
    completed.tail.reset();
    completed.owner_continues = false;
    completed.publish_state = false;
    if (completed.first_new_prefix_page >= static_cast<std::int32_t>(completed.prefix_hashes.size())) {
        out.completed_pages.reset();
    }
    return out;
}

}  // namespace

bool CacheCoordinator::writableImageSlot(std::size_t group_index, const ImageTable& shape,
                                         std::int32_t slot_index) const {
    if (GroupIsState(group_index)) {
        return slot_index == 1;  // the live slot; slot 0 is only ever read
    }
    // The tail's blocks: the slots of the page the computed frontier ends
    // inside. The request writes the last of them and later publishes all of
    // them under the page's hash, so none may stay the tail entry's block.
    const std::int32_t block_granularity = geometry_[group_index].BlockGranularity();
    const std::int32_t num_tokens = shape.num_blocks * block_granularity - shape.available_tokens;
    const std::int32_t tail_start = (num_tokens - num_tokens % prefix_granularity_) / block_granularity;
    return slot_index >= tail_start;
}

void CacheCoordinator::ReleaseStateInputs(std::span<BlockTable> tables) {
    _assert(tables.size() == groups_.size(), "tables/groups size mismatch");
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i) && tables[i].NumBlocks() == 2) {
            groups_[i].Allocator().ReleaseStateInput(tables[i]);
        }
    }
}

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::Admit(
    PrefixProbe&& prefix, std::span<const GroupDemand> demands, const RequestProgress& progress,
    std::optional<std::uint64_t> request_access_epoch) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    for (std::size_t i = 0; i < demands.size(); ++i) {
        _assert(demands[i].table != nullptr, "group demand requires a block table");
        _assert(GroupIsState(i) == std::holds_alternative<StateStep>(demands[i].extent),
                "a State group takes a StateStep; a token-spanned group a dense or sparse extent");
    }
    validateProgress(progress);

    // A replayable group claims no hit pages, so before a hit its table is
    // empty: materialize it as a sparse private suffix from the replay
    // window's first token -- the slots below stay null holes, as
    // absolute-slot tables require -- and the model regenerates the rows.
    // Closed groups keep their dense demand beyond the hit. A demand that
    // already names a sparse suffix is a remote landing (the peer's retained
    // tail) and is left alone: nothing is regenerated here.
    std::vector<GroupDemand> replayed;
    const std::int32_t hit_tokens = std::max(prefix.device.num_common_tokens, prefix.host.num_common_tokens);
    if (replay_window_tokens_ > 0 && hit_tokens > 0) {
        const std::int32_t replay_begin = hit_tokens - ReplayTokens(hit_tokens);
        replayed.assign(demands.begin(), demands.end());
        for (std::size_t i = 0; i < replayed.size(); ++i) {
            const auto* dense = std::get_if<DenseGrowth>(&replayed[i].extent);
            if (!GroupIsReplayable(static_cast<std::int32_t>(i)) || dense == nullptr) {
                continue;
            }
            _assert(replayed[i].table->NumBlocks() == 0, "a replayable group holds no hit pages at admission");
            // The dense growth was relative to an empty table, so the hit plus
            // the growth is the absolute extent.
            replayed[i].extent = SparseSuffix{
                .extent_tokens = hit_tokens + dense->num_tokens,
                .first_block = replay_begin / geometry_[i].BlockGranularity(),
            };
        }
        demands = replayed;
    }

    // The publication this admission carries. An owner that keeps writing
    // past a tail it publishes needs a fresh copy of every published block;
    // when the pool cannot fund those copies the owner keeps its partial
    // block private instead and publishes nothing at this point -- neither
    // the tail nor the state checkpoint keyed by it -- so a request whose
    // reserve covers its generation never waits on a block it did not
    // reserve. Its finish publishes the frontier later.
    RequestProgress effective = progress;
    const bool copies_in =
        progress.completed_pages && progress.completed_pages->owner_continues && progress.completed_pages->tail;
    std::optional<AdmissionPlan> candidate =
        planAdmission(groups_, geometry_, prefix_granularity_, pool_, host_pool_,
                      copies_in ? PrefixProbe{prefix} : PrefixProbe{std::move(prefix)}, demands, effective,
                      /*shapes=*/{}, /*claimed=*/{});
    if (!candidate && copies_in) {
        effective = withoutTailPublication(progress);
        candidate = planAdmission(groups_, geometry_, prefix_granularity_, pool_, host_pool_, std::move(prefix),
                                  demands, effective, /*shapes=*/{}, /*claimed=*/{});
    }
    if (!candidate) {
        return std::nullopt;
    }
    AdmissionPlan plan = std::move(*candidate);

    if (request_access_epoch.has_value()) {
        _assert(*request_access_epoch > 0 && *request_access_epoch <= next_access_epoch_,
                "request access epoch was not issued by this coordinator");
    }
    const std::uint64_t access_epoch = request_access_epoch.has_value() ? *request_access_epoch : ++next_access_epoch_;
    // The boundaries inside a block, per tier: the hitter's table ends in a
    // private copy of the Device tail's partial block, or in a fresh page the
    // Host tail is loaded into.
    const std::int32_t device_tail_tokens = plan.prefix.device.tail ? plan.prefix.device.tail->tail_tokens : 0;
    const std::int32_t host_tail_tokens = plan.prefix.host.tail ? plan.prefix.host.tail->tail_tokens : 0;
    const std::int32_t admitted_tokens =
        std::max(plan.prefix.device.num_common_tokens, plan.prefix.host.num_common_tokens);
    const std::int32_t kv_tokens = std::max(plan.prefix.device.kv_tokens, plan.prefix.host.kv_tokens);
    const bool state_prefix_cached = plan.prefix.state_at_prompt_end.has_value();
    AcquiredPrefix acquired_prefix = acquirePrefix(std::move(plan.prefix), access_epoch);
    AdmissionResult result{
        .device_prefix_tokens = acquired_prefix.device.num_common_tokens,
        .host_prefix_tokens = acquired_prefix.host.num_common_tokens,
        .junction_tokens = kv_tokens > admitted_tokens ? kv_tokens : 0,
        .state_prefix_cached = state_prefix_cached,
        .access_epoch = access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(groups_.size()),
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (GroupIsState(i)) {
            continue;  // a State hit is installed below, as slot 0
        }
        if (acquired_prefix.device.num_common_tokens > 0) {
            groups_[i].Allocator().ClaimHitBlocks(*demands[i].table, std::move(acquired_prefix.device.per_group[i]));
        }
    }
    // A State group's first admission shapes its two slots: slot 0 the one
    // checkpoint the probe matched on the Device (a Host checkpoint lands by
    // load below; the D's prompt-end checkpoint is claimed here), else null
    // until the step below installs the zero state; slot 1 is acquired with
    // the step below. The shaping is the first rotation: a shaped-now table
    // rotates nothing further.
    std::vector<bool> shaped_now(groups_.size(), false);
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        if (!GroupIsState(i) || demands[i].table->NumBlocks() != 0) {
            continue;
        }
        const StateStep::Kind kind = std::get<StateStep>(demands[i].extent).kind;
        if (kind != StateStep::Kind::kRotate && kind != StateStep::Kind::kLand) {
            continue;
        }
        shaped_now[i] = true;
        CacheBlockRef input;
        for (CacheBlockRef& block : acquired_prefix.device.per_group[i].blocks) {
            if (block) {
                _assert(!input, "a State group matches one checkpoint");
                input = std::move(block);
            }
        }
        if (!acquired_prefix.state_at_prompt_end.empty() && acquired_prefix.state_at_prompt_end[i]) {
            _assert(!input, "a State group matches one checkpoint");
            input = std::move(acquired_prefix.state_at_prompt_end[i]);
        }
        groups_[i].Allocator().ShapeStateTable(*demands[i].table, std::move(input));
    }
    // Gives the table a private block in `slot`: a copy of the block there
    // (the hitter's copy-on-write, the owner's copy-in), or a fresh zeroed
    // one where the slot was a hole.
    const auto privatize_slot = [&](std::size_t i, BlockTable& table, std::int32_t slot,
                                    std::int32_t available_tokens) {
        GroupAllocator& allocator = groups_[i].Allocator();
        const CacheBlockRef replaced = allocator.SwapInFreshBlock(pool_, table, slot, available_tokens);
        const std::int32_t destination =
            allocator.ResolveCacheBlockId(table.Blocks()[static_cast<std::size_t>(slot)]->Location());
        if (replaced) {
            result.pages_to_copy[i].emplace_back(allocator.ResolveCacheBlockId(replaced->Location()), destination);
        } else {
            result.new_page_ids[i].push_back(destination);
        }
    };
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
        BlockTable& table = *demands[i].table;
        if (effective.completed_pages) {
            cacheDeviceCompletedBlocksForGroup(i, table, *effective.completed_pages, access_epoch);
        }
        if (GroupIsState(i)) {
            // The step's release: slot 0 leaves the table (the index keeps it
            // when published, else it returns to the pool and may fund this
            // very admission); a rotation moves the block the last forward
            // wrote -- published just above when the point was a checkpoint
            // -- into slot 0 and leaves slot 1 for the acquisition below.
            if (table.NumBlocks() == 0) {
                continue;
            }
            switch (std::get<StateStep>(demands[i].extent).kind) {
                case StateStep::Kind::kRotate:
                    if (!shaped_now[i]) {
                        groups_[i].Allocator().RotateState(table);
                    }
                    break;
                case StateStep::Kind::kRelease:
                    groups_[i].Allocator().ReleaseStateInput(table);
                    break;
                case StateStep::Kind::kKeep:
                case StateStep::Kind::kLand:
                    break;
            }
            continue;
        }
        if (effective.num_computed_tokens) {
            groups_[i].Allocator().ReclaimExpired(
                pool_, table, geometry_[i].ExpiredBlocksAt(groups_[i].Spec(), *effective.num_computed_tokens));
        }
    }
    if (effective.completed_pages) {
        recordPublishedTail(*effective.completed_pages);
    }
    for (const auto& [group_id, location] : prospective_victims) {
        if (!evictCachedBlock(group_id, location)) {
            FatalCheck(!pool_.IsOccupied(location), "admission victim changed before acquisition");
        }
    }

    for (std::size_t i = 0; i < groups_.size(); ++i) {
        // A group's bucket-exact blocks (the owner's copy-in, a hitter's tail
        // copies, the Host extension) are acquired before its dense growth,
        // which is what the planner's fit assumes.
        const GroupDemand& demand = demands[i];
        BlockTable& table = *demand.table;
        const CompletedPages* completed = effective.completed_pages ? &*effective.completed_pages : nullptr;
        // An owner that keeps writing past the publication point moves off
        // every block it just published: a published block is never written
        // again and never carries a second key. (A State group's published
        // slot 1 moves to slot 0 by rotation instead.)
        if (!GroupIsState(i) && completed != nullptr && completed->owner_continues && completed->tail &&
            effective.num_computed_tokens) {
            const TailSlots slots = tailSlots(geometry_[i], prefix_granularity_, *effective.num_computed_tokens);
            for (std::int32_t slot = slots.first; slot < std::min(slots.past_end, table.NumBlocks()); ++slot) {
                if (groups_[i].Index().Contains(table.Blocks()[static_cast<std::size_t>(slot)])) {
                    privatize_slot(i, table, slot, table.AvailableTokens());
                }
            }
        }
        // A first admission's table ends in the tail's blocks: shared full
        // pages, then this request's own copies.
        if (!GroupIsState(i) && device_tail_tokens > 0 && table.NumBlocks() > 0) {
            const TailSlots slots =
                tailSlots(geometry_[i], prefix_granularity_, acquired_prefix.device.num_common_tokens);
            _assert(slots.past_end == table.NumBlocks(), "the Device tail's blocks end the claimed table");
            // A sliding group's tail blocks below its window stay holes.
            const std::int32_t first =
                slots.first + FirstTailBlockNeeded(static_cast<std::int32_t>(i), device_tail_tokens);
            for (std::int32_t slot = first; slot < slots.past_end; ++slot) {
                privatize_slot(i, table, slot, slots.available_tokens);
            }
        }
        if (GroupIsState(i)) {
            const StateStep& step = std::get<StateStep>(demand.extent);
            GroupAllocator& allocator = groups_[i].Allocator();
            if (table.NumBlocks() == 0) {
                continue;  // never shaped: a step other than a rotation leaves it so
            }
            // A Host checkpoint is loaded into a fresh slot 0.
            if (!acquired_prefix.host.per_group.empty()) {
                for (CacheBlockRef& host_block : acquired_prefix.host.per_group[i].blocks) {
                    if (!host_block) {
                        continue;
                    }
                    _assert(!table.Blocks()[0], "a State group matches one checkpoint");
                    std::vector<CacheBlockRef> destination = pool_.AcquireBlocksInBuckets(
                        groups_[i].Id(), std::array<std::int32_t, 1>{allocator.BucketOf(host_block->Location())});
                    FatalCheck(destination.size() == 1, "admission plan no longer fits the block pool");
                    allocator.InstallStateInput(table, std::move(destination.front()));
                    result.load_pairs.push_back(BlockTransfer{.group_id = groups_[i].Id(),
                                                              .source = std::move(host_block),
                                                              .destination = table.Blocks()[0]});
                }
            }
            // A prompt with no checkpoint to read starts from the zero state:
            // a fresh zeroed block in slot 0, so the request holds its two
            // blocks from its first admission on.
            if (step.kind == StateStep::Kind::kRotate && shaped_now[i] && !table.Blocks()[0]) {
                FatalCheck(allocator.AcquireStateInput(pool_, table), "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(table.Blocks()[0]->Location()));
            }
            if (step.kind == StateStep::Kind::kRotate || step.kind == StateStep::Kind::kLand) {
                FatalCheck(allocator.AcquireStateLive(pool_, table), "admission plan no longer fits the block pool");
                result.new_page_ids[i].push_back(allocator.ResolveCacheBlockId(table.Blocks()[1]->Location()));
            }
            continue;
        }
        if (!acquired_prefix.host.per_group.empty() && !acquired_prefix.host.per_group[i].blocks.empty()) {
            const std::int32_t available = host_tail_tokens > 0 ? tailSlots(geometry_[i], prefix_granularity_,
                                                                            acquired_prefix.host.num_common_tokens)
                                                                      .available_tokens
                                                                : 0;
            // A Host tail's blocks land on fresh destinations (one load row
            // each); a sliding group's tail blocks below its window stay holes.
            groups_[i].Allocator().AppendHostExtension(
                pool_, table, std::move(acquired_prefix.host.per_group[i].blocks), result.load_pairs, available);
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

std::optional<CacheCoordinator::AdmissionResult> CacheCoordinator::Restore(const RetractionImage& image,
                                                                           std::span<const GroupDemand> demands,
                                                                           std::uint64_t request_access_epoch) {
    _assert(demands.size() == groups_.size(), "demands/groups size mismatch");
    _assert(image.tables.size() == groups_.size(), "image/groups size mismatch");
    _assert(request_access_epoch > 0 && request_access_epoch <= next_access_epoch_,
            "request access epoch was not issued by this coordinator");
    // A Host L2 slot whose key still has a Device-cached canonical block is
    // claimed instead of copied: the same bytes by construction (the victim's
    // own block, cache-only since it was freed, or the canonical block its
    // table already pointed at). Those blocks are protected from the planner's
    // eviction like prefix hits; every other slot is a bucket-constrained
    // fresh block.
    // A published slot the resumed request will never write again is claimed
    // from the Device index when its key is still cached there; the slot it
    // keeps writing -- the partial page at its computed frontier -- always
    // gets a fresh copy, so the published block stays immutable.
    const auto claimable = [&](std::size_t i, const ImageTable& shape, const ImageSlot& slot) {
        return slot.InHostCache() && !writableImageSlot(i, shape, slot.slot_index);
    };
    std::vector<PlannedShape> shapes(groups_.size());
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> protected_locations;
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        _assert(demands[i].table != nullptr && demands[i].table->NumBlocks() == 0,
                "a restore rebuilds into an empty table");
        const GroupAllocator& allocator = groups_[i].Allocator();
        const ImageTable& shape = image.tables[i];
        shapes[i].blocks_by_bucket.assign(static_cast<std::size_t>(allocator.ShardCount()), 0);
        shapes[i].available_tokens = shape.available_tokens;
        for (const ImageSlot& slot : shape.slots) {
            _assert(static_cast<bool>(slot.block), "image slot lost its block");
            const CacheBlockRef canonical =
                claimable(i, shape, slot) ? groups_[i].Index().Find(pool_, slot.key) : CacheBlockRef{};
            if (canonical) {
                protected_locations.emplace_back(groups_[i].Id(), canonical->Location());
                continue;
            }
            ++shapes[i].blocks_by_bucket[static_cast<std::size_t>(allocator.BucketOf(slot.block->Location()))];
        }
    }

    std::optional<AdmissionPlan> candidate =
        planAdmission(groups_, geometry_, prefix_granularity_, pool_, host_pool_, ProbePrefix({}), demands,
                      RequestProgress{}, shapes, protected_locations);
    if (!candidate) {
        return std::nullopt;
    }
    // Claim before evicting, as Admit does: a claimed entry is pinned, so a
    // victim's eviction cascade (a tail's companions, a page's state
    // checkpoints) cannot take a block the image is about to resume on.
    std::vector<std::vector<CacheBlockRef>> claimed(groups_.size());
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        const ImageTable& shape = image.tables[i];
        claimed[i].resize(shape.slots.size());
        for (std::size_t j = 0; j < shape.slots.size(); ++j) {
            const ImageSlot& slot = shape.slots[j];
            if (!claimable(i, shape, slot) || !groups_[i].Index().Find(pool_, slot.key)) {
                continue;
            }
            // The planner protected this entry; claim it like a prefix hit.
            const std::array<CacheKey, 1> keys{slot.key};
            PrefixMatch hit = groups_[i].Index().AcquireMatched(pool_, keys, /*begin_blocks=*/0,
                                                                GroupPrefixProbe{.hits = {1}}, request_access_epoch);
            claimed[i][j] = std::move(hit.blocks.front());
        }
    }
    // Nothing is published or reclaimed here, so every planned victim is a
    // cache-only block the planner saw as evictable (or a tail companion an
    // earlier victim's eviction already took).
    for (const auto& [group_id, location] : candidate->victims) {
        if (!evictCachedBlock(group_id, location)) {
            FatalCheck(!pool_.IsOccupied(location), "restore victim changed before acquisition");
        }
    }

    AdmissionResult result{
        .access_epoch = request_access_epoch,
        .new_page_ids = std::vector<std::vector<std::int32_t>>(groups_.size()),
        .pages_to_copy = std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>>(groups_.size()),
    };
    for (std::size_t i = 0; i < groups_.size(); ++i) {
        GroupAllocator& allocator = groups_[i].Allocator();
        BlockTable& table = *demands[i].table;
        const ImageTable& shape = image.tables[i];
        FatalCheck(allocator.AcquireShape(pool_, table, shape, std::move(claimed[i])),
                   "restore plan no longer fits the block pool");
        for (const ImageSlot& slot : shape.slots) {
            const CacheBlockRef& destination = table.Blocks()[static_cast<std::size_t>(slot.slot_index)];
            if (slot.InHostCache() && groups_[i].Index().Contains(destination)) {
                continue;  // claimed: already the published Device block, nothing to copy
            }
            // The writable slot's row is unkeyed: its destination is the
            // request's private copy and must not be republished at the ACK.
            const bool republish = claimable(i, shape, slot);
            BlockTransfer transfer{
                .group_id = groups_[i].Id(),
                .source = slot.block,
                .destination = destination,
                .key = republish ? slot.key : CacheKey{},
                .logical_block_index = slot.logical_block_index,
                .boundary_kind = slot.boundary_kind,
            };
            // A published slot comes back from Host L2 (and is re-published at
            // the ACK); a private slot from the snapshot pool. Neither is
            // listed in new_page_ids: the copy fills the whole block, as a
            // Host hit's load-back does.
            if (slot.InHostCache()) {
                result.load_pairs.push_back(std::move(transfer));
            } else {
                result.snapshot_pairs.push_back(std::move(transfer));
            }
        }
        if (GroupIsState(i)) {
            continue;  // the imaged two slots are the whole table
        }
        // Only the reserve appended beyond the imaged shape is fresh, unfilled
        // Device memory for the plan to zero.
        const std::int32_t first_new_block = table.NumBlocks();
        FatalCheck(allocator.Acquire(pool_, table, geometry_[i].PlanAcquire(table, demands[i])),
                   "restore plan no longer fits the block pool");
        const std::span<const CacheBlockRef> blocks = table.Blocks();
        for (std::int32_t block = first_new_block; block < table.NumBlocks(); ++block) {
            result.new_page_ids[i].push_back(
                allocator.ResolveCacheBlockId(blocks[static_cast<std::size_t>(block)]->Location()));
        }
    }
    return result;
}

}  // namespace tokenspeed
