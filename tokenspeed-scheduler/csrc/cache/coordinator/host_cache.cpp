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

#include "cache/coordinator/host_cache.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "cache/coordinator/eviction_candidates.h"
#include "cache/coordinator/eviction_plan.h"
#include "cache/coordinator/flat_hash_map.h"
#include "cache/core/block_placement_demand.h"
#include "utils.h"

namespace tokenspeed {

namespace {

class HostEvictionPlanner {
public:
    struct Member {
        std::uint32_t group_id;
        CacheBlockLocation location;
    };
    struct VictimSet {
        std::vector<Member> members;
    };

    HostEvictionPlanner(const CacheStore& store, const AdmissionHistory& history, const std::vector<CacheGroup>& groups,
                        const BlockPool& pool)
        : store_{store},
          groups_{groups},
          pool_{pool},
          plan_{pool, groups},
          candidates_{groups, pool, history.CheckpointAgingFor(CacheTier::kHost)} {}

    bool Fits(std::span<const BlockPlacementDemand> block_demands) const { return plan_.Fits(block_demands); }

    void Seed(std::uint32_t group_id, CacheBlockLocation location) {
        taken_.insert(location);
        plan_.Remove(group_id, location);
    }

    bool Fit(std::span<const BlockPlacementDemand> block_demands) {
        const CacheEvictionCandidates::LocationSet no_request_reclaims{&scratch_};
        return plan_.SelectUntilFits(block_demands, [&] {
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
                // Host State eviction takes the other unpinned State groups at this boundary.
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
    HostCache::HostAllocationStats Stats() const {
        return {.group_scans = candidates_.GroupScans(),
                .block_candidates = candidates_.ScannedCandidates(),
                .parent_candidates = plan_.TouchedParents()};
    }

private:
    const CacheStore& store_;
    const std::vector<CacheGroup>& groups_;
    const BlockPool& pool_;
    CacheEvictionPlan plan_;
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

}  // namespace

std::vector<HostCache::StoreCandidate> HostCache::TakePendingStores() {
    auto pending = std::exchange(pending_stores_, {});
    std::erase_if(pending, [this](const StoreCandidate& candidate) {
        const auto& index = store_.Groups()[candidate.key.group_id].Index();
        auto location = store_.LocationOf(store_.DevicePool(), candidate.key.group_id, candidate.key);
        if (!location) return false;
        auto metadata = index.MetadataFor(store_.DevicePool(), *location);
        return metadata && metadata->session && metadata->session->ended;
    });
    return pending;
}

void HostCache::QueueCachedBlocksForStore(std::span<const std::string> content_hashes) {
    if (store_.HostPool() == nullptr) {
        return;
    }
    for (const CacheGroup& group : store_.Groups()) {
        const PrefixCacheIndex& index = group.Index();
        // A tail hash holds fewer blocks than a page; Contains filters.
        for (CacheKey& key : store_.KeysForGroup(content_hashes, group.Id())) {
            if (!index.Contains(store_.DevicePool(), key)) {
                continue;
            }
            // Read only with demotion on: off, this stays one lookup.
            if (evictor_.DemotesCheckpoints() &&
                evictor_.IsDemotedEntry(index.Inspect(store_.DevicePool(), key)->metadata)) {
                continue;
            }
            pending_stores_.push_back(StoreCandidate{.key = std::move(key)});
        }
    }
}

HostCache::HostAllocationBatch HostCache::AcquireHostBlocks(std::span<const std::uint32_t> group_ids,
                                                            std::span<const std::int32_t> buckets) {
    _assert(store_.HostPool() != nullptr, "AcquireHostBlocks requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");
    HostAllocationBatch batch;
    batch.blocks.resize(group_ids.size());
    batch.stats.requested = group_ids.size();
    _assert(group_ids.size() <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "Host allocation batch request exceeds int32 range");
    const auto acquire_in_bucket = [&](std::uint32_t group_id, std::int32_t bucket) {
        const std::array<std::int32_t, 1> one{bucket};
        std::vector<CacheBlockRef> refs = store_.HostPool()->AcquireBlocksInBuckets(group_id, one);
        return refs.empty() ? CacheBlockRef{} : std::move(refs.front());
    };
    std::vector<std::size_t> unresolved;
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < store_.Groups().size(), "Host block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < store_.Groups()[group_ids[row]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
        if (!batch.blocks[row]) unresolved.push_back(row);
    }
    if (!unresolved.empty()) {
        std::vector<HostEvictionPlanner::VictimSet> victims;
        std::vector<std::size_t> accepted;
        {
            HostEvictionPlanner planner{store_, history_, store_.Groups(), *store_.HostPool()};
            std::vector<BlockPlacementDemand> block_demands(store_.Groups().size());
            for (std::size_t i = 0; i < store_.Groups().size(); ++i)
                block_demands[i].blocks_by_bucket.resize(store_.Groups()[i].Allocator().ShardCount());
            for (const std::size_t row : unresolved) {
                auto& bucket_blocks = block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
                ++bucket_blocks;
                if (planner.Fit(block_demands))
                    accepted.push_back(row);
                else
                    --bucket_blocks;
            }
            planner.Minimize(block_demands);
            victims = planner.TakeVictims();
            batch.stats = planner.Stats();
            batch.stats.requested = group_ids.size();
        }
        for (const auto& set : victims) {
            for (const auto& [group_id, location] : set.members) {
                if (store_.Groups()[group_id].Index().Contains(*store_.HostPool(), location)) {
                    FatalCheck(evictor_.EvictHostCachedBlock(group_id, location, &CacheEvictor::HostEvictionStats::lru),
                               "a planned Host victim became pinned");
                }
            }
        }
        for (const std::size_t row : accepted) {
            batch.blocks[row] = acquire_in_bucket(group_ids[row], buckets[row]);
            FatalCheck(static_cast<bool>(batch.blocks[row]), "planned Host placement failed");
        }
    }
    batch.stats.allocated = static_cast<std::size_t>(
        std::ranges::count_if(batch.blocks, [](const CacheBlockRef& block) { return static_cast<bool>(block); }));
    batch.stats.unallocated = batch.stats.requested - batch.stats.allocated;
    return batch;
}

CacheBlockRef HostCache::AcquireHostBlock(std::uint32_t group_id, std::int32_t bucket) {
    const std::array groups{group_id};
    const std::array buckets{bucket};
    HostAllocationBatch batch = AcquireHostBlocks(groups, buckets);
    return batch.blocks.empty() ? CacheBlockRef{} : std::move(batch.blocks.front());
}

HostCache::HostAllocationBatch HostCache::AcquireHostBoundaries(std::span<const std::uint32_t> group_ids,
                                                                std::span<const std::int32_t> buckets,
                                                                std::span<const HostSegmentRows> boundaries) {
    return acquireHostPlacement(planHostBoundaries(group_ids, buckets, boundaries, write_back_evictions_), group_ids,
                                buckets, boundaries);
}

std::size_t HostCache::NumHostBoundariesThatFit(std::span<const std::uint32_t> group_ids,
                                                std::span<const std::int32_t> buckets,
                                                std::span<const HostSegmentRows> boundaries) const {
    return planHostBoundaries(group_ids, buckets, boundaries, write_back_evictions_).num_placed;
}

HostCache::HostPlacementPlan HostCache::planHostBoundaries(std::span<const std::uint32_t> group_ids,
                                                           std::span<const std::int32_t> buckets,
                                                           std::span<const HostSegmentRows> boundaries,
                                                           bool place_by_segments) const {
    _assert(store_.HostPool() != nullptr, "Host placement requires a host pool");
    _assert(group_ids.size() == buckets.size(), "Host allocation needs one bucket per requested block");
    std::size_t next_row = 0;
    for (const HostSegmentRows& boundary : boundaries) {
        _assert(boundary.first_row == next_row && boundary.past_end_row >= boundary.first_row,
                "Host boundaries cover the rows in order");
        next_row = boundary.past_end_row;
    }
    _assert(next_row == group_ids.size(), "Host boundaries cover every row");
    std::vector<BlockPlacementDemand> block_demands(store_.Groups().size());
    for (std::size_t i = 0; i < store_.Groups().size(); ++i)
        block_demands[i].blocks_by_bucket.resize(store_.Groups()[i].Allocator().ShardCount());
    for (std::size_t row = 0; row < group_ids.size(); ++row) {
        _assert(group_ids[row] < store_.Groups().size(), "Host block group id out of range");
        _assert(buckets[row] >= 0 && buckets[row] < store_.Groups()[group_ids[row]].Allocator().ShardCount(),
                "Host block bucket is outside the group's shard count");
        ++block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
    }
    HostEvictionPlanner planner{store_, history_, store_.Groups(), *store_.HostPool()};
    HostPlacementPlan plan;
    if (planner.Fits(block_demands)) {
        plan.num_placed = boundaries.size();
        return plan;
    }
    if (place_by_segments) {
        plan = planHostSegments(group_ids, buckets, boundaries);
        if (plan.num_placed == boundaries.size()) return plan;
    }
    for (BlockPlacementDemand& block_demand : block_demands) std::ranges::fill(block_demand.blocks_by_bucket, 0);
    const std::size_t segment_rows = plan.num_placed == 0 ? 0 : boundaries[plan.num_placed - 1].past_end_row;
    for (std::size_t row = 0; row < segment_rows; ++row) ++block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
    for (const auto& victim : plan.victims) planner.Seed(victim.group_id, victim.location);

    for (const HostSegmentRows& boundary : boundaries.subspan(plan.num_placed)) {
        for (std::size_t row = boundary.first_row; row < boundary.past_end_row; ++row)
            ++block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
        if (!planner.Fit(block_demands)) {
            for (std::size_t row = boundary.first_row; row < boundary.past_end_row; ++row)
                --block_demands[group_ids[row]].blocks_by_bucket[buckets[row]];
            break;
        }
        ++plan.num_placed;
    }
    planner.Minimize(block_demands);
    for (const auto& set : planner.Victims()) {
        for (const auto& [group_id, location] : set.members) {
            plan.victims.push_back(
                {.group_id = group_id, .location = location, .reason = &CacheEvictor::HostEvictionStats::lru});
        }
    }
    return plan;
}

bool HostCache::ContainsHostCachedBlock(const CacheKey& key) const {
    if (store_.HostPool() == nullptr) {
        return false;
    }
    _assert(key.group_id < store_.Groups().size(), "host cache key group id out of range");
    return store_.Groups()[key.group_id].Index().Contains(*store_.HostPool(), key);
}

bool HostCache::IsHostCachedBlock(CacheBlockLocation location) const {
    if (store_.HostPool() == nullptr) {
        return false;
    }
    return std::ranges::any_of(
        store_.Groups(), [&](const CacheGroup& group) { return group.Index().Contains(*store_.HostPool(), location); });
}

std::int32_t HostCache::NumHostCachedBlocks() const {
    if (store_.HostPool() == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : store_.Groups()) {
        count += group.Index().NumEntries(*store_.HostPool());
    }
    return count;
}

std::int32_t HostCache::NumPinnedHostCachedBlocks() const {
    if (store_.HostPool() == nullptr) {
        return 0;
    }
    std::int32_t count = 0;
    for (const CacheGroup& group : store_.Groups()) {
        count += group.Index().NumPinnedEntries(*store_.HostPool());
    }
    return count;
}

void HostCache::CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key,
                               std::optional<PrefixCacheIndex::CachedBlockMetadata> inherited_metadata) {
    _assert(store_.HostPool() != nullptr, "CacheHostBlock requires a host pool");
    _assert(key.group_id < store_.Groups().size(), "CacheHostBlock group id out of range");
    PrefixCacheIndex& index = store_.Groups()[key.group_id].Index();
    PrefixCacheIndex::CachedBlockMetadata metadata{
        .eviction = {.logical_block_index = -1, .boundary_kind = CacheBoundaryKind::kChunk},
    };
    if (inherited_metadata) {
        metadata = *inherited_metadata;
        // Registering touches an existing entry with the epoch given: never
        // age one a Host hit of this very build may just have touched.
        if (const std::optional<CacheBlockLocation> existing =
                store_.LocationOf(*store_.HostPool(), key.group_id, key)) {
            metadata.eviction.last_access_epoch =
                std::max(metadata.eviction.last_access_epoch,
                         index.MetadataFor(*store_.HostPool(), *existing)->eviction.last_access_epoch);
        }
    } else {
        metadata.eviction.last_access_epoch = history_.IssueAccessEpoch();
    }
    // The entry holds the node of its hash: a store's the node its Device
    // publication linked, while either tier holds the hash; an L3 GET's, or a
    // store's whose Device entry left before the ACK, one it starts without a
    // parent, until a publication through its hash gives it its boundary (and,
    // past page 0, links it).
    index.Register(*store_.HostPool(), block_ref, key, metadata.eviction.last_access_epoch,
                   metadata.eviction.logical_block_index, metadata.eviction.boundary_kind, /*newly_cached=*/nullptr,
                   metadata.session);
    // A copy a demotion reached on its way (TierTransferManager::
    // DemoteInFlightEvictions) lands demoted, unless the entry is newer.
    if (metadata.demoted) {
        index.MarkDemoted(*store_.HostPool(), key, CacheStore::kDemotedAccessEpoch);
    }
    if (store_.GroupIsState(key.group_id)) {
        BoundaryTree::Node* node = store_.Boundaries().Find(key.content_hash);
        _assert(node != nullptr, "an indexed checkpoint has a tree node");
        node->latest_state_access_epoch =
            std::max({node->latest_state_access_epoch, metadata.eviction.last_access_epoch, metadata.max_access_epoch});
    }
}

void HostCache::CacheHostTail(const ContentHash& prior, const TailPublication& tail) {
    // A failed/partial GET can leave only some groups present. Record a hint
    // only when Host holds data under it; ordinary matching still validates
    // every group's exact boundary before claiming anything.
    bool on_host = false;
    store_.ForEachBoundaryKey(
        CacheKey{.content_hash = tail.hash}, tail.tail_tokens,
        [&](std::size_t, const CacheKey& key) { on_host = on_host || ContainsHostCachedBlock(key); });
    if (on_host) {
        evictor_.RecordTail(prior, tail);
    }
}

std::vector<HostPublication> HostCache::SettleEvictedSegment(CacheEvictor::EvictedSegment&& segment,
                                                             bool placed_on_host) {
    std::vector<HostPublication> published;
    // In eviction order, each hold dropped as soon as its member is decided: a
    // checkpoint boundary is decided before the pages above it, which judge it
    // as the Host then holds it. Within a boundary the order does not matter:
    // its members are cancelled alike, and while one of its checkpoints is
    // still held or was published the boundary is a checkpoint boundary on the
    // Host.
    for (CacheEvictor::EvictedBlock& member : segment.members) {
        const ContentHash& hash = member.key.content_hash;
        _assert(static_cast<bool>(member.pending), "a recorded store holds its tree node until it settles");
        BoundaryTree::Node& node = member.pending.node();
        // The cap dropped the tail's hint while the copy was out, or no
        // checkpoint boundary at or under the boundary is left on either tier:
        // no probe could resume from it on the Host.
        const bool cancelled = (node.tail_tokens > 0 && !store_.Boundaries().Tails().Contains(hash)) ||
                               (store_.NumStateGroups() > 0 && !node.HasCheckpointBoundaryInSubtree());
        const bool publishes = placed_on_host && !cancelled;
        if (publishes) {
            _assert(static_cast<bool>(member.host_block), "a placed segment carries a Host block for every member");
            CacheHostBlock(member.host_block, member.key, member.metadata);
            node.written_back_group_bits |= std::uint64_t{1} << member.key.group_id;
            published.push_back(HostPublication{.key = member.key, .block = member.host_block});
        } else if (placed_on_host && member.host_block &&
                   store_.Groups()[member.key.group_id].Index().Contains(*store_.HostPool(),
                                                                         member.host_block->Location())) {
            // A cancelled member the Host held already leaves it once nothing
            // else holds it; one in use waits for the collection.
            const CacheBlockLocation location = member.host_block->Location();
            member.host_block.reset();
            evictor_.EvictHostCachedBlock(member.key.group_id, location, &CacheEvictor::HostEvictionStats::dead_entry);
        }
        // The node may leave here; it is not read below.
        member.pending.reset();
        // What was not published is judged again by the next collection.
        if (!publishes) {
            store_.Boundaries().QueueDeadCandidate(hash);
        }
    }
    return published;
}

bool HostCache::DemoteHostEntryHeldByWriteBack(const CacheKey& key, std::uint32_t write_back_holds) {
    return store_.HostPool() != nullptr &&
           store_.Groups()[key.group_id].Index().Demote(*store_.HostPool(), key, CacheStore::kDemotedAccessEpoch,
                                                        write_back_holds);
}

bool HostCache::IsClearable() const {
    if (store_.HostPool() == nullptr) {
        return true;
    }
    for (const CacheGroup& group : store_.Groups()) {
        if (group.Index().NumPinnedEntries(*store_.HostPool()) != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace tokenspeed
