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
#include <iterator>
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
#include "cache/coordinator/tier_eviction_planner.h"
#include "cache/core/block_placement_demand.h"
#include "utils.h"
#include "cache/coordinator/gds_tiered.h"
#include <numeric>

namespace tokenspeed {

std::vector<HostCache::StoreCandidate> HostCache::TakePendingStores(std::size_t max_candidates) {
    const auto past_end =
        pending_stores_.begin() + static_cast<std::ptrdiff_t>(std::min(max_candidates, pending_stores_.size()));
    std::vector<StoreCandidate> pending(std::make_move_iterator(pending_stores_.begin()),
                                        std::make_move_iterator(past_end));
    pending_stores_.erase(pending_stores_.begin(), past_end);
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
    if (store_.UsesGds()) return acquireGdsBlocks(group_ids, buckets, true, false);
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
        std::vector<TierEvictionPlanner::VictimSet> victims;
        std::vector<std::size_t> accepted;
        {
            TierEvictionPlanner planner{store_, history_, *store_.HostPool(), CacheTier::kHost};
            std::vector<BlockPlacementDemand> block_demands = EmptyBlockDemands(store_);
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
            const TierEvictionPlanner::Stats stats = planner.GetStats();
            batch.stats.group_scans = stats.group_scans;
            batch.stats.block_candidates = stats.block_candidates;
            batch.stats.parent_candidates = stats.parent_candidates;
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

HostCache::HostAllocationBatch HostCache::AcquireHostCopies(std::span<const std::uint32_t> groups,
                                                            std::span<const std::int32_t> buckets) {
    return store_.UsesGds() ? acquireGdsBlocks(groups, buckets, false, false) : AcquireHostBlocks(groups, buckets);
}

HostCache::HostAllocationBatch HostCache::AcquireHostLanding(std::span<const std::uint32_t> groups,
                                                             std::span<const std::int32_t> buckets) {
    if (store_.UsesGds()) return acquireGdsBlocks(groups, buckets, true, true);
    _assert(groups.size() == buckets.size(), "one Host bucket per landing block");
    HostAllocationBatch batch;
    batch.blocks.resize(groups.size());
    batch.stats.requested = groups.size();
    batch.stats.unallocated = groups.size();
    std::vector<TierEvictionPlanner::VictimSet> victims;
    {
        TierEvictionPlanner planner{store_, history_, *store_.HostPool(), CacheTier::kHost};
        auto demand = EmptyBlockDemands(store_);
        for (std::size_t i = 0; i < groups.size(); ++i) ++demand[groups[i]].blocks_by_bucket[buckets[i]];
        if (!planner.Fit(demand)) return batch;
        planner.Minimize(demand);
        victims = planner.TakeVictims();
    }
    for (const auto& set : victims)
        for (const auto& [group, location] : set.members)
            if (store_.Groups()[group].Index().Contains(*store_.HostPool(), location))
                FatalCheck(evictor_.EvictHostCachedBlock(group, location, &CacheEvictor::HostEvictionStats::lru),
                           "a planned Host landing victim became pinned");
    for (std::size_t i = 0; i < groups.size(); ++i) {
        auto blocks = store_.HostPool()->AcquireBlocksInBuckets(groups[i], buckets.subspan(i, 1));
        FatalCheck(blocks.size() == 1, "a whole Host landing must fit its preview");
        batch.blocks[i] = std::move(blocks.front());
    }
    batch.stats.allocated = groups.size();
    batch.stats.unallocated = 0;
    return batch;
}

HostCache::HostAllocationBatch HostCache::acquireGdsBlocks(std::span<const std::uint32_t> groups,
                                                           std::span<const std::int32_t> buckets,
                                                           bool allow_logical_loss, bool whole) {
    _assert(groups.size() == buckets.size(), "one Host bucket per requested block");
    std::vector<std::size_t> ends(groups.size());
    std::iota(ends.begin(), ends.end(), 1);
    if (whole && !ends.empty()) ends = {groups.size()};
    auto selected = PlanTieredHostAllocation(store_, groups, buckets, ends, allow_logical_loss, false);
    if (whole && !std::ranges::all_of(selected.accepted, [](bool accepted) { return accepted; })) {
        HostAllocationBatch rejected;
        rejected.blocks.resize(groups.size());
        rejected.stats.requested = groups.size();
        rejected.stats.unallocated = groups.size();
        return rejected;
    }
    const auto victims = TieredVictims(selected.plan, std::pmr::get_default_resource());
    FatalCheck(evictor_
                   .EvictPlanned(victims, false,
                                 CacheEvictor::VictimReasons{
                                     .victim = &CacheEvictor::DeviceEvictionStats::admission,
                                     .stranded_suffix = &CacheEvictor::DeviceEvictionStats::stranded_suffix,
                                     .request_reclaim = &CacheEvictor::DeviceEvictionStats::request_reclaim})
                   .empty(),
               "a joint Host plan cannot reclaim live Device owners");
    HostAllocationBatch batch;
    batch.blocks.resize(groups.size());
    batch.stats.requested = groups.size();
    for (std::size_t g = 0; g < store_.Groups().size(); ++g) {
        std::vector<std::size_t> rows;
        std::vector<std::int32_t> owners;
        for (std::size_t row = 0; row < groups.size(); ++row)
            if (selected.accepted[row] && groups[row] == g) {
                rows.push_back(row);
                owners.push_back(buckets[row]);
            }
        auto blocks = store_.HostPool()->AcquireBlocksInBuckets(static_cast<std::uint32_t>(g), owners);
        FatalCheck(blocks.size() == rows.size(), "joint Host placement did not fit its preview");
        for (std::size_t i = 0; i < rows.size(); ++i) batch.blocks[rows[i]] = std::move(blocks[i]);
        batch.stats.allocated += rows.size();
    }
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
    if (store_.UsesGds()) return planHostSegments(group_ids, buckets, boundaries);
    TierEvictionPlanner planner{store_, history_, *store_.HostPool(), CacheTier::kHost};
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

void HostCache::TouchHostKeys(std::span<const CacheKey> keys) {
    _assert(store_.HostPool() != nullptr, "TouchHostKeys requires a host pool");
    const std::uint64_t access_epoch = history_.IssueAccessEpoch();
    for (const CacheKey& key : keys) {
        _assert(key.group_id < store_.Groups().size(), "Host touch key group id out of range");
        PrefixCacheIndex& index = store_.Groups()[key.group_id].Index();
        // TouchExisting refreshes the entries of the keys whose source block
        // is present: the entry's own block stands for it.
        const CacheBlockRef block = index.Find(*store_.HostPool(), key);
        if (!block) {
            continue;
        }
        index.TouchExisting(*store_.HostPool(), std::span{&key, 1}, std::span{&block, 1}, access_epoch,
                            /*session=*/nullptr);
        // A hit keeps its tail young under the per-prior cap.
        store_.Boundaries().TouchTail(key.content_hash);
    }
}

bool HostCache::TryEvictHostEntry(const CacheKey& key) {
    _assert(store_.HostPool() != nullptr, "TryEvictHostEntry requires a host pool");
    _assert(key.group_id < store_.Groups().size(), "Host entry key group id out of range");
    const std::optional<CacheBlockLocation> location = store_.LocationOf(*store_.HostPool(), key.group_id, key);
    return !location ||
           evictor_.EvictHostCachedBlock(key.group_id, *location, &CacheEvictor::HostEvictionStats::released);
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
