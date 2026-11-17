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

// HostCache: Host L2. Its blocks are acquired here -- one per (group, bucket),
// segments that arrive whole, a retraction image's boundaries -- by plans that
// choose the Host victims before anything is evicted; Host entries are
// published, found and counted here; an eviction write-back's segments settle
// here; and the stores queued for the Host wait here until a drain takes
// them. It works on the store, evicts through the evictor and dates by the
// admission history.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cache/coordinator/admission_history.h"
#include "cache/coordinator/cache_evictor.h"
#include "cache/coordinator/cache_store.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {

class HostCache {
public:
    // write_back_evictions: the Host write-back policy (CacheCoordinator).
    HostCache(CacheStore& store, CacheEvictor& evictor, AdmissionHistory& history, bool write_back_evictions)
        : store_{store}, evictor_{evictor}, history_{history}, write_back_evictions_{write_back_evictions} {}
    HostCache(const HostCache&) = delete;
    HostCache& operator=(const HostCache&) = delete;
    HostCache(HostCache&&) = delete;
    HostCache& operator=(HostCache&&) = delete;

    struct StoreCandidate {
        CacheKey key;
    };
    // Ends a segment's Host store transaction, member by member in eviction
    // order: with placed_on_host, each member reaches the Host index as its
    // Device entry was (epoch -- the newer of that and an existing Host entry's
    // -- logical block index and boundary kind), unless it is cancelled: a tail
    // whose hint the per-prior cap dropped meanwhile, or a boundary with no
    // checkpoint boundary at or under it any more on either tier. A cancelled
    // member already on the Host leaves it once nothing else holds it. Without
    // placed_on_host (no Host placement), nothing is published. Either way
    // every member's hold on its tree node ends and a node left dead is queued
    // for collection. Returns the entries published, the block canonical for
    // each key.
    std::vector<HostPublication> SettleEvictedSegment(CacheEvictor::EvictedSegment&& segment, bool placed_on_host);
    // Demotes key's Host entry, as CacheCoordinator::DemoteEntries does, while the segments of
    // an eviction write-back hold it write_back_holds times to publish into
    // it: those holds do not keep it from the demotion, any other does. False
    // when the Host does not hold key, something else holds it, or it is
    // demoted there already.
    bool DemoteHostEntryHeldByWriteBack(const CacheKey& key, std::uint32_t write_back_holds);
    struct HostAllocationStats {
        std::size_t requested{0};
        std::size_t allocated{0};
        std::size_t unallocated{0};
        // Prefix-index enumerations, including groups that can donate capacity.
        std::size_t group_scans{0};
        // Unpinned entries visited while collecting replacement candidates.
        std::size_t block_candidates{0};
        // Distinct parents whose occupancy was simulated during replacement.
        std::size_t parent_candidates{0};
    };
    struct HostAllocationBatch {
        std::vector<CacheBlockRef> blocks;
        HostAllocationStats stats;
    };
    // Queue every already-published Device cache entry under these
    // content hashes (chain hashes and tail hashes alike) for D2H Store.
    // Missing keys and an absent Host tier are silently skipped, and so are
    // demoted entries (SetCheckpointDemotion): free Host space would take
    // them back.
    void QueueCachedBlocksForStore(std::span<const std::string> content_hashes);
    std::vector<StoreCandidate> TakePendingStores();
    // One Host block per (group, bucket), with free capacity assigned first
    // in input order. The shared eviction plan funds remaining rows globally;
    // failed rows and unnecessary tentative victims evict nothing. A State
    // checkpoint takes its unpinned State siblings at the same boundary along.
    // A shortfall leaves an empty reference, without blocking later rows.
    HostAllocationBatch AcquireHostBlocks(std::span<const std::uint32_t> group_ids,
                                          std::span<const std::int32_t> buckets);
    // One segment of an AcquireHostSegments batch, or one boundary of a
    // retraction image's (NumHostBoundariesThatFit): rows [first_row,
    // past_end_row) of its group_ids and buckets.
    struct HostSegmentRows {
        std::size_t first_row{0};
        std::size_t past_end_row{0};
    };
    // Host blocks for segments that must arrive whole, each block in the
    // bucket of the Device block it mirrors. The segments cover the rows in
    // order, the one to keep first. It plans, then evicts, then allocates:
    // the rows of every segment fit jointly against the capacity the chosen
    // victims leave, a segment is placed whole or not at all -- the last ones
    // go while the rest cannot fit -- and nothing is evicted that the placed
    // segments do not need. A victim is a Host checkpoint boundary (a
    // BoundaryTree node with has_checkpoint set on the Host: some State group
    // holds a checkpoint there) with the Host pages above it that only it
    // depends on, up to one in use, and the
    // Host pages under them, or only its Host State while another checkpoint
    // on either tier still reads its history (including a Device copy at the
    // same boundary), or if a history pin blocks the segment while every
    // State group at the boundary is unpinned; a Host
    // page no Host checkpoint boundary depends on (without State groups: a
    // page no Host page continues, a Host leaf) goes with the Host pages under
    // it. The climb stops before another checkpoint on either tier. A Host
    // history candidate can still leave at its own rank under pressure;
    // removing State does not take a Device checkpoint's history with it.
    // Chosen victims never contain a member in use. A pending write-back
    // of the parent page also blocks the victim: the page would otherwise
    // arrive with nothing under it. Victims go
    // by class: dead entries (a tail no probe enumerates, or incomplete State
    // and history with no checkpoint on either tier depending on them) first;
    // then exact Device duplicates and continued nonaligned input checkpoints
    // (a shared State access epoch proves the continuation), newest Device
    // copy first; then safe closed-group KV duplicates, held Device copies
    // first and newer Device copies first within each ownership class; then
    // Host-only cache, least recently used first. A Host-only checkpoint
    // boundary whose checkpoints are all demoted (SetCheckpointDemotion) goes
    // after the held copies and before the unheld ones. A missing Device window
    // still protects every closed-group Host page in the required suffix.
    // Once a victim took every Host page under the page
    // above it, that page is a candidate too. The restore pass then returns,
    // newest first, each victim the placed segments can do without and whose
    // parent page stays. A dropped segment's rows hold empty references. A
    // retraction image's leg under the write-back policy plans its boundaries
    // this way first (NumHostBoundariesThatFit).
    HostAllocationBatch AcquireHostSegments(std::span<const std::uint32_t> group_ids,
                                            std::span<const std::int32_t> buckets,
                                            std::span<const HostSegmentRows> segments);
    // How many of `boundaries` -- a retraction image's rows by content hash,
    // covering the rows in order, the shallowest first -- the Host can take
    // whole from the first on, each row in its Device block's bucket. Nothing
    // is evicted or acquired; AcquireHostBoundaries carries out the same plan.
    // Uses AcquireHostBlocks' planner, accepting whole boundaries instead of
    // individual rows. The write-back policy first tries its segment plan,
    // without placing fewer boundaries than the block plan alone would.
    std::size_t NumHostBoundariesThatFit(std::span<const std::uint32_t> group_ids,
                                         std::span<const std::int32_t> buckets,
                                         std::span<const HostSegmentRows> boundaries) const;
    // Places what NumHostBoundariesThatFit counts, by the plan it makes: its
    // victims are evicted, then the rows of the boundaries it counts take
    // the room they leave, a boundary whole or not at all. The rows of the
    // later boundaries hold empty references.
    HostAllocationBatch AcquireHostBoundaries(std::span<const std::uint32_t> group_ids,
                                              std::span<const std::int32_t> buckets,
                                              std::span<const HostSegmentRows> boundaries);
    CacheBlockRef AcquireHostBlock(std::uint32_t group_id, std::int32_t bucket);
    bool ContainsHostCachedBlock(const CacheKey& key) const;
    bool IsHostCachedBlock(CacheBlockLocation location) const;
    std::int32_t NumHostCachedBlocks() const;
    std::int32_t NumPinnedHostCachedBlocks() const;
    // Publishes a Host block under key; a key that already has a Host entry
    // re-points block_ref to it. inherited_metadata carries the Device entry's
    // metadata an eviction write-back publishes under (the newer of its
    // epoch and an existing entry's; a demoted one lands demoted unless the
    // existing entry is newer); nullopt publishes as a fresh access with no
    // logical block index, at a chunk boundary.
    void CacheHostBlock(CacheBlockRef& block_ref, const CacheKey& key,
                        std::optional<PrefixCacheIndex::CachedBlockMetadata> inherited_metadata);
    void CacheHostTail(const ContentHash& prior, const TailPublication& tail);
    // Queues a newly published Device entry for D2H Store (write-through).
    void QueueStore(CacheKey key) { pending_stores_.push_back(StoreCandidate{.key = std::move(key)}); }
    // Drops the queued stores no drain took (ClearDeviceCache), counting them.
    void DropPendingStores() {
        store_candidates_dropped_by_flush_ += pending_stores_.size();
        pending_stores_.clear();
    }
    // Whether there is a Host tier at all.
    bool HasPool() const { return store_.HostPool() != nullptr; }
    std::int32_t NumFreeLcmBlocks() const {
        return store_.HostPool() == nullptr ? 0 : store_.HostPool()->NumEmptyLcmBlocks();
    }
    // Whether no Host entry is pinned, so ClearCache may drop them all.
    bool IsClearable() const;
    // The store candidates ClearDeviceCache dropped before any drain took
    // them (accounting only).
    std::uint64_t StoreCandidatesDroppedByFlush() const { return store_candidates_dropped_by_flush_; }

private:
    // A Host placement planned against a shadow before anything is evicted:
    // how many of the leading segments or boundaries the Host places, and
    // the entries to evict for them, in eviction order (a checkpoint's State
    // siblings after it), each with the reason it counts under.
    struct HostPlacementPlan {
        struct Victim {
            std::uint32_t group_id{0};
            CacheBlockLocation location;
            CacheEvictor::HostEvictionCounts CacheEvictor::HostEvictionStats::*reason{nullptr};
        };
        std::size_t num_placed{0};
        std::vector<Victim> victims;
    };
    struct HostKvDuplicateCandidate {
        std::uint32_t group_id{0};
        CacheBlockLocation location;
        std::int32_t boundary_tokens{0};
        // A request holds the Device copy. The visit yields every held copy
        // before the unheld ones; a Host plan with demoted checkpoint
        // boundaries next stops at the first unheld one.
        bool device_in_use{false};
    };
    // Unpinned Host KV copies backed by a contiguous Device KV prefix,
    // excluding pages a sliding group can force into a Host suffix. Visit in
    // priority order and stop as soon as the callback returns false.
    // Defined in host_segments.cpp; callers must stay in that translation
    // unit unless the template definition is moved into a header.
    template <typename Visitor>
    void visitHostKvDuplicateCandidates(Visitor&& visit) const;
    // The plan AcquireHostSegments carries out and planHostBoundaries'
    // segment step reuses.
    HostPlacementPlan planHostSegments(std::span<const std::uint32_t> group_ids, std::span<const std::int32_t> buckets,
                                       std::span<const HostSegmentRows> segments) const;
    // Carries out a plan of planHostSegments or planHostBoundaries: evicts its
    // victims, then the rows of the segments it places take the room they
    // leave. The rows of the later segments hold empty references.
    HostAllocationBatch acquireHostPlacement(const HostPlacementPlan& plan, std::span<const std::uint32_t> group_ids,
                                             std::span<const std::int32_t> buckets,
                                             std::span<const HostSegmentRows> segments);
    // Shared block planning with whole-boundary acceptance. The write-back
    // policy first tries its segment plan when place_by_segments is set.
    HostPlacementPlan planHostBoundaries(std::span<const std::uint32_t> group_ids,
                                         std::span<const std::int32_t> buckets,
                                         std::span<const HostSegmentRows> boundaries, bool place_by_segments) const;

    CacheStore& store_;
    CacheEvictor& evictor_;
    AdmissionHistory& history_;
    bool write_back_evictions_{false};
    std::vector<StoreCandidate> pending_stores_;
    // Accounting only.
    std::uint64_t store_candidates_dropped_by_flush_{0};
};

}  // namespace tokenspeed
