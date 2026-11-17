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

// CacheEvictor: how an entry leaves either tier's index -- on the Device
// planned victims with their cascades, one entry alone, a tail's blocks
// together; on the Host an entry with its State siblings -- and, under the
// write-back policy, the records Device evictions leave for the plan's
// write-back drain (the segments, the keys on their way to the Host), with
// the eviction counters of both tiers. Choosing the victims stays with the
// planners (AdmissionPlanner, the Host placement). It works on the
// CacheStore it is given.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_store.h"
#include "cache/coordinator/victim_set.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "cache/prefix/boundary_tree.h"
#include "cache/prefix/prefix_index.h"

namespace tokenspeed {

class CacheEvictor {
public:
    // The write-back an eviction recorded for one Device entry, from the
    // eviction until its segment settles (HostCache::SettleEvictedSegment). Meanwhile
    // the entry's boundary node counts it as on its way to the Host -- a Host
    // checkpoint for a State group's block -- which keeps the node and its
    // tail hint, and a probe takes its key as held by the Host
    // (PrefixProber::EvictionStoresExtendPrefix). Move-only; an empty one holds nothing. The
    // evictor must outlive it; a segment dropped unsettled releases its
    // members' stores with it.
    class PendingHostStore {
    public:
        PendingHostStore() noexcept = default;
        // Records an eviction write-back of `key`'s block under `node` (a
        // State checkpoint when `checkpoint`).
        PendingHostStore(CacheEvictor& evictor, BoundaryTree::Node& node, const CacheKey& key, bool checkpoint);
        PendingHostStore(const PendingHostStore&) = delete;
        PendingHostStore& operator=(const PendingHostStore&) = delete;
        PendingHostStore(PendingHostStore&& other) noexcept;
        PendingHostStore& operator=(PendingHostStore&& other) noexcept;
        ~PendingHostStore() { reset(); }

        explicit operator bool() const noexcept { return static_cast<bool>(node_); }
        // The boundary node the store holds; the store must not be empty.
        BoundaryTree::Node& node() const noexcept { return node_.node(); }
        // Releases the store now and leaves it empty.
        void reset() noexcept;

    private:
        CacheEvictor* evictor_{nullptr};
        BoundaryTree::Hold node_;
        // The key's element of the evictor's pending_eviction_keys_, which
        // is node-based, so it stays put while any record holds it.
        const CacheKey* key_{nullptr};
    };
    // A Device entry an admission, a restore or a compaction evicted under the write-back
    // policy, captured before the index let the block go: its page may be
    // granted again in this same plan, so the copy rides the plan's
    // stream-ordered lane ahead of that reuse, and nothing here pins the
    // Device block.
    struct EvictedBlock {
        CacheKey key;
        // The page id and the bucket follow from the location alone.
        CacheBlockLocation device_location;
        // Read before the eviction; the Host entry inherits it.
        PrefixCacheIndex::CachedBlockMetadata metadata;
        // The key's Host entry, held from the eviction on when the Host had
        // one; otherwise the drain sets the Host block that carries it.
        CacheBlockRef host_block;
        // Accounting only: an earlier write-back published this hash's block of
        // this group on the Host, so copying it again is a recopy.
        bool written_back_before{false};
        // Held from the eviction until the segment settles.
        PendingHostStore pending;
    };
    // One Host store transaction: what one victim set of an admission, a
    // restore or a compaction evicted, or several that depend on each other once
    // TakeSegments merged them. Its members publish together, only
    // once every copy they need has landed.
    struct EvictedSegment {
        // In eviction order: the deepest boundary first, each boundary's
        // blocks together.
        std::vector<EvictedBlock> members;
        // The newest Device access epoch among the members, raised as each
        // member is recorded.
        std::uint64_t newest_access_epoch{0};
    };
    // Device entries evicted for one reason, each also counted as
    // host_resident when the Host index still held its key (never without a
    // Host pool).
    struct DeviceEvictionCounts {
        std::uint64_t blocks{0};
        std::uint64_t host_resident{0};
    };
    // Device evictions over every cache group, by what started them. The
    // blocks a victim takes along (its tail's companions, the state
    // checkpoints under its hash) count under its reason. Only blocks that left
    // the index count; a companion still referenced elsewhere (a request
    // table, a copy awaiting its ACK) stays indexed.
    struct DeviceEvictionStats {
        // An admission's planned victims.
        DeviceEvictionCounts admission;
        // The slots the admitted request itself releases at this step (its
        // State input, its expired window blocks) and what they take along;
        // also a superseded input an admission, a restore or a compaction evicts
        // (MarkSupersededStateInput).
        DeviceEvictionCounts request_reclaim;
        // The blocks a segment takes from under its chosen victim so no Device
        // page outlives its parent; only when evicting by segments.
        DeviceEvictionCounts stranded_suffix;
        // A restore's planned victims.
        DeviceEvictionCounts restore;
        // ClearDeviceCache.
        DeviceEvictionCounts flush;
        // The per-prior tail cap dropping a tail; with no single victim, it
        // evicts every block it drops itself.
        DeviceEvictionCounts tail_cap;
        // The collection of dead prefix entries, which like the tail cap evicts
        // every block it drops itself; only when evicting by segments.
        DeviceEvictionCounts dead_entry_collection;
        // A lone request's compaction clearing a slot it moves a block into
        // (LoneRequestAdmission::Compact), with what the entry takes along.
        DeviceEvictionCounts compaction;
        // Subset of request_reclaim considered for write-back, including
        // entries later dropped as dead or demoted. Superseded inputs are
        // excluded.
        std::uint64_t request_reclaim_for_write_back{0};
    };
    // The counter a planned victim's eviction counts under, by what its plan
    // marked it as (VictimSet::Member).
    struct VictimReasons {
        DeviceEvictionCounts DeviceEvictionStats::*victim;
        DeviceEvictionCounts DeviceEvictionStats::*stranded_suffix;
        DeviceEvictionCounts DeviceEvictionStats::*request_reclaim;
    };
    // A planned victim EvictPlanned found no unpinned entry at: an
    // earlier victim's cascade took it, or something still holds its block
    // (an admission's own released slot until its table lets go).
    struct UnevictedVictim {
        std::uint32_t group_id;
        CacheBlockLocation location;
        DeviceEvictionCounts DeviceEvictionStats::*reason;
        EvictedSegment* write_back_segment;
    };
    // What EvictCachedBlockAlone took out of the index.
    struct EvictedEntry {
        CacheKey key;
        // Another Device entry under the key's content hash is still indexed
        // (another group's, or another block of the page): only then can a
        // cascade find anything. Every Device entry holds its hash's tree
        // node, whose count tells this before the entry leaves.
        bool device_companions_left{false};
    };

    // Host entries evicted for one reason, each also counted as
    // device_resident when the Device index still held its key.
    struct HostEvictionCounts {
        std::uint64_t blocks{0};
        std::uint64_t device_resident{0};
    };
    // Host evictions over every cache group, by why the entry left. The other
    // State groups' checkpoints a State entry takes along at its hash count
    // under the same reason.
    struct HostEvictionStats {
        // HostCache::AcquireHostBlocks' global LRU took it, as one child or with a whole
        // parent, whoever asked (a publication or finish store, an L3
        // prefetch); a retraction image's leg took it block by block in that
        // ranking (HostCache::AcquireHostBoundaries); or a segment plan
        // (HostCache::AcquireHostSegments, or that leg's segment step) took it as
        // Host-only cache.
        HostEvictionCounts lru;
        // A segment plan took it as a duplicate of a boundary the Device
        // resumes.
        HostEvictionCounts duplicate;
        // A segment plan took a nonaligned input checkpoint whose request
        // continued to a deeper Device checkpoint. Aligned checkpoints keep
        // their State LRU rank.
        HostEvictionCounts superseded;
        // A segment plan took it as a dead entry (no checkpoint on either
        // tier depends on it), CollectDeadEntries evicted it, or a settled
        // write-back cancelled it (a held member).
        HostEvictionCounts dead_entry;
        // The per-prior tail cap dropped its tail, entry by entry after the
        // tail's Device blocks (device_resident when the Device kept a pinned
        // companion).
        HostEvictionCounts tail_cap;
        // ClearCache, after the Device clear, so never device_resident.
        HostEvictionCounts flush;
        // A segment plan reclaimed a prefix-closed KV copy independently of
        // the boundary's checkpoints, while the Device still held its key.
        HostEvictionCounts duplicate_kv;
    };

    explicit CacheEvictor(CacheStore& store) : store_{store} {}
    // Pending stores and recorded segments point at the evictor.
    CacheEvictor(const CacheEvictor&) = delete;
    CacheEvictor& operator=(const CacheEvictor&) = delete;
    CacheEvictor(CacheEvictor&&) = delete;
    CacheEvictor& operator=(CacheEvictor&&) = delete;

    // Evicts planned victims in plan order, each with its cascade
    // (EvictCachedBlock), counted under `reasons`. With write_back each set is
    // recorded as one segment of the records (TakeSegments); one whose every block was
    // dropped stays empty for the caller to erase. The one loop that commits
    // an admission's, a restore's, a compaction's or a cache clear's victims;
    // it prefetches for the victims a few places ahead (hints only).
    std::vector<UnevictedVictim> EvictPlanned(std::span<const VictimSet> victims, bool write_back,
                                              const VictimReasons& reasons);
    // Device eviction of a planned or flushed victim, with its cascade (its
    // tail's companions or, for a page's KV block or one State group's
    // checkpoint, every State group's checkpoint under its hash); `reason` and
    // write_back_segment as for EvictCachedBlockAlone. A companion still
    // pinned elsewhere stays.
    bool EvictCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                          DeviceEvictionCounts DeviceEvictionStats::*reason, EvictedSegment* write_back_segment);
    // Device eviction of one entry without its cascade, the one place a
    // Device entry leaves the index: it reports the eviction, counts it under
    // `reason` (the DeviceEvictionStats field of what started the eviction;
    // eviction order and cascades never depend on it); the entry's hold on its
    // tree node goes with it. It returns the entry that left the index,
    // nullopt when it is pinned or gone. A superseded input (MarkSupersededStateInput) an
    // admission, rotation, restore or compaction evicts counts as
    // request_reclaim and is never recorded; a demoted entry
    // (SetCheckpointDemotion) is never recorded either, whatever evicts it,
    // and counts under `reason` (DemotedEntriesNotWrittenBack). With
    // write_back_segment (an admission's, a restore's or a compaction's cache victim under the
    // write-back policy) any other entry that left is recorded there -- its
    // metadata read before it left, its Host entry held when the Host has one;
    // the segment's newest_access_epoch rises to its epoch -- and its
    // PendingHostStore holds the tree node, tail hint included, until the
    // segment settles. It is judged and recorded before the entry leaves, so
    // the node never looks empty in between. One no probe could resume from
    // on the Host is dropped instead (DeadEntriesNotWrittenBack). nullptr
    // drops it.
    std::optional<EvictedEntry> EvictCachedBlockAlone(std::uint32_t group_id, CacheBlockLocation location,
                                                      DeviceEvictionCounts DeviceEvictionStats::*reason,
                                                      EvictedSegment* write_back_segment);

    // The segments this plan's admissions, restores and compactions recorded, in the
    // order they were recorded, with every segment that hangs under the KV
    // pages of a later one merged into it (members concatenated in recording
    // order). Two admissions of one build may each take a segment where the
    // second climbs through pages the first one's segment hangs under; placed
    // apart, the first could reach the Host without the prefix it needs.
    // Segments recording the same key (evicted, published again and evicted
    // again in one build) are merged too: the drain copies the key once. The
    // write-back drain is its one consumer, once per plan.
    std::vector<EvictedSegment> TakeSegments();
    // Records a published tail's hint under the page it extends (`prior`).
    // When the per-prior cap drops another tail for it, that tail's entries
    // leave both tiers (tail_cap) -- no probe enumerates a dropped tail, so
    // none could match again -- what still stays in use goes once released.
    void RecordTail(const ContentHash& prior, const TailPublication& tail);
    // Drops the recorded segments that ended up empty (each admission's,
    // restore's or compaction's own), so none is ever left empty.
    void DropEmptySegments();
    bool HasSegments() const { return !evicted_segments_.empty(); }
    // Drops every recorded segment, releasing its members' pending stores.
    void ClearSegments() { evicted_segments_.clear(); }
    // The keys of the recorded write-backs, from recording through
    // settlement: probes take them as keys the Host holds.
    const std::unordered_set<CacheKey, CacheKeyHash>& PendingKeys() const { return pending_eviction_keys_; }

    // Demoted entries (SetCheckpointDemotion) are never written back.
    void SetCheckpointDemotion(bool enabled) { demotes_checkpoints_ = enabled; }
    bool DemotesCheckpoints() const { return demotes_checkpoints_; }
    // Whether an entry counts as demoted (SetCheckpointDemotion): a demotion
    // marked it and no access touched it since. A demoted session's entry or a
    // superseded input at CacheStore::kDemotedAccessEpoch does not.
    bool IsDemotedEntry(const PrefixCacheIndex::CachedBlockMetadata& metadata) const {
        return demotes_checkpoints_ && metadata.demoted;
    }

    // Evicts an unpinned Host entry and, for a State checkpoint, the other
    // State groups' unpinned entries at its hash. Each entry that left counts
    // under HostEvictions().*reason, as device_resident too when the Device
    // index still holds its key. False when the entry itself is pinned or
    // gone; then nothing is evicted.
    bool EvictHostCachedBlock(std::uint32_t group_id, CacheBlockLocation location,
                              HostEvictionCounts HostEvictionStats::*reason);
    // ClearCache's eviction of one Host entry, once the whole Host tier was
    // found clearable: no cascade, counted under flush.
    void FlushHostEntry(std::uint32_t group_id, CacheBlockLocation location);

    // Cumulative, never reset (accounting only).
    const DeviceEvictionStats& DeviceEvictions() const { return device_evictions_; }
    const HostEvictionStats& HostEvictions() const { return host_evictions_; }
    std::uint64_t DeadEntriesNotWrittenBack() const { return dead_entries_not_written_back_; }
    std::uint64_t DemotedEntriesNotWrittenBack() const { return demoted_entries_not_written_back_; }

private:
    // Evicts every group's Device entries under a tail hash: a tail's blocks
    // are usable only together, so they leave together. `reason` and
    // write_back_segment as for EvictCachedBlockAlone.
    void EvictTailEntries(const ContentHash& hash, std::int32_t tail_tokens,
                          DeviceEvictionCounts DeviceEvictionStats::*reason, EvictedSegment* write_back_segment);
    // Releases a pending store's key (PendingHostStore).
    void releasePendingKey(const CacheKey& key) noexcept;

    CacheStore& store_;
    // The segments this plan's admissions, restores and compactions recorded
    // (write_back_evictions only), empty outside a build: the plan's drain
    // takes them. A deque, so an eviction keeps pointing at the
    // segments it records into while it adds more. Never an empty one: each
    // call erases its own that ended up empty.
    std::deque<EvictedSegment> evicted_segments_;
    // The keys of the recorded write-backs, from recording through settlement,
    // including the remainder of the plan before its drain; only
    // PendingHostStore adds and removes them. Probes take the set as keys the
    // Host holds, so it stays a set; a key recorded more than once counts its
    // additional records beside it.
    std::unordered_set<CacheKey, CacheKeyHash> pending_eviction_keys_;
    std::unordered_map<CacheKey, std::size_t, CacheKeyHash> pending_eviction_extra_refs_;
    bool demotes_checkpoints_{false};
    std::uint64_t dead_entries_not_written_back_{0};
    std::uint64_t demoted_entries_not_written_back_{0};
    DeviceEvictionStats device_evictions_;
    HostEvictionStats host_evictions_;
};

}  // namespace tokenspeed
