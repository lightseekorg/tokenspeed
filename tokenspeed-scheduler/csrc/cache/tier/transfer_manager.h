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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/transfer.h"
#include "resource/allocator/req_pool_allocator.h"

namespace tokenspeed {

// Which call drained a Host L2 store (not which code queued its candidates):
// boundary publication or finish through StartPendingStores, a retraction
// image through StartRetractionStores, what an admission or a restore evicted
// under the write-back policy through StartEvictionStores. Accounting only --
// it selects the counters a store lands in, never reaches the runtime (which
// branches on StoreSourceGuard), and no scheduling decision reads it.
enum class HostStoreCause : std::uint8_t { kPublication, kFinish, kRetraction, kEviction };

// Why a retraction image cannot be held: no snapshot pool at all (the null
// page alone is configured, so nothing is ever imaged), no blob slot free, a
// pool too small for the image even with every published page on the Host,
// or a Host L2 too small to take the published pages the pool cannot.
enum class ImageShortfall : std::uint8_t { kNoPool, kBlobSlot, kSnapshotPool, kHostL2 };

// Owns the mechanics and asynchronous lifetime of transfers between Device and
// Host cache tiers. Scheduling policy and request state transitions stay in
// Scheduler.
class TierTransferManager {
public:
    explicit TierTransferManager(CacheCoordinator& coordinator);

    // Drains the coordinator's pending store candidates into one write-back
    // op. `guard` says how the Device sources are protected while the copy is
    // in flight (see StoreSourceGuard); every candidate drained by this call
    // gets the same guard. Boundary publication and finish drain with
    // kPinnedUntilAck; a retraction's stream-ordered L2 leg does not use the
    // queue -- StartRetractionStores builds it from the victim's tables.
    // `cause` (kPublication or kFinish) only picks the counters. A finish
    // passes its final publication, nullopt for a live request's
    // publications and a finish that computed nothing: the boundary it
    // leaves is stored ahead of the queued candidates, and its Host copies
    // -- already cached, carried by a store in flight that publishes them at
    // its ACK, or stored now -- are dated by the batch's epochs
    // (CacheCoordinator::TakeStoreEpochs); no other Host entry is refreshed.
    // Its copies are placed newest first, each only in free capacity or in
    // place of an older entry.
    std::optional<WriteBackOperation> StartPendingStores(
        StoreSourceGuard guard, HostStoreCause cause,
        const std::optional<CacheCoordinator::FinalPublication>& publication);
    // Drains the segments this plan's admissions and restores recorded under
    // the write-back policy (CacheCoordinator::TakeEvictedSegments) into one
    // stream-ordered write-back op: their Device pages may be granted away
    // in this same plan, so the runtime orders the copies ahead of the plan's
    // page reuse. Segments go newest Device access first, so the oldest are
    // the ones dropped when the Host cannot place every one
    // (AcquireHostSegments); a dropped segment publishes nothing. A member
    // the Host already holds is not copied, nor is one an unacknowledged
    // store already carries (the segment waits for that store too) or one
    // whose key another member of its segment copies. Each segment publishes
    // once every op it waits for is acknowledged, less the members that
    // became dead meanwhile; one that waits for none settles here. nullopt
    // when nothing is copied.
    std::optional<WriteBackOperation> StartEvictionStores();
    LoadBackOperation StartPrefixLoad(std::vector<BlockTransfer> block_transfers);

    // A retraction image and the two store ops that fill it. The L2 leg is a
    // stream-ordered write-back of the published pages not yet on Host (keys
    // already Host-cached or carried by an in-flight store that publishes them
    // at its ACK are pinned, not copied again), dated like a finish's (the
    // retraction's boundary, TakeRetractionEpochs); the tail leg is the
    // snapshot store of every other data slot plus the slot-state blob.
    // pending_store_ops lists every op the image waits for -- both legs' own
    // ops and any earlier in-flight store carrying one of its keys.
    struct RetractionStores {
        RetractionImage image;
        std::optional<WriteBackOperation> host_store;
        SnapshotStoreOperation snapshot_store;
        std::vector<std::uint32_t> pending_store_ops;
    };
    // A retraction image's Host L2 leg, planned before anything is acquired
    // or evicted for it, so the victim probe and the retraction run one plan.
    struct RetractionImageSplit {
        // Per group, the published slots the Host serves already, held so the
        // plan never takes them: the key's Host entry, or the block of an
        // unacknowledged store that publishes it at its ACK
        // (InFlightWriteBack::publishes_at_ack).
        std::vector<std::vector<ImageSlot>> host_slots;
        // The ops of the stores whose blocks host_slots holds.
        std::vector<std::uint32_t> pending_store_ops;
        // The other published slots, the rows to place: each row's group, its
        // Device block's bucket and the slot.
        std::vector<std::uint32_t> group_ids;
        std::vector<std::int32_t> buckets;
        std::vector<ImageSlot> slots;
        // The rows by boundary, one per content hash with every group's rows
        // at it, the shallowest first.
        std::vector<CacheCoordinator::HostSegmentRows> boundaries;
        // The Host takes the rows of the first num_host_boundaries
        // (CacheCoordinator::NumHostBoundariesThatFit), the snapshot pool every
        // later one.
        std::size_t num_host_boundaries{0};
        // kSnapshotPool when the pool cannot hold the image even with every
        // published slot on the Host (without a Host pool, every slot is the
        // pool's), kHostL2 when it can but not with the rows the Host does not
        // take, else unset.
        std::optional<ImageShortfall> shortfall;
    };
    // `published` names the data slots of [0, num_tokens) that are published
    // entries (CacheCoordinator::PublishedDataSlots, or PublishedDataSlotsAfter
    // for a publication still to come: the split reads keys, never the
    // BoundaryTree). Nothing is mutated or counted.
    RetractionImageSplit SplitRetractionImage(std::span<const BlockTable> tables, std::int32_t num_tokens,
                                              std::vector<std::vector<ImageSlot>> published) const;
    // The epochs a retraction's L2 leg dates its Host copies with: those of
    // the victim's final publication (CacheCoordinator::TakeStoreEpochs),
    // taken before its image is split, so the boundary's Host copies are
    // refreshed before the plan picks Host victims, and the boundary keys an
    // earlier store still carries are re-dated, so that store's ACK
    // publishes them at these epochs. The refresh stands when the split then
    // refuses the image. Nothing is dated without a Host pool.
    CacheCoordinator::StoreEpochs TakeRetractionEpochs(const CacheCoordinator::FinalPublication& publication);
    // Takes the image of a quiescent victim's tables ([0, num_tokens) are its
    // computed tokens) as `split` planned it -- with no shortfall and nothing
    // changed since -- and starts both store legs: the Host takes the rows of
    // the first split.num_host_boundaries boundaries, evicting the victims
    // the plan chose, and the later rows ride the snapshot pool. The image
    // therefore fits; nullopt (the pool refusing it after all) is a safety
    // net no plan reaches. The L2 leg's tickets are dated by `epochs`
    // (TakeRetractionEpochs, taken before the split).
    // blob_slot is the slot-state blob's arena slot the tail leg exports into;
    // the store shares it with the Retracted state so an abort before the ACK
    // cannot hand the slot to a new victim while the export still writes it.
    std::optional<RetractionStores> StartRetractionStores(const std::string& request_id,
                                                          std::int32_t request_pool_index,
                                                          std::shared_ptr<SnapshotSlotIndex> blob_slot,
                                                          std::span<const BlockTable> tables, std::int32_t num_tokens,
                                                          RetractionImageSplit split,
                                                          CacheCoordinator::StoreEpochs epochs);
    // Refreshes the Host copies of the boundary a request resumed from
    // (CacheCoordinator::RefreshHostBoundary) and re-dates the ones an
    // earlier store still carries, so its ACK publishes them no older than
    // access_epoch.
    void RefreshHostBoundary(std::span<const std::string> prefix_hashes, const std::optional<TailPublication>& tail,
                             std::uint64_t access_epoch);
    // One restore op for both legs: load_pairs are the Host L2 rows (keyed),
    // snapshot_pairs the snapshot-pool rows. Both ends of every pair, the blob
    // slot (shared) and the request-pool row the blob is imported into
    // (owned here until the ACK, then handed back by CompleteSnapshotRestore)
    // stay pinned until the ACK, so an abort while Restoring cannot re-grant a
    // page, slot or row the copy is still writing. access_epoch is the
    // request's, which the ACK's republication dates the L2 rows with.
    SnapshotRestoreOperation StartSnapshotRestore(const std::string& request_id, ReqPoolIndex request_pool_index,
                                                  std::shared_ptr<SnapshotSlotIndex> blob_slot,
                                                  std::vector<BlockTransfer> load_pairs,
                                                  std::vector<BlockTransfer> snapshot_pairs,
                                                  std::uint64_t access_epoch);

    // Publishes every ticket's Host entry (an eviction write-back's tickets
    // excepted: their segments publish them) and drops the op's pins, then
    // settles every in-flight segment for which this was the last op it
    // waited on. Returns the entries as published -- the block canonical for
    // each key after the index possibly redirected the publication to an
    // existing entry -- so a retraction image pinned on a ticket's block can
    // follow the redirect.
    std::vector<HostPublication> CompleteWriteBack(std::uint32_t op_id);
    // A prefix load landed: drops the op's pins. Its destinations were
    // published at admission (every source is a Host-warm entry).
    void CompleteLoadBack(std::uint32_t op_id);
    // Returns the request the op belonged to (nullopt for an unknown or
    // duplicate ACK) so the scheduler can advance its FSM.
    std::optional<std::string> CompleteSnapshotStore(std::uint32_t op_id);
    // The request an in-flight restore belongs to (nullopt for an unknown or
    // duplicate ACK). Looked up before CompleteSnapshotRestore so the
    // scheduler can register the KV-event descriptors the republication
    // mutates and decide whether the request is still there to resume.
    std::optional<std::string> SnapshotRestoreRequest(std::uint32_t op_id) const;
    // Drops the restore's pins and returns the request-pool row the op
    // imported into (invalid for an unknown or duplicate op id): the caller
    // installs it in the resumed state, or lets it drop when the request was
    // finished or aborted while restoring. With publish, the L2-tier
    // destinations are republished into the Device prefix index first; the
    // caller passes false for a dead request (its token descriptors are gone
    // with it, and the pages return to the pool with the pins).
    ReqPoolIndex CompleteSnapshotRestore(std::uint32_t op_id, bool publish);

    bool HasLoadBacksInFlight() const { return !load_backs_.empty(); }
    // Pinned stores hold Device capacity that returns by itself at the ACK;
    // the scheduler defers retraction while any is in flight rather than
    // sacrificing a request for capacity that is about to free.
    bool HasPinnedStoresInFlight() const;
    bool HasAnyInFlight() const {
        return !write_backs_.empty() || !load_backs_.empty() || !snapshot_stores_.empty() ||
               !snapshot_restores_.empty();
    }

    // Cumulative Host L2 counters, never reset. Accounting only: no
    // scheduling decision reads them. What a drain takes lands in exactly one
    // outcome each (stored counts in blocks):
    // - StartPendingStores, per candidate: host_cached | in_flight |
    //   evicted_before_drain | no_placement | stored. A final publication's
    //   boundary blocks stored ahead of the candidates land in
    //   no_placement | stored, as does the candidate queued for each.
    // - StartRetractionStores, per published slot: host_cached | in_flight |
    //   snapshot_fallback | stored.
    // - StartEvictionStores, per segment member: host_cached | in_flight |
    //   no_placement (its segment was dropped) | stored.
    struct WriteBackStats {
        std::uint64_t ops{0};
        std::vector<std::uint64_t> blocks;      // stored, by cache group index
        std::uint64_t host_cached{0};           // already a Host entry
        std::uint64_t in_flight{0};             // on an unacknowledged store, or queued twice in one drain
        std::uint64_t evicted_before_drain{0};  // left the Device index before the drain
        std::uint64_t no_placement{0};          // no Host block in its group and bucket
        std::uint64_t snapshot_fallback{0};     // retraction: the slot rides the snapshot pool instead
        // Retraction: images the pool refused after their split (no plan
        // does), one each; their slots land in no outcome.
        std::uint64_t image_refused{0};
        // Eviction: of the stored blocks, those an earlier eviction had
        // written back already (the Host dropped that copy since).
        std::uint64_t recopied{0};
        // Eviction: placed members not published when their segment settled (a
        // tail whose hint the cap dropped, a boundary with no checkpoint
        // boundary at or under it any more).
        std::uint64_t cancelled{0};
    };
    // Host L2 -> Device copies: of prefix-hit loads (one op per first
    // admission the Host wins), and of restores (one op per restore that has
    // L2 rows, counting those rows only, not the snapshot pool's).
    struct LoadBackStats {
        std::uint64_t ops{0};
        std::vector<std::uint64_t> blocks;  // by cache group index
    };
    const WriteBackStats& WriteBackStatsFor(HostStoreCause cause) const {
        return write_back_stats_[static_cast<std::size_t>(cause)];
    }
    const LoadBackStats& PrefixLoadStats() const { return prefix_load_stats_; }
    const LoadBackStats& RestoreLoadStats() const { return restore_load_stats_; }

private:
    // A store ticket always pins its Host destination; the ACK publishes that
    // entry (CacheHostBlock) at access_epoch, the date its batch gave it (a
    // later batch whose boundary holds the key re-dates it), except on an
    // eviction write-back, whose segments publish it as its Device entry was.
    // Whether it also pins the Device source is the op's StoreSourceGuard:
    // kPinnedUntilAck keeps the source cached and unevictable until the ACK,
    // kStreamOrdered leaves it empty and relies on the runtime ordering the
    // copy ahead of any reuse.
    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
        std::uint64_t access_epoch{0};
    };

    struct InFlightWriteBack {
        StoreSourceGuard guard;
        std::vector<StoreTicket> tickets;
        // Whether the ACK publishes the tickets. An eviction write-back's
        // publish through the segments that wait for this op instead, which
        // may wait for later ops or be cancelled.
        bool publishes_at_ack{false};
    };
    // An eviction write-back's segment between its drain and its last ACK:
    // every member carries the Host block it publishes, and pending_store_ops
    // lists the ops still out among those carrying its members (its own
    // drain's, and earlier ones it borrowed a member from).
    struct InFlightSegment {
        CacheCoordinator::EvictedSegment segment;
        std::vector<std::uint32_t> pending_store_ops;
    };

    // A tail store pins its snapshot-pool destinations and the blob slot it
    // exports into until the ACK (the image pins them too, for longer); a
    // restore pins both ends of every row, the blob slot it imports from and
    // the request-pool row it imports into.
    struct InFlightSnapshotStore {
        std::string request_id;
        std::vector<CacheBlockRef> destinations;
        std::shared_ptr<SnapshotSlotIndex> blob_slot;
    };
    struct InFlightSnapshotRestore {
        std::string request_id;
        std::vector<BlockTransfer> transfers;
        std::shared_ptr<SnapshotSlotIndex> blob_slot;
        ReqPoolIndex request_pool_index;
        std::uint64_t access_epoch{0};
    };
    std::uint32_t nextOpId() { return next_op_id_++; }
    LoadBackOperation startLoadBack(std::vector<BlockTransfer> block_transfers);
    std::vector<CacheTransfer> resolveTransfers(std::span<const BlockTransfer> block_transfers) const;
    // The Host block an unacknowledged store already carries for each key, with
    // that store's op id and publishes_at_ack (InFlightWriteBack). Built once
    // per drain from write_backs_ (rather than mirrored in a second container)
    // and looked up per block; a key a store publishing at its ACK carries
    // names that store.
    struct InFlightHostBlock {
        CacheBlockRef block;
        std::uint32_t op_id{0};
        bool publishes_at_ack{false};
    };
    using InFlightHostBlocks = std::unordered_map<CacheKey, InFlightHostBlock, CacheKeyHash>;
    InFlightHostBlocks inFlightHostBlocks() const;
    // A boundary's keys Host does not hold, with their epochs.
    using BoundaryEpochs = std::unordered_map<CacheKey, std::uint64_t, CacheKeyHash>;
    // Re-dates the boundary keys an earlier store still carries, never back:
    // its ACK publishes them no older than these epochs. An eviction
    // write-back's tickets publish nothing at the ACK and are left as they
    // are.
    void redateInFlight(const BoundaryEpochs& boundary);

    CacheCoordinator& coordinator_;
    std::unordered_map<std::uint32_t, InFlightWriteBack> write_backs_;
    // In drain order. Each waits for at least one op of write_backs_.
    std::vector<InFlightSegment> in_flight_segments_;
    // Each transfer pins both tiers until the runtime acknowledges the copy.
    std::unordered_map<std::uint32_t, std::vector<BlockTransfer>> load_backs_;
    std::unordered_map<std::uint32_t, InFlightSnapshotStore> snapshot_stores_;
    std::unordered_map<std::uint32_t, InFlightSnapshotRestore> snapshot_restores_;
    std::uint32_t next_op_id_{0};
    // Indexed by HostStoreCause; every per-group vector is sized at construction.
    std::array<WriteBackStats, 4> write_back_stats_;
    LoadBackStats prefix_load_stats_;
    LoadBackStats restore_load_stats_;
};

}  // namespace tokenspeed
