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

// Owns the mechanics and asynchronous lifetime of transfers between Device and
// Host cache tiers. Scheduling policy and request state transitions stay in
// Scheduler.
class TierTransferManager {
public:
    explicit TierTransferManager(CacheCoordinator& coordinator) : coordinator_{coordinator} {}

    // Drains the coordinator's pending store candidates into one write-back
    // op. `guard` says how the Device sources are protected while the copy is
    // in flight (see StoreSourceGuard); every candidate drained by this call
    // gets the same guard. Boundary publication and finish drain with
    // kPinnedUntilAck; a retraction's stream-ordered L2 leg does not use the
    // queue -- StartRetractionStores builds it from the victim's tables.
    std::optional<WriteBackOperation> StartPendingStores(StoreSourceGuard guard);
    LoadBackOperation StartPrefixLoad(std::vector<BlockTransfer> block_transfers);

    // The pre-admission L3 fill of one Submitted request: one op over the
    // plan's rows, whose Host blocks the op pins until its ACK (the request
    // pins them too, and may be aborted meanwhile).
    PrefetchOperation StartPrefetch(const std::string& request_id, CacheCoordinator::PrefetchPlan plan);
    // What a prefetch's ACK leaves behind: the Host entries now published
    // (pinned for the request to hold until its admission acquires them) and
    // the keys whose objects did not land (to forget from the storage shadow).
    struct PrefetchCompleted {
        std::string request_id;
        std::vector<CacheBlockRef> published;
        std::vector<CacheKey> unlanded;
    };
    // Publishes the first landed_pages prefix pages' rows as Host entries and
    // drops the op; nullopt for an unknown or duplicate op id, and for a
    // landed_pages outside [0, pages] (logged; the op then stays in flight
    // untouched). Publication happens whether or not the request still
    // exists: the bytes landed.
    std::optional<PrefetchCompleted> CompletePrefetch(std::uint32_t op_id, std::int32_t landed_pages);

    // A retraction image and the two store ops that fill it. The L2 leg is a
    // stream-ordered write-back of the published pages not yet on Host (keys
    // already Host-cached or carried by an in-flight store are pinned, not
    // copied again); the tail leg is the snapshot store of every other data
    // slot plus the slot-state blob. pending_store_ops lists every op the
    // image waits for -- both legs' own ops and any earlier in-flight store
    // carrying one of its keys.
    struct RetractionStores {
        RetractionImage image;
        std::optional<WriteBackOperation> host_store;
        SnapshotStoreOperation snapshot_store;
        std::vector<std::uint32_t> pending_store_ops;
    };
    // Takes the image of a quiescent victim's tables ([0, num_tokens) are its
    // computed tokens) and starts both store legs. nullopt when the image
    // cannot be held: a published slot whose Host L2 block cannot be acquired
    // (after evicting unpinned entries) falls back to the snapshot pool, and
    // the retraction is refused only when the pool cannot hold tail plus
    // fallback -- then nothing is kept, though unpinned Host entries evicted
    // for the attempt stay evicted.
    // blob_slot is the slot-state blob's arena slot the tail leg exports into;
    // the store shares it with the Retracted state so an abort before the ACK
    // cannot hand the slot to a new victim while the export still writes it.
    std::optional<RetractionStores> StartRetractionStores(const std::string& request_id,
                                                          std::int32_t request_pool_index,
                                                          std::shared_ptr<SnapshotSlotIndex> blob_slot,
                                                          std::span<const BlockTable> tables, std::int32_t num_tokens);
    // One restore op for both legs: load_pairs are the Host L2 rows (keyed),
    // snapshot_pairs the snapshot-pool rows. Both ends of every pair, the blob
    // slot (shared) and the request-pool row the blob is imported into
    // (owned here until the ACK, then handed back by CompleteSnapshotRestore)
    // stay pinned until the ACK, so an abort while Restoring cannot re-grant a
    // page, slot or row the copy is still writing.
    SnapshotRestoreOperation StartSnapshotRestore(const std::string& request_id, ReqPoolIndex request_pool_index,
                                                  std::shared_ptr<SnapshotSlotIndex> blob_slot,
                                                  std::vector<BlockTransfer> load_pairs,
                                                  std::vector<BlockTransfer> snapshot_pairs);

    // Publishes every ticket's Host entry and drops the op's pins. Returns
    // the entries as published -- the block canonical for each key after the
    // index possibly redirected the publication to an existing entry -- so a
    // retraction image pinned on a ticket's block can follow the redirect.
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
        return !write_backs_.empty() || !load_backs_.empty() || !prefetches_.empty() || !snapshot_stores_.empty() ||
               !snapshot_restores_.empty();
    }

private:
    // A store ticket always pins its Host destination; the ACK's one job is
    // publishing that entry (CacheHostBlock). Whether it also pins the Device
    // source is the op's StoreSourceGuard: kPinnedUntilAck keeps the source
    // cached and unevictable until the ACK, kStreamOrdered leaves it empty
    // and relies on the runtime ordering the copy ahead of any reuse.
    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
    };

    struct InFlightWriteBack {
        StoreSourceGuard guard;
        std::vector<StoreTicket> tickets;
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
    };
    // A prefetch pins its Host destinations until the ACK, which publishes
    // the landed prefix (page_row_ends maps a landed page count to a row
    // prefix) and lets the rest go.
    struct InFlightPrefetch {
        std::string request_id;
        std::vector<CacheCoordinator::PrefetchRow> rows;
        std::vector<std::size_t> page_row_ends;
    };

    std::uint32_t nextOpId() { return next_op_id_++; }
    LoadBackOperation startLoadBack(std::vector<BlockTransfer> block_transfers);
    std::vector<CacheTransfer> resolveTransfers(std::span<const BlockTransfer> block_transfers) const;
    // The Host block an unacknowledged store already carries for each key,
    // with that store's op id. Built once per retraction from write_backs_
    // (rather than mirrored in a second container) and looked up per slot.
    struct InFlightHostBlock {
        CacheBlockRef block;
        std::uint32_t op_id{0};
    };
    using InFlightHostBlocks = std::unordered_map<CacheKey, InFlightHostBlock, CacheKeyHash>;
    InFlightHostBlocks inFlightHostBlocks() const;

    CacheCoordinator& coordinator_;
    std::unordered_map<std::uint32_t, InFlightWriteBack> write_backs_;
    // Each transfer pins both tiers until the runtime acknowledges the copy.
    std::unordered_map<std::uint32_t, std::vector<BlockTransfer>> load_backs_;
    std::unordered_map<std::uint32_t, InFlightSnapshotStore> snapshot_stores_;
    std::unordered_map<std::uint32_t, InFlightSnapshotRestore> snapshot_restores_;
    std::unordered_map<std::uint32_t, InFlightPrefetch> prefetches_;
    std::uint32_t next_op_id_{0};
};

}  // namespace tokenspeed
