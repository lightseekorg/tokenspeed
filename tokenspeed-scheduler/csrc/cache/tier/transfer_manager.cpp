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

#include "cache/tier/transfer_manager.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <utility>

#include "utils.h"

namespace tokenspeed {

std::optional<WriteBackOperation> TierTransferManager::StartPendingStores(StoreSourceGuard guard) {
    std::vector<CacheKey> keys;
    std::vector<CacheBlockRef> device_block_refs;
    std::vector<std::uint32_t> group_ids;
    std::vector<std::int32_t> buckets;
    // Keys already travelling: every ticket of every in-flight write-back.
    // Derived from write_backs_ on demand rather than mirrored in a second
    // container that would have to be kept in step with it. Candidates join
    // the same set so a key queued twice in one round is stored once.
    std::unordered_set<CacheKey, CacheKeyHash> storing_keys;
    for (const auto& [op_id, write_back] : write_backs_) {
        for (const StoreTicket& ticket : write_back.tickets) {
            storing_keys.insert(ticket.key);
        }
    }
    for (auto& candidate : coordinator_.TakePendingStores()) {
        if (coordinator_.ContainsHostCachedBlock(candidate.key) || !storing_keys.insert(candidate.key).second) {
            continue;
        }

        CacheBlockRef device_block_ref = coordinator_.AcquireDeviceCachedBlock(candidate.key);
        if (!device_block_ref) {
            continue;
        }

        group_ids.push_back(candidate.key.group_id);
        // The Host copy lives in its Device source's bucket (same owner).
        buckets.push_back(coordinator_.Allocator(static_cast<std::int32_t>(candidate.key.group_id))
                              .BucketOf(device_block_ref->Location()));
        keys.push_back(std::move(candidate.key));
        device_block_refs.push_back(std::move(device_block_ref));
    }

    if (keys.empty()) {
        return std::nullopt;
    }

    CacheCoordinator::HostAllocationBatch host_allocation = coordinator_.AcquireHostBlocks(group_ids, buckets);
    _assert(host_allocation.blocks.size() == keys.size(), "Host allocation result must stay aligned");

    const bool pin_source = guard == StoreSourceGuard::kPinnedUntilAck;
    std::vector<CacheTransfer> transfers;
    std::vector<StoreTicket> tickets;
    transfers.reserve(host_allocation.stats.allocated);
    tickets.reserve(host_allocation.stats.allocated);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CacheBlockRef& host_block_ref = host_allocation.blocks[i];
        if (!host_block_ref) {
            continue;
        }
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(group_ids[i]));
        _assert(manager.BucketOf(device_block_refs[i]->Location()) == manager.BucketOf(host_block_ref->Location()),
                "Host store pairs blocks of different owners");
        transfers.push_back(CacheTransfer{
            .group_id = group_ids[i],
            .source_page = manager.ResolveCacheBlockId(device_block_refs[i]->Location()),
            .destination_page = manager.ResolveCacheBlockId(host_block_ref->Location()),
            .content_hash = keys[i].content_hash,
            .page_offset = keys[i].page_offset,
        });
        // A stream-ordered store resolves the source page id and lets the
        // reference go: the forward thread's stream orders the copy ahead of
        // any later reuse. A pinned store keeps it until the ACK.
        tickets.push_back(StoreTicket{
            std::move(keys[i]),
            pin_source ? std::move(device_block_refs[i]) : CacheBlockRef{},
            std::move(host_block_ref),
        });
    }

    if (transfers.empty()) {
        return std::nullopt;
    }
    const std::uint32_t op_id = nextOpId();
    const bool inserted = write_backs_.emplace(op_id, InFlightWriteBack{guard, std::move(tickets)}).second;
    _assert(inserted, "duplicate store op id");
    return WriteBackOperation{
        .op_id = op_id,
        .transfers = std::move(transfers),
        .source_pinned = pin_source,
    };
}

bool TierTransferManager::HasPinnedStoresInFlight() const {
    return std::ranges::any_of(
        write_backs_, [](const auto& entry) { return entry.second.guard == StoreSourceGuard::kPinnedUntilAck; });
}

LoadBackOperation TierTransferManager::StartPrefixLoad(std::vector<BlockTransfer> block_transfers) {
    _assert(!block_transfers.empty(), "prefix load requires at least one block transfer");
    for (const BlockTransfer& pair : block_transfers) {
        _assert(pair.prefetch_from_storage || coordinator_.IsHostCachedBlock(pair.source->Location()),
                "pinned Host block lost its cache entry before load emission");
    }
    return startLoadBack(std::move(block_transfers));
}

LoadBackOperation TierTransferManager::startLoadBack(std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted = load_backs_.emplace(op_id, std::move(block_transfers)).second;
    _assert(inserted, "duplicate loadback op id");
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::vector<HostPublication> TierTransferManager::CompleteWriteBack(std::uint32_t op_id) {
    // The runtime emits this ACK only after the asynchronous copy completes.
    // Transfer errors terminate the runtime and must never publish cache state.
    auto it = write_backs_.find(op_id);
    if (it == write_backs_.end()) {
        return {};
    }
    std::vector<StoreTicket> stores = std::move(it->second.tickets);
    write_backs_.erase(it);
    // Publishing the Host entry also drops the tickets' Device pins (if any)
    // when `stores` goes out of scope: the source is evictable again. When
    // the key already has a canonical Host entry (an L3 prefetch of it landed
    // first), Register re-points the ticket's reference to that entry and the
    // ticket's own block goes unindexed; the returned publication names the
    // canonical block so the images pinned on the ticket's block follow.
    std::vector<HostPublication> published;
    published.reserve(stores.size());
    for (StoreTicket& ticket : stores) {
        coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key);
        published.push_back(HostPublication{.key = ticket.key, .block = ticket.host_block_ref});
    }
    return published;
}

void TierTransferManager::CompleteLoadBack(std::uint32_t op_id, bool success) {
    auto it = load_backs_.find(op_id);
    if (it == load_backs_.end()) {
        return;
    }
    // A missed batch_get_into must not publish empty Host or Device pages.
    // Host-warm destinations of a mixed L3 hash were not CacheFullBlocks'd at
    // admit (the hash had an L3 prefetch sibling). Publish every keyed filled
    // destination. Host-only L2 load-backs leave key empty; those pages were
    // already published at admit. CacheHostBlock remains prefetch-only
    // because Host-warm sources are already in the Host index.
    for (BlockTransfer& transfer : it->second) {
        if (!success) {
            continue;
        }
        if (transfer.prefetch_from_storage && transfer.source) {
            coordinator_.CacheHostBlock(transfer.source, transfer.key);
        }
        if (transfer.destination && !transfer.key.content_hash.empty()) {
            coordinator_.CacheDeviceBlock(transfer.destination, transfer.key);
        }
    }
    load_backs_.erase(it);
}

TierTransferManager::InFlightHostBlocks TierTransferManager::inFlightHostBlocks() const {
    InFlightHostBlocks by_key;
    for (const auto& [op_id, write_back] : write_backs_) {
        for (const StoreTicket& ticket : write_back.tickets) {
            // StartPendingStores dedupes against in-flight keys, so each key
            // travels on at most one ticket.
            by_key.emplace(ticket.key, InFlightHostBlock{.block = ticket.host_block_ref, .op_id = op_id});
        }
    }
    return by_key;
}

std::optional<TierTransferManager::RetractionStores> TierTransferManager::StartRetractionStores(
    const std::string& request_id, std::int32_t request_pool_index, std::int32_t snapshot_slot,
    std::span<const BlockTable> tables, std::int32_t num_tokens) {
    const std::int32_t num_groups = coordinator_.NumGroups();
    std::vector<std::vector<ImageSlot>> published = coordinator_.PublishedDataSlots(tables, num_tokens);
    std::vector<std::vector<ImageSlot>> host_slots(static_cast<std::size_t>(num_groups));
    std::vector<std::uint32_t> pending_store_ops;
    const auto wait_for = [&pending_store_ops](std::uint32_t op_id) {
        if (std::find(pending_store_ops.begin(), pending_store_ops.end(), op_id) == pending_store_ops.end()) {
            pending_store_ops.push_back(op_id);
        }
    };

    // The L2 leg. A published slot rides Host L2 as its prefix entry: the
    // existing entry when the key is already Host-cached, the in-flight
    // ticket's block when an earlier store carries it (the image then waits
    // for that store too), else a Host block acquired in the Device block's
    // bucket and copied by this retraction's own write-back.
    struct Pending {
        std::uint32_t group_id;
        ImageSlot slot;
    };
    std::vector<Pending> to_allocate;
    std::vector<std::uint32_t> allocate_groups;
    std::vector<std::int32_t> allocate_buckets;
    if (coordinator_.HasHostPool()) {
        const InFlightHostBlocks in_flight = inFlightHostBlocks();
        for (std::int32_t g = 0; g < num_groups; ++g) {
            const GroupAllocator& allocator = coordinator_.Allocator(g);
            for (ImageSlot& slot : published[static_cast<std::size_t>(g)]) {
                if (CacheBlockRef cached = coordinator_.FindHostCachedBlock(slot.key)) {
                    slot.block = std::move(cached);
                    host_slots[static_cast<std::size_t>(g)].push_back(std::move(slot));
                    continue;
                }
                if (const auto it = in_flight.find(slot.key); it != in_flight.end()) {
                    wait_for(it->second.op_id);
                    slot.block = it->second.block;
                    host_slots[static_cast<std::size_t>(g)].push_back(std::move(slot));
                    continue;
                }
                allocate_groups.push_back(static_cast<std::uint32_t>(g));
                allocate_buckets.push_back(allocator.BucketOf(slot.block->Location()));
                to_allocate.push_back(Pending{.group_id = static_cast<std::uint32_t>(g), .slot = std::move(slot)});
            }
        }
    }
    std::vector<StoreTicket> tickets;
    std::vector<BlockTransfer> host_store_pairs;
    if (!to_allocate.empty()) {
        CacheCoordinator::HostAllocationBatch host_blocks =
            coordinator_.AcquireHostBlocks(allocate_groups, allocate_buckets);
        for (std::size_t i = 0; i < to_allocate.size(); ++i) {
            CacheBlockRef& host_block = host_blocks.blocks[i];
            if (!host_block) {
                continue;  // L2 is full of pinned entries: this slot rides the snapshot pool instead
            }
            ImageSlot& slot = to_allocate[i].slot;
            host_store_pairs.push_back(BlockTransfer{
                .group_id = to_allocate[i].group_id,
                .source = slot.block,
                .destination = host_block,
                .key = slot.key,
            });
            // The ticket's Host pin is dropped at the ACK, which publishes the
            // entry; the image's own pin on the same block outlives it.
            tickets.push_back(StoreTicket{slot.key, CacheBlockRef{}, host_block});
            slot.block = std::move(host_block);
            host_slots[to_allocate[i].group_id].push_back(std::move(slot));
        }
        for (std::vector<ImageSlot>& slots : host_slots) {
            std::ranges::sort(slots, {}, &ImageSlot::slot_index);
        }
    }

    // The tail leg: every other data slot, in the snapshot pool.
    std::optional<CacheCoordinator::ImageTaken> taken = coordinator_.TakeImage(tables, num_tokens, host_slots);
    if (!taken) {
        return std::nullopt;  // the acquired Host blocks and tickets die here, unused
    }

    RetractionStores stores{.image = std::move(taken->image)};
    if (!tickets.empty()) {
        const std::uint32_t op_id = nextOpId();
        std::vector<CacheTransfer> transfers = resolveTransfers(host_store_pairs);
        for (std::size_t i = 0; i < transfers.size(); ++i) {
            transfers[i].content_hash = host_store_pairs[i].key.content_hash;
            transfers[i].page_offset = host_store_pairs[i].key.page_offset;
        }
        const bool inserted =
            write_backs_.emplace(op_id, InFlightWriteBack{StoreSourceGuard::kStreamOrdered, std::move(tickets)}).second;
        _assert(inserted, "duplicate store op id");
        stores.host_store = WriteBackOperation{
            .op_id = op_id,
            .transfers = std::move(transfers),
            .source_pinned = false,
        };
        wait_for(op_id);
    }
    {
        const std::uint32_t op_id = nextOpId();
        std::vector<CacheBlockRef> destinations;
        destinations.reserve(taken->store_pairs.size());
        for (const BlockTransfer& pair : taken->store_pairs) {
            destinations.push_back(pair.destination);
        }
        stores.snapshot_store = SnapshotStoreOperation{
            .op_id = op_id,
            .request_id = request_id,
            .request_pool_index = request_pool_index,
            .snapshot_slot = snapshot_slot,
            .transfers = resolveTransfers(taken->store_pairs),
        };
        const bool inserted = snapshot_stores_
                                  .emplace(op_id, InFlightSnapshotStore{.request_id = request_id,
                                                                        .destinations = std::move(destinations)})
                                  .second;
        _assert(inserted, "duplicate snapshot store op id");
        wait_for(op_id);
    }
    stores.pending_store_ops = std::move(pending_store_ops);
    return stores;
}

SnapshotRestoreOperation TierTransferManager::StartSnapshotRestore(const std::string& request_id,
                                                                   std::int32_t request_pool_index,
                                                                   std::int32_t snapshot_slot,
                                                                   std::vector<BlockTransfer> load_pairs,
                                                                   std::vector<BlockTransfer> snapshot_pairs) {
    for (const BlockTransfer& pair : load_pairs) {
        _assert(!pair.key.content_hash.empty() && coordinator_.IsHostCachedBlock(pair.source->Location()),
                "a restore's Host L2 row must come from a published Host entry");
    }
    SnapshotRestoreOperation op{
        .op_id = nextOpId(),
        .request_id = request_id,
        .request_pool_index = request_pool_index,
        .snapshot_slot = snapshot_slot,
        .transfers = resolveTransfers(load_pairs),
        .source_tier = std::vector<HostTier>(load_pairs.size(), HostTier::kL2),
    };
    std::vector<CacheTransfer> pool_rows = resolveTransfers(snapshot_pairs);
    op.transfers.insert(op.transfers.end(), std::make_move_iterator(pool_rows.begin()),
                        std::make_move_iterator(pool_rows.end()));
    op.source_tier.insert(op.source_tier.end(), snapshot_pairs.size(), HostTier::kSnapshotPool);
    std::vector<BlockTransfer> transfers = std::move(load_pairs);
    transfers.insert(transfers.end(), std::make_move_iterator(snapshot_pairs.begin()),
                     std::make_move_iterator(snapshot_pairs.end()));
    const bool inserted =
        snapshot_restores_
            .emplace(op.op_id, InFlightSnapshotRestore{.request_id = request_id, .transfers = std::move(transfers)})
            .second;
    _assert(inserted, "duplicate snapshot restore op id");
    return op;
}

std::optional<std::string> TierTransferManager::CompleteSnapshotStore(std::uint32_t op_id) {
    auto it = snapshot_stores_.find(op_id);
    if (it == snapshot_stores_.end()) {
        return std::nullopt;
    }
    std::string request_id = std::move(it->second.request_id);
    snapshot_stores_.erase(it);
    return request_id;
}

std::optional<std::string> TierTransferManager::SnapshotRestoreRequest(std::uint32_t op_id) const {
    const auto it = snapshot_restores_.find(op_id);
    return it == snapshot_restores_.end() ? std::nullopt : std::optional<std::string>{it->second.request_id};
}

void TierTransferManager::CompleteSnapshotRestore(std::uint32_t op_id, bool publish) {
    auto it = snapshot_restores_.find(op_id);
    if (it == snapshot_restores_.end()) {
        return;
    }
    // The L2-tier rows are the request's own prefix pages, copied back whole:
    // publish them like an ordinary load-back's destinations.
    if (publish) {
        for (BlockTransfer& transfer : it->second.transfers) {
            if (transfer.destination && !transfer.key.content_hash.empty()) {
                coordinator_.CacheDeviceBlock(transfer.destination, transfer.key);
            }
        }
    }
    snapshot_restores_.erase(it);
}

std::vector<CacheTransfer> TierTransferManager::resolveTransfers(std::span<const BlockTransfer> block_transfers) const {
    std::vector<CacheTransfer> transfers;
    transfers.reserve(block_transfers.size());
    for (const BlockTransfer& block_transfer : block_transfers) {
        _assert(block_transfer.source && block_transfer.destination,
                "cache transfer requires pinned source and destination blocks");
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(block_transfer.group_id));
        // Every tier transfer pairs blocks of equal residue: under page-cyclic
        // sharding the rank that owns one end owns the other.
        _assert(manager.BucketOf(block_transfer.source->Location()) ==
                    manager.BucketOf(block_transfer.destination->Location()),
                "cache transfer pairs blocks of different owners");
        transfers.push_back(CacheTransfer{
            .group_id = block_transfer.group_id,
            .source_page = manager.ResolveCacheBlockId(block_transfer.source->Location()),
            .destination_page = manager.ResolveCacheBlockId(block_transfer.destination->Location()),
            .content_hash = block_transfer.key.content_hash,
            .page_offset = block_transfer.key.page_offset,
            .prefetch_from_storage = block_transfer.prefetch_from_storage,
        });
    }
    return transfers;
}

}  // namespace tokenspeed
