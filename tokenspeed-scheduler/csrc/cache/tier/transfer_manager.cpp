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

#include <utility>

#include "utils.h"

namespace tokenspeed {

std::vector<std::pair<std::uint32_t, CacheBlockLocation>> TierTransferManager::DeviceLocationsReleasedOnStoreAck()
    const {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> locations;
    for (const auto& [_, stores] : write_backs_) {
        for (const StoreTicket& ticket : stores) {
            locations.emplace_back(ticket.key.group_id, ticket.device_block_ref->Location());
        }
    }
    return locations;
}

std::optional<WriteBackOperation> TierTransferManager::StartPendingStores() {
    std::vector<CacheTransfer> transfers;
    std::vector<StoreTicket> tickets;
    std::unordered_set<CacheKey, CacheKeyHash> batch_keys;
    for (auto& candidate : coordinator_.TakePendingStores()) {
        if (coordinator_.ContainsHostCachedBlock(candidate.key) || store_keys_.contains(candidate.key) ||
            !batch_keys.insert(candidate.key).second) {
            continue;
        }

        CacheBlockRef device_block_ref = coordinator_.AcquireDeviceCachedBlock(candidate.key);
        if (!device_block_ref) {
            continue;
        }
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(candidate.key.group_id));
        CacheBlockRef host_block_ref = coordinator_.AcquireHostBlock(candidate.key.group_id);
        if (!host_block_ref) {
            continue;
        }
        transfers.push_back(CacheTransfer{
            .group_id = candidate.key.group_id,
            .source_page = manager.ResolveCacheBlockId(device_block_ref->Location()),
            .destination_page = manager.ResolveCacheBlockId(host_block_ref->Location()),
            .content_hash = candidate.key.content_hash,
            .page_offset = candidate.key.page_offset,
        });
        tickets.push_back(StoreTicket{
            std::move(candidate.key),
            std::move(device_block_ref),
            std::move(host_block_ref),
        });
    }

    if (transfers.empty()) {
        return std::nullopt;
    }
    const std::uint32_t op_id = nextOpId();
    for (const StoreTicket& ticket : tickets) {
        store_keys_.insert(ticket.key);
    }
    const bool inserted = write_backs_.emplace(op_id, std::move(tickets)).second;
    _assert(inserted, "duplicate store op id");
    return WriteBackOperation{op_id, std::move(transfers)};
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

void TierTransferManager::CompleteWriteBack(std::uint32_t op_id) {
    // The runtime emits this ACK only after the asynchronous copy completes.
    // Transfer errors terminate the runtime and must never publish cache state.
    auto it = write_backs_.find(op_id);
    if (it == write_backs_.end()) {
        return;
    }
    std::vector<StoreTicket> stores = std::move(it->second);
    write_backs_.erase(it);
    for (const StoreTicket& ticket : stores) {
        store_keys_.erase(ticket.key);
    }
    for (StoreTicket& ticket : stores) {
        coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key);
    }
}

void TierTransferManager::CompleteLoadBack(std::uint32_t op_id) {
    auto it = load_backs_.find(op_id);
    if (it == load_backs_.end()) {
        return;
    }
    for (BlockTransfer& transfer : it->second) {
        if (transfer.prefetch_from_storage && transfer.source) {
            coordinator_.CacheHostBlock(transfer.source, transfer.key);
        }
    }
    load_backs_.erase(it);
}

std::vector<CacheTransfer> TierTransferManager::resolveTransfers(std::span<const BlockTransfer> block_transfers) const {
    std::vector<CacheTransfer> transfers;
    transfers.reserve(block_transfers.size());
    for (const BlockTransfer& block_transfer : block_transfers) {
        _assert(block_transfer.source && block_transfer.destination,
                "cache transfer requires pinned source and destination blocks");
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(block_transfer.group_id));
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
