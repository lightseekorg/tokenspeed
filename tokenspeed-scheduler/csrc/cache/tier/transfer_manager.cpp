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

std::optional<WriteBackOperation> TierTransferManager::StartPendingStores() {
    std::vector<CacheKey> keys;
    std::vector<CacheBlockRef> device_block_refs;
    std::vector<std::uint32_t> group_ids;
    std::unordered_set<CacheKey, CacheKeyHash> batch_keys;
    for (auto& candidate : coordinator_.TakePendingStores()) {
        if (coordinator_.ContainsHostCachedBlock(candidate.key) || store_keys_.contains(candidate.key) ||
            !batch_keys.insert(candidate.key).second) {
            continue;
        }

        const CacheBlockRef device_block_ref = coordinator_.AcquireDeviceCachedBlock(candidate.key);
        if (!device_block_ref) {
            continue;
        }

        group_ids.push_back(candidate.key.group_id);
        keys.push_back(std::move(candidate.key));
        device_block_refs.push_back(std::move(device_block_ref));
    }

    if (keys.empty()) {
        return std::nullopt;
    }

    CacheCoordinator::HostAllocationBatch host_allocation = coordinator_.AcquireHostBlocks(group_ids);
    _assert(host_allocation.blocks.size() == keys.size(), "Host allocation result must stay aligned");

    std::vector<CacheTransfer> transfers;
    std::vector<StoreTicket> tickets;
    transfers.reserve(host_allocation.stats.allocated);
    tickets.reserve(host_allocation.stats.allocated);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CacheBlockRef& host_block_ref = host_allocation.blocks[i];
        if (!host_block_ref) {
            continue;
        }
        // The source page id is resolved here and not pinned: the forward
        // thread's stream orders the copy ahead of any later reuse.
        const GroupAllocator& manager = coordinator_.Allocator(static_cast<std::int32_t>(group_ids[i]));
        transfers.push_back(CacheTransfer{
            .group_id = group_ids[i],
            .source_page = manager.ResolveCacheBlockId(device_block_refs[i]->Location()),
            .destination_page = manager.ResolveCacheBlockId(host_block_ref->Location()),
        });
        tickets.push_back(StoreTicket{
            std::move(keys[i]),
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
        _assert(coordinator_.IsHostCachedBlock(pair.source->Location()),
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
    load_backs_.erase(op_id);
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
        });
    }
    return transfers;
}

}  // namespace tokenspeed
