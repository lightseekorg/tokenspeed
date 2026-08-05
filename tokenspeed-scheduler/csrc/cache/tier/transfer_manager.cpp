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
#include <ranges>
#include <utility>

#include "utils.h"

namespace tokenspeed {

std::vector<std::pair<std::uint32_t, CacheBlockLocation>>
TierTransferManager::DeviceLocationsReleasedOnStoreAck() const {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> locations;
    for (const auto& [_, tickets] : stores_) {
        for (const StoreTicket& ticket : tickets) {
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
        const KvCacheManager& manager =
            coordinator_.GroupManager(static_cast<std::int32_t>(candidate.key.group_id));
        CacheBlockRef host_block_ref = coordinator_.AcquireHostBlockForStore(candidate.key.group_id);
        if (!host_block_ref) {
            continue;
        }
        transfers.push_back(CacheTransfer{
            .group_id = candidate.key.group_id,
            .source_page = manager.ResolveKernelPageId(device_block_ref->Location()),
            .destination_page = manager.ResolveKernelPageId(host_block_ref->Location()),
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
    const bool inserted = stores_.emplace(op_id, std::move(tickets)).second;
    _assert(inserted, "duplicate store op id");
    return WriteBackOperation{op_id, std::move(transfers)};
}

LoadBackOperation TierTransferManager::StartPrefixLoad(std::vector<BlockTransfer> block_transfers) {
    _assert(!block_transfers.empty(), "prefix load requires at least one block transfer");
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    for (const BlockTransfer& pair : block_transfers) {
        _assert(coordinator_.IsHostCachedBlock(pair.source->Location()),
                "pinned Host block lost its cache entry before load emission");
    }
    const std::uint32_t op_id = nextOpId();
    const bool inserted = prefix_loads_.emplace(op_id, std::move(block_transfers)).second;
    _assert(inserted, "duplicate loadback op id");
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::optional<PreparedSnapshotTransfer> TierTransferManager::PrepareOffload(
    std::span<const BlockTable> device_tables) {
    _assert(device_tables.size() == static_cast<std::size_t>(coordinator_.NumGroups()),
            "snapshot requires one Device table per cache group");
    PreparedSnapshotTransfer prepared;
    prepared.destination_tables.reserve(device_tables.size());
    for (std::size_t group_index = 0; group_index < device_tables.size(); ++group_index) {
        const std::uint32_t group_id = static_cast<std::uint32_t>(group_index);
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(group_index));
        std::vector<CacheBlockRef> destination_blocks;
        destination_blocks.reserve(static_cast<std::size_t>(device_tables[group_index].NumBlocks()));
        for (const CacheBlockRef& source : device_tables[group_index].Blocks()) {
            if (!source) {
                destination_blocks.emplace_back();
                continue;
            }
            CacheBlockRef destination = host_pool_.AcquireBlock(group_id, manager.CacheBlocksPerLcmBlock());
            if (!destination) {
                return std::nullopt;
            }
            prepared.block_transfers.push_back(BlockTransfer{
                .group_id = group_id,
                .source = source,
                .destination = destination,
            });
            destination_blocks.push_back(std::move(destination));
        }
        prepared.destination_tables.push_back(BlockTable::FromBlocks(
            std::move(destination_blocks), device_tables[group_index].AvailableTokens()));
    }
    return prepared;
}

std::optional<PreparedSnapshotTransfer> TierTransferManager::PrepareRestore(std::span<const BlockTable> host_tables) {
    _assert(host_tables.size() == static_cast<std::size_t>(coordinator_.NumGroups()),
            "snapshot requires one Host table per cache group");
    std::vector<BlockTable> dense_tables(host_tables.size());
    std::vector<GroupDemand> demands;
    demands.reserve(host_tables.size());
    for (std::size_t group_index = 0; group_index < host_tables.size(); ++group_index) {
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(group_index));
        const std::int32_t block_count = static_cast<std::int32_t>(
            std::ranges::count_if(host_tables[group_index].Blocks(), [](const CacheBlockRef& block_ref) {
                return static_cast<bool>(block_ref);
            }));
        demands.push_back(GroupDemand{
            .table = &dense_tables[group_index],
            .num_tokens = block_count * manager.CacheBlockTokens(),
        });
    }
    std::optional<KvCacheCoordinator::AdmissionResult> allocation =
        coordinator_.Admit(coordinator_.ProbePrefix({}), demands);
    if (!allocation) {
        return std::nullopt;
    }

    PreparedSnapshotTransfer prepared;
    prepared.new_device_page_ids = std::move(allocation->new_page_ids);
    prepared.destination_tables.reserve(host_tables.size());
    for (std::size_t group_index = 0; group_index < host_tables.size(); ++group_index) {
        const std::uint32_t group_id = static_cast<std::uint32_t>(group_index);
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(group_index));
        std::vector<CacheBlockRef> allocated = std::move(dense_tables[group_index]).TakeBlocks();
        auto next = allocated.begin();
        std::vector<CacheBlockRef> destination_blocks;
        destination_blocks.reserve(static_cast<std::size_t>(host_tables[group_index].NumBlocks()));
        for (const CacheBlockRef& source : host_tables[group_index].Blocks()) {
            if (!source) {
                destination_blocks.emplace_back();
                continue;
            }
            _assert(next != allocated.end(), "snapshot allocation returned too few Device blocks");
            CacheBlockRef destination = std::move(*next++);
            prepared.block_transfers.push_back(BlockTransfer{
                .group_id = group_id,
                .source = source,
                .destination = destination,
            });
            destination_blocks.push_back(std::move(destination));
        }
        _assert(next == allocated.end(), "snapshot allocation returned extra Device blocks");
        prepared.destination_tables.push_back(BlockTable::FromBlocks(
            std::move(destination_blocks), host_tables[group_index].AvailableTokens()));
    }
    return prepared;
}

WriteBackOperation TierTransferManager::StartOffload(std::string request_id,
                                                     std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted =
        retraction_offloads_
            .emplace(op_id, SnapshotTransfer{std::move(request_id), std::move(block_transfers)})
            .second;
    _assert(inserted, "duplicate retraction cache op id");
    return WriteBackOperation{op_id, std::move(transfers)};
}

LoadBackOperation TierTransferManager::StartRestore(std::string request_id,
                                                    std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted =
        recovery_loads_.emplace(op_id, SnapshotTransfer{std::move(request_id), std::move(block_transfers)}).second;
    _assert(inserted, "duplicate recovery cache op id");
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::optional<std::string> TierTransferManager::CompleteWriteBack(std::uint32_t op_id, bool success) {
    if (auto it = retraction_offloads_.find(op_id); it != retraction_offloads_.end()) {
        std::string request_id = std::move(it->second.request_id);
        retraction_offloads_.erase(it);
        return request_id;
    }

    auto it = stores_.find(op_id);
    if (it == stores_.end()) {
        return std::nullopt;
    }
    for (const StoreTicket& ticket : it->second) {
        store_keys_.erase(ticket.key);
    }
    std::vector<StoreTicket> tickets = std::move(it->second);
    stores_.erase(it);
    for (StoreTicket& ticket : tickets) {
        if (success) {
            coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key);
        }
    }
    return std::nullopt;
}

std::optional<std::string> TierTransferManager::CompleteLoadBack(std::uint32_t op_id, bool success) {
    if (auto it = recovery_loads_.find(op_id); it != recovery_loads_.end()) {
        _assert(success, "snapshot recovery H2D must not fail");
        std::string request_id = std::move(it->second.request_id);
        recovery_loads_.erase(it);
        return request_id;
    }

    auto it = prefix_loads_.find(op_id);
    if (it == prefix_loads_.end()) {
        return std::nullopt;
    }
    _assert(success, "host loadback failed: host bytes integrity");
    prefix_loads_.erase(it);
    return std::nullopt;
}

std::vector<CacheTransfer> TierTransferManager::resolveTransfers(
    std::span<const BlockTransfer> block_transfers) const {
    std::vector<CacheTransfer> transfers;
    transfers.reserve(block_transfers.size());
    for (const BlockTransfer& block_transfer : block_transfers) {
        _assert(block_transfer.source && block_transfer.destination,
                "cache transfer requires pinned source and destination blocks");
        const KvCacheManager& manager =
            coordinator_.GroupManager(static_cast<std::int32_t>(block_transfer.group_id));
        transfers.push_back(CacheTransfer{
            .group_id = block_transfer.group_id,
            .source_page = manager.ResolveKernelPageId(block_transfer.source->Location()),
            .destination_page = manager.ResolveKernelPageId(block_transfer.destination->Location()),
        });
    }
    return transfers;
}

}  // namespace tokenspeed
