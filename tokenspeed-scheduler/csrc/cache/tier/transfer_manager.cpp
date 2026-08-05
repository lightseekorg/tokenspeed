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
#include <limits>
#include <ranges>
#include <utility>

#include "utils.h"

namespace tokenspeed {

std::vector<std::pair<std::uint32_t, CacheBlockLocation>> TierTransferManager::DeviceLocationsReleasedOnStoreAck()
    const {
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> locations;
    for (const auto& [_, write_back] : write_backs_) {
        for (const StoreTicket& ticket : write_back.stores) {
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
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(candidate.key.group_id));
        CacheBlockRef host_block_ref = coordinator_.AcquireHostBlock(candidate.key.group_id);
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
    const bool inserted =
        write_backs_.emplace(op_id, WriteBackTicket{.stores = std::move(tickets), .retraction = std::nullopt}).second;
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

std::optional<PreparedRetraction> TierTransferManager::PrepareRetraction(
    std::span<const std::string> page_hashes, std::uint64_t access_epoch,
    std::span<const BlockTable> device_tables) {
    _assert(device_tables.size() == static_cast<std::size_t>(coordinator_.NumGroups()),
            "retraction requires one Device table per cache group");
    std::optional<CoordinatorMatch> host_match =
        coordinator_.TryAcquireRetractionHostPrefix(page_hashes, access_epoch, device_tables);
    if (!host_match) {
        return std::nullopt;
    }
    PreparedRetraction prepared;
    prepared.host_prefix_tokens = host_match->num_common_tokens;
    prepared.host_tables.reserve(device_tables.size());
    for (std::size_t group_index = 0; group_index < device_tables.size(); ++group_index) {
        const std::uint32_t group_id = static_cast<std::uint32_t>(group_index);
        std::vector<CacheBlockRef>& matched_blocks = host_match->per_group[group_index].blocks;
        std::vector<CacheBlockRef> destination_blocks;
        destination_blocks.reserve(static_cast<std::size_t>(device_tables[group_index].NumBlocks()));
        for (std::size_t block_index = 0; block_index < device_tables[group_index].Blocks().size(); ++block_index) {
            const CacheBlockRef& source = device_tables[group_index].Blocks()[block_index];
            if (!source) {
                destination_blocks.emplace_back();
                continue;
            }
            if (block_index < matched_blocks.size() && matched_blocks[block_index]) {
                destination_blocks.push_back(std::move(matched_blocks[block_index]));
                continue;
            }
            CacheBlockRef destination = coordinator_.AcquireHostBlock(group_id);
            FatalCheck(static_cast<bool>(destination),
                       "Retraction Host allocation diverged from its capacity precheck");
            prepared.transfers.push_back(BlockTransfer{
                .group_id = group_id,
                .source = source,
                .destination = destination,
            });
            destination_blocks.push_back(std::move(destination));
        }
        prepared.host_tables.push_back(
            BlockTable::FromBlocks(std::move(destination_blocks), device_tables[group_index].AvailableTokens()));
    }
    return prepared;
}

std::optional<PreparedRecovery> TierTransferManager::PrepareRecovery(
    std::span<const std::string> page_hashes, std::uint64_t access_epoch, std::span<const BlockTable> host_tables,
    std::int32_t decode_reserve_tokens) {
    _assert(host_tables.size() == static_cast<std::size_t>(coordinator_.NumGroups()),
            "retraction recovery requires one Host table per cache group");
    _assert(decode_reserve_tokens >= 0, "retraction recovery reserve must be non-negative");
    std::vector<BlockTable> device_tables(host_tables.size());
    std::vector<GroupDemand> demands;
    demands.reserve(host_tables.size());
    KvCacheCoordinator::PrefixProbe probe = coordinator_.ProbeDecodeDevicePrefix(page_hashes);
    const std::int32_t device_prefix_tokens = probe.device.num_common_tokens;
    for (std::size_t group_index = 0; group_index < host_tables.size(); ++group_index) {
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(group_index));
        const BlockTable& host_table = host_tables[group_index];
        const std::int64_t consumed_tokens_wide =
            static_cast<std::int64_t>(host_table.NumBlocks()) * manager.CacheBlockTokens() -
            host_table.AvailableTokens();
        _assert(consumed_tokens_wide >= 0 && consumed_tokens_wide <= std::numeric_limits<std::int32_t>::max(),
                "retraction recovery token extent exceeds int32 range");
        const std::int32_t consumed_tokens = static_cast<std::int32_t>(consumed_tokens_wide);
        GroupDemand demand{
            .table = &device_tables[group_index],
            .num_tokens = std::max(consumed_tokens - device_prefix_tokens, 0),
            .reserve_tokens = decode_reserve_tokens,
        };
        if (coordinator_.GroupKind(static_cast<std::int32_t>(group_index)) == AttnKind::kMambaState) {
            const std::int32_t prefix_blocks = device_prefix_tokens / manager.CacheBlockTokens();
            const std::size_t search_begin = std::min<std::size_t>(static_cast<std::size_t>(prefix_blocks),
                                                                   host_table.Blocks().size());
            const auto first_live = std::find_if(host_table.Blocks().begin() + search_begin,
                                                 host_table.Blocks().end(), [](const CacheBlockRef& block_ref) {
                                                     return static_cast<bool>(block_ref);
                                                 });
            demand.num_tokens = consumed_tokens;
            if (first_live != host_table.Blocks().end()) {
                demand.materialized_suffix_start = static_cast<std::int32_t>(first_live - host_table.Blocks().begin());
            }
        }
        demands.push_back(demand);
    }
    std::optional<KvCacheCoordinator::AdmissionResult> allocation =
        coordinator_.Admit(std::move(probe), demands, access_epoch);
    if (!allocation) {
        return std::nullopt;
    }

    PreparedRecovery prepared;
    prepared.new_device_page_ids = std::move(allocation->new_page_ids);
    prepared.device_tables.reserve(host_tables.size());
    for (std::size_t group_index = 0; group_index < host_tables.size(); ++group_index) {
        const std::uint32_t group_id = static_cast<std::uint32_t>(group_index);
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(group_index));
        const std::int32_t prefix_blocks = device_prefix_tokens / manager.CacheBlockTokens();
        const auto& destination_blocks = device_tables[group_index].Blocks();
        _assert(destination_blocks.size() >= host_tables[group_index].Blocks().size(),
                "retraction recovery allocated too few Device blocks");
        for (std::size_t block_index = 0; block_index < host_tables[group_index].Blocks().size(); ++block_index) {
            const CacheBlockRef& source = host_tables[group_index].Blocks()[block_index];
            const CacheBlockRef& destination = destination_blocks[block_index];
            if (!source || !destination || static_cast<std::int32_t>(block_index) < prefix_blocks) {
                continue;
            }
            prepared.transfers.push_back(BlockTransfer{
                .group_id = group_id,
                .source = source,
                .destination = destination,
            });
        }
        prepared.device_tables.push_back(std::move(device_tables[group_index]));
    }
    return prepared;
}

WriteBackOperation TierTransferManager::StartRetraction(std::string request_id,
                                                        std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted = write_backs_
                              .emplace(op_id, WriteBackTicket{.stores = {},
                                                              .retraction = RetractionTicket{
                                                                  std::move(request_id), std::move(block_transfers)}})
                              .second;
    _assert(inserted, "duplicate retraction cache op id");
    return WriteBackOperation{op_id, std::move(transfers)};
}

LoadBackOperation TierTransferManager::StartRecovery(std::vector<BlockTransfer> block_transfers) {
    _assert(!block_transfers.empty(), "retraction recovery load requires at least one block transfer");
    return startLoadBack(std::move(block_transfers));
}

LoadBackOperation TierTransferManager::startLoadBack(std::vector<BlockTransfer> block_transfers) {
    std::vector<CacheTransfer> transfers = resolveTransfers(block_transfers);
    const std::uint32_t op_id = nextOpId();
    const bool inserted = load_backs_.emplace(op_id, std::move(block_transfers)).second;
    _assert(inserted, "duplicate loadback op id");
    return LoadBackOperation{op_id, std::move(transfers)};
}

std::optional<std::string> TierTransferManager::CompleteWriteBack(std::uint32_t op_id) {
    auto it = write_backs_.find(op_id);
    if (it == write_backs_.end()) {
        return std::nullopt;
    }
    WriteBackTicket write_back = std::move(it->second);
    write_backs_.erase(it);
    if (write_back.retraction) {
        return std::move(write_back.retraction->request_id);
    }
    for (const StoreTicket& ticket : write_back.stores) {
        store_keys_.erase(ticket.key);
    }
    for (StoreTicket& ticket : write_back.stores) {
        coordinator_.CacheHostBlock(ticket.host_block_ref, ticket.key);
    }
    return std::nullopt;
}

bool TierTransferManager::HasStoresInFlight() const {
    return std::ranges::any_of(write_backs_, [](const auto& entry) { return !entry.second.stores.empty(); });
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
        const KvCacheManager& manager = coordinator_.GroupManager(static_cast<std::int32_t>(block_transfer.group_id));
        transfers.push_back(CacheTransfer{
            .group_id = block_transfer.group_id,
            .source_page = manager.ResolveKernelPageId(block_transfer.source->Location()),
            .destination_page = manager.ResolveKernelPageId(block_transfer.destination->Location()),
        });
    }
    return transfers;
}

}  // namespace tokenspeed
