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
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/coordinator/kv_cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/tier/transfer.h"

namespace tokenspeed {

// Temporary ownership between fallible snapshot allocation and the request's
// committed state transition. Destruction rolls back an unstarted copy.
struct PreparedSnapshotTransfer {
    std::vector<BlockTable> destination_tables;
    std::vector<BlockTransfer> block_transfers;
    std::vector<std::vector<std::int32_t>> new_device_page_ids;
};

// Owns the mechanics and asynchronous lifetime of transfers between Device and
// Host cache tiers. Scheduling policy and request state transitions stay in
// Scheduler.
class TierTransferManager {
public:
    TierTransferManager(KvCacheCoordinator& coordinator, BlockPool& host_pool)
        : coordinator_{coordinator}, host_pool_{host_pool} {}

    std::optional<WriteBackOperation> StartPendingStores();
    LoadBackOperation StartPrefixLoad(std::vector<BlockTransfer> block_transfers);

    std::optional<PreparedSnapshotTransfer> PrepareOffload(std::span<const BlockTable> device_tables);
    std::optional<PreparedSnapshotTransfer> PrepareRestore(std::span<const BlockTable> host_tables);
    WriteBackOperation StartOffload(std::string request_id, std::vector<BlockTransfer> block_transfers);
    LoadBackOperation StartRestore(std::string request_id, std::vector<BlockTransfer> block_transfers);

    // Returns a request id only when the completed operation belongs to a
    // snapshot migration. Ordinary cache operations are retired internally.
    std::optional<std::string> CompleteWriteBack(std::uint32_t op_id, bool success);
    std::optional<std::string> CompleteLoadBack(std::uint32_t op_id, bool success);

    bool HasStoresInFlight() const { return !stores_.empty(); }
    bool HasPrefixLoadsInFlight() const { return !prefix_loads_.empty(); }
    bool HasSnapshotsInFlight() const { return !retraction_offloads_.empty() || !recovery_loads_.empty(); }
    bool HasAnyInFlight() const {
        return HasStoresInFlight() || HasPrefixLoadsInFlight() || HasSnapshotsInFlight();
    }
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> DeviceLocationsReleasedOnStoreAck() const;

private:
    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
    };

    struct SnapshotTransfer {
        std::string request_id;
        std::vector<BlockTransfer> block_transfers;
    };

    std::uint32_t nextOpId() { return next_op_id_++; }
    std::vector<CacheTransfer> resolveTransfers(std::span<const BlockTransfer> block_transfers) const;

    KvCacheCoordinator& coordinator_;
    BlockPool& host_pool_;
    std::unordered_map<std::uint32_t, std::vector<StoreTicket>> stores_;
    std::unordered_set<CacheKey, CacheKeyHash> store_keys_;
    std::unordered_map<std::uint32_t, std::vector<BlockTransfer>> prefix_loads_;
    std::unordered_map<std::uint32_t, SnapshotTransfer> retraction_offloads_;
    std::unordered_map<std::uint32_t, SnapshotTransfer> recovery_loads_;
    std::uint32_t next_op_id_{0};
};

}  // namespace tokenspeed
