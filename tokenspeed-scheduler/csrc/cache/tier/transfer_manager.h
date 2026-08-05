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
#include "cache/tier/transfer.h"

namespace tokenspeed {

// Temporary ownership between fallible Host allocation and the committed
// retraction transition. Destruction rolls back an unstarted writeback.
struct PreparedRetraction {
    std::vector<BlockTable> host_tables;
    std::vector<BlockTransfer> transfers;
    std::int32_t host_prefix_tokens{0};
};

// Temporary ownership between fallible Device admission and the committed
// recovery transition. Destruction rolls back an unstarted loadback.
struct PreparedRecovery {
    std::vector<BlockTable> device_tables;
    std::vector<BlockTransfer> transfers;
    std::vector<std::vector<std::int32_t>> new_device_page_ids;
};

// Owns the mechanics and asynchronous lifetime of transfers between Device and
// Host cache tiers. Scheduling policy and request state transitions stay in
// Scheduler.
class TierTransferManager {
public:
    explicit TierTransferManager(KvCacheCoordinator& coordinator) : coordinator_{coordinator} {}

    std::optional<WriteBackOperation> StartPendingStores();
    LoadBackOperation StartPrefixLoad(std::vector<BlockTransfer> block_transfers);

    std::optional<PreparedRetraction> PrepareRetraction(std::span<const std::string> page_hashes,
                                                        std::uint64_t access_epoch,
                                                        std::span<const BlockTable> device_tables);
    std::optional<PreparedRecovery> PrepareRecovery(std::span<const std::string> page_hashes,
                                                    std::uint64_t access_epoch,
                                                    std::span<const BlockTable> host_tables,
                                                    std::int32_t decode_reserve_tokens);
    WriteBackOperation StartRetraction(std::string request_id, std::vector<BlockTransfer> block_transfers);
    LoadBackOperation StartRecovery(std::vector<BlockTransfer> block_transfers);

    // Returns a request id only for a retraction writeback. Ordinary cache
    // stores are retired internally.
    std::optional<std::string> CompleteWriteBack(std::uint32_t op_id);
    void CompleteLoadBack(std::uint32_t op_id);

    bool HasStoresInFlight() const;
    bool HasLoadBacksInFlight() const { return !load_backs_.empty(); }
    bool HasAnyInFlight() const { return !write_backs_.empty() || !load_backs_.empty(); }
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> DeviceLocationsReleasedOnStoreAck() const;

private:
    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
    };

    struct RetractionTicket {
        std::string request_id;
        // Pins both tiers until the runtime acknowledges the copy.
        std::vector<BlockTransfer> transfers;
    };

    struct WriteBackTicket {
        std::vector<StoreTicket> stores;
        std::optional<RetractionTicket> retraction;
    };

    std::uint32_t nextOpId() { return next_op_id_++; }
    LoadBackOperation startLoadBack(std::vector<BlockTransfer> block_transfers);
    std::vector<CacheTransfer> resolveTransfers(std::span<const BlockTransfer> block_transfers) const;

    KvCacheCoordinator& coordinator_;
    std::unordered_map<std::uint32_t, WriteBackTicket> write_backs_;
    std::unordered_set<CacheKey, CacheKeyHash> store_keys_;
    // Each transfer pins both tiers until the runtime acknowledges the copy.
    std::unordered_map<std::uint32_t, std::vector<BlockTransfer>> load_backs_;
    std::uint32_t next_op_id_{0};
};

}  // namespace tokenspeed
