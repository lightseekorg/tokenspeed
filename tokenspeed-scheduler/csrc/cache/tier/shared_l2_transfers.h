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
#include <deque>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/tier/shared_l2.h"
#include "cache/tier/transfer.h"

namespace tokenspeed {

// A rank's side of a shared L2 pool: proxy Host entries for its lookups and
// load-backs, and writes of newly published Device blocks.
class SharedL2Transfers {
public:
    SharedL2Transfers(CacheCoordinator& coordinator, SharedL2Config config);

    // Mirrors a lookup's hit into proxy Host entries held until ReleaseLookup or
    // HoldLoad; prior_hash precedes the hit's tail. Returns the blocks created.
    std::vector<SharedL2ProxyPage> ApplyLookup(const SharedL2LookupResult& result, const ContentHash& prior_hash);
    bool LookedUp(const std::string& request_id) const { return lookups_.contains(request_id); }
    // The request no longer needs its lookup: admitted, or gone.
    void ReleaseLookup(const std::string& request_id);
    // A load-back from proxy entries holds them until its completion.
    void HoldLoad(const LoadBackOperation& load);
    void ReleaseLoad(std::uint32_t op_id);

    // Takes the coordinator's newly published Device blocks and Device-hit
    // keys; a published State checkpoint adds its lineage to the next write.
    void Collect();
    void Publish(const CompletedPages& completed);
    void StartWrites();
    // A cache clear: queued blocks are released unwritten.
    void DropWrites();
    void Complete(const cache::SharedL2WriteDone& event);

    std::vector<SharedL2Operation> TakeCommands() { return std::exchange(commands_, {}); }
    bool HasPending() const;
    // Writes the runtime has not completed.
    bool HasInFlight() const { return !writes_.empty(); }

private:
    struct ProxyEntry {
        std::uint64_t ticket{0};
        std::int32_t page{0};
        std::int32_t block_id{0};
        CacheBlockRef block;
        std::int32_t holders{0};
    };

    static std::uint64_t blockKey(std::uint32_t group_id, std::int32_t block_id) {
        return (static_cast<std::uint64_t>(group_id) << 32U) | static_cast<std::uint32_t>(block_id);
    }
    void hold(std::span<const CacheKey> keys);
    void release(std::span<const CacheKey> keys);
    std::uint64_t nextOpId() { return ++next_op_id_; }

    CacheCoordinator& coordinator_;
    SharedL2Config config_;
    std::unordered_map<CacheKey, ProxyEntry, CacheKeyHash> proxy_;
    // A load-back names its sources by proxy block.
    std::unordered_map<std::uint64_t, CacheKey> keys_by_block_;
    // Proxy entries alive per lookup ticket; the ticket is released with its
    // last entry.
    std::unordered_map<std::uint64_t, std::int32_t> ticket_entries_;
    std::unordered_map<std::string, std::vector<CacheKey>> lookups_;
    std::unordered_map<std::uint32_t, std::vector<CacheKey>> loads_;
    std::unordered_map<std::uint64_t, std::vector<CacheBlockRef>> writes_;
    std::deque<std::pair<CacheKey, CacheBlockRef>> queued_writes_;
    std::vector<SharedL2Publication> publications_;
    std::vector<SharedL2Operation> commands_;
    std::uint64_t next_op_id_{0};
};

}  // namespace tokenspeed
