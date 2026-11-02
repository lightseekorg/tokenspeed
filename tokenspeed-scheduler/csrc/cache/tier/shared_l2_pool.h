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
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

// One Host L2 pool shared by many prefill ranks: the Host L2 cache code over pages
// it names but holds no bytes of. Pins and reservations end with their client.
class SharedL2Pool {
public:
    using ClientId = std::uint64_t;
    using Ticket = std::uint64_t;
    static constexpr std::size_t kMaxPendingPublications = 4096;

    struct Slot {
        CacheKey key;
        std::int32_t page{0};
    };
    // The pool's longest reusable prefix of a prompt (num_tokens 0 when it has
    // none); its slots stay pinned until the client releases the ticket.
    struct Hit {
        Ticket ticket{0};
        std::int32_t num_tokens{0};
        // How far the pool's KV pages reach, checkpoint or not.
        std::int32_t kv_tokens{0};
        std::optional<TailPublication> tail;
        std::vector<Slot> slots;
    };
    // A page, or why there is none.
    static constexpr std::int32_t kHeld = -1;    // the pool holds it or another client is writing it
    static constexpr std::int32_t kNoSlot = -2;  // every slot is pinned or reserved
    // Aligned with the requested keys: the page to write, or why there is none.
    struct Reservation {
        Ticket ticket{0};
        std::vector<std::int32_t> pages;
    };
    // A State checkpoint and the tail it ends in; it takes effect once the pool
    // holds their blocks, which may arrive in a later commit.
    struct Publication {
        std::vector<std::int32_t> tokens;
        std::vector<std::string> prefix_hashes;
        std::optional<TailPublication> tail;
    };
    struct Stats {
        std::int32_t total_lcm_blocks{0};
        std::int32_t free_lcm_blocks{0};
        std::int32_t cached_blocks{0};
        std::int32_t pinned_blocks{0};
        std::size_t reserved_blocks{0};
        std::uint64_t demoted_victims{0};
    };

    // Groups must be replicated. When fewer than half of headroom_lcm_blocks are
    // empty, a reservation evicts back up to it in one pass.
    SharedL2Pool(std::vector<CacheGroupSpec> specs, std::int32_t prefix_granularity, std::int32_t num_lcm_blocks,
                 std::int32_t headroom_lcm_blocks);
    SharedL2Pool(const SharedL2Pool&) = delete;
    SharedL2Pool& operator=(const SharedL2Pool&) = delete;

    Hit Probe(ClientId client, const CacheCoordinator::PromptPrefix& prompt);
    void Release(ClientId client, Ticket ticket);
    Reservation Reserve(ClientId client, std::span<const CacheKey> keys);
    void Commit(ClientId client, Ticket ticket, std::span<const Publication> publications);
    void Touch(std::span<const CacheKey> keys);
    void Disconnect(ClientId client);
    // Empties the pool between workloads; false while any client holds pins
    // or reservations.
    bool Clear();

    bool Contains(const CacheKey& key) const { return cache_.ContainsHostCachedBlock(key); }
    Stats GetStats() const;

private:
    struct Writes {
        std::vector<CacheKey> keys;
        std::vector<CacheBlockRef> blocks;
    };
    struct Client {
        std::map<Ticket, std::vector<CacheBlockRef>> lookups;
        std::map<Ticket, Writes> writes;
    };

    void endWrites(Writes& writes);
    static ContentHash publicationHash(const Publication& publication);
    bool publicationStored(const Publication& publication) const;
    void applyPublication(const Publication& publication);
    void keepHeadroom();

    // The coordinator requires a Device pool; the pool's owner never uses it.
    BlockPool unused_device_;
    BlockPool pool_;
    CacheCoordinator cache_;
    std::int32_t prefix_granularity_;
    std::int32_t headroom_lcm_blocks_;
    // The group with the fewest blocks per LCM block: whole LCM blocks are
    // requested through it.
    std::uint32_t headroom_group_{0};
    std::int32_t headroom_packing_{1};
    std::unordered_map<ClientId, Client> clients_;
    std::unordered_set<CacheKey, CacheKeyHash> writing_;
    std::size_t reserved_blocks_{0};
    // Publications whose blocks are not all stored yet, oldest first.
    std::unordered_map<ContentHash, Publication> pending_publications_;
    std::deque<ContentHash> pending_order_;
    Ticket next_ticket_{0};
};

}  // namespace tokenspeed
