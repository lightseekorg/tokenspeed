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

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache/coordinator/cache_coordinator.h"
#include "cache/core/block_pool.h"
#include "cache/core/cache_types.h"

namespace tokenspeed {

struct SharedL2PoolTestAccess;

// One Host L2 pool shared by many prefill ranks, as its owner process keeps
// it: the Host L2 cache code (prefix index, tails, checkpoint retention,
// eviction) over pages it names but holds no bytes of. Ranks are its clients.
// A client pins what a lookup hit until it releases the ticket, and reserves
// the pages it writes until it commits them; both end with the client
// (Disconnect).
class SharedL2Pool {
public:
    using ClientId = std::uint64_t;
    using Ticket = std::uint64_t;

    // A pool page: `pool_page` is the block id in the pool's Host layout of
    // the key's group, `bucket` the shard it lies in (always 0 until the pool
    // holds sharded groups).
    struct Page {
        CacheKey key;
        std::int32_t bucket{0};
        std::int32_t pool_page{0};
    };
    // The pool's longest reusable prefix of a prompt (num_tokens 0 when it has
    // none); its pages stay pinned until the client releases the ticket
    // (0: nothing is pinned).
    struct Hit {
        Ticket ticket{0};
        std::int32_t num_tokens{0};
        // How far the pool's KV pages reach, checkpoint or not.
        std::int32_t kv_tokens{0};
        std::optional<TailPublication> tail;
        std::vector<Page> pages;
    };
    // A page, or why there is none, in the words of the local Host store
    // path's counters (WriteBackStats). A key answered kInFlight whose writer
    // leaves before its commit is then neither held nor being written: its
    // rank offers it again only when a request that published it finishes.
    static constexpr std::int32_t kHostCached = -1;   // the pool holds the key
    static constexpr std::int32_t kNoPlacement = -2;  // no page of its group and bucket is free
    static constexpr std::int32_t kInFlight = -3;     // another reservation writes it, or this one names it twice
    // Aligned with the requested keys: the page to write, or why there is
    // none. Ticket 0 reserves nothing.
    struct Reservation {
        Ticket ticket{0};
        std::vector<std::int32_t> pages;
    };
    struct Stats {
        std::int32_t num_lcm_blocks{0};
        std::int32_t empty_lcm_blocks{0};
        std::int32_t cached_blocks{0};
        std::int32_t pinned_blocks{0};
        std::size_t reserved_blocks{0};
        // Lookup tickets not released yet.
        std::size_t open_lookups{0};
        std::uint64_t evicted_blocks{0};
        std::size_t checkpoints{0};
    };

    // Groups must be replicated. host_group_packing lays the pool out as a
    // rank's Host tier is (SchedulerConfig::host_group_packing). When fewer
    // than half of headroom_lcm_blocks are empty, a reservation evicts back up
    // to it in one pass. The retention window counts every lookup, hit or
    // miss, over all clients.
    SharedL2Pool(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                 std::vector<std::int32_t> host_group_packing, std::int32_t num_lcm_blocks,
                 std::int32_t headroom_lcm_blocks, std::int64_t checkpoint_retention_requests,
                 std::int32_t checkpoint_retention_leaf_distance);
    SharedL2Pool(const SharedL2Pool&) = delete;
    SharedL2Pool& operator=(const SharedL2Pool&) = delete;

    // The prompt carries its tokens up to max_tokens.
    Hit Lookup(ClientId client, const PromptPrefix& prompt);
    // Ends a hit's pins. Throws for a ticket no lookup of the client holds.
    void Release(ClientId client, Ticket ticket);
    // A key the pool holds, one another write reserved or a repeated one gets
    // no page (kHostCached, kInFlight). The keys the pool holds are refreshed, as a rank's
    // Device publication refreshes its own Host copies, before pages are
    // taken for the others: the reservation does not evict them.
    Reservation Reserve(ClientId client, std::span<const CacheKey> keys, std::span<const std::int32_t> buckets);
    // Publishes a reservation's blocks, with the checkpoints
    // whose State block is among them (CacheCoordinator::CacheWrittenHostBlocks).
    // A checkpoint's access epoch names the request that published it
    // on its rank: checkpoints one request published are no step
    // apart in the retention ranking (CheckpointTree). Throws, with no
    // change, for a ticket under which the client reserves nothing.
    void Commit(ClientId client, Ticket ticket, std::span<const CheckpointTree::Checkpoint> checkpoints);
    void Touch(std::span<const CacheKey> keys);
    // Ends everything the client holds. The pool keeps no list of clients: a
    // call under this id afterwards starts it anew, so the service in front
    // of the pool refuses a client that has left.
    void Disconnect(ClientId client);
    // Empties the pool between workloads; false while any client holds pins
    // or reservations.
    bool ClearCache();

    Stats GetStats() const;

private:
    friend struct SharedL2PoolTestAccess;

    // One reservation: the blocks granted and their keys, which no other
    // write reserves meanwhile. Dropping it frees the keys, and the blocks
    // unless a commit registered them.
    class Write {
    public:
        explicit Write(std::unordered_set<CacheKey, CacheKeyHash>& writing) : writing_{&writing} {}
        Write(const Write&) = delete;
        Write& operator=(const Write&) = delete;
        Write(Write&&) noexcept = default;
        Write& operator=(Write&&) = delete;
        ~Write() {
            for (const CacheKey& key : keys_) {
                writing_->erase(key);
            }
        }

        void Add(const CacheKey& key, CacheBlockRef block) {
            writing_->insert(key);
            keys_.push_back(key);
            blocks_.push_back(std::move(block));
        }
        std::span<const CacheKey> Keys() const noexcept { return keys_; }
        std::span<CacheBlockRef> Blocks() noexcept { return blocks_; }

    private:
        std::unordered_set<CacheKey, CacheKeyHash>* writing_;
        std::vector<CacheKey> keys_;
        std::vector<CacheBlockRef> blocks_;
    };
    struct Client {
        // The blocks each lookup pinned.
        std::map<Ticket, std::vector<CacheBlockRef>> lookups;
        std::map<Ticket, Write> writes;
    };

    void validateKeys(std::span<const CacheKey> keys) const;
    void keepHeadroom();

    // The coordinator requires a Device pool; the pool's owner never uses it.
    BlockPool unused_device_block_pool_;
    BlockPool host_block_pool_;
    CacheCoordinator cache_;
    std::int32_t prefix_granularity_;
    std::int32_t headroom_lcm_blocks_;
    // The group with the fewest blocks per LCM block: whole LCM blocks are
    // requested through it.
    std::uint32_t headroom_group_{0};
    std::int32_t headroom_packing_{1};
    // The keys of every client's reservations; only Write adds and removes
    // them, so it outlives clients_.
    std::unordered_set<CacheKey, CacheKeyHash> writing_;
    std::unordered_map<ClientId, Client> clients_;
    Ticket next_ticket_{0};
};

}  // namespace tokenspeed
