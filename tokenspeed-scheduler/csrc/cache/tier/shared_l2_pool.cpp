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

#include "cache/tier/shared_l2_pool.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "utils.h"

namespace tokenspeed {
namespace {

std::vector<std::int32_t> DevicePacking(std::span<const CacheGroupSpec> specs) {
    std::vector<std::int32_t> packing;
    packing.reserve(specs.size());
    for (const CacheGroupSpec& spec : specs) {
        if (spec.shard_count != 1) {
            throw std::invalid_argument("a shared L2 pool holds replicated groups only");
        }
        packing.push_back(spec.cache_blocks_per_lcm_block);
    }
    return packing;
}

std::vector<std::int32_t> HostPacking(std::span<const CacheGroupSpec> specs, std::vector<std::int32_t> packing) {
    if (packing.size() != specs.size() || std::ranges::any_of(packing, [](std::int32_t slots) { return slots <= 0; })) {
        throw std::invalid_argument("a shared L2 pool needs a positive Host packing for every cache group");
    }
    return packing;
}

}  // namespace

SharedL2Pool::SharedL2Pool(std::span<const CacheGroupSpec> specs, std::int32_t prefix_granularity,
                           std::vector<std::int32_t> host_group_packing, std::int32_t num_lcm_blocks,
                           std::int32_t headroom_lcm_blocks, std::int64_t checkpoint_retention_requests,
                           std::int32_t checkpoint_retention_leaf_distance)
    : unused_device_(1, DevicePacking(specs)),
      pool_(num_lcm_blocks, HostPacking(specs, std::move(host_group_packing))),
      cache_{MakeCoordinator(specs, prefix_granularity, unused_device_, /*enable_l3_storage=*/false, &pool_,
                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false,
                             /*write_back_evictions=*/false, checkpoint_retention_requests,
                             checkpoint_retention_leaf_distance)},
      prefix_granularity_{prefix_granularity},
      headroom_lcm_blocks_{headroom_lcm_blocks} {
    if (headroom_lcm_blocks < 0 || headroom_lcm_blocks >= num_lcm_blocks) {
        throw std::invalid_argument("a shared L2 pool's headroom must leave room for cached blocks");
    }
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const std::int32_t packing = pool_.Packing(static_cast<std::uint32_t>(i));
        if (i == 0 || packing < headroom_packing_) {
            headroom_group_ = static_cast<std::uint32_t>(i);
            headroom_packing_ = packing;
        }
    }
}

SharedL2Pool::Hit SharedL2Pool::Probe(ClientId client, const PromptPrefix& prompt) {
    if (prompt.max_tokens < 0 || prompt.tokens.size() < static_cast<std::size_t>(prompt.max_tokens)) {
        throw std::invalid_argument("a shared L2 lookup's tokens must reach its bound");
    }
    Hit hit;
    // The Device tier is empty, so the Host suffix of the probe starts at the
    // prompt's first token: the Host tier alone.
    PrefixProbe probe = cache_.Prober().ProbePromptPrefix(prompt);
    hit.kv_tokens = probe.host.kv_tokens;
    hit.num_tokens = probe.host.num_common_tokens;
    if (probe.host.tail) {
        hit.tail = TailPublication{.tail_tokens = probe.host.tail->tail_tokens, .hash = probe.host.tail->hash};
    }
    // Every lookup is one request of the retention window, as every new
    // request is on a rank: one that hits nothing pins nothing.
    CoordinatorMatch match = cache_.AcquireHostPrefix(std::move(probe));
    if (hit.num_tokens == 0) {
        return hit;
    }
    std::vector<CacheBlockRef> pins;
    for (std::int32_t i = 0; i < cache_.NumGroups(); ++i) {
        const GroupAllocator& allocator = cache_.Allocator(i);
        for (CacheBlockRef& block : match.per_group[static_cast<std::size_t>(i)].blocks) {
            if (!block) {
                continue;
            }
            const std::optional<CacheKey> key = cache_.GroupPrefixIndex(i).KeyOf(pool_, block);
            FatalCheck(key.has_value(), "a matched shared L2 block has no entry");
            hit.slots.push_back(Slot{.key = *key,
                                     .bucket = allocator.BucketOf(block->Location()),
                                     .page = allocator.ResolveCacheBlockId(*block)});
            pins.push_back(std::move(block));
        }
    }
    hit.ticket = ++next_ticket_;
    clients_[client].lookups.emplace(hit.ticket, std::move(pins));
    return hit;
}

void SharedL2Pool::Release(ClientId client, Ticket ticket) {
    if (const auto it = clients_.find(client); it != clients_.end()) {
        it->second.lookups.erase(ticket);
    }
}

void SharedL2Pool::validateKeys(std::span<const CacheKey> keys) const {
    for (const CacheKey& key : keys) {
        if (key.namespace_id != kDefaultCacheNamespaceId ||
            key.group_id >= static_cast<std::uint32_t>(cache_.NumGroups())) {
            throw std::invalid_argument("a shared L2 key names a cache group of the pool, in the default namespace");
        }
    }
}

SharedL2Pool::Reservation SharedL2Pool::Reserve(ClientId client, std::span<const CacheKey> keys,
                                                std::span<const std::int32_t> buckets) {
    if (keys.size() != buckets.size()) {
        throw std::invalid_argument("a shared L2 reservation names one bucket per key");
    }
    validateKeys(keys);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (buckets[i] < 0 ||
            buckets[i] >= cache_.Allocator(static_cast<std::int32_t>(keys[i].group_id)).ShardCount()) {
            throw std::invalid_argument("a shared L2 bucket lies outside its group's shard count");
        }
    }
    Reservation result;
    result.pages.assign(keys.size(), kHeld);
    // The rank published these keys on its Device. The ones the pool holds
    // are refreshed, as that publication refreshes a rank's own Host copies,
    // before pages are acquired for the others, which evicts the entries
    // least recently used.
    cache_.Host().TouchHostBlocks(keys);
    std::vector<std::size_t> wanted;
    std::vector<std::uint32_t> wanted_groups;
    std::vector<std::int32_t> wanted_buckets;
    std::unordered_set<CacheKey, CacheKeyHash> requested;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (cache_.Host().ContainsHostCachedBlock(keys[i]) || writing_.contains(keys[i]) ||
            !requested.insert(keys[i]).second) {
            continue;
        }
        wanted.push_back(i);
        wanted_groups.push_back(keys[i].group_id);
        wanted_buckets.push_back(buckets[i]);
    }
    if (wanted.empty()) {
        return result;
    }
    HostCache::HostAllocationBatch batch = cache_.Host().AcquireHostBlocks(wanted_groups, wanted_buckets);
    Write write{writing_};
    for (std::size_t j = 0; j < wanted.size(); ++j) {
        CacheBlockRef& block = batch.blocks[j];
        if (!block) {
            result.pages[wanted[j]] = kNoSlot;
            continue;
        }
        const CacheKey& key = keys[wanted[j]];
        result.pages[wanted[j]] = cache_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(*block);
        write.Add(key, std::move(block));
    }
    if (write.Keys().empty()) {
        return result;
    }
    result.ticket = ++next_ticket_;
    clients_[client].writes.emplace(result.ticket, std::move(write));
    keepHeadroom();
    return result;
}

void SharedL2Pool::keepHeadroom() {
    const std::int32_t empty = pool_.NumEmptyLcmBlocks();
    if (std::int64_t{empty} * 2 >= headroom_lcm_blocks_) {
        return;
    }
    // Allocating the whole headroom evicts as a write would (empty blocks go
    // first); dropping the allocation leaves them empty. The headroom is
    // smaller than the pool, whose blocks of a group BlockPool keeps within
    // int32: so is this request.
    const auto blocks = static_cast<std::size_t>(headroom_lcm_blocks_) * static_cast<std::size_t>(headroom_packing_);
    const std::vector<std::uint32_t> groups(blocks, headroom_group_);
    const std::vector<std::int32_t> buckets(blocks, 0);
    cache_.Host().AcquireHostBlocks(groups, buckets);
}

bool SharedL2Pool::Commit(ClientId client, Ticket ticket, std::span<const CheckpointTree::Lineage> checkpoints) {
    std::vector<CompletedPages> completed;
    completed.reserve(checkpoints.size());
    for (const CheckpointTree::Lineage& checkpoint : checkpoints) {
        const std::int32_t tail_tokens = checkpoint.tail ? checkpoint.tail->tail_tokens : 0;
        // A lineage comes off the wire: its token count is held to the int32
        // range before it is narrowed, here and where the checkpoint is
        // recorded (CheckpointTree::Observe), and its pages are compared by
        // division, which no number of hashes overflows.
        if (!std::in_range<std::int32_t>(checkpoint.tokens.size())) {
            throw std::invalid_argument("a shared L2 publication's tokens must fit the int32 token range");
        }
        const auto page = static_cast<std::size_t>(prefix_granularity_);
        if (checkpoint.tokens.empty() ||
            (checkpoint.tail &&
             (tail_tokens <= 0 || tail_tokens >= prefix_granularity_ || checkpoint.tail->hash.empty())) ||
            checkpoint.tokens.size() / page != checkpoint.prefix_hashes.size() ||
            checkpoint.tokens.size() % page != static_cast<std::size_t>(tail_tokens)) {
            throw std::invalid_argument("a shared L2 publication's tokens must cover its pages and tail exactly");
        }
        completed.push_back(
            CompletedPages{.tokens = checkpoint.tokens,
                           .prefix_hashes = checkpoint.prefix_hashes,
                           // Every page may be new to the pool: the link starts at the deepest
                           // page already linked.
                           .first_new_prefix_page = static_cast<std::int32_t>(checkpoint.prefix_hashes.size()),
                           .tail = checkpoint.tail,
                           .publish_state = true,
                           .boundary_kind = CacheBoundaryKind::kEndpoint});
    }
    const auto known = clients_.find(client);
    if (known == clients_.end()) {
        return false;
    }
    auto reserved = known->second.writes.extract(ticket);
    if (reserved.empty()) {
        return false;
    }
    Write& write = reserved.mapped();
    std::vector<BoundaryTree::Hold> linked;
    linked.reserve(completed.size());
    for (const CompletedPages& checkpoint : completed) {
        linked.push_back(cache_.Store().Boundaries().LinkPublication(checkpoint));
    }
    for (std::size_t i = 0; i < write.Keys().size(); ++i) {
        cache_.Host().CacheHostBlock(write.Blocks()[i], write.Keys()[i], /*inherited_metadata=*/std::nullopt);
    }
    // A boundary nothing registered under leaves again.
    linked.clear();
    for (std::size_t i = 0; i < completed.size(); ++i) {
        const CompletedPages& checkpoint = completed[i];
        // Two ranks issue the same access epochs: the client tells them apart.
        cache_.RecordCheckpoint(checkpoint, checkpoints[i].access_epoch ^ (client * 0x9E3779B97F4A7C15ULL));
        if (checkpoint.tail) {
            cache_.Host().CacheHostTail(
                checkpoint.prefix_hashes.empty() ? ContentHash{} : ContentHash{checkpoint.prefix_hashes.back()},
                *checkpoint.tail);
        }
    }
    return true;
}

void SharedL2Pool::Touch(std::span<const CacheKey> keys) {
    validateKeys(keys);
    cache_.Host().TouchHostBlocks(keys);
}

void SharedL2Pool::Disconnect(ClientId client) {
    clients_.erase(client);
}

bool SharedL2Pool::Clear() {
    for (const auto& [client, state] : clients_) {
        if (!state.lookups.empty() || !state.writes.empty()) {
            return false;
        }
    }
    return cache_.ClearCache();
}

SharedL2Pool::Stats SharedL2Pool::GetStats() const {
    Stats stats{
        .total_lcm_blocks = pool_.NumLcmBlocks(),
        .free_lcm_blocks = pool_.NumEmptyLcmBlocks(),
        .cached_blocks = cache_.Host().NumHostCachedBlocks(),
        .pinned_blocks = cache_.Host().NumPinnedHostCachedBlocks(),
        .reserved_blocks = writing_.size(),
        .evicted_blocks = cache_.Evictor().HostEvictions().lru.blocks,
        .checkpoints = cache_.Checkpoints().Size(),
    };
    for (const auto& [client, state] : clients_) {
        stats.lookups += state.lookups.size();
    }
    return stats;
}

}  // namespace tokenspeed
