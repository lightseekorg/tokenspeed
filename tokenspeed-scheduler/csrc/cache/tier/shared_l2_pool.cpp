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

#include <utility>

#include "utils.h"

namespace tokenspeed {
namespace {

std::vector<std::int32_t> SlotsPerGroup(std::span<const CacheGroupSpec> specs) {
    std::vector<std::int32_t> slots;
    slots.reserve(specs.size());
    for (const CacheGroupSpec& spec : specs) {
        _assert(spec.shard_count == 1, "a shared L2 pool holds replicated groups only");
        slots.push_back(spec.cache_blocks_per_lcm_block);
    }
    return slots;
}

}  // namespace

SharedL2Pool::SharedL2Pool(std::vector<CacheGroupSpec> specs, std::int32_t prefix_granularity,
                           std::int32_t num_lcm_blocks, std::int32_t headroom_lcm_blocks)
    : unused_device_(1, SlotsPerGroup(specs)),
      pool_(num_lcm_blocks, SlotsPerGroup(specs)),
      cache_{MakeCoordinator(specs, prefix_granularity, unused_device_, /*enable_l3_storage=*/false, &pool_,
                             /*snapshot_pool=*/nullptr, /*stream_device_cache_to_host=*/false)},
      prefix_granularity_{prefix_granularity},
      headroom_lcm_blocks_{headroom_lcm_blocks} {
    _assert(headroom_lcm_blocks >= 0 && headroom_lcm_blocks < num_lcm_blocks,
            "the headroom must leave room for cached blocks");
    for (std::size_t i = 0; i < specs.size(); ++i) {
        if (i == 0 || specs[i].cache_blocks_per_lcm_block < headroom_packing_) {
            headroom_group_ = static_cast<std::uint32_t>(i);
            headroom_packing_ = specs[i].cache_blocks_per_lcm_block;
        }
    }
    cache_.ReserveHostIndex();
}

SharedL2Pool::Hit SharedL2Pool::Probe(ClientId client, const CacheCoordinator::PromptPrefix& prompt) {
    CacheCoordinator::PrefixProbe probe = cache_.ProbeHostPrefix(prompt);
    Hit hit{.kv_tokens = probe.host.kv_tokens};
    if (probe.host.num_common_tokens == 0) {
        return hit;
    }
    hit.num_tokens = probe.host.num_common_tokens;
    if (probe.host.tail) {
        hit.tail = TailPublication{.tail_tokens = probe.host.tail->tail_tokens, .hash = probe.host.tail->hash};
    }
    CoordinatorMatch match = cache_.AcquireHostPrefix(std::move(probe));
    std::vector<CacheBlockRef> pins;
    for (std::int32_t i = 0; i < cache_.NumGroups(); ++i) {
        for (CacheBlockRef& block : match.per_group[static_cast<std::size_t>(i)].blocks) {
            if (!block) {
                continue;
            }
            const std::optional<CacheKey> key = cache_.GroupPrefixIndex(i).KeyOf(pool_, block);
            _assert(key.has_value(), "a matched shared L2 block has no entry");
            hit.slots.push_back(Slot{.key = *key, .page = cache_.Allocator(i).ResolveCacheBlockId(block->Location())});
            pins.push_back(std::move(block));
        }
    }
    hit.ticket = ++next_ticket_;
    clients_[client].lookups.emplace(hit.ticket, std::move(pins));
    return hit;
}

void SharedL2Pool::Release(ClientId client, Ticket ticket) {
    auto it = clients_.find(client);
    _assert(it != clients_.end() && it->second.lookups.erase(ticket) == 1, "unknown shared L2 lookup ticket");
}

SharedL2Pool::Reservation SharedL2Pool::Reserve(ClientId client, std::span<const CacheKey> keys) {
    Reservation result;
    result.pages.assign(keys.size(), kHeld);
    std::vector<std::size_t> wanted;
    std::vector<std::uint32_t> groups;
    std::unordered_set<CacheKey, CacheKeyHash> requested;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (cache_.ContainsHostCachedBlock(keys[i]) || writing_.contains(keys[i]) ||
            !requested.insert(keys[i]).second) {
            continue;
        }
        wanted.push_back(i);
        groups.push_back(keys[i].group_id);
    }
    if (wanted.empty()) {
        return result;
    }
    const std::vector<std::int32_t> buckets(groups.size(), 0);
    CacheCoordinator::HostAllocationBatch batch = cache_.AcquireHostBlocks(groups, buckets);
    Writes writes;
    for (std::size_t j = 0; j < wanted.size(); ++j) {
        CacheBlockRef& block = batch.blocks[j];
        if (!block) {
            result.pages[wanted[j]] = kNoSlot;
            continue;
        }
        const CacheKey& key = keys[wanted[j]];
        result.pages[wanted[j]] =
            cache_.Allocator(static_cast<std::int32_t>(key.group_id)).ResolveCacheBlockId(block->Location());
        writing_.insert(key);
        writes.keys.push_back(key);
        writes.blocks.push_back(std::move(block));
    }
    if (writes.keys.empty()) {
        return result;
    }
    reserved_blocks_ += writes.keys.size();
    result.ticket = ++next_ticket_;
    clients_[client].writes.emplace(result.ticket, std::move(writes));
    keepHeadroom();
    return result;
}

void SharedL2Pool::keepHeadroom() {
    const std::int32_t empty = pool_.NumEmptyLcmBlocks();
    if (empty * 2 >= headroom_lcm_blocks_) {
        return;
    }
    // Allocating the whole headroom evicts as a write would (empty blocks go
    // first); dropping the allocation leaves them empty.
    const auto blocks = static_cast<std::size_t>(headroom_lcm_blocks_) * headroom_packing_;
    const std::vector<std::uint32_t> groups(blocks, headroom_group_);
    const std::vector<std::int32_t> buckets(blocks, 0);
    cache_.AcquireHostBlocks(groups, buckets);
}

void SharedL2Pool::Commit(ClientId client, Ticket ticket, std::span<const Publication> publications) {
    for (const Publication& publication : publications) {
        _assert(publication.tokens.size() ==
                    publication.prefix_hashes.size() * static_cast<std::size_t>(prefix_granularity_) +
                        static_cast<std::size_t>(publication.tail ? publication.tail->tail_tokens : 0),
                "a shared L2 publication's tokens must cover its prefix and tail exactly");
        if (publicationStored(publication)) {
            applyPublication(publication);
            continue;
        }
        const ContentHash hash = publicationHash(publication);
        if (pending_publications_.insert_or_assign(hash, publication).second) pending_order_.push_back(hash);
        while (pending_order_.size() > kMaxPendingPublications) {
            pending_publications_.erase(pending_order_.front());
            pending_order_.pop_front();
        }
    }
    if (ticket == 0) {
        return;
    }
    auto it = clients_.find(client);
    _assert(it != clients_.end() && it->second.writes.contains(ticket), "unknown shared L2 write ticket");
    Writes writes = std::move(it->second.writes.at(ticket));
    it->second.writes.erase(ticket);
    for (std::size_t i = 0; i < writes.keys.size(); ++i) {
        cache_.CacheHostBlock(writes.blocks[i], writes.keys[i]);
    }
    for (const CacheKey& key : writes.keys) {
        const auto pending = pending_publications_.find(key.content_hash);
        if (pending != pending_publications_.end() && publicationStored(pending->second)) {
            applyPublication(pending->second);
            pending_publications_.erase(pending);
        }
    }
    endWrites(writes);
}

ContentHash SharedL2Pool::publicationHash(const Publication& publication) {
    return publication.tail ? publication.tail->hash : publication.prefix_hashes.back();
}

bool SharedL2Pool::publicationStored(const Publication& publication) const {
    const ContentHash hash = publicationHash(publication);
    for (std::int32_t i = 0; i < cache_.NumGroups(); ++i) {
        const auto group = static_cast<std::uint32_t>(i);
        if (cache_.GroupIsState(static_cast<std::size_t>(i))) {
            if (!cache_.ContainsHostCachedBlock({.group_id = group, .content_hash = hash})) {
                return false;
            }
            continue;
        }
        if (!publication.tail || cache_.GroupIsReplayable(i)) continue;
        for (std::int32_t offset = 0; offset < cache_.TailBlocksFor(i, publication.tail->tail_tokens); ++offset) {
            if (!cache_.ContainsHostCachedBlock({.group_id = group, .content_hash = hash, .page_offset = offset})) {
                return false;
            }
        }
    }
    return true;
}

void SharedL2Pool::applyPublication(const Publication& publication) {
    const ContentHash prior = publication.prefix_hashes.empty() ? ContentHash{} : publication.prefix_hashes.back();
    if (publication.tail) {
        cache_.CacheHostTail(prior, *publication.tail);
    }
    cache_.RecordCheckpoint(CompletedPages{.tokens = publication.tokens,
                                           .prefix_hashes = publication.prefix_hashes,
                                           .tail = publication.tail,
                                           .publish_state = true,
                                           .boundary_kind = CacheBoundaryKind::kEndpoint});
}

void SharedL2Pool::Touch(std::span<const CacheKey> keys) {
    cache_.TouchHostBlocks(keys);
}

void SharedL2Pool::Disconnect(ClientId client) {
    auto it = clients_.find(client);
    if (it == clients_.end()) {
        return;
    }
    for (auto& [ticket, writes] : it->second.writes) {
        endWrites(writes);
    }
    clients_.erase(it);
}

bool SharedL2Pool::Clear() {
    for (const auto& [client, state] : clients_) {
        if (!state.lookups.empty() || !state.writes.empty()) return false;
    }
    if (!cache_.ClearCache()) return false;
    pending_publications_.clear();
    pending_order_.clear();
    return true;
}

void SharedL2Pool::endWrites(Writes& writes) {
    for (const CacheKey& key : writes.keys) {
        writing_.erase(key);
    }
    reserved_blocks_ -= writes.keys.size();
}

SharedL2Pool::Stats SharedL2Pool::GetStats() const {
    return Stats{
        .total_lcm_blocks = pool_.NumLcmBlocks(),
        .free_lcm_blocks = pool_.NumEmptyLcmBlocks(),
        .cached_blocks = cache_.NumHostCachedBlocks(),
        .pinned_blocks = cache_.NumPinnedHostCachedBlocks(),
        .reserved_blocks = reserved_blocks_,
        .demoted_victims = cache_.CheckpointRetentionStats().host_demoted_victims,
    };
}

}  // namespace tokenspeed
