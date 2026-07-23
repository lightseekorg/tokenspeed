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

#include <algorithm>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_block_ref.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

struct ParentEvictionCandidate {
    std::int32_t lcm_block_id{0};
    std::uint64_t last_access{0};
    std::int32_t occupied_count{0};

    auto operator<=>(const ParentEvictionCandidate& other) const noexcept {
        return std::tie(last_access, occupied_count, lcm_block_id) <=>
               std::tie(other.last_access, other.occupied_count, other.lcm_block_id);
    }
    bool operator==(const ParentEvictionCandidate&) const noexcept = default;
};

// Per-attention-type token policy plus cache metadata for one group. Physical
// placement remains entirely in BlockPool.
class KvCacheManager {
private:
    struct CacheBlockLocationHash {
        std::size_t operator()(CacheBlockLocation location) const noexcept {
            const std::size_t parent = std::hash<std::int32_t>{}(location.lcm_block_id);
            const std::size_t slot = std::hash<std::int32_t>{}(location.slot_index);
            return parent ^ (slot + 0x9e3779b9U + (parent << 6U) + (parent >> 2U));
        }
    };

    struct CacheEntry {
        CacheKey key;
        CacheBlockRef block;
        std::uint64_t last_access{0};
    };

    using CacheEntries = std::list<CacheEntry>;
    using CacheEntryIterator = CacheEntries::iterator;
    using ConstCacheEntryIterator = CacheEntries::const_iterator;

    struct PoolCache {
        CacheEntries entries_by_recency;
        std::unordered_map<std::string, CacheEntryIterator> prefix_index;
        std::unordered_map<CacheBlockLocation, CacheEntryIterator, CacheBlockLocationHash> reverse_index;
    };

public:
    explicit KvCacheManager(std::int32_t cache_block_tokens, std::int32_t cache_blocks_per_lcm_block = 1,
                            GroupId group_id = 0)
        : cache_block_tokens_{cache_block_tokens},
          cache_blocks_per_lcm_block_{cache_blocks_per_lcm_block},
          group_id_{group_id} {
        _assert(cache_block_tokens > 0, "cache_block_tokens must be > 0");
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
    }
    virtual ~KvCacheManager() = default;

    KvCacheManager(const KvCacheManager&) = delete;
    KvCacheManager& operator=(const KvCacheManager&) = delete;

    std::int32_t CacheBlockTokens() const noexcept { return cache_block_tokens_; }
    std::int32_t CacheBlocksPerLcmBlock() const noexcept { return cache_blocks_per_lcm_block_; }
    GroupId GroupIdValue() const noexcept { return group_id_; }

    std::int32_t ResolveKernelPageId(CacheBlockLocation location) const {
        _assert(location.lcm_block_id > 0, "LCM block id must be > 0");
        _assert(0 <= location.slot_index && location.slot_index < cache_blocks_per_lcm_block_,
                "cache block slot is out of range");
        const std::int64_t page_id =
            1 + (static_cast<std::int64_t>(location.lcm_block_id) - 1) * cache_blocks_per_lcm_block_ +
            location.slot_index;
        _assert(page_id <= std::numeric_limits<std::int32_t>::max(), "kernel page id exceeds int32 range");
        return static_cast<std::int32_t>(page_id);
    }
    std::vector<std::int32_t> BlockTablePageIds(const BlockTable& table) const {
        std::vector<std::int32_t> ids;
        ids.reserve(static_cast<std::size_t>(table.NumBlocks()));
        for (const CacheBlockRef& block : table.Blocks()) {
            ids.push_back(block ? ResolveKernelPageId(block->Location()) : 0);
        }
        return ids;
    }

    virtual bool MatchIsPrefixClosed() const = 0;
    virtual PrefixProbe Probe(const BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                              std::int32_t max_blocks) const = 0;

    PrefixMatch AcquireMatchedBlocks(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                                     const PrefixProbe& probe, std::uint64_t& next_recency) {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        PrefixMatch match;
        match.blocks.resize(probe.hits.size());
        PoolCache* cache = findPoolCache(pool);
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            _assert(cache != nullptr, "cached pool disappeared between match probe and acquisition");
            CacheEntryIterator entry = findEntry(*cache, keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(entry != cache->entries_by_recency.end(),
                    "cached block disappeared between match probe and acquisition");
            touch(*cache, entry, next_recency);
            match.blocks[i] = entry->block;
            ++match.num_hit_blocks;
        }
        return match;
    }

    PrefixMatch Match(BlockPool& pool, std::span<const std::string> keys, std::int32_t begin_blocks,
                      std::int32_t max_blocks, std::uint64_t& next_recency) {
        return AcquireMatchedBlocks(pool, keys, begin_blocks, Probe(pool, keys, begin_blocks, max_blocks),
                                    next_recency);
    }

    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        table.blocks_ = std::move(hit.blocks);
    }

    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens) {
        if (num_tokens <= 0) {
            return true;
        }
        if (num_tokens <= table.tail_avail_) {
            table.tail_avail_ -= num_tokens;
            return true;
        }
        const std::int32_t over = num_tokens - table.tail_avail_;
        const std::int32_t num_pages = (over + cache_block_tokens_ - 1) / cache_block_tokens_;
        std::vector<CacheBlockRef> new_blocks = pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, num_pages);
        if (static_cast<std::int32_t>(new_blocks.size()) < num_pages) {
            return false;
        }
        table.blocks_.reserve(table.blocks_.size() + new_blocks.size());
        for (CacheBlockRef& block : new_blocks) {
            table.blocks_.push_back(std::move(block));
        }
        const std::int32_t used_in_tail = over % cache_block_tokens_;
        table.tail_avail_ = (used_in_tail == 0) ? 0 : cache_block_tokens_ - used_in_tail;
        return true;
    }

    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<CacheBlockRef>&& host_blocks,
                             std::vector<BlockTransfer>& load_pairs) {
        _assert(table.tail_avail_ == 0, "host extension must append on a full-page boundary");
        for (CacheBlockRef& host_block : host_blocks) {
            if (!host_block) {
                table.blocks_.emplace_back();
                continue;
            }
            const bool acquired = Acquire(pool, table, cache_block_tokens_);
            _assert(acquired, "pre-checked Acquire must succeed");
            load_pairs.push_back(BlockTransfer{
                .group_id = group_id_,
                .source = std::move(host_block),
                .destination = table.blocks_.back(),
            });
        }
    }

    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.tail_avail_) {
            return 0;
        }
        const std::int32_t over = num_tokens - table.tail_avail_;
        return (over + cache_block_tokens_ - 1) / cache_block_tokens_;
    }

    virtual bool RegistersAlignedFinalPageOnly() const { return false; }

    void CacheBlock(BlockPool& pool, CacheBlockRef& block, const std::string& block_hash, std::uint64_t& next_recency,
                    std::vector<std::pair<std::string, CacheBlockRef>>* newly_cached = nullptr) {
        _assert(block && block.IsOwnedBy(pool), "cache block must belong to the target pool");
        _assert(!block_hash.empty(), "block hash must not be empty");
        PoolCache& cache = poolCache(pool);
        CacheEntryIterator existing = findEntry(cache, block->Location());
        if (existing != cache.entries_by_recency.end()) {
            _assert(existing->key.content_hash == block_hash, "one cache block location cannot change content hash");
            touch(cache, existing, next_recency);
            return;
        }
        CacheEntryIterator canonical = findEntry(cache, block_hash);
        if (canonical != cache.entries_by_recency.end()) {
            touch(cache, canonical, next_recency);
            block = canonical->block;
            return;
        }

        cache.entries_by_recency.push_back(CacheEntry{
            .key = CacheKey{.group_id = group_id_, .content_hash = block_hash},
            .block = block,
            .last_access = ++next_recency,
        });
        CacheEntryIterator entry = std::prev(cache.entries_by_recency.end());
        cache.prefix_index.emplace(entry->key.content_hash, entry);
        cache.reverse_index.emplace(entry->block->Location(), entry);
        if (newly_cached != nullptr) {
            newly_cached->emplace_back(block_hash, block);
        }
    }

    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const std::string> block_hashes,
                         std::uint64_t& next_recency, std::int32_t first_slot = 0,
                         std::vector<std::pair<std::string, CacheBlockRef>>* newly_cached = nullptr) {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(
            static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(block_hashes.size()) <= table.NumBlocks(),
            "hash range exceeds table size");
        for (std::size_t j = 0; j < block_hashes.size(); ++j) {
            CacheBlockRef& block = table.blocks_[static_cast<std::size_t>(first_slot) + j];
            if (!block) {
                continue;
            }
            CacheBlock(pool, block, block_hashes[j], next_recency, newly_cached);
        }
    }

    bool ContainsCachedBlock(const BlockPool& pool, const std::string& key) const {
        const PoolCache* cache = findPoolCache(pool);
        return cache != nullptr && findEntry(*cache, key) != cache->entries_by_recency.end();
    }
    bool ContainsCachedBlock(const BlockPool& pool, CacheBlockLocation location) const {
        const PoolCache* cache = findPoolCache(pool);
        return cache != nullptr && findEntry(*cache, location) != cache->entries_by_recency.end();
    }
    bool IsCachedBlockFree(const BlockPool& pool, const std::string& key) const {
        const PoolCache* cache = findPoolCache(pool);
        if (cache == nullptr) {
            return false;
        }
        ConstCacheEntryIterator entry = findEntry(*cache, key);
        return entry != cache->entries_by_recency.end() && entry->block.use_count() == 1;
    }
    std::int32_t NumCachedBlocks(const BlockPool& pool) const {
        const PoolCache* cache = findPoolCache(pool);
        return cache == nullptr ? 0 : static_cast<std::int32_t>(cache->entries_by_recency.size());
    }
    std::int32_t NumPinnedCachedBlocks(const BlockPool& pool) const {
        const PoolCache* cache = findPoolCache(pool);
        if (cache == nullptr) {
            return 0;
        }
        return static_cast<std::int32_t>(std::ranges::count_if(
            cache->entries_by_recency, [](const CacheEntry& entry) { return entry.block.use_count() > 1; }));
    }

    std::vector<CacheBlockLocation> EvictableBlockLocations(const BlockPool& pool) const {
        const PoolCache* cache = findPoolCache(pool);
        if (cache == nullptr) {
            return {};
        }
        std::vector<CacheBlockLocation> locations;
        for (const CacheEntry& entry : cache->entries_by_recency) {
            if (entry.block.use_count() == 1) {
                locations.push_back(entry.block->Location());
            }
        }
        return locations;
    }

    bool EvictCachedBlock(const BlockPool& pool, CacheBlockLocation location) {
        PoolCache* cache = findPoolCache(pool);
        if (cache == nullptr) {
            return false;
        }
        CacheEntryIterator entry = findEntry(*cache, location);
        if (entry == cache->entries_by_recency.end() || entry->block.use_count() != 1) {
            return false;
        }
        eraseEntry(*cache, entry);
        return true;
    }

    bool ParentIsFullyEvictable(const BlockPool& pool, std::int32_t lcm_block_id) const {
        const std::vector<CacheBlockLocation> locations = pool.OccupiedLocations(lcm_block_id);
        if (locations.empty()) {
            return false;
        }
        const PoolCache* cache = findPoolCache(pool);
        if (cache == nullptr) {
            return false;
        }
        return std::ranges::all_of(locations, [&](CacheBlockLocation location) {
            ConstCacheEntryIterator entry = findEntry(*cache, location);
            return entry != cache->entries_by_recency.end() && entry->block.use_count() == 1;
        });
    }

    std::vector<ParentEvictionCandidate> CollectEvictableParents(const BlockPool& pool) const {
        std::vector<ParentEvictionCandidate> candidates;
        for (std::int32_t lcm_block_id = 1; lcm_block_id <= pool.NumLcmBlocks(); ++lcm_block_id) {
            if (pool.BoundGroup(lcm_block_id) != group_id_ || !ParentIsFullyEvictable(pool, lcm_block_id)) {
                continue;
            }
            const std::vector<CacheBlockLocation> locations = pool.OccupiedLocations(lcm_block_id);
            const PoolCache* cache = findPoolCache(pool);
            _assert(cache != nullptr, "fully evictable parent must have a pool cache");
            std::uint64_t parent_recency = 0;
            for (CacheBlockLocation location : locations) {
                ConstCacheEntryIterator entry = findEntry(*cache, location);
                _assert(entry != cache->entries_by_recency.end(),
                        "fully evictable parent must have one entry per occupied child");
                parent_recency = std::max(parent_recency, entry->last_access);
            }
            candidates.push_back(ParentEvictionCandidate{
                .lcm_block_id = lcm_block_id,
                .last_access = parent_recency,
                .occupied_count = static_cast<std::int32_t>(locations.size()),
            });
        }
        std::ranges::sort(candidates);
        return candidates;
    }

    bool EvictParent(const BlockPool& pool, std::int32_t lcm_block_id) {
        if (pool.BoundGroup(lcm_block_id) != group_id_ || !ParentIsFullyEvictable(pool, lcm_block_id)) {
            return false;
        }
        PoolCache* cache = findPoolCache(pool);
        _assert(cache != nullptr, "fully evictable parent must have a pool cache");
        std::vector<CacheEntryIterator> entries;
        for (CacheBlockLocation location : pool.OccupiedLocations(lcm_block_id)) {
            CacheEntryIterator entry = findEntry(*cache, location);
            _assert(entry != cache->entries_by_recency.end(),
                    "fully evictable parent must have one entry per occupied child");
            entries.push_back(entry);
        }
        for (CacheEntryIterator entry : entries) {
            eraseEntry(*cache, entry);
        }
        _assert(pool.BoundGroup(lcm_block_id) == std::nullopt, "evicting every child must unbind the parent");
        return true;
    }

    virtual void ReclaimExpired(BlockPool& /*pool*/, BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/) {}
    virtual std::int32_t BlocksReclaimableAt(const BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/,
                                             bool /*count_uncached*/) const {
        return 0;
    }

    void Free(BlockTable& table) {
        for (auto it = table.blocks_.rbegin(); it != table.blocks_.rend(); ++it) {
            it->reset();
        }
        table.blocks_.clear();
        table.tail_avail_ = 0;
    }

protected:
    bool ContainsCachedBlock(const CacheBlockRef& block) const {
        if (!block) {
            return false;
        }
        return std::ranges::any_of(pool_caches_, [&](const auto& item) {
            auto entry = item.second.reverse_index.find(block->Location());
            return entry != item.second.reverse_index.end() && entry->second->block == block;
        });
    }

    std::int32_t cache_block_tokens_;
    std::int32_t cache_blocks_per_lcm_block_;
    GroupId group_id_;

private:
    PoolCache& poolCache(const BlockPool& pool) { return pool_caches_.try_emplace(&pool).first->second; }
    PoolCache* findPoolCache(const BlockPool& pool) {
        auto it = pool_caches_.find(&pool);
        return it == pool_caches_.end() ? nullptr : &it->second;
    }
    const PoolCache* findPoolCache(const BlockPool& pool) const {
        auto it = pool_caches_.find(&pool);
        return it == pool_caches_.end() ? nullptr : &it->second;
    }
    CacheEntryIterator findEntry(PoolCache& cache, const std::string& key) {
        auto entry = cache.prefix_index.find(key);
        return entry == cache.prefix_index.end() ? cache.entries_by_recency.end() : entry->second;
    }
    CacheEntryIterator findEntry(PoolCache& cache, CacheBlockLocation location) {
        auto entry = cache.reverse_index.find(location);
        return entry == cache.reverse_index.end() ? cache.entries_by_recency.end() : entry->second;
    }
    ConstCacheEntryIterator findEntry(const PoolCache& cache, const std::string& key) const {
        auto entry = cache.prefix_index.find(key);
        return entry == cache.prefix_index.end() ? cache.entries_by_recency.end() : entry->second;
    }
    ConstCacheEntryIterator findEntry(const PoolCache& cache, CacheBlockLocation location) const {
        auto entry = cache.reverse_index.find(location);
        return entry == cache.reverse_index.end() ? cache.entries_by_recency.end() : entry->second;
    }
    void eraseEntry(PoolCache& cache, CacheEntryIterator entry) {
        cache.prefix_index.erase(entry->key.content_hash);
        cache.reverse_index.erase(entry->block->Location());
        cache.entries_by_recency.erase(entry);
    }
    void touch(PoolCache& cache, CacheEntryIterator entry, std::uint64_t& next_recency) {
        entry->last_access = ++next_recency;
        cache.entries_by_recency.splice(cache.entries_by_recency.end(), cache.entries_by_recency, entry);
    }

    // Caches are pool-scoped because the same Manager can serve device and
    // host tiers. Every referenced BlockPool must outlive this Manager.
    std::unordered_map<const BlockPool*, PoolCache> pool_caches_;
};

}  // namespace tokenspeed
