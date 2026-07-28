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
#include <cstdint>
#include <iterator>
#include <limits>
#include <list>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/block_pool.h"
#include "cache/cache_block_ref.h"
#include "cache/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// Per-attention-type token policy plus cache metadata for one group. Physical
// placement remains entirely in BlockPool.
class KvCacheManager {
public:
    // Read-only admission snapshot from one cache-index lookup; owns no block.
    struct CachedBlockMetadata {
        std::uint64_t last_access_epoch;
        std::int32_t logical_block_index;
        CacheBoundaryKind boundary_kind;
        bool was_acquired;
    };

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
    GroupId Id() const noexcept { return group_id_; }

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
        for (const CacheBlockRef& block_ref : table.Blocks()) {
            ids.push_back(block_ref ? ResolveKernelPageId(block_ref->Location()) : 0);
        }
        return ids;
    }

    virtual bool MatchIsPrefixClosed() const = 0;
    virtual std::int32_t BoundaryLookbackBlocks() const = 0;
    virtual GroupPrefixProbe Probe(const BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                                   std::int32_t max_blocks) const = 0;

    PrefixMatch AcquireMatchedBlocks(BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
                                     const GroupPrefixProbe& probe, std::uint64_t access_epoch) {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        PrefixMatch match;
        match.blocks.resize(probe.hits.size());
        CacheEntries* cache_index = findCacheEntries(pool);
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            _assert(cache_index != nullptr, "cached pool disappeared between match probe and acquisition");
            CacheEntryIterator entry_it = findEntry(*cache_index, keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(entry_it != cache_index->entries.end(),
                    "cached block disappeared between match probe and acquisition");
            entry_it->was_acquired = true;
            entry_it->last_access_epoch = access_epoch;
            match.blocks[i] = entry_it->block_ref;
        }
        return match;
    }

    std::vector<CacheBlockLocation> MatchedBlockLocations(const BlockPool& pool, std::span<const CacheKey> keys,
                                                          std::int32_t begin_blocks,
                                                          const GroupPrefixProbe& probe) const {
        _assert(begin_blocks >= 0 && static_cast<std::size_t>(begin_blocks) + probe.hits.size() <= keys.size(),
                "matched block range is out of bounds");
        std::vector<CacheBlockLocation> locations;
        locations.reserve(static_cast<std::size_t>(std::ranges::count(probe.hits, std::uint8_t{1})));
        const CacheEntries* cache_index = findCacheEntries(pool);
        for (std::size_t i = 0; i < probe.hits.size(); ++i) {
            if (probe.hits[i] == 0) {
                continue;
            }
            _assert(cache_index != nullptr, "cached pool disappeared between match probes");
            ConstCacheEntryIterator entry_it =
                findEntry(*cache_index, keys[static_cast<std::size_t>(begin_blocks) + i]);
            _assert(entry_it != cache_index->entries.end(), "cached block disappeared between match probes");
            locations.push_back(entry_it->block_ref->Location());
        }
        return locations;
    }

    void ClaimHitBlocks(BlockTable& table, PrefixMatch&& hit) {
        _assert(table.blocks_.empty(), "ClaimHitBlocks requires a fresh (empty) table");
        table.blocks_ = std::move(hit.blocks);
    }

    bool Acquire(BlockPool& pool, BlockTable& table, std::int32_t num_tokens, std::int32_t reserve_tokens = 0) {
        _assert(num_tokens >= 0 && reserve_tokens >= 0, "token demand and reserve must be non-negative");
        const std::int32_t num_pages = BlocksNeededFor(table, num_tokens + reserve_tokens);
        if (num_pages == 0) {
            table.available_tokens_ -= num_tokens;
            return true;
        }
        table.blocks_.reserve(table.blocks_.size() + static_cast<std::size_t>(num_pages));
        std::vector<CacheBlockRef> new_block_refs =
            pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, num_pages);
        if (static_cast<std::int32_t>(new_block_refs.size()) < num_pages) {
            return false;
        }
        appendBlocks(table, num_tokens, std::move(new_block_refs));
        return true;
    }

    void AppendHostExtension(BlockPool& pool, BlockTable& table, std::vector<CacheBlockRef>&& host_block_refs,
                             std::vector<BlockTransfer>& load_pairs) {
        _assert(table.available_tokens_ == 0, "host extension must append on a full-page boundary");
        const std::int32_t num_pages = static_cast<std::int32_t>(std::ranges::count_if(
            host_block_refs, [](const CacheBlockRef& block_ref) { return static_cast<bool>(block_ref); }));
        table.blocks_.reserve(table.blocks_.size() + host_block_refs.size());
        std::vector<CacheBlockRef> destination_refs =
            pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, num_pages);
        FatalCheck(static_cast<std::int32_t>(destination_refs.size()) == num_pages,
                   "admission plan no longer fits the block pool");
        auto destination_it = destination_refs.begin();
        for (CacheBlockRef& host_block_ref : host_block_refs) {
            if (!host_block_ref) {
                table.blocks_.emplace_back();
                continue;
            }
            _assert(destination_it != destination_refs.end(), "missing host extension destination");
            table.blocks_.push_back(std::move(*destination_it));
            ++destination_it;
            load_pairs.push_back(BlockTransfer{
                .group_id = group_id_,
                .source = std::move(host_block_ref),
                .destination = table.blocks_.back(),
            });
        }
        _assert(destination_it == destination_refs.end(), "unused host extension destination");
    }

    std::int32_t BlocksNeededFor(const BlockTable& table, std::int32_t num_tokens) const {
        if (num_tokens <= table.available_tokens_) {
            return 0;
        }
        const std::int32_t over = num_tokens - table.available_tokens_;
        return (over + cache_block_tokens_ - 1) / cache_block_tokens_;
    }

    std::int32_t BlocksNeededFor(const BlockTable& table, const GroupDemand& demand) const {
        if (demand.materialized_suffix_start < 0) {
            return BlocksNeededFor(table, demand.num_tokens + demand.reserve_tokens);
        }

        // Decode-side prefix acquisition may have already installed aligned
        // null holes for state. They carry no ownership and remain safe to
        // extend sparsely up to the remote endpoint snapshot.
        _assert(table.available_tokens_ == 0, "sparse suffix materialization requires a page boundary");
        _assert(table.blocks_.size() <= static_cast<std::size_t>(demand.materialized_suffix_start),
                "sparse suffix overlaps the existing block table");
        _assert(std::ranges::all_of(table.blocks_, [](const CacheBlockRef& block_ref) { return !block_ref; }),
                "sparse suffix prefix must contain only null holes");
        _assert(demand.num_tokens > 0 && demand.reserve_tokens >= 0,
                "sparse suffix materialization requires a positive extent");
        const std::int64_t extent = static_cast<std::int64_t>(demand.num_tokens) + demand.reserve_tokens;
        _assert(extent <= std::numeric_limits<std::int32_t>::max(), "sparse suffix extent exceeds int32 range");
        const std::int32_t last_block = static_cast<std::int32_t>((extent - 1) / cache_block_tokens_);
        _assert(demand.materialized_suffix_start <= last_block,
                "materialized suffix starts beyond the requested extent");
        return last_block - demand.materialized_suffix_start + 1;
    }

    bool Acquire(BlockPool& pool, BlockTable& table, const GroupDemand& demand) {
        if (demand.materialized_suffix_start < 0) {
            return Acquire(pool, table, demand.num_tokens, demand.reserve_tokens);
        }

        const std::int32_t num_blocks = BlocksNeededFor(table, demand);
        std::vector<CacheBlockRef> block_refs =
            pool.AcquireBlocks(group_id_, cache_blocks_per_lcm_block_, num_blocks);
        if (static_cast<std::int32_t>(block_refs.size()) != num_blocks) {
            return false;
        }

        const std::int64_t extent = static_cast<std::int64_t>(demand.num_tokens) + demand.reserve_tokens;
        const std::int32_t logical_blocks = static_cast<std::int32_t>(
            (extent + cache_block_tokens_ - 1) / cache_block_tokens_);
        table.blocks_.resize(static_cast<std::size_t>(logical_blocks));
        for (std::size_t i = 0; i < block_refs.size(); ++i) {
            table.blocks_[static_cast<std::size_t>(demand.materialized_suffix_start) + i] =
                std::move(block_refs[i]);
        }
        table.available_tokens_ = logical_blocks * cache_block_tokens_ - demand.num_tokens;
        return true;
    }

    void RegisterCachedBlock(BlockPool& pool, CacheBlockRef& block_ref, const CacheKey& key,
                             std::uint64_t access_epoch, std::int32_t logical_block_index = -1,
                             CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk,
                             std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached = nullptr) {
        _assert(block_ref && block_ref.IsOwnedBy(pool), "cache block must belong to the target pool");
        validateKey(key);
        CacheEntries& cache_index = cacheEntries(pool);
        CacheEntryIterator existing_it = findEntry(cache_index, block_ref->Location());
        if (existing_it != cache_index.entries.end()) {
            _assert(existing_it->key == key, "one cache block location cannot change cache key");
            if (existing_it->boundary_kind < boundary_kind) {
                existing_it->boundary_kind = boundary_kind;
            }
            existing_it->last_access_epoch = access_epoch;
            return;
        }
        CacheEntryIterator canonical_it = findEntry(cache_index, key);
        if (canonical_it != cache_index.entries.end()) {
            if (canonical_it->boundary_kind < boundary_kind) {
                canonical_it->boundary_kind = boundary_kind;
            }
            canonical_it->last_access_epoch = access_epoch;
            block_ref = canonical_it->block_ref;
            return;
        }

        cache_index.entries.push_back(CacheEntry{
            .key = key,
            .block_ref = block_ref,
            .last_access_epoch = access_epoch,
            .logical_block_index = logical_block_index,
            .boundary_kind = boundary_kind,
        });
        CacheEntryIterator entry_it = std::prev(cache_index.entries.end());
        cache_index.by_key.emplace(entry_it->key, entry_it);
        cache_index.by_location.emplace(entry_it->block_ref->Location(), entry_it);
        if (newly_cached != nullptr) {
            newly_cached->emplace_back(key, block_ref);
        }
    }

    void CacheFullBlocks(BlockPool& pool, BlockTable& table, std::span<const CacheKey> keys,
                         std::uint64_t access_epoch, std::int32_t first_slot = 0,
                         CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk,
                         std::vector<std::pair<CacheKey, CacheBlockRef>>* newly_cached = nullptr) {
        _assert(first_slot >= 0, "first_slot must be >= 0");
        _assert(static_cast<std::int64_t>(first_slot) + static_cast<std::int64_t>(keys.size()) <= table.NumBlocks(),
                "key range exceeds table size");
        for (std::size_t j = 0; j < keys.size(); ++j) {
            CacheBlockRef& block_ref = table.blocks_[static_cast<std::size_t>(first_slot) + j];
            if (!block_ref) {
                continue;
            }
            RegisterCachedBlock(pool, block_ref, keys[j], access_epoch,
                                first_slot + static_cast<std::int32_t>(j), boundary_kind, newly_cached);
        }
    }

    bool ContainsCachedBlock(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, key) != cache_index->entries.end();
    }
    bool ContainsCachedBlock(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, location) != cache_index->entries.end();
    }
    std::optional<CachedBlockMetadata> CachedBlockMetadataFor(const BlockPool& pool,
                                                              CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end()) {
            return std::nullopt;
        }
        return CachedBlockMetadata{
            .last_access_epoch = entry_it->last_access_epoch,
            .logical_block_index = entry_it->logical_block_index,
            .boundary_kind = entry_it->boundary_kind,
            .was_acquired = entry_it->was_acquired,
        };
    }
    std::int32_t NumCachedBlocks(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index == nullptr ? 0 : static_cast<std::int32_t>(cache_index->entries.size());
    }
    std::int32_t NumPinnedCachedBlocks(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return 0;
        }
        return static_cast<std::int32_t>(std::ranges::count_if(
            cache_index->entries,
            [](const CacheEntry& cache_entry) { return cache_entry.block_ref.use_count() > 1; }));
    }

    std::vector<CacheBlockLocation> EvictableBlockLocations(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return {};
        }
        std::vector<CacheBlockLocation> locations;
        for (const CacheEntry& cache_entry : cache_index->entries) {
            if (cache_entry.block_ref.unique()) {
                locations.push_back(cache_entry.block_ref->Location());
            }
        }
        return locations;
    }

    bool EvictCachedBlock(const BlockPool& pool, CacheBlockLocation location) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        CacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
            return false;
        }
        eraseEntry(*cache_index, entry_it);
        return true;
    }

    bool ParentIsFullyEvictable(const BlockPool& pool, std::int32_t lcm_block_id) const {
        if (pool.OccupiedCount(lcm_block_id) == 0) {
            return false;
        }
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        for (std::int32_t slot = 0; slot < cache_blocks_per_lcm_block_; ++slot) {
            const CacheBlockLocation location{.lcm_block_id = lcm_block_id, .slot_index = slot};
            if (!pool.IsOccupied(location)) {
                continue;
            }
            ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
            if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
                return false;
            }
        }
        return true;
    }

    virtual void ReclaimExpired(BlockPool& /*pool*/, BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/) {}
    virtual std::int32_t BlocksReclaimableAt(const BlockTable& /*table*/, std::int32_t /*num_computed_tokens*/,
                                             bool /*count_uncached*/) const {
        return 0;
    }
    virtual std::vector<CacheBlockLocation> ReclaimableBlockLocationsAt(const BlockTable& /*table*/,
                                                                        std::int32_t /*num_computed_tokens*/) const {
        return {};
    }

    void ConsumeAvailable(BlockTable& table, std::int32_t num_tokens) {
        _assert(num_tokens >= 0 && num_tokens <= table.available_tokens_,
                "token demand exceeds the available capacity");
        table.available_tokens_ -= num_tokens;
    }

    void Free(BlockTable& table) {
        // Release the logical suffix first so newly emptied LCM parents enter
        // the FIFO free queue in deterministic table order.
        for (auto it = table.blocks_.rbegin(); it != table.blocks_.rend(); ++it) {
            it->reset();
        }
        table.blocks_.clear();
        table.available_tokens_ = 0;
    }

protected:
    bool ContainsCachedBlock(const CacheBlockRef& block_ref) const {
        if (!block_ref) {
            return false;
        }
        return std::ranges::any_of(cache_entries_by_pool_, [&](const auto& item) {
            auto index_it = item.second.by_location.find(block_ref->Location());
            return index_it != item.second.by_location.end() && index_it->second->block_ref == block_ref;
        });
    }

    std::int32_t cache_block_tokens_;
    std::int32_t cache_blocks_per_lcm_block_;
    GroupId group_id_;

private:
    struct CacheEntry {
        CacheKey key;
        CacheBlockRef block_ref;
        std::uint64_t last_access_epoch{0};
        // Position in the request's logical prefix. Host-only entries may not
        // have a device-table position yet.
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        // Set only after a successful request admission acquires this entry.
        bool was_acquired{false};
    };

    using CacheEntryList = std::list<CacheEntry>;
    using CacheEntryIterator = CacheEntryList::iterator;
    using ConstCacheEntryIterator = CacheEntryList::const_iterator;

    struct CacheEntries {
        // Owns each CacheEntry once. The maps are non-owning secondary indices
        // into stable list nodes for key and location lookup. Global eviction
        // order is derived by AdmissionPlanner from CacheEntry metadata.
        CacheEntryList entries;
        std::unordered_map<CacheKey, CacheEntryIterator, CacheKeyHash> by_key;
        std::unordered_map<CacheBlockLocation, CacheEntryIterator, CacheBlockLocationHash> by_location;
    };

    void appendBlocks(BlockTable& table, std::int32_t num_tokens, std::vector<CacheBlockRef> block_refs) {
        const std::int32_t added_tokens = static_cast<std::int32_t>(block_refs.size()) * cache_block_tokens_;
        _assert(num_tokens <= table.available_tokens_ + added_tokens,
                "allocated blocks do not cover the immediate token demand");
        for (CacheBlockRef& block_ref : block_refs) {
            table.blocks_.push_back(std::move(block_ref));
        }
        table.available_tokens_ += added_tokens - num_tokens;
    }

    CacheEntries& cacheEntries(const BlockPool& pool) {
        return cache_entries_by_pool_.try_emplace(&pool).first->second;
    }
    CacheEntries* findCacheEntries(const BlockPool& pool) {
        auto it = cache_entries_by_pool_.find(&pool);
        return it == cache_entries_by_pool_.end() ? nullptr : &it->second;
    }
    const CacheEntries* findCacheEntries(const BlockPool& pool) const {
        auto it = cache_entries_by_pool_.find(&pool);
        return it == cache_entries_by_pool_.end() ? nullptr : &it->second;
    }
    void validateKey(const CacheKey& key) const {
        _assert(key.group_id == group_id_, "cache key group does not match manager");
        _assert(!key.content_hash.empty(), "cache key content hash must not be empty");
    }
    CacheEntryIterator findEntry(CacheEntries& cache_index, const CacheKey& key) {
        validateKey(key);
        auto index_it = cache_index.by_key.find(key);
        return index_it == cache_index.by_key.end() ? cache_index.entries.end() : index_it->second;
    }
    CacheEntryIterator findEntry(CacheEntries& cache_index, CacheBlockLocation location) {
        auto index_it = cache_index.by_location.find(location);
        return index_it == cache_index.by_location.end() ? cache_index.entries.end() : index_it->second;
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, const CacheKey& key) const {
        validateKey(key);
        auto index_it = cache_index.by_key.find(key);
        return index_it == cache_index.by_key.end() ? cache_index.entries.end() : index_it->second;
    }
    ConstCacheEntryIterator findEntry(const CacheEntries& cache_index, CacheBlockLocation location) const {
        auto index_it = cache_index.by_location.find(location);
        return index_it == cache_index.by_location.end() ? cache_index.entries.end() : index_it->second;
    }
    void eraseEntry(CacheEntries& cache_index, CacheEntryIterator entry_it) {
        cache_index.by_key.erase(entry_it->key);
        cache_index.by_location.erase(entry_it->block_ref->Location());
        cache_index.entries.erase(entry_it);
    }

    // Indices are pool-scoped because the same Manager can serve device and
    // host tiers. Every referenced BlockPool must outlive this Manager.
    std::unordered_map<const BlockPool*, CacheEntries> cache_entries_by_pool_;
};

}  // namespace tokenspeed
