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
#include <list>
#include <map>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/core/block_pool.h"
#include "cache/core/block_table.h"
#include "cache/core/cache_block_ref.h"
#include "cache/core/cache_types.h"
#include "utils.h"

namespace tokenspeed {

// One cache group's prefix-reuse index: CacheKey -> canonical CacheBlock.
// It decides WHAT is reusable; placement and allocation stay in BlockPool and
// GroupAllocator. Indices are pool-scoped because the same group serves the
// Device and Host tiers; every referenced BlockPool must outlive this index.
class PrefixCacheIndex {
private:
    struct CacheEntry;
    struct CacheEntries;
    using CacheEntryList = std::list<CacheEntry>;
    using CacheEntryIterator = CacheEntryList::iterator;
    using ConstCacheEntryIterator = CacheEntryList::const_iterator;

    // Index key: oldest access epoch first, with location as a stable tie-break.
    struct EvictionOrder {
        std::uint64_t last_access_epoch{0};
        std::int32_t lcm_block_id{0};
        std::int32_t slot_index{0};

        auto operator<=>(const EvictionOrder&) const noexcept = default;
    };

    using EvictionOrderMap = std::map<EvictionOrder, CacheEntryIterator>;

public:
    // Read-only admission snapshot from one index lookup; owns no block.
    struct CachedBlockMetadata {
        std::uint64_t last_access_epoch{0};
        std::int32_t logical_block_index{-1};
        CacheBoundaryKind boundary_kind{CacheBoundaryKind::kChunk};
        bool was_acquired{false};
    };

    // Non-owning traversal state, bound on first use to one index/pool. The
    // index and pool must outlive it; do not insert, erase, or re-key entries
    // in that tier until traversal is finished. Use a fresh cursor per plan.
    class EvictionCursor {
        friend class PrefixCacheIndex;
        const CacheEntries* cache_index_{nullptr};
        EvictionOrderMap::const_iterator next_;
    };

    // One unpinned entry, paired with the metadata an eviction policy reads.
    struct EvictionCandidate {
        CacheBlockLocation location;
        CachedBlockMetadata metadata;
    };

    explicit PrefixCacheIndex(std::uint32_t group_id) : group_id_{group_id} {}

    PrefixCacheIndex(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex(PrefixCacheIndex&&) = default;
    PrefixCacheIndex& operator=(PrefixCacheIndex&&) = default;

    std::uint32_t GroupId() const noexcept { return group_id_; }

    // Registers block_ref under key. If key already has a canonical block,
    // block_ref is replaced with a reference to that block.
    void Register(const BlockPool& pool, CacheBlockRef& block_ref, const CacheKey& key, std::uint64_t access_epoch,
                  std::int32_t logical_block_index = -1, CacheBoundaryKind boundary_kind = CacheBoundaryKind::kChunk,
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
            touchEntry(cache_index, existing_it, access_epoch);
            return;
        }
        CacheEntryIterator canonical_it = findEntry(cache_index, key);
        if (canonical_it != cache_index.entries.end()) {
            if (canonical_it->boundary_kind < boundary_kind) {
                canonical_it->boundary_kind = boundary_kind;
            }
            touchEntry(cache_index, canonical_it, access_epoch);
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
        cache_index.by_eviction_order.emplace(evictionOrder(*entry_it), entry_it);
        if (newly_cached != nullptr) {
            newly_cached->emplace_back(key, block_ref);
        }
    }

    void RegisterFullBlocks(const BlockPool& pool, BlockTable& table, std::span<const CacheKey> keys,
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
            Register(pool, block_ref, keys[j], access_epoch, first_slot + static_cast<std::int32_t>(j), boundary_kind,
                     newly_cached);
        }
    }

    bool Contains(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, key) != cache_index->entries.end();
    }
    bool Contains(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index != nullptr && findEntry(*cache_index, location) != cache_index->entries.end();
    }
    // Any-tier lookup that also checks identity, not just location.
    bool Contains(const CacheBlockRef& block_ref) const {
        if (!block_ref) {
            return false;
        }
        return std::ranges::any_of(cache_entries_by_pool_, [&](const auto& item) {
            auto index_it = item.second.by_location.find(block_ref->Location());
            return index_it != item.second.by_location.end() && index_it->second->block_ref == block_ref;
        });
    }

    CacheBlockRef Find(const BlockPool& pool, const CacheKey& key) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return {};
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, key);
        return entry_it == cache_index->entries.end() ? CacheBlockRef{} : entry_it->block_ref;
    }

    std::optional<CachedBlockMetadata> MetadataFor(const BlockPool& pool, CacheBlockLocation location) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        ConstCacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end()) {
            return std::nullopt;
        }
        return metadataOf(*entry_it);
    }

    // Appends all unpinned entries of the next epoch containing candidates.
    // Skips fully pinned epochs and returns false at exhaustion. The cursor
    // advances continuously, without a tree lookup for each skipped epoch.
    // Complete epochs let admission apply its policy's tie-breaks locally.
    bool NextEvictionEpoch(const BlockPool& pool, EvictionCursor& cursor, std::vector<EvictionCandidate>& out) const {
        if (cursor.cache_index_ == nullptr) {
            cursor.cache_index_ = findCacheEntries(pool);
            if (cursor.cache_index_ == nullptr) {
                return false;
            }
            cursor.next_ = cursor.cache_index_->by_eviction_order.begin();
        }
        const auto end = cursor.cache_index_->by_eviction_order.end();
        while (cursor.next_ != end) {
            const std::uint64_t epoch = cursor.next_->first.last_access_epoch;
            const std::size_t initial_size = out.size();
            for (; cursor.next_ != end && cursor.next_->first.last_access_epoch == epoch; ++cursor.next_) {
                const CacheEntry& cache_entry = *cursor.next_->second;
                if (cache_entry.block_ref.unique()) {
                    out.push_back(EvictionCandidate{
                        .location = cache_entry.block_ref->Location(),
                        .metadata = metadataOf(cache_entry),
                    });
                }
            }
            if (out.size() != initial_size) {
                return true;
            }
        }
        return false;
    }

    std::int32_t NumEntries(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        return cache_index == nullptr ? 0 : static_cast<std::int32_t>(cache_index->entries.size());
    }
    std::int32_t NumPinnedEntries(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return 0;
        }
        return static_cast<std::int32_t>(std::ranges::count_if(
            cache_index->entries, [](const CacheEntry& cache_entry) { return cache_entry.block_ref.use_count() > 1; }));
    }

    // Every unpinned entry of one tier, in no particular order. Callers that
    // only want the oldest few should use NextEvictionEpoch() instead; this is
    // for the ones that genuinely need the whole set. The metadata rides along
    // because recovering it afterwards would cost a lookup per entry.
    std::vector<EvictionCandidate> EvictableCandidates(const BlockPool& pool) const {
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return {};
        }
        std::vector<EvictionCandidate> candidates;
        candidates.reserve(cache_index->entries.size());
        for (const CacheEntry& cache_entry : cache_index->entries) {
            if (cache_entry.block_ref.unique()) {
                candidates.push_back(EvictionCandidate{
                    .location = cache_entry.block_ref->Location(),
                    .metadata = metadataOf(cache_entry),
                });
            }
        }
        return candidates;
    }

    std::optional<CacheKey> Evict(const BlockPool& pool, CacheBlockLocation location) {
        CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return std::nullopt;
        }
        CacheEntryIterator entry_it = findEntry(*cache_index, location);
        if (entry_it == cache_index->entries.end() || !entry_it->block_ref.unique()) {
            return std::nullopt;
        }
        CacheKey key = entry_it->key;
        eraseEntry(*cache_index, entry_it);
        return key;
    }

    // True when every occupied child of the LCM parent is an unpinned entry of
    // this index, i.e. evicting the parent loses only reusable cache.
    bool ParentIsFullyEvictable(const BlockPool& pool, std::int32_t lcm_block_id,
                                std::int32_t cache_blocks_per_lcm_block) const {
        if (pool.OccupiedCount(lcm_block_id) == 0) {
            return false;
        }
        const CacheEntries* cache_index = findCacheEntries(pool);
        if (cache_index == nullptr) {
            return false;
        }
        for (std::int32_t slot = 0; slot < cache_blocks_per_lcm_block; ++slot) {
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

    // Pins the probed hits: marks them acquired at access_epoch and returns
    // owning references aligned with probe.hits.
    PrefixMatch AcquireMatched(const BlockPool& pool, std::span<const CacheKey> keys, std::int32_t begin_blocks,
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
            touchEntry(*cache_index, entry_it, access_epoch);
            match.blocks[i] = entry_it->block_ref;
        }
        return match;
    }

    std::vector<CacheBlockLocation> MatchedLocations(const BlockPool& pool, std::span<const CacheKey> keys,
                                                     std::int32_t begin_blocks, const GroupPrefixProbe& probe) const {
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

    struct CacheEntries {
        // Owns each CacheEntry once. The maps are non-owning secondary indices
        // into stable list nodes. by_eviction_order additionally keeps the
        // entries sorted by age, so a caller reaches the oldest ones without
        // visiting the rest; AdmissionPlanner still owns the policy that ranks
        // entries within one access epoch.
        CacheEntryList entries;
        std::unordered_map<CacheKey, CacheEntryIterator, CacheKeyHash> by_key;
        std::unordered_map<CacheBlockLocation, CacheEntryIterator, CacheBlockLocationHash> by_location;
        EvictionOrderMap by_eviction_order;
    };

    static CachedBlockMetadata metadataOf(const CacheEntry& cache_entry) noexcept {
        return CachedBlockMetadata{
            .last_access_epoch = cache_entry.last_access_epoch,
            .logical_block_index = cache_entry.logical_block_index,
            .boundary_kind = cache_entry.boundary_kind,
            .was_acquired = cache_entry.was_acquired,
        };
    }
    static EvictionOrder evictionOrder(const CacheEntry& cache_entry) {
        const CacheBlockLocation location = cache_entry.block_ref->Location();
        return EvictionOrder{
            .last_access_epoch = cache_entry.last_access_epoch,
            .lcm_block_id = location.lcm_block_id,
            .slot_index = location.slot_index,
        };
    }
    // The only writer of last_access_epoch: an entry's location never moves, so
    // re-keying it in the eviction order is a plain erase-then-insert.
    void touchEntry(CacheEntries& cache_index, CacheEntryIterator entry_it, std::uint64_t access_epoch) {
        if (entry_it->last_access_epoch == access_epoch) {
            return;
        }
        cache_index.by_eviction_order.erase(evictionOrder(*entry_it));
        entry_it->last_access_epoch = access_epoch;
        cache_index.by_eviction_order.emplace(evictionOrder(*entry_it), entry_it);
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
        _assert(key.group_id == group_id_, "cache key group does not match index");
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
        cache_index.by_eviction_order.erase(evictionOrder(*entry_it));
        cache_index.by_key.erase(entry_it->key);
        cache_index.by_location.erase(entry_it->block_ref->Location());
        cache_index.entries.erase(entry_it);
    }

    std::uint32_t group_id_;
    std::unordered_map<const BlockPool*, CacheEntries> cache_entries_by_pool_;
};

}  // namespace tokenspeed
